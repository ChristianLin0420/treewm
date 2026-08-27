#!/usr/bin/env python3
"""Run one fresh Exp21 25k bridge slot with exact resume and safe requeue."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from campaign import (
    CAMPAIGN_DIR,
    ContractError,
    REPOSITORY_ROOT,
    STAGE_TARGET,
    atomic_json,
    file_sha256,
    load_manifest,
    read_json,
    require,
    run_at,
    stable_hash,
    trainer_command,
)
from metric_boundary import recover_metric_boundary


OBJECTIVE = "treewm_v2_grounded_gauge_all_ten_bridge_v2"
GRACEFUL_EXIT_CODE = 75
CANCEL_EXIT_CODE = 143


def classify_child_exit(
    status: int,
    *,
    cancel_requested: bool,
    cancel_latch_exists: bool,
    requeue_requested: bool,
) -> str:
    if status == 0:
        return "complete"
    if cancel_requested or cancel_latch_exists:
        return "cancelled"
    if status == GRACEFUL_EXIT_CODE or requeue_requested:
        return "requeue"
    return "failed"


class SignalState:
    """Forward lifecycle signals to the trainer and durably latch intent."""

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.child: subprocess.Popen[Any] | None = None
        self.requeue_requested = False
        self.cancel_requested = False

    @property
    def cancel_latch(self) -> Path:
        return self.state_dir / "CANCEL_REQUESTED"

    @property
    def requeue_latch(self) -> Path:
        return self.state_dir / "REQUEUE_REQUESTED.json"

    def _forward(self, signum: int) -> None:
        if self.child is not None and self.child.poll() is None:
            try:
                self.child.send_signal(signum)
            except ProcessLookupError:
                pass

    def request_requeue(self, signum: int = signal.SIGUSR1, _frame: object = None) -> None:
        if self.cancel_requested or self.cancel_latch.exists() or self.requeue_requested:
            return
        self.requeue_requested = True
        atomic_json(self.requeue_latch, {
            "schema_version": 1,
            "status": "checkpoint_requested",
            "signal": signal.Signals(signum).name,
            "unix_time": time.time(),
        })
        self._forward(signal.SIGUSR1)

    def request_cancel(self, signum: int = signal.SIGTERM, _frame: object = None) -> None:
        calling = Path(os.environ.get(
            "TREEWM_REQUEUE_CALLING_MARKER",
            str(self.state_dir / "REQUEUE_CALLING.json"),
        ))
        if calling.exists() or self.cancel_requested:
            return
        self.cancel_requested = True
        atomic_json(self.cancel_latch, {
            "schema_version": 1,
            "status": "cancel_requested",
            "signal": signal.Signals(signum).name,
            "unix_time": time.time(),
        })
        self._forward(signal.SIGTERM)

    def forward_latched_intent(self) -> None:
        """Close the signal window between handler installation and child creation."""
        if self.cancel_requested or self.cancel_latch.exists():
            self.cancel_requested = True
            self._forward(signal.SIGTERM)
        elif self.requeue_requested or self.requeue_latch.exists():
            self.requeue_requested = True
            self._forward(signal.SIGUSR1)


def _rank_state_complete(payload: Mapping[str, Any]) -> bool:
    identity = payload.get("run_identity") or {}
    rank_states = payload.get("rank_states") or []
    if int(identity.get("world_size", -1)) != 1 or len(rank_states) != 1:
        return False
    state = rank_states[0] or {}
    if not isinstance(state, Mapping):
        return False
    loader = state.get("loader") or {}
    streams = state.get("rng_streams") or {}
    try:
        from treewm.logging.metrics import MetricTracker

        MetricTracker().load_state_dict(state.get("metric_tracker"))
    except (TypeError, ValueError):
        return False
    return bool(
        int(state.get("rank", -1)) == 0
        and state.get("rng_state")
        and set(streams) >= {"planner", "eval", "viz"}
        and state.get("horizon_generator") is not None
        and set(loader) >= {"epoch", "batches_yielded_in_epoch", "epoch_generator_state"}
        and loader.get("epoch_generator_state") is not None
        and set(payload.get("checkpoint_manager") or {}) >= {"best_success", "best_val_loss"}
    )


def verify_checkpoint(
    path: Path,
    launch: Mapping[str, Any],
    *,
    expected_step: int | None = None,
) -> dict[str, Any]:
    """Verify a single-rank exact-resume payload and every campaign seal."""
    import torch

    require(path.is_file() and not path.is_symlink(), f"checkpoint missing/symlinked: {path}")
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ContractError(f"checkpoint cannot be loaded: {exc}") from exc
    from treewm.utils.checkpoint import validate_exact_resume_payload

    try:
        validate_exact_resume_payload(
            payload,
            expected_identity=payload.get("run_identity"),
            expected_world_size=1,
            require_cuda_rng=True,
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"checkpoint exact-resume payload is invalid: {exc}") from exc

    completed = int(payload.get("completed_updates", -1))
    identity = payload.get("run_identity") or {}
    config = payload.get("config") or {}
    run = launch["run"]
    hashes = launch["hashes"]
    expected_arm = f"exp20-{run['selected_arm']}-all-ten"
    checks = (
        payload.get("schema_version") == 2,
        0 <= completed <= STAGE_TARGET,
        expected_step is None or completed == expected_step,
        int(payload.get("step", -1)) == completed,
        int(payload.get("next_step", -1)) == completed,
        payload.get("optimizer") is not None,
        payload.get("scheduler") is not None,
        _rank_state_complete(payload),
        identity.get("run_name") == run["run_name"],
        identity.get("setting") == run["setting_id"],
        int(identity.get("seed", -1)) == int(run["seed"]),
        identity.get("objective_version") == OBJECTIVE,
        int(identity.get("total_steps", -1)) == STAGE_TARGET,
        int(identity.get("scheduler_total_steps", -1)) == 1_000_000,
        identity.get("protocol_sha256") == hashes["run_protocol_sha256"],
        identity.get("code_sha256") == hashes["source_sha256"],
        identity.get("runtime_sha256") == hashes["runtime_sha256"],
        identity.get("recipe_code_sha256") == hashes["recipe_code_sha256"],
        identity.get("recipe_runtime_sha256") == hashes["recipe_runtime_sha256"],
        identity.get("campaign_source_sha256") == hashes["source_sha256"],
        identity.get("campaign_protocol_sha256") == hashes["package_protocol_sha256"],
        identity.get("campaign_config_sha256") == hashes["config_sha256"],
        identity.get("campaign_input_contract_sha256") == hashes["input_contract_sha256"],
        identity.get("campaign_factorial_arm") == expected_arm,
        identity.get("campaign_prerequisite_binding_sha256") == hashes["exp20_binding_sha256"],
        identity.get("campaign_selected_recipe_sha256") == hashes["selected_recipe_sha256"],
        identity.get("data_manifest_sha256") == hashes["data_manifest_sha256"],
        identity.get("calibration_sha256") == hashes["calibration_sha256"],
        identity.get("future_recipe_sha256") == hashes["future_recipe_sha256"],
        identity.get("evaluation_seed_tables_sha256") == hashes["evaluation_seed_tables_sha256"],
        identity.get("final_seed_table_sha256") == hashes["final_seed_table_sha256"],
        config.get("campaign_source_sha256") == hashes["source_sha256"],
        config.get("campaign_protocol_sha256") == hashes["package_protocol_sha256"],
        config.get("campaign_config_sha256") == hashes["config_sha256"],
        config.get("campaign_prerequisite_binding_sha256") == hashes["exp20_binding_sha256"],
        config.get("campaign_exp20_binding_sha256") == hashes["exp20_binding_sha256"],
        config.get("campaign_selected_recipe_sha256") == hashes["selected_recipe_sha256"],
    )
    require(all(checks), "checkpoint exact-resume identity is incomplete or differs")
    return {
        "schema_version": 1,
        "status": "checkpoint_verified",
        "completed_updates": completed,
        "reason": payload.get("reason"),
        "identity_sha256": payload.get("identity_sha256"),
        "evaluation_seed_tables_sha256": identity.get("evaluation_seed_tables_sha256"),
        "final_seed_table_sha256": identity.get("final_seed_table_sha256"),
        "checkpoint": str(path),
        "checkpoint_sha256": file_sha256(path),
        "launch_sha256": launch["launch_sha256"],
        "unix_time": time.time(),
    }


def verify_stage_marker(run_dir: Path, launch: Mapping[str, Any]) -> dict[str, Any]:
    path = run_dir / "stage-gates" / f"AWAITING_GATE_{STAGE_TARGET}.json"
    marker = read_json(path)
    checkpoint = verify_checkpoint(run_dir / "checkpoints/latest.pt", launch, expected_step=STAGE_TARGET)
    checks = (
        marker.get("schema_version") == 1,
        marker.get("status") == "awaiting_external_stage_gate",
        marker.get("objective_version") == OBJECTIVE,
        int(marker.get("completed_updates", -1)) == STAGE_TARGET,
        int(marker.get("step", -1)) == STAGE_TARGET,
        int(marker.get("total_steps", -1)) == STAGE_TARGET,
        int(marker.get("scheduler_total_steps", -1)) == 1_000_000,
        marker.get("identity_sha256") == checkpoint["identity_sha256"],
        marker.get("checkpoint") == "checkpoints/latest.pt",
        marker.get("checkpoint_sha256") == checkpoint["checkpoint_sha256"],
        marker.get("evaluation_seed_tables_sha256") == checkpoint["evaluation_seed_tables_sha256"],
        not (run_dir / "COMPLETED.json").exists(),
    )
    require(all(checks), "trainer boundary marker differs")
    return {"marker": str(path), **checkpoint}


def seal_or_validate(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        require(read_json(path) == dict(value), f"immutable artifact differs: {path}")
    else:
        atomic_json(path, value)


def validate_requeue_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    target: int = STAGE_TARGET,
) -> None:
    """Allow an exact-boundary preemption so resume can publish its marker."""
    require(int(checkpoint["completed_updates"]) <= target, "preemption checkpoint passed boundary without marker")
    require(str(checkpoint.get("reason", "")).startswith("graceful-stop:"), "requeue lacks graceful checkpoint")


def _cancel_before_launch(state_dir: Path, launch: Mapping[str, Any]) -> int:
    run_dir = Path(launch["run"]["run_directory"])
    checkpoint_path = run_dir / "checkpoints/latest.pt"
    checkpoint: dict[str, Any] | None = None
    if checkpoint_path.is_file():
        persistent = run_dir / "GAUGE_BRIDGE_LAUNCH.json"
        require(persistent.is_file() and read_json(persistent) == launch, "pre-cancel checkpoint is not owned")
        checkpoint = verify_checkpoint(checkpoint_path, launch)
    record = {
        "schema_version": 1,
        **(checkpoint or {}),
        "status": "cancelled_before_resume_with_verified_checkpoint" if checkpoint else "cancelled_before_trainer_launch",
        "stage_target": STAGE_TARGET,
        "run": dict(launch["run"]),
        "checkpoint_available": checkpoint is not None,
        "unix_time": time.time(),
    }
    record["cancel_sha256"] = stable_hash(record)
    atomic_json(state_dir / "CANCELLED.json", record)
    return CANCEL_EXIT_CODE


def run_worker(args: argparse.Namespace) -> int:
    root = args.repo_root.resolve()
    manifest = load_manifest(args.manifest)
    run = run_at(manifest, args.index)
    launch = trainer_command(manifest, run, repo_root=root)
    required_env = (
        ("TREEWM_EXPECTED_SOURCE_SHA256", "source_sha256"),
        ("TREEWM_EXPECTED_RUNTIME_SHA256", "runtime_sha256"),
        ("TREEWM_EXPECTED_PACKAGE_PROTOCOL_SHA256", "package_protocol_sha256"),
        ("TREEWM_EXPECTED_EXP20_BINDING_SHA256", "exp20_binding_sha256"),
        ("TREEWM_EXPECTED_SELECTED_RECIPE_SHA256", "selected_recipe_sha256"),
    )
    for env_name, hash_name in required_env:
        require(os.environ.get(env_name) == launch["hashes"][hash_name], f"{env_name} missing/differs")
    require(os.environ.get("TREEWM_EXPECTED_STAGE_TARGET") == str(STAGE_TARGET), "sealed stage target differs")

    state_dir = args.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    state = SignalState(state_dir)
    signal.signal(signal.SIGUSR1, state.request_requeue)
    signal.signal(signal.SIGTERM, state.request_cancel)
    signal.signal(signal.SIGINT, state.request_cancel)
    if state.cancel_latch.exists():
        return _cancel_before_launch(state_dir, launch)

    run_dir = Path(launch["run"]["run_directory"])
    persistent = run_dir / "GAUGE_BRIDGE_LAUNCH.json"
    if run_dir.exists() and not persistent.exists():
        raise ContractError("fresh Exp21 run refuses a pre-existing unowned directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    prior = read_json(persistent) if persistent.exists() else None
    require(prior in (None, launch), "run directory belongs to another launch")
    if prior is None:
        atomic_json(persistent, launch)

    stage_launch = {
        "schema_version": 1,
        "status": "fresh_stage_launch_sealed",
        "stage_target": STAGE_TARGET,
        "index": run.index,
        "launch_sha256": launch["launch_sha256"],
        "package_protocol_sha256": launch["hashes"]["package_protocol_sha256"],
        "exp20_binding_sha256": launch["hashes"]["exp20_binding_sha256"],
        "selected_recipe_sha256": launch["hashes"]["selected_recipe_sha256"],
    }
    stage_path = run_dir / "stage-gates" / f"STAGE_LAUNCH_{STAGE_TARGET}.json"
    seal_or_validate(stage_path, stage_launch)
    seal_or_validate(state_dir / "launch.json", {**stage_launch, "launch": launch})

    complete_path = run_dir / "stage-gates" / f"STAGE_COMPLETE_{STAGE_TARGET}.json"
    if complete_path.exists():
        marker = verify_stage_marker(run_dir, launch)
        complete = read_json(complete_path)
        require(complete.get("checkpoint_sha256") == marker["checkpoint_sha256"], "existing completion differs")
        atomic_json(state_dir / "WORKER_COMPLETE.json", complete)
        return 0

    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in launch["environment"].items()})
    environment["TREEWM_STOP_AFTER_UPDATE"] = str(STAGE_TARGET)
    state.child = subprocess.Popen(launch["argv"], cwd=root, env=environment)
    state.forward_latched_intent()
    while state.child.poll() is None:
        if state.cancel_latch.exists() and not state.cancel_requested:
            state.request_cancel()
        elif state.requeue_latch.exists() and not state.requeue_requested:
            state.request_requeue()
        time.sleep(0.5)
    status = int(state.child.returncode)
    state.child = None
    disposition = classify_child_exit(
        status,
        cancel_requested=state.cancel_requested,
        cancel_latch_exists=state.cancel_latch.exists(),
        requeue_requested=state.requeue_requested,
    )
    if disposition == "complete":
        marker = verify_stage_marker(run_dir, launch)
        complete = {
            "schema_version": 1,
            "status": "stage_complete_awaiting_campaign_gate",
            "campaign_id": manifest["campaign_id"],
            "index": run.index,
            "setting_id": run.setting_id,
            "seed": run.seed,
            "selected_arm": launch["run"]["selected_arm"],
            "stage_target": STAGE_TARGET,
            "launch_sha256": launch["launch_sha256"],
            "package_protocol_sha256": launch["hashes"]["package_protocol_sha256"],
            "source_sha256": launch["hashes"]["source_sha256"],
            "runtime_sha256": launch["hashes"]["runtime_sha256"],
            "exp20_binding_sha256": launch["hashes"]["exp20_binding_sha256"],
            "selected_recipe_sha256": launch["hashes"]["selected_recipe_sha256"],
            "identity_sha256": marker["identity_sha256"],
            "checkpoint_sha256": marker["checkpoint_sha256"],
            "evaluation_seed_tables_sha256": marker["evaluation_seed_tables_sha256"],
            "final_seed_table_sha256": marker["final_seed_table_sha256"],
            "completed_unix_time": time.time(),
        }
        complete["stage_complete_sha256"] = stable_hash(complete)
        atomic_json(complete_path, complete)
        atomic_json(state_dir / "WORKER_COMPLETE.json", complete)
        return 0
    if disposition == "cancelled":
        checkpoint = verify_checkpoint(run_dir / "checkpoints/latest.pt", launch)
        require(checkpoint.get("reason") == "graceful-stop:SIGTERM", "cancellation lacks SIGTERM checkpoint")
        require(int(checkpoint["completed_updates"]) < STAGE_TARGET, "cancellation raced past boundary")
        record = {
            **checkpoint,
            "schema_version": 1,
            "status": "cancelled_with_verified_checkpoint",
            "campaign_id": manifest["campaign_id"],
            "stage_target": STAGE_TARGET,
            "index": run.index,
            "setting_id": run.setting_id,
            "seed": run.seed,
            "child_exit_code": status,
            "unix_time": time.time(),
        }
        record["cancel_sha256"] = stable_hash(record)
        atomic_json(run_dir / f"stage-gates/CANCELLED_STAGE_{STAGE_TARGET}.json", record)
        atomic_json(state_dir / "CANCELLED.json", record)
        return CANCEL_EXIT_CODE
    if disposition == "requeue":
        checkpoint = verify_checkpoint(run_dir / "checkpoints/latest.pt", launch)
        validate_requeue_checkpoint(checkpoint)
        metric_recovery = recover_metric_boundary(
            run_dir / "checkpoints/latest.pt",
            run_dir,
            launch,
        )
        checkpoint = verify_checkpoint(run_dir / "checkpoints/latest.pt", launch)
        atomic_json(state_dir / "READY_FOR_REQUEUE.json", {
            **checkpoint,
            "metric_boundary_recovery": metric_recovery,
        })
        return GRACEFUL_EXIT_CODE
    atomic_json(state_dir / "FAILED.json", {
        "schema_version": 1,
        "status": "unexpected_child_failure",
        "child_exit_code": status,
        "unix_time": time.time(),
    })
    return status or 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--stage-target", type=int, choices=(STAGE_TARGET,), required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--manifest", type=Path, default=CAMPAIGN_DIR / "manifest.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    require(args.stage_target == STAGE_TARGET, "stage target differs")
    return run_worker(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"Exp21 worker contract error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
