#!/usr/bin/env python3
"""Run one exact grounded-formal lifecycle stage with durable requeue semantics."""

from __future__ import annotations

import argparse
import json
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
    STAGE_TARGETS,
    atomic_json,
    file_sha256,
    load_manifest,
    read_json,
    require,
    run_at,
    stable_hash,
    trainer_command,
)


GRACEFUL_EXIT_CODE = 75
CANCEL_EXIT_CODE = 143


def classify_child_exit(
    status: int,
    *,
    cancel_requested: bool,
    cancel_latch_exists: bool,
    requeue_requested: bool,
) -> str:
    """Resolve signal races without downgrading a verified stage completion."""
    if status == 0:
        return "complete"
    if cancel_requested or cancel_latch_exists:
        return "cancelled"
    if status == GRACEFUL_EXIT_CODE or requeue_requested:
        return "requeue"
    return "failed"


class SignalState:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.child: subprocess.Popen | None = None
        self.requeue_requested = False
        self.cancel_requested = False

    @property
    def cancel_latch(self) -> Path:
        return self.state_dir / "CANCEL_REQUESTED"

    def _forward(self, signum: int) -> None:
        if self.child is not None and self.child.poll() is None:
            try:
                self.child.send_signal(signum)
            except ProcessLookupError:
                pass

    def request_requeue(self, signum: int, _frame: object) -> None:
        if self.cancel_requested or self.cancel_latch.exists():
            return
        self.requeue_requested = True
        atomic_json(
            self.state_dir / "REQUEUE_REQUESTED.json",
            {"schema_version": 1, "status": "checkpoint_requested", "signal": signal.Signals(signum).name, "unix_time": time.time()},
        )
        self._forward(signal.SIGUSR1)

    def request_cancel(self, signum: int, _frame: object) -> None:
        calling = Path(
            os.environ.get(
                "TREEWM_REQUEUE_CALLING_MARKER",
                str(self.state_dir / "REQUEUE_CALLING.json"),
            )
        )
        if calling.exists():
            return
        self.cancel_requested = True
        atomic_json(
            self.cancel_latch,
            {"schema_version": 1, "status": "cancel_requested", "signal": signal.Signals(signum).name, "unix_time": time.time()},
        )
        self._forward(signal.SIGTERM)


def _rank_state_complete(payload: Mapping[str, Any]) -> bool:
    identity = payload.get("run_identity") or {}
    world_size = int(identity.get("world_size", -1))
    rank_states = payload.get("rank_states") or []
    if world_size != 1 or len(rank_states) != 1:
        return False
    state = rank_states[0] or {}
    loader = state.get("loader") or {}
    streams = state.get("rng_streams") or {}
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
    import torch

    if not path.is_file() or path.is_symlink():
        raise ContractError(f"latest checkpoint is unavailable or symlinked: {path}")
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
    checks = (
        payload.get("schema_version") == 2,
        0 <= completed <= 1_000_000,
        expected_step is None or completed == expected_step,
        int(payload.get("step", -1)) == completed,
        int(payload.get("next_step", -1)) == completed,
        payload.get("optimizer") is not None,
        payload.get("scheduler") is not None,
        _rank_state_complete(payload),
        identity.get("run_name") == run["run_name"],
        identity.get("setting") == run["setting_id"],
        int(identity.get("seed", -1)) == int(run["seed"]),
        identity.get("objective_version") == "treewm_v2_grounded_repair_formal_v1",
        identity.get("protocol_sha256") == hashes["run_protocol_sha256"],
        identity.get("code_sha256") == hashes["source_sha256"],
        identity.get("runtime_sha256") == hashes["runtime_sha256"],
        identity.get("campaign_source_sha256") == hashes["source_sha256"],
        identity.get("campaign_protocol_sha256") == hashes["package_protocol_sha256"],
        identity.get("campaign_config_sha256") == hashes["config_sha256"],
        identity.get("campaign_input_contract_sha256") == hashes["input_contract_sha256"],
        identity.get("campaign_prerequisite_binding_sha256")
        == hashes["prerequisite_binding_sha256"],
        identity.get("campaign_selected_recipe_sha256")
        == hashes["selected_recipe_sha256"],
        identity.get("campaign_factorial_arm") == f"exp16-{hashes['selected_arm']}",
        identity.get("calibration_sha256") == hashes["calibration_sha256"],
        identity.get("future_recipe_sha256") == hashes["future_recipe_sha256"],
        identity.get("evaluation_seed_protocol_sha256") == hashes["evaluation_seed_protocol_sha256"],
        identity.get("final_seed_table_sha256") == hashes["final_seed_table_sha256"],
        identity.get("monitor_seed_table_sha256") == hashes["monitor_seed_table_sha256"],
        config.get("campaign_source_sha256") == hashes["source_sha256"],
        config.get("campaign_protocol_sha256") == hashes["package_protocol_sha256"],
        config.get("campaign_config_sha256") == hashes["config_sha256"],
        config.get("campaign_prerequisite_binding_sha256")
        == hashes["prerequisite_binding_sha256"],
        config.get("campaign_selected_recipe_sha256")
        == hashes["selected_recipe_sha256"],
    )
    if not all(checks):
        raise ContractError("checkpoint exact-resume identity is incomplete or differs")
    return {
        "schema_version": 1,
        "status": "checkpoint_verified",
        "completed_updates": completed,
        "reason": payload.get("reason"),
        "identity_sha256": payload.get("identity_sha256"),
        "evaluation_seed_tables_sha256": identity.get("evaluation_seed_tables_sha256"),
        "final_seed_table_sha256": identity.get("final_seed_table_sha256"),
        "monitor_seed_table_sha256": identity.get("monitor_seed_table_sha256"),
        "checkpoint": str(path),
        "checkpoint_sha256": file_sha256(path),
        "launch_sha256": launch["launch_sha256"],
        "unix_time": time.time(),
    }


def verify_stage_marker(run_dir: Path, target: int, launch: Mapping[str, Any]) -> dict[str, Any]:
    path = run_dir / "stage-gates" / f"AWAITING_GATE_{target}.json"
    marker = read_json(path)
    checkpoint = verify_checkpoint(run_dir / "checkpoints" / "latest.pt", launch, expected_step=target)
    require(marker.get("schema_version") == 1, "trainer stage-marker schema differs")
    require(marker.get("status") == "awaiting_external_stage_gate", "trainer did not await gate")
    require(marker.get("objective_version") == "treewm_v2_grounded_repair_formal_v1", "marker objective differs")
    require(int(marker.get("completed_updates", -1)) == target == int(marker.get("step", -1)), "marker step differs")
    require(int(marker.get("total_steps", -1)) == 1_000_000, "marker scientific horizon differs")
    require(int(marker.get("scheduler_total_steps", -1)) == 1_000_000, "marker scheduler horizon differs")
    require(marker.get("identity_sha256") == checkpoint["identity_sha256"], "marker identity differs")
    require(marker.get("checkpoint") == "checkpoints/latest.pt", "marker checkpoint path differs")
    require(marker.get("checkpoint_sha256") == checkpoint["checkpoint_sha256"], "marker checkpoint hash differs")
    require(marker.get("evaluation_seed_tables_sha256") == checkpoint["evaluation_seed_tables_sha256"], "marker seed table differs")
    require(not (run_dir / "COMPLETED.json").exists(), "stage lifecycle illegally ran terminal evaluation")
    return {"marker": str(path), **checkpoint}


def _previous_target(target: int) -> int | None:
    position = STAGE_TARGETS.index(target)
    return STAGE_TARGETS[position - 1] if position else None


def verify_previous_gate(manifest: Mapping[str, Any], target: int, launch: Mapping[str, Any]) -> dict[str, Any] | None:
    previous = _previous_target(target)
    if previous is None:
        return None
    path = Path(manifest["paths"]["run_root"]) / "state" / "stage-gates" / f"STAGE_GATE_{previous}.json"
    gate = read_json(path)
    claimed = gate.get("gate_sha256")
    body = dict(gate)
    body.pop("gate_sha256", None)
    require(claimed == stable_hash(body), "previous stage gate content hash differs")
    require(gate.get("status") == "accepted" and int(gate.get("stage_target", -1)) == previous, "previous stage was not accepted")
    require(gate.get("package_protocol_sha256") == launch["hashes"]["package_protocol_sha256"], "previous gate protocol differs")
    require(gate.get("evaluation_source_sha256") == launch["hashes"]["evaluation_source_sha256"], "previous gate evaluation source differs")
    require(
        gate.get("prerequisite_binding_sha256")
        == launch["hashes"]["prerequisite_binding_sha256"],
        "previous gate prerequisite binding differs",
    )
    require(
        gate.get("selected_recipe_sha256")
        == launch["hashes"]["selected_recipe_sha256"],
        "previous gate selected recipe differs",
    )
    runs = gate.get("runs") or []
    match = next((row for row in runs if int(row.get("index", -1)) == int(launch["run"]["index"])), None)
    require(len(runs) == 40 and match is not None, "previous gate lacks exact 40-run coverage")
    require(match.get("launch_sha256") == launch["launch_sha256"], "previous gate launch differs")
    return {
        "stage_target": previous,
        "gate_sha256": gate["gate_sha256"],
        "identity_sha256": match["identity_sha256"],
        "checkpoint_sha256": match["checkpoint_sha256"],
    }


def seal_or_validate_stage_launch(
    path: Path,
    value: Mapping[str, Any],
    *,
    validate_first_boundary,
) -> bool:
    """Bind the prior boundary once; requeues may legitimately advance beyond it."""
    if path.exists():
        require(read_json(path) == dict(value), "stage launch changed across requeue")
        return False
    validate_first_boundary()
    atomic_json(path, value)
    return True


def run_worker(args: argparse.Namespace) -> int:
    require(args.stage_target in STAGE_TARGETS, "invalid lifecycle stage target")
    root = args.repo_root.resolve()
    manifest = load_manifest(args.manifest)
    run = run_at(manifest, args.index)
    launch = trainer_command(manifest, run, repo_root=root)
    for env_name, hash_name in (
        ("TREEWM_EXPECTED_SOURCE_SHA256", "source_sha256"),
        ("TREEWM_EXPECTED_EVALUATION_SOURCE_SHA256", "evaluation_source_sha256"),
        ("TREEWM_EXPECTED_RUNTIME_SHA256", "runtime_sha256"),
        ("TREEWM_EXPECTED_PACKAGE_PROTOCOL_SHA256", "package_protocol_sha256"),
        ("TREEWM_EXPECTED_PREREQUISITE_BINDING_SHA256", "prerequisite_binding_sha256"),
        ("TREEWM_EXPECTED_SELECTED_RECIPE_SHA256", "selected_recipe_sha256"),
    ):
        expected = os.environ.get(env_name)
        require(expected is not None and expected == launch["hashes"][hash_name], f"{env_name} missing/differs")
    expected_stage = os.environ.get("TREEWM_EXPECTED_STAGE_TARGET")
    require(expected_stage is not None and int(expected_stage) == args.stage_target, "sealed stage target differs")
    previous_gate = verify_previous_gate(manifest, args.stage_target, launch)

    state_dir = args.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    state = SignalState(state_dir)
    signal.signal(signal.SIGUSR1, state.request_requeue)
    signal.signal(signal.SIGTERM, state.request_cancel)
    signal.signal(signal.SIGINT, state.request_cancel)
    if state.cancel_latch.exists():
        return CANCEL_EXIT_CODE

    run_dir = Path(launch["run"]["run_directory"])
    persistent_launch = run_dir / "FORMAL_LAUNCH.json"
    if args.stage_target == STAGE_TARGETS[0] and run_dir.exists() and not persistent_launch.exists():
        raise ContractError("fresh first stage refuses a pre-existing unowned run directory")
    run_dir.mkdir(parents=True, exist_ok=True)
    prior_launch = read_json(persistent_launch) if persistent_launch.exists() else None
    require(prior_launch in (None, launch), "run directory belongs to another launch")
    if prior_launch is None:
        atomic_json(persistent_launch, launch)
    stage_launch = {
        "schema_version": 1,
        "status": "stage_launch_sealed",
        "stage_target": args.stage_target,
        "launch_sha256": launch["launch_sha256"],
        "package_protocol_sha256": launch["hashes"]["package_protocol_sha256"],
        "previous_gate": previous_gate,
    }
    stage_path = run_dir / "stage-gates" / f"STAGE_LAUNCH_{args.stage_target}.json"
    def validate_first_boundary() -> None:
        if previous_gate is not None:
            checkpoint = verify_checkpoint(
                run_dir / "checkpoints" / "latest.pt",
                launch,
                expected_step=int(previous_gate["stage_target"]),
            )
            require(previous_gate["identity_sha256"] == checkpoint["identity_sha256"], "previous gate checkpoint identity differs")
            require(previous_gate["checkpoint_sha256"] == checkpoint["checkpoint_sha256"], "previous gate checkpoint bytes differ")
    seal_or_validate_stage_launch(
        stage_path,
        stage_launch,
        validate_first_boundary=validate_first_boundary,
    )
    atomic_json(state_dir / "launch.json", {**stage_launch, "launch": launch})

    complete_path = run_dir / "stage-gates" / f"STAGE_COMPLETE_{args.stage_target}.json"
    if complete_path.exists():
        record = verify_stage_marker(run_dir, args.stage_target, launch)
        require(read_json(complete_path).get("checkpoint_sha256") == record["checkpoint_sha256"], "existing stage completion differs")
        atomic_json(state_dir / "WORKER_COMPLETE.json", read_json(complete_path))
        return 0

    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in launch["environment"].items()})
    environment["TREEWM_STOP_AFTER_UPDATE"] = str(args.stage_target)
    state.child = subprocess.Popen(launch["argv"], cwd=root, env=environment)
    while state.child.poll() is None:
        if state.cancel_latch.exists() and not state.cancel_requested:
            state.cancel_requested = True
            state._forward(signal.SIGTERM)
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
        record = verify_stage_marker(run_dir, args.stage_target, launch)
        complete = {
            "schema_version": 1,
            "status": "stage_complete_awaiting_campaign_gate",
            "campaign_id": manifest["campaign_id"],
            "index": run.index,
            "setting_id": run.setting_id,
            "seed": run.seed,
            "stage_target": args.stage_target,
            "launch_sha256": launch["launch_sha256"],
            "package_protocol_sha256": launch["hashes"]["package_protocol_sha256"],
            "source_sha256": launch["hashes"]["source_sha256"],
            "evaluation_source_sha256": launch["hashes"]["evaluation_source_sha256"],
            "runtime_sha256": launch["hashes"]["runtime_sha256"],
            "prerequisite_binding_sha256": launch["hashes"]["prerequisite_binding_sha256"],
            "selected_recipe_sha256": launch["hashes"]["selected_recipe_sha256"],
            "selected_arm": launch["hashes"]["selected_arm"],
            "identity_sha256": record["identity_sha256"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "evaluation_seed_tables_sha256": record["evaluation_seed_tables_sha256"],
            "final_seed_table_sha256": record["final_seed_table_sha256"],
            "monitor_seed_table_sha256": record["monitor_seed_table_sha256"],
            "completed_unix_time": time.time(),
        }
        complete["stage_complete_sha256"] = stable_hash(complete)
        atomic_json(complete_path, complete)
        atomic_json(state_dir / "WORKER_COMPLETE.json", complete)
        return 0
    if disposition == "cancelled":
        atomic_json(state_dir / "CANCELLED.json", {"schema_version": 1, "status": "cancelled_without_requeue", "child_exit_code": status, "unix_time": time.time()})
        return CANCEL_EXIT_CODE
    if disposition == "requeue":
        record = verify_checkpoint(run_dir / "checkpoints" / "latest.pt", launch)
        require(int(record["completed_updates"]) < args.stage_target, "preemption checkpoint reached/passed stage without marker")
        require(str(record.get("reason", "")).startswith("graceful-stop:"), "exit 75 is not a graceful-stop checkpoint")
        atomic_json(state_dir / "READY_FOR_REQUEUE.json", record)
        return GRACEFUL_EXIT_CODE
    atomic_json(state_dir / "FAILED.json", {"schema_version": 1, "status": "unexpected_child_failure", "child_exit_code": status, "unix_time": time.time()})
    return status or 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--stage-target", type=int, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--manifest", type=Path, default=CAMPAIGN_DIR / "manifest.json")
    return parser


if __name__ == "__main__":
    try:
        raise SystemExit(run_worker(_parser().parse_args()))
    except ContractError as exc:
        print(f"grounded-formal worker error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
