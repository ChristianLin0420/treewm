#!/usr/bin/env python3
"""Run one repair-pilot model with durable cancellation and exact requeue checks."""

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
    actual_evaluation_bank,
    atomic_json,
    inference_profile,
    load_manifest,
    run_at,
    trainer_command,
)


GRACEFUL_EXIT_CODE = 75
CANCEL_EXIT_CODE = 143


def read_optional_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"existing identity artifact is unreadable: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"existing identity artifact is not a JSON object: {path}")
    return value


class SignalState:
    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir
        self.requeue_requested = False
        self.cancel_requested = False
        self.child: subprocess.Popen | None = None

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
        atomic_json(self.state_dir / "REQUEUE_REQUESTED.json", {
            "status": "checkpoint_requested",
            "signal": signal.Signals(signum).name,
            "unix_time": time.time(),
        })
        self._forward(signal.SIGUSR1)

    def request_cancel(self, signum: int, _frame: object) -> None:
        if any(self.state_dir.glob("requeue/*/REQUEUE_CALLING.json")):
            return
        self.cancel_requested = True
        atomic_json(self.cancel_latch, {
            "status": "cancel_requested",
            "signal": signal.Signals(signum).name,
            "unix_time": time.time(),
        })
        self._forward(signal.SIGTERM)


def verify_checkpoint(path: Path, launch: Mapping[str, Any]) -> dict[str, Any]:
    """Prove an exit-75 checkpoint is atomic, exact-resumable, and identity-complete."""
    import torch

    if not path.is_file() or path.is_symlink():
        raise ContractError(f"graceful exit has no regular checkpoint: {path}")
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ContractError(f"checkpoint cannot be loaded: {exc}") from exc
    completed = int(payload.get("completed_updates", -1))
    identity = payload.get("run_identity") or {}
    config = payload.get("config") or {}
    expected = launch["run"]
    hashes = launch["hashes"]
    rank_states = payload.get("rank_states") or []
    rank_zero = next((state for state in rank_states if int(state.get("rank", -1)) == 0), None)
    loader = (rank_zero or {}).get("loader") or {}
    rng_streams = (rank_zero or {}).get("rng_streams") or {}
    exact_resume_state = bool(
        len(rank_states) == 1
        and rank_zero is not None
        and (rank_zero.get("rng_state") or {})
        and set(rng_streams) >= {"planner", "eval", "viz"}
        and rank_zero.get("horizon_generator") is not None
        and set(loader) >= {"epoch", "batches_yielded_in_epoch", "epoch_generator_state"}
        and loader.get("epoch_generator_state") is not None
        and set(payload.get("checkpoint_manager") or {}) >= {"best_success", "best_val_loss"}
    )
    checks = (
        payload.get("schema_version") == 2,
        0 <= completed <= 25_000,
        int(payload.get("step", -1)) == completed,
        int(payload.get("next_step", -1)) == completed,
        payload.get("optimizer") is not None,
        payload.get("scheduler") is not None,
        exact_resume_state,
        identity.get("run_name") == expected["run_name"],
        identity.get("setting") == expected["setting_id"],
        int(identity.get("seed", -1)) == int(expected["seed"]),
        identity.get("objective_version") == "treewm_v2_grounded_repair_pilot_v1",
        int(identity.get("total_steps", -1)) == 25_000,
        int(identity.get("scheduler_total_steps", -1)) == 1_000_000,
        identity.get("code_sha256") == hashes["source_sha256"],
        identity.get("runtime_sha256") == hashes["runtime_sha256"],
        identity.get("recipe_code_sha256") == hashes["recipe_code_sha256"],
        identity.get("recipe_runtime_sha256") == hashes["recipe_runtime_sha256"],
        identity.get("campaign_source_sha256") == hashes["source_sha256"],
        identity.get("campaign_protocol_sha256") == hashes["package_protocol_sha256"],
        identity.get("campaign_config_sha256") == hashes["config_sha256"],
        identity.get("campaign_input_contract_sha256") == hashes["input_contract_sha256"],
        identity.get("campaign_factorial_arm") == expected["arm_id"],
        identity.get("data_manifest_sha256") == hashes["data_manifest_sha256"],
        identity.get("calibration_sha256") == hashes["calibration_sha256"],
        identity.get("future_recipe_sha256") == hashes["future_recipe_sha256"],
        identity.get("evaluation_seed_tables_sha256") == hashes["evaluation_seed_tables_sha256"],
        identity.get("final_seed_table_sha256") == hashes["final_seed_table_sha256"],
        config.get("campaign_source_sha256") == hashes["source_sha256"],
        config.get("campaign_protocol_sha256") == hashes["package_protocol_sha256"],
        config.get("campaign_config_sha256") == hashes["config_sha256"],
    )
    if not all(checks):
        raise ContractError("checkpoint exact-resume identity is incomplete or drifted")
    stat = path.stat()
    return {
        "status": "checkpoint_verified",
        "completed_updates": completed,
        "reason": payload.get("reason"),
        "identity_sha256": payload.get("identity_sha256"),
        "checkpoint": str(path),
        "checkpoint_size": stat.st_size,
        "checkpoint_mtime_ns": stat.st_mtime_ns,
        "launch_sha256": launch["launch_sha256"],
        "unix_time": time.time(),
    }


def verify_completion(
    path: Path,
    launch: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    payload = read_optional_json(path)
    if payload is None:
        raise ContractError(f"trainer returned success without completion sentinel: {path}")
    identity = payload.get("run_identity") or {}
    metrics = payload.get("final_evaluation") or {}
    expected = launch["run"]
    hashes = launch["hashes"]
    scientific = manifest["scientific_contract"]
    profile = inference_profile(manifest)
    run_dir = path.parent
    progress = read_optional_json(run_dir / str(payload.get("final_eval_progress", "")))
    bank = actual_evaluation_bank(manifest)
    expected_episode_rows = [
        (task_index, task_id, episode_index, seed_value)
        for task_index, (task_id, seeds) in enumerate(zip(bank["task_ids"], bank["seeds"], strict=True))
        for episode_index, seed_value in enumerate(seeds)
    ]
    progress_rows = [
        (
            row.get("task_index"),
            row.get("task_id"),
            row.get("episode_index"),
            row.get("episode_seed"),
        )
        for row in (progress or {}).get("completed_results", [])
    ]
    checks = (
        payload.get("schema_version") == 1,
        payload.get("status") == "complete",
        payload.get("objective_version") == "treewm_v2_grounded_repair_pilot_v1",
        int(payload.get("completed_updates", -1)) == 25_000,
        int(payload.get("scheduler_total_steps", -1)) == 1_000_000,
        int(payload.get("final_eval_step", -1)) == 25_000,
        identity.get("run_name") == expected["run_name"],
        identity.get("setting") == expected["setting_id"],
        int(identity.get("seed", -1)) == int(expected["seed"]),
        identity.get("campaign_factorial_arm") == expected["arm_id"],
        identity.get("code_sha256") == hashes["source_sha256"],
        identity.get("runtime_sha256") == hashes["runtime_sha256"],
        identity.get("recipe_code_sha256") == hashes["recipe_code_sha256"],
        identity.get("recipe_runtime_sha256") == hashes["recipe_runtime_sha256"],
        identity.get("campaign_source_sha256") == hashes["source_sha256"],
        identity.get("campaign_protocol_sha256") == hashes["package_protocol_sha256"],
        identity.get("campaign_config_sha256") == hashes["config_sha256"],
        identity.get("campaign_input_contract_sha256") == hashes["input_contract_sha256"],
        identity.get("data_manifest_sha256") == hashes["data_manifest_sha256"],
        identity.get("calibration_sha256") == hashes["calibration_sha256"],
        identity.get("future_recipe_sha256") == hashes["future_recipe_sha256"],
        identity.get("evaluation_seed_tables_sha256") == hashes["evaluation_seed_tables_sha256"],
        identity.get("final_seed_table_sha256") == hashes["final_seed_table_sha256"],
        payload.get("protocol_sha256") == hashes["run_protocol_sha256"],
        payload.get("code_sha256") == hashes["source_sha256"],
        payload.get("runtime_sha256") == hashes["runtime_sha256"],
        payload.get("data_manifest_sha256") == hashes["data_manifest_sha256"],
        payload.get("calibration_sha256") == hashes["calibration_sha256"],
        payload.get("future_recipe_sha256") == hashes["future_recipe_sha256"],
        payload.get("evaluation_seed_tables_sha256") == hashes["evaluation_seed_tables_sha256"],
        payload.get("final_seed_table_sha256") == hashes["final_seed_table_sha256"],
        payload.get("scorer") == profile["scorer"],
        payload.get("task_ids") == list(scientific.get("task_ids", [1, 2, 3, 4, 5])),
        int(payload.get("episodes_per_task", -1)) == 5,
        int(metrics.get("eval/num_episodes", -1)) == 25,
        all(int(metrics.get(f"eval/task{task}/num_episodes", -1)) == 5 for task in range(1, 6)),
        progress is not None,
        (progress or {}).get("status") == "complete",
        (progress or {}).get("identity_sha256") == payload.get("identity_sha256"),
        (progress or {}).get("seed_table_sha256") == hashes["final_seed_table_sha256"],
        progress_rows == expected_episode_rows,
        (progress or {}).get("metrics") == metrics,
    )
    if not all(checks):
        raise ContractError("completion sentinel, final evaluation, or identity is incomplete/drifted")
    return {
        "status": "worker_complete",
        "completion": str(path),
        "identity_sha256": payload.get("identity_sha256"),
        "actual_evaluation_bank_sha256": bank["sha256"],
        "launch_sha256": launch["launch_sha256"],
        "unix_time": time.time(),
    }


def run_worker(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    run = run_at(manifest, args.index)
    state_dir = args.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    state = SignalState(state_dir)
    signal.signal(signal.SIGUSR1, state.request_requeue)
    signal.signal(signal.SIGTERM, state.request_cancel)
    signal.signal(signal.SIGINT, state.request_cancel)
    if state.cancel_latch.exists():
        return CANCEL_EXIT_CODE

    launch = trainer_command(manifest, run, repo_root=args.repo_root)
    for env_name, hash_name in (
        ("TREEWM_EXPECTED_SOURCE_SHA256", "source_sha256"),
        ("TREEWM_EXPECTED_RUNTIME_SHA256", "runtime_sha256"),
        ("TREEWM_EXPECTED_PACKAGE_PROTOCOL_SHA256", "package_protocol_sha256"),
    ):
        expected_hash = os.environ.get(env_name)
        if expected_hash and launch["hashes"][hash_name] != expected_hash:
            raise ContractError(f"{env_name} differs after the array launch was sealed")

    launch_path = state_dir / "launch.json"
    existing = read_optional_json(launch_path)
    if existing is not None and existing != launch:
        raise ContractError("source/config/protocol changed across requeue")
    atomic_json(launch_path, launch)
    run_dir = Path(launch["run"]["run_directory"])
    run_dir.mkdir(parents=True, exist_ok=True)
    persistent_launch = run_dir / "PILOT_LAUNCH.json"
    prior = read_optional_json(persistent_launch)
    if prior is not None and prior != launch:
        raise ContractError("run directory belongs to a different launch contract")
    atomic_json(persistent_launch, launch)

    completion_path = run_dir / "COMPLETED.json"
    if completion_path.exists():
        atomic_json(state_dir / "WORKER_COMPLETE.json", verify_completion(completion_path, launch, manifest))
        return 0

    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in launch["environment"].items()})
    state.child = subprocess.Popen(launch["argv"], cwd=args.repo_root, env=environment)
    while state.child.poll() is None:
        if state.cancel_latch.exists() and not state.cancel_requested:
            state.cancel_requested = True
            state._forward(signal.SIGTERM)
        time.sleep(0.5)
    status = int(state.child.returncode)
    state.child = None

    if state.cancel_requested or state.cancel_latch.exists():
        atomic_json(state_dir / "CANCELLED.json", {
            "status": "cancelled_without_requeue",
            "child_exit_code": status,
            "unix_time": time.time(),
        })
        return CANCEL_EXIT_CODE
    if status == GRACEFUL_EXIT_CODE or state.requeue_requested:
        record = verify_checkpoint(run_dir / "checkpoints" / "latest.pt", launch)
        if not str(record.get("reason", "")).startswith("graceful-stop:"):
            raise ContractError("exit-75 checkpoint is not marked as a graceful stop")
        atomic_json(state_dir / "READY_FOR_REQUEUE.json", record)
        return GRACEFUL_EXIT_CODE
    if status == 0:
        atomic_json(state_dir / "WORKER_COMPLETE.json", verify_completion(completion_path, launch, manifest))
        return 0
    atomic_json(state_dir / "FAILED.json", {
        "status": "unexpected_child_failure",
        "child_exit_code": status,
        "unix_time": time.time(),
    })
    return status or 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--manifest", type=Path, default=CAMPAIGN_DIR / "manifest.json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    return run_worker(_parser().parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"repair-pilot worker error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
