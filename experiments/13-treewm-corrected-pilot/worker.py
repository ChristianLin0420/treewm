#!/usr/bin/env python3
"""Run one corrected-pilot model with durable cancel and exact-resume checks."""

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
    atomic_json,
    load_manifest,
    run_at,
    trainer_command,
)


GRACEFUL_EXIT_CODE = 75
CANCEL_EXIT_CODE = 143


def read_json(path: Path) -> dict[str, Any] | None:
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
        child = self.child
        if child is not None and child.poll() is None:
            try:
                child.send_signal(signum)
            except ProcessLookupError:
                pass

    def request_requeue(self, signum: int, _frame: object) -> None:
        if self.cancel_requested or self.cancel_latch.exists():
            return
        self.requeue_requested = True
        atomic_json(
            self.state_dir / "REQUEUE_REQUESTED.json",
            {"status": "checkpoint_requested", "signal": signal.Signals(signum).name, "unix_time": time.time()},
        )
        self._forward(signal.SIGUSR1)

    def request_cancel(self, signum: int, _frame: object) -> None:
        # Slurm tears down the old allocation with TERM after an exact scontrol requeue.
        # Once that call is durably marked, it must not be reinterpreted as user cancel.
        if (self.state_dir / "REQUEUE_CALLING.json").exists():
            return
        self.cancel_requested = True
        atomic_json(
            self.cancel_latch,
            {"status": "cancel_requested", "signal": signal.Signals(signum).name, "unix_time": time.time()},
        )
        self._forward(signal.SIGTERM)


def verify_checkpoint(path: Path, launch: Mapping[str, Any]) -> dict[str, Any]:
    """Prove the trainer published an atomic, identity-complete exact-resume point."""
    import torch

    if not path.is_file() or path.is_symlink():
        raise ContractError(f"graceful exit has no regular checkpoint: {path}")
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:  # older PyTorch without mmap keyword
            payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ContractError(f"checkpoint cannot be loaded: {exc}") from exc
    completed = int(payload.get("completed_updates", -1))
    identity = payload.get("run_identity") or {}
    config = payload.get("config") or {}
    expected = launch["run"]
    hashes = launch["hashes"]
    rank_states = payload.get("rank_states") or []
    rank_zero = next(
        (state for state in rank_states if int(state.get("rank", -1)) == 0), None
    )
    loader = (rank_zero or {}).get("loader") or {}
    rng_streams = (rank_zero or {}).get("rng_streams") or {}
    exact_resume_state = bool(
        len(rank_states) == 1
        and rank_zero is not None
        and (rank_zero.get("rng_state") or {})
        and set(rng_streams) >= {"planner", "eval", "viz"}
        and rank_zero.get("horizon_generator") is not None
        and set(loader) >= {
            "epoch",
            "batches_yielded_in_epoch",
            "epoch_generator_state",
        }
        and loader.get("epoch_generator_state") is not None
        and set(payload.get("checkpoint_manager") or {})
        >= {"best_success", "best_val_loss"}
    )
    required = (
        payload.get("schema_version") == 2,
        0 <= completed <= 12_000,
        int(payload.get("step", -1)) == completed,
        int(payload.get("next_step", -1)) == completed,
        payload.get("optimizer") is not None,
        payload.get("scheduler") is not None,
        exact_resume_state,
        identity.get("run_name") == expected["run_name"],
        identity.get("setting") == expected["setting_id"],
        int(identity.get("seed", -1)) == int(expected["seed"]),
        identity.get("objective_version") == "treewm_v2_grounded_pilot_v1",
        identity.get("code_sha256") == hashes["source_sha256"],
        identity.get("runtime_sha256") == hashes["runtime_sha256"],
        identity.get("recipe_code_sha256") == hashes["compatible_recipe_code_sha256"],
        identity.get("recipe_runtime_sha256") == hashes["compatible_recipe_runtime_sha256"],
        identity.get("campaign_source_sha256") == hashes["source_sha256"],
        identity.get("campaign_protocol_sha256") == hashes["package_protocol_sha256"],
        identity.get("campaign_config_sha256") == hashes["config_sha256"],
        identity.get("campaign_input_contract_sha256")
        == hashes["compatible_input_contract_sha256"],
        identity.get("campaign_factorial_arm") == expected["arm_id"],
        identity.get("calibration_sha256") == hashes["calibration_sha256"],
        identity.get("future_recipe_sha256") == hashes["future_recipe_sha256"],
        config.get("campaign_config_sha256") == hashes["config_sha256"],
        config.get("campaign_source_sha256") == hashes["source_sha256"],
        config.get("campaign_protocol_sha256") == hashes["package_protocol_sha256"],
    )
    if not all(required):
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


def verify_completion(path: Path, launch: Mapping[str, Any]) -> dict[str, Any]:
    payload = read_json(path)
    if payload is None:
        raise ContractError(f"trainer returned success without completion sentinel: {path}")
    identity = payload.get("run_identity") or {}
    metrics = payload.get("final_evaluation") or {}
    expected = launch["run"]
    hashes = launch["hashes"]
    if not (
        payload.get("schema_version") == 1
        and payload.get("status") == "complete"
        and int(payload.get("completed_updates", -1)) == 12_000
        and int(payload.get("final_eval_step", -1)) == 12_000
        and payload.get("objective_version") == "treewm_v2_grounded_pilot_v1"
        and payload.get("run_name", identity.get("run_name")) == expected["run_name"]
        and identity.get("setting") == expected["setting_id"]
        and int(identity.get("seed", -1)) == int(expected["seed"])
        and identity.get("code_sha256") == hashes["source_sha256"]
        and identity.get("runtime_sha256") == hashes["runtime_sha256"]
        and identity.get("recipe_code_sha256") == hashes["compatible_recipe_code_sha256"]
        and identity.get("recipe_runtime_sha256") == hashes["compatible_recipe_runtime_sha256"]
        and identity.get("campaign_source_sha256") == hashes["source_sha256"]
        and identity.get("campaign_protocol_sha256") == hashes["package_protocol_sha256"]
        and identity.get("campaign_config_sha256") == hashes["config_sha256"]
        and identity.get("campaign_input_contract_sha256")
        == hashes["compatible_input_contract_sha256"]
        and identity.get("campaign_factorial_arm") == expected["arm_id"]
        and payload.get("calibration_sha256") == hashes["calibration_sha256"]
        and payload.get("future_recipe_sha256") == hashes["future_recipe_sha256"]
        and int(metrics.get("eval/num_episodes", -1)) == 5
        and all(int(metrics.get(f"eval/task{task}/num_episodes", -1)) == 1 for task in range(1, 6))
    ):
        raise ContractError("completion sentinel is incomplete or does not match launch identity")
    return {
        "status": "worker_complete",
        "completion": str(path),
        "identity_sha256": payload.get("identity_sha256"),
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
    expected_source = os.environ.get("TREEWM_EXPECTED_SOURCE_SHA256")
    if expected_source and launch["hashes"]["source_sha256"] != expected_source:
        raise ContractError("trainer source changed after the array launch was sealed")
    expected_runtime = os.environ.get("TREEWM_EXPECTED_RUNTIME_SHA256")
    if expected_runtime and launch["hashes"]["runtime_sha256"] != expected_runtime:
        raise ContractError("trainer runtime changed after the array launch was sealed")
    expected_protocol = os.environ.get("TREEWM_EXPECTED_PACKAGE_PROTOCOL_SHA256")
    if (
        expected_protocol
        and launch["hashes"]["package_protocol_sha256"] != expected_protocol
    ):
        raise ContractError("pilot package changed after the array launch was sealed")
    launch_path = state_dir / "launch.json"
    existing = read_json(launch_path)
    if existing is not None and existing != launch:
        raise ContractError("source/config/protocol changed across requeue")
    atomic_json(launch_path, launch)
    run_dir = Path(launch["run"]["run_directory"])
    run_dir.mkdir(parents=True, exist_ok=True)
    persistent_launch = run_dir / "PILOT_LAUNCH.json"
    prior = read_json(persistent_launch)
    if prior is not None and prior != launch:
        raise ContractError("run directory belongs to a different launch contract")
    atomic_json(persistent_launch, launch)

    completion_path = run_dir / "COMPLETED.json"
    if completion_path.exists():
        atomic_json(state_dir / "WORKER_COMPLETE.json", verify_completion(completion_path, launch))
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
        atomic_json(
            state_dir / "CANCELLED.json",
            {"status": "cancelled_without_requeue", "child_exit_code": status, "unix_time": time.time()},
        )
        return CANCEL_EXIT_CODE
    if status == GRACEFUL_EXIT_CODE or state.requeue_requested:
        record = verify_checkpoint(run_dir / "checkpoints" / "latest.pt", launch)
        if not str(record.get("reason", "")).startswith("graceful-stop:"):
            raise ContractError("exit 75 checkpoint is not marked as a graceful stop")
        atomic_json(state_dir / "READY_FOR_REQUEUE.json", record)
        return GRACEFUL_EXIT_CODE
    if status == 0:
        atomic_json(state_dir / "WORKER_COMPLETE.json", verify_completion(completion_path, launch))
        return 0
    atomic_json(
        state_dir / "FAILED.json",
        {"status": "unexpected_child_failure", "child_exit_code": status, "unix_time": time.time()},
    )
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
        print(f"corrected-pilot worker error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
