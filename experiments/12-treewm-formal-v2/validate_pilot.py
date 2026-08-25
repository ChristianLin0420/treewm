#!/usr/bin/env python3
"""Run and fail-closed validate the ten-setting, seed-zero TreeWM-v2 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from campaign import (
    expand_runs,
    load_data_contract,
    load_manifest,
    protocol_sha256,
    run_directory,
    trainer_command,
)


GRACEFUL_EXIT_CODE = 75
MAX_SHARED_GRADIENT_SHARE = 0.80
AUDIT_STEP = 5000
MAX_MEAN_DATA_WAIT_FRACTION = 0.50
WARN_MEAN_DATA_WAIT_FRACTION = 0.35
SHARED_AUDIT_MODULES = (
    "encoder",
    "branch_transformer",
    "dynamics",
    "controllability",
)
REQUIRED_AUDIT_MODULES = (
    *SHARED_AUDIT_MODULES,
    "contextual_gain",
)
FORMAL_ACTIVE_TERMS = frozenset(
    {
        "state", "action", "horizon", "bind", "coverage", "redundancy",
        "keep", "uncertainty", "recursive", "reconstruction", "control", "expand",
    }
)
NONDEGENERACY_THRESHOLDS: dict[str, dict[str, float | str]] = {
    "control/q_pair_distance_mean": {"operator": ">", "threshold": 0.05},
    "control/q_near_collapse_fraction": {"operator": "<", "threshold": 0.95},
    "expansion/predicted_gain_std": {"operator": ">", "threshold": 1e-4},
    "expansion/target_gain_std": {"operator": ">", "threshold": 1e-4},
    "tree/effective_branching_factor": {"operator": ">", "threshold": 0.5},
    "tree/effective_branching_factor:max": {"operator": "<=", "threshold": 4.0},
    "tree/support_recall": {"operator": ">", "threshold": 0.1},
    "expansion/nodes_generated": {"operator": ">=", "threshold": 32.0},
    "expansion/budget_shortfall": {"operator": ">=", "threshold": 0.0},
    "expansion/budget_shortfall:max": {"operator": "<=", "threshold": 32.0},
}


class PilotError(RuntimeError):
    pass


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def seed_zero_run(manifest: Mapping[str, Any], setting_index: int):
    settings = manifest["settings"]
    if not 0 <= setting_index < len(settings):
        raise PilotError("pilot setting index is out of range")
    setting_id = settings[setting_index]["id"]
    candidates = [
        run for run in expand_runs(manifest)
        if run.setting_id == setting_id and int(run.seed) == 0
    ]
    if len(candidates) != 1:
        raise PilotError(f"expected one seed-zero run for {setting_id}, got {len(candidates)}")
    return candidates[0]


def replace_override(command: list[str], name: str, value: object) -> None:
    prefix = f"{name}="
    matches = [index for index, token in enumerate(command) if token.startswith(prefix)]
    if len(matches) != 1:
        raise PilotError(f"expected exactly one {name} override, got {len(matches)}")
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    else:
        rendered = str(value)
    command[matches[0]] = f"{name}={rendered}"


def pilot_launch_spec(
    manifest: Mapping[str, Any], run, *, repo_root: Path, data_root: Path,
    cache_root: Path, pilot_root: Path, python: str,
) -> tuple[list[str], dict[str, str]]:
    command, environment = trainer_command(
        manifest, run, python_executable=python, repo_root=repo_root,
        run_root=pilot_root, data_root=data_root, cache_root=cache_root,
        wandb_project=manifest["logging"]["pilot_wandb_project"], wandb_mode="online",
    )
    command = [str(value) for value in command]
    environment = {str(key): str(value) for key, value in environment.items()}
    training = manifest["training"]
    evaluation = manifest["evaluation"]
    replace_override(command, "train.steps", int(training["pilot_updates"]))
    replace_override(command, "train.scheduler_total_steps", 1_000_000)
    replace_override(command, "train.ckpt_every", int(training["pilot_checkpoint_every_updates"]))
    replace_override(command, "train.eval_every", 100_000_000)
    replace_override(command, "eval.final_episodes_per_task", int(evaluation["pilot_final_episodes_per_task"]))
    replace_override(command, "train.viz_every", 100_000_000)
    replace_override(command, "train.viz_every_early", 100_000_000)
    replace_override(command, "resume", "auto")
    joined = " ".join(command).lower()
    required = (
        "experiment=treewm_v2", "objective_version=treewm_v2_rms_rank_v1",
        "arm=treewm", "train.gradient_checkpointing=true", "retrieval.enabled=false",
        f"train.steps={int(training['pilot_updates'])}",
        "train.scheduler_total_steps=1000000",
    )
    if any(token not in joined for token in required):
        raise PilotError("pilot lost a TreeWM-v2 objective invariant")
    if "v1" in str(pilot_root).lower() or "v1" in environment.get("WANDB_RUN_GROUP", "").lower():
        raise PilotError("pilot refuses a v1 run/W&B namespace")
    if environment.get("WANDB_PROJECT") != manifest["logging"]["pilot_wandb_project"]:
        raise PilotError("pilot W&B project is not isolated")
    if environment.get("WANDB_RUN_GROUP") != manifest["logging"]["pilot_wandb_group"]:
        raise PilotError("pilot W&B group is not isolated")
    for name in (
        "TREEWM_DATA_SHA256", "TREEWM_CALIBRATION_SHA256",
        "TREEWM_FUTURE_RECIPE_SHA256", "TREEWM_DATA_CONTRACT_SHA256",
        "WANDB_RUN_ID",
    ):
        value = environment.get(name, "")
        if not value or (name.startswith("TREEWM_") and len(value) != 64):
            raise PilotError(f"pilot launch lacks {name}")
    if any("KEY" in key.upper() or "TOKEN" in key.upper() for key in environment):
        raise PilotError("pilot launch environment contains credential material")
    return command, environment


class ForwardSignals:
    def __init__(self) -> None:
        self.child: subprocess.Popen | None = None
        self.stop_signal: signal.Signals | None = None

    def handle_usr1(self, _signum: int, _frame: object) -> None:
        self.stop_signal = signal.SIGUSR1
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(signal.SIGUSR1)

    def handle_term(self, _signum: int, _frame: object) -> None:
        self.stop_signal = signal.SIGTERM
        if self.child is not None and self.child.poll() is None:
            self.child.send_signal(signal.SIGTERM)


def run_setting(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    run = seed_zero_run(manifest, args.setting_index)
    command, environment = pilot_launch_spec(
        manifest, run, repo_root=args.repo_root, data_root=args.data_root,
        cache_root=args.cache_root, pilot_root=args.pilot_root, python=args.python,
    )
    run_dir = run_directory(args.pilot_root, run)
    attempt_dir = run_dir / "pilot-attempts"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    stem = f"job{os.environ.get('SLURM_JOB_ID', 'local')}.restart{os.environ.get('SLURM_RESTART_COUNT', '0')}"
    atomic_json(
        attempt_dir / f"{stem}.json",
        {"schema_version": 2, "setting_id": run.setting_id, "seed": 0,
         "command": command, "started_unix_time": time.time()},
    )
    forwarder = ForwardSignals()
    signal.signal(signal.SIGUSR1, forwarder.handle_usr1)
    signal.signal(signal.SIGTERM, forwarder.handle_term)
    child_env = os.environ.copy()
    child_env.update(environment)
    with (attempt_dir / f"{stem}.log").open("a", encoding="utf-8", buffering=1) as log:
        forwarder.child = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
            env=child_env,
        )
        return_code = forwarder.child.wait()
    return int(return_code)


def _read_scalars(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError as exc:
        raise PilotError("TensorBoard event reader is unavailable") from exc
    events = sorted(run_dir.glob("events.out.tfevents.*"))
    if not events:
        raise PilotError(f"pilot emitted no TensorBoard event file in {run_dir}")
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    result: dict[str, list[tuple[int, float]]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        values = [(int(event.step), float(event.value)) for event in accumulator.Scalars(tag)]
        if values:
            result[tag] = values
    return result


def _recent_mean(scalars: Mapping[str, list[tuple[int, float]]], tag: str, count: int = 10) -> float:
    values = scalars.get(tag, [])
    if not values:
        raise PilotError(f"pilot telemetry lacks {tag}")
    tail = [value for _, value in values[-count:]]
    if not all(math.isfinite(value) for value in tail):
        raise PilotError(f"pilot telemetry {tag} is non-finite")
    return sum(tail) / len(tail)


def validate_recent_nondegeneracy(
    scalars: Mapping[str, list[tuple[int, float]]], setting_id: str,
) -> dict[str, float]:
    observations: dict[str, float] = {}
    for threshold_name, rule in NONDEGENERACY_THRESHOLDS.items():
        tag = threshold_name.removesuffix(":max")
        value = observations.setdefault(tag, _recent_mean(scalars, tag))
        threshold = float(rule["threshold"])
        operator = str(rule["operator"])
        passed = (
            (operator == ">" and value > threshold)
            or (operator == "<" and value < threshold)
            or (operator == "<=" and value <= threshold)
            or (operator == ">=" and value >= threshold)
        )
        if not passed:
            raise PilotError(
                f"{setting_id}: nondegeneracy gate {tag}={value:.6g} "
                f"does not satisfy {operator}{threshold:.6g}"
            )
    if abs(
        observations["expansion/nodes_generated"]
        + observations["expansion/budget_shortfall"]
        - 64.0
    ) > 1e-4:
        raise PilotError(
            f"{setting_id}: expansion budget telemetry does not sum to node budget 64"
        )
    return observations


def validate_recent_loss_telemetry(
    scalars: Mapping[str, list[tuple[int, float]]], setting_id: str,
) -> None:
    raw_tags = sorted(tag for tag in scalars if tag.startswith("train/loss_raw/"))
    effective_tags = sorted(
        tag for tag in scalars if tag.startswith("train/loss_effective/")
    )
    raw_names = {tag.rsplit("/", 1)[-1] for tag in raw_tags}
    effective_names = {tag.rsplit("/", 1)[-1] for tag in effective_tags}
    if raw_names != FORMAL_ACTIVE_TERMS or effective_names != FORMAL_ACTIVE_TERMS:
        raise PilotError(
            f"{setting_id}: raw/effective telemetry term set differs from exact "
            f"formal objective (raw={sorted(raw_names)}, effective={sorted(effective_names)})"
        )
    total_by_step = dict(scalars.get("train/loss_total", []))
    effective_by_name = {
        tag.rsplit("/", 1)[-1]: dict(values)
        for tag, values in scalars.items()
        if tag.startswith("train/loss_effective/")
    }
    common_steps = set(total_by_step)
    for values in effective_by_name.values():
        common_steps.intersection_update(values)
    if not common_steps:
        raise PilotError(f"{setting_id}: no common objective accounting step")
    for step in sorted(common_steps)[-10:]:
        expected = sum(values[step] for values in effective_by_name.values())
        actual = total_by_step[step]
        if abs(expected - actual) > 5e-4 * max(1.0, abs(expected), abs(actual)):
            raise PilotError(f"{setting_id}: effective terms do not sum to total at {step}")


def validate_gradient_audit(payload: Mapping[str, Any], setting_id: str) -> None:
    if payload.get("schema_version") != 2 or payload.get("status") != "passed":
        raise PilotError(f"{setting_id}: gradient audit did not pass")
    if payload.get("setting_id") != setting_id or int(payload.get("audit_step", -1)) != AUDIT_STEP:
        raise PilotError(f"{setting_id}: gradient audit identity mismatch")
    if int(payload.get("checkpoint_completed_updates", -1)) != AUDIT_STEP:
        raise PilotError(f"{setting_id}: gradient audit was not run on the 5k checkpoint")
    if int(payload.get("scheduler_total_steps", -1)) != 1_000_000:
        raise PilotError(f"{setting_id}: gradient audit did not retain the formal LR horizon")
    claimed_hash = payload.get("artifact_sha256")
    unhashed_payload = dict(payload)
    unhashed_payload.pop("artifact_sha256", None)
    if (
        not isinstance(claimed_hash, str)
        or len(claimed_hash) != 64
        or hashlib.sha256(
            json.dumps(
                unhashed_payload, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest() != claimed_hash
    ):
        raise PilotError(f"{setting_id}: gradient audit artifact hash is invalid")
    batches = payload.get("batch_audits")
    if not isinstance(batches, list) or len(batches) < 3:
        raise PilotError(f"{setting_id}: gradient audit requires three fixed real batches")
    artifact_terms = payload.get("active_terms")
    if (
        not isinstance(artifact_terms, list)
        or len(artifact_terms) != len(FORMAL_ACTIVE_TERMS)
        or set(artifact_terms) != FORMAL_ACTIVE_TERMS
    ):
        raise PilotError(f"{setting_id}: audit artifact has the wrong active-term union")
    dataset_size = int(payload.get("dataset_size", -1))
    selection_seed = int(payload.get("dataset_selection_seed", -1))
    expected_positions = None
    if dataset_size >= 48 and selection_seed == 0:
        from gpu_preflight import representative_dataset_positions

        expected_positions = representative_dataset_positions(dataset_size)
    if expected_positions is None:
        raise PilotError(f"{setting_id}: gradient audit sampling identity is invalid")
    actual_positions = [batch.get("dataset_positions") for batch in batches]
    if actual_positions != expected_positions:
        raise PilotError(f"{setting_id}: gradient audit batches are not representative samples")
    if payload.get("dataset_selection_sha256") != hashlib.sha256(
        json.dumps(actual_positions, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest():
        raise PilotError(f"{setting_id}: gradient audit selection hash is invalid")
    for batch_index, batch in enumerate(batches):
        positions = batch.get("dataset_positions")
        anchors = batch.get("anchor_indices")
        if (
            not isinstance(positions, list)
            or not isinstance(anchors, list)
            or len(positions) != 16
            or len(anchors) != 16
            or batch.get("batch_sha256") != hashlib.sha256(
                json.dumps(
                    {"dataset_positions": positions, "anchor_indices": anchors},
                    sort_keys=True, separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
        ):
            raise PilotError(f"{setting_id}: batch {batch_index} selection identity is invalid")
        metrics = batch.get("metrics")
        active_terms = batch.get("active_terms")
        if (
            not isinstance(metrics, dict)
            or not isinstance(active_terms, list)
            or len(active_terms) != len(FORMAL_ACTIVE_TERMS)
            or set(active_terms) != FORMAL_ACTIVE_TERMS
        ):
            raise PilotError(
                f"{setting_id}: batch {batch_index} active terms differ from exact formal objective"
            )
        for key, value in metrics.items():
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise PilotError(f"{setting_id}: non-finite audit metric {key}")
        for module in REQUIRED_AUDIT_MODULES:
            norm = float(metrics.get(f"gradient_audit/objective_norm/{module}", 0.0))
            if norm <= 0:
                raise PilotError(f"{setting_id}: batch {batch_index} {module} gradient is zero")
        for module in SHARED_AUDIT_MODULES:
            shares = [
                float(value) for key, value in metrics.items()
                if key.startswith(f"gradient_audit/share/{module}/")
            ]
            maximum = max(shares, default=1.0)
            if not shares or maximum > MAX_SHARED_GRADIENT_SHARE:
                raise PilotError(
                    f"{setting_id}: batch {batch_index} {module} loss-gradient share "
                    f"{maximum:.6f} exceeds preregistered {MAX_SHARED_GRADIENT_SHARE:.2f}"
                )


def validate_setting(manifest: Mapping[str, Any], run, args: argparse.Namespace) -> dict[str, Any]:
    run_dir = run_directory(args.pilot_root, run)
    _command, expected_environment = pilot_launch_spec(
        manifest, run, repo_root=args.repo_root, data_root=args.data_root,
        cache_root=args.cache_root, pilot_root=args.pilot_root, python=args.python,
    )
    try:
        completion = json.loads((run_dir / "COMPLETED.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"{run.setting_id}: invalid pilot completion: {exc}") from exc
    expected_updates = int(manifest["training"]["pilot_updates"])
    expected_episodes = int(manifest["evaluation"]["pilot_final_episodes_per_task"])
    identity = completion.get("run_identity") or {}
    if (
        completion.get("status") != "complete"
        or completion.get("objective_version") != "treewm_v2_rms_rank_v1"
        or int(completion.get("completed_updates", -1)) != expected_updates
        or int(completion.get("episodes_per_task", -1)) != expected_episodes
        or completion.get("gradient_checkpointing") is not True
        or identity.get("wandb_project") != manifest["logging"]["pilot_wandb_project"]
        or identity.get("wandb_group") != manifest["logging"]["pilot_wandb_group"]
        or identity.get("wandb_id") != expected_environment["WANDB_RUN_ID"]
        or int(identity.get("total_steps", -1)) != expected_updates
        or int(identity.get("scheduler_total_steps", -1)) != 1_000_000
        or int(completion.get("scheduler_total_steps", -1)) != 1_000_000
        or identity.get("protocol_sha256") != expected_environment["TREEWM_PROTOCOL_SHA256"]
        or identity.get("code_sha256") != expected_environment["TREEWM_CODE_SHA256"]
        or identity.get("runtime_sha256") != expected_environment["TREEWM_RUNTIME_SHA256"]
        or completion.get("protocol_sha256") != expected_environment["TREEWM_PROTOCOL_SHA256"]
        or completion.get("code_sha256") != expected_environment["TREEWM_CODE_SHA256"]
        or completion.get("runtime_sha256") != expected_environment["TREEWM_RUNTIME_SHA256"]
        or completion.get("data_manifest_sha256") != expected_environment["TREEWM_DATA_SHA256"]
        or completion.get("calibration_sha256") != expected_environment["TREEWM_CALIBRATION_SHA256"]
        or completion.get("future_recipe_sha256") != expected_environment["TREEWM_FUTURE_RECIPE_SHA256"]
        or completion.get("wandb_id") != expected_environment["WANDB_RUN_ID"]
        or completion.get("identity_sha256") != hashlib.sha256(
            json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
    ):
        raise PilotError(f"{run.setting_id}: completion does not match v2 pilot protocol")
    contract = load_data_contract(
        manifest, next(item for item in manifest["settings"] if item["id"] == run.setting_id),
        data_root=args.data_root, cache_root=args.cache_root,
    )
    if identity.get("data_manifest_sha256") != contract["data_manifest_sha256"]:
        raise PilotError(f"{run.setting_id}: pilot used the wrong data contract")
    if identity.get("calibration_sha256") != contract["calibration_sha256"]:
        raise PilotError(f"{run.setting_id}: pilot used the wrong calibration contract")
    if identity.get("future_recipe_sha256") != contract["future_recipe_sha256"]:
        raise PilotError(f"{run.setting_id}: pilot used the wrong future-recipe contract")

    audit_path = args.pilot_root / "state" / "gradient-audits" / f"{run.setting_id}.json"
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"{run.setting_id}: missing gradient audit: {exc}") from exc
    validate_gradient_audit(audit, run.setting_id)
    if (
        audit.get("protocol_sha256") != expected_environment["TREEWM_PROTOCOL_SHA256"]
        or audit.get("code_sha256") != expected_environment["TREEWM_CODE_SHA256"]
        or audit.get("runtime_sha256") != expected_environment["TREEWM_RUNTIME_SHA256"]
        or audit.get("data_manifest_sha256") != contract["data_manifest_sha256"]
        or audit.get("calibration_sha256") != contract["calibration_sha256"]
        or audit.get("future_recipe_sha256") != contract["future_recipe_sha256"]
    ):
        raise PilotError(f"{run.setting_id}: gradient audit provenance differs from live contract")
    checkpoint_path = run_dir / "checkpoints" / "latest.pt"
    if (
        not checkpoint_path.is_file()
        or hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
        != audit.get("pilot_checkpoint_sha256")
    ):
        raise PilotError(f"{run.setting_id}: gradient audit is stale for the 5k checkpoint")

    scalars = _read_scalars(run_dir)
    for tag, values in scalars.items():
        if any(not math.isfinite(value) for _, value in values):
            raise PilotError(f"{run.setting_id}: non-finite scalar in {tag}")
    # Scalar composition is an accounting check only.  Dominance is gated above by
    # per-term shared-encoder gradients, not by incomparable scalar magnitudes.
    validate_recent_loss_telemetry(scalars, run.setting_id)

    calibration_gates = {
        "data/num_retrieved": (18.0, math.inf),
        "data/horizon_target_normalized_entropy": (0.65, 1.000001),
        "data/horizon_target_occupied_classes": (4.0, 5.000001),
        "data/num_modes": (1.5, 3.5),
        "data/multimode_anchor_fraction": (0.40, 1.000001),
        "data/retrieval_fallback": (-1e-9, 0.01),
        "data/mode_truncation_fraction": (-1e-9, 0.05),
        "control/endpoint_fallback": (-1e-9, 1e-9),
    }
    recent: dict[str, float] = {}
    for tag, (lower, upper) in calibration_gates.items():
        value = _recent_mean(scalars, tag)
        recent[tag] = value
        if not lower <= value <= upper:
            raise PilotError(f"{run.setting_id}: {tag}={value:.6g} outside [{lower}, {upper}]")
    recent_nondegeneracy = validate_recent_nondegeneracy(scalars, run.setting_id)
    # Module snapshots are emitted only at the log-boundary step. With formal
    # gain stride 4 and log interval 50, that step can never be gain-active
    # (step % 4 == 0 versus step % 50 == 49). Gate the four always-active shared
    # modules here and use the interval-averaged, separately clipped gain norm for
    # the frozen-prior/contextual-only gain parameter group below.
    for module in SHARED_AUDIT_MODULES:
        tag = f"train/grad_norm_module/{module}"
        if _recent_mean(scalars, tag) <= 0:
            raise PilotError(f"{run.setting_id}: no live training gradient for {module}")
    if _recent_mean(scalars, "train/grad_norm_gain") <= 0:
        raise PilotError(f"{run.setting_id}: no interval-averaged contextual-gain gradient")

    data_wait = _recent_mean(scalars, "train/data_wait_frac")
    if data_wait > MAX_MEAN_DATA_WAIT_FRACTION:
        raise PilotError(
            f"{run.setting_id}: ten-worker recipe loader remains bottlenecked "
            f"(recent data_wait_frac={data_wait:.3f} > {MAX_MEAN_DATA_WAIT_FRACTION:.2f})"
        )
    data_wait_status = "warning" if data_wait > WARN_MEAN_DATA_WAIT_FRACTION else "passed"

    payload = {
        "schema_version": 2,
        "status": "passed",
        "setting_id": run.setting_id,
        "seed": 0,
        "completed_updates": expected_updates,
        "objective_version": "treewm_v2_rms_rank_v1",
        "protocol_sha256": expected_environment["TREEWM_PROTOCOL_SHA256"],
        "code_sha256": expected_environment["TREEWM_CODE_SHA256"],
        "runtime_sha256": expected_environment["TREEWM_RUNTIME_SHA256"],
        "data_manifest_sha256": contract["data_manifest_sha256"],
        "calibration_sha256": contract["calibration_sha256"],
        "future_recipe_sha256": contract["future_recipe_sha256"],
        "gradient_audit_sha256": audit.get("artifact_sha256"),
        "max_shared_module_gradient_share": max(
            float(value)
            for batch in audit["batch_audits"]
            for key, value in batch["metrics"].items()
            if any(key.startswith(f"gradient_audit/share/{module}/") for module in SHARED_AUDIT_MODULES)
        ),
        "recent_calibration_telemetry": recent,
        "nondegeneracy_thresholds": NONDEGENERACY_THRESHOLDS,
        "recent_nondegeneracy_telemetry": recent_nondegeneracy,
        "data_loader_workers": int(manifest["training"]["data_loader_workers"]),
        "recent_data_wait_fraction": data_wait,
        "data_wait_status": data_wait_status,
        "validated_unix_time": time.time(),
    }
    atomic_json(args.pilot_root / "state" / "pilot-passes" / f"{run.setting_id}.json", payload)
    return payload


def validate_all(manifest: Mapping[str, Any], args: argparse.Namespace) -> None:
    passes = []
    for index in range(len(manifest["settings"])):
        passes.append(validate_setting(manifest, seed_zero_run(manifest, index), args))
    if (
        len({payload["code_sha256"] for payload in passes}) != 1
        or len({payload["runtime_sha256"] for payload in passes}) != 1
    ):
        raise PilotError("pilot settings used mixed code/runtime identities")
    atomic_json(
        args.pilot_root / "state" / "PILOT_ACCEPTED.json",
        {
            "schema_version": 2,
            "status": "passed",
            "campaign_id": manifest["campaign_id"],
            "settings": [payload["setting_id"] for payload in passes],
            "campaign_protocol_sha256": protocol_sha256(manifest),
            "pilot_protocol_sha256_by_setting": {
                payload["setting_id"]: payload["protocol_sha256"]
                for payload in passes
            },
            "code_sha256": passes[0]["code_sha256"],
            "runtime_sha256": passes[0]["runtime_sha256"],
            "data_manifest_sha256_by_setting": {
                payload["setting_id"]: payload["data_manifest_sha256"]
                for payload in passes
            },
            "calibration_sha256_by_setting": {
                payload["setting_id"]: payload["calibration_sha256"]
                for payload in passes
            },
            "future_recipe_sha256_by_setting": {
                payload["setting_id"]: payload["future_recipe_sha256"]
                for payload in passes
            },
            "gradient_audit_sha256_by_setting": {
                payload["setting_id"]: payload["gradient_audit_sha256"]
                for payload in passes
            },
            "nondegeneracy_thresholds": NONDEGENERACY_THRESHOLDS,
            "max_shared_module_gradient_share": max(
                payload["max_shared_module_gradient_share"] for payload in passes
            ),
            "accepted_unix_time": time.time(),
        },
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run-setting", action="store_true")
    mode.add_argument("--validate-setting", action="store_true")
    mode.add_argument("--validate-all", action="store_true")
    parser.add_argument("--setting-index", type=int)
    parser.add_argument("--manifest", type=Path, default=here / "manifest.json")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", default=os.environ.get("TREEWM_PYTHON", sys.executable))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("TREEWM_DATA_ROOT", here / "data")))
    parser.add_argument("--cache-root", type=Path, default=Path(os.environ.get("TREEWM_CACHE", here / "cache")))
    parser.add_argument("--pilot-root", type=Path, default=Path(os.environ.get("TREEWM_PILOT_ROOT", repo_root / "outputs" / "treewm-50task-v2-pilot")))
    args = parser.parse_args(argv)
    if not args.validate_all and args.setting_index is None:
        parser.error("--setting-index is required")
    for name in ("manifest", "repo_root", "data_root", "cache_root", "pilot_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        manifest = load_manifest(args.manifest)
        if args.run_setting:
            return run_setting(args)
        if args.validate_setting:
            validate_setting(manifest, seed_zero_run(manifest, args.setting_index), args)
        else:
            validate_all(manifest, args)
        return 0
    except (PilotError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"pilot failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
