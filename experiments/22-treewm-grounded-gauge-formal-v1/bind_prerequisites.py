#!/usr/bin/env python3
"""Replay Exp20 and Exp21 raw gates and seal their exact selected recipe for Exp22."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping, Sequence

from campaign import (
    CAMPAIGN_DIR,
    ContractError,
    PINNED_FORMAL_PYTHON,
    PREREQUISITE_BINDINGS_PATH,
    PROTOCOL_LOCK_PATH,
    REPOSITORY_ROOT,
    atomic_json,
    file_sha256,
    load_compatible_input,
    load_manifest,
    protocol_sha256,
    read_json,
    require,
    stable_hash,
)
import raw_exp20_recompute


def sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def self_hash(value: Mapping[str, Any], key: str, label: str) -> str:
    body = dict(value)
    claimed = body.pop(key, None)
    require(sha(claimed) and claimed == stable_hash(body), f"{label} self-hash differs")
    return str(claimed)


def contains_forbidden(value: object, tokens: Sequence[str]) -> bool:
    if isinstance(value, Mapping):
        return any(contains_forbidden(key, tokens) or contains_forbidden(item, tokens) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return any(contains_forbidden(item, tokens) for item in value)
    if isinstance(value, str):
        lowered = value.lower()
        return any(token.lower() in lowered for token in tokens)
    return False


def reject_forbidden(value: object, tokens: Sequence[str], label: str) -> None:
    require(not contains_forbidden(value, tokens), f"{label} contains forbidden Exp14-18 ancestry/fields")


def run_json(command: Sequence[str], *, cwd: Path, label: str) -> dict[str, Any]:
    """Run a sealed prerequisite verifier and parse its single JSON stdout object."""
    result = subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env={
            **os.environ,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        },
    )
    require(result.returncode == 0, f"{label} failed: {result.stderr.strip()}")
    try:
        value = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise ContractError(f"{label} emitted non-JSON output: {exc}") from exc
    require(isinstance(value, dict), f"{label} did not emit one JSON object")
    return value


def verify_exp21_protocol(exp21_dir: Path) -> str:
    lock = exp21_dir / "protocol.sha256"
    require(lock.is_file() and not lock.is_symlink(), "Exp21 protocol lock missing/symlinked")
    locked = lock.read_text(encoding="utf-8").strip()
    require(sha(locked), "Exp21 protocol lock is malformed")
    output = run_json(
        [PINNED_FORMAL_PYTHON, "-c", (
            "import json,pathlib,sys;"
            f"p=pathlib.Path({str(exp21_dir / 'campaign.py')!r});"
            "sys.path.insert(0,str(p.parent));"
            "import campaign as m;"
            "print(json.dumps({'live':m.protocol_sha256(p.parent)}))"
        )],
        cwd=REPOSITORY_ROOT,
        label="Exp21 live protocol recomputation",
    )
    require(output.get("live") == locked, "Exp21 live package bytes differ from protocol lock")
    return locked


def file_record(path: Path) -> dict[str, Any]:
    require(path.is_file() and not path.is_symlink(), f"raw evidence missing/symlinked: {path}")
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": file_sha256(path)}


def event_records(run_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(run_dir.glob("events.out.tfevents.*")) or sorted(run_dir.rglob("events.out.tfevents.*"))
    require(paths, f"no TensorBoard raw evidence in {run_dir}")
    return [file_record(path) for path in paths]


def checkpoint_record(run_dir: Path, expected_sha256: str) -> dict[str, Any]:
    record = file_record(run_dir / "checkpoints" / "latest.pt")
    require(record["sha256"] == expected_sha256, f"checkpoint bytes differ from accepted row: {run_dir}")
    return record


def launch_record(path: Path, expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    record = file_record(path)
    launch = read_json(path)
    self_hash(launch, "launch_sha256", f"launch {path}")
    require(launch.get("launch_sha256") == expected_sha256, f"launch hash differs from accepted row: {path}")
    return record, launch


def _exp20_evidence(
    exp20_binding: Mapping[str, Any],
    gate_5000: Mapping[str, Any],
    gate_25000: Mapping[str, Any],
    forbidden_tokens: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any], list[str], dict[str, str]]:
    evidence = exp20_binding["exp20"].get("raw_evidence") or []
    require(len(evidence) == 30, "recomputed Exp20 binding lacks all 30 raw runs")
    by_key = {(row["setting_id"], row["arm_id"], int(row["seed"])): row for row in evidence}
    rows_5000 = gate_5000.get("runs") or []
    require(len(rows_5000) == 30, "Exp20 5k gate lacks all 30 raw rows")
    result_5000: dict[str, Any] = {}
    bound: dict[str, str] = {}
    launches: dict[tuple[str, str, int], dict[str, Any]] = {}
    selected_arm = str(gate_25000.get("selected_arm"))
    for row in rows_5000:
        key = (str(row["setting_id"]), str(row["arm_id"]), int(row["seed"]))
        raw = by_key.get(key)
        require(raw is not None and raw.get("launch_sha256") == row.get("launch_sha256"), f"Exp20 raw launch/gate row differs: {key}")
        launch_path = Path(raw["launch_path"])
        launch_file, launch = launch_record(launch_path, str(row["launch_sha256"]))
        reject_forbidden(launch, forbidden_tokens, f"Exp20 raw launch {key}")
        launches[key] = launch
        events = [file_record(Path(item["path"])) for item in raw.get("event_files") or []]
        require(events and events == raw.get("event_files"), f"Exp20 raw event bytes changed: {key}")
        run_dir = Path(raw["run_directory"])
        checkpoint, lifecycle_files = _exp20_stage_completion(
            run_dir,
            launch,
            row,
            target=5_000,
            stage_slot=int(row["stage_slot"]),
            current_checkpoint=key[1] != selected_arm,
            forbidden_tokens=forbidden_tokens,
        )
        # Selected runs advance to 25k, so the current checkpoint is bound below.
        result_5000["|".join(map(str, key))] = {
            "launch_file_sha256": launch_file["sha256"],
            "launch_sha256": launch["launch_sha256"],
            "event_evidence_sha256": stable_hash(events),
            "checkpoint_sha256": str(row["checkpoint_sha256"]),
            "checkpoint_currently_at_5000": checkpoint is not None,
        }
        bound[launch_file["path"]] = launch_file["sha256"]
        bound.update(lifecycle_files)
        for item in events:
            bound[item["path"]] = item["sha256"]

    selected = gate_25000.get("selected_runs") or []
    skipped = gate_25000.get("skipped_runs") or []
    require(len(selected) == 10 and len(skipped) == 10, "Exp20 terminal selected/skipped coverage differs")
    gate_5000_hash = str(gate_5000["gate_sha256"])
    row_5000_by_index = {int(item["index"]): item for item in rows_5000}
    result_25000: dict[str, Any] = {}
    for row in selected:
        key = (str(row["setting_id"]), str(row["arm_id"]), int(row["seed"]))
        raw = by_key[key]
        previous_row = row_5000_by_index[int(row["index"])]
        checkpoint, lifecycle_files = _exp20_stage_completion(
            Path(raw["run_directory"]),
            launches[key],
            row,
            target=25_000,
            stage_slot=int(row["stage_slot"]),
            current_checkpoint=True,
            forbidden_tokens=forbidden_tokens,
            expected_previous_gate={
                "stage_target": 5_000,
                "gate_sha256": gate_5000_hash,
                "selected_arm": selected_arm,
                "selected": True,
                "identity_sha256": previous_row["identity_sha256"],
                "checkpoint_sha256": previous_row["checkpoint_sha256"],
            },
        )
        events = [file_record(Path(item["path"])) for item in raw["event_files"]]
        result_25000["|".join(map(str, key))] = {
            "launch_file_sha256": str(raw["launch_file_sha256"]),
            "launch_sha256": str(row["launch_sha256"]),
            "event_evidence_sha256": stable_hash(events),
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        }
        bound.update(lifecycle_files)
    for row in skipped:
        key = (str(row["setting_id"]), str(row["arm_id"]), int(row["seed"]))
        raw = by_key[key]
        launch = launches[key]
        skip_path = Path(raw["run_directory"]) / "stage-gates/SKIPPED_BY_SELECTION_25000.json"
        require(skip_path.is_file() and not skip_path.is_symlink(), f"Exp20 selection skip missing/linked: {key}")
        skip = read_json(skip_path)
        reject_forbidden(skip, forbidden_tokens, f"Exp20 selection skip {key}")
        self_hash(skip, "skip_sha256", f"Exp20 selection skip {key}")
        require(
            skip.get("schema_version") == 1
            and skip.get("status") == "skipped_by_immutable_5000_selection"
            and skip.get("campaign_id") == launch["campaign_id"]
            and int(skip.get("stage_target", -1)) == 25_000
            and int(skip.get("stage_slot", -1)) == int(row["stage_slot"])
            and int(skip.get("index", -1)) == int(row["index"])
            and skip.get("setting_id") == key[0]
            and skip.get("arm_id") == key[1]
            and int(skip.get("seed", -1)) == key[2]
            and skip.get("selected_arm") == selected_arm
            and skip.get("gate_sha256") == gate_5000_hash
            and skip.get("launch_sha256") == row["launch_sha256"]
            and skip.get("identity_sha256") == row_5000_by_index[int(row["index"])]["identity_sha256"]
            and skip.get("checkpoint_sha256") == row["checkpoint_sha256"]
            and int(skip.get("completed_updates", -1)) == 5_000
            and skip.get("trainer_launched") is False
            and skip.get("skip_sha256") == row["skip_sha256"],
            f"Exp20 selection skip differs: {key}",
        )
        bound[str(skip_path.resolve())] = file_sha256(skip_path)
    skipped_names = [f"{row['setting_id']}|{row['arm_id']}|{int(row['seed'])}" for row in skipped]
    return result_5000, result_25000, sorted(skipped_names), bound


def _metric_at(
    metrics: Mapping[str, Mapping[int, float]], tag: str, step: int
) -> float | None:
    value = metrics.get(tag, {}).get(step)
    return float(value) if raw_exp20_recompute.finite(value) else None


def _exp21_outcome_summary(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    gate = manifest["stage_acceptance"]
    settings = [str(row["id"]) for row in manifest["settings"]]
    seeds = list(manifest["design"]["seeds"])
    episodes = float(gate["outcome_episodes_per_run"])
    for row in rows:
        outcome = row["outcome"]
        require(
            all(raw_exp20_recompute.finite(outcome.get(name)) for name in (
                "num_episodes", "successes", "success_rate", "distance_reduction_frac"
            )),
            "Exp21 outcome telemetry is nonfinite/incomplete",
        )
        require(outcome["num_episodes"] == episodes, "Exp21 outcome episode count differs")
        successes = float(outcome["successes"])
        require(successes.is_integer() and 0 <= successes <= episodes, "Exp21 success count invalid")
        require(abs(float(outcome["success_rate"]) - successes / episodes) <= 1e-6, "Exp21 success rate/count differs")
        task_rows = outcome.get("tasks") or []
        require([task.get("task_id") for task in task_rows] == [1, 2, 3, 4, 5], "Exp21 task coverage differs")
        require(all(task.get("num_episodes") == 1.0 for task in task_rows), "Exp21 task episode count differs")
        require(
            all(raw_exp20_recompute.finite(task.get(name)) for task in task_rows for name in ("num_episodes", "successes", "success_rate")),
            "Exp21 per-task outcome telemetry is nonfinite/incomplete",
        )
        require(abs(sum(float(task["successes"]) for task in task_rows) - successes) <= 1e-6, "Exp21 task successes differ")
        require(all(float(task["successes"]) in (0.0, 1.0) and float(task["success_rate"]) == float(task["successes"]) for task in task_rows), "Exp21 task success accounting differs")
    per_seed: dict[str, Any] = {}
    for seed in seeds:
        selected = [row for row in rows if int(row["seed"]) == seed]
        require(len(selected) == 10, f"Exp21 seed {seed} coverage differs")
        successes = sum(float(row["outcome"]["successes"]) for row in selected)
        progress = sum(float(row["outcome"]["distance_reduction_frac"]) for row in selected) / len(selected)
        require(successes >= int(gate["min_total_successes_per_seed"]), f"Exp21 seed {seed} all-zero success")
        require(progress > float(gate["min_mean_distance_reduction_per_seed_exclusive"]), f"Exp21 seed {seed} nonpositive progress")
        per_seed[str(seed)] = {"successes": successes, "mean_distance_reduction_frac": progress}
    per_setting: dict[str, Any] = {}
    both_success = 0
    both_progress = 0
    for setting in settings:
        selected = [row for row in rows if row["setting_id"] == setting]
        require({int(row["seed"]) for row in selected} == set(seeds), f"Exp21 {setting} seed coverage differs")
        success = all(float(row["outcome"]["successes"]) > 0 for row in selected)
        progress = all(float(row["outcome"]["distance_reduction_frac"]) > 0 for row in selected)
        both_success += success
        both_progress += progress
        per_setting[setting] = {"both_seed_nonzero_success": success, "both_seed_positive_progress": progress}
    require(both_success >= int(gate["min_settings_with_both_seed_success"]), "Exp21 replicated success quorum failed")
    require(both_progress >= int(gate["min_settings_with_both_seed_positive_progress"]), "Exp21 6/10 progress quorum failed")
    return {
        "per_seed": per_seed,
        "per_setting": per_setting,
        "settings_with_both_seed_nonzero_success": both_success,
        "settings_with_both_seed_positive_progress": both_progress,
        "total_successes": sum(float(row["outcome"]["successes"]) for row in rows),
        "macro_success_rate": sum(float(row["outcome"]["success_rate"]) for row in rows) / len(rows),
        "macro_distance_reduction_frac": sum(float(row["outcome"]["distance_reduction_frac"]) for row in rows) / len(rows),
    }


def _render_override(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if value is None:
        return "null"
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(_render_override(item) for item in value) + "]"
    return str(value)


def _argv_overrides(argv: Sequence[str], label: str) -> tuple[dict[str, str], list[str]]:
    require(len(argv) >= 3, f"{label} argv is too short")
    result: dict[str, str] = {}
    tokens: list[str] = []
    for token in argv[2:]:
        require(isinstance(token, str) and "=" in token and not token.startswith("--"), f"{label} has non-Hydra argv token")
        key, value = token.split("=", 1)
        require(key and key not in result, f"{label} has duplicate override {key}")
        result[key] = value
        tokens.append(token)
    return result, tokens


def _override(name: str, value: object) -> str:
    return f"{name}={_render_override(value)}"


def _expected_exp21_overrides(
    manifest: Mapping[str, Any],
    setting: Mapping[str, Any],
    seed: int,
    contract: Mapping[str, Any],
    recipe: Mapping[str, Any],
    selected_arm: str,
    upstream_binding_sha256: str,
    selected_recipe_sha256: str,
) -> list[str]:
    method = manifest["method"]
    scientific = manifest["scientific_contract"]
    future = scientific["future_config"]
    chosen = contract["chosen_thresholds"]
    return [
        _override("env", setting["env_config"]),
        _override("experiment", method["experiment_config"]),
        _override("arm", method["arm"]),
        _override("objective_version", method["objective_version"]),
        _override("seed", seed),
        _override("train.steps", scientific["optimizer_updates"]),
        _override("train.scheduler_total_steps", scientific["scheduler_total_steps"]),
        _override("train.log_every", scientific["training_log_every_updates"]),
        _override("train.ckpt_every", scientific["checkpoint_every_updates"]),
        _override("train.val_every", scientific["validation_every_updates"]),
        _override("train.diag_every", scientific["diagnostics_every_updates"]),
        _override("train.eval_every", scientific["periodic_evaluation_every_updates"]),
        _override("train.validation_sample_seed", scientific["validation_sample_seed"]),
        _override("train.max_train_anchors", setting["published_union_train_anchors"]),
        _override("train.max_val_anchors", setting["published_union_validation_anchors"]),
        _override("train.num_workers", scientific["data_loader_workers"]),
        _override("train.lr", recipe["world_lr"]),
        _override("train.weight_decay", scientific["world_weight_decay"]),
        _override("train.gradient_checkpointing", scientific["gradient_checkpointing"]),
        _override("train.separate_gain_grad_clip", True),
        _override("train.separate_branch_transformer_grad_clip", recipe["separate_branch_transformer_grad_clip"]),
        _override("train.world_grad_clip", 1.0),
        _override("train.gain_grad_clip", 1.0),
        _override("train.branch_transformer_grad_clip", recipe["branch_transformer_grad_clip"]),
        _override("train.gain_loss_every", scientific["gain_loss_every"]),
        _override("train.gain_lr", scientific["gain_lr"]),
        _override("train.gain_weight_decay", scientific["gain_weight_decay"]),
        _override("train.gain_training_scorers", scientific["gain_training_scorers"]),
        _override("model.dropout", scientific["model_dropout"]),
        _override("model.max_depth", scientific["model_max_depth"]),
        _override("tree.max_depth", scientific["tree_max_depth"]),
        _override("tree.node_budget", method["node_budget"]),
        _override("tree.keep_threshold", scientific["keep_threshold"]),
        _override("tree.scorer", method["scorer"]),
        _override("model.branch_factor", method["branch_factor"]),
        _override("planner.decoded_metric", scientific["planner_decoded_metric"]),
        _override("planner.execute_mode", scientific["planner_execute_mode"]),
        _override("planner.execute_steps", scientific["planner_execute_steps"]),
        _override("planner.max_env_steps", setting["max_episode_steps"]),
        _override("planner.require_first_edge_improvement", method["require_first_edge_improvement"]),
        _override("planner.min_first_edge_improvement", scientific["min_first_edge_improvement"]),
        *[_override(f"future_sets.{name}", value) for name, value in future.items()],
        _override("future_sets.relative_endpoints", setting["relative_endpoints"]),
        _override("future_sets.retrieval_radius", chosen["retrieval_radius"]),
        _override("future_sets.displacement_threshold", chosen["displacement_threshold"]),
        _override("future_sets.cluster_threshold", chosen["cluster_threshold"]),
        _override("+env.task_metric_dims", setting["task_metric_dims"]),
        _override("losses.keep_balance", scientific["keep_balance"]),
        _override("losses.enabled.multistep", scientific["multistep_enabled"]),
        _override("losses.weights.multistep", scientific["multistep_weight"]),
        _override("losses.scheduled_sampling_p", scientific["scheduled_sampling_p"]),
        _override("losses.scheduled_sampling_warmup", scientific["scheduled_sampling_warmup"]),
        _override("losses.scheduled_sampling_granularity", scientific["scheduled_sampling_granularity"]),
        _override("losses.multistep_transition_mode", recipe["transition_mode"]),
        _override("losses.grounded_select_action_weight", recipe["grounded_select_action_weight"]),
        _override("losses.grounded_select_endpoint_weight", recipe["grounded_select_endpoint_weight"]),
        _override("losses.grounded_select_horizon_weight", recipe["grounded_select_horizon_weight"]),
        _override("losses.grounded_loss_latent_weight", recipe["grounded_loss_latent_weight"]),
        _override("losses.grounded_loss_action_weight", recipe["grounded_loss_action_weight"]),
        _override("losses.grounded_loss_horizon_weight", recipe["grounded_loss_horizon_weight"]),
        _override("losses.grounded_loss_endpoint_weight", recipe["grounded_loss_endpoint_weight"]),
        _override("losses.grounded_detach_self_fed_parent", scientific["grounded_detach_self_fed_parent"]),
        _override("losses.multistep_depth_weights", scientific["multistep_depth_weights"]),
        _override("losses.enabled.latent_gauge", recipe["latent_gauge_enabled"]),
        _override("losses.weights.latent_gauge", recipe["latent_gauge_weight"]),
        _override("losses.latent_gauge_epsilon", scientific["latent_gauge_epsilon"]),
        _override("losses.latent_gauge_min_reference_scale", scientific["latent_gauge_min_reference_scale"]),
        _override("eval.task_split", scientific["task_split"]),
        _override("eval.episodes_per_task", scientific["periodic_episodes_per_task"]),
        _override("eval.final_episodes_per_task", scientific["final_episodes_per_task"]),
        _override("eval.seed", scientific["evaluation_seed"]),
        _override("+campaign_input_contract_sha256", contract["contract_sha256"]),
        _override("+campaign_calibration_sha256", contract["calibration_sha256"]),
        _override("+campaign_future_recipe_sha256", contract["future_recipe_sha256"]),
        _override("+campaign_compatible_recipe_code_sha256", manifest["compatible_v2_recipe_input"]["recipe_code_sha256"]),
        _override("+campaign_factorial_arm", f"exp20-{selected_arm}-all-ten"),
        _override("+campaign_prerequisite_binding_sha256", upstream_binding_sha256),
        _override("+campaign_exp20_binding_sha256", upstream_binding_sha256),
        _override("+campaign_selected_recipe_sha256", selected_recipe_sha256),
    ]


def _verify_exp21_source_snapshot(
    exp21_manifest: Mapping[str, Any],
    *,
    package_protocol_sha256: str,
    source_sha256: str,
    runtime_sha256: str,
) -> tuple[Path, dict[str, str]]:
    identity = stable_hash({
        "source_sha256": source_sha256,
        "runtime_sha256": runtime_sha256,
        "package_protocol_sha256": package_protocol_sha256,
    })
    repo = (
        Path(exp21_manifest["paths"]["run_root"])
        / "state/source-snapshots" / identity / "repo"
    )
    marker_path = repo.parent / "SNAPSHOT.json"
    lock_path = repo / "experiments/21-treewm-grounded-gauge-all-ten-bridge-v2/protocol.sha256"
    entry_path = repo / "experiments/21-treewm-grounded-gauge-all-ten-bridge-v2/train_entry.py"
    require(repo.is_dir() and not repo.is_symlink(), "Exp21 accepted source snapshot missing/linked")
    for path, label in ((marker_path, "marker"), (lock_path, "protocol lock"), (entry_path, "trainer entry")):
        require(path.is_file() and not path.is_symlink(), f"Exp21 snapshot {label} missing/linked")
    marker = read_json(marker_path)
    require(
        marker.get("schema_version") == 1
        and marker.get("status") == "sealed_read_only"
        and marker.get("repo_subdirectory") == "repo"
        and marker.get("repo_files_writable") is False
        and marker.get("formal_validation") is False
        and marker.get("trainer_source_sha256") == source_sha256
        and marker.get("runtime_sha256") == runtime_sha256
        and marker.get("package_protocol_sha256") == package_protocol_sha256
        and marker.get("snapshot_identity_sha256") == identity,
        "Exp21 accepted source snapshot marker differs",
    )
    require(lock_path.read_text(encoding="utf-8").strip() == package_protocol_sha256, "Exp21 snapshot protocol lock differs")
    regular_files = [path for path in repo.rglob("*") if path.is_file()]
    require(regular_files and all(not path.is_symlink() for path in regular_files), "Exp21 snapshot contains missing/linked files")
    require(all(path.stat().st_mode & 0o222 == 0 for path in regular_files), "Exp21 snapshot contains writable source")
    from treewm.utils.provenance import trainer_code_fingerprint

    require(trainer_code_fingerprint(repo).get("manifest_sha256") == source_sha256, "Exp21 snapshot trainer source fingerprint differs")
    return repo, {
        str(marker_path.resolve()): file_sha256(marker_path),
        str(lock_path.resolve()): file_sha256(lock_path),
        str(entry_path.resolve()): file_sha256(entry_path),
    }


def _rank_state_complete(payload: Mapping[str, Any]) -> bool:
    identity = payload.get("run_identity") or {}
    rank_states = payload.get("rank_states") or []
    if int(identity.get("world_size", -1)) != 1 or len(rank_states) != 1 or not isinstance(rank_states[0], Mapping):
        return False
    state = rank_states[0]
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


def _validate_exp20_checkpoint(
    path: Path,
    launch: Mapping[str, Any],
    row: Mapping[str, Any],
    target: int,
    forbidden_tokens: Sequence[str],
) -> dict[str, Any]:
    import torch

    require(path.is_file() and not path.is_symlink(), f"Exp20 checkpoint missing/linked: {path}")
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ContractError(f"Exp20 checkpoint cannot be loaded: {exc}") from exc
    from treewm.utils.checkpoint import validate_exact_resume_payload

    try:
        validate_exact_resume_payload(
            payload,
            expected_identity=payload.get("run_identity"),
            expected_world_size=1,
            require_cuda_rng=True,
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"Exp20 checkpoint exact-resume payload is invalid: {exc}") from exc
    identity = payload.get("run_identity") or {}
    config = payload.get("config") or {}
    reject_forbidden(identity, forbidden_tokens, "Exp20 checkpoint run identity")
    reject_forbidden(config, forbidden_tokens, "Exp20 checkpoint config")
    run = launch["run"]
    hashes = launch["hashes"]
    checks = (
        payload.get("schema_version") == 2,
        int(payload.get("completed_updates", -1)) == target,
        int(payload.get("step", -1)) == target,
        int(payload.get("next_step", -1)) == target,
        payload.get("optimizer") is not None,
        payload.get("scheduler") is not None,
        _rank_state_complete(payload),
        sha(payload.get("identity_sha256")),
        payload.get("identity_sha256") == row.get("identity_sha256"),
        identity.get("run_name") == run["run_name"],
        identity.get("setting") == run["setting_id"],
        int(identity.get("seed", -1)) == int(run["seed"]),
        identity.get("objective_version") == "treewm_v2_grounded_gauge_pilot_v2",
        int(identity.get("total_steps", -1)) == 25_000,
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
        identity.get("campaign_factorial_arm") == run["arm_id"],
        identity.get("data_manifest_sha256") == hashes["data_manifest_sha256"],
        identity.get("calibration_sha256") == hashes["calibration_sha256"],
        identity.get("future_recipe_sha256") == hashes["future_recipe_sha256"],
        identity.get("evaluation_seed_tables_sha256") == hashes["evaluation_seed_tables_sha256"],
        identity.get("final_seed_table_sha256") == hashes["final_seed_table_sha256"],
        config.get("campaign_source_sha256") == hashes["source_sha256"],
        config.get("campaign_protocol_sha256") == hashes["package_protocol_sha256"],
        config.get("campaign_config_sha256") == hashes["config_sha256"],
    )
    require(all(checks), f"Exp20 checkpoint exact-resume identity differs: {path}")
    checkpoint_sha256 = file_sha256(path)
    require(checkpoint_sha256 == row.get("checkpoint_sha256"), f"Exp20 checkpoint bytes differ: {path}")
    return {
        "identity_sha256": payload["identity_sha256"],
        "evaluation_seed_tables_sha256": identity["evaluation_seed_tables_sha256"],
        "final_seed_table_sha256": identity["final_seed_table_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
    }


def _exp20_stage_completion(
    run_dir: Path,
    launch: Mapping[str, Any],
    row: Mapping[str, Any],
    *,
    target: int,
    stage_slot: int,
    current_checkpoint: bool,
    forbidden_tokens: Sequence[str],
    expected_previous_gate: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, str]]:
    marker_path = run_dir / "stage-gates" / f"AWAITING_GATE_{target}.json"
    complete_path = run_dir / "stage-gates" / f"STAGE_COMPLETE_{target}.json"
    stage_launch_path = run_dir / "stage-gates" / f"STAGE_LAUNCH_{target}.json"
    for path, label in ((marker_path, "marker"), (complete_path, "completion"), (stage_launch_path, "stage launch")):
        require(path.is_file() and not path.is_symlink(), f"Exp20 {label} missing/linked: {path}")
    marker = read_json(marker_path)
    complete = read_json(complete_path)
    stage_launch = read_json(stage_launch_path)
    reject_forbidden((marker, complete, stage_launch), forbidden_tokens, "Exp20 lifecycle artifacts")
    self_hash(complete, "stage_complete_sha256", "Exp20 stage completion")
    run = launch["run"]
    hashes = launch["hashes"]
    checkpoint = (
        _validate_exp20_checkpoint(
            run_dir / "checkpoints/latest.pt", launch, row, target, forbidden_tokens
        )
        if current_checkpoint else None
    )
    identity_sha256 = (checkpoint or row)["identity_sha256"]
    checkpoint_sha256 = (checkpoint or row)["checkpoint_sha256"]
    require(
        marker.get("schema_version") == 1
        and marker.get("status") == "awaiting_external_stage_gate"
        and marker.get("objective_version") == "treewm_v2_grounded_gauge_pilot_v2"
        and int(marker.get("completed_updates", -1)) == target
        and int(marker.get("step", -1)) == target
        and int(marker.get("total_steps", -1)) == 25_000
        and int(marker.get("scheduler_total_steps", -1)) == 1_000_000
        and marker.get("identity_sha256") == identity_sha256
        and marker.get("checkpoint") == "checkpoints/latest.pt"
        and marker.get("checkpoint_sha256") == checkpoint_sha256
        and marker.get("evaluation_seed_tables_sha256") == hashes["evaluation_seed_tables_sha256"]
        and not (run_dir / "COMPLETED.json").exists(),
        f"Exp20 boundary marker differs: {run_dir}@{target}",
    )
    require(
        complete.get("schema_version") == 1
        and complete.get("status") == "stage_complete_awaiting_campaign_gate"
        and complete.get("campaign_id") == launch["campaign_id"]
        and int(complete.get("stage_slot", -1)) == stage_slot
        and int(complete.get("index", -1)) == int(run["index"])
        and complete.get("setting_id") == run["setting_id"]
        and complete.get("arm_id") == run["arm_id"]
        and int(complete.get("seed", -1)) == int(run["seed"])
        and int(complete.get("stage_target", -1)) == target
        and complete.get("launch_sha256") == launch["launch_sha256"]
        and complete.get("package_protocol_sha256") == hashes["package_protocol_sha256"]
        and complete.get("source_sha256") == hashes["source_sha256"]
        and complete.get("runtime_sha256") == hashes["runtime_sha256"]
        and complete.get("identity_sha256") == identity_sha256
        and complete.get("checkpoint_sha256") == checkpoint_sha256
        and complete.get("evaluation_seed_tables_sha256") == hashes["evaluation_seed_tables_sha256"]
        and complete.get("final_seed_table_sha256") == hashes["final_seed_table_sha256"],
        f"Exp20 stage completion differs: {run_dir}@{target}",
    )
    require(
        stage_launch.get("schema_version") == 1
        and stage_launch.get("status") == "stage_launch_sealed"
        and int(stage_launch.get("stage_target", -1)) == target
        and int(stage_launch.get("stage_slot", -1)) == stage_slot
        and stage_launch.get("launch_sha256") == launch["launch_sha256"]
        and stage_launch.get("package_protocol_sha256") == hashes["package_protocol_sha256"]
        and stage_launch.get("previous_gate")
        == (None if target == 5_000 else dict(expected_previous_gate or {})),
        f"Exp20 stage launch differs: {run_dir}@{target}",
    )
    bound = {
        str(path.resolve()): file_sha256(path)
        for path in (marker_path, complete_path, stage_launch_path)
    }
    if current_checkpoint:
        checkpoint_path = run_dir / "checkpoints/latest.pt"
        bound[str(checkpoint_path.resolve())] = checkpoint_sha256
    return checkpoint, bound


def _validated_exp21_post_update_cadence(
    payload: Mapping[str, Any],
) -> dict[str, int | str | None]:
    """Validate the new Exp21 cadence field without tightening legacy Exp20."""
    from treewm.utils.checkpoint import PostUpdateCadenceState

    try:
        cadence = PostUpdateCadenceState.from_state_dict(
            payload.get("post_update_cadence"),
            require_durable=True,
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"Exp21 checkpoint post-update cadence is invalid: {exc}") from exc
    require(
        cadence.committed_update == 25_000,
        "Exp21 checkpoint post-update cadence is not committed at 25000",
    )
    require(
        cadence.complete,
        "Exp21 terminal checkpoint post-update cadence is incomplete",
    )
    return cadence.state_dict()


def _validate_exp21_checkpoint(
    path: Path,
    launch: Mapping[str, Any],
    row: Mapping[str, Any],
    forbidden_tokens: Sequence[str],
) -> dict[str, Any]:
    import torch

    require(path.is_file() and not path.is_symlink(), f"Exp21 checkpoint missing/linked: {path}")
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ContractError(f"Exp21 checkpoint cannot be loaded: {exc}") from exc
    from treewm.utils.checkpoint import validate_exact_resume_payload

    try:
        validate_exact_resume_payload(
            payload,
            expected_identity=payload.get("run_identity"),
            expected_world_size=1,
            require_cuda_rng=True,
        )
    except (TypeError, ValueError) as exc:
        raise ContractError(f"Exp21 checkpoint exact-resume payload is invalid: {exc}") from exc
    _validated_exp21_post_update_cadence(payload)
    identity = payload.get("run_identity") or {}
    config = payload.get("config") or {}
    reject_forbidden(identity, forbidden_tokens, "Exp21 checkpoint run identity")
    reject_forbidden(config, forbidden_tokens, "Exp21 checkpoint config")
    run = launch["run"]
    hashes = launch["hashes"]
    expected_factor = f"exp20-{run['selected_arm']}-all-ten"
    checks = (
        payload.get("schema_version") == 2,
        int(payload.get("completed_updates", -1)) == 25_000,
        int(payload.get("step", -1)) == 25_000,
        int(payload.get("next_step", -1)) == 25_000,
        payload.get("optimizer") is not None,
        payload.get("scheduler") is not None,
        _rank_state_complete(payload),
        sha(payload.get("identity_sha256")),
        payload.get("identity_sha256") == row.get("identity_sha256"),
        identity.get("run_name") == run["run_name"],
        identity.get("setting") == run["setting_id"],
        int(identity.get("seed", -1)) == int(run["seed"]),
        identity.get("objective_version") == "treewm_v2_grounded_gauge_all_ten_bridge_v2",
        int(identity.get("total_steps", -1)) == 25_000,
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
        identity.get("campaign_factorial_arm") == expected_factor,
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
    require(all(checks), f"Exp21 checkpoint exact-resume identity differs: {path}")
    checkpoint_sha256 = file_sha256(path)
    require(checkpoint_sha256 == row.get("checkpoint_sha256"), f"Exp21 checkpoint bytes differ: {path}")
    return {
        "identity_sha256": payload["identity_sha256"],
        "evaluation_seed_tables_sha256": identity["evaluation_seed_tables_sha256"],
        "final_seed_table_sha256": identity["final_seed_table_sha256"],
        "checkpoint_sha256": checkpoint_sha256,
    }


def _validate_exp21_stage_artifacts(
    run_dir: Path,
    launch: Mapping[str, Any],
    row: Mapping[str, Any],
    forbidden_tokens: Sequence[str],
) -> tuple[dict[str, Any], dict[str, str]]:
    checkpoint_path = run_dir / "checkpoints/latest.pt"
    checkpoint = _validate_exp21_checkpoint(checkpoint_path, launch, row, forbidden_tokens)
    marker_path = run_dir / "stage-gates/AWAITING_GATE_25000.json"
    complete_path = run_dir / "stage-gates/STAGE_COMPLETE_25000.json"
    stage_launch_path = run_dir / "stage-gates/STAGE_LAUNCH_25000.json"
    for path, label in ((marker_path, "marker"), (complete_path, "completion"), (stage_launch_path, "stage launch")):
        require(path.is_file() and not path.is_symlink(), f"Exp21 {label} missing/linked: {path}")
    marker = read_json(marker_path)
    complete = read_json(complete_path)
    stage_launch = read_json(stage_launch_path)
    reject_forbidden((marker, complete, stage_launch), forbidden_tokens, "Exp21 lifecycle artifacts")
    self_hash(complete, "stage_complete_sha256", "Exp21 stage completion")
    run = launch["run"]
    hashes = launch["hashes"]
    require(
        marker.get("schema_version") == 1
        and marker.get("status") == "awaiting_external_stage_gate"
        and marker.get("objective_version") == "treewm_v2_grounded_gauge_all_ten_bridge_v2"
        and int(marker.get("completed_updates", -1)) == 25_000
        and int(marker.get("step", -1)) == 25_000
        and int(marker.get("total_steps", -1)) == 25_000
        and int(marker.get("scheduler_total_steps", -1)) == 1_000_000
        and marker.get("identity_sha256") == checkpoint["identity_sha256"]
        and marker.get("checkpoint") == "checkpoints/latest.pt"
        and marker.get("checkpoint_sha256") == checkpoint["checkpoint_sha256"]
        and marker.get("evaluation_seed_tables_sha256") == checkpoint["evaluation_seed_tables_sha256"]
        and not (run_dir / "COMPLETED.json").exists(),
        f"Exp21 boundary marker differs: {run_dir}",
    )
    require(
        complete.get("schema_version") == 1
        and complete.get("status") == "stage_complete_awaiting_campaign_gate"
        and complete.get("campaign_id") == launch["campaign_id"]
        and int(complete.get("index", -1)) == int(run["index"])
        and complete.get("setting_id") == run["setting_id"]
        and int(complete.get("seed", -1)) == int(run["seed"])
        and complete.get("selected_arm") == run["selected_arm"]
        and int(complete.get("stage_target", -1)) == 25_000
        and complete.get("launch_sha256") == launch["launch_sha256"]
        and complete.get("package_protocol_sha256") == hashes["package_protocol_sha256"]
        and complete.get("source_sha256") == hashes["source_sha256"]
        and complete.get("runtime_sha256") == hashes["runtime_sha256"]
        and complete.get("exp20_binding_sha256") == hashes["exp20_binding_sha256"]
        and complete.get("selected_recipe_sha256") == hashes["selected_recipe_sha256"]
        and complete.get("identity_sha256") == checkpoint["identity_sha256"] == row.get("identity_sha256")
        and complete.get("checkpoint_sha256") == checkpoint["checkpoint_sha256"]
        and complete.get("evaluation_seed_tables_sha256") == hashes["evaluation_seed_tables_sha256"]
        and complete.get("final_seed_table_sha256") == hashes["final_seed_table_sha256"],
        f"Exp21 stage completion differs: {run_dir}",
    )
    require(
        stage_launch == {
            "schema_version": 1,
            "status": "fresh_stage_launch_sealed",
            "stage_target": 25_000,
            "index": run["index"],
            "launch_sha256": launch["launch_sha256"],
            "package_protocol_sha256": hashes["package_protocol_sha256"],
            "exp20_binding_sha256": hashes["exp20_binding_sha256"],
            "selected_recipe_sha256": hashes["selected_recipe_sha256"],
        },
        f"Exp21 stage launch differs: {run_dir}",
    )
    bound = {
        str(path.resolve()): file_sha256(path)
        for path in (checkpoint_path, marker_path, complete_path, stage_launch_path)
    }
    return checkpoint, bound


def _exp21_monitor_bank(exp21_manifest: Mapping[str, Any]) -> dict[str, Any]:
    scientific = exp21_manifest["scientific_contract"]
    task_ids = list(exp21_manifest["design"]["task_ids"])
    episodes = int(scientific["periodic_episodes_per_task"])
    base = int(scientific["evaluation_seed"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "policy": "fixed_cfg_eval_seed_fallback_periodic_monitor",
        "stage_target": 25_000,
        "task_ids": task_ids,
        "episodes_per_task": episodes,
        "seeds": [
            [base + 1_000 * task_index + episode for episode in range(episodes)]
            for task_index, _task_id in enumerate(task_ids)
        ],
    }
    payload["sha256"] = stable_hash(payload)
    return payload


def _exp21_evidence(
    acceptance: Mapping[str, Any],
    exp21_manifest: Mapping[str, Any],
    *,
    selected_arm: str,
    selected_recipe: Mapping[str, Any],
    selected_recipe_sha256: str,
    upstream_binding_sha256: str,
    package_protocol_sha256: str,
    source_sha256: str,
    runtime_sha256: str,
    evaluation_bank_sha256: str,
    forbidden_tokens: Sequence[str],
    snapshot_repo: Path,
) -> tuple[dict[str, Any], dict[str, str], list[dict[str, Any]]]:
    rows = acceptance.get("runs") or []
    require(len(rows) == 20, "Exp21 acceptance lacks all 20 raw rows")
    run_root = Path(exp21_manifest["paths"]["run_root"])
    result: dict[str, Any] = {}
    bound: dict[str, str] = {}
    derived_rows: list[dict[str, Any]] = []
    settings = [str(row["id"]) for row in exp21_manifest["settings"]]
    seeds = list(exp21_manifest["design"]["seeds"])
    monitor = _exp21_monitor_bank(exp21_manifest)
    require(
        acceptance.get("prospective_monitor_bank") == monitor
        and monitor.get("sha256") == evaluation_bank_sha256,
        "Exp21 prospective monitor bank differs from the independently generated bank",
    )
    for row in rows:
        require(isinstance(row, dict) and set(row) == {
            "index", "run_name", "setting_id", "seed", "selected_arm",
            "launch_sha256", "identity_sha256", "checkpoint_sha256",
            "health", "outcome",
        }, "Exp21 acceptance run-row schema differs")
        require(row.get("setting_id") in settings and row.get("seed") in seeds, "Exp21 run row is outside the sealed matrix")
        require(row.get("selected_arm") == selected_arm, "Exp21 run row selected arm differs")
        require(all(sha(row.get(name)) for name in ("launch_sha256", "identity_sha256", "checkpoint_sha256")), "Exp21 run row hash malformed")
        key = f"{row['setting_id']}|{int(row['seed'])}"
        run_dir = run_root / str(row["setting_id"]) / "treewm" / str(row["run_name"])
        launch_file, launch = launch_record(run_dir / "GAUGE_BRIDGE_LAUNCH.json", str(row["launch_sha256"]))
        reject_forbidden(launch, forbidden_tokens, f"Exp21 raw launch {key}")
        setting = str(row["setting_id"])
        seed = int(row["seed"])
        expected_index = settings.index(setting) * len(seeds) + seeds.index(seed)
        require(int(row["index"]) == expected_index, f"Exp21 acceptance row index differs: {key}")
        run = launch.get("run") or {}
        hashes = launch.get("hashes") or {}
        require(launch.get("campaign_id") == exp21_manifest["campaign_id"] and launch.get("formal_validation") is False, f"Exp21 launch claim differs: {key}")
        expected_setting_index = settings.index(setting)
        expected_seed_index = seeds.index(seed)
        setting_row = next(item for item in exp21_manifest["settings"] if item["id"] == setting)
        require(
            run.get("index") == expected_index
            and run.get("setting_index") == expected_setting_index
            and run.get("seed_index") == expected_seed_index
            and run.get("setting_id") == setting
            and run.get("env_config") == setting_row["env_config"]
            and int(run.get("seed", -1)) == seed,
            f"Exp21 launch matrix differs: {key}",
        )
        require(run.get("run_name") == row.get("run_name") and run.get("selected_arm") == selected_arm, f"Exp21 launch selected identity differs: {key}")
        require(
            run.get("wandb_id") == stable_hash({"campaign_id": exp21_manifest["campaign_id"], "setting_id": setting, "seed": seed})[:32]
            and run.get("run_directory") == str(run_dir),
            f"Exp21 run/W&B namespace differs: {key}",
        )
        expected_hashes = {
            "manifest_sha256": stable_hash(exp21_manifest),
            "package_protocol_sha256": package_protocol_sha256,
            "source_sha256": source_sha256,
            "runtime_sha256": runtime_sha256,
            "exp20_binding_sha256": upstream_binding_sha256,
            "selected_recipe_sha256": selected_recipe_sha256,
            "actual_evaluation_bank_sha256": evaluation_bank_sha256,
        }
        require(all(hashes.get(name) == value for name, value in expected_hashes.items()), f"Exp21 launch provenance differs: {key}")
        for name in (
            "config_sha256", "run_protocol_sha256", "input_contract_sha256",
            "data_manifest_sha256", "normalizer_sha256", "train_manifest_sha256",
            "validation_manifest_sha256", "calibration_sha256", "future_recipe_sha256",
            "recipe_code_sha256", "recipe_runtime_sha256",
            "evaluation_seed_tables_sha256", "final_seed_table_sha256",
        ):
            require(sha(hashes.get(name)), f"Exp21 launch {name} malformed: {key}")
        argv = launch.get("argv") or []
        require(isinstance(argv, list) and all(isinstance(value, str) for value in argv), f"Exp21 launch argv differs: {key}")
        overrides, ordered_tokens = _argv_overrides(argv, f"Exp21 {key}")
        method = exp21_manifest["method"]
        scientific = exp21_manifest["scientific_contract"]
        expected_overrides = {
            "env": setting_row["env_config"],
            "experiment": method["experiment_config"],
            "arm": method["arm"],
            "objective_version": method["objective_version"],
            "seed": seed,
            "train.steps": scientific["optimizer_updates"],
            "train.scheduler_total_steps": scientific["scheduler_total_steps"],
            "train.lr": selected_recipe["world_lr"],
            "train.separate_gain_grad_clip": True,
            "train.separate_branch_transformer_grad_clip": selected_recipe["separate_branch_transformer_grad_clip"],
            "train.world_grad_clip": 1.0,
            "train.gain_grad_clip": 1.0,
            "train.branch_transformer_grad_clip": selected_recipe["branch_transformer_grad_clip"],
            "losses.keep_balance": scientific["keep_balance"],
            "losses.enabled.multistep": scientific["multistep_enabled"],
            "losses.scheduled_sampling_p": scientific["scheduled_sampling_p"],
            "losses.scheduled_sampling_warmup": scientific["scheduled_sampling_warmup"],
            "losses.scheduled_sampling_granularity": scientific["scheduled_sampling_granularity"],
            "losses.multistep_transition_mode": selected_recipe["transition_mode"],
            "losses.grounded_select_action_weight": selected_recipe["grounded_select_action_weight"],
            "losses.grounded_select_endpoint_weight": selected_recipe["grounded_select_endpoint_weight"],
            "losses.grounded_select_horizon_weight": selected_recipe["grounded_select_horizon_weight"],
            "losses.grounded_loss_latent_weight": selected_recipe["grounded_loss_latent_weight"],
            "losses.grounded_loss_action_weight": selected_recipe["grounded_loss_action_weight"],
            "losses.grounded_loss_horizon_weight": selected_recipe["grounded_loss_horizon_weight"],
            "losses.grounded_loss_endpoint_weight": selected_recipe["grounded_loss_endpoint_weight"],
            "losses.grounded_detach_self_fed_parent": scientific["grounded_detach_self_fed_parent"],
            "losses.enabled.latent_gauge": selected_recipe["latent_gauge_enabled"],
            "losses.weights.latent_gauge": selected_recipe["latent_gauge_weight"],
            "planner.decoded_metric": scientific["planner_decoded_metric"],
            "planner.execute_mode": scientific["planner_execute_mode"],
            "planner.execute_steps": scientific["planner_execute_steps"],
            "future_sets.recipe_anchor_policy": scientific["future_config"]["recipe_anchor_policy"],
            "eval.task_split": scientific["task_split"],
            "eval.episodes_per_task": scientific["periodic_episodes_per_task"],
            "eval.final_episodes_per_task": scientific["final_episodes_per_task"],
            "+campaign_factorial_arm": f"exp20-{selected_arm}-all-ten",
            "+campaign_prerequisite_binding_sha256": upstream_binding_sha256,
            "+campaign_exp20_binding_sha256": upstream_binding_sha256,
            "+campaign_selected_recipe_sha256": selected_recipe_sha256,
            "run_root": exp21_manifest["paths"]["run_root"],
            "run_name": row["run_name"],
            "resume": "auto",
            "+campaign_source_sha256": source_sha256,
            "+campaign_protocol_sha256": package_protocol_sha256,
            "+campaign_config_sha256": hashes["config_sha256"],
            "hydra.job.chdir": False,
        }
        require(
            all(overrides.get(name) == _render_override(value) for name, value in expected_overrides.items()),
            f"Exp21 launch exact selected recipe differs: {key}",
        )
        run_root_position = next((position for position, token in enumerate(ordered_tokens) if token.startswith("run_root=")), None)
        require(run_root_position is not None, f"Exp21 launch lacks config boundary: {key}")
        require(
            stable_hash({"schema_version": 1, "overrides": ordered_tokens[:run_root_position]}) == hashes["config_sha256"],
            f"Exp21 launch config hash is not argv-derived: {key}",
        )
        expected_entry = snapshot_repo / "experiments/21-treewm-grounded-gauge-all-ten-bridge-v2/train_entry.py"
        require(
            len(argv) >= 2
            and argv[0] == exp21_manifest["paths"]["python"]
            and Path(argv[1]) == expected_entry,
            f"Exp21 launch did not use its exact immutable snapshot train_entry.py: {key}",
        )
        input_contract = load_compatible_input(
            exp21_manifest, setting_row, verify_files=False
        )
        expected_scientific_overrides = _expected_exp21_overrides(
            exp21_manifest,
            setting_row,
            seed,
            input_contract,
            selected_recipe,
            selected_arm,
            upstream_binding_sha256,
            selected_recipe_sha256,
        )
        expected_argv = [
            exp21_manifest["paths"]["python"],
            str(expected_entry),
            *expected_scientific_overrides,
            _override("run_root", exp21_manifest["paths"]["run_root"]),
            _override("run_name", row["run_name"]),
            _override("resume", "auto"),
            _override("+campaign_source_sha256", source_sha256),
            _override("+campaign_protocol_sha256", package_protocol_sha256),
            _override("+campaign_config_sha256", hashes["config_sha256"]),
            _override("hydra.run.dir", run_dir / "hydra"),
            _override("hydra.job.chdir", False),
        ]
        require(argv == expected_argv, f"Exp21 full ordered trainer argv differs: {key}")
        require(
            hashes["config_sha256"] == stable_hash({
                "schema_version": 1,
                "overrides": expected_scientific_overrides,
            }),
            f"Exp21 full expected config hash differs: {key}",
        )
        for hash_name, contract_name in (
            ("input_contract_sha256", "contract_sha256"),
            ("data_manifest_sha256", "data_manifest_sha256"),
            ("normalizer_sha256", "normalizer_sha256"),
            ("train_manifest_sha256", "train_manifest_sha256"),
            ("validation_manifest_sha256", "validation_manifest_sha256"),
            ("calibration_sha256", "calibration_sha256"),
            ("future_recipe_sha256", "future_recipe_sha256"),
        ):
            require(hashes[hash_name] == input_contract[contract_name], f"Exp21 input {hash_name} differs: {key}")
        require(
            hashes["recipe_code_sha256"] == exp21_manifest["compatible_v2_recipe_input"]["recipe_code_sha256"]
            and hashes["recipe_runtime_sha256"] == exp21_manifest["compatible_v2_recipe_input"]["recipe_runtime_sha256"],
            f"Exp21 compatible recipe provenance differs: {key}",
        )
        expected_run_protocol = stable_hash({
            "schema_version": 1,
            "campaign_id": exp21_manifest["campaign_id"],
            "package_protocol_sha256": package_protocol_sha256,
            "source_sha256": source_sha256,
            "runtime_sha256": runtime_sha256,
            "config_sha256": hashes["config_sha256"],
            "exp20_binding_sha256": upstream_binding_sha256,
            "selected_recipe_sha256": selected_recipe_sha256,
            "input_contract_sha256": hashes["input_contract_sha256"],
            "data_manifest_sha256": hashes["data_manifest_sha256"],
            "normalizer_sha256": hashes["normalizer_sha256"],
            "train_manifest_sha256": hashes["train_manifest_sha256"],
            "validation_manifest_sha256": hashes["validation_manifest_sha256"],
            "calibration_sha256": hashes["calibration_sha256"],
            "future_recipe_sha256": hashes["future_recipe_sha256"],
            "actual_evaluation_bank_sha256": evaluation_bank_sha256,
        })
        require(hashes["run_protocol_sha256"] == expected_run_protocol, f"Exp21 run protocol differs: {key}")
        from treewm.evaluation.rollout import build_evaluation_seed_tables

        expected_seed_tables = build_evaluation_seed_tables(
            scientific["evaluation_seed_protocol_sha256"],
            seed,
            exp21_manifest["design"]["task_ids"],
            scientific["periodic_episodes_per_task"],
            scientific["final_episodes_per_task"],
        )
        require(
            hashes["evaluation_seed_tables_sha256"] == expected_seed_tables["sha256"]
            and hashes["final_seed_table_sha256"] == expected_seed_tables["final"]["sha256"],
            f"Exp21 launch seed-table hashes differ: {key}",
        )
        environment = launch.get("environment") or {}
        require(
            environment.get("TREEWM_PROTOCOL_SHA256") == hashes["run_protocol_sha256"]
            and environment.get("TREEWM_CODE_SHA256") == source_sha256
            and environment.get("TREEWM_ACTIVE_SOURCE_SHA256") == source_sha256
            and environment.get("TREEWM_RUNTIME_SHA256") == runtime_sha256
            and environment.get("TREEWM_CONFIG_SHA256") == hashes["config_sha256"]
            and environment.get("TREEWM_PREREQUISITE_BINDING_SHA256") == upstream_binding_sha256
            and environment.get("TREEWM_SELECTED_RECIPE_SHA256") == selected_recipe_sha256
            and environment.get("TREEWM_RECIPE_CODE_SHA256") == hashes["recipe_code_sha256"]
            and environment.get("TREEWM_RECIPE_RUNTIME_SHA256") == hashes["recipe_runtime_sha256"]
            and environment.get("TREEWM_DATA_SHA256") == hashes["data_manifest_sha256"]
            and environment.get("TREEWM_CALIBRATION_SHA256") == hashes["calibration_sha256"]
            and environment.get("TREEWM_FUTURE_RECIPE_SHA256") == hashes["future_recipe_sha256"]
            and environment.get("TREEWM_DATA_CONTRACT_SHA256") == hashes["input_contract_sha256"]
            and environment.get("TREEWM_EVALUATION_SEED_PROTOCOL_SHA256") == scientific["evaluation_seed_protocol_sha256"]
            and environment.get("TREEWM_EXPECTED_FINAL_SEED_TABLE_SHA256") == expected_seed_tables["final"]["sha256"]
            and environment.get("WANDB_PROJECT") == exp21_manifest["logging"]["wandb_project"]
            and environment.get("WANDB_RUN_GROUP") == exp21_manifest["logging"]["wandb_group"]
            and environment.get("WANDB_RUN_ID") == run["wandb_id"],
            f"Exp21 launch environment/provenance differs: {key}",
        )
        events = event_records(run_dir)
        metrics = raw_exp20_recompute.event_scalars([Path(item["path"]) for item in events])
        derived_health = raw_exp20_recompute.evaluate_exp20_metrics(exp21_manifest, metrics, 25_000, selected_arm)
        reported_health = row.get("health") or {}
        exp21_only_health = {
            name: value for name, value in derived_health.items()
            if name not in {
                "structural_integrity_passed", "structural_integrity_gates",
                "clip_coefficients_valid", "clip_saturation_bounded",
            }
        }
        require(reported_health == exp21_only_health, f"Exp21 health differs from complete raw-event derivation: {key}")
        require(derived_health["candidate_passed"], f"Exp21 raw method/gauge gate failed: {key}")
        expected_outcome = {
            "num_episodes": _metric_at(metrics, "eval/num_episodes", 25_000),
            "successes": _metric_at(metrics, "eval/successes", 25_000),
            "success_rate": _metric_at(metrics, "eval/success_rate", 25_000),
            "distance_reduction_frac": _metric_at(metrics, "eval/distance_reduction_frac", 25_000),
            "prospective_monitor_bank_sha256": evaluation_bank_sha256,
            "tasks": [
                {
                    "task_id": task_id,
                    "episode_seed": monitor["seeds"][task_id - 1][0],
                    "num_episodes": _metric_at(metrics, f"eval/task{task_id}/num_episodes", 25_000),
                    "successes": _metric_at(metrics, f"eval/task{task_id}/successes", 25_000),
                    "success_rate": _metric_at(metrics, f"eval/task{task_id}/success_rate", 25_000),
                }
                for task_id in (1, 2, 3, 4, 5)
            ],
        }
        require(row.get("outcome") == expected_outcome, f"Exp21 outcome differs from raw events: {key}")
        checkpoint, lifecycle_files = _validate_exp21_stage_artifacts(
            run_dir, launch, row, forbidden_tokens
        )
        result[key] = {
            "launch_file_sha256": launch_file["sha256"],
            "launch_sha256": launch["launch_sha256"],
            "event_evidence_sha256": stable_hash(events),
            "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        }
        bound[launch_file["path"]] = launch_file["sha256"]
        bound.update(lifecycle_files)
        for item in events:
            bound[item["path"]] = item["sha256"]
        derived_rows.append(dict(row))
    require(len(result) == 20, "Exp21 raw evidence matrix duplicated")
    require(_exp21_outcome_summary(exp21_manifest, derived_rows) == acceptance.get("outcome"), "Exp21 aggregate outcome differs from raw rows")
    return result, bound, derived_rows


def build_binding(manifest: Mapping[str, Any]) -> dict[str, Any]:
    prerequisites = manifest["prerequisites"]
    exp20_contract = prerequisites["exp20"]
    exp21_contract = prerequisites["exp21"]
    tokens = tuple(prerequisites["forbidden_ancestry_tokens"])
    exp21_dir = REPOSITORY_ROOT / "experiments/21-treewm-grounded-gauge-all-ten-bridge-v2"
    exp21_protocol = verify_exp21_protocol(exp21_dir)
    exp21_manifest_path = exp21_dir / "manifest.json"
    exp21_manifest = read_json(exp21_manifest_path)
    require(stable_hash(exp21_manifest) == exp21_contract["manifest_sha256"], "Exp21 manifest differs from pin")
    require(exp21_manifest.get("campaign_id") == exp21_contract["campaign_id"], "Exp21 manifest identity differs")

    gate_5000_path = Path(exp20_contract["stage_5000_gate_path"])
    gate_25000_path = Path(exp20_contract["acceptance_path"])
    exp21_upstream_path = Path(exp21_contract["exp20_binding_path"])
    exp21_acceptance_path = Path(exp21_contract["acceptance_path"])
    for path, label in (
        (gate_5000_path, "Exp20 5k gate"),
        (gate_25000_path, "Exp20 25k acceptance"),
        (exp21_upstream_path, "Exp21 Exp20 binding"),
        (exp21_acceptance_path, "Exp21 acceptance"),
    ):
        require(path.is_file() and not path.is_symlink(), f"{label} missing/symlinked: {path}")

    gate_5000 = read_json(gate_5000_path)
    gate_25000 = read_json(gate_25000_path)
    upstream = read_json(exp21_upstream_path)
    acceptance = read_json(exp21_acceptance_path)
    for value, label in ((gate_5000, "Exp20 5k gate"), (gate_25000, "Exp20 acceptance"), (upstream, "Exp21 upstream binding"), (acceptance, "Exp21 acceptance")):
        reject_forbidden(value, tokens, label)

    gate_5000_hash = self_hash(gate_5000, "gate_sha256", "Exp20 5k gate")
    gate_25000_hash = self_hash(gate_25000, "gate_sha256", "Exp20 acceptance")
    upstream_hash = self_hash(upstream, "binding_sha256", "Exp21 upstream binding")
    acceptance_hash = self_hash(acceptance, "acceptance_sha256", "Exp21 acceptance")
    require(set(acceptance) == {
        "schema_version", "status", "campaign_id", "formal_validation",
        "stage_target", "claim", "selected_arm", "selected_recipe",
        "selected_recipe_sha256", "method_runs_passing",
        "required_method_runs", "outcome", "prospective_monitor_bank",
        "runs", "provenance", "acceptance_sha256",
    }, "Exp21 acceptance top-level schema differs")
    require(
        acceptance.get("schema_version") == 1
        and acceptance.get("campaign_id") == exp21_contract["campaign_id"]
        and int(acceptance.get("stage_target", -1)) == 25_000
        and acceptance.get("claim") == "Bounded all-ten 25k gauge bridge only; never a 1M or formal result.",
        "Exp21 acceptance identity/claim differs",
    )

    # Exp22 carries its own copy of the complete Exp20 scalar/gate derivation.
    # Recompute from the thirty TensorBoard streams rather than accepting either
    # Exp20's gate booleans or Exp21's upstream binding claims.
    local_exp20_contract = dict(exp20_contract)
    local_exp20_contract["required_status"] = exp20_contract["required_acceptance_status"]
    local_exp20_contract["forbidden_ancestry_tokens"] = list(tokens)
    exp20_manifest = raw_exp20_recompute.load_pinned_exp20_manifest(local_exp20_contract)
    exp20_metrics, local_raw_evidence = raw_exp20_recompute.collect_raw_evidence(
        local_exp20_contract, exp20_manifest
    )
    local_selected_arm, local_5000_rows = raw_exp20_recompute.recompute_stage_5000(
        local_exp20_contract, gate_5000, exp20_manifest, exp20_metrics
    )
    raw_exp20_recompute.recompute_acceptance(
        local_exp20_contract,
        gate_25000,
        gate_5000_hash,
        local_selected_arm,
        local_5000_rows,
        exp20_manifest,
        exp20_metrics,
    )

    # Exp21's published upstream receipt must agree with the independent local
    # collection, but no Exp21 executable is trusted as an authorization oracle.
    require(
        upstream.get("exp20", {}).get("raw_evidence") == local_raw_evidence,
        "Exp21 raw-evidence inventory differs from Exp22's independent Exp20 collection",
    )

    require(gate_5000.get("status") == exp20_contract["required_stage_5000_status"], "Exp20 5k status differs")
    require(gate_25000.get("status") == exp20_contract["required_acceptance_status"], "Exp20 25k status differs")
    require(acceptance.get("status") == exp21_contract["required_status"], "Exp21 status differs")
    require(gate_5000.get("formal_validation") is False and gate_25000.get("formal_validation") is False and acceptance.get("formal_validation") is False, "bounded prerequisites claim formal validation")
    for key in ("package_protocol_sha256", "source_sha256", "runtime_sha256", "actual_evaluation_bank_sha256"):
        require(gate_5000.get(key) == gate_25000.get(key) == exp20_contract[key], f"Exp20 {key} differs")
    selected_arm = str(gate_25000.get("selected_arm"))
    require(selected_arm == local_selected_arm, "Exp22-local Exp20 selection differs")
    require(gate_5000.get("selected_arm") == selected_arm == upstream.get("selected_arm") == acceptance.get("selected_arm"), "Exp20/Exp21 selected arms differ")
    require(selected_arm in ("G", "GS"), "selected arm is not G/GS")
    recipe = prerequisites["allowed_selected_recipes"][selected_arm]
    recipe_hash = stable_hash(recipe)
    require(upstream.get("selected_recipe") == acceptance.get("selected_recipe") == recipe, "accepted recipe differs from exact selected-arm recipe")
    require(upstream.get("selected_recipe_sha256") == acceptance.get("selected_recipe_sha256") == recipe_hash, "accepted recipe hash differs")
    provenance = acceptance.get("provenance") or {}
    require(provenance.get("package_protocol_sha256") == exp21_protocol, "Exp21 protocol provenance differs from live accepted lock")
    require(provenance.get("source_sha256") == exp21_contract["source_sha256"], "Exp21 source provenance differs")
    require(provenance.get("runtime_sha256") == exp21_contract["runtime_sha256"], "Exp21 runtime provenance differs")
    require(provenance.get("actual_evaluation_bank_sha256") == exp21_contract["actual_evaluation_bank_sha256"], "Exp21 evaluation bank differs")
    require(provenance.get("exp20_binding_sha256") == upstream_hash, "Exp21 acceptance does not bind exact Exp20 replay")
    require(acceptance.get("method_runs_passing") == acceptance.get("required_method_runs") == 20, "Exp21 method gate is not 20/20")
    outcome = acceptance.get("outcome") or {}
    require(outcome.get("settings_with_both_seed_positive_progress") >= 6, "Exp21 6/10 progress quorum failed")
    require(all(float(row["successes"]) >= 1 and float(row["mean_distance_reduction_frac"]) > 0 for row in (outcome.get("per_seed") or {}).values()), "Exp21 per-seed success/progress failed")
    require(outcome.get("settings_with_both_seed_nonzero_success") >= 1, "Exp21 replicated success failed")

    exp20_5k_evidence, exp20_25k_evidence, skipped, bound_files = _exp20_evidence(
        upstream, gate_5000, gate_25000, tokens
    )
    exp20_snapshot_repo = raw_exp20_recompute._expected_exp20_snapshot_repo(
        local_exp20_contract, exp20_manifest
    )
    for path in (
        exp20_snapshot_repo.parent / "SNAPSHOT.json",
        exp20_snapshot_repo / "scripts/train.py",
        exp20_snapshot_repo / "experiments/20-treewm-grounded-gauge-pilot-v2/protocol.sha256",
        REPOSITORY_ROOT / "experiments/20-treewm-grounded-gauge-pilot-v2/manifest.json",
        REPOSITORY_ROOT / "experiments/20-treewm-grounded-gauge-pilot-v2/protocol.sha256",
    ):
        require(path.is_file() and not path.is_symlink(), f"Exp20 source/protocol evidence missing/linked: {path}")
        bound_files[str(path.resolve())] = file_sha256(path)
    exp21_snapshot_repo, exp21_snapshot_files = _verify_exp21_source_snapshot(
        exp21_manifest,
        package_protocol_sha256=exp21_protocol,
        source_sha256=exp21_contract["source_sha256"],
        runtime_sha256=exp21_contract["runtime_sha256"],
    )
    exp21_evidence, exp21_files, exp21_derived_rows = _exp21_evidence(
        acceptance,
        exp21_manifest,
        selected_arm=selected_arm,
        selected_recipe=recipe,
        selected_recipe_sha256=recipe_hash,
        upstream_binding_sha256=upstream_hash,
        package_protocol_sha256=exp21_protocol,
        source_sha256=exp21_contract["source_sha256"],
        runtime_sha256=exp21_contract["runtime_sha256"],
        evaluation_bank_sha256=exp21_contract["actual_evaluation_bank_sha256"],
        forbidden_tokens=tokens,
        snapshot_repo=exp21_snapshot_repo,
    )
    bound_files.update(exp21_snapshot_files)
    bound_files.update(exp21_files)
    for path in (gate_5000_path, gate_25000_path, exp21_upstream_path, exp21_acceptance_path, exp21_manifest_path, exp21_dir / "protocol.sha256"):
        bound_files[str(path.resolve())] = file_sha256(path)

    exp20_payload = {
        "campaign_id": exp20_contract["campaign_id"],
        "stage_5000_status": gate_5000["status"],
        "acceptance_status": gate_25000["status"],
        "stage_5000_gate_sha256": gate_5000_hash,
        "stage_5000_gate_file_sha256": file_sha256(gate_5000_path),
        "acceptance_gate_sha256": gate_25000_hash,
        "acceptance_file_sha256": file_sha256(gate_25000_path),
        "manifest_sha256": exp20_contract["manifest_sha256"],
        "package_protocol_sha256": exp20_contract["package_protocol_sha256"],
        "source_sha256": exp20_contract["source_sha256"],
        "runtime_sha256": exp20_contract["runtime_sha256"],
        "actual_evaluation_bank_sha256": exp20_contract["actual_evaluation_bank_sha256"],
        "selected_arm": selected_arm,
        "selected_recipe": recipe,
        "selected_recipe_sha256": recipe_hash,
        "stage_5000_run_evidence": exp20_5k_evidence,
        "stage_25000_run_evidence": exp20_25k_evidence,
        "stage_25000_skipped_runs": skipped,
        "raw_recomputation_sha256": stable_hash({
            "selected_arm": local_selected_arm,
            "stage_5000_gate_sha256": gate_5000_hash,
            "acceptance_gate_sha256": gate_25000_hash,
            "raw_evidence": local_raw_evidence,
        }),
    }
    exp21_payload = {
        "campaign_id": exp21_contract["campaign_id"],
        "acceptance_status": acceptance["status"],
        "acceptance_sha256": acceptance_hash,
        "acceptance_file_sha256": file_sha256(exp21_acceptance_path),
        "exp20_binding_sha256": upstream_hash,
        "exp20_binding_file_sha256": file_sha256(exp21_upstream_path),
        "manifest_sha256": exp21_contract["manifest_sha256"],
        "package_protocol_sha256": exp21_protocol,
        "source_sha256": exp21_contract["source_sha256"],
        "runtime_sha256": exp21_contract["runtime_sha256"],
        "actual_evaluation_bank_sha256": exp21_contract["actual_evaluation_bank_sha256"],
        "selected_arm": selected_arm,
        "selected_recipe": recipe,
        "selected_recipe_sha256": recipe_hash,
        "run_evidence": exp21_evidence,
        "raw_recomputation_sha256": stable_hash({
            "runs": exp21_derived_rows,
            "outcome": _exp21_outcome_summary(exp21_manifest, exp21_derived_rows),
        }),
    }
    binding: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "status": "sealed_accepted_exp20_and_exp21_raw_recomputed",
        "formal_submission_allowed": True,
        "selection_policy": "derive_exact_exp21_recipe_after_independent_exp20_and_exp21_raw_recomputation",
        "selected_arm": selected_arm,
        "selected_recipe": recipe,
        "selected_recipe_sha256": recipe_hash,
        "exp20": exp20_payload,
        "exp21": exp21_payload,
        "bound_files": dict(sorted(bound_files.items())),
    }
    reject_forbidden(binding, tokens, "new Exp22 prerequisite binding")
    binding["binding_sha256"] = stable_hash(binding)
    return binding


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CAMPAIGN_DIR / "manifest.json")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    binding = build_binding(manifest)
    print(json.dumps(binding, sort_keys=True, indent=2))
    if not args.publish:
        print("dry-run only: Exp22 binding/protocol unchanged", file=sys.stderr)
        return 0
    existing = read_json(PREREQUISITE_BINDINGS_PATH)
    require(existing.get("status") == "unsealed_waiting_for_accepted_exp20_and_exp21" or existing == binding, "Exp22 binding already sealed to different bytes")
    atomic_json(PREREQUISITE_BINDINGS_PATH, binding)
    atomic_text(PROTOCOL_LOCK_PATH, protocol_sha256(CAMPAIGN_DIR) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"Exp22 prerequisite binding error: {exc}", file=sys.stderr)
        raise SystemExit(2)
