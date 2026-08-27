#!/usr/bin/env python3
"""Recompute Exp20's raw G/GS decision and seal only that acceptance into Exp21."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

from campaign import (
    BINDING_PATH,
    CAMPAIGN_DIR,
    ContractError,
    PROTOCOL_LOCK_PATH,
    REPOSITORY_ROOT,
    atomic_json,
    file_sha256,
    load_manifest,
    protocol_sha256,
    read_json,
    require,
    selected_recipe,
    stable_hash,
)


COMMON_GRADIENT_TAGS = (
    "train/grad_norm_world",
    "train/grad_norm_gain",
    "train/grad_clip_coefficient_world",
    "train/grad_clip_coefficient_gain",
)
SEPARATE_CLIP_TAGS = (
    "train/grad_norm_world_rest",
    "train/grad_norm_branch_transformer",
    "train/grad_clip_coefficient_world_rest",
    "train/grad_clip_coefficient_branch_transformer",
)
GAIN_TAGS = (
    "expansion/gain_rank_correlation",
    "expansion/gain_pairwise_accuracy",
    "expansion/gain_eligible_decision_fraction",
    "expansion/gain_ordered_pair_count",
    "expansion/gain_pair_coverage_fraction",
)


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def self_hash(value: Mapping[str, Any], key: str, label: str) -> str:
    claimed = value.get(key)
    body = dict(value)
    body.pop(key, None)
    require(sha(claimed) and claimed == stable_hash(body), f"{label} self-hash differs")
    return str(claimed)


def reject_forbidden_ancestry(value: object, tokens: Sequence[str], label: str) -> None:
    def visit(node: object) -> bool:
        if isinstance(node, Mapping):
            return any(visit(key) or visit(item) for key, item in node.items())
        if isinstance(node, (list, tuple)):
            return any(visit(item) for item in node)
        if isinstance(node, str):
            lowered = node.lower()
            return any(token.lower() in lowered for token in tokens)
        return False
    require(not visit(value), f"{label} contains forbidden Exp15/Exp16/Exp18 ancestry")


def event_paths(run_dir: Path) -> list[Path]:
    paths = sorted(run_dir.glob("events.out.tfevents.*")) or sorted(run_dir.rglob("events.out.tfevents.*"))
    require(bool(paths), f"no Exp20 TensorBoard evidence in {run_dir}")
    require(all(path.is_file() and not path.is_symlink() for path in paths), f"Exp20 event evidence missing/symlinked in {run_dir}")
    return paths


def event_scalars(paths: Sequence[Path]) -> dict[str, dict[int, float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    merged: dict[str, dict[int, tuple[float, float]]] = {}
    for path in paths:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        try:
            accumulator.Reload()
        except Exception as exc:
            raise ContractError(f"unreadable Exp20 TensorBoard evidence {path}: {exc}") from exc
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                candidate = (float(event.wall_time), float(event.value))
                previous = merged.setdefault(tag, {}).get(int(event.step))
                if previous is None or candidate[0] >= previous[0]:
                    merged[tag][int(event.step)] = candidate
    return {tag: {step: value for step, (_wall, value) in values.items()} for tag, values in merged.items()}


def _values(metrics: Mapping[str, Mapping[int, float]], tag: str, target: int, *, recent: int | None = None) -> list[float]:
    lower = max(0, target - recent) if recent is not None else 0
    return [float(value) for step, value in sorted(metrics.get(tag, {}).items()) if lower <= step <= target]


def _last(metrics: Mapping[str, Mapping[int, float]], tag: str, target: int) -> tuple[int, float] | None:
    values = [(int(step), float(value)) for step, value in metrics.get(tag, {}).items() if step <= target]
    return max(values) if values else None


def _at(metrics: Mapping[str, Mapping[int, float]], tag: str, step: int) -> float | None:
    value = metrics.get(tag, {}).get(step)
    return float(value) if finite(value) else None


def _expected_axis(target: int, cadence: int, window: int) -> tuple[int, ...]:
    lower = max(cadence, target - window)
    first = ((lower + cadence - 1) // cadence) * cadence
    return tuple(range(first, target + 1, cadence))


def _complete_series(metrics: Mapping[str, Mapping[int, float]], tag: str, axis: Sequence[int]) -> list[float] | None:
    values = {int(step): float(value) for step, value in metrics.get(tag, {}).items() if step in axis}
    if tuple(sorted(values)) != tuple(axis):
        return None
    ordered = [values[step] for step in axis]
    return ordered if all(finite(value) for value in ordered) else None


def evaluate_exp20_metrics(
    exp20_manifest: Mapping[str, Any],
    metrics: Mapping[str, Mapping[int, float]],
    target: int,
    arm_id: str,
) -> dict[str, Any]:
    """Independently reproduce Exp20's complete gate from raw scalar events."""
    gate = exp20_manifest["stage_acceptance"]
    required = tuple(gate["required_finite_tags"])
    training_tags = set(gate["training_exact_target_tags"])
    samples = {tag: _last(metrics, tag, target) for tag in required}
    last = {tag: sample[1] if sample is not None else None for tag, sample in samples.items()}
    last_step = {tag: sample[0] if sample is not None else None for tag, sample in samples.items()}
    expected_step = {
        tag: target if tag in training_tags else target - target % int(gate["validation_diagnostic_every_updates"])
        for tag in required
    }
    finite_coverage = all(finite(value) for value in last.values())
    target_appropriate = all(last_step[tag] == expected_step[tag] for tag in required)
    fixed_counts = _values(metrics, "data/validation_fixed_sample_count", target)
    fixed_validation = bool(fixed_counts and all(finite(v) and v > 0 for v in fixed_counts) and len(set(fixed_counts)) == 1)

    gradient_axis = _expected_axis(target, int(gate["training_every_updates"]), min(int(gate["gradient_recent_window_updates"]), target))
    common = {tag: _complete_series(metrics, tag, gradient_axis) for tag in COMMON_GRADIENT_TAGS}
    separate = {tag: _complete_series(metrics, tag, gradient_axis) for tag in SEPARATE_CLIP_TAGS} if arm_id == "GS" else {}
    gradient_map = {**common, **separate}
    complete_gradient_axis = all(values is not None for values in gradient_map.values())
    norm_tags = ["train/grad_norm_world", "train/grad_norm_gain"]
    clip_tags = ["train/grad_clip_coefficient_world", "train/grad_clip_coefficient_gain"]
    if arm_id == "GS":
        norm_tags.extend(("train/grad_norm_world_rest", "train/grad_norm_branch_transformer"))
        clip_tags.extend(("train/grad_clip_coefficient_world_rest", "train/grad_clip_coefficient_branch_transformer"))
    norm_values = [value for tag in norm_tags for value in (gradient_map.get(tag) or [])]
    clip_values = [value for tag in clip_tags for value in (gradient_map.get(tag) or [])]
    gradients_nonzero = bool(complete_gradient_axis and norm_values and all(value > float(gate["min_gradient_norm"]) for value in norm_values))
    low_clip_by_tag = {
        tag: (sum(value < float(gate["min_clip_coefficient"]) for value in (gradient_map.get(tag) or [])) / len(gradient_map.get(tag) or []) if gradient_map.get(tag) else None)
        for tag in clip_tags
    }
    low_clip_fraction = max((float(value) for value in low_clip_by_tag.values()), default=math.inf) if all(finite(value) for value in low_clip_by_tag.values()) else None
    clip_coefficients_valid = bool(
        complete_gradient_axis and clip_values
        and all(0.0 < value <= 1.0 for value in clip_values)
    )
    clipping_saturation_bounded = bool(
        clip_coefficients_valid
        and finite(low_clip_fraction)
        and all(float(value) <= float(gate["max_clip_fraction_below_threshold"]) for value in low_clip_by_tag.values())
    )
    clipping_bounded = bool(clip_coefficients_valid and clipping_saturation_bounded)

    gauge_axis = _expected_axis(target, int(gate["training_every_updates"]), min(int(gate["gauge_recent_window_updates"]), target))
    recent_ratio = _complete_series(metrics, "latent_gauge/min_ratio", gauge_axis)
    root_ratio = last.get("latent_gauge/root/ratio")
    future_ratio = last.get("latent_gauge/future/ratio")
    min_ratio = last.get("latent_gauge/min_ratio")
    ratio_consistent = bool(
        finite(root_ratio) and finite(future_ratio) and finite(min_ratio)
        and float(root_ratio) > 0 and float(future_ratio) > 0 and float(min_ratio) > 0
        and float(min_ratio) <= min(float(root_ratio), float(future_ratio)) + 1e-5
    )
    reference_valid = bool(
        last.get("latent_gauge/reference_sealed") == float(gate["reference_sealed"])
        and last.get("latent_gauge/reference_update") == float(gate["reference_update"])
        and finite(last.get("latent_gauge/root/reference")) and finite(last.get("latent_gauge/future/reference"))
        and float(last["latent_gauge/root/reference"]) >= float(exp20_manifest["scientific_contract"]["latent_gauge_min_reference_scale"])
        and float(last["latent_gauge/future/reference"]) >= float(exp20_manifest["scientific_contract"]["latent_gauge_min_reference_scale"])
    )
    recent_min_ratio = min(recent_ratio) if recent_ratio else None
    gauge_absolute = bool(
        recent_ratio is not None and ratio_consistent and reference_valid
        and float(min_ratio) >= float(gate["min_scale_ratio"])
        and float(recent_min_ratio) >= float(gate["min_scale_ratio"])
    )

    val_values = _values(metrics, "val/loss_total", target)
    validation_stable = bool(val_values and all(finite(v) and v >= 0 for v in val_values) and val_values[-1] <= min(val_values) * (1 + float(gate["max_validation_regret_fraction"])))
    self_fed = _values(metrics, "val/loss_multistep_self_fed", target)
    self_fed_stable = bool(self_fed and all(finite(v) and v >= 0 for v in self_fed) and self_fed[-1] <= min(self_fed) * (1 + float(gate["max_self_fed_multistep_validation_regret_fraction"])))
    fractions = [last.get(f"data/validation_horizon_label_fraction_h{horizon}") for horizon in (4, 8, 16, 32, 64)]
    distribution_valid = bool(all(finite(v) and float(v) >= 0 for v in fractions) and abs(sum(float(v) for v in fractions) - 1) <= float(gate["horizon_label_fraction_sum_tolerance"]))
    prior_entropy = -sum(float(v) * math.log(max(float(v), 1e-12)) for v in fractions) if distribution_valid else None
    horizon_loss = last.get("val/loss_horizon")
    horizon_pass = bool(finite(horizon_loss) and finite(prior_entropy) and float(horizon_loss) < float(gate["horizon_uniform_cross_entropy"]) and float(horizon_loss) < float(prior_entropy))
    q_pass = bool(
        last.get("control/retrieval_uses_task_metric_endpoint") == 1.0
        and finite(last.get("control/q_advantage_over_z")) and finite(last.get("control/q_advantage_over_random_proj"))
        and float(last["control/q_advantage_over_z"]) > float(gate["min_q_advantage"])
        and float(last["control/q_advantage_over_random_proj"]) > float(gate["min_q_advantage"])
    )
    recent_gain = {tag: _values(metrics, tag, target, recent=min(5_000, target)) for tag in GAIN_TAGS}
    gain_mean = {tag: statistics.fmean(values) if values and all(finite(v) for v in values) else None for tag, values in recent_gain.items()}
    gain_pass = bool(
        finite(gain_mean[GAIN_TAGS[0]]) and float(gain_mean[GAIN_TAGS[0]]) >= float(gate["min_gain_rank_correlation"])
        and finite(gain_mean[GAIN_TAGS[1]]) and float(gain_mean[GAIN_TAGS[1]]) >= float(gate["min_gain_pairwise_accuracy"])
        and finite(gain_mean[GAIN_TAGS[2]]) and float(gain_mean[GAIN_TAGS[2]]) >= float(gate["min_gain_eligible_decision_fraction"])
        and finite(gain_mean[GAIN_TAGS[3]]) and float(gain_mean[GAIN_TAGS[3]]) >= float(gate["min_gain_ordered_pair_count"])
        and finite(gain_mean[GAIN_TAGS[4]]) and float(gain_mean[GAIN_TAGS[4]]) >= float(gate["min_gain_pair_coverage_fraction"])
    )
    support_pass = bool(
        finite(last.get("tree/support_recall")) and finite(last.get("tree/support_precision"))
        and float(last["tree/support_recall"]) >= float(gate["min_support_recall"])
        and float(last["tree/support_precision"]) >= float(gate["min_support_precision"])
    )
    integrity_gates = {
        "required_finite_telemetry": finite_coverage,
        "target_appropriate_telemetry": target_appropriate,
        "fixed_common_validation_sample": fixed_validation,
        "complete_recent_gradient_axis": complete_gradient_axis,
        "nonzero_world_gain_and_required_split_gradients": gradients_nonzero,
        "bounded_gradient_clipping": clipping_bounded,
        "gauge_reference_sealed_at_update_zero": reference_valid,
        "gauge_ratio_consistent": ratio_consistent,
        "complete_recent_gauge_axis": recent_ratio is not None,
    }
    structural_integrity_gates = {
        "required_finite_telemetry": finite_coverage,
        "target_appropriate_telemetry": target_appropriate,
        "fixed_common_validation_sample": fixed_validation,
        "complete_recent_gradient_axis": complete_gradient_axis,
        "nonzero_world_gain_and_required_split_gradients": gradients_nonzero,
        "valid_gradient_clip_coefficients": clip_coefficients_valid,
        "gauge_reference_sealed_at_update_zero": reference_valid,
        "gauge_ratio_consistent": ratio_consistent,
        "complete_recent_gauge_axis": recent_ratio is not None,
    }
    method_gates = {
        "validation_nonregression": validation_stable,
        "self_fed_multistep_validation_nonregression": self_fed_stable,
        "horizon_ce_below_uniform_and_empirical_prior": horizon_pass,
        "q_advantage": q_pass,
        "gain_rank_pair_eligibility_and_coverage": gain_pass,
        "support_recall_and_precision": support_pass,
    }
    integrity_passed = all(integrity_gates.values())
    structural_integrity_passed = all(structural_integrity_gates.values())
    method_passed = all(method_gates.values())
    return {
        "integrity_passed": integrity_passed,
        "structural_integrity_passed": structural_integrity_passed,
        "method_passed": method_passed,
        "gauge_absolute_passed": gauge_absolute,
        "candidate_passed": bool(arm_id in {"G", "GS"} and integrity_passed and method_passed and gauge_absolute),
        "integrity_gates": integrity_gates,
        "structural_integrity_gates": structural_integrity_gates,
        "method_gates": method_gates,
        "last": last,
        "last_step": last_step,
        "expected_last_step": expected_step,
        "recent_gauge_min_ratio": recent_min_ratio,
        "recent_gauge_samples": len(recent_ratio or []),
        "recent_gradient_samples": len(gradient_axis),
        "clip_fraction_below_threshold": low_clip_fraction,
        "clip_fraction_below_threshold_by_tag": low_clip_by_tag,
        "clip_coefficients_valid": clip_coefficients_valid,
        "clip_saturation_bounded": clipping_saturation_bounded,
        "recent_gain_mean": gain_mean,
        "horizon_empirical_prior_entropy": prior_entropy,
    }


def recompute_health(
    exp20_manifest: Mapping[str, Any],
    metrics: Mapping[str, Mapping[int, float]],
    row: Mapping[str, Any],
    *,
    target: int,
) -> dict[str, Any]:
    derived = evaluate_exp20_metrics(exp20_manifest, metrics, target, str(row.get("arm_id")))
    require(row.get("health") == derived, f"Exp20 {row.get('run_name')}@{target} health differs from raw TensorBoard evidence")
    return derived


def load_pinned_exp20_manifest(contract: Mapping[str, Any]) -> dict[str, Any]:
    path = REPOSITORY_ROOT / "experiments/20-treewm-grounded-gauge-pilot-v2/manifest.json"
    lock = REPOSITORY_ROOT / "experiments/20-treewm-grounded-gauge-pilot-v2/protocol.sha256"
    manifest = read_json(path)
    require(stable_hash(manifest) == contract["manifest_sha256"], "live Exp20 manifest differs from pinned package")
    require(lock.read_text(encoding="utf-8").strip() == contract["package_protocol_sha256"], "live Exp20 protocol lock differs")
    require(manifest.get("campaign_id") == contract["campaign_id"], "Exp20 manifest campaign differs")
    require(manifest.get("formal_validation") is False, "Exp20 manifest claims formal validation")
    require(manifest.get("classification") == "bounded_causal_gauge_pilot_v2", "Exp20 classification differs")
    design = manifest.get("design") or {}
    require(design.get("stage_5000_arms") == contract["raw_recomputation"]["arms"], "Exp20 arm design differs")
    require(design.get("seeds") == contract["raw_recomputation"]["seeds"], "Exp20 seed design differs")
    require(design.get("promotion_precedence") == contract["selection_precedence"], "Exp20 precedence differs")
    require(str(design.get("fresh_start_policy", "")).startswith("All thirty 5k runs start from scratch"), "Exp20 fresh-start design differs")
    return manifest


def _selected_exp20_recipe(exp20_manifest: Mapping[str, Any], arm: str) -> dict[str, Any]:
    matches = [row for row in exp20_manifest.get("arms", []) if row.get("id") == arm]
    require(len(matches) == 1 and matches[0].get("promotable") is True, "Exp20 selected arm recipe missing/nonpromotable")
    return {key: value for key, value in matches[0].items() if key != "promotable"}


def _expected_exp20_snapshot_repo(
    contract: Mapping[str, Any], exp20_manifest: Mapping[str, Any]
) -> Path:
    identity = stable_hash({
        "source_sha256": contract["source_sha256"],
        "runtime_sha256": contract["runtime_sha256"],
        "package_protocol_sha256": contract["package_protocol_sha256"],
    })
    return (
        Path(exp20_manifest["paths"]["run_root"])
        / "state/source-snapshots"
        / identity
        / "repo"
    )


def _verify_exp20_source_snapshot(
    contract: Mapping[str, Any], exp20_manifest: Mapping[str, Any]
) -> Path:
    repo = _expected_exp20_snapshot_repo(contract, exp20_manifest)
    identity = repo.parent.name
    marker_path = repo.parent / "SNAPSHOT.json"
    trainer_path = repo / "scripts/train.py"
    require(repo.is_dir() and not repo.is_symlink(), "Exp20 source snapshot repository missing/linked")
    require(marker_path.is_file() and not marker_path.is_symlink(), "Exp20 source snapshot marker missing/linked")
    require(trainer_path.is_file() and not trainer_path.is_symlink(), "Exp20 snapshot direct trainer missing/linked")
    marker = read_json(marker_path)
    require(marker.get("schema_version") == 1, "Exp20 snapshot marker schema differs")
    require(marker.get("status") == "sealed_read_only", "Exp20 snapshot is not sealed read-only")
    require(marker.get("repo_subdirectory") == "repo", "Exp20 snapshot repository name differs")
    require(marker.get("repo_files_writable") is False, "Exp20 snapshot marker permits writable files")
    require(marker.get("formal_validation") is False, "Exp20 snapshot claims formal validation")
    require(marker.get("trainer_source_sha256") == contract["source_sha256"], "Exp20 snapshot source marker differs")
    require(marker.get("runtime_sha256") == contract["runtime_sha256"], "Exp20 snapshot runtime marker differs")
    require(marker.get("package_protocol_sha256") == contract["package_protocol_sha256"], "Exp20 snapshot protocol marker differs")
    require(marker.get("snapshot_identity_sha256") == identity, "Exp20 snapshot identity marker differs")
    snapshot_lock = repo / "experiments/20-treewm-grounded-gauge-pilot-v2/protocol.sha256"
    require(
        snapshot_lock.is_file()
        and not snapshot_lock.is_symlink()
        and snapshot_lock.read_text(encoding="utf-8").strip() == contract["package_protocol_sha256"],
        "Exp20 snapshot package lock differs",
    )
    regular_files = [path for path in repo.rglob("*") if path.is_file()]
    require(regular_files, "Exp20 source snapshot is empty")
    require(all(not path.is_symlink() for path in regular_files), "Exp20 source snapshot contains linked files")
    require(all(path.stat().st_mode & 0o222 == 0 for path in regular_files), "Exp20 source snapshot contains writable files")
    from treewm.utils.provenance import trainer_code_fingerprint

    source = trainer_code_fingerprint(repo)
    require(source.get("manifest_sha256") == contract["source_sha256"], "Exp20 snapshot trainer source differs")
    return repo


def _validate_exp20_launch(
    launch: Mapping[str, Any],
    contract: Mapping[str, Any],
    exp20_manifest: Mapping[str, Any],
    key: tuple[str, str, int],
    run_name: str,
) -> None:
    claimed = launch.get("launch_sha256")
    body = dict(launch)
    body.pop("launch_sha256", None)
    require(sha(claimed) and claimed == stable_hash(body), f"Exp20 {run_name} launch self-hash differs")
    reject_forbidden_ancestry(launch, contract["forbidden_ancestry_tokens"], f"Exp20 {run_name} launch")
    setting, arm, seed = key
    run = launch.get("run") or {}
    hashes = launch.get("hashes") or {}
    require(launch.get("campaign_id") == contract["campaign_id"] and launch.get("formal_validation") is False, f"Exp20 {run_name} launch identity differs")
    require(run.get("setting_id") == setting and run.get("arm_id") == arm and run.get("seed") == seed and run.get("run_name") == run_name, f"Exp20 {run_name} run identity differs")
    setting_index = contract["raw_recomputation"]["settings"].index(setting)
    arm_index = contract["raw_recomputation"]["arms"].index(arm)
    seed_index = contract["raw_recomputation"]["seeds"].index(seed)
    require(run.get("index") == ((setting_index * 3) + arm_index) * 2 + seed_index, f"Exp20 {run_name} run index differs")
    for name in ("manifest_sha256", "package_protocol_sha256", "source_sha256", "runtime_sha256", "actual_evaluation_bank_sha256"):
        require(hashes.get(name) == contract[name], f"Exp20 {run_name} launch {name} differs")
    argv = launch.get("argv")
    require(isinstance(argv, list) and all(isinstance(value, str) for value in argv), f"Exp20 {run_name} argv differs")
    required = {
        "experiment=treewm_v2_grounded_gauge_pilot_v2",
        "objective_version=treewm_v2_grounded_gauge_pilot_v2",
        "train.steps=25000",
        "train.scheduler_total_steps=1000000",
        "resume=auto",
        f"+campaign_factorial_arm={arm}",
    }
    require(required.issubset(set(argv)), f"Exp20 {run_name} launch recipe/fresh objective differs")
    trainer_path = Path(argv[1]) if len(argv) >= 2 else Path()
    expected_trainer_path = _expected_exp20_snapshot_repo(contract, exp20_manifest) / "scripts/train.py"
    require(
        len(argv) >= 2
        and argv[0] == exp20_manifest["paths"]["python"]
        and trainer_path == expected_trainer_path,
        f"Exp20 {run_name} did not use its exact sealed direct scripts/train.py",
    )


def collect_raw_evidence(
    contract: Mapping[str, Any],
    exp20_manifest: Mapping[str, Any],
) -> tuple[
    dict[tuple[str, str, int], dict[str, dict[int, float]]],
    list[dict[str, Any]],
]:
    raw = contract["raw_recomputation"]
    _verify_exp20_source_snapshot(contract, exp20_manifest)
    run_root = Path(str(exp20_manifest["paths"]["run_root"]))
    require(run_root.resolve() == Path(contract["stage_5000_gate_path"]).parents[2].resolve(), "Exp20 run root/gate path differ")
    metrics_by_key: dict[tuple[str, str, int], dict[str, dict[int, float]]] = {}
    evidence: list[dict[str, Any]] = []
    for setting in raw["settings"]:
        for arm in raw["arms"]:
            for seed in raw["seeds"]:
                key = (setting, arm, seed)
                run_name = f"gauge-v2-launch2-{setting}-arm{arm.lower()}-seed{seed}"
                run_dir = run_root / setting / "treewm" / run_name
                paths = event_paths(run_dir)
                launch_path = run_dir / "GAUGE_PILOT_V2_LAUNCH.json"
                require(launch_path.is_file() and not launch_path.is_symlink(), f"Exp20 launch evidence missing: {launch_path}")
                launch_file_before = file_sha256(launch_path)
                launch = read_json(launch_path)
                _validate_exp20_launch(launch, contract, exp20_manifest, key, run_name)
                event_files = [
                    {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": file_sha256(path)}
                    for path in paths
                ]
                metrics_by_key[key] = event_scalars(paths)
                require(file_sha256(launch_path) == launch_file_before, f"Exp20 launch evidence changed while binding: {launch_path}")
                for record in event_files:
                    path = Path(record["path"])
                    require(path.stat().st_size == record["size"] and file_sha256(path) == record["sha256"], f"Exp20 event evidence changed while binding: {path}")
                evidence.append({
                    "setting_id": setting,
                    "arm_id": arm,
                    "seed": seed,
                    "run_name": run_name,
                    "run_directory": str(run_dir.resolve()),
                    "launch_path": str(launch_path.resolve()),
                    "launch_file_sha256": launch_file_before,
                    "launch_sha256": launch["launch_sha256"],
                    "event_files": event_files,
                })
    require(len(metrics_by_key) == raw["stage_5000_runs"] and len(evidence) == raw["stage_5000_runs"], "Exp20 raw evidence matrix differs")
    return metrics_by_key, evidence


def recompute_stage_5000(
    contract: Mapping[str, Any],
    gate: Mapping[str, Any],
    exp20_manifest: Mapping[str, Any],
    metrics_by_key: Mapping[tuple[str, str, int], Mapping[str, Mapping[int, float]]],
) -> tuple[str, dict[tuple[str, str, int], Mapping[str, Any]]]:
    raw = contract["raw_recomputation"]
    require(set(gate) == {
        "schema_version", "status", "campaign_id", "formal_validation", "stage_target",
        "selected_arm", "selection_precedence", "nonpromotable_arm", "candidate_summary",
        "runs", "package_protocol_sha256", "source_sha256", "runtime_sha256",
        "actual_evaluation_bank_sha256", "gate_sha256",
    }, "Exp20 5k top-level schema differs")
    require(gate.get("schema_version") == 1, "Exp20 5k schema differs")
    require(gate.get("campaign_id") == contract["campaign_id"], "Exp20 5k campaign differs")
    require(gate.get("status") == "accepted_for_selected_continuation", "Exp20 5k did not accept continuation")
    require(gate.get("formal_validation") is False and gate.get("stage_target") == 5_000, "Exp20 5k claim/target differs")
    require(gate.get("selection_precedence") == contract["selection_precedence"], "Exp20 5k precedence differs")
    require(gate.get("nonpromotable_arm") == "N", "Exp20 N control became promotable")
    for key in ("package_protocol_sha256", "source_sha256", "runtime_sha256", "actual_evaluation_bank_sha256"):
        require(gate.get(key) == contract[key], f"Exp20 5k {key} differs")
    records = gate.get("runs")
    require(isinstance(records, list) and len(records) == raw["stage_5000_runs"], "Exp20 5k raw run count differs")
    expected_keys = {
        (setting, arm, seed)
        for setting in raw["settings"]
        for arm in raw["arms"]
        for seed in raw["seeds"]
    }
    keyed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    derived: dict[tuple[str, str, int], dict[str, Any]] = {}
    for record in records:
        require(isinstance(record, dict), "Exp20 5k raw row is not an object")
        key = (str(record.get("setting_id")), str(record.get("arm_id")), int(record.get("seed", -1)))
        require(key in expected_keys and key not in keyed, "Exp20 5k matrix is missing/extra/duplicated")
        setting_index = raw["settings"].index(key[0])
        arm_index = raw["arms"].index(key[1])
        seed_index = raw["seeds"].index(key[2])
        index = ((setting_index * 3) + arm_index) * 2 + seed_index
        require(record.get("index") == index and record.get("stage_slot") == index, f"Exp20 5k {key} index differs")
        require(record.get("run_name") == f"gauge-v2-launch2-{key[0]}-arm{key[1].lower()}-seed{key[2]}", f"Exp20 5k {key} run name differs")
        for hash_key in ("launch_sha256", "identity_sha256", "checkpoint_sha256"):
            require(sha(record.get(hash_key)), f"Exp20 5k {key} bad {hash_key}")
        keyed[key] = record
        derived[key] = recompute_health(exp20_manifest, metrics_by_key[key], record, target=5_000)
        require(derived[key]["structural_integrity_passed"], f"Exp20 5k structural integrity failed for {key}")
    require(set(keyed) == expected_keys, "Exp20 5k matrix is incomplete")

    summary: dict[str, Any] = {}
    for arm in ("G", "GS"):
        cells = [derived[(setting, arm, seed)] for setting in raw["settings"] for seed in raw["seeds"]]
        deltas = [
            derived[(setting, arm, seed)]["recent_gauge_min_ratio"]
            - derived[(setting, "N", seed)]["recent_gauge_min_ratio"]
            for setting in raw["settings"]
            for seed in raw["seeds"]
        ]
        mean_delta = statistics.fmean(deltas)
        universal_count = sum(cell["candidate_passed"] for cell in cells)
        causal = mean_delta > float(raw["min_paired_mean_ratio_delta_vs_n"])
        summary[arm] = {
            "universal_cells_passed": universal_count,
            "required_cells": 10,
            "paired_ratio_deltas_vs_n": deltas,
            "paired_mean_ratio_delta_vs_n": mean_delta,
            "causal_scale_retention_passed": causal,
            "eligible": universal_count == 10 and causal,
        }
    require(gate.get("candidate_summary") == summary, "Exp20 5k candidate summary differs from raw rows")
    selected = next((arm for arm in contract["selection_precedence"] if summary[arm]["eligible"]), None)
    require(selected in ("G", "GS") and gate.get("selected_arm") == selected, "Exp20 selected clipping mode differs from recomputed G-then-GS rule")
    return str(selected), keyed


def recompute_outcome(
    raw: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    expected_episodes = float(raw["outcome_episodes_per_run"])
    for row in rows:
        outcome = row.get("outcome") or {}
        require(
            outcome.get("num_episodes") == expected_episodes
            and all(finite(outcome.get(key)) for key in ("successes", "success_rate", "distance_reduction_frac")),
            "Exp20 25k outcome telemetry is incomplete",
        )
        successes = float(outcome["successes"])
        require(successes.is_integer() and 0 <= successes <= expected_episodes, "Exp20 success count invalid")
        require(abs(float(outcome["success_rate"]) - successes / expected_episodes) <= 1e-6, "Exp20 success rate/count mismatch")
    per_seed: dict[str, Any] = {}
    for seed in raw["seeds"]:
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        successes = sum(float(row["outcome"]["successes"]) for row in seed_rows)
        progress = statistics.fmean(float(row["outcome"]["distance_reduction_frac"]) for row in seed_rows)
        require(successes >= raw["min_total_successes_per_seed"], f"Exp20 seed {seed} has all-zero success")
        require(progress > raw["min_mean_distance_reduction_per_seed_exclusive"], f"Exp20 seed {seed} has nonpositive progress")
        per_seed[str(seed)] = {"successes": successes, "mean_distance_reduction_frac": progress}
    per_setting: dict[str, Any] = {}
    both_success = 0
    both_progress = 0
    for setting in raw["settings"]:
        setting_rows = [row for row in rows if row["setting_id"] == setting]
        require({int(row["seed"]) for row in setting_rows} == set(raw["seeds"]), f"Exp20 {setting} lacks both seeds")
        success = all(float(row["outcome"]["successes"]) > 0 for row in setting_rows)
        progress = all(float(row["outcome"]["distance_reduction_frac"]) > 0 for row in setting_rows)
        both_success += success
        both_progress += progress
        per_setting[setting] = {"both_seed_nonzero_success": success, "both_seed_positive_progress": progress}
    require(both_success >= raw["min_settings_with_both_seed_success"], "Exp20 replicated success quorum failed")
    require(both_progress >= raw["min_settings_with_both_seed_positive_progress"], "Exp20 replicated progress quorum failed")
    return {
        "per_seed": per_seed,
        "per_setting": per_setting,
        "settings_with_both_seed_nonzero_success": both_success,
        "settings_with_both_seed_positive_progress": both_progress,
        "total_successes": sum(float(row["outcome"]["successes"]) for row in rows),
        "macro_success_rate": statistics.fmean(float(row["outcome"]["success_rate"]) for row in rows),
        "macro_distance_reduction_frac": statistics.fmean(float(row["outcome"]["distance_reduction_frac"]) for row in rows),
    }


def recompute_acceptance(
    contract: Mapping[str, Any],
    gate: Mapping[str, Any],
    stage_5000_hash: str,
    selected_arm: str,
    stage_5000_rows: Mapping[tuple[str, str, int], Mapping[str, Any]],
    exp20_manifest: Mapping[str, Any],
    metrics_by_key: Mapping[tuple[str, str, int], Mapping[str, Mapping[int, float]]],
) -> None:
    raw = contract["raw_recomputation"]
    require(set(gate) == {
        "schema_version", "status", "campaign_id", "formal_validation", "stage_target",
        "selected_arm", "stage_5000_gate_sha256", "selected_runs", "skipped_runs",
        "outcome", "package_protocol_sha256", "source_sha256", "runtime_sha256",
        "actual_evaluation_bank_sha256", "gate_sha256",
    }, "Exp20 acceptance top-level schema differs")
    require(gate.get("schema_version") == 1 and gate.get("campaign_id") == contract["campaign_id"], "Exp20 acceptance identity differs")
    require(gate.get("status") == contract["required_status"], "Exp20 was not accepted")
    require(gate.get("formal_validation") is False and gate.get("stage_target") == 25_000, "Exp20 acceptance claim/target differs")
    require(gate.get("selected_arm") == selected_arm, "Exp20 5k/25k selected arms differ")
    require(gate.get("stage_5000_gate_sha256") == stage_5000_hash, "Exp20 acceptance does not bind exact 5k gate")
    for key in ("package_protocol_sha256", "source_sha256", "runtime_sha256", "actual_evaluation_bank_sha256"):
        require(gate.get(key) == contract[key], f"Exp20 acceptance {key} differs")
    selected_rows = gate.get("selected_runs")
    skipped_rows = gate.get("skipped_runs")
    require(isinstance(selected_rows, list) and len(selected_rows) == raw["stage_25000_selected_runs"], "Exp20 selected terminal rows differ")
    require(isinstance(skipped_rows, list) and len(skipped_rows) == raw["stage_25000_skipped_runs"], "Exp20 skipped terminal rows differ")
    expected_selected = {(setting, selected_arm, seed) for setting in raw["settings"] for seed in raw["seeds"]}
    actual_selected: set[tuple[str, str, int]] = set()
    for row in selected_rows:
        key = (str(row.get("setting_id")), str(row.get("arm_id")), int(row.get("seed", -1)))
        require(key in expected_selected and key not in actual_selected, "Exp20 selected terminal matrix differs")
        setting_index = raw["settings"].index(key[0])
        arm_index = raw["arms"].index(key[1])
        seed_index = raw["seeds"].index(key[2])
        index = ((setting_index * 3) + arm_index) * 2 + seed_index
        stage_slot = setting_index * 4 + (0 if selected_arm == "G" else 2) + seed_index
        prior = stage_5000_rows[key]
        require(row.get("index") == index and row.get("stage_slot") == stage_slot, f"Exp20 selected terminal {key} index/slot differs")
        require(row.get("run_name") == f"gauge-v2-launch2-{key[0]}-arm{key[1].lower()}-seed{key[2]}", f"Exp20 selected terminal {key} run name differs")
        require(row.get("launch_sha256") == prior.get("launch_sha256"), f"Exp20 selected terminal {key} launch changed across stages")
        require(row.get("identity_sha256") == prior.get("identity_sha256"), f"Exp20 selected terminal {key} identity changed across stages")
        require(recompute_health(exp20_manifest, metrics_by_key[key], row, target=25_000)["candidate_passed"], f"Exp20 selected terminal cell {key} failed")
        for hash_key in ("launch_sha256", "identity_sha256", "checkpoint_sha256"):
            require(sha(row.get(hash_key)), f"Exp20 selected terminal {key} bad {hash_key}")
        expected_outcome = {
            "num_episodes": _at(metrics_by_key[key], "eval/num_episodes", 25_000),
            "successes": _at(metrics_by_key[key], "eval/successes", 25_000),
            "success_rate": _at(metrics_by_key[key], "eval/success_rate", 25_000),
            "distance_reduction_frac": _at(metrics_by_key[key], "eval/distance_reduction_frac", 25_000),
        }
        require(row.get("outcome") == expected_outcome, f"Exp20 selected terminal {key} outcome differs from raw events")
        actual_selected.add(key)
    require(actual_selected == expected_selected, "Exp20 selected terminal coverage incomplete")
    other = "GS" if selected_arm == "G" else "G"
    expected_skipped = {(setting, other, seed) for setting in raw["settings"] for seed in raw["seeds"]}
    actual_skipped: set[tuple[str, str, int]] = set()
    for row in skipped_rows:
        key = (str(row.get("setting_id")), str(row.get("arm_id")), int(row.get("seed", -1)))
        require(key in expected_skipped and key not in actual_skipped, "Exp20 skipped terminal matrix differs")
        setting_index = raw["settings"].index(key[0])
        arm_index = raw["arms"].index(key[1])
        seed_index = raw["seeds"].index(key[2])
        index = ((setting_index * 3) + arm_index) * 2 + seed_index
        stage_slot = setting_index * 4 + (0 if key[1] == "G" else 2) + seed_index
        prior = stage_5000_rows[key]
        require(row.get("index") == index and row.get("stage_slot") == stage_slot, f"Exp20 skipped {key} index/slot differs")
        require(row.get("launch_sha256") == prior.get("launch_sha256"), f"Exp20 skipped {key} launch changed")
        require(row.get("checkpoint_sha256") == prior["checkpoint_sha256"], f"Exp20 skipped {key} checkpoint advanced/changed")
        require(all(sha(row.get(name)) for name in ("launch_sha256", "checkpoint_sha256", "skip_sha256")), f"Exp20 skipped {key} hashes malformed")
        actual_skipped.add(key)
    require(actual_skipped == expected_skipped, "Exp20 skipped terminal coverage incomplete")
    outcome = recompute_outcome(raw, selected_rows)
    require(gate.get("outcome") == outcome, "Exp20 outcome summary differs from raw terminal rows")


def build_binding(
    manifest: Mapping[str, Any], stage_5000_path: Path, acceptance_path: Path
) -> dict[str, Any]:
    for path, label in ((stage_5000_path, "Exp20 5k gate"), (acceptance_path, "Exp20 acceptance")):
        require(path.is_file() and not path.is_symlink(), f"{label} missing/symlinked: {path}")
    contract = manifest["prerequisite"]
    exp20_manifest = load_pinned_exp20_manifest(contract)
    stage_5000 = read_json(stage_5000_path)
    acceptance = read_json(acceptance_path)
    reject_forbidden_ancestry(stage_5000, contract["forbidden_ancestry_tokens"], "Exp20 5k gate")
    reject_forbidden_ancestry(acceptance, contract["forbidden_ancestry_tokens"], "Exp20 acceptance")
    gate_5000_hash = self_hash(stage_5000, "gate_sha256", "Exp20 5k gate")
    acceptance_hash = self_hash(acceptance, "gate_sha256", "Exp20 acceptance")
    metrics_by_key, raw_evidence = collect_raw_evidence(contract, exp20_manifest)
    selected_arm, rows = recompute_stage_5000(contract, stage_5000, exp20_manifest, metrics_by_key)
    evidence_launches = {
        (row["setting_id"], row["arm_id"], row["seed"]): row["launch_sha256"]
        for row in raw_evidence
    }
    require(
        all(row.get("launch_sha256") == evidence_launches[key] for key, row in rows.items()),
        "Exp20 gate launch hashes differ from raw launch evidence",
    )
    recompute_acceptance(
        contract, acceptance, gate_5000_hash, selected_arm, rows,
        exp20_manifest, metrics_by_key,
    )
    recipe = selected_recipe(manifest, selected_arm)
    require(recipe == _selected_exp20_recipe(exp20_manifest, selected_arm), "Exp21 recipe differs from the selected Exp20 arm recipe")
    exp20 = {
        "campaign_id": contract["campaign_id"],
        "accepted_status": acceptance["status"],
        "stage_5000_gate_path": str(stage_5000_path.resolve()),
        "stage_5000_gate_file_sha256": file_sha256(stage_5000_path),
        "stage_5000_gate_sha256": gate_5000_hash,
        "acceptance_path": str(acceptance_path.resolve()),
        "acceptance_file_sha256": file_sha256(acceptance_path),
        "acceptance_gate_sha256": acceptance_hash,
        "manifest_sha256": contract["manifest_sha256"],
        "package_protocol_sha256": contract["package_protocol_sha256"],
        "source_sha256": contract["source_sha256"],
        "runtime_sha256": contract["runtime_sha256"],
        "actual_evaluation_bank_sha256": contract["actual_evaluation_bank_sha256"],
        "selected_arm": selected_arm,
        "raw_evidence": raw_evidence,
        "raw_evidence_sha256": stable_hash(raw_evidence),
    }
    binding: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "status": "sealed_exp20_acceptance",
        "launch_allowed": True,
        "selection_policy": "consume_recomputed_exp20_G_then_GS_without_bridge_recipe_selection",
        "exp20": exp20,
        "selected_arm": selected_arm,
        "selected_recipe": recipe,
        "selected_recipe_sha256": stable_hash(recipe),
    }
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
    parser.add_argument("--stage-5000-gate", type=Path)
    parser.add_argument("--acceptance", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    contract = manifest["prerequisite"]
    binding = build_binding(
        manifest,
        args.stage_5000_gate or Path(contract["stage_5000_gate_path"]),
        args.acceptance or Path(contract["acceptance_path"]),
    )
    print(json.dumps(binding, sort_keys=True, indent=2))
    if not args.publish:
        print("dry-run only: binding and protocol lock unchanged", file=sys.stderr)
        return 0
    existing = read_json(BINDING_PATH)
    require(existing.get("status") == "unsealed_waiting_for_exp20_acceptance" or existing == binding, "binding already sealed to different bytes")
    atomic_json(BINDING_PATH, binding)
    atomic_text(PROTOCOL_LOCK_PATH, protocol_sha256(CAMPAIGN_DIR) + "\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"Exp20 binding error: {exc}", file=sys.stderr)
        raise SystemExit(2)
