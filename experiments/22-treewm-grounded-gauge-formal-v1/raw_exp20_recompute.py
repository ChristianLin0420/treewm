#!/usr/bin/env python3
"""Exp22-local independent replay of the complete Exp20 raw G/GS gate chain."""

from __future__ import annotations

import math
from pathlib import Path
import statistics
from typing import Any, Mapping, Sequence

from campaign import (
    ContractError,
    REPOSITORY_ROOT,
    file_sha256,
    load_compatible_input,
    read_json,
    require,
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
    require(not visit(value), f"{label} contains forbidden Exp14-18 ancestry")


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
    gauge_tags = (
        "latent_gauge/root/scale",
        "latent_gauge/root/reference",
        "latent_gauge/root/ratio",
        "latent_gauge/future/scale",
        "latent_gauge/future/reference",
        "latent_gauge/future/ratio",
        "latent_gauge/min_ratio",
    )
    gauge_series = {tag: _complete_series(metrics, tag, gauge_axis) for tag in gauge_tags}
    complete_gauge_axis = bool(gauge_axis and all(values is not None for values in gauge_series.values()))
    recent_ratio = gauge_series["latent_gauge/min_ratio"]
    root_ratio = last.get("latent_gauge/root/ratio")
    future_ratio = last.get("latent_gauge/future/ratio")
    min_ratio = last.get("latent_gauge/min_ratio")
    def close(actual: float, expected: float) -> bool:
        return abs(actual - expected) <= 1e-5 * max(1.0, abs(actual), abs(expected))

    recent_ratio_consistent = bool(
        complete_gauge_axis
        and all(
            root_scale > 0
            and root_reference >= float(exp20_manifest["scientific_contract"]["latent_gauge_min_reference_scale"])
            and future_scale > 0
            and future_reference >= float(exp20_manifest["scientific_contract"]["latent_gauge_min_reference_scale"])
            and root_ratio_value > 0
            and future_ratio_value > 0
            and min_ratio_value > 0
            and close(root_ratio_value, root_scale / root_reference)
            and close(future_ratio_value, future_scale / future_reference)
            and close(min_ratio_value, min(root_ratio_value, future_ratio_value))
            for (
                root_scale,
                root_reference,
                root_ratio_value,
                future_scale,
                future_reference,
                future_ratio_value,
                min_ratio_value,
            ) in zip(*(gauge_series[tag] or [] for tag in gauge_tags), strict=True)
        )
    )
    ratio_consistent = bool(
        finite(root_ratio) and finite(future_ratio) and finite(min_ratio)
        and float(root_ratio) > 0 and float(future_ratio) > 0 and float(min_ratio) > 0
        and float(min_ratio) <= min(float(root_ratio), float(future_ratio)) + 1e-5
        and recent_ratio_consistent
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
        "complete_recent_gauge_axis": complete_gauge_axis,
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
        "complete_recent_gauge_axis": complete_gauge_axis,
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
        require("=" in token and not token.startswith("--"), f"{label} has non-Hydra argv token")
        key, value = token.split("=", 1)
        require(key and key not in result, f"{label} has duplicate override {key}")
        result[key] = value
        tokens.append(token)
    return result, tokens


def _override(name: str, value: object) -> str:
    return f"{name}={_render_override(value)}"


def _expected_exp20_overrides(
    manifest: Mapping[str, Any],
    setting: Mapping[str, Any],
    arm: Mapping[str, Any],
    seed: int,
    contract: Mapping[str, Any],
) -> list[str]:
    method = manifest["method"]
    scientific = manifest["scientific_contract"]
    future = scientific["future_config"]
    chosen = contract["chosen_thresholds"]
    choice = manifest["inference_choice"]
    profile = choice["profiles"][choice["profile"]]
    return [
        _override("env", setting["env_config"]),
        _override("experiment", method["experiment_config"]),
        _override("arm", method["arm"]),
        _override("objective_version", method["objective_version"]),
        _override("seed", seed),
        _override("train.steps", scientific["optimizer_updates"]),
        _override("train.scheduler_total_steps", scientific["scheduler_total_steps"]),
        _override("train.ckpt_every", scientific["checkpoint_every_updates"]),
        _override("train.val_every", scientific["validation_every_updates"]),
        _override("train.diag_every", scientific["diagnostics_every_updates"]),
        _override("train.eval_every", scientific["periodic_evaluation_every_updates"]),
        _override("train.validation_sample_seed", scientific["validation_sample_seed"]),
        _override("train.max_train_anchors", setting["published_union_train_anchors"]),
        _override("train.max_val_anchors", setting["published_union_validation_anchors"]),
        _override("train.num_workers", scientific["data_loader_workers"]),
        _override("train.lr", arm["world_lr"]),
        _override("train.weight_decay", scientific["world_weight_decay"]),
        _override("train.gradient_checkpointing", scientific["gradient_checkpointing"]),
        _override("train.separate_gain_grad_clip", True),
        _override("train.separate_branch_transformer_grad_clip", arm["separate_branch_transformer_grad_clip"]),
        _override("train.world_grad_clip", 1.0),
        _override("train.gain_grad_clip", 1.0),
        _override("train.branch_transformer_grad_clip", arm["branch_transformer_grad_clip"]),
        _override("train.gain_loss_every", scientific["gain_loss_every"]),
        _override("train.gain_lr", scientific["gain_lr"]),
        _override("train.gain_weight_decay", scientific["gain_weight_decay"]),
        _override("train.gain_training_scorers", scientific["gain_training_scorers"]),
        _override("train.viz_every", 25_000),
        _override("train.viz_every_early", 1_000),
        _override("train.viz_early_until", 2_000),
        _override("model.dropout", scientific["model_dropout"]),
        _override("model.max_depth", scientific["model_max_depth"]),
        _override("tree.max_depth", scientific["tree_max_depth"]),
        _override("tree.node_budget", method["node_budget"]),
        _override("tree.keep_threshold", scientific["keep_threshold"]),
        _override("tree.scorer", profile["scorer"]),
        _override("model.branch_factor", method["branch_factor"]),
        _override("planner.decoded_metric", scientific["planner_decoded_metric"]),
        _override("planner.execute_mode", scientific["planner_execute_mode"]),
        _override("planner.execute_steps", scientific["planner_execute_steps"]),
        _override("planner.max_env_steps", setting["max_episode_steps"]),
        _override("planner.require_first_edge_improvement", profile["require_first_edge_improvement"]),
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
        _override("losses.multistep_transition_mode", arm["transition_mode"]),
        _override("losses.grounded_select_action_weight", arm["grounded_select_action_weight"]),
        _override("losses.grounded_select_endpoint_weight", arm["grounded_select_endpoint_weight"]),
        _override("losses.grounded_select_horizon_weight", arm["grounded_select_horizon_weight"]),
        _override("losses.grounded_loss_latent_weight", arm["grounded_loss_latent_weight"]),
        _override("losses.grounded_loss_action_weight", arm["grounded_loss_action_weight"]),
        _override("losses.grounded_loss_horizon_weight", arm["grounded_loss_horizon_weight"]),
        _override("losses.grounded_loss_endpoint_weight", arm["grounded_loss_endpoint_weight"]),
        _override("losses.grounded_detach_self_fed_parent", scientific["grounded_detach_self_fed_parent"]),
        _override("losses.multistep_depth_weights", scientific["multistep_depth_weights"]),
        _override("losses.enabled.latent_gauge", arm["latent_gauge_enabled"]),
        _override("losses.weights.latent_gauge", arm["latent_gauge_weight"]),
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
        _override("+campaign_factorial_arm", arm["id"]),
    ]


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
    for name in (
        "config_sha256", "run_protocol_sha256", "input_contract_sha256",
        "data_manifest_sha256", "normalizer_sha256", "train_manifest_sha256",
        "validation_manifest_sha256", "calibration_sha256", "future_recipe_sha256",
        "recipe_code_sha256", "recipe_runtime_sha256",
        "evaluation_seed_tables_sha256", "final_seed_table_sha256",
    ):
        require(sha(hashes.get(name)), f"Exp20 {run_name} launch {name} is malformed")
    argv = launch.get("argv")
    require(isinstance(argv, list) and all(isinstance(value, str) for value in argv), f"Exp20 {run_name} argv differs")
    overrides, ordered_tokens = _argv_overrides(argv, f"Exp20 {run_name}")
    setting_row = next(row for row in exp20_manifest["settings"] if row["id"] == setting)
    arm_row = next(row for row in exp20_manifest["arms"] if row["id"] == arm)
    method = exp20_manifest["method"]
    scientific = exp20_manifest["scientific_contract"]
    expected = {
        "env": setting_row["env_config"],
        "experiment": method["experiment_config"],
        "arm": method["arm"],
        "objective_version": method["objective_version"],
        "seed": seed,
        "train.steps": scientific["optimizer_updates"],
        "train.scheduler_total_steps": scientific["scheduler_total_steps"],
        "train.lr": arm_row["world_lr"],
        "train.separate_gain_grad_clip": True,
        "train.separate_branch_transformer_grad_clip": arm_row["separate_branch_transformer_grad_clip"],
        "train.world_grad_clip": 1.0,
        "train.gain_grad_clip": 1.0,
        "train.branch_transformer_grad_clip": arm_row["branch_transformer_grad_clip"],
        "losses.keep_balance": scientific["keep_balance"],
        "losses.enabled.multistep": scientific["multistep_enabled"],
        "losses.scheduled_sampling_p": scientific["scheduled_sampling_p"],
        "losses.scheduled_sampling_warmup": scientific["scheduled_sampling_warmup"],
        "losses.scheduled_sampling_granularity": scientific["scheduled_sampling_granularity"],
        "losses.multistep_transition_mode": arm_row["transition_mode"],
        "losses.grounded_select_action_weight": arm_row["grounded_select_action_weight"],
        "losses.grounded_select_endpoint_weight": arm_row["grounded_select_endpoint_weight"],
        "losses.grounded_select_horizon_weight": arm_row["grounded_select_horizon_weight"],
        "losses.grounded_loss_latent_weight": arm_row["grounded_loss_latent_weight"],
        "losses.grounded_loss_action_weight": arm_row["grounded_loss_action_weight"],
        "losses.grounded_loss_horizon_weight": arm_row["grounded_loss_horizon_weight"],
        "losses.grounded_loss_endpoint_weight": arm_row["grounded_loss_endpoint_weight"],
        "losses.grounded_detach_self_fed_parent": scientific["grounded_detach_self_fed_parent"],
        "losses.enabled.latent_gauge": arm_row["latent_gauge_enabled"],
        "losses.weights.latent_gauge": arm_row["latent_gauge_weight"],
        "planner.decoded_metric": scientific["planner_decoded_metric"],
        "planner.execute_mode": scientific["planner_execute_mode"],
        "planner.execute_steps": scientific["planner_execute_steps"],
        "future_sets.recipe_anchor_policy": scientific["future_config"]["recipe_anchor_policy"],
        "eval.task_split": scientific["task_split"],
        "eval.episodes_per_task": scientific["periodic_episodes_per_task"],
        "eval.final_episodes_per_task": scientific["final_episodes_per_task"],
        "+campaign_factorial_arm": arm,
        "run_root": exp20_manifest["paths"]["run_root"],
        "run_name": run_name,
        "resume": "auto",
        "+campaign_source_sha256": contract["source_sha256"],
        "+campaign_protocol_sha256": contract["package_protocol_sha256"],
        "+campaign_config_sha256": hashes["config_sha256"],
        "hydra.job.chdir": False,
    }
    require(
        all(overrides.get(name) == _render_override(value) for name, value in expected.items()),
        f"Exp20 {run_name} exact arm/gauge/clipping/loss recipe differs",
    )
    run_root_position = next(
        (position for position, token in enumerate(ordered_tokens) if token.startswith("run_root=")),
        None,
    )
    require(run_root_position is not None, f"Exp20 {run_name} lacks run_root boundary")
    require(
        stable_hash({"schema_version": 1, "overrides": ordered_tokens[:run_root_position]})
        == hashes["config_sha256"],
        f"Exp20 {run_name} config hash is not exact argv-derived",
    )
    trainer_path = Path(argv[1]) if len(argv) >= 2 else Path()
    expected_trainer_path = _expected_exp20_snapshot_repo(contract, exp20_manifest) / "scripts/train.py"
    require(
        len(argv) >= 2
        and argv[0] == exp20_manifest["paths"]["python"]
        and trainer_path == expected_trainer_path,
        f"Exp20 {run_name} did not use its exact sealed direct scripts/train.py",
    )
    input_contract = load_compatible_input(exp20_manifest, setting_row, verify_files=False)
    expected_scientific_overrides = _expected_exp20_overrides(
        exp20_manifest, setting_row, arm_row, seed, input_contract
    )
    expected_argv = [
        exp20_manifest["paths"]["python"],
        str(expected_trainer_path),
        *expected_scientific_overrides,
        _override("run_root", exp20_manifest["paths"]["run_root"]),
        _override("run_name", run_name),
        _override("resume", "auto"),
        _override("+campaign_source_sha256", contract["source_sha256"]),
        _override("+campaign_protocol_sha256", contract["package_protocol_sha256"]),
        _override("+campaign_config_sha256", hashes["config_sha256"]),
        _override("hydra.run.dir", Path(exp20_manifest["paths"]["run_root"]) / setting / "treewm" / run_name / "hydra"),
        _override("hydra.job.chdir", False),
    ]
    require(argv == expected_argv, f"Exp20 {run_name} full ordered trainer argv differs")
    require(
        hashes["config_sha256"] == stable_hash({
            "schema_version": 1,
            "overrides": expected_scientific_overrides,
        }),
        f"Exp20 {run_name} full expected config hash differs",
    )
    for name in (
        "input_contract_sha256", "data_manifest_sha256", "normalizer_sha256",
        "train_manifest_sha256", "validation_manifest_sha256",
        "calibration_sha256", "future_recipe_sha256",
    ):
        require(hashes[name] == input_contract[name.replace("input_contract", "contract")], f"Exp20 {run_name} input {name} differs")
    scientific = exp20_manifest["scientific_contract"]
    task_ids = list(exp20_manifest["design"]["task_ids"])
    base = int(scientific["evaluation_seed"])
    episodes = int(scientific["periodic_episodes_per_task"])
    actual_bank: dict[str, Any] = {
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
    actual_bank["sha256"] = stable_hash(actual_bank)
    require(actual_bank["sha256"] == contract["actual_evaluation_bank_sha256"], f"Exp20 {run_name} monitor bank differs")
    from treewm.evaluation.rollout import build_evaluation_seed_tables

    seed_tables = build_evaluation_seed_tables(
        scientific["evaluation_seed_protocol_sha256"],
        seed,
        task_ids,
        scientific["periodic_episodes_per_task"],
        scientific["final_episodes_per_task"],
    )
    require(
        hashes["evaluation_seed_tables_sha256"] == seed_tables["sha256"]
        and hashes["final_seed_table_sha256"] == seed_tables["final"]["sha256"],
        f"Exp20 {run_name} evaluation seed tables differ",
    )
    require(
        hashes["recipe_code_sha256"] == exp20_manifest["compatible_v2_recipe_input"]["recipe_code_sha256"]
        and hashes["recipe_runtime_sha256"] == exp20_manifest["compatible_v2_recipe_input"]["recipe_runtime_sha256"],
        f"Exp20 {run_name} compatible recipe provenance differs",
    )
    expected_run_protocol = stable_hash({
        "schema_version": 1,
        "campaign_id": exp20_manifest["campaign_id"],
        "package_protocol_sha256": contract["package_protocol_sha256"],
        "source_sha256": contract["source_sha256"],
        "runtime_sha256": contract["runtime_sha256"],
        "config_sha256": hashes["config_sha256"],
        "input_contract_sha256": hashes["input_contract_sha256"],
        "data_manifest_sha256": hashes["data_manifest_sha256"],
        "normalizer_sha256": hashes["normalizer_sha256"],
        "train_manifest_sha256": hashes["train_manifest_sha256"],
        "validation_manifest_sha256": hashes["validation_manifest_sha256"],
        "calibration_sha256": hashes["calibration_sha256"],
        "future_recipe_sha256": hashes["future_recipe_sha256"],
        "evaluation_seed_protocol_sha256": scientific["evaluation_seed_protocol_sha256"],
        "actual_evaluation_bank_sha256": actual_bank["sha256"],
    })
    require(hashes["run_protocol_sha256"] == expected_run_protocol, f"Exp20 {run_name} run protocol differs")
    environment = launch.get("environment") or {}
    require(
        environment.get("TREEWM_CODE_SHA256") == contract["source_sha256"]
        and environment.get("TREEWM_ACTIVE_SOURCE_SHA256") == contract["source_sha256"]
        and environment.get("TREEWM_RUNTIME_SHA256") == contract["runtime_sha256"]
        and environment.get("TREEWM_CONFIG_SHA256") == hashes["config_sha256"]
        and environment.get("TREEWM_PROTOCOL_SHA256") == hashes["run_protocol_sha256"]
        and environment.get("TREEWM_DATA_CONTRACT_SHA256") == input_contract["contract_sha256"]
        and environment.get("TREEWM_EVALUATION_SEED_PROTOCOL_SHA256") == scientific["evaluation_seed_protocol_sha256"]
        and environment.get("TREEWM_EXPECTED_FINAL_SEED_TABLE_SHA256") == seed_tables["final"]["sha256"]
        and environment.get("WANDB_PROJECT") == exp20_manifest["logging"]["wandb_project"]
        and environment.get("WANDB_RUN_GROUP") == exp20_manifest["logging"]["wandb_group"],
        f"Exp20 {run_name} launch environment/provenance differs",
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
