#!/usr/bin/env python3
"""Aggregate exactly 40 stage completions and publish an immutable health gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Mapping, Sequence

from campaign import (
    CAMPAIGN_DIR,
    ContractError,
    REPOSITORY_ROOT,
    STAGE_TARGETS,
    TRAINING_RUNS,
    atomic_json,
    expand_runs,
    load_manifest,
    read_json,
    run_directory,
    stable_hash,
    trainer_command,
)
from worker import verify_stage_marker


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def event_scalars(run_dir: Path) -> dict[str, dict[int, float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    merged: dict[str, dict[int, tuple[float, float]]] = {}
    paths = sorted(run_dir.glob("events.out.tfevents.*"))
    if not paths:
        paths = sorted(run_dir.rglob("events.out.tfevents.*"))
    if not paths:
        raise ContractError(f"no TensorBoard scalar artifacts in {run_dir}")
    for path in paths:
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        try:
            accumulator.Reload()
        except Exception as exc:
            raise ContractError(f"unreadable TensorBoard event file {path}: {exc}") from exc
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                candidate = (float(event.wall_time), float(event.value))
                previous = merged.setdefault(tag, {}).get(int(event.step))
                if previous is None or candidate[0] >= previous[0]:
                    merged[tag][int(event.step)] = candidate
    return {tag: {step: value for step, (_wall, value) in values.items()} for tag, values in merged.items()}


def _values(metrics: Mapping[str, Mapping[int, float]], tag: str, target: int, *, recent: int | None = None) -> list[float]:
    lower = max(0, target - recent) if recent is not None else 0
    # Preserve NaN/Inf so a newer broken event cannot be hidden by an older finite
    # value.  Callers decide whether a window must be wholly finite.
    return [
        float(value)
        for step, value in sorted(metrics.get(tag, {}).items())
        if lower <= step <= target
    ]


def _last_sample(
    metrics: Mapping[str, Mapping[int, float]], tag: str, target: int
) -> tuple[int, float] | None:
    values = [(step, value) for step, value in metrics.get(tag, {}).items() if step <= target]
    if not values:
        return None
    step, value = max(values)
    return int(step), float(value)


def _expected_axis(target: int, cadence: int, window: int) -> tuple[int, ...]:
    lower = max(cadence, target - window)
    first = ((lower + cadence - 1) // cadence) * cadence
    return tuple(range(first, target + 1, cadence))


def _at(metrics: Mapping[str, Mapping[int, float]], tag: str, step: int) -> float | None:
    value = metrics.get(tag, {}).get(step)
    return float(value) if finite(value) else None


def _complete_series(
    metrics: Mapping[str, Mapping[int, float]], tag: str, axis: Sequence[int]
) -> list[float] | None:
    values = {int(step): float(value) for step, value in metrics.get(tag, {}).items() if step in axis}
    if tuple(sorted(values)) != tuple(axis):
        return None
    ordered = [values[step] for step in axis]
    return ordered if all(finite(value) for value in ordered) else None


def evaluate_metrics(
    manifest: Mapping[str, Any],
    metrics: Mapping[str, Mapping[int, float]],
    target: int,
    selected_arm: str,
) -> dict[str, Any]:
    """Apply target-fresh numerical/gauge health and the preregistered 25k method gate."""
    if selected_arm not in ("G", "GS"):
        raise ContractError("stage gate selected arm is not G/GS")
    gate = manifest["stage_acceptance"]
    freshness = gate["telemetry_freshness"]
    conditional = tuple(freshness["selected_arm_conditional_training_tags"][selected_arm])
    required = tuple(gate["required_finite_tags"]) + conditional
    training_tags = set(freshness["training_exact_target_tags"]) | set(conditional)
    validation_cadence = int(freshness["validation_diagnostic_every_updates"])
    expected_validation_step = target - target % validation_cadence
    samples = {tag: _last_sample(metrics, tag, target) for tag in required}
    last = {tag: sample[1] if sample is not None else None for tag, sample in samples.items()}
    last_step = {tag: sample[0] if sample is not None else None for tag, sample in samples.items()}
    expected_last_step = {tag: target if tag in training_tags else expected_validation_step for tag in required}
    finite_coverage = all(finite(value) for value in last.values())
    target_appropriate = all(last_step[tag] == expected_last_step[tag] for tag in required)

    fixed_counts = _values(metrics, "data/validation_fixed_sample_count", target)
    fixed_validation = bool(
        fixed_counts
        and all(finite(value) and value > 0 for value in fixed_counts)
        and len({float(value) for value in fixed_counts}) == 1
    )
    cadence = int(freshness["training_every_updates"])
    gradient_window = min(int(gate["gradient_recent_window_updates"]), target)
    gradient_axis = _expected_axis(target, cadence, gradient_window)
    common_tags = (
        "train/grad_norm_world",
        "train/grad_norm_gain",
        "train/grad_clip_coefficient_world",
        "train/grad_clip_coefficient_gain",
    )
    gradient_tags = common_tags + conditional
    gradient_series = {tag: _complete_series(metrics, tag, gradient_axis) for tag in gradient_tags}
    complete_gradient_axis = bool(gradient_axis and all(values is not None for values in gradient_series.values()))
    norm_tags = ["train/grad_norm_world", "train/grad_norm_gain"]
    clip_tags = ["train/grad_clip_coefficient_world", "train/grad_clip_coefficient_gain"]
    if selected_arm == "GS":
        norm_tags.extend(("train/grad_norm_world_rest", "train/grad_norm_branch_transformer"))
        clip_tags.extend(("train/grad_clip_coefficient_world_rest", "train/grad_clip_coefficient_branch_transformer"))
    norm_values = [value for tag in norm_tags for value in (gradient_series.get(tag) or [])]
    clip_values = [value for tag in clip_tags for value in (gradient_series.get(tag) or [])]
    gradients_nonzero = bool(
        complete_gradient_axis
        and norm_values
        and all(value > float(gate["min_gradient_norm"]) for value in norm_values)
    )
    clip_coefficients_valid = bool(
        complete_gradient_axis and clip_values and all(0.0 < value <= 1.0 for value in clip_values)
    )
    low_clip_by_tag = {
        tag: (
            sum(value < float(gate["min_clip_coefficient"]) for value in (gradient_series.get(tag) or []))
            / len(gradient_series.get(tag) or [])
            if gradient_series.get(tag)
            else None
        )
        for tag in clip_tags
    }
    clipping_saturation_bounded = bool(
        clip_coefficients_valid
        and all(finite(value) and float(value) <= float(gate["max_clip_fraction_below_threshold"]) for value in low_clip_by_tag.values())
    )

    gauge_window = min(int(gate["gauge_recent_window_updates"]), target)
    gauge_axis = _expected_axis(target, cadence, gauge_window)
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
            and root_reference >= float(manifest["scientific_contract"]["latent_gauge_min_reference_scale"])
            and future_scale > 0
            and future_reference >= float(manifest["scientific_contract"]["latent_gauge_min_reference_scale"])
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
        and finite(last.get("latent_gauge/root/reference"))
        and finite(last.get("latent_gauge/future/reference"))
        and float(last["latent_gauge/root/reference"]) >= float(manifest["scientific_contract"]["latent_gauge_min_reference_scale"])
        and float(last["latent_gauge/future/reference"]) >= float(manifest["scientific_contract"]["latent_gauge_min_reference_scale"])
    )
    recent_min_ratio = min(recent_ratio) if recent_ratio else None
    gauge_absolute = bool(
        recent_ratio is not None and ratio_consistent and reference_valid
        and float(min_ratio) >= float(gate["min_scale_ratio"])
        and finite(recent_min_ratio) and float(recent_min_ratio) >= float(gate["min_scale_ratio"])
    )

    enforce = target == int(gate["scientific_gate_stage"])
    val_values = _values(metrics, "val/loss_total", target)
    validation_stable_raw = bool(
        val_values and all(finite(value) and value >= 0 for value in val_values)
        and val_values[-1] <= min(val_values) * (1 + float(gate["max_validation_regret_fraction"]))
    )
    self_fed = _values(metrics, "val/loss_multistep_self_fed", target)
    self_fed_stable_raw = bool(
        self_fed and all(finite(value) and value >= 0 for value in self_fed)
        and self_fed[-1] <= min(self_fed) * (1 + float(gate["max_self_fed_multistep_validation_regret_fraction"]))
    )
    fractions = [last.get(f"data/validation_horizon_label_fraction_h{horizon}") for horizon in (4, 8, 16, 32, 64)]
    distribution_valid = bool(
        all(finite(value) and float(value) >= 0 for value in fractions)
        and abs(sum(float(value) for value in fractions) - 1) <= float(gate["horizon_label_fraction_sum_tolerance"])
    )
    prior_entropy = -sum(float(value) * math.log(max(float(value), 1e-12)) for value in fractions) if distribution_valid else None
    horizon_loss = last.get("val/loss_horizon")
    horizon_raw = bool(
        finite(horizon_loss) and finite(prior_entropy)
        and float(horizon_loss) < float(gate["horizon_uniform_cross_entropy"])
        and float(horizon_loss) < float(prior_entropy)
    )
    q_raw = bool(
        last.get("control/retrieval_uses_task_metric_endpoint") == 1.0
        and finite(last.get("control/q_advantage_over_z"))
        and finite(last.get("control/q_advantage_over_random_proj"))
        and float(last["control/q_advantage_over_z"]) > float(gate["min_q_advantage"])
        and float(last["control/q_advantage_over_random_proj"]) > float(gate["min_q_advantage"])
    )
    gain_tags = (
        "expansion/gain_rank_correlation",
        "expansion/gain_pairwise_accuracy",
        "expansion/gain_eligible_decision_fraction",
        "expansion/gain_ordered_pair_count",
        "expansion/gain_pair_coverage_fraction",
    )
    gain_values = {tag: _values(metrics, tag, target, recent=min(5_000, target)) for tag in gain_tags}
    gain_mean = {tag: statistics.fmean(values) if values and all(finite(value) for value in values) else None for tag, values in gain_values.items()}
    gain_raw = bool(
        finite(gain_mean[gain_tags[0]]) and float(gain_mean[gain_tags[0]]) >= float(gate["min_gain_rank_correlation"])
        and finite(gain_mean[gain_tags[1]]) and float(gain_mean[gain_tags[1]]) >= float(gate["min_gain_pairwise_accuracy"])
        and finite(gain_mean[gain_tags[2]]) and float(gain_mean[gain_tags[2]]) >= float(gate["min_gain_eligible_decision_fraction"])
        and finite(gain_mean[gain_tags[3]]) and float(gain_mean[gain_tags[3]]) >= float(gate["min_gain_ordered_pair_count"])
        and finite(gain_mean[gain_tags[4]]) and float(gain_mean[gain_tags[4]]) >= float(gate["min_gain_pair_coverage_fraction"])
    )
    support_raw = bool(
        finite(last.get("tree/support_recall")) and float(last["tree/support_recall"]) >= float(gate["min_support_recall"])
        and finite(last.get("tree/support_precision")) and float(last["tree/support_precision"]) >= float(gate["min_support_precision"])
    )
    integrity_gates = {
        "required_finite_telemetry": finite_coverage,
        "target_appropriate_telemetry": target_appropriate,
        "fixed_common_validation_sample": fixed_validation,
        "complete_recent_gradient_axis": complete_gradient_axis,
        "nonzero_world_gain_and_required_split_gradients": gradients_nonzero,
        "valid_gradient_clip_coefficients": clip_coefficients_valid,
        "bounded_gradient_clipping_per_tag": clipping_saturation_bounded,
        "gauge_reference_sealed_at_update_zero": reference_valid,
        "gauge_ratio_consistent": ratio_consistent,
        "complete_recent_gauge_axis": complete_gauge_axis,
        "absolute_gauge_scale_floor": gauge_absolute,
    }
    raw_scientific = {
        "validation_nonregression": validation_stable_raw,
        "self_fed_multistep_validation_nonregression": self_fed_stable_raw,
        "horizon_ce_below_uniform_and_empirical_prior": horizon_raw,
        "q_advantage": q_raw,
        "gain_rank_pair_and_eligibility_coverage": gain_raw,
        "support_recall_and_precision": support_raw,
    }
    scientific_gates = {key: (value if enforce else True) for key, value in raw_scientific.items()}
    integrity_passed = all(integrity_gates.values())
    scientific_passed = all(scientific_gates.values())
    return {
        "integrity_passed": integrity_passed,
        "scientific_passed": scientific_passed,
        "passed": integrity_passed and scientific_passed,
        "integrity_gates": integrity_gates,
        "scientific_gates": scientific_gates,
        "scientific_raw_observations": raw_scientific,
        "last": last,
        "last_step": last_step,
        "expected_last_step": expected_last_step,
        "recent_gain_mean": gain_mean,
        "recent_gradient_window_updates": gradient_window,
        "recent_gradient_expected_sample_count": len(gradient_axis),
        "recent_gradient_sample_count": {tag: len(values or []) for tag, values in gradient_series.items()},
        "clip_fraction_below_threshold_by_tag": low_clip_by_tag,
        "recent_gauge_window_updates": gauge_window,
        "recent_gauge_expected_sample_count": len(gauge_axis),
        "recent_gauge_sample_count": {tag: len(values or []) for tag, values in gauge_series.items()},
        "recent_gauge_min_ratio": recent_min_ratio,
        "horizon_empirical_prior_entropy": prior_entropy,
    }


def validate_stage_complete(path: Path, expected_launch: Mapping[str, Any], target: int) -> dict[str, Any]:
    value = read_json(path)
    claimed = value.get("stage_complete_sha256")
    body = dict(value)
    body.pop("stage_complete_sha256", None)
    if claimed != stable_hash(body):
        raise ContractError(f"stage completion hash differs: {path}")
    run = expected_launch["run"]
    hashes = expected_launch["hashes"]
    checks = (
        value.get("schema_version") == 1,
        value.get("status") == "stage_complete_awaiting_campaign_gate",
        int(value.get("index", -1)) == int(run["index"]),
        value.get("setting_id") == run["setting_id"],
        int(value.get("seed", -1)) == int(run["seed"]),
        int(value.get("stage_target", -1)) == target,
        value.get("launch_sha256") == expected_launch["launch_sha256"],
        value.get("package_protocol_sha256") == hashes["package_protocol_sha256"],
        value.get("source_sha256") == hashes["source_sha256"],
        value.get("evaluation_source_sha256") == hashes["evaluation_source_sha256"],
        value.get("runtime_sha256") == hashes["runtime_sha256"],
        value.get("prerequisite_binding_sha256") == hashes["prerequisite_binding_sha256"],
        value.get("selected_recipe_sha256") == hashes["selected_recipe_sha256"],
        value.get("selected_arm") == hashes["selected_arm"],
        SHA(value.get("identity_sha256")),
        SHA(value.get("checkpoint_sha256")),
        value.get("evaluation_seed_tables_sha256") == hashes["evaluation_seed_tables_sha256"],
        value.get("final_seed_table_sha256") == hashes["final_seed_table_sha256"],
        value.get("monitor_seed_table_sha256") == hashes["monitor_seed_table_sha256"],
    )
    if not all(checks):
        raise ContractError(f"stage completion identity differs: {path}")
    return value


def validate_scientific_setting_coverage(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], target: int
) -> dict[str, int] | None:
    if target != int(manifest["stage_acceptance"]["scientific_gate_stage"]):
        return None
    coverage = {
        setting_id: sum(
            bool(row["health"]["scientific_passed"])
            for row in rows
            if row["setting_id"] == setting_id
        )
        for setting_id in {str(row["setting_id"]) for row in rows}
    }
    failed = {setting: count for setting, count in coverage.items() if count < 3}
    if failed:
        raise ContractError(
            "scientific stage gate requires >=3/4 seeds per setting; "
            f"failed={json.dumps(failed, sort_keys=True)}"
        )
    return coverage


def validate_outcome_sanity(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], target: int
) -> dict[str, Any] | None:
    if target != int(manifest["stage_acceptance"]["outcome_sanity_stage"]):
        return None
    expected_episodes = int(manifest["stage_acceptance"]["monitor_episodes_per_run"])
    if any(
        row["monitor"]["num_episodes"] != float(expected_episodes)
        or not finite(row["monitor"]["successes"])
        or not finite(row["monitor"]["success_rate"])
        or not finite(row["monitor"]["distance_reduction_frac"])
        or not float(row["monitor"]["successes"]).is_integer()
        or not 0.0 <= float(row["monitor"]["successes"]) <= float(expected_episodes)
        or abs(
            float(row["monitor"]["success_rate"])
            - float(row["monitor"]["successes"]) / float(expected_episodes)
        )
        > 1e-6
        for row in rows
    ):
        raise ContractError("100k outcome rail lacks exact/self-consistent paired 5-episode monitor coverage")
    for setting_id in {str(row["setting_id"]) for row in rows}:
        if len({row["monitor_seed_table_sha256"] for row in rows if row["setting_id"] == setting_id}) != 1:
            raise ContractError(f"{setting_id}: training seeds did not share one monitor bank")
    successes = sum(float(row["monitor"]["successes"]) for row in rows)
    if successes < int(manifest["stage_acceptance"]["min_fleet_monitor_successes"]):
        raise ContractError(f"100k outcome sanity rejected: fleet successes={successes}/200")
    mean_progress = statistics.fmean(
        float(row["monitor"]["distance_reduction_frac"]) for row in rows
    )
    positive_runs = sum(
        float(row["monitor"]["distance_reduction_frac"]) > 0.0 for row in rows
    )
    if mean_progress <= float(
        manifest["stage_acceptance"]["min_fleet_mean_distance_reduction"]
    ):
        raise ContractError(
            f"100k outcome sanity rejected: fleet mean progress={mean_progress}"
        )
    if positive_runs < int(
        manifest["stage_acceptance"]["min_runs_with_positive_progress"]
    ):
        raise ContractError(
            f"100k outcome sanity rejected: positive-progress runs={positive_runs}"
        )
    return {
        "episodes": expected_episodes * TRAINING_RUNS,
        "successes": successes,
        "macro_success_rate": statistics.fmean(float(row["monitor"]["success_rate"]) for row in rows),
        "macro_distance_reduction_frac": mean_progress,
        "runs_with_positive_progress": positive_runs,
    }


def SHA(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def build_gate(manifest: Mapping[str, Any], target: int, repo_root: Path) -> dict[str, Any]:
    if target not in STAGE_TARGETS:
        raise ContractError("invalid stage target")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    runs = expand_runs(manifest)
    launches = [trainer_command(manifest, run, repo_root=repo_root) for run in runs]
    for run, launch in zip(runs, launches, strict=True):
        run_dir = run_directory(manifest, run)
        complete_path = run_dir / "stage-gates" / f"STAGE_COMPLETE_{target}.json"
        try:
            complete = validate_stage_complete(complete_path, launch, target)
            marker = verify_stage_marker(run_dir, target, launch)
            if complete["checkpoint_sha256"] != marker["checkpoint_sha256"]:
                raise ContractError("worker completion/checkpoint hash differs")
            scalars = event_scalars(run_dir)
            health = evaluate_metrics(manifest, scalars, target, str(launch["hashes"]["selected_arm"]))
            if not health["integrity_passed"]:
                raise ContractError("one or more per-run structural/integrity gates failed")
            rows.append({
                "index": run.index,
                "setting_id": run.setting_id,
                "seed": run.seed,
                "launch_sha256": launch["launch_sha256"],
                "identity_sha256": complete["identity_sha256"],
                "checkpoint_sha256": complete["checkpoint_sha256"],
                "final_seed_table_sha256": complete["final_seed_table_sha256"],
                "monitor_seed_table_sha256": complete["monitor_seed_table_sha256"],
                "health": health,
                "monitor": {
                    "num_episodes": _at(scalars, "eval/num_episodes", target),
                    "successes": _at(scalars, "eval/successes", target),
                    "success_rate": _at(scalars, "eval/success_rate", target),
                    "distance_reduction_frac": _at(scalars, "eval/distance_reduction_frac", target),
                },
            })
        except (ContractError, OSError, ValueError) as exc:
            failures.append({"index": run.index, "run_name": run.run_name, "error": str(exc)})
    if failures or len(rows) != TRAINING_RUNS:
        raise ContractError(f"stage {target} rejected: {len(rows)}/40 healthy; failures={json.dumps(failures, sort_keys=True)}")
    scientific_setting_coverage = validate_scientific_setting_coverage(manifest, rows, target)
    outcome_summary = validate_outcome_sanity(manifest, rows, target)
    hashes = {
        key: {
            launch["hashes"][key]
            for launch in launches
        }
        for key in (
            "package_protocol_sha256",
            "source_sha256",
            "evaluation_source_sha256",
            "runtime_sha256",
            "prerequisite_binding_sha256",
            "selected_recipe_sha256",
            "selected_arm",
        )
    }
    if any(len(values) != 1 for values in hashes.values()):
        raise ContractError("stage run provenance hashes differ")
    gate: dict[str, Any] = {
        "schema_version": 1,
        "status": "accepted",
        "campaign_id": manifest["campaign_id"],
        "stage_target": target,
        "expected_runs": TRAINING_RUNS,
        "package_protocol_sha256": next(iter(hashes["package_protocol_sha256"])),
        "source_sha256": next(iter(hashes["source_sha256"])),
        "evaluation_source_sha256": next(iter(hashes["evaluation_source_sha256"])),
        "runtime_sha256": next(iter(hashes["runtime_sha256"])),
        "prerequisite_binding_sha256": next(iter(hashes["prerequisite_binding_sha256"])),
        "selected_recipe_sha256": next(iter(hashes["selected_recipe_sha256"])),
        "selected_arm": next(iter(hashes["selected_arm"])),
        "runs": rows,
        "outcome_sanity": outcome_summary,
        "scientific_setting_coverage": scientific_setting_coverage,
    }
    gate["gate_sha256"] = stable_hash(gate)
    return gate


def publish_gate(path: Path, gate: Mapping[str, Any]) -> None:
    if path.exists():
        if read_json(path) != dict(gate):
            raise ContractError(f"immutable stage gate already differs: {path}")
        return
    atomic_json(path, gate)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-target", type=int, choices=STAGE_TARGETS, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--manifest", type=Path, default=CAMPAIGN_DIR / "manifest.json")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    gate = build_gate(manifest, args.stage_target, args.repo_root.resolve())
    print(json.dumps(gate, sort_keys=True, indent=2))
    if args.publish:
        output = Path(manifest["paths"]["run_root"]) / "state" / "stage-gates" / f"STAGE_GATE_{args.stage_target}.json"
        publish_gate(output, gate)
        print(f"published {output}", file=sys.stderr)
    else:
        print("dry-run only: gate not published", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"grounded-formal stage gate error: {exc}", file=sys.stderr)
        raise SystemExit(2)
