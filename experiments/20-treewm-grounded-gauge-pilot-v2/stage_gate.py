#!/usr/bin/env python3
"""Publish immutable 5k causal-selection and 25k outcome gates for Exp20."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

from campaign import (
    ARM_IDS,
    CAMPAIGN_DIR,
    CONTINUATION_ARM_IDS,
    CONTINUATION_RUNS,
    ContractError,
    REPOSITORY_ROOT,
    RUNS,
    SEEDS,
    SETTING_IDS,
    STAGE_TARGETS,
    atomic_json,
    continuation_runs,
    expand_runs,
    load_manifest,
    read_json,
    require,
    run_directory,
    stable_hash,
    trainer_command,
)
from worker import verify_checkpoint, verify_stage_marker


SEPARATE_CLIP_TAGS = (
    "train/grad_norm_world_rest",
    "train/grad_norm_branch_transformer",
    "train/grad_clip_coefficient_world_rest",
    "train/grad_clip_coefficient_branch_transformer",
)
COMMON_GRADIENT_TAGS = (
    "train/grad_norm_world",
    "train/grad_norm_gain",
    "train/grad_clip_coefficient_world",
    "train/grad_clip_coefficient_gain",
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
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


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
    return {
        tag: {step: value for step, (_wall, value) in values.items()}
        for tag, values in merged.items()
    }


def _values(
    metrics: Mapping[str, Mapping[int, float]],
    tag: str,
    target: int,
    *,
    recent: int | None = None,
) -> list[float]:
    lower = max(0, target - recent) if recent is not None else 0
    return [
        float(value)
        for step, value in sorted(metrics.get(tag, {}).items())
        if lower <= step <= target
    ]


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


def _complete_series(
    metrics: Mapping[str, Mapping[int, float]],
    tag: str,
    axis: Sequence[int],
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
    arm_id: str,
) -> dict[str, Any]:
    gate = manifest["stage_acceptance"]
    required = tuple(gate["required_finite_tags"])
    training_tags = set(gate["training_exact_target_tags"])
    samples = {tag: _last(metrics, tag, target) for tag in required}
    last = {tag: (sample[1] if sample is not None else None) for tag, sample in samples.items()}
    last_step = {tag: (sample[0] if sample is not None else None) for tag, sample in samples.items()}
    expected_step = {
        tag: target if tag in training_tags else target - target % int(gate["validation_diagnostic_every_updates"])
        for tag in required
    }
    finite_coverage = all(finite(value) for value in last.values())
    target_appropriate = all(last_step[tag] == expected_step[tag] for tag in required)

    fixed_counts = _values(metrics, "data/validation_fixed_sample_count", target)
    fixed_validation = bool(
        fixed_counts
        and all(finite(value) and value > 0 for value in fixed_counts)
        and len(set(fixed_counts)) == 1
    )

    gradient_window = min(int(gate["gradient_recent_window_updates"]), target)
    gradient_axis = _expected_axis(target, int(gate["training_every_updates"]), gradient_window)
    common_gradient = {tag: _complete_series(metrics, tag, gradient_axis) for tag in COMMON_GRADIENT_TAGS}
    separate_gradient = (
        {tag: _complete_series(metrics, tag, gradient_axis) for tag in SEPARATE_CLIP_TAGS}
        if arm_id == "GS"
        else {}
    )
    complete_gradient_axis = all(values is not None for values in (*common_gradient.values(), *separate_gradient.values()))
    norm_tags = ["train/grad_norm_world", "train/grad_norm_gain"]
    clip_tags = ["train/grad_clip_coefficient_world", "train/grad_clip_coefficient_gain"]
    if arm_id == "GS":
        norm_tags.extend(("train/grad_norm_world_rest", "train/grad_norm_branch_transformer"))
        clip_tags.extend(("train/grad_clip_coefficient_world_rest", "train/grad_clip_coefficient_branch_transformer"))
    gradient_map = {**common_gradient, **separate_gradient}
    norm_values = [value for tag in norm_tags for value in (gradient_map.get(tag) or [])]
    clip_values = [value for tag in clip_tags for value in (gradient_map.get(tag) or [])]
    gradients_nonzero = bool(
        complete_gradient_axis
        and norm_values
        and all(value > float(gate["min_gradient_norm"]) for value in norm_values)
    )
    low_clip_fraction_by_tag = {
        tag: (
            sum(value < float(gate["min_clip_coefficient"]) for value in (gradient_map.get(tag) or []))
            / len(gradient_map.get(tag) or [])
            if gradient_map.get(tag)
            else None
        )
        for tag in clip_tags
    }
    low_clip_fraction = (
        max(float(value) for value in low_clip_fraction_by_tag.values())
        if low_clip_fraction_by_tag
        and all(finite(value) for value in low_clip_fraction_by_tag.values())
        else None
    )
    clip_coefficients_valid = bool(
        complete_gradient_axis
        and clip_values
        and all(0.0 < value <= 1.0 for value in clip_values)
    )
    clipping_saturation_bounded = bool(
        clip_coefficients_valid
        and finite(low_clip_fraction)
        and all(
            float(value) <= float(gate["max_clip_fraction_below_threshold"])
            for value in low_clip_fraction_by_tag.values()
        )
    )
    clipping_bounded = bool(clip_coefficients_valid and clipping_saturation_bounded)

    gauge_window = min(int(gate["gauge_recent_window_updates"]), target)
    gauge_axis = _expected_axis(target, int(gate["training_every_updates"]), gauge_window)
    recent_ratio = _complete_series(metrics, "latent_gauge/min_ratio", gauge_axis)
    root_ratio = last.get("latent_gauge/root/ratio")
    future_ratio = last.get("latent_gauge/future/ratio")
    min_ratio = last.get("latent_gauge/min_ratio")
    ratio_consistent = bool(
        finite(root_ratio)
        and finite(future_ratio)
        and finite(min_ratio)
        and float(root_ratio) > 0.0
        and float(future_ratio) > 0.0
        and float(min_ratio) > 0.0
        # Tracker values are 50-update means.  E[min(root,future)] need not
        # equal min(E[root],E[future]), but it cannot exceed it.
        and float(min_ratio) <= min(float(root_ratio), float(future_ratio)) + 1e-5
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
        recent_ratio is not None
        and ratio_consistent
        and reference_valid
        and float(min_ratio) >= float(gate["min_scale_ratio"])
        and float(recent_min_ratio) >= float(gate["min_scale_ratio"])
    )

    val_values = _values(metrics, "val/loss_total", target)
    validation_stable = bool(
        val_values
        and all(finite(value) and value >= 0.0 for value in val_values)
        and val_values[-1] <= min(val_values) * (1.0 + float(gate["max_validation_regret_fraction"]))
    )
    self_fed_values = _values(metrics, "val/loss_multistep_self_fed", target)
    self_fed_stable = bool(
        self_fed_values
        and all(finite(value) and value >= 0.0 for value in self_fed_values)
        and self_fed_values[-1] <= min(self_fed_values) * (1.0 + float(gate["max_self_fed_multistep_validation_regret_fraction"]))
    )
    horizon_fractions = [
        last.get(f"data/validation_horizon_label_fraction_h{horizon}")
        for horizon in (4, 8, 16, 32, 64)
    ]
    horizon_distribution_valid = bool(
        all(finite(value) and float(value) >= 0.0 for value in horizon_fractions)
        and abs(sum(float(value) for value in horizon_fractions) - 1.0)
        <= float(gate["horizon_label_fraction_sum_tolerance"])
    )
    horizon_prior_entropy = (
        -sum(float(value) * math.log(max(float(value), 1e-12)) for value in horizon_fractions)
        if horizon_distribution_valid
        else None
    )
    horizon_loss = last.get("val/loss_horizon")
    horizon_pass = bool(
        finite(horizon_loss)
        and finite(horizon_prior_entropy)
        and float(horizon_loss) < float(gate["horizon_uniform_cross_entropy"])
        and float(horizon_loss) < float(horizon_prior_entropy)
    )
    q_pass = bool(
        last.get("control/retrieval_uses_task_metric_endpoint") == 1.0
        and finite(last.get("control/q_advantage_over_z"))
        and finite(last.get("control/q_advantage_over_random_proj"))
        and float(last["control/q_advantage_over_z"]) > float(gate["min_q_advantage"])
        and float(last["control/q_advantage_over_random_proj"]) > float(gate["min_q_advantage"])
    )
    gain_window = min(5_000, target)
    recent_gain = {
        tag: _values(metrics, tag, target, recent=gain_window)
        for tag in GAIN_TAGS
    }
    gain_mean = {
        tag: statistics.fmean(values) if values and all(finite(value) for value in values) else None
        for tag, values in recent_gain.items()
    }
    gain_pass = bool(
        finite(gain_mean["expansion/gain_rank_correlation"])
        and float(gain_mean["expansion/gain_rank_correlation"]) >= float(gate["min_gain_rank_correlation"])
        and finite(gain_mean["expansion/gain_pairwise_accuracy"])
        and float(gain_mean["expansion/gain_pairwise_accuracy"]) >= float(gate["min_gain_pairwise_accuracy"])
        and finite(gain_mean["expansion/gain_eligible_decision_fraction"])
        and float(gain_mean["expansion/gain_eligible_decision_fraction"]) >= float(gate["min_gain_eligible_decision_fraction"])
        and finite(gain_mean["expansion/gain_ordered_pair_count"])
        and float(gain_mean["expansion/gain_ordered_pair_count"]) >= float(gate["min_gain_ordered_pair_count"])
        and finite(gain_mean["expansion/gain_pair_coverage_fraction"])
        and float(gain_mean["expansion/gain_pair_coverage_fraction"]) >= float(gate["min_gain_pair_coverage_fraction"])
    )
    support_pass = bool(
        finite(last.get("tree/support_recall"))
        and finite(last.get("tree/support_precision"))
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
    # N is the deliberately nonpromotable causal control.  Its finite, complete,
    # exact-boundary execution must be trustworthy, but observed clipping saturation,
    # scale collapse, or method underperformance is the causal outcome being measured.
    # G/GS still consume the full integrity dictionary below without threshold changes.
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
    promotable = arm_id in CONTINUATION_ARM_IDS
    return {
        "integrity_passed": integrity_passed,
        "structural_integrity_passed": structural_integrity_passed,
        "method_passed": method_passed,
        "gauge_absolute_passed": gauge_absolute,
        "candidate_passed": bool(promotable and integrity_passed and method_passed and gauge_absolute),
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
        "clip_fraction_below_threshold_by_tag": low_clip_fraction_by_tag,
        "clip_coefficients_valid": clip_coefficients_valid,
        "clip_saturation_bounded": clipping_saturation_bounded,
        "recent_gain_mean": gain_mean,
        "horizon_empirical_prior_entropy": horizon_prior_entropy,
    }


def validate_stage_complete(
    path: Path,
    launch: Mapping[str, Any],
    target: int,
    stage_slot: int,
) -> dict[str, Any]:
    value = read_json(path)
    claimed = value.get("stage_complete_sha256")
    body = dict(value)
    body.pop("stage_complete_sha256", None)
    require(claimed == stable_hash(body), f"stage completion hash differs: {path}")
    run = launch["run"]
    hashes = launch["hashes"]
    checks = (
        value.get("schema_version") == 1,
        value.get("status") == "stage_complete_awaiting_campaign_gate",
        int(value.get("stage_slot", -1)) == stage_slot,
        int(value.get("index", -1)) == int(run["index"]),
        value.get("setting_id") == run["setting_id"],
        value.get("arm_id") == run["arm_id"],
        int(value.get("seed", -1)) == int(run["seed"]),
        int(value.get("stage_target", -1)) == target,
        value.get("launch_sha256") == launch["launch_sha256"],
        value.get("package_protocol_sha256") == hashes["package_protocol_sha256"],
        value.get("source_sha256") == hashes["source_sha256"],
        value.get("runtime_sha256") == hashes["runtime_sha256"],
        sha(value.get("identity_sha256")),
        sha(value.get("checkpoint_sha256")),
        value.get("evaluation_seed_tables_sha256") == hashes["evaluation_seed_tables_sha256"],
        value.get("final_seed_table_sha256") == hashes["final_seed_table_sha256"],
    )
    require(all(checks), f"stage completion identity differs: {path}")
    return value


def _read_gate(path: Path, *, expected_target: int) -> dict[str, Any]:
    value = read_json(path)
    claimed = value.get("gate_sha256")
    body = dict(value)
    body.pop("gate_sha256", None)
    require(claimed == stable_hash(body), f"gate hash differs: {path}")
    require(int(value.get("stage_target", -1)) == expected_target, "gate target differs")
    return value


def validate_skip(
    path: Path,
    launch: Mapping[str, Any],
    stage_slot: int,
    gate_5000: Mapping[str, Any],
) -> dict[str, Any]:
    value = read_json(path)
    claimed = value.get("skip_sha256")
    body = dict(value)
    body.pop("skip_sha256", None)
    require(claimed == stable_hash(body), f"selection skip hash differs: {path}")
    run = launch["run"]
    prior = next(
        (row for row in gate_5000["runs"] if int(row["index"]) == int(run["index"])),
        None,
    )
    checks = (
        value.get("schema_version") == 1,
        value.get("status") == "skipped_by_immutable_5000_selection",
        int(value.get("stage_target", -1)) == STAGE_TARGETS[1],
        int(value.get("stage_slot", -1)) == stage_slot,
        int(value.get("index", -1)) == int(run["index"]),
        value.get("setting_id") == run["setting_id"],
        value.get("arm_id") == run["arm_id"],
        int(value.get("seed", -1)) == int(run["seed"]),
        value.get("selected_arm") == gate_5000["selected_arm"],
        value.get("gate_sha256") == gate_5000["gate_sha256"],
        value.get("launch_sha256") == launch["launch_sha256"],
        value.get("trainer_launched") is False,
        int(value.get("completed_updates", -1)) == STAGE_TARGETS[0],
        prior is not None,
        value.get("identity_sha256") == (prior or {}).get("identity_sha256"),
        value.get("checkpoint_sha256") == (prior or {}).get("checkpoint_sha256"),
    )
    require(all(checks), f"selection skip identity differs: {path}")
    return value


def _common_hashes(launches: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    result: dict[str, str] = {}
    for key in ("source_sha256", "runtime_sha256", "package_protocol_sha256", "actual_evaluation_bank_sha256"):
        values = {str(launch["hashes"][key]) for launch in launches}
        require(len(values) == 1, f"fleet {key} differs")
        result[key] = next(iter(values))
    return result


def _validate_common_samples(rows: Sequence[Mapping[str, Any]], launches: Sequence[Mapping[str, Any]]) -> None:
    by_index = {int(launch["run"]["index"]): launch for launch in launches}
    for setting_id in SETTING_IDS:
        setting_rows = [row for row in rows if row["setting_id"] == setting_id]
        require(setting_rows, f"{setting_id}: no rows")
        validation_hashes = {
            by_index[int(row["index"])]["hashes"]["validation_manifest_sha256"]
            for row in setting_rows
        }
        require(len(validation_hashes) == 1, f"{setting_id}: validation sample source differs")
        fixed_counts = {
            float(row["health"]["last"]["data/validation_fixed_sample_count"])
            for row in setting_rows
        }
        require(len(fixed_counts) == 1, f"{setting_id}: fixed validation count differs")


def _stage_5000_gate(manifest: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    runs = expand_runs(manifest)
    launches = [trainer_command(manifest, run, repo_root=repo_root) for run in runs]
    for stage_slot, (run, launch) in enumerate(zip(runs, launches, strict=True)):
        try:
            run_dir = run_directory(manifest, run)
            complete = validate_stage_complete(
                run_dir / "stage-gates" / f"STAGE_COMPLETE_{STAGE_TARGETS[0]}.json",
                launch,
                STAGE_TARGETS[0],
                stage_slot,
            )
            marker = verify_stage_marker(run_dir, STAGE_TARGETS[0], launch)
            require(complete["checkpoint_sha256"] == marker["checkpoint_sha256"], "worker/marker checkpoint differs")
            health = evaluate_metrics(manifest, event_scalars(run_dir), STAGE_TARGETS[0], run.arm_id)
            require(health["structural_integrity_passed"], "structural execution integrity gate failed")
            rows.append({
                "index": run.index,
                "stage_slot": stage_slot,
                "run_name": run.run_name,
                "setting_id": run.setting_id,
                "arm_id": run.arm_id,
                "seed": run.seed,
                "launch_sha256": launch["launch_sha256"],
                "identity_sha256": complete["identity_sha256"],
                "checkpoint_sha256": complete["checkpoint_sha256"],
                "health": health,
            })
        except (ContractError, OSError, ValueError) as exc:
            failures.append({"stage_slot": stage_slot, "run_name": run.run_name, "error": str(exc)})
    require(not failures and len(rows) == RUNS, f"5k gate requires 30/30 exact rows: {json.dumps(failures, sort_keys=True)}")
    _validate_common_samples(rows, launches)

    by_key = {(row["setting_id"], int(row["seed"]), row["arm_id"]): row for row in rows}
    candidate_summary: dict[str, Any] = {}
    for arm_id in CONTINUATION_ARM_IDS:
        arm_rows = [row for row in rows if row["arm_id"] == arm_id]
        require(len(arm_rows) == 10, f"{arm_id}: candidate coverage is not 10/10")
        deltas = [
            float(by_key[(setting, seed, arm_id)]["health"]["recent_gauge_min_ratio"])
            - float(by_key[(setting, seed, "N")]["health"]["recent_gauge_min_ratio"])
            for setting in SETTING_IDS
            for seed in SEEDS
        ]
        mean_delta = statistics.fmean(deltas)
        universal = all(bool(row["health"]["candidate_passed"]) for row in arm_rows)
        causal = mean_delta > float(manifest["stage_acceptance"]["min_paired_mean_ratio_delta_vs_n"])
        candidate_summary[arm_id] = {
            "universal_cells_passed": sum(bool(row["health"]["candidate_passed"]) for row in arm_rows),
            "required_cells": 10,
            "paired_ratio_deltas_vs_n": deltas,
            "paired_mean_ratio_delta_vs_n": mean_delta,
            "causal_scale_retention_passed": causal,
            "eligible": bool(universal and causal),
        }
    selected = next(
        (arm for arm in manifest["design"]["promotion_precedence"] if candidate_summary[arm]["eligible"]),
        None,
    )
    require(selected in CONTINUATION_ARM_IDS, "neither G nor GS cleared the universal absolute+causal 5k gate")
    hashes = _common_hashes(launches)
    gate = {
        "schema_version": 1,
        "status": "accepted_for_selected_continuation",
        "campaign_id": manifest["campaign_id"],
        "formal_validation": False,
        "stage_target": STAGE_TARGETS[0],
        "selected_arm": selected,
        "selection_precedence": list(manifest["design"]["promotion_precedence"]),
        "nonpromotable_arm": "N",
        "candidate_summary": candidate_summary,
        "runs": rows,
        **hashes,
    }
    gate["gate_sha256"] = stable_hash(gate)
    return gate


def _outcome_summary(
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    gate = manifest["stage_acceptance"]
    expected_episodes = float(gate["outcome_episodes_per_run"])
    for row in rows:
        outcome = row["outcome"]
        require(
            outcome["num_episodes"] == expected_episodes
            and finite(outcome["successes"])
            and finite(outcome["success_rate"])
            and finite(outcome["distance_reduction_frac"]),
            "25k outcome telemetry is incomplete",
        )
        successes = float(outcome["successes"])
        require(successes.is_integer() and 0.0 <= successes <= expected_episodes, "success count invalid")
        require(abs(float(outcome["success_rate"]) - successes / expected_episodes) <= 1e-6, "success rate/count mismatch")

    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        seed_rows = [row for row in rows if int(row["seed"]) == seed]
        successes = sum(float(row["outcome"]["successes"]) for row in seed_rows)
        mean_progress = statistics.fmean(float(row["outcome"]["distance_reduction_frac"]) for row in seed_rows)
        require(successes >= int(gate["min_total_successes_per_seed"]), f"seed {seed}: all-zero success")
        require(mean_progress > float(gate["min_mean_distance_reduction_per_seed"]), f"seed {seed}: nonpositive mean progress")
        per_seed[str(seed)] = {
            "successes": successes,
            "mean_distance_reduction_frac": mean_progress,
        }

    both_success = 0
    both_progress = 0
    per_setting: dict[str, Any] = {}
    for setting in SETTING_IDS:
        setting_rows = [row for row in rows if row["setting_id"] == setting]
        require({int(row["seed"]) for row in setting_rows} == set(SEEDS), f"{setting}: missing replicated seed")
        replicated_success = all(float(row["outcome"]["successes"]) > 0.0 for row in setting_rows)
        replicated_progress = all(float(row["outcome"]["distance_reduction_frac"]) > 0.0 for row in setting_rows)
        both_success += replicated_success
        both_progress += replicated_progress
        per_setting[setting] = {
            "both_seed_nonzero_success": replicated_success,
            "both_seed_positive_progress": replicated_progress,
        }
    require(both_success >= int(gate["min_settings_with_both_seed_success"]), "no replicated nonzero-SR setting quorum")
    require(both_progress >= int(gate["min_settings_with_both_seed_positive_progress"]), "replicated positive-progress setting quorum failed")
    return {
        "per_seed": per_seed,
        "per_setting": per_setting,
        "settings_with_both_seed_nonzero_success": both_success,
        "settings_with_both_seed_positive_progress": both_progress,
        "total_successes": sum(float(row["outcome"]["successes"]) for row in rows),
        "macro_success_rate": statistics.fmean(float(row["outcome"]["success_rate"]) for row in rows),
        "macro_distance_reduction_frac": statistics.fmean(float(row["outcome"]["distance_reduction_frac"]) for row in rows),
    }


def _stage_25000_gate(manifest: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    gate_5000 = _read_gate(
        Path(manifest["paths"]["run_root"]) / "state" / "stage-gates" / f"STAGE_GATE_{STAGE_TARGETS[0]}.json",
        expected_target=STAGE_TARGETS[0],
    )
    require(gate_5000.get("status") == "accepted_for_selected_continuation", "5k continuation was not accepted")
    selected = gate_5000.get("selected_arm")
    require(selected in CONTINUATION_ARM_IDS, "5k selected arm is invalid")

    selected_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    runs = continuation_runs(manifest)
    launches = [trainer_command(manifest, run, repo_root=repo_root) for run in runs]
    for stage_slot, (run, launch) in enumerate(zip(runs, launches, strict=True)):
        run_dir = run_directory(manifest, run)
        try:
            if run.arm_id != selected:
                skip = validate_skip(
                    run_dir / "stage-gates" / f"SKIPPED_BY_SELECTION_{STAGE_TARGETS[1]}.json",
                    launch,
                    stage_slot,
                    gate_5000,
                )
                checkpoint = verify_checkpoint(
                    run_dir / "checkpoints" / "latest.pt",
                    launch,
                    expected_step=STAGE_TARGETS[0],
                )
                require(checkpoint["checkpoint_sha256"] == skip["checkpoint_sha256"], "skipped checkpoint advanced/changed")
                skipped_rows.append({
                    "index": run.index,
                    "stage_slot": stage_slot,
                    "setting_id": run.setting_id,
                    "arm_id": run.arm_id,
                    "seed": run.seed,
                    "launch_sha256": launch["launch_sha256"],
                    "checkpoint_sha256": checkpoint["checkpoint_sha256"],
                    "skip_sha256": skip["skip_sha256"],
                })
                continue

            complete = validate_stage_complete(
                run_dir / "stage-gates" / f"STAGE_COMPLETE_{STAGE_TARGETS[1]}.json",
                launch,
                STAGE_TARGETS[1],
                stage_slot,
            )
            marker = verify_stage_marker(run_dir, STAGE_TARGETS[1], launch)
            require(complete["checkpoint_sha256"] == marker["checkpoint_sha256"], "worker/marker checkpoint differs")
            scalars = event_scalars(run_dir)
            health = evaluate_metrics(manifest, scalars, STAGE_TARGETS[1], run.arm_id)
            require(health["candidate_passed"], "selected terminal cell failed gauge/method/integrity")
            selected_rows.append({
                "index": run.index,
                "stage_slot": stage_slot,
                "run_name": run.run_name,
                "setting_id": run.setting_id,
                "arm_id": run.arm_id,
                "seed": run.seed,
                "launch_sha256": launch["launch_sha256"],
                "identity_sha256": complete["identity_sha256"],
                "checkpoint_sha256": complete["checkpoint_sha256"],
                "health": health,
                "outcome": {
                    "num_episodes": _at(scalars, "eval/num_episodes", STAGE_TARGETS[1]),
                    "successes": _at(scalars, "eval/successes", STAGE_TARGETS[1]),
                    "success_rate": _at(scalars, "eval/success_rate", STAGE_TARGETS[1]),
                    "distance_reduction_frac": _at(scalars, "eval/distance_reduction_frac", STAGE_TARGETS[1]),
                },
            })
        except (ContractError, OSError, ValueError) as exc:
            failures.append({"stage_slot": stage_slot, "run_name": run.run_name, "error": str(exc)})

    require(not failures, f"25k gate has invalid slots: {json.dumps(failures, sort_keys=True)}")
    require(len(selected_rows) == 10 and len(skipped_rows) == 10, "25k selected/skip coverage is not 10+10")
    require({row["setting_id"] for row in selected_rows} == set(SETTING_IDS), "selected arm lacks setting coverage")
    require(all(sum(row["setting_id"] == setting for row in selected_rows) == 2 for setting in SETTING_IDS), "selected arm lacks two seeds per setting")
    _validate_common_samples(selected_rows, launches)
    outcome = _outcome_summary(manifest, selected_rows)
    hashes = _common_hashes(launches)
    require(gate_5000["package_protocol_sha256"] == hashes["package_protocol_sha256"], "5k/25k protocol differs")
    require(gate_5000["source_sha256"] == hashes["source_sha256"], "5k/25k source differs")
    require(gate_5000["runtime_sha256"] == hashes["runtime_sha256"], "5k/25k runtime differs")
    gate = {
        "schema_version": 1,
        "status": "accepted_for_fresh_formal_campaign_design",
        "campaign_id": manifest["campaign_id"],
        "formal_validation": False,
        "stage_target": STAGE_TARGETS[1],
        "selected_arm": selected,
        "stage_5000_gate_sha256": gate_5000["gate_sha256"],
        "selected_runs": selected_rows,
        "skipped_runs": skipped_rows,
        "outcome": outcome,
        **hashes,
    }
    gate["gate_sha256"] = stable_hash(gate)
    return gate


def build_gate(manifest: Mapping[str, Any], target: int, repo_root: Path) -> dict[str, Any]:
    require(target in STAGE_TARGETS, "invalid stage target")
    return (
        _stage_5000_gate(manifest, repo_root)
        if target == STAGE_TARGETS[0]
        else _stage_25000_gate(manifest, repo_root)
    )


def publish_gate(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        require(read_json(path) == dict(value), f"immutable stage gate already differs: {path}")
        return
    atomic_json(path, value)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-target", type=int, choices=STAGE_TARGETS, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--manifest", type=Path, default=CAMPAIGN_DIR / "manifest.json")
    parser.add_argument("--publish", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    gate = build_gate(manifest, args.stage_target, args.repo_root.resolve())
    print(json.dumps(gate, sort_keys=True, indent=2))
    if args.publish:
        output = (
            Path(manifest["paths"]["run_root"])
            / "state"
            / "stage-gates"
            / f"STAGE_GATE_{args.stage_target}.json"
        )
        publish_gate(output, gate)
        print(f"published {output}", file=sys.stderr)
    else:
        print("dry-run only: gate not published", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"grounded-gauge-pilot stage gate error: {exc}", file=sys.stderr)
        raise SystemExit(2)
