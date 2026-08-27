#!/usr/bin/env python3
"""Strict preregistered health, outcome, and scale-selection report for bridge 16."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

from campaign import (
    ARM_IDS,
    CAMPAIGN_DIR,
    ContractError,
    REPOSITORY_ROOT,
    RUNS,
    SEEDS,
    SETTING_IDS,
    actual_evaluation_bank,
    atomic_json,
    expand_runs,
    load_manifest,
    manifest_sha256,
    run_directory,
    stable_hash,
    trainer_command,
    verify_protocol_lock,
    verify_source_snapshot,
)
from worker import read_optional_json, verify_completion


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def event_scalars(run_dir: Path) -> dict[str, dict[int, float]]:
    """Merge requeue-created event files and retain the latest wall-time per step."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    paths = sorted(run_dir.glob("events.out.tfevents.*"))
    if not paths:
        paths = sorted(run_dir.rglob("events.out.tfevents.*"))
    if not paths:
        raise ContractError(f"no TensorBoard scalar artifacts in {run_dir}")
    merged: dict[str, dict[int, tuple[float, float]]] = {}
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


def _at(metrics: Mapping[str, Mapping[int, float]], tag: str, step: int) -> float | None:
    value = metrics.get(tag, {}).get(step)
    return float(value) if finite(value) else None


def _last(metrics: Mapping[str, Mapping[int, float]], tag: str, target: int = 25_000) -> float | None:
    values = [(step, value) for step, value in metrics.get(tag, {}).items() if step <= target and finite(value)]
    return float(max(values)[1]) if values else None


def _window(
    metrics: Mapping[str, Mapping[int, float]],
    tag: str,
    *,
    lower: int,
    upper: int = 25_000,
) -> list[float]:
    return [
        float(value)
        for step, value in sorted(metrics.get(tag, {}).items())
        if lower <= step <= upper and finite(value)
    ]


def _strict_regret(values: Sequence[float], fraction: float) -> bool:
    return bool(values and values[-1] <= min(values) * (1.0 + float(fraction)))


def evaluate_run(
    manifest: Mapping[str, Any],
    run,
    *,
    repo_root: Path,
) -> dict[str, Any]:
    run_dir = run_directory(manifest, run)
    launch = read_optional_json(run_dir / "BRIDGE_LAUNCH.json")
    expected_launch = trainer_command(manifest, run, repo_root=repo_root)
    if launch is None or launch != expected_launch:
        raise ContractError(f"{run.run_name}: launch contract is missing or differs")
    completion_path = run_dir / "COMPLETED.json"
    completion = read_optional_json(completion_path)
    if completion is None:
        raise ContractError(f"{run.run_name}: completion sentinel is missing")
    verified_completion = verify_completion(completion_path, expected_launch, manifest)
    metrics = event_scalars(run_dir)
    acceptance = manifest["acceptance"]
    checkpoint_steps = list(acceptance["checkpoint_steps"])

    required_terminal = (
        "val/loss_total",
        "val/loss_horizon",
        "val/loss_multistep",
        "val/loss_multistep_self_fed",
        "data/validation_fixed_sample_count",
        "data/validation_horizon_label_fraction_h4",
        "data/validation_horizon_label_fraction_h8",
        "data/validation_horizon_label_fraction_h16",
        "data/validation_horizon_label_fraction_h32",
        "data/validation_horizon_label_fraction_h64",
        "control/q_advantage_over_z",
        "control/q_advantage_over_random_proj",
        "control/retrieval_uses_task_metric_endpoint",
        "expansion/gain_rank_correlation",
        "expansion/gain_pairwise_accuracy",
        "expansion/gain_eligible_decision_fraction",
        "expansion/gain_ordered_pair_count",
        "expansion/gain_pair_coverage_fraction",
        "tree/support_recall",
        "tree/support_precision",
        "train/grad_norm_world",
        "train/grad_norm_gain",
        "train/grad_clip_coefficient_world",
        "train/grad_clip_coefficient_gain",
    )
    terminal = {tag: _last(metrics, tag) for tag in required_terminal}
    terminal_telemetry = all(finite(value) for value in terminal.values())

    fixed_values = [_at(metrics, "data/validation_fixed_sample_count", step) for step in checkpoint_steps]
    fixed_validation = bool(
        all(finite(value) and float(value) > 0.0 for value in fixed_values)
        and len({float(value) for value in fixed_values}) == 1
    )
    val_series = [_at(metrics, "val/loss_total", step) for step in checkpoint_steps]
    self_fed_series = [_at(metrics, "val/loss_multistep_self_fed", step) for step in checkpoint_steps]
    exact_validation_axis = all(finite(value) for value in [*val_series, *self_fed_series])

    midpoint = {
        "num_episodes": _at(metrics, "eval/num_episodes", 12_500),
        "successes": _at(metrics, "eval/successes", 12_500),
        "success_rate": _at(metrics, "eval/success_rate", 12_500),
        "distance_reduction_frac": _at(metrics, "eval/distance_reduction_frac", 12_500),
    }
    midpoint_present = bool(
        all(finite(value) for value in midpoint.values())
        and int(float(midpoint["num_episodes"])) == 5
    )
    final_eval = completion.get("final_evaluation") or {}
    final = {
        "num_episodes": final_eval.get("eval/num_episodes"),
        "successes": final_eval.get("eval/successes"),
        "success_rate": final_eval.get("eval/success_rate"),
        "distance_reduction_frac": final_eval.get("eval/distance_reduction_frac"),
    }
    final_present = bool(
        all(finite(value) for value in final.values())
        and int(float(final["num_episodes"])) == 25
        and float(final["successes"]).is_integer()
        and 0.0 <= float(final["successes"]) <= 25.0
        and abs(float(final["success_rate"]) - float(final["successes"]) / 25.0) <= 1e-6
    )
    bank_match = verified_completion["actual_evaluation_bank_sha256"] == actual_evaluation_bank(manifest)["sha256"]
    integrity_gates = {
        "exact_launch_completion_and_provenance": True,
        "identical_terminal_evaluation_bank": bank_match,
        "fixed_common_validation_sample": fixed_validation,
        "exact_1k_validation_axis": exact_validation_axis,
        "finite_terminal_method_telemetry": terminal_telemetry,
        "midpoint_and_terminal_rollouts": midpoint_present and final_present,
    }

    val_float = [float(value) for value in val_series if finite(value)]
    self_fed_float = [float(value) for value in self_fed_series if finite(value)]
    validation_pass = exact_validation_axis and _strict_regret(
        val_float, acceptance["max_validation_regret_fraction"]
    )
    self_fed_pass = exact_validation_axis and _strict_regret(
        self_fed_float,
        acceptance["max_self_fed_multistep_validation_regret_fraction"],
    )
    horizon_fractions = [
        terminal[f"data/validation_horizon_label_fraction_h{horizon}"]
        for horizon in (4, 8, 16, 32, 64)
    ]
    horizon_distribution = bool(
        all(finite(value) and float(value) >= 0.0 for value in horizon_fractions)
        and abs(sum(float(value) for value in horizon_fractions) - 1.0)
        <= float(acceptance["horizon_label_fraction_sum_tolerance"])
    )
    horizon_prior = (
        -sum(float(value) * math.log(max(float(value), 1e-12)) for value in horizon_fractions)
        if horizon_distribution
        else None
    )
    horizon_loss = terminal["val/loss_horizon"]
    horizon_pass = bool(
        finite(horizon_loss)
        and finite(horizon_prior)
        and float(horizon_loss) < float(acceptance["horizon_uniform_cross_entropy"])
        and float(horizon_loss) < float(horizon_prior)
    )
    q_pass = bool(
        terminal["control/retrieval_uses_task_metric_endpoint"] == 1.0
        and float(terminal["control/q_advantage_over_z"] or -math.inf) > float(acceptance["min_q_advantage"])
        and float(terminal["control/q_advantage_over_random_proj"] or -math.inf) > float(acceptance["min_q_advantage"])
    )
    lower = 25_000 - int(acceptance["gain_recent_window_updates"])
    gain_tags = (
        "expansion/gain_rank_correlation",
        "expansion/gain_pairwise_accuracy",
        "expansion/gain_eligible_decision_fraction",
        "expansion/gain_ordered_pair_count",
        "expansion/gain_pair_coverage_fraction",
    )
    gain_windows = {tag: _window(metrics, tag, lower=lower) for tag in gain_tags}
    gain_mean = {
        tag: statistics.fmean(values) if values else None
        for tag, values in gain_windows.items()
    }
    gain_pass = bool(
        all(finite(value) for value in gain_mean.values())
        and float(gain_mean["expansion/gain_rank_correlation"]) >= float(acceptance["min_gain_rank_correlation"])
        and float(gain_mean["expansion/gain_pairwise_accuracy"]) >= float(acceptance["min_gain_pairwise_accuracy"])
        and float(gain_mean["expansion/gain_eligible_decision_fraction"]) >= float(acceptance["min_gain_eligible_decision_fraction"])
        and float(gain_mean["expansion/gain_ordered_pair_count"]) >= float(acceptance["min_gain_ordered_pair_count"])
        and float(gain_mean["expansion/gain_pair_coverage_fraction"]) >= float(acceptance["min_gain_pair_coverage_fraction"])
    )
    support_pass = bool(
        float(terminal["tree/support_recall"] or -math.inf) >= float(acceptance["min_support_recall"])
        and float(terminal["tree/support_precision"] or -math.inf) >= float(acceptance["min_support_precision"])
    )
    world_norm = _window(metrics, "train/grad_norm_world", lower=lower)
    gain_norm = _window(metrics, "train/grad_norm_gain", lower=lower)
    world_clip = _window(metrics, "train/grad_clip_coefficient_world", lower=lower)
    gain_clip = _window(metrics, "train/grad_clip_coefficient_gain", lower=lower)
    clip_values = [*world_clip, *gain_clip]
    low_clip_fraction = (
        sum(value < float(acceptance["min_clip_coefficient"]) for value in clip_values) / len(clip_values)
        if clip_values
        else None
    )
    gradient_pass = bool(
        world_norm
        and gain_norm
        and clip_values
        and all(value > float(acceptance["min_gradient_norm"]) for value in [*world_norm, *gain_norm])
        and finite(low_clip_fraction)
        and float(low_clip_fraction) <= float(acceptance["max_clip_fraction_below_threshold"])
    )
    scientific_gates = {
        "validation_regret_le_1p10": validation_pass,
        "self_fed_multistep_regret_le_1p10": self_fed_pass,
        "horizon_below_uniform_and_empirical_prior": horizon_pass,
        "q_advantage": q_pass,
        "gain_rank_pair_eligibility_and_coverage": gain_pass,
        "support_recall_and_precision": support_pass,
        "aggregate_shared_module_gradients_and_clipping": gradient_pass,
    }
    return {
        "index": run.index,
        "setting_id": run.setting_id,
        "arm_id": run.arm_id,
        "seed": run.seed,
        "run_name": run.run_name,
        "integrity_gates": integrity_gates,
        "integrity_pass": all(integrity_gates.values()),
        "scientific_gates": scientific_gates,
        "scientific_pass": all(scientific_gates.values()),
        "metrics": {
            "validation_final": val_series[-1],
            "validation_min": min(val_float) if val_float else None,
            "self_fed_final": self_fed_series[-1],
            "self_fed_min": min(self_fed_float) if self_fed_float else None,
            "horizon_loss": horizon_loss,
            "horizon_empirical_prior": horizon_prior,
            "gain_recent_mean": gain_mean,
            "support_recall": terminal["tree/support_recall"],
            "support_precision": terminal["tree/support_precision"],
            "clip_fraction_below_threshold": low_clip_fraction,
            "midpoint": midpoint,
            "final": final,
        },
        "error": None,
    }


def failed_record(run, error: Exception) -> dict[str, Any]:
    return {
        "index": run.index,
        "setting_id": run.setting_id,
        "arm_id": run.arm_id,
        "seed": run.seed,
        "run_name": run.run_name,
        "integrity_gates": {},
        "integrity_pass": False,
        "scientific_gates": {},
        "scientific_pass": False,
        "metrics": {"midpoint": {}, "final": {}},
        "error": str(error),
    }


def aggregate_acceptance(
    manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Apply the locked global F-then-H selection rule without domain tuning."""
    acceptance = manifest["acceptance"]
    expected_keys = {
        (setting, arm, seed)
        for setting in SETTING_IDS
        for arm in ARM_IDS
        for seed in SEEDS
    }
    keyed = {
        (str(record["setting_id"]), str(record["arm_id"]), int(record["seed"])): record
        for record in records
    }
    key_exact = set(keyed) == expected_keys and len(records) == RUNS
    integrity_count = sum(bool(record.get("integrity_pass")) for record in records)
    integrity_40_of_40 = key_exact and integrity_count == int(acceptance["integrity_runs_required"])
    full = str(acceptance["preferred_arm"])
    half = str(acceptance["fallback_arm"])
    arm_setting_seed_pass_count = {
        arm: {
            setting: (
                sum(
                    bool(keyed[(setting, arm, seed)].get("scientific_pass"))
                    for seed in SEEDS
                )
                if key_exact
                else 0
            )
            for setting in SETTING_IDS
        }
        for arm in ARM_IDS
    }
    arm_setting_pass = {
        arm: {
            setting: count == int(acceptance["seeds_per_setting_required"])
            for setting, count in counts.items()
        }
        for arm, counts in arm_setting_seed_pass_count.items()
    }
    arm_scientific_runs_passing = {
        arm: (
            sum(
                bool(keyed[(setting, arm, seed)].get("scientific_pass"))
                for setting in SETTING_IDS
                for seed in SEEDS
            )
            if key_exact
            else 0
        )
        for arm in ARM_IDS
    }

    paired: list[dict[str, Any]] = []
    if key_exact:
        for setting in SETTING_IDS:
            for seed in SEEDS:
                full_final = keyed[(setting, full, seed)]["metrics"]["final"]
                half_final = keyed[(setting, half, seed)]["metrics"]["final"]
                full_success = full_final.get("success_rate")
                half_success = half_final.get("success_rate")
                full_progress = full_final.get("distance_reduction_frac")
                half_progress = half_final.get("distance_reduction_frac")
                paired.append({
                    "setting_id": setting,
                    "seed": seed,
                    "success_delta_half_minus_full": (
                        float(half_success) - float(full_success)
                        if finite(half_success) and finite(full_success)
                        else None
                    ),
                    "distance_reduction_delta_half_minus_full": (
                        float(half_progress) - float(full_progress)
                        if finite(half_progress) and finite(full_progress)
                        else None
                    ),
                })

    def paired_mean(key: str) -> float | None:
        values = [row.get(key) for row in paired]
        return statistics.fmean(float(value) for value in values) if values and all(finite(value) for value in values) else None

    success_delta = paired_mean("success_delta_half_minus_full")
    progress_delta = paired_mean("distance_reduction_delta_half_minus_full")
    arm_metrics: dict[str, dict[str, Any]] = {}
    for arm in ARM_IDS:
        rows = [
            keyed[(setting, arm, seed)]
            for setting in SETTING_IDS
            for seed in SEEDS
        ] if key_exact else []
        successes = [row["metrics"]["final"].get("successes") for row in rows]
        progress = [row["metrics"]["final"].get("distance_reduction_frac") for row in rows]
        arm_metrics[arm] = {
            "total_successes": (
                sum(float(value) for value in successes)
                if successes and all(finite(value) for value in successes)
                else None
            ),
            "mean_distance_reduction": (
                statistics.fmean(float(value) for value in progress)
                if progress and all(finite(value) for value in progress)
                else None
            ),
            "runs_with_positive_progress": sum(
                finite(value) and float(value) > 0.0 for value in progress
            ),
            "finite_terminal_outcomes": bool(
                successes
                and progress
                and all(finite(value) for value in [*successes, *progress])
            ),
        }

    def outcome_gates(arm: str) -> dict[str, bool]:
        metrics = arm_metrics[arm]
        return {
            "finite_terminal_outcomes": bool(metrics["finite_terminal_outcomes"]),
            "not_all_zero_success": bool(
                finite(metrics["total_successes"])
                and float(metrics["total_successes"])
                >= float(acceptance["min_arm_total_successes"])
            ),
            "positive_mean_progress": bool(
                finite(metrics["mean_distance_reduction"])
                and float(metrics["mean_distance_reduction"])
                > float(acceptance["min_arm_mean_distance_reduction"])
            ),
            "positive_progress_run_quorum": bool(
                int(metrics["runs_with_positive_progress"])
                >= int(acceptance["min_arm_runs_with_positive_progress"])
            ),
        }

    arm_outcome_gates = {arm: outcome_gates(arm) for arm in ARM_IDS}
    arm_scientific_quorum_gates = {
        arm: {
            "scientific_runs_at_least_18_of_20": (
                arm_scientific_runs_passing[arm]
                >= int(acceptance["min_arm_scientific_runs_passing"])
            ),
            "both_seed_settings_at_least_8_of_10": (
                sum(arm_setting_pass[arm].values())
                >= int(acceptance["min_arm_settings_with_both_seeds_passing"])
            ),
            "every_setting_has_at_least_one_passing_seed": (
                sum(
                    count >= 1
                    for count in arm_setting_seed_pass_count[arm].values()
                )
                >= int(
                    acceptance[
                        "min_arm_settings_with_at_least_one_seed_passing"
                    ]
                )
            ),
        }
        for arm in ARM_IDS
    }
    full_scientific_quorum = all(arm_scientific_quorum_gates[full].values())
    half_scientific_quorum = all(arm_scientific_quorum_gates[half].values())
    half_noninferior = {
        "success_noninferior_to_full": bool(
            finite(success_delta)
            and float(success_delta)
            >= float(acceptance["min_paired_fallback_success_delta_vs_full"])
        ),
        "distance_reduction_noninferior_to_full": bool(
            finite(progress_delta)
            and float(progress_delta)
            >= float(acceptance["min_paired_fallback_distance_reduction_delta_vs_full"])
        ),
    }
    full_eligible = bool(
        integrity_40_of_40
        and full_scientific_quorum
        and all(arm_outcome_gates[full].values())
    )
    half_eligible = bool(
        integrity_40_of_40
        and half_scientific_quorum
        and all(arm_outcome_gates[half].values())
        and all(half_noninferior.values())
    )
    selected_arm = full if full_eligible else half if half_eligible else None
    accepted = selected_arm is not None
    arms_by_id = {str(arm["id"]): arm for arm in manifest["arms"]}
    selected_recipe = dict(arms_by_id[selected_arm]) if selected_arm else None
    selected_recipe_sha256 = stable_hash(selected_recipe) if selected_recipe is not None else None
    aggregate_gates = {
        "integrity_40_of_40": integrity_40_of_40,
        "full_scientific_quorum": full_scientific_quorum,
        "half_scientific_quorum": half_scientific_quorum,
        "full_outcome_gates": all(arm_outcome_gates[full].values()),
        "half_outcome_gates": all(arm_outcome_gates[half].values()),
        "half_success_noninferior_to_full": half_noninferior["success_noninferior_to_full"],
        "half_distance_reduction_noninferior_to_full": half_noninferior["distance_reduction_noninferior_to_full"],
        "full_eligible": full_eligible,
        "half_eligible": half_eligible,
        "deterministic_full_then_half_selection": bool(
            acceptance["selection_precedence"] == ["F", "H"]
            and acceptance["no_domain_specific_tuning"]
            and acceptance["no_posthoc_arm_selection"]
            and selected_arm == (full if full_eligible else half if half_eligible else None)
        ),
    }
    return {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "status": (
            "selected_full_for_fresh_formal_campaign_design"
            if selected_arm == full
            else "selected_half_for_fresh_formal_campaign_design"
            if selected_arm == half
            else "rejected_no_recipe_selected"
        ),
        "accepted": accepted,
        "formal_validation": False,
        "claim": "Bounded all-ten 25k bridge result only; never formal validation or a 1M result.",
        "selection_precedence": [full, half],
        "selected_arm": selected_arm,
        "selected_recipe": selected_recipe,
        "selected_recipe_sha256": selected_recipe_sha256,
        "integrity_runs_passing": integrity_count,
        "arm_scientific_runs_passing": arm_scientific_runs_passing,
        "arm_setting_seed_pass_count": arm_setting_seed_pass_count,
        "arm_setting_pass": arm_setting_pass,
        "arm_scientific_quorum_gates": arm_scientific_quorum_gates,
        "arm_outcome_gates": arm_outcome_gates,
        "half_noninferiority_gates": half_noninferior,
        "aggregate_gates": aggregate_gates,
        "aggregate_metrics": {
            "by_arm": arm_metrics,
            "paired_mean_success_delta_half_minus_full": success_delta,
            "paired_mean_distance_reduction_delta_half_minus_full": progress_delta,
        },
        "paired_comparisons": paired,
        "missing_or_extra_keys": sorted(expected_keys.symmetric_difference(keyed)),
        "actual_evaluation_bank_sha256": actual_evaluation_bank(manifest)["sha256"],
        "runs": list(records),
    }


def seal_report(
    manifest: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    repo_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Bind a decision and its formal-consumer artifact to exact bridge inputs."""
    launches = {
        run.run_name: trainer_command(manifest, run, repo_root=repo_root)
        for run in expand_runs(manifest)
    }
    common: dict[str, str] = {}
    for key in (
        "source_sha256",
        "runtime_sha256",
        "package_protocol_sha256",
        "actual_evaluation_bank_sha256",
        "exp15_prerequisite_sha256",
    ):
        values = {launch["hashes"][key] for launch in launches.values()}
        if len(values) != 1:
            raise ContractError(f"report launch {key} differs across runs")
        common[key] = next(iter(values))
    expected_prerequisite = os.environ.get(
        "TREEWM_EXPECTED_EXP15_PREREQUISITE_SHA256"
    )
    if (
        expected_prerequisite
        and expected_prerequisite != common["exp15_prerequisite_sha256"]
    ):
        raise ContractError("exp15 prerequisite changed after bridge submission")
    sealed = dict(report)
    prerequisites = {
        stable_hash(launch["exp15_prerequisite"]): launch["exp15_prerequisite"]
        for launch in launches.values()
    }
    if len(prerequisites) != 1:
        raise ContractError("exp15 prerequisite differs across bridge launches")
    sealed["exp15_prerequisite"] = next(iter(prerequisites.values()))
    sealed["exp15_prerequisite_sha256"] = common["exp15_prerequisite_sha256"]
    sealed["selection_rule_sha256"] = stable_hash({
        "schema_version": 1,
        "design": manifest["design"],
        "acceptance": manifest["acceptance"],
        "arms": manifest["arms"],
        "setting_ids": list(SETTING_IDS),
        "seeds": list(SEEDS),
    })
    sealed["provenance"] = {
        "manifest_sha256": manifest_sha256(manifest),
        **common,
        "exp15_prerequisite": sealed["exp15_prerequisite"],
        "run_config_sha256": {
            name: launch["hashes"]["config_sha256"]
            for name, launch in sorted(launches.items())
        },
        "run_input_contract_sha256": {
            name: launch["hashes"]["input_contract_sha256"]
            for name, launch in sorted(launches.items())
        },
    }
    sealed["report_sha256"] = stable_hash(sealed)
    acceptance_bytes = (
        json.dumps(sealed, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    selected_arm = sealed.get("selected_arm")
    selected_configs = {
        name: launch["hashes"]["config_sha256"]
        for name, launch in sorted(launches.items())
        if launch["run"]["arm_id"] == selected_arm
    }
    selection: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "status": "selected_recipe" if sealed["accepted"] else "no_recipe_selected",
        "selected": bool(sealed["accepted"]),
        "formal_validation": False,
        "selected_arm": selected_arm,
        "selected_recipe": sealed.get("selected_recipe"),
        "selected_recipe_sha256": sealed.get("selected_recipe_sha256"),
        "selection_rule_sha256": sealed["selection_rule_sha256"],
        "bridge_acceptance_sha256": sealed["report_sha256"],
        "acceptance_sha256": hashlib.sha256(acceptance_bytes).hexdigest(),
        "manifest_sha256": sealed["provenance"]["manifest_sha256"],
        "package_protocol_sha256": common["package_protocol_sha256"],
        "source_sha256": common["source_sha256"],
        "runtime_sha256": common["runtime_sha256"],
        "actual_evaluation_bank_sha256": common["actual_evaluation_bank_sha256"],
        "exp15_prerequisite": sealed["exp15_prerequisite"],
        "exp15_prerequisite_sha256": common["exp15_prerequisite_sha256"],
        "selected_run_config_sha256": selected_configs,
    }
    selection["artifact_sha256"] = stable_hash(selection)
    return sealed, selection


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# TreeWM grounded repair all-ten bridge v1 report",
        "",
        f"Decision: **{report['status']}**.",
        "",
        "> This is a bounded all-ten 25k bridge, not formal validation or a 1M result.",
        "",
        f"Integrity-complete runs: {report['integrity_runs_passing']}/40.",
        f"Selected global arm: {report['selected_arm'] or 'none'} (precedence: F, then H).",
        f"Scientific runs passing: F={report['arm_scientific_runs_passing']['F']}/20; H={report['arm_scientific_runs_passing']['H']}/20.",
        "Required selected-arm science: >=18/20, >=8/10 both-seed settings, and >=1 passing seed in every setting.",
        "Required selected-arm outcomes: >=1 success, mean DRF > 0, and >=12/20 positive-progress runs.",
        "",
        "| Setting | Full both-seed pass | Half both-seed pass |",
        "|---|---:|---:|",
    ]
    lines.extend(
        f"| {setting} | {'yes' if report['arm_setting_pass']['F'][setting] else 'no'} | {'yes' if report['arm_setting_pass']['H'][setting] else 'no'} |"
        for setting in SETTING_IDS
    )
    lines.extend(["", "Aggregate gates:", ""])
    lines.extend(
        f"- {name}: {'pass' if passed else 'fail'}"
        for name, passed in report["aggregate_gates"].items()
    )
    return "\n".join(lines) + "\n"


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CAMPAIGN_DIR / "manifest.json")
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    manifest = load_manifest(args.manifest)
    verify_protocol_lock(root / "experiments" / "16-treewm-grounded-repair-all-ten-bridge-v1")
    if root.parent.joinpath("SNAPSHOT.json").exists():
        verify_source_snapshot(root)
    records: list[dict[str, Any]] = []
    for run in expand_runs(manifest):
        try:
            records.append(evaluate_run(manifest, run, repo_root=root))
        except (ContractError, OSError, ValueError, KeyError) as exc:
            records.append(failed_record(run, exc))
    report, selection = seal_report(
        manifest,
        aggregate_acceptance(manifest, records),
        repo_root=root,
    )
    print(json.dumps({
        "status": report["status"],
        "accepted": report["accepted"],
        "formal_validation": report["formal_validation"],
        "integrity_runs_passing": report["integrity_runs_passing"],
        "selected_arm": report["selected_arm"],
        "arm_scientific_runs_passing": report["arm_scientific_runs_passing"],
        "report_sha256": report["report_sha256"],
    }, sort_keys=True, indent=2))
    if args.publish:
        output = args.output_dir or Path(manifest["paths"]["run_root"]) / "reports"
        atomic_json(output / "acceptance.json", report)
        atomic_text(output / "acceptance.md", markdown_report(report))
        atomic_json(output / "SELECTED_RECIPE.json", selection)
    else:
        print("dry-run only: report artifacts not published", file=sys.stderr)
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"all-ten bridge report error: {exc}", file=sys.stderr)
        raise SystemExit(2)
