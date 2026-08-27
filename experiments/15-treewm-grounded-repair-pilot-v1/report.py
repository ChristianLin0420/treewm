#!/usr/bin/env python3
"""Strict preregistered health and outcome report for repair pilot 15."""

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
    run_directory,
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
    launch = read_optional_json(run_dir / "PILOT_LAUNCH.json")
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
    candidate = str(acceptance["scientific_candidate_arm"])
    control = str(acceptance["matched_control_arm"])
    setting_pass = {
        setting: bool(
            key_exact
            and sum(
                bool(keyed[(setting, candidate, seed)].get("scientific_pass"))
                for seed in SEEDS
            )
            >= int(acceptance["candidate_seeds_per_passing_setting"])
        )
        for setting in SETTING_IDS
    }
    settings_passing = sum(setting_pass.values())

    paired: list[dict[str, Any]] = []
    if key_exact:
        for setting in SETTING_IDS:
            for seed in SEEDS:
                candidate_final = keyed[(setting, candidate, seed)]["metrics"]["final"]
                control_final = keyed[(setting, control, seed)]["metrics"]["final"]
                candidate_success = candidate_final.get("success_rate")
                control_success = control_final.get("success_rate")
                candidate_progress = candidate_final.get("distance_reduction_frac")
                control_progress = control_final.get("distance_reduction_frac")
                paired.append({
                    "setting_id": setting,
                    "seed": seed,
                    "success_delta_candidate_minus_control": (
                        float(candidate_success) - float(control_success)
                        if finite(candidate_success) and finite(control_success)
                        else None
                    ),
                    "distance_reduction_delta_candidate_minus_control": (
                        float(candidate_progress) - float(control_progress)
                        if finite(candidate_progress) and finite(control_progress)
                        else None
                    ),
                })

    def paired_mean(key: str) -> float | None:
        values = [row.get(key) for row in paired]
        return statistics.fmean(float(value) for value in values) if values and all(finite(value) for value in values) else None

    success_delta = paired_mean("success_delta_candidate_minus_control")
    progress_delta = paired_mean("distance_reduction_delta_candidate_minus_control")
    candidate_rows = [
        keyed[(setting, candidate, seed)]
        for setting in SETTING_IDS
        for seed in SEEDS
    ] if key_exact else []
    candidate_successes = [row["metrics"]["final"].get("successes") for row in candidate_rows]
    candidate_progress = [row["metrics"]["final"].get("distance_reduction_frac") for row in candidate_rows]
    total_successes = (
        sum(float(value) for value in candidate_successes)
        if candidate_successes and all(finite(value) for value in candidate_successes)
        else None
    )
    mean_progress = (
        statistics.fmean(float(value) for value in candidate_progress)
        if candidate_progress and all(finite(value) for value in candidate_progress)
        else None
    )
    positive_runs = sum(finite(value) and float(value) > 0.0 for value in candidate_progress)
    aggregate_gates = {
        "integrity_40_of_40": integrity_40_of_40,
        "preregistered_candidate_setting_quorum": settings_passing >= int(acceptance["candidate_settings_required"]),
        "candidate_not_all_zero_success": finite(total_successes) and float(total_successes) >= float(acceptance["min_candidate_total_successes"]),
        "candidate_positive_mean_progress": finite(mean_progress) and float(mean_progress) > float(acceptance["min_candidate_mean_distance_reduction"]),
        "candidate_positive_progress_run_quorum": positive_runs >= int(acceptance["min_candidate_runs_with_positive_progress"]),
        "candidate_success_noninferior_to_control": finite(success_delta) and float(success_delta) >= float(acceptance["min_paired_success_delta_vs_control"]),
        "candidate_distance_reduction_noninferior_to_control": finite(progress_delta) and float(progress_delta) >= float(acceptance["min_paired_distance_reduction_delta_vs_control"]),
        "adaptive_selection_disabled": bool(acceptance["no_adaptive_arm_selection"] and candidate == "C" and control == "A"),
    }
    accepted = all(aggregate_gates.values())
    return {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "status": "accepted_for_fresh_formal_campaign_design" if accepted else "rejected_or_incomplete",
        "accepted": accepted,
        "formal_validation": False,
        "claim": "Bounded repair-pilot result only; never formal validation or a 1M result.",
        "preregistered_candidate_arm": candidate,
        "matched_control_arm": control,
        "sensitivity_arms_are_nonpromotable": ["B", "D"],
        "integrity_runs_passing": integrity_count,
        "candidate_setting_pass": setting_pass,
        "candidate_settings_passing": settings_passing,
        "aggregate_gates": aggregate_gates,
        "aggregate_metrics": {
            "candidate_total_successes": total_successes,
            "candidate_mean_distance_reduction": mean_progress,
            "candidate_runs_with_positive_progress": positive_runs,
            "paired_mean_success_delta_candidate_minus_control": success_delta,
            "paired_mean_distance_reduction_delta_candidate_minus_control": progress_delta,
        },
        "paired_comparisons": paired,
        "missing_or_extra_keys": sorted(expected_keys.symmetric_difference(keyed)),
        "actual_evaluation_bank_sha256": actual_evaluation_bank(manifest)["sha256"],
        "runs": list(records),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# TreeWM grounded repair pilot v1 report",
        "",
        f"Decision: **{report['status']}**.",
        "",
        "> This is a bounded 25k repair pilot, not formal validation or a 1M result.",
        "",
        f"Integrity-complete runs: {report['integrity_runs_passing']}/40.",
        f"Preregistered candidate: arm {report['preregistered_candidate_arm']}; matched control: arm {report['matched_control_arm']}.",
        f"Candidate settings passing both seeds: {report['candidate_settings_passing']}/5 (required: 4).",
        "",
        "| Setting | Candidate both-seed scientific pass |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {setting} | {'yes' if passed else 'no'} |"
        for setting, passed in report["candidate_setting_pass"].items()
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
    verify_protocol_lock(root / "experiments" / "15-treewm-grounded-repair-pilot-v1")
    if root.parent.joinpath("SNAPSHOT.json").exists():
        verify_source_snapshot(root)
    records: list[dict[str, Any]] = []
    for run in expand_runs(manifest):
        try:
            records.append(evaluate_run(manifest, run, repo_root=root))
        except (ContractError, OSError, ValueError, KeyError) as exc:
            records.append(failed_record(run, exc))
    report = aggregate_acceptance(manifest, records)
    print(json.dumps({
        "status": report["status"],
        "accepted": report["accepted"],
        "formal_validation": report["formal_validation"],
        "integrity_runs_passing": report["integrity_runs_passing"],
        "candidate_settings_passing": report["candidate_settings_passing"],
    }, sort_keys=True, indent=2))
    if args.publish:
        output = args.output_dir or Path(manifest["paths"]["run_root"]) / "report"
        atomic_json(output / "acceptance.json", report)
        atomic_text(output / "acceptance.md", markdown_report(report))
    else:
        print("dry-run only: report artifacts not published", file=sys.stderr)
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"repair-pilot report error: {exc}", file=sys.stderr)
        raise SystemExit(2)
