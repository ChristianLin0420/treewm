#!/usr/bin/env python3
"""Strict, outcome-aware acceptance report for corrected pilot 13.

This report can accept a recipe for another bounded experiment.  It always labels the
result as non-formal and has no code path that launches or promotes 1M training.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

from campaign import (
    CAMPAIGN_DIR,
    ContractError,
    REPOSITORY_ROOT,
    SEEDS,
    atomic_json,
    expand_runs,
    load_manifest,
    run_directory,
    stable_hash,
    trainer_command,
    verify_protocol_lock,
)


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _event_scalars(run_dir: Path) -> dict[str, dict[int, float]]:
    """Merge all requeue-created TensorBoard files, keeping the latest wall-time."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    merged: dict[str, dict[int, tuple[float, float]]] = {}
    for path in sorted(run_dir.glob("events.out.tfevents.*")):
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        try:
            accumulator.Reload()
        except Exception as exc:
            raise ContractError(f"unreadable TensorBoard event file {path}: {exc}") from exc
        for tag in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(tag):
                previous = merged.setdefault(tag, {}).get(int(event.step))
                candidate = (float(event.wall_time), float(event.value))
                if previous is None or candidate[0] >= previous[0]:
                    merged[tag][int(event.step)] = candidate
    return {
        tag: {step: value for step, (_wall, value) in steps.items()}
        for tag, steps in merged.items()
    }


def _at(metrics: Mapping[str, Mapping[int, float]], tag: str, step: int) -> float | None:
    value = metrics.get(tag, {}).get(step)
    return float(value) if finite(value) else None


def _last(metrics: Mapping[str, Mapping[int, float]], tag: str, at_most: int = 12_000) -> float | None:
    candidates = [(step, value) for step, value in metrics.get(tag, {}).items() if step <= at_most and finite(value)]
    return float(max(candidates)[1]) if candidates else None


def _tail(metrics: Mapping[str, Mapping[int, float]], tag: str, after: int = 10_000) -> list[float]:
    return [float(value) for step, value in sorted(metrics.get(tag, {}).items()) if step >= after and finite(value)]


def _completion(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "COMPLETED.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"missing/invalid completion {path}: {exc}") from exc
    return value


def evaluate_run(
    manifest: Mapping[str, Any],
    run,
    metrics,
    completion,
    *,
    repo_root: Path = REPOSITORY_ROOT,
) -> dict[str, Any]:
    gate = manifest["acceptance"]
    launch_path = run_directory(manifest, run) / "PILOT_LAUNCH.json"
    try:
        launch = json.loads(launch_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"missing/invalid launch contract {launch_path}: {exc}") from exc
    expected_launch = trainer_command(manifest, run, repo_root=repo_root)
    launch_exact = launch == expected_launch
    identity = completion.get("run_identity") or {}
    final_eval = completion.get("final_evaluation") or {}
    hashes = expected_launch["hashes"]
    identity_hash_valid = bool(
        completion.get("identity_sha256")
        and stable_hash(identity) == completion.get("identity_sha256")
    )
    complete = bool(
        launch_exact
        and identity_hash_valid
        and completion.get("status") == "complete"
        and int(completion.get("completed_updates", -1)) == 12_000
        and int(completion.get("final_eval_step", -1)) == 12_000
        and completion.get("objective_version") == "treewm_v2_grounded_pilot_v1"
        and identity.get("run_name") == run.run_name
        and completion.get("protocol_sha256") == hashes["run_protocol_sha256"]
        and completion.get("code_sha256") == hashes["source_sha256"]
        and completion.get("runtime_sha256") == hashes["runtime_sha256"]
        and completion.get("recipe_code_sha256")
        == hashes["compatible_recipe_code_sha256"]
        and completion.get("recipe_runtime_sha256")
        == hashes["compatible_recipe_runtime_sha256"]
        and identity.get("protocol_sha256") == hashes["run_protocol_sha256"]
        and identity.get("code_sha256") == hashes["source_sha256"]
        and identity.get("runtime_sha256") == hashes["runtime_sha256"]
        and identity.get("recipe_code_sha256")
        == hashes["compatible_recipe_code_sha256"]
        and identity.get("recipe_runtime_sha256")
        == hashes["compatible_recipe_runtime_sha256"]
        and identity.get("calibration_sha256") == hashes["calibration_sha256"]
        and identity.get("future_recipe_sha256") == hashes["future_recipe_sha256"]
        and identity.get("campaign_source_sha256") == hashes["source_sha256"]
        and identity.get("campaign_protocol_sha256")
        == hashes["package_protocol_sha256"]
        and identity.get("campaign_config_sha256") == hashes["config_sha256"]
        and identity.get("campaign_input_contract_sha256")
        == hashes["compatible_input_contract_sha256"]
        and identity.get("campaign_factorial_arm") == run.arm_id
        and int(final_eval.get("eval/num_episodes", -1)) == 5
        and all(
            int(final_eval.get(f"eval/task{task}/num_episodes", -1)) == 1
            for task in range(1, 6)
        )
    )

    checkpoint_steps = list(gate["checkpoint_steps"])
    val_loss = {step: _at(metrics, "val/loss_total", step) for step in checkpoint_steps}
    val_present = all(finite(value) for value in val_loss.values())
    stable_values = [float(val_loss[step]) for step in checkpoint_steps if step >= 6000 and finite(val_loss[step])]
    val_stable = bool(
        val_present
        and stable_values
        and stable_values[-1]
        <= min(stable_values) * (1.0 + float(gate["max_final_validation_regret_fraction"]))
    )
    fixed_counts = {
        _at(metrics, "data/validation_fixed_sample_count", step) for step in [0, *checkpoint_steps]
    }
    fixed_validation = None not in fixed_counts and len(fixed_counts) == 1 and next(iter(fixed_counts)) > 0

    horizon_ce = _at(metrics, "val/loss_horizon", 12_000)
    probabilities = [
        _at(metrics, f"data/validation_horizon_label_fraction_h{horizon}", 12_000)
        for horizon in (4, 8, 16, 32, 64)
    ]
    label_distribution_valid = all(finite(value) and float(value) >= 0 for value in probabilities)
    probability_sum = sum(float(value) for value in probabilities if finite(value))
    label_distribution_valid = label_distribution_valid and abs(probability_sum - 1.0) <= 0.02
    majority_prior_ce = (
        -sum(float(p) * math.log(max(float(p), 1e-12)) for p in probabilities)
        if label_distribution_valid
        else None
    )
    horizon_pass = bool(
        finite(horizon_ce)
        and finite(majority_prior_ce)
        and float(horizon_ce) < float(gate["horizon_uniform_ce"])
        and float(horizon_ce) < float(majority_prior_ce)
    )

    q_z = _at(metrics, "control/q_advantage_over_z", 12_000)
    q_random = _at(metrics, "control/q_advantage_over_random_proj", 12_000)
    q_measurement = _at(metrics, "control/retrieval_uses_task_metric_endpoint", 12_000)
    q_pass = bool(
        q_measurement == 1.0
        and finite(q_z)
        and finite(q_random)
        and float(q_z) > float(gate["min_q_advantage"])
        and float(q_random) > float(gate["min_q_advantage"])
    )

    gain_rho = _last(metrics, "expansion/gain_rank_correlation")
    gain_pairwise = _last(metrics, "expansion/gain_pairwise_accuracy")
    gain_pass = bool(
        finite(gain_rho)
        and finite(gain_pairwise)
        and float(gain_rho) >= float(gate["min_gain_rank_correlation"])
        and float(gain_pairwise) >= float(gate["min_gain_pairwise_accuracy"])
    )
    support_recall = _last(metrics, "tree/support_recall")
    support_precision = _last(metrics, "tree/support_precision")
    support_pass = bool(
        finite(support_recall)
        and finite(support_precision)
        and float(support_recall) >= float(gate["min_support_recall"])
        and float(support_precision) >= float(gate["min_support_precision"])
    )

    world_norm = _tail(metrics, "train/grad_norm_world")
    gain_norm = _tail(metrics, "train/grad_norm_gain")
    world_clip = _tail(metrics, "train/grad_clip_coefficient_world")
    gain_clip = _tail(metrics, "train/grad_clip_coefficient_gain")
    clip_values = [*world_clip, *gain_clip]
    low_clip_fraction = (
        sum(value < float(gate["min_clip_coefficient"]) for value in clip_values) / len(clip_values)
        if clip_values
        else None
    )
    gradient_pass = bool(
        world_norm
        and gain_norm
        and clip_values
        and all(finite(value) and value > float(gate["min_gradient_norm"]) for value in [*world_norm, *gain_norm])
        and finite(low_clip_fraction)
        and float(low_clip_fraction) <= float(gate["max_clip_fraction_below_threshold"])
    )
    progress_6k = _at(metrics, "eval/distance_reduction_frac", 6000)
    success_6k = _at(metrics, "eval/success_rate", 6000)
    progress_12k = final_eval.get("eval/distance_reduction_frac")
    success_12k = final_eval.get("eval/success_rate")
    rollout_present = all(finite(value) for value in (progress_6k, success_6k, progress_12k, success_12k))

    gates = {
        "complete_identity": complete,
        "fixed_validation": fixed_validation,
        "validation_stability": val_stable,
        "horizon_ce_below_uniform_and_empirical_prior": horizon_pass,
        "q_advantage_over_z_and_random": q_pass,
        "gain_rank_and_pairwise": gain_pass,
        "support_recall_and_precision": support_pass,
        "gradients_and_clipping": gradient_pass,
        "rollouts_at_6k_and_12k": rollout_present,
    }
    return {
        "index": run.index,
        "setting_id": run.setting_id,
        "arm_id": run.arm_id,
        "seed": run.seed,
        "run_name": run.run_name,
        "gates": gates,
        "launch_exact": launch_exact,
        "identity_hash_valid": identity_hash_valid,
        "internal_pass": all(gates.values()),
        "metrics": {
            "validation_loss": val_loss,
            "horizon_ce": horizon_ce,
            "empirical_prior_ce": majority_prior_ce,
            "q_advantage_over_z": q_z,
            "q_advantage_over_random": q_random,
            "gain_rank_correlation": gain_rho,
            "gain_pairwise_accuracy": gain_pairwise,
            "support_recall": support_recall,
            "support_precision": support_precision,
            "clip_fraction_below_threshold": low_clip_fraction,
            "progress_6k": progress_6k,
            "progress_12k": progress_12k,
            "success_6k": success_6k,
            "success_12k": success_12k,
        },
    }


def aggregate_acceptance(manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gate = manifest["acceptance"]
    keyed = {(record["setting_id"], record["arm_id"], record["seed"]): record for record in records}
    candidate = str(gate["candidate_arm"])
    controls = "r1-g0"
    expected_keys = {
        (setting["id"], arm["id"], seed)
        for setting in manifest["settings"]
        for arm in manifest["factorial"]["arms"]
        for seed in SEEDS
    }
    if set(keyed) != expected_keys:
        return {
            "status": "rejected_or_incomplete",
            "accepted": False,
            "formal_validation": False,
            "claim": "Bounded corrected-pilot result only; never formal validation.",
            "candidate_arm": candidate,
            "candidate_setting_internal_pass": {
                setting["id"]: False for setting in manifest["settings"]
            },
            "candidate_settings_passing": 0,
            "comparison_gates": {
                "candidate_progress_vs_regularized_control": False,
                "candidate_progress_6k_to_12k": False,
                "candidate_success_noninferior": False,
            },
            "comparison_means": {},
            "paired_comparisons": [],
            "missing_or_extra_run_keys": sorted(expected_keys.symmetric_difference(keyed)),
            "runs": list(records),
        }
    setting_internal: dict[str, bool] = {}
    comparisons: list[dict[str, Any]] = []
    for setting in manifest["settings"]:
        setting_id = setting["id"]
        setting_internal[setting_id] = all(
            bool(keyed[(setting_id, candidate, seed)]["internal_pass"]) for seed in SEEDS
        )
        for seed in SEEDS:
            treatment = keyed[(setting_id, candidate, seed)]["metrics"]
            control = keyed[(setting_id, controls, seed)]["metrics"]
            comparisons.append(
                {
                    "setting_id": setting_id,
                    "seed": seed,
                    "progress_delta_vs_regularized_control": (
                        float(treatment["progress_12k"]) - float(control["progress_12k"])
                        if finite(treatment["progress_12k"]) and finite(control["progress_12k"])
                        else None
                    ),
                    "candidate_progress_delta_6k_to_12k": (
                        float(treatment["progress_12k"]) - float(treatment["progress_6k"])
                        if finite(treatment["progress_12k"]) and finite(treatment["progress_6k"])
                        else None
                    ),
                    "success_delta_vs_regularized_control": (
                        float(treatment["success_12k"]) - float(control["success_12k"])
                        if finite(treatment["success_12k"]) and finite(control["success_12k"])
                        else None
                    ),
                }
            )
    def paired_mean(key: str) -> float | None:
        values = [row[key] for row in comparisons]
        return statistics.fmean(float(value) for value in values) if all(finite(value) for value in values) else None

    progress_vs_control = paired_mean("progress_delta_vs_regularized_control")
    progress_over_time = paired_mean("candidate_progress_delta_6k_to_12k")
    success_vs_control = paired_mean("success_delta_vs_regularized_control")
    comparison_gates = {
        "candidate_progress_vs_regularized_control": finite(progress_vs_control)
        and float(progress_vs_control) > float(gate["min_rollout_progress_delta_vs_matched_control"]),
        "candidate_progress_6k_to_12k": finite(progress_over_time)
        and float(progress_over_time) >= float(gate["min_rollout_progress_delta_6k_to_12k"]),
        "candidate_success_noninferior": finite(success_vs_control) and float(success_vs_control) >= 0.0,
    }
    settings_passing = sum(setting_internal.values())
    accepted = bool(
        len(records) == 32
        and all(record["gates"]["complete_identity"] for record in records)
        and settings_passing >= int(gate["min_candidate_settings_passing_internal_gates"])
        and all(comparison_gates.values())
    )
    return {
        "status": "accepted_for_next_bounded_pilot" if accepted else "rejected_or_incomplete",
        "accepted": accepted,
        "formal_validation": False,
        "claim": "Bounded corrected-pilot result only; never formal validation.",
        "candidate_arm": candidate,
        "candidate_setting_internal_pass": setting_internal,
        "candidate_settings_passing": settings_passing,
        "comparison_gates": comparison_gates,
        "comparison_means": {
            "progress_delta_vs_regularized_control": progress_vs_control,
            "candidate_progress_delta_6k_to_12k": progress_over_time,
            "success_delta_vs_regularized_control": success_vs_control,
        },
        "paired_comparisons": comparisons,
        "runs": list(records),
    }


def markdown_report(report: Mapping[str, Any]) -> str:
    lines = [
        "# TreeWM corrected pilot report",
        "",
        f"Decision: **{report['status']}**.",
        "",
        "> This is a bounded diagnostic pilot. It is not formal validation and cannot by itself authorize a 1M campaign.",
        "",
        f"Candidate settings passing all internal gates: {report['candidate_settings_passing']}/4.",
        "",
        "| Setting | Internal pass |",
        "|---|---:|",
    ]
    lines.extend(
        f"| {setting} | {'yes' if passed else 'no'} |"
        for setting, passed in report["candidate_setting_internal_pass"].items()
    )
    lines.extend(["", "Comparison gates:", ""])
    lines.extend(
        f"- {name}: {'pass' if passed else 'fail'}"
        for name, passed in report["comparison_gates"].items()
    )
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CAMPAIGN_DIR / "manifest.json")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    verify_protocol_lock(CAMPAIGN_DIR)
    records = []
    for run in expand_runs(manifest):
        directory = run_directory(manifest, run)
        records.append(
            evaluate_run(
                manifest,
                run,
                _event_scalars(directory),
                _completion(directory),
                repo_root=args.repo_root.resolve(),
            )
        )
    report = aggregate_acceptance(manifest, records)
    output = args.output_dir or Path(manifest["paths"]["run_root"]) / "report"
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / "acceptance.json", report)
    (output / "acceptance.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "accepted", "formal_validation")}, indent=2))
    return 0 if report["accepted"] else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"corrected-pilot report error: {exc}", file=sys.stderr)
        raise SystemExit(2)
