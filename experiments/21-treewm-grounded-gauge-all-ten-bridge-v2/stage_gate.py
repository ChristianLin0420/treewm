#!/usr/bin/env python3
"""Publish the immutable 20/20 method and replicated-outcome Exp21 acceptance."""

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
    RUNS,
    SEEDS,
    SETTING_IDS,
    STAGE_TARGET,
    TASK_IDS,
    actual_evaluation_bank,
    atomic_json,
    expand_runs,
    load_exp20_binding,
    load_manifest,
    read_json,
    require,
    run_directory,
    stable_hash,
    trainer_command,
)
from worker import verify_stage_marker


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


def event_scalars(run_dir: Path) -> dict[str, dict[int, float]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    merged: dict[str, dict[int, tuple[float, float]]] = {}
    paths = sorted(run_dir.glob("events.out.tfevents.*")) or sorted(run_dir.rglob("events.out.tfevents.*"))
    require(bool(paths), f"no TensorBoard scalar artifacts in {run_dir}")
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


def evaluate_metrics(
    manifest: Mapping[str, Any], metrics: Mapping[str, Mapping[int, float]], arm_id: str
) -> dict[str, Any]:
    gate = manifest["stage_acceptance"]
    required = tuple(gate["required_finite_tags"])
    training_tags = set(gate["training_exact_target_tags"])
    samples = {tag: _last(metrics, tag, STAGE_TARGET) for tag in required}
    last = {tag: sample[1] if sample is not None else None for tag, sample in samples.items()}
    last_step = {tag: sample[0] if sample is not None else None for tag, sample in samples.items()}
    expected_step = {
        tag: STAGE_TARGET if tag in training_tags else STAGE_TARGET - STAGE_TARGET % int(gate["validation_diagnostic_every_updates"])
        for tag in required
    }
    finite_coverage = all(finite(value) for value in last.values())
    target_appropriate = all(last_step[tag] == expected_step[tag] for tag in required)
    fixed_counts = _values(metrics, "data/validation_fixed_sample_count", STAGE_TARGET)
    fixed_validation = bool(fixed_counts and all(finite(value) and value > 0 for value in fixed_counts) and len(set(fixed_counts)) == 1)

    gradient_axis = _expected_axis(STAGE_TARGET, int(gate["training_every_updates"]), int(gate["gradient_recent_window_updates"]))
    common_gradient = {tag: _complete_series(metrics, tag, gradient_axis) for tag in COMMON_GRADIENT_TAGS}
    separate_gradient = {tag: _complete_series(metrics, tag, gradient_axis) for tag in SEPARATE_CLIP_TAGS} if arm_id == "GS" else {}
    gradient_map = {**common_gradient, **separate_gradient}
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
        tag: (
            sum(value < float(gate["min_clip_coefficient"]) for value in (gradient_map.get(tag) or [])) / len(gradient_map.get(tag) or [])
            if gradient_map.get(tag) else None
        )
        for tag in clip_tags
    }
    low_clip_fraction = max((float(value) for value in low_clip_by_tag.values()), default=math.inf) if all(finite(value) for value in low_clip_by_tag.values()) else None
    clipping_bounded = bool(
        complete_gradient_axis
        and clip_values
        and all(0.0 < value <= 1.0 for value in clip_values)
        and finite(low_clip_fraction)
        and all(float(value) <= float(gate["max_clip_fraction_below_threshold"]) for value in low_clip_by_tag.values())
    )

    gauge_axis = _expected_axis(STAGE_TARGET, int(gate["training_every_updates"]), int(gate["gauge_recent_window_updates"]))
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
        and finite(last.get("latent_gauge/root/reference"))
        and finite(last.get("latent_gauge/future/reference"))
        and float(last["latent_gauge/root/reference"]) >= float(manifest["scientific_contract"]["latent_gauge_min_reference_scale"])
        and float(last["latent_gauge/future/reference"]) >= float(manifest["scientific_contract"]["latent_gauge_min_reference_scale"])
    )
    recent_min_ratio = min(recent_ratio) if recent_ratio else None
    gauge_absolute = bool(
        recent_ratio is not None and ratio_consistent and reference_valid
        and float(min_ratio) >= float(gate["min_scale_ratio"])
        and float(recent_min_ratio) >= float(gate["min_scale_ratio"])
    )

    val_values = _values(metrics, "val/loss_total", STAGE_TARGET)
    validation_stable = bool(val_values and all(finite(value) and value >= 0 for value in val_values) and val_values[-1] <= min(val_values) * (1 + float(gate["max_validation_regret_fraction"])))
    self_fed = _values(metrics, "val/loss_multistep_self_fed", STAGE_TARGET)
    self_fed_stable = bool(self_fed and all(finite(value) and value >= 0 for value in self_fed) and self_fed[-1] <= min(self_fed) * (1 + float(gate["max_self_fed_multistep_validation_regret_fraction"])))
    fractions = [last.get(f"data/validation_horizon_label_fraction_h{horizon}") for horizon in (4, 8, 16, 32, 64)]
    distribution_valid = bool(all(finite(value) and float(value) >= 0 for value in fractions) and abs(sum(float(value) for value in fractions) - 1) <= float(gate["horizon_label_fraction_sum_tolerance"]))
    prior_entropy = -sum(float(value) * math.log(max(float(value), 1e-12)) for value in fractions) if distribution_valid else None
    horizon_loss = last.get("val/loss_horizon")
    horizon_pass = bool(finite(horizon_loss) and finite(prior_entropy) and float(horizon_loss) < float(gate["horizon_uniform_cross_entropy"]) and float(horizon_loss) < float(prior_entropy))
    q_pass = bool(
        last.get("control/retrieval_uses_task_metric_endpoint") == 1.0
        and finite(last.get("control/q_advantage_over_z")) and finite(last.get("control/q_advantage_over_random_proj"))
        and float(last["control/q_advantage_over_z"]) > float(gate["min_q_advantage"])
        and float(last["control/q_advantage_over_random_proj"]) > float(gate["min_q_advantage"])
    )
    recent_gain = {tag: _values(metrics, tag, STAGE_TARGET, recent=5_000) for tag in GAIN_TAGS}
    gain_mean = {tag: statistics.fmean(values) if values and all(finite(value) for value in values) else None for tag, values in recent_gain.items()}
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
    method_gates = {
        "validation_nonregression": validation_stable,
        "self_fed_multistep_validation_nonregression": self_fed_stable,
        "horizon_ce_below_uniform_and_empirical_prior": horizon_pass,
        "q_advantage": q_pass,
        "gain_rank_pair_eligibility_and_coverage": gain_pass,
        "support_recall_and_precision": support_pass,
    }
    integrity_passed = all(integrity_gates.values())
    method_passed = all(method_gates.values())
    return {
        "integrity_passed": integrity_passed,
        "method_passed": method_passed,
        "gauge_absolute_passed": gauge_absolute,
        "candidate_passed": bool(integrity_passed and method_passed and gauge_absolute),
        "integrity_gates": integrity_gates,
        "method_gates": method_gates,
        "last": last,
        "last_step": last_step,
        "expected_last_step": expected_step,
        "recent_gauge_min_ratio": recent_min_ratio,
        "recent_gauge_samples": len(recent_ratio or []),
        "recent_gradient_samples": len(gradient_axis),
        "clip_fraction_below_threshold": low_clip_fraction,
        "clip_fraction_below_threshold_by_tag": low_clip_by_tag,
        "recent_gain_mean": gain_mean,
        "horizon_empirical_prior_entropy": prior_entropy,
    }


def validate_stage_complete(path: Path, launch: Mapping[str, Any]) -> dict[str, Any]:
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
        value.get("campaign_id") == launch["campaign_id"],
        value.get("index") == run["index"],
        value.get("setting_id") == run["setting_id"],
        value.get("seed") == run["seed"],
        value.get("selected_arm") == run["selected_arm"],
        value.get("stage_target") == STAGE_TARGET,
        value.get("launch_sha256") == launch["launch_sha256"],
        value.get("package_protocol_sha256") == hashes["package_protocol_sha256"],
        value.get("source_sha256") == hashes["source_sha256"],
        value.get("runtime_sha256") == hashes["runtime_sha256"],
        value.get("exp20_binding_sha256") == hashes["exp20_binding_sha256"],
        value.get("selected_recipe_sha256") == hashes["selected_recipe_sha256"],
        all(sha(value.get(key)) for key in ("identity_sha256", "checkpoint_sha256")),
        value.get("evaluation_seed_tables_sha256") == hashes["evaluation_seed_tables_sha256"],
        value.get("final_seed_table_sha256") == hashes["final_seed_table_sha256"],
    )
    require(all(checks), f"stage completion identity differs: {path}")
    return value


def outcome_summary(manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    gate = manifest["stage_acceptance"]
    episodes = float(gate["outcome_episodes_per_run"])
    for row in rows:
        outcome = row["outcome"]
        require(outcome["num_episodes"] == episodes and all(finite(outcome[key]) for key in ("successes", "success_rate", "distance_reduction_frac")), "25k outcome telemetry incomplete")
        successes = float(outcome["successes"])
        require(successes.is_integer() and 0 <= successes <= episodes, "success count invalid")
        require(abs(float(outcome["success_rate"]) - successes / episodes) <= 1e-6, "success rate/count mismatch")
        task_rows = outcome.get("tasks")
        require(isinstance(task_rows, list) and [task.get("task_id") for task in task_rows] == list(TASK_IDS), "25k outcome task coverage differs")
        bank = actual_evaluation_bank(manifest)
        require(
            all(task.get("num_episodes") == 1.0 and task.get("episode_seed") == bank["seeds"][index][0] for index, task in enumerate(task_rows)),
            "25k outcome episode count/seed bank differs",
        )
        require(all(finite(task.get("successes")) and finite(task.get("success_rate")) for task in task_rows), "25k per-task outcome telemetry incomplete")
        require(all(float(task["successes"]).is_integer() and float(task["successes"]) in (0.0, 1.0) and abs(float(task["success_rate"]) - float(task["successes"])) <= 1e-6 for task in task_rows), "25k per-task success telemetry inconsistent")
        require(abs(sum(float(task["successes"]) for task in task_rows) - successes) <= 1e-6, "25k task/aggregate successes differ")
        require(abs(statistics.fmean(float(task["success_rate"]) for task in task_rows) - float(outcome["success_rate"])) <= 1e-6, "25k task/aggregate SR differs")
        require(outcome.get("prospective_monitor_bank_sha256") == bank["sha256"], "25k outcome bank differs")
    per_seed: dict[str, Any] = {}
    for seed in SEEDS:
        seed_rows = [row for row in rows if row["seed"] == seed]
        successes = sum(float(row["outcome"]["successes"]) for row in seed_rows)
        progress = statistics.fmean(float(row["outcome"]["distance_reduction_frac"]) for row in seed_rows)
        require(successes >= int(gate["min_total_successes_per_seed"]), f"seed {seed}: all-zero success")
        require(progress > float(gate["min_mean_distance_reduction_per_seed_exclusive"]), f"seed {seed}: nonpositive mean progress")
        per_seed[str(seed)] = {"successes": successes, "mean_distance_reduction_frac": progress}
    both_success = 0
    both_progress = 0
    per_setting: dict[str, Any] = {}
    for setting in SETTING_IDS:
        setting_rows = [row for row in rows if row["setting_id"] == setting]
        require({row["seed"] for row in setting_rows} == set(SEEDS), f"{setting}: missing replicated seed")
        success = all(float(row["outcome"]["successes"]) > 0 for row in setting_rows)
        progress = all(float(row["outcome"]["distance_reduction_frac"]) > 0 for row in setting_rows)
        both_success += success
        both_progress += progress
        per_setting[setting] = {"both_seed_nonzero_success": success, "both_seed_positive_progress": progress}
    require(both_success >= int(gate["min_settings_with_both_seed_success"]), "no setting has replicated nonzero success")
    require(both_progress >= int(gate["min_settings_with_both_seed_positive_progress"]), "6/10 replicated positive-progress quorum failed")
    return {
        "per_seed": per_seed,
        "per_setting": per_setting,
        "settings_with_both_seed_nonzero_success": both_success,
        "settings_with_both_seed_positive_progress": both_progress,
        "total_successes": sum(float(row["outcome"]["successes"]) for row in rows),
        "macro_success_rate": statistics.fmean(float(row["outcome"]["success_rate"]) for row in rows),
        "macro_distance_reduction_frac": statistics.fmean(float(row["outcome"]["distance_reduction_frac"]) for row in rows),
    }


def build_acceptance(manifest: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    binding = load_exp20_binding(manifest, repo_root / "experiments/21-treewm-grounded-gauge-all-ten-bridge-v2/exp20_binding.json")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    launches = [trainer_command(manifest, run, repo_root=repo_root) for run in expand_runs(manifest)]
    for run, launch in zip(expand_runs(manifest), launches, strict=True):
        run_dir = run_directory(manifest, run)
        try:
            complete = validate_stage_complete(run_dir / "stage-gates" / f"STAGE_COMPLETE_{STAGE_TARGET}.json", launch)
            marker = verify_stage_marker(run_dir, launch)
            require(complete["checkpoint_sha256"] == marker["checkpoint_sha256"], "worker/marker checkpoint differs")
            scalars = event_scalars(run_dir)
            health = evaluate_metrics(manifest, scalars, str(binding["selected_arm"]))
            require(health["candidate_passed"], "method/gauge/integrity gate failed")
            rows.append({
                "index": run.index,
                "run_name": run.run_name,
                "setting_id": run.setting_id,
                "seed": run.seed,
                "selected_arm": binding["selected_arm"],
                "launch_sha256": launch["launch_sha256"],
                "identity_sha256": complete["identity_sha256"],
                "checkpoint_sha256": complete["checkpoint_sha256"],
                "health": health,
                "outcome": {
                    "num_episodes": _at(scalars, "eval/num_episodes", STAGE_TARGET),
                    "successes": _at(scalars, "eval/successes", STAGE_TARGET),
                    "success_rate": _at(scalars, "eval/success_rate", STAGE_TARGET),
                    "distance_reduction_frac": _at(scalars, "eval/distance_reduction_frac", STAGE_TARGET),
                    "prospective_monitor_bank_sha256": actual_evaluation_bank(manifest)["sha256"],
                    "tasks": [
                        {
                            "task_id": task_id,
                            "episode_seed": actual_evaluation_bank(manifest)["seeds"][task_index][0],
                            "num_episodes": _at(scalars, f"eval/task{task_id}/num_episodes", STAGE_TARGET),
                            "successes": _at(scalars, f"eval/task{task_id}/successes", STAGE_TARGET),
                            "success_rate": _at(scalars, f"eval/task{task_id}/success_rate", STAGE_TARGET),
                        }
                        for task_index, task_id in enumerate(TASK_IDS)
                    ],
                },
            })
        except (ContractError, OSError, ValueError) as exc:
            failures.append({"index": run.index, "run_name": run.run_name, "error": str(exc)})
    require(not failures and len(rows) == RUNS, f"25k bridge requires 20/20 exact method rows: {json.dumps(failures, sort_keys=True)}")
    require({(row["setting_id"], row["seed"]) for row in rows} == {(setting, seed) for setting in SETTING_IDS for seed in SEEDS}, "20-run matrix differs")
    for setting in SETTING_IDS:
        hashes = {launch["hashes"]["validation_manifest_sha256"] for launch in launches if launch["run"]["setting_id"] == setting}
        counts = {float(row["health"]["last"]["data/validation_fixed_sample_count"]) for row in rows if row["setting_id"] == setting}
        require(len(hashes) == 1 and len(counts) == 1, f"{setting}: representative validation differs across seeds")
    outcome = outcome_summary(manifest, rows)
    provenance: dict[str, str] = {}
    for key in ("package_protocol_sha256", "source_sha256", "runtime_sha256", "exp20_binding_sha256", "selected_recipe_sha256", "actual_evaluation_bank_sha256"):
        values = {launch["hashes"][key] for launch in launches}
        require(len(values) == 1, f"fleet {key} differs")
        provenance[key] = next(iter(values))
    acceptance: dict[str, Any] = {
        "schema_version": 1,
        "status": "accepted_for_later_1m_formal_campaign_design",
        "campaign_id": manifest["campaign_id"],
        "formal_validation": False,
        "stage_target": STAGE_TARGET,
        "claim": "Bounded all-ten 25k gauge bridge only; never a 1M or formal result.",
        "selected_arm": binding["selected_arm"],
        "selected_recipe": binding["selected_recipe"],
        "selected_recipe_sha256": binding["selected_recipe_sha256"],
        "method_runs_passing": RUNS,
        "required_method_runs": RUNS,
        "outcome": outcome,
        "prospective_monitor_bank": actual_evaluation_bank(manifest),
        "runs": rows,
        "provenance": provenance,
    }
    acceptance["acceptance_sha256"] = stable_hash(acceptance)
    return acceptance


def publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        require(read_json(path) == dict(value), f"immutable acceptance differs: {path}")
    else:
        atomic_json(path, value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--manifest", type=Path, default=CAMPAIGN_DIR / "manifest.json")
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    acceptance = build_acceptance(manifest, args.repo_root.resolve())
    print(json.dumps(acceptance, sort_keys=True, indent=2))
    if args.publish:
        publish(Path(manifest["paths"]["run_root"]) / "reports/acceptance.json", acceptance)
    else:
        print("dry-run only: acceptance not published", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"gauge bridge acceptance error: {exc}", file=sys.stderr)
        raise SystemExit(2)
