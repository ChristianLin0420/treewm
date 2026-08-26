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
    return [float(value) for step, value in sorted(metrics.get(tag, {}).items()) if lower <= step <= target and finite(value)]


def _last(metrics: Mapping[str, Mapping[int, float]], tag: str, target: int) -> float | None:
    values = [(step, value) for step, value in metrics.get(tag, {}).items() if step <= target and finite(value)]
    return float(max(values)[1]) if values else None


def _at(metrics: Mapping[str, Mapping[int, float]], tag: str, step: int) -> float | None:
    value = metrics.get(tag, {}).get(step)
    return float(value) if finite(value) else None


def evaluate_metrics(manifest: Mapping[str, Any], metrics: Mapping[str, Mapping[int, float]], target: int) -> dict[str, Any]:
    gate = manifest["stage_acceptance"]
    required = list(gate["required_finite_tags"])
    last = {tag: _last(metrics, tag, target) for tag in required}
    finite_coverage = all(finite(value) for value in last.values())
    fixed_counts = _values(metrics, "data/validation_fixed_sample_count", target)
    fixed_validation = bool(fixed_counts and len({float(value) for value in fixed_counts}) == 1 and fixed_counts[0] > 0)
    gradient_pass = bool(
        finite(last.get("train/grad_norm_world"))
        and finite(last.get("train/grad_norm_gain"))
        and float(last["train/grad_norm_world"]) > float(gate["min_gradient_norm"])
        and float(last["train/grad_norm_gain"]) > float(gate["min_gradient_norm"])
    )
    val_values = _values(metrics, "val/loss_total", target)
    validation_stable = bool(
        val_values
        and val_values[-1] <= min(val_values) * (1.0 + float(gate["max_validation_regret_fraction"]))
    )
    self_fed_multistep_values = _values(metrics, "val/loss_multistep_self_fed", target)
    self_fed_multistep_stable = bool(
        self_fed_multistep_values
        and self_fed_multistep_values[-1]
        <= min(self_fed_multistep_values)
        * (1.0 + float(gate["max_self_fed_multistep_validation_regret_fraction"]))
    )
    enforce = target >= int(gate["enforce_scientific_thresholds_from_stage"])
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
        not enforce
        or (
            finite(horizon_loss)
            and finite(horizon_prior_entropy)
            and float(horizon_loss) < float(gate["horizon_uniform_cross_entropy"])
            and float(horizon_loss) < float(horizon_prior_entropy)
        )
    )
    window = min(5_000, target)
    recent_tags = (
        "expansion/gain_rank_correlation",
        "expansion/gain_pairwise_accuracy",
        "expansion/gain_eligible_decision_fraction",
        "expansion/gain_ordered_pair_count",
        "expansion/gain_pair_coverage_fraction",
    )
    recent = {tag: _values(metrics, tag, target, recent=window) for tag in recent_tags}
    recent_mean = {tag: statistics.fmean(values) if values else None for tag, values in recent.items()}
    q_pass = bool(
        not enforce
        or (
            float(last.get("control/retrieval_uses_task_metric_endpoint", 0.0)) == 1.0
            and float(last.get("control/q_advantage_over_z", -math.inf)) > float(gate["min_q_advantage"])
            and float(last.get("control/q_advantage_over_random_proj", -math.inf)) > float(gate["min_q_advantage"])
        )
    )
    gain_pass = bool(
        not enforce
        or (
            recent_mean["expansion/gain_rank_correlation"] is not None
            and recent_mean["expansion/gain_pairwise_accuracy"] is not None
            and recent_mean["expansion/gain_eligible_decision_fraction"] is not None
            and recent_mean["expansion/gain_ordered_pair_count"] is not None
            and recent_mean["expansion/gain_pair_coverage_fraction"] is not None
            and float(recent_mean["expansion/gain_rank_correlation"]) >= float(gate["min_gain_rank_correlation"])
            and float(recent_mean["expansion/gain_pairwise_accuracy"]) >= float(gate["min_gain_pairwise_accuracy"])
            and float(recent_mean["expansion/gain_eligible_decision_fraction"]) >= float(gate["min_gain_eligible_decision_fraction"])
            and float(recent_mean["expansion/gain_ordered_pair_count"]) >= float(gate["min_gain_ordered_pair_count"])
            and float(recent_mean["expansion/gain_pair_coverage_fraction"]) >= float(gate["min_gain_pair_coverage_fraction"])
        )
    )
    support_pass = bool(
        not enforce
        or (
            float(last.get("tree/support_recall", -math.inf)) >= float(gate["min_support_recall"])
            and float(last.get("tree/support_precision", -math.inf)) >= float(gate["min_support_precision"])
        )
    )
    integrity_gates = {
        "required_finite_telemetry": finite_coverage,
        "fixed_validation_sample": fixed_validation,
        "nonzero_world_and_gain_gradients": gradient_pass,
    }
    scientific_gates = {
        "validation_nonregression": validation_stable,
        "self_fed_multistep_validation_nonregression": self_fed_multistep_stable,
        "horizon_ce_below_uniform_and_empirical_prior": horizon_pass,
        "q_advantage": q_pass,
        "gain_rank_pair_and_eligibility_coverage": gain_pass,
        "support_recall_and_precision": support_pass,
    }
    return {
        "integrity_passed": all(integrity_gates.values()),
        "scientific_passed": all(scientific_gates.values()),
        "passed": all(integrity_gates.values()) and all(scientific_gates.values()),
        "integrity_gates": integrity_gates,
        "scientific_gates": scientific_gates,
        "last": last,
        "recent_window_updates": window,
        "recent_gain_mean": recent_mean,
        "horizon_empirical_prior_entropy": horizon_prior_entropy,
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
        SHA(value.get("identity_sha256")),
        SHA(value.get("checkpoint_sha256")),
        SHA(value.get("evaluation_seed_tables_sha256")),
        SHA(value.get("final_seed_table_sha256")),
        SHA(value.get("monitor_seed_table_sha256")),
    )
    if not all(checks):
        raise ContractError(f"stage completion identity differs: {path}")
    return value


def validate_scientific_setting_coverage(
    manifest: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], target: int
) -> dict[str, int] | None:
    if target < int(manifest["stage_acceptance"]["enforce_scientific_thresholds_from_stage"]):
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
    if target < int(manifest["stage_acceptance"]["outcome_sanity_stage"]):
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
    return {
        "episodes": expected_episodes * TRAINING_RUNS,
        "successes": successes,
        "macro_success_rate": statistics.fmean(float(row["monitor"]["success_rate"]) for row in rows),
        "macro_distance_reduction_frac": statistics.fmean(float(row["monitor"]["distance_reduction_frac"]) for row in rows),
        "runs_with_positive_progress": sum(float(row["monitor"]["distance_reduction_frac"]) > 0 for row in rows),
    }


def SHA(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def build_gate(manifest: Mapping[str, Any], target: int, repo_root: Path) -> dict[str, Any]:
    if target not in STAGE_TARGETS:
        raise ContractError("invalid stage target")
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for run in expand_runs(manifest):
        launch = trainer_command(manifest, run, repo_root=repo_root)
        run_dir = run_directory(manifest, run)
        complete_path = run_dir / "stage-gates" / f"STAGE_COMPLETE_{target}.json"
        try:
            complete = validate_stage_complete(complete_path, launch, target)
            marker = verify_stage_marker(run_dir, target, launch)
            if complete["checkpoint_sha256"] != marker["checkpoint_sha256"]:
                raise ContractError("worker completion/checkpoint hash differs")
            scalars = event_scalars(run_dir)
            health = evaluate_metrics(manifest, scalars, target)
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
            trainer_command(manifest, run, repo_root=repo_root)["hashes"][key]
            for run in expand_runs(manifest)
        }
        for key in (
            "package_protocol_sha256",
            "source_sha256",
            "evaluation_source_sha256",
            "runtime_sha256",
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
