#!/usr/bin/env python3
"""Strict TreeWM 10-setting x 5-task x 4-seed final reporter.

TreeWM trains 40 goal-agnostic dataset models.  Each completion contains all five
task evaluations (50 episodes each), yielding 200 task/seed cells and 10,000 raw
episodes.  This reporter refuses partial, stale, reordered, or metric-only results;
it validates the immutable completion and recomputes success from every raw episode.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from campaign import (
    completion_is_valid,
    expand_runs,
    live_contract,
    load_data_contract,
    load_manifest,
    protocol_sha256,
    run_directory,
    setting_for_run,
)


T_CRITICAL_95_DF3 = 3.182446305284263


class ReportError(RuntimeError):
    pass


def _finite_fraction(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReportError(f"{label}: expected a numeric fraction") from exc
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ReportError(f"{label}: value {result!r} is outside [0, 1]")
    return result


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-7, abs_tol=1e-7)


def load_task_seed_values(
    manifest: Mapping[str, Any],
    run_root: Path,
    *,
    repo_root: Path,
    data_root: Path,
    cache_root: Path,
) -> tuple[dict[tuple[str, int, int], float], dict[str, str]]:
    """Validate all 40 completions and derive all 200 cells from raw episodes."""

    task_ids = [int(value) for value in manifest["axes"]["task_ids"]]
    episodes_per_task = int(manifest["evaluation"]["final_episodes_per_task"])
    expected_order = [
        (task_index, task_id, episode_index)
        for task_index, task_id in enumerate(task_ids)
        for episode_index in range(episodes_per_task)
    ]
    values: dict[tuple[str, int, int], float] = {}
    data_hashes: dict[str, str] = {}
    failures: list[str] = []
    seen_wandb_ids: set[str] = set()

    for run in expand_runs(manifest):
        run_dir = run_directory(run_root, run)
        try:
            setting = setting_for_run(manifest, run)
            data_contract = load_data_contract(
                manifest, setting, data_root=data_root, cache_root=cache_root
            )
            data_sha = str(data_contract["data_manifest_sha256"])
            previous = data_hashes.setdefault(run.setting_id, data_sha)
            if previous != data_sha:
                raise ReportError(f"{run.run_id}: setting data identity changed across seeds")
            if not completion_is_valid(
                run_dir,
                manifest,
                run,
                repo_root=repo_root,
                data_manifest_sha256=data_sha,
                cache_root=cache_root,
            ):
                raise ReportError(f"{run.run_id}: missing or invalid COMPLETED.json")
            completion = json.loads((run_dir / "COMPLETED.json").read_text(encoding="utf-8"))
            if completion["wandb_id"] in seen_wandb_ids:
                raise ReportError(f"{run.run_id}: duplicate W&B ID {completion['wandb_id']}")
            seen_wandb_ids.add(completion["wandb_id"])
            progress_path = run_dir / completion["final_eval_progress"]
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            raw = progress.get("completed_results")
            if not isinstance(raw, list) or len(raw) != len(expected_order):
                raise ReportError(
                    f"{run.run_id}: expected {len(expected_order)} raw episodes, "
                    f"found {len(raw) if isinstance(raw, list) else 'non-list'}"
                )
            observed_order: list[tuple[int, int, int]] = []
            for ordinal, episode in enumerate(raw):
                if not isinstance(episode, dict):
                    raise ReportError(f"{run.run_id}: raw episode {ordinal} is not an object")
                observed_order.append(
                    (
                        int(episode.get("task_index", -1)),
                        int(episode.get("task_id", -1)),
                        int(episode.get("episode_index", -1)),
                    )
                )
                _finite_fraction(episode.get("success"), f"{run.run_id} episode {ordinal} success")
            if observed_order != expected_order:
                raise ReportError(f"{run.run_id}: final episodes are not the deterministic 5x50 order")

            metrics = completion["final_evaluation"]
            raw_all_successes = [_finite_fraction(item["success"], run.run_id) for item in raw]
            global_success = statistics.fmean(raw_all_successes)
            if not _close(
                global_success,
                _finite_fraction(metrics.get("eval/success_rate"), f"{run.run_id} global metric"),
            ):
                raise ReportError(f"{run.run_id}: global success metric disagrees with raw episodes")
            if int(metrics.get("eval/num_episodes", -1)) != len(expected_order):
                raise ReportError(f"{run.run_id}: global episode count is not 250")

            for task_index, task_id in enumerate(task_ids):
                task_raw = raw[
                    task_index * episodes_per_task : (task_index + 1) * episodes_per_task
                ]
                raw_successes = [_finite_fraction(item["success"], run.run_id) for item in task_raw]
                success = statistics.fmean(raw_successes)
                metric_prefix = f"eval/task{task_id}"
                metric_success = _finite_fraction(
                    metrics.get(f"{metric_prefix}/success_rate"),
                    f"{run.run_id} task {task_id} metric",
                )
                if not _close(success, metric_success):
                    raise ReportError(
                        f"{run.run_id}: task {task_id} metric {metric_success} "
                        f"disagrees with raw {success}"
                    )
                if int(metrics.get(f"{metric_prefix}/num_episodes", -1)) != episodes_per_task:
                    raise ReportError(f"{run.run_id}: task {task_id} episode count is not 50")
                metric_successes = metrics.get(f"{metric_prefix}/successes")
                if metric_successes is not None and not _close(float(metric_successes), sum(raw_successes)):
                    raise ReportError(f"{run.run_id}: task {task_id} success count disagrees with raw")
                key = (run.setting_id, task_id, run.seed)
                if key in values:
                    raise ReportError(f"duplicate task/seed cell {key}")
                values[key] = success
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, ReportError) as exc:
            failures.append(str(exc))

    if failures:
        preview = "\n  ".join(failures[:30])
        extra = f"\n  ... and {len(failures) - 30} more" if len(failures) > 30 else ""
        raise ReportError(
            f"refusing partial aggregation; {len(failures)} run(s) failed:\n  {preview}{extra}"
        )
    if len(seen_wandb_ids) != 40 or len(values) != 200:
        raise ReportError(
            f"expected 40 unique W&B runs and 200 task/seed cells, got "
            f"{len(seen_wandb_ids)} and {len(values)}"
        )
    return values, data_hashes


def summarize(samples: Iterable[float]) -> dict[str, Any]:
    values = list(samples)
    if len(values) != 4:
        raise ReportError(f"seed CI requires exactly four paired values, got {len(values)}")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    half_width = T_CRITICAL_95_DF3 * standard_deviation / math.sqrt(4)
    return {
        "n_seeds": 4,
        "seed_values": values,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "ci95_half_width": half_width,
        "mean_percent": 100.0 * mean,
        "ci95_half_width_percent": 100.0 * half_width,
    }


def build_report(
    manifest: Mapping[str, Any],
    values: Mapping[tuple[str, int, int], float],
    *,
    repo_root: Path,
    data_hashes: Mapping[str, str],
) -> dict[str, Any]:
    setting_ids = [setting["id"] for setting in manifest["settings"]]
    task_ids = [int(value) for value in manifest["axes"]["task_ids"]]
    seeds = [int(value) for value in manifest["axes"]["seeds"]]
    expected = {
        (setting_id, task_id, seed)
        for setting_id in setting_ids
        for task_id in task_ids
        for seed in seeds
    }
    if set(values) != expected:
        raise ReportError("task/seed matrix has a gap, duplicate, or out-of-protocol cell")

    per_task: list[dict[str, Any]] = []
    for setting_id in setting_ids:
        for task_id in task_ids:
            per_task.append(
                {
                    "setting_id": setting_id,
                    "task_id": task_id,
                    **summarize(values[(setting_id, task_id, seed)] for seed in seeds),
                }
            )
    per_setting: list[dict[str, Any]] = []
    for setting_id in setting_ids:
        seed_averages = [
            statistics.fmean(values[(setting_id, task_id, seed)] for task_id in task_ids)
            for seed in seeds
        ]
        per_setting.append(
            {"setting_id": setting_id, "task_count": 5, **summarize(seed_averages)}
        )
    overall_seed_averages = [
        statistics.fmean(
            values[(setting_id, task_id, seed)]
            for setting_id in setting_ids
            for task_id in task_ids
        )
        for seed in seeds
    ]
    return {
        "schema_version": 1,
        "status": "complete",
        "campaign_id": manifest["campaign_id"],
        "protocol_sha256": protocol_sha256(manifest),
        "generated_unix_time": time.time(),
        "method": dict(manifest["method"]),
        "scientific_unit": (
            "40 dataset-level TreeWM models; each model evaluated on five built-in tasks"
        ),
        "training_anchor_protocol": (
            "300,000 anchors selected uniformly without replacement from each full valid "
            "transition universe, with a fixed 50,000-item future retrieval pool; cache, "
            "global normalization, and source universe use the complete underlying release"
        ),
        "validated_model_runs": 40,
        "validated_task_seed_cells": 200,
        "validated_raw_episodes": 10_000,
        "code_runtime_contract": live_contract(repo_root),
        "data_manifest_sha256_by_setting": dict(data_hashes),
        "success_unit": "fraction; *_percent is fraction multiplied by 100",
        "ci_method": "two-sided 95% Student-t interval across four paired training seeds (df=3)",
        "per_task": per_task,
        "per_setting": per_setting,
        "aggregate_50_task": {"task_count": 50, **summarize(overall_seed_averages)},
    }


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_report(
    report: Mapping[str, Any],
    values: Mapping[tuple[str, int, int], float],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_json = json.dumps(report, sort_keys=True, indent=2) + "\n"
    atomic_text(output_dir / "report.json", report_json)

    task_seed = io.StringIO(newline="")
    seed_writer = csv.writer(task_seed)
    seed_writer.writerow(("setting_id", "task_id", "seed", "success_rate", "episodes"))
    for (setting_id, task_id, seed), success in sorted(values.items()):
        seed_writer.writerow((setting_id, task_id, seed, success, 50))
    atomic_text(output_dir / "task_seed.csv", task_seed.getvalue())

    per_task = io.StringIO(newline="")
    fieldnames = (
        "setting_id",
        "task_id",
        "n_seeds",
        "mean",
        "standard_deviation",
        "ci95_half_width",
        "mean_percent",
        "ci95_half_width_percent",
    )
    writer = csv.DictWriter(per_task, fieldnames=fieldnames)
    writer.writeheader()
    for row in report["per_task"]:
        writer.writerow({key: row[key] for key in fieldnames})
    atomic_text(output_dir / "per_task.csv", per_task.getvalue())

    lines = [
        "# Formal TreeWM 50-task report",
        "",
        "Validated 40 dataset-level TreeWM models, 200 task/seed cells, and 10,000 raw episodes.",
        "",
        "Training uses 300,000 anchors selected uniformly without replacement from each full "
        "valid transition universe with a fixed 50,000-item future retrieval pool; this is "
        "the disclosed task-aligned TreeWM anchor-cap protocol.",
        "",
        f"Protocol SHA256: `{report['protocol_sha256']}`.",
        "",
        "Intervals are two-sided 95% Student-t intervals across four paired training seeds.",
        "",
        "| setting | five-task success (%) | 95% CI half-width (pp) |",
        "|---|---:|---:|",
    ]
    for row in report["per_setting"]:
        lines.append(
            f"| {row['setting_id']} | {row['mean_percent']:.3f} | "
            f"{row['ci95_half_width_percent']:.3f} |"
        )
    overall = report["aggregate_50_task"]
    lines.extend(
        [
            "",
            f"50-task macro-average: **{overall['mean_percent']:.3f}% "
            f"+/- {overall['ci95_half_width_percent']:.3f} pp**.",
            "",
            "This is a goal-conditioned dataset-model protocol, not 200 task-specific agents.",
        ]
    )
    atomic_text(output_dir / "summary.md", "\n".join(lines) + "\n")

    artifact_hashes = {
        name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        for name in ("report.json", "task_seed.csv", "per_task.csv", "summary.md")
    }
    atomic_text(
        output_dir / "REPORT_COMPLETED.json",
        json.dumps(
            {
                "schema_version": 1,
                "status": "complete",
                "protocol_sha256": report["protocol_sha256"],
                "validated_model_runs": 40,
                "validated_task_seed_cells": 200,
                "validated_raw_episodes": 10_000,
                "artifacts": artifact_hashes,
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=here / "manifest.json")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("TREEWM_RUN_ROOT", repo_root / "outputs" / "treewm-50task-1m-v1")))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("TREEWM_DATA_ROOT", here / "data")))
    parser.add_argument("--cache-root", type=Path, default=Path(os.environ.get("TREEWM_CACHE", here / "cache")))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    for name in ("manifest", "repo_root", "run_root", "data_root", "cache_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    if args.output_dir is None:
        args.output_dir = args.run_root / "report"
    args.output_dir = args.output_dir.expanduser().resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        values, data_hashes = load_task_seed_values(
            manifest,
            args.run_root,
            repo_root=args.repo_root,
            data_root=args.data_root,
            cache_root=args.cache_root,
        )
        report = build_report(
            manifest, values, repo_root=args.repo_root, data_hashes=data_hashes
        )
        write_report(report, values, args.output_dir)
        print(
            f"strict TreeWM report complete: {args.output_dir} "
            f"({report['aggregate_50_task']['mean_percent']:.3f}%)"
        )
        return 0
    except (ReportError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"aggregation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
