#!/usr/bin/env python3
"""Strict 200-run RQL final-evaluation aggregator."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from campaign import completion_is_valid, expand_runs, load_manifest, manifest_sha256, run_directory


T_CRITICAL_95_DF3 = 3.182446305284263
SUCCESS_KEYS = ("success", "success_rate", "episode.success")


class ReportError(RuntimeError):
    pass


def _success_value(metrics: Mapping[str, Any], run_id: str) -> float:
    for key in SUCCESS_KEYS:
        if key in metrics:
            try:
                value = float(metrics[key])
            except (TypeError, ValueError) as exc:
                raise ReportError(f"{run_id}: final metric {key!r} is not numeric") from exc
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ReportError(f"{run_id}: final success {value!r} is outside [0, 1]")
            return value
    raise ReportError(f"{run_id}: none of the expected final success keys are present: {SUCCESS_KEYS}")


def load_final_successes(manifest: Mapping[str, Any], run_root: Path) -> dict[tuple[str, int, int], float]:
    values: dict[tuple[str, int, int], float] = {}
    failures: list[str] = []
    for run in expand_runs(manifest):
        directory = run_directory(run_root, run)
        if not completion_is_valid(directory, manifest, run):
            failures.append(f"{run.run_id}: missing or invalid COMPLETED.json")
            continue
        try:
            completion = json.loads((directory / "COMPLETED.json").read_text(encoding="utf-8"))
            values[(run.setting_id, run.task_id, run.seed)] = _success_value(
                completion["final_evaluation"], run.run_id
            )
        except (OSError, json.JSONDecodeError, KeyError, ReportError) as exc:
            failures.append(str(exc))
    if failures:
        preview = "\n  ".join(failures[:30])
        suffix = f"\n  ... and {len(failures) - 30} more" if len(failures) > 30 else ""
        raise ReportError(f"refusing partial aggregation; {len(failures)} runs failed validation:\n  {preview}{suffix}")
    if len(values) != 200:
        raise ReportError(f"expected 200 final successes, found {len(values)}")
    return values


def summarize(samples: Iterable[float]) -> dict[str, Any]:
    values = list(samples)
    if len(values) != 4:
        raise ReportError(f"95% seed CI requires exactly four values, got {len(values)}")
    mean = statistics.fmean(values)
    standard_deviation = statistics.stdev(values)
    half_width = T_CRITICAL_95_DF3 * standard_deviation / math.sqrt(len(values))
    return {
        "n_seeds": len(values),
        "seed_values": values,
        "mean": mean,
        "standard_deviation": standard_deviation,
        "ci95_half_width": half_width,
        "mean_percent": 100.0 * mean,
        "ci95_half_width_percent": 100.0 * half_width,
    }


def build_report(manifest: Mapping[str, Any], values: Mapping[tuple[str, int, int], float]) -> dict[str, Any]:
    task_ids = manifest["axes"]["task_ids"]
    seeds = manifest["axes"]["seeds"]
    setting_ids = [setting["id"] for setting in manifest["settings"]]

    per_task: list[dict[str, Any]] = []
    for setting_id in setting_ids:
        for task_id in task_ids:
            summary = summarize(values[(setting_id, task_id, seed)] for seed in seeds)
            per_task.append({"setting_id": setting_id, "task_id": task_id, **summary})

    per_setting: list[dict[str, Any]] = []
    for setting_id in setting_ids:
        seed_averages = [
            statistics.fmean(values[(setting_id, task_id, seed)] for task_id in task_ids)
            for seed in seeds
        ]
        per_setting.append({"setting_id": setting_id, "tasks": 5, **summarize(seed_averages)})

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
        "campaign_id": manifest["campaign_id"],
        "protocol_sha256": manifest_sha256(manifest),
        "generated_unix_time": time.time(),
        "success_unit": "fraction; *_percent fields are fraction multiplied by 100",
        "ci_method": "two-sided 95% Student-t CI across four seed-level values (df=3)",
        "validated_runs": len(values),
        "per_task": per_task,
        "per_setting": per_setting,
        "aggregate_50_task": {"tasks": 50, **summarize(overall_seed_averages)},
    }


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_report(report: Mapping[str, Any], output_dir: Path) -> None:
    atomic_text(output_dir / "report.json", json.dumps(report, sort_keys=True, indent=2) + "\n")

    import io

    stream = io.StringIO(newline="")
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
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    for row in report["per_task"]:
        writer.writerow({key: row[key] for key in fieldnames})
    atomic_text(output_dir / "per_task.csv", stream.getvalue())

    lines = [
        "# RQL 50-task final report",
        "",
        f"Validated runs: {report['validated_runs']}/200.",
        f"Protocol SHA256: `{report['protocol_sha256']}`.",
        "",
        "All intervals are two-sided 95% Student-t intervals across four seed-level values.",
        "",
        "| setting | five-task success (%) | 95% CI half-width |",
        "|---|---:|---:|",
    ]
    for row in report["per_setting"]:
        lines.append(
            f"| {row['setting_id']} | {row['mean_percent']:.3f} | {row['ci95_half_width_percent']:.3f} |"
        )
    overall = report["aggregate_50_task"]
    lines.extend(
        [
            "",
            f"50-task aggregate success: **{overall['mean_percent']:.3f}% ± "
            f"{overall['ci95_half_width_percent']:.3f}%** (95% CI half-width).",
            "",
        ]
    )
    atomic_text(output_dir / "summary.md", "\n".join(lines))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=here / "manifest.json")
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("RQL_RUN_ROOT", here / "output")))
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.output_dir is None:
        args.output_dir = args.run_root / "report"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        values = load_final_successes(manifest, args.run_root.resolve())
        report = build_report(manifest, values)
        write_report(report, args.output_dir.resolve())
        overall = report["aggregate_50_task"]
        print(
            f"report complete: {overall['mean_percent']:.3f}% +/- "
            f"{overall['ci95_half_width_percent']:.3f}% (95% CI half-width)"
        )
        return 0
    except (OSError, ValueError, ReportError) as exc:
        print(f"aggregation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
