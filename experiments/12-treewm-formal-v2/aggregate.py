#!/usr/bin/env python3
"""Strictly aggregate all 40 TreeWM-v2 completions and 10,000 raw episodes."""

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
    load_data_contract,
    load_manifest,
    protocol_sha256,
    run_directory,
)
from treewm.utils.provenance import runtime_fingerprint, trainer_code_fingerprint


class ReportError(RuntimeError):
    pass


def live_contract(repo_root: Path) -> dict[str, str]:
    return {
        "code_sha256": trainer_code_fingerprint(repo_root)["manifest_sha256"],
        "runtime_sha256": runtime_fingerprint()["sha256"],
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


def load_task_seed_values(
    manifest: Mapping[str, Any], run_root: Path, *, repo_root: Path,
    data_root: Path, cache_root: Path,
) -> tuple[dict[tuple[str, int, int], float], dict[str, dict[str, str]]]:
    task_ids = [int(value) for value in manifest["axes"]["task_ids"]]
    episodes_per_task = int(manifest["evaluation"]["final_episodes_per_task"])
    values: dict[tuple[str, int, int], float] = {}
    contract_hashes: dict[str, dict[str, str]] = {}
    settings = {str(setting["id"]): setting for setting in manifest["settings"]}
    for setting_id, setting in settings.items():
        contract = load_data_contract(
            manifest, setting, data_root=data_root, cache_root=cache_root
        )
        contract_hashes[setting_id] = {
            name: str(contract[name])
            for name in ("data_manifest_sha256", "calibration_sha256", "future_recipe_sha256")
            if name in contract
        }

    for run in expand_runs(manifest):
        run_dir = run_directory(run_root, run)
        if not completion_is_valid(
            run_dir, manifest, run, repo_root=repo_root, cache_root=cache_root
        ):
            raise ReportError(f"invalid or incomplete formal run: {run.run_id}")
        try:
            completion = json.loads((run_dir / "COMPLETED.json").read_text(encoding="utf-8"))
            progress_path = run_dir / str(completion["final_eval_progress"])
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
        except (KeyError, FileNotFoundError, OSError, json.JSONDecodeError) as exc:
            raise ReportError(f"{run.run_id}: unreadable final evaluation: {exc}") from exc
        _require(completion.get("objective_version") == "treewm_v2_rms_rank_v1", f"{run.run_id}: wrong objective")
        _require(progress.get("status") == "complete", f"{run.run_id}: final evaluation incomplete")
        _require(progress.get("identity_sha256") == completion.get("identity_sha256"), f"{run.run_id}: evaluation identity differs")
        rows = progress.get("completed_results")
        _require(isinstance(rows, list), f"{run.run_id}: raw episode rows missing")
        expected_order = [
            (task_index, task_id, episode_index)
            for task_index, task_id in enumerate(task_ids)
            for episode_index in range(episodes_per_task)
        ]
        actual_order = [
            (row.get("task_index"), row.get("task_id"), row.get("episode_index"))
            for row in rows
        ]
        _require(actual_order == expected_order, f"{run.run_id}: raw episodes violate deterministic 5x50 order")
        metrics = progress.get("metrics") or {}
        _require(int(metrics.get("eval/num_episodes", -1)) == len(expected_order), f"{run.run_id}: episode count differs")
        for task_id in task_ids:
            task_rows = [row for row in rows if int(row["task_id"]) == task_id]
            successes = [float(row["success"]) for row in task_rows]
            _require(len(successes) == episodes_per_task, f"{run.run_id}: task {task_id} episode gap")
            _require(all(value in (0.0, 1.0) for value in successes), f"{run.run_id}: non-binary success")
            mean = sum(successes) / episodes_per_task
            logged = float(metrics.get(f"eval/task{task_id}/success_rate", float("nan")))
            count = float(metrics.get(f"eval/task{task_id}/successes", float("nan")))
            _require(math.isfinite(logged) and abs(logged - mean) <= 1e-12, f"{run.run_id}: task metric/raw mismatch")
            _require(math.isfinite(count) and abs(count - sum(successes)) <= 1e-12, f"{run.run_id}: task count/raw mismatch")
            key = (run.setting_id, task_id, int(run.seed))
            _require(key not in values, f"duplicate task/seed cell: {key}")
            values[key] = mean
    return values, contract_hashes


def summarize(values: Iterable[float]) -> dict[str, float | int]:
    data = [float(value) for value in values]
    if not data:
        raise ReportError("cannot summarize an empty sample")
    mean = statistics.fmean(data)
    sd = statistics.stdev(data) if len(data) > 1 else 0.0
    # Formal protocol always has four seeds (df=3).
    critical = 3.182446305284263 if len(data) == 4 else 0.0
    half_width = critical * sd / math.sqrt(len(data))
    return {
        "n_seeds": len(data), "mean": mean, "standard_deviation": sd,
        "ci95_half_width": half_width, "mean_percent": 100.0 * mean,
        "ci95_half_width_percent": 100.0 * half_width,
    }


def build_report(
    manifest: Mapping[str, Any], values: Mapping[tuple[str, int, int], float], *,
    repo_root: Path, data_hashes: Mapping[str, Any],
) -> dict[str, Any]:
    setting_ids = [str(setting["id"]) for setting in manifest["settings"]]
    task_ids = [int(value) for value in manifest["axes"]["task_ids"]]
    seeds = [int(value) for value in manifest["axes"]["seeds"]]
    expected = {(setting, task, seed) for setting in setting_ids for task in task_ids for seed in seeds}
    _require(set(values) == expected, "task/seed matrix has a gap, duplicate, or out-of-protocol cell")
    per_task = [
        {"setting_id": setting, "task_id": task,
         **summarize(values[(setting, task, seed)] for seed in seeds)}
        for setting in setting_ids for task in task_ids
    ]
    per_setting = []
    for setting in setting_ids:
        seed_means = [statistics.fmean(values[(setting, task, seed)] for task in task_ids) for seed in seeds]
        per_setting.append({"setting_id": setting, "task_count": 5, **summarize(seed_means)})
    overall = [
        statistics.fmean(values[(setting, task, seed)] for setting in setting_ids for task in task_ids)
        for seed in seeds
    ]
    return {
        "schema_version": 2, "status": "complete",
        "campaign_id": manifest["campaign_id"],
        "objective_version": "treewm_v2_rms_rank_v1",
        "protocol_sha256": protocol_sha256(manifest),
        "generated_unix_time": time.time(),
        "method": dict(manifest["method"]),
        "scientific_unit": "40 dataset-level TreeWM-v2 models; each evaluated on five built-in tasks",
        "validated_model_runs": 40,
        "validated_task_seed_cells": 200,
        "validated_raw_episodes": 10_000,
        "code_runtime_contract": live_contract(repo_root),
        "data_calibration_recipe_contract_by_setting": dict(data_hashes),
        "success_unit": "fraction; *_percent is fraction multiplied by 100",
        "ci_method": "two-sided 95% Student-t interval across four paired training seeds (df=3)",
        "per_task": per_task, "per_setting": per_setting,
        "aggregate_50_task": {"task_count": 50, **summarize(overall)},
    }


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_report(report: Mapping[str, Any], values: Mapping[tuple[str, int, int], float], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_text(output_dir / "report.json", json.dumps(report, sort_keys=True, indent=2) + "\n")
    task_seed = io.StringIO(newline="")
    writer = csv.writer(task_seed)
    writer.writerow(("setting_id", "task_id", "seed", "success_rate", "episodes"))
    for (setting, task, seed), success in sorted(values.items()):
        writer.writerow((setting, task, seed, success, 50))
    atomic_text(output_dir / "task_seed.csv", task_seed.getvalue())
    per_task = io.StringIO(newline="")
    fieldnames = ("setting_id", "task_id", "n_seeds", "mean", "standard_deviation", "ci95_half_width", "mean_percent", "ci95_half_width_percent")
    dict_writer = csv.DictWriter(per_task, fieldnames=fieldnames)
    dict_writer.writeheader()
    for row in report["per_task"]:
        dict_writer.writerow({name: row[name] for name in fieldnames})
    atomic_text(output_dir / "per_task.csv", per_task.getvalue())
    lines = [
        "# Formal TreeWM-v2 50-task report", "",
        "Validated 40 models, 200 task/seed cells, and 10,000 raw episodes.", "",
        f"Objective: `{report['objective_version']}`.",
        f"Protocol SHA256: `{report['protocol_sha256']}`.", "",
        "| setting | five-task success (%) | 95% CI half-width (pp) |", "|---|---:|---:|",
    ]
    for row in report["per_setting"]:
        lines.append(f"| {row['setting_id']} | {row['mean_percent']:.3f} | {row['ci95_half_width_percent']:.3f} |")
    overall = report["aggregate_50_task"]
    lines.extend(["", f"50-task macro-average: **{overall['mean_percent']:.3f}% +/- {overall['ci95_half_width_percent']:.3f} pp**."])
    atomic_text(output_dir / "summary.md", "\n".join(lines) + "\n")
    names = ("report.json", "task_seed.csv", "per_task.csv", "summary.md")
    hashes = {name: hashlib.sha256((output_dir / name).read_bytes()).hexdigest() for name in names}
    atomic_text(
        output_dir / "REPORT_COMPLETED.json",
        json.dumps({"schema_version": 2, "status": "complete", "protocol_sha256": report["protocol_sha256"],
                    "objective_version": report["objective_version"], "validated_model_runs": 40,
                    "validated_task_seed_cells": 200, "validated_raw_episodes": 10_000,
                    "artifacts": hashes}, sort_keys=True, indent=2) + "\n",
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=here / "manifest.json")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("TREEWM_RUN_ROOT", repo_root / "outputs" / "treewm-50task-1m-v2")))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("TREEWM_DATA_ROOT", here / "data")))
    parser.add_argument("--cache-root", type=Path, default=Path(os.environ.get("TREEWM_CACHE", here / "cache")))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    for name in ("manifest", "repo_root", "run_root", "data_root", "cache_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    args.output_dir = (args.output_dir or args.run_root / "report").expanduser().resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
        values, hashes = load_task_seed_values(
            manifest, args.run_root, repo_root=args.repo_root,
            data_root=args.data_root, cache_root=args.cache_root,
        )
        report = build_report(manifest, values, repo_root=args.repo_root, data_hashes=hashes)
        write_report(report, values, args.output_dir)
        print(f"strict TreeWM-v2 report complete: {args.output_dir}")
        return 0
    except (ReportError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"aggregation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
