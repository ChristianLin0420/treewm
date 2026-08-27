#!/usr/bin/env python3
"""Strictly aggregate all 200 paired learned/BFS final-evaluation cells."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence

from campaign import (
    CAMPAIGN_DIR,
    ContractError,
    FINAL_EVAL_TASKS,
    REPOSITORY_ROOT,
    SEEDS,
    SETTING_IDS,
    TASK_IDS,
    atomic_json,
    eval_at,
    expand_runs,
    load_manifest,
    load_seed_table,
    read_json,
    stable_hash,
    trainer_command,
)
from final_eval import _progress_identity, single_task_seed_table


def finite(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def seed_t_summary(values: Sequence[float], t_critical: float = 3.182446) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    if len(numeric) != 4 or not all(finite(value) for value in numeric):
        raise ContractError("seed-level inference requires exactly four finite replicates")
    mean = statistics.fmean(numeric)
    sample_sd = statistics.stdev(numeric)
    half_width = float(t_critical) * sample_sd / math.sqrt(len(numeric))
    return {
        "n": len(numeric),
        "mean": mean,
        "sample_sd": sample_sd,
        "t_critical": float(t_critical),
        "degrees_of_freedom": len(numeric) - 1,
        "ci95_lower": mean - half_width,
        "ci95_upper": mean + half_width,
    }


def validate_episode_success_accounting(
    episode_successes: Sequence[object], reported_successes: object, label: str
) -> list[bool]:
    if any(not isinstance(value, bool) for value in episode_successes):
        raise ContractError(f"{label}: episode success is not boolean")
    if not finite(reported_successes):
        raise ContractError(f"{label}: reported successes are non-finite")
    values = list(episode_successes)
    if float(reported_successes) != float(sum(values)):
        raise ContractError(f"{label}: episode successes disagree with metrics")
    return values


def validate_exact_episode_count(value: object, expected: int, label: str) -> None:
    if not finite(value) or float(value) != float(expected):
        raise ContractError(f"{label}: metric episode count differs")


def load_final_gate(manifest: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(manifest["paths"]["run_root"]) / "state" / "stage-gates" / "STAGE_GATE_1000000.json"
    gate = read_json(path)
    claimed = gate.get("gate_sha256")
    body = dict(gate)
    body.pop("gate_sha256", None)
    if claimed != stable_hash(body):
        raise ContractError("final training-stage gate hash differs")
    if gate.get("status") != "accepted" or int(gate.get("stage_target", -1)) != 1_000_000 or len(gate.get("runs") or []) != 40:
        raise ContractError("final training stage lacks exact accepted 40-run gate")
    if not isinstance(gate.get("prerequisite_binding_sha256"), str) or not isinstance(
        gate.get("selected_recipe_sha256"), str
    ):
        raise ContractError("final training stage lacks prerequisite/recipe binding")
    return gate


def _validate_result(
    result: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    spec,
    launch: Mapping[str, Any],
    gate_row: Mapping[str, Any],
    seed_table: Mapping[str, Any],
) -> dict[str, Any]:
    claimed = result.get("result_sha256")
    body = dict(result)
    body.pop("result_sha256", None)
    if claimed != stable_hash(body):
        raise ContractError(f"eval {spec.index}: result hash differs")
    identity = result.get("identity") or {}
    expected_task_table = single_task_seed_table(seed_table, spec.task_id)
    expected_identity = _progress_identity(
        manifest,
        spec,
        launch,
        {
            "identity_sha256": gate_row.get("identity_sha256"),
            "checkpoint_sha256": gate_row.get("checkpoint_sha256"),
        },
        expected_task_table,
    )
    identity_body = dict(identity)
    identity_claim = identity_body.pop("eval_contract_sha256", None)
    checks = (
        result.get("schema_version") == 1,
        result.get("status") == "complete",
        identity == expected_identity,
        int(identity.get("eval_index", -1)) == spec.index,
        int(identity.get("training_index", -1)) == spec.training_index,
        identity.get("setting_id") == spec.run.setting_id,
        int(identity.get("training_seed", -1)) == spec.run.seed,
        int(identity.get("task_id", -1)) == spec.task_id,
        identity.get("rails") == ["learned", "bfs"],
        int(identity.get("episodes_per_task_per_rail", -1)) == 50,
        identity.get("launch_sha256") == launch["launch_sha256"] == gate_row.get("launch_sha256"),
        identity.get("identity_sha256") == gate_row.get("identity_sha256"),
        identity.get("checkpoint_sha256") == gate_row.get("checkpoint_sha256"),
        identity.get("task_seed_table_sha256") == expected_task_table["sha256"],
        identity.get("full_final_seed_table_sha256") == seed_table["sha256"] == gate_row.get("final_seed_table_sha256"),
        identity.get("evaluation_seed_protocol_sha256") == seed_table["protocol_sha256"],
        identity.get("package_protocol_sha256") == launch["hashes"]["package_protocol_sha256"],
        identity.get("package_seed_table_sha256") == launch["hashes"]["package_seed_table_sha256"],
        identity.get("source_sha256") == launch["hashes"]["source_sha256"],
        identity.get("evaluation_source_sha256") == launch["hashes"]["evaluation_source_sha256"],
        identity.get("runtime_sha256") == launch["hashes"]["runtime_sha256"],
        identity_claim == stable_hash(identity_body),
    )
    if not all(checks):
        raise ContractError(f"eval {spec.index}: result identity differs")
    expected_seeds = expected_task_table["seeds"][0]
    rows: dict[str, Any] = {}
    success_vectors: dict[str, list[bool]] = {}
    rails = result.get("rails") or {}
    if set(rails) != {"learned", "bfs"} or identity.get("rails") != ["learned", "bfs"]:
        raise ContractError(f"eval {spec.index}: rail coverage/order differs")
    for rail in ("learned", "bfs"):
        episodes = rails[rail].get("episodes") or []
        metrics = rails[rail].get("metrics") or {}
        actual_seed_identity = [
            (row.get("task_id"), row.get("episode_index"), row.get("episode_seed"))
            for row in episodes
        ]
        expected_seed_identity = [(spec.task_id, episode, seed) for episode, seed in enumerate(expected_seeds)]
        if len(episodes) != 50 or actual_seed_identity != expected_seed_identity:
            raise ContractError(f"eval {spec.index}/{rail}: locked episode coverage differs")
        validate_exact_episode_count(
            metrics.get("eval/num_episodes"), 50, f"eval {spec.index}/{rail}"
        )
        success = metrics.get("eval/successes")
        success_rate = metrics.get("eval/success_rate")
        progress = metrics.get("eval/distance_reduction_frac")
        if not all(finite(value) for value in (success, success_rate, progress)):
            raise ContractError(f"eval {spec.index}/{rail}: non-finite headline metric")
        if not 0 <= float(success) <= 50 or abs(float(success_rate) - float(success) / 50.0) > 1e-6:
            raise ContractError(f"eval {spec.index}/{rail}: success metrics are inconsistent")
        rows[rail] = {
            "successes": float(success),
            "success_rate": float(success_rate),
            "distance_reduction_frac": float(progress),
        }
        success_vectors[rail] = validate_episode_success_accounting(
            [row.get("success") for row in episodes],
            success,
            f"eval {spec.index}/{rail}",
        )
    rows["paired"] = {
        "learned_only_success": sum(a and not b for a, b in zip(success_vectors["learned"], success_vectors["bfs"], strict=True)),
        "bfs_only_success": sum(b and not a for a, b in zip(success_vectors["learned"], success_vectors["bfs"], strict=True)),
        "both_success": sum(a and b for a, b in zip(success_vectors["learned"], success_vectors["bfs"], strict=True)),
        "neither_success": sum(not a and not b for a, b in zip(success_vectors["learned"], success_vectors["bfs"], strict=True)),
    }
    return rows


def aggregate(manifest: Mapping[str, Any], repo_root: Path) -> dict[str, Any]:
    gate = load_final_gate(manifest)
    gate_rows = {int(row["index"]): row for row in gate["runs"]}
    launches = {run.index: trainer_command(manifest, run, repo_root=repo_root) for run in expand_runs(manifest)}
    evaluation_source_hashes = {
        launch["hashes"]["evaluation_source_sha256"] for launch in launches.values()
    }
    if len(evaluation_source_hashes) != 1 or gate.get("evaluation_source_sha256") not in evaluation_source_hashes:
        raise ContractError("final aggregate evaluation-source provenance differs")
    seed_bundle = load_seed_table(manifest, repo_root / "experiments" / "22-treewm-grounded-gauge-formal-v1" / "eval_seed_table.json")
    result_root = Path(manifest["paths"]["final_eval_root"]) / "results"
    expected_paths = {result_root / f"{index:03d}.json" for index in range(FINAL_EVAL_TASKS)}
    actual_paths = set(result_root.glob("[0-9][0-9][0-9].json"))
    if actual_paths != expected_paths:
        missing = sorted(str(path) for path in expected_paths - actual_paths)
        unexpected = sorted(str(path) for path in actual_paths - expected_paths)
        raise ContractError(f"final results are not exact 200 coverage; missing={missing}, unexpected={unexpected}")
    cells: list[dict[str, Any]] = []
    for index in range(FINAL_EVAL_TASKS):
        spec = eval_at(manifest, index)
        result = read_json(result_root / f"{index:03d}.json")
        rails = _validate_result(
            result,
            manifest=manifest,
            spec=spec,
            launch=launches[spec.training_index],
            gate_row=gate_rows[spec.training_index],
            seed_table=seed_bundle["settings"][spec.run.setting_id],
        )
        cells.append({
            "index": index,
            "training_index": spec.training_index,
            "setting_id": spec.run.setting_id,
            "training_seed": spec.run.seed,
            "task_id": spec.task_id,
            "result_sha256": result["result_sha256"],
            "rails": rails,
        })

    def summary(rows: Sequence[Mapping[str, Any]], rail: str) -> dict[str, float]:
        successes = sum(float(row["rails"][rail]["successes"]) for row in rows)
        episodes = len(rows) * 50
        return {
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / episodes,
            "macro_cell_success_rate": statistics.fmean(float(row["rails"][rail]["success_rate"]) for row in rows),
            "macro_distance_reduction_frac": statistics.fmean(float(row["rails"][rail]["distance_reduction_frac"]) for row in rows),
        }

    overall = {rail: summary(cells, rail) for rail in ("learned", "bfs")}
    paired_discordance = {
        key: sum(int(row["rails"]["paired"][key]) for row in cells)
        for key in ("learned_only_success", "bfs_only_success", "both_success", "neither_success")
    }
    per_setting: dict[str, Any] = {}
    for setting in SETTING_IDS:
        selected = [row for row in cells if row["setting_id"] == setting]
        per_setting[setting] = {rail: summary(selected, rail) for rail in ("learned", "bfs")}
        per_setting[setting]["learned_minus_bfs_success_rate"] = (
            per_setting[setting]["learned"]["success_rate"] - per_setting[setting]["bfs"]["success_rate"]
        )
    per_task = {
        str(task): {rail: summary([row for row in cells if row["task_id"] == task], rail) for rail in ("learned", "bfs")}
        for task in TASK_IDS
    }
    criterion = manifest["final_evaluation"]["promotion_criterion"]
    learned_delta = overall["learned"]["success_rate"] - overall["bfs"]["success_rate"]
    settings_noninferior = sum(per_setting[setting]["learned_minus_bfs_success_rate"] >= 0.0 for setting in SETTING_IDS)
    seed_rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        selected = [row for row in cells if int(row["training_seed"]) == seed]
        if len(selected) != 50:
            raise ContractError(f"training seed {seed}: primary-unit coverage is not 50 cells")
        learned = statistics.fmean(float(row["rails"]["learned"]["success_rate"]) for row in selected)
        bfs = statistics.fmean(float(row["rails"]["bfs"]["success_rate"]) for row in selected)
        seed_rows.append({
            "training_seed": seed,
            "cells": len(selected),
            "episodes_per_rail": len(selected) * 50,
            "learned_macro_success_rate": learned,
            "bfs_macro_success_rate": bfs,
            "learned_minus_bfs_macro_success_rate": learned - bfs,
        })

    def t_summary(values: Sequence[float]) -> dict[str, Any]:
        if len(values) != int(criterion["training_seed_replicates"]):
            raise ContractError("primary inference does not contain four training seeds")
        return seed_t_summary(values, float(criterion["t_critical_975_df3"]))

    seed_inference = {
        "unit": "training_seed",
        "replicates": seed_rows,
        "learned_macro_success_rate": t_summary([row["learned_macro_success_rate"] for row in seed_rows]),
        "bfs_macro_success_rate": t_summary([row["bfs_macro_success_rate"] for row in seed_rows]),
        "paired_learned_minus_bfs": t_summary([row["learned_minus_bfs_macro_success_rate"] for row in seed_rows]),
    }
    gates = {
        "learned_has_success": overall["learned"]["successes"] >= int(criterion["min_learned_total_successes"]),
        "paired_seed_delta_ci_lower_nonnegative": seed_inference["paired_learned_minus_bfs"]["ci95_lower"] >= float(criterion["min_paired_seed_delta_ci_lower"]),
        "learned_setting_noninferiority_coverage": settings_noninferior >= int(criterion["min_settings_learned_noninferior"]),
    }
    output: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "campaign_id": manifest["campaign_id"],
        "formal_validation": True,
        "training_stage_gate_sha256": gate["gate_sha256"],
        "package_protocol_sha256": gate["package_protocol_sha256"],
        "evaluation_source_sha256": next(iter(evaluation_source_hashes)),
        "evaluation_seed_table_sha256": seed_bundle["sha256"],
        "result_cells": FINAL_EVAL_TASKS,
        "episodes_per_rail": FINAL_EVAL_TASKS * 50,
        "overall": overall,
        "pooled_episode_summary_is_descriptive_only": True,
        "primary_seed_level_inference": seed_inference,
        "paired_episode_discordance": paired_discordance,
        "learned_minus_bfs_success_rate": learned_delta,
        "settings_learned_noninferior": settings_noninferior,
        "per_setting": per_setting,
        "per_task": per_task,
        "promotion_gates": gates,
        "promotion_eligible": all(gates.values()),
        "adaptive_selection": False,
        "cells": cells,
    }
    output["aggregate_sha256"] = stable_hash(output)
    return output


def publish(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if read_json(path) != dict(value):
            raise ContractError(f"existing aggregate differs: {path}")
        return
    atomic_json(path, value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--manifest", type=Path, default=CAMPAIGN_DIR / "manifest.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    value = aggregate(manifest, args.repo_root.resolve())
    print(json.dumps(value, sort_keys=True, indent=2))
    if args.publish:
        destination = args.output or Path(manifest["paths"]["final_eval_root"]) / "aggregate.json"
        publish(destination, value)
        print(f"published {destination}", file=sys.stderr)
    else:
        print("dry-run only: aggregate not published", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"grounded-formal aggregate error: {exc}", file=sys.stderr)
        raise SystemExit(2)
