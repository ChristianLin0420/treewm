#!/usr/bin/env python3
"""Validate exp15/exp16 decisions and seal their exact bytes into formal campaign 17.

The command is read-only unless --publish is supplied. It never chooses a recipe:
it accepts only the deterministic F/H decision already published by exp16.
"""

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
    CAMPAIGN_DIR,
    ContractError,
    PREREQUISITE_BINDINGS_PATH,
    PROTOCOL_LOCK_PATH,
    SETTING_IDS,
    atomic_json,
    file_sha256,
    load_manifest,
    protocol_sha256,
    read_json,
    require,
    stable_hash,
)


def _sha(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _self_hash(payload: Mapping[str, Any], key: str, label: str) -> str:
    claimed = payload.get(key)
    body = dict(payload)
    body.pop(key, None)
    require(_sha(claimed) and claimed == stable_hash(body), f"{label} self-hash differs")
    return str(claimed)


def _regular_file(path: Path, label: str) -> None:
    require(path.is_file() and not path.is_symlink(), f"{label} is missing or symlinked: {path}")


def _close(left: object, right: object, *, tolerance: float = 1e-12) -> bool:
    return bool(
        _finite(left)
        and _finite(right)
        and math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)
    )


def _exact_boolean_gates(
    record: Mapping[str, Any],
    key: str,
    expected_names: Sequence[str],
    label: str,
) -> dict[str, bool]:
    gates = record.get(key)
    require(
        isinstance(gates, dict)
        and set(gates) == set(expected_names)
        and all(type(value) is bool for value in gates.values()),
        f"{label} {key} schema differs",
    )
    return {name: bool(gates[name]) for name in expected_names}


def _recompute_exp15_claims(
    contract: Mapping[str, Any],
    report: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> set[tuple[str, str, int]]:
    """Derive every exp15 acceptance claim from its complete raw run records."""
    raw = contract["raw_report_recomputation"]
    settings = tuple(str(value) for value in raw["settings"])
    arms = tuple(str(value) for value in raw["arms"])
    seeds = tuple(int(value) for value in raw["seeds"])
    expected_keys = {
        (setting, arm, seed)
        for setting in settings
        for arm in arms
        for seed in seeds
    }
    require(len(records) == len(expected_keys) == 40, "exp15 report does not contain 40 raw runs")

    keyed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    integrity_pass: dict[tuple[str, str, int], bool] = {}
    scientific_pass: dict[tuple[str, str, int], bool] = {}
    for record in records:
        require(isinstance(record, dict), "exp15 raw run record is not an object")
        setting = record.get("setting_id")
        arm = record.get("arm_id")
        seed = record.get("seed")
        require(
            isinstance(setting, str)
            and isinstance(arm, str)
            and isinstance(seed, int)
            and not isinstance(seed, bool),
            "exp15 raw run identity is malformed",
        )
        key = (setting, arm, seed)
        require(key in expected_keys and key not in keyed, "exp15 raw run matrix is missing, extra, or duplicated")
        setting_index = settings.index(setting)
        arm_index = arms.index(arm)
        seed_index = seeds.index(seed)
        expected_index = ((setting_index * len(arms)) + arm_index) * len(seeds) + seed_index
        require(record.get("index") == expected_index, f"exp15 {key} array index differs")
        require(
            record.get("run_name") == f"repair-{setting}-arm{arm.lower()}-seed{seed}",
            f"exp15 {key} run name differs",
        )
        integrity = _exact_boolean_gates(
            record,
            "integrity_gates",
            raw["required_integrity_gates"],
            f"exp15 {key}",
        )
        scientific = _exact_boolean_gates(
            record,
            "scientific_gates",
            raw["required_scientific_gates"],
            f"exp15 {key}",
        )
        derived_integrity = all(integrity.values())
        derived_scientific = all(scientific.values())
        require(
            record.get("integrity_pass") is derived_integrity,
            f"exp15 {key} integrity_pass differs from raw gates",
        )
        require(
            record.get("scientific_pass") is derived_scientific,
            f"exp15 {key} scientific_pass differs from raw gates",
        )
        require(record.get("error") is None, f"exp15 accepted raw run {key} reports an error")

        metrics = record.get("metrics")
        final = metrics.get("final") if isinstance(metrics, dict) else None
        terminal_keys = {
            "num_episodes",
            "successes",
            "success_rate",
            "distance_reduction_frac",
        }
        require(
            isinstance(final, dict)
            and set(final) == terminal_keys
            and all(_finite(final[name]) for name in terminal_keys),
            f"exp15 {key} terminal outcomes are incomplete or non-finite",
        )
        episodes = float(final["num_episodes"])
        successes = float(final["successes"])
        require(
            episodes == float(raw["terminal_episodes_per_run"])
            and successes.is_integer()
            and 0.0 <= successes <= episodes
            and _close(final["success_rate"], successes / episodes, tolerance=1e-6),
            f"exp15 {key} terminal success counts/rate are inconsistent",
        )
        keyed[key] = record
        integrity_pass[key] = derived_integrity
        scientific_pass[key] = derived_scientific

    require(set(keyed) == expected_keys, "exp15 raw run matrix is incomplete")
    integrity_count = sum(integrity_pass.values())
    candidate = str(contract["required_candidate_arm"])
    control = str(contract["required_control_arm"])
    setting_pass = {
        setting: sum(scientific_pass[(setting, candidate, seed)] for seed in seeds)
        >= int(raw["candidate_seeds_per_passing_setting"])
        for setting in settings
    }
    settings_passing = sum(setting_pass.values())

    paired: list[dict[str, Any]] = []
    for setting in settings:
        for seed in seeds:
            candidate_final = keyed[(setting, candidate, seed)]["metrics"]["final"]
            control_final = keyed[(setting, control, seed)]["metrics"]["final"]
            paired.append(
                {
                    "setting_id": setting,
                    "seed": seed,
                    "success_delta_candidate_minus_control": float(candidate_final["success_rate"])
                    - float(control_final["success_rate"]),
                    "distance_reduction_delta_candidate_minus_control": float(
                        candidate_final["distance_reduction_frac"]
                    )
                    - float(control_final["distance_reduction_frac"]),
                }
            )
    success_delta = statistics.fmean(
        row["success_delta_candidate_minus_control"] for row in paired
    )
    progress_delta = statistics.fmean(
        row["distance_reduction_delta_candidate_minus_control"] for row in paired
    )
    candidate_rows = [
        keyed[(setting, candidate, seed)]
        for setting in settings
        for seed in seeds
    ]
    candidate_successes = [
        float(row["metrics"]["final"]["successes"]) for row in candidate_rows
    ]
    candidate_progress = [
        float(row["metrics"]["final"]["distance_reduction_frac"])
        for row in candidate_rows
    ]
    total_successes = sum(candidate_successes)
    mean_progress = statistics.fmean(candidate_progress)
    positive_runs = sum(value > 0.0 for value in candidate_progress)
    aggregate_gates = {
        "integrity_40_of_40": integrity_count == int(contract["required_integrity_runs"]),
        "preregistered_candidate_setting_quorum": settings_passing
        >= int(raw["candidate_settings_required"]),
        "candidate_not_all_zero_success": total_successes
        >= float(raw["min_candidate_total_successes"]),
        "candidate_positive_mean_progress": mean_progress
        > float(raw["min_candidate_mean_distance_reduction_exclusive"]),
        "candidate_positive_progress_run_quorum": positive_runs
        >= int(raw["min_candidate_runs_with_positive_progress"]),
        "candidate_success_noninferior_to_control": success_delta
        >= float(raw["min_paired_success_delta_vs_control"]),
        "candidate_distance_reduction_noninferior_to_control": progress_delta
        >= float(raw["min_paired_distance_reduction_delta_vs_control"]),
        "adaptive_selection_disabled": bool(
            raw["no_adaptive_arm_selection"] and candidate == "C" and control == "A"
        ),
    }
    accepted = all(aggregate_gates.values())
    require(report.get("integrity_runs_passing") == integrity_count, "exp15 integrity count differs from raw runs")
    require(report.get("candidate_setting_pass") == setting_pass, "exp15 candidate setting quorum differs from raw runs")
    require(report.get("candidate_settings_passing") == settings_passing, "exp15 candidate setting count differs from raw runs")
    require(
        report.get("sensitivity_arms_are_nonpromotable")
        == raw["nonpromotable_sensitivity_arms"],
        "exp15 sensitivity-arm claim differs",
    )
    require(report.get("aggregate_gates") == aggregate_gates, "exp15 aggregate gates differ from raw runs")
    require(report.get("accepted") is accepted, "exp15 accepted claim differs from raw runs")
    require(
        report.get("status")
        == ("accepted_for_fresh_formal_campaign_design" if accepted else "rejected_or_incomplete"),
        "exp15 status differs from recomputed acceptance",
    )
    expected_metrics = {
        "candidate_total_successes": total_successes,
        "candidate_mean_distance_reduction": mean_progress,
        "candidate_runs_with_positive_progress": positive_runs,
        "paired_mean_success_delta_candidate_minus_control": success_delta,
        "paired_mean_distance_reduction_delta_candidate_minus_control": progress_delta,
    }
    reported_metrics = report.get("aggregate_metrics")
    require(
        isinstance(reported_metrics, dict)
        and set(reported_metrics) == set(expected_metrics)
        and all(_close(reported_metrics[key], value) for key, value in expected_metrics.items()),
        "exp15 aggregate metrics differ from raw runs",
    )
    reported_paired = report.get("paired_comparisons")
    require(
        isinstance(reported_paired, list) and len(reported_paired) == len(paired),
        "exp15 paired comparison matrix differs from raw runs",
    )
    for observed, expected in zip(reported_paired, paired):
        require(
            isinstance(observed, dict)
            and set(observed) == set(expected)
            and observed.get("setting_id") == expected["setting_id"]
            and observed.get("seed") == expected["seed"]
            and _close(
                observed.get("success_delta_candidate_minus_control"),
                expected["success_delta_candidate_minus_control"],
            )
            and _close(
                observed.get("distance_reduction_delta_candidate_minus_control"),
                expected["distance_reduction_delta_candidate_minus_control"],
            ),
            "exp15 paired comparisons differ from raw runs",
        )
    require(report.get("missing_or_extra_keys") == [], "exp15 missing/extra claim differs from raw matrix")
    return expected_keys


def validate_exp15(manifest: Mapping[str, Any], acceptance_path: Path, plan_path: Path) -> dict[str, Any]:
    _regular_file(acceptance_path, "exp15 acceptance")
    _regular_file(plan_path, "exp15 launch plan")
    report = read_json(acceptance_path)
    contract = manifest["prerequisites"]["exp15"]
    require(report.get("schema_version") == 1, "exp15 acceptance schema differs")
    require(report.get("campaign_id") == contract["campaign_id"], "exp15 campaign differs")
    require(report.get("status") == contract["required_status"], "exp15 was not accepted")
    require(report.get("accepted") is True, "exp15 accepted flag is false")
    require(report.get("formal_validation") is False, "exp15 incorrectly claims formal validation")
    require(report.get("preregistered_candidate_arm") == contract["required_candidate_arm"], "exp15 candidate differs")
    require(report.get("matched_control_arm") == contract["required_control_arm"], "exp15 matched control differs")
    require(
        report.get("integrity_runs_passing") == contract["required_integrity_runs"],
        "exp15 lacks 40/40 integrity",
    )
    records = report.get("runs")
    require(isinstance(records, list), "exp15 report runs are missing")
    expected_keys = _recompute_exp15_claims(contract, report, records)

    plan = read_json(plan_path)
    plan_hash = _self_hash(plan, "plan_sha256", "exp15 launch plan")
    require(plan.get("schema_version") == 1, "exp15 launch-plan schema differs")
    require(plan.get("campaign_id") == contract["campaign_id"], "exp15 plan campaign differs")
    require(
        plan.get("status") == "sealed_bounded_repair_pilot_plan",
        "exp15 launch-plan status differs",
    )
    require(plan.get("formal_validation") is False, "exp15 plan incorrectly claims formal validation")
    plan_records = plan.get("runs") or []
    plan_keys = {
        (
            str(record.get("setting_id")),
            str(record.get("arm_id")),
            int(record.get("seed", -1)),
        )
        for record in plan_records
    }
    require(
        len(plan_records) == 40 and plan_keys == expected_keys,
        "exp15 launch plan run matrix differs",
    )
    common = plan.get("common_hashes") or {}
    for key in (
        "package_protocol_sha256",
        "source_sha256",
        "runtime_sha256",
        "actual_evaluation_bank_sha256",
    ):
        require(_sha(common.get(key)), f"exp15 plan has malformed {key}")
    for key in ("package_protocol_sha256", "source_sha256", "runtime_sha256"):
        require(common[key] == contract[key], f"exp15 pinned {key} differs")
    require(
        report.get("actual_evaluation_bank_sha256")
        == common["actual_evaluation_bank_sha256"],
        "exp15 acceptance/launch-plan evaluation bank differs",
    )
    exp16_compatible: dict[str, Any] = {
        "schema_version": 1,
        "status": "accepted_exp15_prerequisite",
        "campaign_id": contract["campaign_id"],
        "accepted_status": report["status"],
        "candidate_arm": report["preregistered_candidate_arm"],
        "integrity_runs_passing": report["integrity_runs_passing"],
        "acceptance_path": str(acceptance_path.resolve()),
        "acceptance_sha256": file_sha256(acceptance_path),
        "acceptance_canonical_sha256": stable_hash(report),
        "launch_plan_path": str(plan_path.resolve()),
        "launch_plan_sha256": file_sha256(plan_path),
        "launch_plan_canonical_sha256": plan_hash,
        "package_protocol_sha256": common["package_protocol_sha256"],
        "source_sha256": common["source_sha256"],
        "runtime_sha256": common["runtime_sha256"],
        "actual_evaluation_bank_sha256": common["actual_evaluation_bank_sha256"],
    }
    exp16_compatible["prerequisite_sha256"] = stable_hash(exp16_compatible)
    return {
        "campaign_id": contract["campaign_id"],
        "acceptance_path": str(acceptance_path.resolve()),
        "acceptance_file_sha256": file_sha256(acceptance_path),
        "launch_plan_path": str(plan_path.resolve()),
        "launch_plan_file_sha256": file_sha256(plan_path),
        "launch_plan_sha256": plan_hash,
        "package_protocol_sha256": common["package_protocol_sha256"],
        "trainer_source_sha256": common["source_sha256"],
        "runtime_sha256": common["runtime_sha256"],
        "actual_evaluation_bank_sha256": common["actual_evaluation_bank_sha256"],
        "accepted_status": report["status"],
        "candidate_arm": report["preregistered_candidate_arm"],
        "exp16_prerequisite": exp16_compatible,
        "exp16_prerequisite_sha256": exp16_compatible["prerequisite_sha256"],
    }


def _selected_recipe_from_artifact(artifact: Mapping[str, Any]) -> Mapping[str, Any]:
    recipe = artifact.get("selected_recipe")
    require(isinstance(recipe, dict), "exp16 selected recipe payload is missing")
    return recipe


def validate_exp16(
    manifest: Mapping[str, Any],
    acceptance_path: Path,
    selected_recipe_path: Path,
) -> dict[str, Any]:
    _regular_file(acceptance_path, "exp16 acceptance")
    _regular_file(selected_recipe_path, "exp16 selected recipe")
    acceptance = read_json(acceptance_path)
    artifact = read_json(selected_recipe_path)
    contract = manifest["prerequisites"]["exp16"]
    quorum = contract["required_selection_quorum"]
    bridge_seeds = tuple(int(seed) for seed in quorum["bridge_seeds"])
    require(acceptance.get("schema_version") == 1, "exp16 acceptance schema differs")
    require(acceptance.get("campaign_id") == contract["campaign_id"], "exp16 campaign differs")
    require(acceptance.get("status") in contract["accepted_statuses"], "exp16 selected no formal recipe")
    require(acceptance.get("accepted") is True, "exp16 accepted flag is false")
    require(acceptance.get("formal_validation") is False, "exp16 incorrectly claims formal validation")
    acceptance_report_hash = _self_hash(acceptance, "report_sha256", "exp16 acceptance")
    selected_arm = acceptance.get("selected_arm")
    require(selected_arm in contract["allowed_selected_arms"], "exp16 selected arm is not F/H")
    expected_status = (
        "selected_full_for_fresh_formal_campaign_design"
        if selected_arm == "F"
        else "selected_half_for_fresh_formal_campaign_design"
    )
    require(acceptance.get("status") == expected_status, "exp16 status/selected arm differ")
    require(
        acceptance.get("selection_precedence") == contract["selection_precedence"],
        "exp16 selection precedence differs",
    )

    acceptance_file_hash = file_sha256(acceptance_path)
    require(artifact.get("schema_version") == 1, "exp16 selected-recipe schema differs")
    require(artifact.get("campaign_id") == contract["campaign_id"], "exp16 recipe campaign differs")
    require(artifact.get("status") == "selected_recipe", "exp16 recipe artifact did not select a recipe")
    require(artifact.get("selected") is True, "exp16 recipe artifact selected flag is false")
    require(artifact.get("formal_validation") is False, "exp16 recipe incorrectly claims formal validation")
    selected_recipe_artifact_hash = _self_hash(
        artifact, "artifact_sha256", "exp16 selected-recipe artifact"
    )
    require(artifact.get("selected_arm") == selected_arm, "exp16 artifacts disagree on selected arm")
    require(
        artifact.get("acceptance_sha256") == acceptance_file_hash,
        "exp16 recipe does not bind the exact acceptance bytes",
    )
    require(
        artifact.get("bridge_acceptance_sha256") == acceptance_report_hash,
        "exp16 recipe does not bind the canonical acceptance report",
    )
    recipe = _selected_recipe_from_artifact(artifact)
    recipe_hash = stable_hash(recipe)
    require(acceptance.get("selected_recipe") == recipe, "exp16 acceptance/recipe payloads differ")
    require(acceptance.get("selected_recipe_sha256") == recipe_hash, "exp16 acceptance recipe hash differs")
    require(artifact.get("selected_recipe_sha256") == recipe_hash, "exp16 artifact recipe hash differs")
    require(
        artifact.get("selection_rule_sha256") == acceptance.get("selection_rule_sha256")
        and _sha(artifact.get("selection_rule_sha256")),
        "exp16 selection-rule hashes differ",
    )

    expected_losses = manifest["prerequisites"]["allowed_selected_recipes"][selected_arm]
    for key, value in expected_losses.items():
        require(float(recipe.get(key, -1.0)) == float(value), f"exp16 selected {key} differs")
    common = manifest["method"]["grounded_multistep"]
    expected_common = {
        "transition_mode": common["transition_mode"],
        "grounded_select_action_weight": common["selector_weights"]["action"],
        "grounded_select_endpoint_weight": common["selector_weights"]["endpoint"],
        "grounded_select_horizon_weight": common["selector_weights"]["horizon"],
    }
    for key, value in expected_common.items():
        require(recipe.get(key) == value, f"exp16 selected {key} differs")
    expected_recipe = {
        "id": selected_arm,
        "label": (
            "grounded-conservative-full-loss-scale-lr3e-5"
            if selected_arm == "F"
            else "grounded-conservative-half-loss-scale-lr3e-5"
        ),
        "world_lr": 3e-5,
        **expected_common,
        "grounded_loss_scale": 1.0 if selected_arm == "F" else 0.5,
        **expected_losses,
    }
    require(recipe == expected_recipe, "exp16 selected recipe schema/values differ")

    exp15_prerequisite = acceptance.get("exp15_prerequisite")
    require(isinstance(exp15_prerequisite, dict), "exp16 lacks its exp15 prerequisite binding")
    exp15_prerequisite_hash = _self_hash(
        exp15_prerequisite,
        "prerequisite_sha256",
        "exp16 embedded exp15 prerequisite",
    )
    require(
        artifact.get("exp15_prerequisite") == exp15_prerequisite
        and acceptance.get("exp15_prerequisite_sha256") == exp15_prerequisite_hash
        and artifact.get("exp15_prerequisite_sha256") == exp15_prerequisite_hash,
        "exp16 acceptance/artifact exp15 prerequisite bindings differ",
    )

    provenance = acceptance.get("provenance") or {}
    for key in (
        "manifest_sha256",
        "package_protocol_sha256",
        "source_sha256",
        "runtime_sha256",
        "actual_evaluation_bank_sha256",
        "exp15_prerequisite_sha256",
    ):
        require(_sha(provenance.get(key)), f"exp16 acceptance has malformed provenance {key}")
        require(
            artifact.get(key) == provenance[key],
            f"exp16 acceptance/artifact provenance {key} differs",
        )
    for key in (
        "manifest_sha256",
        "package_protocol_sha256",
        "source_sha256",
        "runtime_sha256",
    ):
        require(
            provenance[key] == contract[key],
            f"exp16 pinned provenance {key} differs",
        )
    require(
        acceptance.get("actual_evaluation_bank_sha256")
        == provenance["actual_evaluation_bank_sha256"],
        "exp16 acceptance evaluation-bank hashes differ",
    )
    require(
        provenance.get("exp15_prerequisite") == exp15_prerequisite,
        "exp16 provenance exp15 prerequisite payload differs",
    )

    records = acceptance.get("runs") or []
    require(len(records) == 40, "exp16 acceptance lacks exactly 40 runs")
    run_names = [record.get("run_name") for record in records]
    require(
        all(isinstance(name, str) and name for name in run_names)
        and len(set(run_names)) == 40,
        "exp16 acceptance run names are missing or duplicated",
    )
    keyed = {
        (
            str(record.get("setting_id")),
            str(record.get("arm_id")),
            int(record.get("seed", -1)),
        ): record
        for record in records
    }
    expected_keys = {
        (setting, arm, seed)
        for setting in SETTING_IDS
        for arm in contract["allowed_selected_arms"]
        for seed in bridge_seeds
    }
    require(
        len(keyed) == 40 and set(keyed) == expected_keys,
        "exp16 acceptance run design is not exact 10 settings x 2 arms x 2 seeds",
    )
    arms_by_name = {str(record["run_name"]): record.get("arm_id") for record in records}
    require(
        set(arms_by_name.values()) == set(contract["allowed_selected_arms"])
        and sum(arm == selected_arm for arm in arms_by_name.values())
        == int(quorum["runs_per_arm"]),
        "exp16 acceptance arm/run coverage differs",
    )
    require(
        acceptance.get("integrity_runs_passing") == int(quorum["integrity_runs"])
        and all(record.get("integrity_pass") is True for record in records),
        "exp16 acceptance lacks 40/40 run integrity",
    )
    scientific_counts: dict[str, int] = {}
    settings_with_both: dict[str, int] = {}
    settings_with_one: dict[str, int] = {}
    scientific_eligible: dict[str, bool] = {}
    outcome_eligible: dict[str, bool] = {}
    setting_seed_pass_counts: dict[str, dict[str, int]] = {}
    setting_passes: dict[str, dict[str, bool]] = {}
    scientific_quorum_gates: dict[str, dict[str, bool]] = {}
    outcome_gates: dict[str, dict[str, bool]] = {}
    for arm in contract["allowed_selected_arms"]:
        per_setting = {
            setting: sum(
                keyed[(setting, arm, seed)].get("scientific_pass") is True
                for seed in bridge_seeds
            )
            for setting in SETTING_IDS
        }
        scientific_counts[arm] = sum(per_setting.values())
        settings_with_both[arm] = sum(count == 2 for count in per_setting.values())
        settings_with_one[arm] = sum(count >= 1 for count in per_setting.values())
        setting_seed_pass_counts[arm] = per_setting
        setting_passes[arm] = {
            setting: count == len(bridge_seeds)
            for setting, count in per_setting.items()
        }
        scientific_quorum_gates[arm] = {
            "scientific_runs_at_least_18_of_20": scientific_counts[arm]
            >= int(quorum["min_scientific_runs_per_arm"]),
            "both_seed_settings_at_least_8_of_10": settings_with_both[arm]
            >= int(quorum["min_settings_with_both_seeds"]),
            "every_setting_has_at_least_one_passing_seed": settings_with_one[arm]
            >= int(quorum["min_settings_with_at_least_one_seed"]),
        }
        scientific_eligible[arm] = all(scientific_quorum_gates[arm].values())
        terminal = [
            keyed[(setting, arm, seed)].get("metrics", {}).get("final", {})
            for setting in SETTING_IDS
            for seed in bridge_seeds
        ]
        successes = [row.get("successes") for row in terminal]
        progress = [row.get("distance_reduction_frac") for row in terminal]
        success_rates = [row.get("success_rate") for row in terminal]
        require(
            all(_finite(value) for value in [*successes, *progress, *success_rates]),
            f"exp16 arm {arm} terminal outcomes are incomplete/non-finite",
        )
        outcome_gates[arm] = {
            "finite_terminal_outcomes": True,
            "not_all_zero_success": sum(float(value) for value in successes)
            >= float(quorum["min_total_successes_per_arm"]),
            "positive_mean_progress": statistics.fmean(
                float(value) for value in progress
            )
            > float(quorum["min_mean_distance_reduction_exclusive"]),
            "positive_progress_run_quorum": sum(
                float(value) > 0.0 for value in progress
            )
            >= int(quorum["min_positive_progress_runs_per_arm"]),
        }
        outcome_eligible[arm] = all(outcome_gates[arm].values())

    success_deltas = [
        float(keyed[(setting, "H", seed)]["metrics"]["final"]["success_rate"])
        - float(keyed[(setting, "F", seed)]["metrics"]["final"]["success_rate"])
        for setting in SETTING_IDS
        for seed in bridge_seeds
    ]
    progress_deltas = [
        float(
            keyed[(setting, "H", seed)]["metrics"]["final"][
                "distance_reduction_frac"
            ]
        )
        - float(
            keyed[(setting, "F", seed)]["metrics"]["final"][
                "distance_reduction_frac"
            ]
        )
        for setting in SETTING_IDS
        for seed in bridge_seeds
    ]
    half_noninferiority_gates = {
        "success_noninferior_to_full": statistics.fmean(success_deltas)
        >= float(quorum["min_half_minus_full_success_rate"]),
        "distance_reduction_noninferior_to_full": statistics.fmean(progress_deltas)
        >= float(quorum["min_half_minus_full_distance_reduction"]),
    }
    half_noninferior = all(half_noninferiority_gates.values())
    arm_eligible = {
        "F": scientific_eligible["F"] and outcome_eligible["F"],
        "H": scientific_eligible["H"] and outcome_eligible["H"] and half_noninferior,
    }
    derived_selected_arm = (
        "F" if arm_eligible["F"] else "H" if arm_eligible["H"] else None
    )
    require(
        selected_arm == derived_selected_arm,
        "exp16 selected arm differs from the exact preregistered quorum and F-then-H rule",
    )
    require(
        acceptance.get("arm_scientific_runs_passing") == scientific_counts,
        "exp16 reported scientific run counts differ from exact records",
    )
    require(
        acceptance.get("arm_setting_seed_pass_count") == setting_seed_pass_counts
        and acceptance.get("arm_setting_pass") == setting_passes
        and acceptance.get("arm_scientific_quorum_gates")
        == scientific_quorum_gates
        and acceptance.get("arm_outcome_gates") == outcome_gates
        and acceptance.get("half_noninferiority_gates")
        == half_noninferiority_gates,
        "exp16 reported quorum/outcome gate details differ from exact records",
    )
    expected_aggregate_gates = {
        "integrity_40_of_40": True,
        "full_scientific_quorum": scientific_eligible["F"],
        "half_scientific_quorum": scientific_eligible["H"],
        "full_outcome_gates": outcome_eligible["F"],
        "half_outcome_gates": outcome_eligible["H"],
        "half_success_noninferior_to_full": half_noninferiority_gates[
            "success_noninferior_to_full"
        ],
        "half_distance_reduction_noninferior_to_full": half_noninferiority_gates[
            "distance_reduction_noninferior_to_full"
        ],
        "full_eligible": arm_eligible["F"],
        "half_eligible": arm_eligible["H"],
        "deterministic_full_then_half_selection": True,
    }
    require(
        acceptance.get("aggregate_gates") == expected_aggregate_gates,
        "exp16 aggregate acceptance gates differ from the exact F-then-H decision",
    )
    run_configs = provenance.get("run_config_sha256")
    run_inputs = provenance.get("run_input_contract_sha256")
    require(
        isinstance(run_configs, dict)
        and set(run_configs) == set(run_names)
        and all(_sha(value) for value in run_configs.values()),
        "exp16 acceptance lacks the exact 40 run-config hashes",
    )
    require(
        isinstance(run_inputs, dict)
        and set(run_inputs) == set(run_names)
        and all(_sha(value) for value in run_inputs.values()),
        "exp16 acceptance lacks the exact 40 input-contract hashes",
    )
    selected_configs = artifact.get("selected_run_config_sha256")
    expected_selected_configs = {
        name: run_configs[name]
        for name in sorted(run_names)
        if arms_by_name[name] == selected_arm
    }
    require(
        isinstance(selected_configs, dict)
        and len(selected_configs) == 20
        and selected_configs == expected_selected_configs,
        "exp16 selected recipe does not bind the selected arm's exact 20 configs",
    )
    return {
        "campaign_id": contract["campaign_id"],
        "acceptance_path": str(acceptance_path.resolve()),
        "acceptance_file_sha256": acceptance_file_hash,
        "acceptance_report_sha256": acceptance_report_hash,
        "selected_recipe_path": str(selected_recipe_path.resolve()),
        "selected_recipe_file_sha256": file_sha256(selected_recipe_path),
        "selected_recipe_artifact_sha256": selected_recipe_artifact_hash,
        "accepted_status": acceptance["status"],
        "selected_arm": selected_arm,
        "selected_recipe": dict(recipe),
        "selected_recipe_sha256": recipe_hash,
        "selection_rule_sha256": artifact["selection_rule_sha256"],
        "manifest_sha256": artifact["manifest_sha256"],
        "package_protocol_sha256": artifact["package_protocol_sha256"],
        "source_sha256": artifact["source_sha256"],
        "runtime_sha256": artifact["runtime_sha256"],
        "actual_evaluation_bank_sha256": artifact["actual_evaluation_bank_sha256"],
        "exp15_prerequisite": dict(exp15_prerequisite),
        "exp15_prerequisite_sha256": exp15_prerequisite_hash,
        "run_config_sha256": dict(sorted(run_configs.items())),
        "run_input_contract_sha256": dict(sorted(run_inputs.items())),
        "selected_run_config_sha256": dict(sorted(selected_configs.items())),
    }


def build_binding(
    manifest: Mapping[str, Any],
    *,
    exp15_acceptance: Path,
    exp15_plan: Path,
    exp16_acceptance: Path,
    exp16_selected_recipe: Path,
) -> dict[str, Any]:
    exp15 = validate_exp15(manifest, exp15_acceptance, exp15_plan)
    exp16 = validate_exp16(manifest, exp16_acceptance, exp16_selected_recipe)
    require(
        exp16["exp15_prerequisite"] == exp15["exp16_prerequisite"]
        and exp16["exp15_prerequisite_sha256"]
        == exp15["exp16_prerequisite_sha256"],
        "exp16 did not consume the exact exp15 acceptance/launch-plan bytes bound by exp17",
    )
    binding: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "status": "sealed_accepted_prerequisites",
        "formal_submission_allowed": True,
        "selection_policy": "consume_exp16_deterministic_selection_without_formal_outcome_selection",
        "exp15": exp15,
        "exp16": exp16,
        "selected_arm": exp16["selected_arm"],
        "selected_recipe": exp16["selected_recipe"],
        "selected_recipe_sha256": exp16["selected_recipe_sha256"],
    }
    binding["binding_sha256"] = stable_hash(binding)
    return binding


def _atomic_text(path: Path, value: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CAMPAIGN_DIR / "manifest.json")
    parser.add_argument("--exp15-acceptance", type=Path)
    parser.add_argument("--exp15-plan", type=Path)
    parser.add_argument("--exp16-acceptance", type=Path)
    parser.add_argument("--exp16-selected-recipe", type=Path)
    parser.add_argument("--publish", action="store_true")
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    exp15 = manifest["prerequisites"]["exp15"]
    exp16 = manifest["prerequisites"]["exp16"]
    binding = build_binding(
        manifest,
        exp15_acceptance=args.exp15_acceptance or Path(exp15["acceptance_path"]),
        exp15_plan=args.exp15_plan or Path(exp15["launch_plan_path"]),
        exp16_acceptance=args.exp16_acceptance or Path(exp16["acceptance_path"]),
        exp16_selected_recipe=args.exp16_selected_recipe or Path(exp16["selected_recipe_path"]),
    )
    print(json.dumps(binding, sort_keys=True, indent=2))
    if not args.publish:
        print("dry-run only: prerequisite binding and protocol lock were not changed", file=sys.stderr)
        return 0
    existing = read_json(PREREQUISITE_BINDINGS_PATH)
    require(
        existing.get("status") == "unsealed_waiting_for_accepted_exp15_and_exp16"
        or existing == binding,
        "prerequisite binding was already sealed to different bytes",
    )
    atomic_json(PREREQUISITE_BINDINGS_PATH, binding)
    _atomic_text(PROTOCOL_LOCK_PATH, protocol_sha256(CAMPAIGN_DIR) + "\n")
    print(f"sealed {PREREQUISITE_BINDINGS_PATH} and refreshed {PROTOCOL_LOCK_PATH}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"formal prerequisite binding error: {exc}", file=sys.stderr)
        raise SystemExit(2)
