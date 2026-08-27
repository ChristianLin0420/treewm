from __future__ import annotations

from pathlib import Path

import pytest

import bind_prerequisites
import campaign


def _accepted_exp15_pair(tmp_path: Path) -> tuple[Path, Path]:
    manifest = campaign.load_manifest()
    contract = manifest["prerequisites"]["exp15"]
    raw = contract["raw_report_recomputation"]
    records = []
    for setting_index, setting in enumerate(raw["settings"]):
        for arm_index, arm in enumerate(raw["arms"]):
            for seed_index, seed in enumerate(raw["seeds"]):
                index = ((setting_index * len(raw["arms"])) + arm_index) * len(
                    raw["seeds"]
                ) + seed_index
                records.append(
                    {
                        "index": index,
                        "setting_id": setting,
                        "arm_id": arm,
                        "seed": seed,
                        "run_name": f"repair-{setting}-arm{arm.lower()}-seed{seed}",
                        "integrity_gates": {
                            name: True for name in raw["required_integrity_gates"]
                        },
                        "integrity_pass": True,
                        "scientific_gates": {
                            name: True for name in raw["required_scientific_gates"]
                        },
                        "scientific_pass": True,
                        "metrics": {
                            "final": {
                                "num_episodes": 25.0,
                                "successes": 1.0,
                                "success_rate": 0.04,
                                "distance_reduction_frac": 0.1,
                            }
                        },
                        "error": None,
                    }
                )
    paired = [
        {
            "setting_id": setting,
            "seed": seed,
            "success_delta_candidate_minus_control": 0.0,
            "distance_reduction_delta_candidate_minus_control": 0.0,
        }
        for setting in raw["settings"]
        for seed in raw["seeds"]
    ]
    gates = {name: True for name in contract["required_aggregate_gates"]}
    report = {
        "schema_version": 1,
        "campaign_id": contract["campaign_id"],
        "status": contract["required_status"],
        "accepted": True,
        "formal_validation": False,
        "preregistered_candidate_arm": contract["required_candidate_arm"],
        "matched_control_arm": contract["required_control_arm"],
        "sensitivity_arms_are_nonpromotable": raw[
            "nonpromotable_sensitivity_arms"
        ],
        "integrity_runs_passing": 40,
        "candidate_setting_pass": {setting: True for setting in raw["settings"]},
        "candidate_settings_passing": len(raw["settings"]),
        "aggregate_gates": gates,
        "aggregate_metrics": {
            "candidate_total_successes": 10.0,
            "candidate_mean_distance_reduction": 0.1,
            "candidate_runs_with_positive_progress": 10,
            "paired_mean_success_delta_candidate_minus_control": 0.0,
            "paired_mean_distance_reduction_delta_candidate_minus_control": 0.0,
        },
        "paired_comparisons": paired,
        "missing_or_extra_keys": [],
        "actual_evaluation_bank_sha256": "5" * 64,
        "runs": records,
    }
    acceptance_path = tmp_path / "exp15-acceptance.json"
    campaign.atomic_json(acceptance_path, report)
    common_hashes = {
        "package_protocol_sha256": contract["package_protocol_sha256"],
        "source_sha256": contract["source_sha256"],
        "runtime_sha256": contract["runtime_sha256"],
        "actual_evaluation_bank_sha256": report[
            "actual_evaluation_bank_sha256"
        ],
    }
    plan = {
        "schema_version": 1,
        "campaign_id": contract["campaign_id"],
        "status": "sealed_bounded_repair_pilot_plan",
        "formal_validation": False,
        "common_hashes": common_hashes,
        "runs": [
            {
                "setting_id": record["setting_id"],
                "arm_id": record["arm_id"],
                "seed": record["seed"],
            }
            for record in records
        ],
    }
    plan["plan_sha256"] = campaign.stable_hash(plan)
    plan_path = tmp_path / "exp15-launch-plan.json"
    campaign.atomic_json(plan_path, plan)
    return acceptance_path, plan_path


def test_exp15_binding_recomputes_complete_raw_report(tmp_path: Path) -> None:
    acceptance_path, plan_path = _accepted_exp15_pair(tmp_path)
    result = bind_prerequisites.validate_exp15(
        campaign.load_manifest(), acceptance_path, plan_path
    )
    assert result["accepted_status"] == "accepted_for_fresh_formal_campaign_design"
    assert result["candidate_arm"] == "C"


def test_exp15_binding_rejects_forged_per_run_scientific_pass(tmp_path: Path) -> None:
    acceptance_path, plan_path = _accepted_exp15_pair(tmp_path)
    report = campaign.read_json(acceptance_path)
    record = next(row for row in report["runs"] if row["arm_id"] == "C")
    record["scientific_gates"]["q_advantage"] = False
    campaign.atomic_json(acceptance_path, report)
    with pytest.raises(campaign.ContractError, match="scientific_pass differs from raw gates"):
        bind_prerequisites.validate_exp15(
            campaign.load_manifest(), acceptance_path, plan_path
        )


def test_exp15_binding_rejects_forged_candidate_quorum_claims(tmp_path: Path) -> None:
    acceptance_path, plan_path = _accepted_exp15_pair(tmp_path)
    report = campaign.read_json(acceptance_path)
    candidate_rows = [row for row in report["runs"] if row["arm_id"] == "C"]
    for record in (candidate_rows[0], candidate_rows[2]):
        record["scientific_gates"]["q_advantage"] = False
        record["scientific_pass"] = False
    campaign.atomic_json(acceptance_path, report)
    with pytest.raises(campaign.ContractError, match="candidate setting quorum differs"):
        bind_prerequisites.validate_exp15(
            campaign.load_manifest(), acceptance_path, plan_path
        )


def test_exp15_binding_rejects_forged_outcome_gates(tmp_path: Path) -> None:
    acceptance_path, plan_path = _accepted_exp15_pair(tmp_path)
    report = campaign.read_json(acceptance_path)
    for record in report["runs"]:
        if record["arm_id"] == "C":
            record["metrics"]["final"]["successes"] = 0.0
            record["metrics"]["final"]["success_rate"] = 0.0
    campaign.atomic_json(acceptance_path, report)
    with pytest.raises(campaign.ContractError, match="aggregate gates differ from raw runs"):
        bind_prerequisites.validate_exp15(
            campaign.load_manifest(), acceptance_path, plan_path
        )


def test_exp15_binding_rejects_forged_integrity_pass(tmp_path: Path) -> None:
    acceptance_path, plan_path = _accepted_exp15_pair(tmp_path)
    report = campaign.read_json(acceptance_path)
    report["runs"][0]["integrity_gates"][
        "exact_launch_completion_and_provenance"
    ] = False
    campaign.atomic_json(acceptance_path, report)
    with pytest.raises(campaign.ContractError, match="integrity_pass differs from raw gates"):
        bind_prerequisites.validate_exp15(
            campaign.load_manifest(), acceptance_path, plan_path
        )


def _accepted_exp16_pair(
    tmp_path: Path,
    *,
    failed_full_scientific_keys: set[tuple[str, int]] | None = None,
) -> tuple[Path, Path]:
    manifest = campaign.load_manifest()
    selected_arm = "F"
    common = manifest["method"]["grounded_multistep"]
    recipe = {
        "id": selected_arm,
        "label": "grounded-conservative-full-loss-scale-lr3e-5",
        "world_lr": 3e-5,
        "transition_mode": common["transition_mode"],
        "grounded_select_action_weight": common["selector_weights"]["action"],
        "grounded_select_endpoint_weight": common["selector_weights"]["endpoint"],
        "grounded_select_horizon_weight": common["selector_weights"]["horizon"],
        "grounded_loss_scale": 1.0,
        **manifest["prerequisites"]["allowed_selected_recipes"][selected_arm],
    }
    records = []
    run_configs = {}
    run_inputs = {}
    for setting in campaign.SETTING_IDS:
        for arm in ("F", "H"):
            for seed in (102, 103):
                name = f"bridge-{setting}-arm{arm.lower()}-seed{seed}"
                scientific_pass = not (
                    arm == "F"
                    and (setting, seed) in (failed_full_scientific_keys or set())
                )
                records.append(
                    {
                        "run_name": name,
                        "setting_id": setting,
                        "arm_id": arm,
                        "seed": seed,
                        "integrity_pass": True,
                        "scientific_pass": scientific_pass,
                        "metrics": {
                            "final": {
                                "successes": 1.0,
                                "success_rate": 0.2,
                                "distance_reduction_frac": 0.1,
                            }
                        },
                    }
                )
                run_configs[name] = campaign.stable_hash({"config": name})
                run_inputs[name] = campaign.stable_hash({"input": setting})
    exp15_prerequisite = {
        "schema_version": 1,
        "status": "accepted_exp15_prerequisite",
        "campaign_id": "treewm-grounded-repair-pilot-v1",
        "acceptance_sha256": "7" * 64,
        "launch_plan_sha256": "8" * 64,
    }
    exp15_prerequisite["prerequisite_sha256"] = campaign.stable_hash(
        exp15_prerequisite
    )
    common_hashes = {
        "manifest_sha256": manifest["prerequisites"]["exp16"]["manifest_sha256"],
        "package_protocol_sha256": manifest["prerequisites"]["exp16"]["package_protocol_sha256"],
        "source_sha256": manifest["prerequisites"]["exp16"]["source_sha256"],
        "runtime_sha256": manifest["prerequisites"]["exp16"]["runtime_sha256"],
        "actual_evaluation_bank_sha256": "5" * 64,
        "exp15_prerequisite_sha256": exp15_prerequisite["prerequisite_sha256"],
    }
    selection_rule = "6" * 64
    keyed = {
        (record["setting_id"], record["arm_id"], record["seed"]): record
        for record in records
    }
    setting_counts = {
        arm: {
            setting: sum(
                keyed[(setting, arm, seed)]["scientific_pass"]
                for seed in (102, 103)
            )
            for setting in campaign.SETTING_IDS
        }
        for arm in ("F", "H")
    }
    scientific_counts = {
        arm: sum(setting_counts[arm].values()) for arm in ("F", "H")
    }
    setting_pass = {
        arm: {
            setting: count == 2 for setting, count in setting_counts[arm].items()
        }
        for arm in ("F", "H")
    }
    scientific_quorum = {
        arm: {
            "scientific_runs_at_least_18_of_20": scientific_counts[arm] >= 18,
            "both_seed_settings_at_least_8_of_10": sum(
                setting_pass[arm].values()
            )
            >= 8,
            "every_setting_has_at_least_one_passing_seed": all(
                count >= 1 for count in setting_counts[arm].values()
            ),
        }
        for arm in ("F", "H")
    }
    outcome_gates = {
        arm: {
            "finite_terminal_outcomes": True,
            "not_all_zero_success": True,
            "positive_mean_progress": True,
            "positive_progress_run_quorum": True,
        }
        for arm in ("F", "H")
    }
    half_noninferiority = {
        "success_noninferior_to_full": True,
        "distance_reduction_noninferior_to_full": True,
    }
    acceptance = {
        "schema_version": 1,
        "campaign_id": manifest["prerequisites"]["exp16"]["campaign_id"],
        "status": "selected_full_for_fresh_formal_campaign_design",
        "accepted": True,
        "formal_validation": False,
        "selection_precedence": ["F", "H"],
        "selected_arm": selected_arm,
        "selected_recipe": recipe,
        "selected_recipe_sha256": campaign.stable_hash(recipe),
        "selection_rule_sha256": selection_rule,
        "exp15_prerequisite": exp15_prerequisite,
        "exp15_prerequisite_sha256": exp15_prerequisite["prerequisite_sha256"],
        "integrity_runs_passing": 40,
        "arm_scientific_runs_passing": scientific_counts,
        "arm_setting_seed_pass_count": setting_counts,
        "arm_setting_pass": setting_pass,
        "arm_scientific_quorum_gates": scientific_quorum,
        "arm_outcome_gates": outcome_gates,
        "half_noninferiority_gates": half_noninferiority,
        "aggregate_gates": {
            "integrity_40_of_40": True,
            "full_scientific_quorum": True,
            "half_scientific_quorum": True,
            "full_outcome_gates": True,
            "half_outcome_gates": True,
            "half_success_noninferior_to_full": True,
            "half_distance_reduction_noninferior_to_full": True,
            "full_eligible": True,
            "half_eligible": True,
            "deterministic_full_then_half_selection": True,
        },
        "actual_evaluation_bank_sha256": common_hashes["actual_evaluation_bank_sha256"],
        "runs": records,
        "provenance": {
            **common_hashes,
            "exp15_prerequisite": exp15_prerequisite,
            "run_config_sha256": run_configs,
            "run_input_contract_sha256": run_inputs,
        },
    }
    acceptance["report_sha256"] = campaign.stable_hash(acceptance)
    acceptance_path = tmp_path / "acceptance.json"
    campaign.atomic_json(acceptance_path, acceptance)

    selected_configs = {
        name: value
        for name, value in sorted(run_configs.items())
        if "-armf-" in name
    }
    artifact = {
        "schema_version": 1,
        "campaign_id": acceptance["campaign_id"],
        "status": "selected_recipe",
        "selected": True,
        "formal_validation": False,
        "selected_arm": selected_arm,
        "selected_recipe": recipe,
        "selected_recipe_sha256": acceptance["selected_recipe_sha256"],
        "selection_rule_sha256": selection_rule,
        "bridge_acceptance_sha256": acceptance["report_sha256"],
        "acceptance_sha256": campaign.file_sha256(acceptance_path),
        "exp15_prerequisite": exp15_prerequisite,
        **common_hashes,
        "selected_run_config_sha256": selected_configs,
    }
    artifact["artifact_sha256"] = campaign.stable_hash(artifact)
    artifact_path = tmp_path / "SELECTED_RECIPE.json"
    campaign.atomic_json(artifact_path, artifact)
    return acceptance_path, artifact_path


def test_exp16_binding_verifies_exact_and_canonical_hashes_and_plural_maps(tmp_path: Path):
    acceptance_path, artifact_path = _accepted_exp16_pair(tmp_path)
    result = bind_prerequisites.validate_exp16(
        campaign.load_manifest(), acceptance_path, artifact_path
    )
    assert result["acceptance_file_sha256"] == campaign.file_sha256(acceptance_path)
    assert result["selected_recipe_artifact_sha256"]
    assert len(result["run_config_sha256"]) == 40
    assert len(result["run_input_contract_sha256"]) == 40
    assert len(result["selected_run_config_sha256"]) == 20


def test_exp16_binding_rejects_selected_config_subset_tamper(tmp_path: Path):
    acceptance_path, artifact_path = _accepted_exp16_pair(tmp_path)
    artifact = campaign.read_json(artifact_path)
    artifact["selected_run_config_sha256"].pop(next(iter(artifact["selected_run_config_sha256"])))
    artifact.pop("artifact_sha256")
    artifact["artifact_sha256"] = campaign.stable_hash(artifact)
    campaign.atomic_json(artifact_path, artifact)
    with pytest.raises(campaign.ContractError, match="exact 20 configs"):
        bind_prerequisites.validate_exp16(
            campaign.load_manifest(), acceptance_path, artifact_path
        )


def test_exp16_binding_accepts_exact_18_of_20_cross_setting_quorum(tmp_path: Path):
    acceptance_path, artifact_path = _accepted_exp16_pair(
        tmp_path,
        failed_full_scientific_keys={
            (campaign.SETTING_IDS[0], 102),
            (campaign.SETTING_IDS[1], 102),
        },
    )
    result = bind_prerequisites.validate_exp16(
        campaign.load_manifest(), acceptance_path, artifact_path
    )
    assert result["selected_arm"] == "F"
