from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

import campaign


def test_exact_selected_gsep_formal_matrix() -> None:
    manifest = campaign.load_manifest()
    runs = campaign.expand_runs(manifest)
    assert len(runs) == 40
    assert {run.arm for run in runs} == {"GSEP"}
    assert [run.seed for run in runs[:4]] == [240, 241, 242, 243]
    assert [run.setting_id for run in runs[::4]] == [row[0] for row in campaign.SETTING_SPECS]
    assert len({run.run_name for run in runs}) == len({run.wandb_id for run in runs}) == 40
    assert not set(campaign.SEEDS) & (set(range(4)) | set(range(100, 112)) | set(range(220, 224)))
    assert manifest["design"]["seed_assignment_census"]["prior_training_assignment_matches"] == 0
    assert manifest["design"]["scratch_only"] is True
    assert manifest["design"]["reuse_upstream_checkpoints"] is False
    assert manifest["design"]["reuse_upstream_outputs"] is False
    assert manifest["design"]["reuse_upstream_wandb"] is False


def test_exact_selected_prefix_weights_and_distinct_formal_identity() -> None:
    manifest = campaign.load_manifest()
    assert manifest["objective"]["weights"] == {
        "action": 0.033368419,
        "latent": 0.027645085,
        "endpoint": 0.011350645,
    }
    assert manifest["objective"]["optimizer_updates"] == 1_000_000
    assert manifest["objective"]["scheduler_total_steps"] == 1_000_000
    assert manifest["objective"]["objective_version"] == "treewm_v2_grounded_executable_prefix_formal_v1"
    assert manifest["objective"]["objective_version"] != "treewm_v2_grounded_executable_prefix_pilot_v1"
    assert manifest["campaign_id"] != campaign.UPSTREAM_CAMPAIGN_ID
    assert campaign.UPSTREAM_CAMPAIGN_ID not in manifest["paths"]["run_root"]


def test_all_40_staged_afterok_dag_is_exact() -> None:
    manifest = campaign.load_manifest()
    dag = campaign.scheduler_dag(manifest)
    assert [(row.name, row.kind, row.elements, row.dependency, row.dependency_type) for row in dag] == [
        ("train_2000", "array", 40, None, None),
        ("gate_2000", "all_40_gate", 1, "train_2000", "afterok"),
        ("train_25000", "array", 40, "gate_2000", "afterok"),
        ("gate_25000", "all_40_gate", 1, "train_25000", "afterok"),
        ("train_100000", "array", 40, "gate_25000", "afterok"),
        ("gate_100000", "all_40_gate", 1, "train_100000", "afterok"),
        ("train_1000000", "array", 40, "gate_100000", "afterok"),
        ("gate_1000000", "all_40_gate", 1, "train_1000000", "afterok"),
        ("heldout_eval", "array", 200, "gate_1000000", "afterok"),
        ("aggregate", "seed_level_paired_t", 1, "heldout_eval", "afterok"),
        ("formal_report", "immutable_report", 1, "aggregate", "afterok"),
    ]


def test_heldout_matrix_and_primary_inference_are_exact() -> None:
    manifest = campaign.load_manifest()
    rows = campaign.expand_final_evaluations(manifest)
    assert len(rows) == 200
    assert [(row.index, row.training_index, row.task_id) for row in rows[:6]] == [
        (0, 0, 1), (1, 0, 2), (2, 0, 3), (3, 0, 4), (4, 0, 5), (5, 1, 1)
    ]
    assert campaign.eval_at(manifest, 199).training_index == 39
    assert campaign.eval_at(manifest, 199).task_id == 5
    final = manifest["final_evaluation"]
    assert final["rails"] == ["learned", "bfs"]
    assert final["episodes_per_cell_per_rail"] == 50
    assert final["same_ordered_episode_seeds_across_rails"] is True
    assert final["adaptive_selection"] is False
    assert final["primary_inference"] == {
        "unit": "training_seed",
        "replicates": 4,
        "cells_per_seed": 50,
        "episodes_per_seed_per_rail": 2500,
        "paired_quantity": "learned_minus_bfs_macro_success_rate",
        "two_sided_confidence_level": 0.95,
        "degrees_of_freedom": 3,
        "t_critical": 3.182446,
        "seed_replicate_aggregation": "unweighted_mean_of_exactly_50_cell_success_rates",
        "missingness_or_cell_selection_allowed": False,
        "raw_episode_success_accounting_must_match_cell_summary": True,
        "rail_episode_seed_order_must_match_exactly": True,
    }
    summary = campaign.paired_t_summary([0.0, 0.0, 0.0, 0.0])
    assert summary == {
        "n": 4,
        "degrees_of_freedom": 3,
        "mean": 0.0,
        "standard_error": 0.0,
        "ci95_lower": 0.0,
        "ci95_upper": 0.0,
    }


def test_launch7_dependency_and_all_ten_gap_are_explicit() -> None:
    manifest = campaign.load_manifest()
    dependency = manifest["launch_dependency"]
    assert dependency["required_status"] == "accepted_engineering_pilot"
    assert set(dependency["pilot_covered_settings"]) == campaign.PILOT_COVERED_SETTINGS
    assert set(dependency["additional_formal_settings_requiring_outcome_blind_audits"]) == campaign.ADDITIONAL_FORMAL_SETTINGS
    assert campaign.ADDITIONAL_FORMAL_SETTINGS == {
        "cube-double", "cube-triple", "antmaze-giant", "humanoidmaze-medium", "humanoidmaze-large"
    }
    status = campaign.upstream_binding_status(manifest)
    assert status["accepted"] is False
    assert status["status"] == "awaiting_launch7_accepted_engineering_pilot"
    with pytest.raises(campaign.ContractError, match="blocked until Launch7"):
        campaign.assert_launch_authorized(manifest)


def test_scientific_protocol_remains_unsealed_and_accepted_looking_binding_is_inert(tmp_path: Path) -> None:
    manifest = campaign.load_manifest()
    assert manifest["scientific_protocol_sealed"] is False
    assert manifest["per_setting_contract_state"] == "unbound_pending_all_ten_audits"
    assert all(row["formal_contract_sha256"] is None for row in manifest["design"]["settings"])
    forged = tmp_path / "binding.json"
    forged.write_text(
        json.dumps({"schema_version": 1, "status": "sealed_accepted_engineering_pilot"}) + "\n",
        encoding="utf-8",
    )
    status = campaign.upstream_binding_status(manifest, forged)
    assert status["accepted"] is False
    assert status["status"] == "accepted_looking_binding_unusable_in_design_phase"
    with pytest.raises(campaign.ContractError, match="hardened Launch7 report binder"):
        campaign.assert_launch_authorized(manifest, forged)


def test_threshold_operators_weight_safety_and_rail_parity_are_frozen() -> None:
    manifest = campaign.load_manifest()
    stage_25k = manifest["stage_acceptance"]["stage_25000"]
    assert stage_25k["operator_semantics"]["q_advantage"] == "strictly_greater_than_zero"
    assert stage_25k["operator_semantics"]["horizon_cross_entropy"] == "strictly_less_than_both_ln5_and_empirical_prior_entropy"
    assert stage_25k["operator_semantics"]["ratio_intervals"] == "closed_inclusive"
    assert manifest["stage_acceptance"]["stage_100000"]["fleet_mean_distance_reduction_strictly_greater_than"] == 0.0

    safety = manifest["fixed_weight_safety_contract"]
    assert safety["expected_rows"] == 80
    assert safety["per_component_median_base_gradient_fraction_max"] == 0.03
    assert safety["aggregate_every_row_group_base_gradient_fraction_max"] == 0.10
    assert safety["optimizer_steps"] == 0
    assert safety["outcomes_or_rollouts_read"] is False
    assert safety["retuning_allowed"] is False

    parity = manifest["final_evaluation"]["rail_parity_contract"]
    assert parity["only_allowed_difference"] == "tree_config.scorer=learned|bfs"
    assert parity["same_exact_1m_checkpoint"] is True
    assert parity["rail_specific_tuning"] is False
    assert parity["order_dependent_rng"] is False
    setting = manifest["final_evaluation"]["setting_noninferiority"]
    assert setting["denominator_settings"] == 10
    assert setting["required_noninferior_settings"] == 7
    assert setting["per_setting_episodes_per_rail"] == 1000


def test_hardening_port_contract_closes_known_exp22_gaps() -> None:
    manifest = campaign.load_manifest()
    patterns = set(manifest["required_exp23_hardening_port"]["required_patterns"])
    assert {
        "exclusive_submission_claim_and_append_only_journals",
        "scheduler_control_plane_observation_binding",
        "exact_id_submit_reconciliation_rollback_and_cancel",
        "whole_array_afterok_dependency_reconciliation",
        "strict_append_only_scalar_identity_and_conflict_rejection",
        "identical_duplicate_inventory_or_upstream_suppression",
        "dense_50_update_gain_and_support_axis",
        "authenticated_wandb_leaf_symlink_policy",
        "immutable_report_triplet_and_commit",
        "formal_objective_registered_in_v2_gauge_prefix_formal_staged_authorization_sets",
        "hardened_launch7_report_quartet_binder",
        "execution_ready_manifest_exact_schema_no_unvalidated_fields",
    } <= patterns
    assert manifest["required_exp23_hardening_port"]["runtime_files_present"] is False
    assert manifest["required_exp23_hardening_port"]["protocol_lock_present"] is False


def test_manifest_mutations_fail_closed() -> None:
    manifest = campaign.load_manifest()
    mutations = (
        lambda value: value["design"].update(selected_arm="GS"),
        lambda value: value["design"].update(seeds=[0, 1, 2, 3]),
        lambda value: value["design"].update(reuse_upstream_checkpoints=True),
        lambda value: value["objective"]["weights"].update(action=0.5),
        lambda value: value["lifecycle"].update(stage_targets=[2_000, 25_000, 1_000_000]),
        lambda value: value["lifecycle"].update(all_40_required_at_every_gate=False),
        lambda value: value["final_evaluation"].update(episodes_per_cell_per_rail=5),
        lambda value: value["final_evaluation"]["primary_inference"].update(unit="episode"),
        lambda value: value["launch_dependency"].update(formal_submission_allowed=True),
        lambda value: value["required_exp23_hardening_port"].update(runtime_files_present=True),
    )
    for mutate in mutations:
        changed = copy.deepcopy(manifest)
        mutate(changed)
        with pytest.raises(campaign.ContractError):
            campaign.validate_manifest(changed)


def test_submit_test_only_is_read_only_and_submit_fails_closed(tmp_path: Path) -> None:
    package = campaign.PACKAGE_DIR
    before = sorted((path.relative_to(package), path.stat().st_mtime_ns, path.stat().st_size) for path in package.rglob("*") if path.is_file())
    command = [sys.executable, "-I", "-S", "-B", str(package / "submit.py"), "--test-only"]
    result = subprocess.run(command, cwd=tmp_path, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    report = json.loads(result.stdout)
    assert report["status"] == "geometry_and_global_recipe_locked_execution_blocked"
    assert report["submitted"] is False
    assert report["persistent_writes_performed"] is False
    assert report["scheduler_calls"] == []
    assert report["snapshot_created"] is False
    assert report["training_runs"] == 40 and report["final_eval_cells"] == 200
    assert not list(tmp_path.iterdir())
    after = sorted((path.relative_to(package), path.stat().st_mtime_ns, path.stat().st_size) for path in package.rglob("*") if path.is_file())
    assert after == before

    rejected = subprocess.run(
        [sys.executable, "-I", "-S", "-B", str(package / "submit.py"), "--submit"],
        cwd=tmp_path,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert rejected.returncode == 2
    assert "EXP24_BLOCKED" in rejected.stderr
    assert not list(tmp_path.iterdir())
