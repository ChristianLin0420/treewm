from __future__ import annotations

import copy
import json
import os
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
    assert "claim_policy" not in manifest
    assert "Submission authority requires" in manifest["submission_authority_policy"]
    assert "all four staged all-40 gates" in manifest["scientific_claim_policy"]


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


def test_launch7_negative_and_future_launch8_dependencies_are_explicit() -> None:
    manifest = campaign.load_manifest()
    dependency = manifest["launch_dependency"]
    negative = manifest["launch7_negative_dependency"]
    assert negative["binding_state"] == "sealed_authenticated_terminal_negative_no_reuse"
    assert negative["negative_binding_sha256"] == (
        "629610c2bb677f53ee3acb75a8bcd1e3089bee78a4c43600a944e4290f5148bd"
    )
    assert negative["accepted"] is negative["reusable"] is False
    assert dependency["campaign_id"] == "treewm-executable-prefix-repair-pilot-v1-launch8"
    assert dependency["required_status"] == "accepted_engineering_pilot"
    assert dependency["launch7_positive_authority_forbidden"] is True
    assert set(dependency["pilot_covered_settings"]) == campaign.PILOT_COVERED_SETTINGS
    assert set(dependency["additional_formal_settings_requiring_outcome_blind_audits"]) == campaign.ADDITIONAL_FORMAL_SETTINGS
    assert campaign.ADDITIONAL_FORMAL_SETTINGS == {
        "cube-double", "cube-triple", "antmaze-giant", "humanoidmaze-medium", "humanoidmaze-large"
    }
    status = campaign.prerequisite_binding_status(manifest)
    assert status["accepted"] is False
    assert status["status"] == "launch7_negative_authenticated_launch8_positive_unbound"
    with pytest.raises(campaign.ContractError, match="Launch8 semantic adapter"):
        campaign.assert_launch_authorized(manifest)


def test_scientific_protocol_remains_unsealed_and_positive_placeholder_is_exact(tmp_path: Path) -> None:
    manifest = campaign.load_manifest()
    assert manifest["scientific_protocol_sealed"] is False
    assert manifest["per_setting_contract_state"] == "unbound_pending_all_ten_audits"
    assert all(row["formal_contract_sha256"] is None for row in manifest["design"]["settings"])
    assert manifest["m2a_authority"]["formal_submission_allowed"] is False
    assert manifest["m2a_authority"]["execution_readiness_ready"] is False
    assert manifest["m2a_authority"]["launch7_terminal_negative_binding_state"] == (
        "sealed_authenticated_terminal_negative_no_reuse"
    )
    assert manifest["m2a_authority"]["accepted_engineering_pilot_binding_state"] == "unbound"
    forged = tmp_path / "binding.json"
    forged.write_text(
        json.dumps({"schema_version": 1, "status": "sealed_accepted_engineering_pilot"}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(campaign.ContractError, match="placeholder differs"):
        campaign.prerequisite_binding_status(manifest, forged)
    with pytest.raises(campaign.ContractError, match="placeholder differs"):
        campaign.assert_launch_authorized(manifest, forged)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("status", "accepted_engineering_pilot"),
        ("formal_submission_allowed", True),
        ("adapter_file_sha256", "a" * 64),
        ("adapter_runtime_file_sha256", "b" * 64),
        ("report_commit_file_sha256", "c" * 64),
        ("binding_sha256", "d" * 64),
    ],
)
def test_future_positive_placeholder_field_tampering_is_rejected(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    manifest = campaign.load_manifest()
    positive = json.loads(campaign.ACCEPTED_PILOT_BINDING_PATH.read_text())
    positive[field] = value
    path = tmp_path / "positive.json"
    path.write_text(json.dumps(positive) + "\n", encoding="utf-8")
    with pytest.raises(campaign.ContractError, match="placeholder differs"):
        campaign.prerequisite_binding_status(manifest, path)


def test_launch7_negative_binding_or_evidence_tampering_is_rejected(tmp_path: Path) -> None:
    manifest = campaign.load_manifest()
    negative = json.loads(campaign.LAUNCH7_NEGATIVE_BINDING_PATH.read_text())
    for mutate in (
        lambda row: row.update(status="awaiting_authenticated_terminal_negative_record"),
        lambda row: row.update(formal_submission_allowed=True),
        lambda row: row["evidence"].update(raw_file_sha256="0" * 64),
        lambda row: row["bound_semantics"].update(reporter_state="COMPLETED"),
        lambda row: row.update(negative_binding_sha256="f" * 64),
    ):
        changed = copy.deepcopy(negative)
        mutate(changed)
        path = tmp_path / "negative.json"
        path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
        with pytest.raises(campaign.ContractError):
            campaign.prerequisite_binding_status(manifest, negative_path=path)
    rehashed = copy.deepcopy(negative)
    rehashed["bound_semantics"]["reporter_state"] = "COMPLETED"
    body = {key: value for key, value in rehashed.items() if key != "negative_binding_sha256"}
    rehashed["negative_binding_sha256"] = campaign.stable_hash(body)
    with pytest.raises(campaign.ContractError, match="semantic binding differs"):
        campaign.validate_launch7_negative_binding(rehashed)

    evidence_target = tmp_path / campaign.LAUNCH7_NEGATIVE_EVIDENCE_RELATIVE
    evidence_target.parent.mkdir(parents=True)
    evidence = json.loads(
        (campaign.REPOSITORY_ROOT / campaign.LAUNCH7_NEGATIVE_EVIDENCE_RELATIVE).read_text()
    )
    evidence["terminal_zero_active_evidence"]["active_job_count"] = 1
    evidence_target.write_text(json.dumps(evidence) + "\n", encoding="utf-8")
    with pytest.raises(campaign.ContractError, match="bytes/canonical hash differ"):
        campaign.validate_launch7_negative_binding(negative, evidence_root=tmp_path)


def test_readiness_milestones_are_unique_acyclic_parallel_branches() -> None:
    milestones = campaign.load_manifest()["milestones"]
    ids = [row["id"] for row in milestones]
    assert len(ids) == len(set(ids))
    seen: set[str] = set()
    for row in milestones:
        assert set(row["depends_on"]) <= seen
        seen.add(row["id"])
    outcome_blind = {
        "m2a_runtime_and_interpreter",
        "m2b_shared_objective_config_authorization",
        "m2c_all_ten_outcome_blind_audits_contracts",
        "m2d_heldout_seed_and_feasibility",
        "m2e_scientific_adapters",
    }
    by_id = {row["id"]: row for row in milestones}
    assert all(
        "m3_launch8_positive_binding" not in by_id[milestone]["depends_on"]
        for milestone in outcome_blind
    )
    assert set(by_id["m4_protocol_and_execution_readiness_join"]["depends_on"]) >= (
        outcome_blind
        | {
            "m1_launch7_terminal_negative_binding",
            "m2f_launch8_versioned_adapter",
            "m3_launch8_positive_binding",
        }
    )


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
        "authenticated_launch7_terminal_negative_no_reuse_binding",
        "versioned_future_engineering_pilot_positive_adapter_after_protocol_freeze",
        "execution_ready_manifest_exact_schema_no_unvalidated_fields",
    } <= patterns
    assert manifest["required_exp23_hardening_port"]["runtime_files_present"] is True
    assert manifest["required_exp23_hardening_port"]["runtime_execution_ready"] is False
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
        lambda value: value["required_exp23_hardening_port"].update(runtime_execution_ready=True),
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
    assert report["status"] == "m2a_orders_0_3_runtime_authority_scaffold_execution_blocked"
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

    unisolated = subprocess.run(
        [sys.executable, str(package / "submit.py"), "--submit"],
        cwd=tmp_path,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert unisolated.returncode == 2
    assert "exact pinned Python 3.11 with -I -S -B" in unisolated.stderr
    assert not list(tmp_path.iterdir())

    without_bytecode_flag_environment = {
        key: value for key, value in os.environ.items() if key != "PYTHONDONTWRITEBYTECODE"
    }
    for mode in ("--snapshot-test", "--submit"):
        missing_b = subprocess.run(
            [sys.executable, "-I", "-S", str(package / "submit.py"), mode],
            cwd=tmp_path,
            env=without_bytecode_flag_environment,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert missing_b.returncode == 2
        assert "exact pinned Python 3.11 with -I -S -B" in missing_b.stderr
        assert not list(tmp_path.iterdir())
