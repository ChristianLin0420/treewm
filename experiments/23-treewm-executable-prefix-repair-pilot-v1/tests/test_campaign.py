from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
SPEC = importlib.util.spec_from_file_location("exp23_campaign", PACKAGE / "campaign.py")
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


def contracts():
    manifest = campaign.read_json(PACKAGE / "manifest.json")
    lock = campaign.read_json(PACKAGE / "weight_audit.lock.json")
    return manifest, lock


def test_full_static_contract_and_matrix():
    manifest, lock = contracts()
    campaign.validate_manifest(manifest, lock, REPO)
    cells = campaign.expand_matrix(manifest)
    assert len(cells) == 20
    assert cells[0].index == 0
    assert (cells[0].setting, cells[0].arm, cells[0].seed) == (
        "antmaze-large", "GS", 110
    )
    assert (cells[-1].setting, cells[-1].arm, cells[-1].seed) == (
        "cube-quadruple-100m", "GSEP", 111
    )
    assert [cell.index for cell in cells] == list(range(20))
    assert manifest["campaign_id"] == campaign.CAMPAIGN_ID
    assert all(cell.run_name.startswith("exp23-launch4-") for cell in cells)


def test_launch4_namespace_and_ordered_superseded_aborts_are_exact():
    manifest, lock = contracts()
    superseded = manifest["superseded_launches"]
    assert superseded == campaign.SUPERSEDED_LAUNCHES
    assert [row["campaign_id"] for row in superseded] == [
        "treewm-executable-prefix-repair-pilot-v1",
        "treewm-executable-prefix-repair-pilot-v1-launch2",
        "treewm-executable-prefix-repair-pilot-v1-launch3",
    ]
    launch1, launch2, launch3 = superseded
    assert launch1["source_commit_claimed_by_journal"] is False
    assert launch1["snapshot"] == {
        "inventory_sha256": "6767520819d42ef8866712023211b2f1bc8d236db3ffc836c8dae429b4e5b326",
        "file_count": 137,
        "independently_matched_files": 137,
        "all_files_match": True,
    }
    assert launch2["source_commit"] == "0fd89949a092bd9bbf12b16e3efb058850d50c86"
    assert launch2["package_protocol_sha256"] == (
        "6472ca50fcbc1eaa35c4388876bf627f0f2c03d8310fd05e336f0204c0f49516"
    )
    assert launch2["manifest_raw_sha256"] == (
        "44911238bb06b10b46abbf58a8fe33019e0c107a6d760f8954a9a416382be776"
    )
    assert launch2["manifest_canonical_sha256"] == (
        "d124566d5834a62028f7416756c7cca36e6e63ae256e72d8c7c788412d558b00"
    )
    assert launch2["snapshot"] == {
        "inventory_sha256": "4aef86836e7fb683ace18cdd7588fd6b3904bdb9877e5c9a08146e76e49e2a76",
        "file_count": 137,
        "independently_matched_files": 137,
        "all_files_match": True,
    }
    assert launch2["claim_token"] == (
        "a2ea8575200dc47b4e3de67863a0f429d2397ae35fbbd2e7948e9322ffb64802"
    )
    assert launch2["journal_sha256"] == {
        "0000_CLAIMED.json": "e353f69fb6a397d1095f3f5b81ca717a5887b6d4d55fa0961ca38faa3460b6dc",
        "0001_SNAPSHOT_SEALED.json": "be9a29c112f56dc0ace53847b99e3892ab964ee410ca2ecfc2a0fd9d39179bdc",
        "9998_OUTER_ABORTED.json": "48eed85ce12306a35927e3ac8b539be28dc52e107465fac9f5546c378c99cb99",
    }
    assert launch3["status"] == "aborted_after_contract_before_scheduler_submission"
    assert launch3["source_commit"] == "ca979a2b0329d6775793cd8ce51d57a9200e6b8a"
    assert launch3["source_commit_claimed_by_contract"] is True
    assert launch3["source_commit_claimed_by_journal"] is False
    assert launch3["package_protocol_sha256"] == (
        "6178ed54273d13c88fce750414131c98d002b394231618f11f6d8d6a1a3fb49a"
    )
    assert launch3["manifest_raw_sha256"] == (
        "92a9e17b78075805503907e4d9b71b732b54f5127da981ca17524feba650f74a"
    )
    assert launch3["manifest_canonical_sha256"] == (
        "7259a76924ac6f4566541a00f74923c67151619b36411048dd82ee20747a8d09"
    )
    assert launch3["snapshot"] == {
        "inventory_sha256": "7e237d31f9d49e3b55d0e0598c299b6064ab16b9caa77b62329c9bb8a2839eae",
        "file_count": 137,
        "independently_matched_files": 137,
        "all_files_match": True,
    }
    assert launch3["preserved_tree"] == {
        "regular_file_count": 163,
        "symlink_count": 0,
        "snapshot_file_count": 137,
        "launch_file_count": 20,
        "contract_file_count": 1,
        "journal_file_count": 5,
    }
    assert launch3["claim_token"] == (
        "2741418c7e528a0b64b8115cafa46cfac391abd53312cf29b0ffc8a4e1afca4d"
    )
    assert launch3["contract_sha256"] == launch3["submission_sha256"] == (
        "0cd594c8a49499b5e3d10a09ddbf3b89f981264be67bb603dc64836568a1b4c2"
    )
    assert launch3["journal_sha256"] == {
        "0000_CLAIMED.json": "25d607cef5aaf49e932e86a56c2272d37be6936010f089f8e7230bb44166be28",
        "0001_SNAPSHOT_SEALED.json": "d4acf9fb3fa6af98adedf45abd0269d1e491abd48b3cb992509b7641ad05b4c6",
        "0002_CONTRACT_SEALED.json": "3936a8717d096a93cf7d0eb3dea5293e32c14cdd0b986dd7a738f9501a92a044",
        "9999_ABORTED.json": "915dd5a3869a6784f3eb9d8e0d564a16b96fb17689c3d31f5a2e21431365199a",
        "9998_OUTER_ABORTED.json": "e65e9b39d268c2497e98e1d66b0cadd7f80b0f53c98a243057cc03ca399b47b0",
    }
    assert launch3["failure_phase"] == "first_sanitized_squeue_before_any_sbatch"
    assert launch3["actual_sbatch_calls"] == 0
    assert launch3["known_job_ids"] == []
    assert launch3["job_ids_by_role"] == {"train": [], "report": []}
    assert launch3["submission_contract_committed"] is True
    assert launch3["no_job_proof"]["source_order_at_commit"] == {
        "contract_sealed_first": True,
        "train_absence_check_line": 3112,
        "report_absence_check_line": 3113,
        "first_sbatch_call_line": 3114,
        "recorded_failure": "first_train_absence_check",
    }
    observation = launch3["no_job_proof"]["current_scheduler_observation"]
    assert observation["environment"] == {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "SLURM_CONF": "/cm/shared/apps/slurm/var/etc/cs-oci-ord/slurm.conf",
    }
    assert observation["history_start_utc"] == "2026-08-01"
    assert observation["squeue_matching_rows"] == 0
    assert observation["sacct_matching_rows"] == 0
    for row in superseded[:2]:
        assert row["submission_sha256"] is None
        assert row["known_job_ids"] == []
    for row in superseded:
        for key in (
            "submission_receipt_committed",
            "scientific_run_started",
            "checkpoint_created",
            "wandb_run_created",
            "results_consumed",
            "checkpoints_consumed",
            "reuse_allowed",
            "resume_allowed",
            "retry_allowed",
            "recovery_allowed",
        ):
            assert row[key] is False
        assert row["optimizer_updates"] == 0
    assert launch2["submission_contract_committed"] is False
    assert manifest["paths"]["run_root"].endswith(
        "/outputs/treewm-executable-prefix-repair-pilot-v1-launch4"
    )
    assert manifest["paths"]["transaction_lock"] == (
        "outputs/.exp23-d3765ecc9f5b5f7a.transaction.lock"
    )
    assert all(manifest["paths"]["run_root"] != row["run_root"] for row in superseded)
    assert manifest["logging"]["wandb_project"].endswith("-launch4")
    assert manifest["logging"]["wandb_group"].endswith("-launch4")

    tampered = copy.deepcopy(manifest)
    tampered["paths"]["run_root"] = launch2["run_root"]
    with pytest.raises(campaign.ContractError):
        campaign.validate_manifest(tampered, lock, REPO)
    tampered = copy.deepcopy(manifest)
    tampered["superseded_launches"][1]["resume_allowed"] = True
    with pytest.raises(campaign.ContractError):
        campaign.validate_manifest(tampered, lock, REPO)


def test_each_matched_pair_differs_only_in_three_audited_weights():
    manifest, lock = contracts()
    cells = campaign.expand_matrix(manifest)
    for setting in campaign.SETTINGS:
        for seed in campaign.SEEDS:
            pair = [cell for cell in cells if cell.setting == setting and cell.seed == seed]
            left, right = [campaign.cell_overrides(cell, manifest, lock) for cell in pair]
            differing = {key for key in left if left[key] != right[key]}
            assert differing == set(campaign.WEIGHT_KEYS)
            for key in campaign.WEIGHT_KEYS:
                assert left[key] == 0.0
                assert right[key] > 0.0
            for name in campaign.PREFIX_TERMS:
                assert left[f"losses.enabled.{name}"] is True
                assert right[f"losses.enabled.{name}"] is True


def test_audited_tuple_and_fail_closed_tree_binding_are_exact():
    manifest, lock = contracts()
    assert lock["derived"]["weights"] == {
        "executable_prefix_action": 0.033368419,
        "executable_prefix_latent": 0.027645085,
        "executable_prefix_endpoint": 0.011350645,
    }
    assert lock["derived"]["post_scale_max_aggregate_ratio"] < 0.10
    assert len(lock["checkpoint_sha256"]) == 10
    assert len(lock["batch_sha256"]) == 20
    api = lock["fail_closed_api_binding"]
    assert api["effective_tree_config_required"] is True
    assert api["audit_call"] == "tree_config_for(cfg.arm, cfg_utils.tree_config(cfg), model)"
    source = (PACKAGE / "weight_audit.py").read_text(encoding="utf-8")
    assert "tree_cfg=tree_config_for(" in source
    assert hashlib.sha256((PACKAGE / "weight_audit.py").read_bytes()).hexdigest() == lock["source_sha256"]["audit"]


def test_bounds_are_per_setting_env_arrays_not_an_unbound_scalar_default():
    _, lock = contracts()
    for setting, contract in lock["action_bounds"].items():
        low = np.full((contract["action_dim"],), contract["lower"], dtype=np.float32)
        high = np.full((contract["action_dim"],), contract["upper"], dtype=np.float32)
        assert hashlib.sha256(low.tobytes()).hexdigest() == contract["lower_sha256"], setting
        assert hashlib.sha256(high.tobytes()).hexdigest() == contract["upper_sha256"], setting
        assert np.all(low < high)


def test_prefix_rule_accepts_short_logged_continuations_and_forbids_mean_four_gate():
    manifest, lock = contracts()
    campaign.validate_manifest(manifest, lock, REPO)
    rule = manifest["scientific_contract"]["prefix_length_rule"]
    target_rule = manifest["acceptance"]["prefix_structural_gates"]["target_rule"]
    assert "min(4" in rule
    assert "never compared to 4" in target_rule
    tampered = copy.deepcopy(manifest)
    tampered["acceptance"]["prefix_structural_gates"]["target_rule"] = "require prefix_steps_mean == 4"
    with pytest.raises(campaign.ContractError):
        campaign.validate_manifest(tampered, lock, REPO)


def test_one_continuous_launch_has_no_stage_stop_and_binds_all_four_audits():
    manifest, lock = contracts()
    cell = campaign.expand_matrix(manifest)[0]
    launch = campaign.trainer_command(
        manifest,
        lock,
        cell,
        repo_root=REPO,
        package_protocol_sha256="0" * 64,
    )
    assert "TREEWM_STOP_AFTER_UPDATE" not in launch["environment"]
    assert all("TREEWM_STOP_AFTER_UPDATE" not in item for item in launch["argv"])
    assert launch["environment"]["TREEWM_RESOLVED_CONFIG_SHA256"] == manifest[
        "resolved_config_contract"
    ]["artifact_sha256"]
    assert launch["environment"]["TREEWM_CAUSAL_PARITY_SHA256"] == manifest[
        "causal_parity_contract"
    ]["artifact_sha256"]
    assert launch["environment"]["MUJOCO_GL"] == "egl"
    assert launch["environment"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    for key in (
        "weight_audit_artifact_sha256",
        "prefix_target_artifact_sha256",
        "resolved_config_artifact_sha256",
        "causal_parity_artifact_sha256",
    ):
        assert launch["hashes"][key] == {
            "weight_audit_artifact_sha256": manifest["weight_audit"]["artifact_sha256"],
            "prefix_target_artifact_sha256": manifest["prefix_target_contract"]["artifact_sha256"],
            "resolved_config_artifact_sha256": manifest["resolved_config_contract"]["artifact_sha256"],
            "causal_parity_artifact_sha256": manifest["causal_parity_contract"]["artifact_sha256"],
        }[key]


def test_prefix_target_lock_proves_exact_logged_horizon_set():
    target_lock = campaign.read_json(PACKAGE / "prefix_target.lock.json")
    for row in target_lock["settings"].values():
        histogram = row["logged_selected_horizon_histogram"]
        assert set(histogram) == {"4", "8", "16", "32", "64"}
        assert sum(histogram.values()) == row["matched_branch_count"]


def test_protocol_inventory_is_closed_and_deterministic():
    assert campaign.PROTOCOL_FILES == (
        "manifest.json",
        "campaign.py",
        "gate.py",
        "weight_audit.py",
        "weight_audit.lock.json",
        "prefix_target_audit.py",
        "prefix_target.lock.json",
        "resolved_config_audit.py",
        "resolved_config.lock.json",
        "causal_parity_audit.py",
        "causal_parity.lock.json",
        "train_entry.py",
        "worker.py",
        "train.slurm",
        "submit.py",
        "cancel.py",
        "report.py",
        "report.slurm",
        "README.md",
        "tests/test_campaign.py",
        "tests/test_gate.py",
        "tests/test_lifecycle.py",
        "tests/test_orchestration.py",
    )
    assert len(campaign.PROTOCOL_FILES) == len(set(campaign.PROTOCOL_FILES))
    assert "protocol.sha256" not in campaign.PROTOCOL_FILES
    assert campaign.SNAPSHOT_IMPORT_FILES == {
        "scripts/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    }
    campaign.validate_snapshot_import_files(REPO)
    assert campaign.protocol_sha256(PACKAGE) == campaign.protocol_sha256(PACKAGE)
