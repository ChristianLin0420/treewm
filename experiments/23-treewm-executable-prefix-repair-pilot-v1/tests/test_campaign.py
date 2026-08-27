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
