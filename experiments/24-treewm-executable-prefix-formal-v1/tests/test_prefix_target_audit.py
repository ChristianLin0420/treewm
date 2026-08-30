from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest

import data_authority_common as common
import future_recipe_audit
import prefix_target_audit as audit
from data_authority_fixtures import build_future_fixture


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _future_lock(fixture):
    return future_recipe_audit.run(
        contract_root_path=fixture["contract_root"].absolute(),
        input_lock=None,
        ledger=(fixture["setting"],),
        require_all_ten=False,
    )


def test_exact_5120_anchor_target_derivation_and_lock_chain(tmp_path: Path):
    fixture = build_future_fixture(tmp_path)
    future_lock = _future_lock(fixture)
    result = audit.run(
        project_root_path=PROJECT_ROOT,
        contract_root_path=fixture["contract_root"].absolute(),
        future_lock=future_lock,
        ledger=(fixture["setting"],),
        require_all_ten=False,
    )
    target = result["settings"]["toy"]
    assert target["anchor_count"] == 5120
    assert target["all_anchors_have_match"] is True
    assert target["matched_branch_count"] == 5120
    assert target["prefix_length_histogram"] == {"1": 0, "2": 0, "3": 0, "4": 5120}
    assert target["prefix_action_scalar_count"] == 5120 * 4
    assert result["future_recipe_audit_artifact_sha256"] == future_lock["artifact_sha256"]
    body = dict(result)
    assert body.pop("artifact_sha256") == common.stable_hash(body)


def test_sampler_reproduction_matches_protocol_implementation():
    torch = pytest.importorskip("torch")
    from treewm.data.samplers import FixedRepresentativeSampler

    class Dataset:
        def __len__(self):
            return 6000

    expected = FixedRepresentativeSampler(
        Dataset(), batch_size=256, num_batches=20, seed=1701
    ).global_indices.numpy()
    actual = audit.fixed_representative_positions(6000)
    assert np.array_equal(actual, expected)
    assert hashlib_sha(actual) == hashlib_sha(expected)


def hashlib_sha(value):
    import hashlib

    return hashlib.sha256(value.astype("<i8", copy=False).tobytes()).hexdigest()


def test_missing_anchor_match_is_rejected(tmp_path: Path):
    fixture = build_future_fixture(tmp_path)
    records = fixture["records"].copy()
    positions = audit.fixed_representative_positions(len(records))
    records["mode_valid"][int(positions[0])] = 0
    with pytest.raises(common.DataAuthorityError, match="retained-mode count differs"):
        audit.derive_target(
            records,
            setting_id="toy",
            action_dim=1,
            validation_manifest_sha256="2" * 64,
            future_recipe_sha256=fixture["setting"]["future_recipe_sha256"],
        )


def test_duplicate_or_out_of_range_representatives_are_rejected(tmp_path: Path):
    fixture = build_future_fixture(tmp_path)
    records = fixture["records"].copy()
    position = int(audit.fixed_representative_positions(len(records))[0])
    records["mode_valid"][position, :2] = 1
    records["mode_rep"][position, :2] = 0
    with pytest.raises(common.DataAuthorityError, match="repeats a representative"):
        audit.derive_target(
            records,
            setting_id="toy",
            action_dim=1,
            validation_manifest_sha256="2" * 64,
            future_recipe_sha256=fixture["setting"]["future_recipe_sha256"],
        )


@pytest.mark.parametrize(
    ("attack", "message"),
    [
        ("moved", "not a contiguous prefix"),
        ("unsampled", "not a present neighbor"),
    ],
)
def test_prefix_derivation_rejects_moved_or_unsampled_neighbors(
    tmp_path: Path, attack: str, message: str
):
    fixture = build_future_fixture(tmp_path)
    records = fixture["records"].copy()
    position = int(audit.fixed_representative_positions(len(records))[0])
    if attack == "moved":
        records["neighbors"][position, 0] = -1
        records["neighbors"][position, 1] = records["anchor"][position]
        records["fut_valid"][position, 0] = 0
        records["fut_valid"][position, 1] = 1
        records["mode_rep"][position, 0] = 1
    else:
        records["mode_rep"][position, 0] = 1
    with pytest.raises(common.DataAuthorityError, match=message):
        audit.derive_target(
            records,
            setting_id="toy",
            action_dim=1,
            validation_manifest_sha256="2" * 64,
            future_recipe_sha256=fixture["setting"]["future_recipe_sha256"],
        )


def test_tampered_future_lock_is_rejected_before_records_open(tmp_path: Path):
    fixture = build_future_fixture(tmp_path)
    future_lock = _future_lock(fixture)
    future_lock["settings"]["toy"]["splits"]["val"]["records_sha256"] = "0" * 64
    with pytest.raises(common.DataAuthorityError, match="self-hash differs"):
        audit.run(
            project_root_path=PROJECT_ROOT,
            contract_root_path=fixture["contract_root"].absolute(),
            future_lock=future_lock,
            ledger=(fixture["setting"],),
            require_all_ten=False,
        )


@pytest.mark.parametrize(
    ("relative", "replacement"),
    [
        (("val/manifest.json", "size"), None),
        (("val/records.npy", "mtime_ns"), None),
        (("val/records.npy", "sha256"), "not-a-sha256"),
    ],
)
def test_prefix_future_inventory_requires_exact_keys_and_scalar_types(
    tmp_path: Path, relative, replacement
):
    fixture = build_future_fixture(tmp_path)
    future_lock = _future_lock(fixture)
    row = future_lock["settings"]["toy"]["inventory"][relative[0]]
    row[relative[1]] = (
        float(row[relative[1]]) if replacement is None else replacement
    )
    future_lock.pop("artifact_sha256")
    future_lock["artifact_sha256"] = common.stable_hash(future_lock)
    with pytest.raises(common.DataAuthorityError):
        audit.run(
            project_root_path=PROJECT_ROOT,
            contract_root_path=fixture["contract_root"].absolute(),
            future_lock=future_lock,
            ledger=(fixture["setting"],),
            require_all_ten=False,
        )


def test_prefix_future_inventory_rejects_extra_key(tmp_path: Path):
    fixture = build_future_fixture(tmp_path)
    future_lock = _future_lock(fixture)
    future_lock["settings"]["toy"]["inventory"]["val/manifest.json"]["extra"] = 1
    future_lock.pop("artifact_sha256")
    future_lock["artifact_sha256"] = common.stable_hash(future_lock)
    with pytest.raises(common.DataAuthorityError, match="key set differs"):
        audit.run(
            project_root_path=PROJECT_ROOT,
            contract_root_path=fixture["contract_root"].absolute(),
            future_lock=future_lock,
            ledger=(fixture["setting"],),
            require_all_ten=False,
        )


def test_all_ten_mode_rejects_a_subset_ledger(tmp_path: Path):
    fixture = build_future_fixture(tmp_path)
    with pytest.raises(common.DataAuthorityError, match="finalized all-ten ledger"):
        audit.run(
            project_root_path=PROJECT_ROOT,
            contract_root_path=fixture["contract_root"].absolute(),
            future_lock={},
            ledger=(fixture["setting"],),
            require_all_ten=True,
        )


def test_prefix_all_ten_guard_rejects_type_equivalent_float_ledger(tmp_path: Path):
    ledger = json.loads(common.canonical_json(list(common.SETTINGS)))
    ledger[0]["task_metric_dims"][0] = float(ledger[0]["task_metric_dims"][0])
    with pytest.raises(common.DataAuthorityError, match="finalized all-ten ledger"):
        audit.run(
            project_root_path=PROJECT_ROOT,
            contract_root_path=tmp_path / "missing-contract",
            future_lock={},
            ledger=ledger,
            require_all_ten=True,
        )


def test_unsealed_prefix_placeholder_is_not_a_seal():
    path = Path(audit.__file__).with_name("prefix_target.lock.unsealed.json")
    value = json.loads(path.read_text())
    assert value["artifact_sha256"] is None
    with pytest.raises(common.DataAuthorityError):
        common.validate_sealed_result(value, audit_id=audit.AUDIT_ID, status=audit.STATUS)
