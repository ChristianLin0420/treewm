from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import data_authority_common as common
import input_contract_audit as audit
from data_authority_fixtures import build_input_fixture
from data_authority_fixtures import build_sharded_input_fixture


def _run(fixture, *, verify_content=True):
    return audit.run(
        data_root_path=fixture["data_root"].absolute(),
        cache_root_path=fixture["cache_root"].absolute(),
        contract_root_path=fixture["contract_root"].absolute(),
        ledger=(fixture["setting"],),
        require_all_ten=False,
        verify_content=verify_content,
    )


def test_exact_source_cache_content_and_self_hash(tmp_path: Path):
    fixture = build_input_fixture(tmp_path)
    result = _run(fixture)
    assert result["status"] == audit.STATUS
    assert result["all_settings_count"] == 1
    assert result["settings"]["toy"]["content_verification"] == (
        "exact_source_projection_and_normalization"
    )
    body = dict(result)
    claimed = body.pop("artifact_sha256")
    assert common.stable_hash(body) == claimed


@pytest.mark.parametrize("verify_content", [False, True])
def test_tiny_sharded_source_cache_is_valid_in_both_content_modes(
    tmp_path: Path, verify_content: bool
):
    fixture = build_sharded_input_fixture(tmp_path)
    result = _run(fixture, verify_content=verify_content)
    row = result["settings"]["toy-sharded"]
    assert row["source_file_count"] == 4
    assert row["train_transitions"] == 8
    assert row["validation_transitions"] == 8


def _set_coherent_wrong_sharded_trajectory_count(fixture, split: str) -> None:
    field = "train_trajectories" if split == "train" else "validation_trajectories"
    manifest_path = fixture["cache_dir"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] += 1
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    for array_name in (f"{split}_starts.npy", f"{split}_lengths.npy"):
        values = np.load(fixture["cache_dir"] / array_name)
        extension = values[-1:] + (2 if array_name.endswith("starts.npy") else 0)
        np.save(fixture["cache_dir"] / array_name, np.concatenate((values, extension)))
    contract_path = fixture["contract_root"] / "data/toy-sharded.json"
    contract = json.loads(contract_path.read_text())
    contract[field] += 1
    contract["raw_cache_manifest_file_sha256"] = common.source_file_sha256(manifest_path)
    contract.pop("contract_sha256")
    contract["contract_sha256"] = common.stable_hash(contract)
    fixture["setting"]["input_contract_sha256"] = contract["contract_sha256"]
    contract_path.write_text(json.dumps(contract, sort_keys=True, indent=2) + "\n")


@pytest.mark.parametrize("split", ["train", "val"])
@pytest.mark.parametrize("verify_content", [False, True])
def test_sharded_trajectory_totals_are_recounted_in_both_content_modes(
    tmp_path: Path, split: str, verify_content: bool
):
    fixture = build_sharded_input_fixture(tmp_path)
    _set_coherent_wrong_sharded_trajectory_count(fixture, split)
    with pytest.raises(common.DataAuthorityError, match="source trajectory totals differ"):
        _run(fixture, verify_content=verify_content)


@pytest.mark.parametrize("replacement", [4.0, True, "4"])
def test_standard_cache_shape_counts_require_exact_integers(
    tmp_path: Path, replacement: object
):
    fixture = build_input_fixture(tmp_path)
    manifest = json.loads((fixture["cache_dir"] / "manifest.json").read_text())
    manifest["shapes"]["train_obs"][0] = replacement
    with pytest.raises(common.DataAuthorityError):
        audit.validate_cache_manifest(manifest, fixture["contract"], fixture["setting"])


@pytest.mark.parametrize("replacement", [2.0, True, "2"])
def test_sharded_expected_shards_requires_an_exact_integer(
    tmp_path: Path, replacement: object
):
    fixture = build_sharded_input_fixture(tmp_path)
    manifest = json.loads((fixture["cache_dir"] / "manifest.json").read_text())
    manifest["recipe"]["expected_shards"] = replacement
    with pytest.raises(common.DataAuthorityError, match="expected_shards"):
        audit.validate_cache_manifest(manifest, fixture["contract"], fixture["setting"])


@pytest.mark.parametrize("attack", ["extra", "symlink", "hardlink", "fifo"])
def test_cache_inventory_rejects_extra_alias_and_special_files(tmp_path: Path, attack: str):
    fixture = build_input_fixture(tmp_path)
    target = fixture["cache_dir"] / "train_obs.npy"
    if attack == "extra":
        (fixture["cache_dir"] / "unclaimed.npy").write_bytes(b"x")
    elif attack == "symlink":
        target.unlink()
        target.symlink_to("val_obs.npy")
    elif attack == "hardlink":
        alias = fixture["cache_dir"] / "alias"
        os.link(target, alias)
    else:
        target.unlink()
        os.mkfifo(target)
    with pytest.raises((common.DataAuthorityError, OSError)):
        _run(fixture)


def test_source_cache_content_mismatch_is_rejected_even_without_expected_npy_hash(tmp_path: Path):
    fixture = build_input_fixture(tmp_path)
    path = fixture["cache_dir"] / "train_obs.npy"
    value = np.load(path, mmap_mode="r+")
    value[0, 0] += 1
    value.flush()
    del value
    with pytest.raises(common.DataAuthorityError, match="observations differ"):
        _run(fixture)


@pytest.mark.parametrize(
    ("field", "claimed"),
    [("train_trajectories", 999), ("validation_trajectories", 998)],
)
def test_claimed_trajectory_totals_are_recounted_without_content_mode(
    tmp_path: Path, field: str, claimed: int
):
    fixture = build_input_fixture(tmp_path)
    contract_path = fixture["contract_root"] / "data/toy.json"
    contract = json.loads(contract_path.read_text())
    contract[field] = claimed
    contract.pop("contract_sha256")
    contract["contract_sha256"] = common.stable_hash(contract)
    fixture["setting"]["input_contract_sha256"] = contract["contract_sha256"]
    contract_path.write_text(json.dumps(contract, sort_keys=True, indent=2) + "\n")
    with pytest.raises(common.DataAuthorityError, match="source trajectory totals differ"):
        _run(fixture, verify_content=False)


def test_contract_rejects_extra_key_wrong_type_and_path_escape(tmp_path: Path):
    fixture = build_input_fixture(tmp_path)
    contract_path = fixture["contract_root"] / "data/toy.json"
    contract = json.loads(contract_path.read_text())
    contract["invented"] = True
    with pytest.raises(common.DataAuthorityError, match="key set differs"):
        audit.validate_contract(contract, fixture["setting"])
    contract.pop("invented")
    contract["train_transitions"] = True
    contract.pop("contract_sha256")
    contract["contract_sha256"] = common.stable_hash(contract)
    setting = dict(fixture["setting"])
    setting["input_contract_sha256"] = contract["contract_sha256"]
    with pytest.raises(common.DataAuthorityError, match="not an integer"):
        audit.validate_contract(contract, setting)
    with pytest.raises(common.DataAuthorityError, match="escapes"):
        common.lexical_relative_to("/tmp/escape", fixture["contract_root"], "attack path")


def test_duplicate_json_key_and_nonfinite_value_are_rejected():
    with pytest.raises(common.DataAuthorityError, match="duplicate JSON key"):
        common.parse_json_bytes(b'{"x":1,"x":2}', "duplicate")
    with pytest.raises(common.DataAuthorityError, match="non-finite"):
        common.parse_json_bytes(b'{"x":NaN}', "nan")


def test_stable_open_detects_in_place_mutation(tmp_path: Path):
    root_path = tmp_path / "root"
    root_path.mkdir()
    path = root_path / "value.bin"
    path.write_bytes(b"before")
    root = common.SecureRoot(root_path.absolute(), "mutation root")
    source = root.open_regular("value.bin")
    try:
        assert source.sha256()
        path.write_bytes(b"after!")
        with pytest.raises(common.DataAuthorityError, match="mutated"):
            source.verify_stable()
    finally:
        try:
            source.close()
        except common.DataAuthorityError:
            pass
        try:
            root.close()
        except common.DataAuthorityError:
            pass


def test_secure_root_exact_tree_unchanged_passes_close_rescan(tmp_path: Path):
    tree = tmp_path / "exact-tree"
    nested = tree / "nested"
    nested.mkdir(parents=True)
    (tree / "top.bin").write_bytes(b"top")
    (nested / "value.bin").write_bytes(b"nested")
    with common.SecureRoot(tree.absolute(), "unchanged exact tree") as root:
        root.require_exact_tree(
            files=("top.bin", "nested/value.bin"), directories=("nested",)
        )


@pytest.mark.parametrize("nested", [False, True], ids=("top", "nested"))
@pytest.mark.parametrize("attack", ["add", "delete", "replace"])
def test_secure_root_close_rejects_same_tick_persistent_tree_mutation(
    tmp_path: Path, nested: bool, attack: str
):
    tree = tmp_path / "exact-tree"
    governed = tree / "nested" if nested else tree
    governed.mkdir(parents=True)
    value = governed / "value.bin"
    value.write_bytes(b"identity-preserving-content")
    relative = "nested/value.bin" if nested else "value.bin"
    root = common.SecureRoot(tree.absolute(), "hostile exact tree")
    root.require_exact_tree(
        files=(relative,), directories=("nested",) if nested else ()
    )
    root_identity = common.file_identity(os.stat(tree))
    if attack == "add":
        (governed / "extra.bin").write_bytes(b"extra")
    elif attack == "delete":
        value.unlink()
    else:
        replacement = tmp_path / "replacement.bin"
        replacement.write_bytes(value.read_bytes())
        before = value.stat()
        os.chmod(replacement, before.st_mode & 0o777)
        os.utime(
            replacement,
            ns=(before.st_atime_ns, before.st_mtime_ns),
        )
        old_inode = before.st_ino
        os.replace(replacement, value)
        assert value.stat().st_ino != old_inode
    if nested:
        # Mutating a child directory never changes the retained root's own stat.
        assert common.file_identity(os.stat(tree)) == root_identity
    with pytest.raises(common.DataAuthorityError):
        root.close()


def test_unsealed_placeholder_cannot_be_accepted_as_lock():
    path = Path(audit.__file__).with_name("input_contract.lock.unsealed.json")
    value = json.loads(path.read_text())
    assert value["artifact_sha256"] is None
    with pytest.raises(common.DataAuthorityError):
        common.validate_sealed_result(value, audit_id=audit.AUDIT_ID, status=audit.STATUS)


@pytest.mark.parametrize("replacement", [True, 1.0, "1"])
def test_sealed_lock_schema_requires_exact_integer_even_after_rehash(replacement: object):
    value = common.seal_result({"payload": "x"}, audit_id="test", status="complete")
    value["schema_version"] = replacement
    value.pop("artifact_sha256")
    value["artifact_sha256"] = common.stable_hash(value)
    with pytest.raises(common.DataAuthorityError, match="not an integer"):
        common.validate_sealed_result(value, audit_id="test", status="complete")


def test_input_all_ten_guard_rejects_type_equivalent_float_ledger(tmp_path: Path):
    ledger = json.loads(common.canonical_json(list(common.SETTINGS)))
    ledger[0]["published_union_train_anchors"] = float(
        ledger[0]["published_union_train_anchors"]
    )
    with pytest.raises(common.DataAuthorityError, match="finalized all-ten ledger"):
        audit.run(
            data_root_path=tmp_path / "missing-data",
            cache_root_path=tmp_path / "missing-cache",
            contract_root_path=tmp_path / "missing-contract",
            ledger=ledger,
            require_all_ten=True,
            verify_content=False,
        )
