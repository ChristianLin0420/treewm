from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import data_authority_common as common
import future_recipe_audit as audit
from data_authority_fixtures import build_future_fixture


def _run(fixture):
    return audit.run(
        contract_root_path=fixture["contract_root"].absolute(),
        input_lock=None,
        ledger=(fixture["setting"],),
        require_all_ten=False,
    )


def _set_nested(value, path, replacement):
    cursor = value
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement


def _validate_rehashed_calibration(fixture, path, replacement):
    calibration = json.loads(json.dumps(fixture["calibration"]))
    _set_nested(calibration, path, replacement)
    calibration.pop("contract_sha256")
    calibration["contract_sha256"] = common.stable_hash(calibration)
    setting = dict(fixture["setting"])
    setting["calibration_sha256"] = calibration["contract_sha256"]
    contract = dict(fixture["contract"])
    contract["calibration_sha256"] = calibration["contract_sha256"]
    return audit._validate_calibration(calibration, setting, contract)


def _validate_rehashed_child(fixture, path, replacement):
    child_path = fixture["contract_root"] / "future-recipes/toy/val/manifest.json"
    child = json.loads(child_path.read_text())
    _set_nested(child, path, replacement)
    child["identity_sha256"] = common.stable_hash(child["identity"])
    child.pop("recipe_sha256")
    child["recipe_sha256"] = common.stable_hash(child)
    composite = json.loads(json.dumps(fixture["composite"]))
    composite["validation_recipe_sha256"] = child["recipe_sha256"]
    return audit._validate_child(
        child,
        setting=fixture["setting"],
        contract=fixture["contract"],
        composite=composite,
        calibration=fixture["calibration"],
        split="val",
    )


def test_composite_children_records_calibration_code_runtime_are_bound(tmp_path: Path):
    fixture = build_future_fixture(tmp_path)
    result = _run(fixture)
    row = result["settings"]["toy"]
    assert result["status"] == audit.STATUS
    assert row["code_sha256"] == common.RECIPE_CODE_SHA256
    assert row["runtime_sha256"] == common.RECIPE_RUNTIME_SHA256
    assert row["splits"]["train"]["anchor_count"] == 6000
    assert row["splits"]["val"]["anchor_count"] == 6000
    body = dict(result)
    assert body.pop("artifact_sha256") == common.stable_hash(body)


@pytest.mark.parametrize("attack", ["missing", "extra", "symlink", "hardlink"])
def test_recipe_tree_rejects_missing_extra_and_aliases(tmp_path: Path, attack: str):
    fixture = build_future_fixture(tmp_path)
    recipe = fixture["contract_root"] / "future-recipes/toy"
    target = recipe / "val/records.npy"
    if attack == "missing":
        target.unlink()
    elif attack == "extra":
        (recipe / "val/unclaimed.bin").write_bytes(b"x")
    elif attack == "symlink":
        target.unlink()
        target.symlink_to("../train/records.npy")
    else:
        os.link(target, recipe / "val/alias.npy")
    with pytest.raises((common.DataAuthorityError, OSError)):
        _run(fixture)


def test_child_manifest_rejects_extra_key_even_with_recomputed_self_hash(tmp_path: Path):
    fixture = build_future_fixture(tmp_path)
    path = fixture["contract_root"] / "future-recipes/toy/val/manifest.json"
    child = json.loads(path.read_text())
    child["invented"] = 1
    child.pop("recipe_sha256")
    child["recipe_sha256"] = common.stable_hash(child)
    with pytest.raises(common.DataAuthorityError, match="key set differs"):
        audit._validate_child(
            child,
            setting=fixture["setting"],
            contract=fixture["contract"],
            composite=fixture["composite"],
            calibration=fixture["calibration"],
            split="val",
        )


def test_records_content_is_checked_beyond_manifest_byte_hash(tmp_path: Path):
    fixture = build_future_fixture(tmp_path)
    path = fixture["contract_root"] / "future-recipes/toy/val/records.npy"
    records = np.load(path, mmap_mode="r+")
    records["mode_valid"][0, 0] = 7
    records.flush()
    del records
    # A hostile publisher can update all local self-hashes; semantic validation still fails.
    manifest_path = path.parent / "manifest.json"
    child = json.loads(manifest_path.read_text())
    child["records_sha256"] = common.source_file_sha256(path)
    child["records_mtime_ns"] = path.stat().st_mtime_ns
    child.pop("recipe_sha256")
    child["recipe_sha256"] = common.stable_hash(child)
    manifest_path.write_text(json.dumps(child, sort_keys=True, indent=2) + "\n")
    composite_path = path.parents[1] / "manifest.json"
    composite = json.loads(composite_path.read_text())
    composite["validation_recipe_sha256"] = child["recipe_sha256"]
    composite.pop("recipe_sha256")
    composite["recipe_sha256"] = common.stable_hash(composite)
    composite_path.write_text(json.dumps(composite, sort_keys=True, indent=2) + "\n")
    contract_path = fixture["contract_root"] / "data/toy.json"
    contract = json.loads(contract_path.read_text())
    contract["future_recipe_sha256"] = composite["recipe_sha256"]
    contract.pop("contract_sha256")
    contract["contract_sha256"] = common.stable_hash(contract)
    contract_path.write_text(json.dumps(contract, sort_keys=True, indent=2) + "\n")
    setting = dict(fixture["setting"])
    setting["future_recipe_sha256"] = composite["recipe_sha256"]
    setting["input_contract_sha256"] = contract["contract_sha256"]
    with pytest.raises(common.DataAuthorityError, match="mode_valid is not binary"):
        audit.run(
            contract_root_path=fixture["contract_root"].absolute(),
            input_lock=None,
            ledger=(setting,),
            require_all_ten=False,
        )


def test_calibration_index_hash_and_offsets_are_independently_verified(tmp_path: Path):
    fixture = build_future_fixture(tmp_path)
    calibration = dict(fixture["calibration"])
    calibration["sample_indices"] = list(calibration["sample_indices"])
    calibration["sample_indices"][-1] += 1
    calibration.pop("contract_sha256")
    calibration["contract_sha256"] = common.stable_hash(calibration)
    setting = dict(fixture["setting"])
    setting["calibration_sha256"] = calibration["contract_sha256"]
    contract = dict(fixture["contract"])
    contract["calibration_sha256"] = calibration["contract_sha256"]
    with pytest.raises(common.DataAuthorityError, match="sample byte hash differs"):
        audit._validate_calibration(calibration, setting, contract)


@pytest.mark.parametrize(
    ("path", "replacement", "message"),
    [
        (("config", "query_multiplier"), 6.0, "not an integer"),
        (("cluster", "chosen_index"), False, "not an integer"),
        (("horizon", "candidates", 0, "occupied_classes"), "5", "not an integer"),
        (("retrieval", "chosen", "max_retrieved"), 24.0, "not an integer"),
        (("selected_continuation_count",), "4096", "not an integer"),
        (("sample_indices", 0), False, "not an integer"),
    ],
)
def test_calibration_integer_fields_reject_bool_float_and_string(
    tmp_path: Path, path, replacement, message: str
):
    fixture = build_future_fixture(tmp_path)
    with pytest.raises(common.DataAuthorityError, match=message):
        _validate_rehashed_calibration(fixture, path, replacement)


def test_calibration_failed_gate_is_rejected_even_after_rehash(tmp_path: Path):
    fixture = build_future_fixture(tmp_path)
    with pytest.raises(common.DataAuthorityError, match="did not pass"):
        _validate_rehashed_calibration(
            fixture, ("gates", "mean_retrieved", "passed"), False
        )


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("record_dtype", 1, 2, 0), 24.0),
        (("identity", "task_metric_dims", 0), 0.0),
        (("identity", "task_metric_dims", 0), False),
        (("identity", "xy_dims", 0), 0.0),
        (("identity", "xy_dims", 0), False),
        (("identity", "future_config", "horizons", 0), 4.0),
        (("identity", "chosen_thresholds", "cluster_threshold"), 1),
    ],
)
def test_child_nested_dtype_dims_horizons_and_thresholds_are_type_exact(
    tmp_path: Path, path, replacement
):
    fixture = build_future_fixture(tmp_path)
    with pytest.raises(common.DataAuthorityError):
        _validate_rehashed_child(fixture, path, replacement)


def test_calibration_and_composite_nested_threshold_joins_are_type_exact(tmp_path: Path):
    fixture = build_future_fixture(tmp_path)
    contract = json.loads(json.dumps(fixture["contract"]))
    contract["chosen_thresholds"]["cluster_threshold"] = 1
    with pytest.raises(common.DataAuthorityError, match="chosen thresholds differ"):
        audit._validate_calibration(fixture["calibration"], fixture["setting"], contract)

    composite = json.loads(json.dumps(fixture["composite"]))
    composite["chosen_thresholds"]["cluster_threshold"] = 1
    composite.pop("recipe_sha256")
    composite["recipe_sha256"] = common.stable_hash(composite)
    setting = dict(fixture["setting"])
    setting["future_recipe_sha256"] = composite["recipe_sha256"]
    with pytest.raises(common.DataAuthorityError, match="chosen_thresholds"):
        audit._validate_composite(composite, setting, fixture["contract"])


@pytest.mark.parametrize("attack", ["summary_drift", "failed_gate"])
def test_all_settings_row_is_exact_calibration_projection_and_all_gates_pass(
    tmp_path: Path, attack: str
):
    fixture = build_future_fixture(tmp_path)
    path = fixture["contract_root"] / "calibration/ALL_SETTINGS.json"
    summary = json.loads(path.read_text())
    row = summary["settings"][0]
    if attack == "summary_drift":
        row["chosen"]["retrieval_radius"] += 0.25
        message = "row differs from validated calibration"
    else:
        row["gates"]["mean_retrieved"]["passed"] = False
        message = "did not pass"
    summary.pop("summary_sha256")
    summary["summary_sha256"] = common.stable_hash(summary)
    path.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n")
    with pytest.raises(common.DataAuthorityError, match=message):
        _run(fixture)


@pytest.mark.parametrize(
    ("attack", "message"),
    [
        ("moved", "not a contiguous prefix"),
        ("unsampled_rep", "does not gather a present neighbor"),
        ("duplicate", "repeats a present neighbor"),
        ("future_on_absent", "future-valid slot has no present neighbor"),
    ],
)
def test_records_reject_moved_duplicate_and_unsampled_neighbors(
    tmp_path: Path, attack: str, message: str
):
    fixture = build_future_fixture(tmp_path)
    records = fixture["records"].copy()
    if attack == "moved":
        records["neighbors"][0, 0] = -1
        records["neighbors"][0, 1] = records["anchor"][0]
        records["fut_valid"][0, 0] = 0
        records["fut_valid"][0, 1] = 1
        records["cluster"][0, 0] = -1
        records["cluster"][0, 1] = 0
        records["mode_rep"][0, 0] = 1
    elif attack == "unsampled_rep":
        records["mode_rep"][0, 0] = 1
    elif attack == "duplicate":
        records["neighbors"][0, 1] = records["neighbors"][0, 0]
    else:
        records["fut_valid"][0, 1] = 1
    child = json.loads(
        (fixture["contract_root"] / "future-recipes/toy/val/manifest.json").read_text()
    )
    with pytest.raises(common.DataAuthorityError, match=message):
        audit._validate_record_content(
            records, child=child, contract=fixture["contract"], label="hostile records"
        )


@pytest.mark.parametrize(
    ("attack", "message"),
    [
        ("candidate_upper_bound", "candidate counts are outside"),
        ("raw_mode_upper_bound", "raw-mode counts differ"),
    ],
)
def test_record_count_fields_enforce_protocol_upper_bounds(
    tmp_path: Path, attack: str, message: str
):
    fixture = build_future_fixture(tmp_path)
    records = fixture["records"].copy()
    if attack == "candidate_upper_bound":
        records["retrieval_num_candidates"][0] = 65535
    else:
        records["modes_raw"][0] = 255
        records["modes_truncated"][0] = 254
    child = json.loads(
        (fixture["contract_root"] / "future-recipes/toy/val/manifest.json").read_text()
    )
    with pytest.raises(common.DataAuthorityError, match=message):
        audit._validate_record_content(
            records, child=child, contract=fixture["contract"], label="hostile records"
        )


def test_candidate_upper_bound_includes_a_self_insert_after_the_145_item_query(
    tmp_path: Path,
):
    fixture = build_future_fixture(tmp_path)
    child = json.loads(
        (fixture["contract_root"] / "future-recipes/toy/val/manifest.json").read_text()
    )
    records = fixture["records"].copy()
    records["retrieval_num_candidates"][0] = 146
    audit._validate_record_content(
        records, child=child, contract=fixture["contract"], label="boundary records"
    )
    records["retrieval_num_candidates"][0] = 147
    with pytest.raises(common.DataAuthorityError, match="candidate counts are outside"):
        audit._validate_record_content(
            records, child=child, contract=fixture["contract"], label="boundary records"
        )


def test_unsealed_future_placeholder_is_not_a_seal():
    path = Path(audit.__file__).with_name("future_recipe.lock.unsealed.json")
    value = json.loads(path.read_text())
    assert value["status"] == "unsealed"
    with pytest.raises(common.DataAuthorityError):
        common.validate_sealed_result(value, audit_id=audit.AUDIT_ID, status=audit.STATUS)


def test_future_all_ten_guard_rejects_type_equivalent_float_ledger(tmp_path: Path):
    ledger = json.loads(common.canonical_json(list(common.SETTINGS)))
    ledger[2]["expected_shards"] = float(ledger[2]["expected_shards"])
    with pytest.raises(common.DataAuthorityError, match="finalized all-ten ledger"):
        audit.run(
            contract_root_path=tmp_path / "missing-contract",
            input_lock=None,
            ledger=ledger,
            require_all_ten=True,
        )
