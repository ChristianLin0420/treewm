#!/usr/bin/env python3
"""Audit every byte that gives Exp24 its all-ten offline-data authority.

The audit is outcome-blind and stdout-only.  It authenticates the ten published input
contracts, their declared source NPZ files, and every NPY in the materialized read-only
cache.  It also independently checks that cache arrays are exactly the trajectory-safe
projection and normalization of the source archives.
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import data_authority_common as common


AUDIT_ID = "treewm_exp24_input_contract_authority_v1"
STATUS = "sealed_outcome_blind_all_ten_input_contract_authority"
CONTRACT_KEYS = {
    "action_dim", "calibration_path", "calibration_sha256", "campaign_id",
    "campaign_protocol_sha256", "chosen_thresholds", "contract_sha256",
    "data_manifest_sha256", "dataset_kind", "env_name", "future_recipe_manifest",
    "future_recipe_sha256", "normalizer_sha256", "objective_version", "obs_dim",
    "raw_cache_manifest", "raw_cache_manifest_file_sha256", "raw_cache_read_only",
    "schema_version", "setting_id", "source_files", "source_name", "status",
    "train_manifest_sha256", "train_recipe_rows", "train_trajectories",
    "train_transitions", "validation_manifest_sha256", "validation_recipe_rows",
    "validation_trajectories", "validation_transitions",
}
THRESHOLD_KEYS = {"cluster_threshold", "displacement_threshold", "retrieval_radius"}
STANDARD_CACHE_KEYS = {
    "built_s", "dataset", "key", "norm_stats", "recipe", "shapes", "source_files",
    "source_manifest_sha256", "version",
}
SHARDED_CACHE_KEYS = {
    "action_dim", "cache_key", "dataset_name", "norm_stats", "obs_dim", "recipe",
    "schema_version", "source_dataset", "source_files", "source_manifest_sha256",
    "train_shards", "train_trajectories", "train_transitions", "validation_shards",
    "validation_trajectories", "validation_transitions",
}
NORM_KEYS = {"act_mean", "act_std", "obs_mean", "obs_std"}
STANDARD_ARRAY_NAMES = tuple(
    f"{split}_{field}.npy"
    for split in ("train", "val")
    for field in ("obs", "act", "terminals", "obs_norm", "act_norm")
)
SHARDED_ARRAY_NAMES = tuple(
    f"{split}_{field}.npy"
    for split in ("train", "val")
    for field in (
        "obs", "act", "terminals", "obs_norm", "act_norm", "traj_id",
        "remaining", "starts", "lengths",
    )
)


def _normalizer_sha256(norm_stats: Mapping[str, Sequence[float]]) -> str:
    import struct

    digest = hashlib.sha256(b"treewm-normalizer-float32-v1\n")
    for key in sorted(norm_stats):
        values = norm_stats[key]
        digest.update(key.encode("utf-8") + b"\0" + struct.pack("<Q", len(values)))
        for value in values:
            digest.update(struct.pack("<f", float(value)))
    return digest.hexdigest()


def _validate_thresholds(value: object, label: str) -> dict[str, Any]:
    row = common.require_exact_keys(value, THRESHOLD_KEYS, label)
    for key in sorted(THRESHOLD_KEYS):
        common.require_number(row[key], f"{label}.{key}")
        common.require(float(row[key]) > 0.0, f"{label}.{key} is not positive")
    return dict(row)


def _validate_source_rows(
    value: object, setting: Mapping[str, Any], label: str
) -> list[dict[str, Any]]:
    common.require(isinstance(value, list) and bool(value), f"{label} is empty or not an array")
    sharded = setting["dataset_kind"] == "sharded_100m_full"
    expected_keys = {"index", "mtime_ns", "path", "sha256", "size", "split"}
    rows: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for position, item in enumerate(value):
        row = dict(common.require_exact_keys(item, expected_keys, f"{label}[{position}]"))
        split = common.require_string(row["split"], f"{label}[{position}].split")
        common.require(split in {"train", "val"}, f"{label}[{position}] split differs")
        path = common.safe_relative(common.require_string(row["path"], f"{label}[{position}].path"),
                                    f"{label}[{position}].path")
        common.require(str(path).startswith(str(setting["data_subdir"]) + "/"),
                       f"{label}[{position}] is outside the setting data subdirectory")
        common.require(str(path) not in seen_paths, f"{label} repeats a source path")
        seen_paths.add(str(path))
        common.require_int(row["size"], f"{label}[{position}].size", minimum=1)
        common.require_int(row["mtime_ns"], f"{label}[{position}].mtime_ns", minimum=1)
        common.require_sha256(row["sha256"], f"{label}[{position}].sha256")
        if sharded:
            common.require_int(row["index"], f"{label}[{position}].index", minimum=0)
        else:
            common.require(row["index"] is None, f"{label}[{position}].index is not null")
        rows.append(row)
    if sharded:
        count = common.require_int(
            setting["expected_shards"], f"{label} finalized expected_shards", minimum=1
        )
        expected = [(index, split) for index in range(count) for split in ("train", "val")]
        actual = [(row["index"], row["split"]) for row in rows]
        common.require(actual == expected, f"{label} does not exactly enumerate every shard/split")
    else:
        common.require([row["split"] for row in rows] == ["train", "val"],
                       f"{label} is not the exact train/val pair")
        expected_names = [
            f"{setting['data_subdir']}/{setting['source_name']}.npz",
            f"{setting['data_subdir']}/{setting['source_name']}-val.npz",
        ]
        common.require([row["path"] for row in rows] == expected_names,
                       f"{label} standard filenames differ")
    return rows


def validate_contract(value: object, setting: Mapping[str, Any]) -> dict[str, Any]:
    label = f"{setting['id']} input contract"
    contract = dict(common.require_exact_keys(value, CONTRACT_KEYS, label))
    common.require_int(contract["schema_version"], f"{label}.schema_version")
    common.require(contract["schema_version"] == 2, f"{label} schema version differs")
    expected_scalars = {
        "status": "complete",
        "campaign_id": common.CAMPAIGN_ID,
        "objective_version": common.OBJECTIVE_VERSION,
        "campaign_protocol_sha256": common.CAMPAIGN_PROTOCOL_SHA256,
        "setting_id": setting["id"],
        "env_name": setting["env_name"],
        "source_name": setting["source_name"],
        "dataset_kind": setting["dataset_kind"],
        "calibration_sha256": setting["calibration_sha256"],
        "future_recipe_sha256": setting["future_recipe_sha256"],
        "raw_cache_read_only": True,
    }
    for key, expected in expected_scalars.items():
        common.require(type(contract[key]) is type(expected) and contract[key] == expected,
                       f"{label}.{key} differs from the finalized ledger")
    for key in (
        "contract_sha256", "data_manifest_sha256", "normalizer_sha256",
        "raw_cache_manifest_file_sha256", "train_manifest_sha256",
        "validation_manifest_sha256",
    ):
        common.require_sha256(contract[key], f"{label}.{key}")
    common.require(contract["contract_sha256"] == setting["input_contract_sha256"],
                   f"{label} expected identity differs")
    body = dict(contract)
    claimed = body.pop("contract_sha256")
    common.require(common.stable_hash(body) == claimed, f"{label} canonical self-hash differs")
    _validate_thresholds(contract["chosen_thresholds"], f"{label}.chosen_thresholds")
    for key in (
        "obs_dim", "action_dim", "train_recipe_rows", "train_trajectories",
        "train_transitions", "validation_recipe_rows", "validation_trajectories",
        "validation_transitions",
    ):
        common.require_int(contract[key], f"{label}.{key}", minimum=1)
    common.require(contract["train_recipe_rows"] == setting["published_union_train_anchors"],
                   f"{label} train recipe population differs")
    common.require(contract["validation_recipe_rows"] == setting["published_union_validation_anchors"],
                   f"{label} validation recipe population differs")
    common.require(contract["train_manifest_sha256"] != contract["validation_manifest_sha256"],
                   f"{label} train and validation identities overlap")
    source_rows = _validate_source_rows(contract["source_files"], setting, f"{label}.source_files")
    source_manifest_rows = []
    for row in source_rows:
        normalized = dict(row)
        normalized["path"] = str(
            PurePosixPath(row["path"]).relative_to(PurePosixPath(setting["data_subdir"]))
        )
        if setting["dataset_kind"] == "standard":
            normalized.pop("index")
        source_manifest_rows.append(normalized)
    common.require(common.stable_hash(source_manifest_rows) == contract["data_manifest_sha256"],
                   f"{label} source inventory hash differs")
    for key in ("calibration_path", "future_recipe_manifest", "raw_cache_manifest"):
        common.require_string(contract[key], f"{label}.{key}")
        common.require(Path(contract[key]).is_absolute(), f"{label}.{key} is not absolute")
    return contract


def _validate_norm_stats(value: object, contract: Mapping[str, Any], label: str) -> dict[str, list[float]]:
    stats = dict(common.require_exact_keys(value, NORM_KEYS, label))
    expected_lengths = {
        "obs_mean": contract["obs_dim"], "obs_std": contract["obs_dim"],
        "act_mean": contract["action_dim"], "act_std": contract["action_dim"],
    }
    for key, count in expected_lengths.items():
        values = stats[key]
        common.require(isinstance(values, list) and len(values) == count,
                       f"{label}.{key} length differs")
        for index, item in enumerate(values):
            common.require_number(item, f"{label}.{key}[{index}]")
        if key.endswith("_std"):
            common.require(all(float(item) > 0.0 for item in values),
                           f"{label}.{key} is not strictly positive")
    common.require(_normalizer_sha256(stats) == contract["normalizer_sha256"],
                   f"{label} float32 state hash differs")
    return stats


def validate_cache_manifest(
    value: object, contract: Mapping[str, Any], setting: Mapping[str, Any]
) -> dict[str, Any]:
    label = f"{setting['id']} raw cache manifest"
    sharded = setting["dataset_kind"] == "sharded_100m_full"
    keys = SHARDED_CACHE_KEYS if sharded else STANDARD_CACHE_KEYS
    manifest = dict(common.require_exact_keys(value, keys, label))
    if sharded:
        common.require_int(manifest["schema_version"], f"{label}.schema_version")
        common.require(manifest["schema_version"] == 1, f"{label} schema differs")
        expected = {
            "dataset_name": setting["source_name"],
            "source_dataset": Path(setting["data_subdir"]).name,
            "source_manifest_sha256": contract["data_manifest_sha256"],
            "train_shards": setting["expected_shards"],
            "validation_shards": setting["expected_shards"],
            "train_transitions": contract["train_transitions"],
            "validation_transitions": contract["validation_transitions"],
            "train_trajectories": contract["train_trajectories"],
            "validation_trajectories": contract["validation_trajectories"],
            "obs_dim": contract["obs_dim"],
            "action_dim": contract["action_dim"],
        }
        for key, expected_value in expected.items():
            common.require(type(manifest[key]) is type(expected_value) and manifest[key] == expected_value,
                           f"{label}.{key} differs")
        common.require(isinstance(manifest["cache_key"], str) and len(manifest["cache_key"]) == 20,
                       f"{label}.cache_key differs")
    else:
        common.require_int(manifest["version"], f"{label}.version")
        common.require(manifest["version"] == 4, f"{label} cache version differs")
        common.require(manifest["dataset"] == setting["source_name"], f"{label}.dataset differs")
        common.require(manifest["source_manifest_sha256"] == contract["data_manifest_sha256"],
                       f"{label}.source_manifest_sha256 differs")
        common.require(isinstance(manifest["key"], str) and len(manifest["key"]) == 16,
                       f"{label}.key differs")
        shapes = common.require_exact_keys(manifest["shapes"], {"train_obs", "val_obs"},
                                           f"{label}.shapes")
        train_shape = common.require_int_list(
            shapes["train_obs"], f"{label}.shapes.train_obs", minimum=1
        )
        validation_shape = common.require_int_list(
            shapes["val_obs"], f"{label}.shapes.val_obs", minimum=1
        )
        common.require(len(train_shape) == 2, f"{label}.shapes.train_obs rank differs")
        common.require(len(validation_shape) == 2, f"{label}.shapes.val_obs rank differs")
        common.require(train_shape == [contract["train_transitions"], contract["obs_dim"]],
                       f"{label}.shapes.train_obs differs")
        common.require(validation_shape == [contract["validation_transitions"], contract["obs_dim"]],
                       f"{label}.shapes.val_obs differs")
        common.require_number(manifest["built_s"], f"{label}.built_s")
    _validate_norm_stats(manifest["norm_stats"], contract, f"{label}.norm_stats")
    if sharded:
        recipe = common.require_exact_keys(
            manifest["recipe"],
            {
                "cache_version", "dataset_name", "dtype", "eps", "expected_shards",
                "normalization", "regular_dataset_semantics", "shard_file_stem",
                "source_directory", "source_stat_sha256",
            },
            f"{label}.recipe",
        )
        common.require_int(recipe["cache_version"], f"{label}.recipe.cache_version")
        common.require(recipe["cache_version"] == 2, f"{label}.recipe.cache_version differs")
        common.require(recipe["dataset_name"] == setting["source_name"],
                       f"{label}.recipe.dataset_name differs")
        common.require(recipe["dtype"] == "float32", f"{label}.recipe.dtype differs")
        common.require_number(recipe["eps"], f"{label}.recipe.eps")
        common.require(float(recipe["eps"]) == 1e-6, f"{label}.recipe.eps differs")
        expected_shards = common.require_int(
            recipe["expected_shards"], f"{label}.recipe.expected_shards", minimum=1
        )
        common.require(expected_shards == setting["expected_shards"],
                       f"{label}.recipe.expected_shards differs")
        common.require(recipe["normalization"] == "global_train_mean_std",
                       f"{label}.recipe.normalization differs")
        common.require_bool(recipe["regular_dataset_semantics"],
                            f"{label}.recipe.regular_dataset_semantics")
        common.require(recipe["regular_dataset_semantics"] is True,
                       f"{label}.recipe.regular_dataset_semantics differs")
        common.require_string(recipe["shard_file_stem"], f"{label}.recipe.shard_file_stem")
        common.require_string(recipe["source_directory"], f"{label}.recipe.source_directory")
        common.require_sha256(recipe["source_stat_sha256"], f"{label}.recipe.source_stat_sha256")
    else:
        recipe = common.require_exact_keys(
            manifest["recipe"], {"dtype", "eps", "norm", "source_stat_sha256"},
            f"{label}.recipe",
        )
        common.require(recipe["dtype"] == "float32", f"{label}.recipe.dtype differs")
        common.require_number(recipe["eps"], f"{label}.recipe.eps")
        common.require(float(recipe["eps"]) == 1e-6, f"{label}.recipe.eps differs")
        common.require(recipe["norm"] == "mean_std", f"{label}.recipe.norm differs")
        common.require_sha256(recipe["source_stat_sha256"], f"{label}.recipe.source_stat_sha256")
    source_rows = manifest["source_files"]
    common.require(isinstance(source_rows, list), f"{label}.source_files is not an array")
    normalized_rows = []
    for row in source_rows:
        normalized = dict(row)
        normalized["path"] = f"{setting['data_subdir']}/{row['path']}"
        if not sharded:
            normalized["index"] = None
        normalized_rows.append(normalized)
    common.require(normalized_rows == contract["source_files"],
                   f"{label} source inventory differs from the contract")
    common.require(common.stable_hash(source_rows) == manifest["source_manifest_sha256"],
                   f"{label} source manifest canonical hash differs")
    stat_rows = [{key: item for key, item in row.items() if key != "sha256"} for row in source_rows]
    common.require(common.stable_hash(stat_rows) == recipe["source_stat_sha256"],
                   f"{label} source stat inventory hash differs")
    return manifest


def _array_schema(contract: Mapping[str, Any], sharded: bool, split: str, field: str):
    import numpy as np

    transitions = contract["train_transitions"] if split == "train" else contract["validation_transitions"]
    trajectories = contract["train_trajectories"] if split == "train" else contract["validation_trajectories"]
    if field in {"obs", "obs_norm"}:
        return (transitions, contract["obs_dim"]), np.dtype("<f4")
    if field in {"act", "act_norm"}:
        return (transitions, contract["action_dim"]), np.dtype("<f4")
    if field == "terminals":
        return (transitions,), np.dtype("|b1" if sharded else "<f4")
    common.require(sharded, f"unexpected standard cache field {field}")
    if field in {"traj_id", "remaining"}:
        return (transitions,), np.dtype("<i4")
    if field == "starts":
        return (trajectories,), np.dtype("<i8")
    if field == "lengths":
        return (trajectories,), np.dtype("<i4")
    raise common.DataAuthorityError(f"unknown cache field {field}")


def _raw_member_names(setting: Mapping[str, Any]) -> set[str]:
    names = {"observations.npy", "actions.npy", "terminals.npy", "qpos.npy", "qvel.npy"}
    if setting["id"].startswith("puzzle-") or setting["id"] == "scene":
        names.add("button_states.npy")
    return names


def _audit_cache_and_sources(
    *,
    data_root: common.SecureRoot,
    cache_root: common.SecureRoot,
    cache_root_path: Path,
    contract: Mapping[str, Any],
    setting: Mapping[str, Any],
    verify_content: bool,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    import numpy as np

    setting_id = setting["id"]
    manifest_relative = common.lexical_relative_to(
        contract["raw_cache_manifest"], cache_root_path, f"{setting_id} cache manifest"
    )
    common.require(manifest_relative.name == "manifest.json",
                   f"{setting_id} cache manifest basename differs")
    cache_directory = manifest_relative.parent
    array_names = SHARDED_ARRAY_NAMES if setting["dataset_kind"] == "sharded_100m_full" else STANDARD_ARRAY_NAMES
    with cache_root.subroot(
        cache_directory, f"{setting_id} cache directory"
    ) as cache_directory_root:
        cache_directory_root.require_exact_tree(
            files=("manifest.json", *array_names), directories=()
        )
        cache_manifest, cache_manifest_row = cache_directory_root.read_json("manifest.json")
        common.require(cache_manifest_row["sha256"] == contract["raw_cache_manifest_file_sha256"],
                       f"{setting_id} cache manifest byte hash differs")
        cache_manifest = validate_cache_manifest(cache_manifest, contract, setting)
        inventory: dict[str, Any] = {"manifest.json": cache_manifest_row}
        with ExitStack() as stack:
            arrays: dict[str, Any] = {}
            for name in array_names:
                source = stack.enter_context(
                    cache_directory_root.open_regular(name, f"{setting_id} cache {name}")
                )
                npy = stack.enter_context(common.StableNpy(source, f"{setting_id} cache {name}"))
                split, field_with_suffix = name.split("_", 1)
                field = field_with_suffix[:-4]
                expected_shape, expected_dtype = _array_schema(
                    contract, setting["dataset_kind"] == "sharded_100m_full", split, field
                )
                common.require(tuple(npy.array.shape) == expected_shape,
                               f"{setting_id} cache {name} shape differs")
                common.require(npy.array.dtype == expected_dtype,
                               f"{setting_id} cache {name} dtype differs")
                common.require(not npy.fortran_order, f"{setting_id} cache {name} is Fortran ordered")
                digest = source.sha256()
                inventory[name] = common.inventory_row(source, digest=digest)
                arrays[name] = npy.array

            source_inventory: dict[str, Any] = {}
            cursors = {"train": 0, "val": 0}
            # Trajectory cardinality is an input-contract fact, not an optional
            # cache-content check.  Count every source trajectory even in the
            # validation-only/hash-only mode.
            trajectory_counts = {"train": 0, "val": 0}
            for source_row in contract["source_files"]:
                relative = common.safe_relative(source_row["path"], f"{setting_id} source path")
                with data_root.open_regular(relative, f"{setting_id} source {relative}") as source:
                    digest = source.sha256()
                    common.require(digest == source_row["sha256"],
                                   f"{setting_id} source {relative} digest differs")
                    common.require(source.size == source_row["size"],
                                   f"{setting_id} source {relative} size differs")
                    common.require(source.mtime_ns == source_row["mtime_ns"],
                                   f"{setting_id} source {relative} mtime differs")
                    source_inventory[str(relative)] = common.inventory_row(source, digest=digest)
                    archive, archive_handle = common.validate_npz_members(source, f"{setting_id} source {relative}")
                    try:
                        common.require(set(archive.namelist()) == _raw_member_names(setting),
                                       f"{setting_id} source {relative} NPZ key set differs")
                        observations = common.load_npz_array(archive, "observations", str(relative))
                        actions = common.load_npz_array(archive, "actions", str(relative))
                        terminals = common.load_npz_array(archive, "terminals", str(relative))
                        common.require(observations.ndim == 2
                                       and observations.shape[1] == contract["obs_dim"]
                                       and observations.dtype == np.dtype("<f4"),
                                       f"{setting_id} source {relative} observations schema differs")
                        common.require(actions.ndim == 2
                                       and actions.shape[1] == contract["action_dim"]
                                       and actions.dtype == np.dtype("<f4"),
                                       f"{setting_id} source {relative} actions schema differs")
                        common.require(terminals.ndim == 1 and terminals.dtype == np.dtype("|b1"),
                                       f"{setting_id} source {relative} terminals schema differs")
                        common.require(len(observations) == len(actions) == len(terminals),
                                       f"{setting_id} source {relative} row counts differ")
                        common.require(bool(terminals[-1]),
                                       f"{setting_id} source {relative} lacks final terminal state")
                        for extra_key in sorted(_raw_member_names(setting) - {
                            "observations.npy", "actions.npy", "terminals.npy"
                        }):
                            extra = common.load_npz_array(archive, extra_key[:-4], str(relative))
                            common.require(extra.ndim >= 1 and len(extra) == len(terminals),
                                           f"{setting_id} source {relative} {extra_key} schema differs")
                            if extra_key == "button_states.npy":
                                common.require(extra.dtype == np.dtype("<i8"),
                                               f"{setting_id} source {relative} button dtype differs")
                            else:
                                common.require(extra.dtype == np.dtype("<f4"),
                                               f"{setting_id} source {relative} state dtype differs")
                        ends = np.flatnonzero(terminals)
                        starts = np.concatenate((np.array([0], dtype=np.int64), ends[:-1] + 1))
                        lengths = ends.astype(np.int64) - starts
                        common.require(np.all(lengths > 0),
                                       f"{setting_id} source {relative} has an empty trajectory")
                        mask = ~terminals
                        split = source_row["split"]
                        lo = cursors[split]
                        hi = lo + int(mask.sum())
                        trajectory_lo = trajectory_counts[split]
                        trajectory_hi = trajectory_lo + len(lengths)
                        common.require(hi <= len(arrays[f"{split}_obs.npy"]),
                                       f"{setting_id} source {relative} exceeds cache rows")
                        if verify_content:
                            common.require(np.array_equal(arrays[f"{split}_obs.npy"][lo:hi], observations[mask]),
                                           f"{setting_id} cache observations differ from {relative}")
                            common.require(np.array_equal(arrays[f"{split}_act.npy"][lo:hi], actions[mask]),
                                           f"{setting_id} cache actions differ from {relative}")
                            expected_terminals = np.zeros(hi - lo, dtype=np.bool_)
                            expected_terminals[np.cumsum(lengths) - 1] = True
                            common.require(np.array_equal(arrays[f"{split}_terminals.npy"][lo:hi], expected_terminals),
                                           f"{setting_id} cache terminals differ from {relative}")
                            norm = cache_manifest["norm_stats"]
                            for field, mean_key, std_key in (
                                ("obs", "obs_mean", "obs_std"),
                                ("act", "act_mean", "act_std"),
                            ):
                                raw = arrays[f"{split}_{field}.npy"]
                                normalized = arrays[f"{split}_{field}_norm.npy"]
                                mean = np.asarray(norm[mean_key], dtype=np.float32)
                                std = np.asarray(norm[std_key], dtype=np.float32)
                                for section in common.chunks(hi - lo, 100_000):
                                    source_slice = slice(lo + section.start, lo + section.stop)
                                    expected = ((raw[source_slice] - mean) / std).astype(np.float32)
                                    common.require(np.array_equal(normalized[source_slice], expected, equal_nan=True),
                                                   f"{setting_id} {split} {field} normalization differs")
                            if setting["dataset_kind"] == "sharded_100m_full":
                                absolute_starts = lo + np.cumsum(np.r_[0, lengths[:-1]], dtype=np.int64)
                                common.require(np.array_equal(arrays[f"{split}_starts.npy"][trajectory_lo:trajectory_hi],
                                                              absolute_starts),
                                               f"{setting_id} cache trajectory starts differ")
                                common.require(np.array_equal(arrays[f"{split}_lengths.npy"][trajectory_lo:trajectory_hi],
                                                              lengths.astype(np.int32)),
                                               f"{setting_id} cache trajectory lengths differ")
                                expected_ids = np.repeat(
                                    np.arange(trajectory_lo, trajectory_hi, dtype=np.int32), lengths
                                )
                                expected_remaining = np.concatenate([
                                    np.arange(int(length) - 1, -1, -1, dtype=np.int32)
                                    for length in lengths
                                ])
                                common.require(np.array_equal(arrays[f"{split}_traj_id.npy"][lo:hi], expected_ids),
                                               f"{setting_id} cache trajectory ids differ")
                                common.require(np.array_equal(arrays[f"{split}_remaining.npy"][lo:hi], expected_remaining),
                                               f"{setting_id} cache remaining counts differ")
                        trajectory_counts[split] = trajectory_hi
                        cursors[split] = hi
                    finally:
                        archive.close()
                        archive_handle.close()
            common.require(cursors == {
                "train": contract["train_transitions"],
                "val": contract["validation_transitions"],
            }, f"{setting_id} source coverage does not equal cache transitions")
            common.require(trajectory_counts == {
                "train": contract["train_trajectories"],
                "val": contract["validation_trajectories"],
            }, f"{setting_id} source trajectory totals differ from the input contract")
    return cache_manifest, source_inventory, inventory


def audit_setting(
    *,
    data_root: common.SecureRoot,
    cache_root: common.SecureRoot,
    cache_root_path: Path,
    contract_root_path: Path,
    contract_root: common.SecureRoot,
    setting: Mapping[str, Any],
    verify_content: bool,
) -> dict[str, Any]:
    setting_id = setting["id"]
    contract_relative = PurePosixPath("data") / f"{setting_id}.json"
    contract, contract_inventory = contract_root.read_json(contract_relative)
    contract = validate_contract(contract, setting)
    with ExitStack() as source_scope:
        if setting["dataset_kind"] == "sharded_100m_full":
            source_root = source_scope.enter_context(
                data_root.subroot(
                    setting["data_subdir"],
                    f"{setting_id} sharded source directory",
                )
            )
            source_root.require_exact_tree(
                files=tuple(
                    str(
                        PurePosixPath(row["path"]).relative_to(
                            PurePosixPath(setting["data_subdir"])
                        )
                    )
                    for row in contract["source_files"]
                ),
                directories=(),
            )
        common.require(
            common.lexical_relative_to(contract["calibration_path"], contract_root_path,
                                       f"{setting_id} calibration path")
            == PurePosixPath("calibration") / f"{setting_id}.json",
            f"{setting_id} calibration path differs",
        )
        common.require(
            common.lexical_relative_to(contract["future_recipe_manifest"], contract_root_path,
                                       f"{setting_id} recipe path")
            == PurePosixPath("future-recipes") / setting_id / "manifest.json",
            f"{setting_id} future recipe path differs",
        )
        cache_manifest, source_inventory, cache_inventory = _audit_cache_and_sources(
            data_root=data_root,
            cache_root=cache_root,
            cache_root_path=cache_root_path,
            contract=contract,
            setting=setting,
            verify_content=verify_content,
        )
    inventories = {
        "contract": contract_inventory,
        "source_files": source_inventory,
        "cache_files": cache_inventory,
    }
    return {
        "setting_id": setting_id,
        "input_contract_sha256": contract["contract_sha256"],
        "source_manifest_sha256": contract["data_manifest_sha256"],
        "cache_manifest_file_sha256": contract["raw_cache_manifest_file_sha256"],
        "normalizer_sha256": contract["normalizer_sha256"],
        "train_transitions": contract["train_transitions"],
        "validation_transitions": contract["validation_transitions"],
        "source_file_count": len(source_inventory),
        "cache_file_count": len(cache_inventory),
        "content_verification": "exact_source_projection_and_normalization" if verify_content else "disabled",
        "inventory": inventories,
        "inventory_sha256": common.stable_hash(inventories),
        "cache_manifest_semantic_sha256": common.stable_hash(cache_manifest),
    }


def run(
    *,
    data_root_path: Path,
    cache_root_path: Path,
    contract_root_path: Path,
    ledger: Sequence[Mapping[str, Any]] = common.SETTINGS,
    require_all_ten: bool = True,
    verify_content: bool = True,
) -> dict[str, Any]:
    settings = tuple(dict(row) for row in ledger)
    ids = [row["id"] for row in settings]
    common.require(len(ids) == len(set(ids)), "input audit ledger repeats a setting")
    if require_all_ten:
        common.require(
            common.canonical_json(list(settings)).encode("ascii")
            == common.canonical_json(list(common.SETTINGS)).encode("ascii"),
            "input audit is not using the finalized all-ten ledger",
        )
    with common.SecureRoot(data_root_path, "registered source-data root") as data_root, \
         common.SecureRoot(cache_root_path, "registered raw-cache root") as cache_root, \
         common.SecureRoot(contract_root_path, "registered compatible-contract root") as contract_root:
        # The data-contract directory is governed exclusively by these ten settings.
        with contract_root.subroot(
            "data", "all-ten data-contract directory"
        ) as data_directory:
            data_directory.require_exact_tree(
                files=tuple(f"{setting_id}.json" for setting_id in ids), directories=()
            )
            rows = {
                setting["id"]: audit_setting(
                    data_root=data_root,
                    cache_root=cache_root,
                    cache_root_path=cache_root_path,
                    contract_root_path=contract_root_path,
                    contract_root=contract_root,
                    setting=setting,
                    verify_content=verify_content,
                )
                for setting in settings
            }
    body = {
        "classification": "outcome_blind_read_only_full_byte_inventory_and_source_cache_content_equivalence",
        "setting_order": ids,
        "settings": rows,
        "all_settings_count": len(rows),
        "full_content_verification": bool(verify_content),
        "ledger_sha256": common.stable_hash(list(settings)),
        "inventory_root_sha256": common.stable_hash({key: row["inventory_sha256"] for key, row in rows.items()}),
        "source_sha256": common.source_file_sha256(Path(__file__)),
    }
    return common.seal_result(body, audit_id=AUDIT_ID, status=STATUS)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--contract-root", type=Path, required=True)
    parser.add_argument("--expected-lock", type=Path)
    args = parser.parse_args()
    try:
        result = run(
            data_root_path=args.data_root.absolute(),
            cache_root_path=args.cache_root.absolute(),
            contract_root_path=args.contract_root.absolute(),
        )
        if args.expected_lock is not None:
            lock_root = common.SecureRoot(args.expected_lock.absolute().parent, "input audit lock parent")
            try:
                lock, _row = lock_root.read_json(args.expected_lock.name, "input audit lock")
            finally:
                lock_root.close()
            common.validate_or_compare_lock(result, lock, audit_id=AUDIT_ID, status=STATUS)
    except Exception as exc:
        print(f"input contract audit failed: {exc}", file=sys.stderr)
        return 1
    print("EXP24_INPUT_CONTRACT_AUDIT=" + common.canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
