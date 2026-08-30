#!/usr/bin/env python3
"""Derive the exact all-ten 5,120-anchor executable-prefix target authority.

Only published validation recipe records are sampled.  The audit builds no model,
performs no rollout, reads no run directory, and has no outcome-bearing input.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import data_authority_common as common
import future_recipe_audit
import input_contract_audit


AUDIT_ID = "treewm_exp24_prefix_target_authority_v1"
STATUS = "sealed_outcome_blind_all_ten_fixed_validation_prefix_targets"
VALIDATION_SAMPLE_SEED = 1701
BATCH_SIZE = 256
VALIDATION_BATCHES = 20
ANCHOR_COUNT = BATCH_SIZE * VALIDATION_BATCHES
PREFIX_WIDTH = 64
PREFIX_CAP = 4
LOGGED_HORIZONS = (4, 8, 16, 32, 64)
SAMPLER_SOURCE_RELATIVE = Path("treewm/data/samplers.py")
SAMPLER_SOURCE_SHA256 = "ae3dc07cf7a4a1a673faadb20b3dc72b235b5f2c673650027375b1f4dab7f681"


def _validate_inventory_row(value: object, label: str) -> dict[str, Any]:
    row = dict(common.require_exact_keys(value, {"sha256", "size", "mtime_ns"}, label))
    common.require_sha256(row["sha256"], f"{label}.sha256")
    common.require_int(row["size"], f"{label}.size", minimum=0)
    common.require_int(row["mtime_ns"], f"{label}.mtime_ns", minimum=0)
    return row


def fixed_representative_positions(
    dataset_size: int,
    *,
    batch_size: int = BATCH_SIZE,
    num_batches: int = VALIDATION_BATCHES,
    seed: int = VALIDATION_SAMPLE_SEED,
) -> Any:
    """Independent byte-for-byte reproduction of FixedRepresentativeSampler v1."""
    import torch

    common.require_int(dataset_size, "validation dataset size", minimum=1)
    common.require_int(batch_size, "fixed sampler batch size", minimum=1)
    common.require_int(num_batches, "fixed sampler batch count", minimum=1)
    common.require_int(seed, "fixed sampler seed", minimum=0)
    sample_size = batch_size * num_batches
    common.require(sample_size <= dataset_size,
                   "fixed validation sample exceeds validation recipe population")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    stratum = torch.arange(sample_size, dtype=torch.int64)
    starts = torch.div(stratum * dataset_size, sample_size, rounding_mode="floor")
    ends = torch.div((stratum + 1) * dataset_size, sample_size, rounding_mode="floor")
    widths = ends - starts
    common.require(bool(torch.all(widths > 0)), "fixed sampler has an empty stratum")
    offsets = torch.floor(
        torch.rand(sample_size, generator=generator) * widths.float()
    ).to(torch.int64)
    selected = starts + offsets
    by_batch = selected.view(batch_size, num_batches).transpose(0, 1)
    ordered_batches = []
    for values in by_batch:
        ordered_batches.append(values[torch.randperm(batch_size, generator=generator)])
    positions = torch.stack(ordered_batches).reshape(-1).numpy().astype("<i8", copy=False)
    common.require(len(positions) == sample_size and len(set(positions.tolist())) == sample_size,
                   "fixed sampler positions are not exact and unique")
    common.require(int(positions.min()) >= 0 and int(positions.max()) < dataset_size,
                   "fixed sampler position escapes validation recipe")
    return positions


def _sampler_summary(positions: Any, dataset_size: int) -> dict[str, Any]:
    import numpy as np

    fractions = positions.astype(np.float64) / max(dataset_size - 1, 1)
    quantiles = np.quantile(fractions, [0.0, 0.25, 0.5, 0.75, 1.0])
    return {
        "sampler": "fixed_representative_stratified_permutation_v1",
        "seed": VALIDATION_SAMPLE_SEED,
        "dataset_size": dataset_size,
        "global_sample_size": len(positions),
        "batch_size_per_rank": BATCH_SIZE,
        "num_batches": VALIDATION_BATCHES,
        "num_replicas": 1,
        "indices_sha256": hashlib.sha256(positions.tobytes(order="C")).hexdigest(),
        "anchor_rank_fraction_quantiles": {
            key: float(value)
            for key, value in zip(("q00", "q25", "q50", "q75", "q100"), quantiles, strict=True)
        },
    }


def derive_target(
    records: Any,
    *,
    setting_id: str,
    action_dim: int,
    validation_manifest_sha256: str,
    future_recipe_sha256: str,
) -> dict[str, Any]:
    import numpy as np

    common.require(records.ndim == 1 and records.dtype.names is not None,
                   f"{setting_id} validation records schema differs")
    common.require(list(records.dtype.descr) == [
        tuple([row[0], row[1], tuple(row[2])]) if len(row) == 3 else tuple(row)
        for row in future_recipe_audit.expected_record_dtype()
    ], f"{setting_id} validation record dtype differs")
    common.require_int(action_dim, f"{setting_id} action_dim", minimum=1)
    common.require_sha256(validation_manifest_sha256,
                          f"{setting_id} validation_manifest_sha256")
    common.require_sha256(future_recipe_sha256, f"{setting_id} future_recipe_sha256")
    positions = fixed_representative_positions(len(records))
    selected = records[positions]
    anchors = np.asarray(selected["anchor"], dtype="<i8")
    counts = np.zeros(ANCHOR_COUNT, dtype="u1")
    lengths = np.zeros((ANCHOR_COUNT, 4), dtype="u1")
    logged_horizons = np.zeros((ANCHOR_COUNT, 4), dtype="u1")
    masks = np.zeros((ANCHOR_COUNT, 4, PREFIX_WIDTH), dtype="u1")
    action_steps = 0
    branch_count = 0
    for row_index, row in enumerate(selected):
        neighbors = np.asarray(row["neighbors"], dtype=np.int64)
        present = neighbors >= 0
        present_count = int(present.sum())
        common.require(np.all((neighbors == -1) | (neighbors >= 0)),
                       f"{setting_id} anchor {row_index} neighbor sentinel differs")
        common.require(np.array_equal(
            present, np.arange(len(neighbors), dtype=np.int64) < present_count
        ), f"{setting_id} anchor {row_index} present neighbors are not a contiguous prefix")
        common.require(len(set(neighbors[present].tolist())) == present_count,
                       f"{setting_id} anchor {row_index} repeats a present neighbor")
        future_valid = np.asarray(row["fut_valid"], dtype=np.uint8)
        common.require(np.all((future_valid == 0) | (future_valid == 1)),
                       f"{setting_id} anchor {row_index} future validity is not binary")
        common.require(np.all(~future_valid.astype(bool) | present),
                       f"{setting_id} anchor {row_index} future-valid slot is not present")
        mode_valid = np.asarray(row["mode_valid"], dtype=np.uint8)
        common.require(np.all((mode_valid == 0) | (mode_valid == 1)),
                       f"{setting_id} anchor {row_index} mode validity is not binary")
        representatives = np.asarray(row["mode_rep"], dtype=np.int64)[mode_valid.astype(bool)]
        count = len(representatives)
        common.require(1 <= count <= 4,
                       f"{setting_id} anchor {row_index} retained-mode count differs")
        common.require(len(set(representatives.tolist())) == count,
                       f"{setting_id} anchor {row_index} repeats a representative")
        common.require(np.all((representatives >= 0) & (representatives < 24)),
                       f"{setting_id} anchor {row_index} representative range differs")
        branch_rows: list[tuple[int, int, bytes, Any]] = []
        for representative in representatives.tolist():
            common.require(bool(present[representative]),
                           f"{setting_id} selected representative is not a present neighbor")
            common.require(int(row["fut_valid"][representative]) == 1,
                           f"{setting_id} selected representative is not future-valid")
            horizon_index = int(row["horizon_idx"][representative])
            common.require(0 <= horizon_index < len(LOGGED_HORIZONS),
                           f"{setting_id} selected horizon index differs")
            logged_horizon = LOGGED_HORIZONS[horizon_index]
            length = min(PREFIX_CAP, logged_horizon)
            common.require(length == PREFIX_CAP,
                           f"{setting_id} prefix cap is not covered by the logged horizon")
            mask = (np.arange(PREFIX_WIDTH, dtype=np.int64) < length).astype("u1")
            branch_rows.append((logged_horizon, length, mask.tobytes(), mask))
        branch_rows.sort(key=lambda value: (value[0], value[1], value[2]))
        counts[row_index] = count
        for branch_index, (logged_horizon, length, _bytes, mask) in enumerate(branch_rows):
            logged_horizons[row_index, branch_index] = logged_horizon
            lengths[row_index, branch_index] = length
            masks[row_index, branch_index] = mask
            action_steps += length
            branch_count += 1
    common.require(len(anchors) == ANCHOR_COUNT and bool(np.all(counts > 0)),
                   f"{setting_id} does not cover every one of 5,120 anchors")
    common.require(branch_count == int(counts.sum()), f"{setting_id} branch denominator differs")
    prefix_histogram = {
        str(length): int((lengths == length).sum())
        for length in range(1, PREFIX_CAP + 1)
    }
    horizon_histogram = {
        str(horizon): int((logged_horizons == horizon).sum())
        for horizon in LOGGED_HORIZONS
    }
    sampler = _sampler_summary(positions, len(records))
    target = {
        "setting_id": setting_id,
        "validation_manifest_sha256": validation_manifest_sha256,
        "future_recipe_sha256": future_recipe_sha256,
        "validation_population": len(records),
        "validation_sample_seed": VALIDATION_SAMPLE_SEED,
        "fixed_validation_sampler": sampler,
        "batch_size": BATCH_SIZE,
        "num_batches": VALIDATION_BATCHES,
        "anchor_count": ANCHOR_COUNT,
        "positions_sha256": common.array_sha256(positions),
        "anchors_sha256": common.array_sha256(anchors),
        "branches_per_anchor_sha256": common.array_sha256(counts),
        "sorted_prefix_lengths_sha256": common.array_sha256(lengths),
        "sorted_logged_selected_horizons_sha256": common.array_sha256(logged_horizons),
        "sorted_prefix_masks_hmax64_sha256": common.array_sha256(masks),
        "matched_branch_count": branch_count,
        "prefix_length_histogram": prefix_histogram,
        "logged_selected_horizon_histogram": horizon_histogram,
        "prefix_action_step_count": action_steps,
        "action_dim": action_dim,
        "prefix_action_scalar_count": action_steps * action_dim,
        "all_anchors_have_match": True,
    }
    target["validation_sample_sha256"] = common.stable_hash({
        "setting_id": setting_id,
        "validation_manifest_sha256": validation_manifest_sha256,
        "validation_sample_seed": VALIDATION_SAMPLE_SEED,
        "batch_size": BATCH_SIZE,
        "num_batches": VALIDATION_BATCHES,
        "positions_sha256": target["positions_sha256"],
        "anchors_sha256": target["anchors_sha256"],
    })
    target["target_contract_sha256"] = common.stable_hash(target)
    return target


def _future_setting(
    future_lock: Mapping[str, Any], setting_id: str, expected_ids: set[str]
) -> Mapping[str, Any]:
    rows = future_lock.get("settings")
    common.require(isinstance(rows, dict) and set(rows) == expected_ids,
                   "future lock settings key set differs")
    row = rows[setting_id]
    common.require(isinstance(row, dict), f"future lock {setting_id} row differs")
    return row


def audit_setting(
    *,
    contract_root: common.SecureRoot,
    setting: Mapping[str, Any],
    future_row: Mapping[str, Any],
) -> dict[str, Any]:
    setting_id = setting["id"]
    contract_value, _contract_inventory = contract_root.read_json(f"data/{setting_id}.json")
    contract = input_contract_audit.validate_contract(contract_value, setting)
    common.require(future_row.get("future_recipe_sha256") == setting["future_recipe_sha256"],
                   f"{setting_id} future lock recipe identity differs")
    common.require(future_row.get("calibration_sha256") == setting["calibration_sha256"],
                   f"{setting_id} future lock calibration identity differs")
    split = future_row.get("splits", {}).get("val")
    common.require(isinstance(split, dict), f"{setting_id} future lock validation split differs")
    with contract_root.subroot(
        PurePosixPath("future-recipes") / setting_id / "val",
        f"{setting_id} prefix validation recipe root",
    ) as recipe_root:
        recipe_root.require_exact_tree(
            files=(".build.lock", "manifest.json", "records.npy"), directories=()
        )
        with recipe_root.open_regular(
            ".build.lock", f"{setting_id} validation build lock"
        ) as lock_source:
            common.require(
                lock_source.size == 0,
                f"{setting_id} validation build lock is not empty",
            )
            lock_inventory = common.inventory_row(lock_source)
        expected_lock_inventory = _validate_inventory_row(
            future_row["inventory"]["val/.build.lock"],
            f"{setting_id} future-lock validation build-lock inventory",
        )
        common.require(
            common.canonical_json(lock_inventory).encode("ascii")
            == common.canonical_json(expected_lock_inventory).encode("ascii"),
            f"{setting_id} validation build-lock stat differs from future lock",
        )
        child_value, child_inventory = recipe_root.read_json("manifest.json")
        expected_child_inventory = _validate_inventory_row(
            future_row["inventory"]["val/manifest.json"],
            f"{setting_id} future-lock validation manifest inventory",
        )
        common.require(common.canonical_json(child_inventory).encode("ascii")
                       == common.canonical_json(expected_child_inventory).encode("ascii"),
                       f"{setting_id} validation manifest bytes differ from future lock")
        common.require(child_value.get("recipe_sha256") == split.get("recipe_sha256")
                       and child_value.get("records_sha256") == split.get("records_sha256"),
                       f"{setting_id} validation child identity differs from future lock")
        with recipe_root.open_regular("records.npy", f"{setting_id} validation records") as source:
            digest = source.sha256()
            common.require(digest == split["records_sha256"],
                           f"{setting_id} validation records differ from future lock")
            expected_inventory = _validate_inventory_row(
                future_row["inventory"]["val/records.npy"],
                f"{setting_id} future-lock validation records inventory",
            )
            common.require(
                common.canonical_json(common.inventory_row(source, digest=digest)).encode("ascii")
                == common.canonical_json(expected_inventory).encode("ascii"),
                           f"{setting_id} validation records stat differs from future lock")
            with common.StableNpy(source, f"{setting_id} validation records") as records:
                target = derive_target(
                    records.array,
                    setting_id=setting_id,
                    action_dim=contract["action_dim"],
                    validation_manifest_sha256=contract["validation_manifest_sha256"],
                    future_recipe_sha256=setting["future_recipe_sha256"],
                )
    target["validation_records_sha256"] = split["records_sha256"]
    target["future_recipe_audit_inventory_sha256"] = future_row["inventory_sha256"]
    target["target_contract_sha256"] = common.stable_hash(
        {key: value for key, value in target.items() if key != "target_contract_sha256"}
    )
    return target


def run(
    *,
    project_root_path: Path,
    contract_root_path: Path,
    future_lock: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]] = common.SETTINGS,
    require_all_ten: bool = True,
) -> dict[str, Any]:
    import torch

    settings = tuple(dict(row) for row in ledger)
    if require_all_ten:
        common.require(
            common.canonical_json(list(settings)).encode("ascii")
            == common.canonical_json(list(common.SETTINGS)).encode("ascii"),
            "prefix audit is not using finalized all-ten ledger",
        )
    validated_future = common.validate_sealed_result(
        future_lock, audit_id=future_recipe_audit.AUDIT_ID, status=future_recipe_audit.STATUS
    )
    common.require(validated_future.get("setting_order") == [row["id"] for row in settings],
                   "future lock setting order differs")
    sampler_path = project_root_path / SAMPLER_SOURCE_RELATIVE
    common.require(common.source_file_sha256(sampler_path) == SAMPLER_SOURCE_SHA256,
                   "fixed representative sampler source bytes differ")
    with common.SecureRoot(contract_root_path, "registered compatible-contract root") as contract_root:
        rows = {
            setting["id"]: audit_setting(
                contract_root=contract_root,
                setting=setting,
                future_row=_future_setting(
                    validated_future, setting["id"], {row["id"] for row in settings}
                ),
            )
            for setting in settings
        }
    common.require(all(row["anchor_count"] == ANCHOR_COUNT for row in rows.values()),
                   "one or more settings lack exactly 5,120 anchors")
    body = {
        "classification": "outcome_blind_read_only_exact_all_ten_fixed_validation_recipe_target_derivation",
        "setting_order": [row["id"] for row in settings],
        "settings": rows,
        "all_settings_count": len(rows),
        "anchors_per_setting": ANCHOR_COUNT,
        "total_anchor_count": ANCHOR_COUNT * len(rows),
        "validation_sample_seed": VALIDATION_SAMPLE_SEED,
        "batch_size": BATCH_SIZE,
        "validation_batches": VALIDATION_BATCHES,
        "prefix_width": PREFIX_WIDTH,
        "prefix_cap": PREFIX_CAP,
        "logged_horizons": list(LOGGED_HORIZONS),
        "future_recipe_audit_artifact_sha256": validated_future["artifact_sha256"],
        "sampler_source_sha256": SAMPLER_SOURCE_SHA256,
        "torch_version": torch.__version__,
        "ledger_sha256": common.stable_hash(list(settings)),
        "target_root_sha256": common.stable_hash({
            setting_id: row["target_contract_sha256"] for setting_id, row in rows.items()
        }),
        "source_sha256": common.source_file_sha256(Path(__file__)),
    }
    return common.seal_result(body, audit_id=AUDIT_ID, status=STATUS)


def _read_lock(path: Path, label: str) -> dict[str, Any]:
    with common.SecureRoot(path.absolute().parent, f"{label} parent") as root:
        value, _row = root.read_json(path.name, label)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--contract-root", type=Path, required=True)
    parser.add_argument("--future-lock", type=Path, required=True)
    parser.add_argument("--expected-lock", type=Path)
    args = parser.parse_args()
    try:
        future_lock = _read_lock(args.future_lock, "future-recipe audit lock")
        result = run(
            project_root_path=args.project_root.absolute(),
            contract_root_path=args.contract_root.absolute(),
            future_lock=future_lock,
        )
        if args.expected_lock is not None:
            lock = _read_lock(args.expected_lock, "prefix-target audit lock")
            common.validate_or_compare_lock(result, lock, audit_id=AUDIT_ID, status=STATUS)
    except Exception as exc:
        print(f"prefix target audit failed: {exc}", file=sys.stderr)
        return 1
    print("EXP24_PREFIX_TARGET_AUDIT=" + common.canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
