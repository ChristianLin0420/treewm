#!/usr/bin/env python3
"""Print the outcome-blind fixed-validation executable-prefix target contract.

The audit reads the immutable published-union validation recipes used by Exp20 and
selects the exact Exp23 validation sample.  It never builds a model, reads an outcome,
performs a rollout, or writes an artifact.  Per-anchor mode counts, sorted prefix
lengths, and full 64-wide masks are byte-hashed so a report cannot establish its own
denominator from an arbitrary nonempty subset.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
from typing import Any


sys.dont_write_bytecode = True
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
WEIGHT_AUDIT = PACKAGE_DIR / "weight_audit.py"
SETTINGS = (
    "antmaze-large",
    "scene",
    "puzzle-3x3",
    "puzzle-4x4-100m",
    "cube-quadruple-100m",
)
VALIDATION_SAMPLE_SEED = 1701
BATCH_SIZE = 256
VALIDATION_BATCHES = 20
PREFIX_WIDTH = 64
PREFIX_CAP = 4
LOGGED_HORIZONS = (4, 8, 16, 32, 64)


class PrefixTargetAuditError(RuntimeError):
    pass


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _array_hash(array: Any) -> str:
    value = array.copy(order="C")
    header = canonical_json(
        {"dtype": value.dtype.str, "shape": list(value.shape)}
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _load_weight_audit():
    spec = importlib.util.spec_from_file_location("exp23_weight_audit", WEIGHT_AUDIT)
    if spec is None or spec.loader is None:
        raise PrefixTargetAuditError("cannot load sealed weight-audit helpers")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def run(project_root: Path) -> dict[str, Any]:
    # Importing this module first also enforces its pre-NumPy/torch thread contract.
    audit = _load_weight_audit()
    import numpy as np
    import torch
    from treewm.data.samplers import FixedRepresentativeSampler

    if any(os.environ.get(name) != "1" for name in audit.DETERMINISM_ENVIRONMENT):
        raise PrefixTargetAuditError("weight-audit numerical environment is not sealed")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise PrefixTargetAuditError("torch thread pools are not single-threaded")

    output_root = project_root / "outputs/treewm-grounded-gauge-pilot-v2-launch2"
    rows: dict[str, Any] = {}
    for setting in SETTINGS:
        candidates = sorted(
            (output_root / setting / "treewm").glob("*armgs-seed108")
        )
        if len(candidates) != 1:
            raise PrefixTargetAuditError(f"{setting}: sealed Exp20 GS source is missing")
        run_dir = candidates[0]
        launch = audit.read_json(run_dir / "GAUGE_PILOT_V2_LAUNCH.json")
        try:
            payload = torch.load(
                run_dir / "checkpoints/latest.pt",
                map_location="cpu", weights_only=False, mmap=True,
            )
        except TypeError:
            payload = torch.load(
                run_dir / "checkpoints/latest.pt", map_location="cpu",
                weights_only=False,
            )
        bound = audit.read_json(PACKAGE_DIR / "weight_audit.lock.json")[
            "action_bounds"
        ][setting]
        cfg = audit.prepare_cfg(
            payload["config"], seed=110,
            lower=float(bound["lower"]), upper=float(bound["upper"]),
        )
        candidate_horizons = tuple(int(value) for value in cfg.future_sets.horizons)
        if candidate_horizons != LOGGED_HORIZONS:
            raise PrefixTargetAuditError(f"{setting}: sealed continuation horizons differ")
        environment = {
            **{str(key): str(value) for key, value in launch["environment"].items()},
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "WANDB_MODE": "disabled",
        }
        with audit.patched_environment(environment):
            train_ds, val_ds, _normalizer, _domain, identity = (
                audit.load_read_only_datasets(cfg, launch)
            )
            del train_ds
            sampler = FixedRepresentativeSampler(
                val_ds,
                batch_size=BATCH_SIZE,
                num_batches=VALIDATION_BATCHES,
                seed=VALIDATION_SAMPLE_SEED,
            )
            positions = sampler.global_indices.cpu().numpy().astype("<i8", copy=False)
            anchors = np.empty((len(positions),), dtype="<i8")
            counts = np.zeros((len(positions),), dtype="u1")
            lengths = np.zeros((len(positions), 4), dtype="u1")
            logged_horizons = np.zeros((len(positions), 4), dtype="u1")
            masks = np.zeros((len(positions), 4, PREFIX_WIDTH), dtype="u1")
            action_steps = 0
            for row_index, position in enumerate(positions.tolist()):
                sample = val_ds[int(position)]
                anchors[row_index] = int(sample["anchor_index"].item())
                valid = sample["mode_valid"].bool()
                reps = sample["mode_rep"].long()[valid]
                count = int(reps.numel())
                if not 1 <= count <= 4:
                    raise PrefixTargetAuditError(
                        f"{setting}: anchor {row_index} has {count} retained modes"
                    )
                branch_rows: list[tuple[int, int, bytes, np.ndarray]] = []
                for representative in reps.tolist():
                    horizon_index = int(sample["fut_horizon_idx"][representative].item())
                    if not 0 <= horizon_index < len(candidate_horizons):
                        raise PrefixTargetAuditError(
                            f"{setting}: logged continuation horizon index is invalid"
                        )
                    logged_horizon = candidate_horizons[horizon_index]
                    if logged_horizon not in LOGGED_HORIZONS:
                        raise PrefixTargetAuditError(
                            f"{setting}: impossible logged continuation horizon"
                        )
                    length = int(
                        sample["fut_executable_prefix_len"][representative].item()
                    )
                    mask = (
                        sample["fut_executable_prefix_action_mask"][representative]
                        .bool().cpu().numpy().astype("u1", copy=False)
                    )
                    if (
                        mask.shape != (PREFIX_WIDTH,)
                        or not 1 <= length <= PREFIX_CAP
                        or not np.array_equal(
                            mask,
                            (np.arange(PREFIX_WIDTH) < length).astype("u1"),
                        )
                    ):
                        raise PrefixTargetAuditError(
                            f"{setting}: malformed executable prefix target"
                        )
                    if length != min(PREFIX_CAP, logged_horizon):
                        raise PrefixTargetAuditError(
                            f"{setting}: prefix length does not match logged horizon"
                        )
                    branch_rows.append((logged_horizon, length, mask.tobytes(), mask))
                branch_rows.sort(key=lambda value: (value[0], value[1], value[2]))
                counts[row_index] = count
                for branch_index, (logged_horizon, length, _raw, mask) in enumerate(branch_rows):
                    logged_horizons[row_index, branch_index] = logged_horizon
                    lengths[row_index, branch_index] = length
                    masks[row_index, branch_index] = mask
                    action_steps += length

            action_dim = int(bound["action_dim"])
            histogram = {
                str(length): int((lengths == length).sum())
                for length in range(1, PREFIX_CAP + 1)
            }
            horizon_histogram = {
                str(horizon): int((logged_horizons == horizon).sum())
                for horizon in LOGGED_HORIZONS
            }
            validation_manifest = str(
                launch["hashes"]["validation_manifest_sha256"]
            )
            target = {
                "setting_id": setting,
                "validation_manifest_sha256": validation_manifest,
                "future_recipe_sha256": identity["future_recipe_sha256"],
                "validation_population": int(len(val_ds)),
                "validation_sample_seed": VALIDATION_SAMPLE_SEED,
                "fixed_validation_sampler": sampler.summary(),
                "batch_size": BATCH_SIZE,
                "num_batches": VALIDATION_BATCHES,
                "anchor_count": int(len(positions)),
                "positions_sha256": _array_hash(positions),
                "anchors_sha256": _array_hash(anchors),
                "branches_per_anchor_sha256": _array_hash(counts),
                "sorted_prefix_lengths_sha256": _array_hash(lengths),
                "sorted_logged_selected_horizons_sha256": _array_hash(logged_horizons),
                "sorted_prefix_masks_hmax64_sha256": _array_hash(masks),
                "matched_branch_count": int(counts.sum()),
                "prefix_length_histogram": histogram,
                "logged_selected_horizon_histogram": horizon_histogram,
                "prefix_action_step_count": int(action_steps),
                "action_dim": action_dim,
                "prefix_action_scalar_count": int(action_steps * action_dim),
                "all_anchors_have_match": bool((counts > 0).all()),
            }
            target["validation_sample_sha256"] = stable_hash(
                {
                    "setting_id": setting,
                    "validation_manifest_sha256": validation_manifest,
                    "validation_sample_seed": VALIDATION_SAMPLE_SEED,
                    "batch_size": BATCH_SIZE,
                    "num_batches": VALIDATION_BATCHES,
                    "positions_sha256": target["positions_sha256"],
                    "anchors_sha256": target["anchors_sha256"],
                }
            )
            target["target_contract_sha256"] = stable_hash(target)
            rows[setting] = target
            del val_ds

    result = {
        "schema_version": 1,
        "status": "frozen_outcome_blind_fixed_validation_prefix_targets",
        "audit_id": "treewm_exp23_prefix_target_contract_v1",
        "settings": rows,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "weight_audit_artifact_sha256": audit.read_json(
            PACKAGE_DIR / "weight_audit.lock.json"
        )["result_identity"]["artifact_sha256"],
    }
    result["artifact_sha256"] = stable_hash(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    try:
        result = run(args.project_root.resolve())
    except Exception as exc:
        print(f"prefix target audit failed: {exc}", file=sys.stderr)
        return 1
    print("EXP23_PREFIX_TARGET_AUDIT=" + canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
