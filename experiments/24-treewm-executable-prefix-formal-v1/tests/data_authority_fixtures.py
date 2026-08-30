from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

import data_authority_common as common
import future_recipe_audit
import input_contract_audit


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base_fixture(root: Path) -> dict[str, Any]:
    data_root = root / "source"
    cache_root = root / "cache"
    contract_root = root / "contracts"
    standard = data_root / "standard"
    cache_dir = cache_root / "toy-v0__0123456789abcdef"
    standard.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    (contract_root / "data").mkdir(parents=True)

    source_rows = []
    processed: dict[str, dict[str, np.ndarray]] = {}
    for split, suffix, shift in (("train", "", 0.0), ("val", "-val", 100.0)):
        observations = (np.arange(12, dtype=np.float32).reshape(6, 2) + shift)
        actions = (np.arange(6, dtype=np.float32).reshape(6, 1) / 10 + shift)
        terminals = np.asarray([False, False, True, False, False, True], dtype=np.bool_)
        path = standard / f"toy-v0{suffix}.npz"
        np.savez(
            path,
            observations=observations,
            actions=actions,
            terminals=terminals,
            qpos=observations.copy(),
            qvel=observations.copy(),
        )
        info = path.stat()
        source_rows.append({
            "index": None,
            "split": split,
            "path": f"standard/{path.name}",
            "size": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "sha256": _sha(path),
        })
        mask = ~terminals
        terminal_projection = np.zeros(int(mask.sum()), dtype=np.bool_)
        terminal_projection[[1, 3]] = True
        processed[split] = {
            "obs": observations[mask],
            "act": actions[mask],
            "terminals": terminal_projection,
        }

    train = processed["train"]
    norm = {
        "obs_mean": train["obs"].mean(0).astype(np.float32),
        "obs_std": (train["obs"].std(0) + 1e-6).astype(np.float32),
        "act_mean": np.zeros(1, dtype=np.float32),
        "act_std": (train["act"].std(0) + 1e-6).astype(np.float32),
    }
    for split in ("train", "val"):
        row = processed[split]
        np.save(cache_dir / f"{split}_obs.npy", row["obs"].astype(np.float32))
        np.save(cache_dir / f"{split}_act.npy", row["act"].astype(np.float32))
        np.save(cache_dir / f"{split}_terminals.npy", row["terminals"].astype(np.float32))
        np.save(cache_dir / f"{split}_obs_norm.npy",
                ((row["obs"] - norm["obs_mean"]) / norm["obs_std"]).astype(np.float32))
        np.save(cache_dir / f"{split}_act_norm.npy",
                ((row["act"] - norm["act_mean"]) / norm["act_std"]).astype(np.float32))

    cache_source_rows = []
    for row in source_rows:
        cache_row = {**row, "path": Path(row["path"]).name}
        cache_row.pop("index")
        cache_source_rows.append(cache_row)
    stat_rows = [{key: value for key, value in row.items() if key != "sha256"}
                 for row in cache_source_rows]
    manifest = {
        "version": 4,
        "dataset": "toy-v0",
        "key": "0123456789abcdef",
        "recipe": {
            "norm": "mean_std", "eps": 1e-6, "dtype": "float32",
            "source_stat_sha256": common.stable_hash(stat_rows),
        },
        "built_s": 0.1,
        "norm_stats": {key: value.tolist() for key, value in norm.items()},
        "shapes": {"train_obs": [4, 2], "val_obs": [4, 2]},
        "source_files": cache_source_rows,
        "source_manifest_sha256": common.stable_hash(cache_source_rows),
    }
    cache_manifest_path = cache_dir / "manifest.json"
    _write_json(cache_manifest_path, manifest)
    chosen = {
        "cluster_threshold": 1.0,
        "displacement_threshold": 1.0,
        "retrieval_radius": 1.0,
    }
    contract = {
        "action_dim": 1,
        "calibration_path": str((contract_root / "calibration/toy.json").absolute()),
        "calibration_sha256": "a" * 64,
        "campaign_id": common.CAMPAIGN_ID,
        "campaign_protocol_sha256": common.CAMPAIGN_PROTOCOL_SHA256,
        "chosen_thresholds": chosen,
        "data_manifest_sha256": common.stable_hash(cache_source_rows),
        "dataset_kind": "standard",
        "env_name": "toy-v0",
        "future_recipe_manifest": str((contract_root / "future-recipes/toy/manifest.json").absolute()),
        "future_recipe_sha256": "b" * 64,
        "normalizer_sha256": input_contract_audit._normalizer_sha256(manifest["norm_stats"]),
        "objective_version": common.OBJECTIVE_VERSION,
        "obs_dim": 2,
        "raw_cache_manifest": str(cache_manifest_path.absolute()),
        "raw_cache_manifest_file_sha256": _sha(cache_manifest_path),
        "raw_cache_read_only": True,
        "schema_version": 2,
        "setting_id": "toy",
        "source_files": source_rows,
        "source_name": "toy-v0",
        "status": "complete",
        "train_manifest_sha256": "1" * 64,
        "train_recipe_rows": 4,
        "train_trajectories": 2,
        "train_transitions": 4,
        "validation_manifest_sha256": "2" * 64,
        "validation_recipe_rows": 4,
        "validation_trajectories": 2,
        "validation_transitions": 4,
    }
    contract["contract_sha256"] = common.stable_hash(contract)
    _write_json(contract_root / "data/toy.json", contract)
    setting = {
        "id": "toy", "env_name": "toy-v0", "source_name": "toy-v0",
        "dataset_kind": "standard", "data_subdir": "standard", "expected_shards": None,
        "task_metric_dims": [0], "relative_endpoints": True,
        "input_contract_sha256": contract["contract_sha256"],
        "calibration_sha256": contract["calibration_sha256"],
        "future_recipe_sha256": contract["future_recipe_sha256"],
        "published_union_train_anchors": contract["train_recipe_rows"],
        "published_union_validation_anchors": contract["validation_recipe_rows"],
    }
    return {
        "data_root": data_root,
        "cache_root": cache_root,
        "cache_dir": cache_dir,
        "contract_root": contract_root,
        "contract": contract,
        "setting": setting,
    }


def build_input_fixture(root: Path) -> dict[str, Any]:
    return _base_fixture(root)


def build_sharded_input_fixture(root: Path, shard_count: int = 2) -> dict[str, Any]:
    """Build a small, semantically real sharded source/cache/contract authority."""
    data_root = root / "source"
    data_subdir = Path("100m/toy-sharded-v0")
    source_dir = data_root / data_subdir
    cache_root = root / "cache"
    cache_dir = cache_root / "toy-sharded-v0__0123456789abcdef0123"
    contract_root = root / "contracts"
    source_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    (contract_root / "data").mkdir(parents=True)

    source_rows: list[dict[str, Any]] = []
    projected: dict[str, list[dict[str, np.ndarray]]] = {"train": [], "val": []}
    for index in range(shard_count):
        for split in ("train", "val"):
            shift = float(index * 1000 + (100 if split == "val" else 0))
            observations = np.arange(12, dtype=np.float32).reshape(6, 2) + shift
            actions = np.arange(6, dtype=np.float32).reshape(6, 1) / 10 + shift
            terminals = np.asarray([False, False, True, False, False, True], dtype=np.bool_)
            filename = f"toy-sharded-v0-{index:03d}-{split}.npz"
            path = source_dir / filename
            np.savez(
                path,
                observations=observations,
                actions=actions,
                terminals=terminals,
                qpos=observations.copy(),
                qvel=observations.copy(),
            )
            info = path.stat()
            source_rows.append({
                "index": index,
                "split": split,
                "path": (data_subdir / filename).as_posix(),
                "size": info.st_size,
                "mtime_ns": info.st_mtime_ns,
                "sha256": _sha(path),
            })
            mask = ~terminals
            projected[split].append({
                "obs": observations[mask],
                "act": actions[mask],
                "terminals": np.asarray([False, True, False, True], dtype=np.bool_),
            })

    combined: dict[str, dict[str, np.ndarray]] = {}
    for split in ("train", "val"):
        obs = np.concatenate([row["obs"] for row in projected[split]], axis=0)
        act = np.concatenate([row["act"] for row in projected[split]], axis=0)
        terminals = np.concatenate([row["terminals"] for row in projected[split]], axis=0)
        trajectory_count = shard_count * 2
        lengths = np.full(trajectory_count, 2, dtype=np.int32)
        starts = np.arange(trajectory_count, dtype=np.int64) * 2
        traj_id = np.repeat(np.arange(trajectory_count, dtype=np.int32), 2)
        remaining = np.tile(np.asarray([1, 0], dtype=np.int32), trajectory_count)
        combined[split] = {
            "obs": obs.astype(np.float32),
            "act": act.astype(np.float32),
            "terminals": terminals,
            "starts": starts,
            "lengths": lengths,
            "traj_id": traj_id,
            "remaining": remaining,
        }

    train = combined["train"]
    norm = {
        "obs_mean": train["obs"].mean(0).astype(np.float32),
        "obs_std": (train["obs"].std(0) + 1e-6).astype(np.float32),
        "act_mean": np.zeros(1, dtype=np.float32),
        "act_std": (train["act"].std(0) + 1e-6).astype(np.float32),
    }
    for split in ("train", "val"):
        row = combined[split]
        np.save(cache_dir / f"{split}_obs.npy", row["obs"])
        np.save(cache_dir / f"{split}_act.npy", row["act"])
        np.save(cache_dir / f"{split}_terminals.npy", row["terminals"])
        np.save(
            cache_dir / f"{split}_obs_norm.npy",
            ((row["obs"] - norm["obs_mean"]) / norm["obs_std"]).astype(np.float32),
        )
        np.save(
            cache_dir / f"{split}_act_norm.npy",
            ((row["act"] - norm["act_mean"]) / norm["act_std"]).astype(np.float32),
        )
        for field in ("traj_id", "remaining", "starts", "lengths"):
            np.save(cache_dir / f"{split}_{field}.npy", row[field])

    cache_source_rows = []
    for row in source_rows:
        cache_row = dict(row)
        cache_row["path"] = Path(row["path"]).name
        cache_source_rows.append(cache_row)
    stat_rows = [
        {key: value for key, value in row.items() if key != "sha256"}
        for row in cache_source_rows
    ]
    transitions = shard_count * 4
    trajectories = shard_count * 2
    manifest = {
        "schema_version": 1,
        "dataset_name": "toy-sharded-v0",
        "source_dataset": "toy-sharded-v0",
        "cache_key": "0123456789abcdef0123",
        "recipe": {
            "cache_version": 2,
            "dataset_name": "toy-sharded-v0",
            "dtype": "float32",
            "eps": 1e-6,
            "expected_shards": shard_count,
            "normalization": "global_train_mean_std",
            "regular_dataset_semantics": True,
            "shard_file_stem": "toy-sharded-v0",
            "source_directory": str(source_dir.absolute()),
            "source_stat_sha256": common.stable_hash(stat_rows),
        },
        "norm_stats": {key: value.tolist() for key, value in norm.items()},
        "source_files": cache_source_rows,
        "source_manifest_sha256": common.stable_hash(cache_source_rows),
        "train_shards": shard_count,
        "validation_shards": shard_count,
        "train_transitions": transitions,
        "validation_transitions": transitions,
        "train_trajectories": trajectories,
        "validation_trajectories": trajectories,
        "obs_dim": 2,
        "action_dim": 1,
    }
    cache_manifest_path = cache_dir / "manifest.json"
    _write_json(cache_manifest_path, manifest)
    chosen = {
        "cluster_threshold": 1.0,
        "displacement_threshold": 1.0,
        "retrieval_radius": 1.0,
    }
    contract = {
        "action_dim": 1,
        "calibration_path": str((contract_root / "calibration/toy-sharded.json").absolute()),
        "calibration_sha256": "a" * 64,
        "campaign_id": common.CAMPAIGN_ID,
        "campaign_protocol_sha256": common.CAMPAIGN_PROTOCOL_SHA256,
        "chosen_thresholds": chosen,
        "data_manifest_sha256": common.stable_hash(cache_source_rows),
        "dataset_kind": "sharded_100m_full",
        "env_name": "toy-sharded-v0",
        "future_recipe_manifest": str(
            (contract_root / "future-recipes/toy-sharded/manifest.json").absolute()
        ),
        "future_recipe_sha256": "b" * 64,
        "normalizer_sha256": input_contract_audit._normalizer_sha256(manifest["norm_stats"]),
        "objective_version": common.OBJECTIVE_VERSION,
        "obs_dim": 2,
        "raw_cache_manifest": str(cache_manifest_path.absolute()),
        "raw_cache_manifest_file_sha256": _sha(cache_manifest_path),
        "raw_cache_read_only": True,
        "schema_version": 2,
        "setting_id": "toy-sharded",
        "source_files": source_rows,
        "source_name": "toy-sharded-v0",
        "status": "complete",
        "train_manifest_sha256": "1" * 64,
        "train_recipe_rows": transitions,
        "train_trajectories": trajectories,
        "train_transitions": transitions,
        "validation_manifest_sha256": "2" * 64,
        "validation_recipe_rows": transitions,
        "validation_trajectories": trajectories,
        "validation_transitions": transitions,
    }
    contract["contract_sha256"] = common.stable_hash(contract)
    _write_json(contract_root / "data/toy-sharded.json", contract)
    setting = {
        "id": "toy-sharded",
        "env_name": "toy-sharded-v0",
        "source_name": "toy-sharded-v0",
        "dataset_kind": "sharded_100m_full",
        "data_subdir": data_subdir.as_posix(),
        "expected_shards": shard_count,
        "task_metric_dims": [0],
        "relative_endpoints": True,
        "input_contract_sha256": contract["contract_sha256"],
        "calibration_sha256": contract["calibration_sha256"],
        "future_recipe_sha256": contract["future_recipe_sha256"],
        "published_union_train_anchors": contract["train_recipe_rows"],
        "published_union_validation_anchors": contract["validation_recipe_rows"],
    }
    return {
        "data_root": data_root,
        "cache_root": cache_root,
        "cache_dir": cache_dir,
        "contract_root": contract_root,
        "contract": contract,
        "setting": setting,
    }


def _calibration(setting: dict[str, Any], contract: dict[str, Any]) -> dict[str, Any]:
    samples = list(range(4096))
    pool = list(range(50000))
    selected = list(range(4096))
    offsets = list(range(4097))
    chosen = contract["chosen_thresholds"]
    config = json.loads(common.canonical_json(future_recipe_audit.FORMAL_CALIBRATION_CONFIG))
    horizon_histogram = [820, 819, 819, 819, 819]
    gates = {
        "eligible_23rd_neighbor": {"passed": True, "rule": "<= 0.1", "value": 0.0},
        "mean_retained_modes": {
            "passed": True, "rule": "in [1.5, 3.5]", "value": 2.0,
        },
        "mean_retrieved": {"passed": True, "rule": ">= 18.0", "value": 24.0},
        "mode_truncation_fraction": {
            "passed": True, "rule": "<= 0.05", "value": 0.0,
        },
        "multimode_anchor_fraction": {
            "passed": True, "rule": ">= 0.4", "value": 1.0,
        },
        "normalized_horizon_entropy": {
            "passed": True, "rule": ">= 0.65", "value": 1.0,
        },
        "occupied_horizon_classes": {"passed": True, "rule": ">= 4", "value": 5.0},
        "retrieval_fallback_fraction": {
            "passed": True, "rule": "<= 0.01", "value": 0.0,
        },
    }
    value = {
        "schema_version": 1,
        "algorithm_version": "synthetic-v1",
        "implementation_sha256": "c" * 64,
        "numpy_version": np.__version__,
        "scipy_version": "synthetic",
        "status": "complete",
        "setting_id": setting["id"],
        "split": "train",
        "train_manifest_sha256": contract["train_manifest_sha256"],
        "normalizer_sha256": contract["normalizer_sha256"],
        "input_units": "training-normalizer standardized observations",
        "metric_mode": "rms_v2",
        "relative_endpoints": setting["relative_endpoints"],
        "xy_dims": [0],
        "task_metric_dims": setting["task_metric_dims"],
        "config": config,
        "anchor_population": 50000,
        "sample_seed": 0,
        "sample_indices": samples,
        "sample_indices_sha256": hashlib.sha256(np.asarray(samples, dtype="<i8").tobytes()).hexdigest(),
        "retrieval_pool_indices": pool,
        "retrieval_pool_indices_sha256": hashlib.sha256(np.asarray(pool, dtype="<i8").tobytes()).hexdigest(),
        "eligible_query_count_histogram": {"24": 4096},
        "selected_continuation_count": len(selected),
        "selected_continuation_indices": selected,
        "selected_continuation_indices_sha256": hashlib.sha256(np.asarray(selected, dtype="<i8").tobytes()).hexdigest(),
        "selected_continuation_offsets": offsets,
        "retrieval": {
            "chosen": {
                "fallback_fraction": 0.0, "max_retrieved": 24,
                "mean_retrieved": 24.0, "min_retrieved": 24,
                "radius": chosen["retrieval_radius"], "retrieved_histogram": {"24": 4096},
            },
            "chosen_radius": chosen["retrieval_radius"],
            "fallback_component": {
                "distance_count": 4096, "distances_sha256": "d" * 64,
                "order_statistic": 1, "quantile": 0.99,
                "radius": chosen["retrieval_radius"],
            },
            "insufficient_anchor_count": 0,
            "insufficient_anchor_fraction": 0.0,
            "kth_distance_count": 4096,
            "kth_distance_sha256": "e" * 64,
            "maximum_insufficient_anchor_fraction": 0.1,
            "quantile_method": "higher",
            "query_k": 145,
            "report_candidates": [{
                "fallback_fraction": 0.0, "max_retrieved": 24,
                "mean_retrieved": 24.0, "min_retrieved": 24,
                "quantile": quantile, "radius": chosen["retrieval_radius"],
                "retrieved_histogram": {"24": 4096},
            } for quantile in config["radius_report_quantiles"]],
            "required_nonself_neighbors": 23,
            "rule": (
                "max(higher q=0.90 of the 23th non-self eligible neighbor RMS distance, "
                "higher q=0.99 of nearest eligible non-self RMS distance)"
            ),
            "sufficient_anchor_count": 4096,
            "support_count_component": {
                "distance_count": 4096, "distances_sha256": "f" * 64,
                "order_statistic": 23, "quantile": 0.9,
                "radius": chosen["retrieval_radius"],
            },
        },
        "horizon": {
            "candidate_quantiles": config["horizon_candidate_quantiles"],
            "candidates": [{
                "histogram": horizon_histogram,
                "max_class_fraction": max(horizon_histogram) / sum(horizon_histogram),
                "normalized_entropy": 1.0, "occupied_classes": 5,
                "threshold": chosen["displacement_threshold"],
            }],
            "chosen_histogram": horizon_histogram,
            "chosen_index": 0,
            "chosen_threshold": chosen["displacement_threshold"],
            "normalized_entropy": 1.0,
            "occupied_classes": 5,
            "quantile_method": "higher",
            "rule": future_recipe_audit.HORIZON_RULE,
            "threshold_boundary": future_recipe_audit.HORIZON_BOUNDARY,
        },
        "cluster": {
            "candidate_quantiles": config["cluster_candidate_quantiles"],
            "candidates": [{
                "mean_raw_modes": 2.0, "mean_retained_modes": 2.0,
                "median_retained_modes": 2.0, "multimode_anchor_fraction": 1.0,
                "raw_mode_histogram": {"2": 4096},
                "retained_mode_histogram": {"2": 4096},
                "threshold": chosen["cluster_threshold"], "truncation_fraction": 0.0,
            }],
            "chosen_index": 0,
            "chosen_raw_mode_histogram": {"2": 4096},
            "chosen_retained_mode_histogram": {"2": 4096},
            "chosen_threshold": chosen["cluster_threshold"],
            "had_truncation_feasible_candidate": True,
            "mean_retained_modes": 2.0,
            "median_retained_modes": 2.0,
            "multimode_anchor_fraction": 1.0,
            "quantile_method": "higher",
            "rule": future_recipe_audit.CLUSTER_RULE,
            "truncation_fraction": 0.0,
        },
        "chosen": chosen,
        "gates": gates,
    }
    value["contract_sha256"] = common.stable_hash(value)
    return value


def _record_array(count: int) -> np.ndarray:
    dtype = np.dtype([
        tuple([row[0], row[1], tuple(row[2])]) if len(row) == 3 else tuple(row)
        for row in future_recipe_audit.expected_record_dtype()
    ])
    records = np.zeros(count, dtype=dtype)
    records["anchor"] = np.arange(count, dtype=np.int64)
    records["neighbors"] = -1
    records["neighbors"][:, 0] = records["anchor"]
    records["horizon_idx"][:, 0] = np.arange(count, dtype=np.uint64) % 5
    records["fut_valid"][:, 0] = 1
    records["cluster"] = -1
    records["cluster"][:, 0] = 0
    records["mode_rep"] = -1
    records["mode_rep"][:, 0] = 0
    records["mode_mass"][:, 0] = 1.0
    records["mode_valid"][:, 0] = 1
    records["num_retrieved"] = 1
    records["retrieval_num_candidates"] = 1
    records["retrieval_num_valid"] = 1
    records["modes_raw"] = 1
    records["modes_retained"] = 1
    return records


def _child_manifest(
    path: Path,
    *,
    records: np.ndarray,
    split: str,
    setting: dict[str, Any],
    contract: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    path.mkdir(parents=True)
    (path / ".build.lock").write_bytes(b"")
    records_path = path / "records.npy"
    np.save(records_path, records)
    info = records_path.stat()
    anchors = records["anchor"].astype("<i8", copy=False)
    anchor_hash = hashlib.sha256(anchors.tobytes()).hexdigest()
    identity = {
        "anchor_sets": {
            f"seed{seed}": {"count": len(records), "sha256": anchor_hash}
            for seed in range(4)
        },
        "anchor_union_count": len(records),
        "anchor_union_sha256": anchor_hash,
        "calibration_sha256": calibration["contract_sha256"],
        "chosen_thresholds": contract["chosen_thresholds"],
        "code_sha256": common.RECIPE_CODE_SHA256,
        "future_config": {
            "cluster_method": "average",
            "cluster_threshold": contract["chosen_thresholds"]["cluster_threshold"],
            "displacement_threshold": contract["chosen_thresholds"]["displacement_threshold"],
            "fixed_horizon": 32,
            "h_max": 64,
            "horizon_rule": "displacement",
            "horizons": [4, 8, 16, 32, 64],
            "include_self": True,
            "max_modes": 4,
            "metric_mode": "rms_v2",
            "multi_step_depth": 3,
            "num_neighbors": 24,
            "query_multiplier": 6,
            "relative_endpoints": setting["relative_endpoints"],
            "retrieval_pool": 50000,
            "retrieval_radius": contract["chosen_thresholds"]["retrieval_radius"],
            "time_exclusion": 50,
        },
        "normalizer_sha256": contract["normalizer_sha256"],
        "numpy_version": np.__version__,
        "recipe_version": "treewm_compact_future_recipe_v1",
        "runtime_sha256": common.RECIPE_RUNTIME_SHA256,
        "schema_version": 1,
        "scipy_version": "synthetic",
        "source_manifest_sha256": contract["data_manifest_sha256"],
        "split": split,
        "split_manifest_sha256": contract[
            "train_manifest_sha256" if split == "train" else "validation_manifest_sha256"
        ],
        "task_metric_dims": setting["task_metric_dims"],
        "xy_dims": calibration["xy_dims"],
    }
    child = {
        "schema_version": 1,
        "status": "complete",
        "identity": identity,
        "identity_sha256": common.stable_hash(identity),
        "records_file": "records.npy",
        "records_size": info.st_size,
        "records_mtime_ns": info.st_mtime_ns,
        "records_sha256": _sha(records_path),
        "record_dtype": future_recipe_audit.expected_record_dtype(),
        "record_count": len(records),
    }
    child["recipe_sha256"] = common.stable_hash(child)
    _write_json(path / "manifest.json", child)
    return child


def build_future_fixture(root: Path, count: int = 6000) -> dict[str, Any]:
    fixture = _base_fixture(root)
    contract = fixture["contract"]
    setting = fixture["setting"]
    contract.update({
        "train_recipe_rows": count,
        "validation_recipe_rows": count,
        "train_transitions": max(count + 100, 50100),
        "validation_transitions": max(count + 100, 50100),
        "train_trajectories": 2,
        "validation_trajectories": 2,
    })
    setting["published_union_train_anchors"] = count
    setting["published_union_validation_anchors"] = count
    calibration = _calibration(setting, contract)
    setting["calibration_sha256"] = calibration["contract_sha256"]
    contract["calibration_sha256"] = calibration["contract_sha256"]
    calibration_root = fixture["contract_root"] / "calibration"
    calibration_root.mkdir()
    _write_json(calibration_root / "toy.json", calibration)
    records = _record_array(count)
    recipe_root = fixture["contract_root"] / "future-recipes/toy"
    train = _child_manifest(
        recipe_root / "train", records=records, split="train", setting=setting,
        contract=contract, calibration=calibration,
    )
    val = _child_manifest(
        recipe_root / "val", records=records, split="val", setting=setting,
        contract=contract, calibration=calibration,
    )
    composite = {
        "schema_version": 1,
        "status": "complete",
        "recipe_version": "treewm_compact_future_recipe_v1",
        "train_recipe_sha256": train["recipe_sha256"],
        "validation_recipe_sha256": val["recipe_sha256"],
        "train_manifest": "train/manifest.json",
        "validation_manifest": "val/manifest.json",
        "source_manifest_sha256": contract["data_manifest_sha256"],
        "normalizer_sha256": contract["normalizer_sha256"],
        "calibration_sha256": calibration["contract_sha256"],
        "chosen_thresholds": contract["chosen_thresholds"],
        "train_manifest_sha256": contract["train_manifest_sha256"],
        "validation_manifest_sha256": contract["validation_manifest_sha256"],
        "code_sha256": common.RECIPE_CODE_SHA256,
        "runtime_sha256": common.RECIPE_RUNTIME_SHA256,
    }
    composite["recipe_sha256"] = common.stable_hash(composite)
    _write_json(recipe_root / "manifest.json", composite)
    contract["future_recipe_sha256"] = composite["recipe_sha256"]
    setting["future_recipe_sha256"] = composite["recipe_sha256"]
    contract.pop("contract_sha256", None)
    contract["contract_sha256"] = common.stable_hash(contract)
    setting["input_contract_sha256"] = contract["contract_sha256"]
    _write_json(fixture["contract_root"] / "data/toy.json", contract)
    summary = {
        "campaign_id": common.CAMPAIGN_ID,
        "campaign_protocol_sha256": common.CAMPAIGN_PROTOCOL_SHA256,
        "schema_version": 1,
        "setting_count": 1,
        "settings": [{
            "setting_id": "toy",
            "calibration_sha256": calibration["contract_sha256"],
            "chosen": calibration["chosen"],
            "gates": calibration["gates"],
            "horizon_histogram": calibration["horizon"]["chosen_histogram"],
            "raw_mode_histogram": calibration["cluster"]["chosen_raw_mode_histogram"],
            "retained_mode_histogram": calibration["cluster"]["chosen_retained_mode_histogram"],
            "retrieval": calibration["retrieval"]["chosen"],
        }],
        "status": "complete",
    }
    summary["summary_sha256"] = common.stable_hash(summary)
    _write_json(calibration_root / "ALL_SETTINGS.json", summary)
    fixture.update({
        "contract": contract,
        "setting": setting,
        "calibration": calibration,
        "composite": composite,
        "records": records,
    })
    return fixture
