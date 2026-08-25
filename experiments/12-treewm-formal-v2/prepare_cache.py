#!/usr/bin/env python3
"""Publish train-only calibrations and compact future recipes from the v1 raw cache.

The approved raw cache is opened read-only.  Exp12 writes only beneath its separate
contract root.  Staging is deliberately two phase: all ten calibrations must pass and
be summarized before any future recipe is materialized.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
from typing import Any, Mapping

import numpy as np

from campaign import (
    calibration_contract_path,
    data_contract_path,
    live_contract,
    load_data_contract,
    load_manifest,
    normalizer_sha256,
    protocol_sha256,
    raw_cache_manifest_path,
    recipe_root_path,
    required_dataset_files,
    stable_hash,
    train_inventory_sha256,
    _validate_current_source_files,
)
from calibration import CalibrationConfig, calibrate_future_metrics, validate_contract


GRACEFUL_EXIT_CODE = 75
STOP_REQUESTED = False


class GracefulCacheStop(RuntimeError):
    pass


def _request_stop(signum: int, frame: object) -> None:
    del signum, frame
    global STOP_REQUESTED
    STOP_REQUESTED = True


def _check_stop() -> None:
    if STOP_REQUESTED:
        raise GracefulCacheStop("scheduler requested a safe v2 cache-stage stop")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


@dataclass(frozen=True)
class RawSplit:
    obs_norm: np.ndarray
    act_norm: np.ndarray
    index: Any


@dataclass(frozen=True)
class RawCache:
    path: Path
    manifest: dict[str, Any]
    train: RawSplit
    val: RawSplit


def _standard_index(terminals: np.ndarray):
    from treewm.data.ogbench_dataset import TrajectoryIndex

    return TrajectoryIndex.from_terminals(terminals)


def _sharded_index(path: Path, prefix: str):
    from treewm.data.sharded_ogbench import MemmapTrajectoryIndex

    load = lambda name: np.load(path / f"{prefix}_{name}.npy", mmap_mode="r")
    return MemmapTrajectoryIndex(
        traj_id=load("traj_id"),
        steps_remaining=load("remaining"),
        starts=load("starts"),
        lengths=load("lengths"),
    )


def load_raw_cache_readonly(cache_root: Path, setting: Mapping[str, Any]) -> RawCache:
    """Open the already materialized raw arrays without invoking any cache builder."""
    manifest_path = raw_cache_manifest_path(cache_root, setting)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = manifest_path.parent
    load = lambda split, name: np.load(path / f"{split}_{name}.npy", mmap_mode="r")
    if setting["dataset_kind"] == "sharded_100m_full":
        train_index = _sharded_index(path, "train")
        val_index = _sharded_index(path, "val")
    else:
        train_index = _standard_index(load("train", "terminals"))
        val_index = _standard_index(load("val", "terminals"))
    cache = RawCache(
        path=path,
        manifest=payload,
        train=RawSplit(load("train", "obs_norm"), load("train", "act_norm"), train_index),
        val=RawSplit(load("val", "obs_norm"), load("val", "act_norm"), val_index),
    )
    if cache.train.obs_norm.shape[1] != setting["obs_dim"]:
        raise ValueError(f"raw cache observation dimension drifted for {setting['id']}")
    if cache.train.act_norm.shape[1] != setting["action_dim"]:
        raise ValueError(f"raw cache action dimension drifted for {setting['id']}")
    return cache


def _split_inventory_sha256(
    source_files: list[Mapping[str, Any]], split: str
) -> str:
    entries = [
        {
            "split": split,
            "index": entry.get("index"),
            "path": entry.get("path"),
            "size": entry.get("size"),
            "sha256": entry.get("sha256"),
        }
        for entry in source_files
        if entry.get("split") == split
    ]
    if not entries:
        raise ValueError(f"raw cache has no {split} source inventory")
    return stable_hash(entries)


def _raw_identity(
    manifest: Mapping[str, Any],
    setting: Mapping[str, Any],
    *,
    data_root: Path,
    cache_root: Path,
) -> tuple[RawCache, dict[str, Any]]:
    cache = load_raw_cache_readonly(cache_root, setting)
    source_files = list(cache.manifest.get("source_files") or [])
    current = _validate_current_source_files(manifest, setting, data_root, source_files)
    raw_manifest = cache.path / "manifest.json"
    identity = {
        "data_manifest_sha256": cache.manifest["source_manifest_sha256"],
        "train_manifest_sha256": train_inventory_sha256(source_files),
        "validation_manifest_sha256": _split_inventory_sha256(source_files, "val"),
        "normalizer_sha256": normalizer_sha256(cache.manifest["norm_stats"]),
        "raw_cache_manifest": str(raw_manifest),
        "raw_cache_manifest_file_sha256": hashlib.sha256(raw_manifest.read_bytes()).hexdigest(),
        "source_files": current,
    }
    return cache, identity


def calibrate_setting(
    manifest: Mapping[str, Any],
    setting_index: int,
    *,
    data_root: Path,
    cache_root: Path,
    contract_root: Path,
) -> dict[str, Any]:
    setting = manifest["settings"][setting_index]
    cache, identity = _raw_identity(
        manifest, setting, data_root=data_root, cache_root=cache_root
    )
    path = calibration_contract_path(contract_root, setting["id"])
    if path.is_file():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            validate_contract(
                payload,
                expected_config=CalibrationConfig(),
                expected_setting_id=setting["id"],
                expected_train_manifest_sha256=identity["train_manifest_sha256"],
                expected_normalizer_sha256=identity["normalizer_sha256"],
                expected_xy_dims=setting["xy_dims"],
                expected_task_metric_dims=setting["task_metric_dims"],
                expected_relative_endpoints=setting["relative_endpoints"],
            )
            return payload
        except (ValueError, OSError, json.JSONDecodeError):
            # A failed or stale calibration is never overwritten under the same
            # protocol namespace; require an operator to inspect/remove it.
            raise ValueError(f"existing calibration is stale or failed: {path}")
    _check_stop()
    return calibrate_future_metrics(
        cache.train.obs_norm,
        cache.train.index,
        setting_id=setting["id"],
        train_manifest_sha256=identity["train_manifest_sha256"],
        normalizer_sha256=identity["normalizer_sha256"],
        xy_dims=setting["xy_dims"],
        task_metric_dims=setting["task_metric_dims"],
        relative_endpoints=setting["relative_endpoints"],
        config=CalibrationConfig(),
        output_path=path,
        enforce_gates=True,
        stop_callback=_check_stop,
    )


def calibration_summary_path(contract_root: Path) -> Path:
    return contract_root / "calibration" / "ALL_SETTINGS.json"


def validate_all_calibrations(
    manifest: Mapping[str, Any],
    *,
    data_root: Path,
    cache_root: Path,
    contract_root: Path,
) -> dict[str, Any]:
    """Hard gate all ten train-only calibrations before any recipe is built."""
    settings: list[dict[str, Any]] = []
    for setting in manifest["settings"]:
        _check_stop()
        _, identity = _raw_identity(
            manifest, setting, data_root=data_root, cache_root=cache_root
        )
        path = calibration_contract_path(contract_root, setting["id"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_contract(
            payload,
            expected_config=CalibrationConfig(),
            expected_setting_id=setting["id"],
            expected_train_manifest_sha256=identity["train_manifest_sha256"],
            expected_normalizer_sha256=identity["normalizer_sha256"],
            expected_xy_dims=setting["xy_dims"],
            expected_task_metric_dims=setting["task_metric_dims"],
            expected_relative_endpoints=setting["relative_endpoints"],
        )
        settings.append(
            {
                "setting_id": setting["id"],
                "calibration_sha256": payload["contract_sha256"],
                "chosen": payload["chosen"],
                "gates": payload["gates"],
                "retrieval": payload["retrieval"]["chosen"],
                "horizon_histogram": payload["horizon"]["chosen_histogram"],
                "raw_mode_histogram": payload["cluster"]["chosen_raw_mode_histogram"],
                "retained_mode_histogram": payload["cluster"]["chosen_retained_mode_histogram"],
            }
        )
    summary: dict[str, Any] = {
        "schema_version": 1,
        "status": "complete",
        "campaign_id": manifest["campaign_id"],
        "campaign_protocol_sha256": protocol_sha256(manifest),
        "setting_count": len(settings),
        "settings": settings,
    }
    summary["summary_sha256"] = stable_hash(summary)
    _atomic_json(calibration_summary_path(contract_root), summary)
    return summary


def _validate_summary(
    manifest: Mapping[str, Any], contract_root: Path
) -> dict[str, Any]:
    path = calibration_summary_path(contract_root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    body = dict(payload)
    claimed = body.pop("summary_sha256", None)
    if (
        stable_hash(body) != claimed
        or payload.get("status") != "complete"
        or payload.get("campaign_id") != manifest["campaign_id"]
        or payload.get("campaign_protocol_sha256") != protocol_sha256(manifest)
        or payload.get("setting_count") != 10
        or [item.get("setting_id") for item in payload.get("settings", ())]
        != [item["id"] for item in manifest["settings"]]
    ):
        raise ValueError("all-setting calibration summary is missing or stale")
    return payload


def _future_config(manifest: Mapping[str, Any], setting: Mapping[str, Any], chosen):
    from treewm.data.future_sets import FutureSetConfig

    training = manifest["training"]
    return FutureSetConfig(
        num_neighbors=training["future_num_neighbors"],
        query_multiplier=training["future_query_multiplier"],
        time_exclusion=training["future_time_exclusion"],
        retrieval_radius=float(chosen["retrieval_radius"]),
        include_self=True,
        metric_mode="rms_v2",
        horizons=tuple(training["future_horizons"]),
        h_max=training["future_h_max"],
        horizon_rule=training["future_horizon_rule"],
        displacement_threshold=float(chosen["displacement_threshold"]),
        fixed_horizon=32,
        relative_endpoints=bool(setting["relative_endpoints"]),
        cluster_threshold=float(chosen["cluster_threshold"]),
        cluster_method="average",
        max_modes=manifest["objective"]["max_modes"],
        multi_step_depth=3,
        retrieval_pool=training["future_retrieval_pool"],
    )


def build_setting_recipe(
    manifest: Mapping[str, Any],
    setting_index: int,
    *,
    data_root: Path,
    cache_root: Path,
    contract_root: Path,
) -> dict[str, Any]:
    """Build train and validation recipe unions only after all calibrations pass."""
    _validate_summary(manifest, contract_root)
    setting = manifest["settings"][setting_index]
    cache, identity = _raw_identity(
        manifest, setting, data_root=data_root, cache_root=cache_root
    )
    calibration_path = calibration_contract_path(contract_root, setting["id"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    validate_contract(
        calibration,
        expected_config=CalibrationConfig(),
        expected_setting_id=setting["id"],
        expected_train_manifest_sha256=identity["train_manifest_sha256"],
        expected_normalizer_sha256=identity["normalizer_sha256"],
        expected_xy_dims=setting["xy_dims"],
        expected_task_metric_dims=setting["task_metric_dims"],
        expected_relative_endpoints=setting["relative_endpoints"],
    )
    from treewm.data.future_recipe import (
        build_or_load_split_recipe,
        formal_anchor_sets,
        publish_recipe_manifest,
    )

    train_sets, val_sets = formal_anchor_sets(
        cache.train.index,
        cache.val.index,
        max_train_anchors=manifest["training"]["max_train_anchors"],
        max_val_anchors=manifest["training"]["max_validation_anchors"],
    )
    cfg = _future_config(manifest, setting, calibration["chosen"])
    live = live_contract(Path(__file__).resolve().parents[2])
    root = recipe_root_path(contract_root, setting["id"])
    train_recipe = build_or_load_split_recipe(
        root / "train",
        split="train",
        obs_norm=cache.train.obs_norm,
        act_norm=cache.train.act_norm,
        index=cache.train.index,
        cfg=cfg,
        xy_dims=setting["xy_dims"],
        task_metric_dims=setting["task_metric_dims"],
        anchor_sets=train_sets,
        source_manifest_sha256=identity["data_manifest_sha256"],
        split_manifest_sha256=identity["train_manifest_sha256"],
        normalizer_sha256=identity["normalizer_sha256"],
        calibration_sha256=calibration["contract_sha256"],
        chosen_thresholds=calibration["chosen"],
        code_sha256=live["code_sha256"],
        runtime_sha256=live["runtime_sha256"],
        stop_callback=_check_stop,
    )
    _check_stop()
    val_recipe = build_or_load_split_recipe(
        root / "val",
        split="val",
        obs_norm=cache.val.obs_norm,
        act_norm=cache.val.act_norm,
        index=cache.val.index,
        cfg=cfg,
        xy_dims=setting["xy_dims"],
        task_metric_dims=setting["task_metric_dims"],
        anchor_sets=val_sets,
        source_manifest_sha256=identity["data_manifest_sha256"],
        split_manifest_sha256=identity["validation_manifest_sha256"],
        normalizer_sha256=identity["normalizer_sha256"],
        calibration_sha256=calibration["contract_sha256"],
        chosen_thresholds=calibration["chosen"],
        code_sha256=live["code_sha256"],
        runtime_sha256=live["runtime_sha256"],
        stop_callback=_check_stop,
    )
    return publish_recipe_manifest(
        root, train_manifest=train_recipe, val_manifest=val_recipe
    )


def publish_data_contract(
    manifest: Mapping[str, Any],
    setting_index: int,
    *,
    data_root: Path,
    cache_root: Path,
    contract_root: Path,
) -> dict[str, Any]:
    setting = manifest["settings"][setting_index]
    cache, identity = _raw_identity(
        manifest, setting, data_root=data_root, cache_root=cache_root
    )
    calibration_path = calibration_contract_path(contract_root, setting["id"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    recipe_manifest_path = recipe_root_path(contract_root, setting["id"]) / "manifest.json"
    recipe = json.loads(recipe_manifest_path.read_text(encoding="utf-8"))
    from treewm.data.future_recipe import validate_recipe_manifest

    validate_recipe_manifest(
        recipe_manifest_path.parent,
        recipe,
        expected_source_manifest_sha256=identity["data_manifest_sha256"],
        expected_normalizer_sha256=identity["normalizer_sha256"],
        expected_calibration_sha256=calibration["contract_sha256"],
        expected_thresholds=calibration["chosen"],
        expected_train_manifest_sha256=identity["train_manifest_sha256"],
        expected_validation_manifest_sha256=identity["validation_manifest_sha256"],
        expected_code_sha256=live_contract(Path(__file__).resolve().parents[2])["code_sha256"],
        expected_runtime_sha256=live_contract(Path(__file__).resolve().parents[2])["runtime_sha256"],
        verify_file_hash=False,
    )
    payload: dict[str, Any] = {
        "schema_version": 2,
        "status": "complete",
        "campaign_id": manifest["campaign_id"],
        "objective_version": manifest["method"]["objective_version"],
        "campaign_protocol_sha256": protocol_sha256(manifest),
        "setting_id": setting["id"],
        "env_name": setting["env_name"],
        "source_name": setting["source_name"],
        "dataset_kind": setting["dataset_kind"],
        "raw_cache_read_only": True,
        **identity,
        "calibration_path": str(calibration_path),
        "calibration_sha256": calibration["contract_sha256"],
        "chosen_thresholds": calibration["chosen"],
        "future_recipe_manifest": str(recipe_manifest_path),
        "future_recipe_sha256": recipe["recipe_sha256"],
        "train_recipe_rows": int(
            json.loads((recipe_manifest_path.parent / "train/manifest.json").read_text())["record_count"]
        ),
        "validation_recipe_rows": int(
            json.loads((recipe_manifest_path.parent / "val/manifest.json").read_text())["record_count"]
        ),
        "train_transitions": int(len(cache.train.obs_norm)),
        "validation_transitions": int(len(cache.val.obs_norm)),
        "train_trajectories": int(cache.train.index.num_trajectories),
        "validation_trajectories": int(cache.val.index.num_trajectories),
        "obs_dim": int(cache.train.obs_norm.shape[1]),
        "action_dim": int(cache.train.act_norm.shape[1]),
    }
    payload["contract_sha256"] = stable_hash(payload)
    _atomic_json(data_contract_path(contract_root, setting["id"]), payload)
    # Re-open through the same strict interface consumed by submit/dispatcher.
    return load_data_contract(
        manifest,
        setting,
        data_root=data_root,
        cache_root=cache_root,
        contract_root=contract_root,
    )


def _locked_paths(args, manifest):
    data_root = Path(args.data_root or manifest["paths"]["data_root"]).expanduser().absolute()
    cache_root = Path(args.cache_root or manifest["paths"]["raw_cache_root"]).expanduser().absolute()
    contract_root = Path(args.contract_root or manifest["paths"]["contract_root"]).expanduser().absolute()
    if str(cache_root) != manifest["paths"]["raw_cache_root"]:
        raise ValueError("raw cache root is immutable")
    if str(data_root) != manifest["paths"]["data_root"]:
        raise ValueError("source data root is immutable")
    if str(contract_root) != manifest["paths"]["contract_root"]:
        raise ValueError("v2 contract root is immutable")
    return data_root, cache_root, contract_root


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=here / "manifest.json")
    parser.add_argument("--phase", choices=("calibrate", "validate-all", "recipe"), required=True)
    parser.add_argument("--setting-index", type=int)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--contract-root", type=Path)
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    data_root, cache_root, contract_root = _locked_paths(args, manifest)
    if args.phase != "validate-all" and (
        args.setting_index is None or not 0 <= args.setting_index < len(manifest["settings"])
    ):
        parser.error("calibrate/recipe require --setting-index in [0,10)")

    signal.signal(signal.SIGUSR1, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    contract_root.mkdir(parents=True, exist_ok=True)
    try:
        if args.phase == "calibrate":
            result = calibrate_setting(
                manifest,
                int(args.setting_index),
                data_root=data_root,
                cache_root=cache_root,
                contract_root=contract_root,
            )
        elif args.phase == "validate-all":
            result = validate_all_calibrations(
                manifest,
                data_root=data_root,
                cache_root=cache_root,
                contract_root=contract_root,
            )
        else:
            setting_index = int(args.setting_index)
            build_setting_recipe(
                manifest,
                setting_index,
                data_root=data_root,
                cache_root=cache_root,
                contract_root=contract_root,
            )
            result = publish_data_contract(
                manifest,
                setting_index,
                data_root=data_root,
                cache_root=cache_root,
                contract_root=contract_root,
            )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except GracefulCacheStop:
        return GRACEFUL_EXIT_CODE


if __name__ == "__main__":
    raise SystemExit(main())
