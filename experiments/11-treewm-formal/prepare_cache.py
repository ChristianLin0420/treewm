#!/usr/bin/env python3
"""Materialize one immutable TreeWM data cache and publish its source contract.

Run one setting per stage rank.  Cache builders use advisory locks and atomic manifests,
so a duplicate invocation either waits or reuses the finished cache.  The two 100M
settings stream every official shard; this script never offers a subsampling mode.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import sys
from typing import Any

from campaign import (
    data_contract_path,
    load_data_contract,
    load_manifest,
    protocol_sha256,
    required_dataset_files,
)


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
        raise GracefulCacheStop("scheduler requested a safe cache-stage stop")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _space_preflight(source_paths: list[Path], cache_root: Path) -> None:
    """Conservative filesystem-capacity check before creating large mmap arrays."""
    source_bytes = sum(path.stat().st_size for path in source_paths)
    cache_root.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(cache_root).free
    # NPZs include simulator-only qpos/qvel arrays which the cache omits. Three times
    # compressed source size nevertheless covers raw+normalised arrays and a restarted
    # in-progress build with a safety margin for these official releases.
    required = source_bytes * 3 + 10 * 1024**3
    if free_bytes < required:
        raise OSError(
            f"insufficient filesystem capacity for full cache: free={free_bytes} "
            f"conservative_required={required}"
        )


def prepare_setting(
    manifest: dict[str, Any],
    setting_index: int,
    *,
    data_root: Path,
    cache_root: Path,
) -> dict[str, Any]:
    setting = manifest["settings"][setting_index]
    # A validated contract is a durable success sentinel. It also verifies that all
    # current source size/mtime values still match the content-digested cache build.
    try:
        return load_data_contract(
            manifest, setting, data_root=data_root, cache_root=cache_root
        )
    except ValueError:
        pass

    source_paths = required_dataset_files(manifest, data_root, setting)
    missing = [path for path in source_paths if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise FileNotFoundError(f"missing source data for {setting['id']}: {missing[:5]}")
    _space_preflight(source_paths, cache_root)
    _check_stop()

    if setting["dataset_kind"] == "standard":
        from treewm.data.shared_cache import build_or_load

        cache = build_or_load(
            setting["source_name"],
            dataset_dir=str(data_root / setting["data_subdir"]),
            root=cache_root,
        )
    elif setting["dataset_kind"] == "sharded_100m_full":
        from treewm.data.sharded_ogbench import build_or_load_sharded_cache

        cache = build_or_load_sharded_cache(
            setting["source_name"],
            dataset_dir=data_root / setting["data_subdir"],
            cache_root=cache_root,
            expected_shards=100,
            stop_callback=_check_stop,
        )
    else:  # protected independently by validate_manifest; defensive at stage boundary
        raise ValueError(f"unsupported dataset kind {setting['dataset_kind']!r}")

    _check_stop()
    by_name = {Path(entry["path"]).name: entry for entry in cache.source_files or []}
    entries: list[dict[str, Any]] = []
    for source_path in source_paths:
        cached = by_name.get(source_path.name)
        if cached is None:
            raise ValueError(f"cache manifest omitted source {source_path.name}")
        stat = source_path.stat()
        if stat.st_size != cached["size"] or stat.st_mtime_ns != cached["mtime_ns"]:
            raise ValueError(f"source changed while cache was built: {source_path}")
        entries.append(
            {
                "path": str(source_path.relative_to(data_root)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": cached["sha256"],
            }
        )

    payload = {
        "schema_version": 1,
        "status": "complete",
        "campaign_id": manifest["campaign_id"],
        "protocol_sha256": protocol_sha256(manifest),
        "setting_id": setting["id"],
        "env_name": setting["env_name"],
        "source_name": setting["source_name"],
        "dataset_kind": setting["dataset_kind"],
        "data_manifest_sha256": cache.source_manifest_sha256,
        "cache_key": cache.key,
        "cache_path": str(cache.path),
        "cache_manifest": str(cache.path / "manifest.json"),
        "source_files": entries,
        "train_transitions": int(len(cache.train.obs)),
        "validation_transitions": int(len(cache.val.obs)),
        "obs_dim": int(cache.train.obs.shape[1]),
        "action_dim": int(cache.train.act.shape[1]),
        "full_transition_corpus": setting["dataset_kind"] == "sharded_100m_full",
    }
    if setting["dataset_kind"] == "sharded_100m_full":
        payload["train_trajectories"] = int(cache.train.trajectory_index.num_trajectories)
        payload["validation_trajectories"] = int(cache.val.trajectory_index.num_trajectories)
    if payload["obs_dim"] != setting["obs_dim"] or payload["action_dim"] != setting["action_dim"]:
        raise ValueError(f"cache dimensions do not match protocol for {setting['id']}")
    if setting["dataset_kind"] == "sharded_100m_full" and (
        payload["train_transitions"] != setting["expected_train_transitions"]
        or payload["validation_transitions"] != setting["expected_validation_transitions"]
        or payload["train_trajectories"] != setting["expected_train_trajectories"]
        or payload["validation_trajectories"] != setting["expected_validation_trajectories"]
    ):
        raise ValueError(f"full-shard transition counts do not match protocol for {setting['id']}")
    _atomic_json(data_contract_path(cache_root, setting["id"]), payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=here / "manifest.json")
    parser.add_argument("--setting-index", type=int, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--cache-root", type=Path)
    args = parser.parse_args(argv)
    manifest = load_manifest(args.manifest)
    if not 0 <= args.setting_index < len(manifest["settings"]):
        parser.error("--setting-index must be in [0, 10)")
    data_root = (args.data_root or Path(manifest["paths"]["data_root"])).expanduser().resolve()
    cache_root = (args.cache_root or Path(manifest["paths"]["cache_root"])).expanduser().resolve()

    signal.signal(signal.SIGUSR1, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    try:
        payload = prepare_setting(
            manifest, args.setting_index, data_root=data_root, cache_root=cache_root
        )
    except GracefulCacheStop as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return GRACEFUL_EXIT_CODE
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
