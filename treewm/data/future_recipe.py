"""Compact, resumable future-set recipes for formal TreeWM v2 training.

The expensive cKDTree query and average-linkage clustering are deterministic functions
of an anchor.  A recipe stores only their compact decisions.  Dense action chunks,
masks, and endpoints are reconstructed on demand from the immutable raw memmaps, so
the cache stays a few GB rather than scaling with ``anchors * 24 * 64 * action_dim``.
"""

from __future__ import annotations

from dataclasses import asdict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import struct
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from treewm.data.future_sets import FutureSetBuilder, FutureSetConfig
from treewm.data.ogbench_dataset import uniform_anchor_ranks


SCHEMA_VERSION = 1
RECIPE_VERSION = "treewm_compact_future_recipe_v1"
FORMAL_TRAIN_SEEDS = (0, 1, 2, 3)
FORMAL_VALIDATION_SELECTION_SEEDS = (1, 2, 3, 4)


class FutureRecipeError(ValueError):
    """A recipe is incomplete, stale, or inconsistent with the requested dataset."""


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def indices_sha256(indices: np.ndarray) -> str:
    values = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def normalizer_state_sha256(norm_stats: Mapping[str, Sequence[float]]) -> str:
    """Hash the exact float32 state consumed by ``Normalizer.from_state_dict``."""
    digest = hashlib.sha256(b"treewm-normalizer-float32-v1\n")
    for key in sorted(norm_stats):
        values = list(norm_stats[key])
        digest.update(key.encode("utf-8") + b"\0" + struct.pack("<Q", len(values)))
        for value in values:
            digest.update(struct.pack("<f", float(value)))
    return digest.hexdigest()


def file_sha256(path: Path, stop_callback: Callable[[], None] | None = None) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            if stop_callback is not None:
                stop_callback()
            digest.update(block)
    return digest.hexdigest()


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


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def anchors_for_seed(index: Any, max_anchors: int, seed: int) -> np.ndarray:
    """Reproduce ``ChunkDataset`` anchor selection without constructing the dataset."""
    minimum_horizon = 4
    counts = np.maximum(np.asarray(index.lengths, dtype=np.int64) - minimum_horizon, 0)
    population = int(counts.sum(dtype=np.int64))
    size = min(int(max_anchors), population)
    if size == 0:
        return np.empty(0, dtype=np.int64)
    if size == population:
        parts = [
            np.arange(int(start), int(start) + int(count), dtype=np.int64)
            for start, count in zip(index.starts, counts, strict=True)
            if count
        ]
        return np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
    ranks = uniform_anchor_ranks(population, size, int(seed))
    cumulative = np.cumsum(counts, dtype=np.int64)
    trajectories = np.searchsorted(cumulative, ranks, side="right")
    previous = np.where(trajectories == 0, 0, cumulative[trajectories - 1])
    return (
        np.asarray(index.starts, dtype=np.int64)[trajectories] + ranks - previous
    ).astype(np.int64, copy=False)


def formal_anchor_sets(
    train_index: Any,
    val_index: Any,
    *,
    max_train_anchors: int = 300_000,
    max_val_anchors: int = 30_000,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    train = {
        f"seed{seed}": anchors_for_seed(train_index, max_train_anchors, seed)
        for seed in FORMAL_TRAIN_SEEDS
    }
    # ChunkDataset intentionally uses training seed+1 for validation selection.
    val = {
        f"seed{seed}": anchors_for_seed(val_index, max_val_anchors, seed + 1)
        for seed in FORMAL_TRAIN_SEEDS
    }
    return train, val


def _record_dtype(cfg: FutureSetConfig) -> np.dtype:
    k = int(cfg.num_neighbors)
    c = int(cfg.max_modes)
    d = int(cfg.multi_step_depth)
    return np.dtype(
        [
            ("anchor", "<i8"),
            ("neighbors", "<i8", (k,)),
            ("horizon_idx", "u1", (k,)),
            ("fut_valid", "u1", (k,)),
            ("cluster", "i1", (k,)),
            ("mode_rep", "i1", (c,)),
            ("mode_mass", "<f4", (c,)),
            ("mode_valid", "u1", (c,)),
            ("ms_horizon_idx", "u1", (d,)),
            ("ms_valid", "u1", (d,)),
            ("future_diversity", "<f4"),
            ("num_retrieved", "u1"),
            ("retrieval_num_candidates", "<u2"),
            ("retrieval_num_valid", "u1"),
            ("retrieval_mean_distance", "<f4"),
            ("retrieval_fallback", "u1"),
            ("retrieval_truncated", "u1"),
            ("retrieval_query_saturated", "u1"),
            ("modes_raw", "u1"),
            ("modes_retained", "u1"),
            ("modes_truncated", "u1"),
        ],
        align=False,
    )


def _recipe_row(builder: FutureSetBuilder, anchor_index: int) -> dict[str, Any]:
    """Compute exactly the compact decisions made by ``FutureSetBuilder.build``."""
    cfg = builder.cfg
    t = int(anchor_index)
    rng = np.random.default_rng(t)
    builder._last_retrieval_distances = np.empty(0, dtype=np.float32)
    builder._last_retrieval_query_saturated = False
    neighbors_raw = np.asarray(builder._neighbors(t), dtype=np.int64)
    retrieval_num_candidates = len(neighbors_raw)
    neighbors = neighbors_raw[: cfg.num_neighbors]
    used = len(neighbors)
    distances = np.asarray(builder._last_retrieval_distances, dtype=np.float32)[:used]

    k = int(cfg.num_neighbors)
    neighbor_slots = np.full(k, -1, dtype=np.int64)
    neighbor_slots[:used] = neighbors
    horizon_idx = np.zeros(k, dtype=np.uint8)
    fut_valid = np.zeros(k, dtype=np.uint8)
    metric_endpoints = np.zeros((k, len(builder.task_metric_dims)), dtype=np.float32)
    horizons = np.asarray(cfg.horizons, dtype=np.int64)
    horizon_lookup = {int(h): i for i, h in enumerate(horizons)}
    anchor = builder.obs_norm[t]
    for slot, continuation in enumerate(neighbors):
        continuation = int(continuation)
        horizon = builder._pick_horizon(continuation, rng)
        endpoint = builder.obs_norm[continuation + horizon]
        if cfg.relative_endpoints:
            endpoint = endpoint.copy()
            endpoint[builder.xy_dims] = anchor[builder.xy_dims] + (
                endpoint[builder.xy_dims]
                - builder.obs_norm[continuation][builder.xy_dims]
            )
        metric_endpoints[slot] = builder._metric_coordinates(endpoint)
        horizon_idx[slot] = horizon_lookup[horizon]
        fut_valid[slot] = 1

    labels, representatives, masses, raw_modes = builder._cluster(
        metric_endpoints[:used], rng, return_raw_count=True
    )
    cluster = np.full(k, -1, dtype=np.int8)
    cluster[:used] = labels.astype(np.int8, copy=False)
    fut_valid[:used] = (labels >= 0).astype(np.uint8)
    c_max = int(cfg.max_modes)
    mode_rep = np.full(c_max, -1, dtype=np.int8)
    mode_mass = np.zeros(c_max, dtype=np.float32)
    mode_valid = np.zeros(c_max, dtype=np.uint8)
    retained_modes = min(len(representatives), c_max)
    mode_rep[:retained_modes] = representatives[:retained_modes].astype(np.int8)
    mode_mass[:retained_modes] = masses[:retained_modes]
    mode_valid[:retained_modes] = 1

    raw_metric = metric_endpoints[:used]
    if len(raw_metric) > 1:
        diversity = float(
            builder._metric_distance(raw_metric[:, None, :], raw_metric[None, :, :]).mean()
        )
    else:
        diversity = 0.0
    nonself_distances = distances[distances > 0]
    mean_distance = float(nonself_distances.mean()) if len(nonself_distances) else 0.0
    query_saturated = bool(builder._last_retrieval_query_saturated)
    retrieval_truncated = retrieval_num_candidates > used or query_saturated
    fallback = int(sum(int(index) != t for index in neighbors) == 0)

    d_max = int(cfg.multi_step_depth)
    ms_horizon_idx = np.zeros(d_max, dtype=np.uint8)
    ms_valid = np.zeros(d_max, dtype=np.uint8)
    cursor = t
    for depth in range(d_max):
        if builder._remaining[cursor] < int(horizons.min()):
            break
        horizon = builder._pick_horizon(cursor, rng)
        if horizon > int(builder._remaining[cursor]):
            break
        ms_horizon_idx[depth] = horizon_lookup[horizon]
        ms_valid[depth] = 1
        cursor += horizon

    return {
        "anchor": t,
        "neighbors": neighbor_slots,
        "horizon_idx": horizon_idx,
        "fut_valid": fut_valid,
        "cluster": cluster,
        "mode_rep": mode_rep,
        "mode_mass": mode_mass,
        "mode_valid": mode_valid,
        "ms_horizon_idx": ms_horizon_idx,
        "ms_valid": ms_valid,
        "future_diversity": np.float32(diversity),
        "num_retrieved": used,
        "retrieval_num_candidates": retrieval_num_candidates,
        "retrieval_num_valid": used,
        "retrieval_mean_distance": np.float32(mean_distance),
        "retrieval_fallback": fallback,
        "retrieval_truncated": int(retrieval_truncated),
        "retrieval_query_saturated": int(query_saturated),
        "modes_raw": int(raw_modes),
        "modes_retained": int(retained_modes),
        "modes_truncated": int(max(raw_modes - retained_modes, 0)),
    }


def _identity(
    *,
    split: str,
    cfg: FutureSetConfig,
    xy_dims: Sequence[int],
    task_metric_dims: Sequence[int],
    anchor_sets: Mapping[str, np.ndarray],
    source_manifest_sha256: str,
    split_manifest_sha256: str,
    normalizer_sha256: str,
    calibration_sha256: str,
    chosen_thresholds: Mapping[str, float],
    code_sha256: str,
    runtime_sha256: str,
) -> tuple[dict[str, Any], np.ndarray]:
    import scipy

    if split not in {"train", "val"}:
        raise FutureRecipeError("recipe split must be train or val")
    normalized_sets = {
        name: np.asarray(values, dtype=np.int64) for name, values in sorted(anchor_sets.items())
    }
    if not normalized_sets or any(len(values) == 0 for values in normalized_sets.values()):
        raise FutureRecipeError("every formal seed anchor set must be nonempty")
    union = np.unique(np.concatenate(list(normalized_sets.values()))).astype(np.int64)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "recipe_version": RECIPE_VERSION,
        "split": split,
        "source_manifest_sha256": source_manifest_sha256,
        "split_manifest_sha256": split_manifest_sha256,
        "normalizer_sha256": normalizer_sha256,
        "calibration_sha256": calibration_sha256,
        "chosen_thresholds": dict(chosen_thresholds),
        "future_config": asdict(cfg),
        "xy_dims": [int(dim) for dim in xy_dims],
        "task_metric_dims": [int(dim) for dim in task_metric_dims],
        "code_sha256": code_sha256,
        "runtime_sha256": runtime_sha256,
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "anchor_sets": {
            name: {"count": int(len(values)), "sha256": indices_sha256(values)}
            for name, values in normalized_sets.items()
        },
        "anchor_union_count": int(len(union)),
        "anchor_union_sha256": indices_sha256(union),
    }
    return identity, union


def _build_or_load_split_recipe_unlocked(
    root: str | Path,
    *,
    split: str,
    obs_norm: np.ndarray,
    act_norm: np.ndarray,
    index: Any,
    cfg: FutureSetConfig,
    xy_dims: Sequence[int],
    task_metric_dims: Sequence[int],
    anchor_sets: Mapping[str, np.ndarray],
    source_manifest_sha256: str,
    split_manifest_sha256: str,
    normalizer_sha256: str,
    calibration_sha256: str,
    chosen_thresholds: Mapping[str, float],
    code_sha256: str,
    runtime_sha256: str,
    stop_callback: Callable[[], None] | None = None,
    chunk_size: int = 256,
) -> dict[str, Any]:
    """Build one split recipe with a durable row cursor, or validate an existing one."""
    root = Path(root).expanduser().absolute()
    root.mkdir(parents=True, exist_ok=True)
    identity, anchors = _identity(
        split=split,
        cfg=cfg,
        xy_dims=xy_dims,
        task_metric_dims=task_metric_dims,
        anchor_sets=anchor_sets,
        source_manifest_sha256=source_manifest_sha256,
        split_manifest_sha256=split_manifest_sha256,
        normalizer_sha256=normalizer_sha256,
        calibration_sha256=calibration_sha256,
        chosen_thresholds=chosen_thresholds,
        code_sha256=code_sha256,
        runtime_sha256=runtime_sha256,
    )
    identity_sha256 = stable_hash(identity)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        validate_split_recipe_manifest(
            root,
            payload,
            expected_identity_sha256=identity_sha256,
            verify_file_hash=False,
        )
        return payload

    dtype = _record_dtype(cfg)
    state_path = root / ".build_state.json"
    build_path = root / ".records.build.npy"
    final_path = root / "records.npy"
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("identity_sha256") != identity_sha256:
            raise FutureRecipeError("partial recipe identity does not match requested build")
        next_row = int(state.get("next_row", -1))
        if not 0 <= next_row <= len(anchors):
            raise FutureRecipeError("partial recipe cursor/file is corrupt")
        if build_path.is_file() and not final_path.exists():
            records = np.lib.format.open_memmap(build_path, mode="r+")
        elif (
            next_row == len(anchors)
            and final_path.is_file()
            and not build_path.exists()
        ):
            # A signal may interrupt the potentially long content hash after the
            # durable promotion.  Keep the cursor until the manifest is published
            # so the next allocation can resume finalization without rebuilding.
            records = np.load(final_path, mmap_mode="r")
        else:
            raise FutureRecipeError("partial recipe cursor/file is corrupt")
        if records.dtype != dtype or records.shape != (len(anchors),):
            raise FutureRecipeError("partial recipe array schema drifted")
    else:
        if build_path.exists() or final_path.exists():
            raise FutureRecipeError("orphaned recipe file has no durable cursor")
        records = np.lib.format.open_memmap(
            build_path, mode="w+", dtype=dtype, shape=(len(anchors),)
        )
        records["anchor"] = anchors
        records.flush()
        with build_path.open("rb") as handle:
            os.fsync(handle.fileno())
        next_row = 0
        _atomic_json(
            state_path,
            {
                "schema_version": SCHEMA_VERSION,
                "identity_sha256": identity_sha256,
                "next_row": 0,
                "total_rows": int(len(anchors)),
            },
        )

    builder = FutureSetBuilder(
        obs_norm=obs_norm,
        act_norm=act_norm,
        index=index,
        cfg=cfg,
        xy_dims=tuple(int(dim) for dim in xy_dims),
        task_metric_dims=tuple(int(dim) for dim in task_metric_dims),
    )
    for start in range(next_row, len(anchors), int(chunk_size)):
        stop = min(start + int(chunk_size), len(anchors))
        for row in range(start, stop):
            if stop_callback is not None:
                stop_callback()
            values = _recipe_row(builder, int(anchors[row]))
            for field in records.dtype.names or ():
                records[field][row] = values[field]
        records.flush()
        descriptor = os.open(build_path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _atomic_json(
            state_path,
            {
                "schema_version": SCHEMA_VERSION,
                "identity_sha256": identity_sha256,
                "next_row": int(stop),
                "total_rows": int(len(anchors)),
            },
        )
    del records
    if build_path.is_file():
        os.replace(build_path, final_path)
        _fsync_directory(root)
    elif not final_path.is_file():
        raise FutureRecipeError("recipe records disappeared before finalization")
    stat = final_path.stat()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "identity": identity,
        "identity_sha256": identity_sha256,
        "records_file": final_path.name,
        "records_size": stat.st_size,
        "records_mtime_ns": stat.st_mtime_ns,
        "records_sha256": file_sha256(final_path, stop_callback),
        "record_dtype": dtype.descr,
        "record_count": int(len(anchors)),
    }
    payload = json.loads(_canonical_json(payload))
    payload["recipe_sha256"] = stable_hash(payload)
    _atomic_json(manifest_path, payload)
    state_path.unlink(missing_ok=True)
    _fsync_directory(root)
    return payload


def build_or_load_split_recipe(root: str | Path, *args, **kwargs) -> dict[str, Any]:
    """Serialize duplicate builders with a setting/split-local advisory lock."""
    path = Path(root).expanduser().absolute()
    path.mkdir(parents=True, exist_ok=True)
    with (path / ".build.lock").open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return _build_or_load_split_recipe_unlocked(path, *args, **kwargs)


def validate_split_recipe_manifest(
    root: str | Path,
    payload: Mapping[str, Any],
    *,
    expected_identity_sha256: str | None = None,
    verify_file_hash: bool = False,
) -> None:
    root = Path(root).expanduser().absolute()
    claimed = payload.get("recipe_sha256")
    body = dict(payload)
    body.pop("recipe_sha256", None)
    if not isinstance(claimed, str) or stable_hash(body) != claimed:
        raise FutureRecipeError("split recipe manifest hash mismatch")
    identity = payload.get("identity") or {}
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "complete"
        or identity.get("recipe_version") != RECIPE_VERSION
        or stable_hash(identity) != payload.get("identity_sha256")
        or payload.get("record_count") != identity.get("anchor_union_count")
    ):
        raise FutureRecipeError("split recipe identity/count drifted")
    if expected_identity_sha256 is not None and payload.get("identity_sha256") != expected_identity_sha256:
        raise FutureRecipeError("split recipe does not match the live immutable identity")
    records_path = root / str(payload.get("records_file", ""))
    stat = records_path.stat()
    if stat.st_size != payload.get("records_size") or stat.st_mtime_ns != payload.get("records_mtime_ns"):
        raise FutureRecipeError("split recipe records changed after publication")
    if verify_file_hash and file_sha256(records_path) != payload.get("records_sha256"):
        raise FutureRecipeError("split recipe records content hash mismatch")


def publish_recipe_manifest(
    root: str | Path,
    *,
    train_manifest: Mapping[str, Any],
    val_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    root = Path(root).expanduser().absolute()
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "recipe_version": RECIPE_VERSION,
        "train_recipe_sha256": train_manifest["recipe_sha256"],
        "validation_recipe_sha256": val_manifest["recipe_sha256"],
        "train_manifest": "train/manifest.json",
        "validation_manifest": "val/manifest.json",
        "source_manifest_sha256": train_manifest["identity"]["source_manifest_sha256"],
        "normalizer_sha256": train_manifest["identity"]["normalizer_sha256"],
        "calibration_sha256": train_manifest["identity"]["calibration_sha256"],
        "chosen_thresholds": train_manifest["identity"]["chosen_thresholds"],
        "train_manifest_sha256": train_manifest["identity"]["split_manifest_sha256"],
        "validation_manifest_sha256": val_manifest["identity"]["split_manifest_sha256"],
        "code_sha256": train_manifest["identity"]["code_sha256"],
        "runtime_sha256": train_manifest["identity"]["runtime_sha256"],
    }
    if any(
        val_manifest["identity"][key] != payload[key]
        for key in (
            "source_manifest_sha256",
            "normalizer_sha256",
            "calibration_sha256",
            "chosen_thresholds",
            "code_sha256",
            "runtime_sha256",
        )
    ):
        raise FutureRecipeError("train/validation recipe identities disagree")
    payload["recipe_sha256"] = stable_hash(payload)
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing != payload:
            raise FutureRecipeError("existing composite recipe identity drifted")
        return existing
    _atomic_json(manifest_path, payload)
    return payload


def validate_recipe_manifest(
    root: str | Path,
    payload: Mapping[str, Any],
    *,
    expected_source_manifest_sha256: str | None = None,
    expected_normalizer_sha256: str | None = None,
    expected_calibration_sha256: str | None = None,
    expected_thresholds: Mapping[str, float] | None = None,
    expected_train_manifest_sha256: str | None = None,
    expected_validation_manifest_sha256: str | None = None,
    expected_code_sha256: str | None = None,
    expected_runtime_sha256: str | None = None,
    verify_file_hash: bool = False,
) -> None:
    root = Path(root).expanduser().absolute()
    claimed = payload.get("recipe_sha256")
    body = dict(payload)
    body.pop("recipe_sha256", None)
    if not isinstance(claimed, str) or stable_hash(body) != claimed:
        raise FutureRecipeError("composite future recipe manifest hash mismatch")
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("status") != "complete"
        or payload.get("recipe_version") != RECIPE_VERSION
    ):
        raise FutureRecipeError("composite future recipe is incomplete")
    expected = {
        "source_manifest_sha256": expected_source_manifest_sha256,
        "normalizer_sha256": expected_normalizer_sha256,
        "calibration_sha256": expected_calibration_sha256,
        "chosen_thresholds": expected_thresholds,
        "train_manifest_sha256": expected_train_manifest_sha256,
        "validation_manifest_sha256": expected_validation_manifest_sha256,
        "code_sha256": expected_code_sha256,
        "runtime_sha256": expected_runtime_sha256,
    }
    if any(value is not None and payload.get(key) != value for key, value in expected.items()):
        raise FutureRecipeError("composite future recipe immutable identity drifted")
    for split, manifest_key, sha_key in (
        ("train", "train_manifest", "train_recipe_sha256"),
        ("val", "validation_manifest", "validation_recipe_sha256"),
    ):
        path = root / str(payload.get(manifest_key, ""))
        child = json.loads(path.read_text(encoding="utf-8"))
        validate_split_recipe_manifest(path.parent, child, verify_file_hash=verify_file_hash)
        if child.get("recipe_sha256") != payload.get(sha_key) or child["identity"]["split"] != split:
            raise FutureRecipeError(f"{split} recipe does not match composite manifest")


class FutureRecipe:
    """Read-only split recipe that reconstructs the exact builder result."""

    def __init__(
        self,
        root: str | Path,
        *,
        expected_recipe_sha256: str | None = None,
        expected_source_manifest_sha256: str | None = None,
        expected_calibration_sha256: str | None = None,
    ) -> None:
        self.root = Path(root).expanduser().absolute()
        payload = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        validate_split_recipe_manifest(self.root, payload, verify_file_hash=False)
        if expected_recipe_sha256 and payload.get("recipe_sha256") != expected_recipe_sha256:
            raise FutureRecipeError("split recipe SHA does not match injected campaign identity")
        identity = payload["identity"]
        if expected_source_manifest_sha256 and identity.get("source_manifest_sha256") != expected_source_manifest_sha256:
            raise FutureRecipeError("split recipe source identity drifted")
        if expected_calibration_sha256 and identity.get("calibration_sha256") != expected_calibration_sha256:
            raise FutureRecipeError("split recipe calibration identity drifted")
        self.manifest = payload
        self.recipe_sha256 = payload["recipe_sha256"]
        self.identity = identity
        self.records = np.load(self.root / payload["records_file"], mmap_mode="r")
        self.anchors = self.records["anchor"]
        if len(self.anchors) > 1 and np.any(self.anchors[1:] <= self.anchors[:-1]):
            raise FutureRecipeError("recipe anchors are not strictly increasing")

    def contains_all(self, anchors: np.ndarray) -> bool:
        anchors = np.asarray(anchors, dtype=np.int64)
        positions = np.searchsorted(self.anchors, anchors)
        return bool(
            np.all(positions < len(self.anchors))
            and np.array_equal(self.anchors[positions], anchors)
        )

    def _row(self, anchor: int) -> np.void:
        position = int(np.searchsorted(self.anchors, int(anchor)))
        if position >= len(self.anchors) or int(self.anchors[position]) != int(anchor):
            raise FutureRecipeError(f"anchor {anchor} is absent from the formal recipe")
        return self.records[position]

    def build(
        self,
        anchor: int,
        *,
        obs_norm: np.ndarray,
        act_norm: np.ndarray,
        index: Any,
    ) -> dict[str, np.ndarray]:
        row = self._row(anchor)
        identity = self.identity
        cfg = FutureSetConfig(**identity["future_config"])
        xy_dims = np.asarray(identity["xy_dims"], dtype=np.int64)
        metric_dims = np.asarray(identity["task_metric_dims"], dtype=np.int64)
        horizons = np.asarray(cfg.horizons, dtype=np.int64)
        k = cfg.num_neighbors
        obs_dim = int(obs_norm.shape[1])
        act_dim = int(act_norm.shape[1])
        fut_actions = np.zeros((k, cfg.h_max, act_dim), dtype=np.float32)
        fut_mask = np.zeros((k, cfg.h_max), dtype=np.float32)
        fut_endpoint = np.zeros((k, obs_dim), dtype=np.float32)
        fut_metric_endpoint = np.zeros((k, len(metric_dims)), dtype=np.float32)
        fut_horizon_idx = np.asarray(row["horizon_idx"], dtype=np.int64).copy()
        fut_horizon_len = np.zeros(k, dtype=np.float32)
        neighbors = np.asarray(row["neighbors"], dtype=np.int64)
        valid_neighbors = neighbors >= 0
        anchor_obs = obs_norm[int(anchor)]
        for slot in np.flatnonzero(valid_neighbors):
            continuation = int(neighbors[slot])
            horizon = int(horizons[fut_horizon_idx[slot]])
            endpoint = obs_norm[continuation + horizon]
            if cfg.relative_endpoints:
                endpoint = endpoint.copy()
                endpoint[xy_dims] = anchor_obs[xy_dims] + (
                    endpoint[xy_dims] - obs_norm[continuation][xy_dims]
                )
            fut_endpoint[slot] = endpoint
            fut_metric_endpoint[slot] = endpoint[metric_dims]
            fut_actions[slot, :horizon] = act_norm[continuation : continuation + horizon]
            fut_mask[slot, :horizon] = 1.0
            fut_horizon_len[slot] = float(horizon)

        d_max = cfg.multi_step_depth
        ms_actions = np.zeros((d_max, cfg.h_max, act_dim), dtype=np.float32)
        ms_mask = np.zeros((d_max, cfg.h_max), dtype=np.float32)
        ms_obs = np.tile(anchor_obs.astype(np.float32), (d_max, 1))
        ms_horizon_idx = np.asarray(row["ms_horizon_idx"], dtype=np.int64).copy()
        ms_valid = np.asarray(row["ms_valid"], dtype=np.float32).copy()
        cursor = int(anchor)
        for depth in np.flatnonzero(ms_valid > 0):
            horizon = int(horizons[ms_horizon_idx[depth]])
            if int(index.remaining_at(np.asarray([cursor], dtype=np.int64))[0]) < horizon:
                raise FutureRecipeError("recipe multistep cursor crosses a trajectory boundary")
            ms_actions[depth, :horizon] = act_norm[cursor : cursor + horizon]
            ms_mask[depth, :horizon] = 1.0
            ms_obs[depth] = obs_norm[cursor + horizon]
            cursor += horizon

        return {
            "anchor_index": np.int64(anchor),
            "task_metric_dims": metric_dims.astype(np.int64, copy=True),
            "ms_actions": ms_actions,
            "ms_action_mask": ms_mask,
            "ms_obs": ms_obs,
            "ms_horizon_idx": ms_horizon_idx,
            "ms_valid": ms_valid,
            "obs": anchor_obs.astype(np.float32),
            "fut_actions": fut_actions,
            "fut_action_mask": fut_mask,
            "fut_endpoint": fut_endpoint,
            "fut_metric_endpoint": fut_metric_endpoint,
            "fut_horizon_idx": fut_horizon_idx,
            "fut_horizon_len": fut_horizon_len,
            "fut_valid": np.asarray(row["fut_valid"], dtype=np.float32).copy(),
            "fut_cluster": np.asarray(row["cluster"], dtype=np.int64).copy(),
            "mode_rep": np.asarray(row["mode_rep"], dtype=np.int64).copy(),
            "mode_mass": np.asarray(row["mode_mass"], dtype=np.float32).copy(),
            "mode_valid": np.asarray(row["mode_valid"], dtype=np.float32).copy(),
            "num_modes": np.int64(row["modes_retained"]),
            "future_diversity": np.float32(row["future_diversity"]),
            "num_retrieved": np.int64(row["num_retrieved"]),
            "retrieval_num_candidates": np.int64(row["retrieval_num_candidates"]),
            "retrieval_num_valid": np.int64(row["retrieval_num_valid"]),
            "retrieval_mean_distance": np.float32(row["retrieval_mean_distance"]),
            "retrieval_fallback": np.float32(row["retrieval_fallback"]),
            "retrieval_truncated": np.float32(row["retrieval_truncated"]),
            "retrieval_query_saturated": np.float32(row["retrieval_query_saturated"]),
            "modes_raw": np.int64(row["modes_raw"]),
            "modes_retained": np.int64(row["modes_retained"]),
            "modes_truncated": np.int64(row["modes_truncated"]),
        }
