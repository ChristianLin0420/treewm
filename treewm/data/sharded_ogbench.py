"""Coherent memory-mapped view of the official 100M OGBench shards.

The released 100M datasets are 100 train shards plus 100 paired validation shards.  They
cannot be passed to :func:`ogbench.make_env_and_datasets`, which expects one monolithic
``.npz`` pair, and concatenating them in RAM is not viable.  This module streams the exact
numeric shard sequence into one read-only mmap-backed corpus while preserving every
trajectory boundary.

No transitions are sampled here.  ``ChunkDataset`` applies its configured uniform anchor
cap over the full corpus, and ``FutureSetBuilder`` independently caps its retrieval pool.
The normalizer is fitted once over all train transitions; validation never contributes to
it.  A manifest records every source file, size and SHA256 so a cache cannot be mistaken
for a different local copy of the 100M release.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

from treewm.data.shared_cache import SharedCache, SharedSplit


CACHE_VERSION = 2
EXPECTED_SHARDS = 100
COPY_CHUNK_ROWS = 262_144
SHA_CHUNK_BYTES = 16 * 1024 * 1024


class ShardedDatasetError(ValueError):
    """The official shard set or a materialized cache is inconsistent."""


def shard_file_stem(dataset_name: str) -> str:
    """Map the release label to the actual official shard filename stem.

    OGBench labels the source directory ``*-100m-v0`` but publishes files such as
    ``puzzle-4x4-play-v0-000.npz``.  Keeping both identities explicit prevents the
    base simulator from being confused with the 100M data source.
    """
    if dataset_name.endswith("-100m-v0"):
        return dataset_name.removesuffix("-100m-v0") + "-v0"
    return dataset_name


@dataclass(frozen=True)
class ShardPair:
    index: int
    train: Path
    val: Path


@dataclass
class MemmapTrajectoryIndex:
    """Trajectory metadata shared by every process through read-only memmaps."""

    traj_id: np.ndarray
    steps_remaining: np.ndarray
    starts: np.ndarray
    lengths: np.ndarray

    @property
    def num_trajectories(self) -> int:
        return int(len(self.starts))

    def remaining_at(self, indices: np.ndarray) -> np.ndarray:
        """Indexed remaining-transition lookup without a full-corpus temporary."""
        return np.asarray(self.steps_remaining[np.asarray(indices, dtype=np.int64)])


def canonical_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def discover_shards(
    directory: str | Path,
    dataset_name: str,
    expected_shards: int = EXPECTED_SHARDS,
) -> list[ShardPair]:
    """Return the exact numeric train/validation sequence and reject extras or gaps."""

    root = Path(directory).expanduser().resolve()
    if not root.is_dir():
        raise ShardedDatasetError(f"100M shard directory does not exist: {root}")
    if not dataset_name.endswith("-v0"):
        raise ShardedDatasetError(f"base dataset name must end in -v0: {dataset_name!r}")
    stem = shard_file_stem(dataset_name)
    expected: dict[str, tuple[int, bool]] = {}
    for index in range(expected_shards):
        expected[f"{stem}-{index:03d}.npz"] = (index, False)
        expected[f"{stem}-{index:03d}-val.npz"] = (index, True)

    actual = {path.name: path for path in root.glob("*.npz")}
    missing = sorted(set(expected) - set(actual))
    extras = sorted(set(actual) - set(expected))
    if missing or extras:
        raise ShardedDatasetError(
            f"{root} is not the exact {expected_shards}+{expected_shards} shard set; "
            f"missing={missing[:5]} extras={extras[:5]}"
        )
    return [
        ShardPair(
            index=index,
            train=actual[f"{stem}-{index:03d}.npz"],
            val=actual[f"{stem}-{index:03d}-val.npz"],
        )
        for index in range(expected_shards)
    ]


def is_official_shard_directory(
    directory: str | Path | None,
    dataset_name: str,
    expected_shards: int = EXPECTED_SHARDS,
) -> bool:
    if directory is None:
        return False
    root = Path(directory).expanduser()
    stem = shard_file_stem(dataset_name)
    return (
        root.is_dir()
        and (root / f"{stem}-000.npz").is_file()
        and (root / f"{stem}-{expected_shards - 1:03d}-val.npz").is_file()
    )


def source_stat_inventory(pairs: Sequence[ShardPair], root: Path) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for pair in pairs:
        for split, path in (("train", pair.train), ("val", pair.val)):
            stat = path.stat()
            out.append(
                {
                    "index": pair.index,
                    "split": split,
                    "path": str(path.relative_to(root)),
                    "size": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
    return out


def file_sha256(
    path: Path, stop_callback: Callable[[], None] | None = None
) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            if stop_callback is not None:
                stop_callback()
            block = handle.read(SHA_CHUNK_BYTES)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _raw_trajectory_layout(terminals: np.ndarray, path: Path) -> tuple[np.ndarray, np.ndarray]:
    terminals = np.asarray(terminals, dtype=bool).reshape(-1)
    ends = np.flatnonzero(terminals)
    if len(ends) == 0 or int(ends[-1]) != len(terminals) - 1:
        raise ShardedDatasetError(
            f"{path}: raw OGBench shard must end with a terminal state"
        )
    starts = np.concatenate((np.array([0], dtype=np.int64), ends[:-1] + 1))
    # The raw terminal row is the state after the last valid transition and is excluded,
    # matching ogbench.load_dataset(compact_dataset=False).
    lengths = ends.astype(np.int64) - starts
    if np.any(lengths <= 0):
        raise ShardedDatasetError(f"{path}: empty trajectory in terminal layout")
    return starts, lengths


def _scan_shard_layouts(
    paths: Iterable[Path], stop_callback: Callable[[], None] | None
) -> list[dict[str, int]]:
    layouts: list[dict[str, int]] = []
    for path in paths:
        if stop_callback is not None:
            stop_callback()
        with np.load(path, allow_pickle=False) as archive:
            starts, lengths = _raw_trajectory_layout(archive["terminals"], path)
        layouts.append(
            {"transitions": int(lengths.sum()), "trajectories": int(len(starts))}
        )
    return layouts


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry updates; tolerate filesystems that reject dir fsync."""
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def _flush_array(array: np.memmap) -> None:
    """Make mmap contents durable before advancing the resumable cursor."""
    array.flush()
    descriptor = os.open(str(array.filename), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _open_array(path: Path, shape: tuple[int, ...], dtype) -> np.memmap:
    path.unlink(missing_ok=True)
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)


def _open_build_array(
    path: Path, shape: tuple[int, ...], dtype, *, resume: bool
) -> np.memmap:
    if not resume:
        return _open_array(path, shape, dtype)
    if not path.is_file():
        raise ShardedDatasetError(f"resumable build array vanished: {path}")
    array = np.load(path, mmap_mode="r+")
    if tuple(array.shape) != tuple(shape) or np.dtype(array.dtype) != np.dtype(dtype):
        raise ShardedDatasetError(
            f"resumable build array schema drift at {path}: {array.shape}/{array.dtype}"
        )
    return array


def _stream_split(
    paths: Sequence[Path],
    layouts: Sequence[dict[str, int]],
    prefix: str,
    build_dir: Path,
    total_transitions: int,
    total_trajectories: int,
    obs_dim: int,
    action_dim: int,
    *,
    collect_stats: bool,
    stop_callback: Callable[[], None] | None,
    next_shard: int,
    prior_stats: dict[str, object] | None,
    progress_callback: Callable[[int, dict[str, object]], None],
) -> dict[str, object]:
    if len(paths) != len(layouts) or not 0 <= next_shard <= len(paths):
        raise ShardedDatasetError(f"{prefix}: invalid resumable shard cursor {next_shard}")
    resume = next_shard > 0
    obs_out = _open_build_array(
        build_dir / f".{prefix}_obs.build.npy", (total_transitions, obs_dim), np.float32,
        resume=resume,
    )
    act_out = _open_build_array(
        build_dir / f".{prefix}_act.build.npy", (total_transitions, action_dim), np.float32,
        resume=resume,
    )
    terminals_out = _open_build_array(
        build_dir / f".{prefix}_terminals.build.npy", (total_transitions,), np.bool_,
        resume=resume,
    )
    traj_id_out = _open_build_array(
        build_dir / f".{prefix}_traj_id.build.npy", (total_transitions,), np.int32,
        resume=resume,
    )
    remaining_out = _open_build_array(
        build_dir / f".{prefix}_remaining.build.npy", (total_transitions,), np.int32,
        resume=resume,
    )
    starts_out = _open_build_array(
        build_dir / f".{prefix}_starts.build.npy", (total_trajectories,), np.int64,
        resume=resume,
    )
    lengths_out = _open_build_array(
        build_dir / f".{prefix}_lengths.build.npy", (total_trajectories,), np.int32,
        resume=resume,
    )

    def prior(name: str, dimension: int) -> np.ndarray:
        if prior_stats is None:
            return np.zeros(dimension, dtype=np.float64)
        return np.asarray(prior_stats[name], dtype=np.float64)

    obs_sum = prior("obs_sum", obs_dim)
    obs_sq = prior("obs_sq", obs_dim)
    act_sum = prior("act_sum", action_dim)
    act_sq = prior("act_sq", action_dim)
    row_cursor = sum(layout["transitions"] for layout in layouts[:next_shard])
    traj_cursor = sum(layout["trajectories"] for layout in layouts[:next_shard])

    arrays = (obs_out, act_out, terminals_out, traj_id_out, remaining_out, starts_out, lengths_out)
    for shard_index in range(next_shard, len(paths)):
        path = paths[shard_index]
        if stop_callback is not None:
            stop_callback()
        with np.load(path, allow_pickle=False) as archive:
            raw_obs = np.asarray(archive["observations"], dtype=np.float32)
            raw_act = np.asarray(archive["actions"], dtype=np.float32)
            raw_terminals = np.asarray(archive["terminals"], dtype=bool)
            starts, lengths = _raw_trajectory_layout(raw_terminals, path)
            if raw_obs.ndim != 2 or raw_obs.shape[1] != obs_dim:
                raise ShardedDatasetError(f"{path}: observation shape drifted to {raw_obs.shape}")
            if raw_act.ndim != 2 or raw_act.shape[1] != action_dim:
                raise ShardedDatasetError(f"{path}: action shape drifted to {raw_act.shape}")

            # Exclude each raw terminal state.  The mask is shard-local, so concatenation
            # cannot create a transition across a shard or trajectory boundary.
            mask = ~raw_terminals
            obs = raw_obs[mask]
            act = raw_act[mask]
            count = len(obs)
            expected = int(lengths.sum())
            layout = layouts[shard_index]
            if count != expected or count != layout["transitions"] or len(lengths) != layout["trajectories"]:
                raise ShardedDatasetError(f"{path}: mask produced {count}, expected {expected}")
            lo, hi = row_cursor, row_cursor + count
            obs_out[lo:hi] = obs
            act_out[lo:hi] = act
            terminals_out[lo:hi] = False

            local_cursor = 0
            for length in lengths.tolist():
                end = local_cursor + int(length)
                terminals_out[lo + end - 1] = True
                traj_id_out[lo + local_cursor : lo + end] = traj_cursor
                remaining_out[lo + local_cursor : lo + end] = np.arange(
                    int(length) - 1, -1, -1, dtype=np.int32
                )
                starts_out[traj_cursor] = lo + local_cursor
                lengths_out[traj_cursor] = int(length)
                local_cursor = end
                traj_cursor += 1

            if collect_stats:
                for chunk_lo in range(0, count, COPY_CHUNK_ROWS):
                    if stop_callback is not None:
                        stop_callback()
                    chunk_hi = min(chunk_lo + COPY_CHUNK_ROWS, count)
                    ob = obs[chunk_lo:chunk_hi].astype(np.float64, copy=False)
                    ac = act[chunk_lo:chunk_hi].astype(np.float64, copy=False)
                    obs_sum += ob.sum(axis=0, dtype=np.float64)
                    obs_sq += np.square(ob, dtype=np.float64).sum(axis=0, dtype=np.float64)
                    act_sum += ac.sum(axis=0, dtype=np.float64)
                    act_sq += np.square(ac, dtype=np.float64).sum(axis=0, dtype=np.float64)
            row_cursor = hi
            for array in arrays:
                _flush_array(array)
            stats = {
                "count": row_cursor,
                "obs_sum": obs_sum,
                "obs_sq": obs_sq,
                "act_sum": act_sum,
                "act_sq": act_sq,
            }
            progress_callback(shard_index + 1, stats)

    if row_cursor != total_transitions or traj_cursor != total_trajectories:
        raise ShardedDatasetError(
            f"{prefix}: wrote rows/trajectories {row_cursor}/{traj_cursor}, expected "
            f"{total_transitions}/{total_trajectories}"
        )
    for array in arrays:
        _flush_array(array)
    del obs_out, act_out, terminals_out, traj_id_out, remaining_out, starts_out, lengths_out
    return {
        "count": total_transitions,
        "obs_sum": obs_sum,
        "obs_sq": obs_sq,
        "act_sum": act_sum,
        "act_sq": act_sq,
    }


def _normalizer_from_stats(stats: dict[str, object], eps: float) -> dict[str, np.ndarray]:
    count = int(stats["count"])
    if count <= 0:
        raise ShardedDatasetError("cannot normalize an empty train split")
    obs_mean = np.asarray(stats["obs_sum"], dtype=np.float64) / count
    act_mean_for_std = np.asarray(stats["act_sum"], dtype=np.float64) / count
    obs_var = np.asarray(stats["obs_sq"], dtype=np.float64) / count - np.square(obs_mean)
    act_var = np.asarray(stats["act_sq"], dtype=np.float64) / count - np.square(act_mean_for_std)
    return {
        "obs_mean": obs_mean.astype(np.float32),
        "obs_std": (np.sqrt(np.maximum(obs_var, 0.0)) + eps).astype(np.float32),
        # Preserve TreeWM's existing action-normalization contract: zero centre, empirical
        # standard deviation around the data mean.
        "act_mean": np.zeros_like(act_mean_for_std, dtype=np.float32),
        "act_std": (np.sqrt(np.maximum(act_var, 0.0)) + eps).astype(np.float32),
    }


def _write_normalized(
    prefix: str,
    build_dir: Path,
    norm: dict[str, np.ndarray],
    stop_callback,
    *,
    next_row: int,
    progress_callback: Callable[[int], None],
) -> None:
    obs = np.load(build_dir / f".{prefix}_obs.build.npy", mmap_mode="r")
    act = np.load(build_dir / f".{prefix}_act.build.npy", mmap_mode="r")
    obs_norm = _open_build_array(
        build_dir / f".{prefix}_obs_norm.build.npy", obs.shape, np.float32,
        resume=next_row > 0,
    )
    act_norm = _open_build_array(
        build_dir / f".{prefix}_act_norm.build.npy", act.shape, np.float32,
        resume=next_row > 0,
    )
    if not 0 <= next_row <= len(obs):
        raise ShardedDatasetError(f"{prefix}: invalid normalization cursor {next_row}")
    for lo in range(next_row, len(obs), COPY_CHUNK_ROWS):
        if stop_callback is not None:
            stop_callback()
        hi = min(lo + COPY_CHUNK_ROWS, len(obs))
        obs_norm[lo:hi] = (obs[lo:hi] - norm["obs_mean"]) / norm["obs_std"]
        act_norm[lo:hi] = (act[lo:hi] - norm["act_mean"]) / norm["act_std"]
        _flush_array(obs_norm)
        _flush_array(act_norm)
        progress_callback(hi)
    _flush_array(obs_norm)
    _flush_array(act_norm)
    del obs, act, obs_norm, act_norm


def _promote_arrays(build_dir: Path) -> None:
    for prefix in ("train", "val"):
        for name in (
            "obs",
            "act",
            "terminals",
            "obs_norm",
            "act_norm",
            "traj_id",
            "remaining",
            "starts",
            "lengths",
        ):
            source = build_dir / f".{prefix}_{name}.build.npy"
            destination = build_dir / f"{prefix}_{name}.npy"
            if source.exists():
                os.replace(source, destination)
            elif not destination.exists():
                raise ShardedDatasetError(f"cache promotion lost both {source} and {destination}")
    _fsync_directory(build_dir)


def _load_split(root: Path, prefix: str) -> SharedSplit:
    load = lambda name: np.load(root / f"{prefix}_{name}.npy", mmap_mode="r")
    split = SharedSplit(
        obs=load("obs"),
        act=load("act"),
        terminals=load("terminals"),
        obs_norm=load("obs_norm"),
        act_norm=load("act_norm"),
        path=root,
    )
    split.trajectory_index = MemmapTrajectoryIndex(
        traj_id=load("traj_id"),
        steps_remaining=load("remaining"),
        starts=load("starts"),
        lengths=load("lengths"),
    )
    return split


def _load_cache(root: Path, manifest: dict[str, object], was_hit: bool) -> SharedCache:
    norm = {key: np.asarray(value, dtype=np.float32) for key, value in manifest["norm_stats"].items()}
    cache = SharedCache(
        key=str(manifest["cache_key"]),
        path=root,
        train=_load_split(root, "train"),
        val=_load_split(root, "val"),
        norm_stats=norm,
        was_hit=was_hit,
    )
    cache.source_manifest_sha256 = str(manifest["source_manifest_sha256"])
    cache.source_files = list(manifest["source_files"])
    cache.dataset_kind = "sharded_100m_full"
    return cache


def build_or_load_sharded_cache(
    dataset_name: str,
    dataset_dir: str | Path,
    *,
    cache_root: str | Path | None = None,
    expected_shards: int = EXPECTED_SHARDS,
    eps: float = 1e-6,
    verify_digests: bool = False,
    stop_callback: Callable[[], None] | None = None,
) -> SharedCache:
    """Materialize or load the full official shard corpus without sampling transitions."""

    source_root = Path(dataset_dir).expanduser().resolve()
    pairs = discover_shards(source_root, dataset_name, expected_shards)
    stat_inventory = source_stat_inventory(pairs, source_root)
    source_stat_sha256 = canonical_json_hash(stat_inventory)
    recipe = {
        "cache_version": CACHE_VERSION,
        "dataset_name": dataset_name,
        "shard_file_stem": shard_file_stem(dataset_name),
        "source_directory": str(source_root),
        "expected_shards": expected_shards,
        "source_stat_sha256": source_stat_sha256,
        "normalization": "global_train_mean_std",
        "regular_dataset_semantics": True,
        "dtype": "float32",
        "eps": eps,
    }
    cache_key = canonical_json_hash(recipe)[:20]
    root = Path(cache_root or os.environ.get("TREEWM_CACHE", Path.home() / ".cache" / "treewm"))
    root = root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = root / f"{source_root.name}__full__{cache_key}"
    destination.mkdir(parents=True, exist_ok=True)
    manifest_path = destination / "manifest.json"
    lock_path = root / f".{source_root.name}__full.lock"

    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("recipe") != recipe:
                raise ShardedDatasetError(f"cache recipe mismatch at {manifest_path}")
            if verify_digests:
                expected = {entry["path"]: entry["sha256"] for entry in manifest["source_files"]}
                for pair in pairs:
                    for path in (pair.train, pair.val):
                        relative = str(path.relative_to(source_root))
                        if file_sha256(path, stop_callback) != expected.get(relative):
                            raise ShardedDatasetError(f"source digest mismatch: {path}")
            return _load_cache(destination, manifest, was_hit=True)

        train_paths = [pair.train for pair in pairs]
        val_paths = [pair.val for pair in pairs]
        state_path = destination / ".build_state.json"
        if state_path.is_file():
            try:
                state = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ShardedDatasetError(f"corrupt resumable state {state_path}") from exc
            if state.get("schema_version") != 1 or state.get("recipe") != recipe:
                raise ShardedDatasetError(f"resumable state recipe mismatch: {state_path}")
        else:
            # Scanning terminals is cheap relative to materialization. Persist the exact
            # per-shard layout before allocating any large file.
            train_layouts = _scan_shard_layouts(train_paths, stop_callback)
            val_layouts = _scan_shard_layouts(val_paths, stop_callback)
            with np.load(train_paths[0], allow_pickle=False) as first:
                obs_dim = int(first["observations"].shape[1])
                action_dim = int(first["actions"].shape[1])
            for name in (
                "obs", "act", "terminals", "obs_norm", "act_norm",
                "traj_id", "remaining", "starts", "lengths",
            ):
                for prefix in ("train", "val"):
                    (destination / f".{prefix}_{name}.build.npy").unlink(missing_ok=True)
                    (destination / f"{prefix}_{name}.npy").unlink(missing_ok=True)
            state = {
                "schema_version": 1,
                "recipe": recipe,
                "train_layouts": train_layouts,
                "val_layouts": val_layouts,
                "obs_dim": obs_dim,
                "action_dim": action_dim,
                "train_next_shard": 0,
                "val_next_shard": 0,
                "train_stats": None,
                "val_stats": None,
                "train_norm_next_row": 0,
                "val_norm_next_row": 0,
                "arrays_promoted": False,
                "source_files": [],
            }
            _atomic_json(state_path, state)

        train_layouts = list(state["train_layouts"])
        val_layouts = list(state["val_layouts"])
        if len(train_layouts) != expected_shards or len(val_layouts) != expected_shards:
            raise ShardedDatasetError("resumable layout does not cover every official shard")
        train_steps = sum(int(layout["transitions"]) for layout in train_layouts)
        train_trajectories = sum(int(layout["trajectories"]) for layout in train_layouts)
        val_steps = sum(int(layout["transitions"]) for layout in val_layouts)
        val_trajectories = sum(int(layout["trajectories"]) for layout in val_layouts)
        obs_dim = int(state["obs_dim"])
        action_dim = int(state["action_dim"])

        # Promotion deliberately persists the directory entries before advancing the
        # JSON cursor.  A machine failure in that small interval can therefore leave
        # some (or all) final arrays present while ``arrays_promoted`` is still false.
        # Recovery is safe only after both stream and normalization cursors are
        # complete; finish the idempotent promotion before reopening final arrays.
        if not state["arrays_promoted"]:
            promotion_started = any(destination.glob("train_*.npy")) or any(
                destination.glob("val_*.npy")
            )
            normalization_complete = (
                int(state["train_next_shard"]) == len(train_layouts)
                and int(state["val_next_shard"]) == len(val_layouts)
                and int(state["train_norm_next_row"]) == train_steps
                and int(state["val_norm_next_row"]) == val_steps
            )
            if promotion_started:
                if not normalization_complete:
                    raise ShardedDatasetError(
                        "final cache arrays appeared before all durable build cursors completed"
                    )
                _promote_arrays(destination)
                state["arrays_promoted"] = True
                _atomic_json(state_path, state)

        if state["arrays_promoted"]:
            # A stop during the (potentially long) source-digest phase resumes after
            # promotion. Re-expose the final arrays read-only through temporary build
            # symlinks so the completed stream/normalization cursors can be validated
            # without rewriting a single row.
            for name in (
                "obs", "act", "terminals", "obs_norm", "act_norm",
                "traj_id", "remaining", "starts", "lengths",
            ):
                for prefix in ("train", "val"):
                    build_path = destination / f".{prefix}_{name}.build.npy"
                    final_path = destination / f"{prefix}_{name}.npy"
                    if not build_path.exists():
                        if not final_path.exists():
                            raise ShardedDatasetError(f"promoted cache array vanished: {final_path}")
                        build_path.symlink_to(final_path.name)

        def serialize_stats(stats: dict[str, object]) -> dict[str, object]:
            return {
                "count": int(stats["count"]),
                "obs_sum": np.asarray(stats["obs_sum"]).tolist(),
                "obs_sq": np.asarray(stats["obs_sq"]).tolist(),
                "act_sum": np.asarray(stats["act_sum"]).tolist(),
                "act_sq": np.asarray(stats["act_sq"]).tolist(),
            }

        def train_progress(next_shard: int, stats: dict[str, object]) -> None:
            state["train_next_shard"] = next_shard
            state["train_stats"] = serialize_stats(stats)
            _atomic_json(state_path, state)

        train_stats = _stream_split(
            train_paths,
            train_layouts,
            "train",
            destination,
            train_steps,
            train_trajectories,
            obs_dim,
            action_dim,
            collect_stats=True,
            stop_callback=stop_callback,
            next_shard=int(state["train_next_shard"]),
            prior_stats=state.get("train_stats"),
            progress_callback=train_progress,
        )

        def val_progress(next_shard: int, stats: dict[str, object]) -> None:
            state["val_next_shard"] = next_shard
            state["val_stats"] = serialize_stats(stats)
            _atomic_json(state_path, state)

        _stream_split(
            val_paths,
            val_layouts,
            "val",
            destination,
            val_steps,
            val_trajectories,
            obs_dim,
            action_dim,
            collect_stats=False,
            stop_callback=stop_callback,
            next_shard=int(state["val_next_shard"]),
            prior_stats=state.get("val_stats"),
            progress_callback=val_progress,
        )
        norm = _normalizer_from_stats(train_stats, eps)

        def train_norm_progress(next_row: int) -> None:
            state["train_norm_next_row"] = next_row
            _atomic_json(state_path, state)

        _write_normalized(
            "train", destination, norm, stop_callback,
            next_row=int(state["train_norm_next_row"]),
            progress_callback=train_norm_progress,
        )

        def val_norm_progress(next_row: int) -> None:
            state["val_norm_next_row"] = next_row
            _atomic_json(state_path, state)

        _write_normalized(
            "val", destination, norm, stop_callback,
            next_row=int(state["val_norm_next_row"]),
            progress_callback=val_norm_progress,
        )
        if not state["arrays_promoted"]:
            _promote_arrays(destination)
            state["arrays_promoted"] = True
            _atomic_json(state_path, state)

        source_files: list[dict[str, object]] = list(state["source_files"])
        source_paths = [path for pair in pairs for path in (pair.train, pair.val)]
        if len(source_files) > len(stat_inventory):
            raise ShardedDatasetError("resumable source digest cursor exceeds inventory")
        for source_index in range(len(source_files), len(stat_inventory)):
            entry = stat_inventory[source_index]
            pair_path = source_paths[source_index]
            if stop_callback is not None:
                stop_callback()
            source_files.append(
                {**entry, "sha256": file_sha256(pair_path, stop_callback)}
            )
            state["source_files"] = source_files
            _atomic_json(state_path, state)
        if source_stat_inventory(pairs, source_root) != stat_inventory:
            raise ShardedDatasetError(
                "100M source files changed while the materialized cache was built"
            )
        source_manifest_sha256 = canonical_json_hash(source_files)
        manifest: dict[str, object] = {
            "schema_version": 1,
            "cache_key": cache_key,
            "recipe": recipe,
            "dataset_name": dataset_name,
            "source_dataset": source_root.name,
            "source_manifest_sha256": source_manifest_sha256,
            "source_files": source_files,
            "train_shards": expected_shards,
            "validation_shards": expected_shards,
            "train_transitions": train_steps,
            "validation_transitions": val_steps,
            "train_trajectories": train_trajectories,
            "validation_trajectories": val_trajectories,
            "obs_dim": obs_dim,
            "action_dim": action_dim,
            "norm_stats": {key: value.tolist() for key, value in norm.items()},
        }
        _atomic_json(manifest_path, manifest)
        state_path.unlink(missing_ok=True)
        for temporary in destination.glob(".*.build.npy"):
            # At this point these can only be digest-resume symlinks. Never unlink a
            # regular build file silently: that would conceal a promotion bug.
            if temporary.is_symlink():
                temporary.unlink()
        _fsync_directory(destination)
        return _load_cache(destination, manifest, was_hit=False)
