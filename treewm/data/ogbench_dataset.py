"""OGBench dataset loading, trajectory indexing, normalisation and chunk sampling.

Dataset layout (verified against ``ogbench.utils.load_dataset`` with
``compact_dataset=False``)::

                          |<--- traj 1 --->|  |<--- traj 2 --->|
    'observations'      : [s0, s1, s2, s3,     s0, s1, s2, s3, ...]
    'actions'           : [a0, a1, a2, a3,     a0, a1, a2, a3, ...]
    'next_observations' : [s1, s2, s3, s4,     s1, s2, s3, s4, ...]
    'terminals'         : [ 0,  0,  0,  1,      0,  0,  0,  1, ...]

so ``terminals[t] == 1`` marks the last *transition* of a trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from treewm.data.future_sets import FutureSetBuilder, FutureSetConfig


def load_ogbench(
    dataset_name: str,
    dataset_dir: str | None = None,
    env_only: bool = False,
    **env_kwargs: Any,
):
    """Thin wrapper over ``ogbench.make_env_and_datasets``.

    Returns ``(env, train_dataset, val_dataset)``, or just ``env`` when ``env_only``.
    """
    import ogbench

    kwargs: dict[str, Any] = dict(dataset_name=dataset_name, compact_dataset=False, **env_kwargs)
    if dataset_dir:
        kwargs["dataset_dir"] = str(Path(dataset_dir).expanduser())
    if env_only:
        return ogbench.make_env_and_datasets(env_only=True, **kwargs)
    return ogbench.make_env_and_datasets(**kwargs)


@dataclass
class TrajectoryIndex:
    """Per-timestep trajectory bookkeeping derived from ``terminals``."""

    traj_id: np.ndarray  # [N] int32
    traj_start: np.ndarray  # [N] int64, first index of this timestep's trajectory
    traj_end: np.ndarray  # [N] int64, last index (inclusive) of this trajectory
    starts: np.ndarray = field(repr=False, default=None)  # [T] trajectory start indices
    lengths: np.ndarray = field(repr=False, default=None)  # [T]

    @classmethod
    def from_terminals(cls, terminals: np.ndarray) -> "TrajectoryIndex":
        terminals = np.asarray(terminals).reshape(-1)
        n = len(terminals)
        ends = np.flatnonzero(terminals > 0.5)
        if len(ends) == 0 or ends[-1] != n - 1:
            # Trailing partial trajectory: treat the final index as a terminal.
            ends = np.concatenate([ends, [n - 1]])
        starts = np.concatenate([[0], ends[:-1] + 1])
        lengths = ends - starts + 1

        traj_id = np.repeat(np.arange(len(starts), dtype=np.int32), lengths)
        traj_start = np.repeat(starts, lengths).astype(np.int64)
        traj_end = np.repeat(ends, lengths).astype(np.int64)
        assert len(traj_id) == n, f"trajectory index covers {len(traj_id)} of {n} steps"
        return cls(traj_id=traj_id, traj_start=traj_start, traj_end=traj_end, starts=starts, lengths=lengths)

    @property
    def steps_remaining(self) -> np.ndarray:
        """Number of transitions available after each index, within its trajectory."""
        return self.traj_end - np.arange(len(self.traj_id), dtype=np.int64)

    def remaining_at(self, indices: np.ndarray) -> np.ndarray:
        """Return remaining-transition counts only for ``indices``.

        The indexed form matters for the 100M corpus: callers that need 100k retrieval
        keys must not accidentally allocate an additional full-corpus temporary.
        """
        indices = np.asarray(indices, dtype=np.int64)
        return self.traj_end[indices] - indices

    @property
    def num_trajectories(self) -> int:
        return len(self.starts)


@dataclass
class Normalizer:
    """Per-dimension mean/std normalisation, fitted on the training split only."""

    obs_mean: np.ndarray
    obs_std: np.ndarray
    act_mean: np.ndarray
    act_std: np.ndarray

    @classmethod
    def fit(cls, observations: np.ndarray, actions: np.ndarray, eps: float = 1e-6) -> "Normalizer":
        return cls(
            obs_mean=observations.mean(0).astype(np.float32),
            obs_std=(observations.std(0) + eps).astype(np.float32),
            act_mean=np.zeros(actions.shape[1], dtype=np.float32),
            # Actions live in [-1, 1] and are already centred; only scale is normalised
            # so that the action MSE stays comparable to the env's own action units.
            act_std=(actions.std(0) + eps).astype(np.float32),
        )

    def norm_obs(self, obs: np.ndarray) -> np.ndarray:
        return ((obs - self.obs_mean) / self.obs_std).astype(np.float32)

    def denorm_obs(self, obs: np.ndarray) -> np.ndarray:
        return (obs * self.obs_std + self.obs_mean).astype(np.float32)

    def norm_act(self, act: np.ndarray) -> np.ndarray:
        return ((act - self.act_mean) / self.act_std).astype(np.float32)

    def denorm_act(self, act: np.ndarray) -> np.ndarray:
        return (act * self.act_std + self.act_mean).astype(np.float32)

    def state_dict(self) -> dict[str, np.ndarray]:
        return {
            "obs_mean": self.obs_mean,
            "obs_std": self.obs_std,
            "act_mean": self.act_mean,
            "act_std": self.act_std,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "Normalizer":
        return cls(**{k: np.asarray(v, dtype=np.float32) for k, v in state.items()})


def uniform_anchor_ranks(population: int, size: int, seed: int) -> np.ndarray:
    """Uniform sample without replacement using memory bounded by ``size``.

    Some NumPy versions materialize work proportional to ``population`` for
    ``Generator.choice(replace=False)``.  That can be an 800MB transient for 100M
    anchors in each of sixteen concurrent trainers.  Rejection sampling is symmetric
    over the population and has negligible collisions at the formal 300k/100M ratio.
    """
    if not 0 <= size <= population:
        raise ValueError(f"cannot sample {size} ranks from population {population}")
    rng = np.random.default_rng(seed)
    if size == 0:
        return np.empty(0, dtype=np.int64)
    if population <= 10_000_000:
        return np.sort(rng.choice(population, size=size, replace=False)).astype(
            np.int64, copy=False
        )
    selected = np.empty(0, dtype=np.int64)
    while len(selected) < size:
        needed = size - len(selected)
        draws = rng.integers(0, population, size=max(needed * 2, 1024), dtype=np.int64)
        selected = np.unique(np.concatenate((selected, draws)))
    if len(selected) > size:
        selected = rng.choice(selected, size=size, replace=False)
    return np.sort(selected).astype(np.int64, copy=False)


class ChunkDataset(Dataset):
    """Anchor states paired with a retrieved *set* of alternative supported futures.

    A single offline trajectory provides only one continuation, so each item bundles
    the anchor's own future together with continuations retrieved from nearby dataset
    states (spec section 11). Those continuations are clustered into discrete
    controllability *modes* by :class:`~treewm.data.future_sets.FutureSetBuilder`, which
    is what gives KEEP/support and mass separate, well-defined targets.
    """

    def __init__(
        self,
        dataset: dict[str, np.ndarray],
        normalizer: Normalizer,
        future_cfg: FutureSetConfig,
        xy_dims: tuple[int, ...] = (0, 1),
        max_anchors: int | None = None,
        seed: int = 0,
        cache_future_sets: bool = False,
        shared: "SharedSplit | None" = None,
        task_metric_dims: tuple[int, ...] | None = None,
    ) -> None:
        if shared is not None:
            # Bind the memory-mapped arrays directly. ascontiguousarray would COPY them,
            # which is what silently defeated a previous caching attempt: the cache was
            # read, then every process materialised its own multi-GB duplicate anyway.
            self.obs, self.act = shared.obs, shared.act
            self.obs_norm, self.act_norm = shared.obs_norm, shared.act_norm
            # The full 100M cache persists trajectory metadata as read-only mmaps. Do
            # not reconstruct multi-GB trajectory arrays in every process.
            self.index = getattr(shared, "trajectory_index", None)
            if self.index is None:
                self.index = TrajectoryIndex.from_terminals(shared.terminals)
            self.cache_backend = "mmap"
        else:
            self.obs = np.ascontiguousarray(dataset["observations"], dtype=np.float32)
            self.act = np.ascontiguousarray(dataset["actions"], dtype=np.float32)
            self.index = TrajectoryIndex.from_terminals(dataset["terminals"])
            self.obs_norm = normalizer.norm_obs(self.obs)
            self.act_norm = normalizer.norm_act(self.act)
            self.cache_backend = "in_process"
        self.normalizer = normalizer
        self.cfg = future_cfg
        self.xy_dims = tuple(xy_dims)
        self.task_metric_dims = (
            self.xy_dims if task_metric_dims is None else tuple(task_metric_dims)
        )
        self.obs_dim = self.obs.shape[1]
        self.act_dim = self.act.shape[1]

        min_h = int(min(future_cfg.horizons))
        valid_counts = np.maximum(
            np.asarray(self.index.lengths, dtype=np.int64) - min_h, 0
        )
        total_valid = int(valid_counts.sum(dtype=np.int64))
        if max_anchors is not None and total_valid > int(max_anchors):
            # Sample ranks in the exact full valid-anchor universe, then map the ranks
            # through trajectory-local offsets. This stays uniform without allocating
            # a 100M-element mask and flattened index array first.
            ranks = uniform_anchor_ranks(total_valid, int(max_anchors), seed)
            cumulative = np.cumsum(valid_counts, dtype=np.int64)
            trajectories = np.searchsorted(cumulative, ranks, side="right")
            previous = np.where(trajectories == 0, 0, cumulative[trajectories - 1])
            anchors = (
                np.asarray(self.index.starts, dtype=np.int64)[trajectories]
                + ranks
                - previous
            )
        else:
            # Small/uncapped datasets keep the historical explicit anchor array.
            parts = [
                np.arange(int(start), int(start) + int(count), dtype=np.int64)
                for start, count in zip(self.index.starts, valid_counts, strict=True)
                if count
            ]
            anchors = np.concatenate(parts) if parts else np.empty(0, dtype=np.int64)
        self.total_valid_anchors = total_valid
        self.anchors = anchors

        # Future sets are a deterministic function of the anchor index, and with
        # 150k anchors over a 300k-step run each anchor is rebuilt ~500 times. Caching
        # turns the dataloader from the bottleneck into a lookup after one epoch --
        # essential for AntMaze, where a k-d tree query in 29-D dominates the step.
        self._cache: dict[int, dict] | None = {} if cache_future_sets else None

        self.builder = FutureSetBuilder(
            obs_norm=self.obs_norm,
            act_norm=self.act_norm,
            index=self.index,
            cfg=future_cfg,
            xy_dims=self.xy_dims,
            task_metric_dims=self.task_metric_dims,
        )
        self.future_recipe = None

    def attach_future_recipe(self, recipe) -> None:
        """Attach a complete read-only recipe covering every selected anchor."""
        if not recipe.contains_all(self.anchors):
            raise ValueError("formal future recipe does not cover every selected anchor")
        self.future_recipe = recipe
        self._cache = None

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        t = int(self.anchors[i])
        if self.future_recipe is not None:
            item = self.future_recipe.build(
                t, obs_norm=self.obs_norm, act_norm=self.act_norm, index=self.index
            )
        elif self._cache is None:
            item = self.builder.build(t)
        else:
            item = self._cache.get(t)
            if item is None:
                item = self.builder.build(t)
                self._cache[t] = item
        return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v) for k, v in item.items()}

    # ------------------------------------------------------------------ helpers

    def observation_at(self, t: int) -> np.ndarray:
        return self.obs[t]

    def summary(self) -> dict[str, Any]:
        return {
            "num_steps": int(len(self.obs)),
            "num_trajectories": int(self.index.num_trajectories),
            "num_anchors": int(len(self.anchors)),
            "total_valid_anchors": int(self.total_valid_anchors),
            "obs_dim": int(self.obs_dim),
            "act_dim": int(self.act_dim),
            "median_traj_len": int(np.median(self.index.lengths)),
        }


def build_datasets(
    dataset_name: str,
    future_cfg: FutureSetConfig,
    dataset_dir: str | None = None,
    xy_dims: tuple[int, ...] = (0, 1),
    max_train_anchors: int | None = None,
    max_val_anchors: int | None = None,
    seed: int = 0,
    cache_future_sets: bool = False,
    shared_cache: bool = False,
    dataset_kind: str | None = None,
    source_name: str | None = None,
    expected_shards: int = 100,
    cache_root: str | None = None,
    data_manifest_sha256: str | None = None,
    task_metric_dims: tuple[int, ...] | None = None,
) -> tuple[Any, ChunkDataset, ChunkDataset, Normalizer]:
    """Load ``dataset_name`` and build train/val :class:`ChunkDataset` objects.

    With ``shared_cache`` the observation/action arrays and normalisation statistics come
    from a memory-mapped on-disk cache built exactly once (see
    :mod:`treewm.data.shared_cache`), so concurrent jobs share one physical copy.
    """
    source_name = source_name or dataset_name
    sharded = dataset_kind == "sharded_100m_full"
    if dataset_kind not in (None, "standard", "sharded_100m_full"):
        raise ValueError(f"unsupported dataset_kind: {dataset_kind!r}")
    if dataset_kind is None and source_name != dataset_name:
        # Backward-compatible inference for callers composed before the formal env
        # schema was introduced. Formal campaign calls always pass the explicit kind.
        sharded = source_name.endswith("-100m-v0")

    if sharded:
        if not shared_cache:
            raise ValueError("sharded_100m_full requires future_sets.shared_cache=true")
        if max_train_anchors is None or max_val_anchors is None:
            raise ValueError("sharded_100m_full requires bounded train and validation anchors")
        from treewm.data.sharded_ogbench import build_or_load_sharded_cache

        cache = build_or_load_sharded_cache(
            source_name,
            dataset_dir=dataset_dir,
            cache_root=cache_root,
            expected_shards=int(expected_shards),
        )
        # OGBench registers only the base simulator (without the `-100m` data suffix).
        env = load_ogbench(dataset_name, dataset_dir=None, env_only=True)
        normalizer = Normalizer.from_state_dict(cache.norm_stats)
        train_ds = ChunkDataset(
            {}, normalizer, future_cfg, xy_dims=xy_dims,
            max_anchors=max_train_anchors, seed=seed,
            cache_future_sets=cache_future_sets, shared=cache.train,
            task_metric_dims=task_metric_dims,
        )
        val_ds = ChunkDataset(
            {}, normalizer, future_cfg, xy_dims=xy_dims,
            max_anchors=max_val_anchors, seed=seed + 1, shared=cache.val,
            task_metric_dims=task_metric_dims,
        )
        train_ds.cache_metrics = cache.assert_consumed_by(train_ds, val_ds)
        train_ds.manifest_sha256 = cache.source_manifest_sha256
        train_ds.source_manifest_sha256 = cache.source_manifest_sha256
        train_ds.dataset_kind = cache.dataset_kind
        train_ds.source_files = cache.source_files
        if data_manifest_sha256 and data_manifest_sha256 != train_ds.manifest_sha256:
            raise ValueError("injected data manifest SHA256 does not match full-shard cache")
        _attach_future_recipes_if_requested(
            train_ds, val_ds, normalizer, future_cfg, train_ds.manifest_sha256
        )
        return env, train_ds, val_ds, normalizer

    if shared_cache:
        from treewm.data.shared_cache import build_or_load

        cache = build_or_load(dataset_name, dataset_dir=dataset_dir, root=cache_root)
        env = load_ogbench(dataset_name, dataset_dir=dataset_dir, env_only=True)
        normalizer = Normalizer.from_state_dict(cache.norm_stats)
        train_ds = ChunkDataset(
            {}, normalizer, future_cfg, xy_dims=xy_dims, max_anchors=max_train_anchors,
            seed=seed, cache_future_sets=cache_future_sets, shared=cache.train,
            task_metric_dims=task_metric_dims,
        )
        val_ds = ChunkDataset(
            {}, normalizer, future_cfg, xy_dims=xy_dims, max_anchors=max_val_anchors,
            seed=seed + 1, shared=cache.val, task_metric_dims=task_metric_dims,
        )
        # Fails loudly if the arrays were copied rather than mapped.
        train_ds.cache_metrics = cache.assert_consumed_by(train_ds, val_ds)
        manifest_sha256 = getattr(cache, "source_manifest_sha256", "")
        if manifest_sha256:
            train_ds.manifest_sha256 = manifest_sha256
            train_ds.source_manifest_sha256 = manifest_sha256
        if data_manifest_sha256 and data_manifest_sha256 != manifest_sha256:
            raise ValueError("injected data manifest SHA256 does not match standard cache")
        _attach_future_recipes_if_requested(
            train_ds, val_ds, normalizer, future_cfg, manifest_sha256
        )
        return env, train_ds, val_ds, normalizer

    env, train, val = load_ogbench(dataset_name, dataset_dir=dataset_dir)
    normalizer = Normalizer.fit(train["observations"], train["actions"])
    train_ds = ChunkDataset(
        train, normalizer, future_cfg, xy_dims=xy_dims, max_anchors=max_train_anchors, seed=seed,
        cache_future_sets=cache_future_sets, task_metric_dims=task_metric_dims,
    )
    val_ds = ChunkDataset(
        val, normalizer, future_cfg, xy_dims=xy_dims, max_anchors=max_val_anchors,
        seed=seed + 1, task_metric_dims=task_metric_dims,
    )
    return env, train_ds, val_ds, normalizer


def _attach_future_recipes_if_requested(
    train_ds: ChunkDataset,
    val_ds: ChunkDataset,
    normalizer: Normalizer,
    future_cfg: FutureSetConfig,
    source_manifest_sha256: str,
) -> None:
    """Wire formal-v2 recipes without changing historical dataset semantics."""
    recipe_root_value = os.environ.get("TREEWM_FUTURE_RECIPE_ROOT")
    calibration_sha256 = os.environ.get("TREEWM_CALIBRATION_SHA256")
    if not recipe_root_value:
        if calibration_sha256 and future_cfg.metric_mode == "rms_v2":
            raise ValueError("formal v2 supplied calibration identity but no recipe root")
        return
    if future_cfg.metric_mode != "rms_v2":
        raise ValueError("compact formal recipes require future_sets.metric_mode=rms_v2")
    expected_recipe_sha256 = os.environ.get("TREEWM_FUTURE_RECIPE_SHA256")
    if not expected_recipe_sha256 or not calibration_sha256:
        raise ValueError("formal recipe use requires injected recipe and calibration SHA256")
    from treewm.data.future_recipe import (
        FutureRecipe,
        normalizer_state_sha256,
        validate_recipe_manifest,
    )

    root = Path(recipe_root_value).expanduser().absolute()
    payload = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if payload.get("recipe_sha256") != expected_recipe_sha256:
        raise ValueError("injected future recipe SHA256 does not match its manifest")
    normalizer_sha256 = normalizer_state_sha256(normalizer.state_dict())
    validate_recipe_manifest(
        root,
        payload,
        expected_source_manifest_sha256=source_manifest_sha256,
        expected_normalizer_sha256=normalizer_sha256,
        expected_calibration_sha256=calibration_sha256,
        expected_code_sha256=os.environ.get("TREEWM_CODE_SHA256"),
        expected_runtime_sha256=os.environ.get("TREEWM_RUNTIME_SHA256"),
    )
    train_recipe = FutureRecipe(
        root / "train",
        expected_recipe_sha256=payload["train_recipe_sha256"],
        expected_source_manifest_sha256=source_manifest_sha256,
        expected_calibration_sha256=calibration_sha256,
    )
    val_recipe = FutureRecipe(
        root / "val",
        expected_recipe_sha256=payload["validation_recipe_sha256"],
        expected_source_manifest_sha256=source_manifest_sha256,
        expected_calibration_sha256=calibration_sha256,
    )
    train_ds.attach_future_recipe(train_recipe)
    val_ds.attach_future_recipe(val_recipe)
    train_ds.future_recipe_sha256 = expected_recipe_sha256
    train_ds.calibration_sha256 = calibration_sha256
    val_ds.future_recipe_sha256 = expected_recipe_sha256
    val_ds.calibration_sha256 = calibration_sha256
    train_ds.cache_metrics = {
        **getattr(train_ds, "cache_metrics", {}),
        "future_recipe/consumed": 1.0,
        "future_recipe/train_rows": float(len(train_recipe.records)),
        "future_recipe/validation_rows": float(len(val_recipe.records)),
    }
