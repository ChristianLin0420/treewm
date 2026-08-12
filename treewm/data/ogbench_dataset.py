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
    ) -> None:
        self.obs = np.ascontiguousarray(dataset["observations"], dtype=np.float32)
        self.act = np.ascontiguousarray(dataset["actions"], dtype=np.float32)
        self.index = TrajectoryIndex.from_terminals(dataset["terminals"])
        self.normalizer = normalizer
        self.cfg = future_cfg
        self.xy_dims = tuple(xy_dims)
        self.obs_dim = self.obs.shape[1]
        self.act_dim = self.act.shape[1]

        self.obs_norm = normalizer.norm_obs(self.obs)
        self.act_norm = normalizer.norm_act(self.act)

        min_h = int(min(future_cfg.horizons))
        remaining = self.index.steps_remaining
        self.valid_mask = remaining >= min_h
        anchors = np.flatnonzero(self.valid_mask)
        if max_anchors is not None and len(anchors) > max_anchors:
            rng = np.random.default_rng(seed)
            anchors = np.sort(rng.choice(anchors, size=max_anchors, replace=False))
        self.anchors = anchors

        self.builder = FutureSetBuilder(
            obs_norm=self.obs_norm,
            act_norm=self.act_norm,
            index=self.index,
            cfg=future_cfg,
            xy_dims=self.xy_dims,
        )

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        t = int(self.anchors[i])
        item = self.builder.build(t)
        return {k: torch.from_numpy(v) if isinstance(v, np.ndarray) else torch.tensor(v) for k, v in item.items()}

    # ------------------------------------------------------------------ helpers

    def observation_at(self, t: int) -> np.ndarray:
        return self.obs[t]

    def summary(self) -> dict[str, Any]:
        return {
            "num_steps": int(len(self.obs)),
            "num_trajectories": int(self.index.num_trajectories),
            "num_anchors": int(len(self.anchors)),
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
) -> tuple[Any, ChunkDataset, ChunkDataset, Normalizer]:
    """Load ``dataset_name`` and build train/val :class:`ChunkDataset` objects."""
    env, train, val = load_ogbench(dataset_name, dataset_dir=dataset_dir)
    normalizer = Normalizer.fit(train["observations"], train["actions"])
    train_ds = ChunkDataset(
        train, normalizer, future_cfg, xy_dims=xy_dims, max_anchors=max_train_anchors, seed=seed
    )
    val_ds = ChunkDataset(
        val, normalizer, future_cfg, xy_dims=xy_dims, max_anchors=max_val_anchors, seed=seed + 1
    )
    return env, train_ds, val_ds, normalizer
