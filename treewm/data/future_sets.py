"""Local reachable-future sets and their decomposition into controllability modes.

This module answers the question the rest of the codebase depends on: *what is a
distinct future mode?*

An offline trajectory gives one continuation per state, so alternatives are obtained by
retrieving dataset states near the anchor and borrowing their continuations
(spec section 11). Those continuations are then clustered by where they *end up*, and
each cluster is one mode. That yields two separate, non-interchangeable targets:

    support (kappa)  1 per cluster, regardless of how many samples it holds
    mass    (rho)    cluster frequency

which is the support-vs-frequency distinction of spec section 13. A mode holding 5% of
the samples is exactly as "supported" as one holding 80%; only its mass differs. Any
implementation that ranks clusters by size and truncates would silently delete rare
valid modes, so truncation here is *random*, never mass-ordered.

Nothing in this module reads maze topology.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class FutureSetConfig:
    """All retrieval/clustering knobs. Every one is exposed in ``configs/base.yaml``."""

    num_neighbors: int = 24
    query_multiplier: int = 6
    time_exclusion: int = 50
    retrieval_radius: float = 1.0
    include_self: bool = True

    horizons: tuple[int, ...] = (4, 8, 16, 32, 64)
    h_max: int = 64
    horizon_rule: str = "displacement"  # displacement | fixed | random
    displacement_threshold: float = 0.15
    fixed_horizon: int = 32

    relative_endpoints: bool = True
    cluster_threshold: float = 0.12
    cluster_method: str = "average"
    max_modes: int = 8
    # Track A: supervise the anchor's own continuation at depths 1..multi_step_depth so
    # recursion is trained on chained predictions, not only single edges.
    multi_step_depth: int = 3
    # Size of the retrieval pool. k-d trees degrade toward a linear scan in high
    # dimensions, and AntMaze observations are 29-D: querying 145 neighbours over 1M
    # points measured 0.37 it/s (9h for a 12k-step run). Subsampling the pool restores
    # sublinear behaviour; it changes which neighbours are found, not what a mode is.
    # 0 keeps the whole dataset (the PointMaze default).
    retrieval_pool: int = 0

    def __post_init__(self) -> None:
        self.horizons = tuple(int(h) for h in self.horizons)
        assert len(self.horizons) > 0, "at least one candidate horizon is required"
        assert self.h_max >= max(self.horizons), "h_max must cover the largest candidate horizon"
        assert self.horizon_rule in {"displacement", "fixed", "random"}
        assert self.max_modes >= 1

    @property
    def num_horizons(self) -> int:
        return len(self.horizons)


class FutureSetBuilder:
    """Builds one anchor's future set. Safe to use inside DataLoader workers."""

    def __init__(
        self,
        obs_norm: np.ndarray,
        act_norm: np.ndarray,
        index: Any,
        cfg: FutureSetConfig,
        xy_dims: tuple[int, ...] = (0, 1),
    ) -> None:
        self.obs_norm = obs_norm
        self.act_norm = act_norm
        self.index = index
        self.cfg = cfg
        self.xy_dims = np.asarray(xy_dims, dtype=np.int64)
        self.obs_dim = obs_norm.shape[1]
        self.act_dim = act_norm.shape[1]
        self.horizons = np.asarray(cfg.horizons, dtype=np.int64)
        self._remaining = index.steps_remaining
        self._tree = None  # built lazily: one per worker process

    # ------------------------------------------------------------------- lookup

    @property
    def tree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree

            # Retrieval uses normalised *raw* state, which is stable from step 0
            # (spec section 11 recommends this over a moving learned latent).
            pool = int(self.cfg.retrieval_pool)
            if pool and pool < len(self.obs_norm):
                rng = np.random.default_rng(0)  # fixed: the pool must not vary by worker
                self._pool_idx = np.sort(rng.choice(len(self.obs_norm), size=pool, replace=False))
                self._tree = cKDTree(self.obs_norm[self._pool_idx])
            else:
                self._pool_idx = None
                self._tree = cKDTree(self.obs_norm)
        return self._tree

    def _neighbors(self, t: int) -> np.ndarray:
        cfg = self.cfg
        k = min(cfg.num_neighbors * cfg.query_multiplier + 1, len(self.obs_norm))
        dists, idxs = self.tree.query(self.obs_norm[t], k=k)
        idxs = np.atleast_1d(idxs)
        dists = np.atleast_1d(dists)
        if getattr(self, "_pool_idx", None) is not None:
            idxs = self._pool_idx[idxs]  # map pool positions back to dataset indices

        min_h = int(self.horizons.min())
        keep = (
            (dists <= cfg.retrieval_radius)
            & (self._remaining[idxs] >= min_h)
            & (idxs != t)
        )
        # Temporal neighbours from the same trajectory share a continuation, so they
        # would inflate the dominant mode without adding an alternative.
        same_traj = self.index.traj_id[idxs] == self.index.traj_id[t]
        too_close = np.abs(idxs.astype(np.int64) - t) < cfg.time_exclusion
        keep &= ~(same_traj & too_close)

        selected = idxs[keep][: cfg.num_neighbors - (1 if cfg.include_self else 0)]
        if cfg.include_self:
            # The anchor's own continuation is a genuine supported future and anchors
            # the set; it is always slot 0.
            selected = np.concatenate([[t], selected])
        return selected.astype(np.int64)

    # ------------------------------------------------------------------ horizon

    def _pick_horizon(self, c: int, rng: np.random.Generator) -> int:
        cfg = self.cfg
        remaining = int(self._remaining[c])
        feasible = self.horizons[self.horizons <= remaining]
        if len(feasible) == 0:
            return int(self.horizons.min())

        if cfg.horizon_rule == "fixed":
            h = min(cfg.fixed_horizon, int(feasible.max()))
            return int(feasible[np.abs(feasible - h).argmin()])
        if cfg.horizon_rule == "random":
            return int(rng.choice(feasible))

        # "displacement": the duration of a transition mode is how long it takes the
        # state to actually move somewhere, which makes the horizon head predict
        # something meaningful instead of a constant.
        base = self.obs_norm[c]
        for h in feasible:
            disp = np.linalg.norm(self.obs_norm[c + int(h)] - base)
            if disp > cfg.displacement_threshold:
                return int(h)
        return int(feasible.max())

    # ---------------------------------------------------------------- clustering

    def _cluster(self, endpoints: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cluster endpoints into modes.

        Returns ``(labels [M], rep_idx [C], mass [C])`` with labels in ``[0, C)``.
        """
        m = len(endpoints)
        if m == 1:
            return np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64), np.ones(1, dtype=np.float32)

        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        d = pdist(endpoints)
        if not np.isfinite(d).all() or d.size == 0:
            labels = np.zeros(m, dtype=np.int64)
        else:
            z = linkage(d, method=self.cfg.cluster_method)
            labels = fcluster(z, t=self.cfg.cluster_threshold, criterion="distance").astype(np.int64) - 1

        num = int(labels.max()) + 1
        reps = np.zeros(num, dtype=np.int64)
        mass = np.zeros(num, dtype=np.float32)
        for c in range(num):
            members = np.flatnonzero(labels == c)
            centroid = endpoints[members].mean(0)
            reps[c] = members[np.argmin(np.linalg.norm(endpoints[members] - centroid, axis=1))]
            mass[c] = len(members) / m

        if num > self.cfg.max_modes:
            # Random, NOT mass-ordered. Dropping the smallest clusters would delete
            # exactly the rare-but-valid modes this project exists to preserve.
            chosen = np.sort(rng.choice(num, size=self.cfg.max_modes, replace=False))
            remap = -np.ones(num, dtype=np.int64)
            remap[chosen] = np.arange(len(chosen))
            labels = remap[labels]
            reps, mass = reps[chosen], mass[chosen]
            mass = mass / max(mass.sum(), 1e-8)
        return labels, reps, mass

    # ---------------------------------------------------------------------- api

    def build(self, t: int) -> dict[str, np.ndarray]:
        cfg = self.cfg
        rng = np.random.default_rng(t)  # deterministic per anchor -> reproducible eval
        neighbors = self._neighbors(t)
        m_used = len(neighbors)

        m = cfg.num_neighbors
        fut_actions = np.zeros((m, cfg.h_max, self.act_dim), dtype=np.float32)
        fut_mask = np.zeros((m, cfg.h_max), dtype=np.float32)
        fut_endpoint = np.zeros((m, self.obs_dim), dtype=np.float32)
        fut_h_idx = np.zeros(m, dtype=np.int64)
        fut_h_len = np.zeros(m, dtype=np.float32)
        fut_valid = np.zeros(m, dtype=np.float32)

        anchor = self.obs_norm[t]
        h_lookup = {int(h): i for i, h in enumerate(self.horizons)}

        for slot, c in enumerate(neighbors):
            c = int(c)
            h = self._pick_horizon(c, rng)
            end = self.obs_norm[c + h]
            if cfg.relative_endpoints:
                # Transfer the neighbour's *displacement* onto the anchor: the local
                # chart assumption. Non-positional dims are taken from the neighbour
                # directly, which is what generalises to AntMaze.
                ep = end.copy()
                ep[self.xy_dims] = anchor[self.xy_dims] + (end[self.xy_dims] - self.obs_norm[c][self.xy_dims])
            else:
                ep = end.copy()
            fut_endpoint[slot] = ep
            fut_actions[slot, :h] = self.act_norm[c : c + h]
            fut_mask[slot, :h] = 1.0
            fut_h_idx[slot] = h_lookup[h]
            fut_h_len[slot] = float(h)
            fut_valid[slot] = 1.0

        labels_used, reps_used, mass_used = self._cluster(fut_endpoint[:m_used], rng)

        cluster = -np.ones(m, dtype=np.int64)
        cluster[:m_used] = labels_used
        # Slots dropped by max_modes truncation become invalid rather than silently
        # joining another mode.
        fut_valid[:m_used] = np.where(labels_used >= 0, fut_valid[:m_used], 0.0)

        c_max = cfg.max_modes
        mode_rep = -np.ones(c_max, dtype=np.int64)
        mode_mass = np.zeros(c_max, dtype=np.float32)
        mode_valid = np.zeros(c_max, dtype=np.float32)
        num_modes = min(len(reps_used), c_max)
        mode_rep[:num_modes] = reps_used[:num_modes]
        mode_mass[:num_modes] = mass_used[:num_modes]
        mode_valid[:num_modes] = 1.0

        valid_eps = fut_endpoint[fut_valid > 0]
        if len(valid_eps) > 1:
            spread = float(np.linalg.norm(valid_eps[:, None] - valid_eps[None], axis=-1).mean())
        else:
            spread = 0.0

        # ---- multi-step chain along the anchor's own trajectory (Track A) ----
        d_max = cfg.multi_step_depth
        ms_actions = np.zeros((d_max, cfg.h_max, self.act_dim), dtype=np.float32)
        ms_mask = np.zeros((d_max, cfg.h_max), dtype=np.float32)
        ms_obs = np.tile(anchor.astype(np.float32), (d_max, 1))
        ms_h_idx = np.zeros(d_max, dtype=np.int64)
        ms_valid = np.zeros(d_max, dtype=np.float32)
        cursor = t
        for d in range(d_max):
            if self._remaining[cursor] < int(self.horizons.min()):
                break
            h = self._pick_horizon(cursor, rng)
            if h > int(self._remaining[cursor]):
                break
            ms_actions[d, :h] = self.act_norm[cursor : cursor + h]
            ms_mask[d, :h] = 1.0
            ms_obs[d] = self.obs_norm[cursor + h]
            ms_h_idx[d] = h_lookup[h]
            ms_valid[d] = 1.0
            cursor = cursor + h

        return {
            "anchor_index": np.int64(t),
            "ms_actions": ms_actions,
            "ms_action_mask": ms_mask,
            "ms_obs": ms_obs,
            "ms_horizon_idx": ms_h_idx,
            "ms_valid": ms_valid,
            "obs": anchor.astype(np.float32),
            "fut_actions": fut_actions,
            "fut_action_mask": fut_mask,
            "fut_endpoint": fut_endpoint,
            "fut_horizon_idx": fut_h_idx,
            "fut_horizon_len": fut_h_len,
            "fut_valid": fut_valid,
            "fut_cluster": cluster,
            "mode_rep": mode_rep,
            "mode_mass": mode_mass,
            "mode_valid": mode_valid,
            "num_modes": np.int64(num_modes),
            # Empirical local future diversity -- the model-free quantity that
            # `control/branching_future_diversity_corr` correlates against.
            "future_diversity": np.float32(spread),
            "num_retrieved": np.int64(m_used),
        }


def gather_mode_targets(batch: dict[str, Any]) -> dict[str, Any]:
    """Pull per-mode representative futures out of a collated batch.

    Returns tensors shaped ``[B, C_max, ...]`` holding, for each mode, the *executable
    representative* future -- one real action chunk, never an average of several
    (spec section 10).
    """
    import torch

    rep = batch["mode_rep"].clamp_min(0)  # [B, C]
    b, c = rep.shape

    def _gather(x: "torch.Tensor") -> "torch.Tensor":
        idx = rep.view(b, c, *([1] * (x.dim() - 2))).expand(b, c, *x.shape[2:])
        return torch.gather(x, 1, idx)

    return {
        "actions": _gather(batch["fut_actions"]),
        "action_mask": _gather(batch["fut_action_mask"]),
        "endpoint": _gather(batch["fut_endpoint"]),
        "horizon_idx": torch.gather(batch["fut_horizon_idx"], 1, rep),
        "horizon_len": torch.gather(batch["fut_horizon_len"], 1, rep),
        "mass": batch["mode_mass"],
        "valid": batch["mode_valid"],
    }
