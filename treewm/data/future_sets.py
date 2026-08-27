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


def bounded_uniform_indices(population: int, size: int, seed: int = 0) -> np.ndarray:
    """Uniform sample without replacement using memory proportional to ``size``.

    NumPy is free to allocate work proportional to ``population`` for
    ``choice(population, replace=False)``.  That is unacceptable when the retrieval
    source contains 100M transitions and the desired fixed pool contains only 50k.
    Rejection sampling is symmetric and has negligible collision overhead at that
    sampling ratio.
    """
    population = int(population)
    size = int(size)
    if not 0 <= size <= population:
        raise ValueError(f"cannot sample {size} indices from population {population}")
    if size == 0:
        return np.empty(0, dtype=np.int64)
    rng = np.random.default_rng(seed)
    if population <= 10_000_000:
        return np.sort(rng.choice(population, size=size, replace=False)).astype(
            np.int64, copy=False
        )
    selected = np.empty(0, dtype=np.int64)
    while len(selected) < size:
        needed = size - len(selected)
        draws = rng.integers(
            0, population, size=max(needed * 2, 1024), dtype=np.int64
        )
        selected = np.unique(np.concatenate((selected, draws)))
    if len(selected) > size:
        selected = rng.choice(selected, size=size, replace=False)
    return np.sort(selected).astype(np.int64, copy=False)


@dataclass
class FutureSetConfig:
    """All retrieval/clustering knobs. Every one is exposed in ``configs/base.yaml``."""

    num_neighbors: int = 24
    query_multiplier: int = 6
    time_exclusion: int = 50
    retrieval_radius: float = 1.0
    include_self: bool = True
    # ``rms_v2`` makes every Euclidean threshold independent of the number of
    # coordinates used by dividing distances by sqrt(dimension).  ``legacy_l2`` is
    # retained only so old experiment snapshots can still be inspected explicitly.
    # Legacy remains the library default so historical configs do not silently change
    # units. Formal v2 must pin ``rms_v2`` explicitly in its immutable protocol.
    metric_mode: str = "legacy_l2"  # rms_v2 | legacy_l2

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
    # Prospective, opt-in executable-prefix target. Zero preserves the historical item
    # schema and materialization loop exactly. A positive value materializes the logged
    # endpoint reached after min(value, selected_horizon) actions from the same
    # continuation that supplies each future slot.
    executable_prefix_steps: int = 0
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
        assert self.num_neighbors >= 1
        assert self.query_multiplier >= 1
        assert self.metric_mode in {"rms_v2", "legacy_l2"}
        raw_prefix_steps = self.executable_prefix_steps
        if isinstance(raw_prefix_steps, bool):
            raise ValueError("executable_prefix_steps must be an integer")
        try:
            self.executable_prefix_steps = int(raw_prefix_steps)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("executable_prefix_steps must be an integer") from exc
        if self.executable_prefix_steps != raw_prefix_steps:
            raise ValueError("executable_prefix_steps must be an integer")
        if self.executable_prefix_steps < 0:
            raise ValueError("executable_prefix_steps must be non-negative")
        if self.executable_prefix_steps > 0:
            if self.executable_prefix_steps > self.h_max:
                raise ValueError("executable_prefix_steps cannot exceed h_max")
            if self.horizons.count(self.executable_prefix_steps) != 1:
                raise ValueError(
                    "executable_prefix_steps must be one unique candidate horizon"
                )
        if not np.isfinite(self.retrieval_radius) or self.retrieval_radius < 0:
            raise ValueError("retrieval_radius must be finite and non-negative")
        if not np.isfinite(self.displacement_threshold) or self.displacement_threshold < 0:
            raise ValueError("displacement_threshold must be finite and non-negative")
        if not np.isfinite(self.cluster_threshold) or self.cluster_threshold <= 0:
            raise ValueError("cluster_threshold must be finite and positive")

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
        task_metric_dims: tuple[int, ...] | None = None,
    ) -> None:
        self.obs_norm = obs_norm
        self.act_norm = act_norm
        self.index = index
        self.cfg = cfg
        self.xy_dims = np.asarray(xy_dims, dtype=np.int64)
        self.obs_dim = obs_norm.shape[1]
        self.act_dim = act_norm.shape[1]
        metric_dims = xy_dims if task_metric_dims is None else task_metric_dims
        self.task_metric_dims = np.asarray(metric_dims, dtype=np.int64)
        if self.task_metric_dims.ndim != 1 or len(self.task_metric_dims) == 0:
            raise ValueError("task_metric_dims must contain at least one observation coordinate")
        if len(np.unique(self.task_metric_dims)) != len(self.task_metric_dims):
            raise ValueError("task_metric_dims must not contain duplicate coordinates")
        if self.task_metric_dims.min() < 0 or self.task_metric_dims.max() >= self.obs_dim:
            raise ValueError(
                f"task_metric_dims {tuple(metric_dims)} are outside observation width {self.obs_dim}"
            )
        if self.xy_dims.ndim != 1 or (
            len(self.xy_dims) and (self.xy_dims.min() < 0 or self.xy_dims.max() >= self.obs_dim)
        ):
            raise ValueError(f"xy_dims {tuple(xy_dims)} are outside observation width {self.obs_dim}")
        self.horizons = np.asarray(cfg.horizons, dtype=np.int64)
        self._remaining = index.steps_remaining
        self._tree = None  # built lazily: one per worker process

    def _distance_scale(self, dimension: int) -> float:
        """Denominator that converts L2 into the configured metric units."""
        if self.cfg.metric_mode == "rms_v2":
            return float(np.sqrt(dimension))
        return 1.0

    def _metric_coordinates(self, observations: np.ndarray) -> np.ndarray:
        """Standardized task coordinates used for horizons, modes and diversity."""
        return observations[..., self.task_metric_dims]

    def _metric_distance(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """L2/RMS-L2 over task coordinates, preserving all leading dimensions."""
        delta = np.asarray(left) - np.asarray(right)
        return np.linalg.norm(delta, axis=-1) / self._distance_scale(delta.shape[-1])

    # ------------------------------------------------------------------- lookup

    @property
    def tree(self):
        if self._tree is None:
            from scipy.spatial import cKDTree

            # Retrieval uses the full standardized raw state, which is stable from
            # step 0 (spec section 11 recommends this over a moving learned latent).
            # Dividing returned distances by sqrt(D) below gives RMS-L2 without
            # materializing a scaled copy of a potentially multi-million-row corpus;
            # uniform scalar scaling does not change nearest-neighbour ordering.
            pool = int(self.cfg.retrieval_pool)
            if pool and pool < len(self.obs_norm):
                # Fixed and bounded-memory: the pool must not vary by worker, and a
                # 50k selection must not allocate an O(100M) permutation per process.
                self._pool_idx = bounded_uniform_indices(len(self.obs_norm), pool, seed=0)
                # Index the memmap before constructing the tree. A 50k pool must never
                # materialize or permute the full 100M-row source in every worker.
                retrieval_obs = np.asarray(self.obs_norm[self._pool_idx])
                self._tree = cKDTree(retrieval_obs)
            else:
                self._pool_idx = None
                self._tree = cKDTree(self.obs_norm)
        return self._tree

    def _neighbors(self, t: int) -> np.ndarray:
        cfg = self.cfg
        k = min(cfg.num_neighbors * cfg.query_multiplier + 1, len(self.obs_norm))
        dists, idxs = self.tree.query(self.obs_norm[t], k=k)
        idxs = np.atleast_1d(idxs)
        dists = np.atleast_1d(dists) / self._distance_scale(self.obs_dim)
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

        # Deliberately return every eligible result in the queried pool. ``build`` caps
        # the materialized tensor only after recording the raw candidate count.
        selected = idxs[keep]
        selected_distances = dists[keep].astype(np.float32, copy=False)
        if cfg.include_self:
            # The anchor's own continuation is a genuine supported future and anchors
            # the set; it is always slot 0.
            selected = np.concatenate([[t], selected])
            selected_distances = np.concatenate(
                [np.zeros(1, dtype=np.float32), selected_distances]
            )
        # A saturated query means there may be additional in-radius candidates beyond
        # query_multiplier's bounded search. This is reported separately rather than
        # being silently presented as an exact exhaustive count.
        self._last_retrieval_distances = selected_distances
        self._last_retrieval_query_saturated = bool(
            k < len(self.obs_norm) and len(dists) == k and float(dists[-1]) <= cfg.retrieval_radius
        )
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
        base = self._metric_coordinates(self.obs_norm[c])
        for h in feasible:
            endpoint = self._metric_coordinates(self.obs_norm[c + int(h)])
            disp = float(self._metric_distance(endpoint, base))
            if disp > cfg.displacement_threshold:
                return int(h)
        return int(feasible.max())

    # ---------------------------------------------------------------- clustering

    def _cluster(
        self,
        endpoints: np.ndarray,
        rng: np.random.Generator,
        *,
        return_raw_count: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Cluster endpoints into modes.

        Returns ``(labels [M], rep_idx [C], mass [C])`` with labels in ``[0, C)``.
        """
        m = len(endpoints)
        if m == 0:
            empty_i = np.empty(0, dtype=np.int64)
            empty_f = np.empty(0, dtype=np.float32)
            result = (empty_i, empty_i.copy(), empty_f)
            return (*result, 0) if return_raw_count else result
        if m == 1:
            result = (
                np.zeros(1, dtype=np.int64),
                np.zeros(1, dtype=np.int64),
                np.ones(1, dtype=np.float32),
            )
            return (*result, 1) if return_raw_count else result

        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import pdist

        d = pdist(endpoints) / self._distance_scale(endpoints.shape[-1])
        if not np.isfinite(d).all() or d.size == 0:
            labels = np.zeros(m, dtype=np.int64)
        else:
            z = linkage(d, method=self.cfg.cluster_method)
            labels = fcluster(z, t=self.cfg.cluster_threshold, criterion="distance").astype(np.int64) - 1

        num = int(labels.max()) + 1
        raw_num = num
        reps = np.zeros(num, dtype=np.int64)
        mass = np.zeros(num, dtype=np.float32)
        for c in range(num):
            members = np.flatnonzero(labels == c)
            centroid = endpoints[members].mean(0)
            reps[c] = members[
                np.argmin(self._metric_distance(endpoints[members], centroid))
            ]
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
        result = (labels, reps, mass)
        return (*result, raw_num) if return_raw_count else result

    # ---------------------------------------------------------------------- api

    def build(self, t: int) -> dict[str, np.ndarray]:
        cfg = self.cfg
        rng = np.random.default_rng(t)  # deterministic per anchor -> reproducible eval
        # `_neighbors` intentionally returns its uncapped eligible set so truncation is
        # visible. Keep calling the public-ish method here because several dataset tests
        # and adapters replace it with a deterministic retrieval fixture.
        self._last_retrieval_distances = np.empty(0, dtype=np.float32)
        self._last_retrieval_query_saturated = False
        neighbors_raw = np.asarray(self._neighbors(t), dtype=np.int64)
        retrieval_num_candidates = len(neighbors_raw)
        neighbors = neighbors_raw[: cfg.num_neighbors]
        m_used = len(neighbors)
        distances = np.asarray(self._last_retrieval_distances, dtype=np.float32)[:m_used]

        m = cfg.num_neighbors
        fut_actions = np.zeros((m, cfg.h_max, self.act_dim), dtype=np.float32)
        fut_mask = np.zeros((m, cfg.h_max), dtype=np.float32)
        fut_endpoint = np.zeros((m, self.obs_dim), dtype=np.float32)
        fut_metric_endpoint = np.zeros(
            (m, len(self.task_metric_dims)), dtype=np.float32
        )
        fut_h_idx = np.zeros(m, dtype=np.int64)
        fut_h_len = np.zeros(m, dtype=np.float32)
        fut_valid = np.zeros(m, dtype=np.float32)

        prefix_steps = int(cfg.executable_prefix_steps)
        if prefix_steps > 0:
            fut_prefix_endpoint = np.zeros((m, self.obs_dim), dtype=np.float32)
            fut_prefix_metric_endpoint = np.zeros(
                (m, len(self.task_metric_dims)), dtype=np.float32
            )
            fut_prefix_mask = np.zeros((m, cfg.h_max), dtype=np.float32)
            fut_prefix_h_idx = np.zeros(m, dtype=np.int64)
            fut_prefix_len = np.zeros(m, dtype=np.float32)

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
            fut_metric_endpoint[slot] = self._metric_coordinates(ep)
            fut_actions[slot, :h] = self.act_norm[c : c + h]
            fut_mask[slot, :h] = 1.0
            fut_h_idx[slot] = h_lookup[h]
            fut_h_len[slot] = float(h)
            fut_valid[slot] = 1.0
            if prefix_steps > 0:
                prefix_len = min(prefix_steps, h)
                prefix_end = self.obs_norm[c + prefix_len]
                if cfg.relative_endpoints:
                    prefix_end = prefix_end.copy()
                    prefix_end[self.xy_dims] = anchor[self.xy_dims] + (
                        prefix_end[self.xy_dims] - self.obs_norm[c][self.xy_dims]
                    )
                fut_prefix_endpoint[slot] = prefix_end
                fut_prefix_metric_endpoint[slot] = self._metric_coordinates(
                    prefix_end
                )
                fut_prefix_mask[slot, :prefix_len] = 1.0
                fut_prefix_h_idx[slot] = h_lookup[prefix_len]
                fut_prefix_len[slot] = float(prefix_len)

        labels_used, reps_used, mass_used, num_modes_raw = self._cluster(
            fut_metric_endpoint[:m_used], rng, return_raw_count=True
        )

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

        # Diversity describes the raw retrieved support before max_modes truncation.
        # The task-coordinate RMS metric is the same one used by clustering/horizons.
        raw_metric_eps = fut_metric_endpoint[:m_used]
        if len(raw_metric_eps) > 1:
            spread = float(
                self._metric_distance(
                    raw_metric_eps[:, None, :], raw_metric_eps[None, :, :]
                ).mean()
            )
        else:
            spread = 0.0

        nonself_distances = distances[distances > 0]
        retrieval_mean_distance = (
            float(nonself_distances.mean()) if len(nonself_distances) else 0.0
        )
        query_saturated = bool(self._last_retrieval_query_saturated)
        retrieval_truncated = retrieval_num_candidates > m_used or query_saturated
        retrieval_fallback = int(
            sum(int(idx) != int(t) for idx in neighbors) == 0
        )

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

        item = {
            "anchor_index": np.int64(t),
            # Repeated per item so a collated batch is self-describing to loss and
            # diagnostic code; every item in one dataset has the same index vector.
            "task_metric_dims": self.task_metric_dims.astype(np.int64, copy=True),
            "ms_actions": ms_actions,
            "ms_action_mask": ms_mask,
            "ms_obs": ms_obs,
            "ms_horizon_idx": ms_h_idx,
            "ms_valid": ms_valid,
            "obs": anchor.astype(np.float32),
            "fut_actions": fut_actions,
            "fut_action_mask": fut_mask,
            "fut_endpoint": fut_endpoint,
            "fut_metric_endpoint": fut_metric_endpoint,
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
            # V2 retrieval/mode telemetry. Counts are recorded both before and after
            # materialization/mode caps; query saturation says when the raw candidate
            # count is only a lower bound because query_multiplier bounded the search.
            "retrieval_num_candidates": np.int64(retrieval_num_candidates),
            "retrieval_num_valid": np.int64(m_used),
            "retrieval_mean_distance": np.float32(retrieval_mean_distance),
            "retrieval_fallback": np.float32(retrieval_fallback),
            "retrieval_truncated": np.float32(retrieval_truncated),
            "retrieval_query_saturated": np.float32(query_saturated),
            "modes_raw": np.int64(num_modes_raw),
            "modes_retained": np.int64(num_modes),
            "modes_truncated": np.int64(max(num_modes_raw - num_modes, 0)),
        }
        if prefix_steps > 0:
            item.update(
                {
                    "fut_executable_prefix_endpoint": fut_prefix_endpoint,
                    "fut_executable_prefix_metric_endpoint": (
                        fut_prefix_metric_endpoint
                    ),
                    "fut_executable_prefix_action_mask": fut_prefix_mask,
                    "fut_executable_prefix_horizon_idx": fut_prefix_h_idx,
                    "fut_executable_prefix_len": fut_prefix_len,
                }
            )
        return item


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

    out = {
        "actions": _gather(batch["fut_actions"]),
        "action_mask": _gather(batch["fut_action_mask"]),
        "endpoint": _gather(batch["fut_endpoint"]),
        "horizon_idx": torch.gather(batch["fut_horizon_idx"], 1, rep),
        "horizon_len": torch.gather(batch["fut_horizon_len"], 1, rep),
        "mass": batch["mode_mass"],
        "valid": batch["mode_valid"],
    }
    if "fut_metric_endpoint" in batch:
        out["metric_endpoint"] = _gather(batch["fut_metric_endpoint"])
    prefix_fields = {
        "executable_prefix_endpoint": "fut_executable_prefix_endpoint",
        "executable_prefix_metric_endpoint": (
            "fut_executable_prefix_metric_endpoint"
        ),
        "executable_prefix_action_mask": "fut_executable_prefix_action_mask",
        "executable_prefix_horizon_idx": (
            "fut_executable_prefix_horizon_idx"
        ),
        "executable_prefix_len": "fut_executable_prefix_len",
    }
    present = [source in batch for source in prefix_fields.values()]
    if any(present) and not all(present):
        raise ValueError("executable-prefix future tensors must be present as one set")
    if all(present):
        for target, source in prefix_fields.items():
            out[target] = _gather(batch[source])
    return out
