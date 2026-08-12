"""Maze topology helpers.

Everything here is used for **evaluation, visualisation and goal sampling only**.
No training loss may consume maze topology (spec section 11 / anti-goal section 28):
the mode clustering in :mod:`treewm.data.future_sets` operates on raw states, and this
module exists to *validate* those clusters after the fact, not to produce them.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MazeSpec:
    """Grid topology plus the affine map to world coordinates.

    OGBench locomaze places cell ``(i, j)`` at ``xy = (j*unit - offset_x, i*unit - offset_y)``;
    verified against ``env.unwrapped.task_infos`` for pointmaze-medium.
    """

    maze_map: np.ndarray  # [H, W] int, 1 = wall, 0 = free
    unit: float
    offset_x: float
    offset_y: float

    @classmethod
    def from_env(cls, env) -> "MazeSpec":
        u = env.unwrapped
        return cls(
            maze_map=np.asarray(u.maze_map, dtype=np.int32),
            unit=float(u._maze_unit),
            offset_x=float(u._offset_x),
            offset_y=float(u._offset_y),
        )

    @property
    def shape(self) -> tuple[int, int]:
        return self.maze_map.shape

    def ij_to_xy(self, i: int | np.ndarray, j: int | np.ndarray) -> np.ndarray:
        x = np.asarray(j) * self.unit - self.offset_x
        y = np.asarray(i) * self.unit - self.offset_y
        return np.stack([x, y], axis=-1).astype(np.float32)

    def xy_to_ij(self, xy: np.ndarray) -> np.ndarray:
        """Nearest cell index for each xy. Shape [..., 2] -> [..., 2] as (i, j)."""
        xy = np.asarray(xy, dtype=np.float64)
        j = np.rint((xy[..., 0] + self.offset_x) / self.unit).astype(np.int64)
        i = np.rint((xy[..., 1] + self.offset_y) / self.unit).astype(np.int64)
        h, w = self.shape
        return np.stack([np.clip(i, 0, h - 1), np.clip(j, 0, w - 1)], axis=-1)

    def free_cells(self) -> np.ndarray:
        """[N, 2] array of (i, j) for every traversable cell."""
        return np.argwhere(self.maze_map == 0).astype(np.int64)

    def is_free(self, i: int, j: int) -> bool:
        h, w = self.shape
        return 0 <= i < h and 0 <= j < w and self.maze_map[i, j] == 0

    def geodesic_field(self, source_ij: tuple[int, int]) -> np.ndarray:
        """BFS distance in cells from ``source_ij``; unreachable cells are ``inf``."""
        h, w = self.shape
        dist = np.full((h, w), np.inf, dtype=np.float64)
        si, sj = source_ij
        if not self.is_free(si, sj):
            return dist
        dist[si, sj] = 0.0
        queue = deque([(si, sj)])
        while queue:
            i, j = queue.popleft()
            for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ni, nj = i + di, j + dj
                if self.is_free(ni, nj) and not np.isfinite(dist[ni, nj]):
                    dist[ni, nj] = dist[i, j] + 1.0
                    queue.append((ni, nj))
        return dist

    def all_pairs_geodesic(self) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(cells [N,2], dist [N,N])`` in cell units."""
        cells = self.free_cells()
        index = {(int(i), int(j)): k for k, (i, j) in enumerate(cells)}
        n = len(cells)
        dist = np.full((n, n), np.inf, dtype=np.float64)
        for k, (i, j) in enumerate(cells):
            field = self.geodesic_field((int(i), int(j)))
            for (ci, cj), m in index.items():
                dist[k, m] = field[ci, cj]
        return cells, dist

    def junction_degree(self) -> np.ndarray:
        """Number of free 4-neighbours per cell; walls get 0.

        This is the hand-free proxy for "corridor vs junction" used to sanity-check the
        branching-factor heatmap (spec section 20). It is a *diagnostic*, never a target.
        """
        h, w = self.shape
        deg = np.zeros((h, w), dtype=np.int32)
        for i in range(h):
            for j in range(w):
                if self.maze_map[i, j] != 0:
                    continue
                deg[i, j] = sum(self.is_free(i + di, j + dj) for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        return deg


def sample_hard_goal_pairs(
    spec: MazeSpec,
    num_pairs: int,
    percentile: float = 75.0,
    seed: int = 0,
) -> list[dict]:
    """Sample start/goal pairs whose geodesic separation is in the top ``percentile``.

    This is the "hard split" of the evaluation protocol: pairs that are far apart in
    *maze* distance, not euclidean distance, so a straight-line dash cannot solve them.
    """
    cells, dist = spec.all_pairs_geodesic()
    finite = np.isfinite(dist) & (dist > 0)
    if not finite.any():
        raise ValueError("maze has no connected free-cell pairs")
    threshold = np.percentile(dist[finite], percentile)
    candidates = np.argwhere(finite & (dist >= threshold))
    rng = np.random.default_rng(seed)
    if len(candidates) == 0:
        raise ValueError("no start/goal pairs above the geodesic threshold")
    picks = rng.choice(len(candidates), size=min(num_pairs, len(candidates)), replace=False)

    pairs = []
    for p in picks:
        a, b = candidates[p]
        si, sj = cells[a]
        gi, gj = cells[b]
        pairs.append(
            {
                "init_ij": (int(si), int(sj)),
                "goal_ij": (int(gi), int(gj)),
                "init_xy": spec.ij_to_xy(int(si), int(sj)).tolist(),
                "goal_xy": spec.ij_to_xy(int(gi), int(gj)).tolist(),
                "geodesic": float(dist[a, b]),
            }
        )
    pairs.sort(key=lambda d: -d["geodesic"])
    return pairs


def env_task_pairs(env) -> list[dict]:
    """The environment's own built-in tasks, in OGBench's ordering (task_id 1..N)."""
    infos = env.unwrapped.task_infos
    return [
        {
            "task_id": k + 1,
            "task_name": info["task_name"],
            "init_ij": tuple(info["init_ij"]),
            "goal_ij": tuple(info["goal_ij"]),
            "init_xy": list(info["init_xy"]),
            "goal_xy": list(info["goal_xy"]),
        }
        for k, info in enumerate(infos)
    ]
