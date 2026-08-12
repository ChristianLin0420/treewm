"""Coverage in a globally comparable space.

Mode cluster ids from :mod:`treewm.data.future_sets` are *per-anchor local labels* --
anchor A's "mode 2" has nothing to do with anchor B's "mode 2" -- so they cannot be
used to measure how much of the world a whole tree covers. Coverage therefore uses a
global quantisation of normalised state: two futures count as the same covered region
iff they land in the same cell.

This gives one yardstick used in three places, which is deliberate:

  * the training target for the expansion-gain head (marginal new cells),
  * ``expansion/controllability_coverage`` during training,
  * the evaluation metric against the simulator's true reachable set,

so a model cannot look good on the metric by redefining the space it is scored in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class StateQuantizer:
    """Uniform grid over selected (normalised) state dimensions."""

    resolution: float = 0.2
    dims: tuple[int, ...] = (0, 1)

    def cell_ids(self, states: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
        """Map ``[..., obs_dim]`` states to integer cell ids ``[...]``.

        Ids are produced by mixing per-dimension bin indices with distinct large odd
        multipliers; collisions are possible in principle but astronomically unlikely
        for the ranges involved, and only ever cause a slight *under*-count of coverage.
        """
        if torch.is_tensor(states):
            binned = torch.floor(states[..., list(self.dims)] / self.resolution).to(torch.int64)
            mult = torch.tensor(
                [_MULTIPLIERS[i % len(_MULTIPLIERS)] for i in range(len(self.dims))],
                device=states.device,
                dtype=torch.int64,
            )
            return (binned * mult).sum(-1)
        binned = np.floor(np.asarray(states)[..., list(self.dims)] / self.resolution).astype(np.int64)
        mult = np.array([_MULTIPLIERS[i % len(_MULTIPLIERS)] for i in range(len(self.dims))], dtype=np.int64)
        return (binned * mult).sum(-1)

    def num_unique(self, states: np.ndarray | torch.Tensor) -> int:
        ids = self.cell_ids(states)
        if torch.is_tensor(ids):
            return int(torch.unique(ids).numel())
        return int(len(np.unique(ids)))


_MULTIPLIERS = (1, 73856093, 19349663, 83492791, 39916801, 15485863, 32452843, 49979687)


def batched_new_cell_counts(
    candidate_cells: torch.Tensor,
    candidate_valid: torch.Tensor,
    covered_cells: torch.Tensor,
    covered_valid: torch.Tensor,
) -> torch.Tensor:
    """Count candidate cells not already present in the covered set.

    Args:
        candidate_cells: ``[B, N, M]`` cell ids reachable from each of N candidates.
        candidate_valid: ``[B, N, M]`` float mask.
        covered_cells: ``[B, C]`` cell ids already covered by the tree.
        covered_valid: ``[B, C]`` float mask.

    Returns:
        ``[B, N]`` count of distinct *new* cells per candidate.
    """
    b, n, m = candidate_cells.shape
    # Is each candidate cell already covered?
    already = (candidate_cells.unsqueeze(-1) == covered_cells.view(b, 1, 1, -1)) & (
        covered_valid.view(b, 1, 1, -1) > 0
    )
    already = already.any(-1)  # [B, N, M]

    # De-duplicate within a candidate's own M cells: only the first occurrence counts.
    same = candidate_cells.unsqueeze(-1) == candidate_cells.unsqueeze(-2)  # [B,N,M,M]
    earlier = torch.tril(torch.ones(m, m, device=candidate_cells.device, dtype=torch.bool), diagonal=-1)
    dup = (same & earlier.view(1, 1, m, m)).any(-1)

    fresh = (candidate_valid > 0) & (~already) & (~dup)
    return fresh.float().sum(-1)


def unique_cells_per_row(cells: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Number of distinct cells covered by each tree. ``[B, N] -> [B]``.

    This is the tree-level coverage quantity: how many distinct regions of the world a
    tree's nodes actually reach. Counting nodes instead would reward a model for
    predicting the same place 64 times.
    """
    b, n = cells.shape
    sentinel = torch.iinfo(cells.dtype).max
    masked = torch.where(valid > 0, cells, torch.full_like(cells, sentinel))
    ordered, _ = torch.sort(masked, dim=1)
    counts = (valid > 0).sum(1)  # [B]

    positions = torch.arange(1, n, device=cells.device).view(1, -1)
    is_new = ordered[:, 1:] != ordered[:, :-1]
    within = positions < counts.view(-1, 1)
    return (is_new & within).sum(1) + (counts > 0).long()


def coverage_fraction(
    predicted_states: np.ndarray,
    reference_cells: set[int],
    quantizer: StateQuantizer,
) -> float:
    """Fraction of a reference reachable-set's cells hit by predicted states."""
    if not reference_cells:
        return 0.0
    hit = set(np.unique(quantizer.cell_ids(predicted_states)).tolist())
    return len(hit & reference_cells) / len(reference_cells)


def pairwise_mean_distance(x: torch.Tensor, valid: torch.Tensor | None = None) -> torch.Tensor:
    """Mean pairwise L2 distance over the last-but-one dim. ``x``: ``[B, N, D]``."""
    b, n, _ = x.shape
    d = torch.cdist(x, x)  # [B, N, N]
    mask = ~torch.eye(n, device=x.device, dtype=torch.bool)
    mask = mask.unsqueeze(0).expand(b, n, n)
    if valid is not None:
        v = (valid > 0).float()
        mask = mask & (v.unsqueeze(1) * v.unsqueeze(2)).bool()
    count = mask.float().sum((1, 2)).clamp_min(1.0)
    return (d * mask.float()).sum((1, 2)) / count
