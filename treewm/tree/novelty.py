"""Min-novelty: distance from a node to the nearest node already in the tree.

    G*(n | T) = min_{j in T, j != n} d(q_n, q_j)

This replaces the retrieval-based marginal-coverage gain target. The previous target
counted how many new quantised regions a node's *dataset neighbours* covered; the head
predicted it well (rank correlation +0.43..+0.53) yet allocating by it lost to a
parameter-free novelty rule and anti-scaled with budget. Min-novelty is the signal the
winning heuristic actually acts on, so learning it makes "direct vs learned" a
one-variable comparison.

Both metric spaces are first-class here. q-novelty being better than z-novelty is a
hypothesis to test, not an assumption -- the previous run found q no better than z (and
no better than a random projection of z) at future-set retrieval.
"""

from __future__ import annotations

import torch

BIG = 1e9


def pairwise_min_distance(dist: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Min distance from each node to any *other* valid node. ``[B, N, N] -> [B, N]``.

    Nodes with no valid neighbour (a lone root) get 0 rather than ``BIG``: at the first
    expansion every frontier node is equally novel, and a huge constant would make the
    regression target scale-unstable.
    """
    b, n, _ = dist.shape
    mask_cols = (valid > 0).unsqueeze(1).expand(b, n, n)
    eye = torch.eye(n, device=dist.device, dtype=torch.bool).unsqueeze(0).expand(b, n, n)
    usable = mask_cols & ~eye

    masked = dist.masked_fill(~usable, BIG)
    out = masked.min(dim=-1).values
    return torch.where(usable.any(-1), out, torch.zeros_like(out))


def q_novelty(tree, q_distance_cdist) -> torch.Tensor:
    """``min_j d_q(q_n, q_j)`` for every node. ``[B, N]``."""
    dist = q_distance_cdist(tree.q.float(), tree.q.float())  # [B, N, N]
    return pairwise_min_distance(dist, tree.valid.float())


def z_novelty(tree) -> torch.Tensor:
    """``min_j ||z_n - z_j||_2`` for every node. ``[B, N]``."""
    dist = torch.cdist(tree.latent.float(), tree.latent.float())
    return pairwise_min_distance(dist, tree.valid.float())


def novelty_of(tree, space: str, q_distance_cdist=None) -> torch.Tensor:
    if space == "q":
        assert q_distance_cdist is not None, "q-novelty needs a q cdist function"
        return q_novelty(tree, q_distance_cdist)
    if space == "z":
        return z_novelty(tree)
    raise ValueError(f"unknown novelty space {space!r}; options: q | z")


def node_features(tree, space: str) -> torch.Tensor:
    """Flat per-node features the learned head consumes. ``[B, N, F]``."""
    if space == "q":
        b, n = tree.q.shape[:2]
        return tree.q.float().reshape(b, n, -1)
    if space == "z":
        return tree.latent.float()
    raise ValueError(f"unknown novelty space {space!r}")


def feature_dim(space: str, z_dim: int, q_dim: int, num_scales: int) -> int:
    return q_dim * num_scales if space == "q" else z_dim


@torch.no_grad()
def redundant_fraction(tree, space: str, q_distance_cdist=None, threshold: float = 0.05) -> float:
    """Fraction of valid nodes whose nearest tree neighbour is closer than ``threshold``.

    An expansion that lands on top of somewhere the tree already reaches spent budget for
    nothing, which is precisely what a coverage-allocating policy is supposed to avoid.
    """
    nov = novelty_of(tree, space, q_distance_cdist)
    valid = tree.valid
    if not bool(valid.any()):
        return 0.0
    return float(((nov < threshold) & valid).float().sum() / valid.float().sum())
