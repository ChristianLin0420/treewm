"""Frontier scoring strategies -- the single axis along which the tree arms differ.

Every tree arm shares one expansion loop and one set of network weights; only the
function that ranks the frontier changes. Keeping them in one file makes the controlled
comparison auditable at a glance:

    bfs           uniform breadth-first, no allocation           (FixedTreeWM)
    random        random frontier choice, matched budget         (RandomTreeWM)
    uncertainty   expand where the model is least certain        (UncertaintyTreeWM)
    heuristic     greedy q-novelty vs the tree context, NO learned parameters
    learned       g_psi(q, c_T, depth, kappa, sigma)             (TreeWM)

``heuristic`` is the control that isolates the word *learned* in "learned compute
allocation". BFS and random are non-adaptive and uncertainty keys off a different
signal, so without this arm a TreeWM win over FixedTreeWM is equally well explained by
"any novelty heuristic beats breadth-first".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from treewm.tree.node import BatchedTree

NEG_INF = -1e9


@dataclass
class ScoringContext:
    """Everything a scorer may read. Deliberately small and explicit."""

    context: torch.Tensor | None = None  # [B, S, q_dim] pooled tree context
    gain_head: torch.nn.Module | None = None
    q_distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
    generator: torch.Generator | None = None
    step: int = 0


def _mask_scores(scores: torch.Tensor, frontier: torch.Tensor) -> torch.Tensor:
    return torch.where(frontier, scores, torch.full_like(scores, NEG_INF))


def bfs_score(tree: BatchedTree, frontier: torch.Tensor, ctx: ScoringContext) -> torch.Tensor:
    """Shallowest first, ties broken by creation order -- exact breadth-first."""
    key = tree.depth.float() * tree.capacity + tree.order.clamp_min(0).float()
    return _mask_scores(-key, frontier)


def random_score(tree: BatchedTree, frontier: torch.Tensor, ctx: ScoringContext) -> torch.Tensor:
    noise = torch.rand(
        tree.valid.shape, device=tree.valid.device, generator=ctx.generator, dtype=torch.float32
    )
    return _mask_scores(noise, frontier)


def uncertainty_score(tree: BatchedTree, frontier: torch.Tensor, ctx: ScoringContext) -> torch.Tensor:
    return _mask_scores(tree.uncertainty.float(), frontier)


def heuristic_score(tree: BatchedTree, frontier: torch.Tensor, ctx: ScoringContext) -> torch.Tensor:
    """Greedy novelty: distance in q-space from the pooled tree context.

    No learned parameters, no training signal -- this is the null model for adaptive
    allocation.
    """
    assert ctx.q_distance is not None, "heuristic scorer needs a q distance function"
    if ctx.context is None:
        return _mask_scores(torch.zeros_like(tree.uncertainty), frontier)
    b, n = tree.valid.shape
    ctx_q = ctx.context.unsqueeze(1).expand(b, n, *ctx.context.shape[1:])
    dist = ctx.q_distance(tree.q.float(), ctx_q.float())  # [B, N]
    return _mask_scores(dist, frontier)


def learned_score(tree: BatchedTree, frontier: torch.Tensor, ctx: ScoringContext) -> torch.Tensor:
    """``g_psi(q_n, c_T, depth, kappa, sigma)`` -- predicted marginal coverage gain."""
    assert ctx.gain_head is not None, "learned scorer needs an ExpansionGainHead"
    gain = ctx.gain_head(
        tree.q.float(),
        ctx.context.float() if ctx.context is not None else None,
        tree.depth,
        tree.keep_score.float(),
        tree.uncertainty.float(),
    )
    return _mask_scores(gain, frontier)


SCORERS: dict[str, Callable[[BatchedTree, torch.Tensor, ScoringContext], torch.Tensor]] = {
    "bfs": bfs_score,
    "random": random_score,
    "uncertainty": uncertainty_score,
    "heuristic": heuristic_score,
    "learned": learned_score,
}


def get_scorer(name: str):
    if name not in SCORERS:
        raise ValueError(f"unknown frontier scorer {name!r}; options: {sorted(SCORERS)}")
    return SCORERS[name]


def select_topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-k frontier slots per tree.

    Returns ``(indices [B, k], valid [B, k])``. Slots whose score is ``NEG_INF`` are
    marked invalid so a tree with a small frontier does not re-expand a node.
    """
    k = max(1, min(k, scores.shape[1]))
    values, indices = torch.topk(scores, k=k, dim=1)
    return indices, values > NEG_INF / 2
