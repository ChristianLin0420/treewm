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

    context: torch.Tensor | None = None  # [B, S, q_dim] pooled q context (heuristic)
    context_flat: torch.Tensor | None = None  # [B, F] pooled context for the gain head
    gain_head: torch.nn.Module | None = None
    q_distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
    q_cdist: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None
    generator: torch.Generator | None = None
    step: int = 0
    novelty_space: str = "q"  # q | z -- metric space for novelty scorers and the head
    depth_penalty: float = 0.0  # lambda in  S = novelty - lambda * depth
    # Goal-directed allocation. The goal reaches the *frontier ordering only* -- the
    # branch network, dynamics and q never see it, so the world model stays
    # goal-independent (spec section 8). A tree built this way is, however, no longer
    # reusable across goals.
    goal_obs: torch.Tensor | None = None  # [B, obs_dim] normalised goal observation
    decoder: torch.nn.Module | None = None
    alpha: float = 0.0  # diversity weight in  S = -d_goal + alpha * novelty_q
    budget_fraction: float = 0.0  # fraction of the node budget already spent (D6)
    broad_fraction: float = 0.45  # D6 switch point: broad below, goal-directed above
    depth_pools: int = 3  # D2 number of depth strata


def _mask_scores(scores: torch.Tensor, frontier: torch.Tensor) -> torch.Tensor:
    return torch.where(frontier, scores, torch.full_like(scores, NEG_INF))


def bfs_score(tree: BatchedTree, frontier: torch.Tensor, ctx: ScoringContext) -> torch.Tensor:
    """Shallowest first, ties broken by creation order -- exact breadth-first."""
    key = tree.depth.float() * tree.capacity + tree.order.clamp_min(0).float()
    return _mask_scores(-key, frontier)


def random_score(tree: BatchedTree, frontier: torch.Tensor, ctx: ScoringContext) -> torch.Tensor:
    # Fail loudly rather than fall back to the global stream: silently sharing it with
    # training and visualisation is exactly what made results depend on viz cadence.
    assert ctx.generator is not None, (
        "random frontier expansion requires an explicit generator; pass one from "
        "RngStreams so logging cannot perturb training or planning"
    )
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


def novelty_q_score(tree: BatchedTree, frontier: torch.Tensor, ctx: ScoringContext) -> torch.Tensor:
    """Direct ``min_j d_q(q_n, q_j)`` -- the exact signal the learned head must predict."""
    from treewm.tree.novelty import q_novelty

    assert ctx.q_cdist is not None, "q-novelty scorer needs a q cdist function"
    return _mask_scores(q_novelty(tree, ctx.q_cdist), frontier)


def novelty_z_score(tree: BatchedTree, frontier: torch.Tensor, ctx: ScoringContext) -> torch.Tensor:
    """Direct ``min_j ||z_n - z_j||`` -- the state-space control for q-novelty."""
    from treewm.tree.novelty import z_novelty

    return _mask_scores(z_novelty(tree), frontier)


def novelty_q_penalized_score(tree: BatchedTree, frontier: torch.Tensor, ctx: ScoringContext) -> torch.Tensor:
    """``S = min_j d_q(q_n, q_j) - lambda * depth``.

    Trades novelty against the depth at which the world model stops being reliable,
    without needing a learned uncertainty model.
    """
    from treewm.tree.novelty import q_novelty

    assert ctx.q_cdist is not None, "penalised q-novelty needs a q cdist function"
    score = q_novelty(tree, ctx.q_cdist) - ctx.depth_penalty * tree.depth.float()
    return _mask_scores(score, frontier)


def _goal_distance(tree: BatchedTree, ctx: ScoringContext) -> torch.Tensor:
    """Decoded-position distance from every node to the goal. ``[B, N]``."""
    assert ctx.decoder is not None and ctx.goal_obs is not None, (
        "goal-directed scorers need a decoder and a normalised goal observation"
    )
    node_obs = ctx.decoder(tree.latent)  # [B, N, obs_dim]
    return torch.linalg.vector_norm(node_obs - ctx.goal_obs.unsqueeze(1), dim=-1)


def goal_score(tree: BatchedTree, frontier: torch.Tensor, ctx: ScoringContext) -> torch.Tensor:
    """Pure goal-directed best-first: expand whatever is closest to the goal."""
    return _mask_scores(-_goal_distance(tree, ctx), frontier)


def goal_novelty_score(tree: BatchedTree, frontier: torch.Tensor, ctx: ScoringContext) -> torch.Tensor:
    """``S = -d_goal(n) + alpha * min_j d_q(q_n, q_j)``.

    q-novelty as a *diversity bonus inside* goal-directed search rather than as the sole
    objective -- the question is whether it stops the search collapsing into one greedy
    corridor without dragging it into the far corners of the maze.
    """
    from treewm.tree.novelty import q_novelty

    assert ctx.q_cdist is not None, "goal+novelty scorer needs a q cdist function"
    score = -_goal_distance(tree, ctx) + ctx.alpha * q_novelty(tree, ctx.q_cdist)
    return _mask_scores(score, frontier)


def depth_balanced_random_score(tree: BatchedTree, frontier: torch.Tensor, ctx: ScoringContext) -> torch.Tensor:
    """D2 -- random, but with the budget explicitly spread across depth strata.

    Tests whether Random wins simply because of the depth distribution it happens to
    produce. Frontier nodes are split into shallow/medium/deep pools by their depth
    relative to the deepest frontier node; a random score is offset per pool so pools are
    visited round-robin rather than in proportion to their size (deep pools are always
    larger, which is what biases plain Random).
    """
    noise = torch.rand(tree.valid.shape, device=tree.valid.device, generator=ctx.generator,
                       dtype=torch.float32)
    depth = tree.depth.float()
    max_d = depth.masked_fill(~frontier, 0).max(dim=1, keepdim=True).values.clamp_min(1.0)
    pool = (depth / max_d * ctx.depth_pools).floor().clamp(0, ctx.depth_pools - 1)
    # Rotate which pool is favoured each iteration so all strata get expanded.
    favoured = ctx.step % ctx.depth_pools
    bonus = (pool == favoured).float() * 10.0
    return _mask_scores(noise + bonus, frontier)


def root_quota_score(tree: BatchedTree, frontier: torch.Tensor, ctx: ScoringContext) -> torch.Tensor:
    """D3 -- guarantee every root subtree recursive budget before any can dominate.

    Score = -(nodes already in this node's root subtree) + noise, so the least-developed
    subtree is always expanded next. Once ``budget_fraction`` passes ``broad_fraction``
    the quota is released and selection falls back to plain random.
    """
    noise = torch.rand(tree.valid.shape, device=tree.valid.device, generator=ctx.generator,
                       dtype=torch.float32)
    if ctx.budget_fraction >= ctx.broad_fraction:
        return _mask_scores(noise, frontier)

    b, n = tree.valid.shape
    rb = tree.root_branch  # [B, N], -1 at root
    # size of each node's own root subtree
    same = (rb.unsqueeze(2) == rb.unsqueeze(1)) & tree.valid.unsqueeze(1) & (rb.unsqueeze(2) >= 0)
    subtree_size = same.float().sum(-1)  # [B, N]
    return _mask_scores(-subtree_size + noise, frontier)


def diverse_goal_score(tree: BatchedTree, frontier: torch.Tensor, ctx: ScoringContext) -> torch.Tensor:
    """D5 -- goal-directed with an explicit root-subtree diversity term.

    One simple normalised formulation (no alpha sweep): the goal distance is normalised by
    the frontier's own spread, and a node is penalised in proportion to how much of the
    tree its root subtree already occupies.
    """
    d = _goal_distance(tree, ctx)
    masked = d.masked_fill(~frontier, float("nan"))
    lo = torch.nan_to_num(masked, nan=float("inf")).min(1, keepdim=True).values
    hi = torch.nan_to_num(masked, nan=float("-inf")).max(1, keepdim=True).values
    norm_d = (d - lo) / (hi - lo).clamp_min(1e-6)

    rb = tree.root_branch
    same = (rb.unsqueeze(2) == rb.unsqueeze(1)) & tree.valid.unsqueeze(1) & (rb.unsqueeze(2) >= 0)
    share = same.float().sum(-1) / tree.valid.float().sum(1, keepdim=True).clamp_min(1.0)
    return _mask_scores(-(norm_d + share), frontier)


def broad_to_focused_score(tree: BatchedTree, frontier: torch.Tensor, ctx: ScoringContext) -> torch.Tensor:
    """D6 -- balanced/broad expansion first, decoded-goal-directed once the tree exists."""
    if ctx.budget_fraction < ctx.broad_fraction:
        return root_quota_score(tree, frontier, ctx)
    return goal_score(tree, frontier, ctx)


def learned_score(tree: BatchedTree, frontier: torch.Tensor, ctx: ScoringContext) -> torch.Tensor:
    """Set-aware ``g_psi(feat_n, {feat_j}, depth, kappa, sigma)`` gain.

    Feature space follows ``ctx.novelty_space`` so the learned arm consumes exactly the
    representation whose novelty it is trained to predict.  The full valid tree set is
    supplied to query-conditioned cross-attention; the exact novelty value is not an
    input.
    """
    from treewm.tree.novelty import node_features

    assert ctx.gain_head is not None, "learned scorer needs an ExpansionGainHead"
    features = node_features(tree, ctx.novelty_space)
    if getattr(ctx.gain_head, "set_aware_enabled", False):
        gain = ctx.gain_head(
            features,
            features,
            tree.depth,
            tree.keep_score.float(),
            tree.uncertainty.float(),
            context_valid=tree.valid,
            exclude_self=True,
        )
    else:
        # Exact v1 compatibility: retain the configured mean/max pooled token.
        gain = ctx.gain_head(
            features,
            ctx.context_flat.float() if ctx.context_flat is not None else None,
            tree.depth,
            tree.keep_score.float(),
            tree.uncertainty.float(),
        )
    return _mask_scores(gain, frontier)


SCORERS: dict[str, Callable[[BatchedTree, torch.Tensor, ScoringContext], torch.Tensor]] = {
    "bfs": bfs_score,
    "random": random_score,
    "uncertainty": uncertainty_score,
    "heuristic": heuristic_score,  # mean-pooled context distance (the original arm)
    "novelty_q": novelty_q_score,
    "novelty_z": novelty_z_score,
    "novelty_q_penalized": novelty_q_penalized_score,
    "goal": goal_score,
    "goal_novelty": goal_novelty_score,
    "depth_balanced": depth_balanced_random_score,
    "root_quota": root_quota_score,
    "diverse_goal": diverse_goal_score,
    "broad_to_focused": broad_to_focused_score,
    "learned": learned_score,
}


# Scorers whose ranking depends on the goal. Tree generation must be handed a goal
# observation for these, and only these.
GOAL_AWARE_SCORERS = ("goal", "goal_novelty", "diverse_goal", "broad_to_focused")


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
