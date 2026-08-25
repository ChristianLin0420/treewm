"""Batched global best-first expansion under a hard node budget.

Single-node best-first wastes a GPU, so ``expansion_batch_size`` frontier nodes are
expanded per iteration (spec section 9). The loop is shared by *every* tree arm; the arm
selects a frontier scorer and nothing else, which is what makes "learned allocation" a
one-variable comparison.

Budget semantics are strict: a tree never exceeds ``node_budget`` nodes, and when the
final iteration's children overflow, the least-supported ones are dropped rather than
whichever happened to be last in memory order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

import torch

from treewm.tree.frontier import ScoringContext, get_scorer, select_topk
from treewm.tree.node import BatchedTree


@dataclass
class TreeConfig:
    node_budget: int = 64
    expansion_batch_size: int = 4
    max_depth: int = 16
    branch_factor: int = 4
    context_pooling: str = "mean"  # none | mean | max
    scorer: str = "learned"
    depth_penalty: float = 0.0
    alpha: float = 0.0  # diversity weight for the goal_novelty scorer
    broad_fraction: float = 0.45  # D6 broad -> focused switch point
    depth_pools: int = 3  # D2 depth strata
    # Explicit user override; when set it wins over the arm's default scorer.
    scorer_override: str | None = None  # lambda for the novelty_q_penalized scorer
    # ``None`` preserves the historical admit-all behaviour.  V2 inference pins 0.5:
    # children below threshold are pruned, with a per-parent top-1 fallback so every
    # valid expansion can still make progress.
    keep_threshold: float | None = None

    def __post_init__(self) -> None:
        assert self.node_budget >= 1
        assert self.expansion_batch_size >= 1
        assert self.branch_factor >= 1
        assert self.context_pooling in {"none", "mean", "max"}
        if self.keep_threshold is not None and not 0.0 <= self.keep_threshold <= 1.0:
            raise ValueError("keep_threshold must be in [0, 1] or None")


class BranchGenerator(Protocol):
    """What the expansion loop needs from a model: expand latents into K children."""

    def expand_nodes(self, z: torch.Tensor, depth: torch.Tensor) -> dict[str, torch.Tensor]:
        """``z``: ``[M, z_dim]``, ``depth``: ``[M]`` -> dict of ``[M, K, ...]``."""
        ...


@dataclass
class ExpansionTrace:
    """Per-iteration record, used for expansion diagnostics and gain supervision."""

    frontier_sizes: list[int] = field(default_factory=list)
    selected_scores: list[torch.Tensor] = field(default_factory=list)
    num_iterations: int = 0
    budget_reached: bool = False
    # Mean novelty of the frontier immediately before and after each expansion batch.
    # A healthy allocator should keep finding novel frontier nodes; a collapsing one
    # drives frontier novelty toward zero and then spends budget on redundant nodes.
    frontier_novelty_before: list[float] = field(default_factory=list)
    frontier_novelty_after: list[float] = field(default_factory=list)
    # Best decoded goal distance in the tree after each expansion batch; its decrease
    # per batch is 'goal progress per expansion'.
    best_goal_distance: list[float] = field(default_factory=list)
    # Detached per-iteration state for training the gain head against the *partial*
    # tree it will actually face at inference, rather than the finished tree.
    snapshots: list[dict[str, torch.Tensor]] = field(default_factory=list)


@torch.no_grad()
def _noop_grad_context():  # pragma: no cover - trivial
    yield


def generate_tree(
    model: BranchGenerator,
    root_z: torch.Tensor,
    root_q: torch.Tensor,
    cfg: TreeConfig,
    q_distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    gain_head: torch.nn.Module | None = None,
    generator: torch.Generator | None = None,
    on_iteration: Callable[[dict[str, Any]], None] | None = None,
    h_max: int = 64,
    action_dim: int = 2,
    q_cdist: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    novelty_space: str = "q",
    collect_snapshots: bool = False,
    track_novelty: bool = False,
    goal_obs: torch.Tensor | None = None,
    decoder: torch.nn.Module | None = None,
) -> tuple[BatchedTree, ExpansionTrace]:
    """Grow a batch of trees to exactly ``cfg.node_budget`` nodes (frontier permitting).

    Args:
        model: provides ``expand_nodes``.
        root_z: ``[B, z_dim]``
        root_q: ``[B, S, q_dim]``
        on_iteration: optional hook receiving the frontier, scores and selection each
            iteration. The trainer uses it to attach the expansion-gain loss without
            duplicating this loop.
    """
    tree = BatchedTree.initialize(root_z, root_q, cfg.node_budget, h_max, action_dim)
    scorer = get_scorer(cfg.scorer)
    trace = ExpansionTrace()

    # Each iteration adds at most expansion_batch_size * K nodes; this bound is a
    # safety net, not the termination condition.
    max_iters = cfg.node_budget

    for step in range(1, max_iters + 1):
        if int(tree.num_nodes.min()) >= cfg.node_budget:
            trace.budget_reached = True
            break

        frontier = tree.expandable_frontier(cfg.max_depth)
        if not bool(frontier.any()):
            break

        use_ctx = cfg.context_pooling != "none"
        ctx = ScoringContext(
            context=tree.context(cfg.context_pooling) if use_ctx else None,
            context_flat=tree.context_features(novelty_space, cfg.context_pooling) if use_ctx else None,
            gain_head=gain_head,
            q_distance=q_distance,
            q_cdist=q_cdist,
            generator=generator,
            step=step,
            novelty_space=novelty_space,
            depth_penalty=cfg.depth_penalty,
            goal_obs=goal_obs,
            decoder=decoder,
            alpha=cfg.alpha,
            budget_fraction=float(tree.num_nodes.float().mean().item()) / max(cfg.node_budget, 1),
            broad_fraction=cfg.broad_fraction,
            depth_pools=cfg.depth_pools,
        )
        scores = scorer(tree, frontier, ctx)

        if track_novelty or collect_snapshots:
            from treewm.tree.novelty import node_features, novelty_of

            target = novelty_of(tree, novelty_space, q_cdist)
            if track_novelty:
                fmask = frontier.float()
                trace.frontier_novelty_before.append(
                    float((target * fmask).sum() / fmask.sum().clamp_min(1.0))
                )
            if collect_snapshots:
                trace.snapshots.append(
                    {
                        "feats": node_features(tree, novelty_space).detach(),
                        "context": ctx.context_flat.detach() if ctx.context_flat is not None else None,
                        "depth": tree.depth.clone(),
                        "keep": tree.keep_score.float().detach(),
                        "sigma": tree.uncertainty.float().detach(),
                        "valid": tree.valid.clone(),
                        "frontier": frontier.clone(),
                        "target": target.detach(),
                    }
                )

        remaining = int((cfg.node_budget - tree.num_nodes).clamp_min(0).max())
        take = max(1, min(cfg.expansion_batch_size, remaining))
        sel_idx, sel_valid = select_topk(scores, take)

        # Record the score that actually drove each expansion decision.
        chosen_scores = torch.gather(scores, 1, sel_idx).to(tree.expansion_gain.dtype)
        tree.expansion_gain.scatter_(
            1, sel_idx, torch.where(sel_valid, chosen_scores, torch.gather(tree.expansion_gain, 1, sel_idx))
        )

        b, e = sel_idx.shape
        z_sel = torch.gather(
            tree.latent, 1, sel_idx.unsqueeze(-1).expand(b, e, tree.latent.shape[-1])
        )
        d_sel = torch.gather(tree.depth, 1, sel_idx)

        child = model.expand_nodes(z_sel.reshape(b * e, -1), d_sel.reshape(b * e))
        child = {k: v.view(b, e, *v.shape[1:]) for k, v in child.items()}

        k = child["latent"].shape[2]
        child_valid = sel_valid.unsqueeze(-1).expand(b, e, k).float()

        if on_iteration is not None:
            on_iteration(
                {
                    "tree": tree,
                    "frontier": frontier,
                    "scores": scores,
                    "selected": sel_idx,
                    "selected_valid": sel_valid,
                    "context": ctx.context,
                    "step": step,
                }
            )

        tree.add_children(
            sel_idx,
            child,
            cfg.node_budget,
            step,
            child_valid=child_valid,
            parent_valid=sel_valid,
            keep_threshold=cfg.keep_threshold,
        )

        if track_novelty:
            from treewm.tree.novelty import novelty_of

            after = novelty_of(tree, novelty_space, q_cdist)
            fmask_after = tree.expandable_frontier(cfg.max_depth).float()
            trace.frontier_novelty_after.append(
                float((after * fmask_after).sum() / fmask_after.sum().clamp_min(1.0))
            )

        if goal_obs is not None and decoder is not None:
            with torch.no_grad():
                d_goal = torch.linalg.vector_norm(
                    decoder(tree.latent) - goal_obs.unsqueeze(1), dim=-1
                ).masked_fill(~tree.valid, float("inf"))
                trace.best_goal_distance.append(float(d_goal.min(1).values.mean().item()))

        trace.frontier_sizes.append(int(frontier.sum(1).float().mean().item()))
        trace.selected_scores.append(chosen_scores.detach())
        trace.num_iterations = step

    trace.budget_reached = bool((tree.num_nodes >= cfg.node_budget).all())
    assert int(tree.num_nodes.max()) <= cfg.node_budget, "node budget violated"
    return tree, trace


def expansion_metrics(tree: BatchedTree, trace: ExpansionTrace, cfg: TreeConfig) -> dict[str, float]:
    """Scalars for the ``expansion/*`` namespace."""
    valid = tree.valid
    depth = tree.depth.float()
    counts = valid.float().sum(1)

    mean_depth = (depth * valid.float()).sum(1) / counts.clamp_min(1.0)
    expanded_depth = (depth * tree.expanded.float()).sum(1) / tree.expanded.float().sum(1).clamp_min(1.0)

    return {
        "expansion/nodes_generated": float(counts.mean().item()),
        "expansion/max_depth": float(depth.masked_fill(~valid, 0).max(1).values.mean().item()),
        "expansion/mean_depth": float(mean_depth.mean().item()),
        "expansion/expanded_depth": float(expanded_depth.mean().item()),
        "expansion/frontier_size": float(
            sum(trace.frontier_sizes) / max(len(trace.frontier_sizes), 1)
        ),
        "expansion/iterations": float(trace.num_iterations),
        "expansion/budget_shortfall": float(
            (cfg.node_budget - counts).clamp_min(0).mean().item()
        ),
        "expansion/predicted_gain_mean": float(
            torch.cat(trace.selected_scores, dim=1).mean().item() if trace.selected_scores else 0.0
        ),
    }
