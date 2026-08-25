"""Novelty-target supervision for the expansion-gain head.

Replaces the retrieval-based marginal-coverage target with the signal the winning
heuristic actually acts on:

    G*(n | T) = min_{j in T} d(q_n, q_j)      (or d(z_n, z_j))

Two properties matter for isolating the previous failure:

1. **Targets come from the partial tree.** The head is supervised at each expansion
   iteration against the tree as it exists at that moment -- the same state it faces at
   inference. Regressing against the *finished* tree would train it on a distribution it
   never sees, which is a plausible contributor to the earlier anti-scaling.

2. **Only the head learns.** Snapshot features are detached, so the encoder, branch
   transformer and dynamics receive no gradient from this loss. Every arm therefore
   shares identical world-model training and differs *only* in how the frontier is
   ranked, which is the one variable under test.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from treewm.logging.metrics import pearson_correlation, rank_correlation


def frontier_gain_objective(
    predicted: torch.Tensor,
    target: torch.Tensor,
    frontier: torch.Tensor,
    *,
    rank_weight: float = 1.0,
    calibration_weight: float = 0.1,
    tie_tolerance: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Pairwise ranking plus calibration, averaged equally per decision.

    Every batch row is one frontier decision regardless of its frontier size.  Ranking
    depends only on target ordering and is therefore invariant to positive rescaling of
    novelty.  A small Huber term anchors score calibration without letting large
    frontiers or target units dominate.
    """
    if predicted.shape != target.shape or predicted.shape != frontier.shape:
        raise ValueError("predicted, target and frontier must have identical [B, N] shape")
    if rank_weight < 0 or calibration_weight < 0:
        raise ValueError("gain rank/calibration weights must be non-negative")

    decision_losses: list[torch.Tensor] = []
    rank_losses: list[torch.Tensor] = []
    calibration_losses: list[torch.Tensor] = []
    calibration_mae: list[torch.Tensor] = []
    rank_scores: list[float] = []
    pearson_scores: list[float] = []
    top1_scores: list[torch.Tensor] = []
    regrets: list[torch.Tensor] = []
    pair_accuracy: list[torch.Tensor] = []
    pred_spreads: list[torch.Tensor] = []
    target_spreads: list[torch.Tensor] = []
    pred_stds: list[torch.Tensor] = []
    target_stds: list[torch.Tensor] = []
    pred_means: list[torch.Tensor] = []
    target_means: list[torch.Tensor] = []
    frontier_sizes: list[float] = []

    for row in range(predicted.shape[0]):
        valid = frontier[row] > 0
        if not bool(valid.any()):
            continue
        pred = predicted[row, valid].float()
        tgt = target[row, valid].float().detach()
        frontier_sizes.append(float(pred.numel()))

        calibration = F.smooth_l1_loss(pred, tgt)
        calibration_losses.append(calibration)
        calibration_mae.append((pred - tgt).abs().mean())

        if pred.numel() > 1:
            target_delta = tgt.unsqueeze(1) - tgt.unsqueeze(0)
            pred_delta = pred.unsqueeze(1) - pred.unsqueeze(0)
            ordered = target_delta > tie_tolerance
            if bool(ordered.any()):
                rank = F.softplus(-pred_delta[ordered]).mean()
                accuracy = (pred_delta[ordered] > 0).float().mean()
            else:
                rank = pred.sum() * 0.0
                accuracy = pred.sum() * 0.0
        else:
            rank = pred.sum() * 0.0
            accuracy = pred.sum() * 0.0
        rank_losses.append(rank)
        pair_accuracy.append(accuracy)
        decision_losses.append(rank_weight * rank + calibration_weight * calibration)

        with torch.no_grad():
            choice = pred.argmax()
            best = tgt.max()
            worst = tgt.min()
            top1_scores.append((tgt[choice] >= best - tie_tolerance).float())
            regrets.append((best - tgt[choice]) / (best - worst).clamp_min(tie_tolerance))
            pred_spreads.append(pred.max() - pred.min())
            target_spreads.append(best - worst)
            pred_stds.append(pred.std(unbiased=False))
            target_stds.append(tgt.std(unbiased=False))
            pred_means.append(pred.mean())
            target_means.append(tgt.mean())
            rank_scores.append(rank_correlation(pred, tgt) if pred.numel() > 1 else 0.0)
            pearson_scores.append(pearson_correlation(pred, tgt) if pred.numel() > 1 else 0.0)

    if not decision_losses:
        zero = predicted.sum() * 0.0
        return zero, {
            "decision_count": 0.0,
            "loss_rank": 0.0,
            "loss_calibration": 0.0,
        }

    def mean_tensor(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).mean()

    loss = mean_tensor(decision_losses)
    metrics = {
        "decision_count": float(len(decision_losses)),
        "mean_frontier_size": float(sum(frontier_sizes) / len(frontier_sizes)),
        "loss_rank": float(mean_tensor(rank_losses).detach().item()),
        "loss_calibration": float(mean_tensor(calibration_losses).detach().item()),
        "calibration_mae": float(mean_tensor(calibration_mae).detach().item()),
        "rank_correlation": float(sum(rank_scores) / len(rank_scores)),
        "pearson_correlation": float(sum(pearson_scores) / len(pearson_scores)),
        "top1_accuracy": float(mean_tensor(top1_scores).item()),
        "normalized_regret": float(mean_tensor(regrets).item()),
        "pairwise_accuracy": float(mean_tensor(pair_accuracy).item()),
        "predicted_spread": float(mean_tensor(pred_spreads).item()),
        "target_spread": float(mean_tensor(target_spreads).item()),
        "predicted_std": float(mean_tensor(pred_stds).item()),
        "target_std": float(mean_tensor(target_stds).item()),
        "predicted_mean": float(mean_tensor(pred_means).item()),
        "target_mean": float(mean_tensor(target_means).item()),
    }
    return loss, metrics


def _root_branch_prior_objective(
    model,
    z0: torch.Tensor,
    space: str,
    *,
    rank_weight: float,
    calibration_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train the context-free branch prior on root-frontier novelty only."""
    with torch.no_grad():
        root_children = model.predict_children(z0)
        branch_embedding = root_children["branch"].embedding.detach()
        if space == "q":
            root = model.q_of(z0).unsqueeze(1)
            nodes = torch.cat((root, root_children["q"]), dim=1)
            distance = model.q_cdist(nodes.float(), nodes.float())
        elif space == "z":
            nodes = torch.cat((z0.unsqueeze(1), root_children["latent"]), dim=1)
            distance = torch.cdist(nodes.float(), nodes.float())
        else:
            raise ValueError(f"unknown novelty space {space!r}; options: q | z")
        k = branch_embedding.shape[1]
        child_rows = distance[:, 1 : k + 1].clone()
        child_index = torch.arange(k, device=z0.device)
        child_rows[:, child_index, child_index + 1] = float("inf")
        prior_target = child_rows.min(-1).values.detach()

    prior_pred = model.heads.gain_head(branch_embedding).squeeze(-1)
    valid = torch.ones_like(prior_target, dtype=torch.bool)
    return frontier_gain_objective(
        prior_pred,
        prior_target,
        valid,
        rank_weight=rank_weight,
        calibration_weight=calibration_weight,
    )


def novelty_gain_loss(
    model,
    z0: torch.Tensor,
    tree_cfg,
    space: str = "q",
    generator: torch.Generator | None = None,
    rank_weight: float = 0.0,
    calibration_weight: float = 1.0,
    branch_prior_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train ``g_psi`` to predict min-novelty over the frontier.

    Returns ``(loss, metrics)``. Metrics carry both Pearson and Spearman correlation --
    the head can be well calibrated in value yet wrong in *ordering*, and best-first
    expansion only consumes the ordering.
    """
    with torch.no_grad():
        tree, trace = model.generate(
            z0, tree_cfg, generator=generator, collect_snapshots=True, track_novelty=True
        )

    if not trace.snapshots:
        zero = z0.sum() * 0.0
        return zero, {}

    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    contextual_numerator = z0.sum() * 0.0
    decision_count = 0
    decision_metric_sums: dict[str, float] = {}
    for snap in trace.snapshots:
        if getattr(model.gain_head, "set_aware_enabled", False):
            pred = model.gain_head(
                snap["feats"],
                snap["feats"],
                snap["depth"],
                snap["keep"],
                snap["sigma"],
                context_valid=snap["valid"],
                exclude_self=True,
            )
        else:
            pred = model.gain_head(
                snap["feats"], snap["context"], snap["depth"], snap["keep"], snap["sigma"]
            )
        sel = snap["frontier"]
        if not bool(sel.any()):
            continue
        preds.append(pred[sel])
        targets.append(snap["target"][sel])
        if rank_weight > 0:
            decision_loss, decision_metrics = frontier_gain_objective(
                pred,
                snap["target"],
                sel,
                rank_weight=rank_weight,
                calibration_weight=calibration_weight,
            )
            count = int(decision_metrics.get("decision_count", 0.0))
            if count:
                contextual_numerator = contextual_numerator + decision_loss * count
                decision_count += count
                for key, value in decision_metrics.items():
                    if key != "decision_count":
                        decision_metric_sums[key] = decision_metric_sums.get(key, 0.0) + value * count

    if not preds:
        zero = z0.sum() * 0.0
        return zero, {}

    pred_flat = torch.cat(preds)
    tgt_flat = torch.cat(targets)
    if rank_weight > 0:
        contextual_loss = contextual_numerator / max(decision_count, 1)
    else:
        # Exact v1 pointwise objective and node-weighted reduction.
        contextual_loss = calibration_weight * F.smooth_l1_loss(pred_flat, tgt_flat)

    prior_loss = contextual_loss.detach() * 0.0
    prior_metrics: dict[str, float] = {}
    if branch_prior_weight > 0:
        prior_loss, prior_metrics = _root_branch_prior_objective(
            model,
            z0,
            space,
            rank_weight=rank_weight,
            calibration_weight=calibration_weight,
        )
    loss = contextual_loss + branch_prior_weight * prior_loss

    with torch.no_grad():
        metrics = {
            "expansion/loss_novelty_gain": float(loss.item()),
            "expansion/loss_novelty_gain_contextual": float(contextual_loss.item()),
            "expansion/loss_branch_gain_prior": float(prior_loss.item()),
            "expansion/effective_branch_gain_prior": float((branch_prior_weight * prior_loss).item()),
            "expansion/predicted_gain_mean": float(pred_flat.mean().item()),
            "expansion/target_gain_mean": float(tgt_flat.mean().item()),
            "expansion/gain_mae": float((pred_flat - tgt_flat).abs().mean().item()),
            "expansion/gain_rank_correlation": rank_correlation(pred_flat, tgt_flat),
            "expansion/gain_pearson_correlation": pearson_correlation(pred_flat, tgt_flat),
            "expansion/target_gain_std": float(tgt_flat.std(unbiased=False).item()),
            "expansion/predicted_gain_std": float(pred_flat.std(unbiased=False).item()),
        }
        if rank_weight > 0 and decision_count:
            averaged = {
                key: value / decision_count for key, value in decision_metric_sums.items()
            }
            metrics.update(
                {
                    "expansion/gain_decision_count": float(decision_count),
                    "expansion/gain_mean_frontier_size": averaged["mean_frontier_size"],
                    "expansion/loss_novelty_gain_rank": averaged["loss_rank"],
                    "expansion/loss_novelty_gain_calibration": averaged["loss_calibration"],
                    "expansion/gain_calibration_mae": averaged["calibration_mae"],
                    "expansion/gain_rank_correlation": averaged["rank_correlation"],
                    "expansion/gain_pearson_correlation": averaged["pearson_correlation"],
                    "expansion/gain_top1_accuracy": averaged["top1_accuracy"],
                    "expansion/gain_normalized_regret": averaged["normalized_regret"],
                    "expansion/gain_pairwise_accuracy": averaged["pairwise_accuracy"],
                    "expansion/predicted_gain_spread": averaged["predicted_spread"],
                    "expansion/target_gain_spread": averaged["target_spread"],
                    "expansion/predicted_gain_std": averaged["predicted_std"],
                    "expansion/target_gain_std": averaged["target_std"],
                }
            )
        if prior_metrics:
            metrics.update(
                {
                    f"expansion/branch_prior_{key}": value
                    for key, value in prior_metrics.items()
                }
            )
        if trace.frontier_novelty_before:
            before = trace.frontier_novelty_before
            after = trace.frontier_novelty_after or before
            metrics["expansion/frontier_novelty_before"] = float(sum(before) / len(before))
            metrics["expansion/frontier_novelty_after"] = float(sum(after) / len(after))
            metrics["expansion/frontier_novelty_first"] = float(before[0])
            metrics["expansion/frontier_novelty_last"] = float(before[-1])
            # Falling frontier novelty means the allocator is running out of distinct
            # places to go -- the signature of the collapse seen with the old target.
            metrics["expansion/frontier_novelty_decay"] = float(before[-1] - before[0])
        metrics.update(tree_expansion_metrics(model, tree, space))
    return loss, metrics


@torch.no_grad()
def tree_expansion_metrics(model, tree, space: str = "q") -> dict[str, float]:
    """Depth, budget use and redundancy for a generated tree.

    Physical coverage is intentionally not inferred here.  The old implementation
    quantised decoded observation dimensions ``(0, 1)`` at a hard-coded resolution,
    which happened to resemble maze coordinates but was meaningless for manipulation
    and puzzle domains.  Task-aware endpoint fidelity is reported by ``total.py`` from
    the explicit ``task_metric_dims`` contract instead.
    """
    from treewm.tree.novelty import novelty_of, redundant_fraction

    valid = tree.valid
    nodes = valid.float().sum(1).clamp_min(1.0)
    depth = tree.depth.float()
    capacity = float(tree.capacity)

    out = {
        # These are acceptance-critical for the KEEP-gated v2 allocator.  A model
        # whose KEEP logits are all low still grows a depth-limited top-1 chain;
        # depth/entropy alone therefore cannot distinguish that collapse from a
        # tree that actually uses its advertised node budget.
        "expansion/nodes_generated": float(nodes.mean().item()),
        "expansion/budget_shortfall": float((capacity - nodes).clamp_min(0).mean().item()),
        "expansion/budget_fill_fraction": float((nodes / capacity).mean().item()),
        "expansion/mean_depth": float(((depth * valid.float()).sum(1) / nodes).mean().item()),
        "expansion/max_depth": float(depth.masked_fill(~valid, 0).max(1).values.mean().item()),
        "expansion/depth_std": float(
            (((depth - (depth * valid.float()).sum(1, keepdim=True) / nodes.unsqueeze(1)) ** 2
              * valid.float()).sum(1) / nodes).sqrt().mean().item()
        ),
        "expansion/redundant_expansion_fraction": redundant_fraction(
            tree, space, model.q_cdist if space == "q" else None
        ),
        "expansion/frontier_novelty_mean": float(
            (novelty_of(tree, space, model.q_cdist if space == "q" else None) * valid.float()).sum(1).div(nodes).mean().item()
        ),
    }

    return out
