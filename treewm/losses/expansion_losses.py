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


def novelty_gain_loss(
    model,
    z0: torch.Tensor,
    tree_cfg,
    space: str = "q",
    generator: torch.Generator | None = None,
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
    for snap in trace.snapshots:
        pred = model.gain_head(
            snap["feats"], snap["context"], snap["depth"], snap["keep"], snap["sigma"]
        )
        sel = snap["frontier"]
        if not bool(sel.any()):
            continue
        preds.append(pred[sel])
        targets.append(snap["target"][sel])

    if not preds:
        zero = z0.sum() * 0.0
        return zero, {}

    pred_flat = torch.cat(preds)
    tgt_flat = torch.cat(targets)
    loss = F.smooth_l1_loss(pred_flat, tgt_flat)

    with torch.no_grad():
        metrics = {
            "expansion/loss_novelty_gain": float(loss.item()),
            "expansion/predicted_gain_mean": float(pred_flat.mean().item()),
            "expansion/target_gain_mean": float(tgt_flat.mean().item()),
            "expansion/gain_mae": float((pred_flat - tgt_flat).abs().mean().item()),
            "expansion/gain_rank_correlation": rank_correlation(pred_flat, tgt_flat),
            "expansion/gain_pearson_correlation": pearson_correlation(pred_flat, tgt_flat),
            "expansion/target_gain_std": float(tgt_flat.std().item()),
            "expansion/predicted_gain_std": float(pred_flat.std().item()),
        }
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
    """Coverage, depth distribution and redundancy for a generated tree."""
    from treewm.evaluation.coverage import unique_cells_per_row
    from treewm.tree.novelty import novelty_of, redundant_fraction

    valid = tree.valid
    nodes = valid.float().sum(1).clamp_min(1.0)
    depth = tree.depth.float()

    out = {
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

    if model.decoder is not None:
        from treewm.evaluation.coverage import StateQuantizer

        quant = StateQuantizer(resolution=0.2, dims=(0, 1))
        cells = quant.cell_ids(model.decoder(tree.latent))
        covered = unique_cells_per_row(cells, valid.float()).float()
        expanded = tree.expanded.float().sum(1).clamp_min(1.0)
        out["expansion/controllability_coverage"] = float(covered.mean().item())
        out["expansion/controllability_coverage_per_node"] = float((covered / nodes).mean().item())
        out["expansion/coverage_per_expanded_node"] = float((covered / expanded).mean().item())
    return out
