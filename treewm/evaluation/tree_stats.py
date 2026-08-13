"""Structural tree summaries (spec section 15).

These describe *how the tree is shaped*, not whether the final success number moved.
Most of the night's questions -- does K=8 make new routes or duplicates, does a goal beam
eat one subtree, does Random keep a healthier depth profile -- are answered here rather
than by success alone.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def structural_summary(tree, model=None, normalizer=None, prefix: str = "tree") -> dict[str, float]:
    """Per-depth and per-subtree structure for one batch of trees."""
    valid = tree.valid
    b, n = valid.shape
    depth = tree.depth
    counts = valid.float().sum(1).clamp_min(1.0)
    out: dict[str, float] = {}

    max_depth = int(depth.masked_fill(~valid, 0).max().item())
    out[f"{prefix}/max_depth"] = float(max_depth)
    out[f"{prefix}/mean_depth"] = float(((depth.float() * valid.float()).sum(1) / counts).mean())

    # --- effective branching factor and horizon, per depth -------------------------
    for d in range(0, min(max_depth, 6) + 1):
        at_d = valid & (depth == d)
        if not bool(at_d.any()):
            continue
        n_at_d = at_d.float().sum(1)
        n_children = (valid & (depth == d + 1)).float().sum(1)
        out[f"{prefix}/effective_branching_factor_d{d}"] = float(
            (n_children / n_at_d.clamp_min(1.0)).mean()
        )
        horiz = tree.action_mask.sum(-1)
        out[f"{prefix}/horizon_d{d}"] = float(
            (horiz * at_d.float()).sum() / at_d.float().sum().clamp_min(1.0)
        )
        out[f"{prefix}/count_d{d}"] = float(n_at_d.mean())

    # --- root-subtree utilisation ---------------------------------------------------
    rb = tree.root_branch
    uniq, top2 = [], []
    for i in range(b):
        ids = rb[i][valid[i]]
        ids = ids[ids >= 0]
        if ids.numel() == 0:
            continue
        vals, cnt = torch.unique(ids, return_counts=True)
        uniq.append(float(vals.numel()))
        share = (cnt.float() / cnt.sum()).sort(descending=True).values
        top2.append(float(share[: min(2, share.numel())].sum()))
    if uniq:
        out[f"{prefix}/unique_root_subtrees_explored"] = float(np.mean(uniq))
        # Near 1.0 means the budget collapsed into one or two subtrees.
        out[f"{prefix}/top2_root_subtree_fraction"] = float(np.mean(top2))

    # --- diversity by depth ----------------------------------------------------------
    if model is not None and getattr(model, "decoder", None) is not None:
        pos = model.decoder(tree.latent).float()
        for d in range(1, min(max_depth, 5) + 1):
            at_d = valid & (depth == d)
            if int(at_d.sum()) < 2:
                continue
            per_tree = [
                float(torch.cdist(pos[i][at_d[i]], pos[i][at_d[i]]).mean())
                for i in range(b) if int(at_d[i].sum()) >= 2
            ]
            if per_tree:
                out[f"{prefix}/pairwise_endpoint_diversity_d{d}"] = float(np.mean(per_tree))

    act = tree.action_chunk.flatten(2).float()
    for d in range(1, min(max_depth, 5) + 1):
        at_d = valid & (depth == d)
        if int(at_d.sum()) < 2:
            continue
        per_tree = [
            float(torch.cdist(act[i][at_d[i]], act[i][at_d[i]]).mean())
            for i in range(b) if int(at_d[i].sum()) >= 2
        ]
        if per_tree:
            out[f"{prefix}/pairwise_action_diversity_d{d}"] = float(np.mean(per_tree))

    # --- expansion order vs depth ----------------------------------------------------
    ok = valid & (tree.order >= 0)
    if bool(ok.any()):
        o = tree.order.float()[ok]
        dd = depth.float()[ok]
        if o.numel() > 1 and float(o.std()) > 0 and float(dd.std()) > 0:
            # ~1.0 means strictly deepening over time; ~0 means depth-agnostic ordering.
            out[f"{prefix}/expansion_order_vs_depth"] = float(
                ((o - o.mean()) * (dd - dd.mean())).mean() / (o.std() * dd.std())
            )
    return out


@torch.no_grad()
def depth_histogram(tree) -> np.ndarray:
    d = tree.depth[tree.valid].float().cpu().numpy()
    return d if d.size else np.zeros(1, dtype=np.float32)
