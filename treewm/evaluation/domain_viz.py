"""Tree diagnostics for non-spatial domains.

The PointMaze renders plot decoded node positions on the maze floor. Cube, scene and
puzzle have no such floor, and projecting their observations onto the first two dims
would draw a picture of the robot's joint angles -- visually plausible and completely
uninformative about whether the branches are meaningful alternatives.

So each family gets a render of the quantity its task actually constrains:

    cube / scene   predicted object positions at every tree node, per-object target error
    puzzle         predicted board configuration per recursive depth
    locomotion     handled by the existing xy renders in treewm/evaluation/tree_viz.py

The question these are meant to answer is the one the whole wave exists for: do the K
branches at a node represent *different executable futures*, or has the model collapsed
onto one continuation with cosmetic variation?
"""

from __future__ import annotations

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@torch.no_grad()
def decode_nodes(model, tree, normalizer) -> np.ndarray:
    """[N, obs_dim] decoded observation at each tree node, in raw (unnormalised) units."""
    obs_n = model.decoder(tree.latent)[0].float().cpu().numpy()
    return normalizer.denorm_obs(obs_n) if hasattr(normalizer, "denorm_obs") else obs_n


def _valid_nodes(tree) -> np.ndarray:
    return tree.valid[0].cpu().numpy()


def view_object_tree(model, tree, normalizer, domain, goal_obs: np.ndarray,
                     title: str = "", selected: int | None = None):
    """Predicted object positions across the tree, one panel per tracked object.

    A tree that has learned genuine alternatives shows branches fanning to *different*
    object placements; a collapsed tree shows one cloud.
    """
    nodes = decode_nodes(model, tree, normalizer)
    valid = _valid_nodes(tree)
    parent = tree.parent_index[0].cpu().numpy()
    depth = tree.depth[0].cpu().numpy()
    gv_all = np.asarray(goal_obs, dtype=np.float32)

    subs = domain.subgoals or ((0, len(domain.goal_dims)),)
    n_obj = len(subs)
    fig, axes = plt.subplots(1, n_obj, figsize=(4.2 * n_obj, 4.0), squeeze=False)
    dims = np.asarray(domain.goal_dims)

    for k, (lo, hi) in enumerate(subs):
        ax = axes[0][k]
        sel_dims = dims[lo:hi]
        pts = nodes[:, sel_dims]
        goal_pt = gv_all[sel_dims]
        # Project to the first two constrained dims; for a cube these are x and y.
        if pts.shape[1] == 1:
            xs, ys = pts[:, 0], np.zeros_like(pts[:, 0])
            gx, gy = goal_pt[0], 0.0
        else:
            xs, ys = pts[:, 0], pts[:, 1]
            gx, gy = goal_pt[0], goal_pt[1]

        for n in range(len(xs)):
            if not valid[n] or parent[n] < 0:
                continue
            p = int(parent[n])
            ax.plot([xs[p], xs[n]], [ys[p], ys[n]], color="0.6", lw=0.7, zorder=1)
        sc = ax.scatter(xs[valid], ys[valid], c=depth[valid], cmap="viridis", s=26, zorder=3)
        ax.scatter([xs[0]], [ys[0]], marker="*", s=180, color="crimson", zorder=5, label="root")
        ax.scatter([gx], [gy], marker="X", s=180, color="green", zorder=5, label="goal")
        if selected is not None and valid[selected]:
            chain, cur = [], int(selected)
            while cur >= 0:
                chain.append(cur)
                cur = int(parent[cur])
            chain = chain[::-1]
            ax.plot(xs[chain], ys[chain], color="orange", lw=2.4, zorder=4)
        err = float(np.linalg.norm(pts[valid] - goal_pt, axis=1).min()) if valid.any() else float("nan")
        ax.set_title(f"object {k}  best node error {err:.3f}", fontsize=9)
        ax.grid(alpha=.3)
        if k == 0:
            ax.legend(fontsize=7, loc="best")
        fig.colorbar(sc, ax=ax, label="depth")
    fig.suptitle(title or "predicted object configuration at tree nodes", fontsize=11)
    fig.tight_layout()
    return fig


def _puzzle_grid_shape(domain, grid: tuple[int, int] | None) -> tuple[int, int]:
    """Resolve and validate a board shape from the number of puzzle cells."""
    cells = len(domain.subgoals)
    if cells <= 0:
        raise ValueError("puzzle visualisation requires at least one subgoal cell")
    if grid is None:
        side = int(np.sqrt(cells))
        if side * side != cells:
            raise ValueError(f"cannot infer a square puzzle grid from {cells} cells")
        return side, side
    if len(grid) != 2 or any(int(value) <= 0 for value in grid):
        raise ValueError("puzzle grid must contain two positive dimensions")
    shape = int(grid[0]), int(grid[1])
    if shape[0] * shape[1] != cells:
        raise ValueError(f"puzzle grid {shape} does not match {cells} cells")
    return shape


def view_board_by_depth(model, tree, normalizer, domain, goal_obs: np.ndarray,
                        grid: tuple[int, int] | None = None, title: str = ""):
    """Predicted puzzle board per recursive depth, beside the target board.

    Each cell is the argmax of that button's one-hot block, so this shows whether deeper
    recursion predicts progressively different *configurations* rather than re-predicting
    the current board.
    """
    nodes = decode_nodes(model, tree, normalizer)
    valid = _valid_nodes(tree)
    depth = tree.depth[0].cpu().numpy()
    dims = np.asarray(domain.goal_dims)
    gv = np.asarray(goal_obs, dtype=np.float32)
    grid = _puzzle_grid_shape(domain, grid)

    def board(vec: np.ndarray) -> np.ndarray:
        cells = [int(np.argmax(vec[dims[lo:hi]])) for lo, hi in domain.subgoals]
        return np.asarray(cells, dtype=float).reshape(grid)

    depths = sorted({int(d) for d, v in zip(depth, valid) if v})
    fig, axes = plt.subplots(1, len(depths) + 1, figsize=(2.5 * (len(depths) + 1), 2.9),
                             squeeze=False)
    for i, d in enumerate(depths):
        idx = np.flatnonzero(valid & (depth == d))
        # modal predicted board at this depth
        boards = np.stack([board(nodes[n]) for n in idx])
        modal = (boards.mean(0) > 0.5).astype(float)
        ax = axes[0][i]
        ax.imshow(modal, cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_title(f"depth {d}  (n={len(idx)})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
        for (r, c), v in np.ndenumerate(modal):
            ax.text(c, r, int(v), ha="center", va="center", fontsize=9)
    ax = axes[0][-1]
    gb = board(gv)
    ax.imshow(gb, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_title("target", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    for (r, c), v in np.ndenumerate(gb):
        ax.text(c, r, int(v), ha="center", va="center", fontsize=9)
    fig.suptitle(title or "predicted board configuration by recursive depth", fontsize=11)
    fig.tight_layout()
    return fig


def branch_divergence(model, tree, normalizer, domain) -> dict[str, float]:
    """Scalar test of whether siblings are genuinely different futures.

    ``sibling_spread`` is the mean pairwise distance between sibling nodes in goal-dim
    space, normalised by the spread of the whole tree. Near zero means the K branches
    collapsed onto one continuation -- which would make 'recursive multimodal prediction'
    an empty label regardless of what the success numbers say.
    """
    nodes = decode_nodes(model, tree, normalizer)
    valid = _valid_nodes(tree)
    parent = tree.parent_index[0].cpu().numpy()
    dims = np.asarray(domain.goal_dims)
    gv = nodes[:, dims]

    spreads, groups = [], {}
    for n in range(len(gv)):
        if valid[n] and parent[n] >= 0:
            groups.setdefault(int(parent[n]), []).append(n)
    for _, sibs in groups.items():
        if len(sibs) < 2:
            continue
        pts = gv[sibs]
        d = np.linalg.norm(pts[:, None] - pts[None, :], axis=-1)
        spreads.append(d[np.triu_indices(len(sibs), 1)].mean())
    tree_spread = float(np.linalg.norm(gv[valid] - gv[valid].mean(0), axis=1).mean()) if valid.any() else 0.0
    sib = float(np.mean(spreads)) if spreads else 0.0
    return {
        "tree/sibling_spread": sib,
        "tree/global_spread": tree_spread,
        "tree/sibling_spread_ratio": float(sib / tree_spread) if tree_spread > 1e-9 else 0.0,
        "tree/num_branch_points": float(len(spreads)),
    }
