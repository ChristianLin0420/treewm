"""Tree visualisation suite (spec sections 13-14).

Every view renders the *same* generated tree from a fixed anchor so runs are visually
comparable across the night. The anchors are fixed once per environment and reused by
every run, which is what makes "did K=8 create new routes or duplicates?" answerable by
flipping between two runs in TensorBoard.

Views:
    tree_xy_depth            nodes coloured by depth
    tree_xy_expansion_order  nodes coloured by when they were expanded
    tree_xy_goal_distance    nodes coloured by decoded goal distance
    tree_xy_root_subtree     nodes grouped by root branch
    tree_topology            pure graph layout, no physical coordinates
    tree_horizon             nodes coloured by predicted action-chunk horizon
    tree_selected_path       selected path highlighted over a faded tree
    predicted_vs_grounded    predicted endpoint vs simulator-executed endpoint
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass
class AnchorSet:
    """Fixed diagnostic scenarios, identical across every run."""

    starts: np.ndarray  # [A, obs_dim]
    goals: np.ndarray  # [A, obs_dim]
    names: list[str]

    def __len__(self) -> int:
        return len(self.starts)


def build_anchors(maze_spec, num: int = 8, percentile: float = 70.0, seed: int = 0) -> AnchorSet:
    """Deterministic start/goal pairs spread over the maze, sorted by geodesic distance."""
    from treewm.data.maze_utils import sample_hard_goal_pairs

    pairs = sample_hard_goal_pairs(maze_spec, num_pairs=num * 3, percentile=percentile, seed=seed)
    pairs = sorted(pairs, key=lambda d: d["geodesic"])
    picks = np.linspace(0, len(pairs) - 1, num).astype(int)
    chosen = [pairs[i] for i in picks]
    return AnchorSet(
        starts=np.array([c["init_xy"] for c in chosen], dtype=np.float32),
        goals=np.array([c["goal_xy"] for c in chosen], dtype=np.float32),
        names=[f"a{i}_geo{int(c['geodesic'])}" for i, c in enumerate(chosen)],
    )


def _draw_maze(ax, maze_spec) -> None:
    for i, j in np.argwhere(maze_spec.maze_map == 1):
        c = maze_spec.ij_to_xy(int(i), int(j))
        ax.add_patch(plt.Rectangle(
            (c[0] - maze_spec.unit / 2, c[1] - maze_spec.unit / 2),
            maze_spec.unit, maze_spec.unit, color="0.88", zorder=0))


def _draw_edges(ax, xy, parent, valid, lw=0.6, color="0.55", alpha=1.0) -> None:
    for n in range(len(xy)):
        if not valid[n] or parent[n] < 0:
            continue
        p = int(parent[n])
        ax.plot([xy[p, 0], xy[n, 0]], [xy[p, 1], xy[n, 1]], color=color, lw=lw, alpha=alpha, zorder=1)


@dataclass
class TreeRender:
    """Everything a view needs, extracted once from a tree."""

    xy: np.ndarray
    valid: np.ndarray
    parent: np.ndarray
    depth: np.ndarray
    order: np.ndarray
    root_branch: np.ndarray
    horizon: np.ndarray
    keep: np.ndarray
    score: np.ndarray
    goal_xy: np.ndarray
    start_xy: np.ndarray
    selected_path: list[int]

    @classmethod
    def from_tree(cls, model, tree, normalizer, goal_xy, start_xy, index=0, selected=None):
        with torch.no_grad():
            decoded = model.decoder(tree.latent[index]).float().cpu().numpy()
        xy = normalizer.denorm_obs(decoded)[:, :2]
        parent = tree.parent_index[index].cpu().numpy()
        path: list[int] = []
        if selected is not None:
            cur = int(selected)
            while cur > 0:
                path.append(cur)
                cur = int(parent[cur])
            path.append(0)
            path.reverse()
        return cls(
            xy=xy,
            valid=tree.valid[index].cpu().numpy(),
            parent=parent,
            depth=tree.depth[index].cpu().numpy(),
            order=tree.order[index].cpu().numpy(),
            root_branch=tree.root_branch[index].cpu().numpy(),
            horizon=tree.action_mask[index].sum(-1).float().cpu().numpy(),
            keep=tree.keep_score[index].float().cpu().numpy(),
            score=tree.expansion_gain[index].float().cpu().numpy(),
            goal_xy=np.asarray(goal_xy, dtype=np.float32),
            start_xy=np.asarray(start_xy, dtype=np.float32),
            selected_path=path,
        )


def _base_fig(r: TreeRender, maze_spec, title: str):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    _draw_maze(ax, maze_spec)
    _draw_edges(ax, r.xy, r.parent, r.valid)
    ax.scatter(*r.start_xy[:2], marker="*", s=200, color="crimson", zorder=5, label="root")
    ax.scatter(*r.goal_xy[:2], marker="X", s=200, color="green", zorder=5, label="goal")
    ax.set_title(title, fontsize=9)
    ax.set_aspect("equal")
    return fig, ax


def _finish(fig, ax, sc=None, label=""):
    if sc is not None:
        fig.colorbar(sc, ax=ax, label=label, fraction=0.046)
    ax.legend(loc="upper right", fontsize=7)
    fig.tight_layout()
    return fig


def view_depth(r: TreeRender, maze_spec, title="depth"):
    fig, ax = _base_fig(r, maze_spec, f"{title} | nodes={int(r.valid.sum())} max_d={int(r.depth[r.valid].max())}")
    sc = ax.scatter(r.xy[r.valid, 0], r.xy[r.valid, 1], c=r.depth[r.valid], s=24, cmap="viridis", zorder=3)
    return _finish(fig, ax, sc, "depth")


def view_expansion_order(r: TreeRender, maze_spec, title="expansion order"):
    fig, ax = _base_fig(r, maze_spec, title)
    sc = ax.scatter(r.xy[r.valid, 0], r.xy[r.valid, 1], c=r.order[r.valid], s=24, cmap="plasma", zorder=3)
    return _finish(fig, ax, sc, "expansion batch")


def view_goal_distance(r: TreeRender, maze_spec, title="goal distance"):
    d = np.linalg.norm(r.xy - r.goal_xy[None, :2], axis=1)
    fig, ax = _base_fig(r, maze_spec, f"{title} | best={d[r.valid].min():.1f}")
    sc = ax.scatter(r.xy[r.valid, 0], r.xy[r.valid, 1], c=d[r.valid], s=24, cmap="coolwarm_r", zorder=3)
    return _finish(fig, ax, sc, "decoded goal distance")


def view_root_subtree(r: TreeRender, maze_spec, title="root subtree"):
    fig, ax = _base_fig(r, maze_spec, f"{title} | unique={len(set(r.root_branch[r.valid].tolist())) - 1}")
    rb = r.root_branch[r.valid]
    sc = ax.scatter(r.xy[r.valid, 0], r.xy[r.valid, 1], c=rb, s=24, cmap="tab20", zorder=3)
    return _finish(fig, ax, sc, "root branch id")


def view_horizon(r: TreeRender, maze_spec, title="predicted horizon"):
    fig, ax = _base_fig(r, maze_spec, title)
    sc = ax.scatter(r.xy[r.valid, 0], r.xy[r.valid, 1], c=r.horizon[r.valid], s=24, cmap="cividis", zorder=3)
    return _finish(fig, ax, sc, "chunk length")


def view_selected_path(r: TreeRender, maze_spec, title="selected path"):
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    _draw_maze(ax, maze_spec)
    _draw_edges(ax, r.xy, r.parent, r.valid, lw=0.5, color="0.75", alpha=0.6)
    ax.scatter(r.xy[r.valid, 0], r.xy[r.valid, 1], s=12, color="0.6", zorder=2)
    if r.selected_path:
        pts = r.xy[np.asarray(r.selected_path)]
        ax.plot(pts[:, 0], pts[:, 1], color="orange", lw=3.0, zorder=4, label="selected path")
        ax.scatter(pts[-1, 0], pts[-1, 1], s=110, color="orange", edgecolor="k", zorder=5, label="selected leaf")
    ax.scatter(*r.start_xy[:2], marker="*", s=200, color="crimson", zorder=6, label="root")
    ax.scatter(*r.goal_xy[:2], marker="X", s=200, color="green", zorder=6, label="goal")
    ax.set_title(f"{title} | len={len(r.selected_path)}", fontsize=9)
    ax.set_aspect("equal")
    return _finish(fig, ax)


def view_topology(r: TreeRender, title="topology"):
    """Pure tree layout: x = order within depth, y = -depth. No physical coordinates."""
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    pos = {}
    for d in sorted(set(r.depth[r.valid].tolist())):
        nodes = [n for n in range(len(r.valid)) if r.valid[n] and r.depth[n] == d]
        for i, n in enumerate(nodes):
            pos[n] = (i - len(nodes) / 2.0, -float(d))
    for n, (x, y) in pos.items():
        p = int(r.parent[n])
        if p >= 0 and p in pos:
            ax.plot([pos[p][0], x], [pos[p][1], y], color="0.6", lw=0.6, zorder=1)
    xs = [pos[n][0] for n in pos]; ys = [pos[n][1] for n in pos]
    cs = [r.root_branch[n] for n in pos]
    sc = ax.scatter(xs, ys, c=cs, s=22, cmap="tab20", zorder=2)
    if r.selected_path:
        px = [pos[n][0] for n in r.selected_path if n in pos]
        py = [pos[n][1] for n in r.selected_path if n in pos]
        ax.plot(px, py, color="orange", lw=2.5, zorder=3)
    ax.set_xlabel("index within depth"); ax.set_ylabel("-depth")
    ax.set_title(f"{title} | nodes={int(r.valid.sum())}", fontsize=9)
    fig.colorbar(sc, ax=ax, label="root branch", fraction=0.046)
    fig.tight_layout()
    return fig


def view_predicted_vs_grounded(r: TreeRender, actual_xy, grounded, maze_spec, title="predicted vs grounded"):
    fig, ax = _base_fig(r, maze_spec, title)
    ok = r.valid & grounded
    ax.scatter(r.xy[ok, 0], r.xy[ok, 1], s=22, color="tab:blue", zorder=3, label="predicted")
    ax.scatter(actual_xy[ok, 0], actual_xy[ok, 1], s=22, color="tab:red", marker="s", zorder=3, label="actual")
    for n in np.flatnonzero(ok):
        ax.plot([r.xy[n, 0], actual_xy[n, 0]], [r.xy[n, 1], actual_xy[n, 1]],
                color="0.4", lw=0.5, alpha=0.7, zorder=2)
    err = np.linalg.norm(r.xy[ok] - actual_xy[ok], axis=1)
    ax.set_title(f"{title} | mean err={err.mean():.2f}", fontsize=9)
    return _finish(fig, ax)


def expansion_video(model, tree_frames, normalizer, maze_spec, goal_xy, start_xy, fps: int = 2):
    """``[1, T, C, H, W]`` uint8 tensor for ``SummaryWriter.add_video``.

    One frame per expansion batch: root, batch 1, batch 2, ..., final tree. Makes it
    possible to see *when* a policy commits to one subtree rather than only the end state.
    """
    frames = []
    for r in tree_frames:
        fig, ax = _base_fig(r, maze_spec, f"expansion step {int(r.order[r.valid].max())}")
        sc = ax.scatter(r.xy[r.valid, 0], r.xy[r.valid, 1], c=r.depth[r.valid], s=24,
                        cmap="viridis", vmin=0, vmax=8, zorder=3)
        _finish(fig, ax, sc, "depth")
        fig.canvas.draw()
        buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
        frames.append(buf)
    if not frames:
        return None
    h = min(f.shape[0] for f in frames); w = min(f.shape[1] for f in frames)
    stack = np.stack([f[:h, :w] for f in frames])  # [T, H, W, C]
    video = torch.from_numpy(stack).permute(0, 3, 1, 2).unsqueeze(0)  # [1, T, C, H, W]
    return video
