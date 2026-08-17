"""Scientific diagnostics and qualitative visualisations.

The two most important functions here are not plots:

``q_vs_z_retrieval`` runs the controllability premise check with a dimension-matched
control, and ``branching_diversity_correlation`` measures whether the model's effective
branching factor tracks *empirical* local future diversity -- the major diagnostic of
spec section 19-H. Geometry-based checks are secondary and never enter a loss.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from treewm.logging.metrics import rank_correlation
from treewm.losses.controllability_losses import retrieval_precision


@torch.no_grad()
def q_vs_z_retrieval(model, batch: dict[str, torch.Tensor], k: int = 5) -> dict[str, float]:
    """Do q-space distances predict future-set similarity better than z-space?

    Three embeddings are compared on the *same* task with the *same* metric:

        d_q       the learned controllability distance
        d_z       the full state latent
        d_rand    a frozen random projection of z down to q_dim

    The third is the control that makes the comparison honest. ``q = C(z)`` is a
    deterministic function of ``z`` and so cannot hold more information about the
    future; if q beats z but not the random projection, the apparent win is
    dimensionality, not controllability structure.
    """
    z = model.encode(batch["obs"])
    q = model.q_of(z)
    endpoints, valid = batch["fut_endpoint"], batch["fut_valid"]

    def q_dist(_: torch.Tensor) -> torch.Tensor:
        b = q.shape[0]
        return model.q_distance(
            q.unsqueeze(1).expand(b, b, *q.shape[1:]), q.unsqueeze(0).expand(b, b, *q.shape[1:])
        )

    prec_q = retrieval_precision(q, endpoints, valid, q_dist, k)
    prec_z = retrieval_precision(z, endpoints, valid, lambda e: torch.cdist(e, e), k)
    rand = model.z_control_projection(z)
    prec_rand = retrieval_precision(rand, endpoints, valid, lambda e: torch.cdist(e, e), k)

    return {
        "control/local_future_set_retrieval_precision": prec_q,
        "control/retrieval_precision_q": prec_q,
        "control/retrieval_precision_z": prec_z,
        "control/retrieval_precision_random_proj": prec_rand,
        # The two quantities that decide the premise. Both must be positive.
        "control/q_advantage_over_z": prec_q - prec_z,
        "control/q_advantage_over_random_proj": prec_q - prec_rand,
    }


@torch.no_grad()
def branching_diversity_correlation(
    model, batch: dict[str, torch.Tensor], keep_threshold: float = 0.5
) -> dict[str, float]:
    """Correlate effective branching factor with empirical local future diversity.

    If the model is doing what the hypothesis says, it should open more branches exactly
    where the data actually offers more distinct futures. The reference is data-derived
    (``future_diversity`` and ``num_modes`` from the retrieved future set), not maze
    geometry.
    """
    z = model.encode(batch["obs"])
    out = model.branch(z)
    ebf = (out.keep > keep_threshold).float().sum(-1)

    diversity = batch["future_diversity"].float()
    num_modes = batch["num_modes"].float()
    return {
        "control/branching_future_diversity_corr": rank_correlation(ebf, diversity),
        "control/branching_num_modes_corr": rank_correlation(ebf, num_modes),
        "control/equivalent_branch_merge_rate": float(
            (1.0 - ebf / max(out.keep.shape[-1], 1)).mean().item()
        ),
    }


def _grid_states(maze_spec, normalizer, samples_per_cell: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Sample valid xy positions across all free maze cells."""
    cells = maze_spec.free_cells()
    offsets = np.linspace(-0.35, 0.35, samples_per_cell)
    pts = []
    for i, j in cells:
        base = maze_spec.ij_to_xy(int(i), int(j))
        for dx in offsets:
            for dy in offsets:
                pts.append(base + np.array([dx, dy]) * maze_spec.unit)
    xy = np.asarray(pts, dtype=np.float32)
    return xy, normalizer.norm_obs(xy)


@torch.no_grad()
def _scatter_heatmap(xy: np.ndarray, values: np.ndarray, maze_spec, title: str, cmap: str = "viridis"):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    walls = np.argwhere(maze_spec.maze_map == 1)
    for i, j in walls:
        c = maze_spec.ij_to_xy(int(i), int(j))
        ax.add_patch(
            plt.Rectangle(
                (c[0] - maze_spec.unit / 2, c[1] - maze_spec.unit / 2),
                maze_spec.unit, maze_spec.unit, color="0.85", zorder=0,
            )
        )
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=values, s=14, cmap=cmap, zorder=2)
    fig.colorbar(sc, ax=ax)
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


@torch.no_grad()
def branching_factor_heatmap(model, maze_spec, normalizer, device, keep_threshold: float = 0.5):
    """``viz/branching_factor_heatmap``: corridors should branch less than junctions."""
    xy, norm = _grid_states(maze_spec, normalizer)
    obs = torch.from_numpy(norm).to(device)
    ebf = []
    for i in range(0, len(obs), 4096):
        out = model.branch(model.encode(obs[i : i + 4096]))
        ebf.append((out.keep > keep_threshold).float().sum(-1).cpu().numpy())
    return _scatter_heatmap(xy, np.concatenate(ebf), maze_spec, "effective branching factor")


@torch.no_grad()
def expansion_gain_heatmap(model, maze_spec, normalizer, device):
    """``viz/expansion_gain_heatmap`` using the context-free per-branch gain prior."""
    xy, norm = _grid_states(maze_spec, normalizer)
    obs = torch.from_numpy(norm).to(device)
    gains = []
    for i in range(0, len(obs), 4096):
        out = model.branch(model.encode(obs[i : i + 4096]))
        gains.append(out.gain_prior.max(-1).values.float().cpu().numpy())
    return _scatter_heatmap(xy, np.concatenate(gains), maze_spec, "predicted expansion gain", cmap="magma")


@torch.no_grad()
def tree_plot(model, tree, normalizer, maze_spec, index: int = 0, title: str = "generated tree"):
    """Render one generated tree in physical XY with edges, duration and KEEP."""
    if model.decoder is None:
        raise ValueError("tree_plot requires the reconstruction decoder")
    states = normalizer.denorm_obs(model.decoder(tree.latent[index]).float().cpu().numpy())
    valid = tree.valid[index].cpu().numpy()
    parent = tree.parent_index[index].cpu().numpy()
    keep = tree.keep_score[index].float().cpu().numpy()
    order = tree.order[index].cpu().numpy()
    horizon = tree.action_mask[index].sum(-1).float().cpu().numpy()

    fig, ax = plt.subplots(figsize=(6.5, 6))
    for i, j in np.argwhere(maze_spec.maze_map == 1):
        c = maze_spec.ij_to_xy(int(i), int(j))
        ax.add_patch(
            plt.Rectangle(
                (c[0] - maze_spec.unit / 2, c[1] - maze_spec.unit / 2),
                maze_spec.unit, maze_spec.unit, color="0.88", zorder=0,
            )
        )

    for n in range(len(states)):
        if not valid[n] or parent[n] < 0:
            continue
        p = parent[n]
        ax.plot(
            [states[p, 0], states[n, 0]], [states[p, 1], states[n, 1]],
            color="0.5", lw=0.5 + 2.0 * float(horizon[n]) / max(horizon.max(), 1), zorder=1,
        )
    sc = ax.scatter(
        states[valid, 0], states[valid, 1], c=keep[valid], s=26, cmap="viridis", vmin=0, vmax=1, zorder=3
    )
    ax.scatter(states[0, 0], states[0, 1], marker="*", s=220, color="crimson", zorder=4, label="root")
    fig.colorbar(sc, ax=ax, label="KEEP")
    ax.set_title(f"{title} (nodes={int(valid.sum())}, max order={int(order.max())})")
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


@torch.no_grad()
def q_pca_plot(model, batch, normalizer, maze_spec=None):
    """``viz/q_pca``: PCA of controllability embeddings, coloured by geometry.

    Colour is *only* a visualisation aid -- position labels are never trained on
    (spec section 20).
    """
    z = model.encode(batch["obs"])
    q = model.q_of(z).flatten(1).float().cpu().numpy()
    xy = normalizer.denorm_obs(batch["obs"].float().cpu().numpy())

    q = q - q.mean(0, keepdims=True)
    try:
        _, _, vt = np.linalg.svd(q, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    proj = q @ vt[:2].T

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, colour, name in ((axes[0], xy[:, 0], "x"), (axes[1], xy[:, 1], "y")):
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=colour, s=12, cmap="coolwarm")
        fig.colorbar(sc, ax=ax, label=name)
        ax.set_title(f"q-space PCA (coloured by {name})")
    fig.tight_layout()
    return fig


@torch.no_grad()
def planning_plot(model, tree, normalizer, maze_spec, start_xy, goal_xy, path_nodes, executed=None):
    """``viz/planning_example``: tree, goal, selected leaf and executed prefix."""
    fig = tree_plot(model, tree, normalizer, maze_spec, 0, "planning example")
    ax = fig.axes[0]
    states = normalizer.denorm_obs(model.decoder(tree.latent[0]).float().cpu().numpy())
    ax.scatter(*goal_xy[:2], marker="X", s=220, color="green", zorder=5, label="goal")
    if path_nodes:
        pts = states[np.asarray(path_nodes)]
        ax.plot(pts[:, 0], pts[:, 1], color="orange", lw=2.5, zorder=4, label="selected path")
    if executed is not None and len(executed):
        ex = np.asarray(executed)
        ax.plot(ex[:, 0], ex[:, 1], color="black", lw=2.0, ls="--", zorder=6, label="executed")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return fig


@torch.no_grad()
def geometry_sanity(model, batch, maze_spec, normalizer, keep_threshold: float = 0.5) -> dict[str, float]:
    """Secondary check: does branching correlate with maze junction structure?

    Reported alongside the data-derived correlation because geometry only partly
    determines future diversity -- measured at ~0.3 rank correlation on
    pointmaze-medium before any training -- so a weak value here is expected and is not
    evidence against the hypothesis on its own.
    """
    z = model.encode(batch["obs"])
    ebf = (model.branch(z).keep > keep_threshold).float().sum(-1).cpu().numpy()
    xy = normalizer.denorm_obs(batch["obs"].float().cpu().numpy())
    ij = maze_spec.xy_to_ij(xy)
    degree = maze_spec.junction_degree()[ij[:, 0], ij[:, 1]]
    return {"control/branching_junction_degree_corr": rank_correlation(ebf, degree)}

def interaction_sanity(model, batch, domain, normalizer, keep_threshold: float = 0.5) -> dict[str, float]:
    """Non-maze counterpart of :func:`geometry_sanity`.

    ``branching_diversity_correlation`` is the primary check, but it compares the model
    against ``future_diversity`` produced by the *same* retrieval machinery that built its
    training targets. If retrieval is broken, both go wrong together and agree -- so a
    cross-check that does not touch that pipeline is needed. In a maze that role is played
    by junction degree, which is ground truth from the map.

    The manipulation analogue of a junction is proximity to something you can act on:
    futures genuinely diverge when the gripper can grasp a cube (push, lift, rotate,
    ignore), and collapse to "move somewhere" in free space. So branching should
    anti-correlate with effector-to-nearest-object distance, giving a *positive*
    correlation with proximity.

    Like the maze version this is a weak secondary signal -- geometry only partly
    determines future diversity -- so a small value is expected and is not on its own
    evidence against the hypothesis. A strongly negative value would be, since it would
    mean the model opens branches precisely where nothing can be done.

    Returns ``{}`` where the observation exposes no actionable object position (puzzle
    reports button press depth, not button location; the two mazes use geometry_sanity).
    """
    if not domain.effector_dims or not domain.object_dims:
        return {}
    z = model.encode(batch["obs"])
    ebf = (model.branch(z).keep > keep_threshold).float().sum(-1).cpu().numpy()
    obs = normalizer.denorm_obs(batch["obs"].float().cpu().numpy())

    eff = obs[:, list(domain.effector_dims)]
    dists = np.stack([np.linalg.norm(obs[:, list(od)] - eff, axis=1)
                      for od in domain.object_dims], axis=1)
    nearest = dists.min(axis=1)
    return {
        "control/branching_object_proximity_corr": rank_correlation(ebf, -nearest),
        "control/effector_object_distance_mean": float(nearest.mean()),
    }
