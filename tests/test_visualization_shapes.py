"""Regression tests for full-observation maze and variable-size puzzle renders."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import torch

from treewm.data.ogbench_dataset import Normalizer
from treewm.evaluation import diagnostics, domain_viz, tree_viz
from treewm.evaluation.domains import get_domain


def _normalizer(obs_dim: int) -> Normalizer:
    return Normalizer(
        obs_mean=np.linspace(-1.0, 1.0, obs_dim, dtype=np.float32),
        obs_std=np.linspace(0.5, 1.5, obs_dim, dtype=np.float32),
        act_mean=np.zeros(1, dtype=np.float32),
        act_std=np.ones(1, dtype=np.float32),
    )


def test_antmaze_heatmap_grid_embeds_xy_in_full_observation():
    class Maze:
        unit = 1.0

        @staticmethod
        def free_cells():
            return np.asarray([[2, 3]])

        @staticmethod
        def ij_to_xy(i, j):
            return np.asarray([float(i), float(j)], dtype=np.float32)

    normalizer = _normalizer(29)
    xy, obs_n = diagnostics._grid_states(Maze(), normalizer, samples_per_cell=3)
    raw = normalizer.denorm_obs(obs_n)
    assert xy.shape == (9, 2)
    assert obs_n.shape == (9, 29)
    assert np.allclose(raw[:, :2], xy)
    assert np.allclose(raw[:, 2:], normalizer.obs_mean[None, 2:])


def test_maze_xy_anchors_expand_to_encoder_observation_width():
    anchors = tree_viz.AnchorSet(
        starts=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        goals=np.asarray([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        names=["first", "second"],
    )
    normalizer = _normalizer(69)
    expanded = tree_viz.expand_xy_anchors(anchors, normalizer)
    assert expanded.starts.shape == (2, 69)
    assert expanded.goals.shape == (2, 69)
    assert np.array_equal(expanded.starts[:, :2], anchors.starts)
    assert np.array_equal(expanded.goals[:, :2], anchors.goals)
    assert np.allclose(normalizer.norm_obs(expanded.starts)[:, 2:], 0.0, atol=1e-6)
    assert expanded.names == anchors.names


def test_puzzle_4x4_board_render_infers_four_by_four_grid():
    domain = get_domain("puzzle-4x4-play-v0")
    nodes = np.zeros((2, domain.obs_dim), dtype=np.float32)
    goal = np.zeros(domain.obs_dim, dtype=np.float32)
    for cell, (lo, hi) in enumerate(domain.subgoals):
        dim_block = np.asarray(domain.goal_dims[lo:hi])
        nodes[:, dim_block[cell % len(dim_block)]] = 1.0
        goal[dim_block[(cell + 1) % len(dim_block)]] = 1.0

    class Model:
        @staticmethod
        def decoder(latent):
            return latent

    class Tree:
        latent = torch.from_numpy(nodes[None])
        valid = torch.tensor([[True, True]])
        depth = torch.tensor([[0, 1]])

    class IdentityNormalizer:
        @staticmethod
        def denorm_obs(value):
            return value

    fig = domain_viz.view_board_by_depth(
        Model(), Tree(), IdentityNormalizer(), domain, goal
    )
    try:
        assert len(fig.axes) == 3  # depth 0, depth 1, target
        assert all(tuple(axis.images[0].get_array().shape) == (4, 4) for axis in fig.axes)
    finally:
        plt.close(fig)
