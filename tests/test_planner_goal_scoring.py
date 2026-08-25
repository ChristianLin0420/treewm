"""Focused checks for decoded planner goal metrics."""

from __future__ import annotations

import numpy as np
import torch

from treewm.evaluation.domains import Domain
from treewm.planning.goal_planner import PlannerConfig, decoded_goal_scores


def _domain_raw_scores(
    nodes_n: torch.Tensor,
    goal_n: torch.Tensor,
    domain: Domain,
    *,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    return decoded_goal_scores(
        nodes_n,
        goal_n,
        decoded_metric="domain_raw",
        goal_dims=torch.tensor(domain.goal_dims),
        goal_metric=domain.goal_metric,
        subgoals=domain.subgoals,
        obs_mean=mean,
        obs_std=std,
    )


def test_domain_raw_l2_reverses_anisotropic_normalized_ordering():
    """Raw task units, not dataset variance, must determine continuous proximity."""
    domain = Domain("anisotropic", "test", (0, 1), "l2", obs_dim=2, action_dim=1)
    mean = torch.zeros(2)
    std = torch.tensor([10.0, 1.0])
    goal_n = torch.zeros(1, 2)
    # Normalised L2 prefers node 0 (0.2 < 1.0), but in raw units node 1 is closer
    # (1.0 < 2.0) because the first coordinate has ten times the physical scale.
    nodes_n = torch.tensor([[[0.2, 0.0], [0.0, 1.0]]])

    historical = decoded_goal_scores(
        nodes_n,
        goal_n,
        decoded_metric="normalized_l2",
        goal_dims=torch.tensor(domain.goal_dims),
    )
    corrected = _domain_raw_scores(nodes_n, goal_n, domain, mean=mean, std=std)

    assert int(historical.argmin(1).item()) == 0
    assert int(corrected.argmin(1).item()) == 1
    np.testing.assert_allclose(corrected.numpy(), [[2.0, 1.0]], rtol=0, atol=1e-6)

    # The tensor implementation agrees with Domain.distance after denormalisation.
    goal_raw = (goal_n[0] * std + mean).numpy()
    nodes_raw = (nodes_n[0] * std + mean).numpy()
    expected = [domain.distance(node, goal_raw) for node in nodes_raw]
    np.testing.assert_allclose(corrected[0].numpy(), expected, rtol=0, atol=1e-6)


def test_onehot_hamming_dominates_soft_confidence():
    """No confidence tie-break can outweigh one additional categorical error."""
    domain = Domain(
        "buttons",
        "puzzle",
        (0, 1, 2, 3),
        "onehot",
        obs_dim=4,
        action_dim=1,
        subgoals=((0, 2), (2, 4)),
    )
    mean = torch.zeros(4)
    std = torch.ones(4)
    goal = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    nodes = torch.tensor(
        [[
            # Both argmaxes are correct but confidence is deliberately very weak.
            [0.5001, 0.4999, 0.5001, 0.4999],
            # One exact block and one confidently wrong block.
            [0.0, 1.0, 1.0, 0.0],
            # Same Hamming count as node 0, with exact one-hot confidence.
            [1.0, 0.0, 1.0, 0.0],
        ]]
    )

    scores = _domain_raw_scores(nodes, goal, domain, mean=mean, std=std)

    assert 0.0 < float(scores[0, 0]) < 1.0e-3
    assert float(scores[0, 1]) >= 1.0
    assert float(scores[0, 0]) < float(scores[0, 1])
    assert float(scores[0, 2]) < float(scores[0, 0])
    assert [domain.hamming(node.numpy(), goal[0].numpy()) for node in nodes[0]] == [0, 1, 0]


def test_historical_planner_default_is_explicit():
    assert PlannerConfig().decoded_metric == "normalized_l2"
