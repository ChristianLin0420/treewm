"""Focused checks for decoded planner goal metrics."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from treewm.evaluation.domains import Domain
from treewm.planning.goal_planner import (
    GoalPlanner,
    PlannerConfig,
    decoded_goal_scores,
    first_edge_improvement_mask,
    first_root_edge_indices,
)
from treewm.tree.expansion import TreeConfig
from treewm.tree.node import BatchedTree


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
    assert not PlannerConfig().require_first_edge_improvement


def test_first_edge_guard_uses_executable_root_edge_not_deep_endpoint():
    # 0=root; 1 and 2 are root children; 3 descends from 1, 4 descends from 2.
    parents = torch.tensor([[-1, 0, 0, 1, 2]])
    valid = torch.ones_like(parents, dtype=torch.bool)
    scores = torch.tensor([[5.0, 6.0, 4.0, 1.0, 3.0]])

    first = first_root_edge_indices(parents)
    # The decoded root is deliberately wrong (50); the guard must use the actual
    # current-observation distance (5) supplied separately.
    scores[:, 0] = 50.0
    first = first_root_edge_indices(parents)
    executable_scores = scores.gather(1, first)
    allowed = first_edge_improvement_mask(
        executable_scores,
        valid,
        current_observation_score=torch.tensor([5.0]),
    )

    assert first.tolist() == [[0, 1, 2, 1, 2]]
    # Deep node 3 looks best in isolation, but executing its root edge (node 1) would
    # regress from 5 to 6, so the whole subtree is rejected. Nodes 2/4 are safe.
    assert allowed.tolist() == [[False, False, True, False, True]]


def test_first_edge_guard_is_fail_closed_and_validates_contract():
    parents = torch.tensor([[-1, 0, 1, -1]])
    valid = torch.tensor([[True, True, True, False]])
    scores = torch.tensor([[1.0, 1.0, 0.0, -100.0]])

    assert not first_edge_improvement_mask(
        scores.gather(1, first_root_edge_indices(parents)),
        valid,
        current_observation_score=torch.tensor([1.0]),
        minimum_improvement=0.1,
    ).any()
    with np.testing.assert_raises(ValueError):
        first_root_edge_indices(torch.tensor([[-1, 2, 0, 0]]))
    with np.testing.assert_raises(ValueError):
        first_edge_improvement_mask(
            scores,
            valid,
            current_observation_score=torch.tensor([1.0]),
            minimum_improvement=-1.0,
        )


def test_first_edge_guard_rejects_bad_current_score_shape():
    scores = torch.tensor([[1.0, 0.5]])
    parents = torch.tensor([[-1, 0]])
    valid = torch.ones_like(parents, dtype=torch.bool)
    with np.testing.assert_raises(ValueError):
        first_edge_improvement_mask(
            scores,
            valid,
            current_observation_score=torch.tensor([[1.0]]),
        )


def test_guard_scores_the_executed_four_step_prefix_and_maps_root_branches():
    class FakeModel:
        cfg = type(
            "Cfg",
            (),
            {"horizons": (4, 8, 16), "branch_factor": 2},
        )()
        executed_horizons = None

        def predict_children(self, root_z, depth, horizon_override):
            assert depth.tolist() == [0]
            self.executed_horizons = torch.tensor(self.cfg.horizons)[horizon_override]
            # Prefix predictions for original root branches 0 and 1.
            return {"latent": torch.tensor([[[3.0], [1.0]]])}

        def decoder(self, latent):
            return latent

    planner = object.__new__(GoalPlanner)
    planner.cfg = PlannerConfig(
        decoded_metric="normalized_l2",
        execute_mode="clipped",
        execute_steps=4,
    )
    planner.domain = None
    planner.goal_dims = torch.tensor([0])
    planner.obs_mean = None
    planner.obs_std = None
    tree = type(
        "Tree",
        (),
        {
            "root_branch": torch.tensor([[-1, 0, 1, 0]]),
            "valid": torch.tensor([[True, True, True, True]]),
        },
    )()
    model = FakeModel()
    scores = planner._executable_first_edge_scores(
        model,
        torch.zeros(1, 1),
        tree,
        torch.zeros(1, 1),
    )
    assert model.executed_horizons.tolist() == [[4, 4]]
    assert scores.tolist() == [[3.0, 3.0, 1.0, 3.0]]


def test_plan_rejects_full_horizon_winner_when_its_executed_prefix_regresses():
    class IdentityNormalizer:
        obs_mean = np.zeros(1, dtype=np.float32)
        obs_std = np.ones(1, dtype=np.float32)

        @staticmethod
        def norm_obs(value):
            return np.asarray(value, dtype=np.float32)

        @staticmethod
        def denorm_act(value):
            return np.asarray(value, dtype=np.float32)

    tree = BatchedTree.initialize(
        torch.tensor([[5.0]]),
        torch.zeros(1, 1, 1),
        capacity=5,
        h_max=16,
        action_dim=1,
    )
    # The deep endpoint in root branch 0 looks best (distance 0.5), but its true e4
    # prefix will be predicted at distance 6 from the goal and must reject slots 1/3.
    # Root branch 1 has a safe e4 prefix (distance 3), so its deep slot 4 should win.
    tree.latent[0, :, 0] = torch.tensor([5.0, 1.0, 4.0, 0.5, 2.0])
    tree.valid[:] = True
    tree.parent_index[0] = torch.tensor([-1, 0, 0, 1, 2])
    tree.root_branch[0] = torch.tensor([-1, 0, 1, 0, 1])
    tree.depth[0] = torch.tensor([0, 1, 1, 2, 2])
    tree.num_nodes[0] = 5
    tree.action_mask[0, 1:3, :8] = 1.0
    tree.action_chunk[0, 1, :8, 0] = -0.75
    tree.action_chunk[0, 2, :8, 0] = torch.tensor(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )

    class FakeModel:
        cfg = type(
            "Cfg",
            (),
            {
                "horizons": (4, 8, 16),
                "branch_factor": 2,
                "action_dim": 1,
            },
        )()

        def __init__(self):
            self.executed_horizons = None
            self.prefix_latent = torch.tensor([[[6.0], [3.0]]])

        @staticmethod
        def encode(obs):
            return obs

        @staticmethod
        def decoder(latent):
            return latent

        @staticmethod
        def generate(*_args, **_kwargs):
            return tree, None

        def predict_children(self, root_z, depth, horizon_override):
            assert root_z.tolist() == [[5.0]]
            assert depth.tolist() == [0]
            self.executed_horizons = torch.tensor(self.cfg.horizons)[horizon_override]
            return {"latent": self.prefix_latent}

    model = FakeModel()
    guarded = GoalPlanner(
        model,
        IdentityNormalizer(),
        TreeConfig(node_budget=5, branch_factor=2, max_depth=2),
        PlannerConfig(
            decoded_metric="normalized_l2",
            execute_mode="clipped",
            execute_steps=4,
            require_first_edge_improvement=True,
        ),
        device=torch.device("cpu"),
    )
    plan = guarded.plan(
        np.asarray([5.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
    )

    assert model.executed_horizons.tolist() == [[4, 4]]
    assert plan.selected_node == 4
    assert tree.root_branch[0, plan.selected_node].item() == 1
    np.testing.assert_allclose(
        plan.actions[:, 0], [0.1, 0.2, 0.3, 0.4], rtol=0, atol=1e-7
    )
    # Five search-tree nodes plus the two actually evaluated e4 prefix successors.
    assert plan.num_nodes == 7
    assert plan.guard_applied
    assert plan.guard_candidate_count == 4
    assert plan.guard_accepted_count == 2
    assert not plan.guard_rejected_all
    assert plan.guard_best_predicted_improvement == pytest.approx(2.0)
    assert plan.guard_selected_predicted_improvement == pytest.approx(2.0)

    # Both executable root prefixes now regress from the actual current score of five.
    # Endpoint slot 3 still looks excellent, so only the guard can make this no-action.
    model.prefix_latent = torch.tensor([[[6.0], [7.0]]])
    rejected = guarded.plan(
        np.asarray([5.0], dtype=np.float32),
        np.asarray([0.0], dtype=np.float32),
    )
    assert rejected.actions.shape == (0, 1)
    assert rejected.guard_applied
    assert rejected.guard_candidate_count == 4
    assert rejected.guard_accepted_count == 0
    assert rejected.guard_rejected_all
    assert rejected.guard_best_predicted_improvement == pytest.approx(-1.0)
    assert rejected.guard_selected_predicted_improvement == 0.0


def test_root_only_latent_planner_fails_closed_without_decoded_score():
    class IdentityNormalizer:
        obs_mean = np.zeros(1, dtype=np.float32)
        obs_std = np.ones(1, dtype=np.float32)

        @staticmethod
        def norm_obs(value):
            return np.asarray(value, dtype=np.float32)

        @staticmethod
        def denorm_act(value):
            return np.asarray(value, dtype=np.float32)

    tree = BatchedTree.initialize(
        torch.tensor([[5.0]]),
        torch.zeros(1, 1, 1),
        capacity=1,
        h_max=4,
        action_dim=1,
    )

    class LatentModel:
        cfg = type("Cfg", (), {"action_dim": 1})()
        decoder = None

        @staticmethod
        def encode(obs):
            return obs

        @staticmethod
        def generate(*_args, **_kwargs):
            return tree, None

    planner = GoalPlanner(
        LatentModel(),
        IdentityNormalizer(),
        TreeConfig(node_budget=1),
        PlannerConfig(score_space="latent", exclude_root=True),
        device=torch.device("cpu"),
    )
    plan = planner.plan(
        np.asarray([5.0], dtype=np.float32),
        np.asarray([2.0], dtype=np.float32),
    )

    assert plan.selected_node == 0
    assert plan.actions.shape == (0, 1)
    assert plan.goal_distance == 3.0
    assert plan.num_nodes == 1
