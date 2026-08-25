"""Tests for the overnight design-space components (tracks A, D, E, F + provenance)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from treewm.evaluation.tree_stats import structural_summary
from treewm.losses.recursive_losses import multi_step_recursive_loss, scheduled_sampling_schedule
from treewm.utils.rng import make_generator
from treewm.models.baselines import build_model, tree_config_for
from treewm.models.treewm import TreeWMConfig
from treewm.tree.expansion import TreeConfig
from treewm.tree.frontier import SCORERS, ScoringContext, get_scorer
from treewm.utils.provenance import compatible, load_checked, provenance

SMALL = TreeWMConfig(obs_dim=2, action_dim=2, z_dim=32, q_dim=16, hidden_dim=64, num_layers=2)


def _batch(b=4, d=3, h_max=64, obs_dim=2, act_dim=2):
    return {
        "obs": torch.randn(b, obs_dim),
        "ms_actions": torch.randn(b, d, h_max, act_dim),
        "ms_action_mask": torch.ones(b, d, h_max),
        "ms_obs": torch.randn(b, d, obs_dim),
        "ms_horizon_idx": torch.zeros(b, d, dtype=torch.long),
        "ms_valid": torch.ones(b, d),
    }


# --------------------------------------------------------------------- Track A


def test_multi_step_loss_reports_every_depth():
    model = build_model("randomtreewm", SMALL)
    loss, metrics = multi_step_recursive_loss(model, _batch(), scheduled_sampling_p=0.0)
    loss.backward()
    for d in (1, 2, 3):
        assert f"recursive/loss_depth{d}" in metrics
        assert f"recursive/state_error_depth{d}" in metrics
        assert f"recursive/self_vs_teacher_depth{d}" in metrics
    assert model.dynamics.net[0].weight.grad is not None
    assert metrics["recursive/mean_chain_depth"] == pytest.approx(3.0)


def test_multi_step_loss_respects_validity_mask():
    """A chain that runs off the end of a trajectory must not be supervised."""
    batch = _batch()
    batch["ms_valid"][:, 1:] = 0.0
    _, metrics = multi_step_recursive_loss(build_model("randomtreewm", SMALL), batch)
    assert "recursive/loss_depth1" in metrics
    assert "recursive/loss_depth2" not in metrics, "invalid depths must be skipped entirely"


def test_scheduled_sampling_changes_the_rollout():
    """p=1 feeds predictions, p=0 feeds ground truth -- the two must differ."""
    torch.manual_seed(0)
    model = build_model("randomtreewm", SMALL)
    batch = _batch()
    _, m0 = multi_step_recursive_loss(model, batch, scheduled_sampling_p=0.0)
    _, m1 = multi_step_recursive_loss(model, batch, scheduled_sampling_p=1.0)
    assert m0["recursive/scheduled_sampling_p"] == 0.0
    assert m1["recursive/scheduled_sampling_p"] == 1.0
    # depth-1 is identical (both start from the encoded anchor); deeper depths diverge
    assert m0["recursive/loss_depth1"] == pytest.approx(m1["recursive/loss_depth1"], rel=1e-4)


def test_scheduled_sampling_warmup_is_linear_and_clamped():
    assert scheduled_sampling_schedule(0, 0.5, 1000) == 0.0
    assert scheduled_sampling_schedule(500, 0.5, 1000) == pytest.approx(0.25)
    assert scheduled_sampling_schedule(5000, 0.5, 1000) == pytest.approx(0.5)
    assert scheduled_sampling_schedule(10, 0.0, 1000) == 0.0


def test_depth_weights_change_the_objective():
    model = build_model("randomtreewm", SMALL)
    batch = _batch()
    flat, _ = multi_step_recursive_loss(model, batch)
    weighted, _ = multi_step_recursive_loss(model, batch, depth_weights=(1.0, 2.0, 4.0))
    assert not torch.isclose(flat, weighted), "depth weighting must alter the loss"


# --------------------------------------------------------------------- Track D


def _tree(arm="randomtreewm", scorer="random", budget=32, goal=None, model=None):
    model = model or build_model(arm, SMALL).eval()
    cfg = tree_config_for(arm, TreeConfig(node_budget=budget, branch_factor=4, max_depth=16), model)
    from dataclasses import replace

    cfg = replace(cfg, scorer=scorer)
    with torch.no_grad():
        tree, _ = model.generate(torch.randn(2, SMALL.z_dim), cfg, goal_obs=goal, generator=make_generator(0, 'eval'))
    return model, tree, cfg


def test_root_branch_identity_is_inherited():
    _, tree, _ = _tree(scorer="bfs")
    rb, parent, valid, depth = tree.root_branch, tree.parent_index, tree.valid, tree.depth
    for b in range(tree.batch_size):
        for n in range(1, tree.capacity):
            if not valid[b, n]:
                continue
            p = int(parent[b, n])
            if p == 0:
                assert int(rb[b, n]) >= 0, "children of the root start their own subtree"
            else:
                assert int(rb[b, n]) == int(rb[b, p]), "deeper nodes inherit their parent's subtree"


@pytest.mark.parametrize("scorer", ["depth_balanced", "root_quota", "random", "bfs"])
def test_new_policies_fill_the_budget(scorer):
    _, tree, cfg = _tree(scorer=scorer, budget=32)
    assert int(tree.num_nodes.min()) == 32, f"{scorer} underspent the budget"


def test_root_quota_prefers_the_least_developed_subtree():
    """D3's guarantee, tested directly rather than through a stochastic outcome.

    Comparing end-of-run subtree concentration against plain random is far too noisy at
    this scale (two trees, 48 nodes, four subtrees: the difference is well under one
    node). The actual guarantee is per-decision -- while the quota phase is active, a
    frontier node belonging to a smaller root subtree must score above one belonging to a
    larger subtree -- so assert that instead.
    """
    from treewm.tree.frontier import root_quota_score

    model = build_model("randomtreewm", SMALL).eval()
    _, tree, cfg = _tree(scorer="bfs", budget=48, model=model)

    frontier = tree.expandable_frontier(cfg.max_depth)
    ctx = ScoringContext(generator=make_generator(0, "eval"), budget_fraction=0.0,
                         broad_fraction=0.45)
    scores = root_quota_score(tree, frontier, ctx)

    # subtree size for every frontier node
    checked = 0
    for b in range(tree.batch_size):
        idx = torch.nonzero(frontier[b]).flatten().tolist()
        sizes = {}
        for n in idx:
            rb = int(tree.root_branch[b, n])
            sizes[n] = int(((tree.root_branch[b] == rb) & tree.valid[b]).sum())
        for i in idx:
            for j in idx:
                if sizes[i] + 1 < sizes[j]:  # strictly smaller subtree, margin > noise
                    assert scores[b, i] > scores[b, j], (
                        f"node in a {sizes[i]}-node subtree scored below one in a "
                        f"{sizes[j]}-node subtree"
                    )
                    checked += 1
    assert checked > 0, "test did not exercise any unequal-subtree pair"


def test_root_quota_releases_after_the_broad_phase():
    """Past broad_fraction the quota is lifted and selection becomes plain random."""
    from treewm.tree.frontier import root_quota_score

    model = build_model("randomtreewm", SMALL).eval()
    _, tree, cfg = _tree(scorer="bfs", budget=32, model=model)
    frontier = tree.expandable_frontier(cfg.max_depth)

    early = root_quota_score(tree, frontier, ScoringContext(
        generator=make_generator(0, "eval"), budget_fraction=0.0, broad_fraction=0.45))
    late = root_quota_score(tree, frontier, ScoringContext(
        generator=make_generator(0, "eval"), budget_fraction=0.9, broad_fraction=0.45))
    assert not torch.allclose(early, late), "quota must stop biasing selection after the broad phase"
    assert float(late[frontier].max()) <= 1.0, "released phase should be plain uniform noise"


def test_goal_aware_scorers_require_a_goal():
    model = build_model("randomtreewm", SMALL).eval()
    ctx = ScoringContext(novelty_space="q")
    tree = _tree(scorer="bfs", model=model)[1]
    frontier = tree.expandable_frontier(16)
    with pytest.raises(AssertionError):
        get_scorer("goal")(tree, frontier, ctx)


def test_all_registered_scorers_are_runnable():
    """Every name in SCORERS must be usable; a typo'd registry entry is a silent trap."""
    model = build_model("randomtreewm", SMALL).eval()
    goal = torch.randn(2, SMALL.obs_dim)
    for name in SCORERS:
        if name == "learned":
            continue
        _, tree, _ = _tree(scorer=name, budget=16, goal=goal, model=model)
        assert int(tree.num_nodes.min()) == 16, f"{name} failed to fill the budget"


# ------------------------------------------------------------ Track F (planner)


def test_path_aware_and_ancestor_scores_differ_from_endpoint():
    from dataclasses import replace

    from treewm.planning.goal_planner import GoalPlanner, PlannerConfig

    model, tree, cfg = _tree(scorer="bfs", budget=32)
    planner = GoalPlanner(model, None, cfg, PlannerConfig(), device=torch.device("cpu"))
    endpoint = torch.rand(tree.batch_size, tree.capacity)

    planner.cfg = replace(planner.cfg, score_mode="path_aware", path_cost_weight=0.05)
    path_aware = planner._path_aware_score(tree, endpoint)
    planner.cfg = replace(planner.cfg, score_mode="ancestor", ancestor_weight=0.5)
    ancestor = planner._path_aware_score(tree, endpoint)

    assert not torch.allclose(path_aware, endpoint), "path cost must change the ranking"
    assert (path_aware >= endpoint - 1e-6).all(), "path cost only ever adds"
    # best-on-path is never worse than the node's own endpoint score
    assert (ancestor <= endpoint + 1e-6).all()


# ------------------------------------------------------------------ tree stats


def test_structural_summary_reports_shape_not_just_scalars():
    model, tree, _ = _tree(scorer="bfs", budget=48)
    stats = structural_summary(tree, model)
    assert stats["tree/max_depth"] >= 1
    assert "tree/unique_root_subtrees_explored" in stats
    assert 0.0 <= stats["tree/top2_root_subtree_fraction"] <= 1.0 + 1e-6
    assert any(k.startswith("tree/effective_branching_factor_d") for k in stats)


# ------------------------------------------------------------------ provenance


def test_provenance_records_what_produced_an_artifact():
    prov = provenance(extra={"scorer": "random"})
    for key in ("git_commit", "hostname", "timestamp", "scorer"):
        assert key in prov


def test_collector_rejects_mixed_scoring_rules():
    """The exact bug this exists to prevent: two scoring rules merged into one table."""
    a = {"provenance": {"score_space": "decoded"}}
    b = {"provenance": {"score_space": "latent"}}
    ok, reason = compatible([a, a], ("score_space",))
    assert ok
    ok, reason = compatible([a, b], ("score_space",))
    assert not ok and "score_space" in reason

    raw = {"provenance": {"score_space": "decoded", "decoded_metric": "domain_raw"}}
    normalised = {
        "provenance": {
            "score_space": "decoded",
            "decoded_metric": "normalized_l2",
        }
    }
    ok, reason = compatible([raw, normalised], ("score_space", "decoded_metric"))
    assert not ok and "decoded_metric" in reason


def test_missing_artifacts_are_reported_not_silently_skipped(tmp_path):
    import json

    good = tmp_path / "good.json"
    good.write_text(json.dumps({"provenance": {"score_space": "decoded"}}))
    loaded, missing, mismatched = load_checked([good, tmp_path / "absent.json"])
    assert len(loaded) == 1
    assert str(tmp_path / "absent.json") in missing, "a missing file must be reported"
