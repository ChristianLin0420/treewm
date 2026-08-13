"""Min-novelty target, novelty scorers, and the learned-vs-direct pairing."""

from __future__ import annotations

import pytest
import torch

from treewm.logging.metrics import pearson_correlation, rank_correlation
from treewm.models.baselines import NOVELTY_ARMS, NOVELTY_PAIRS, build_model, tree_config_for
from treewm.models.treewm import TreeWMConfig
from treewm.tree.expansion import TreeConfig
from treewm.tree.novelty import feature_dim, pairwise_min_distance, z_novelty

SMALL = TreeWMConfig(obs_dim=2, action_dim=2, z_dim=32, q_dim=16, hidden_dim=64, num_layers=2)


def test_pairwise_min_distance_excludes_self_and_invalid():
    # Three points on a line at 0, 1, 10; the third is far from both others.
    pts = torch.tensor([[[0.0], [1.0], [10.0]]])
    dist = torch.cdist(pts, pts)
    valid = torch.ones(1, 3)
    nov = pairwise_min_distance(dist, valid)
    assert torch.allclose(nov[0], torch.tensor([1.0, 1.0, 9.0])), nov

    # Masking the middle point out changes its neighbours' novelty.
    valid = torch.tensor([[1.0, 0.0, 1.0]])
    nov = pairwise_min_distance(dist, valid)
    assert torch.allclose(nov[0, 0], torch.tensor(10.0))
    assert torch.allclose(nov[0, 2], torch.tensor(10.0))


def test_lone_node_gets_zero_not_a_huge_constant():
    """A tree with one valid node has no neighbour; the target must stay bounded."""
    dist = torch.zeros(1, 3, 3)
    valid = torch.tensor([[1.0, 0.0, 0.0]])
    nov = pairwise_min_distance(dist, valid)
    assert float(nov[0, 0]) == 0.0, "lone node must not receive a 1e9 regression target"


def test_z_novelty_matches_manual_computation():
    model = build_model("noveltyz", SMALL).eval()
    cfg = tree_config_for("noveltyz", TreeConfig(node_budget=16, branch_factor=4, max_depth=16), model)
    with torch.no_grad():
        tree, _ = model.generate(torch.randn(2, SMALL.z_dim), cfg)
    nov = z_novelty(tree)
    d = torch.cdist(tree.latent.float(), tree.latent.float())
    for b in range(tree.batch_size):
        for n in range(tree.capacity):
            if not tree.valid[b, n]:
                continue
            others = [m for m in range(tree.capacity) if m != n and tree.valid[b, m]]
            if not others:
                continue
            expected = min(float(d[b, n, m]) for m in others)
            assert abs(float(nov[b, n]) - expected) < 1e-4


def test_novelty_scorer_selects_the_most_novel_frontier_node():
    from treewm.tree.frontier import novelty_z_score, ScoringContext, select_topk

    torch.manual_seed(0)  # otherwise the tree depends on suite-wide RNG order
    model = build_model("noveltyz", SMALL).eval()
    cfg = tree_config_for("noveltyz", TreeConfig(node_budget=32, branch_factor=4, max_depth=16), model)
    with torch.no_grad():
        tree, _ = model.generate(torch.randn(1, SMALL.z_dim), cfg)

    frontier = tree.expandable_frontier(cfg.max_depth)
    scores = novelty_z_score(tree, frontier, ScoringContext(novelty_space="z"))
    idx, _ = select_topk(scores, 1)
    nov = z_novelty(tree)[0]
    masked = torch.where(frontier[0], nov, torch.full_like(nov, -1e9))

    # Compare the selected node's novelty to the maximum, not the index: several
    # frontier nodes can tie exactly, and then argmax and top-k may disagree on which
    # index to return while both being correct.
    chosen_novelty = float(masked[int(idx[0, 0])])
    assert chosen_novelty == pytest.approx(float(masked.max()), abs=1e-5), (
        "top-k must pick a node whose novelty equals the frontier maximum"
    )


@pytest.mark.parametrize("arm", list(NOVELTY_ARMS))
def test_novelty_arms_build_with_matching_feature_dim(arm):
    model = build_model(arm, SMALL)
    space = model.cfg.novelty_space
    expected = feature_dim(space, SMALL.z_dim, SMALL.q_dim, model.controllability.num_scales)
    assert model.gain_head.feat_dim == expected, (
        f"{arm}: head consumes {model.gain_head.feat_dim}-d features but its novelty space "
        f"{space!r} produces {expected}-d -- the head would learn the wrong representation"
    )


@pytest.mark.parametrize("direct,learned", list(NOVELTY_PAIRS))
def test_paired_arms_share_a_metric_space(direct, learned):
    """The learned arm must optimise exactly the signal its direct partner acts on."""
    a, b = build_model(direct, SMALL), build_model(learned, SMALL)
    assert a.cfg.novelty_space == b.cfg.novelty_space
    assert a.gain_head.feat_dim == b.gain_head.feat_dim


def test_novelty_gain_loss_trains_only_the_head():
    """World-model training must stay identical across arms.

    If this loss reached the encoder or branch transformer, the learned arms would have
    a different world model than the direct ones and the comparison would no longer
    isolate the allocation policy.
    """
    from treewm.losses.expansion_losses import novelty_gain_loss

    model = build_model("learnedq", SMALL)
    cfg = tree_config_for("learnedq", TreeConfig(node_budget=16, branch_factor=4, max_depth=16), model)
    loss, metrics = novelty_gain_loss(model, torch.randn(4, SMALL.z_dim), cfg, space="q")
    loss.backward()

    assert model.gain_head.net[0].weight.grad is not None, "gain head must receive gradient"
    for name, module in (("encoder", model.encoder), ("branch", model.branch_transformer),
                         ("dynamics", model.dynamics), ("controllability", model.controllability)):
        for pname, p in module.named_parameters():
            assert p.grad is None, f"{name}.{pname} received gradient from the gain loss"
    assert "expansion/gain_rank_correlation" in metrics
    assert "expansion/gain_pearson_correlation" in metrics


def test_correlations_report_both_ordering_and_magnitude():
    # Monotone but non-linear: perfect ordering, imperfect magnitude.
    a = torch.tensor([1.0, 2.0, 3.0, 4.0])
    b = torch.tensor([1.0, 4.0, 9.0, 16.0])
    assert rank_correlation(a, b) == pytest.approx(1.0)
    assert pearson_correlation(a, b) < 1.0
    assert pearson_correlation(a, a) == pytest.approx(1.0)
    assert pearson_correlation(torch.zeros(4), a) == 0.0
