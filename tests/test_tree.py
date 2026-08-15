"""Tree indexing, node-budget enforcement, expansion ordering and matching."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from treewm.utils.rng import make_generator
from treewm.models.baselines import ARMS, build_model, tree_config_for
from treewm.models.treewm import TreeWMConfig
from treewm.tree.expansion import TreeConfig
from treewm.tree.frontier import bfs_score, select_topk
from treewm.tree.matching import MatchingConfig, greedy_match, hungarian_match
from treewm.tree.node import BatchedTree

SMALL = TreeWMConfig(obs_dim=2, action_dim=2, z_dim=32, q_dim=16, hidden_dim=64, num_layers=2)


def test_branch_tensor_shapes():
    model = build_model("treewm", SMALL)
    z = torch.randn(6, SMALL.z_dim)
    out = model.branch(z)
    k = SMALL.branch_factor
    assert out.embedding.shape == (6, k, SMALL.hidden_dim)
    assert out.action.shape == (6, k, SMALL.h_max, SMALL.action_dim)
    assert out.horizon_logits.shape == (6, k, len(SMALL.horizons))
    assert out.keep_logit.shape == (6, k)
    assert out.uncertainty.shape == (6, k)
    assert (out.uncertainty > 0).all(), "sigma must be positive"
    assert torch.allclose(out.mass.sum(-1), torch.ones(6), atol=1e-5), "mass normalises over siblings"

    child = model.predict_children(z)
    assert child["latent"].shape == (6, k, SMALL.z_dim)
    assert child["q"].shape == (6, k, model.controllability.num_scales, SMALL.q_dim)


def test_tree_parent_child_indexing():
    model = build_model("treewm", SMALL).eval()
    cfg = TreeConfig(node_budget=32, expansion_batch_size=2, branch_factor=4, max_depth=8, scorer="bfs")
    with torch.no_grad():
        tree, _ = model.generate(torch.randn(3, SMALL.z_dim), cfg, generator=make_generator(0, 'eval'))

    parent = tree.parent_index
    valid = tree.valid
    assert (parent[:, 0] == -1).all(), "root has no parent"
    for b in range(tree.batch_size):
        for n in range(1, tree.capacity):
            if not valid[b, n]:
                continue
            p = int(parent[b, n])
            assert 0 <= p < n, "children must point at an earlier, existing slot"
            assert bool(valid[b, p]), "parent must be a valid node"
            assert int(tree.depth[b, n]) == int(tree.depth[b, p]) + 1


def test_node_budget_enforced_for_every_arm():
    for budget in (8, 16, 64):
        for arm in ARMS:
            model = build_model(arm, SMALL, k_max=64).eval()
            base = TreeConfig(node_budget=budget, expansion_batch_size=4, max_depth=16, branch_factor=4)
            cfg = tree_config_for(arm, base, model)
            with torch.no_grad():
                tree, _ = model.generate(torch.randn(2, SMALL.z_dim), cfg, generator=make_generator(0, 'eval'))
            counts = tree.num_nodes
            assert int(counts.max()) <= budget, f"{arm} exceeded budget {budget}"
            assert int(tree.valid.sum(1).max()) <= budget
            # With an unbounded frontier every arm should spend the full budget.
            assert int(counts.min()) == budget, f"{arm} underspent budget {budget}: {counts.tolist()}"


def test_shape_of_spend_differs_by_arm():
    """SingleWM is a chain, FlatKWM is depth-1, tree arms are in between."""
    budget = 64
    depths = {}
    for arm in ("singlewm", "flatkwm", "fixedtreewm", "treewm"):
        model = build_model(arm, SMALL, k_max=256).eval()
        cfg = tree_config_for(arm, TreeConfig(node_budget=budget, branch_factor=4, max_depth=16), model)
        with torch.no_grad():
            tree, _ = model.generate(torch.randn(2, SMALL.z_dim), cfg, generator=make_generator(0, 'eval'))
        depths[arm] = int(tree.depth.masked_fill(~tree.valid, 0).max())
    assert depths["singlewm"] == budget - 1, "SingleWM must spend the budget on depth"
    assert depths["flatkwm"] == 1, "FlatKWM must spend the budget on breadth"
    assert 1 < depths["fixedtreewm"] < budget - 1
    assert 1 < depths["treewm"] < budget - 1


def test_best_first_expansion_ordering_is_breadth_first_for_bfs():
    model = build_model("fixedtreewm", SMALL).eval()
    cfg = TreeConfig(node_budget=40, expansion_batch_size=1, branch_factor=4, max_depth=8, scorer="bfs")
    with torch.no_grad():
        tree, _ = model.generate(torch.randn(1, SMALL.z_dim), cfg, generator=make_generator(0, 'eval'))
    expanded = tree.expanded[0] & tree.valid[0]
    depths = tree.depth[0][expanded].tolist()
    # Breadth-first: the sequence of expanded depths is non-decreasing.
    assert depths == sorted(depths), f"bfs expanded out of depth order: {depths}"


@pytest.mark.parametrize("arm", list(ARMS))
def test_every_arm_generates_under_autocast(arm):
    """Regression: tree generation must work in the dtype training actually uses.

    Frontier scorers are a mix -- bfs/random/uncertainty compute in float32 while the
    learned gain head returns bf16 under autocast. When the tree's bookkeeping fields
    followed the root latent's dtype, five of seven arms crashed in scatter_ and only
    the learned scorer survived, so a float32-only test suite saw nothing wrong.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = build_model(arm, SMALL, k_max=32).to(device).eval()
    cfg = tree_config_for(arm, TreeConfig(node_budget=16, branch_factor=4, max_depth=16), model)

    with torch.autocast(device_type=device, dtype=dtype, enabled=(device == "cuda")):
        with torch.no_grad():
            z = model.encode(torch.randn(2, SMALL.obs_dim, device=device))
            tree, _ = model.generate(z, cfg, generator=make_generator(0, 'eval', device))
    assert int(tree.num_nodes.min()) == 16
    assert tree.expansion_gain.dtype == torch.float32
    assert tree.keep_score.dtype == torch.float32


@pytest.mark.parametrize("arm", ["treewm", "fixedtreewm", "singlewm", "flatkwm"])
def test_every_node_carries_an_executable_action_chunk(arm):
    """Regression: no valid non-root node may have an empty action chunk.

    Rejected children (padding from a partially-valid expansion batch) used to land on
    the same destination slot as a genuine child, and the duplicate scatter blanked real
    nodes. The planner then found a zero-length chunk, executed a single primitive
    action, and replanned ~415 times per 500-step episode instead of ~31.
    """
    model = build_model(arm, SMALL, k_max=32).eval()
    cfg = tree_config_for(arm, TreeConfig(node_budget=48, expansion_batch_size=4, branch_factor=4,
                                          max_depth=48), model)
    with torch.no_grad():
        tree, _ = model.generate(torch.randn(3, SMALL.z_dim), cfg, generator=make_generator(0, 'eval'))

    non_root = tree.valid.clone()
    non_root[:, 0] = False
    lengths = tree.action_mask.sum(-1)
    assert (lengths[non_root] > 0).all(), (
        f"{arm}: {int((lengths[non_root] == 0).sum())} valid nodes have an empty action chunk"
    )
    # The mask must agree with the node's own predicted horizon.
    horizons = model.horizons.to(lengths.device)
    expected = horizons[tree.horizon_idx.clamp(0, len(horizons) - 1)].float()
    assert torch.equal(lengths[non_root], expected[non_root]), "mask length must match horizon_idx"


def test_frontier_topk_marks_padding_invalid():
    scores = torch.tensor([[1.0, -1e9, -1e9, 5.0]])
    idx, valid = select_topk(scores, 3)
    assert idx.shape == (1, 3)
    assert valid[0, 0] and valid[0, 1]
    assert not valid[0, 2], "slots with -inf score must be flagged invalid"


def test_hungarian_matching_is_optimal_and_one_to_one():
    cost = torch.tensor([[[0.1, 5.0, 5.0], [5.0, 0.2, 5.0], [5.0, 5.0, 0.3], [9.0, 9.0, 9.0]]])
    valid = torch.ones(1, 3)
    b2m, m2b = hungarian_match(cost, valid)
    assert b2m[0, 0] == 0 and b2m[0, 1] == 1 and b2m[0, 2] == 2
    assert b2m[0, 3] == -1, "surplus branch stays unmatched"
    assert sorted(m2b[0].tolist()) == [0, 1, 2]


def test_matching_respects_invalid_targets():
    cost = torch.rand(2, 4, 5)
    valid = torch.tensor([[1.0, 1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0]])
    for fn in (hungarian_match, greedy_match):
        b2m, m2b = fn(cost, valid)
        assert (m2b[0, 2:] == -1).all(), "invalid modes must never be matched"
        assert (m2b[1, 1:] == -1).all()
        assert int((b2m[0] >= 0).sum()) == 2
        assert int((b2m[1] >= 0).sum()) == 1


def test_greedy_and_hungarian_agree_on_separable_costs():
    cost = torch.tensor([[[0.0, 9.0], [9.0, 0.0]]])
    valid = torch.ones(1, 2)
    h, _ = hungarian_match(cost, valid)
    g, _ = greedy_match(cost, valid)
    assert h.tolist() == g.tolist()


def test_path_to_root_reaches_root():
    model = build_model("fixedtreewm", SMALL).eval()
    cfg = TreeConfig(node_budget=32, expansion_batch_size=2, branch_factor=4, max_depth=8, scorer="bfs")
    with torch.no_grad():
        tree, _ = model.generate(torch.randn(2, SMALL.z_dim), cfg, generator=make_generator(0, 'eval'))
    target = torch.tensor([20, 15])
    chain = tree.path_to_root(target)
    assert (chain[0] == 0).all(), "path must start at the root"
    assert (chain[-1] == target).all(), "path must end at the requested node"
