"""Unit contracts for v2 matching, support calibration, and child admission."""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from treewm.losses.support_losses import (
    coverage_loss,
    keep_loss,
    redundancy_loss,
    support_metrics,
)
from treewm.tree.matching import (
    LARGE_COST,
    MatchingConfig,
    assigned_cost_metrics,
    branch_mode_cost,
)
from treewm.tree.node import BatchedTree


def _flat_cdist(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.cdist(left.float().flatten(2), right.float().flatten(2))


def _matching_inputs(action_dim: int = 2) -> dict[str, torch.Tensor]:
    pred_z = torch.tensor([[[-1.5, 0.5], [1.0, -0.5]]])
    tgt_z = torch.tensor([[[-2.0, 0.0], [2.0, 0.0]]])
    pred_q = torch.zeros(1, 2, 1, 2)
    tgt_q = torch.zeros(1, 2, 1, 2)
    pred_action = torch.zeros(1, 2, 4, action_dim)
    tgt_action = torch.ones(1, 2, 4, action_dim)
    # Garbage in the padded tail must not affect action matching.
    tgt_action[:, :, 2:] = 1000.0
    target_mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0]]]).expand(1, 2, 4).clone()
    return {
        "pred_z": pred_z,
        "pred_q": pred_q,
        "pred_action": pred_action,
        "pred_horizon_idx": torch.tensor([[0, 4]]),
        "tgt_z": tgt_z,
        "tgt_q": tgt_q,
        "tgt_action": tgt_action,
        "tgt_action_mask": target_mask,
        "tgt_horizon_idx": torch.tensor([[4, 0]]),
        "tgt_valid": torch.ones(1, 2),
    }


def test_rms_v2_matching_is_scale_translation_and_action_width_invariant():
    cfg = MatchingConfig(normalization_version="rms_v2", num_horizons=5)
    inputs = _matching_inputs(action_dim=2)
    cost, components = branch_mode_cost(
        **inputs, cfg=cfg, q_cdist=_flat_cdist, return_components=True
    )

    transformed = dict(inputs)
    transformed["pred_z"] = inputs["pred_z"] * 7.0 + 31.0
    transformed["tgt_z"] = inputs["tgt_z"] * 7.0 + 31.0
    cost_scaled, components_scaled = branch_mode_cost(
        **transformed, cfg=cfg, q_cdist=_flat_cdist, return_components=True
    )
    torch.testing.assert_close(components_scaled["z"], components["z"], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(cost_scaled, cost, atol=2e-6, rtol=2e-6)
    assert components_scaled["latent_target_centered_rms"] == pytest.approx(
        7.0 * float(components["latent_target_centered_rms"]), rel=1e-6
    )

    wide = _matching_inputs(action_dim=8)
    _, wide_components = branch_mode_cost(
        **wide, cfg=cfg, q_cdist=_flat_cdist, return_components=True
    )
    torch.testing.assert_close(wide_components["action"], components["action"])
    assert torch.allclose(components["action"], torch.ones_like(components["action"]), atol=2e-6)
    assert float(components["horizon"][0, 0, 0]) == pytest.approx(1.0)


def test_matching_components_mask_invalid_targets_and_report_assigned_units():
    cfg = MatchingConfig(normalization_version="rms_v2", num_horizons=5)
    inputs = _matching_inputs()
    inputs["tgt_valid"] = torch.tensor([[1.0, 0.0]])
    cost, components = branch_mode_cost(
        **inputs, cfg=cfg, q_cdist=_flat_cdist, return_components=True
    )
    assert torch.all(cost[:, :, 1] == LARGE_COST)
    for name in ("z", "q", "action", "horizon"):
        assert torch.all(components[name][:, :, 1] == LARGE_COST)

    branch_to_mode = torch.tensor([[0, -1]])
    matched = torch.tensor([[1.0, 0.0]])
    metrics = assigned_cost_metrics(components, branch_to_mode, matched, cfg)
    assert metrics["matching/assigned_total_cost"] == pytest.approx(
        sum(metrics[f"matching/assigned_{name}_weighted_cost"] for name in ("z", "q", "action", "horizon"))
    )
    assert sum(
        metrics[f"matching/assigned_{name}_weighted_share"]
        for name in ("z", "q", "action", "horizon")
    ) == pytest.approx(1.0)
    assert metrics["matching/latent_target_centered_rms"] >= 1e-3

    empty = assigned_cost_metrics(
        components, torch.full((1, 2), -1), torch.zeros(1, 2), cfg
    )
    assert empty["matching/assigned_total_cost"] == 0.0
    assert all(math.isfinite(value) for value in empty.values())


def test_coverage_is_detached_one_to_one_and_has_explicit_v2_units():
    # Both modes coincide. One branch can cover the first for free, but one-to-one
    # assignment forces the second branch (distance 2 -> normalised q cost 1).
    pred = torch.tensor([[[[0.0]], [[2.0]]]], requires_grad=True)
    modes = torch.tensor([[[[0.0]], [[0.0]]]], requires_grad=True)
    valid = torch.ones(1, 2)
    loss = coverage_loss(pred, modes, valid, _flat_cdist, "rms_v2", space="q")
    assert float(loss.detach()) == pytest.approx(0.5)
    loss.backward()
    assert pred.grad is not None and float(pred.grad.abs().sum()) > 0
    assert modes.grad is None

    # Relative z RMS uses target scale, so global changes of latent units and origin do
    # not alter the objective.
    z_modes = torch.tensor([[[[-2.0, 0.0]], [[2.0, 0.0]]]])
    z_pred = z_modes + torch.tensor([[[[0.25, -0.25]], [[-0.5, 0.5]]]])
    base = coverage_loss(z_pred, z_modes, valid, _flat_cdist, "rms_v2", space="z")
    changed = coverage_loss(
        z_pred * 9.0 + 17.0,
        z_modes * 9.0 + 17.0,
        valid,
        _flat_cdist,
        "rms_v2",
        space="z",
    )
    torch.testing.assert_close(changed, base, atol=2e-6, rtol=2e-6)


def test_coverage_reuses_canonical_joint_assignment_instead_of_q_rematching():
    # Q-only matching would choose identity at zero cost. The canonical joint matcher
    # (which also sees z/action/horizon) chose the swap, so coverage must follow that
    # same supervision rather than contradicting the other branch losses.
    pred = torch.tensor([[[[0.0]], [[2.0]]]], requires_grad=True)
    modes = torch.tensor([[[[0.0]], [[2.0]]]])
    valid = torch.ones(1, 2)
    canonical = torch.tensor([[1, 0]])
    rematched = coverage_loss(
        pred, modes, valid, _flat_cdist, "rms_v2", space="q"
    )
    assigned = coverage_loss(
        pred,
        modes,
        valid,
        _flat_cdist,
        "rms_v2",
        space="q",
        branch_to_mode=canonical,
    )
    assert float(rematched.detach()) == pytest.approx(0.0)
    assert float(assigned.detach()) == pytest.approx(1.0)
    assigned.backward()
    assert pred.grad is not None and float(pred.grad.abs().sum()) > 0

    with pytest.raises(ValueError, match="invalid mode"):
        coverage_loss(
            pred.detach(),
            modes,
            torch.tensor([[1.0, 0.0]]),
            _flat_cdist,
            "rms_v2",
            branch_to_mode=canonical,
        )


def test_redundancy_uses_only_detached_matched_pairs_and_is_k_invariant():
    logits = torch.full((2, 3), -20.0, requires_grad=True)
    keep = torch.sigmoid(logits)
    q = torch.zeros(2, 3, 1, 2, requires_grad=True)
    matched = torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    loss = redundancy_loss(q, keep, _flat_cdist, matched=matched)
    assert float(loss.detach()) == pytest.approx(1.0)
    loss.backward()
    assert logits.grad is None, "redundancy must not train the KEEP head"

    # Adding arbitrary inactive branches cannot dilute or change the eligible pair.
    q_wide = torch.cat([q.detach(), torch.full((2, 4, 1, 2), 100.0)], dim=1)
    keep_wide = torch.cat([keep.detach(), torch.ones(2, 4)], dim=1)
    matched_wide = torch.cat([matched, torch.zeros(2, 4)], dim=1)
    wide = redundancy_loss(q_wide, keep_wide, _flat_cdist, matched=matched_wide)
    assert float(wide) == pytest.approx(float(loss.detach()))


def _support_inputs():
    keep = torch.tensor([[0.9, 0.2, 0.9, 0.8]])
    matched = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    target_mass = torch.tensor([[0.80, 0.15, 0.05, 0.0]])
    branch_target_mass = torch.tensor([[0.80, 0.15, 0.05, 0.0]])
    mode_valid = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
    mode_to_branch = torch.tensor([[0, 1, 2, -1]])
    pred_q = torch.tensor([[[[0.0]], [[0.5]], [[1.0]], [[1.5]]]])
    pred_z = pred_q.squeeze(-1)
    mass_pred = torch.tensor([[0.70, 0.10, 0.10, 0.10]])
    return (
        keep,
        mass_pred,
        matched,
        target_mass,
        branch_target_mass,
        mode_valid,
        mode_to_branch,
        pred_q,
        pred_z,
        _flat_cdist,
    )


def test_keep_calibration_and_keep_aware_support_telemetry():
    logits = torch.tensor([[2.0, -1.0, 0.0]])
    targets = torch.tensor([[1.0, 0.0, 1.0]])
    torch.testing.assert_close(
        keep_loss(logits, targets, balance="standard"),
        F.binary_cross_entropy_with_logits(logits, targets),
    )

    metrics = support_metrics(*_support_inputs())
    assert metrics["tree/assignment_capacity_recall"] == 1.0
    assert metrics["tree/support_recall"] == pytest.approx(2 / 3)
    assert metrics["tree/mode_keep_recall"] == pytest.approx(2 / 3)
    assert metrics["tree/branch_keep_precision"] == pytest.approx(2 / 3)
    assert metrics["tree/branch_keep_recall"] == pytest.approx(2 / 3)
    assert metrics["tree/rare_mode_recall"] == 1.0
    assert metrics["tree/common_mode_recall"] == pytest.approx(0.5)
    assert metrics["tree/num_modes"] == 3.0
    assert metrics["tree/num_rare_modes"] == 1.0
    assert metrics["tree/num_common_modes"] == 2.0
    assert metrics["tree/num_multimode_anchors"] == 1.0
    assert metrics["tree/keep_brier"] == pytest.approx(
        float(((torch.tensor([[0.9, 0.2, 0.9, 0.8]]) - torch.tensor([[1.0, 1.0, 1.0, 0.0]])) ** 2).mean())
    )


def test_rare_mode_recall_requires_keep_and_zero_denominator_is_reported():
    args = list(_support_inputs())
    # Mode 2 is valid, rare, and assigned, but its branch is pruned by KEEP.
    args[0] = torch.tensor([[0.9, 0.9, 0.1, 0.1]])
    missed = support_metrics(*args)
    assert missed["tree/assignment_capacity_recall"] == 1.0
    assert missed["tree/rare_mode_recall"] == 0.0
    assert missed["tree/num_rare_modes"] == 1.0

    # Exact-threshold mass is common, so this batch has no rare denominator. The metric
    # stays finite at zero and the explicit count makes that zero unambiguous.
    args[3] = torch.tensor([[0.70, 0.20, 0.10, 0.0]])
    args[4] = args[3].clone()
    no_rare = support_metrics(*args)
    assert no_rare["tree/rare_mode_recall"] == 0.0
    assert no_rare["tree/num_rare_modes"] == 0.0


def test_mass_metrics_are_finite_calibrated_and_expose_uncovered_target():
    metrics = support_metrics(*_support_inputs())
    assert metrics["tree/mass_ce"] == pytest.approx(
        metrics["tree/mass_target_entropy"] + metrics["tree/mass_kl"], rel=1e-6
    )
    for name in ("mass_ce", "mass_target_entropy", "mass_kl", "mass_tv", "mass_brier"):
        assert math.isfinite(metrics[f"tree/{name}"])
    assert metrics["tree/mass_uncovered_target"] == 0.0

    args = list(_support_inputs())
    args[2] = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    args[4] = torch.tensor([[0.80, 0.15, 0.0, 0.0]])
    args[6] = torch.tensor([[0, 1, -1, -1]])
    uncovered = support_metrics(*args)
    assert uncovered["tree/mass_uncovered_target"] == pytest.approx(0.05)


def test_disabled_mass_head_emits_no_random_calibration_telemetry():
    metrics = support_metrics(*_support_inputs(), mass_enabled=False)
    assert "tree/mass_ce" not in metrics
    assert "tree/mass_tv" not in metrics
    assert "stochastic/mass_calibration_error" not in metrics
    assert "stochastic/support_frequency_decoupling" not in metrics
    assert metrics["tree/support_recall"] == pytest.approx(2 / 3)


def test_support_and_redundancy_no_match_paths_are_finite_zeros():
    keep = torch.tensor([[0.3, 0.3]], requires_grad=True)
    q = torch.zeros(1, 2, 1, 1, requires_grad=True)
    no_pairs = redundancy_loss(
        q, keep, _flat_cdist, matched=torch.zeros(1, 2)
    )
    assert float(no_pairs.detach()) == 0.0
    no_pairs.backward()
    assert keep.grad is None

    metrics = support_metrics(
        keep.detach(),
        torch.tensor([[0.5, 0.5]]),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.full((1, 2), -1),
        q.detach(),
        q.detach().squeeze(-2),
        _flat_cdist,
    )
    assert metrics["tree/num_modes"] == 0.0
    assert metrics["tree/support_recall"] == 0.0
    assert metrics["tree/mass_uncovered_target"] == 0.0
    assert all(math.isfinite(value) for value in metrics.values())


def _children(batch: int, parents: int, scores: torch.Tensor) -> dict[str, torch.Tensor]:
    k = scores.shape[-1]
    return {
        "latent": torch.zeros(batch, parents, k, 2),
        "q": torch.zeros(batch, parents, k, 1, 2),
        "action_chunk": torch.ones(batch, parents, k, 2, 1),
        "action_mask": torch.ones(batch, parents, k, 2),
        "horizon_idx": torch.zeros(batch, parents, k, dtype=torch.long),
        "keep_score": scores,
        "mass": torch.full((batch, parents, k), 1.0 / k),
        "uncertainty": torch.zeros(batch, parents, k),
        "expansion_gain": torch.zeros(batch, parents, k),
    }


def test_keep_threshold_admission_and_per_parent_top1_fallback():
    tree = BatchedTree.initialize(
        root_z=torch.zeros(1, 2),
        root_q=torch.zeros(1, 1, 2),
        capacity=8,
        h_max=2,
        action_dim=1,
    )
    scores = torch.tensor([[[0.9, 0.5, 0.2], [0.4, 0.3, 0.1]]])
    admitted = tree.add_children(
        parent_idx=torch.tensor([[0, 0]]),
        child=_children(1, 2, scores),
        budget=8,
        step=1,
        child_valid=torch.ones(1, 2, 3),
        parent_valid=torch.ones(1, 2, dtype=torch.bool),
        keep_threshold=0.5,
    )
    assert admitted.view(1, 2, 3).tolist() == [[[True, True, False], [True, False, False]]]
    assert int(tree.num_nodes) == 4
    assert sorted(tree.keep_score[tree.valid].tolist()) == pytest.approx([0.0, 0.4, 0.5, 0.9])

    # A padded/invalid selected parent must not receive the top-1 fallback.
    padded_tree = BatchedTree.initialize(
        root_z=torch.zeros(2, 2),
        root_q=torch.zeros(2, 1, 2),
        capacity=4,
        h_max=2,
        action_dim=1,
    )
    low = torch.tensor([[[0.4, 0.3]], [[0.4, 0.3]]])
    padded = padded_tree.add_children(
        parent_idx=torch.zeros(2, 1, dtype=torch.long),
        child=_children(2, 1, low),
        budget=4,
        step=1,
        child_valid=torch.ones(2, 1, 2),
        parent_valid=torch.tensor([[True], [False]]),
        keep_threshold=0.5,
    )
    assert padded.tolist() == [[True, False], [False, False]]
