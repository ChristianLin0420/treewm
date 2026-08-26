"""Focused regressions for the v2 controllability and expansion objectives."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.train import (
    TrainingStepModule,
    build_scheduler,
    formal_v2_objective_contract,
    gradients_finite,
    malformed_sha256_names,
    objective_finite,
    preserve_global_rng_state,
    required_formal_provenance_hashes,
)
from treewm.losses.controllability_losses import future_set_distance_loss
from treewm.losses.expansion_losses import (
    frontier_gain_objective,
    novelty_gain_loss,
    tree_expansion_metrics,
)
from treewm.losses.total import (
    LossConfig,
    LossWeights,
    assemble_loss_terms,
    audit_effective_loss_gradients,
)
from treewm.models.baselines import build_model, tree_config_for
from treewm.models.branch_transformer import ExpansionGainHead
from treewm.models.treewm import TreeWMConfig
from treewm.tree.expansion import TreeConfig
from treewm.tree.frontier import ScoringContext, learned_score
from treewm.tree.matching import MatchingConfig
from treewm.tree.node import BatchedTree


def _euclidean_q_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm((a - b).flatten(-2), dim=-1)


def test_expansion_metrics_expose_keep_collapse_budget_fill():
    tree = BatchedTree.initialize(
        root_z=torch.randn(2, 8),
        root_q=torch.randn(2, 1, 4),
        capacity=64,
        h_max=4,
        action_dim=2,
    )
    tree.valid[0, :17] = True  # top-1 depth chain under an all-low KEEP head
    tree.valid[1, :] = True
    tree.num_nodes.copy_(torch.tensor([17, 64]))

    metrics = tree_expansion_metrics(SimpleNamespace(decoder=None), tree, space="z")
    assert metrics["expansion/nodes_generated"] == pytest.approx(40.5)
    assert metrics["expansion/budget_shortfall"] == pytest.approx(23.5)
    assert metrics["expansion/budget_fill_fraction"] == pytest.approx(40.5 / 64.0)


def test_fixed_scheduler_horizon_makes_short_pilot_an_exact_formal_prefix():
    def factors(steps: int, scheduler_total_steps: int | None):
        parameter = torch.nn.Parameter(torch.zeros(()))
        optimizer = torch.optim.AdamW([parameter], lr=3e-4)
        cfg = SimpleNamespace(
            train=SimpleNamespace(
                steps=steps,
                scheduler_total_steps=scheduler_total_steps,
                warmup_steps=1000,
                min_lr_scale=0.1,
            )
        )
        # SimpleNamespace has no ``get``; match OmegaConf's mapping API explicitly.
        cfg.train.get = lambda name: getattr(cfg.train, name)
        return build_scheduler(optimizer, cfg).lr_lambdas[0]

    pilot = factors(5_000, 1_000_000)
    formal = factors(1_000_000, 1_000_000)
    for update in (0, 1, 999, 1_000, 2_000, 4_999, 5_000):
        assert pilot(update) == pytest.approx(formal(update), abs=0.0)
    assert factors(5_000, None)(5_000) != pytest.approx(formal(5_000))


def test_stochastic_validation_context_cannot_change_training_rng_streams():
    def seed_all() -> None:
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)

    def draws():
        return random.random(), float(np.random.random()), torch.rand(5)

    seed_all()
    expected = draws()
    seed_all()
    with preserve_global_rng_state():
        for _ in range(8):
            draws()
            torch.randperm(257)
    actual = draws()
    assert actual[:2] == expected[:2]
    torch.testing.assert_close(actual[2], expected[2], rtol=0, atol=0)


def test_control_rms_target_is_dimension_invariant_bounded_and_detached():
    q = torch.tensor(
        [[[1.0, 0.0]], [[0.0, 1.0]], [[-1.0, 0.0]]], requires_grad=True
    )
    endpoints = torch.tensor(
        [[[0.0], [1.0]], [[2.0], [3.0]], [[8.0], [9.0]]], requires_grad=True
    )
    valid = torch.ones(3, 2)

    loss_1, metrics_1 = future_set_distance_loss(
        q,
        endpoints,
        valid,
        _euclidean_q_distance,
        target_transform="rms_tanh",
        metric_weight=1.0,
        rank_weight=0.5,
    )
    duplicated = endpoints.detach().repeat(1, 1, 7).requires_grad_(True)
    loss_7, metrics_7 = future_set_distance_loss(
        q,
        duplicated,
        valid,
        _euclidean_q_distance,
        target_transform="rms_tanh",
        metric_weight=1.0,
        rank_weight=0.5,
    )

    assert float(loss_1.detach()) == pytest.approx(float(loss_7.detach()), abs=1e-6)
    assert metrics_1["control/raw_rms_p90"] == pytest.approx(
        metrics_7["control/raw_rms_p90"], abs=1e-6
    )
    assert metrics_1["control/target_max"] <= 2.0
    loss_1.backward()
    assert q.grad is not None and torch.isfinite(q.grad).all()
    assert endpoints.grad is None, "task-metric targets must be stop-gradient"


def test_frontier_rank_loss_rewards_order_and_weights_decisions_equally():
    target = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    frontier = torch.ones_like(target, dtype=torch.bool)
    good, _ = frontier_gain_objective(
        target.clone(), target, frontier, rank_weight=1.0, calibration_weight=0.0
    )
    bad, _ = frontier_gain_objective(
        target.flip(-1), target, frontier, rank_weight=1.0, calibration_weight=0.0
    )
    rescaled, _ = frontier_gain_objective(
        target.clone(), target * 19.0, frontier, rank_weight=1.0, calibration_weight=0.0
    )
    assert float(good) < float(bad)
    assert float(good) == pytest.approx(float(rescaled), abs=1e-7)

    predicted = torch.tensor([[1.0, 0.0, 99.0, 99.0], [0.0, 3.0, 2.0, 1.0]])
    targets = torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 2.0, 3.0]])
    masks = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]], dtype=torch.bool)
    combined, _ = frontier_gain_objective(
        predicted, targets, masks, rank_weight=1.0, calibration_weight=0.0
    )
    row_losses = []
    for row in range(2):
        row_loss, _ = frontier_gain_objective(
            predicted[row : row + 1],
            targets[row : row + 1],
            masks[row : row + 1],
            rank_weight=1.0,
            calibration_weight=0.0,
        )
        row_losses.append(row_loss)
    assert float(combined) == pytest.approx(float(torch.stack(row_losses).mean()), abs=1e-7)


def test_frontier_gain_excludes_unrankable_rows_and_credits_prediction_ties():
    predicted = torch.tensor(
        [[91.0, 0.0, 0.0], [37.0, -9.0, 4.0], [0.0, 0.0, 1.0]],
        requires_grad=True,
    )
    target = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 1.00004, 1.00009], [0.0, 0.001, 0.01]]
    )
    frontier = torch.tensor(
        [[1, 0, 0], [1, 1, 1], [1, 1, 1]], dtype=torch.bool
    )
    loss, metrics = frontier_gain_objective(
        predicted,
        target,
        frontier,
        rank_weight=1.0,
        calibration_weight=1.0,
    )
    eligible_loss, _ = frontier_gain_objective(
        predicted[2:],
        target[2:],
        frontier[2:],
        rank_weight=1.0,
        calibration_weight=1.0,
    )

    assert float(loss) == pytest.approx(float(eligible_loss), abs=1e-7)
    assert metrics["decision_count"] == 3
    assert metrics["ranking_decision_count"] == 1
    assert metrics["calibration_decision_count"] == 1
    assert metrics["eligible_decision_fraction"] == pytest.approx(1 / 3)
    assert metrics["total_pair_count"] == 6
    assert metrics["ordered_pair_count"] == 3
    assert metrics["target_tie_fraction"] == pytest.approx(0.5)
    assert metrics["predicted_tie_fraction"] == pytest.approx(1 / 3)
    assert metrics["pairwise_accuracy"] == pytest.approx(5 / 6)
    loss.backward()
    assert torch.equal(predicted.grad[:2], torch.zeros_like(predicted.grad[:2]))


def test_frontier_gain_adaptive_tie_band_is_scale_relative():
    predicted = torch.tensor([[0.0, 0.1, 1.0]])
    frontier = torch.ones_like(predicted, dtype=torch.bool)
    for scale in (1.0, 100.0):
        target = scale * torch.tensor([[0.0, 0.04, 1.0]])
        _, metrics = frontier_gain_objective(predicted, target, frontier)
        assert metrics["ordered_pair_count"] == 2
        assert metrics["target_tie_fraction"] == pytest.approx(1 / 3)
        assert metrics["mean_target_tie_band"] == pytest.approx(0.05 * scale)


def test_frontier_gain_rank_does_not_chain_smooth_adjacent_values_into_one_tie():
    target = torch.linspace(0.0, 1.0, 26).unsqueeze(0)
    frontier = torch.ones_like(target, dtype=torch.bool)
    _, metrics = frontier_gain_objective(target.clone(), target, frontier)

    assert metrics["ordered_pair_count"] == 300
    assert metrics["pair_coverage_fraction"] == pytest.approx(300 / 325)
    assert metrics["pairwise_accuracy"] == pytest.approx(1.0)
    assert metrics["rank_correlation"] == pytest.approx(1.0)


def test_gain_loss_disables_autocast_and_records_deterministic_behavior_mix(monkeypatch):
    class RecordingHead(torch.nn.Module):
        set_aware_enabled = True

        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(1, 1)
            self.autocast_enabled = []

        def forward(self, node_feats, *_args, **_kwargs):
            self.autocast_enabled.append(torch.is_autocast_enabled("cpu"))
            return self.linear(node_feats.float()).squeeze(-1)

    class RecordingModel:
        def __init__(self):
            self.gain_head = RecordingHead()
            self.calls = []

        def generate(self, z, cfg, **_kwargs):
            self.calls.append((cfg.scorer, z[:, 0].tolist()))
            batch = z.shape[0]
            feats = torch.tensor([[-1.0, 0.0, 2.0]]).repeat(batch, 1).unsqueeze(-1)
            snapshot = {
                "feats": feats,
                "context": None,
                "depth": torch.zeros(batch, 3, dtype=torch.long),
                "keep": torch.ones(batch, 3),
                "sigma": torch.ones(batch, 3),
                "valid": torch.ones(batch, 3, dtype=torch.bool),
                "frontier": torch.ones(batch, 3, dtype=torch.bool),
                "target": torch.tensor([[0.1, 0.3, 0.9]]).repeat(batch, 1),
            }
            trace = SimpleNamespace(
                snapshots=[snapshot],
                frontier_novelty_before=[],
                frontier_novelty_after=[],
            )
            return object(), trace

    monkeypatch.setattr(
        "treewm.losses.expansion_losses.tree_expansion_metrics", lambda *_args: {}
    )
    model = RecordingModel()
    z0 = torch.arange(4.0).unsqueeze(-1)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        loss, metrics = novelty_gain_loss(
            model,
            z0,
            TreeConfig(node_budget=3, scorer="learned"),
            rank_weight=1.0,
            calibration_weight=0.0,
            training_scorers=("learned", "novelty_q"),
        )

    assert model.calls == [("learned", [0.0, 2.0]), ("novelty_q", [1.0, 3.0])]
    assert model.gain_head.autocast_enabled == [False, False]
    assert loss.dtype == torch.float32
    assert metrics["expansion/gain_objective_fp32"] == 1.0
    assert metrics["expansion/gain_training_scorer_learned_fraction"] == 0.5
    assert metrics["expansion/gain_training_scorer_novelty_q_fraction"] == 0.5
    assert metrics["expansion/gain_ranking_decision_count"] == 4

    class NoFrontierModel(RecordingModel):
        def generate(self, z, cfg, **kwargs):
            tree, trace = super().generate(z, cfg, **kwargs)
            trace.snapshots[0]["frontier"].zero_()
            return tree, trace

    no_frontier = NoFrontierModel()
    empty_z = z0.clone().requires_grad_(True)
    empty_loss, empty_metrics = novelty_gain_loss(
        no_frontier,
        empty_z,
        TreeConfig(node_budget=3, scorer="learned"),
        rank_weight=1.0,
    )
    assert float(empty_loss) == 0.0
    assert empty_metrics == {}
    empty_loss.backward()
    assert torch.equal(empty_z.grad, torch.zeros_like(empty_z))


def test_learned_frontier_allocation_is_fp32_inside_training_autocast():
    class RecordingHead(torch.nn.Module):
        set_aware_enabled = True

        def __init__(self):
            super().__init__()
            self.autocast_enabled: list[bool] = []
            self.input_dtypes: list[torch.dtype] = []

        def forward(self, node_feats, *_args, **_kwargs):
            self.autocast_enabled.append(torch.is_autocast_enabled("cpu"))
            self.input_dtypes.append(node_feats.dtype)
            return node_feats.sum(-1)

    tree = BatchedTree.initialize(
        torch.zeros(1, 1),
        torch.ones(1, 1, 1),
        capacity=2,
        h_max=4,
        action_dim=1,
    )
    frontier = tree.expandable_frontier()
    head = RecordingHead()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        scores = learned_score(
            tree,
            frontier,
            ScoringContext(gain_head=head, novelty_space="q"),
        )
    assert head.autocast_enabled == [False]
    assert head.input_dtypes == [torch.float32]
    assert scores.dtype == torch.float32


def test_set_gain_excludes_self_and_uses_nonself_set_geometry():
    torch.manual_seed(7)
    head = ExpansionGainHead(feat_dim=1, hidden_dim=16, num_attention_heads=4)
    head.set_set_aware(True)
    query = torch.tensor([[[-1.0], [0.0], [1.0]]], requires_grad=True)
    context = query.detach().clone().requires_grad_(True)
    valid = torch.ones(1, 3, dtype=torch.bool)
    metadata = torch.zeros(1, 3)
    score = head(
        query,
        context,
        metadata.long(),
        metadata,
        metadata,
        context_valid=valid,
        exclude_self=True,
    )
    score[0, 1].backward()
    assert torch.equal(context.grad[0, 1], torch.zeros_like(context.grad[0, 1]))
    assert context.grad[0, [0, 2]].abs().sum() > 0

    # Both sets have mean zero and the same middle query, but nearest-other distance
    # differs. A pooled-context scorer cannot distinguish them; set attention can.
    set_near = torch.tensor([[[-1.0], [0.0], [1.0]]])
    set_far = torch.tensor([[[-4.0], [0.0], [4.0]]])
    score_near = head(
        set_near, set_near, metadata.long(), metadata, metadata,
        context_valid=valid, exclude_self=True,
    )[0, 1]
    score_far = head(
        set_far, set_far, metadata.long(), metadata, metadata,
        context_valid=valid, exclude_self=True,
    )[0, 1]
    assert not torch.allclose(score_near, score_far)

    lone_valid = torch.tensor([[1, 0, 0]], dtype=torch.bool)
    lone = head(
        set_near, set_near, metadata.long(), metadata, metadata,
        context_valid=lone_valid, exclude_self=True,
    )
    assert torch.isfinite(lone).all(), "lone-root attention must not be all-masked"


def test_set_gain_head_can_overfit_synthetic_nearest_neighbor_ranking():
    """The scorer/objective pair must have enough signal to fit its declared target."""
    torch.manual_seed(123)
    batch_size, nodes = 12, 6
    base = torch.tensor([0.0, 0.02, 0.12, 0.4, 1.0, 2.0]).view(nodes, 1)
    features = torch.stack([base[torch.randperm(nodes)] for _ in range(batch_size)])
    target = torch.cdist(features, features)
    target = target.masked_fill(
        torch.eye(nodes, dtype=torch.bool).unsqueeze(0), float("inf")
    ).min(-1).values
    valid = torch.ones(batch_size, nodes, dtype=torch.bool)
    metadata = torch.zeros(batch_size, nodes)
    head = ExpansionGainHead(feat_dim=1, hidden_dim=32, num_attention_heads=4)
    head.set_set_aware(True)
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-3)

    for _ in range(100):
        predicted = head(
            features,
            features,
            metadata.long(),
            metadata,
            metadata,
            context_valid=valid,
            exclude_self=True,
        )
        loss, _ = frontier_gain_objective(
            predicted, target, valid, rank_weight=1.0, calibration_weight=0.0
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        predicted = head(
            features,
            features,
            metadata.long(),
            metadata,
            metadata,
            context_valid=valid,
            exclude_self=True,
        )
        _, metrics = frontier_gain_objective(predicted, target, valid)
    assert metrics["pairwise_accuracy"] > 0.95
    assert metrics["rank_correlation"] > 0.90


def test_v2_gain_loss_reaches_only_contextual_and_enabled_prior_heads():
    cfg = TreeWMConfig(
        obs_dim=2,
        action_dim=2,
        z_dim=16,
        q_dim=8,
        hidden_dim=32,
        num_layers=1,
        scales=(("only", 8, 1.0),),
    )
    model = build_model("treewm", cfg)
    model.gain_head.set_set_aware(True)
    tree_cfg = tree_config_for(
        "treewm", TreeConfig(node_budget=12, branch_factor=4, max_depth=8), model
    )
    loss, metrics = novelty_gain_loss(
        model,
        torch.randn(3, cfg.z_dim),
        tree_cfg,
        space="q",
        rank_weight=1.0,
        calibration_weight=0.0,
        branch_prior_weight=0.25,
        training_scorers=("learned", "novelty_q"),
    )
    loss.backward()
    assert model.gain_head.query_proj.weight.grad is not None
    assert any(parameter.grad is not None for parameter in model.heads.gain_head.parameters())
    assert not any(parameter.requires_grad for parameter in model.gain_head.net.parameters())
    for name, parameter in (
        list(model.gain_head.named_parameters())
        + [(f"branch_prior.{name}", parameter) for name, parameter in model.heads.gain_head.named_parameters()]
    ):
        if parameter.requires_grad:
            assert parameter.grad is not None, f"trainable gain parameter {name} is unreachable"
            assert torch.isfinite(parameter.grad).all(), f"non-finite gain gradient: {name}"
    for module in (
        model.encoder,
        model.branch_transformer,
        model.dynamics,
        model.controllability,
    ):
        assert all(parameter.grad is None for parameter in module.parameters())
    assert metrics["expansion/gain_training_scorer_learned_fraction"] == pytest.approx(2 / 3)
    assert metrics["expansion/gain_training_scorer_novelty_q_fraction"] == pytest.approx(1 / 3)


def test_objective_assembly_and_gain_stride_are_exact(monkeypatch):
    weights = LossWeights(state=2.0, expand=3.0)
    cfg = LossConfig(weights=weights, warmup={"expand": 10})
    state = torch.tensor(5.0, requires_grad=True)
    expansion = torch.tensor(7.0, requires_grad=True)
    terms = assemble_loss_terms({"state": state, "expand": expansion}, cfg, step=5)
    assert float(terms.total) == pytest.approx(2.0 * 5.0 + 3.0 * 0.5 * 7.0)
    assert torch.equal(terms.total, sum(terms.effective.values()))

    model = torch.nn.Linear(1, 1)

    def fake_branch(*_args, **_kwargs):
        raw = {"state": model.weight.sum() * 0 + 2.0}
        branch_terms = assemble_loss_terms(raw, cfg, step=int(_kwargs["step"]))
        return branch_terms.total, {}, {"z": torch.ones(2, 1)}, branch_terms

    def fake_gain(*_args, **_kwargs):
        return model.weight.sum() * 0 + 4.0, {}

    monkeypatch.setattr("scripts.train.compute_branch_losses", fake_branch)
    monkeypatch.setattr("scripts.train.novelty_gain_loss", fake_gain)
    step_module = TrainingStepModule(
        model=model,
        loss_cfg=cfg,
        match_cfg=None,
        gain_tree_cfg=None,
        latent_index=None,
        quantizer=None,
        train_cfg=SimpleNamespace(gain_loss_every=2, gain_batch_size=2),
        model_cfg=SimpleNamespace(novelty_space="q"),
        losses_cfg=SimpleNamespace(gain_target="novelty"),
    )
    batch = {"obs": torch.ones(2, 1)}
    inactive, inactive_metrics, _ = step_module(batch, 1, None)
    active, active_metrics, _ = step_module(batch, 2, None)
    assert float(inactive) == pytest.approx(inactive_metrics["train/loss_total_backward"])
    assert float(active) == pytest.approx(active_metrics["train/loss_total_backward"])
    assert inactive_metrics["train/loss_effective/expand"] == 0.0
    assert active_metrics["train/loss_effective/expand"] == pytest.approx(2.4)
    assert inactive_metrics["train/gain_active"] == 0.0
    assert active_metrics["train/gain_active"] == 1.0


def test_formal_v2_active_step_reaches_every_trainable_parameter():
    torch.manual_seed(11)
    batch_size, modes, h_max = 4, 2, 64
    model_cfg = TreeWMConfig(
        obs_dim=2,
        action_dim=1,
        z_dim=16,
        q_dim=8,
        hidden_dim=32,
        encoder_hidden=32,
        num_layers=1,
        num_heads=4,
        branch_factor=4,
        h_max=h_max,
        horizons=(4, 8, 16, 32, 64),
        scales=(("only", 2, 1.0),),
        max_depth=16,
        use_depth_embedding=False,
    )
    model = build_model("treewm", model_cfg)
    model.gain_head.set_set_aware(True)
    model.tree_signature.requires_grad_(False)
    model.heads.mass_head.requires_grad_(False)
    model.heads.gain_head.requires_grad_(False)
    enabled = dict(LossConfig().enabled)
    enabled["mass"] = False
    loss_cfg = LossConfig(
        weights=LossWeights(mass=0.0),
        enabled=enabled,
        keep_balance=False,
        control_target_transform="rms_tanh",
        control_endpoint_key="fut_metric_endpoint",
        control_allow_endpoint_fallback=False,
        control_require_single_scale=True,
        control_metric_weight=1.0,
        control_rank_weight=1.0,
        detach_world_targets=True,
        bind_negative_margin=0.1,
        gain_set_context=True,
        gain_rank_weight=1.0,
        gain_calibration_weight=0.0,
        gain_branch_prior_weight=0.0,
        control_batch=batch_size,
        recursive_batch=32,
        warmup={},
    )
    tree_cfg = tree_config_for(
        "treewm",
        TreeConfig(
            node_budget=10,
            expansion_batch_size=2,
            branch_factor=4,
            max_depth=16,
            keep_threshold=0.5,
        ),
        model,
    )
    formal_train_cfg = SimpleNamespace(
        gain_loss_every=1,
        gain_lr=3.0e-4,
        gain_weight_decay=0.0,
        gain_training_scorers=("learned", "novelty_q"),
    )
    contract = formal_v2_objective_contract(
        model,
        loss_cfg,
        MatchingConfig(normalization_version="rms_v2", num_horizons=5),
        SimpleNamespace(metric_mode="rms_v2", num_horizons=5),
        tree_cfg,
        separate_gain_clip=True,
        train_cfg=formal_train_cfg,
    )
    assert all(contract.values()), contract
    loss_cfg.future_scale = 0.5
    assert not formal_v2_objective_contract(
        model,
        loss_cfg,
        MatchingConfig(normalization_version="rms_v2", num_horizons=5),
        SimpleNamespace(metric_mode="rms_v2", num_horizons=5),
        tree_cfg,
        separate_gain_clip=True,
        train_cfg=formal_train_cfg,
    )["unit_future_scale"]
    loss_cfg.future_scale = 1.0
    module = TrainingStepModule(
        model=model,
        loss_cfg=loss_cfg,
        match_cfg=MatchingConfig(
            normalization_version="rms_v2", num_horizons=5
        ),
        gain_tree_cfg=tree_cfg,
        latent_index=None,
        quantizer=None,
        train_cfg=SimpleNamespace(gain_loss_every=1, gain_batch_size=batch_size),
        model_cfg=SimpleNamespace(novelty_space="q"),
        losses_cfg=SimpleNamespace(gain_target="novelty"),
    )
    endpoint = torch.randn(batch_size, modes, model_cfg.obs_dim)
    action_mask = torch.zeros(batch_size, modes, h_max)
    action_mask[:, 0, :4] = 1
    action_mask[:, 1, :8] = 1
    batch = {
        "obs": torch.randn(batch_size, model_cfg.obs_dim),
        "fut_actions": torch.randn(batch_size, modes, h_max, model_cfg.action_dim),
        "fut_action_mask": action_mask,
        "fut_endpoint": endpoint,
        "fut_metric_endpoint": endpoint.clone(),
        "fut_horizon_idx": torch.tensor([[0, 1]]).expand(batch_size, modes),
        "fut_horizon_len": torch.tensor([[4.0, 8.0]]).expand(batch_size, modes),
        "fut_valid": torch.ones(batch_size, modes),
        "mode_rep": torch.tensor([[0, 1]]).expand(batch_size, modes),
        "mode_mass": torch.tensor([[0.6, 0.4]]).expand(batch_size, modes),
        "mode_valid": torch.ones(batch_size, modes),
        "num_modes": torch.full((batch_size,), modes),
        "future_diversity": torch.ones(batch_size),
    }
    loss, metrics, _ = module(
        batch, step=10, planner_generator=torch.Generator().manual_seed(3)
    )
    assert float(loss.detach()) == pytest.approx(
        metrics["train/loss_total_backward"], rel=1e-6
    )
    loss.backward()
    unreachable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()))
    ]
    assert unreachable == []


def test_nonfinite_gradient_detection_fails_closed():
    assert objective_finite(torch.tensor(1.0))
    assert not objective_finite(torch.tensor(float("nan")))
    assert not objective_finite(torch.tensor(float("inf")))
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    parameter.grad = torch.tensor(float("nan"))
    assert not gradients_finite([parameter])
    parameter.grad = torch.tensor(float("inf"))
    assert not gradients_finite([parameter])
    parameter.grad = torch.tensor(3.0)
    assert gradients_finite([parameter])


def test_effective_gradient_audit_is_bounded_nondestructive_and_gated():
    torch.manual_seed(5)
    encoder = torch.nn.Linear(3, 4)
    branch = torch.nn.Linear(4, 2)
    value = encoder(torch.randn(6, 3))
    prediction = branch(value)
    raw = {
        "state": prediction.square().mean(),
        "control": value.square().mean(),
        "action": prediction.abs().mean(),
    }
    cfg = LossConfig(weights=LossWeights(state=2.0, control=0.5, action=1.5))
    terms = assemble_loss_terms(raw, cfg, step=100)
    result = audit_effective_loss_gradients(
        terms,
        {"encoder": encoder, "branch": branch},
        max_terms=3,
        max_gradient_share=1.0,
    )
    assert result.active_terms == ("state", "control", "action")
    assert result.passed and 0.0 <= result.max_share <= 1.0
    assert result.metrics["gradient_audit/num_terms"] == 3.0
    for term in result.active_terms:
        assert f"gradient_audit/effective_norm/shared/{term}" in result.metrics
        assert f"gradient_audit/cosine_total/encoder/{term}" in result.metrics
    assert all(parameter.grad is None for parameter in encoder.parameters())
    assert all(parameter.grad is None for parameter in branch.parameters())

    # Rebuild the graph because the default pilot path deliberately releases it.
    value = encoder(torch.randn(6, 3))
    prediction = branch(value)
    gated_terms = assemble_loss_terms(
        {
            "state": prediction.square().mean(),
            "control": value.square().mean(),
            "action": prediction.abs().mean(),
        },
        cfg,
        step=100,
    )
    with pytest.raises(ValueError, match="gradient-share gate failed"):
        audit_effective_loss_gradients(
            gated_terms,
            {"encoder": encoder, "branch": branch},
            max_gradient_share=1e-6,
            fail_on_excess=True,
        )


def test_formal_v2_requires_explicit_calibration_hash():
    digest = "a" * 64
    missing = required_formal_provenance_hashes(
        "treewm_v2_rms_rank_v1",
        protocol_sha256=digest,
        code_sha256=digest,
        runtime_sha256=digest,
        calibration_sha256="",
        future_recipe_sha256="",
    )
    assert malformed_sha256_names(missing) == [
        "TREEWM_CALIBRATION_SHA256",
        "TREEWM_FUTURE_RECIPE_SHA256",
    ]
    valid = required_formal_provenance_hashes(
        "treewm_v2_rms_rank_v1",
        protocol_sha256=digest,
        code_sha256=digest,
        runtime_sha256=digest,
        calibration_sha256="b" * 64,
        future_recipe_sha256="c" * 64,
    )
    assert malformed_sha256_names(valid) == []
    legacy = required_formal_provenance_hashes(
        "treewm_v1",
        protocol_sha256=digest,
        code_sha256=digest,
        runtime_sha256=digest,
        calibration_sha256="",
        future_recipe_sha256="",
    )
    assert "TREEWM_CALIBRATION_SHA256" not in legacy
    assert "TREEWM_FUTURE_RECIPE_SHA256" not in legacy
