"""Contracts for the opt-in executable-prefix prospective repair."""

from __future__ import annotations

import copy
from pathlib import Path

from hydra import compose, initialize_config_dir
import numpy as np
import pytest
import torch

from scripts.train import (
    validate_executable_prefix_configuration,
    validate_objective_version,
)
from treewm.data.future_sets import FutureSetBuilder, FutureSetConfig
from treewm.data.ogbench_dataset import ChunkDataset, Normalizer, TrajectoryIndex
from treewm.losses.executable_prefix import (
    EXECUTABLE_PREFIX_SCHEMA_VERSION,
    EXECUTABLE_PREFIX_TRAIN_METRIC_PREFIX,
    EXECUTABLE_PREFIX_VALIDATION_METRIC_PREFIX,
    executable_prefix_losses,
)
from treewm.losses.total import (
    LossConfig,
    LossWeights,
    compute_branch_losses,
)
from treewm.models.treewm import TreeWM, TreeWMConfig
from treewm.planning.action_execution import (
    project_normalized_actions,
    uniform_action_bounds,
)
from treewm.planning.goal_planner import GoalPlanner, PlannerConfig
from treewm.tree.matching import MatchingConfig, gather_matched_targets, match
from treewm.utils import config as cfg_utils


OBJECTIVE = "treewm_v2_grounded_executable_prefix_pilot_v1"


def _index(lengths: tuple[int, ...]) -> TrajectoryIndex:
    terminals = np.zeros(sum(lengths), dtype=np.float32)
    cursor = 0
    for length in lengths:
        cursor += length
        terminals[cursor - 1] = 1.0
    return TrajectoryIndex.from_terminals(terminals)


def test_same_continuation_prefix_target_uses_min_four_and_exact_masks():
    observations = np.stack(
        [
            np.arange(16, dtype=np.float32),
            np.arange(16, dtype=np.float32) * 10.0,
            np.arange(16, dtype=np.float32) * -3.0,
        ],
        axis=-1,
    )
    actions = np.arange(32, dtype=np.float32).reshape(16, 2)
    cfg = FutureSetConfig(
        num_neighbors=2,
        query_multiplier=1,
        time_exclusion=0,
        retrieval_radius=1.0,
        include_self=False,
        horizons=(2, 4, 6),
        h_max=6,
        horizon_rule="fixed",
        fixed_horizon=4,
        relative_endpoints=False,
        cluster_threshold=0.01,
        max_modes=2,
        multi_step_depth=1,
        executable_prefix_steps=4,
    )
    builder = FutureSetBuilder(
        observations,
        actions,
        _index((8, 8)),
        cfg,
        xy_dims=(0, 1),
        task_metric_dims=(0, 2),
    )
    builder._neighbors = lambda _: np.asarray([0, 8], dtype=np.int64)
    builder._pick_horizon = lambda continuation, _: 2 if continuation == 0 else 6
    item = builder.build(0)

    assert item["fut_executable_prefix_endpoint"].shape == (2, 3)
    assert item["fut_executable_prefix_metric_endpoint"].shape == (2, 2)
    assert item["fut_executable_prefix_action_mask"].shape == (2, 6)
    np.testing.assert_array_equal(item["fut_executable_prefix_len"], [2.0, 4.0])
    np.testing.assert_array_equal(
        item["fut_executable_prefix_action_mask"].sum(-1), [2.0, 4.0]
    )
    np.testing.assert_array_equal(
        item["fut_executable_prefix_horizon_idx"], [0, 1]
    )
    # Each endpoint and its logged action prefix come from exactly the same c.
    np.testing.assert_array_equal(
        item["fut_executable_prefix_endpoint"][0], observations[2]
    )
    np.testing.assert_array_equal(
        item["fut_executable_prefix_endpoint"][1], observations[12]
    )
    np.testing.assert_array_equal(item["fut_actions"][0, :2], actions[0:2])
    np.testing.assert_array_equal(item["fut_actions"][1, :4], actions[8:12])

    # Relative endpoints reuse the sealed full-endpoint local chart: only the logged
    # displacement c->c+p is transferred to the anchor, while nuisance state remains
    # the actual c+p observation.
    relative_cfg = copy.deepcopy(cfg)
    relative_cfg.relative_endpoints = True
    relative = FutureSetBuilder(
        observations,
        actions,
        _index((8, 8)),
        relative_cfg,
        xy_dims=(0, 1),
        task_metric_dims=(0, 2),
    )
    relative._neighbors = builder._neighbors
    relative._pick_horizon = builder._pick_horizon
    relative_item = relative.build(0)
    np.testing.assert_array_equal(
        relative_item["fut_executable_prefix_endpoint"][1, :2],
        observations[0, :2] + (observations[12, :2] - observations[8, :2]),
    )
    assert (
        relative_item["fut_executable_prefix_endpoint"][1, 2]
        == observations[12, 2]
    )


def test_disabled_builder_preserves_old_keys_values_and_rng_decisions():
    rng = np.random.default_rng(17)
    observations = rng.normal(size=(24, 3)).astype(np.float32)
    actions = rng.normal(size=(24, 2)).astype(np.float32)
    common = dict(
        num_neighbors=2,
        query_multiplier=1,
        time_exclusion=0,
        retrieval_radius=100.0,
        horizons=(4, 8),
        h_max=8,
        horizon_rule="random",
        relative_endpoints=False,
        cluster_threshold=100.0,
        max_modes=2,
        multi_step_depth=1,
    )
    disabled = FutureSetBuilder(
        observations, actions, _index((12, 12)), FutureSetConfig(**common)
    )
    active = FutureSetBuilder(
        observations,
        actions,
        _index((12, 12)),
        FutureSetConfig(**common, executable_prefix_steps=4),
    )
    for builder in (disabled, active):
        builder._neighbors = lambda _: np.asarray([0, 12], dtype=np.int64)
    old = disabled.build(0)
    prospective = active.build(0)
    assert not any("executable_prefix" in key for key in old)
    assert set(prospective) - set(old) == {
        "fut_executable_prefix_endpoint",
        "fut_executable_prefix_metric_endpoint",
        "fut_executable_prefix_action_mask",
        "fut_executable_prefix_horizon_idx",
        "fut_executable_prefix_len",
    }
    for key, value in old.items():
        assert np.array_equal(value, prospective[key]), key


def test_canonical_numpy_torch_projection_matches_historical_planner_math():
    actions = np.asarray([[4.0, -4.0], [0.5, -0.25]], dtype=np.float32)
    mean = np.asarray([0.1, -0.2], dtype=np.float32)
    std = np.asarray([0.5, 0.25], dtype=np.float32)
    lower = np.asarray([-1.0, -1.0], dtype=np.float32)
    upper = np.asarray([1.0, 1.0], dtype=np.float32)
    numpy_projection = project_normalized_actions(
        actions,
        action_mean=mean,
        action_std=std,
        action_lower_bound=lower,
        action_upper_bound=upper,
    )
    expected_raw = actions * std + mean
    expected_applied = np.clip(expected_raw, lower, upper)
    np.testing.assert_array_equal(numpy_projection.raw_env, expected_raw)
    np.testing.assert_array_equal(numpy_projection.applied_env, expected_applied)

    action_tensor = torch.tensor(actions, requires_grad=True)
    torch_projection = project_normalized_actions(
        action_tensor,
        action_mean=torch.tensor(mean),
        action_std=torch.tensor(std),
        action_lower_bound=torch.tensor(lower),
        action_upper_bound=torch.tensor(upper),
    )
    np.testing.assert_allclose(
        torch_projection.raw_env.detach().numpy(), numpy_projection.raw_env, atol=0, rtol=0
    )
    np.testing.assert_allclose(
        torch_projection.applied_env.detach().numpy(),
        numpy_projection.applied_env,
        atol=0,
        rtol=0,
    )
    torch_projection.applied_normalized.sum().backward()
    assert action_tensor.grad is not None
    assert float(action_tensor.grad[0].abs().sum()) == 0.0  # both coordinates clipped
    assert float(action_tensor.grad[1].abs().sum()) > 0.0


class _LinearPrefixModel(torch.nn.Module):
    def __init__(self, width: int = 4) -> None:
        super().__init__()
        self.register_buffer("horizons", torch.tensor([2, 4], dtype=torch.long))
        self.encoder = torch.nn.Linear(width, width, bias=False)
        self.decoder = torch.nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.encoder.weight.copy_(torch.eye(width))
            self.decoder.weight.copy_(torch.eye(width))

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        return self.encoder(observation)

    def dynamics(
        self,
        parent_z: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        horizon_idx: torch.Tensor,
        branch_embedding: torch.Tensor,
    ) -> torch.Tensor:
        del horizon_idx
        action_delta = (actions * action_mask.unsqueeze(-1)).mean(-2)
        padded = torch.nn.functional.pad(
            action_delta, (0, parent_z.shape[-1] - action_delta.shape[-1])
        )
        return parent_z.unsqueeze(1) + padded + 0.1 * branch_embedding


def _loss_inputs(*, onehot: bool = False):
    batch, branches, horizon, action_dim, width = 2, 2, 4, 2, 4
    parent_obs = torch.tensor(
        [[0.0, 0.0, 1.0, 0.0], [1.0, -1.0, 0.0, 1.0]],
        dtype=torch.float32,
    )
    raw_action = torch.tensor(
        [
            [[[3.0, -4.0]] * 4, [[0.5, 0.25]] * 4],
            [[[0.1, -0.2]] * 4, [[-0.3, 0.4]] * 4],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    target_action = torch.zeros_like(raw_action)
    prefix_mask = torch.tensor(
        [[[1, 1, 1, 1], [1, 1, 0, 0]], [[1, 1, 1, 1], [1, 1, 1, 1]]],
        dtype=torch.float32,
    )
    endpoint = parent_obs.unsqueeze(1).expand(batch, branches, width).clone()
    endpoint[:, :, 0] += torch.tensor([[1.0, 9.0], [0.5, -0.5]])
    endpoint[:, :, 1] += torch.tensor([[0.0, 9.0], [0.25, 0.5]])
    dims = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    metric_endpoint = endpoint.index_select(-1, dims[0])
    if onehot:
        dims = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
        metric_endpoint = endpoint.clone()
        subgoals = torch.tensor([[[0, 2], [2, 4]]] * batch)
        kind = torch.ones(batch, dtype=torch.long)
    else:
        subgoals = torch.empty(batch, 0, 2, dtype=torch.long)
        kind = torch.zeros(batch, dtype=torch.long)
    return {
        "parent_obs": parent_obs,
        "raw_predicted_action": raw_action,
        "target_action": target_action,
        "prefix_action_mask": prefix_mask,
        "prefix_horizon_idx": torch.tensor([[1, 0], [1, 1]]),
        "prefix_target_endpoint": endpoint,
        "prefix_target_metric_endpoint": metric_endpoint,
        "prefix_length": prefix_mask.sum(-1),
        "matched": torch.tensor([[1.0, 0.0], [1.0, 1.0]]),
        "task_metric_dims": dims,
        "action_mean": torch.zeros(batch, action_dim),
        "action_std": torch.full((batch, action_dim), 0.5),
        "observation_mean": torch.zeros(batch, width),
        "observation_std": torch.ones(batch, width),
        "task_metric_kind": kind,
        "task_subgoals": subgoals,
        "action_lower_bound": -1.0,
        "action_upper_bound": 1.0,
        "branch_embedding": torch.zeros(batch, branches, width, requires_grad=True),
    }


def _run_loss(model, values):
    return executable_prefix_losses(
        model,
        parent_z=model.encode(values["parent_obs"]),
        parent_obs=values["parent_obs"],
        **{key: value for key, value in values.items() if key != "parent_obs"},
    )


def test_three_components_mask_unmatched_and_raw_action_retains_clipped_gradient():
    model = _LinearPrefixModel()
    values = _loss_inputs()
    values["target_action"] = values["target_action"].detach().requires_grad_(True)
    values["prefix_target_endpoint"] = (
        values["prefix_target_endpoint"].detach().requires_grad_(True)
    )
    values["prefix_target_metric_endpoint"] = (
        values["prefix_target_metric_endpoint"].detach().requires_grad_(True)
    )
    values["raw_predicted_action"].retain_grad()
    result = _run_loss(model, values)
    assert result.action.ndim == result.latent.ndim == result.endpoint.ndim == 0
    assert all(torch.isfinite(value) for value in (result.action, result.latent, result.endpoint))
    total = 0.5 * result.action + 0.25 * result.latent + 0.5 * result.endpoint
    total.backward()
    gradient = values["raw_predicted_action"].grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert float(gradient[0, 0].abs().sum()) > 0.0  # raw MSE pulls clipped actions inward
    assert float(gradient[0, 1].abs().sum()) == 0.0  # unmatched branch is fully masked
    assert values["target_action"].grad is None
    assert values["prefix_target_endpoint"].grad is None
    assert values["prefix_target_metric_endpoint"].grad is None
    assert result.metrics["train/executable_prefix/action_clipped_fraction"] > 0.0
    assert result.metrics["train/executable_prefix/action_finite_fraction"] == 1.0
    assert result.metrics[
        "train/executable_prefix/matched_branches_per_anchor"
    ] == pytest.approx(1.5)
    assert result.metrics[
        "train/executable_prefix/action_scalars_per_anchor"
    ] == pytest.approx(12.0)
    assert result.metrics[
        "train/executable_prefix/action_raw_finite_scalars_per_anchor"
    ] == pytest.approx(12.0)
    assert set(result.artifacts) >= {
        "raw_action_env",
        "applied_action_env",
        "applied_action_normalized",
        "predicted_prefix_endpoint",
        "target_prefix_endpoint",
        "prefix_action_mask",
        "matched",
    }


def test_prefix_loss_rejects_noncontiguous_mask_or_length_horizon_drift():
    model = _LinearPrefixModel()
    noncontiguous = _loss_inputs()
    noncontiguous["prefix_action_mask"][0, 0] = torch.tensor([1, 0, 1, 1])
    with pytest.raises(ValueError, match="contiguous exact prefix"):
        _run_loss(model, noncontiguous)

    wrong_horizon = _loss_inputs()
    wrong_horizon["prefix_horizon_idx"][0, 0] = 0  # h2 for a four-step mask
    with pytest.raises(ValueError, match="exact available length"):
        _run_loss(model, wrong_horizon)


def test_unmatched_padding_may_have_zero_prefix_and_invalid_horizon_index():
    model = _LinearPrefixModel()
    values = _loss_inputs()
    values["prefix_action_mask"][0, 1].zero_()
    values["prefix_length"][0, 1] = 0
    values["prefix_horizon_idx"][0, 1] = -1
    result = _run_loss(model, values)
    assert all(
        torch.isfinite(value)
        for value in (result.action, result.latent, result.endpoint)
    )


def test_hungarian_tie_is_first_index_stable_and_only_matched_branch_gets_gradient():
    branch_to_mode, _ = match(
        torch.zeros(1, 2, 1),
        torch.ones(1, 1),
        type("Cfg", (), {"method": "hungarian"})(),
    )
    assert branch_to_mode.tolist() == [[0, -1]]
    gathered = gather_matched_targets(
        {"value": torch.tensor([[[7.0]]])}, branch_to_mode
    )
    assert gathered["matched"].tolist() == [[1.0, 0.0]]

    model = _LinearPrefixModel()
    values = _loss_inputs()
    values = {
        key: (value[:1].clone() if torch.is_tensor(value) and value.shape[:1] == (2,) else value)
        for key, value in values.items()
    }
    values["matched"] = gathered["matched"]
    values["raw_predicted_action"] = values["raw_predicted_action"].detach().requires_grad_(True)
    values["branch_embedding"] = values["branch_embedding"].detach().requires_grad_(True)
    result = _run_loss(model, values)
    (result.action + result.latent + result.endpoint).backward()
    assert float(values["raw_predicted_action"].grad[0, 0].abs().sum()) > 0.0
    assert float(values["raw_predicted_action"].grad[0, 1].abs().sum()) == 0.0


def test_compute_branch_losses_exposes_three_raw_terms_and_monitor_zero_graph():
    observations = np.stack(
        [np.linspace(0.0, 1.0, 16), np.linspace(1.0, -1.0, 16)], axis=-1
    ).astype(np.float32)
    actions = np.stack(
        [np.linspace(-0.8, 0.8, 16), np.linspace(0.7, -0.7, 16)], axis=-1
    ).astype(np.float32)
    terminals = np.zeros(16, dtype=np.float32)
    terminals[[7, 15]] = 1.0
    normalizer = Normalizer.fit(observations, actions)
    future = FutureSetConfig(
        num_neighbors=1,
        query_multiplier=1,
        time_exclusion=0,
        retrieval_radius=10.0,
        horizons=(4,),
        h_max=4,
        horizon_rule="fixed",
        fixed_horizon=4,
        relative_endpoints=False,
        cluster_threshold=1.0,
        max_modes=1,
        multi_step_depth=1,
        executable_prefix_steps=4,
    )
    dataset = ChunkDataset(
        {"observations": observations, "actions": actions, "terminals": terminals},
        normalizer,
        future,
        xy_dims=(0, 1),
        max_anchors=2,
        seed=0,
        task_metric_dims=(0, 1),
        task_goal_metric="l2",
        task_subgoals=((0, 2),),
    )
    batch = torch.utils.data.default_collate([dataset[0], dataset[1]])
    model = TreeWM(
        TreeWMConfig(
            obs_dim=2,
            action_dim=2,
            z_dim=8,
            q_dim=4,
            hidden_dim=16,
            encoder_hidden=16,
            num_layers=1,
            num_heads=2,
            branch_factor=2,
            h_max=4,
            horizons=(4,),
            scales=(("short", 4, 1.0),),
            max_depth=2,
            dropout=0.0,
        )
    )
    enabled = {name: False for name in LossConfig().enabled}
    enabled.update(
        {
            "executable_prefix_action": True,
            "executable_prefix_latent": True,
            "executable_prefix_endpoint": True,
        }
    )
    config = LossConfig(
        weights=LossWeights(
            executable_prefix_action=0.0,
            executable_prefix_latent=0.0,
            executable_prefix_endpoint=0.0,
        ),
        enabled=enabled,
        executable_action_lower_bound=-1.0,
        executable_action_upper_bound=1.0,
    )
    total, metrics, artifacts, terms = compute_branch_losses(
        model,
        batch,
        config,
        MatchingConfig(num_horizons=1),
        return_loss_terms=True,
    )
    assert tuple(terms.raw) == (
        "executable_prefix_action",
        "executable_prefix_latent",
        "executable_prefix_endpoint",
    )
    assert float(total) == pytest.approx(0.0)
    assert total.requires_grad
    assert all(torch.isfinite(value) for value in terms.raw.values())
    assert "train/executable_prefix/predicted_vs_actual_guard_metric_error" in metrics
    assert artifacts["predicted_prefix_endpoint"].shape[:2] == (2, 2)
    total.backward()


def test_bf16_path_is_fp32_finite_and_onehot_guard_telemetry_is_emitted():
    model = _LinearPrefixModel()
    values = _loss_inputs(onehot=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        result = _run_loss(model, values)
        total = result.action + result.latent + result.endpoint
    assert result.action.dtype == result.latent.dtype == result.endpoint.dtype == torch.float32
    assert torch.isfinite(total)
    total.backward()
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )
    assert result.metrics["train/executable_prefix/goal_metric_onehot"] == 1.0
    assert "train/executable_prefix/predicted_vs_actual_hamming" in result.metrics
    assert "train/executable_prefix/predicted_vs_actual_guard_metric_error" in result.metrics


def test_exact_hamming_telemetry_uses_denormalized_domain_coordinates():
    class FixedEndpointModel(_LinearPrefixModel):
        def dynamics(self, parent_z, actions, action_mask, horizon_idx, branch_embedding):
            del actions, action_mask, horizon_idx, branch_embedding
            endpoint = torch.tensor(
                [-10.0, 1.0, -10.0, 1.0],
                dtype=parent_z.dtype,
                device=parent_z.device,
            )
            return endpoint.view(1, 1, 4).expand(parent_z.shape[0], 2, 4)

    model = FixedEndpointModel()
    values = _loss_inputs(onehot=True)
    values["prefix_target_endpoint"] = torch.tensor(
        [[[-9.0, 0.0, -9.0, 0.0]] * 2] * 2,
        dtype=torch.float32,
    )
    values["prefix_target_metric_endpoint"] = values[
        "prefix_target_endpoint"
    ].clone()
    values["observation_mean"] = torch.tensor(
        [[10.0, 0.0, 10.0, 0.0]] * 2,
        dtype=torch.float32,
    )
    result = _run_loss(model, values)
    # Both normalized blocks choose index 1. In raw coordinates the target chooses
    # index 0 and the prediction index 1, so both categorical subgoals mismatch.
    matched_hamming = result.artifacts["predicted_vs_actual_hamming"][
        values["matched"] > 0
    ]
    assert torch.equal(matched_hamming, torch.full_like(matched_hamming, 2.0))
    guard = result.artifacts["predicted_vs_actual_guard_metric_error"][
        values["matched"] > 0
    ]
    assert torch.equal(torch.floor(guard), matched_hamming)


def test_planner_guard_uses_same_applied_normalized_projection_as_training():
    class FakeModel:
        cfg = type(
            "Cfg",
            (),
            {"horizons": (4, 8), "branch_factor": 2, "action_dim": 2, "h_max": 8},
        )()
        horizons = torch.tensor([4, 8])

        def branch(self, root_z, depth):
            del depth
            action = torch.tensor(
                [[[[4.0, -4.0]] * 8, [[0.25, -0.5]] * 8]],
                dtype=torch.float32,
            )
            return type(
                "Branch", (), {"action": action, "embedding": torch.zeros(1, 2, 1)}
            )()

        def predict_children(self, *args, **kwargs):
            raise AssertionError("sealed planner must not use the legacy raw-action path")

        def dynamics(self, root_z, actions, mask, horizon_idx, embedding):
            del root_z, mask, horizon_idx, embedding
            self.seen_actions = actions.detach().clone()
            return actions[:, :, 0, :1]

        def decoder(self, latent):
            return latent

    normalizer = Normalizer(
        obs_mean=np.zeros(1, dtype=np.float32),
        obs_std=np.ones(1, dtype=np.float32),
        act_mean=np.asarray([0.0, 0.0], dtype=np.float32),
        act_std=np.asarray([0.5, 0.25], dtype=np.float32),
    )
    model = FakeModel()
    planner = object.__new__(GoalPlanner)
    planner.cfg = PlannerConfig(
        decoded_metric="normalized_l2",
        execute_mode="clipped",
        execute_steps=4,
        action_lower_bound=-1.0,
        action_upper_bound=1.0,
    )
    planner.normalizer = normalizer
    planner._sealed_action_bounds = True
    planner.action_lower_bound = np.full(2, -1.0, dtype=np.float32)
    planner.action_upper_bound = np.full(2, 1.0, dtype=np.float32)
    planner.domain = None
    planner.goal_dims = torch.tensor([0])
    planner.obs_mean = None
    planner.obs_std = None
    tree = type(
        "Tree",
        (),
        {
            "root_branch": torch.tensor([[-1, 0, 1]]),
            "valid": torch.tensor([[True, True, True]]),
        },
    )()
    planner._executable_first_edge_scores(
        model, torch.zeros(1, 1), tree, torch.zeros(1, 1)
    )
    raw = model.branch(torch.zeros(1, 1), torch.zeros(1, dtype=torch.long)).action
    lower, upper = uniform_action_bounds(2, -1.0, 1.0, like=raw)
    expected = project_normalized_actions(
        raw,
        action_mean=torch.tensor(normalizer.act_mean),
        action_std=torch.tensor(normalizer.act_std),
        action_lower_bound=lower,
        action_upper_bound=upper,
    ).applied_normalized
    torch.testing.assert_close(model.seen_actions, expected, rtol=0, atol=0)


def test_exact_resume_reproduces_second_update_and_prefix_loss_consumes_no_rng():
    def make_state():
        model = _LinearPrefixModel()
        action = torch.nn.Parameter(_loss_inputs()["raw_predicted_action"].detach())
        optimizer = torch.optim.Adam([*model.parameters(), action], lr=1.0e-3)
        return model, action, optimizer

    def update(model, action, optimizer):
        values = _loss_inputs()
        values["raw_predicted_action"] = action
        before = torch.get_rng_state().clone()
        result = _run_loss(model, values)
        loss = 0.5 * result.action + 0.25 * result.latent + 0.5 * result.endpoint
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        assert torch.equal(before, torch.get_rng_state())
        return loss.detach().clone()

    model_a, action_a, optimizer_a = make_state()
    update(model_a, action_a, optimizer_a)
    checkpoint = {
        "model": copy.deepcopy(model_a.state_dict()),
        "action": action_a.detach().clone(),
        "optimizer": copy.deepcopy(optimizer_a.state_dict()),
    }
    model_b, action_b, optimizer_b = make_state()
    model_b.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        action_b.copy_(checkpoint["action"])
    optimizer_b.load_state_dict(checkpoint["optimizer"])

    loss_a = update(model_a, action_a, optimizer_a)
    loss_b = update(model_b, action_b, optimizer_b)
    torch.testing.assert_close(loss_a, loss_b, rtol=0, atol=0)
    for left, right in zip(model_a.parameters(), model_b.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    torch.testing.assert_close(action_a, action_b, rtol=0, atol=0)


def _composed_config():
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(
            config_name="base",
            overrides=["experiment=treewm_v2_grounded_executable_prefix_pilot_v1"],
        )


def test_disabled_base_config_retains_historical_key_schema_and_defaults():
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="base")
    assert "executable_prefix_steps" not in cfg.future_sets
    assert "executable_action_lower_bound" not in cfg.losses
    assert "executable_prefix_action" not in cfg.losses.weights
    assert "executable_prefix_action" not in cfg.losses.enabled
    assert "action_lower_bound" not in cfg.planner
    assert cfg_utils.future_set_config(cfg).executable_prefix_steps == 0
    assert cfg_utils.loss_config(cfg).weights.executable_prefix_action == 0.0
    assert cfg_utils.planner_config(cfg).action_lower_bound is None


def test_registered_config_is_bounded_explicit_and_fails_closed_on_bounds_or_mixed_weights():
    cfg = _composed_config()
    assert cfg.objective_version == OBJECTIVE
    assert cfg.future_sets.executable_prefix_steps == 4
    assert cfg.planner.action_lower_bound == cfg.losses.executable_action_lower_bound == -1.0
    assert cfg.planner.action_upper_bound == cfg.losses.executable_action_upper_bound == 1.0
    assert (
        cfg.losses.weights.executable_prefix_action,
        cfg.losses.weights.executable_prefix_latent,
        cfg.losses.weights.executable_prefix_endpoint,
    ) == pytest.approx((0.5, 0.25, 0.5))
    validate_objective_version(OBJECTIVE, 25_000)
    with pytest.raises(ValueError, match="bounded diagnostic objective"):
        validate_objective_version(OBJECTIVE, 25_001)

    future = cfg_utils.future_set_config(cfg)
    loss = cfg_utils.loss_config(cfg)
    planner = cfg_utils.planner_config(cfg)
    tree = cfg_utils.tree_config(cfg)
    action_space = type(
        "ActionSpace",
        (),
        {
            "low": np.full(int(cfg.env.action_dim), -1.0, dtype=np.float32),
            "high": np.full(int(cfg.env.action_dim), 1.0, dtype=np.float32),
        },
    )()
    model = type(
        "Model",
        (),
        {
            "decoder": object(),
            "cfg": type(
                "ModelConfig",
                (),
                {
                    "horizons": (4, 8, 16, 32, 64),
                    "h_max": 64,
                    "max_depth": 3,
                    "action_dim": int(cfg.env.action_dim),
                },
            )(),
            "horizons": torch.tensor([4, 8, 16, 32, 64]),
        },
    )()
    validate_executable_prefix_configuration(
        OBJECTIVE,
        loss,
        future,
        planner,
        tree_cfg=tree,
        action_space=action_space,
        model=model,
    )
    with pytest.raises(ValueError, match="restricted to a registered bounded objective"):
        validate_executable_prefix_configuration(
            "treewm_v2_grounded_gauge_pilot_v2",
            loss,
            future,
            planner,
        )

    monitor = copy.deepcopy(loss)
    monitor.weights.executable_prefix_action = 0.0
    monitor.weights.executable_prefix_latent = 0.0
    monitor.weights.executable_prefix_endpoint = 0.0
    validate_executable_prefix_configuration(
        OBJECTIVE,
        monitor,
        future,
        planner,
        tree_cfg=tree,
        action_space=action_space,
        model=model,
    )
    mixed = copy.deepcopy(loss)
    mixed.weights.executable_prefix_latent = 0.0
    with pytest.raises(ValueError, match="all-zero monitor-only or an all-positive"):
        validate_executable_prefix_configuration(
            OBJECTIVE,
            mixed,
            future,
            planner,
            tree_cfg=tree,
            action_space=action_space,
            model=model,
        )
    scheduled = copy.deepcopy(loss)
    scheduled.warmup["executable_prefix_action"] = 100
    with pytest.raises(ValueError, match="cannot warm up or decay"):
        validate_executable_prefix_configuration(
            OBJECTIVE,
            scheduled,
            future,
            planner,
            tree_cfg=tree,
            action_space=action_space,
            model=model,
        )
    wrong_space = copy.deepcopy(action_space)
    wrong_space.high = np.full(int(cfg.env.action_dim), 2.0, dtype=np.float32)
    with pytest.raises(ValueError, match="environment action space"):
        validate_executable_prefix_configuration(
            OBJECTIVE,
            loss,
            future,
            planner,
            tree_cfg=tree,
            action_space=wrong_space,
            model=model,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("score_space", "latent"),
        ("decoded_metric", "normalized_l2"),
        ("execute_mode", "full"),
        ("execute_steps", 8),
        ("require_first_edge_improvement", False),
        ("min_first_edge_improvement", -0.01),
        ("min_first_edge_improvement", float("nan")),
    ],
)
def test_registered_objective_rejects_planner_semantic_drift(field, value):
    cfg = _composed_config()
    future = cfg_utils.future_set_config(cfg)
    loss = cfg_utils.loss_config(cfg)
    planner = cfg_utils.planner_config(cfg)
    setattr(planner, field, value)
    tree = cfg_utils.tree_config(cfg)
    action_dim = int(cfg.env.action_dim)
    action_space = type(
        "ActionSpace",
        (),
        {
            "low": np.full(action_dim, -1.0, dtype=np.float32),
            "high": np.full(action_dim, 1.0, dtype=np.float32),
        },
    )()
    model = type(
        "Model",
        (),
        {
            "decoder": object(),
            "cfg": type(
                "ModelConfig",
                (),
                {
                    "horizons": (4, 8, 16, 32, 64),
                    "h_max": 64,
                    "max_depth": 3,
                    "action_dim": action_dim,
                },
            )(),
            "horizons": torch.tensor([4, 8, 16, 32, 64]),
        },
    )()
    with pytest.raises(ValueError, match="decoded domain_raw planning"):
        validate_executable_prefix_configuration(
            OBJECTIVE,
            loss,
            future,
            planner,
            tree_cfg=tree,
            action_space=action_space,
            model=model,
        )


def test_registered_objective_rejects_horizon_depth_or_action_dimension_drift():
    cfg = _composed_config()
    future = cfg_utils.future_set_config(cfg)
    loss = cfg_utils.loss_config(cfg)
    planner = cfg_utils.planner_config(cfg)
    tree = cfg_utils.tree_config(cfg)
    action_dim = int(cfg.env.action_dim)
    action_space = type(
        "ActionSpace",
        (),
        {
            "low": np.full(action_dim, -1.0, dtype=np.float32),
            "high": np.full(action_dim, 1.0, dtype=np.float32),
        },
    )()

    def model(*, horizons=(4, 8, 16, 32, 64), h_max=64, depth=3, dim=action_dim):
        return type(
            "Model",
            (),
            {
                "decoder": object(),
                "cfg": type(
                    "ModelConfig",
                    (),
                    {
                        "horizons": horizons,
                        "h_max": h_max,
                        "max_depth": depth,
                        "action_dim": dim,
                    },
                )(),
                "horizons": torch.tensor(horizons),
            },
        )()

    future_hmax = copy.deepcopy(future)
    future_hmax.h_max = 128
    with pytest.raises(ValueError, match="formal horizons/h_max"):
        validate_executable_prefix_configuration(
            OBJECTIVE,
            loss,
            future_hmax,
            planner,
            tree_cfg=tree,
            action_space=action_space,
            model=model(),
        )

    with pytest.raises(ValueError, match="sealed model horizons"):
        validate_executable_prefix_configuration(
            OBJECTIVE,
            loss,
            future,
            planner,
            tree_cfg=tree,
            action_space=action_space,
            model=model(horizons=(4, 8, 16, 32, 32)),
        )

    wrong_tree = copy.deepcopy(tree)
    wrong_tree.max_depth = 4
    with pytest.raises(ValueError, match="model/tree depth 3"):
        validate_executable_prefix_configuration(
            OBJECTIVE,
            loss,
            future,
            planner,
            tree_cfg=wrong_tree,
            action_space=action_space,
            model=model(),
        )

    with pytest.raises(ValueError, match="environment action space"):
        validate_executable_prefix_configuration(
            OBJECTIVE,
            loss,
            future,
            planner,
            tree_cfg=tree,
            action_space=action_space,
            model=model(dim=action_dim + 1),
        )


@pytest.mark.parametrize("location", ["loss", "planner"])
def test_nonregistered_objective_rejects_projection_bounds(location):
    cfg = _composed_config()
    loss = cfg_utils.loss_config(cfg)
    planner = cfg_utils.planner_config(cfg)
    future = cfg_utils.future_set_config(cfg)
    for name in (
        "executable_prefix_action",
        "executable_prefix_latent",
        "executable_prefix_endpoint",
    ):
        loss.enabled[name] = False
        setattr(loss.weights, name, 0.0)
    future.executable_prefix_steps = 0
    if location == "loss":
        planner.action_lower_bound = planner.action_upper_bound = None
    else:
        loss.executable_action_lower_bound = loss.executable_action_upper_bound = None
    with pytest.raises(ValueError, match="data/loss/planner fields"):
        validate_executable_prefix_configuration(
            "treewm_v2_grounded_gauge_pilot_v2",
            loss,
            future,
            planner,
        )


def test_metric_schema_maps_train_tags_to_fixed_validation_tags():
    result = _run_loss(_LinearPrefixModel(), _loss_inputs())
    train_tags = set(result.metrics)
    expected_suffixes = {
        "action_raw_env_abs_mean",
        "action_raw_env_rms",
        "action_applied_env_abs_mean",
        "action_applied_env_rms",
        "action_logged_env_abs_mean",
        "action_logged_env_rms",
        "action_clipped_fraction",
        "action_finite_fraction",
        "action_applied_finite_fraction",
        "action_logged_finite_fraction",
        "predicted_vs_actual_normalized_task_rms",
        "predicted_normalized_task_displacement_rms",
        "actual_normalized_task_displacement_rms",
        "predicted_vs_actual_guard_metric_error",
        "predicted_guard_metric_displacement",
        "actual_guard_metric_displacement",
        "prefix_steps_mean",
        "valid_anchor_fraction",
        "matched_branches_per_anchor",
        "action_scalars_per_anchor",
        "action_raw_finite_scalars_per_anchor",
        "action_applied_finite_scalars_per_anchor",
        "action_logged_finite_scalars_per_anchor",
    }
    assert {
        f"train/executable_prefix/{suffix}" for suffix in expected_suffixes
    }.issubset(train_tags)
    fixed_validation_tags = {
        key.replace("train/", "val/")
        for key in train_tags
        if "loss" in key or "executable_prefix/" in key
    }
    assert {
        f"val/executable_prefix/{suffix}" for suffix in expected_suffixes
    }.issubset(fixed_validation_tags)
    assert EXECUTABLE_PREFIX_SCHEMA_VERSION == 1
    assert EXECUTABLE_PREFIX_TRAIN_METRIC_PREFIX == "train/executable_prefix/"
    assert EXECUTABLE_PREFIX_VALIDATION_METRIC_PREFIX == "val/executable_prefix/"
