"""Unit contracts for the v2 data metric and scale-coherent world losses.

These tests intentionally use tiny deterministic fixtures: they validate units,
masking and gradient contracts without network, OGBench, MuJoCo or a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
import torch

from treewm.data.future_sets import (
    FutureSetBuilder,
    FutureSetConfig,
    bounded_uniform_indices,
)
from treewm.data.ogbench_dataset import TrajectoryIndex
from treewm.losses.recursive_losses import (
    _masked_action_distance,
    multi_step_recursive_loss,
)
from treewm.losses.world_losses import (
    bind_loss,
    prediction_metrics,
    recursive_loss,
    state_loss,
    uncertainty_loss,
)


def _index(lengths: tuple[int, ...]) -> TrajectoryIndex:
    terminals = np.zeros(sum(lengths), dtype=np.float32)
    cursor = 0
    for length in lengths:
        cursor += length
        terminals[cursor - 1] = 1.0
    return TrajectoryIndex.from_terminals(terminals)


def _fixed_cfg(**overrides) -> FutureSetConfig:
    values = {
        "num_neighbors": 4,
        "query_multiplier": 3,
        "time_exclusion": 0,
        "retrieval_radius": 1.0,
        "include_self": False,
        "metric_mode": "rms_v2",
        "horizons": (1,),
        "h_max": 1,
        "horizon_rule": "fixed",
        "fixed_horizon": 1,
        "relative_endpoints": False,
        "cluster_threshold": 0.2,
        "max_modes": 4,
        "multi_step_depth": 1,
    }
    values.update(overrides)
    return FutureSetConfig(**values)


def test_full_state_retrieval_rms_is_dimension_replication_invariant():
    # Three two-step trajectories. Candidate 2 is RMS distance sqrt(.5)=.707 from
    # anchor 0; candidate 4 is farther than the .8 threshold.
    obs = np.asarray(
        [[0.0, 0.0], [0.0, 0.0], [0.6, 0.8], [0.6, 0.8], [1.6, 0.0], [1.6, 0.0]],
        dtype=np.float32,
    )
    act = np.zeros((len(obs), 1), dtype=np.float32)
    cfg = _fixed_cfg(retrieval_radius=0.8)
    base = FutureSetBuilder(obs, act, _index((2, 2, 2)), cfg, xy_dims=(0, 1))

    # Replicating every standardized coordinate leaves RMS-L2 exactly unchanged.
    repeated_obs = np.tile(obs, (1, 3))
    repeated = FutureSetBuilder(
        repeated_obs,
        act,
        _index((2, 2, 2)),
        cfg,
        xy_dims=(0, 1),
        task_metric_dims=(0, 1),
    )
    assert base._neighbors(0).tolist() == [2]
    assert repeated._neighbors(0).tolist() == [2]
    assert float(base._last_retrieval_distances[0]) == pytest.approx(
        float(repeated._last_retrieval_distances[0]), rel=1e-6
    )


def test_task_metric_dims_exclude_nuisance_from_modes_and_diversity():
    # The two futures agree in task coordinates 0,1 but disagree enormously in
    # nuisance coordinates 2,3. They must remain one task mode with zero diversity.
    obs = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 50.0, -50.0],
            [0.0, 0.0, 100.0, -100.0],
            [1.0, 0.0, -50.0, 50.0],
        ],
        dtype=np.float32,
    )
    builder = FutureSetBuilder(
        obs,
        np.zeros((4, 1), dtype=np.float32),
        _index((2, 2)),
        _fixed_cfg(num_neighbors=2, cluster_threshold=0.01),
        xy_dims=(2, 3),
        task_metric_dims=(0, 1),
    )
    builder._neighbors = lambda _: np.asarray([0, 2], dtype=np.int64)
    item = builder.build(0)

    assert item["fut_metric_endpoint"].shape == (2, 2)
    np.testing.assert_allclose(item["fut_metric_endpoint"], [[1.0, 0.0], [1.0, 0.0]])
    assert not np.allclose(item["fut_endpoint"][0], item["fut_endpoint"][1])
    assert int(item["modes_raw"]) == 1
    assert float(item["future_diversity"]) == pytest.approx(0.0)


def test_horizon_displacement_uses_task_metric_rms_not_nuisance_dims():
    obs = np.asarray(
        [[0.0, 0.0, 0.0], [0.1, 0.1, 100.0], [1.0, 1.0, -100.0]],
        dtype=np.float32,
    )
    cfg = _fixed_cfg(
        horizons=(1, 2),
        h_max=2,
        horizon_rule="displacement",
        displacement_threshold=0.2,
    )
    builder = FutureSetBuilder(
        obs,
        np.zeros((3, 1), dtype=np.float32),
        _index((3,)),
        cfg,
        xy_dims=(0, 1),
        task_metric_dims=(0, 1),
    )
    # h=1 task RMS is .1 even though nuisance motion is 100; h=2 task RMS is 1.
    assert builder._pick_horizon(0, np.random.default_rng(0)) == 2


def test_retrieval_and_raw_mode_caps_are_reported_before_truncation():
    starts = [0.0, 0.0, 0.0, 0.0, 0.0]
    ends = [0.0, 10.0, 20.0, 30.0, 40.0]
    obs = np.asarray(
        [[start, 0.0] if j % 2 == 0 else [ends[j // 2], 0.0]
         for j, start in enumerate(np.repeat(starts, 2))],
        dtype=np.float32,
    )
    builder = FutureSetBuilder(
        obs,
        np.zeros((10, 1), dtype=np.float32),
        _index((2, 2, 2, 2, 2)),
        _fixed_cfg(num_neighbors=3, max_modes=2, cluster_threshold=0.1),
        xy_dims=(0, 1),
    )
    builder._neighbors = lambda _: np.asarray([0, 2, 4, 6, 8], dtype=np.int64)
    item = builder.build(0)

    assert int(item["retrieval_num_candidates"]) == 5
    assert int(item["retrieval_num_valid"]) == 3
    assert int(item["num_retrieved"]) == 3
    assert bool(item["retrieval_truncated"])
    assert int(item["modes_raw"]) == 3
    assert int(item["modes_retained"]) == 2
    assert int(item["modes_truncated"]) == 1


def test_100m_retrieval_pool_sampling_never_requests_population_permutation(monkeypatch):
    real_default_rng = np.random.default_rng

    class GuardedGenerator:
        def __init__(self, seed):
            self.inner = real_default_rng(seed)

        def integers(self, *args, **kwargs):
            return self.inner.integers(*args, **kwargs)

        def choice(self, source, *args, **kwargs):
            assert not np.isscalar(source), "must not choice(100M, replace=False)"
            return self.inner.choice(source, *args, **kwargs)

    monkeypatch.setattr(np.random, "default_rng", GuardedGenerator)
    first = bounded_uniform_indices(100_000_000, 50_000, seed=17)
    second = bounded_uniform_indices(100_000_000, 50_000, seed=17)
    assert len(first) == 50_000
    assert len(np.unique(first)) == 50_000
    assert first[0] >= 0 and first[-1] < 100_000_000
    np.testing.assert_array_equal(first, second)


def test_100m_retrieval_tree_indexes_pool_before_scaling_full_source():
    class ShapeOnlyObservations:
        shape = (100_000_000, 2)

        def __init__(self):
            self.indexed_rows = 0

        def __len__(self):
            return self.shape[0]

        def __getitem__(self, index):
            assert isinstance(index, np.ndarray), "the bounded pool must be indexed first"
            self.indexed_rows = len(index)
            values = np.empty((len(index), 2), dtype=np.float32)
            values[:, 0] = index % 997
            values[:, 1] = index % 991
            return values

        def __truediv__(self, scale):
            raise AssertionError(f"attempted to materialize the 100M source / {scale}")

    class IndexOnly:
        steps_remaining = np.asarray([1], dtype=np.int64)

    observations = ShapeOnlyObservations()
    builder = FutureSetBuilder(
        observations,
        np.empty((1, 1), dtype=np.float32),
        IndexOnly(),
        _fixed_cfg(retrieval_pool=50_000),
        xy_dims=(0, 1),
    )
    assert builder.tree.n == 50_000
    assert observations.indexed_rows == 50_000


def test_state_loss_is_latent_scale_invariant_and_no_match_safe():
    torch.manual_seed(2)
    target = torch.randn(3, 4, 7)
    pred = target + 0.2 * torch.randn_like(target)
    matched = torch.tensor(
        [[1.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
    )
    base = state_loss(pred, target, matched)
    for factor in (0.1, 10.0):
        torch.testing.assert_close(
            state_loss(pred * factor, target * factor, matched), base, rtol=2e-6, atol=1e-7
        )
    torch.testing.assert_close(
        state_loss(pred + 37.0, target + 37.0, matched), base, rtol=2e-6, atol=1e-7
    )

    pred_no_match = pred.clone().requires_grad_(True)
    zero = state_loss(pred_no_match, target, torch.zeros_like(matched))
    assert torch.isfinite(zero) and float(zero) == 0.0
    zero.backward()
    assert pred_no_match.grad is not None
    assert int(torch.count_nonzero(pred_no_match.grad)) == 0


def test_uncertainty_is_scale_and_unmatched_count_invariant_with_safe_no_match():
    # Two distinct supported targets establish a nondegenerate centered target scale.
    pred = torch.tensor([[[2.0], [0.0]], [[5.0], [0.0]]])
    target = torch.tensor([[[1.0], [0.0]], [[3.0], [0.0]]])
    matched = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    sigma = torch.tensor([[0.25, 0.75], [0.5, 0.75]], requires_grad=True)
    base = uncertainty_loss(sigma, pred, target, matched)
    scaled = uncertainty_loss(sigma, pred * 10.0, target * 10.0, matched)
    torch.testing.assert_close(scaled, base)
    translated = uncertainty_loss(sigma, pred - 123.0, target - 123.0, matched)
    torch.testing.assert_close(translated, base)

    # Replicating the unsupported group does not change its group weight.
    pred_many = torch.cat([pred[:, :1], pred[:, 1:].expand(-1, 5, -1)], dim=1)
    target_many = torch.cat([target[:, :1], target[:, 1:].expand(-1, 5, -1)], dim=1)
    matched_many = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
    )
    sigma_many = torch.tensor(
        [[0.25, 0.75, 0.75, 0.75, 0.75, 0.75],
         [0.5, 0.75, 0.75, 0.75, 0.75, 0.75]]
    )
    torch.testing.assert_close(
        uncertainty_loss(sigma_many, pred_many, target_many, matched_many), base
    )

    no_match_sigma = torch.randn(2, 3, requires_grad=True)
    no_match = uncertainty_loss(
        no_match_sigma, torch.randn(2, 3, 4), torch.randn(2, 3, 4), torch.zeros(2, 3)
    )
    assert torch.isfinite(no_match) and float(no_match) == 0.0
    no_match.backward()
    assert int(torch.count_nonzero(no_match_sigma.grad)) == 0


class _EquivariantDynamics:
    def dynamics(self, z, action, mask, horizon_idx, embedding):
        del action, mask, horizon_idx
        return z.unsqueeze(1) + embedding[..., : z.shape[-1]]


class _ActionDynamics:
    def __init__(self, use_action: bool):
        self.use_action = use_action

    def dynamics(self, z, action, mask, horizon_idx, embedding):
        del mask, horizon_idx, embedding
        if not self.use_action:
            return z.unsqueeze(1).expand(-1, action.shape[1], -1)
        return z.unsqueeze(1) + action[..., 0, : z.shape[-1]]


def test_bind_loss_is_scale_invariant_and_swap_margin_detects_action_ignoring():
    z = torch.randn(2, 3)
    embedding = torch.randn(2, 2, 3)
    target = z.unsqueeze(1) + embedding
    action = torch.randn(2, 2, 2, 1)
    mask = torch.ones(2, 2, 2)
    horizon = torch.zeros(2, 2, dtype=torch.long)
    matched = torch.ones(2, 2)
    base = bind_loss(
        _EquivariantDynamics(), z, embedding, action, mask, horizon, target, matched
    )
    scaled = bind_loss(
        _EquivariantDynamics(), z * 10, embedding * 10, action, mask, horizon,
        target * 10, matched,
    )
    torch.testing.assert_close(scaled, base, atol=1e-7, rtol=1e-6)
    translated = bind_loss(
        _EquivariantDynamics(), z + 19.0, embedding, action, mask, horizon,
        target + 19.0, matched,
    )
    torch.testing.assert_close(translated, base, atol=1e-7, rtol=1e-6)

    z1 = torch.zeros(1, 1)
    actions = torch.tensor([[[[0.0]], [[1.0]]]])
    target1 = torch.tensor([[[0.0], [1.0]]])
    args = (
        z1,
        torch.zeros(1, 2, 1),
        actions,
        torch.ones(1, 2, 1),
        torch.zeros(1, 2, dtype=torch.long),
        target1,
        torch.ones(1, 2),
    )
    good, good_metrics = bind_loss(
        _ActionDynamics(True), *args, bind_negative_margin=0.5, return_metrics=True
    )
    ignored, ignored_metrics = bind_loss(
        _ActionDynamics(False), *args, bind_negative_margin=0.5, return_metrics=True
    )
    assert float(good) == pytest.approx(0.0)
    assert float(ignored) > float(good)
    assert good_metrics["bind/eligible_anchors"] == 1.0
    assert good_metrics["bind/achieved_margin"] >= 0.5
    assert ignored_metrics["bind/negative_margin_loss"] == pytest.approx(0.5)

    no_match, no_match_metrics = bind_loss(
        _ActionDynamics(False),
        *args[:-1],
        torch.zeros_like(args[-1]),
        bind_negative_margin=0.5,
        return_metrics=True,
    )
    assert torch.isfinite(no_match) and float(no_match) == 0.0
    assert no_match_metrics["bind/eligible_anchors"] == 0.0
    assert no_match_metrics["bind/negative_margin_loss"] == 0.0


@dataclass
class _BranchResult:
    embedding: torch.Tensor
    action: torch.Tensor
    horizon_logits: torch.Tensor

    def horizon_index(self):
        return self.horizon_logits.argmax(-1)


class _RecursiveModel:
    def __init__(self):
        self.batch_sizes: list[int] = []

    def predict_children(self, z, depth):
        del depth
        self.batch_sizes.append(len(z))
        n = len(z)
        # Pred/target differ only in padded action tail when z is 0/1.
        tail = torch.tensor([0.0, 0.0, 1.0, 1.0], device=z.device)
        action = z[:, :1].view(n, 1, 1, 1) * tail.view(1, 1, 4, 1)
        action = action.expand(n, 2, 4, 1)
        branch = _BranchResult(
            embedding=torch.zeros(n, 2, 3, device=z.device),
            action=action,
            horizon_logits=torch.zeros(n, 2, 1, device=z.device),
        )
        return {
            # Isolate action-tail masking in this fixture: recursive latent predictions
            # agree even though the parent fixture values differ.
            "latent": torch.zeros(n, 2, 3, device=z.device),
            "branch": branch,
            "action_mask": torch.tensor(
                [1.0, 1.0, 0.0, 0.0], device=z.device
            ).view(1, 1, 4).expand(n, 2, 4),
        }


def test_recursive_samples_matched_nodes_directly_and_masks_action_tails():
    model = _RecursiveModel()
    pred = torch.zeros(1, 4, 1, requires_grad=True)
    target = torch.ones(1, 4, 1)
    matched = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    loss, metrics = recursive_loss(
        model, pred, target, matched, max_nodes=1, return_metrics=True
    )
    assert float(loss) == pytest.approx(0.0)
    assert model.batch_sizes == [1, 1]
    assert metrics["recursive/matched_nodes"] == 1.0
    assert metrics["recursive/sampled_nodes"] == 1.0
    assert metrics["recursive/latent_component"] == pytest.approx(0.0)
    assert metrics["recursive/action_component"] == pytest.approx(0.0)

    empty_model = _RecursiveModel()
    empty, empty_metrics = recursive_loss(
        empty_model, pred, target, torch.zeros_like(matched), return_metrics=True
    )
    assert float(empty) == 0.0
    assert empty_model.batch_sizes == []
    assert empty_metrics["recursive/matched_nodes"] == 0.0


class _AffineRecursiveModel:
    def predict_children(self, z, depth):
        del depth
        n = len(z)
        latent = z.unsqueeze(1).expand(n, 2, z.shape[-1])
        branch = _BranchResult(
            embedding=torch.zeros(n, 2, 2, device=z.device),
            action=torch.zeros(n, 2, 2, 1, device=z.device),
            horizon_logits=torch.zeros(n, 2, 1, device=z.device),
        )
        return {
            "latent": latent,
            "branch": branch,
            "action_mask": torch.ones(n, 2, 2, device=z.device),
        }


def test_recursive_child_latent_loss_is_scale_and_translation_invariant():
    pred = torch.tensor([[[0.0, 1.0], [2.0, -1.0]], [[1.0, 3.0], [-2.0, 2.0]]])
    target = torch.tensor([[[1.0, 2.0], [3.0, 0.0]], [[2.0, 5.0], [-1.0, 4.0]]])
    matched = torch.ones(2, 2)
    model = _AffineRecursiveModel()
    base = recursive_loss(model, pred, target, matched)
    torch.testing.assert_close(
        recursive_loss(model, pred * 10.0, target * 10.0, matched), base
    )
    torch.testing.assert_close(
        recursive_loss(model, pred - 71.0, target - 71.0, matched), base
    )


def test_masked_multistep_action_distance_ignores_padding_and_action_width():
    target = torch.zeros(1, 4, 1)
    predicted = torch.zeros(1, 2, 4, 1)
    predicted[:, 0, 2:] = 1000.0
    predicted[:, 1, :2] = 1.0
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    distance = _masked_action_distance(predicted, target, mask)
    torch.testing.assert_close(distance, torch.tensor([[0.0, 1.0]]))

    repeated = _masked_action_distance(
        predicted.expand(-1, -1, -1, 7), target.expand(-1, -1, 7), mask
    )
    torch.testing.assert_close(repeated, distance)


class _MultiStepModel:
    def __init__(self):
        self.depths: list[torch.Tensor] = []

    def encode(self, obs):
        return obs

    def branch(self, z, depth):
        self.depths.append(depth.detach().clone())
        b = len(z)
        return _BranchResult(
            embedding=torch.zeros(b, 1, 2, device=z.device),
            action=torch.zeros(b, 1, 4, 1, device=z.device),
            horizon_logits=torch.zeros(b, 1, 1, device=z.device),
        )

    def dynamics(self, z, action, mask, horizon_idx, embedding):
        del action, mask, horizon_idx, embedding
        return z.unsqueeze(1)


def _multistep_batch(scale: float = 1.0, translation: float = 0.0):
    obs = torch.tensor([[1.0, 2.0], [2.0, -1.0]]) * scale + translation
    ms_obs = torch.tensor(
        [[[2.0, 3.0], [3.0, 5.0], [999.0, 999.0]],
         [[4.0, -2.0], [999.0, 999.0], [999.0, 999.0]]]
    ) * scale + translation
    actions = torch.zeros(2, 3, 4, 1)
    actions[..., 2:, :] = 1000.0  # masked padding must not affect branch choice
    return {
        "obs": obs,
        "ms_actions": actions,
        "ms_action_mask": torch.tensor(
            [[[1.0, 1.0, 0.0, 0.0]] * 3, [[1.0, 1.0, 0.0, 0.0]] * 3]
        ),
        "ms_obs": ms_obs,
        "ms_horizon_idx": torch.zeros(2, 3, dtype=torch.long),
        "ms_valid": torch.tensor([[1.0, 1.0, 0.0], [1.0, 0.0, 0.0]]),
    }


def test_multistep_is_scale_invariant_masks_validity_and_uses_parent_depth():
    model = _MultiStepModel()
    loss, metrics = multi_step_recursive_loss(model, _multistep_batch())
    scaled, _ = multi_step_recursive_loss(_MultiStepModel(), _multistep_batch(scale=10.0))
    torch.testing.assert_close(scaled, loss, rtol=2e-6, atol=1e-7)
    translated, _ = multi_step_recursive_loss(
        _MultiStepModel(), _multistep_batch(translation=55.0)
    )
    torch.testing.assert_close(translated, loss, rtol=2e-6, atol=1e-7)
    assert metrics["recursive/valid_transitions"] == 3.0
    observed_depths = {int(depth[0]) for depth in model.depths}
    assert observed_depths == {0, 1}

    changed_invalid = _multistep_batch()
    changed_invalid["ms_obs"][changed_invalid["ms_valid"] == 0] = -1e9
    changed, _ = multi_step_recursive_loss(_MultiStepModel(), changed_invalid)
    torch.testing.assert_close(changed, loss)


def test_prediction_endpoint_consistency_gathers_last_valid_timestep():
    pred_z = torch.zeros(1, 2, 1)
    target_z = torch.zeros_like(pred_z)
    pred_action = torch.zeros(1, 2, 5, 1)
    target_action = torch.zeros_like(pred_action)
    target_mask = torch.tensor([[[1.0, 1.0, 0.0, 0.0, 0.0], [0.0] * 5]])
    # The eligible branch disagrees at its true endpoint t=1. Garbage at padded t=4
    # must not hide that error; the zero-horizon branch is excluded from the count.
    pred_action[0, 0, 1, 0] = 3.0
    pred_action[0, 0, 4, 0] = 999.0
    target_action[0, 0, 4, 0] = 999.0
    metrics = prediction_metrics(
        pred_z,
        target_z,
        pred_action,
        target_action,
        target_mask,
        torch.zeros(1, 2, 1),
        torch.zeros(1, 2, dtype=torch.long),
        torch.ones(1, 2),
        torch.tensor([2]),
    )
    assert metrics["model/action_endpoint_consistency"] == pytest.approx(9.0)
    assert metrics["model/action_endpoint_consistency_count"] == 1.0
