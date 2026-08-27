"""Latent-scale gauge contracts: geometry, DDP seal, clipping groups and resume."""

from __future__ import annotations

import copy
from dataclasses import replace
import math
from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from scripts.train import (
    gradient_parameter_groups,
    resolve_stage_stop_after,
    split_branch_transformer_parameters,
    validate_latent_gauge_configuration,
    validate_objective_version,
)
from treewm.losses.latent_gauge import (
    LatentGauge,
    centered_rms,
    distributed_centered_rms,
    distributed_centered_rms_reference,
)
from treewm.models.baselines import build_model
from treewm.models.treewm import TreeWMConfig
from treewm.utils.checkpoint import load_checkpoint, save_checkpoint
from treewm.utils.config import loss_config


SMALL = TreeWMConfig(
    obs_dim=2,
    action_dim=1,
    z_dim=8,
    q_dim=4,
    hidden_dim=16,
    encoder_hidden=16,
    num_layers=1,
    num_heads=2,
    branch_factor=2,
    h_max=4,
    horizons=(1, 2, 3, 4),
    scales=(("mixed", 2, 1.0),),
    max_depth=3,
)


def _gauge_config(*overrides: str):
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(
            config_name="base",
            overrides=["experiment=treewm_v2_grounded_gauge_pilot", *overrides],
        )


def test_centered_rms_is_translation_invariant_and_differentiable():
    values = torch.tensor([[-1.0, 1.0], [1.0, 3.0]], requires_grad=True)
    scale = centered_rms(values, epsilon=1.0e-12)
    assert float(scale) == pytest.approx(1.0)
    shifted = centered_rms(values + torch.tensor([100.0, -70.0]), epsilon=1.0e-12)
    torch.testing.assert_close(scale, shifted)
    scale.backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert float(values.grad.abs().sum()) > 0.0


def test_distributed_statistic_uses_stable_two_pass_centering():
    perturbation = torch.tensor(
        [[-0.03125, 0.015625], [0.0234375, -0.0078125], [0.0078125, -0.015625]],
        requires_grad=True,
    )
    values = perturbation + 10_000.0
    expected = centered_rms(values)
    reference = distributed_centered_rms_reference(values)
    current = distributed_centered_rms(values)
    torch.testing.assert_close(reference, expected, rtol=0, atol=0)
    torch.testing.assert_close(current, expected, rtol=0, atol=0)

    gauge = LatentGauge()
    initial_loss, metrics = gauge(
        values,
        values.view(1, 3, 2),
        torch.ones(1, 3),
        step=0,
    )
    assert float(initial_loss) == pytest.approx(0.0, abs=1e-12)
    assert metrics["latent_gauge/min_ratio"] == pytest.approx(1.0, abs=1e-7)


def test_gauge_stays_fp32_under_bf16_autocast_with_finite_gradients():
    root = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], requires_grad=True)
    future = torch.tensor(
        [[[-2.0, 0.0], [2.0, 0.0]]], requires_grad=True
    )
    valid = torch.ones(1, 2)
    gauge = LatentGauge()
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        initial, _ = gauge(root, future, valid, step=0)
        loss, _ = gauge(root * 0.5, future * 0.5, valid, step=1)
    assert initial.dtype == torch.float32
    assert loss.dtype == torch.float32 and bool(torch.isfinite(loss))
    loss.backward()
    assert root.grad is not None and torch.isfinite(root.grad).all()
    assert future.grad is not None and torch.isfinite(future.grad).all()


def test_gauge_is_shrink_only_and_seals_only_at_update_zero():
    root = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    future = torch.tensor([[[-2.0, 0.0], [2.0, 0.0]]])
    valid = torch.ones(1, 2)
    gauge = LatentGauge(epsilon=1.0e-8, min_reference_scale=1.0e-4)
    initial_loss, initial_metrics = gauge(root, future, valid, step=0)
    assert float(initial_loss) == pytest.approx(0.0, abs=1e-12)
    assert initial_metrics["latent_gauge/min_ratio"] == pytest.approx(1.0)
    assert gauge.is_sealed and int(gauge.sealed_update) == 0

    factor = torch.tensor(2.0, requires_grad=True)
    expanded_loss, _ = gauge(root * factor, future * factor, valid, step=1)
    expanded_loss.backward()
    assert float(expanded_loss) == pytest.approx(0.0, abs=1e-12)
    assert float(factor.grad) == pytest.approx(0.0, abs=1e-12)

    factor = torch.tensor(0.5, requires_grad=True)
    shrunken_loss, metrics = gauge(root * factor, future * factor, valid, step=1)
    shrunken_loss.backward()
    assert float(shrunken_loss) == pytest.approx(math.log(2.0) ** 2, rel=1e-6)
    assert metrics["latent_gauge/min_ratio"] == pytest.approx(0.5, rel=1e-6)
    assert float(factor.grad) < 0.0  # gradient descent increases the latent scale

    unsealed = LatentGauge()
    with pytest.raises(ValueError, match="after update zero"):
        unsealed(root, future, valid, step=1)


def test_gauge_prevents_synthetic_scale_collapse_without_forcing_growth():
    base_root = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    base_future = torch.tensor([[[-2.0, 0.0], [2.0, 0.0]]])
    valid = torch.ones(1, 2)

    def optimize(use_gauge: bool) -> float:
        log_scale = torch.nn.Parameter(torch.tensor(0.0))
        optimizer = torch.optim.Adam([log_scale], lr=0.05)
        gauge = LatentGauge() if use_gauge else None
        for step in range(200):
            scale = log_scale.exp()
            collapse_pressure = scale.square()
            loss = collapse_pressure
            if gauge is not None:
                gauge_loss, _ = gauge(
                    base_root * scale,
                    base_future * scale,
                    valid,
                    step=step,
                )
                loss = loss + gauge_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        return float(log_scale.detach().exp())

    collapsed = optimize(False)
    protected = optimize(True)
    assert collapsed < 0.12
    assert protected > 0.5
    assert protected <= 1.05


def _ddp_gauge_worker(rank: int, world_size: int, init_file: str, queue) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    # Root union: x={0,2,4,6}, y=0. Future valid union has unequal rank
    # counts: {(0,2), (4,2), (6,2)}. Both exercise true global statistics.
    root = torch.tensor([[4.0 * rank, 0.0], [4.0 * rank + 2.0, 0.0]])
    future = root.view(1, 2, 2) + torch.tensor([[[0.0, 2.0], [0.0, 2.0]]])
    valid = torch.tensor([[1.0, float(rank)]])

    class _ScaledGauge(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.log_scale = torch.nn.Parameter(torch.tensor(0.0))
            self.gauge = LatentGauge()

        def forward(self, step: int):
            scale = self.log_scale.exp()
            return self.gauge(root * scale, future * scale, valid, step=step)

    model = torch.nn.parallel.DistributedDataParallel(_ScaledGauge())
    initial_loss, metrics = model(0)
    with torch.no_grad():
        model.module.log_scale.fill_(-math.log(2.0))
    shrunken_loss, shrunken_metrics = model(1)
    shrunken_loss.backward()
    queue.put(
        (
            rank,
            float(model.module.gauge.root_reference),
            float(model.module.gauge.future_reference),
            metrics["latent_gauge/min_ratio"],
            float(initial_loss.detach()),
            shrunken_metrics["latent_gauge/min_ratio"],
            float(shrunken_loss.detach()),
            float(model.module.log_scale.grad),
        )
    )
    dist.destroy_process_group()


def test_ddp_current_scale_loss_and_gradient_match_global_population(tmp_path):
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    init_file = str(tmp_path / "gauge-gloo-init")
    processes = [
        ctx.Process(target=_ddp_gauge_worker, args=(rank, 2, init_file, queue))
        for rank in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=90)
    assert all(process.exitcode == 0 for process in processes)
    results = sorted(queue.get(timeout=10) for _ in processes)
    expected_root = math.sqrt(2.5)
    expected_future = math.sqrt(28.0 / 9.0)
    for (
        _,
        root_reference,
        future_reference,
        initial_ratio,
        initial_loss,
        shrunken_ratio,
        shrunken_loss,
        log_scale_grad,
    ) in results:
        assert root_reference == pytest.approx(expected_root, rel=1e-6)
        assert future_reference == pytest.approx(expected_future, rel=1e-6)
        assert initial_ratio == pytest.approx(1.0, rel=1e-6)
        assert initial_loss == pytest.approx(0.0, abs=1e-12)
        assert shrunken_ratio == pytest.approx(0.5, rel=1e-6)
        assert shrunken_loss == pytest.approx(math.log(2.0) ** 2, rel=1e-6)
        # The autograd-aware collective contributes a SUM in backward and DDP's
        # parameter hook averages it, yielding the single global objective gradient.
        assert log_scale_grad == pytest.approx(-2.0 * math.log(2.0), rel=1e-5)
    assert results[0][1:] == pytest.approx(results[1][1:])


def _attach_gauge(model) -> None:
    model.add_module("latent_gauge", LatentGauge())


def test_unsealed_update_zero_checkpoint_state_is_entirely_finite(tmp_path):
    model = build_model("treewm", SMALL)
    _attach_gauge(model)
    assert not model.latent_gauge.is_sealed
    path = save_checkpoint(tmp_path / "update-zero.pt", model=model, step=0)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert all(
        bool(torch.isfinite(value).all()) for value in payload["model"].values()
    )
    assert float(payload["model"]["latent_gauge.root_reference"]) == 0.0
    assert int(payload["model"]["latent_gauge.sealed_update"]) == -1


def _three_group_optimizer(model):
    world, gain = gradient_parameter_groups(model, include_branch_prior=False)
    world_rest, branch = split_branch_transformer_parameters(model, world)
    optimizer = torch.optim.AdamW(
        [
            {"params": world_rest, "name": "world_rest"},
            {"params": branch, "name": "branch_transformer"},
            {"params": gain, "name": "gain"},
        ],
        lr=1.0e-3,
    )
    return optimizer, (world_rest, branch, gain)


def test_branch_group_and_gauge_reference_exactly_resume(tmp_path):
    torch.manual_seed(41)
    model = build_model("treewm", SMALL)
    model.gain_head.set_set_aware(True)
    _attach_gauge(model)
    optimizer, groups = _three_group_optimizer(model)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda update: 1.0 / (update + 1.0)
    )
    assert [group["name"] for group in optimizer.param_groups] == [
        "world_rest",
        "branch_transformer",
        "gain",
    ]
    ids = [{id(parameter) for parameter in group} for group in groups]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert set.union(*ids) == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }

    root = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    future = torch.tensor([[[-2.0, 0.0], [2.0, 0.0]]])
    valid = torch.ones(1, 2)
    model.latent_gauge(root, future, valid, step=0)

    def update(target_model, target_optimizer, target_scheduler):
        target_optimizer.zero_grad(set_to_none=True)
        # Touch every optimizer group deterministically, including the contextual gain.
        loss = sum(
            parameter.float().square().mean()
            for parameter in target_model.parameters()
            if parameter.requires_grad
        )
        loss.backward()
        target_optimizer.step()
        target_scheduler.step()
        return loss.detach()

    update(model, optimizer, scheduler)
    identity = {"run": "gauge-resume", "total_steps": 25_000}
    path = save_checkpoint(
        tmp_path / "latest.pt",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=1,
        extra={"run_identity": identity},
    )
    expected_loss = update(model, optimizer, scheduler)
    expected_lrs = scheduler.get_last_lr()

    torch.manual_seed(999)
    restored = build_model("treewm", SMALL)
    restored.gain_head.set_set_aware(True)
    _attach_gauge(restored)
    restored_optimizer, _ = _three_group_optimizer(restored)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer, lr_lambda=lambda update: 1.0 / (update + 1.0)
    )
    load_checkpoint(
        path,
        restored,
        restored_optimizer,
        restored_scheduler,
        expected_identity=identity,
    )
    assert restored.latent_gauge.is_sealed
    torch.testing.assert_close(
        restored.latent_gauge.root_reference, model.latent_gauge.root_reference
    )
    resumed_loss = update(restored, restored_optimizer, restored_scheduler)
    torch.testing.assert_close(resumed_loss, expected_loss)
    assert restored_scheduler.get_last_lr() == pytest.approx(expected_lrs)
    for expected, actual in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(actual, expected)


def test_gauge_preset_is_bounded_active_and_has_monitor_only_control():
    cfg = _gauge_config()
    assert cfg.objective_version == "treewm_v2_grounded_gauge_pilot_v1"
    assert cfg.train.steps == 25_000
    assert cfg.losses.enabled.latent_gauge is True
    assert cfg.losses.weights.latent_gauge == pytest.approx(1.0)
    assert cfg.train.separate_branch_transformer_grad_clip is False
    assert cfg.train.branch_transformer_grad_clip == pytest.approx(1.0)
    active = loss_config(cfg)
    validate_latent_gauge_configuration(str(cfg.objective_version), active)
    validate_objective_version(str(cfg.objective_version), 25_000)
    assert resolve_stage_stop_after(str(cfg.objective_version), 25_000, "5000") == (
        5_000,
        True,
    )
    assert resolve_stage_stop_after(str(cfg.objective_version), 25_000, "25000") == (
        25_000,
        True,
    )
    with pytest.raises(ValueError, match="one of"):
        resolve_stage_stop_after(str(cfg.objective_version), 25_000, "10000")
    with pytest.raises(ValueError, match="25000-update cap"):
        validate_objective_version(str(cfg.objective_version), 25_001)

    monitor_cfg = _gauge_config(
        "losses.enabled.latent_gauge=false",
        "losses.weights.latent_gauge=0.0",
    )
    monitor = loss_config(monitor_cfg)
    assert not monitor.on("latent_gauge")
    validate_latent_gauge_configuration(str(monitor_cfg.objective_version), monitor)

    invalid = copy.deepcopy(active)
    invalid.weights.latent_gauge = 0.5
    with pytest.raises(ValueError, match="false/0.0 or active true/1.0"):
        validate_latent_gauge_configuration(str(cfg.objective_version), invalid)

    delayed = copy.deepcopy(active)
    delayed.decay["latent_gauge"] = 10_000
    with pytest.raises(ValueError, match="cannot warm up or decay"):
        validate_latent_gauge_configuration(str(cfg.objective_version), delayed)


def test_legacy_config_and_model_state_have_no_gauge_side_effects():
    cfg = _gauge_config("experiment=treewm_v2_grounded_repair_pilot")
    legacy_loss = loss_config(cfg)
    assert not legacy_loss.on("latent_gauge")
    validate_latent_gauge_configuration(str(cfg.objective_version), legacy_loss)
    model = build_model("treewm", replace(SMALL))
    assert not hasattr(model, "latent_gauge")
    assert not any(name.startswith("latent_gauge.") for name in model.state_dict())
