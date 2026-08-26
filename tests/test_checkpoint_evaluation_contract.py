"""Exact-resume checkpoint and protocol-bound evaluation episode contracts."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random

import numpy as np
import pytest
import torch

from treewm.evaluation import rollout
from treewm.utils.checkpoint import (
    CheckpointManager,
    build_checkpoint,
    load_checkpoint,
    save_checkpoint_payload,
    validate_exact_resume_payload,
)
from treewm.utils.seeding import set_rng_state


def _stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _exact_payload(model, optimizer, scheduler, manager: CheckpointManager):
    if not optimizer.state:
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            sum(parameter.square().sum() for parameter in model.parameters()).backward()
            optimizer.step()
            scheduler.step()
    generator_state = torch.Generator().get_state()
    identity = {"run": "exact-unit", "world_size": 1}
    payload = build_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=3,
        epoch=1,
        config={"train": {"steps": 3}},
        extra={
            "completed_updates": 3,
            "next_step": 3,
            "run_identity": identity,
            "identity_sha256": _stable_hash(identity),
            "rank_states": [
                {
                    "rank": 0,
                    "rng_state": {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch": torch.get_rng_state(),
                    },
                    "loader": {
                        "epoch": 1,
                        "batches_yielded_in_epoch": 2,
                        "epoch_generator_state": generator_state.clone(),
                    },
                    "rng_streams": {
                        "planner": generator_state.clone(),
                        "eval": generator_state.clone(),
                        "viz": generator_state.clone(),
                    },
                    "horizon_generator": generator_state.clone(),
                }
            ],
            "checkpoint_manager": manager.state_dict(),
            "normalizer": {},
            "latent_index": None,
            "pending_eval_step": None,
            "final_eval": None,
            "phase": "train",
            "gradient_checkpointing": False,
            "reason": "unit-test-boundary",
        },
    )
    return identity, payload


def test_all_checkpoint_slots_carry_same_complete_post_selection_boundary(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    manager = CheckpointManager(tmp_path)
    assert manager.record_success(0.25)
    assert manager.record_val_loss(1.5)
    identity, payload = _exact_payload(model, optimizer, scheduler, manager)

    manager.save_best_success_payload(payload)
    manager.save_best_val_loss_payload(payload)
    manager.save_latest_payload(payload)

    loaded = []
    for name in ("latest.pt", "best_success.pt", "best_validation_loss.pt"):
        restored = load_checkpoint(
            tmp_path / name,
            restore_rng=False,
            expected_identity=identity,
            expected_world_size=1,
            require_exact_resume=True,
        )
        validate_exact_resume_payload(
            restored, expected_identity=identity, expected_world_size=1
        )
        loaded.append(restored)

    expected_manager = {"best_success": 0.25, "best_val_loss": 1.5}
    assert all(item["checkpoint_manager"] == expected_manager for item in loaded)
    assert all(
        item["rank_states"][0]["loader"]["batches_yielded_in_epoch"] == 2
        for item in loaded
    )
    assert [set(item) for item in loaded] == [set(payload), set(payload), set(payload)]


def test_exact_resume_validation_precedes_any_model_mutation(tmp_path):
    saved_model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        saved_model.weight.fill_(1.0)
        saved_model.bias.fill_(1.0)
    saved_optimizer = torch.optim.AdamW(saved_model.parameters(), lr=1e-3)
    saved_scheduler = torch.optim.lr_scheduler.LambdaLR(
        saved_optimizer, lambda _step: 1.0
    )
    manager = CheckpointManager(tmp_path / "manager")
    identity, invalid = _exact_payload(
        saved_model, saved_optimizer, saved_scheduler, manager
    )
    invalid.pop("rank_states")
    path = save_checkpoint_payload(tmp_path / "incomplete.pt", invalid)

    target = torch.nn.Linear(2, 2)
    with torch.no_grad():
        target.weight.fill_(7.0)
        target.bias.fill_(7.0)
    target_optimizer = torch.optim.AdamW(target.parameters(), lr=3e-3)
    target_scheduler = torch.optim.lr_scheduler.LambdaLR(
        target_optimizer, lambda _step: 0.5
    )
    before_model = copy.deepcopy(target.state_dict())
    before_optimizer = copy.deepcopy(target_optimizer.state_dict())
    before_scheduler = copy.deepcopy(target_scheduler.state_dict())

    with pytest.raises(ValueError, match="lacks exact-resume fields: rank_states"):
        load_checkpoint(
            path,
            target,
            target_optimizer,
            target_scheduler,
            restore_rng=False,
            expected_identity=identity,
            expected_world_size=1,
            require_exact_resume=True,
        )

    for key, value in before_model.items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)
    assert target_optimizer.state_dict() == before_optimizer
    assert target_scheduler.state_dict() == before_scheduler


def test_exact_resume_rejects_late_model_shape_error_before_partial_mutation(tmp_path):
    saved_model = torch.nn.Linear(2, 2)
    saved_optimizer = torch.optim.AdamW(saved_model.parameters(), lr=1e-3)
    saved_scheduler = torch.optim.lr_scheduler.LambdaLR(saved_optimizer, lambda _step: 1.0)
    identity, payload = _exact_payload(
        saved_model,
        saved_optimizer,
        saved_scheduler,
        CheckpointManager(tmp_path / "manager", enabled=False),
    )
    payload["model"]["bias"] = torch.zeros(3)
    path = save_checkpoint_payload(tmp_path / "wrong-shape.pt", payload)

    target = torch.nn.Linear(2, 2)
    with torch.no_grad():
        target.weight.fill_(7.0)
        target.bias.fill_(7.0)
    optimizer = torch.optim.AdamW(target.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    before = copy.deepcopy(target.state_dict())
    with pytest.raises(ValueError, match="model state 'bias' shape differs"):
        load_checkpoint(
            path,
            target,
            optimizer,
            scheduler,
            restore_rng=False,
            expected_identity=identity,
            expected_world_size=1,
            require_exact_resume=True,
        )
    for key, value in before.items():
        torch.testing.assert_close(target.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda payload: payload["rank_states"][0]["rng_state"].pop("numpy"),
            "rank RNG state lacks numpy",
        ),
        (
            lambda payload: payload["rank_states"][0]["loader"].pop(
                "batches_yielded_in_epoch"
            ),
            "rank loader state lacks batches_yielded_in_epoch",
        ),
        (
            lambda payload: payload["rank_states"][0]["loader"].__setitem__(
                "epoch_generator_state", None
            ),
            "epoch-generator state is invalid",
        ),
        (
            lambda payload: payload["rank_states"][0]["rng_streams"].pop("viz"),
            "rank RNG streams lack viz",
        ),
        (
            lambda payload: payload["rank_states"][0].__setitem__(
                "horizon_generator", None
            ),
            "horizon-generator state is invalid",
        ),
        (
            lambda payload: payload["optimizer"]["state"].clear(),
            "optimizer state is empty",
        ),
        (
            lambda payload: payload.__setitem__("scheduler", {}),
            "scheduler state lacks",
        ),
        (
            lambda payload: payload["scheduler"].__setitem__("last_epoch", 2),
            "scheduler last_epoch differs",
        ),
    ],
    ids=(
        "rank-global-rng",
        "loader-offset",
        "loader-generator",
        "named-rng-streams",
        "horizon-generator",
        "optimizer-moments",
        "scheduler-empty",
        "scheduler-position",
    ),
)
def test_exact_resume_rejects_structurally_incomplete_rank_state(
    tmp_path, mutate, match
):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    manager = CheckpointManager(tmp_path, enabled=False)
    _identity, payload = _exact_payload(model, optimizer, scheduler, manager)
    mutate(payload)
    with pytest.raises(ValueError, match=match):
        validate_exact_resume_payload(payload, expected_world_size=1)


def test_exact_resume_allows_unstarted_optimizer_and_loader_at_update_zero(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    manager = CheckpointManager(tmp_path, enabled=False)
    _identity, payload = _exact_payload(model, optimizer, scheduler, manager)
    payload["step"] = payload["completed_updates"] = payload["next_step"] = 0
    payload["optimizer"]["state"].clear()
    payload["scheduler"]["last_epoch"] = 0
    payload["scheduler"]["_step_count"] = 1
    payload["rank_states"][0]["loader"]["epoch_generator_state"] = None
    validate_exact_resume_payload(payload, expected_world_size=1)


def test_exact_resume_can_require_cuda_rng_state_without_loading_a_gpu(tmp_path):
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    _identity, payload = _exact_payload(
        model,
        optimizer,
        scheduler,
        CheckpointManager(tmp_path, enabled=False),
    )
    with pytest.raises(ValueError, match="global RNG state lacks CUDA RNG state"):
        validate_exact_resume_payload(
            payload, expected_world_size=1, require_cuda_rng=True
        )
    cuda_state = [torch.Generator().get_state()]
    payload["rng_state"]["torch_cuda"] = cuda_state
    payload["rank_states"][0]["rng_state"]["torch_cuda"] = cuda_state
    validate_exact_resume_payload(
        payload, expected_world_size=1, require_cuda_rng=True
    )


def test_strict_cuda_rng_restore_requires_presence_and_exact_topology(monkeypatch):
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    restored: list[tuple[int, torch.Tensor]] = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state",
        lambda value, index: restored.append((index, value.clone())),
    )
    with pytest.raises(RuntimeError, match="absent"):
        set_rng_state(copy.deepcopy(state), strict_cuda=True)

    two_device_state = copy.deepcopy(state)
    two_device_state["torch_cuda"] = [torch.zeros(8, dtype=torch.uint8)] * 2
    with pytest.raises(RuntimeError, match="could not be restored exactly"):
        set_rng_state(two_device_state, strict_cuda=True)

    one_device_state = copy.deepcopy(state)
    one_device_state["torch_cuda"] = [torch.arange(8, dtype=torch.uint8)]
    set_rng_state(one_device_state, strict_cuda=True)
    assert len(restored) == 1 and restored[0][0] == 0
    torch.testing.assert_close(restored[0][1], one_device_state["torch_cuda"][0])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_checkpoint_manager_rejects_nonfinite_best_metrics(tmp_path, value):
    manager = CheckpointManager(tmp_path, enabled=False)
    with pytest.raises(ValueError, match="finite"):
        manager.record_success(value)
    with pytest.raises(ValueError, match="finite"):
        manager.record_val_loss(value)
    with pytest.raises(ValueError, match="finite"):
        manager.maybe_save_success(value)
    with pytest.raises(ValueError, match="finite"):
        manager.maybe_save_val_loss(value)


def test_checkpoint_manager_rejects_corrupt_persisted_metrics(tmp_path):
    manager = CheckpointManager(tmp_path, enabled=False)
    with pytest.raises(ValueError, match="best-success"):
        manager.load_state_dict({"best_success": float("nan"), "best_val_loss": 1.0})
    with pytest.raises(ValueError, match="best-validation"):
        manager.load_state_dict({"best_success": 0.5, "best_val_loss": -float("inf")})


def test_protocol_seed_banks_are_paired_across_training_seeds_and_split_disjoint():
    protocol = "a" * 64
    seed_zero = rollout.build_evaluation_seed_tables(protocol, 0, [3, 7], 4, 6)
    seed_one = rollout.build_evaluation_seed_tables(protocol, 1, [3, 7], 4, 6)

    assert seed_zero["training_seed"] == 0
    assert seed_one["training_seed"] == 1
    assert seed_zero["monitor"] == seed_one["monitor"]
    assert seed_zero["final"] == seed_one["final"]
    monitor = {value for row in seed_zero["monitor"]["seeds"] for value in row}
    final = {value for row in seed_zero["final"]["seeds"] for value in row}
    assert monitor.isdisjoint(final)
    assert len(monitor) == 8
    assert len(final) == 12

    different_protocol = rollout.build_evaluation_seed_tables(
        "b" * 64, 0, [3, 7], 4, 6
    )
    assert different_protocol["monitor"]["seeds"] != seed_zero["monitor"]["seeds"]
    assert different_protocol["final"]["seeds"] != seed_zero["final"]["seeds"]

    tampered = copy.deepcopy(seed_zero["monitor"])
    tampered["seeds"][0][0] += 1
    with pytest.raises(ValueError, match="hash differs"):
        rollout.validate_evaluation_seed_table(tampered)


def test_evaluation_records_seed_and_validates_raw_resume_prefix(monkeypatch):
    calls = []

    def fake_episode(_env, _planner, _task, seed, **_kwargs):
        calls.append(seed)
        return rollout.EpisodeResult(
            success=False,
            steps=1,
            replans=1,
            nodes=4,
            final_goal_distance=0.5,
            best_goal_distance=0.25,
        )

    monkeypatch.setattr(rollout, "run_episode", fake_episode)
    tasks = [{"task_id": 3}, {"task_id": 7}]
    table = rollout.build_evaluation_seed_tables("c" * 64, 0, [3, 7], 3, 5)[
        "monitor"
    ]
    expected = [value for row in table["seeds"] for value in row]
    persisted = []
    metrics = rollout.evaluate(
        object(),
        object(),
        tasks,
        episodes_per_task=3,
        episode_seed_table=table,
        expected_episode_seed_split="monitor",
        episode_callback=persisted.append,
    )

    assert calls == expected
    assert [row["episode_seed"] for row in persisted] == expected
    assert metrics["eval/world_model_nodes_per_success"] == 0.0
    assert metrics["eval/world_model_nodes_per_success_defined"] == 0.0
    assert all(math.isfinite(value) for value in metrics.values())

    calls.clear()
    resumed = rollout.evaluate(
        object(),
        object(),
        tasks,
        episodes_per_task=3,
        episode_seed_table=table,
        completed_results=persisted[:4],
    )
    assert calls == expected[4:]
    assert resumed["eval/num_episodes"] == 6

    corrupted = copy.deepcopy(persisted[:4])
    corrupted[-1]["episode_seed"] += 1
    with pytest.raises(ValueError, match="deterministic prefix"):
        rollout.evaluate(
            object(),
            object(),
            tasks,
            episodes_per_task=3,
            episode_seed_table=table,
            completed_results=corrupted,
        )

    with pytest.raises(ValueError, match="split differs"):
        rollout.evaluate(
            object(),
            object(),
            tasks,
            episodes_per_task=3,
            episode_seed_table=table,
            expected_episode_seed_split="final",
        )
