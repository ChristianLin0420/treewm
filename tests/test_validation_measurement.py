"""Regression tests for representative validation and task-metric diagnostics."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from treewm.data.samplers import (
    FixedRepresentativeSampler,
    build_fixed_validation_dataloader,
)
from treewm.evaluation import diagnostics
from scripts.train import fixed_validation_rng


class _IndexDataset(Dataset):
    def __init__(self, size: int) -> None:
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"index": torch.tensor(index, dtype=torch.int64)}


def test_fixed_representative_sampler_is_stable_disjoint_and_spans_every_batch():
    dataset = _IndexDataset(2_400)
    kwargs = dict(batch_size=4, num_batches=3, seed=91, num_replicas=2)
    rank0 = FixedRepresentativeSampler(dataset, rank=0, **kwargs)
    rank1 = FixedRepresentativeSampler(dataset, rank=1, **kwargs)

    assert list(rank0) == list(rank0), "a new iterator must not reshuffle validation"
    assert set(rank0.local_indices.tolist()).isdisjoint(rank1.local_indices.tolist())
    assert set(rank0.local_indices.tolist()) | set(rank1.local_indices.tolist()) == set(
        rank0.global_indices.tolist()
    )
    torch.testing.assert_close(rank0.global_indices, rank1.global_indices, rtol=0, atol=0)

    # Each global batch, including the one reused by diagnostics, covers low and high
    # ranks. The old sequential loader's first batch would contain only ranks 0..7.
    global_batches = rank0.global_indices.view(3, 8).float() / (len(dataset) - 1)
    assert bool((global_batches.min(dim=1).values < 0.20).all())
    assert bool((global_batches.max(dim=1).values > 0.80).all())
    summary = rank0.summary()
    assert summary["anchor_rank_fraction_quantiles"]["q00"] < 0.05
    assert summary["anchor_rank_fraction_quantiles"]["q100"] > 0.95


def test_fixed_validation_loader_repeats_exact_order_without_global_rng_use():
    dataset = _IndexDataset(257)
    torch.manual_seed(1234)
    rng_before = torch.random.get_rng_state().clone()
    loader, sampler = build_fixed_validation_dataloader(
        dataset,
        batch_size=8,
        num_batches=4,
        num_workers=0,
        seed=17,
        generator=torch.Generator().manual_seed(999),
    )
    torch.testing.assert_close(torch.random.get_rng_state(), rng_before, rtol=0, atol=0)

    first = torch.cat([batch["index"] for batch in loader])
    second = torch.cat([batch["index"] for batch in loader])
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    torch.testing.assert_close(first, sampler.local_indices, rtol=0, atol=0)
    assert len(first) == 32


def test_fixed_validation_rng_repeats_measurement_and_restores_training_stream():
    torch.manual_seed(2026)
    state = torch.random.get_rng_state().clone()
    with fixed_validation_rng(7, rank=2):
        first = torch.rand(16)
    torch.testing.assert_close(torch.random.get_rng_state(), state, rtol=0, atol=0)
    with fixed_validation_rng(7, rank=2):
        second = torch.rand(16)
    torch.testing.assert_close(first, second, rtol=0, atol=0)


class _DiagnosticModel:
    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return obs

    def q_of(self, z: torch.Tensor) -> torch.Tensor:
        return z[..., :1]

    def z_control_projection(self, z: torch.Tensor) -> torch.Tensor:
        return z[..., -1:]

    def q_distance(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left - right).square().sum(-1).sqrt()


def test_q_retrieval_uses_metric_endpoint_as_primary_and_labels_full_state_secondary(
    monkeypatch,
):
    metric = torch.full((8, 3, 2), 7.0)
    full = torch.full((8, 3, 9), -4.0)
    calls: list[torch.Tensor] = []

    def fake_precision(embedding, endpoints, valid, distance_fn, k):
        del embedding, valid, distance_fn, k
        calls.append(endpoints)
        return float(len(calls))

    monkeypatch.setattr(diagnostics, "retrieval_precision", fake_precision)
    metrics = diagnostics.q_vs_z_retrieval(
        _DiagnosticModel(),
        {
            "obs": torch.randn(8, 4),
            "fut_metric_endpoint": metric,
            "fut_endpoint": full,
            "fut_valid": torch.ones(8, 3),
        },
        k=2,
    )

    assert all(endpoint is metric for endpoint in calls[:3])
    assert all(endpoint is full for endpoint in calls[3:])
    assert metrics["control/retrieval_precision_q"] == 1.0
    assert metrics["control/full_state_retrieval_precision_q"] == 4.0
    assert metrics["control/retrieval_uses_task_metric_endpoint"] == 1.0


def test_q_retrieval_falls_back_to_full_endpoint_for_legacy_batches(monkeypatch):
    full = torch.randn(8, 3, 5)
    calls: list[torch.Tensor] = []

    def fake_precision(embedding, endpoints, valid, distance_fn, k):
        del embedding, valid, distance_fn, k
        calls.append(endpoints)
        return 0.25

    monkeypatch.setattr(diagnostics, "retrieval_precision", fake_precision)
    metrics = diagnostics.q_vs_z_retrieval(
        _DiagnosticModel(),
        {"obs": torch.randn(8, 4), "fut_endpoint": full, "fut_valid": torch.ones(8, 3)},
        k=2,
    )
    assert len(calls) == 3
    assert all(endpoint is full for endpoint in calls)
    assert metrics["control/retrieval_uses_task_metric_endpoint"] == 0.0
    assert "control/full_state_retrieval_precision_q" not in metrics
