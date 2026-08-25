"""Network/GPU-free tests for the exact full-shard TreeWM data adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from treewm.data.future_sets import FutureSetConfig
from treewm.data.ogbench_dataset import ChunkDataset, Normalizer, uniform_anchor_ranks
from treewm.data.sharded_ogbench import (
    ShardedDatasetError,
    build_or_load_sharded_cache,
    discover_shards,
)


def _write_raw(path: Path, trajectories: list[np.ndarray], offset: float) -> None:
    """Write OGBench raw layout: one excluded terminal-state row per trajectory."""
    observations = []
    actions = []
    terminals = []
    for trajectory in trajectories:
        trajectory = np.asarray(trajectory, dtype=np.float32) + offset
        observations.append(trajectory)
        actions.append(np.full((len(trajectory), 2), offset, dtype=np.float32))
        marker = np.zeros(len(trajectory), dtype=bool)
        marker[-1] = True
        terminals.append(marker)
    np.savez(
        path,
        observations=np.concatenate(observations),
        actions=np.concatenate(actions),
        terminals=np.concatenate(terminals),
    )


def _make_release(root: Path) -> tuple[Path, np.ndarray]:
    source = root / "toy-100m-v0"
    source.mkdir()
    train_kept = []
    for shard in range(2):
        # Raw lengths 5 and 4 -> four and three transitions. Values make it easy to
        # recompute the train-only normalizer independently.
        trajectories = [
            np.arange(15, dtype=np.float32).reshape(5, 3),
            np.arange(12, dtype=np.float32).reshape(4, 3) + 20,
        ]
        _write_raw(source / f"toy-v0-{shard:03d}.npz", trajectories, shard * 100)
        for trajectory in trajectories:
            train_kept.append(trajectory[:-1] + shard * 100)
        # Validation is deliberately far away; it must not enter train statistics.
        _write_raw(
            source / f"toy-v0-{shard:03d}-val.npz",
            [np.arange(15, dtype=np.float32).reshape(5, 3)],
            10_000 + shard * 100,
        )
    return source, np.concatenate(train_kept)


def test_full_sharded_cache_preserves_all_transitions_and_boundaries(tmp_path: Path):
    source, expected_train = _make_release(tmp_path)
    cache = build_or_load_sharded_cache(
        "toy-100m-v0", source, cache_root=tmp_path / "cache", expected_shards=2
    )

    assert cache.dataset_kind == "sharded_100m_full"
    assert len(cache.train.obs) == 14
    assert len(cache.val.obs) == 8
    assert np.array_equal(cache.train.obs, expected_train)
    assert cache.train.trajectory_index.lengths.tolist() == [4, 3, 4, 3]
    assert cache.train.trajectory_index.steps_remaining.tolist() == [3, 2, 1, 0, 2, 1, 0] * 2
    assert np.allclose(cache.norm_stats["obs_mean"], expected_train.mean(axis=0))
    assert float(cache.norm_stats["obs_mean"].max()) < 1_000  # validation excluded
    assert len(cache.source_files) == 4
    assert all(len(entry["sha256"]) == 64 for entry in cache.source_files)
    assert len(cache.source_manifest_sha256) == 64

    manifest = json.loads((cache.path / "manifest.json").read_text())
    assert manifest["source_dataset"] == "toy-100m-v0"
    assert manifest["train_shards"] == manifest["validation_shards"] == 2
    assert manifest["train_transitions"] == 14
    assert manifest["validation_transitions"] == 8

    hit = build_or_load_sharded_cache(
        "toy-100m-v0", source, cache_root=tmp_path / "cache", expected_shards=2
    )
    assert hit.was_hit
    assert hit.source_manifest_sha256 == cache.source_manifest_sha256


def test_uniform_anchor_cap_maps_full_valid_universe_without_large_mask(tmp_path: Path):
    source, _ = _make_release(tmp_path)
    cache = build_or_load_sharded_cache(
        "toy-100m-v0", source, cache_root=tmp_path / "cache", expected_shards=2
    )
    normalizer = Normalizer.from_state_dict(cache.norm_stats)
    cfg = FutureSetConfig(horizons=(1,), h_max=1, num_neighbors=2)
    first = ChunkDataset(
        {}, normalizer, cfg, max_anchors=5, seed=19, shared=cache.train
    )
    second = ChunkDataset(
        {}, normalizer, cfg, max_anchors=5, seed=19, shared=cache.train
    )
    different = ChunkDataset(
        {}, normalizer, cfg, max_anchors=5, seed=20, shared=cache.train
    )

    assert first.total_valid_anchors == sum(length - 1 for length in [4, 3, 4, 3])
    assert len(first.anchors) == 5
    assert np.array_equal(first.anchors, second.anchors)
    assert not np.array_equal(first.anchors, different.anchors)
    assert np.all(cache.train.trajectory_index.remaining_at(first.anchors) >= 1)
    assert not hasattr(first, "valid_mask")


def test_100m_anchor_rank_sampler_is_bounded_deterministic_and_unique():
    first = uniform_anchor_ranks(100_000_000, 300_000, seed=7)
    second = uniform_anchor_ranks(100_000_000, 300_000, seed=7)
    assert np.array_equal(first, second)
    assert len(first) == len(np.unique(first)) == 300_000
    assert int(first[0]) >= 0 and int(first[-1]) < 100_000_000


def test_discovery_rejects_missing_or_extra_shards(tmp_path: Path):
    source, _ = _make_release(tmp_path)
    (source / "toy-v0-001-val.npz").unlink()
    with pytest.raises(ShardedDatasetError, match="missing"):
        discover_shards(source, "toy-100m-v0", expected_shards=2)


def test_standard_shared_cache_binds_content_identity_and_invalidates_stat_drift(
    tmp_path: Path, monkeypatch
):
    from treewm.data import ogbench_dataset
    from treewm.data.shared_cache import build_or_load

    source = tmp_path / "standard"
    source.mkdir()
    (source / "toy-v0.npz").write_bytes(b"train-release")
    (source / "toy-v0-val.npz").write_bytes(b"validation-release")
    train = {
        "observations": np.arange(24, dtype=np.float32).reshape(8, 3),
        "actions": np.arange(16, dtype=np.float32).reshape(8, 2),
        "terminals": np.array([0, 0, 0, 1, 0, 0, 0, 1], dtype=bool),
    }
    val = {
        "observations": np.arange(12, dtype=np.float32).reshape(4, 3) + 100,
        "actions": np.arange(8, dtype=np.float32).reshape(4, 2),
        "terminals": np.array([0, 0, 0, 1], dtype=bool),
    }
    calls = []

    def fake_load(dataset_name, dataset_dir=None, env_only=False, **kwargs):
        calls.append((dataset_name, dataset_dir, env_only))
        return object() if env_only else (object(), train, val)

    monkeypatch.setattr(ogbench_dataset, "load_ogbench", fake_load)
    first = build_or_load("toy-v0", str(source), root=tmp_path / "cache", verbose=False)
    assert len(first.source_manifest_sha256) == 64
    assert [entry["split"] for entry in first.source_files] == ["train", "val"]
    assert all(len(entry["sha256"]) == 64 for entry in first.source_files)
    hit = build_or_load("toy-v0", str(source), root=tmp_path / "cache", verbose=False)
    assert hit.was_hit and hit.key == first.key
    assert len(calls) == 1

    # Stat drift selects a new cache key; it cannot silently reuse a content identity
    # from an earlier local copy.
    (source / "toy-v0.npz").write_bytes(b"changed-train-release")
    changed = build_or_load("toy-v0", str(source), root=tmp_path / "cache", verbose=False)
    assert changed.key != first.key
    assert changed.source_manifest_sha256 != first.source_manifest_sha256
    assert len(calls) == 2


def test_full_shard_build_resumes_from_durable_shard_cursor(tmp_path: Path):
    source, _ = _make_release(tmp_path)
    calls = 0

    def interrupt_after_first_train_shard():
        nonlocal calls
        calls += 1
        # 2 train scans + 2 val scans + shard0 start + its stats chunk; the next
        # callback is shard1, after shard0 arrays/stats and cursor were fsynced.
        if calls == 7:
            raise RuntimeError("simulated USR1")

    with pytest.raises(RuntimeError, match="simulated USR1"):
        build_or_load_sharded_cache(
            "toy-100m-v0",
            source,
            cache_root=tmp_path / "cache",
            expected_shards=2,
            stop_callback=interrupt_after_first_train_shard,
        )
    states = list((tmp_path / "cache").glob("*__full__*/.build_state.json"))
    assert len(states) == 1
    state = json.loads(states[0].read_text())
    assert state["train_next_shard"] == 1
    assert state["train_stats"]["count"] == 7

    cache = build_or_load_sharded_cache(
        "toy-100m-v0", source, cache_root=tmp_path / "cache", expected_shards=2
    )
    assert len(cache.train.obs) == 14
    assert not states[0].exists()
    assert (cache.path / "manifest.json").is_file()


def test_full_shard_build_resumes_after_array_promotion_during_digests(tmp_path: Path):
    source, _ = _make_release(tmp_path)
    cache_root = tmp_path / "cache"

    def interrupt_digest_phase():
        states = list(cache_root.glob("*__full__*/.build_state.json"))
        if not states:
            return
        state = json.loads(states[0].read_text())
        if state.get("arrays_promoted") and len(state.get("source_files", [])) == 1:
            raise RuntimeError("simulated digest-phase USR1")

    with pytest.raises(RuntimeError, match="digest-phase USR1"):
        build_or_load_sharded_cache(
            "toy-100m-v0",
            source,
            cache_root=cache_root,
            expected_shards=2,
            stop_callback=interrupt_digest_phase,
        )
    state_path = next(cache_root.glob("*__full__*/.build_state.json"))
    state = json.loads(state_path.read_text())
    assert state["arrays_promoted"] is True
    assert len(state["source_files"]) == 1

    cache = build_or_load_sharded_cache(
        "toy-100m-v0", source, cache_root=cache_root, expected_shards=2
    )
    assert len(cache.source_files) == 4
    assert not state_path.exists()
    assert not list(cache.path.glob(".*.build.npy"))


def test_full_shard_build_recovers_crash_between_promotion_and_cursor(
    tmp_path: Path, monkeypatch
):
    from treewm.data import sharded_ogbench

    source, _ = _make_release(tmp_path)
    cache_root = tmp_path / "cache"
    original_promote = sharded_ogbench._promote_arrays

    def crash_after_durable_promotion(build_dir):
        original_promote(build_dir)
        raise RuntimeError("simulated promotion crash")

    monkeypatch.setattr(sharded_ogbench, "_promote_arrays", crash_after_durable_promotion)
    with pytest.raises(RuntimeError, match="promotion crash"):
        build_or_load_sharded_cache(
            "toy-100m-v0", source, cache_root=cache_root, expected_shards=2
        )

    state_path = next(cache_root.glob("*__full__*/.build_state.json"))
    state = json.loads(state_path.read_text())
    assert state["arrays_promoted"] is False
    assert state["train_norm_next_row"] == 14
    assert state["val_norm_next_row"] == 8
    assert list(state_path.parent.glob("train_*.npy"))

    monkeypatch.setattr(sharded_ogbench, "_promote_arrays", original_promote)
    cache = build_or_load_sharded_cache(
        "toy-100m-v0", source, cache_root=cache_root, expected_shards=2
    )
    assert len(cache.train.obs) == 14
    assert not state_path.exists()
    assert not list(cache.path.glob(".*.build.npy"))
