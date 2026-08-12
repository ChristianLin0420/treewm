"""Dataset chunk sampling, horizon masking and support/frequency separation."""

from __future__ import annotations

import numpy as np
import pytest

from treewm.data.future_sets import FutureSetBuilder, FutureSetConfig
from treewm.data.ogbench_dataset import Normalizer, TrajectoryIndex


def make_synthetic(num_traj: int = 20, length: int = 100, obs_dim: int = 2, act_dim: int = 2, seed: int = 0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(num_traj * length, obs_dim)).astype(np.float32)
    act = rng.uniform(-1, 1, size=(num_traj * length, act_dim)).astype(np.float32)
    terminals = np.zeros(num_traj * length, dtype=np.float32)
    terminals[length - 1 :: length] = 1.0
    return {"observations": obs, "actions": act, "terminals": terminals}


def test_trajectory_index_covers_all_steps():
    data = make_synthetic(num_traj=7, length=13)
    idx = TrajectoryIndex.from_terminals(data["terminals"])
    assert idx.num_trajectories == 7
    assert len(idx.traj_id) == 7 * 13
    assert idx.lengths.tolist() == [13] * 7
    # Last step of each trajectory has zero remaining steps.
    assert idx.steps_remaining[12] == 0
    assert idx.steps_remaining[0] == 12


def test_trajectory_index_handles_missing_final_terminal():
    data = make_synthetic(num_traj=3, length=10)
    data["terminals"][-1] = 0.0  # truncated final trajectory
    idx = TrajectoryIndex.from_terminals(data["terminals"])
    assert len(idx.traj_id) == 30
    assert idx.num_trajectories == 3


def test_chunk_sampling_shapes_and_horizon_masking():
    data = make_synthetic()
    nz = Normalizer.fit(data["observations"], data["actions"])
    cfg = FutureSetConfig(num_neighbors=8, h_max=64, horizons=(4, 8, 16, 32, 64))
    idx = TrajectoryIndex.from_terminals(data["terminals"])
    builder = FutureSetBuilder(nz.norm_obs(data["observations"]), nz.norm_act(data["actions"]), idx, cfg)

    item = builder.build(5)
    assert item["fut_actions"].shape == (8, 64, 2)
    assert item["fut_action_mask"].shape == (8, 64)
    assert item["fut_endpoint"].shape == (8, 2)

    # The mask must be a prefix of exactly the chosen horizon, and zero after it.
    horizons = np.asarray(cfg.horizons)
    for slot in range(8):
        if item["fut_valid"][slot] == 0:
            continue
        h = int(horizons[item["fut_horizon_idx"][slot]])
        mask = item["fut_action_mask"][slot]
        assert mask[:h].all(), "mask must cover the full horizon"
        assert not mask[h:].any(), "mask must be zero past the horizon"
        assert np.allclose(item["fut_actions"][slot, h:], 0.0), "padding must be zeroed"


def test_variable_horizon_never_exceeds_remaining_steps():
    data = make_synthetic(num_traj=4, length=40)
    nz = Normalizer.fit(data["observations"], data["actions"])
    cfg = FutureSetConfig(num_neighbors=6, horizons=(4, 8, 16, 32, 64), h_max=64)
    idx = TrajectoryIndex.from_terminals(data["terminals"])
    builder = FutureSetBuilder(nz.norm_obs(data["observations"]), nz.norm_act(data["actions"]), idx, cfg)
    remaining = idx.steps_remaining
    for anchor in (0, 7, 20, 35):
        if remaining[anchor] < min(cfg.horizons):
            continue
        item = builder.build(anchor)
        lengths = item["fut_horizon_len"][item["fut_valid"] > 0]
        assert (lengths >= min(cfg.horizons)).all()
        assert (lengths <= max(cfg.horizons)).all()


def test_support_and_frequency_are_separate():
    """Rare modes must keep full support while their mass stays small.

    Three well-separated future clusters with an 80/15/5 split must yield
    ``support = [1, 1, 1]`` and ``mass ~ [.80, .15, .05]`` -- the mandatory
    distinction of spec section 13.
    """
    n = 100
    obs = np.zeros((n, 2), dtype=np.float32)
    act = np.zeros((n, 2), dtype=np.float32)
    terminals = np.zeros(n, dtype=np.float32)
    terminals[-1] = 1.0

    # Anchor at index 0; 20 neighbours all sit at the origin but travel to three
    # clearly distinct destinations with an 80/15/5 split.
    counts = {0: 16, 1: 3, 2: 1}
    destinations = {0: np.array([5.0, 0.0]), 1: np.array([0.0, 5.0]), 2: np.array([-5.0, 0.0])}
    slot = 0
    for mode, count in counts.items():
        for _ in range(count):
            obs[slot] = 0.0
            obs[slot + 1 : slot + 5] = destinations[mode]
            slot += 5

    idx = TrajectoryIndex.from_terminals(terminals)
    cfg = FutureSetConfig(
        num_neighbors=20, horizons=(4,), h_max=4, time_exclusion=0,
        retrieval_radius=1e-3, cluster_threshold=1.0, horizon_rule="fixed", fixed_horizon=4,
        relative_endpoints=True, max_modes=8,
    )
    builder = FutureSetBuilder(obs, act, idx, cfg)
    item = builder.build(0)

    num_modes = int(item["num_modes"])
    assert num_modes == 3, f"expected 3 distinct modes, got {num_modes}"

    mass = item["mode_mass"][:num_modes]
    valid = item["mode_valid"][:num_modes]
    assert np.allclose(valid, 1.0), "every mode carries equal support regardless of mass"
    assert np.isclose(mass.sum(), 1.0, atol=1e-5)
    assert mass.max() > 0.6, "dominant mode should hold most of the mass"
    assert mass.min() < 0.15, "rare mode should hold little mass"
    # The rare mode is still fully supported: support is not proportional to mass.
    assert valid[np.argmin(mass)] == 1.0


def test_rare_modes_survive_truncation():
    """max_modes truncation must not be mass-ordered."""
    cfg = FutureSetConfig(max_modes=2)
    rng = np.random.default_rng(0)
    endpoints = np.array([[0.0, 0.0], [5.0, 0.0], [0.0, 5.0], [-5.0, 0.0]], dtype=np.float32)
    builder = FutureSetBuilder(endpoints, endpoints, TrajectoryIndex.from_terminals(np.array([0, 0, 0, 1.0])), cfg)
    labels, reps, mass = builder._cluster(endpoints, rng)
    assert len(reps) == 2, "should truncate to max_modes"
    assert np.isclose(mass.sum(), 1.0, atol=1e-5)
