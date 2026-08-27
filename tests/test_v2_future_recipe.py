from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

import treewm.data.future_recipe as future_recipe_module
from treewm.data.future_recipe import (
    FutureRecipe,
    FutureRecipeError,
    _record_dtype,
    anchors_for_seed,
    build_or_load_split_recipe,
)
from treewm.data.future_sets import FutureSetBuilder, FutureSetConfig
from treewm.data.ogbench_dataset import Normalizer, TrajectoryIndex


def _split(tmp_path: Path, *, mmap: bool = False):
    trajectories, length, obs_dim, act_dim = 12, 24, 5, 3
    observations = np.zeros((trajectories * length, obs_dim), dtype=np.float32)
    actions = np.zeros((trajectories * length, act_dim), dtype=np.float32)
    terminals = np.zeros(trajectories * length, dtype=np.float32)
    for trajectory in range(trajectories):
        start = trajectory * length
        time = np.arange(length, dtype=np.float32)
        observations[start : start + length] = np.stack(
            [
                np.sin(time / (2 + trajectory % 3)),
                np.cos(time / (3 + trajectory % 4)),
                np.full(length, trajectory % 4),
                time / length,
                np.full(length, trajectory / trajectories),
            ],
            axis=-1,
        )
        actions[start : start + length] = np.stack(
            [np.sin(time), np.cos(time), time / length], axis=-1
        )
        terminals[start + length - 1] = 1
    normalizer = Normalizer.fit(observations, actions)
    obs_norm = normalizer.norm_obs(observations).astype(np.float32)
    act_norm = normalizer.norm_act(actions).astype(np.float32)
    index = TrajectoryIndex.from_terminals(terminals)
    if mmap:
        for name, values in (("obs", obs_norm), ("act", act_norm)):
            mapped = np.lib.format.open_memmap(
                tmp_path / f"{name}.npy", mode="w+", dtype=np.float32, shape=values.shape
            )
            mapped[:] = values
            mapped.flush()
        obs_norm = np.load(tmp_path / "obs.npy", mmap_mode="r")
        act_norm = np.load(tmp_path / "act.npy", mmap_mode="r")
        from treewm.data.sharded_ogbench import MemmapTrajectoryIndex

        mapped_index = {}
        for name, values in (
            ("traj_id", index.traj_id),
            ("remaining", index.steps_remaining),
            ("starts", index.starts),
            ("lengths", index.lengths),
        ):
            values = np.asarray(values)
            mapped = np.lib.format.open_memmap(
                tmp_path / f"{name}.npy", mode="w+", dtype=values.dtype, shape=values.shape
            )
            mapped[:] = values
            mapped.flush()
            mapped_index[name] = np.load(tmp_path / f"{name}.npy", mmap_mode="r")
        index = MemmapTrajectoryIndex(
            traj_id=mapped_index["traj_id"],
            steps_remaining=mapped_index["remaining"],
            starts=mapped_index["starts"],
            lengths=mapped_index["lengths"],
        )
    return obs_norm, act_norm, index, normalizer


def _cfg(executable_prefix_steps: int = 0) -> FutureSetConfig:
    return FutureSetConfig(
        num_neighbors=8,
        query_multiplier=4,
        time_exclusion=0,
        retrieval_radius=10.0,
        metric_mode="rms_v2",
        horizons=(4, 8, 16),
        h_max=16,
        horizon_rule="displacement",
        displacement_threshold=0.3,
        relative_endpoints=True,
        cluster_threshold=0.4,
        max_modes=4,
        multi_step_depth=3,
        executable_prefix_steps=executable_prefix_steps,
        retrieval_pool=0,
    )


def _build(
    tmp_path: Path,
    *,
    mmap: bool = False,
    stop_callback=None,
    chunk_size=3,
    executable_prefix_steps: int = 0,
):
    obs, act, index, normalizer = _split(tmp_path, mmap=mmap)
    cfg = _cfg(executable_prefix_steps)
    anchors = anchors_for_seed(index, 20, seed=0)
    payload = build_or_load_split_recipe(
        tmp_path / "recipe",
        split="train",
        obs_norm=obs,
        act_norm=act,
        index=index,
        cfg=cfg,
        xy_dims=(0, 1),
        task_metric_dims=(0, 1, 2),
        anchor_sets={"seed0": anchors},
        source_manifest_sha256="a" * 64,
        split_manifest_sha256="b" * 64,
        normalizer_sha256="c" * 64,
        calibration_sha256="d" * 64,
        chosen_thresholds={
            "retrieval_radius": cfg.retrieval_radius,
            "displacement_threshold": cfg.displacement_threshold,
            "cluster_threshold": cfg.cluster_threshold,
        },
        code_sha256="e" * 64,
        runtime_sha256="f" * 64,
        stop_callback=stop_callback,
        chunk_size=chunk_size,
    )
    return obs, act, index, cfg, anchors, payload


@pytest.mark.parametrize("mmap", [False, True])
@pytest.mark.parametrize("executable_prefix_steps", [0, 4])
def test_recipe_reconstruction_is_bitwise_identical_to_every_builder_output(
    tmp_path, mmap, executable_prefix_steps
):
    obs, act, index, cfg, anchors, payload = _build(
        tmp_path,
        mmap=mmap,
        executable_prefix_steps=executable_prefix_steps,
    )
    recipe = FutureRecipe(
        tmp_path / "recipe",
        expected_recipe_sha256=payload["recipe_sha256"],
        expected_source_manifest_sha256="a" * 64,
        expected_calibration_sha256="d" * 64,
    )
    assert (
        "executable_prefix_steps" in payload["identity"]["future_config"]
    ) is bool(executable_prefix_steps)
    builder = FutureSetBuilder(
        obs, act, index, cfg, xy_dims=(0, 1), task_metric_dims=(0, 1, 2)
    )
    for anchor in anchors:
        expected = builder.build(int(anchor))
        actual = recipe.build(int(anchor), obs_norm=obs, act_norm=act, index=index)
        assert actual.keys() == expected.keys()
        for key in expected:
            assert actual[key].dtype == expected[key].dtype, key
            assert np.array_equal(actual[key], expected[key]), key


def test_legacy_recipe_rows_can_derive_opt_in_prefix_without_identity_mutation(
    tmp_path,
):
    obs, act, index, cfg, anchors, payload = _build(
        tmp_path, executable_prefix_steps=0
    )
    recipe = FutureRecipe(
        tmp_path / "recipe", expected_recipe_sha256=payload["recipe_sha256"]
    )
    anchor = int(anchors[0])
    historical = recipe.build(anchor, obs_norm=obs, act_norm=act, index=index)
    assert not any("executable_prefix" in key for key in historical)

    active_cfg = replace(cfg, executable_prefix_steps=4)
    expected = FutureSetBuilder(
        obs,
        act,
        index,
        active_cfg,
        xy_dims=(0, 1),
        task_metric_dims=(0, 1, 2),
    ).build(anchor)
    prospective = recipe.build(
        anchor,
        obs_norm=obs,
        act_norm=act,
        index=index,
        executable_prefix_steps=4,
    )
    assert recipe.recipe_sha256 == payload["recipe_sha256"]
    assert prospective.keys() == expected.keys()
    for key in expected:
        assert np.array_equal(prospective[key], expected[key]), key


def test_recipe_build_resumes_only_after_durable_chunk_and_reuses_complete(tmp_path):
    calls = 0

    class Interrupted(RuntimeError):
        pass

    def stop():
        nonlocal calls
        calls += 1
        if calls == 5:
            raise Interrupted

    with pytest.raises(Interrupted):
        _build(tmp_path, stop_callback=stop, chunk_size=3)
    state = json.loads((tmp_path / "recipe" / ".build_state.json").read_text())
    assert state["next_row"] == 3
    obs, act, index, cfg, anchors, payload = _build(tmp_path, chunk_size=3)
    assert payload["record_count"] == len(anchors)
    assert not (tmp_path / "recipe" / ".build_state.json").exists()
    assert not (tmp_path / "recipe" / ".records.build.npy").exists()
    # A complete cache is opened without rebuilding and remains read-only to consumers.
    again = _build(tmp_path, chunk_size=1)[-1]
    assert again == payload
    recipe = FutureRecipe(tmp_path / "recipe")
    assert recipe.records.flags.writeable is False


def test_recipe_resumes_finalization_after_interruption_during_content_hash(
    tmp_path, monkeypatch
):
    real_file_sha256 = future_recipe_module.file_sha256
    calls = 0

    class Interrupted(RuntimeError):
        pass

    def interrupt_first_hash(path, stop_callback=None):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise Interrupted
        return real_file_sha256(path, stop_callback)

    monkeypatch.setattr(future_recipe_module, "file_sha256", interrupt_first_hash)
    with pytest.raises(Interrupted):
        _build(tmp_path)
    state = json.loads((tmp_path / "recipe" / ".build_state.json").read_text())
    assert state["next_row"] == state["total_rows"]
    assert (tmp_path / "recipe" / "records.npy").is_file()
    assert not (tmp_path / "recipe" / ".records.build.npy").exists()
    assert not (tmp_path / "recipe" / "manifest.json").exists()

    monkeypatch.setattr(future_recipe_module, "file_sha256", real_file_sha256)
    payload = _build(tmp_path)[-1]
    assert payload["status"] == "complete"
    assert not (tmp_path / "recipe" / ".build_state.json").exists()


def test_recipe_fails_closed_on_identity_drift_missing_anchor_and_mutation(tmp_path):
    obs, act, index, cfg, anchors, payload = _build(tmp_path)
    recipe = FutureRecipe(tmp_path / "recipe")
    with pytest.raises(FutureRecipeError, match="absent"):
        recipe.build(999_999, obs_norm=obs, act_norm=act, index=index)
    with pytest.raises(FutureRecipeError, match="identity"):
        build_or_load_split_recipe(
            tmp_path / "recipe",
            split="train",
            obs_norm=obs,
            act_norm=act,
            index=index,
            cfg=replace(cfg, cluster_threshold=0.5),
            xy_dims=(0, 1),
            task_metric_dims=(0, 1, 2),
            anchor_sets={"seed0": anchors},
            source_manifest_sha256="a" * 64,
            split_manifest_sha256="b" * 64,
            normalizer_sha256="c" * 64,
            calibration_sha256="d" * 64,
            chosen_thresholds={
                "retrieval_radius": cfg.retrieval_radius,
                "displacement_threshold": cfg.displacement_threshold,
                "cluster_threshold": 0.5,
            },
            code_sha256="e" * 64,
            runtime_sha256="f" * 64,
        )
    records = tmp_path / "recipe" / "records.npy"
    records.touch()
    with pytest.raises(FutureRecipeError, match="changed"):
        FutureRecipe(tmp_path / "recipe")


def test_compact_schema_is_bounded_and_anchor_sampling_matches_exact_valid_universe(tmp_path):
    _, _, index, _ = _split(tmp_path)
    cfg = _cfg()
    assert _record_dtype(cfg).itemsize < 400
    anchors = anchors_for_seed(index, 100, seed=7)
    assert len(anchors) == 100
    assert len(np.unique(anchors)) == 100
    assert (index.remaining_at(anchors) >= 4).all()
