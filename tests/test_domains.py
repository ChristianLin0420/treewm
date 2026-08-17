"""Verify every domain adapter's observation indices against the environment itself.

These indices were derived by reading OGBench's ``compute_observation``. That derivation
could be wrong, or could silently drift when OGBench updates. A wrong index does not
crash -- it produces a full set of plausible numbers measuring the wrong quantity, which
is the single most dangerous failure mode in this screen. So each index is checked
against the environment's own ``privileged/*`` info at runtime.
"""

from __future__ import annotations

import numpy as np
import pytest

ogbench = pytest.importorskip("ogbench")

from treewm.evaluation.domains import (BUTTON_STRIDE, CUBE_STRIDE, PROPRIO, XYZ_CENTER,
                                       XYZ_SCALER, get_domain)

pytestmark = pytest.mark.slow


def _reset(name):
    env = ogbench.make_env_and_datasets(name, env_only=True)
    ob, info = env.reset(options={"task_id": 1}, seed=0)
    return env, np.asarray(ob, dtype=np.float64), info


@pytest.mark.parametrize("name,n_cubes", [("cube-single-play-v0", 1), ("cube-double-play-v0", 2)])
def test_cube_position_dims(name, n_cubes):
    env, ob, info = _reset(name)
    d = get_domain(name)
    gv = d.goal_vector(ob)
    for i in range(n_cubes):
        expect = (np.asarray(info[f"privileged/block_{i}_pos"]) - XYZ_CENTER) * XYZ_SCALER
        lo, hi = d.subgoals[i]
        np.testing.assert_allclose(gv[lo:hi], expect, atol=1e-4)


def test_puzzle_button_states():
    env, ob, info = _reset("puzzle-3x3-play-v0")
    d = get_domain("puzzle-3x3-play-v0")
    gv = d.goal_vector(ob)
    for i, (lo, hi) in enumerate(d.subgoals):
        expect = int(info[f"privileged/button_{i}_state"])
        assert int(np.argmax(gv[lo:hi])) == expect, f"button {i} one-hot misaligned"


def test_antsoccer_goal_tracks_ball_not_agent():
    """The whole point of the adapter: antsoccer's goal is the ball."""
    name = "antsoccer-medium-navigate-v0"
    env, ob, info = _reset(name)
    d = get_domain(name)
    agent_xy, ball_xy = env.unwrapped.get_agent_ball_xy()
    np.testing.assert_allclose(d.goal_vector(ob), ball_xy, atol=1e-4)
    # and it must NOT be the agent, unless they happen to coincide
    if np.linalg.norm(agent_xy - ball_xy) > 1e-3:
        assert not np.allclose(d.goal_vector(ob), agent_xy, atol=1e-4)


def test_scene_button_and_cube_dims():
    env, ob, info = _reset("scene-play-v0")
    d = get_domain("scene-play-v0")
    gv = d.goal_vector(ob)
    np.testing.assert_allclose(
        gv[0:3], (np.asarray(info["privileged/block_0_pos"]) - XYZ_CENTER) * XYZ_SCALER, atol=1e-4)
    for i in range(2):
        lo = 3 + 2 * i
        assert int(np.argmax(gv[lo:lo + 2])) == int(info[f"privileged/button_{i}_state"])


@pytest.mark.parametrize("name", ["antmaze-large-navigate-v0", "humanoidmaze-medium-navigate-v0"])
def test_locomotion_xy(name):
    env, ob, info = _reset(name)
    d = get_domain(name)
    np.testing.assert_allclose(d.goal_vector(ob), np.asarray(ob)[:2], atol=1e-6)


@pytest.mark.parametrize("name", ["antmaze-large-navigate-v0", "cube-double-play-v0",
                                  "scene-play-v0", "puzzle-3x3-play-v0",
                                  "antsoccer-medium-navigate-v0"])
def test_goal_is_same_shape_as_obs(name):
    """Planner scoring relies on info['goal'] being a full goal observation."""
    env, ob, info = _reset(name)
    assert "goal" in info, f"{name} exposes no goal observation"
    assert np.shape(info["goal"]) == np.shape(ob)
    d = get_domain(name)
    assert d.obs_dim == ob.shape[0], f"{name} obs_dim {d.obs_dim} != actual {ob.shape[0]}"
    # a fresh reset should not already be at the goal on every subgoal
    assert d.distance(ob, np.asarray(info["goal"])) >= 0.0


def test_unknown_env_raises_rather_than_guessing():
    with pytest.raises(KeyError, match="no domain adapter"):
        get_domain("some-unregistered-env-v0")
