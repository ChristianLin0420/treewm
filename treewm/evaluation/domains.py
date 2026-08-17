"""Per-environment goal semantics for the cross-family screen.

PointMaze let us assume ``obs[:2]`` is the thing the goal constrains. That assumption is
wrong for five of the seven environments here, and wrong in a way that fails silently --
scoring antsoccer by the *agent's* position would produce a complete set of plausible
numbers measuring the wrong quantity, because the task is to move the **ball**.

Every index below was verified empirically against the environment's own ``privileged/*``
info keys (see ``tests/test_domains.py``), not derived on paper:

    ant.py      get_ob -> [qpos, qvel]                  so obs[0:2] is agent xy
    humanoid.py get_ob -> [xy, joint_angles, ...]       so obs[0:2] is agent xy
    maze.py     antsoccer success is
                ``norm(ball_xy - goal_xy) <= tol`` with ``ball_xy = qpos[-7:-5]``
                                                        so obs[15:17] is BALL xy
    manipspace  compute_observation -> 19-dim proprio prefix
                (joint_pos 6, joint_vel 6, effector_pos 3, cos/sin yaw 2,
                 gripper_opening 1, gripper_contact 1)
                then 9 dims per cube  [pos 3, quat 4, cos_yaw, sin_yaw]
                then 4 dims per button [state one-hot 2, pos 1, vel 1]
                then drawer [pos, vel], window [pos, vel] for scene

Success is always taken from the environment (``info['success']`` /
``_compute_successes()``), never reimplemented here. What this module adds is (a) which
observation dims the planner should score against, and (b) partial-progress signals so a
zero success rate is still informative -- the failure mode that made three AntMaze cycles
uninterpretable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

PROPRIO = 19          # manipspace proprioceptive prefix, identical across cube/scene/puzzle
EFFECTOR = (12, 13, 14)   # effector_pos, after joint_pos(6) + joint_vel(6)
CUBE_STRIDE = 9       # pos 3 + quat 4 + cos_yaw + sin_yaw
BUTTON_STRIDE = 4     # state one-hot 2 + pos + vel
XYZ_CENTER = np.array([0.425, 0.0, 0.0])
XYZ_SCALER = 10.0


@dataclass(frozen=True)
class Domain:
    """How to score, measure and bucket one environment."""

    name: str
    family: str                       # locomotion | manipulation | puzzle | grid
    goal_dims: tuple[int, ...]        # observation dims the goal actually constrains
    goal_metric: str                  # "l2" for continuous, "onehot" for discrete state
    obs_dim: int
    action_dim: int
    # slices of goal_dims belonging to each independent sub-object, for partial progress
    subgoals: tuple[tuple[int, int], ...] = ()
    difficulty_fn: Callable | None = field(default=None, compare=False)
    max_episode_steps: int = 1000
    # For the retrieval-independent branching cross-check (interaction_sanity): the
    # actuator's position and the positions of things it can act on. Futures genuinely
    # diverge near a manipulable object and collapse to "move somewhere" in free space,
    # so this is the manipulation analogue of a maze junction. Empty where no such
    # position exists in the observation.
    effector_dims: tuple[int, ...] = ()
    object_dims: tuple[tuple[int, ...], ...] = ()

    # ---- goal scoring -------------------------------------------------------------
    def goal_vector(self, ob: np.ndarray) -> np.ndarray:
        return np.asarray(ob, dtype=np.float32)[list(self.goal_dims)]

    def distance(self, ob: np.ndarray, goal: np.ndarray) -> float:
        """Scalar goal distance in the domain's own units.

        For one-hot state domains an L2 distance over the one-hot block is monotone in the
        number of mismatched cells, so this doubles as a Hamming-like signal while staying
        differentiable-friendly for the planner.
        """
        a, b = self.goal_vector(ob), self.goal_vector(goal)
        if self.goal_metric == "onehot":
            return float(self.hamming(ob, goal))
        return float(np.linalg.norm(a - b))

    def hamming(self, ob: np.ndarray, goal: np.ndarray) -> int:
        """Mismatched discrete cells, for puzzle/scene style domains."""
        a, b = self.goal_vector(ob), self.goal_vector(goal)
        n = 0
        for lo, hi in self.subgoals:
            if int(np.argmax(a[lo:hi])) != int(np.argmax(b[lo:hi])):
                n += 1
        return n

    def subgoal_fraction(self, ob: np.ndarray, goal: np.ndarray, tol: float = 0.15) -> float:
        """Fraction of independent sub-objects already at their goal value."""
        if not self.subgoals:
            return float("nan")
        a, b = self.goal_vector(ob), self.goal_vector(goal)
        hits = []
        for lo, hi in self.subgoals:
            if self.goal_metric == "onehot":
                hits.append(int(np.argmax(a[lo:hi])) == int(np.argmax(b[lo:hi])))
            else:
                hits.append(float(np.linalg.norm(a[lo:hi] - b[lo:hi])) <= tol)
        return float(np.mean(hits))


def _cube_dims(n_cubes: int) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """Positions only -- orientation is not constrained by the pick-and-place tasks."""
    dims: list[int] = []
    subs: list[tuple[int, int]] = []
    for i in range(n_cubes):
        base = PROPRIO + CUBE_STRIDE * i
        subs.append((len(dims), len(dims) + 3))
        dims.extend(range(base, base + 3))
    return tuple(dims), tuple(subs)


def _button_dims(n_buttons: int, start: int = PROPRIO) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...]]:
    """One-hot state block per button; pos/vel are dynamics, not goal content."""
    dims: list[int] = []
    subs: list[tuple[int, int]] = []
    for i in range(n_buttons):
        base = start + BUTTON_STRIDE * i
        subs.append((len(dims), len(dims) + 2))
        dims.extend((base, base + 1))
    return tuple(dims), tuple(subs)


# ---- difficulty measures -----------------------------------------------------------
# Deliberately NOT a shared geometric "horizon": the spec is explicit that difficulty must
# be domain-appropriate. Each returns a scalar that increases with task difficulty.

def _cube_difficulty(env, ob, goal, domain: "Domain") -> float:
    """Number of cubes that must actually move (a 2-cube task may only need one)."""
    a, b = domain.goal_vector(ob), domain.goal_vector(goal)
    return float(sum(np.linalg.norm(a[lo:hi] - b[lo:hi]) > 0.15 for lo, hi in domain.subgoals))


def _discrete_difficulty(env, ob, goal, domain: "Domain") -> float:
    """Mismatched cells = lower bound on the number of atomic actions required."""
    return float(domain.hamming(ob, goal))


def _maze_difficulty(env, ob, goal, domain: "Domain") -> float:
    """Euclidean start-goal separation; geodesic where a maze map is available."""
    try:
        from treewm.data.maze_utils import MazeSpec
        spec = MazeSpec.from_env(env)
        gv_o, gv_g = domain.goal_vector(ob), domain.goal_vector(goal)
        i0, j0 = spec.xy_to_ij(gv_o[:2])
        i1, j1 = spec.xy_to_ij(gv_g[:2])
        d = spec.geodesic_field((int(i0), int(j0)))[int(i1), int(j1)]
        if np.isfinite(d):
            return float(d * spec.unit)
    except Exception:
        pass
    return float(domain.distance(ob, goal))


def _build_registry() -> dict[str, Domain]:
    reg: dict[str, Domain] = {}

    # ---- locomotion: goal constrains the agent's xy ---------------------------------
    for name, obs_dim, act_dim in [("antmaze-large-navigate-v0", 29, 8),
                                   ("humanoidmaze-medium-navigate-v0", 69, 21)]:
        reg[name] = Domain(name, "locomotion", (0, 1), "l2", obs_dim, act_dim,
                           subgoals=((0, 2),), difficulty_fn=_maze_difficulty)

    # ---- antsoccer: goal constrains the BALL, not the agent -------------------------
    reg["antsoccer-medium-navigate-v0"] = Domain(
        "antsoccer-medium-navigate-v0", "locomotion", (15, 16), "l2", 42, 8,
        subgoals=((0, 2),), difficulty_fn=_maze_difficulty,
        # The ant is the actuator, the ball is the only thing it can act on.
        effector_dims=(0, 1), object_dims=((15, 16),))

    # ---- manipulation: goal constrains cube positions -------------------------------
    for name, n, obs_dim in [("cube-single-play-v0", 1, 28), ("cube-double-play-v0", 2, 37)]:
        dims, subs = _cube_dims(n)
        reg[name] = Domain(name, "manipulation", dims, "l2", obs_dim, 5,
                           subgoals=subs, difficulty_fn=_cube_difficulty,
                           effector_dims=EFFECTOR,
                           object_dims=tuple(tuple(range(PROPRIO + CUBE_STRIDE * i,
                                                         PROPRIO + CUBE_STRIDE * i + 3))
                                             for i in range(n)))

    # ---- scene: one cube + two buttons + drawer + window ----------------------------
    cube_dims, cube_subs = _cube_dims(1)
    btn_dims, btn_subs = _button_dims(2, start=PROPRIO + CUBE_STRIDE)
    drawer_window = (PROPRIO + CUBE_STRIDE + 2 * BUTTON_STRIDE,
                     PROPRIO + CUBE_STRIDE + 2 * BUTTON_STRIDE + 2)
    dims = cube_dims + btn_dims + (drawer_window[0], drawer_window[1])
    off = len(cube_dims)
    subs = cube_subs + tuple((off + lo, off + hi) for lo, hi in btn_subs)
    subs = subs + ((len(dims) - 2, len(dims) - 1), (len(dims) - 1, len(dims)))
    reg["scene-play-v0"] = Domain("scene-play-v0", "manipulation", dims, "l2", 40, 5,
                                  subgoals=subs, difficulty_fn=_cube_difficulty,
                                  # Only the cube has a 3-D position in the observation;
                                  # buttons/drawer/window expose press depth, not location.
                                  effector_dims=EFFECTOR,
                                  object_dims=(tuple(range(PROPRIO, PROPRIO + 3)),))

    # ---- puzzle: nine binary buttons, Hamming distance ------------------------------
    dims, subs = _button_dims(9)
    reg["puzzle-3x3-play-v0"] = Domain("puzzle-3x3-play-v0", "puzzle", dims, "onehot", 55, 5,
                                       subgoals=subs, difficulty_fn=_discrete_difficulty)

    # PointMaze keeps its original semantics so existing cycles stay reproducible.
    for name in ("pointmaze-medium-stitch-v0", "pointmaze-large-stitch-v0",
                 "pointmaze-giant-stitch-v0"):
        reg[name] = Domain(name, "locomotion", (0, 1), "l2", 2, 2,
                           subgoals=((0, 2),), difficulty_fn=_maze_difficulty)
    return reg


REGISTRY: dict[str, Domain] = _build_registry()


def get_domain(env_name: str) -> Domain:
    if env_name not in REGISTRY:
        raise KeyError(f"no domain adapter for {env_name!r}; "
                       f"known: {sorted(REGISTRY)}. Add one rather than falling back to "
                       "obs[:2], which silently measures the wrong quantity.")
    return REGISTRY[env_name]


def progress_metrics(env, domain: Domain, ob, goal, info: dict) -> dict[str, float]:
    """Domain-specific competence signals, so a zero success rate is still informative."""
    out: dict[str, float] = {}
    out["progress/goal_distance"] = domain.distance(ob, goal)
    frac = domain.subgoal_fraction(ob, goal)
    if np.isfinite(frac):
        out["progress/subgoal_fraction"] = frac
    if domain.goal_metric == "onehot":
        out["progress/hamming"] = float(domain.hamming(ob, goal))

    # The environment's own per-subgoal booleans, where it exposes them. Authoritative:
    # we never reimplement success.
    u = getattr(env, "unwrapped", env)
    if hasattr(u, "_compute_successes"):
        try:
            res = u._compute_successes()
            flat = []
            for r in (res if isinstance(res, tuple) else [res]):
                flat.extend(np.atleast_1d(r).astype(bool).tolist())
            if flat:
                out["progress/env_subgoal_fraction"] = float(np.mean(flat))
                out["progress/env_subgoals_met"] = float(np.sum(flat))
        except Exception:
            pass
    return out
