"""Closed-loop goal-reaching evaluation.

Reports success against *generated world-model nodes*, which is the compute-normalised
axis of the whole project: every arm is run at the same node budget, and wall-clock is
recorded as a secondary quantity (spec section 25).

Evaluation is deterministic given a seed: episode seeds are derived from the task index
so that two arms see the same start states.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from treewm.planning.goal_planner import GoalPlanner


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    replans: int
    nodes: int
    final_goal_distance: float
    best_goal_distance: float
    chunk_lengths: list[int] = field(default_factory=list)
    selected_depths: list[int] = field(default_factory=list)
    initial_goal_distance: float = 0.0
    displacement: float = 0.0
    path_length: float = 0.0
    action_magnitude: float = 0.0
    trajectory: list[np.ndarray] = field(default_factory=list)
    progress: dict = field(default_factory=dict)


def run_episode(
    env,
    planner: GoalPlanner,
    task: dict,
    seed: int,
    max_steps: int = 500,
    record_trajectory: bool = False,
    domain=None,
) -> EpisodeResult:
    """One goal-reaching episode with replanning after each executed chunk.

    ``domain`` supplies the environment's own goal semantics. Without it this falls back
    to PointMaze's ``obs[:2]`` convention, which is correct only for the maze families --
    for antsoccer the goal is the ball, for cube/scene/puzzle there is no meaningful xy.
    """
    options: dict = {}
    if "task_id" in task:
        options["task_id"] = int(task["task_id"])
    else:
        options["task_info"] = {
            "task_name": task.get("task_name", "custom"),
            "init_ij": tuple(task["init_ij"]),
            "goal_ij": tuple(task["goal_ij"]),
            "init_xy": tuple(task["init_xy"]),
            "goal_xy": tuple(task["goal_xy"]),
        }
    ob, info = env.reset(options=options, seed=seed)
    goal = np.asarray(info["goal"], dtype=np.float32)

    if domain is None:
        gv = lambda o: np.asarray(o, dtype=np.float32)[:2]
        dist_of = lambda o: float(np.linalg.norm(gv(o) - gv(goal)))
    else:
        gv = domain.goal_vector
        dist_of = lambda o: domain.distance(o, goal)

    start_gv = gv(ob).copy()
    initial_d = dist_of(ob)
    # Baseline for partial progress. Reporting a raw subgoal fraction is misleading:
    # with nine binary buttons, a random state already matches the goal on ~50% of them,
    # so puzzle scored 0.48 while doing nothing. Progress is measured as the fraction of
    # the *remaining* subgoals that were actually closed.
    initial_frac = domain.subgoal_fraction(ob, goal) if domain is not None and domain.subgoals else float("nan")
    prev_gv = start_gv.copy()
    path_len = 0.0
    act_mag: list[float] = []

    steps = replans = nodes = 0
    success = False
    best_dist = float("inf")
    chunks: list[int] = []
    depths: list[int] = []
    traj: list[np.ndarray] = [np.asarray(ob, dtype=np.float32)] if record_trajectory else []
    best_progress = 0.0

    done = False
    while steps < max_steps and not success and not done:
        plan = planner.plan(np.asarray(ob, dtype=np.float32), goal)
        replans += 1
        nodes += plan.num_nodes
        if len(plan.actions) == 0:
            break
        chunks.append(len(plan.actions))
        depths.append(plan.selected_depth)

        for action in plan.actions:
            ob, _, terminated, truncated, info = env.step(action)
            steps += 1
            cur = gv(ob)
            path_len += float(np.linalg.norm(cur - prev_gv))
            prev_gv = cur.copy()
            act_mag.append(float(np.abs(action).mean()))
            if record_trajectory:
                traj.append(np.asarray(ob, dtype=np.float32))
            best_dist = min(best_dist, dist_of(ob))
            if domain is not None and domain.subgoals:
                best_progress = max(best_progress, domain.subgoal_fraction(ob, goal))
            if info.get("success", False):
                success = True
            if terminated or truncated:
                # The episode is over. Previously only the inner loop broke, so the outer
                # loop replanned and kept stepping an environment past its own time
                # limit. cube-single-play truncates at 200 steps while max_env_steps is
                # 500, so every episode ran 300 extra steps at one action per replan --
                # 276-313 replans instead of 32, a 30x eval slowdown, and any "success"
                # recorded after truncation was not a real success.
                done = True
                break
            if success or steps >= max_steps:
                break

    final_dist = dist_of(ob)
    extra = {}
    if domain is not None:
        from treewm.evaluation.domains import progress_metrics

        extra = progress_metrics(env, domain, ob, goal, info)
        extra["progress/best_subgoal_fraction"] = best_progress
        if np.isfinite(initial_frac):
            final_frac = domain.subgoal_fraction(ob, goal)
            room = max(1.0 - initial_frac, 1e-6)
            extra["progress/subgoal_fraction_initial"] = initial_frac
            # 0 = no better than the starting state, 1 = every remaining subgoal closed,
            # negative = actively undid subgoals that started correct.
            extra["progress/subgoal_gain"] = (final_frac - initial_frac) / room
            extra["progress/best_subgoal_gain"] = (best_progress - initial_frac) / room
    return EpisodeResult(
        success=success,
        steps=steps,
        replans=replans,
        nodes=nodes,
        final_goal_distance=final_dist,
        best_goal_distance=min(best_dist, final_dist),
        chunk_lengths=chunks,
        selected_depths=depths,
        initial_goal_distance=initial_d,
        displacement=float(np.linalg.norm(gv(ob) - start_gv)),
        path_length=path_len,
        action_magnitude=float(np.mean(act_mag)) if act_mag else 0.0,
        trajectory=traj,
        progress=extra,
    )


def evaluate(
    env,
    planner: GoalPlanner,
    tasks: list[dict],
    episodes_per_task: int = 5,
    max_steps: int = 500,
    seed: int = 0,
    domain=None,
) -> dict[str, float]:
    """Run the full task set and return the ``eval/*`` namespace."""
    results: list[EpisodeResult] = []
    start = time.perf_counter()
    for t, task in enumerate(tasks):
        for e in range(episodes_per_task):
            # Seed derived from task/episode, not from a global counter, so arms are
            # compared on identical start states.
            results.append(run_episode(env, planner, task, seed=seed + 1000 * t + e,
                                       max_steps=max_steps, domain=domain))
    elapsed = time.perf_counter() - start

    successes = np.array([r.success for r in results], dtype=np.float32)
    steps = np.array([r.steps for r in results], dtype=np.float32)
    nodes = np.array([r.nodes for r in results], dtype=np.float32)
    replans = np.array([r.replans for r in results], dtype=np.float32)
    final = np.array([r.final_goal_distance for r in results], dtype=np.float32)
    best = np.array([r.best_goal_distance for r in results], dtype=np.float32)
    chunk_lens = np.array([c for r in results for c in r.chunk_lengths], dtype=np.float32)
    sel_depths = np.array([d for r in results for d in r.selected_depths], dtype=np.float32)
    disp = np.array([r.displacement for r in results], dtype=np.float32)
    init_d = np.array([r.initial_goal_distance for r in results], dtype=np.float32)
    path_len = np.array([r.path_length for r in results], dtype=np.float32)
    act_mag = np.array([r.action_magnitude for r in results], dtype=np.float32)

    nodes_per_success = float(nodes[successes > 0].mean()) if successes.any() else float("nan")

    return {
        "eval/success_rate": float(successes.mean()),
        "eval/episode_return": float(successes.mean()),
        "eval/goal_distance_final": float(final.mean()),
        "eval/goal_distance_best": float(best.mean()),
        "eval/environment_steps": float(steps.mean()),
        "eval/replans": float(replans.mean()),
        "eval/world_model_nodes_per_replan": float((nodes / replans.clip(1)).mean()),
        "eval/world_model_nodes_total": float(nodes.mean()),
        "eval/world_model_nodes_per_success": nodes_per_success,
        "eval/action_chunk_execution_length": float(chunk_lens.mean()) if chunk_lens.size else 0.0,
        "eval/selected_leaf_depth": float(sel_depths.mean()) if sel_depths.size else 0.0,
        # Locomotion diagnostics -- on AntMaze a zero success rate is uninformative
        # unless we can also say whether the ant moved at all.
        "eval/displacement": float(disp.mean()),
        "eval/initial_goal_distance": float(init_d.mean()),
        "eval/distance_reduction": float((init_d - final).mean()),
        "eval/distance_reduction_frac": float(((init_d - final) / np.maximum(init_d, 1e-6)).mean()),
        "eval/path_length": float(path_len.mean()),
        "eval/action_magnitude": float(act_mag.mean()),
        "eval/fraction_moving": float((disp > 1.0).mean()),
        "eval/planning_wall_clock_s": float(elapsed / max(len(results), 1)),
        "eval/num_episodes": float(len(results)),
        # Domain-specific competence, averaged over episodes. These are what make a zero
        # success rate interpretable -- the failure mode that made three AntMaze cycles
        # uninformative was having only success to look at.
        **{f"eval/{k}": float(np.mean([r.progress[k] for r in results if k in r.progress]))
           for k in sorted({k for r in results for k in r.progress})},
    }


@torch.no_grad()
def sweep_budgets(
    env,
    model,
    normalizer,
    tasks: list[dict],
    budgets: list[int],
    base_tree_cfg,
    planner_cfg,
    arm: str = "treewm",
    episodes_per_task: int = 5,
    max_steps: int = 500,
    seed: int = 0,
) -> dict[int, dict[str, float]]:
    """Success vs node budget -- the primary plot.

    A single trained checkpoint is evaluated at every budget; nothing is retrained, so
    the curve isolates inference-time compute allocation.
    """
    from dataclasses import replace

    from treewm.models.baselines import tree_config_for

    out: dict[int, dict[str, float]] = {}
    for budget in budgets:
        tree_cfg = tree_config_for(arm, replace(base_tree_cfg, node_budget=budget), model)
        planner = GoalPlanner(model, normalizer, tree_cfg, planner_cfg)
        metrics = evaluate(env, planner, tasks, episodes_per_task, max_steps, seed)
        assert metrics["eval/world_model_nodes_per_replan"] <= budget + 1e-6, (
            f"arm {arm} exceeded node budget {budget}"
        )
        out[budget] = metrics
    return out
