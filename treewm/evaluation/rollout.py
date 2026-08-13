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
    trajectory: list[np.ndarray] = field(default_factory=list)


def run_episode(
    env,
    planner: GoalPlanner,
    task: dict,
    seed: int,
    max_steps: int = 500,
    record_trajectory: bool = False,
) -> EpisodeResult:
    """One goal-reaching episode with replanning after each executed chunk."""
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

    steps = replans = nodes = 0
    success = False
    best_dist = float("inf")
    chunks: list[int] = []
    depths: list[int] = []
    traj: list[np.ndarray] = [np.asarray(ob, dtype=np.float32)] if record_trajectory else []

    while steps < max_steps and not success:
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
            if record_trajectory:
                traj.append(np.asarray(ob, dtype=np.float32))
            dist = float(np.linalg.norm(np.asarray(ob, dtype=np.float32)[:2] - goal[:2]))
            best_dist = min(best_dist, dist)
            if info.get("success", False):
                success = True
            if success or terminated or truncated or steps >= max_steps:
                break

    final_dist = float(np.linalg.norm(np.asarray(ob, dtype=np.float32)[:2] - goal[:2]))
    return EpisodeResult(
        success=success,
        steps=steps,
        replans=replans,
        nodes=nodes,
        final_goal_distance=final_dist,
        best_goal_distance=min(best_dist, final_dist),
        chunk_lengths=chunks,
        selected_depths=depths,
        trajectory=traj,
    )


def evaluate(
    env,
    planner: GoalPlanner,
    tasks: list[dict],
    episodes_per_task: int = 5,
    max_steps: int = 500,
    seed: int = 0,
) -> dict[str, float]:
    """Run the full task set and return the ``eval/*`` namespace."""
    results: list[EpisodeResult] = []
    start = time.perf_counter()
    for t, task in enumerate(tasks):
        for e in range(episodes_per_task):
            # Seed derived from task/episode, not from a global counter, so arms are
            # compared on identical start states.
            results.append(run_episode(env, planner, task, seed=seed + 1000 * t + e, max_steps=max_steps))
    elapsed = time.perf_counter() - start

    successes = np.array([r.success for r in results], dtype=np.float32)
    steps = np.array([r.steps for r in results], dtype=np.float32)
    nodes = np.array([r.nodes for r in results], dtype=np.float32)
    replans = np.array([r.replans for r in results], dtype=np.float32)
    final = np.array([r.final_goal_distance for r in results], dtype=np.float32)
    best = np.array([r.best_goal_distance for r in results], dtype=np.float32)
    chunk_lens = np.array([c for r in results for c in r.chunk_lengths], dtype=np.float32)
    sel_depths = np.array([d for r in results for d in r.selected_depths], dtype=np.float32)

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
        "eval/planning_wall_clock_s": float(elapsed / max(len(results), 1)),
        "eval/num_episodes": float(len(results)),
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
