"""Closed-loop goal-reaching evaluation.

Reports success against *generated world-model nodes*, which is the compute-normalised
axis of the whole project: every arm is run at the same node budget, and wall-clock is
recorded as a secondary quantity (spec section 25).

Evaluation is deterministic given a seed: episode seeds are derived from the task index
so that two arms see the same start states.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

from treewm.planning.goal_planner import GoalPlanner


class EvaluationInterrupted(RuntimeError):
    """A deferred lifecycle request interrupted evaluation at a safe boundary."""


SEED_TABLE_SCHEMA_VERSION = 1
MAX_ENVIRONMENT_SEED = 2**31 - 1
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _seed_table_sha256(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body.pop("sha256", None)
    return hashlib.sha256(_canonical_json(body).encode("utf-8")).hexdigest()


def _protocol_seed(
    protocol_sha256: str,
    split: str,
    task_id: int,
    episode_index: int,
    nonce: int,
) -> int:
    digest = hashlib.sha256(
        _canonical_json(
            {
                "protocol_sha256": protocol_sha256,
                "split": split,
                "task_id": int(task_id),
                "episode_index": int(episode_index),
                "nonce": int(nonce),
            }
        ).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % MAX_ENVIRONMENT_SEED


def _build_seed_table(
    protocol_sha256: str,
    split: str,
    task_ids: Sequence[int],
    episodes_per_task: int,
    used: set[int],
) -> dict[str, Any]:
    if split not in {"monitor", "final"}:
        raise ValueError(f"unsupported evaluation seed split: {split!r}")
    if episodes_per_task <= 0:
        raise ValueError("episodes_per_task must be positive")
    rows: list[list[int]] = []
    for task_id in task_ids:
        row: list[int] = []
        for episode_index in range(episodes_per_task):
            nonce = 0
            value = _protocol_seed(
                protocol_sha256,
                split,
                int(task_id),
                episode_index,
                nonce,
            )
            while value in used:
                nonce += 1
                value = _protocol_seed(
                    protocol_sha256,
                    split,
                    int(task_id),
                    episode_index,
                    nonce,
                )
            used.add(value)
            row.append(value)
        rows.append(row)
    table: dict[str, Any] = {
        "schema_version": SEED_TABLE_SCHEMA_VERSION,
        "split": split,
        "protocol_sha256": protocol_sha256,
        "task_ids": [int(value) for value in task_ids],
        "episodes_per_task": int(episodes_per_task),
        "seeds": rows,
    }
    table["sha256"] = _seed_table_sha256(table)
    return table


def validate_evaluation_seed_table(
    table: Mapping[str, Any],
    *,
    split: str | None = None,
    task_ids: Sequence[int] | None = None,
    episodes_per_task: int | None = None,
) -> None:
    """Fail closed on an explicit, protocol-bound episode seed table."""
    if table.get("schema_version") != SEED_TABLE_SCHEMA_VERSION:
        raise ValueError("evaluation seed-table schema differs")
    if _SHA256.fullmatch(str(table.get("protocol_sha256", ""))) is None:
        raise ValueError("evaluation seed table is not protocol-bound")
    if table.get("sha256") != _seed_table_sha256(table):
        raise ValueError("evaluation seed-table hash differs")
    if table.get("split") not in {"monitor", "final"}:
        raise ValueError("evaluation seed-table split is invalid")
    if split is not None and table.get("split") != split:
        raise ValueError("evaluation seed-table split differs")
    expected_tasks = (
        [int(value) for value in task_ids]
        if task_ids is not None
        else [int(value) for value in table.get("task_ids", [])]
    )
    if table.get("task_ids") != expected_tasks or not expected_tasks:
        raise ValueError("evaluation seed-table tasks differ")
    expected_episodes = (
        int(episodes_per_task)
        if episodes_per_task is not None
        else int(table.get("episodes_per_task", -1))
    )
    if int(table.get("episodes_per_task", -1)) != expected_episodes or expected_episodes <= 0:
        raise ValueError("evaluation seed-table episode count differs")
    rows = table.get("seeds")
    if (
        not isinstance(rows, list)
        or len(rows) != len(expected_tasks)
        or any(not isinstance(row, list) or len(row) != expected_episodes for row in rows)
    ):
        raise ValueError("evaluation seed-table shape differs")
    flattened = [value for row in rows for value in row]
    if any(
        not isinstance(value, int) or isinstance(value, bool)
        or not 0 <= value < MAX_ENVIRONMENT_SEED
        for value in flattened
    ):
        raise ValueError("evaluation seed-table value is invalid")
    if len(set(flattened)) != len(flattened):
        raise ValueError("evaluation seed table contains duplicate episode seeds")


def build_evaluation_seed_tables(
    protocol_sha256: str,
    training_seed: int,
    task_ids: Sequence[int],
    monitor_episodes_per_task: int,
    final_episodes_per_task: int,
) -> dict[str, Any]:
    """Build protocol-bound, disjoint monitor/final episode banks.

    The training seed is recorded for provenance but deliberately excluded from episode
    derivation: every model seed and evaluation rail within a sealed campaign must see
    paired environment episodes. Collision handling is deterministic within the union
    of monitor and final banks.
    """
    protocol = str(protocol_sha256)
    if _SHA256.fullmatch(protocol) is None:
        raise ValueError("evaluation seed tables require a lowercase SHA256 protocol")
    tasks = [int(value) for value in task_ids]
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("evaluation seed-table task IDs must be unique and nonempty")
    used: set[int] = set()
    monitor = _build_seed_table(
        protocol,
        "monitor",
        tasks,
        int(monitor_episodes_per_task),
        used,
    )
    final = _build_seed_table(
        protocol,
        "final",
        tasks,
        int(final_episodes_per_task),
        used,
    )
    bundle: dict[str, Any] = {
        "schema_version": SEED_TABLE_SCHEMA_VERSION,
        "protocol_sha256": protocol,
        "training_seed": int(training_seed),
        "monitor": monitor,
        "final": final,
    }
    bundle["sha256"] = _seed_table_sha256(bundle)
    return bundle


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
    no_action_plans: int = 0
    guard_plans: int = 0
    guard_rejections: int = 0
    guard_candidate_count: int = 0
    guard_accepted_count: int = 0
    guard_best_predicted_improvements: list[float] = field(default_factory=list)
    guard_selected_predicted_improvements: list[float] = field(default_factory=list)
    trajectory: list[np.ndarray] = field(default_factory=list)
    progress: dict = field(default_factory=dict)
    task_index: int = -1
    task_id: int = -1
    episode_index: int = -1
    episode_seed: int = -1
    planning_wall_clock_s: float = 0.0


def episode_result_to_dict(result: EpisodeResult) -> dict[str, Any]:
    """JSON-safe representation used by resumable terminal evaluation."""
    def convert(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        return value

    return convert(asdict(result))


def episode_result_from_dict(value: dict[str, Any]) -> EpisodeResult:
    restored = dict(value)
    restored["trajectory"] = [np.asarray(x, dtype=np.float32) for x in restored.get("trajectory", [])]
    return EpisodeResult(**restored)


def run_episode(
    env,
    planner: GoalPlanner,
    task: dict,
    seed: int,
    max_steps: int = 500,
    record_trajectory: bool = False,
    domain=None,
    stop_callback: Callable[[], bool] | None = None,
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
    no_action_plans = 0
    guard_plans = guard_rejections = 0
    guard_candidate_count = guard_accepted_count = 0
    guard_best_predicted_improvements: list[float] = []
    guard_selected_predicted_improvements: list[float] = []
    # Some fixed tasks can reset directly into a solved state. A planner that correctly
    # returns no action under the non-regression guard must not turn that into a false
    # failure merely because success was historically checked only after env.step().
    success = bool(info.get("success", False))
    best_dist = initial_d
    chunks: list[int] = []
    depths: list[int] = []
    traj: list[np.ndarray] = [np.asarray(ob, dtype=np.float32)] if record_trajectory else []
    best_progress = float(initial_frac) if np.isfinite(initial_frac) else 0.0

    done = False
    while steps < max_steps and not success and not done:
        if stop_callback is not None and stop_callback():
            raise EvaluationInterrupted("evaluation interrupted before planning")
        plan = planner.plan(np.asarray(ob, dtype=np.float32), goal)
        replans += 1
        nodes += plan.num_nodes
        if plan.guard_applied:
            guard_plans += 1
            guard_rejections += int(plan.guard_rejected_all)
            guard_candidate_count += int(plan.guard_candidate_count)
            guard_accepted_count += int(plan.guard_accepted_count)
            if plan.guard_candidate_count > 0:
                guard_best_predicted_improvements.append(
                    float(plan.guard_best_predicted_improvement)
                )
            if plan.selected_node != 0:
                guard_selected_predicted_improvements.append(
                    float(plan.guard_selected_predicted_improvement)
                )
        if len(plan.actions) == 0:
            no_action_plans += 1
            break
        chunks.append(len(plan.actions))
        depths.append(plan.selected_depth)

        for action in plan.actions:
            if stop_callback is not None and stop_callback():
                raise EvaluationInterrupted("evaluation interrupted before environment step")
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
        no_action_plans=no_action_plans,
        guard_plans=guard_plans,
        guard_rejections=guard_rejections,
        guard_candidate_count=guard_candidate_count,
        guard_accepted_count=guard_accepted_count,
        guard_best_predicted_improvements=guard_best_predicted_improvements,
        guard_selected_predicted_improvements=(
            guard_selected_predicted_improvements
        ),
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
    stop_callback: Callable[[], bool] | None = None,
    completed_results: list[dict[str, Any]] | None = None,
    episode_callback: Callable[[dict[str, Any]], None] | None = None,
    episode_seed_table: Mapping[str, Any] | None = None,
    expected_episode_seed_split: str | None = None,
) -> dict[str, float]:
    """Run the full task set and return the ``eval/*`` namespace."""
    task_ids = [int(task.get("task_id", index + 1)) for index, task in enumerate(tasks)]
    if episode_seed_table is not None:
        validate_evaluation_seed_table(
            episode_seed_table,
            split=expected_episode_seed_split,
            task_ids=task_ids,
            episodes_per_task=episodes_per_task,
        )
        episode_seeds = episode_seed_table["seeds"]
    else:
        if expected_episode_seed_split is not None:
            raise ValueError("an expected seed split requires an explicit episode seed table")
        episode_seeds = [
            [
                int(seed) + 1000 * task_index + episode_index
                for episode_index in range(episodes_per_task)
            ]
            for task_index in range(len(tasks))
        ]
    results = [episode_result_from_dict(value) for value in (completed_results or [])]
    expected_prefix = [
        (t, int(task.get("task_id", t + 1)), e, int(episode_seeds[t][e]))
        for t, task in enumerate(tasks)
        for e in range(episodes_per_task)
    ]
    actual_prefix = [
        (r.task_index, r.task_id, r.episode_index, r.episode_seed) for r in results
    ]
    if actual_prefix != expected_prefix[: len(actual_prefix)]:
        raise ValueError("completed evaluation results are not a valid deterministic prefix")
    for t, task in enumerate(tasks):
        for e in range(episodes_per_task):
            ordinal = t * episodes_per_task + e
            if ordinal < len(results):
                continue
            if stop_callback is not None and stop_callback():
                raise EvaluationInterrupted("evaluation interrupted between episodes")
            # Seed derived from task/episode, not from a global counter, so arms are
            # compared on identical start states.
            episode_start = time.perf_counter()
            result = run_episode(
                env,
                planner,
                task,
                seed=int(episode_seeds[t][e]),
                max_steps=max_steps,
                domain=domain,
                stop_callback=stop_callback,
            )
            result.task_index = t
            result.task_id = int(task.get("task_id", t + 1))
            result.episode_index = e
            result.episode_seed = int(episode_seeds[t][e])
            result.planning_wall_clock_s = time.perf_counter() - episode_start
            results.append(result)
            if episode_callback is not None:
                episode_callback(episode_result_to_dict(result))
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
    no_action_plans = sum(r.no_action_plans for r in results)
    guard_plans = sum(r.guard_plans for r in results)
    guard_rejections = sum(r.guard_rejections for r in results)
    guard_candidates = sum(r.guard_candidate_count for r in results)
    guard_accepted = sum(r.guard_accepted_count for r in results)
    guard_best_improvements = np.asarray(
        [
            value
            for result in results
            for value in result.guard_best_predicted_improvements
        ],
        dtype=np.float32,
    )
    guard_selected_improvements = np.asarray(
        [
            value
            for result in results
            for value in result.guard_selected_predicted_improvements
        ],
        dtype=np.float32,
    )

    has_success = bool(successes.any())
    nodes_per_success = float(nodes[successes > 0].mean()) if has_success else 0.0

    metrics = {
        "eval/success_rate": float(successes.mean()),
        "eval/successes": float(successes.sum()),
        "eval/episode_return": float(successes.mean()),
        "eval/goal_distance_final": float(final.mean()),
        "eval/goal_distance_best": float(best.mean()),
        "eval/environment_steps": float(steps.mean()),
        "eval/replans": float(replans.mean()),
        "eval/world_model_nodes_per_replan": float((nodes / replans.clip(1)).mean()),
        "eval/world_model_nodes_total": float(nodes.mean()),
        "eval/world_model_nodes_per_success": nodes_per_success,
        "eval/world_model_nodes_per_success_defined": float(has_success),
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
        # A no-action plan ends the episode immediately. Keep this distinct from the
        # guard-specific rejection rate so a root-only/empty tree is diagnosable too.
        "eval/no_action_plan_rate": float(no_action_plans / max(float(replans.sum()), 1.0)),
        "eval/no_action_episode_fraction": float(
            np.mean([result.no_action_plans > 0 for result in results])
        ),
        "eval/guard/plan_fraction": float(guard_plans / max(float(replans.sum()), 1.0)),
        "eval/guard/rejection_rate": float(
            guard_rejections / max(float(guard_plans), 1.0)
        ),
        "eval/guard/candidate_acceptance_rate": float(
            guard_accepted / max(float(guard_candidates), 1.0)
        ),
        "eval/guard/best_predicted_executable_improvement": float(
            guard_best_improvements.mean() if guard_best_improvements.size else 0.0
        ),
        "eval/guard/selected_predicted_executable_improvement": float(
            guard_selected_improvements.mean()
            if guard_selected_improvements.size
            else 0.0
        ),
        "eval/guard/plans": float(guard_plans),
        "eval/guard/rejections": float(guard_rejections),
        "eval/planning_wall_clock_s": float(
            np.mean([result.planning_wall_clock_s for result in results])
        ),
        "eval/num_episodes": float(len(results)),
        # Domain-specific competence, averaged over episodes. These are what make a zero
        # success rate interpretable -- the failure mode that made three AntMaze cycles
        # uninformative was having only success to look at.
        **{f"eval/{k}": float(np.mean([r.progress[k] for r in results if k in r.progress]))
           for k in sorted({k for r in results for k in r.progress})},
    }
    for t, task in enumerate(tasks):
        task_id = int(task.get("task_id", t + 1))
        task_results = [r for r in results if r.task_index == t]
        label = f"task{task_id}"
        metrics[f"eval/{label}/success_rate"] = float(
            np.mean([r.success for r in task_results])
        )
        metrics[f"eval/{label}/successes"] = float(sum(r.success for r in task_results))
        metrics[f"eval/{label}/num_episodes"] = float(len(task_results))
        for key in sorted({key for r in task_results for key in r.progress}):
            metrics[f"eval/{label}/{key}"] = float(
                np.mean([r.progress[key] for r in task_results if key in r.progress])
            )
    return metrics


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
    domain=None,
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
        planner = GoalPlanner(model, normalizer, tree_cfg, planner_cfg, domain=domain)
        metrics = evaluate(
            env, planner, tasks, episodes_per_task, max_steps, seed, domain=domain
        )
        guard_predictions = (
            int(model.cfg.branch_factor)
            if bool(getattr(planner_cfg, "require_first_edge_improvement", False))
            else 0
        )
        effective_bound = budget + guard_predictions
        assert metrics["eval/world_model_nodes_per_replan"] <= effective_bound + 1e-6, (
            f"arm {arm} exceeded effective prediction budget {effective_bound} "
            f"(tree={budget}, guard={guard_predictions})"
        )
        out[budget] = metrics
    return out
