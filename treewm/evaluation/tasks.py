"""Evaluation task sets.

``standard`` is the environment's own five tasks. ``hard`` is the primary split: start/
goal pairs drawn from the top geodesic-distance percentile, so a straight-line dash
cannot solve them and the node budget actually binds.
"""

from __future__ import annotations

from treewm.data.maze_utils import (MazeSpec, env_task_pairs, sample_goal_pairs_by_distance,
                                    sample_hard_goal_pairs)


def has_maze(env) -> bool:
    """Whether the environment has a maze topology at all (used for xy visualisation)."""
    return hasattr(getattr(env, "unwrapped", env), "maze_map")


def supports_hard_split(env) -> bool:
    """Whether we can *synthesise* start/goal pairs for this environment.

    Having a maze is not sufficient. The hard split builds a task_info containing
    ``init_ij``/``goal_ij``, which places a single body. antsoccer places two -- it
    expects ``agent_init_ij`` and ``ball_init_ij`` -- so feeding it a navigate-shaped
    task_info raises KeyError deep inside reset(). Rather than invent a ball placement
    (where should the ball start relative to the agent? any answer is a guess that
    changes the task), such environments fall back to their own built-in tasks.
    """
    if not has_maze(env):
        return False
    infos = getattr(getattr(env, "unwrapped", env), "task_infos", None)
    return bool(infos) and "init_ij" in infos[0] and "goal_ij" in infos[0]


def build_tasks(env, split: str = "hard", num_hard: int = 10, percentile: float = 75.0, seed: int = 0) -> list[dict]:
    if split == "auto":
        # Manipulation, scene and puzzle have no maze topology, so "top geodesic
        # percentile" is undefined there; their own five tasks are the correct split.
        # antsoccer has a maze but a two-body task schema we cannot synthesise.
        split = "hard" if supports_hard_split(env) else "standard"
    if split == "standard":
        return env_task_pairs(env)
    if split == "hard":
        if not supports_hard_split(env):
            raise ValueError(
                "the 'hard' split needs a maze topology and a single-body "
                "init_ij/goal_ij task schema; this environment has neither. "
                "Use split='auto' or 'standard' rather than inventing a geometric "
                "difficulty measure for a non-spatial domain."
            )
        spec = MazeSpec.from_env(env)
        return sample_hard_goal_pairs(spec, num_pairs=num_hard, percentile=percentile, seed=seed)
    if split == "both":
        return env_task_pairs(env) + build_tasks(env, "hard", num_hard, percentile, seed)
    raise ValueError(f"unknown task split {split!r}; options: auto | standard | hard | both")


def describe_tasks(tasks: list[dict]) -> str:
    lines = []
    for i, t in enumerate(tasks):
        geo = t.get("geodesic")
        geo_s = f" geodesic={geo:.0f}" if geo is not None else ""
        if "init_ij" in t and "goal_ij" in t:
            where = f"{tuple(t['init_ij'])} -> {tuple(t['goal_ij'])}"
        else:
            # Manipulation / scene / puzzle tasks are named, not positioned.
            where = f"{t.get('task_name', '?')} (task_id={t.get('task_id', '?')})"
        lines.append(f"  [{i}] {where}{geo_s}")
    return "\n".join(lines)


def build_bucketed_tasks(env, bins: list[tuple[int, float]], per_bin: int = 12,
                         seed: int = 0) -> dict[str, list[dict]]:
    """Difficulty-bucketed evaluation sets keyed by geodesic distance range."""
    return sample_goal_pairs_by_distance(MazeSpec.from_env(env), bins, per_bin, seed)
