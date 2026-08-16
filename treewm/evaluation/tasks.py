"""Evaluation task sets.

``standard`` is the environment's own five tasks. ``hard`` is the primary split: start/
goal pairs drawn from the top geodesic-distance percentile, so a straight-line dash
cannot solve them and the node budget actually binds.
"""

from __future__ import annotations

from treewm.data.maze_utils import (MazeSpec, env_task_pairs, sample_goal_pairs_by_distance,
                                    sample_hard_goal_pairs)


def build_tasks(env, split: str = "hard", num_hard: int = 10, percentile: float = 75.0, seed: int = 0) -> list[dict]:
    if split == "standard":
        return env_task_pairs(env)
    if split == "hard":
        spec = MazeSpec.from_env(env)
        return sample_hard_goal_pairs(spec, num_pairs=num_hard, percentile=percentile, seed=seed)
    if split == "both":
        return env_task_pairs(env) + build_tasks(env, "hard", num_hard, percentile, seed)
    raise ValueError(f"unknown task split {split!r}; options: standard | hard | both")


def describe_tasks(tasks: list[dict]) -> str:
    lines = []
    for i, t in enumerate(tasks):
        geo = t.get("geodesic")
        geo_s = f" geodesic={geo:.0f}" if geo is not None else ""
        lines.append(f"  [{i}] {tuple(t['init_ij'])} -> {tuple(t['goal_ij'])}{geo_s}")
    return "\n".join(lines)


def build_bucketed_tasks(env, bins: list[tuple[int, float]], per_bin: int = 12,
                         seed: int = 0) -> dict[str, list[dict]]:
    """Difficulty-bucketed evaluation sets keyed by geodesic distance range."""
    return sample_goal_pairs_by_distance(MazeSpec.from_env(env), bins, per_bin, seed)
