"""Render generated trees, heatmaps and a planning example from a checkpoint.

    python scripts/visualize_tree.py checkpoint=... tree.node_budget=64

Writes PNGs next to the checkpoint and mirrors them into a TensorBoard ``viz/`` run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treewm.data.maze_utils import MazeSpec
from treewm.data.ogbench_dataset import load_ogbench
from treewm.evaluation import diagnostics as diag
from treewm.evaluation.tasks import build_tasks
from treewm.logging.tensorboard import TreeWMLogger
from treewm.models.baselines import tree_config_for
from treewm.planning.goal_planner import GoalPlanner
from treewm.utils import config as cfg_utils
from treewm.utils.seeding import seed_everything
from scripts.eval import load_run


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint")
    if not checkpoint:
        raise SystemExit("usage: python scripts/visualize_tree.py checkpoint=<path>")

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    seed_everything(int(cfg.seed))
    overrides = OmegaConf.create({"tree": OmegaConf.to_container(cfg.tree, resolve=True)})
    model, normalizer, run_cfg, _ = load_run(str(checkpoint), device, overrides)

    env = load_ogbench(run_cfg.env.name, dataset_dir=run_cfg.env.dataset_dir, env_only=True)
    maze_spec = MazeSpec.from_env(env)
    tree_cfg = tree_config_for(run_cfg.arm, cfg_utils.tree_config(run_cfg), model)

    out_dir = Path(checkpoint).parent.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = TreeWMLogger(Path(checkpoint).parent.parent / "viz", is_main=True)

    if model.decoder is None:
        raise SystemExit("visualisation requires model.reconstruction=true at training time")

    # Heatmaps over all valid maze positions.
    for name, fn in (
        ("branching_factor_heatmap", diag.branching_factor_heatmap),
        ("expansion_gain_heatmap", diag.expansion_gain_heatmap),
    ):
        fig = fn(model, maze_spec, normalizer, device)
        fig.savefig(out_dir / f"{name}.png", dpi=130)
        logger.figure(f"viz/{name}", fig, 0)
        print(f"[viz] {out_dir / (name + '.png')}")

    # Trees from a few fixed evaluation states -- fixed so runs are comparable.
    free = maze_spec.free_cells()
    picks = free[:: max(1, len(free) // 4)][:4]
    for i, (ci, cj) in enumerate(picks):
        xy = maze_spec.ij_to_xy(int(ci), int(cj))[None]
        obs = torch.from_numpy(normalizer.norm_obs(xy)).to(device)
        with torch.no_grad():
            tree, _ = model.generate(model.encode(obs), tree_cfg)
        fig = diag.tree_plot(model, tree, normalizer, maze_spec, 0, f"tree from cell {(int(ci), int(cj))}")
        fig.savefig(out_dir / f"tree_example_{i}.png", dpi=130)
        logger.figure(f"viz/tree_example_{i}", fig, 0)
        print(f"[viz] tree_example_{i}: {int(tree.num_nodes[0])} nodes")

    # One planning example: tree + goal + selected leaf + executed prefix.
    tasks = build_tasks(env, str(run_cfg.eval.task_split), int(run_cfg.eval.num_hard_tasks),
                        float(run_cfg.eval.hard_percentile), int(run_cfg.eval.seed))
    planner = GoalPlanner(model, normalizer, tree_cfg, cfg_utils.planner_config(run_cfg), device)
    task = tasks[0]
    options = {"task_id": int(task["task_id"])} if "task_id" in task else {
        "task_info": {"task_name": task.get("task_name", "custom"),
                      "init_ij": tuple(task["init_ij"]), "goal_ij": tuple(task["goal_ij"]),
                      "init_xy": tuple(task["init_xy"]), "goal_xy": tuple(task["goal_xy"])}
    }
    ob, info = env.reset(options=options, seed=int(run_cfg.eval.seed))
    goal = np.asarray(info["goal"], dtype=np.float32)
    plan = planner.plan(np.asarray(ob, dtype=np.float32), goal, return_tree=True)

    executed = [np.asarray(ob, dtype=np.float32)]
    for action in plan.actions:
        ob, _, term, trunc, _ = env.step(action)
        executed.append(np.asarray(ob, dtype=np.float32))
        if term or trunc:
            break

    chain = plan.tree.path_to_root(torch.tensor([plan.selected_node], device=device))
    path_nodes = sorted({int(c.item()) for c in chain})
    fig = diag.planning_plot(model, plan.tree, normalizer, maze_spec, ob, goal, path_nodes, executed)
    fig.savefig(out_dir / "planning_example.png", dpi=130)
    logger.figure("viz/planning_example", fig, 0)
    logger.close()
    print(f"[viz] planning_example.png (selected node {plan.selected_node}, "
          f"{plan.num_nodes} nodes, goal distance {plan.goal_distance:.3f})")


if __name__ == "__main__":
    main()
