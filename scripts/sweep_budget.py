"""Success vs node budget -- the primary plot of the project.

    python scripts/sweep_budget.py checkpoint=... budgets='[16,32,64,128,256]'

One trained checkpoint is evaluated at every budget, so the curve isolates inference-time
compute allocation with training held fixed. Node count is the compute-normalised axis;
wall-clock is recorded but secondary (spec section 25).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import matplotlib
import torch
from omegaconf import DictConfig, OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treewm.data.ogbench_dataset import load_ogbench
from treewm.evaluation.rollout import sweep_budgets
from treewm.evaluation.tasks import build_tasks
from treewm.logging.tensorboard import TreeWMLogger
from treewm.models.baselines import tree_config_for
from treewm.utils import config as cfg_utils
from treewm.utils.seeding import seed_everything
from scripts.eval import load_run


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint")
    if not checkpoint:
        raise SystemExit("usage: python scripts/sweep_budget.py checkpoint=<path> budgets='[16,32,64]'")

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    seed_everything(int(cfg.seed))

    overrides = OmegaConf.create(
        {"eval": OmegaConf.to_container(cfg.eval, resolve=True),
         "planner": OmegaConf.to_container(cfg.planner, resolve=True)}
    )
    model, normalizer, run_cfg, _ = load_run(str(checkpoint), device, overrides)
    env = load_ogbench(run_cfg.env.name, dataset_dir=run_cfg.env.dataset_dir, env_only=True)
    tasks = build_tasks(
        env, str(run_cfg.eval.task_split), int(run_cfg.eval.num_hard_tasks),
        float(run_cfg.eval.hard_percentile), int(run_cfg.eval.seed),
    )

    budgets = [int(b) for b in (cfg.get("budgets") or run_cfg.eval.budgets)]
    base_tree_cfg = cfg_utils.tree_config(run_cfg)
    print(f"[sweep] arm={run_cfg.arm} budgets={budgets}")

    results = sweep_budgets(
        env, model, normalizer, tasks, budgets, base_tree_cfg,
        cfg_utils.planner_config(run_cfg), arm=str(run_cfg.arm),
        episodes_per_task=int(run_cfg.eval.episodes_per_task),
        max_steps=int(run_cfg.planner.max_env_steps), seed=int(run_cfg.eval.seed),
    )

    run_dir = Path(checkpoint).parent.parent
    payload = {"arm": str(run_cfg.arm), "env": str(run_cfg.env.name),
               "seed": int(run_cfg.seed), "budgets": {str(b): m for b, m in results.items()}}
    out = run_dir / "budget_sweep.json"
    out.write_text(json.dumps(payload, indent=2))

    for budget, metrics in results.items():
        print(f"  budget {budget:4d}  success={metrics['eval/success_rate']:.3f}  "
              f"nodes/replan={metrics['eval/world_model_nodes_per_replan']:.1f}  "
              f"steps={metrics['eval/environment_steps']:.0f}")

    fig, ax = plt.subplots(figsize=(6, 4.2))
    xs = sorted(results)
    ax.plot(xs, [results[b]["eval/success_rate"] for b in xs], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("world-model nodes per replan (budget)")
    ax.set_ylabel("success rate")
    ax.set_title(f"{run_cfg.arm} on {run_cfg.env.short_name} ({run_cfg.eval.task_split} split)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(run_dir / "budget_sweep.png", dpi=130)

    logger = TreeWMLogger(run_dir / "sweep", is_main=True)
    for budget, metrics in results.items():
        logger.scalar("sweep/success_vs_nodes", metrics["eval/success_rate"], budget)
        logger.scalar(f"eval/success_budget_{budget}", metrics["eval/success_rate"], 0)
    logger.figure("viz/budget_sweep", fig, 0)
    logger.close()
    print(f"[sweep] wrote {out} and budget_sweep.png")


if __name__ == "__main__":
    main()
