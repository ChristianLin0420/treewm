"""Diagnostic 1 -- does novelty beat Random once kept inside the reliable horizon?

Open-loop endpoint error reaches roughly a corridor width (4.0 units) by depth ~8, and
novelty-driven expansion sits at mean depth 8-9 while Random sits at 4.7. This sweeps a
hard depth cap and a soft depth penalty over *existing checkpoints* -- no retraining.

    python scripts/depth_sweep.py --runs runs_novelty --dataset pointmaze-medium-stitch
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treewm.data.ogbench_dataset import load_ogbench
from treewm.evaluation.rollout import evaluate
from treewm.evaluation.tasks import build_tasks
from treewm.models.baselines import tree_config_for
from treewm.planning.goal_planner import GoalPlanner
from treewm.utils import config as cfg_utils
from treewm.utils.seeding import seed_everything
from scripts.eval import load_run

REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="experiments/02-novelty-target/runs/novelty")
    p.add_argument("--dataset", default="pointmaze-medium-stitch")
    p.add_argument("--arms", nargs="+", default=["noveltyq", "randomtreewm"])
    p.add_argument("--depths", nargs="+", type=int, default=[3, 4, 5, 6, 8, 16])
    p.add_argument("--penalties", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.2])
    p.add_argument("--budgets", nargs="+", type=int, default=[64, 256])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--out", default="experiments/02-novelty-target/results/novelty")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(0)

    checkpoints = [
        ck for ck in sorted((REPO / args.runs).glob("*/*/*/checkpoints/latest.pt"))
        if ck.parts[-5] == args.dataset and ck.parts[-4] in args.arms
    ]
    print(f"[depth] {len(checkpoints)} checkpoints on {args.dataset}")

    results: dict = defaultdict(lambda: defaultdict(list))
    env = tasks = None

    for ck in checkpoints:
        arm = ck.parts[-4]
        model, normalizer, cfg, _ = load_run(str(ck), device)
        if env is None:
            env = load_ogbench(cfg.env.name, dataset_dir=cfg.env.dataset_dir, env_only=True)
            tasks = build_tasks(env, str(cfg.eval.task_split), int(cfg.eval.num_hard_tasks),
                                float(cfg.eval.hard_percentile), int(cfg.eval.seed))
        base = cfg_utils.tree_config(cfg)
        planner_cfg = cfg_utils.planner_config(cfg)

        for budget in args.budgets:
            for max_depth in args.depths:
                tc = tree_config_for(arm, replace(base, node_budget=budget, max_depth=max_depth), model)
                planner = GoalPlanner(model, normalizer, tc, planner_cfg, device)
                m = evaluate(env, planner, tasks, args.episodes, int(cfg.planner.max_env_steps), 0)
                results[f"{arm}|B{budget}"][f"depth{max_depth}"].append(
                    (m["eval/success_rate"], m["eval/selected_leaf_depth"], m["eval/goal_distance_final"])
                )
            # Soft penalty only makes sense for the novelty scorer.
            if arm == "noveltyq":
                for lam in args.penalties:
                    if lam == 0.0:
                        continue
                    tc = tree_config_for(arm, replace(base, node_budget=budget, max_depth=16), model)
                    tc = replace(tc, scorer="novelty_q_penalized", depth_penalty=lam)
                    planner = GoalPlanner(model, normalizer, tc, planner_cfg, device)
                    m = evaluate(env, planner, tasks, args.episodes, int(cfg.planner.max_env_steps), 0)
                    results[f"{arm}|B{budget}"][f"penalty{lam}"].append(
                        (m["eval/success_rate"], m["eval/selected_leaf_depth"], m["eval/goal_distance_final"])
                    )
        print(f"  done {arm} {ck.parts[-3]}")

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    plain = {k: {kk: vv for kk, vv in v.items()} for k, v in results.items()}
    (out_dir / f"depth_sweep_{args.dataset}.json").write_text(json.dumps(plain, indent=2))

    for key in sorted(results):
        print(f"\n=== {key} ({args.dataset}) ===")
        print(f"{'setting':12s} {'success':>18s} {'sel_leaf_depth':>16s} {'goal_dist':>12s}")
        print("-" * 62)
        for setting in sorted(results[key], key=lambda s: (s.startswith("penalty"), s)):
            arr = np.array(results[key][setting], dtype=float)
            print(f"{setting:12s} {arr[:,0].mean():8.3f}±{arr[:,0].std():<7.3f} "
                  f"{arr[:,1].mean():16.2f} {arr[:,2].mean():12.2f}")
    print(f"\nwrote {out_dir / f'depth_sweep_{args.dataset}.json'}")


if __name__ == "__main__":
    main()
