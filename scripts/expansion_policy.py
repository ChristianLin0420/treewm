"""Expansion-policy diagnostic: is q-novelty useful as a *bonus* inside goal-directed search?

Four allocation policies on identical checkpoints (no retraining):

    random        matched-budget control
    novelty_q     min_j d_q(q_n, q_j)          -- novelty as the sole objective
    goal          -d_goal(n)                   -- decoded-position goal-only best-first
    goal_novelty  -d_goal(n) + alpha*novelty   -- novelty as a diversity bonus

The goal enters the *frontier ordering only*. The branch network, dynamics and q never
see it, so the world model itself stays goal-independent.

Reports success vs budget, best reachable (simulator-grounded) goal distance in each
tree, selected-leaf regret, coverage, mean depth, and goal progress per expansion.
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
from treewm.evaluation.coverage import StateQuantizer, unique_cells_per_row
from treewm.evaluation.grounding import ground_tree, predicted_xy, restore_state, save_state
from treewm.evaluation.rollout import evaluate
from treewm.evaluation.tasks import build_tasks
from treewm.models.baselines import tree_config_for
from treewm.planning.goal_planner import GoalPlanner
from treewm.utils import config as cfg_utils
from treewm.utils.seeding import seed_everything
from scripts.eval import load_run

REPO = Path(__file__).resolve().parents[1]


def policies(alphas: list[float]) -> list[tuple[str, str, float]]:
    out = [("random", "random", 0.0), ("novelty_q", "novelty_q", 0.0), ("goal", "goal", 0.0)]
    out += [(f"goal+a{a}", "goal_novelty", a) for a in alphas]
    return out


@torch.no_grad()
def tree_quality(model, normalizer, env, tasks, tc, quantizer, device, episodes=1, max_replans=4):
    """Grounded tree quality from real episode states.

    Returns best *reachable* goal distance, selected-leaf regret, coverage, mean depth and
    goal progress per expansion.
    """
    stats: dict[str, list[float]] = defaultdict(list)
    for t, task in enumerate(tasks):
        for e in range(episodes):
            options = {"task_id": int(task["task_id"])} if "task_id" in task else {
                "task_info": {"task_name": task.get("task_name", "custom"),
                              "init_ij": tuple(task["init_ij"]), "goal_ij": tuple(task["goal_ij"]),
                              "init_xy": tuple(task["init_xy"]), "goal_xy": tuple(task["goal_xy"])}
            }
            ob, info = env.reset(options=options, seed=1000 * t + e)
            goal = np.asarray(info["goal"], dtype=np.float32)
            goal_n = torch.from_numpy(normalizer.norm_obs(goal[None])).to(device)

            for _ in range(max_replans):
                obs_n = torch.from_numpy(
                    normalizer.norm_obs(np.asarray(ob, dtype=np.float32)[None])
                ).to(device)
                z = model.encode(obs_n)
                tree, trace = model.generate(
                    z, tc, goal_obs=goal_n if tc.scorer in ("goal", "goal_novelty") else None
                )

                valid = tree.valid[0].cpu().numpy()
                pred = predicted_xy(model, tree, normalizer)
                entry = save_state(env)
                actual, grounded = ground_tree(env, tree, normalizer)
                restore_state(env, entry)

                usable = valid.copy(); usable[0] = False
                idx = np.flatnonzero(usable & grounded)
                if idx.size == 0:
                    break
                d_pred = np.linalg.norm(pred[idx] - goal[None, :2], axis=1)
                d_act = np.linalg.norm(actual[idx] - goal[None, :2], axis=1)
                chosen = int(idx[int(np.argmin(d_pred))])

                stats["best_reachable_goal_distance"].append(float(d_act.min()))
                stats["best_predicted_goal_distance"].append(float(d_pred.min()))
                stats["selected_leaf_regret"].append(
                    float(np.linalg.norm(actual[chosen] - goal[:2]) - d_act.min())
                )
                stats["mean_depth"].append(
                    float(tree.depth[0][tree.valid[0]].float().mean().item())
                )
                cells = quantizer.cell_ids(model.decoder(tree.latent))
                cov = float(unique_cells_per_row(cells, tree.valid.float())[0].item())
                stats["coverage"].append(cov)
                if trace.best_goal_distance and len(trace.best_goal_distance) > 1:
                    b = trace.best_goal_distance
                    stats["goal_progress_per_expansion"].append(
                        (b[0] - b[-1]) / max(len(b) - 1, 1)
                    )

                # advance the episode with the chosen path's first chunk
                parent = tree.parent_index[0].cpu().numpy()
                chain, cur = [], chosen
                while cur > 0:
                    chain.append(cur); cur = int(parent[cur])
                if not chain:
                    break
                first = chain[-1]
                n = max(1, min(int(tree.action_mask[0, first].sum().item()), 16))
                acts = normalizer.denorm_act(tree.action_chunk[0, first, :n].float().cpu().numpy())
                done = False
                for a in acts:
                    ob, _, term, trunc, info = env.step(np.clip(a, -1.0, 1.0))
                    if info.get("success", False) or term or trunc:
                        done = True
                        break
                if done:
                    break
    return {k: float(np.mean(v)) for k, v in stats.items() if v}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs_novelty")
    p.add_argument("--dataset", default="pointmaze-medium-stitch")
    p.add_argument("--arm", default="randomtreewm",
                   help="checkpoint to run every policy on; the world model is shared")
    p.add_argument("--alphas", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0])
    p.add_argument("--budgets", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--quality-episodes", type=int, default=1)
    p.add_argument("--out", default="results_policy")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(0)

    checkpoints = [
        ck for ck in sorted((REPO / args.runs).glob("*/*/*/checkpoints/latest.pt"))
        if ck.parts[-5] == args.dataset and ck.parts[-4] == args.arm
    ]
    print(f"[policy] {len(checkpoints)} checkpoints ({args.arm}) on {args.dataset}")

    success: dict = defaultdict(lambda: defaultdict(list))
    quality: dict = defaultdict(lambda: defaultdict(list))
    env = tasks = None

    for ck in checkpoints:
        model, normalizer, cfg, _ = load_run(str(ck), device)
        if env is None:
            env = load_ogbench(cfg.env.name, dataset_dir=cfg.env.dataset_dir, env_only=True)
            tasks = build_tasks(env, str(cfg.eval.task_split), int(cfg.eval.num_hard_tasks),
                                float(cfg.eval.hard_percentile), int(cfg.eval.seed))
        base = cfg_utils.tree_config(cfg)
        planner_cfg = cfg_utils.planner_config(cfg)
        quantizer = StateQuantizer(resolution=float(cfg.retrieval.grid_resolution), dims=(0, 1))

        for name, scorer, alpha in policies(args.alphas):
            for budget in args.budgets:
                tc = tree_config_for(args.arm, replace(base, node_budget=budget), model)
                tc = replace(tc, scorer=scorer, alpha=alpha)
                m = evaluate(env, GoalPlanner(model, normalizer, tc, planner_cfg, device),
                             tasks, args.episodes, int(cfg.planner.max_env_steps), 0)
                success[name][budget].append(m["eval/success_rate"])
                if budget == 64:
                    q = tree_quality(model, normalizer, env, tasks[:5], tc, quantizer, device,
                                     args.quality_episodes)
                    for k, v in q.items():
                        quality[name][k].append(v)
            print(f"  {ck.parts[-3]} {name} done")

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"policy_{args.dataset}.json").write_text(json.dumps(
        {"success": {k: {str(b): v for b, v in d.items()} for k, d in success.items()},
         "quality": {k: {kk: float(np.mean(vv)) for kk, vv in d.items()} for k, d in quality.items()}},
        indent=2))

    print(f"\n=== success vs node budget ({args.dataset}, {args.arm} checkpoints) ===")
    print(f"{'policy':14s} " + " ".join(f"{b:>7d}" for b in args.budgets) + "     AUC")
    print("-" * 62)
    for name, _, _ in policies(args.alphas):
        row = [float(np.mean(success[name][b])) for b in args.budgets]
        print(f"{name:14s} " + " ".join(f"{x:7.3f}" for x in row) + f"   {np.mean(row):6.3f}")

    print(f"\n=== tree quality at budget 64 ({args.dataset}) ===")
    cols = ["best_reachable_goal_distance", "best_predicted_goal_distance",
            "selected_leaf_regret", "coverage", "mean_depth", "goal_progress_per_expansion"]
    hdr = ["best_reach_d", "best_pred_d", "regret", "coverage", "depth", "goal_prog/exp"]
    print(f"{'policy':14s} " + " ".join(f"{h:>14s}" for h in hdr))
    print("-" * 100)
    for name, _, _ in policies(args.alphas):
        cells = []
        for c in cols:
            vals = quality[name].get(c, [])
            cells.append(f"{np.mean(vals):14.3f}" if len(vals) else f"{'n/a':>14s}")
        print(f"{name:14s} " + " ".join(cells))
    print(f"\nwrote {out_dir / f'policy_{args.dataset}.json'}")


if __name__ == "__main__":
    main()
