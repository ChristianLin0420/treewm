"""Diagnostic 2 -- is long-horizon endpoint error the bottleneck?

Generates exactly the trees the planner would generate, then executes every node's
root-to-node chunks in the simulator to learn where each node *actually* lands. Leaf
selection is then run three ways on the identical tree:

    latent     z-distance to the goal latent          (what the deployed planner does)
    predicted  decoded predicted position vs goal xy  (isolates decode vs latent)
    oracle     ACTUAL grounded endpoint vs goal xy    (diagnosis only -- privileged)

If oracle >> latent, the tree already contains reachable futures that reach the goal and
selection cannot identify them: endpoint error is the bottleneck. If oracle is also flat,
the tree does not contain useful futures and maximising coverage is the wrong objective.

    python scripts/oracle_selection.py --runs runs_novelty --dataset pointmaze-medium-stitch
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
from treewm.evaluation.grounding import (
    ground_tree,
    predicted_xy,
    restore_state,
    save_state,
    selection_disagreement,
)
from treewm.evaluation.tasks import build_tasks
from treewm.models.baselines import tree_config_for
from treewm.utils import config as cfg_utils
from treewm.utils.seeding import seed_everything
from scripts.eval import load_run

REPO = Path(__file__).resolve().parents[1]


def episode(env, model, normalizer, tree_cfg, task, goal_seed, mode, max_steps, device, stats,
            ground_on_latent: bool = True):
    """One closed-loop episode. ``mode`` in {latent, predicted, oracle}."""
    options = {"task_id": int(task["task_id"])} if "task_id" in task else {
        "task_info": {"task_name": task.get("task_name", "custom"),
                      "init_ij": tuple(task["init_ij"]), "goal_ij": tuple(task["goal_ij"]),
                      "init_xy": tuple(task["init_xy"]), "goal_xy": tuple(task["goal_xy"])}
    }
    ob, info = env.reset(options=options, seed=goal_seed)
    goal = np.asarray(info["goal"], dtype=np.float32)
    steps, success = 0, False
    depths: list[int] = []

    while steps < max_steps and not success:
        obs_t = torch.from_numpy(normalizer.norm_obs(np.asarray(ob, dtype=np.float32)[None])).to(device)
        with torch.no_grad():
            z = model.encode(obs_t)
            z_goal = model.encode(
                torch.from_numpy(normalizer.norm_obs(goal[None])).to(device)
            )
            tree, _ = model.generate(z, tree_cfg)

        valid = tree.valid[0].cpu().numpy()
        pred_xy = predicted_xy(model, tree, normalizer)

        # Grounding is the expensive part (a full simulator replay of every node), so it
        # runs only where needed: for oracle selection, and once on the latent pass to
        # collect the predicted-vs-actual diagnostics.
        if mode == "oracle" or (mode == "latent" and ground_on_latent):
            entry = save_state(env)
            actual_xy, grounded = ground_tree(env, tree, normalizer)
            restore_state(env, entry)
        else:
            actual_xy, grounded = pred_xy, valid.copy()

        info_sel = selection_disagreement(pred_xy, actual_xy, grounded, valid, goal) if mode == "latent" else {}
        if info_sel:
            for k in ("disagree", "min_predicted_goal_distance", "min_actual_goal_distance",
                      "realised_goal_distance_of_predicted_choice", "regret"):
                stats[k].append(info_sel[k])

        usable = valid.copy()
        usable[0] = False
        idx = np.flatnonzero(usable & grounded)
        if idx.size == 0:
            break

        if mode == "latent":
            with torch.no_grad():
                d = torch.linalg.vector_norm(tree.latent[0] - z_goal[0], dim=-1).cpu().numpy()
            best = int(idx[int(np.argmin(d[idx]))])
        elif mode == "predicted":
            best = int(idx[int(np.argmin(np.linalg.norm(pred_xy[idx] - goal[None, :2], axis=1)))])
        else:
            best = int(idx[int(np.argmin(np.linalg.norm(actual_xy[idx] - goal[None, :2], axis=1)))])

        depths.append(int(tree.depth[0, best].item()))

        # Execute the first chunk on the path from the root to the selected node.
        chain, cur = [], best
        parent = tree.parent_index[0].cpu().numpy()
        while cur > 0:
            chain.append(cur)
            cur = int(parent[cur])
        chain.reverse()
        first = chain[0]
        n_steps = int(tree.action_mask[0, first].sum().item())
        n_steps = max(1, min(n_steps, 16))
        actions = normalizer.denorm_act(
            tree.action_chunk[0, first, :n_steps].float().cpu().numpy()
        )
        for action in actions:
            ob, _, term, trunc, info = env.step(np.clip(action, -1.0, 1.0))
            steps += 1
            if info.get("success", False):
                success = True
            if success or term or trunc or steps >= max_steps:
                break

    return success, steps, float(np.mean(depths)) if depths else 0.0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="experiments/02-novelty-target/runs/novelty")
    p.add_argument("--dataset", default="pointmaze-medium-stitch")
    p.add_argument("--arms", nargs="+", default=["noveltyq", "randomtreewm"])
    p.add_argument("--budgets", nargs="+", type=int, default=[64, 256])
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--seed-tag", default="seed0")
    p.add_argument("--max-tasks", type=int, default=6)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--modes", nargs="+", default=["latent", "predicted", "oracle"])
    p.add_argument("--seed-tags", nargs="+", default=None)
    p.add_argument("--out", default="experiments/02-novelty-target/results/novelty")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(0)

    checkpoints = [
        ck for ck in sorted((REPO / args.runs).glob("*/*/*/checkpoints/latest.pt"))
        if ck.parts[-5] == args.dataset and ck.parts[-4] in args.arms
        and (any(t in ck.parts[-3] for t in args.seed_tags) if args.seed_tags else args.seed_tag in ck.parts[-3])
    ]
    print(f"[oracle] {len(checkpoints)} checkpoints on {args.dataset}")

    table: dict = defaultdict(list)
    diag: dict = defaultdict(lambda: defaultdict(list))
    env = tasks = None

    for ck in checkpoints:
        arm = ck.parts[-4]
        model, normalizer, cfg, _ = load_run(str(ck), device)
        if env is None:
            env = load_ogbench(cfg.env.name, dataset_dir=cfg.env.dataset_dir, env_only=True)
            tasks = build_tasks(env, str(cfg.eval.task_split), int(cfg.eval.num_hard_tasks),
                                float(cfg.eval.hard_percentile), int(cfg.eval.seed))[: args.max_tasks]
        base = cfg_utils.tree_config(cfg)

        for budget in args.budgets:
            tc = tree_config_for(arm, replace(base, node_budget=budget), model)
            for mode in args.modes:
                stats: dict = defaultdict(list)
                succ, dep = [], []
                for t, task in enumerate(tasks):
                    for e in range(args.episodes):
                        s, _, d = episode(env, model, normalizer, tc, task,
                                          1000 * t + e, mode, args.max_steps, device, stats,
                                          ground_on_latent="oracle" in args.modes)
                        succ.append(float(s)); dep.append(d)
                table[f"{arm}|B{budget}|{mode}"].append((float(np.mean(succ)), float(np.mean(dep))))
                for k, v in stats.items():
                    diag[f"{arm}|B{budget}"][k].extend(v)
            print(f"  {arm} B{budget} done")

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "success": {k: v for k, v in table.items()},
        "diagnostics": {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in diag.items()},
    }
    (out_dir / f"oracle_{args.dataset}.json").write_text(json.dumps(payload, indent=2))

    print(f"\n=== leaf-selection mode vs success ({args.dataset}) ===")
    print(f"{'arm/budget':22s} " + " ".join(f"{m:>16s}" for m in args.modes))
    print("-" * 74)
    keys = sorted({k.rsplit("|", 1)[0] for k in table})
    for key in keys:
        cells = []
        for mode in args.modes:
            arr = np.array(table.get(f"{key}|{mode}", []), dtype=float)
            cells.append(f"{arr[:,0].mean():7.3f}±{arr[:,0].std():<6.3f}" if arr.size else "n/a")
        print(f"{key:22s} " + " ".join(f"{c:>16s}" for c in cells))

    print(f"\n=== grounding diagnostics ({args.dataset}) ===")
    hdr = ["disagree", "min_predicted_goal_distance", "min_actual_goal_distance",
           "realised_goal_distance_of_predicted_choice", "regret"]
    print(f"{'arm/budget':22s} {'disagree':>9s} {'min_pred_d':>11s} {'min_act_d':>10s} "
          f"{'realised':>9s} {'regret':>8s}")
    print("-" * 74)
    for key in sorted(diag):
        d = diag[key]
        print(f"{key:22s} " + " ".join(
            f"{np.mean(d[h]):>9.3f}" if h in d else f"{'n/a':>9s}" for h in hdr))
    print(f"\nwrote {out_dir / f'oracle_{args.dataset}.json'}")


if __name__ == "__main__":
    main()
