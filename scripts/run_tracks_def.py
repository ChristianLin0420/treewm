"""Tracks D / E / F on existing checkpoints -- no retraining.

D  expansion policy      random, bfs, depth_balanced, root_quota, goal, diverse_goal,
                         broad_to_focused                    (RandomTreeWM is mandatory)
E  execution cadence     1 / 4 / 8 primitive actions, clipped chunk, full chunk
F  planner interface     endpoint, path_aware (two lambdas), ancestor

Every artifact is written with full provenance so results cannot later be merged across
incompatible settings.
"""

from __future__ import annotations

import argparse
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
from treewm.evaluation.tree_stats import structural_summary
from treewm.models.baselines import tree_config_for
from treewm.planning.goal_planner import GoalPlanner
from treewm.utils import config as cfg_utils
from treewm.utils.provenance import provenance, write_artifact
from treewm.utils.rng import make_generator
from treewm.utils.seeding import seed_everything
from scripts.eval import load_run

REPO = Path(__file__).resolve().parents[1]

D_POLICIES = ["random", "bfs", "depth_balanced", "root_quota", "goal", "diverse_goal",
              "broad_to_focused"]
E_CADENCE = [("fixed", 1), ("fixed", 4), ("fixed", 8), ("clipped", 16), ("full", 64)]
F_MODES = [("endpoint", 0.0), ("path_aware", 0.005), ("path_aware", 0.02), ("ancestor", 0.5)]


@torch.no_grad()
def tree_shape(model, normalizer, env, tasks, tc, device, n=6):
    """Structural summary from real start states under this policy."""
    from treewm.data.maze_utils import MazeSpec
    from treewm.evaluation.tree_viz import build_anchors

    anchors = build_anchors(MazeSpec.from_env(env), num=n)
    obs = torch.from_numpy(normalizer.norm_obs(anchors.starts)).to(device)
    goal = torch.from_numpy(normalizer.norm_obs(anchors.goals)).to(device)
    z = model.encode(obs)
    from treewm.tree.frontier import GOAL_AWARE_SCORERS

    tree, _ = model.generate(z, tc, generator=make_generator(0, 'viz', device),
                             goal_obs=goal if tc.scorer in GOAL_AWARE_SCORERS else None)
    return structural_summary(tree, model, normalizer)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="experiments/02-novelty-target/runs/novelty")
    p.add_argument("--dataset", default="pointmaze-medium-stitch")
    p.add_argument("--arm", default="randomtreewm")
    p.add_argument("--budgets", nargs="+", type=int, default=[32, 64, 128])
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max-tasks", type=int, default=8)
    p.add_argument("--tracks", nargs="+", default=["D", "E", "F"])
    p.add_argument("--out", default="results_tracks")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(0)
    cks = [c for c in sorted((REPO / args.runs).glob("*/*/*/checkpoints/latest.pt"))
           if c.parts[-5] == args.dataset and c.parts[-4] == args.arm]
    print(f"[def] {len(cks)} checkpoints ({args.arm}) on {args.dataset}", flush=True)

    res: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    shapes: dict[str, list] = defaultdict(list)
    env = tasks = None

    for ck in cks:
        model, normalizer, cfg, _ = load_run(str(ck), device)
        if env is None:
            env = load_ogbench(cfg.env.name, dataset_dir=cfg.env.dataset_dir, env_only=True)
            tasks = build_tasks(env, "hard", int(cfg.eval.num_hard_tasks),
                                float(cfg.eval.hard_percentile), 0)[: args.max_tasks]
        base = cfg_utils.tree_config(cfg)
        pcfg = cfg_utils.planner_config(cfg)

        if "D" in args.tracks:
            for pol in D_POLICIES:
                for b in args.budgets:
                    tc = replace(tree_config_for(args.arm, replace(base, node_budget=b), model),
                                 scorer=pol)
                    m = evaluate(env, GoalPlanner(model, normalizer, tc, pcfg, device),
                                 tasks, args.episodes, int(cfg.planner.max_env_steps), 0)
                    res[f"D|{pol}"][str(b)].append(m["eval/success_rate"])
                    if b == 64:
                        res[f"D|{pol}"]["leaf_depth"].append(m["eval/selected_leaf_depth"])
                        res[f"D|{pol}"]["goal_dist"].append(m["eval/goal_distance_final"])
                        shapes[pol].append(tree_shape(model, normalizer, env, tasks, tc, device))
            print(f"  {ck.parts[-3]} track D done", flush=True)

        if "E" in args.tracks:
            tc = tree_config_for(args.arm, replace(base, node_budget=64), model)
            for mode, steps in E_CADENCE:
                pc = replace(pcfg, execute_mode=mode, execute_steps=steps)
                m = evaluate(env, GoalPlanner(model, normalizer, tc, pc, device),
                             tasks, args.episodes, int(cfg.planner.max_env_steps), 0)
                key = f"E|{mode}{steps}"
                res[key]["success"].append(m["eval/success_rate"])
                res[key]["replans"].append(m["eval/replans"])
                res[key]["exec_len"].append(m["eval/action_chunk_execution_length"])
                res[key]["env_steps"].append(m["eval/environment_steps"])
                res[key]["goal_dist"].append(m["eval/goal_distance_final"])
            print(f"  {ck.parts[-3]} track E done", flush=True)

        if "F" in args.tracks:
            tc = tree_config_for(args.arm, replace(base, node_budget=64), model)
            for mode, lam in F_MODES:
                pc = replace(pcfg, score_mode=mode, path_cost_weight=lam,
                             ancestor_weight=lam if mode == "ancestor" else pcfg.ancestor_weight)
                m = evaluate(env, GoalPlanner(model, normalizer, tc, pc, device),
                             tasks, args.episodes, int(cfg.planner.max_env_steps), 0)
                key = f"F|{mode}_{lam}"
                res[key]["success"].append(m["eval/success_rate"])
                res[key]["leaf_depth"].append(m["eval/selected_leaf_depth"])
                res[key]["goal_dist"].append(m["eval/goal_distance_final"])
            print(f"  {ck.parts[-3]} track F done", flush=True)

    payload = {k: {kk: [float(x) for x in vv] for kk, vv in v.items()} for k, v in res.items()}
    payload["_shapes"] = {
        p: {k: float(np.mean([s[k] for s in ss if k in s]))
            for k in {k for s in ss for k in s}} for p, ss in shapes.items()
    }
    prov = provenance(cks[0] if cks else None, None,
                      extra={"arm": args.arm, "dataset": args.dataset,
                             "n_checkpoints": len(cks), "tracks": args.tracks,
                             "episodes": args.episodes, "max_tasks": args.max_tasks})
    out = write_artifact(REPO / args.out / f"tracks_def_{args.dataset}.json", payload, prov)

    for track, title in (("D", "expansion policy"), ("E", "execution cadence"), ("F", "planner score")):
        keys = sorted(k for k in res if k.startswith(f"{track}|"))
        if not keys:
            continue
        print(f"\n=== Track {track}: {title} ({args.dataset}) ===")
        cols = sorted({c for k in keys for c in res[k]}, key=lambda c: (not c.isdigit(), c))
        print(f"{'setting':22s} " + " ".join(f"{c:>11s}" for c in cols))
        print("-" * (22 + 12 * len(cols)))
        for k in keys:
            cells = [f"{np.mean(res[k][c]):11.3f}" if c in res[k] else f"{'-':>11s}" for c in cols]
            print(f"{k.split('|')[1]:22s} " + " ".join(cells))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
