"""Q3 -- do `ancestor` scoring and `root_quota` allocation compose?

Both are inference-time settings, so this is evaluation-only on the existing `C1_short`
checkpoints: a clean 2x2 factorial with no retraining.

    C1_short                       (neither)
    C1_short + ancestor            (planner score only)
    C1_short + root_quota          (allocation only)
    C1_short + ancestor + root_quota

If the two effects are additive they compose; if the combined cell matches the better
single cell, one of them was only useful under the older model.
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
from treewm.models.baselines import tree_config_for
from treewm.planning.goal_planner import GoalPlanner
from treewm.utils import config as cfg_utils
from treewm.utils.provenance import provenance, write_artifact
from treewm.utils.seeding import seed_everything
from scripts.eval import load_run

REPO = Path(__file__).resolve().parents[1]

CELLS = [
    ("base", "random", "endpoint"),
    ("ancestor", "random", "ancestor"),
    ("root_quota", "root_quota", "endpoint"),
    ("both", "root_quota", "ancestor"),
]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="experiments/05-design-space/runs/phase2")
    p.add_argument("--recipe", default="C1_short")
    p.add_argument("--dataset", default="pointmaze-medium-stitch")
    p.add_argument("--budgets", nargs="+", type=int, default=[32, 64, 128])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--out", default="results_q3")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(0)
    cks = [c for c in sorted((REPO / args.runs).glob("*/*/*/checkpoints/latest.pt"))
           if c.parts[-5] == args.dataset and c.parts[-3].startswith(args.recipe)]
    print(f"[q3] {len(cks)} {args.recipe} checkpoints", flush=True)
    if not cks:
        print("[q3] no checkpoints found -- reporting as missing rather than substituting")
        return

    res: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    env = tasks = None
    for ck in cks:
        model, normalizer, cfg, _ = load_run(str(ck), device)
        if env is None:
            env = load_ogbench(cfg.env.name, dataset_dir=cfg.env.dataset_dir, env_only=True)
            tasks = build_tasks(env, "hard", int(cfg.eval.num_hard_tasks),
                                float(cfg.eval.hard_percentile), 0)
        base_tree = cfg_utils.tree_config(cfg)
        base_plan = cfg_utils.planner_config(cfg)

        for name, scorer, score_mode in CELLS:
            for b in args.budgets:
                tc = tree_config_for("randomtreewm", replace(base_tree, node_budget=b), model)
                tc = replace(tc, scorer=scorer)
                pc = replace(base_plan, score_mode=score_mode)
                m = evaluate(env, GoalPlanner(model, normalizer, tc, pc, device),
                             tasks, args.episodes, int(cfg.planner.max_env_steps), 0)
                res[name][str(b)].append(m["eval/success_rate"])
                if b == 64:
                    res[name]["goal_dist"].append(m["eval/goal_distance_final"])
                    res[name]["leaf_depth"].append(m["eval/selected_leaf_depth"])
        print(f"  {ck.parts[-3]} done", flush=True)

    prov = provenance(cks[0], None, extra={"recipe": args.recipe, "cells": [c[0] for c in CELLS],
                                           "n_checkpoints": len(cks)})
    write_artifact(REPO / args.out / f"factorial_{args.recipe}.json",
                   {k: {kk: [float(x) for x in vv] for kk, vv in v.items()} for k, v in res.items()},
                   prov)

    print(f"\n=== Q3 factorial on {args.recipe} ({len(cks)} seeds) ===")
    cols = [str(b) for b in args.budgets] + ["goal_dist", "leaf_depth"]
    print(f"{'cell':14s} " + " ".join(f"{c:>11s}" for c in cols) + f" {'AUC':>8s}")
    print("-" * (14 + 12 * len(cols) + 9))
    for name, _, _ in CELLS:
        cells = [f"{np.mean(res[name][c]):11.3f}" if c in res[name] else f"{'-':>11s}" for c in cols]
        auc = np.mean([np.mean(res[name][str(b)]) for b in args.budgets])
        print(f"{name:14s} " + " ".join(cells) + f" {auc:8.3f}")

    base = np.mean([np.mean(res["base"][str(b)]) for b in args.budgets])
    anc = np.mean([np.mean(res["ancestor"][str(b)]) for b in args.budgets]) - base
    rq = np.mean([np.mean(res["root_quota"][str(b)]) for b in args.budgets]) - base
    both = np.mean([np.mean(res["both"][str(b)]) for b in args.budgets]) - base
    print(f"\n  ancestor alone   {anc:+.3f}")
    print(f"  root_quota alone {rq:+.3f}")
    print(f"  both             {both:+.3f}   (additive would be {anc + rq:+.3f})")
    verdict = ("compose" if both > max(anc, rq) + 0.01 else
               "do NOT compose -- combined is no better than the best single effect")
    print(f"  -> {verdict}")


if __name__ == "__main__":
    main()
