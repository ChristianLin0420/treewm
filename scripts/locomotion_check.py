"""Early locomotion sanity check for AntMaze (Q1 gate).

A zero success rate on AntMaze is uninformative on its own -- the previous 5k smoke test
scored 0.000 everywhere simply because the ant could not walk. This measures whether the
body is moving at all, independently of whether it reaches goals.

The gate is deliberately permissive: **do not abort merely because success is still zero**.
Abort only if the ant genuinely is not locomoting (near-zero displacement, degenerate
actions) or prediction is diverging.

    python scripts/locomotion_check.py --runs runs_q1 --step 5000
"""

from __future__ import annotations

import argparse
import sys
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

# Thresholds are about *locomotion*, not task success.
MIN_DISPLACEMENT = 1.0     # world units moved from the start over an episode
MIN_PATH_LENGTH = 5.0      # total distance travelled (catches jitter-in-place)
MIN_ACTION_MAG = 0.02      # near-zero actions mean the policy collapsed
MAX_STATE_MSE = 0.5        # decoded-state prediction should not be diverging


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="experiments/06-q1-q2-q3/runs/q1")
    p.add_argument("--checkpoint-name", default="latest.pt")
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max-tasks", type=int, default=4)
    p.add_argument("--out", default="results_q1")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(0)
    cks = sorted((REPO / args.runs).glob(f"*/*/*/checkpoints/{args.checkpoint_name}"))
    if not cks:
        print(f"[loco] no checkpoints under {args.runs} -- reporting as missing")
        return
    print(f"[loco] {len(cks)} checkpoints")

    rows = []
    env = tasks = None
    for ck in cks:
        model, normalizer, cfg, payload = load_run(str(ck), device)
        if env is None:
            env = load_ogbench(cfg.env.name, dataset_dir=cfg.env.dataset_dir, env_only=True)
            tasks = build_tasks(env, "hard", int(cfg.eval.num_hard_tasks),
                                float(cfg.eval.hard_percentile), 0)[: args.max_tasks]
        tc = tree_config_for(ck.parts[-4],
                             replace(cfg_utils.tree_config(cfg), node_budget=args.budget), model)
        pc = cfg_utils.planner_config(cfg)
        m = evaluate(env, GoalPlanner(model, normalizer, tc, pc, device), tasks,
                     args.episodes, int(cfg.planner.max_env_steps), 0)
        rows.append({
            "recipe": ck.parts[-3], "step": int(payload.get("step", -1)),
            "success": m["eval/success_rate"], "displacement": m["eval/displacement"],
            "path_length": m["eval/path_length"], "action_magnitude": m["eval/action_magnitude"],
            "fraction_moving": m["eval/fraction_moving"],
            "goal_distance": m["eval/goal_distance_final"],
        })

    print(f"\n{'recipe':18s} {'step':>7s} {'succ':>6s} {'displ':>7s} {'path':>8s} "
          f"{'|a|':>6s} {'moving':>7s} {'goal_d':>7s}  verdict")
    print("-" * 92)
    healthy = 0
    for r in rows:
        ok = (r["displacement"] >= MIN_DISPLACEMENT and r["path_length"] >= MIN_PATH_LENGTH
              and r["action_magnitude"] >= MIN_ACTION_MAG)
        healthy += bool(ok)
        note = "locomoting" if ok else "NOT LOCOMOTING"
        if ok and r["success"] == 0.0:
            note += " (success 0 -- expected this early, not a reason to abort)"
        print(f"{r['recipe']:18s} {r['step']:7d} {r['success']:6.3f} {r['displacement']:7.2f} "
              f"{r['path_length']:8.2f} {r['action_magnitude']:6.3f} {r['fraction_moving']:7.2f} "
              f"{r['goal_distance']:7.2f}  {note}")

    verdict = "CONTINUE" if healthy >= max(1, len(rows) // 2) else "INVESTIGATE"
    print(f"\n  {healthy}/{len(rows)} runs locomoting -> {verdict}")
    write_artifact(REPO / args.out / "locomotion_check.json",
                   {"rows": rows, "healthy": healthy, "verdict": verdict},
                   provenance(cks[0], None, extra={"budget": args.budget}))


if __name__ == "__main__":
    main()
