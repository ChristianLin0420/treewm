"""Task-horizon difficulty curve: where does recursion start to help?

A single hard-goal benchmark that saturates at zero cannot say *where* recursion becomes
valuable. This bins evaluation start/goal pairs by geodesic distance and evaluates every
arm in every bin at matched node budgets, producing:

    success(d)                        per arm
    distance reduction(d)             per arm -- informative even where success is 0
    Delta(d) = recursive - flat       the quantity the recursion claim predicts grows
    h*(d)                             best fixed edge horizon per difficulty bin

Nothing is retrained: this only changes which evaluation pairs are sampled.

    python scripts/difficulty_curve.py --runs runs_ant4 --dataset antmaze-teleport
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treewm.data.ogbench_dataset import load_ogbench
from treewm.evaluation.rollout import evaluate
from treewm.evaluation.tasks import build_bucketed_tasks
from treewm.models.baselines import tree_config_for
from treewm.planning.goal_planner import GoalPlanner
from treewm.utils import config as cfg_utils
from treewm.utils.provenance import provenance, write_artifact
from treewm.utils.rng import make_generator
from treewm.utils.seeding import seed_everything

REPO = Path(__file__).resolve().parents[1]

METRICS = ["eval/success_rate", "eval/distance_reduction", "eval/distance_reduction_frac",
           "eval/goal_distance_final", "eval/initial_goal_distance", "eval/displacement",
           "eval/selected_leaf_depth", "eval/path_length"]


def label_of(run_name: str) -> str:
    return re.sub(r"_s\d+$", "", run_name)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=["experiments/07-horizon-antmaze/runs/ant4"])
    p.add_argument("--bins", nargs="+", type=int, default=[1, 5, 10, 15, 20, 99])
    p.add_argument("--per-bin", type=int, default=10)
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--budgets", nargs="+", type=int, default=[32, 64, 128])
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--out", default="experiments/08-scaling/results/difficulty")
    p.add_argument("--tag", default="antmaze")
    p.add_argument("--include", nargs="+", default=None,
                   help="only evaluate these recipe labels (keeps the grid affordable)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(0)
    from scripts.eval import load_run

    cks = []
    for root in args.runs:
        cks += sorted((REPO / root).glob("*/*/*/checkpoints/latest.pt"))
    if args.include:
        cks = [c for c in cks if label_of(c.parts[-3]) in set(args.include)]
    if not cks:
        print(f"[curve] no checkpoints under {args.runs} -- reporting as missing")
        return
    print(f"[curve] {len(cks)} checkpoints", flush=True)

    bins = [(args.bins[i], args.bins[i + 1]) for i in range(len(args.bins) - 1)]
    res: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    geo: dict[str, float] = {}
    env = buckets = None

    for ck in cks:
        arm = label_of(ck.parts[-3])
        model, normalizer, cfg, _ = load_run(str(ck), device)
        if env is None:
            env = load_ogbench(cfg.env.name, dataset_dir=cfg.env.dataset_dir, env_only=True)
            buckets = build_bucketed_tasks(env, bins, args.per_bin, seed=0)
            for label, tasks in buckets.items():
                if tasks:
                    geo[label] = float(np.mean([t["geodesic_world"] for t in tasks]))
        base = cfg_utils.tree_config(cfg)
        pcfg = cfg_utils.planner_config(cfg)

        for label, tasks in buckets.items():
            if not tasks:
                continue
            for b in args.budgets:
                tc = tree_config_for(ck.parts[-4], replace(base, node_budget=b), model)
                planner = GoalPlanner(model, normalizer, tc, pcfg, device,
                                      generator=make_generator(0, "eval", device))
                m = evaluate(env, planner, tasks, args.episodes, args.max_steps, 0)
                for k in METRICS:
                    if k in m:
                        res[arm][label][f"{k}|b{b}"].append(m[k])
        print(f"  {arm} ({ck.parts[-3]}) done", flush=True)

    labels = [l for l, t in buckets.items() if t]
    arms = sorted(res)

    payload = {a: {l: {k: [float(x) for x in v] for k, v in d.items()}
                   for l, d in res[a].items()} for a in arms}
    payload["_geodesic_world"] = geo
    write_artifact(REPO / args.out / f"difficulty_{args.tag}.json", payload,
                   provenance(cks[0], None, extra={"bins": args.bins, "budgets": args.budgets,
                                                   "per_bin": args.per_bin,
                                                   "episodes": args.episodes}))

    def val(arm, label, metric, b):
        v = res[arm][label].get(f"{metric}|b{b}", [])
        return float(np.mean(v)) if v else float("nan")

    for b in args.budgets:
        print(f"\n=== budget {b}: success by geodesic distance bin ===")
        print(f"{'arm':16s} " + " ".join(f"{l:>10s}" for l in labels))
        print(f"{'(mean world d)':16s} " + " ".join(f"{geo.get(l, float('nan')):10.1f}" for l in labels))
        print("-" * (16 + 11 * len(labels)))
        for a in arms:
            print(f"{a:16s} " + " ".join(f"{val(a, l, 'eval/success_rate', b):10.3f}" for l in labels))
        print(f"{'-- dist reduction (world units) --':>16s}")
        for a in arms:
            print(f"{a:16s} " + " ".join(f"{val(a, l, 'eval/distance_reduction', b):10.2f}" for l in labels))

    # Delta(recursive - flat) where both exist
    flat = next((a for a in arms if "flat" in a.lower()), None)
    rec = [a for a in arms if a != flat]
    if flat and rec:
        print(f"\n=== Delta = recursive - flat, by distance bin ===")
        for b in args.budgets:
            print(f"  budget {b}")
            for a in rec:
                ds = [val(a, l, "eval/success_rate", b) - val(flat, l, "eval/success_rate", b)
                      for l in labels]
                dr = [val(a, l, "eval/distance_reduction", b) - val(flat, l, "eval/distance_reduction", b)
                      for l in labels]
                print(f"    {a:14s} success  " + " ".join(f"{x:+7.3f}" for x in ds))
                print(f"    {'':14s} distred  " + " ".join(f"{x:+7.2f}" for x in dr))

    # h*(d): best horizon per bin, if several H-arms are present
    hs = {a: int(m.group(1)) for a in arms if (m := re.match(r"^H(\d+)$", a))}
    if len(hs) > 1:
        print(f"\n=== h*(d): best fixed horizon per distance bin ===")
        print(f"{'bin':>10s} {'mean world d':>13s} {'best h':>7s}   success by h")
        for l in labels:
            per = {h: np.mean([val(a, l, "eval/success_rate", b) for b in args.budgets])
                   for a, h in hs.items()}
            finite = {h: v for h, v in per.items() if np.isfinite(v)}
            if not finite:
                continue
            best = max(finite, key=finite.get)
            detail = " ".join(f"h{h}={finite[h]:.2f}" for h in sorted(finite))
            print(f"{l:>10s} {geo.get(l, float('nan')):13.1f} {best:7d}   {detail}")

    # ---- plots -------------------------------------------------------------------
    xs = [geo[l] for l in labels]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
    b0 = args.budgets[len(args.budgets) // 2]
    for a in arms:
        axes[0].plot(xs, [val(a, l, "eval/success_rate", b0) for l in labels], marker="o", label=a)
        axes[1].plot(xs, [val(a, l, "eval/distance_reduction", b0) for l in labels], marker="o", label=a)
    axes[0].set_xlabel("geodesic task distance (world units)"); axes[0].set_ylabel("success")
    axes[0].set_title(f"success vs task distance (budget {b0})"); axes[0].grid(alpha=.3); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("geodesic task distance (world units)"); axes[1].set_ylabel("distance reduction")
    axes[1].set_title("distance reduction vs task distance"); axes[1].grid(alpha=.3); axes[1].legend(fontsize=8)
    if flat and rec:
        for a in rec:
            axes[2].plot(xs, [val(a, l, "eval/success_rate", b0) - val(flat, l, "eval/success_rate", b0)
                              for l in labels], marker="o", label=f"{a} - {flat}")
        axes[2].axhline(0, color="0.5", lw=1)
        axes[2].set_title("recursive advantage vs task distance")
        axes[2].legend(fontsize=8)
    axes[2].set_xlabel("geodesic task distance (world units)"); axes[2].set_ylabel("delta success")
    axes[2].grid(alpha=.3)
    fig.tight_layout()
    out = REPO / args.out / f"difficulty_curve_{args.tag}.png"
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
