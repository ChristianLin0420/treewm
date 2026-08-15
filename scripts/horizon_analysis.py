"""Q2 -- horizon usage statistics and interpretability evidence.

The mechanism question is whether the learned model *chooses* horizons in an
interpretable way, or whether simply having short horizons available is the whole story.
This reports, for the learned model against its controls:

  * horizon histogram over the whole tree
  * horizon by tree depth          (are deeper, less certain rollouts shorter?)
  * horizon by root subtree        (does one branch specialise?)
  * horizon by maze location       (long edges in open corridors, short at junctions?)
  * horizon on successful vs failed episodes

The location analysis uses maze geometry for *interpretation only* -- nothing here feeds
a loss.
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

from treewm.data.maze_utils import MazeSpec
from treewm.data.ogbench_dataset import load_ogbench
from treewm.evaluation.tasks import build_tasks
from treewm.evaluation.tree_viz import build_anchors
from treewm.models.baselines import tree_config_for
from treewm.utils import config as cfg_utils
from treewm.utils.provenance import provenance, write_artifact
from treewm.utils.rng import make_generator
from treewm.utils.seeding import seed_everything
from scripts.eval import load_run

REPO = Path(__file__).resolve().parents[1]


@torch.no_grad()
def analyse(model, normalizer, spec, tc, anchors, device) -> dict:
    obs = torch.from_numpy(normalizer.norm_obs(anchors.starts)).to(device)
    goal = torch.from_numpy(normalizer.norm_obs(anchors.goals)).to(device)
    from treewm.tree.frontier import GOAL_AWARE_SCORERS

    z = model.encode(obs)
    tree, _ = model.generate(z, tc, generator=make_generator(0, 'viz', device),
                             goal_obs=goal if tc.scorer in GOAL_AWARE_SCORERS else None)

    valid = tree.valid
    horizon = tree.action_mask.sum(-1).float()
    depth = tree.depth
    rb = tree.root_branch

    out: dict = {}
    h_all = horizon[valid]
    out["histogram"] = np.bincount(h_all.long().cpu().numpy(), minlength=17)[:17].tolist()
    out["mean"] = float(h_all.mean())
    out["std"] = float(h_all.std())
    # A model that always emits the same horizon has zero entropy here; the learned model
    # is only interesting if this is well above zero.
    p = np.asarray(out["histogram"], dtype=float)
    p = p[p > 0] / p.sum()
    out["entropy_nats"] = float(-(p * np.log(p)).sum())

    for d in range(0, 6):
        m = valid & (depth == d)
        if int(m.sum()):
            out[f"mean_at_depth{d}"] = float(horizon[m].mean())

    per_sub = []
    for b in range(tree.batch_size):
        ids = rb[b][valid[b]]
        ids = ids[ids >= 0]
        for u in torch.unique(ids):
            sel = valid[b] & (rb[b] == u)
            if int(sel.sum()):
                per_sub.append(float(horizon[b][sel].mean()))
    if per_sub:
        out["subtree_mean_spread"] = float(np.std(per_sub))

    # horizon vs local maze openness (free 4-neighbours of the node's cell)
    if model.decoder is not None:
        pos = normalizer.denorm_obs(model.decoder(tree.latent).float().cpu().numpy())
        deg = spec.junction_degree()
        hs, ds = [], []
        for b in range(tree.batch_size):
            for n in range(tree.capacity):
                if not bool(valid[b, n]):
                    continue
                ij = spec.xy_to_ij(pos[b, n][:2])
                hs.append(float(horizon[b, n]))
                ds.append(float(deg[ij[0], ij[1]]))
        if len(hs) > 2 and np.std(ds) > 0 and np.std(hs) > 0:
            out["corr_horizon_vs_openness"] = float(np.corrcoef(hs, ds)[0, 1])
            for d in sorted(set(ds)):
                sel = [h for h, dd in zip(hs, ds) if dd == d]
                if len(sel) > 3:
                    out[f"mean_at_openness{int(d)}"] = float(np.mean(sel))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs_q2")
    p.add_argument("--dataset", default="pointmaze-medium-stitch")
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--anchors", type=int, default=8)
    p.add_argument("--out", default="results_q2")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(0)
    cks = sorted((REPO / args.runs).glob("*/*/*/checkpoints/latest.pt"))
    print(f"[horizon] {len(cks)} checkpoints")
    if not cks:
        print("[horizon] none found -- reporting as missing")
        return

    agg: dict[str, list] = defaultdict(list)
    env = spec = anchors = None
    for ck in cks:
        import re

        recipe = re.sub(r"_s\d+$", "", ck.parts[-3])
        model, normalizer, cfg, _ = load_run(str(ck), device)
        if env is None:
            env = load_ogbench(cfg.env.name, dataset_dir=cfg.env.dataset_dir, env_only=True)
            spec = MazeSpec.from_env(env)
            anchors = build_anchors(spec, num=args.anchors)
        tc = tree_config_for("randomtreewm",
                             replace(cfg_utils.tree_config(cfg), node_budget=args.budget), model)
        agg[recipe].append(analyse(model, normalizer, spec, tc, anchors, device))
        print(f"  {recipe} {ck.parts[-3]}")

    merged = {}
    for rec, runs in agg.items():
        keys = {k for r in runs for k in r if k != "histogram"}
        merged[rec] = {k: float(np.mean([r[k] for r in runs if k in r])) for k in keys}
        merged[rec]["histogram"] = np.mean([r["histogram"] for r in runs], axis=0).tolist()

    write_artifact(REPO / args.out / "horizon_analysis.json", merged,
                   provenance(cks[0], None, extra={"budget": args.budget}))

    print(f"\n=== horizon usage (budget {args.budget}) ===")
    cols = ["mean", "std", "entropy_nats", "mean_at_depth1", "mean_at_depth3",
            "subtree_mean_spread", "corr_horizon_vs_openness"]
    print(f"{'recipe':16s} " + " ".join(f"{c[:13]:>14s}" for c in cols))
    print("-" * (16 + 15 * len(cols)))
    for rec in sorted(merged):
        print(f"{rec:16s} " + " ".join(
            f"{merged[rec].get(c, float('nan')):14.3f}" for c in cols))
    print(f"\nwrote {REPO / args.out / 'horizon_analysis.json'}")


if __name__ == "__main__":
    main()
