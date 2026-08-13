"""Controllability coverage vs node budget, per arm.

The second primary plot (spec section 25). Unlike the success sweep this needs no
environment rollouts -- it grows a tree from each of a fixed set of start states and
counts how many distinct quantised regions the tree's nodes reach.

This is the direct test of the project's core claim: if learned allocation works, TreeWM
should convert node budget into *coverage* more efficiently than BFS/random/uncertainty.
All arms share identically-trained encoders and decoders, so the decoded positions are
comparable across arms.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace

from treewm.data.maze_utils import MazeSpec
from treewm.data.ogbench_dataset import load_ogbench
from treewm.evaluation.coverage import StateQuantizer, unique_cells_per_row
from treewm.models.baselines import tree_config_for
from treewm.utils import config as cfg_utils
from treewm.utils.seeding import seed_everything
from scripts.eval import load_run

REPO = Path(__file__).resolve().parents[1]
ARM_ORDER = ["singlewm", "flatkwm", "fixedtreewm", "randomtreewm", "uncertaintytreewm",
             "heuristictreewm", "treewm", "noveltyq", "learnedq", "noveltyz", "learnedz"]


@torch.no_grad()
def coverage_for(model, normalizer, maze_spec, quantizer, tree_cfg, starts, device):
    obs = torch.from_numpy(normalizer.norm_obs(starts)).to(device)
    z = model.encode(obs)
    tree, _ = model.generate(z, tree_cfg)
    states = model.decoder(tree.latent)  # [B, N, obs_dim]
    cells = quantizer.cell_ids(states)
    covered = unique_cells_per_row(cells, tree.valid.float()).float()
    nodes = tree.valid.float().sum(1).clamp_min(1.0)
    return float(covered.mean()), float((covered / nodes).mean()), float(nodes.mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--budgets", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    p.add_argument("--num-starts", type=int, default=64)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(0)
    results: dict[str, dict[str, dict[int, list[float]]]] = {}

    checkpoints = sorted((REPO / args.runs).glob("*/*/*/checkpoints/latest.pt"))
    print(f"[coverage] {len(checkpoints)} checkpoints")

    env_cache: dict[str, tuple] = {}
    for ck in checkpoints:
        dataset, arm = ck.parts[-5], ck.parts[-4]
        model, normalizer, cfg, _ = load_run(str(ck), device)
        if model.decoder is None:
            continue
        if dataset not in env_cache:
            env = load_ogbench(cfg.env.name, dataset_dir=cfg.env.dataset_dir, env_only=True)
            spec = MazeSpec.from_env(env)
            cells = spec.free_cells()
            rng = np.random.default_rng(0)
            pick = rng.choice(len(cells), size=min(args.num_starts, len(cells)), replace=True)
            jitter = rng.uniform(-0.3, 0.3, size=(len(pick), 2)) * spec.unit
            starts = np.stack([spec.ij_to_xy(int(i), int(j)) for i, j in cells[pick]]) + jitter
            env_cache[dataset] = (spec, starts.astype(np.float32))
        spec, starts = env_cache[dataset]

        quantizer = StateQuantizer(
            resolution=float(cfg.retrieval.grid_resolution), dims=tuple(cfg.env.xy_dims)
        )
        base = cfg_utils.tree_config(cfg)
        for budget in args.budgets:
            tc = tree_config_for(arm, replace(base, node_budget=budget), model)
            cov, cov_per_node, nodes = coverage_for(
                model, normalizer, spec, quantizer, tc, starts, device
            )
            assert nodes <= budget + 1e-6, f"{arm} exceeded budget {budget}"
            results.setdefault(dataset, {}).setdefault(arm, {}).setdefault(budget, []).append(cov)
        print(f"  {dataset}/{arm}/{ck.parts[-3]}")

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "coverage_curves.json").write_text(json.dumps(results, indent=2, default=str))

    for dataset, arms in results.items():
        print(f"\n=== controllability coverage (distinct regions) vs budget — {dataset} ===")
        print(f"{'arm':20s} " + " ".join(f"{b:>7d}" for b in args.budgets))
        print("-" * 70)
        fig, ax = plt.subplots(figsize=(7, 4.6))
        for arm in ARM_ORDER:
            if arm not in arms:
                continue
            ys = [float(np.mean(arms[arm][b])) for b in args.budgets]
            print(f"{arm:20s} " + " ".join(f"{y:7.2f}" for y in ys))
            style = dict(marker="o", lw=2.2) if arm == "treewm" else dict(marker=".", lw=1.2, alpha=0.85)
            ax.plot(args.budgets, ys, label=arm, **style)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("node budget")
        ax.set_ylabel("distinct regions covered")
        ax.set_title(f"Coverage vs node budget — {dataset}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"coverage_vs_budget_{dataset}.png"
        fig.savefig(path, dpi=140)
        print(f"[coverage] wrote {path}")


if __name__ == "__main__":
    main()
