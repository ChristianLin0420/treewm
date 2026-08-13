"""Is the coverage real, or is the model hallucinating it?

`coverage_sweep.py` counts distinct regions among *decoded predicted* node positions. If
open-loop prediction error compounds with depth, a policy that expands deeper will score
more "coverage" while actually predicting places the agent cannot reach -- which would
make the coverage metric reward hallucination.

This script executes each tree node's root-to-node action chunks in the simulator and
compares where the agent actually lands against where the model said it would:

    grounded error(d) = || actual_xy - predicted_xy ||   for nodes at depth d
    grounded coverage = distinct regions among *actual* landing positions

Every arm shares the same world model, so any difference between arms comes purely from
*which* nodes they choose to expand.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataclasses import replace

from treewm.data.maze_utils import MazeSpec
from treewm.data.ogbench_dataset import load_ogbench
from treewm.evaluation.coverage import StateQuantizer
from treewm.models.baselines import tree_config_for
from treewm.utils import config as cfg_utils
from treewm.utils.seeding import seed_everything
from scripts.eval import load_run

REPO = Path(__file__).resolve().parents[1]


def replay(env, start_xy, chunks, normalizer):
    """Reset to ``start_xy`` and execute a list of action chunks. Returns final xy."""
    env.reset(seed=0)
    env.unwrapped.set_xy(np.asarray(start_xy, dtype=np.float64))
    ob = env.unwrapped._get_obs() if hasattr(env.unwrapped, "_get_obs") else None
    for chunk in chunks:
        for action in chunk:
            ob, _, term, trunc, _ = env.step(np.clip(action, -1.0, 1.0))
            if term or trunc:
                return np.asarray(ob, dtype=np.float32)[:2]
    return np.asarray(ob, dtype=np.float32)[:2]


@torch.no_grad()
def analyse(model, normalizer, env, spec, tree_cfg, starts, quantizer, max_nodes=12, seed=0):
    device = next(model.parameters()).device
    rng = np.random.default_rng(seed)
    by_depth: dict[int, list[float]] = defaultdict(list)
    pred_cells, real_cells = set(), set()

    for start in starts:
        obs = torch.from_numpy(normalizer.norm_obs(start[None])).to(device)
        tree, _ = model.generate(model.encode(obs), tree_cfg)
        pred_xy = normalizer.denorm_obs(model.decoder(tree.latent[0]).float().cpu().numpy())
        valid = tree.valid[0].cpu().numpy()
        depth = tree.depth[0].cpu().numpy()
        parent = tree.parent_index[0].cpu().numpy()
        chunk = tree.action_chunk[0].float().cpu().numpy()
        mask = tree.action_mask[0].float().cpu().numpy()

        candidates = [n for n in range(len(valid)) if valid[n] and n != 0]
        if not candidates:
            continue
        picks = rng.choice(candidates, size=min(max_nodes, len(candidates)), replace=False)

        for node in picks:
            path, cur = [], int(node)
            while cur > 0:
                path.append(cur)
                cur = int(parent[cur])
            path.reverse()
            chunks = []
            for p in path:
                n_steps = int(mask[p].sum())
                if n_steps:
                    chunks.append(normalizer.denorm_act(chunk[p][:n_steps]))
            if not chunks:
                continue
            actual = replay(env, start, chunks, normalizer)
            predicted = pred_xy[node]
            by_depth[int(depth[node])].append(float(np.linalg.norm(actual - predicted)))
            pred_cells.add(int(quantizer.cell_ids(predicted[None])[0]))
            real_cells.add(int(quantizer.cell_ids(actual[None])[0]))

    return by_depth, len(pred_cells), len(real_cells)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs_novelty")
    p.add_argument("--dataset", default="pointmaze-medium-stitch")
    p.add_argument("--budget", type=int, default=256)
    p.add_argument("--seed-tag", default="seed0")
    p.add_argument("--num-starts", type=int, default=8)
    p.add_argument("--max-nodes", type=int, default=12)
    p.add_argument("--out", default="results_novelty")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(0)
    report: dict[str, dict] = {}

    checkpoints = [
        ck for ck in sorted((REPO / args.runs).glob("*/*/*/checkpoints/latest.pt"))
        if ck.parts[-5] == args.dataset and args.seed_tag in ck.parts[-3]
    ]
    print(f"[grounded] {len(checkpoints)} checkpoints on {args.dataset} @ budget {args.budget}")

    env = spec = starts = None
    for ck in checkpoints:
        arm = ck.parts[-4]
        model, normalizer, cfg, _ = load_run(str(ck), device)
        if model.decoder is None:
            continue
        if env is None:
            env = load_ogbench(cfg.env.name, dataset_dir=cfg.env.dataset_dir, env_only=True)
            spec = MazeSpec.from_env(env)
            cells = spec.free_cells()
            rng = np.random.default_rng(0)
            pick = rng.choice(len(cells), size=min(args.num_starts, len(cells)), replace=False)
            starts = np.stack([spec.ij_to_xy(int(i), int(j)) for i, j in cells[pick]]).astype(np.float32)

        quantizer = StateQuantizer(resolution=float(cfg.retrieval.grid_resolution), dims=(0, 1))
        tc = tree_config_for(arm, replace(cfg_utils.tree_config(cfg), node_budget=args.budget), model)
        by_depth, n_pred, n_real = analyse(
            model, normalizer, env, spec, tc, starts, quantizer, args.max_nodes
        )
        depths = sorted(by_depth)
        report[arm] = {
            "error_by_depth": {int(d): float(np.mean(by_depth[d])) for d in depths},
            "count_by_depth": {int(d): len(by_depth[d]) for d in depths},
            "mean_error": float(np.mean([e for v in by_depth.values() for e in v])),
            "mean_depth": float(np.mean([d for d, v in by_depth.items() for _ in v])),
            "predicted_regions": n_pred,
            "actual_regions": n_real,
            "hallucination_ratio": n_pred / max(n_real, 1),
        }
        print(f"  {arm:16s} mean_depth={report[arm]['mean_depth']:.2f} "
              f"mean_err={report[arm]['mean_error']:6.2f} "
              f"pred_regions={n_pred:3d} actual_regions={n_real:3d} "
              f"ratio={report[arm]['hallucination_ratio']:.2f}")

    out = REPO / args.out / f"grounded_{args.dataset}.json"
    out.write_text(json.dumps(report, indent=2))

    print(f"\n=== open-loop prediction error vs tree depth ({args.dataset}) ===")
    all_depths = sorted({d for r in report.values() for d in r["error_by_depth"]})
    print(f"{'arm':16s} " + " ".join(f"d{d:<6d}" for d in all_depths))
    print("-" * (16 + 8 * len(all_depths)))
    for arm, r in report.items():
        cells = [f"{r['error_by_depth'].get(d, float('nan')):6.2f} " for d in all_depths]
        print(f"{arm:16s} " + " ".join(cells))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
