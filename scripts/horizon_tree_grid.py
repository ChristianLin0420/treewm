"""Side-by-side tree geometry across fixed horizons, from the same anchors.

Answers visually what the scalar tradeoff implies: short edges make dense, local trees
that cannot reach; long edges make sparse, far-reaching trees whose endpoints are wrong.
Every panel uses the same anchor and the same node budget, so only h differs.

    python scripts/horizon_tree_grid.py --horizons 8 16 24 32 --budget 64
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treewm.data.maze_utils import MazeSpec
from treewm.data.ogbench_dataset import load_ogbench
from treewm.evaluation import tree_viz as tv
from treewm.evaluation.grounding import ground_tree, restore_state, save_state
from treewm.models.baselines import tree_config_for
from treewm.utils import config as cfg_utils
from treewm.utils.rng import make_generator
from treewm.utils.seeding import seed_everything
from scripts.eval import load_run

REPO = Path(__file__).resolve().parents[1]


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs_h")
    p.add_argument("--horizons", nargs="+", type=int, default=[8, 16, 24, 32])
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--seed-tag", default="s0")
    p.add_argument("--anchors", nargs="+", type=int, default=[0, 3])
    p.add_argument("--out", default="results_h")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(0)

    picked: dict[int, Path] = {}
    for ck in sorted((REPO / args.runs).glob("*/*/*/checkpoints/latest.pt")):
        m = re.match(r"^H(\d+)_" + args.seed_tag + "$", ck.parts[-3])
        if m and int(m.group(1)) in args.horizons:
            picked[int(m.group(1))] = ck
    missing = [h for h in args.horizons if h not in picked]
    if missing:
        print(f"[grid] missing checkpoints for h={missing} -- reporting as missing")
    hs = [h for h in args.horizons if h in picked]
    if not hs:
        return

    env = spec = anchors = None
    rows = len(args.anchors)
    fig, axes = plt.subplots(rows, len(hs), figsize=(4.1 * len(hs), 4.0 * rows), squeeze=False)
    stats: dict[int, dict] = {}

    for col, h in enumerate(hs):
        model, normalizer, cfg, _ = load_run(str(picked[h]), device)
        if env is None:
            env = load_ogbench(cfg.env.name, dataset_dir=cfg.env.dataset_dir, env_only=True)
            spec = MazeSpec.from_env(env)
            anchors = tv.build_anchors(spec, num=8)
        tc = tree_config_for("randomtreewm",
                             replace(cfg_utils.tree_config(cfg), node_budget=args.budget), model)

        for row, a in enumerate(args.anchors):
            start, goal = anchors.starts[a], anchors.goals[a]
            obs = torch.from_numpy(normalizer.norm_obs(start[None])).to(device)
            goal_n = torch.from_numpy(normalizer.norm_obs(goal[None])).to(device)
            tree, _ = model.generate(model.encode(obs), tc,
                                     generator=make_generator(0, "viz", device))
            node_obs = model.decoder(tree.latent)
            d = torch.linalg.vector_norm(node_obs - goal_n.unsqueeze(1), dim=-1)
            d = d.masked_fill(~tree.valid, float("inf")); d[:, 0] = float("inf")
            sel = int(d.argmin(dim=1).item())
            r = tv.TreeRender.from_tree(model, tree, normalizer, goal, start, 0, sel)

            ax = axes[row][col]
            for i, j in np.argwhere(spec.maze_map == 1):
                c = spec.ij_to_xy(int(i), int(j))
                ax.add_patch(plt.Rectangle((c[0] - spec.unit / 2, c[1] - spec.unit / 2),
                                           spec.unit, spec.unit, color="0.88", zorder=0))
            for n in range(len(r.xy)):
                if not r.valid[n] or r.parent[n] < 0:
                    continue
                pnode = int(r.parent[n])
                ax.plot([r.xy[pnode, 0], r.xy[n, 0]], [r.xy[pnode, 1], r.xy[n, 1]],
                        color="0.55", lw=0.7, zorder=1)
            sc = ax.scatter(r.xy[r.valid, 0], r.xy[r.valid, 1], c=r.depth[r.valid],
                            s=20, cmap="viridis", zorder=3)
            if r.selected_path:
                pts = r.xy[np.asarray(r.selected_path)]
                ax.plot(pts[:, 0], pts[:, 1], color="orange", lw=2.4, zorder=4)
            ax.scatter(*start[:2], marker="*", s=170, color="crimson", zorder=5)
            ax.scatter(*goal[:2], marker="X", s=170, color="green", zorder=5)

            reach = float(np.linalg.norm(r.xy[r.valid] - start[None, :2], axis=1).max())
            if row == 0:
                # one grounded pass per column for the error annotation
                env.reset(seed=a)
                env.unwrapped.set_xy(np.asarray(start, dtype=np.float64))
                entry = save_state(env)
                actual, grounded = ground_tree(env, tree, normalizer)
                restore_state(env, entry)
                ok = r.valid & grounded; ok[0] = False
                err = float(np.linalg.norm(r.xy[ok] - actual[ok], axis=1).mean()) if ok.any() else float("nan")
                stats[h] = {"max_reach": reach, "grounded_err": err,
                            "max_depth": int(r.depth[r.valid].max())}
                ax.set_title(f"h={h}  |  spatial reach {reach:.1f}\nexec err {err:.2f}, "
                             f"max depth {int(r.depth[r.valid].max())}", fontsize=9)
            else:
                ax.set_title(f"h={h}  |  spatial reach {reach:.1f}", fontsize=9)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Tree geometry vs fixed edge horizon (same anchors, same node budget)",
                 fontsize=11)
    fig.tight_layout()
    out = REPO / args.out / "horizon_tree_grid.png"
    fig.savefig(out, dpi=140)
    print(f"[grid] wrote {out}")
    print(f"\n{'h':>4s} {'max spatial reach':>18s} {'grounded err':>13s} {'max depth':>10s}")
    print("-" * 50)
    for h in hs:
        s = stats.get(h, {})
        print(f"{h:4d} {s.get('max_reach', float('nan')):18.2f} "
              f"{s.get('grounded_err', float('nan')):13.2f} {s.get('max_depth', -1):10d}")


if __name__ == "__main__":
    main()
