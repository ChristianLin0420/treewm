"""Fixed-horizon sweep report: control performance vs prediction/execution error.

The hypothesis is an intermediate optimum: short edges are individually easy but need
many recursive compositions; long edges need few compositions but are individually hard.
This reports both sides of that tradeoff against h, and plots them side by side.

    python scripts/horizon_sweep_report.py --runs runs_h --results results_h
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

REPO = Path(__file__).resolve().parents[1]


def horizon_of(recipe: str) -> int | None:
    m = re.match(r"^H(\d+)$", recipe)
    return int(m.group(1)) if m else None


def read_training(runs_root: Path) -> dict[int, dict[str, list[float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for d in sorted(runs_root.glob("*/*/*/")):
        name = d.name
        if name in {"checkpoints", "hparams", "viz", "sweep"} or name.startswith("viz_"):
            continue
        if not any(d.glob("events.out.tfevents.*")):
            continue
        h = horizon_of(re.sub(r"_s\d+$", "", name))
        if h is None:
            continue
        ea = EventAccumulator(str(d), size_guidance={"scalars": 0})
        ea.Reload()
        tags = set(ea.Tags()["scalars"])
        for tag in ["val/loss_total", "model/state_latent_mse", "model/action_mse",
                    "eval/success_rate", "eval/selected_leaf_depth",
                    "eval/action_chunk_execution_length", "tree/mean_depth", "tree/max_depth",
                    "eval/goal_distance_final", "model/state_decode_mse"]:
            if tag in tags:
                vals = [s.value for s in ea.Scalars(tag)]
                out[h][tag].append(float(np.mean(vals[-3:])))
    return out


@torch.no_grad()
def grounded_error(runs_root: Path, budget: int = 64, max_nodes: int = 10) -> dict[int, float]:
    """Executed endpoint error per horizon: how far predictions land from reality."""
    from treewm.data.ogbench_dataset import load_ogbench
    from treewm.evaluation.grounding import ground_tree, predicted_xy, restore_state, save_state
    from treewm.evaluation.tree_viz import build_anchors
    from treewm.data.maze_utils import MazeSpec
    from treewm.models.baselines import tree_config_for
    from treewm.utils import config as cfg_utils
    from treewm.utils.rng import make_generator
    from scripts.eval import load_run

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out: dict[int, list[float]] = defaultdict(list)
    env = spec = anchors = None
    for ck in sorted(runs_root.glob("*/*/*/checkpoints/latest.pt")):
        h = horizon_of(re.sub(r"_s\d+$", "", ck.parts[-3]))
        if h is None:
            continue
        model, normalizer, cfg, _ = load_run(str(ck), device)
        if model.decoder is None:
            continue
        if env is None:
            env = load_ogbench(cfg.env.name, dataset_dir=cfg.env.dataset_dir, env_only=True)
            spec = MazeSpec.from_env(env)
            anchors = build_anchors(spec, num=4)
        tc = tree_config_for("randomtreewm",
                             replace(cfg_utils.tree_config(cfg), node_budget=budget), model)
        errs = []
        for a in range(len(anchors)):
            start = anchors.starts[a]
            env.reset(seed=a)
            env.unwrapped.set_xy(np.asarray(start, dtype=np.float64))
            obs = torch.from_numpy(normalizer.norm_obs(start[None])).to(device)
            tree, _ = model.generate(model.encode(obs), tc,
                                     generator=make_generator(0, "viz", device))
            pred = predicted_xy(model, tree, normalizer)
            entry = save_state(env)
            actual, grounded = ground_tree(env, tree, normalizer)
            restore_state(env, entry)
            ok = tree.valid[0].cpu().numpy() & grounded
            ok[0] = False
            if ok.any():
                errs.append(float(np.linalg.norm(pred[ok] - actual[ok], axis=1).mean()))
        if errs:
            out[h].append(float(np.mean(errs)))
        print(f"  grounded h={h} {ck.parts[-3]}", flush=True)
    return {h: float(np.mean(v)) for h, v in out.items()}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="experiments/07-horizon-antmaze/runs/h")
    p.add_argument("--results", default="experiments/07-horizon-antmaze/results/h")
    p.add_argument("--budgets", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    p.add_argument("--skip-grounded", action="store_true")
    args = p.parse_args()

    runs_root = REPO / args.runs
    train = read_training(runs_root)
    curves_path = REPO / args.results / "budget_curves.json"
    curves = json.loads(curves_path.read_text()) if curves_path.exists() else {}

    per_h_auc: dict[int, list[float]] = {}
    per_h_budget: dict[int, list[float]] = {}
    for key, pts in curves.items():
        h = horizon_of(key.split("|")[1])
        if h is None:
            continue
        n = len(pts[str(args.budgets[0])])
        per_h_auc[h] = [float(np.mean([pts[str(b)][i] for b in args.budgets])) for i in range(n)]
        per_h_budget[h] = [float(np.mean(pts[str(b)])) for b in args.budgets]

    grounded = {} if args.skip_grounded else grounded_error(runs_root)
    hs = sorted(set(train) | set(per_h_auc))

    print(f"\n=== fixed-horizon sweep: control vs prediction error ===")
    hdr = (f"{'h':>4s} {'success AUC':>18s} {'succ@64':>8s} {'val_loss':>9s} {'state_mse':>10s} "
           f"{'action_mse':>11s} {'grounded_err':>13s} {'mean_depth':>11s} {'leaf_depth':>11s} "
           f"{'reach(steps)':>13s}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for h in hs:
        t = train.get(h, {})
        auc = np.mean(per_h_auc[h]) if h in per_h_auc else float("nan")
        sd = np.std(per_h_auc[h], ddof=1) if h in per_h_auc and len(per_h_auc[h]) > 1 else float("nan")
        succ64 = per_h_budget[h][args.budgets.index(64)] if h in per_h_budget and 64 in args.budgets else float("nan")
        leaf = float(np.mean(t.get("eval/selected_leaf_depth", [np.nan])))
        # temporal reach of the selected path, in primitive steps
        reach = leaf * h
        rows.append((h, auc, sd, succ64, float(np.mean(t.get("val/loss_total", [np.nan]))),
                     float(np.mean(t.get("model/state_latent_mse", [np.nan]))),
                     float(np.mean(t.get("model/action_mse", [np.nan]))),
                     grounded.get(h, float("nan")),
                     float(np.mean(t.get("tree/mean_depth", [np.nan]))), leaf, reach))
        print(f"{h:4d} {auc:9.3f}+/-{sd:<6.3f} {succ64:8.3f} {rows[-1][4]:9.3f} {rows[-1][5]:10.5f} "
              f"{rows[-1][6]:11.4f} {rows[-1][7]:13.3f} {rows[-1][8]:11.2f} {leaf:11.2f} {reach:13.1f}")

    finite = [(h, a) for h, a, *_ in rows if np.isfinite(a)]
    if finite:
        best_h, best_a = max(finite, key=lambda r: r[1])
        order = [h for h, _ in sorted(finite, key=lambda r: r[0])]
        aucs = [a for _, a in sorted(finite, key=lambda r: r[0])]
        interior = order.index(best_h) not in (0, len(order) - 1)
        print(f"\n  best h = {best_h} (AUC {best_a:.3f})")
        print(f"  inverted-U (optimum strictly interior)? {'YES' if interior else 'NO'}")
        (REPO / args.results).mkdir(parents=True, exist_ok=True)
        (REPO / args.results / "horizon_summary.json").write_text(json.dumps(
            {"best_h": best_h, "interior_optimum": bool(interior),
             "auc": {str(h): a for h, a in finite},
             "per_seed_auc": {str(h): v for h, v in per_h_auc.items()},
             "grounded_error": {str(h): v for h, v in grounded.items()}}, indent=2))

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        axes[0].errorbar(order, aucs,
                         yerr=[np.std(per_h_auc[h], ddof=1) if len(per_h_auc[h]) > 1 else 0
                               for h in order], marker="o", capsize=3)
        axes[0].set_xlabel("fixed edge horizon h (primitive steps)")
        axes[0].set_ylabel("success AUC (budgets 16-256)")
        axes[0].set_title("control performance vs h")
        axes[0].grid(alpha=0.3)

        ax2 = axes[1]
        vl = [float(np.mean(train[h].get("val/loss_total", [np.nan]))) for h in order]
        ax2.plot(order, vl, marker="s", color="tab:red", label="val loss")
        ax2.set_xlabel("fixed edge horizon h (primitive steps)")
        ax2.set_ylabel("validation loss", color="tab:red")
        ax2.grid(alpha=0.3)
        if grounded:
            ax3 = ax2.twinx()
            ge = [grounded.get(h, np.nan) for h in order]
            ax3.plot(order, ge, marker="^", color="tab:blue", label="grounded endpoint error")
            ax3.set_ylabel("executed endpoint error", color="tab:blue")
        ax2.set_title("prediction / execution error vs h")
        fig.tight_layout()
        out = REPO / args.results / "horizon_tradeoff.png"
        fig.savefig(out, dpi=140)
        print(f"  wrote {out}")

    if curves:
        print(f"\n=== success vs node budget, per horizon ===")
        print(f"{'h':>4s} " + " ".join(f"{b:>7d}" for b in args.budgets))
        print("-" * (4 + 8 * len(args.budgets)))
        for h in sorted(per_h_budget):
            print(f"{h:4d} " + " ".join(f"{v:7.3f}" for v in per_h_budget[h]))


if __name__ == "__main__":
    main()
