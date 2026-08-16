"""Cross-environment scaling report: does the recursion advantage and h*(d) reproduce?

Six panels. The four requested by the report spec:

  1. Recursive - Flat success advantage vs geodesic distance, medium and large overlaid
  2. Optimal h vs task distance, medium and large overlaid
  3. Success heatmap S(h, d), both environments
  4. Recursive advantage vs path turns (compositional complexity)

plus two that are needed to read them honestly:

  5. The same advantage in *distance reduction*. Success-based Delta is bounded above by
     recursive success itself, so once both arms approach the floor Delta must fall for
     arithmetic reasons rather than scientific ones. Distance reduction is uncensored.
  6. The raw success curves, which show where that censoring starts to bite.

Panel 1 uses the *best* recursive horizon per bin, not a fixed h=20: h=20 is optimal on
medium but not on large, so fixing it would handicap one maze and confound the overlay.

Overlaying in *absolute* distance (world units) is the point: both mazes use the same
cell size and dynamics, so if the curves line up there, the effect is a function of
required horizon rather than of maze topology.

    python scripts/scaling_report.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
HORIZONS = [8, 16, 20, 32, 48, 64]
FLAT = "Fh20"
BINS = [(1, 3), (3, 5), (5, 7), (7, 9), (9, 12), (12, 16), (16, 21), (21, 26), (26, 32)]
ENVS = [("medium", "pointmaze", "pointmaze-medium-stitch-v0"),
        ("large", "pointmaze_large", "pointmaze-large-stitch-v0"),
        ("giant", "pointmaze_giant", "pointmaze-giant-stitch-v0")]
COLORS = {"medium": "tab:blue", "large": "tab:red", "giant": "tab:green"}


def turns_by_bin(env_name: str) -> dict[str, float]:
    from treewm.data.ogbench_dataset import load_ogbench
    from treewm.evaluation.tasks import build_bucketed_tasks

    b = build_bucketed_tasks(load_ogbench(env_name, env_only=True), BINS, per_bin=12, seed=0)
    return {k: float(np.mean([t["turns"] for t in v])) for k, v in b.items() if v}


def collect(name: str, tag: str, env_name: str, budgets: list[int], out: str) -> list[dict]:
    """One record per distance bin: distance, turns, per-horizon success, flat baseline."""
    p = REPO / out / f"difficulty_{tag}.json"
    if not p.exists():
        print(f"  missing: {p}")
        return []
    d = json.loads(p.read_text())
    try:
        turns = turns_by_bin(env_name)
    except Exception as exc:
        print(f"  {name}: turns unavailable ({exc})")
        turns = {}
    geo = d["_geodesic_world"]

    def mean_of(arm: str, label: str, metric: str) -> float:
        vals = [np.mean(d[arm][label][f"{metric}|b{b}"]) for b in budgets
                if arm in d and label in d[arm] and f"{metric}|b{b}" in d[arm][label]]
        return float(np.mean(vals)) if vals else float("nan")

    recs = []
    for label in sorted(geo, key=lambda k: geo[k]):
        per = {h: mean_of(f"H{h}", label, "eval/success_rate") for h in HORIZONS
               if f"H{h}" in d and label in d[f"H{h}"]}
        per = {h: v for h, v in per.items() if np.isfinite(v)}
        if not per or FLAT not in d or label not in d[FLAT]:
            continue
        per_dr = {h: mean_of(f"H{h}", label, "eval/distance_reduction") for h in per}
        recs.append({
            "env": name, "label": label, "d": geo[label], "turns": turns.get(label, float("nan")),
            "per": per, "hstar": max(per, key=per.get),
            "best": max(per.values()), "flat": mean_of(FLAT, label, "eval/success_rate"),
            "best_dr": max(per_dr.values()), "flat_dr": mean_of(FLAT, label, "eval/distance_reduction"),
        })
    return recs


def rho(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    return float(spearmanr(x[ok], y[ok]).statistic) if ok.sum() > 2 else float("nan")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", nargs="+", type=int, default=[64, 128])
    p.add_argument("--out", default="results_difficulty")
    args = p.parse_args()

    data = {n: r for n, t, e in ENVS if (r := collect(n, t, e, args.budgets, args.out))}
    if not data:
        print("no difficulty artifacts found")
        return
    pooled = [r for recs in data.values() for r in recs]

    ncol = 3 + (len(data) > 2)
    fig, axes = plt.subplots(2, ncol, figsize=(6 * ncol, 9.5), squeeze=False)

    # ---- 1. Delta success vs distance (best recursive per bin) --------------------
    ax = axes[0][0]
    print("\n=== 1. (Best recursive - Flat) SUCCESS advantage vs distance ===")
    for name, recs in data.items():
        x = [r["d"] for r in recs]
        y = [r["best"] - r["flat"] for r in recs]
        ax.plot(x, y, marker="o", color=COLORS[name], label=name)
        print(f"  {name:7s} d={np.round(x, 1).tolist()}")
        print(f"          delta={np.round(y, 3).tolist()}  spearman={rho(x, y):+.3f}")
    print(f"  POOLED spearman = {rho([r['d'] for r in pooled], [r['best'] - r['flat'] for r in pooled]):+.3f}"
          "   <-- floor-censored, see panel 5")
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xlabel("geodesic task distance (world units)")
    ax.set_ylabel("success(best recursive) - success(flat)")
    ax.set_title("1. Recursion advantage vs distance\n(success; floor-censored at large d)")
    ax.grid(alpha=.3); ax.legend()

    # ---- 2. h*(d) overlaid --------------------------------------------------------
    ax = axes[0][1]
    print("\n=== 2. Optimal horizon vs task distance ===")
    for name, recs in data.items():
        x = [r["d"] for r in recs]
        y = [r["hstar"] for r in recs]
        ax.plot(x, y, marker="s", color=COLORS[name], label=name)
        print(f"  {name:7s} d={np.round(x, 1).tolist()}")
        print(f"          h*={y}")
    print(f"  POOLED h* vs d spearman = {rho([r['d'] for r in pooled], [r['hstar'] for r in pooled]):+.3f}")
    ax.set_xlabel("geodesic task distance (world units)")
    ax.set_ylabel("optimal fixed horizon h*")
    ax.set_yticks(HORIZONS)
    ax.set_title("2. Optimal edge horizon vs distance\n(rises in both, but offset between mazes)")
    ax.grid(alpha=.3); ax.legend()

    # ---- 3. S(h, d) heatmaps ------------------------------------------------------
    print("\n=== 3. S(h,d) heatmaps ===")
    slots = [axes[0][2], axes[1][2], axes[0][3], axes[1][3]]
    for k, (name, recs) in enumerate(data.items()):
        ax = slots[k]
        grid = np.array([[r["per"].get(h, np.nan) for r in recs] for h in HORIZONS], float)
        im = ax.imshow(grid, aspect="auto", cmap="viridis", origin="lower", vmin=0, vmax=1)
        ax.set_xticks(range(len(recs)))
        ax.set_xticklabels([f"{r['d']:.0f}" for r in recs])
        ax.set_yticks(range(len(HORIZONS))); ax.set_yticklabels(HORIZONS)
        ax.set_xlabel("geodesic task distance (world units)"); ax.set_ylabel("fixed horizon h")
        ax.set_title(f"3. success S(h, d) -- {name}")
        for i in range(grid.shape[0]):
            for j in range(grid.shape[1]):
                if np.isfinite(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center",
                            color="w" if grid[i, j] < 0.6 else "k", fontsize=7)
        fig.colorbar(im, ax=ax, label="success")
        print(f"  {name}: h rows {HORIZONS}, d cols {[round(r['d'], 1) for r in recs]}")

    # ---- 4. advantage vs path turns ----------------------------------------------
    ax = axes[1][0]
    print("\n=== 4. Recursion advantage vs compositional complexity (path turns) ===")
    for name, recs in data.items():
        x = [r["turns"] for r in recs]
        y = [r["best_dr"] - r["flat_dr"] for r in recs]
        ax.plot(x, y, marker="^", color=COLORS[name], label=name)
        print(f"  {name:7s} turns={np.round(x, 1).tolist()}")
        print(f"          delta_distred={np.round(y, 2).tolist()}  spearman={rho(x, y):+.3f}")
    pt, pd_ = [r["turns"] for r in pooled], [r["best_dr"] - r["flat_dr"] for r in pooled]
    print(f"  POOLED vs turns = {rho(pt, pd_):+.3f}   vs distance = "
          f"{rho([r['d'] for r in pooled], pd_):+.3f}")
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xlabel("mean path turns (compositional complexity)")
    ax.set_ylabel("distance reduction: recursive - flat")
    ax.set_title("4. Recursion advantage vs path turns")
    ax.grid(alpha=.3); ax.legend()

    # ---- 5. uncensored advantage: distance reduction ------------------------------
    ax = axes[1][1]
    print("\n=== 5. UNCENSORED advantage (distance reduction) vs distance ===")
    for name, recs in data.items():
        x = [r["d"] for r in recs]
        y = [r["best_dr"] - r["flat_dr"] for r in recs]
        ax.plot(x, y, marker="o", color=COLORS[name], label=name)
        print(f"  {name:7s} delta={np.round(y, 2).tolist()}  spearman={rho(x, y):+.3f}")
    print(f"  POOLED spearman = {rho([r['d'] for r in pooled], pd_):+.3f}   <-- the scaling claim")
    ax.axhline(0, color="0.6", lw=1)
    ax.set_xlabel("geodesic task distance (world units)")
    ax.set_ylabel("distance reduction: recursive - flat")
    ax.set_title("5. Recursion advantage, uncensored\n(distance reduction, both mazes)")
    ax.grid(alpha=.3); ax.legend()

    fig.tight_layout()
    out = REPO / args.out / "scaling_report.png"
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
