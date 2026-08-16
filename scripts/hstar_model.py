"""Fit a simple explanatory model for h*(task), pooled across maze layouts.

The two-maze result was ambiguous by construction: medium is short+twisty and large is
long+straight, so task distance and corridor straightness are collinear and no fit on
those two can say which one sets the optimal edge horizon. giant (long+twisty) breaks
that collinearity, which is the only reason a fit is worth running at all.

The candidate predictors correspond to the two competing paper claims:

    d        geodesic task distance      -> "task-horizon-dependent temporal resolution"
    seg      mean straight-segment length -> "environment decision-timescale-dependent"
                                             temporal resolution (the stronger claim)

seg = d / (turns + 1) is the mean run between direction changes. It is the natural
mechanistic candidate: an edge executes an open-loop action chunk that cannot turn
mid-chunk, so the affordable horizon should be set by how far you can go straight,
not by how far you must ultimately travel.

Models are compared by leave-one-environment-out R^2 on log2(h*), which is the honest
test -- in-sample R^2 will always favour the model with more parameters, and the claim
we care about is precisely whether a fit transfers to a *held-out layout*.

    python scripts/hstar_model.py
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]
HORIZONS = [8, 16, 20, 32, 48, 64]
BINS = [(1, 3), (3, 5), (5, 7), (7, 9), (9, 12), (12, 16), (16, 21), (21, 26), (26, 32)]
ENVS = [("medium", "pointmaze", "pointmaze-medium-stitch-v0"),
        ("large", "pointmaze_large", "pointmaze-large-stitch-v0"),
        ("giant", "pointmaze_giant", "pointmaze-giant-stitch-v0")]


def turns_by_bin(env_name: str) -> dict[str, float]:
    from treewm.data.ogbench_dataset import load_ogbench
    from treewm.evaluation.tasks import build_bucketed_tasks

    b = build_bucketed_tasks(load_ogbench(env_name, env_only=True), BINS, per_bin=12, seed=0)
    return {k: float(np.mean([t["turns"] for t in v])) for k, v in b.items() if v}


def collect(budgets: list[int], out: str) -> list[dict]:
    rows = []
    for name, tag, env_name in ENVS:
        p = REPO / out / f"difficulty_{tag}.json"
        if not p.exists():
            print(f"  missing {p} -- {name} excluded")
            continue
        d = json.loads(p.read_text())
        turns = turns_by_bin(env_name)
        geo = d["_geodesic_world"]
        for label in sorted(geo, key=lambda k: geo[k]):
            per = {}
            for h in HORIZONS:
                arm = f"H{h}"
                if arm in d and label in d[arm]:
                    v = [np.mean(d[arm][label][f"eval/success_rate|b{b}"]) for b in budgets
                         if f"eval/success_rate|b{b}" in d[arm][label]]
                    if v:
                        per[h] = float(np.mean(v))
            # a bin where nothing succeeds cannot identify an optimum
            if not per or max(per.values()) <= 0.02 or label not in turns:
                continue
            t = turns[label]
            rows.append({"env": name, "label": label, "d": geo[label], "turns": t,
                         "seg": geo[label] / (t + 1.0), "hstar": max(per, key=per.get),
                         "best": max(per.values()), "per": per})
    return rows


def fit_predict(X_tr, y_tr, X_te):
    """Least-squares with intercept; returns predictions on X_te."""
    A = np.hstack([np.ones((len(X_tr), 1)), X_tr])
    coef, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    return np.hstack([np.ones((len(X_te), 1)), X_te]) @ coef, coef


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budgets", nargs="+", type=int, default=[64, 128])
    p.add_argument("--out", default="results_difficulty")
    args = p.parse_args()

    rows = collect(args.budgets, args.out)
    envs = sorted({r["env"] for r in rows})
    print(f"\n{len(rows)} identifiable bins across {len(envs)} layouts: {envs}")
    print(f"\n{'env':8s} {'bin':>7s} {'d':>6s} {'turns':>6s} {'seg':>6s} {'h*':>4s} {'best':>6s}")
    for r in rows:
        print(f"{r['env']:8s} {r['label']:>7s} {r['d']:6.1f} {r['turns']:6.1f} "
              f"{r['seg']:6.2f} {r['hstar']:4d} {r['best']:6.3f}")

    if len({r["env"] for r in rows}) < 2:
        print("\nneed >=2 layouts to compare models -- stopping")
        return

    # collinearity check: the whole point of adding giant
    d = np.array([r["d"] for r in rows]); seg = np.array([r["seg"] for r in rows])
    print(f"\ncorr(d, seg) pooled = {np.corrcoef(d, seg)[0, 1]:+.3f}  "
          "(near +-1 would mean the layouts still cannot separate the two claims)")

    y = np.log2([r["hstar"] for r in rows])
    feats = {"d": d, "log d": np.log2(d), "seg": seg, "log seg": np.log2(seg),
             "turns": np.array([r["turns"] for r in rows])}
    envarr = np.array([r["env"] for r in rows])

    candidates = [("constant", [])]
    candidates += [(k, [k]) for k in feats]
    candidates += [(f"{a} + {b}", [a, b]) for a, b in combinations(feats, 2)
                   if not (a.startswith("log") ^ b.startswith("log"))]

    print(f"\n=== leave-one-layout-out R^2 on log2(h*) ===")
    print(f"{'model':22s} {'LOLO R^2':>9s} {'in-sample R^2':>14s}   per-layout held-out R^2")
    results = []
    for name, keys in candidates:
        pred = np.zeros_like(y)
        per_env = {}
        for e in envs:
            te = envarr == e
            X_tr = np.column_stack([feats[k][~te] for k in keys]) if keys else np.zeros((int((~te).sum()), 0))
            X_te = np.column_stack([feats[k][te] for k in keys]) if keys else np.zeros((int(te.sum()), 0))
            pred[te], _ = fit_predict(X_tr, y[~te], X_te)
            ss = ((y[te] - pred[te]) ** 2).sum()
            per_env[e] = 1 - ss / max(((y[te] - y[~te].mean()) ** 2).sum(), 1e-9)
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        X = np.column_stack([feats[k] for k in keys]) if keys else np.zeros((len(y), 0))
        ins_pred, coef = fit_predict(X, y, X)
        ins = 1 - ((y - ins_pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        results.append((r2, name, ins, per_env, keys, coef))
    for r2, name, ins, per_env, keys, coef in sorted(results, reverse=True):
        detail = "  ".join(f"{e}={per_env[e]:+.2f}" for e in envs)
        print(f"{name:22s} {r2:9.3f} {ins:14.3f}   {detail}")

    best = max(results)
    print(f"\nbest transferring model: {best[1]}   (LOLO R^2 = {best[0]:.3f})")
    if best[4]:
        terms = "  ".join(f"{c:+.3f}*{k}" for c, k in zip(best[5][1:], best[4]))
        print(f"  log2(h*) = {best[5][0]:+.3f}  {terms}")
    print("\nreading: if a distance-only model transfers, the claim is task-horizon-dependent")
    print("temporal resolution. If seg (straight-run length) is needed, the claim is the")
    print("stronger environment-decision-timescale one. If 'constant' wins, h* does not")
    print("scale at all and both claims fail.")


if __name__ == "__main__":
    main()
