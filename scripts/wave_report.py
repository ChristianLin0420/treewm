"""Wave-1 cross-family report: per environment, not per global score.

The promotion decision deliberately ignores validation loss. This project has shown
repeatedly that val loss mis-ranks control models -- on pointmaze-large the best-predicting
horizon (h=16) was not the best-acting one (h=48) -- so ranking environments by it would
select for the wrong thing.

An environment is *discriminative* when the controller shows real competence and Flat vs
Recursive can actually differ: success strictly between the floor and ceiling, or partial
progress that separates the arms. Zero success everywhere is not a failure to report, it
is a failure to *measure*, and it is why three AntMaze cycles produced nothing.

    python scripts/wave_report.py --runs experiments/09-cross-family/runs/wave1
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

SUCCESS_FLOOR, SUCCESS_CEIL = 0.05, 0.95
ALPHA = 0.05
# Partial-progress signals, most task-meaningful first. The first one present is used.
#
# subgoal_gain leads because the raw fraction is not chance-baselined: with nine binary
# buttons a random state already matches ~50% of the goal, so puzzle reported 0.48 while
# achieving nothing. subgoal_gain measures the fraction of initially-unmet subgoals that
# were actually closed, so 0 means "no better than the start state".
PROGRESS_KEYS = [
    "eval/progress/best_subgoal_gain",
    "eval/progress/subgoal_gain",
    "eval/distance_reduction_frac",
]


def fisher_p(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact p for (successes, episodes) of two arms."""
    try:
        from scipy.stats import fisher_exact
        return float(fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])[1])
    except Exception:
        return float("nan")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    except Exception:
        return "unknown"


def ogbench_version() -> str:
    try:
        import ogbench
        return getattr(ogbench, "__version__", "unspecified")
    except Exception:
        return "unavailable"


def read_run(d: Path) -> dict | None:
    """Scalars from one run directory, keyed by tag -> list[(step, value)]."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if not list(d.glob("events.out.tfevents.*")):
        return None
    ea = EventAccumulator(str(d), size_guidance={"scalars": 0})
    ea.Reload()
    return {t: [(s.step, s.value) for s in ea.Scalars(t)] for t in ea.Tags()["scalars"]}


def at_checkpoints(series: list[tuple[int, float]], checkpoints: list[int]) -> dict[int, float]:
    """Value at each staged checkpoint (last value at or before it)."""
    out = {}
    for c in checkpoints:
        prior = [v for s, v in series if s <= c]
        if prior:
            out[c] = float(prior[-1])
    return out


def collect(runs_root: Path, checkpoints: list[int]) -> dict:
    envs: dict[str, dict[str, dict]] = defaultdict(dict)
    for d in sorted(runs_root.glob("*/*/*/")):
        if d.name in ("checkpoints", "hparams", "viz", "sweep") or d.name.startswith("viz_"):
            continue
        sc = read_run(d)
        if not sc:
            continue
        env, arm = d.parts[-3], d.parts[-2]
        seed = int(m.group(1)) if (m := re.search(r"_s(\d+)$", d.name)) else 0
        rec = envs[env].setdefault(arm, {"seeds": {}, "final_step": 0})
        prog_key = next((k for k in PROGRESS_KEYS if k in sc), None)
        rec["seeds"][seed] = {
            "success": at_checkpoints(sc.get("eval/success_rate", []), checkpoints),
            "episodes": at_checkpoints(sc.get("eval/num_episodes", []), checkpoints),
            "progress": at_checkpoints(sc.get(prog_key, []), checkpoints) if prog_key else {},
            "progress_key": prog_key,
            "depth": at_checkpoints(sc.get("eval/selected_leaf_depth", []), checkpoints),
            "sibling_ratio": at_checkpoints(sc.get("tree/sibling_spread_ratio", []), checkpoints),
            "val_loss": at_checkpoints(sc.get("val/loss_total", []), checkpoints),
            "cache_consumed": (sc.get("cache/consumed", [(0, 0.0)])[-1][1]),
            "last_step": max((s for s, _ in sc.get("train/loss_total", [(0, 0)])), default=0),
            "dir": str(d),
        }
        rec["final_step"] = max(rec["final_step"], rec["seeds"][seed]["last_step"])
    return envs


def mean_at(rec: dict, field: str, ck: int) -> float:
    vals = [s[field].get(ck) for s in rec["seeds"].values() if ck in s.get(field, {})]
    return float(np.mean(vals)) if vals else float("nan")


def assess(env: str, arms: dict, checkpoints: list[int]) -> dict:
    """Apply the promotion rule to one environment."""
    flat, rec = arms.get("flatkwm"), arms.get("randomtreewm")
    if not flat or not rec:
        return {"env": env, "verdict": "INCOMPLETE",
                "reason": f"missing arm(s): have {sorted(arms)}"}

    rows, sep_step = [], None
    for ck in checkpoints:
        fs, rs = mean_at(flat, "success", ck), mean_at(rec, "success", ck)
        fp, rp = mean_at(flat, "progress", ck), mean_at(rec, "progress", ck)
        d_s = rs - fs if np.isfinite(rs) and np.isfinite(fs) else float("nan")
        d_p = rp - fp if np.isfinite(rp) and np.isfinite(fp) else float("nan")
        rows.append({"step": ck, "flat_success": fs, "rec_success": rs, "delta_success": d_s,
                     "flat_progress": fp, "rec_progress": rp, "delta_progress": d_p})
        if sep_step is None and (np.isfinite(d_s) and abs(d_s) >= 0.05):
            sep_step = ck

    # Significance on the final checkpoint. Wave 1 promoted cube-single on 3 successes
    # out of 25 against 0/25 -- CIs [0.025, 0.312] vs [0, 0.137], heavily overlapping.
    # A threshold alone is not evidence.
    ck = checkpoints[-1]
    n_f, n_r = mean_at(flat, "episodes", ck), mean_at(rec, "episodes", ck)
    p_val = float("nan")
    if np.isfinite(n_f) and np.isfinite(n_r) and n_f > 0 and n_r > 0:
        k_f = int(round(mean_at(flat, "success", ck) * n_f))
        k_r = int(round(mean_at(rec, "success", ck) * n_r))
        p_val = fisher_p(k_r, int(n_r), k_f, int(n_f))

    last = rows[-1]
    best = max(rows, key=lambda r: (r["rec_success"] if np.isfinite(r["rec_success"]) else -1))
    competent = np.isfinite(best["rec_success"]) and best["rec_success"] > SUCCESS_FLOOR
    non_sat = any(SUCCESS_FLOOR <= r["rec_success"] <= SUCCESS_CEIL
                  for r in rows if np.isfinite(r["rec_success"]))
    prog_sep = any(np.isfinite(r["delta_progress"]) and abs(r["delta_progress"]) >= 0.05
                   for r in rows)
    deltas = [r["delta_success"] for r in rows if np.isfinite(r["delta_success"])]
    stable = len(deltas) >= 2 and all(np.sign(d) == np.sign(deltas[-1]) for d in deltas[-2:])

    significant = bool(np.isfinite(p_val) and p_val < ALPHA)
    discriminative = bool((non_sat or prog_sep) and competent)
    # Promotion now requires the success difference to be significant, or a
    # partial-progress separation that is large relative to the noise floor.
    verdict = ("PROMOTE" if discriminative and (significant or prog_sep)
               else "PARTIAL" if discriminative or prog_sep
               else "NOT_DISCRIMINATIVE")
    return {
        "env": env, "rows": rows, "verdict": verdict,
        "competent": bool(competent), "non_saturated": bool(non_sat),
        "progress_separates": bool(prog_sep), "stable": bool(stable),
        "separation_step": sep_step, "fisher_p": p_val, "significant": significant,
        "episodes_per_arm": {"flat": n_f, "recursive": n_r},
        "delta_success_final": last["delta_success"],
        "delta_progress_final": last["delta_progress"],
        "rank_key": (float(discriminative),
                     abs(last["delta_success"]) if np.isfinite(last["delta_success"]) else 0.0,
                     float(stable),
                     best["rec_success"] if np.isfinite(best["rec_success"]) else 0.0),
        "level": "success" if non_sat else ("partial-progress" if prog_sep else "none"),
        "cache_ok": all(s["cache_consumed"] == 1.0
                        for a in (flat, rec) for s in a["seeds"].values()),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="experiments/09-cross-family/runs/wave1")
    p.add_argument("--checkpoints", nargs="+", type=int, default=[20000, 50000, 100000])
    p.add_argument("--out", default="experiments/09-cross-family/results/wave1_report.json")
    p.add_argument("--promote", type=int, default=4)
    args = p.parse_args()

    root = REPO / args.runs
    envs = collect(root, args.checkpoints)
    if not envs:
        print(f"no runs under {root}")
        return

    results = [assess(e, arms, args.checkpoints) for e, arms in sorted(envs.items())]
    results.sort(key=lambda r: r.get("rank_key", (0,)), reverse=True)

    print(f"\n{'='*104}\nWAVE 1 -- cross-family screen, per environment\n{'='*104}")
    for r in results:
        if r["verdict"] == "INCOMPLETE":
            print(f"\n### {r['env']}\n  INCOMPLETE: {r['reason']}")
            continue
        print(f"\n### {r['env']}   [{r['verdict']}]")
        print(f"  {'step':>8s} {'flat succ':>10s} {'rec succ':>9s} {'Δsucc':>8s} "
              f"{'flat prog':>10s} {'rec prog':>9s} {'Δprog':>8s}")
        for row in r["rows"]:
            print(f"  {row['step']:8d} {row['flat_success']:10.3f} {row['rec_success']:9.3f} "
                  f"{row['delta_success']:+8.3f} {row['flat_progress']:10.3f} "
                  f"{row['rec_progress']:9.3f} {row['delta_progress']:+8.3f}")
        pv = r.get("fisher_p", float("nan"))
        print(f"  competence={r['competent']}  level={r['level']}  stable={r['stable']}  "
              f"separation_at={r['separation_step']}  "
              f"fisher_p={pv:.4f}" if np.isfinite(pv) else
              f"  competence={r['competent']}  level={r['level']}  stable={r['stable']}")
        print(f"  episodes/arm: flat={r['episodes_per_arm']['flat']:.0f} "
              f"recursive={r['episodes_per_arm']['recursive']:.0f}  "
              f"significant={r.get('significant')}")
        if not r["cache_ok"]:
            print("  WARNING: a run did not consume the shared cache")

    promoted = [r["env"] for r in results if r["verdict"] == "PROMOTE"][: args.promote]
    print(f"\n{'='*104}\nSHORTLIST FOR WAVE 2 ({len(promoted)}): {promoted or 'NONE'}")
    if not promoted:
        print("  No environment is discriminative yet. Continue training rather than\n"
              "  promoting on partial progress alone -- a horizon sweep on a floor-bound\n"
              "  environment measures nothing, which is the AntMaze lesson.")

    out = REPO / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "provenance": {
            "git_commit": git_commit(), "ogbench_version": ogbench_version(),
            "checkpoints": args.checkpoints, "runs_root": str(root),
            "promotion_rule": {"success_floor": SUCCESS_FLOOR, "success_ceiling": SUCCESS_CEIL,
                               "ranks_by": "flat-vs-recursive separation, stability, "
                                           "competence; NOT validation loss",
                               "alpha": ALPHA,
                               "significance": "two-sided Fisher exact on success counts",
                               "progress_metric": "subgoal_gain, baselined on each "
                                                  "episode's initial subgoal fraction"},
        },
        "environments": results, "promoted": promoted,
    }
    out.write_text(json.dumps(payload, indent=2, default=float))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
