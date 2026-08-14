"""Phase-1 screening analysis and promotion.

Promotion is deliberately multi-criteria (spec section 10): a recipe qualifies if it
improves at least two meaningful aspects relative to the A0_random baseline, not if one
scalar moved. Ranking on success alone is what produced the earlier
coverage-vs-success contradiction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[1]

# (tag, higher_is_better, human label)
CRITERIA = [
    ("eval/success_rate", True, "success"),
    ("eval/goal_distance_final", False, "goal_dist"),
    ("model/state_latent_mse", False, "state_mse"),
    ("recursive/state_error_depth3", False, "rec_err_d3"),
    ("tree/effective_branching_factor", True, "EBF"),
    ("tree/mean_pairwise_z_distance", True, "branch_div"),
    ("eval/selected_leaf_depth", True, "leaf_depth"),
    ("val/loss_total", False, "val_loss"),
]


def read_run(run_dir: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    out = {}
    for t in tags:
        try:
            vals = [s.value for s in ea.Scalars(t)]
            out[t] = float(np.mean(vals[-3:])) if len(vals) >= 3 else float(vals[-1])
        except (KeyError, IndexError):
            continue
    out["_steps"] = max((s.step for s in ea.Scalars("train/loss_total")), default=0) if \
        "train/loss_total" in tags else 0
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs_screen")
    p.add_argument("--baseline", default="A0_random_s0")
    p.add_argument("--min-criteria", type=int, default=2)
    p.add_argument("--rel-threshold", type=float, default=0.05)
    p.add_argument("--out", default="results_screen")
    args = p.parse_args()

    runs: dict[str, dict] = {}
    for d in sorted((REPO / args.runs).glob("*/*/*/")):
        if d.name in {"checkpoints", "hparams", "figures", "viz", "sweep"} or d.name.startswith("viz_"):
            continue
        if not any(d.glob("events.out.tfevents.*")):
            continue
        runs[d.name] = read_run(d)
    if not runs:
        print(f"no runs under {args.runs}")
        return
    if args.baseline not in runs:
        print(f"baseline {args.baseline} missing; have {sorted(runs)}")
        return

    base = runs[args.baseline]
    incomplete = {k: v["_steps"] for k, v in runs.items() if v["_steps"] < 0.95 * max(
        r["_steps"] for r in runs.values())}

    print(f"=== Phase-1 screening ({len(runs)} runs, baseline {args.baseline}) ===")
    if incomplete:
        print(f"  INCOMPLETE (fewer steps): {incomplete}")
    header = f"{'recipe':20s} {'steps':>7s} " + " ".join(f"{lbl:>11s}" for _, _, lbl in CRITERIA) + "  wins"
    print(header)
    print("-" * len(header))

    scored = []
    for name in sorted(runs):
        r = runs[name]
        cells, wins = [], 0
        for tag, higher, _ in CRITERIA:
            if tag not in r:
                cells.append(f"{'-':>11s}")
                continue
            cells.append(f"{r[tag]:11.3f}")
            # A criterion the baseline does not log (e.g. recursive depth errors, which
            # only exist for multi-step runs) is still *shown*, but cannot be scored as a
            # win because there is nothing to compare against.
            if name == args.baseline or tag not in base or abs(base[tag]) < 1e-9:
                continue
            rel = (r[tag] - base[tag]) / abs(base[tag]) * (1 if higher else -1)
            if rel > args.rel_threshold:
                wins += 1
        scored.append((name, wins, r))
        mark = "  <- baseline" if name == args.baseline else ""
        print(f"{name:20s} {r['_steps']:7d} " + " ".join(cells) + f"  {wins:4d}{mark}")

    # Sorted by how many criteria improved, so a consumer taking the top-k gets the
    # *strongest* recipes. Alphabetical order silently promoted the weakest two.
    promoted = [n for n, w, _ in sorted(scored, key=lambda x: -x[1])
                if n != args.baseline and w >= args.min_criteria and n not in incomplete]
    print(f"\n=== promotion (>= {args.min_criteria} criteria improved by >{args.rel_threshold:.0%}) ===")
    for n, w, _ in sorted(scored, key=lambda x: -x[1]):
        if n == args.baseline:
            continue
        status = "PROMOTE" if n in promoted else ("incomplete" if n in incomplete else "drop")
        print(f"  {n:20s} criteria={w}  -> {status}")

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "screen_summary.json").write_text(json.dumps(
        {"baseline": args.baseline, "promoted": promoted,
         "runs": {n: {k: v for k, v in r.items()} for n, _, r in scored}},
        indent=2, default=str))
    print(f"\nwrote {out_dir/'screen_summary.json'}")


if __name__ == "__main__":
    main()
