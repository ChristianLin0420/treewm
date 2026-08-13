"""Collect finished runs into the ladder table and the diagnostics table.

    python scripts/aggregate_results.py                      # all datasets
    python scripts/aggregate_results.py --dataset pointmaze-medium-stitch

Reads TensorBoard event files rather than checkpoints, so it works on runs that are
still in progress (it reports the latest logged value and flags incomplete runs).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]

LADDER = ["singlewm", "flatkwm", "fixedtreewm", "treewm"]
CONTROLS = ["randomtreewm", "uncertaintytreewm", "heuristictreewm"]
ARM_ORDER = ["singlewm", "flatkwm", "fixedtreewm", "randomtreewm", "uncertaintytreewm",
             "heuristictreewm", "treewm", "noveltyq", "learnedq", "noveltyz", "learnedz"]

PRIMARY = [
    ("eval/success_rate", "success"),
    ("eval/goal_distance_final", "goal_dist"),
    ("eval/world_model_nodes_per_replan", "nodes/replan"),
    ("eval/action_chunk_execution_length", "chunk_len"),
]

NOVELTY_DIAG = [
    ("expansion/gain_rank_correlation", "spearman"),
    ("expansion/gain_pearson_correlation", "pearson"),
    ("expansion/controllability_coverage", "coverage"),
    ("expansion/coverage_per_expanded_node", "cov/expanded"),
    ("expansion/redundant_expansion_fraction", "redundant"),
    ("expansion/mean_depth", "mean_depth"),
    ("expansion/depth_std", "depth_std"),
    ("expansion/frontier_novelty_decay", "front_decay"),
]

DIAGNOSTICS = [
    ("tree/effective_branching_factor", "EBF"),
    ("tree/rare_mode_recall", "rare_recall"),
    ("tree/support_recall", "supp_recall"),
    ("expansion/controllability_coverage", "coverage"),
    ("expansion/controllability_coverage_per_node", "cov/node"),
    ("expansion/gain_rank_correlation", "gain_rho"),
    ("control/branching_future_diversity_corr", "branch~div"),
    ("control/q_advantage_over_z", "q-z"),
    ("control/q_advantage_over_random_proj", "q-rand"),
]


def read_run(run_dir: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    tags = set(ea.Tags()["scalars"])
    out: dict[str, float] = {}
    for tag in tags:
        try:
            out[tag] = ea.Scalars(tag)[-1].value
        except (KeyError, IndexError):
            continue
    steps = [s.step for s in ea.Scalars("train/loss_total")] if "train/loss_total" in tags else [0]
    out["_last_step"] = max(steps)
    out["_has_eval"] = "eval/success_rate" in tags
    return out


def collect(runs_root: Path) -> dict:
    results: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for run_dir in sorted(runs_root.glob("*/*/*/")):
        if run_dir.name in {"hparams", "checkpoints", "figures", "viz", "sweep"}:
            continue
        if not any(run_dir.glob("events.out.tfevents.*")):
            continue
        dataset, arm = run_dir.parts[-3], run_dir.parts[-2]
        try:
            results[(dataset, arm)].append(read_run(run_dir))
        except Exception as exc:  # a half-written event file must not kill the report
            print(f"  ! skipped {run_dir}: {exc}")
    return results


def agg(values: list[float]) -> tuple[float, float, int]:
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    return float(arr.mean()), float(arr.std()), int(arr.size)


def table(results: dict, dataset: str, keys: list[tuple[str, str]], title: str) -> str:
    arms = [a for a in ARM_ORDER if (dataset, a) in results]
    if not arms:
        return ""
    lines = [f"\n### {title} — {dataset}\n"]
    header = f"{'arm':20s} {'n':>2s} {'steps':>7s} " + " ".join(f"{lbl:>14s}" for _, lbl in keys)
    lines.append(header)
    lines.append("-" * len(header))
    for arm in arms:
        runs = results[(dataset, arm)]
        steps = int(np.median([r["_last_step"] for r in runs]))
        row = f"{arm:20s} {len(runs):2d} {steps:7d} "
        cells = []
        for tag, _ in keys:
            m, s, n = agg([r.get(tag) for r in runs])
            cells.append("           n/a" if n == 0 else f"{m:>8.3f}±{s:<5.3f}")
        lines.append(row + " ".join(cells))
    return "\n".join(lines)


def training_parity(results: dict, dataset: str, tolerance: float = 0.05) -> tuple[bool, str]:
    """Are all arms at comparable training steps?

    Comparing an arm at 11.5k steps against one at 20k is not a controlled comparison,
    and mid-flight aggregation silently does exactly that. The ladder verdict is
    suppressed unless every arm is within ``tolerance`` of the maximum.
    """
    steps = {}
    for arm in ARM_ORDER:
        runs = results.get((dataset, arm), [])
        if runs:
            steps[arm] = int(np.median([r["_last_step"] for r in runs]))
    if not steps:
        return False, "no runs"
    hi = max(steps.values())
    lagging = {a: s for a, s in steps.items() if s < hi * (1 - tolerance)}
    if lagging:
        detail = ", ".join(f"{a}@{s}" for a, s in sorted(lagging.items()))
        return False, f"arms still training (max={hi}): {detail}"
    return True, f"all arms at ~{hi} steps"


def ladder_verdict(results: dict, dataset: str) -> str:
    """State whether each link of the causal chain is supported, on this dataset."""
    def mean_success(arm: str) -> float:
        runs = results.get((dataset, arm), [])
        m, _, n = agg([r.get("eval/success_rate") for r in runs])
        return m if n else float("nan")

    ok, parity = training_parity(results, dataset)
    vals = {a: mean_success(a) for a in ARM_ORDER}
    lines = [f"\n### Ladder verdict — {dataset}\n"]
    lines.append(f"  training parity: {parity}\n")
    if not ok:
        lines.append(
            "  *** VERDICT SUPPRESSED: arms are at different training steps, so any\n"
            "  *** ordering below would confound allocation with training budget.\n"
        )
    steps = [
        ("multimodal futures matter", "flatkwm", "singlewm"),
        ("recursive prediction matters", "fixedtreewm", "flatkwm"),
        ("learned allocation matters", "treewm", "fixedtreewm"),
        ("*learned* beats merely adaptive", "treewm", "heuristictreewm"),
    ]
    for label, a, b in steps:
        va, vb = vals.get(a, float("nan")), vals.get(b, float("nan"))
        if not (np.isfinite(va) and np.isfinite(vb)):
            verdict = "no data"
        elif abs(va - vb) < 1e-9:
            verdict = "TIED"
        else:
            verdict = "supported" if va > vb else "NOT supported"
        if not ok:
            verdict = f"(unreliable) {verdict}"
        lines.append(f"  {label:34s} {a:>18s} {va:6.3f} vs {b:<18s} {vb:6.3f}   -> {verdict}")

    finite = [v for v in vals.values() if np.isfinite(v)]
    if finite and max(finite) < 0.05:
        lines.append(
            "\n  Note: every arm is at or near zero -- these comparisons are vacuous. That is a\n"
            "  non-discriminative outcome measure, not evidence against the hypothesis."
        )
    if finite and min(finite) > 0.9:
        lines.append(
            "\n  Note: every arm is near ceiling -- the split is too easy to separate the arms."
        )
    return "\n".join(lines)


def novelty_verdict(results: dict, dataset: str) -> str:
    """Apply the pre-registered interpretation criteria for the novelty-target rerun."""
    def val(arm: str, tag: str = "eval/success_rate") -> float:
        runs = results.get((dataset, arm), [])
        m, _, n = agg([r.get(tag) for r in runs])
        return m if n else float("nan")

    present = [a for a in ("noveltyq", "learnedq", "noveltyz", "learnedz") if (dataset, a) in results]
    if not present:
        return ""

    ok, parity = training_parity(results, dataset)
    lines = [f"\n### Novelty-target verdict — {dataset}\n", f"  training parity: {parity}\n"]
    if not ok:
        lines.append("  *** SUPPRESSED: unequal training steps.\n")

    dq, lq = val("noveltyq"), val("learnedq")
    dz, lz = val("noveltyz"), val("learnedz")
    rho_q, rho_z = val("learnedq", "expansion/gain_rank_correlation"), val("learnedz", "expansion/gain_rank_correlation")

    def gap(direct: float, learned: float, name: str, rho: float) -> str:
        if not (np.isfinite(direct) and np.isfinite(learned)):
            return f"  {name:14s} no data"
        rel = (learned - direct) / max(abs(direct), 1e-6)
        if abs(rel) <= 0.15:
            verdict = "CLOSED -- bad gain target was the problem; learned allocation viable"
        elif learned > direct:
            verdict = "learned EXCEEDS direct"
        elif np.isfinite(rho) and rho > 0.7:
            verdict = ("STILL FAR BELOW despite high target correlation -> inspect best-first "
                       "batching / tree-context feedback, not the representation")
        else:
            verdict = "STILL BELOW, and target correlation is weak -> the head is not learning the signal"
        return f"  {name:14s} direct={direct:.3f} learned={learned:.3f} (rel {rel:+.0%}, spearman {rho:.2f}) -> {verdict}"

    lines.append("  Q1. Does learned novelty close the gap to its direct heuristic?")
    lines.append(gap(dq, lq, "q-novelty", rho_q))
    lines.append(gap(dz, lz, "z-novelty", rho_z))

    lines.append("\n  Q2. Is q-novelty actually better than z-novelty?")
    if np.isfinite(dq) and np.isfinite(dz):
        rel = (dq - dz) / max(abs(dz), 1e-6)
        verdict = ("q ~= z -- do NOT attribute the gain to controllability-aware q"
                   if abs(rel) <= 0.15 else ("q BETTER than z" if dq > dz else "z BETTER than q"))
        lines.append(f"  direct: q={dq:.3f} vs z={dz:.3f} (rel {rel:+.0%}) -> {verdict}")

    lines.append("\n  Q3. Controls")
    for a in ("randomtreewm", "fixedtreewm"):
        if (dataset, a) in results:
            lines.append(f"  {a:20s} {val(a):.3f}")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--dataset", default=None)
    p.add_argument("--out", default="results")
    args = p.parse_args()

    runs_root = REPO / args.runs
    results = collect(runs_root)
    if not results:
        print(f"no runs found under {runs_root}")
        return

    datasets = sorted({d for d, _ in results}) if args.dataset is None else [args.dataset]
    report: list[str] = []
    for dataset in datasets:
        report.append(table(results, dataset, PRIMARY, "Primary (success vs matched node budget)"))
        report.append(table(results, dataset, DIAGNOSTICS, "Diagnostics"))
        report.append(table(results, dataset, NOVELTY_DIAG, "Novelty-target diagnostics"))
        report.append(ladder_verdict(results, dataset))
        report.append(novelty_verdict(results, dataset))

    text = "\n".join(report)
    print(text)

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(text)

    rows = []
    for (dataset, arm), runs in sorted(results.items()):
        for i, r in enumerate(runs):
            rows.append({"dataset": dataset, "arm": arm, "run": i, "last_step": r["_last_step"],
                         **{k: v for k, v in r.items() if not k.startswith("_")}})
    (out_dir / "all_metrics.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out_dir/'summary.md'} and {out_dir/'all_metrics.json'}")


if __name__ == "__main__":
    main()
