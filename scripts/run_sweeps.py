"""Run the budget sweep for every finished checkpoint, then plot arms together.

    python scripts/run_sweeps.py --budgets 16 32 64 128 256

This produces the primary plot -- success rate vs generated world-model nodes, all arms
on one axis at matched budgets. Training is untouched: a single checkpoint per arm is
evaluated at every budget, so the curve isolates inference-time compute allocation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
ARM_ORDER = ["singlewm", "flatkwm", "fixedtreewm", "randomtreewm", "uncertaintytreewm",
             "heuristictreewm", "treewm"]


def find_checkpoints(runs_root: Path, dataset: str | None) -> list[Path]:
    out = []
    for ck in sorted(runs_root.glob("*/*/*/checkpoints/latest.pt")):
        if dataset and ck.parts[-5] != dataset:
            continue
        out.append(ck)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default="runs")
    p.add_argument("--dataset", default=None)
    p.add_argument("--budgets", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 0, 1, 1])
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--out", default="results")
    p.add_argument("--plot-only", action="store_true",
                   help="skip evaluation, just re-collect existing budget_sweep.json files")
    args = p.parse_args()

    runs_root = REPO / args.runs
    checkpoints = find_checkpoints(runs_root, args.dataset)
    print(f"[sweep] {len(checkpoints)} checkpoints, budgets {args.budgets}")

    import os
    import time

    pending = [] if args.plot_only else list(checkpoints)
    running: list[tuple] = []
    free = list(args.gpus)
    budgets = "[" + ",".join(str(b) for b in args.budgets) + "]"

    while pending or running:
        while pending and free:
            gpu, ck = free.pop(0), pending.pop(0)
            proc = subprocess.Popen(
                [args.python, "scripts/sweep_budget.py", f"checkpoint={ck}", f"budgets={budgets}"],
                cwd=REPO, env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
            )
            running.append((proc, ck, gpu))
        time.sleep(3)
        for entry in list(running):
            proc, ck, gpu = entry
            if proc.poll() is None:
                continue
            running.remove(entry)
            free.append(gpu)
            status = "ok" if proc.returncode == 0 else f"FAILED({proc.returncode})"
            print(f"[sweep] {status} {ck.parts[-4]}/{ck.parts[-3]}")

    # ---------------------------------------------------------------- collect
    curves: dict[tuple[str, str], dict[int, list[float]]] = {}
    for ck in checkpoints:
        payload_path = ck.parent.parent / "budget_sweep.json"
        if not payload_path.exists():
            continue
        data = json.loads(payload_path.read_text())
        # parts: runs/<dataset>/<arm>/<stamp>/checkpoints/latest.pt
        key = (ck.parts[-5], data["arm"])
        curves.setdefault(key, {})
        for budget, metrics in data["budgets"].items():
            curves[key].setdefault(int(budget), []).append(metrics["eval/success_rate"])

    out_dir = REPO / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "budget_curves.json").write_text(
        json.dumps({f"{d}|{a}": v for (d, a), v in curves.items()}, indent=2, default=str)
    )

    for dataset in sorted({d for d, _ in curves}):
        fig, ax = plt.subplots(figsize=(7, 4.6))
        for arm in ARM_ORDER:
            if (dataset, arm) not in curves:
                continue
            pts = curves[(dataset, arm)]
            xs = sorted(pts)
            ys = [sum(pts[x]) / len(pts[x]) for x in xs]
            style = dict(marker="o", lw=2.2) if arm == "treewm" else dict(marker=".", lw=1.2, alpha=0.85)
            ax.plot(xs, ys, label=f"{arm} (n={len(pts[xs[0]])})", **style)
        ax.set_xscale("log", base=2)
        ax.set_xlabel("world-model nodes per replan (matched budget)")
        ax.set_ylabel("success rate")
        ax.set_title(f"Success vs node budget — {dataset}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"success_vs_budget_{dataset}.png"
        fig.savefig(path, dpi=140)
        print(f"[sweep] wrote {path}")


if __name__ == "__main__":
    main()
