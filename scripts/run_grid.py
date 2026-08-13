"""Launch the arm x seed x dataset grid across the available GPUs.

PointMaze models are ~2-5M parameters, so DDP across two A100s buys almost nothing:
the bottleneck is future-set retrieval in the dataloader, not matmul throughput.
Running two *independent* single-GPU jobs concurrently is strictly better utilisation,
which is what this launcher does. ``torchrun`` remains supported for AntMaze and for a
future pixel encoder.

    python scripts/run_grid.py --arms all --seeds 0 1 2 --datasets navigate stitch
    python scripts/run_grid.py --arms treewm singlewm --seeds 0 --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ALL_ARMS = [
    "singlewm",
    "flatkwm",
    "fixedtreewm",
    "randomtreewm",
    "uncertaintytreewm",
    "heuristictreewm",
    "treewm",
]

DATASETS = {
    "navigate": "pointmaze_medium_navigate",
    "stitch": "pointmaze_medium_stitch",
    "large": "pointmaze_large_navigate",
}


def build_jobs(args) -> list[dict]:
    arms = ALL_ARMS if args.arms == ["all"] else args.arms
    jobs = []
    for dataset, seed, arm in itertools.product(args.datasets, args.seeds, arms):
        jobs.append(
            {
                "arm": arm,
                "seed": seed,
                "dataset": dataset,
                "overrides": [
                    f"env={DATASETS[dataset]}",
                    f"arm={arm}",
                    f"seed={seed}",
                    f"train.steps={args.steps}",
                    f"eval.task_split={args.task_split}",
                    f"train.num_workers={args.num_workers}",
                    f"run_root={args.run_root}",
                    *args.extra,
                ],
            }
        )
    return jobs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=["all"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--datasets", nargs="+", default=["navigate"], choices=list(DATASETS))
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--task-split", default="hard")
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    p.add_argument("--num-workers", type=int, default=10)
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--run-root", default="runs")
    p.add_argument("--extra", nargs="*", default=[], help="extra hydra overrides")
    p.add_argument("--log-dir", default=None)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    jobs = build_jobs(args)
    log_dir = REPO / (args.log_dir or f"{args.run_root}/_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[grid] {len(jobs)} jobs over {len(args.gpus)} GPUs, {args.steps} steps each")

    if args.dry_run:
        for j in jobs:
            print(f"  {j['dataset']:9s} {j['arm']:18s} seed={j['seed']}  {' '.join(j['overrides'])}")
        return

    running: list[tuple[subprocess.Popen, dict, int, object]] = []
    free = list(args.gpus)
    pending = list(jobs)
    done = 0
    started = time.time()

    while pending or running:
        while pending and free:
            gpu = free.pop(0)
            job = pending.pop(0)
            tag = f"{job['dataset']}_{job['arm']}_s{job['seed']}"
            handle = open(log_dir / f"{tag}.log", "w")
            proc = subprocess.Popen(
                [args.python, "scripts/train.py", *job["overrides"]],
                cwd=REPO,
                env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            running.append((proc, job, gpu, handle))
            print(f"[grid] start gpu{gpu} {tag}")

        time.sleep(5)
        for entry in list(running):
            proc, job, gpu, handle = entry
            if proc.poll() is None:
                continue
            running.remove(entry)
            handle.close()
            free.append(gpu)
            done += 1
            status = "ok" if proc.returncode == 0 else f"FAILED({proc.returncode})"
            elapsed = (time.time() - started) / 60
            print(
                f"[grid] {status} {job['dataset']}_{job['arm']}_s{job['seed']} "
                f"({done}/{len(jobs)}, {elapsed:.0f} min elapsed)"
            )

    print(f"[grid] all done in {(time.time() - started) / 60:.0f} min")


if __name__ == "__main__":
    main()
