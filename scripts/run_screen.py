"""Phase-1 screening launcher: named recipes, concurrent single-GPU jobs.

Tests *mechanisms* one at a time rather than a Cartesian product. Each recipe is a small
set of Hydra overrides on top of the current defaults, so a recipe named in the morning
report maps to an exactly reproducible command.

Concurrency: starts at ``--per-gpu`` jobs per GPU and measures per-job throughput against
a single-job baseline. If the median drops below ``baseline / --degrade-factor`` the
launcher stops adding jobs beyond a reduced cap -- oversubscription that halves throughput
buys nothing.

    python scripts/run_screen.py --list
    python scripts/run_screen.py --recipes A1_multistep B1_k8 --steps 20000
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SHORT_H = "future_sets.horizons=[2,4,8,16] future_sets.h_max=16"
MED_H = "future_sets.horizons=[4,8,16,32] future_sets.h_max=32"
MULTISTEP = "losses.enabled.multistep=true losses.weights.multistep=1.0"

# recipe -> (track, hydra overrides)
RECIPES: dict[str, tuple[str, str]] = {
    # ---- Track A: recursive robustness -------------------------------------------
    "A0_random":      ("A", "arm=randomtreewm"),
    "A0_flat":        ("A", "arm=flatkwm"),  # mandatory flat control for generalisation
    "A0_bfs":         ("A", "arm=fixedtreewm"),
    "A1_multistep":   ("A", f"arm=randomtreewm {MULTISTEP}"),
    "A2_ss25":        ("A", f"arm=randomtreewm {MULTISTEP} losses.scheduled_sampling_p=0.25"),
    "A2_ss50":        ("A", f"arm=randomtreewm {MULTISTEP} losses.scheduled_sampling_p=0.50"),
    # residual dynamics is already the default (z' = z + delta); this is its control
    "A4_noresidual":  ("A", "arm=randomtreewm model.residual_dynamics=false"),
    # ---- Track B: branch proposal capacity ---------------------------------------
    "B1_k8":          ("B", "arm=randomtreewm model.branch_factor=8"),
    "B2_k16":         ("B", "arm=randomtreewm model.branch_factor=16"),
    "B3_k8_short":    ("B", f"arm=randomtreewm model.branch_factor=8 {SHORT_H}"),
    # ---- Track C: temporal abstraction -------------------------------------------
    "C1_short":       ("C", f"arm=randomtreewm {SHORT_H}"),
    "C2_medium":      ("C", f"arm=randomtreewm {MED_H}"),
    "C4_fixed8":      ("C", "arm=randomtreewm future_sets.horizons=[8] future_sets.h_max=8 "
                            "future_sets.horizon_rule=fixed future_sets.fixed_horizon=8"),
    # ---- Track G: capacity --------------------------------------------------------
    "G1_wide":        ("G", "arm=randomtreewm model.hidden_dim=512"),
    "G2_deep":        ("G", "arm=randomtreewm model.num_layers=6"),
    "G3_z256":        ("G", "arm=randomtreewm model.z_dim=256"),
    # ---- Track H: loss design -----------------------------------------------------
    "H1_bind":        ("H", "arm=randomtreewm losses.weights.bind=3.0"),
    "H2_depthw":      ("H", f"arm=randomtreewm {MULTISTEP} losses.multistep_depth_weights=[1.0,1.5,2.0]"),
    "H3_noredundancy": ("H", "arm=randomtreewm losses.weights.redundancy=0.0"),
}

# ---- Q1: AntMaze generalisation (50k) ---------------------------------------------
ANT: dict[str, tuple[str, str]] = {
    "Q1_flat":        ("Q1", "arm=flatkwm"),
    "Q1_random":      ("Q1", "arm=randomtreewm"),
    "Q1_short":       ("Q1", f"arm=randomtreewm {SHORT_H}"),
    "Q1_short_combo": ("Q1", f"arm=randomtreewm {SHORT_H} tree.scorer=root_quota "
                             "planner.score_mode=ancestor"),
}

# ---- Q2: is the win learned horizon selection, or just short horizons available? ----
# Every variant shares the SAME available set [2,4,8,16]; only selection differs.
Q2: dict[str, tuple[str, str]] = {
    "Q2_learned":  ("Q2", f"arm=randomtreewm {SHORT_H} model.horizon_mode=learned"),
    "Q2_random_h": ("Q2", f"arm=randomtreewm {SHORT_H} model.horizon_mode=random"),
    "Q2_fixed2":   ("Q2", f"arm=randomtreewm {SHORT_H} model.horizon_mode=fixed model.fixed_horizon_index=0"),
    "Q2_fixed4":   ("Q2", f"arm=randomtreewm {SHORT_H} model.horizon_mode=fixed model.fixed_horizon_index=1"),
    "Q2_fixed8":   ("Q2", f"arm=randomtreewm {SHORT_H} model.horizon_mode=fixed model.fixed_horizon_index=2"),
    "Q2_fixed16":  ("Q2", f"arm=randomtreewm {SHORT_H} model.horizon_mode=fixed model.fixed_horizon_index=3"),
    "Q2_depthsched": ("Q2", f"arm=randomtreewm {SHORT_H} model.horizon_mode=depth_schedule"),
}

# Phase-2 combinations, launched only after screening motivates them.
COMBOS: dict[str, tuple[str, str]] = {
    "P_k8_short_ms":  ("combo", f"arm=randomtreewm model.branch_factor=8 {SHORT_H} {MULTISTEP}"),
    "P_k8_ss":        ("combo", f"arm=randomtreewm model.branch_factor=8 {MULTISTEP} "
                                "losses.scheduled_sampling_p=0.25"),
    "P_ms_ss":        ("combo", f"arm=randomtreewm {MULTISTEP} losses.scheduled_sampling_p=0.25"),
    "P_short_ms":     ("combo", f"arm=randomtreewm {SHORT_H} {MULTISTEP}"),
}

ALL = {**RECIPES, **COMBOS, **ANT, **Q2}


def throughput(log: Path) -> float | None:
    """Latest it/s parsed from a tqdm line, or None."""
    try:
        tail = log.read_text(errors="ignore")[-4000:]
    except OSError:
        return None
    hits = re.findall(r"([0-9.]+)it/s", tail)
    return float(hits[-1]) if hits else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--recipes", nargs="+", default=None)
    p.add_argument("--tracks", nargs="+", default=None)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--dataset", default="pointmaze_medium_stitch")
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    p.add_argument("--per-gpu", type=int, default=3)
    p.add_argument("--min-per-gpu", type=int, default=2)
    p.add_argument("--degrade-factor", type=float, default=1.8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--run-root", default="runs_screen")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--extra", nargs="*", default=[])
    p.add_argument("--list", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if args.list:
        for name, (track, ov) in ALL.items():
            print(f"  [{track}] {name:18s} {ov}")
        return

    names = args.recipes or [n for n, (t, _) in ALL.items() if not args.tracks or t in args.tracks]
    jobs = [(n, s) for s in args.seeds for n in names]
    log_dir = REPO / args.run_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    slots = [g for g in args.gpus for _ in range(args.per_gpu)]
    print(f"[screen] {len(jobs)} jobs, {len(slots)} slots ({args.per_gpu}/GPU), {args.steps} steps")
    if args.dry_run:
        for n, s in jobs:
            print(f"  {n:18s} seed={s}  {ALL[n][1]}")
        return

    baseline: float | None = None
    cap = len(slots)
    free = list(slots)
    running: list[tuple] = []
    pending = list(jobs)
    done = 0
    started = time.time()

    while pending or running:
        while pending and free and len(running) < cap:
            gpu = free.pop(0)
            name, seed = pending.pop(0)
            overrides = ALL[name][1].split() + [
                f"env={args.dataset}", f"seed={seed}", f"train.steps={args.steps}",
                f"train.num_workers={args.num_workers}", f"run_root={args.run_root}",
                f"run_name={name}_s{seed}", "eval.task_split=hard", *args.extra,
            ]
            handle = open(log_dir / f"{name}_s{seed}.log", "w")
            proc = subprocess.Popen(
                [args.python, "scripts/train.py", *overrides], cwd=REPO,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
                stdout=handle, stderr=subprocess.STDOUT,
            )
            running.append((proc, name, seed, gpu, handle, log_dir / f"{name}_s{seed}.log"))
            print(f"[screen] start gpu{gpu} {name} seed={seed}  ({len(running)} running)", flush=True)
            time.sleep(2)

        time.sleep(20)

        # Throughput guard: first job alone sets the baseline; if the median rate under
        # load falls below baseline/degrade_factor, stop growing the pool.
        rates = [r for r in (throughput(l) for *_, l in running) if r]
        if rates:
            if baseline is None and len(running) == 1:
                baseline = rates[0]
            elif baseline is not None and len(running) > 1:
                med = sorted(rates)[len(rates) // 2]
                if med < baseline / args.degrade_factor and cap > len(args.gpus) * args.min_per_gpu:
                    cap = max(len(args.gpus) * args.min_per_gpu, len(running) - 1)
                    print(f"[screen] throughput {med:.1f} it/s vs baseline {baseline:.1f}; "
                          f"reducing concurrency cap to {cap}", flush=True)

        for entry in list(running):
            proc, name, seed, gpu, handle, _ = entry
            if proc.poll() is None:
                continue
            running.remove(entry); handle.close(); free.append(gpu); done += 1
            status = "ok" if proc.returncode == 0 else f"FAILED({proc.returncode})"
            print(f"[screen] {status} {name} seed={seed} "
                  f"({done}/{len(jobs)}, {(time.time()-started)/60:.0f} min)", flush=True)

    print(f"[screen] all done in {(time.time()-started)/60:.0f} min", flush=True)


if __name__ == "__main__":
    main()
