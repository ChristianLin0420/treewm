"""Two-A100 job fleet: probe, bin-pack, run, back off, report.

Treats both GPUs as a pool of independent single-GPU jobs (no DDP for screening -- these
models are 2-5M params, so DDP would add communication for no throughput). Concurrency is
*measured*, not assumed: 8 jobs/GPU is the starting hypothesis, and the probe decides
whether it survives contact with humanoidmaze's 69-dim observations.

Two independent backoff triggers, because either alone is insufficient:

* **VRAM** -- bin-pack against probed peak reserved memory, keeping >=12% headroom. On
  OOM the job is marked OOM_RETRY (an infrastructure event, never a scientific result),
  killed cleanly, and requeued at lower concurrency, resuming from checkpoint if one
  exists. At most two OOM retries, so a genuinely too-large config fails loudly rather
  than looping.
* **Throughput** -- a GPU that fits 8 jobs but runs each at a third of its solo rate has
  bought nothing. If the median per-job step time exceeds 1.8x the calibrated single-job
  rate, slots are reduced even though VRAM is fine.

Backoff ladder: 8 -> 6 -> 4 -> 3 -> 2 -> 1.

    python scripts/fleet.py --probe-only
    python scripts/fleet.py --wave 1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LADDER = [8, 6, 4, 3, 2, 1]
VRAM_HEADROOM = 0.12
THROUGHPUT_DEGRADE = 1.8
MAX_OOM_RETRIES = 2
OOM_PAT = re.compile(r"CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|torch\.OutOfMemoryError", re.I)


@dataclass
class Job:
    job_id: str
    env: str          # hydra env config name
    env_name: str     # ogbench dataset name
    arm: str
    seed: int
    steps: int
    overrides: str = ""
    run_root: str = "experiments/09-cross-family/runs/wave1"
    gpu: int | None = None
    status: str = "QUEUED"     # QUEUED RUNNING COMPLETE OOM_RETRY FAILED
    step: int = 0
    peak_vram_gb: float = 0.0
    host_ram_gb: float = 0.0
    steps_per_s: float = 0.0
    oom_retries: int = 0
    started: float = 0.0
    elapsed_s: float = 0.0
    log: str = ""

    @property
    def name(self) -> str:
        return f"{self.arm}_s{self.seed}"


def nvml_free_gb(gpu: int) -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits",
             "-i", str(gpu)], text=True)
        return float(out.strip().split("\n")[0]) / 1024.0
    except Exception:
        return 0.0


def gpu_util(gpu: int) -> float:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits",
             "-i", str(gpu)], text=True)
        return float(out.strip().split("\n")[0])
    except Exception:
        return float("nan")


def train_cmd(job: Job, gpu: int, python: str, resume: bool) -> list[str]:
    run_dir = REPO / job.run_root
    cmd = [python, "-u", str(REPO / "scripts" / "train.py"),
           f"env={job.env}", f"arm={job.arm}", f"seed={job.seed}",
           f"train.steps={job.steps}", f"run_root={run_dir}", f"run_name={job.name}",
           "future_sets.cache=true", "future_sets.shared_cache=true",
           "eval.task_split=auto"]
    if job.overrides:
        cmd += job.overrides.split()
    if resume:
        cmd.append("resume=auto")
    return cmd


# ---------------------------------------------------------------- resource probe
def probe(jobs: list[Job], python: str, steps: int, gpu: int = 0) -> dict[str, dict]:
    """Run each distinct (env, arm) briefly, alone, to calibrate memory and speed."""
    seen: dict[str, Job] = {}
    for j in jobs:
        seen.setdefault(f"{j.env}|{j.arm}", j)
    out: dict[str, dict] = {}
    probe_dir = REPO / "experiments/09-cross-family/probe"
    probe_dir.mkdir(parents=True, exist_ok=True)

    for k, j in seen.items():
        log = probe_dir / f"probe_{j.env}_{j.arm}.log"
        pj = Job(**{**asdict(j), "steps": steps, "run_root": "experiments/09-cross-family/probe/runs"})
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TREEWM_PROBE="1")
        t0 = time.time()
        with open(log, "w") as fh:
            p = subprocess.run(train_cmd(pj, gpu, python, resume=False),
                               stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=REPO)
        txt = log.read_text(errors="ignore")
        vram = max([float(m) for m in re.findall(r"peak_reserved_gb=([0-9.]+)", txt)] or [0.0])
        rss = max([float(m) for m in re.findall(r"host_rss_gb=([0-9.]+)", txt)] or [0.0])
        rate = ([float(m) for m in re.findall(r"([0-9.]+)it/s", txt)] or [0.0])[-1]
        dl = max([float(m) for m in re.findall(r"data_wait_frac=([0-9.]+)", txt)] or [0.0])
        out[k] = {"peak_vram_gb": vram, "host_rss_gb": rss, "steps_per_s": rate,
                  "data_wait_frac": dl, "ok": p.returncode == 0,
                  "gpu_util": gpu_util(gpu), "probe_s": round(time.time() - t0, 1),
                  "log": str(log)}
        print(f"[probe] {k:52s} vram={vram:5.2f}GB rss={rss:5.2f}GB "
              f"{rate:6.2f} it/s dl_wait={dl:.2f} {'ok' if p.returncode == 0 else 'FAILED'}",
              flush=True)
    (probe_dir / "probe.json").write_text(json.dumps(out, indent=2))
    return out


def plan_slots(profiles: dict[str, dict], gpus: list[int], want: int) -> dict[int, int]:
    """Slots per GPU from probed peak VRAM, honouring the headroom requirement."""
    worst = max([p["peak_vram_gb"] for p in profiles.values()] or [1.0])
    worst = max(worst, 0.2)
    slots = {}
    for g in gpus:
        total = nvml_free_gb(g)
        fit = int((total * (1.0 - VRAM_HEADROOM)) // worst)
        chosen = min(want, max(1, fit))
        chosen = min(LADDER, key=lambda x: (abs(x - chosen), -x)) if chosen < want else want
        slots[g] = chosen
        print(f"[plan] gpu{g}: {total:.1f}GB free, worst-case job {worst:.2f}GB, "
              f"headroom {VRAM_HEADROOM:.0%} -> {chosen} slots", flush=True)
    return slots


def lower(n: int) -> int:
    for v in LADDER:
        if v < n:
            return v
    return 1


class Fleet:
    def __init__(self, jobs: list[Job], slots: dict[int, int], python: str,
                 baseline: dict[str, float], out: Path, poll: float = 20.0) -> None:
        self.queue = list(jobs)
        self.slots = dict(slots)
        self.python = python
        self.baseline = baseline
        self.out = out
        self.poll = poll
        self.running: dict[str, tuple[Job, subprocess.Popen, int]] = {}
        self.done: list[Job] = []
        self.log_dir = out.parent / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- dashboard
    def dashboard(self) -> str:
        rows = [f"{'job_id':26s} {'gpu':>3s} {'env':26s} {'arm':10s} {'seed':>4s} "
                f"{'step':>7s} {'vram':>6s} {'ram':>6s} {'it/s':>6s} {'status':10s}"]
        rows.append("-" * 116)
        allj = [j for j, _, _ in self.running.values()] + self.done + self.queue
        for j in allj:
            rows.append(f"{j.job_id:26s} {str(j.gpu if j.gpu is not None else '-'):>3s} "
                        f"{j.env_name[:26]:26s} {j.arm:10s} {j.seed:4d} {j.step:7d} "
                        f"{j.peak_vram_gb:6.2f} {j.host_ram_gb:6.2f} {j.steps_per_s:6.2f} {j.status:10s}")
        occ = " ".join(f"gpu{g}={sum(1 for _, _, gg in self.running.values() if gg == g)}/{n}"
                       for g, n in self.slots.items())
        rows.append(f"\nslots: {occ}   queued={len(self.queue)} done={len(self.done)}")
        return "\n".join(rows)

    def write_status(self) -> None:
        payload = {
            "timestamp": time.time(),
            "slots": self.slots,
            "jobs": [asdict(j) for j in
                     [x for x, _, _ in self.running.values()] + self.done + self.queue],
        }
        tmp = self.out.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.out)
        print("\n" + self.dashboard(), flush=True)

    # -------------------------------------------------------------- scheduling
    def free_gpu(self) -> int | None:
        for g, n in self.slots.items():
            if sum(1 for _, _, gg in self.running.values() if gg == g) < n:
                return g
        return None

    def launch(self, job: Job, gpu: int) -> None:
        job.gpu, job.status, job.started = gpu, "RUNNING", time.time()
        short = job.env_name.replace("-v0", "")
        ckpt = REPO / job.run_root / short / job.arm / job.name / "checkpoints" / "latest.pt"
        log = self.log_dir / f"{job.job_id}.log"
        job.log = str(log)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        fh = open(log, "a")
        p = subprocess.Popen(train_cmd(job, gpu, self.python, resume=ckpt.exists()),
                             stdout=fh, stderr=subprocess.STDOUT, env=env, cwd=REPO,
                             start_new_session=True)
        self.running[job.job_id] = (job, p, gpu)
        print(f"[fleet] start gpu{gpu} {job.job_id}"
              f"{' (resume)' if ckpt.exists() else ''}", flush=True)

    def scrape(self, job: Job) -> None:
        try:
            txt = Path(job.log).read_text(errors="ignore")[-8000:]
        except OSError:
            return
        for pat, attr, cast in [(r"peak_reserved_gb=([0-9.]+)", "peak_vram_gb", float),
                                (r"host_rss_gb=([0-9.]+)", "host_ram_gb", float),
                                (r"([0-9.]+)it/s", "steps_per_s", float),
                                (r"step[= ](\d+)", "step", int)]:
            hits = re.findall(pat, txt)
            if hits:
                setattr(job, attr, cast(hits[-1]))

    def handle_exit(self, job: Job, proc: subprocess.Popen, gpu: int) -> None:
        txt = ""
        try:
            txt = Path(job.log).read_text(errors="ignore")[-20000:]
        except OSError:
            pass
        job.elapsed_s = round(time.time() - job.started, 1)
        if proc.returncode == 0:
            job.status = "COMPLETE"
            self.done.append(job)
            print(f"[fleet] ok {job.job_id} in {job.elapsed_s/60:.0f} min", flush=True)
            return
        if OOM_PAT.search(txt) and job.oom_retries < MAX_OOM_RETRIES:
            # Infrastructure event, not a scientific failure: shrink the GPU and requeue.
            job.oom_retries += 1
            job.status = "OOM_RETRY"
            old = self.slots[gpu]
            self.slots[gpu] = lower(old)
            print(f"[fleet] OOM {job.job_id} (retry {job.oom_retries}/{MAX_OOM_RETRIES}); "
                  f"gpu{gpu} slots {old} -> {self.slots[gpu]}", flush=True)
            job.gpu = None
            self.queue.insert(0, job)
            return
        job.status = "FAILED"
        self.done.append(job)
        tail = "\n".join(txt.strip().split("\n")[-4:])
        print(f"[fleet] FAILED {job.job_id} rc={proc.returncode}\n  {tail}", flush=True)

    def check_throughput(self) -> None:
        """Reduce slots if oversubscription has made each job pathologically slow."""
        for g in list(self.slots):
            rates = [j.steps_per_s for j, _, gg in self.running.values()
                     if gg == g and j.steps_per_s > 0]
            if len(rates) < max(2, self.slots[g] // 2):
                continue
            rates.sort()
            med = rates[len(rates) // 2]
            key_base = max(self.baseline.values()) if self.baseline else 0.0
            if key_base <= 0 or med <= 0:
                continue
            if key_base / med > THROUGHPUT_DEGRADE and self.slots[g] > 1:
                old = self.slots[g]
                self.slots[g] = lower(old)
                print(f"[fleet] throughput {med:.2f} it/s vs baseline {key_base:.2f} "
                      f"(>{THROUGHPUT_DEGRADE}x slower); gpu{g} slots {old} -> {self.slots[g]}",
                      flush=True)

    def run(self) -> None:
        last = 0.0
        while self.queue or self.running:
            while self.queue:
                g = self.free_gpu()
                if g is None:
                    break
                self.launch(self.queue.pop(0), g)
            time.sleep(self.poll)
            for jid in list(self.running):
                job, proc, gpu = self.running[jid]
                self.scrape(job)
                if proc.poll() is not None:
                    del self.running[jid]
                    self.handle_exit(job, proc, gpu)
            self.check_throughput()
            if time.time() - last > 60:
                self.write_status()
                last = time.time()
        self.write_status()
        ok = sum(j.status == "COMPLETE" for j in self.done)
        print(f"\n[fleet] finished: {ok}/{len(self.done)} complete, "
              f"{sum(j.status == 'FAILED' for j in self.done)} failed", flush=True)


# ---------------------------------------------------------------- wave definition
WAVE1_ENVS = [
    ("antmaze_large_navigate", "antmaze-large-navigate-v0"),
    ("humanoidmaze_medium_navigate", "humanoidmaze-medium-navigate-v0"),
    ("antsoccer_medium_navigate", "antsoccer-medium-navigate-v0"),
    ("cube_single_play", "cube-single-play-v0"),
    ("cube_double_play", "cube-double-play-v0"),
    ("scene_play", "scene-play-v0"),
    ("puzzle_3x3_play", "puzzle-3x3-play-v0"),
]

# The only intended conceptual difference is recursion. Both arms: K=4, fixed h=16, same
# encoder/hidden/optimizer/steps/seed/budget, residual latent dynamics, decoded goal
# scoring on the domain's goal dims.
H16 = ("future_sets.horizons=[16] future_sets.h_max=16 future_sets.horizon_rule=fixed "
       "future_sets.fixed_horizon=16 model.horizon_mode=fixed model.fixed_horizon_index=0 "
       "model.branch_factor=4")

# Screen-scale resource settings, applied identically to both arms so they cannot become
# a confound. These are NOT the global defaults -- changing those would silently alter the
# reproducibility of the earlier PointMaze cycles.
#
# retrieval_pool: the default 0 means "whole dataset", which is free in PointMaze's 2-D
# observations and catastrophic here -- a measured 90 ms/anchor on cube-double's 1M x 37,
# i.e. ~23 s per batch. kd-tree query cost grows superlinearly with pool size and
# dimension (50k: 1.7 ms, 400k: 36 ms at D=37), so the pool is capped. With the future-set
# cache on, each anchor is built once and reused ~100x, so this cost is paid once.
#
# num_workers: 32 cores shared by up to 16 jobs is 2 cores each. Requesting 12 would
# oversubscribe 6x and make every job slower than running alone.
RESOURCE = ("future_sets.retrieval_pool=50000 train.max_train_anchors=100000 "
            "train.max_val_anchors=10000 train.num_workers=2")

ARMS = {"flatkwm": f"arm=flatkwm {H16} {RESOURCE}",
        "randomtreewm": f"arm=randomtreewm {H16} {RESOURCE}"}


def wave1_jobs(seeds: list[int], steps: int) -> list[Job]:
    jobs = []
    for cfg, env_name in WAVE1_ENVS:
        for arm, ov in ARMS.items():
            for s in seeds:
                jobs.append(Job(job_id=f"{cfg}|{arm}|s{s}", env=cfg, env_name=env_name,
                                arm=arm, seed=s, steps=steps,
                                overrides=" ".join(ov.split()[1:])))
    return jobs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--wave", type=int, default=1)
    p.add_argument("--seeds", nargs="+", type=int, default=[0])
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--per-gpu", type=int, default=8)
    p.add_argument("--gpus", nargs="+", type=int, default=[0, 1])
    p.add_argument("--probe-steps", type=int, default=300)
    p.add_argument("--probe-only", action="store_true")
    p.add_argument("--skip-probe", action="store_true")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--status", default="experiments/09-cross-family/fleet_status.json")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    jobs = wave1_jobs(args.seeds, args.steps)
    print(f"[fleet] wave {args.wave}: {len(jobs)} jobs "
          f"({len(WAVE1_ENVS)} envs x {len(ARMS)} arms x {len(args.seeds)} seed)")
    if args.dry_run:
        for j in jobs:
            print(f"  {j.job_id:34s} env={j.env_name:32s} {j.overrides}")
        return

    profiles: dict[str, dict] = {}
    if not args.skip_probe:
        print(f"[fleet] probing {args.probe_steps} steps per (env, arm) ...")
        profiles = probe(jobs, args.python, args.probe_steps, gpu=args.gpus[0])
        bad = [k for k, v in profiles.items() if not v["ok"]]
        if bad:
            print(f"[fleet] probe FAILED for {bad} -- fix before launching the wave")
            return
    if args.probe_only:
        return

    slots = plan_slots(profiles, args.gpus, args.per_gpu) if profiles else \
        {g: args.per_gpu for g in args.gpus}
    baseline = {k: v["steps_per_s"] for k, v in profiles.items()} if profiles else {}
    out = REPO / args.status
    out.parent.mkdir(parents=True, exist_ok=True)
    Fleet(jobs, slots, args.python, baseline, out).run()


if __name__ == "__main__":
    main()
