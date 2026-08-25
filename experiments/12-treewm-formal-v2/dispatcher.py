#!/usr/bin/env python3
"""Cancellation-safe 16-rank dispatcher for the formal TreeWM-v2 array.

Each Slurm array element owns one allocation shard.  Every rank owns at most one
single-GPU training run; ranks 8..15 in the final shard are rendezvous participants
only.  USR1/deadline stops are checkpointed and coordinated before the *composite*
array element is requeued.  SIGTERM writes a persistent cancel latch and always wins.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import netrc
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from campaign import (
    EXPECTED_ALLOCATION_SHARDS,
    EXPECTED_WORKERS,
    RunSpec,
    completion_is_valid,
    expand_runs,
    load_manifest,
    run_directory,
    trainer_command,
)


GRACEFUL_EXIT_CODE = 75
CANCEL_EXIT_CODE = 143
POLL_SECONDS = 0.5
ELEMENT_ID = re.compile(r"^[0-9]+(?:_[0-9]+)?$")
ACTION_PRIORITY = {"requeue": 1, "abort": 2, "cancel": 3}


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def durable_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    try:
        fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def slurm_requeue_target(environ: Mapping[str, str]) -> str:
    master, task = environ.get("SLURM_ARRAY_JOB_ID"), environ.get("SLURM_ARRAY_TASK_ID")
    if (master is None) != (task is None):
        raise ValueError("SLURM_ARRAY_JOB_ID and SLURM_ARRAY_TASK_ID must be set together")
    target = f"{master}_{task}" if master is not None else environ.get("SLURM_JOB_ID", "0")
    if ELEMENT_ID.fullmatch(target) is None:
        raise ValueError(f"invalid Slurm element ID: {target!r}")
    return target


def allocation_state_root(
    state_root: Path, job_id: str, allocation_shard: int, allocation_shards: int
) -> Path:
    if ELEMENT_ID.fullmatch(job_id) is None:
        raise ValueError(f"invalid Slurm element ID: {job_id!r}")
    if allocation_shards != EXPECTED_ALLOCATION_SHARDS:
        raise ValueError(f"formal dispatch requires {EXPECTED_ALLOCATION_SHARDS} array elements")
    if not 0 <= allocation_shard < allocation_shards:
        raise ValueError("allocation shard is out of range")
    return (
        state_root.resolve()
        / f"array-job-{job_id}"
        / f"allocation-shard-{allocation_shard:02d}-of-{allocation_shards:02d}"
    )


class RunLockUnavailable(RuntimeError):
    pass


@contextlib.contextmanager
def run_lock(run_dir: Path, *, run: RunSpec, job_id: str, allocation_shard: int, rank: int):
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / ".dispatcher.lock"
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunLockUnavailable(f"run {run.run_id} is already owned: {path}") from exc
        handle.seek(0)
        handle.truncate()
        json.dump(
            {
                "run_id": run.run_id,
                "manifest_index": run.index,
                "job_id": job_id,
                "allocation_shard": allocation_shard,
                "rank": rank,
                "pid": os.getpid(),
                "hostname": os.uname().nodename,
                "acquired_unix_time": time.time(),
            },
            handle,
            sort_keys=True,
        )
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


class SignalState:
    def __init__(self) -> None:
        self.usr1 = False
        self.term = False

    def handle_usr1(self, _signum: int, _frame: object) -> None:
        self.usr1 = True

    def handle_term(self, _signum: int, _frame: object) -> None:
        self.term = True


class RequeueCoordinator:
    """Durable restart-generation barrier with a permanent cancel override."""

    def __init__(
        self,
        state_root: Path,
        *,
        job_id: str,
        restart_count: int,
        rank: int,
        workers: int,
        wait_seconds: int,
        allocation_shard: int,
        allocation_shards: int,
        cancel_latch: Path | None = None,
    ) -> None:
        if workers != EXPECTED_WORKERS:
            raise ValueError(f"formal rendezvous requires {EXPECTED_WORKERS} ranks")
        if ELEMENT_ID.fullmatch(job_id) is None:
            raise ValueError(f"invalid Slurm element ID: {job_id!r}")
        self.state_root = state_root.resolve()
        self.state_root.mkdir(parents=True, exist_ok=True)
        self.job_id = job_id
        self.restart_count = restart_count
        self.rank = rank
        self.workers = workers
        self.wait_seconds = wait_seconds
        self.allocation_shard = allocation_shard
        self.allocation_shards = allocation_shards
        self.directory = self.state_root / "requeue" / str(restart_count)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.request_path = self.directory / "REQUESTED.json"
        self.request_lock = self.directory / ".request.lock"
        self.resolution_lock = self.directory / ".resolution.lock"
        self.cancel_latch = (cancel_latch or self.state_root / "CANCEL_REQUESTED").resolve()
        self.cancelled_path = self.state_root / "CANCELLED.json"
        self.fallback_calling_path = self.directory / "FALLBACK_REQUEUE_CALLING.json"
        self.batch_calling_path = self.directory / "BATCH_REQUEUE_CALLING.json"
        self.signal_state: SignalState | None = None
        self.allocation_stop_epoch = float("inf")

    def _base(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "restart_count": self.restart_count,
            "rank": self.rank,
            "allocation_shard": self.allocation_shard,
            "allocation_shards": self.allocation_shards,
            "unix_time": time.time(),
        }

    def _belongs(self, payload: Mapping[str, Any] | None) -> bool:
        return bool(
            payload
            and payload.get("job_id") == self.job_id
            and payload.get("restart_count") == self.restart_count
            and payload.get("allocation_shard") == self.allocation_shard
            and payload.get("allocation_shards") == self.allocation_shards
        )

    @property
    def cancelled(self) -> bool:
        return self.cancel_latch.exists() or self.cancelled_path.exists()

    @property
    def intentional_requeue_teardown(self) -> bool:
        for path, status in (
            (self.fallback_calling_path, "fallback_scontrol_requeue_calling"),
            (self.batch_calling_path, "batch_scontrol_requeue_calling"),
        ):
            payload = read_json(path)
            if self._belongs(payload) and payload.get("status") == status:
                return True
        return False

    def request(self, action: str, reason: str, *, run_id: str | None = None) -> None:
        if action not in ACTION_PRIORITY:
            raise ValueError(f"unknown coordination action: {action}")
        with self.request_lock.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            existing = read_json(self.request_path)
            if self._belongs(existing) and ACTION_PRIORITY.get(str(existing.get("action")), 0) > ACTION_PRIORITY[action]:
                return
            payload = {**self._base(), "action": action, "reason": reason}
            if run_id is not None:
                payload["run_id"] = run_id
            atomic_json(self.request_path, payload)

    def request_payload(self) -> dict[str, Any] | None:
        payload = read_json(self.request_path)
        return payload if self._belongs(payload) else None

    def cancel(self, reason: str) -> None:
        self.cancel_latch.parent.mkdir(parents=True, exist_ok=True)
        self.cancel_latch.touch(exist_ok=True)
        atomic_json(self.cancelled_path, {**self._base(), "status": "cancelled", "reason": reason})
        self.request("cancel", reason)

    def sync_signals(self, run_id: str | None = None) -> None:
        if self.signal_state is not None and self.signal_state.term:
            if not self.intentional_requeue_teardown:
                self.cancel("slurm_sigterm")
        elif self.cancel_latch.exists() and not self.cancelled_path.exists():
            self.cancel("batch_shell_sigterm")
        elif self.signal_state is not None and self.signal_state.usr1 and self.request_payload() is None:
            self.request("requeue", "slurm_usr1", run_id=run_id)

    def mark_requeue_calling(self, *, fallback: bool) -> bool:
        self.sync_signals()
        request = self.request_payload()
        authorized = read_json(self.directory / "REQUEUE_AUTHORIZED.json")
        if (
            self.cancelled
            or request is None
            or request.get("action") != "requeue"
            or not self._belongs(authorized)
            or authorized.get("status") != "requeue_authorized"
        ):
            return False
        path = self.fallback_calling_path if fallback else self.batch_calling_path
        status = "fallback_scontrol_requeue_calling" if fallback else "batch_scontrol_requeue_calling"
        atomic_json(path, {**self._base(), "status": status})
        return True

    def mark_ready(self, *, run_id: str | None, child_exit_code: int | None) -> None:
        atomic_json(
            self.directory / f"ready.{self.rank}",
            {**self._base(), "status": "durable_and_ready", "run_id": run_id, "child_exit_code": child_exit_code},
        )

    def mark_finished(self) -> None:
        atomic_json(self.directory / f"finished.{self.rank}", {**self._base(), "status": "assigned_run_complete"})

    def _matching(self, prefix: str) -> set[int]:
        found = set()
        for rank in range(self.workers):
            payload = read_json(self.directory / f"{prefix}.{rank}")
            if self._belongs(payload) and payload.get("rank") == rank:
                found.add(rank)
        return found

    def all_finished(self) -> bool:
        return len(self._matching("finished")) == self.workers

    def _wait_ready(self) -> bool:
        deadline = time.monotonic() + self.wait_seconds
        while True:
            self.sync_signals()
            if len(self._matching("ready")) == self.workers:
                return True
            if time.monotonic() >= deadline:
                break
            time.sleep(POLL_SECONDS)
        missing = sorted(set(range(self.workers)) - self._matching("ready"))
        atomic_json(self.directory / "RENDEZVOUS_FAILED.json", {**self._base(), "status": "timeout", "missing_ranks": missing})
        return False

    def _fallback_requeue(self) -> int:
        if not self.mark_requeue_calling(fallback=True):
            return CANCEL_EXIT_CODE if self.cancelled else 1
        try:
            result = subprocess.run(
                ["scontrol", "requeue", self.job_id], check=False, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            result = subprocess.CompletedProcess([], 1, stdout=str(exc))
        if result.returncode != 0:
            durable_unlink(self.fallback_calling_path)
            atomic_json(self.directory / "REQUEUE_FAILED.json", {**self._base(), "status": "scontrol_requeue_failed", "output": result.stdout[-4000:]})
            return 1
        atomic_json(self.directory / "REQUEUE_CALLED.json", {**self._base(), "status": "scontrol_requeue_succeeded", "output": result.stdout[-4000:]})
        return GRACEFUL_EXIT_CODE

    def _resolve(self, *, wait: bool) -> int:
        all_ready = self._wait_ready() if wait else len(self._matching("ready")) == self.workers
        self.sync_signals()
        request = self.request_payload()
        if self.cancelled or (request and request.get("action") == "cancel"):
            atomic_json(self.directory / "CANCEL_RESOLVED.json", {**self._base(), "status": "cancelled_without_requeue"})
            return CANCEL_EXIT_CODE
        if request is None or request.get("action") == "abort":
            atomic_json(self.directory / "ABORTED.json", {**self._base(), "status": "aborted_without_requeue", "all_ready": all_ready, "request": request})
            return 1
        ready = sorted(self._matching("ready"))
        atomic_json(self.directory / "REQUEUE_AUTHORIZED.json", {**self._base(), "status": "requeue_authorized", "all_ready": all_ready, "ready_ranks": ready})
        return GRACEFUL_EXIT_CODE if all_ready else self._fallback_requeue()

    def coordinate_resolution(self) -> int:
        with self.resolution_lock.open("a+", encoding="utf-8") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            self.sync_signals()
            if self.cancelled or self._belongs(read_json(self.directory / "CANCEL_RESOLVED.json")):
                return CANCEL_EXIT_CODE
            if self._belongs(read_json(self.directory / "REQUEUE_CALLED.json")):
                return GRACEFUL_EXIT_CODE
            if self._belongs(read_json(self.directory / "REQUEUE_AUTHORIZED.json")):
                return GRACEFUL_EXIT_CODE
            if self._belongs(read_json(self.directory / "ABORTED.json")):
                return 1
            if self._belongs(read_json(self.directory / "REQUEUE_FAILED.json")):
                return 1
            return self._resolve(wait=True)

    def authorize_as_rank_zero(self) -> int:
        return self.coordinate_resolution()


def resolution_status(
    state_root: Path, *, job_id: str, restart_count: int, rank: int, workers: int,
    allocation_shard: int, allocation_shards: int, cancel_latch: Path,
) -> str:
    coordinator = RequeueCoordinator(
        state_root, job_id=job_id, restart_count=restart_count, rank=rank,
        workers=workers, wait_seconds=0, allocation_shard=allocation_shard,
        allocation_shards=allocation_shards, cancel_latch=cancel_latch,
    )
    if coordinator.cancelled:
        return "cancelled"
    request = coordinator.request_payload()
    if request is None or request.get("action") != "requeue":
        return "invalid"
    called = read_json(coordinator.directory / "REQUEUE_CALLED.json")
    if coordinator._belongs(called) and called.get("status") == "scontrol_requeue_succeeded":
        return "called"
    authorized = read_json(coordinator.directory / "REQUEUE_AUTHORIZED.json")
    ready = authorized.get("ready_ranks") if coordinator._belongs(authorized) else None
    if (
        authorized and authorized.get("status") == "requeue_authorized"
        and isinstance(authorized.get("all_ready"), bool) and isinstance(ready, list)
        and ready == sorted(set(ready))
        and all(isinstance(value, int) and 0 <= value < workers for value in ready)
        and bool(authorized["all_ready"]) == (ready == list(range(workers)))
    ):
        return "authorized"
    return "invalid"


def wandb_credentials_available() -> bool:
    try:
        credentials = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False
    return bool(credentials and credentials[2])


def _launch_spec(args: argparse.Namespace, manifest: Mapping[str, Any], run: RunSpec):
    result = trainer_command(
        manifest, run, python_executable=args.python, repo_root=args.repo_root,
        run_root=args.run_root, data_root=args.data_root, cache_root=args.cache_root,
        wandb_project=args.wandb_project, wandb_mode=args.wandb_mode,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise ValueError("campaign.trainer_command must return (argv, environment)")
    command, environment = result
    command = [str(value) for value in command]
    environment = {str(key): str(value) for key, value in environment.items()}
    joined = " ".join(command).lower()
    if not command or command[0] != str(args.python):
        raise ValueError("formal command changed the exact virtual-environment interpreter")
    required = (
        "scripts/train.py", "arm=treewm", "experiment=treewm_v2",
        "objective_version=treewm_v2_rms_rank_v1", "train.steps=1000000",
        "train.gradient_checkpointing=true", "retrieval.enabled=false",
    )
    if any(token not in command and token not in joined for token in required):
        raise ValueError("formal launch is missing a TreeWM-v2 objective invariant")
    if any(token in joined for token in ("upstream_", "agents/rql", "main.py --agent")):
        raise ValueError("formal command contains a baseline trainer token")
    if environment.get("WANDB_RUN_ID") != run.wandb_id:
        raise ValueError("formal command changed the stable W&B ID")
    if "v1" in str(args.run_root).lower() or "v1" in environment.get("WANDB_RUN_GROUP", "").lower():
        raise ValueError("formal v2 launch refuses a v1 run or W&B namespace")
    if any("KEY" in key.upper() or "TOKEN" in key.upper() for key in environment):
        raise ValueError("trainer environment must not contain credentials")
    return command, environment


def _completion_valid(args: argparse.Namespace, manifest: Mapping[str, Any], run: RunSpec) -> bool:
    return completion_is_valid(
        run_directory(args.run_root, run), manifest, run,
        repo_root=args.repo_root, cache_root=args.cache_root,
    )


def _run_child(command: Sequence[str], environment: Mapping[str, str], run: RunSpec, run_dir: Path, coordinator: RequeueCoordinator) -> int:
    attempt_dir = run_dir / "attempts"
    attempt_dir.mkdir(parents=True, exist_ok=True)
    stem = f"job{coordinator.job_id}.restart{coordinator.restart_count}"
    atomic_json(
        attempt_dir / f"{stem}.json",
        {"schema_version": 2, "run_id": run.run_id, "manifest_index": run.index,
         "job_id": coordinator.job_id, "restart_count": coordinator.restart_count,
         "rank": coordinator.rank, "command": list(command), "started_unix_time": time.time()},
    )
    child_env = os.environ.copy()
    child_env.update(environment)
    forwarded: str | None = None
    with (attempt_dir / f"{stem}.log").open("a", encoding="utf-8", buffering=1) as log:
        child = subprocess.Popen(list(command), stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, env=child_env)
        while child.poll() is None:
            if time.time() >= coordinator.allocation_stop_epoch and coordinator.request_payload() is None:
                coordinator.request("requeue", "allocation_wide_deadline", run_id=run.run_id)
            coordinator.sync_signals(run.run_id)
            request = coordinator.request_payload()
            if request is not None and str(request.get("action")) != forwarded:
                forwarded = str(request.get("action"))
                child.send_signal(signal.SIGUSR1 if forwarded == "requeue" else signal.SIGTERM)
            time.sleep(POLL_SECONDS)
        return int(child.returncode)


def _rendezvous(coordinator: RequeueCoordinator, run_id: str | None, child_exit: int | None) -> int:
    coordinator.mark_ready(run_id=run_id, child_exit_code=child_exit)
    return coordinator.coordinate_resolution()


def dispatch(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    runs = expand_runs(manifest)
    if len(runs) != 40 or args.workers != EXPECTED_WORKERS:
        raise ValueError("formal campaign requires exactly 40 runs and 16 ranks")
    if args.wandb_project != manifest["logging"]["wandb_project"]:
        raise ValueError("W&B project differs from immutable manifest")
    global_index = args.allocation_shard * args.workers + args.worker_index
    assigned = [runs[global_index]] if global_index < len(runs) else []
    if "SLURM_ARRAY_JOB_ID" in os.environ and slurm_requeue_target(os.environ) != args.job_id:
        raise ValueError("dispatcher may requeue only its composite array element")
    if "_" in args.job_id and int(args.job_id.rsplit("_", 1)[1]) != args.allocation_shard:
        raise ValueError("composite element ID does not match allocation shard")
    state = allocation_state_root(args.state_root, args.job_id, args.allocation_shard, args.allocation_shards)
    cancel_latch = args.cancel_latch or state / "CANCEL_REQUESTED"
    if args.dry_run:
        print(f"allocation {args.allocation_shard}/{args.allocation_shards}, rank {args.worker_index}: {len(assigned)} run")
        for run in assigned:
            command, environment = _launch_spec(args, manifest, run)
            status = "skip-complete" if _completion_valid(args, manifest, run) else "run-or-resume"
            print(f"[{run.index:02d}] {status}: env={environment} {shlex.join(command)}")
        return 0
    coordinator = RequeueCoordinator(
        state, job_id=args.job_id, restart_count=args.restart_count, rank=args.worker_index,
        workers=args.workers, wait_seconds=args.rendezvous_timeout,
        allocation_shard=args.allocation_shard, allocation_shards=args.allocation_shards,
        cancel_latch=cancel_latch,
    )
    if coordinator.cancelled:
        return CANCEL_EXIT_CODE
    if args.wandb_mode == "online" and not wandb_credentials_available():
        raise ValueError("W&B netrc authentication is unavailable")
    coordinator.allocation_stop_epoch = args.allocation_stop_epoch
    signal_state = SignalState()
    coordinator.signal_state = signal_state
    signal.signal(signal.SIGUSR1, signal_state.handle_usr1)
    signal.signal(signal.SIGTERM, signal_state.handle_term)
    last_run: str | None = None
    child_exit: int | None = None
    run_complete = not assigned
    for run in assigned:
        last_run = run.run_id
        run_dir = run_directory(args.run_root, run)
        if args.allocation_stop_epoch - time.time() <= args.minimum_child_seconds:
            coordinator.request("requeue", "insufficient_time_for_trainer", run_id=last_run)
            break
        try:
            with run_lock(run_dir, run=run, job_id=args.job_id, allocation_shard=args.allocation_shard, rank=args.worker_index):
                if _completion_valid(args, manifest, run):
                    run_complete = True
                    continue
                command, environment = _launch_spec(args, manifest, run)
                child_exit = _run_child(command, environment, run, run_dir, coordinator)
                coordinator.sync_signals(last_run)
                if child_exit == 0 and _completion_valid(args, manifest, run):
                    run_complete = True
                elif child_exit == GRACEFUL_EXIT_CODE and (run_dir / "checkpoints" / "latest.pt").is_file():
                    if coordinator.request_payload() is None:
                        coordinator.request("requeue", "trainer_graceful_stop", run_id=last_run)
                else:
                    coordinator.request("abort", f"trainer_failed_or_invalid_{child_exit}", run_id=last_run)
        except RunLockUnavailable:
            coordinator.request("abort", "run_lock_contention", run_id=last_run)
    coordinator.sync_signals(last_run)
    if coordinator.cancelled or coordinator.request_payload() is not None:
        return _rendezvous(coordinator, last_run, child_exit)
    if not run_complete:
        coordinator.request("abort", "dispatcher_left_run_incomplete", run_id=last_run)
        return _rendezvous(coordinator, last_run, child_exit)
    coordinator.mark_finished()
    while not coordinator.all_finished():
        if time.time() >= coordinator.allocation_stop_epoch and coordinator.request_payload() is None:
            coordinator.request("requeue", "allocation_deadline_while_idle", run_id=last_run)
        coordinator.sync_signals(last_run)
        if coordinator.cancelled or coordinator.request_payload() is not None:
            return _rendezvous(coordinator, last_run, child_exit)
        time.sleep(POLL_SECONDS)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=here / "manifest.json")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("TREEWM_RUN_ROOT", repo_root / "outputs" / "treewm-50task-1m-v2")))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("TREEWM_DATA_ROOT", here / "data")))
    parser.add_argument("--cache-root", type=Path, default=Path(os.environ.get("TREEWM_CACHE", here / "cache")))
    parser.add_argument("--state-root", type=Path)
    parser.add_argument("--cancel-latch", type=Path)
    parser.add_argument("--python", default=os.environ.get("TREEWM_PYTHON", sys.executable))
    parser.add_argument("--worker-index", type=int, default=int(os.environ.get("SLURM_PROCID", "0")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_NTASKS", EXPECTED_WORKERS)))
    parser.add_argument("--allocation-shard", type=int, default=int(os.environ["SLURM_ARRAY_TASK_ID"]) if "SLURM_ARRAY_TASK_ID" in os.environ else None)
    parser.add_argument("--allocation-shards", type=int, default=int(os.environ.get("TREEWM_ALLOCATION_SHARDS", EXPECTED_ALLOCATION_SHARDS)))
    parser.add_argument("--job-id", default=slurm_requeue_target(os.environ))
    parser.add_argument("--restart-count", type=int, default=int(os.environ.get("SLURM_RESTART_COUNT", "0")))
    parser.add_argument("--rendezvous-timeout", type=int, default=240)
    parser.add_argument("--allocation-stop-epoch", type=float, default=float(os.environ.get("TREEWM_ALLOCATION_STOP_EPOCH", str(time.time() + 13200))))
    parser.add_argument("--minimum-child-seconds", type=int, default=300)
    parser.add_argument("--wandb-project", default=os.environ.get("WANDB_PROJECT", "treewm-50task-formal-v2"))
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resolution-status", action="store_true")
    parser.add_argument("--mark-batch-requeue-calling", action="store_true")
    args = parser.parse_args(argv)
    if args.allocation_shard is None:
        parser.error("--allocation-shard or SLURM_ARRAY_TASK_ID is required")
    if args.state_root is None:
        args.state_root = args.run_root / "state" / "arrays"
    for name in ("manifest", "repo_root", "run_root", "data_root", "cache_root", "state_root"):
        setattr(args, name, getattr(args, name).resolve())
    if args.cancel_latch is not None:
        args.cancel_latch = args.cancel_latch.resolve()
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        state = allocation_state_root(args.state_root, args.job_id, args.allocation_shard, args.allocation_shards)
        cancel_latch = args.cancel_latch or state / "CANCEL_REQUESTED"
        if args.resolution_status:
            print(resolution_status(
                state, job_id=args.job_id, restart_count=args.restart_count,
                rank=args.worker_index, workers=args.workers,
                allocation_shard=args.allocation_shard, allocation_shards=args.allocation_shards,
                cancel_latch=cancel_latch,
            ))
            return 0
        if args.mark_batch_requeue_calling:
            coordinator = RequeueCoordinator(
                state, job_id=args.job_id, restart_count=args.restart_count,
                rank=args.worker_index, workers=args.workers, wait_seconds=0,
                allocation_shard=args.allocation_shard, allocation_shards=args.allocation_shards,
                cancel_latch=cancel_latch,
            )
            return 0 if coordinator.mark_requeue_calling(fallback=False) else 2
        return dispatch(args)
    except (ValueError, OSError) as exc:
        print(f"dispatcher error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
