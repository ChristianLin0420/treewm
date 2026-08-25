#!/usr/bin/env python3
"""Persistent 16-rank dispatcher for one-GPU-per-run RQL training.

Slurm's literal ``--signal=USR1@420`` targets job-step tasks, not the batch
shell.  Each rank therefore catches USR1, forwards it to its active trainer,
waits for the trainer's atomic checkpoint (exit 75), and participates in a
shared-filesystem rendezvous.  Rank 0 invokes ``scontrol requeue`` only after
all 16 ranks have declared themselves safe.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import netrc
import os
import re
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
    build_train_command,
    completion_is_valid,
    expand_runs,
    load_manifest,
    redact_command,
    run_directory,
    worker_runs,
)


REQUEUE_EXIT_CODE = 75
POLL_SECONDS = 1.0
SLURM_ELEMENT_JOB_ID = re.compile(r"^[0-9]+(?:_[0-9]+)?$")


def allocation_state_root(
    state_root: Path,
    job_id: str,
    allocation_shard: int,
    allocation_shards: int,
) -> Path:
    """Give every array element a collision-free coordination namespace."""

    if SLURM_ELEMENT_JOB_ID.fullmatch(job_id) is None:
        raise ValueError(f"invalid Slurm job/array-element ID: {job_id!r}")
    if allocation_shards != EXPECTED_ALLOCATION_SHARDS:
        raise ValueError(f"formal dispatch requires {EXPECTED_ALLOCATION_SHARDS} allocation shards")
    if not 0 <= allocation_shard < allocation_shards:
        raise ValueError(f"allocation shard {allocation_shard} is outside [0, {allocation_shards})")
    return (
        state_root.resolve()
        / f"array-job-{job_id}"
        / f"allocation-shard-{allocation_shard:02d}-of-{allocation_shards:02d}"
    )


def slurm_requeue_target(environ: Mapping[str, str]) -> str:
    """Return one array element, never an array master, as requeue target."""

    array_job_id = environ.get("SLURM_ARRAY_JOB_ID")
    array_task_id = environ.get("SLURM_ARRAY_TASK_ID")
    if (array_job_id is None) != (array_task_id is None):
        raise ValueError("SLURM_ARRAY_JOB_ID and SLURM_ARRAY_TASK_ID must be set together")
    if array_job_id is not None:
        target = f"{array_job_id}_{array_task_id}"
    else:
        target = environ.get("SLURM_JOB_ID", "0")
    if SLURM_ELEMENT_JOB_ID.fullmatch(target) is None:
        raise ValueError(f"invalid Slurm requeue target: {target!r}")
    return target


class RunLockUnavailable(RuntimeError):
    """A second allocation attempted to mutate the same durable run."""


@contextlib.contextmanager
def run_lock(
    run_dir: Path,
    *,
    run: RunSpec,
    job_id: str,
    allocation_shard: int,
    rank: int,
):
    """Hold a nonblocking per-run lease for the complete trainer lifetime."""

    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = run_dir / ".dispatcher.lock"
    with lock_path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RunLockUnavailable(
                f"run {run.run_id} is already owned by another allocation: {lock_path}"
            ) from exc
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


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a small coordination/provenance file with rename durability."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        directory_fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # Some parallel filesystems reject directory fsync; atomic rename still
        # prevents readers from observing partial JSON.
        pass


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


class RequeueCoordinator:
    """Generation-scoped shared-filesystem barrier for preemption safety."""

    def __init__(
        self,
        state_root: Path,
        *,
        job_id: str,
        restart_count: int,
        rank: int,
        workers: int,
        wait_seconds: int,
        allocation_shard: int | None = None,
        allocation_shards: int | None = None,
    ) -> None:
        if workers != EXPECTED_WORKERS:
            raise ValueError(f"formal rendezvous requires {EXPECTED_WORKERS} ranks")
        if SLURM_ELEMENT_JOB_ID.fullmatch(job_id) is None:
            raise ValueError(f"SLURM_JOB_ID must be numeric or an array element ID, got {job_id!r}")
        self.job_id = job_id
        self.restart_count = restart_count
        self.rank = rank
        self.workers = workers
        self.wait_seconds = wait_seconds
        self.allocation_shard = allocation_shard
        self.allocation_shards = allocation_shards
        # Keep this exact hierarchy: state/requeue/$SLURM_RESTART_COUNT/ready.$rank.
        self.directory = state_root.resolve() / "requeue" / str(restart_count)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.request_path = self.directory / "REQUESTED.json"
        self.request_lock = self.directory / ".request.lock"

    def _base(self) -> dict[str, Any]:
        payload = {
            "job_id": self.job_id,
            "restart_count": self.restart_count,
            "rank": self.rank,
            "unix_time": time.time(),
        }
        if self.allocation_shard is not None:
            payload["allocation_shard"] = self.allocation_shard
            payload["allocation_shards"] = self.allocation_shards
        return payload

    def _belongs_to_job(self, payload: Mapping[str, Any] | None) -> bool:
        belongs = bool(
            payload
            and payload.get("job_id") == self.job_id
            and payload.get("restart_count") == self.restart_count
        )
        if self.allocation_shard is not None:
            belongs = bool(
                belongs
                and payload.get("allocation_shard") == self.allocation_shard
                and payload.get("allocation_shards") == self.allocation_shards
            )
        return belongs

    def request(self, reason: str, *, abort: bool = False, run_id: str | None = None) -> None:
        """Publish a global stop request; abort takes precedence over requeue."""

        with self.request_lock.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            existing = read_json(self.request_path)
            existing_abort = self._belongs_to_job(existing) and existing.get("action") == "abort"
            if existing_abort and not abort:
                return
            payload = self._base()
            payload.update({"action": "abort" if abort else "requeue", "reason": reason})
            if run_id is not None:
                payload["run_id"] = run_id
            atomic_json(self.request_path, payload)

    def request_payload(self) -> dict[str, Any] | None:
        payload = read_json(self.request_path)
        return payload if self._belongs_to_job(payload) else None

    def mark_ready(self, *, run_id: str | None, child_exit_code: int | None) -> None:
        payload = self._base()
        payload.update({"status": "ready", "run_id": run_id, "child_exit_code": child_exit_code})
        atomic_json(self.directory / f"ready.{self.rank}", payload)

    def mark_finished(self) -> None:
        payload = self._base()
        payload.update({"status": "assigned_runs_complete"})
        atomic_json(self.directory / f"finished.{self.rank}", payload)

    def _matching_ranks(self, prefix: str) -> set[int]:
        matching: set[int] = set()
        for rank in range(self.workers):
            payload = read_json(self.directory / f"{prefix}.{rank}")
            if self._belongs_to_job(payload) and payload.get("rank") == rank:
                matching.add(rank)
        return matching

    def all_finished(self) -> bool:
        return len(self._matching_ranks("finished")) == self.workers

    def wait_for_all_ready(self) -> bool:
        deadline = time.monotonic() + self.wait_seconds
        while time.monotonic() < deadline:
            ready = self._matching_ranks("ready")
            if len(ready) == self.workers:
                return True
            time.sleep(POLL_SECONDS)
        missing = sorted(set(range(self.workers)) - self._matching_ranks("ready"))
        atomic_json(
            self.directory / "RENDEZVOUS_FAILED.json",
            {**self._base(), "status": "timeout", "missing_ranks": missing},
        )
        print(f"rank 0: requeue rendezvous timed out; missing ranks: {missing}", file=sys.stderr, flush=True)
        return False

    def wait_for_resolution(self) -> int:
        """Nonzero ranks stay alive until rank 0 resolves the rendezvous."""

        while True:
            aborted = read_json(self.directory / "ABORTED.json")
            if self._belongs_to_job(aborted):
                return 1
            called = read_json(self.directory / "REQUEUE_CALLED.json")
            if self._belongs_to_job(called):
                # Normally Slurm kills this process immediately after scontrol.
                # Returning 75 is a fallback and is never treated as auto-requeue.
                return REQUEUE_EXIT_CODE
            failed = read_json(self.directory / "REQUEUE_FAILED.json")
            if self._belongs_to_job(failed):
                return 1
            time.sleep(POLL_SECONDS)

    def resolve_as_rank_zero(self) -> int:
        all_ready = self.wait_for_all_ready()
        request = self.request_payload()
        if request is None or request.get("action") == "abort":
            atomic_json(
                self.directory / "ABORTED.json",
                {**self._base(), "status": "aborted_without_requeue", "request": request},
            )
            return 1

        if not all_ready:
            # A missing/crashed rank cannot acknowledge the barrier.  Requeue
            # anyway for a preemption request: every trainer also writes
            # periodic atomic checkpoints, and this cluster has no RequeueExit
            # policy.  Suppressing scontrol here would terminate the campaign.
            print(
                "rank 0: readiness was incomplete; requeueing from latest periodic checkpoints",
                file=sys.stderr,
                flush=True,
            )

        # argv form plus numeric validation above prevents command injection.
        result = subprocess.run(
            ["scontrol", "requeue", self.job_id],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        if result.returncode != 0:
            atomic_json(
                self.directory / "REQUEUE_FAILED.json",
                {
                    **self._base(),
                    "status": "scontrol_failed",
                    "returncode": result.returncode,
                    "output": result.stdout[-4000:],
                },
            )
            print(f"scontrol requeue failed: {result.stdout}", file=sys.stderr, flush=True)
            return 1
        atomic_json(
            self.directory / "REQUEUE_CALLED.json",
            {**self._base(), "status": "scontrol_requeue_succeeded", "output": result.stdout[-4000:]},
        )
        return REQUEUE_EXIT_CODE


class SignalState:
    def __init__(self) -> None:
        self.usr1_received = False

    def handle_stop(self, signum: int, frame: object) -> None:
        del frame
        self.usr1_received = True


def wandb_credentials_available() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        credentials = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False
    return bool(credentials and credentials[2])


def _attempt_metadata(
    run_dir: Path,
    run: RunSpec,
    command: Sequence[str],
    coordinator: RequeueCoordinator,
) -> Path:
    attempt_dir = run_dir / "attempts"
    attempt_path = attempt_dir / f"job{coordinator.job_id}.restart{coordinator.restart_count}.json"
    atomic_json(
        attempt_path,
        {
            "run_id": run.run_id,
            "manifest_index": run.index,
            "job_id": coordinator.job_id,
            "restart_count": coordinator.restart_count,
            "rank": coordinator.rank,
            "command": list(command),
            "started_unix_time": time.time(),
        },
    )
    return attempt_path


def _run_child(
    command: Sequence[str],
    run: RunSpec,
    run_dir: Path,
    coordinator: RequeueCoordinator,
    signal_state: SignalState,
) -> int:
    """Launch one trainer and forward local/global stop requests."""

    run_dir.mkdir(parents=True, exist_ok=True)
    _attempt_metadata(run_dir, run, command, coordinator)
    log_path = run_dir / "attempts" / f"job{coordinator.job_id}.restart{coordinator.restart_count}.log"
    print(f"rank {coordinator.rank}: launch {run.run_id}", flush=True)
    forwarded = False
    with log_path.open("a", encoding="utf-8", buffering=1) as log_handle:
        child = subprocess.Popen(
            list(command),
            stdin=subprocess.DEVNULL,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        while True:
            if time.time() >= coordinator.allocation_stop_epoch and coordinator.request_payload() is None:
                coordinator.request("allocation_wide_deadline", run_id=run.run_id)
            if signal_state.usr1_received and coordinator.request_payload() is None:
                coordinator.request("slurm_usr1", run_id=run.run_id)
            request = coordinator.request_payload()
            if request is not None and not forwarded and child.poll() is None:
                print(
                    f"rank {coordinator.rank}: forwarding USR1 to {run.run_id}",
                    flush=True,
                )
                child.send_signal(signal.SIGUSR1)
                forwarded = True
            returncode = child.poll()
            if returncode is not None:
                return returncode
            time.sleep(POLL_SECONDS)


def _rendezvous(
    coordinator: RequeueCoordinator,
    *,
    run_id: str | None,
    child_exit_code: int | None,
) -> int:
    coordinator.mark_ready(run_id=run_id, child_exit_code=child_exit_code)
    print(f"rank {coordinator.rank}: durable and ready for coordinator", flush=True)
    if coordinator.rank == 0:
        return coordinator.resolve_as_rank_zero()
    return coordinator.wait_for_resolution()


def dispatch(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    runs = expand_runs(manifest)
    if "SLURM_ARRAY_JOB_ID" in os.environ or "SLURM_ARRAY_TASK_ID" in os.environ:
        expected_requeue_target = slurm_requeue_target(os.environ)
        if args.job_id != expected_requeue_target:
            raise ValueError(
                f"array dispatcher must requeue only {expected_requeue_target}, got {args.job_id}"
            )
    assigned = worker_runs(
        runs,
        args.worker_index,
        args.workers,
        allocation_shard=args.allocation_shard,
        allocation_shards=args.allocation_shards,
    )
    isolated_state_root = allocation_state_root(
        Path(args.state_root),
        args.job_id,
        args.allocation_shard,
        args.allocation_shards,
    )
    if "_" in args.job_id and int(args.job_id.rsplit("_", 1)[1]) != args.allocation_shard:
        raise ValueError(
            f"array requeue target {args.job_id} does not match allocation shard {args.allocation_shard}"
        )

    if args.dry_run:
        print(
            f"allocation {args.allocation_shard}/{args.allocation_shards}, "
            f"worker {args.worker_index}/{args.workers}: {len(assigned)} deterministic runs"
        )
        print(f"isolated state root: {isolated_state_root}")
        for run in assigned:
            command = build_train_command(
                manifest,
                run,
                python_executable=args.python,
                upstream_main=args.upstream_main,
                run_root=args.run_root,
                data_root=args.data_root,
                wandb_project=args.wandb_project,
                wandb_entity=args.wandb_entity,
                wandb_mode=args.wandb_mode,
            )
            status = "skip-complete" if completion_is_valid(run_directory(args.run_root, run), manifest, run) else "run-or-resume"
            print(f"[{run.index:03d}] {status}: {redact_command(command)}")
        return 0

    if args.wandb_mode == "online" and not wandb_credentials_available():
        print("W&B login is required (WANDB_API_KEY or api.wandb.ai in ~/.netrc)", file=sys.stderr)
        return 2

    coordinator = RequeueCoordinator(
        isolated_state_root,
        job_id=args.job_id,
        restart_count=args.restart_count,
        rank=args.worker_index,
        workers=args.workers,
        wait_seconds=args.rendezvous_timeout,
        allocation_shard=args.allocation_shard,
        allocation_shards=args.allocation_shards,
    )
    coordinator.allocation_stop_epoch = args.allocation_stop_epoch
    signal_state = SignalState()
    signal.signal(signal.SIGUSR1, signal_state.handle_stop)
    signal.signal(signal.SIGTERM, signal_state.handle_stop)

    last_run_id: str | None = None
    last_child_exit: int | None = None
    for run in assigned:
        request = coordinator.request_payload()
        seconds_to_stop = args.allocation_stop_epoch - time.time()
        if seconds_to_stop <= args.minimum_child_seconds and request is None:
            coordinator.request("insufficient_time_for_new_trainer", run_id=last_run_id)
            request = coordinator.request_payload()
        if signal_state.usr1_received or request is not None:
            if request is None:
                coordinator.request("slurm_usr1_between_runs", run_id=last_run_id)
            break
        run_dir = run_directory(args.run_root, run)
        try:
            with run_lock(
                run_dir,
                run=run,
                job_id=args.job_id,
                allocation_shard=args.allocation_shard,
                rank=args.worker_index,
            ):
                if completion_is_valid(run_dir, manifest, run):
                    print(f"rank {args.worker_index}: skip complete {run.run_id}", flush=True)
                    continue

                command = build_train_command(
                    manifest,
                    run,
                    python_executable=args.python,
                    upstream_main=args.upstream_main,
                    run_root=args.run_root,
                    data_root=args.data_root,
                    wandb_project=args.wandb_project,
                    wandb_entity=args.wandb_entity,
                    wandb_mode=args.wandb_mode,
                    walltime_seconds_override=max(
                        1,
                        min(manifest["training"]["walltime_seconds"], int(seconds_to_stop)),
                    ),
                )
                last_run_id = run.run_id
                last_child_exit = _run_child(command, run, run_dir, coordinator, signal_state)
                request = coordinator.request_payload()

                if last_child_exit == 0:
                    if not completion_is_valid(run_dir, manifest, run):
                        coordinator.request(
                            "trainer_exited_without_valid_completion",
                            abort=True,
                            run_id=run.run_id,
                        )
                elif last_child_exit == REQUEUE_EXIT_CODE:
                    if not (run_dir / "checkpoint.pkl").is_file() and not completion_is_valid(
                        run_dir, manifest, run
                    ):
                        coordinator.request(
                            "trainer_exit_75_without_checkpoint",
                            abort=True,
                            run_id=run.run_id,
                        )
                    elif request is None:
                        coordinator.request("trainer_walltime", run_id=run.run_id)
                else:
                    coordinator.request(
                        f"trainer_failed_exit_{last_child_exit}",
                        abort=True,
                        run_id=run.run_id,
                    )
        except RunLockUnavailable as exc:
            last_run_id = run.run_id
            print(str(exc), file=sys.stderr, flush=True)
            coordinator.request("run_lock_contention", abort=True, run_id=run.run_id)

        request = coordinator.request_payload()
        if request is not None:
            break

    request = coordinator.request_payload()
    if request is not None or signal_state.usr1_received:
        if request is None:
            coordinator.request("slurm_usr1_after_runs", run_id=last_run_id)
        return _rendezvous(
            coordinator,
            run_id=last_run_id,
            child_exit_code=last_child_exit,
        )

    # A rank with no pending work must stay in the step so a later USR1 can
    # still collect all 16 ready markers.  It exits normally only when every
    # rank's deterministic shard is complete.
    coordinator.mark_finished()
    print(f"rank {args.worker_index}: assigned shard complete; entering campaign barrier", flush=True)
    while not coordinator.all_finished():
        if signal_state.usr1_received and coordinator.request_payload() is None:
            coordinator.request("slurm_usr1_while_idle", run_id=last_run_id)
        if coordinator.request_payload() is not None:
            return _rendezvous(
                coordinator,
                run_id=last_run_id,
                child_exit_code=last_child_exit,
            )
        time.sleep(POLL_SECONDS)
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=here / "manifest.json")
    parser.add_argument("--upstream-main", type=Path, default=here / "upstream_rql" / "main.py")
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("RQL_RUN_ROOT", here / "output")))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("RQL_DATA_ROOT", here / "data")))
    parser.add_argument("--state-root", type=Path, default=None)
    parser.add_argument("--python", default=os.environ.get("RQL_PYTHON", sys.executable))
    parser.add_argument("--worker-index", type=int, default=int(os.environ.get("SLURM_PROCID", "0")))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_NTASKS", EXPECTED_WORKERS)))
    parser.add_argument(
        "--allocation-shard",
        type=int,
        default=(
            int(os.environ["SLURM_ARRAY_TASK_ID"])
            if "SLURM_ARRAY_TASK_ID" in os.environ
            else None
        ),
        help="zero-based Slurm array element; required for formal dispatch",
    )
    parser.add_argument(
        "--allocation-shards",
        type=int,
        default=int(os.environ.get("RQL_ALLOCATION_SHARDS", EXPECTED_ALLOCATION_SHARDS)),
        help="formal array width (must remain 13)",
    )
    parser.add_argument("--job-id", default=slurm_requeue_target(os.environ))
    parser.add_argument("--restart-count", type=int, default=int(os.environ.get("SLURM_RESTART_COUNT", "0")))
    parser.add_argument("--rendezvous-timeout", type=int, default=240)
    parser.add_argument(
        "--allocation-stop-epoch",
        type=float,
        default=float(os.environ.get("RQL_ALLOCATION_STOP_EPOCH", str(time.time() + 13_200))),
        help="shared wall-clock deadline; all ranks checkpoint/requeue at this epoch",
    )
    parser.add_argument("--minimum-child-seconds", type=int, default=300)
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.state_root is None:
        args.state_root = args.run_root / "state"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return dispatch(parse_args(argv))
    except (ValueError, OSError) as exc:
        print(f"dispatcher error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
