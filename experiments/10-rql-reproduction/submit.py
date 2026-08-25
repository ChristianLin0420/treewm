#!/usr/bin/env python3
"""Validate and optionally submit the formal RQL campaign.

Dry-run is the default.  This script never accepts API tokens as arguments and
never prints them; Slurm inherits credentials from the submitter's environment.
"""

from __future__ import annotations

import argparse
import netrc
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from campaign import (
    EXPECTED_ALLOCATION_SHARDS,
    EXPECTED_RUN_COUNT,
    EXPECTED_WORKERS,
    build_train_command,
    completion_is_valid,
    expand_runs,
    load_manifest,
    manifest_sha256,
    run_directory,
    worker_runs,
)


REQUIRED_SBATCH_LINES = (
    "#SBATCH --partition=polar4,polar3,polar,grizzly",
    "#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip",
    "#SBATCH --qos=normal",
    "#SBATCH --time=04:00:00",
    "#SBATCH --requeue",
    "#SBATCH --signal=USR1@420",
    "#SBATCH --nodes=2",
    "#SBATCH --ntasks-per-node=8",
    "#SBATCH --gpus-per-node=8",
    "#SBATCH --cpus-per-task=12",
    "#SBATCH --array=0-12%13",
    "#SBATCH --output=logs/%x_%j.out",
)
REQUIRED_SRUN_SNIPPETS = (
    "--ntasks=16",
    "--ntasks-per-node=8",
    "--gpus-per-task=1",
    "--gpu-bind=single:1",
    "--kill-on-bad-exit=0",
    "--allocation-shard",
    "--allocation-shards",
)
REQUIRED_STAGE_SBATCH_LINES = (
    "#SBATCH --partition=polar4,polar3,polar,grizzly",
    "#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip",
    "#SBATCH --qos=normal",
    "#SBATCH --time=04:00:00",
    "#SBATCH --requeue",
    "#SBATCH --signal=USR1@420",
    "#SBATCH --nodes=1",
    "#SBATCH --ntasks-per-node=1",
    "#SBATCH --gpus-per-node=1",
    "#SBATCH --cpus-per-task=4",
    "#SBATCH --output=logs/%x_%j.out",
)
REQUIRED_AGGREGATE_SBATCH_LINES = (
    "#SBATCH --partition=polar4,polar3,polar,grizzly",
    "#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip",
    "#SBATCH --qos=normal",
    "#SBATCH --time=00:30:00",
    "#SBATCH --nodes=1",
    "#SBATCH --ntasks-per-node=1",
    "#SBATCH --gpus-per-node=1",
    "#SBATCH --cpus-per-task=4",
    "#SBATCH --output=logs/%x_%j.out",
)
TRAINER_FLAGS = (
    "run_dir",
    "run_name",
    "wandb_id",
    "wandb_project",
    "wandb_mode",
    "resume",
    "gradient_checkpointing",
    "walltime_seconds",
    "checkpoint_interval",
    "ogbench_standard_dataset_dir",
    "protocol_sha256",
)
SECRET_PATTERNS = (
    re.compile(r"wandb_v1_[A-Za-z0-9_-]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
)


class ValidationError(RuntimeError):
    pass


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def validate_slurm(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for required in REQUIRED_SBATCH_LINES:
        _assert(lines.count(required) == 1, f"Slurm file must contain exactly one `{required}`")
    for required in REQUIRED_SRUN_SNIPPETS:
        _assert(required in text, f"Slurm srun command is missing {required}")
    _assert("scontrol requeue" in text, "Slurm documentation must make explicit requeue semantics visible")


def validate_stage_slurm(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for required in REQUIRED_STAGE_SBATCH_LINES:
        _assert(lines.count(required) == 1, f"data-stage Slurm file must contain exactly one `{required}`")
    _assert("--download" in text, "data-stage job does not invoke resumable download mode")
    _assert("scontrol requeue" in text, "data-stage job lacks explicit requeue")


def validate_aggregate_slurm(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for required in REQUIRED_AGGREGATE_SBATCH_LINES:
        _assert(lines.count(required) == 1, f"aggregate Slurm file must contain exactly one `{required}`")
    _assert("aggregate.py" in text, "aggregate Slurm file does not invoke the strict reporter")


def validate_trainer_interface(main_path: Path) -> None:
    _assert(main_path.is_file(), f"trainer missing: {main_path}")
    text = main_path.read_text(encoding="utf-8")
    for flag in TRAINER_FLAGS:
        _assert(flag in text, f"trainer does not expose required campaign flag --{flag}")
    _assert("COMPLETED.json" in text, "trainer does not write the completion sentinel")
    _assert("checkpoint.pkl" in text, "trainer does not expose the atomic resume checkpoint")


def scan_for_embedded_secrets(campaign_dir: Path) -> None:
    suffixes = {".py", ".json", ".md", ".sh", ".slurm", ".txt"}
    for path in campaign_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in SECRET_PATTERNS:
            _assert(pattern.search(text) is None, f"embedded credential-like string found in {path}")


def validate_campaign(args: argparse.Namespace) -> tuple[dict, list, int]:
    manifest = load_manifest(args.manifest)
    runs = expand_runs(manifest)
    _assert(len(runs) == EXPECTED_RUN_COUNT, f"expected {EXPECTED_RUN_COUNT} runs")
    ownership: dict[int, tuple[int, int]] = {}
    shard_sizes: list[int] = []
    for allocation_shard in range(EXPECTED_ALLOCATION_SHARDS):
        shard_size = 0
        for rank in range(EXPECTED_WORKERS):
            assigned = worker_runs(
                runs,
                rank,
                allocation_shard=allocation_shard,
                allocation_shards=EXPECTED_ALLOCATION_SHARDS,
            )
            _assert(len(assigned) <= 1, "each GPU rank may own at most one maximally parallel run")
            for run in assigned:
                _assert(run.index not in ownership, f"run {run.index} has duplicate allocation ownership")
                ownership[run.index] = (allocation_shard, rank)
                shard_size += 1
        shard_sizes.append(shard_size)
    _assert(shard_sizes == [16] * 12 + [8], f"unexpected allocation shard sizes: {shard_sizes}")
    _assert(set(ownership) == set(range(EXPECTED_RUN_COUNT)), "array ownership has a gap or duplicate")

    validate_slurm(args.slurm)
    validate_stage_slurm(args.stage_slurm)
    validate_aggregate_slurm(args.aggregate_slurm)
    validate_trainer_interface(args.upstream_main)
    scan_for_embedded_secrets(args.campaign_dir)
    expected_protocol = manifest_sha256(manifest)
    try:
        locked_protocol = args.protocol_lock.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise ValidationError(f"protocol lock missing: {args.protocol_lock}") from exc
    _assert(locked_protocol == expected_protocol, "protocol.sha256 does not match canonical manifest semantics")

    # Construct every argv to catch missing data fields and identity collisions.
    commands = [
        build_train_command(
            manifest,
            run,
            python_executable=args.python,
            upstream_main=args.upstream_main,
            run_root=args.run_root,
            data_root=args.data_root,
            wandb_mode="online",
        )
        for run in runs
    ]
    _assert(all("--gradient_checkpointing=true" in cmd for cmd in commands), "gradient checkpointing is not universal")
    _assert(all("--offline_steps=1000000" in cmd for cmd in commands), "exact 1M training is not universal")
    _assert(all("--eval_episodes=50" in cmd for cmd in commands), "50-episode evaluation is not universal")
    _assert(all("--final_eval_episodes=50" in cmd for cmd in commands), "50-episode final evaluation is not universal")
    _assert(
        all(f"--protocol_sha256={expected_protocol}" in cmd for cmd in commands),
        "canonical protocol fingerprint is not universal",
    )
    complete = sum(completion_is_valid(run_directory(args.run_root, run), manifest, run) for run in runs)
    return manifest, runs, complete


def strict_data_check(args: argparse.Namespace) -> None:
    # Imported lazily so manifest/source validation itself remains dependency-free.
    from prepare_data import check_all_data

    missing = check_all_data(load_manifest(args.manifest), args.data_root)
    if missing:
        preview = "\n  ".join(str(path) for path in missing[:20])
        suffix = f"\n  ... and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise ValidationError(f"data preflight found {len(missing)} missing/invalid files:\n  {preview}{suffix}")


def wandb_credentials_available() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    try:
        credentials = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False
    return bool(credentials and credentials[2])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate and print the inert sbatch command (default)")
    mode.add_argument("--submit", action="store_true", help="submit only after strict data and credential checks")
    parser.add_argument("--require-data", action="store_true", help="make data availability fatal during dry-run")
    parser.add_argument("--manifest", type=Path, default=here / "manifest.json")
    parser.add_argument("--slurm", type=Path, default=here / "train.slurm")
    parser.add_argument("--stage-slurm", type=Path, default=here / "stage_data.slurm")
    parser.add_argument("--aggregate-slurm", type=Path, default=here / "aggregate.slurm")
    prerequisite = parser.add_mutually_exclusive_group()
    prerequisite.add_argument(
        "--stage-data",
        action="store_true",
        help="submit resumable data staging first and hold training with afterok dependency",
    )
    prerequisite.add_argument(
        "--data-job-id",
        help="reuse an existing numeric data-stage job as the training afterok dependency",
    )
    parser.add_argument("--upstream-main", type=Path, default=here / "upstream_rql" / "main.py")
    parser.add_argument("--campaign-dir", type=Path, default=here)
    parser.add_argument("--protocol-lock", type=Path, default=here / "protocol.sha256")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("RQL_DATA_ROOT", here / "data")))
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("RQL_RUN_ROOT", here / "output")))
    parser.add_argument("--python", default=os.environ.get("RQL_PYTHON", sys.executable))
    args = parser.parse_args(argv)
    if not args.submit:
        args.dry_run = True
    for name in ("manifest", "slurm", "stage_slurm", "aggregate_slurm", "upstream_main", "campaign_dir", "protocol_lock", "repo_root", "data_root", "run_root"):
        setattr(args, name, getattr(args, name).resolve())
    if args.data_job_id is not None and not args.data_job_id.isdigit():
        parser.error("--data-job-id must be numeric")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        _, runs, complete = validate_campaign(args)
        if (args.submit and not args.stage_data and args.data_job_id is None) or args.require_data:
            strict_data_check(args)
        if args.submit:
            _assert(
                wandb_credentials_available(),
                "W&B credentials are unavailable (set WANDB_API_KEY or run `wandb login` to populate ~/.netrc)",
            )
            _assert(shlex.split(args.python), "RQL_PYTHON/Python command is empty")
            (args.repo_root / "logs").mkdir(parents=True, exist_ok=True)
            args.run_root.mkdir(parents=True, exist_ok=True)
            environment = os.environ.copy()
            environment["RQL_DATA_ROOT"] = str(args.data_root)
            environment["RQL_RUN_ROOT"] = str(args.run_root)
            environment["RQL_PYTHON"] = args.python
            dependency: list[str] = []
            if args.stage_data:
                stage_result = subprocess.run(
                    ["sbatch", "--parsable", str(args.stage_slurm)],
                    cwd=args.repo_root,
                    env=environment,
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                if stage_result.returncode != 0:
                    print(stage_result.stdout, file=sys.stderr, end="")
                    return stage_result.returncode
                stage_job_id = stage_result.stdout.strip().split(";", 1)[0]
                _assert(stage_job_id.isdigit(), f"could not parse data-stage job ID: {stage_result.stdout!r}")
                print(f"submitted resumable data stage job {stage_job_id}")
                dependency = [f"--dependency=afterok:{stage_job_id}"]
            elif args.data_job_id is not None:
                dependency = [f"--dependency=afterok:{args.data_job_id}"]

            train_result = subprocess.run(
                ["sbatch", "--parsable", *dependency, str(args.slurm)],
                cwd=args.repo_root,
                env=environment,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if train_result.returncode != 0:
                print(train_result.stdout, file=sys.stderr, end="")
                return train_result.returncode
            train_array_id = train_result.stdout.strip().split(";", 1)[0]
            _assert(train_array_id.isdigit(), f"could not parse training array ID: {train_result.stdout!r}")
            print(f"submitted 13-element training array job {train_array_id}")

            report_result = subprocess.run(
                [
                    "sbatch",
                    "--parsable",
                    f"--dependency=afterok:{train_array_id}",
                    str(args.aggregate_slurm),
                ],
                cwd=args.repo_root,
                env=environment,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
            if report_result.returncode != 0:
                print(report_result.stdout, file=sys.stderr, end="")
                return report_result.returncode
            report_job_id = report_result.stdout.strip().split(";", 1)[0]
            _assert(report_job_id.isdigit(), f"could not parse report job ID: {report_result.stdout!r}")
            print(f"submitted strict afterok report job {report_job_id}")
            return 0

        print("validation: OK")
        print(f"campaign: {len(runs)} runs; {complete} complete; {len(runs) - complete} pending")
        print("parallelism: 13 allocations x 16 ranks; shard sizes 16 x 12 and 8 x 1")
        print(f"data root: {args.data_root}")
        print(f"run root: {args.run_root}")
        print(
            "dry-run only: "
            + shlex.join(
                [
                    f"RQL_DATA_ROOT={args.data_root}",
                    f"RQL_RUN_ROOT={args.run_root}",
                    "sbatch",
                    str(args.slurm),
                ]
            )
        )
        if args.stage_data:
            print(
                "staged dry-run: submit stage_data.slurm first, then add "
                "--dependency=afterok:<DATA_JOB_ID> to the training sbatch"
            )
        elif args.data_job_id is not None:
            print(f"staged dry-run: training array depends afterok on existing job {args.data_job_id}")
        if not args.require_data:
            print("data availability was not required; add --require-data before formal submission")
        return 0
    except (ValidationError, ValueError, OSError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
