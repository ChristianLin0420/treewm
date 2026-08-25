#!/usr/bin/env python3
"""Validate and submit the formal TreeWM cache/train/report lifecycle.

Dry-run is the default. Credentials are never accepted as arguments, persisted, or
printed: formal jobs authenticate through the submitting account's ``~/.netrc``.
Unrelated credential-bearing environment variables are removed before every
``sbatch`` call. Formal submission always attaches an ``afterok`` data dependency
and chains the strict report after the complete three-element training array.
"""

from __future__ import annotations

import argparse
import inspect
import netrc
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from campaign import (
    ALLOCATION_SHARDS,
    FORMAL_TASK_IDS,
    WORKERS_PER_ALLOCATION,
    expand_runs,
    load_data_contract,
    load_manifest,
    protocol_sha256,
    required_dataset_files,
    run_for_worker,
    trainer_command,
)


FORMAL_PYTHON = Path(
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/envs/treewm-formal-py311/bin/python"
)
FORMAL_DATA_ROOT = Path(
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/datasets/ogbench-rql-50task"
)
FORMAL_CACHE_ROOT = Path(
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/datasets/treewm-50task-full-cache-v1"
)
REQUIRED_TRAIN_SBATCH = (
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
    "#SBATCH --array=0-2%3",
    "#SBATCH --output=logs/%x_%j.out",
)
REQUIRED_TRAIN_SRUN = (
    "--ntasks=16",
    "--ntasks-per-node=8",
    "--gpus-per-task=1",
    "--gpu-bind=single:1",
    "--kill-on-bad-exit=0",
    "dispatcher.py",
    "--allocation-shard",
    "--allocation-shards",
)
REQUIRED_STAGE_SBATCH = (
    "#SBATCH --partition=polar4,polar3,polar,grizzly",
    "#SBATCH --account=edgeai_tao-ptm_image-foundation-model-clip",
    "#SBATCH --qos=normal",
    "#SBATCH --time=04:00:00",
    "#SBATCH --requeue",
    "#SBATCH --signal=USR1@420",
    "#SBATCH --nodes=1",
    "#SBATCH --ntasks-per-node=1",
    "#SBATCH --gpus-per-node=1",
    "#SBATCH --cpus-per-task=12",
    "#SBATCH --array=0-9%10",
    "#SBATCH --output=logs/%x_%j.out",
)
REQUIRED_REPORT_SBATCH = (
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
SECRET_PATTERNS = (
    re.compile(r"wandb_v1_[A-Za-z0-9_-]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
)
SENSITIVE_ENV_KEY_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")


class ValidationError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _validate_exact_lines(path: Path, required: Sequence[str], label: str) -> str:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for line in required:
        _require(lines.count(line) == 1, f"{label} must contain exactly one `{line}`")
    return text


def validate_slurm(train: Path, stage: Path, report: Path) -> None:
    train_text = _validate_exact_lines(train, REQUIRED_TRAIN_SBATCH, "training Slurm")
    for snippet in REQUIRED_TRAIN_SRUN:
        _require(snippet in train_text, f"training Slurm is missing {snippet}")
    for snippet in (
        'REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"',
        "--resolution-status",
        "CANCEL_REQUESTED",
        "FALLBACK_REQUEUE_CALLING.json",
        "BATCH_REQUEUE_CALLING.json",
        'scontrol requeue "$REQUEUE_TARGET"',
        "gpu_preflight.py",
    ):
        _require(snippet in train_text, f"training Slurm is missing lifecycle guard {snippet}")
    _require("scripts/train.py" not in train_text, "batch shell must launch TreeWM only via dispatcher")

    stage_text = _validate_exact_lines(stage, REQUIRED_STAGE_SBATCH, "cache-stage Slurm")
    for snippet in (
        "prepare_cache.py",
        'scontrol requeue "$REQUEUE_TARGET"',
        "CANCEL_REQUESTED",
        "REQUEUE_CALLING.json",
    ):
        _require(snippet in stage_text, f"cache-stage Slurm is missing {snippet}")
    report_text = _validate_exact_lines(report, REQUIRED_REPORT_SBATCH, "report Slurm")
    _require("aggregate.py" in report_text, "report Slurm does not invoke the strict reporter")


def validate_trainer_interface(repo_root: Path) -> None:
    train_path = repo_root / "scripts" / "train.py"
    text = train_path.read_text(encoding="utf-8")
    for token in (
        "COMPLETED.json",
        "final_eval_progress.json",
        "GRACEFUL_EXIT_CODE",
        "TREEWM_PROTOCOL_SHA256",
        "TREEWM_CODE_SHA256",
        "TREEWM_RUNTIME_SHA256",
        "TREEWM_DATA_SHA256",
        "WANDB_RUN_ID",
        "model.set_gradient_checkpointing",
        'str(cfg.arm) != "treewm"',
        'model.__class__.__name__ != "TreeWM"',
        'str(tree_cfg.scorer) != "learned"',
    ):
        _require(token in text, f"formal trainer interface is missing {token!r}")


def scan_for_embedded_secrets(paths: Sequence[Path]) -> None:
    suffixes = {".py", ".json", ".md", ".sh", ".slurm", ".yaml", ".yml", ".txt"}
    for root in paths:
        candidates = [root] if root.is_file() else list(root.rglob("*"))
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in SECRET_PATTERNS:
                _require(pattern.search(text) is None, f"credential-like string embedded in {path}")


def validate_source_data(manifest: Mapping[str, Any], data_root: Path) -> int:
    paths: list[Path] = []
    for setting in manifest["settings"]:
        paths.extend(required_dataset_files(manifest, data_root, setting))
    _require(len(paths) == 416 and len(set(paths)) == 416, "formal source inventory must be 416 files")
    missing = [path for path in paths if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        preview = "\n  ".join(str(path) for path in missing[:20])
        raise ValidationError(f"source inventory has {len(missing)} missing/empty files:\n  {preview}")
    return len(paths)


def validate_cache_contracts(
    manifest: Mapping[str, Any], data_root: Path, cache_root: Path
) -> tuple[int, list[str]]:
    valid = 0
    errors: list[str] = []
    for setting in manifest["settings"]:
        try:
            load_data_contract(
                manifest, setting, data_root=data_root, cache_root=cache_root
            )
            valid += 1
        except ValueError as exc:
            errors.append(str(exc))
    return valid, errors


def validate_mapping_and_commands(
    args: argparse.Namespace, manifest: Mapping[str, Any], *, contracts_complete: bool
) -> None:
    runs = expand_runs(manifest)
    _require(len(runs) == 40, "formal campaign must contain 40 model runs")
    owned: dict[int, tuple[int, int]] = {}
    active_counts: list[int] = []
    for shard in range(ALLOCATION_SHARDS):
        active = 0
        for rank in range(WORKERS_PER_ALLOCATION):
            run = run_for_worker(manifest, shard, rank)
            if run is not None:
                _require(run.index not in owned, f"run {run.index} has duplicate ownership")
                _require(run.index == 16 * shard + rank, "array ownership formula drifted")
                owned[run.index] = (shard, rank)
                active += 1
        active_counts.append(active)
    _require(sorted(owned) == list(range(40)), "array mapping has a gap")
    _require(active_counts == [16, 16, 8], f"unexpected array shard sizes {active_counts}")

    if contracts_complete:
        commands = [
            trainer_command(
                manifest,
                run,
                python_executable=args.python,
                repo_root=args.repo_root,
                run_root=args.run_root,
                data_root=args.data_root,
                cache_root=args.cache_root,
            )
            for run in runs
        ]
        for run, (argv, environment) in zip(runs, commands, strict=True):
            joined = " ".join(argv).lower()
            _require(argv[0] == str(args.python), "formal virtual-environment path was canonicalized")
            _require(argv[1] == str(args.repo_root / "scripts" / "train.py"), "wrong trainer path")
            _require("arm=treewm" in argv, f"{run.run_id}: arm is not TreeWM")
            _require("train.steps=1000000" in argv, f"{run.run_id}: not exactly 1M updates")
            _require("train.gradient_checkpointing=true" in argv, f"{run.run_id}: remat not pinned")
            _require("eval.final_episodes_per_task=50" in argv, f"{run.run_id}: final50 not pinned")
            _require("tree.scorer=learned" in argv, f"{run.run_id}: learned scorer not pinned")
            _require("upstream_" not in joined and "agents/rql" not in joined, "wrong trainer family")
            _require(environment.get("WANDB_RUN_ID") == run.wandb_id, "unstable W&B ID")
            _require(environment.get("WANDB_PROJECT") == "treewm-50task-formal", "W&B project drift")
            _require(not any("KEY" in key or "TOKEN" in key for key in environment), "secret env key")
    else:
        # The data SHA is intentionally unavailable before the dependent cache stage.
        # Still prove that the shared command constructor explicitly pins every named
        # scientific invariant; dispatcher reconstructs and validates all argv at run time.
        source = inspect.getsource(trainer_command)
        for token in (
            'override("arm", method["arm"])',
            'override("train.steps", training["optimizer_updates"])',
            'override("train.gradient_checkpointing", method["gradient_checkpointing"])',
            'override("eval.final_episodes_per_task", evaluation["final_episodes_per_task"])',
            'override("tree.scorer", method["scorer"])',
            '"WANDB_RUN_ID": run.wandb_id',
        ):
            _require(token in source, f"deferred command constructor is missing {token}")


def wandb_credentials_available() -> bool:
    """Require the filesystem credential that remains available after env scrubbing."""

    try:
        credentials = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False
    return bool(credentials and credentials[2])


def scrub_sensitive_environment(environment: Mapping[str, str]) -> dict[str, str]:
    """Drop ambient credentials before Slurm captures the submission environment."""

    return {
        str(key): str(value)
        for key, value in environment.items()
        if not any(part in str(key).upper() for part in SENSITIVE_ENV_KEY_PARTS)
    }


def validate_runtime(python: Path) -> None:
    _require(python.is_file() and os.access(python, os.X_OK), f"formal Python is not executable: {python}")
    probe = subprocess.run(
        [
            str(python),
            "-c",
            "import os, sys, hydra, numpy, ogbench, torch, wandb; "
            "assert torch.__version__; "
            "assert os.path.abspath(sys.executable) == sys.argv[1], "
            "(sys.executable, sys.argv[1]); print('runtime-imports-ok')",
            str(python),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PYTHONNOUSERSITE": "1"},
    )
    _require(probe.returncode == 0, f"formal runtime import probe failed: {probe.stdout[-2000:]}")


def _parse_job_id(output: str, label: str) -> str:
    job_id = output.strip().split(";", 1)[0]
    _require(job_id.isdigit(), f"could not parse {label} job ID from {output!r}")
    return job_id


def _run_sbatch(
    argv: list[str], *, cwd: Path, environment: Mapping[str, str], label: str, test_only: bool = False
) -> str:
    command = ["sbatch", "--test-only"] if test_only else ["sbatch", "--parsable"]
    safe_environment = scrub_sensitive_environment(environment)
    result = subprocess.run(
        [*command, *argv],
        cwd=cwd,
        env=safe_environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise ValidationError(f"{label} sbatch failed: {result.stdout}")
    return result.stdout


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate without submitting (default)")
    mode.add_argument("--submit", action="store_true", help="submit cache/train/report lifecycle")
    dependency = parser.add_mutually_exclusive_group()
    dependency.add_argument("--stage-data", action="store_true", help="submit the ten-setting cache array")
    dependency.add_argument("--data-job-id", help="existing numeric successful/pending cache-stage job")
    parser.add_argument("--scheduler-test", action="store_true", help="run real sbatch --test-only checks")
    parser.add_argument("--manifest", type=Path, default=here / "manifest.json")
    parser.add_argument("--protocol-lock", type=Path, default=here / "protocol.sha256")
    parser.add_argument("--train-slurm", type=Path, default=here / "train.slurm")
    parser.add_argument("--stage-slurm", type=Path, default=here / "stage_data.slurm")
    parser.add_argument("--report-slurm", type=Path, default=here / "aggregate.slurm")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", type=Path, default=Path(os.environ.get("TREEWM_PYTHON", FORMAL_PYTHON)))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("TREEWM_DATA_ROOT", FORMAL_DATA_ROOT)))
    parser.add_argument("--cache-root", type=Path, default=Path(os.environ.get("TREEWM_CACHE", FORMAL_CACHE_ROOT)))
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("TREEWM_RUN_ROOT", repo_root / "outputs" / "treewm-50task-1m-v1")))
    args = parser.parse_args(argv)
    if not args.submit:
        args.dry_run = True
    if args.data_job_id is not None and not args.data_job_id.isdigit():
        parser.error("--data-job-id must be numeric")
    # The manifest intentionally locks the stable formal-environment symlink. Do
    # not canonicalize it to the environment's implementation target: doing so
    # both changes the recorded command and makes the exact manifest check fail.
    args.python = Path(os.path.abspath(os.fspath(args.python.expanduser())))
    for name in (
        "manifest",
        "protocol_lock",
        "train_slurm",
        "stage_slurm",
        "report_slurm",
        "repo_root",
        "data_root",
        "cache_root",
        "run_root",
    ):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def validate(args: argparse.Namespace) -> tuple[dict[str, Any], int, bool]:
    manifest = load_manifest(args.manifest)
    _require(args.python == FORMAL_PYTHON, "formal Python path is immutable")
    _require(args.data_root == FORMAL_DATA_ROOT.resolve(), "formal source data path is immutable")
    _require(args.cache_root == FORMAL_CACHE_ROOT.resolve(), "formal cache path is immutable")
    _require(args.run_root == Path(manifest["paths"]["run_root"]).resolve(), "formal run root is immutable")
    _require(args.repo_root == Path(__file__).resolve().parents[2], "repo root does not match campaign location")
    _require(manifest["paths"]["python"] == str(args.python), "manifest Python path drifted")
    _require(manifest["paths"]["data_root"] == str(args.data_root), "manifest data root drifted")
    _require(manifest["paths"]["cache_root"] == str(args.cache_root), "manifest cache root drifted")
    locked = args.protocol_lock.read_text(encoding="utf-8").strip()
    _require(
        locked == protocol_sha256(manifest),
        "protocol.sha256 does not match manifest plus executable launch sources",
    )
    validate_slurm(args.train_slurm, args.stage_slurm, args.report_slurm)
    validate_trainer_interface(args.repo_root)
    scan_for_embedded_secrets(
        [args.repo_root / "experiments" / "11-treewm-formal", args.repo_root / "configs" / "env"]
    )
    validate_runtime(args.python)
    source_count = validate_source_data(manifest, args.data_root)
    valid_contracts, cache_errors = validate_cache_contracts(manifest, args.data_root, args.cache_root)
    contracts_complete = valid_contracts == len(manifest["settings"])
    has_dependency = args.stage_data or args.data_job_id is not None
    _require(
        contracts_complete or has_dependency,
        f"only {valid_contracts}/10 cache contracts are valid; use --stage-data or --data-job-id",
    )
    validate_mapping_and_commands(args, manifest, contracts_complete=contracts_complete)
    if cache_errors and has_dependency:
        print(
            f"cache contracts: {valid_contracts}/10 ready; remaining contracts are deferred to afterok stage",
            file=sys.stderr,
        )
    return manifest, source_count, contracts_complete


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest, source_count, contracts_complete = validate(args)
        if args.submit:
            _require(
                args.stage_data or args.data_job_id is not None,
                "formal submission requires --stage-data or --data-job-id for an explicit afterok edge",
            )
            _require(
                wandb_credentials_available(),
                "W&B credentials unavailable (protected api.wandb.ai entry required in ~/.netrc)",
            )
        environment = scrub_sensitive_environment(os.environ)
        environment.update(
            {
                "TREEWM_PYTHON": str(args.python),
                "TREEWM_DATA_ROOT": str(args.data_root),
                "TREEWM_CACHE": str(args.cache_root),
                "TREEWM_RUN_ROOT": str(args.run_root),
                "WANDB_PROJECT": manifest["logging"]["wandb_project"],
                "WANDB_MODE": "online",
                "PYTHONNOUSERSITE": "1",
            }
        )
        (args.repo_root / "logs").mkdir(parents=True, exist_ok=True)
        args.run_root.mkdir(parents=True, exist_ok=True)

        if args.scheduler_test:
            _run_sbatch(
                [str(args.stage_slurm)],
                cwd=args.repo_root,
                environment=environment,
                label="cache-stage test",
                test_only=True,
            )
            _run_sbatch(
                [str(args.train_slurm)],
                cwd=args.repo_root,
                environment=environment,
                label="training test",
                test_only=True,
            )
            _run_sbatch(
                [str(args.report_slurm)],
                cwd=args.repo_root,
                environment=environment,
                label="report test",
                test_only=True,
            )
            print("scheduler validation: all three sbatch --test-only checks accepted")

        if not args.submit:
            print("validation: OK")
            print("campaign: 40 TreeWM models -> 200 task/seed cells -> 10,000 final episodes")
            print("parallelism: three 2-node allocations, shard sizes 16, 16, 8")
            print(f"source inventory: {source_count}/416 files present")
            print(f"cache contracts complete: {contracts_complete}")
            print(f"formal Python: {args.python}")
            print(f"data root: {args.data_root}")
            print(f"cache root: {args.cache_root}")
            print(f"run root: {args.run_root}")
            if args.stage_data:
                print("dry-run lifecycle: cache array -> afterok training array -> afterok report")
            elif args.data_job_id:
                print(f"dry-run lifecycle: afterok:{args.data_job_id} training array -> afterok report")
            else:
                print("cache already complete; add --stage-data or --data-job-id for formal submission")
            return 0

        dependency_job_id: str
        if args.stage_data:
            output = _run_sbatch(
                [str(args.stage_slurm)],
                cwd=args.repo_root,
                environment=environment,
                label="cache-stage",
            )
            dependency_job_id = _parse_job_id(output, "cache-stage")
            print(f"submitted ten-setting resumable cache array {dependency_job_id}")
        else:
            dependency_job_id = str(args.data_job_id)

        train_output = _run_sbatch(
            [f"--dependency=afterok:{dependency_job_id}", str(args.train_slurm)],
            cwd=args.repo_root,
            environment=environment,
            label="training",
        )
        train_job_id = _parse_job_id(train_output, "training array")
        print(f"submitted three-element TreeWM training array {train_job_id}")

        report_output = _run_sbatch(
            [f"--dependency=afterok:{train_job_id}", str(args.report_slurm)],
            cwd=args.repo_root,
            environment=environment,
            label="strict report",
        )
        report_job_id = _parse_job_id(report_output, "report")
        print(f"submitted strict afterok report {report_job_id}")
        return 0
    except (ValidationError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
