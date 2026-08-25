#!/usr/bin/env python3
"""Validate and submit calibration -> pilot gate -> 1M TreeWM-v2 -> report."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import netrc
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from campaign import (
    EXPECTED_ALLOCATION_SHARDS,
    EXPECTED_WORKERS,
    PROTOCOL_SOURCE_FILES,
    expand_runs,
    load_data_contract,
    load_manifest,
    protocol_sha256,
    required_dataset_files,
    trainer_command,
)
from treewm.utils.provenance import trainer_code_fingerprint


FORMAL_PYTHON = Path(
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/envs/treewm-formal-py311/bin/python"
)
FORMAL_DATA_ROOT = Path(
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/datasets/ogbench-rql-50task"
)
APPROVED_RAW_CACHE_ROOT = Path(
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/datasets/treewm-50task-full-cache-v1"
)
REQUIRED_TRAIN_SBATCH = (
    "#SBATCH --partition=polar4,polar3,polar,grizzly",
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
REQUIRED_SINGLE_GPU_ARRAY = (
    "#SBATCH --partition=polar4,polar3,polar,grizzly",
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
SECRET_PATTERNS = (
    re.compile(r"wandb_v1_[A-Za-z0-9_-]{20,}"),
    re.compile(r"hf_[A-Za-z0-9]{20,}"),
)
SENSITIVE_ENV_KEY_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
SOURCE_SNAPSHOT_MARKER = ".treewm-formal-source-snapshot.json"


class ValidationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValidationError(message)


def _canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"),
            ensure_ascii=True, allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _snapshot_source_paths(repo_root: Path) -> list[Path]:
    """Exact executable source inventory; never includes outputs, data, or credentials."""
    root = repo_root.resolve()
    campaign_dir = root / "experiments" / "12-treewm-formal-v2"
    candidates = [
        *(root / "treewm").rglob("*.py"),
        *(root / "configs").rglob("*.yaml"),
        root / "scripts" / "train.py",
        root / "scripts" / "__init__.py",
        campaign_dir / "manifest.json",
        campaign_dir / "protocol.sha256",
        *(campaign_dir / relative for relative in PROTOCOL_SOURCE_FILES),
    ]
    checked: set[Path] = set()
    for candidate in candidates:
        require(
            candidate.is_file() and not candidate.is_symlink(),
            f"snapshot source missing/symlinked: {candidate}",
        )
        path = candidate.resolve()
        require(path.is_relative_to(root), f"snapshot source escapes repository: {path}")
        checked.add(path)
    return sorted(checked)


def _source_hashes(repo_root: Path) -> dict[str, str]:
    root = repo_root.resolve()
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in _snapshot_source_paths(root)
    }


def _snapshot_identity(
    repo_root: Path, manifest: Mapping[str, Any], *, created_unix_time: float,
) -> dict[str, Any]:
    root = repo_root.resolve()
    campaign_dir = root / "experiments" / "12-treewm-formal-v2"
    code = trainer_code_fingerprint(root)
    source_hashes = _source_hashes(root)
    identity: dict[str, Any] = {
        "schema_version": 2,
        "status": "immutable_source_snapshot",
        "campaign_id": manifest["campaign_id"],
        "objective_version": manifest["method"]["objective_version"],
        "protocol_sha256": protocol_sha256(manifest, campaign_dir=campaign_dir),
        "protocol_lock": (campaign_dir / "protocol.sha256").read_text(
            encoding="utf-8"
        ).strip(),
        "code_sha256": code["manifest_sha256"],
        "source_files": source_hashes,
        "source_file_count": len(source_hashes),
        "slurm_log_dir": str(Path(manifest["paths"]["run_root"]) / "logs"),
        "created_unix_time": float(created_unix_time),
        "policy": "read-only protocol-keyed source; no live-worktree dependency",
    }
    identity["snapshot_identity_sha256"] = _canonical_hash(identity)
    return identity


def verify_source_snapshot(repo_root: Path) -> dict[str, Any]:
    """Fail closed if an allocation's protocol-keyed source snapshot changed."""
    root = repo_root.resolve()
    marker_path = root / SOURCE_SNAPSHOT_MARKER
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"immutable source snapshot marker is invalid: {exc}") from exc
    claimed = marker.get("snapshot_identity_sha256")
    body = dict(marker)
    body.pop("snapshot_identity_sha256", None)
    require(
        isinstance(claimed, str) and claimed == _canonical_hash(body),
        "source snapshot marker hash differs",
    )
    manifest = load_manifest(
        root / "experiments" / "12-treewm-formal-v2" / "manifest.json"
    )
    expected = _snapshot_identity(
        root, manifest, created_unix_time=float(marker.get("created_unix_time", -1.0))
    )
    require(marker == expected, "source snapshot content or protocol identity changed")
    require(
        marker["protocol_lock"] == marker["protocol_sha256"],
        "source snapshot protocol lock is stale",
    )
    log_link = root / "logs"
    require(log_link.is_symlink(), "source snapshot Slurm log path is not a symlink")
    require(
        log_link.resolve() == Path(marker["slurm_log_dir"]).resolve(),
        "source snapshot Slurm log target changed",
    )
    require(
        log_link.resolve().is_dir() and os.access(log_link.resolve(), os.W_OK),
        "source snapshot Slurm log target is not writable",
    )
    for relative in marker["source_files"]:
        mode = (root / relative).stat().st_mode
        require(mode & 0o222 == 0, f"source snapshot file is writable: {relative}")
    return marker


def prepare_source_snapshot(
    args: argparse.Namespace, manifest: Mapping[str, Any]
) -> Path:
    """Atomically publish and verify the one source tree used by every dependency."""
    protocol = protocol_sha256(
        manifest, campaign_dir=args.repo_root / "experiments" / "12-treewm-formal-v2"
    )
    require(
        args.protocol_lock.read_text(encoding="utf-8").strip() == protocol,
        "cannot snapshot an unlocked protocol",
    )
    live_code = trainer_code_fingerprint(args.repo_root)["manifest_sha256"]
    parent = args.run_root / "state" / "source-snapshots" / protocol
    destination = parent / "repo"
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / ".create.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if destination.exists():
            marker = verify_source_snapshot(destination)
            require(marker["protocol_sha256"] == protocol, "snapshot protocol path differs")
            require(marker["code_sha256"] == live_code, "existing snapshot differs from locked live source")
            return destination
        temporary = parent / f".repo.tmp.{os.getpid()}"
        if temporary.exists():
            shutil.rmtree(temporary)
        temporary.mkdir(parents=True)
        try:
            for source in _snapshot_source_paths(args.repo_root):
                relative = source.relative_to(args.repo_root)
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
            marker = _snapshot_identity(
                temporary, manifest, created_unix_time=time.time()
            )
            require(marker["protocol_sha256"] == protocol, "copied snapshot protocol drifted")
            require(marker["code_sha256"] == live_code, "copied snapshot trainer source drifted")
            _atomic_json(temporary / SOURCE_SNAPSHOT_MARKER, marker)
            logs_target = args.run_root / "logs"
            logs_target.mkdir(parents=True, exist_ok=True)
            os.symlink(logs_target, temporary / "logs", target_is_directory=True)
            for path in temporary.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            for directory in sorted(
                (
                    path for path in temporary.rglob("*")
                    if path.is_dir() and not path.is_symlink()
                ),
                key=lambda path: len(path.parts), reverse=True,
            ):
                directory.chmod(
                    stat.S_IRUSR | stat.S_IXUSR
                    | stat.S_IRGRP | stat.S_IXGRP
                    | stat.S_IROTH | stat.S_IXOTH
                )
            temporary.chmod(
                stat.S_IRUSR | stat.S_IXUSR
                | stat.S_IRGRP | stat.S_IXGRP
                | stat.S_IROTH | stat.S_IXOTH
            )
            os.replace(temporary, destination)
            fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception:
            if temporary.exists():
                for path in temporary.rglob("*"):
                    if path.is_dir() and not path.is_symlink():
                        path.chmod(stat.S_IRWXU)
                    elif path.is_file() and not path.is_symlink():
                        path.chmod(stat.S_IRUSR | stat.S_IWUSR)
                temporary.chmod(stat.S_IRWXU)
                shutil.rmtree(temporary)
            raise
    verify_source_snapshot(destination)
    return destination


def _exact_lines(path: Path, required: Sequence[str], label: str) -> str:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for line in required:
        require(lines.count(line) == 1, f"{label} must contain exactly one `{line}`")
    return text


def validate_slurm(
    train: Path, stage: Path, calibration_gate: Path, pilot: Path,
    pilot_gate: Path, report: Path,
) -> None:
    train_text = _exact_lines(train, REQUIRED_TRAIN_SBATCH, "formal training Slurm")
    for snippet in (
        'REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"',
        "--ntasks=16", "--ntasks-per-node=8", "--gpus-per-task=1",
        "--gpu-bind=single:1", "--kill-on-bad-exit=0", "dispatcher.py",
        "--allocation-shard", "--allocation-shards", "--resolution-status",
        "CANCEL_REQUESTED", "FALLBACK_REQUEUE_CALLING.json",
        "BATCH_REQUEUE_CALLING.json", 'scontrol requeue "$REQUEUE_TARGET"',
        "gpu_preflight.py",
    ):
        require(snippet in train_text, f"formal training Slurm lacks {snippet}")
    require("scripts/train.py" not in train_text, "batch shell must use the guarded dispatcher")
    stage_text = _exact_lines(stage, REQUIRED_SINGLE_GPU_ARRAY, "calibration stage Slurm")
    for snippet in ("prepare_cache.py", "REQUEUE_CALLING.json", "CANCEL_REQUESTED", 'scontrol requeue "$REQUEUE_TARGET"'):
        require(snippet in stage_text, f"stage Slurm lacks {snippet}")
    calibration_gate_text = calibration_gate.read_text(encoding="utf-8")
    require("--phase validate-all" in calibration_gate_text, "calibration gate does not validate all settings")
    pilot_text = _exact_lines(pilot, REQUIRED_SINGLE_GPU_ARRAY, "pilot Slurm")
    for snippet in ("gpu_preflight.py", "validate_pilot.py", "--audit-setting-index", "REQUEUE_CALLING.json", "CANCEL_REQUESTED", 'scontrol requeue "$REQUEUE_TARGET"'):
        require(snippet in pilot_text, f"pilot Slurm lacks {snippet}")
    require("--validate-all" in pilot_gate.read_text(encoding="utf-8"), "pilot gate does not validate all settings")
    require("aggregate.py" in report.read_text(encoding="utf-8"), "report Slurm does not invoke strict aggregation")
    for path in (train, stage, calibration_gate, pilot, pilot_gate, report):
        text = path.read_text(encoding="utf-8")
        require(
            "--verify-source-snapshot" in text,
            f"{path.name} does not verify the immutable source snapshot",
        )
        require(
            "PYTHONDONTWRITEBYTECODE=1" in text,
            f"{path.name} may write bytecode into the immutable source snapshot",
        )


def scan_for_embedded_secrets(paths: Sequence[Path]) -> None:
    suffixes = {".py", ".json", ".md", ".sh", ".slurm", ".yaml", ".yml", ".txt"}
    for root in paths:
        candidates = [root] if root.is_file() else list(root.rglob("*"))
        for path in candidates:
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in SECRET_PATTERNS:
                require(pattern.search(text) is None, f"credential-like string embedded in {path}")


def scrub_sensitive_environment(environment: Mapping[str, str]) -> dict[str, str]:
    return {
        str(key): str(value) for key, value in environment.items()
        if not any(part in str(key).upper() for part in SENSITIVE_ENV_KEY_PARTS)
    }


def wandb_credentials_available() -> bool:
    try:
        credentials = netrc.netrc().authenticators("api.wandb.ai")
    except (FileNotFoundError, netrc.NetrcParseError, OSError):
        return False
    return bool(credentials and credentials[2])


def validate_runtime(python: Path) -> None:
    require(python.is_file() and os.access(python, os.X_OK), f"formal Python is not executable: {python}")
    result = subprocess.run(
        [str(python), "-c", "import hydra,numpy,ogbench,torch,wandb; print('runtime-ok')"],
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env={**scrub_sensitive_environment(os.environ), "PYTHONNOUSERSITE": "1"},
    )
    require(result.returncode == 0, f"formal runtime import probe failed: {result.stdout[-2000:]}")


def validate_source_data(manifest: Mapping[str, Any], data_root: Path) -> int:
    paths: list[Path] = []
    for setting in manifest["settings"]:
        paths.extend(required_dataset_files(manifest, data_root, setting))
    require(len(paths) == 416 and len(set(paths)) == 416, "source inventory must be exactly 416 files")
    missing = [path for path in paths if not path.is_file() or path.stat().st_size <= 0]
    if missing:
        raise ValidationError(f"source inventory has {len(missing)} missing/empty files: {missing[:5]}")
    return len(paths)


def validate_contracts(
    manifest: Mapping[str, Any], data_root: Path, cache_root: Path
) -> tuple[int, list[str]]:
    count = 0
    errors: list[str] = []
    for setting in manifest["settings"]:
        try:
            contract = load_data_contract(
                manifest, setting, data_root=data_root, cache_root=cache_root
            )
            for key in ("data_manifest_sha256", "calibration_sha256", "future_recipe_sha256"):
                value = str(contract.get(key, ""))
                require(len(value) == 64 and all(ch in "0123456789abcdef" for ch in value), f"{setting['id']}: malformed {key}")
            count += 1
        except (ValueError, ValidationError) as exc:
            errors.append(str(exc))
    return count, errors


def validate_mapping_and_commands(
    args: argparse.Namespace, manifest: Mapping[str, Any], *, contracts_complete: bool
) -> None:
    runs = expand_runs(manifest)
    require(len(runs) == 40, "formal campaign must expand to 40 models")
    owned = {
        shard * EXPECTED_WORKERS + rank
        for shard in range(EXPECTED_ALLOCATION_SHARDS)
        for rank in range(EXPECTED_WORKERS)
        if shard * EXPECTED_WORKERS + rank < len(runs)
    }
    require(owned == set(range(40)), "array ownership has a gap")
    require([16, 16, 8] == [sum(16 * shard + rank < 40 for rank in range(16)) for shard in range(3)], "array shard sizes drifted")
    if not contracts_complete:
        return
    for run in runs:
        command, environment = trainer_command(
            manifest, run, python_executable=str(args.python), repo_root=args.repo_root,
            run_root=args.run_root, data_root=args.data_root, cache_root=args.cache_root,
            wandb_project=manifest["logging"]["wandb_project"], wandb_mode="online",
        )
        command = [str(value) for value in command]
        joined = " ".join(command).lower()
        for token in (
            "experiment=treewm_v2", "objective_version=treewm_v2_rms_rank_v1",
            "arm=treewm", "train.steps=1000000", "train.gradient_checkpointing=true",
            "tree.scorer=learned", "retrieval.enabled=false",
            "eval.final_episodes_per_task=50",
        ):
            require(token in joined, f"{run.run_id}: launch lacks {token}")
        require(command[0] == str(args.python), "formal interpreter path was changed")
        require(environment.get("WANDB_RUN_ID") == run.wandb_id, f"{run.run_id}: unstable formal W&B ID")
        require(environment.get("WANDB_PROJECT") == manifest["logging"]["wandb_project"], "formal W&B project drift")
        require(environment.get("WANDB_RUN_GROUP") == manifest["logging"]["wandb_group"], "formal W&B group drift")
        for key in (
            "TREEWM_DATA_SHA256", "TREEWM_CALIBRATION_SHA256",
            "TREEWM_FUTURE_RECIPE_SHA256", "TREEWM_DATA_CONTRACT_SHA256",
            "TREEWM_CODE_SHA256", "TREEWM_RUNTIME_SHA256", "TREEWM_PROTOCOL_SHA256",
        ):
            value = str(environment.get(key, ""))
            require(len(value) == 64 and all(ch in "0123456789abcdef" for ch in value), f"{run.run_id}: malformed {key}")
        require(not any("KEY" in key.upper() or "TOKEN" in key.upper() for key in environment), "trainer environment contains a credential")
        require("v1" not in environment.get("WANDB_RUN_GROUP", "").lower(), "formal W&B namespace references v1")


def validate_fresh_v2_paths(args: argparse.Namespace, manifest: Mapping[str, Any]) -> None:
    require(args.run_root == Path(manifest["paths"]["run_root"]).resolve(), "formal run root differs from manifest")
    require(args.pilot_root == Path(manifest["paths"]["pilot_run_root"]).resolve(), "pilot root differs from manifest")
    require(args.contract_root == Path(manifest["paths"]["contract_root"]).resolve(), "contract root differs from manifest")
    for label, path in (("formal", args.run_root), ("pilot", args.pilot_root)):
        require("v1" not in str(path).lower(), f"{label} v2 path references v1")
    # The suffix is the schema/version of the *new v2 contract format*, not the old
    # TreeWM-v1 campaign.  It is deliberately separate from the approved raw-cache-v1
    # path and from every v1 run/W&B namespace.
    require(
        "treewm-50task-formal-v2-contracts-" in str(args.contract_root).lower(),
        "v2 contract root is not explicitly namespaced",
    )
    require(args.cache_root == APPROVED_RAW_CACHE_ROOT.resolve(), "only the approved raw cache may be reused")
    require(args.contract_root != args.cache_root, "v2 contracts may not overwrite the raw cache")
    for group_key in ("wandb_group", "pilot_wandb_group"):
        require("v1" not in manifest["logging"][group_key].lower(), f"{group_key} references v1")


def _parse_job_id(output: str, label: str) -> str:
    job_id = output.strip().split(";", 1)[0]
    require(job_id.isdigit(), f"could not parse {label} ID from {output!r}")
    return job_id


def _run_sbatch(
    argv: list[str], *, cwd: Path, environment: Mapping[str, str], label: str,
    test_only: bool = False,
) -> str:
    command = ["sbatch", "--test-only"] if test_only else ["sbatch", "--parsable"]
    result = subprocess.run(
        [*command, *argv], cwd=cwd, env=scrub_sensitive_environment(environment),
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if result.returncode != 0:
        raise ValidationError(f"{label} sbatch failed: {result.stdout}")
    return result.stdout


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    repo_root = here.parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--submit", action="store_true")
    mode.add_argument("--verify-source-snapshot", action="store_true")
    dependency = parser.add_mutually_exclusive_group()
    dependency.add_argument("--stage-data", action="store_true")
    dependency.add_argument("--data-job-id")
    parser.add_argument("--scheduler-test", action="store_true")
    parser.add_argument("--manifest", type=Path, default=here / "manifest.json")
    parser.add_argument("--protocol-lock", type=Path, default=here / "protocol.sha256")
    parser.add_argument("--train-slurm", type=Path, default=here / "train.slurm")
    parser.add_argument("--stage-slurm", type=Path, default=here / "stage_data.slurm")
    parser.add_argument("--calibration-gate-slurm", type=Path, default=here / "calibration_gate.slurm")
    parser.add_argument("--pilot-slurm", type=Path, default=here / "pilot.slurm")
    parser.add_argument("--pilot-gate-slurm", type=Path, default=here / "pilot_gate.slurm")
    parser.add_argument("--report-slurm", type=Path, default=here / "aggregate.slurm")
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--python", type=Path, default=Path(os.environ.get("TREEWM_PYTHON", FORMAL_PYTHON)))
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("TREEWM_DATA_ROOT", FORMAL_DATA_ROOT)))
    parser.add_argument("--cache-root", type=Path, default=Path(os.environ.get("TREEWM_CACHE", APPROVED_RAW_CACHE_ROOT)))
    manifest_preview = json.loads((here / "manifest.json").read_text(encoding="utf-8"))
    parser.add_argument("--contract-root", type=Path, default=Path(os.environ.get("TREEWM_CONTRACT_ROOT", manifest_preview["paths"]["contract_root"])))
    parser.add_argument("--run-root", type=Path, default=Path(os.environ.get("TREEWM_RUN_ROOT", manifest_preview["paths"]["run_root"])))
    parser.add_argument("--pilot-root", type=Path, default=Path(os.environ.get("TREEWM_PILOT_ROOT", manifest_preview["paths"]["pilot_run_root"])))
    args = parser.parse_args(argv)
    if not args.submit and not args.verify_source_snapshot:
        args.dry_run = True
    if args.data_job_id is not None and not args.data_job_id.isdigit():
        parser.error("--data-job-id must be numeric")
    args.python = Path(os.path.abspath(os.fspath(args.python.expanduser())))
    for name in ("manifest", "protocol_lock", "train_slurm", "stage_slurm", "calibration_gate_slurm", "pilot_slurm", "pilot_gate_slurm", "report_slurm", "repo_root", "data_root", "cache_root", "contract_root", "run_root", "pilot_root"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    return args


def validate(args: argparse.Namespace) -> tuple[dict[str, Any], int, bool]:
    manifest = load_manifest(args.manifest)
    require(args.repo_root == Path(__file__).resolve().parents[2], "repository root differs from campaign location")
    require(args.python == FORMAL_PYTHON, "formal Python lexical path is immutable")
    require(args.data_root == FORMAL_DATA_ROOT.resolve(), "formal source root is immutable")
    require(manifest["paths"]["python"] == str(args.python), "manifest Python path drifted")
    validate_fresh_v2_paths(args, manifest)
    locked = args.protocol_lock.read_text(encoding="utf-8").strip()
    require(locked == protocol_sha256(manifest), "protocol lock differs from executable campaign")
    validate_slurm(
        args.train_slurm, args.stage_slurm, args.calibration_gate_slurm,
        args.pilot_slurm, args.pilot_gate_slurm, args.report_slurm,
    )
    scan_for_embedded_secrets([Path(__file__).resolve().parent, args.repo_root / "configs"])
    validate_runtime(args.python)
    source_count = validate_source_data(manifest, args.data_root)
    valid_contracts, errors = validate_contracts(manifest, args.data_root, args.cache_root)
    contracts_complete = valid_contracts == len(manifest["settings"])
    require(contracts_complete or args.stage_data or args.data_job_id is not None,
            f"only {valid_contracts}/10 v2 contracts are ready; stage dependency required")
    validate_mapping_and_commands(args, manifest, contracts_complete=contracts_complete)
    if errors and (args.stage_data or args.data_job_id):
        print(f"contracts ready {valid_contracts}/10; remaining work is behind stage afterok", file=sys.stderr)
    return manifest, source_count, contracts_complete


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.verify_source_snapshot:
            marker = verify_source_snapshot(args.repo_root)
            print(
                "immutable source snapshot: OK "
                f"protocol={marker['protocol_sha256']} code={marker['code_sha256']}"
            )
            return 0
        manifest, source_count, contracts_complete = validate(args)
        if args.submit:
            require(args.stage_data or args.data_job_id is not None, "submission requires an explicit stage dependency")
            require(wandb_credentials_available(), "W&B credentials must be available through ~/.netrc")
        environment = scrub_sensitive_environment(os.environ)
        environment.update({
            "TREEWM_PYTHON": str(args.python), "TREEWM_DATA_ROOT": str(args.data_root),
            "TREEWM_CACHE": str(args.cache_root), "TREEWM_CONTRACT_ROOT": str(args.contract_root),
            "TREEWM_RUN_ROOT": str(args.run_root), "TREEWM_PILOT_ROOT": str(args.pilot_root),
            "WANDB_PROJECT": manifest["logging"]["wandb_project"], "WANDB_MODE": "online",
            "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        })
        (args.repo_root / "logs").mkdir(parents=True, exist_ok=True)
        args.contract_root.mkdir(parents=True, exist_ok=True)
        args.run_root.mkdir(parents=True, exist_ok=True)
        args.pilot_root.mkdir(parents=True, exist_ok=True)
        launch_root = args.repo_root
        if args.submit:
            launch_root = prepare_source_snapshot(args, manifest)
            environment["TREEWM_SOURCE_SNAPSHOT"] = str(launch_root)

        def launch_path(path: Path) -> Path:
            return launch_root / path.relative_to(args.repo_root)

        stage_slurm = launch_path(args.stage_slurm)
        calibration_gate_slurm = launch_path(args.calibration_gate_slurm)
        pilot_slurm = launch_path(args.pilot_slurm)
        pilot_gate_slurm = launch_path(args.pilot_gate_slurm)
        train_slurm = launch_path(args.train_slurm)
        report_slurm = launch_path(args.report_slurm)
        scripts = (
            (stage_slurm, "calibration stage"),
            (calibration_gate_slurm, "calibration gate"),
            (pilot_slurm, "pilot"),
            (pilot_gate_slurm, "pilot gate"), (train_slurm, "formal training"),
            (report_slurm, "report"),
        )
        if args.scheduler_test:
            for script, label in scripts:
                _run_sbatch([str(script)], cwd=launch_root, environment=environment, label=f"{label} test", test_only=True)
            _run_sbatch(
                ["--export=ALL,TREEWM_STAGE_PHASE=recipe", str(stage_slurm)],
                cwd=launch_root, environment=environment,
                label="recipe stage test", test_only=True,
            )
            print("scheduler validation: all seven sbatch --test-only checks accepted")
        if not args.submit:
            print("validation: OK")
            print("campaign: 10 five-thousand-step pilots -> 40 TreeWM-v2 models -> 10,000 final episodes")
            print("formal parallelism: three 2-node allocations, shard sizes 16, 16, 8")
            print(f"source inventory: {source_count}/416; contracts complete: {contracts_complete}")
            print(f"formal root: {args.run_root}")
            print(f"pilot root: {args.pilot_root}")
            print("lifecycle: stage -> pilot array -> gradient/health gate -> 1M array -> report")
            return 0

        if args.stage_data:
            stage_output = _run_sbatch([str(stage_slurm)], cwd=launch_root, environment=environment, label="v2 calibration/recipe stage")
            stage_job = _parse_job_id(stage_output, "stage")
            print(f"submitted ten-setting resumable train-only calibration array {stage_job}")
            calibration_gate_output = _run_sbatch(
                [f"--dependency=afterok:{stage_job}", str(calibration_gate_slurm)],
                cwd=launch_root, environment=environment,
                label="all-setting calibration gate",
            )
            calibration_gate_job = _parse_job_id(calibration_gate_output, "calibration gate")
            print(f"submitted all-setting calibration gate {calibration_gate_job}")
            recipe_output = _run_sbatch(
                [f"--dependency=afterok:{calibration_gate_job}",
                 "--export=ALL,TREEWM_STAGE_PHASE=recipe", str(stage_slurm)],
                cwd=launch_root, environment=environment,
                label="compact future-recipe array",
            )
            dependency_job = _parse_job_id(recipe_output, "recipe array")
            print(f"submitted ten-setting resumable compact recipe array {dependency_job}")
        else:
            dependency_job = str(args.data_job_id)
        pilot_output = _run_sbatch(
            [f"--dependency=afterok:{dependency_job}", str(pilot_slurm)],
            cwd=launch_root, environment=environment, label="v2 pilot array",
        )
        pilot_job = _parse_job_id(pilot_output, "pilot array")
        print(f"submitted ten-setting 5k-step TreeWM-v2 pilot array {pilot_job}")
        gate_output = _run_sbatch(
            [f"--dependency=afterok:{pilot_job}", str(pilot_gate_slurm)],
            cwd=launch_root, environment=environment, label="pilot acceptance gate",
        )
        gate_job = _parse_job_id(gate_output, "pilot gate")
        print(f"submitted strict pilot gradient/health gate {gate_job}")
        train_output = _run_sbatch(
            [f"--dependency=afterok:{gate_job}", str(train_slurm)],
            cwd=launch_root, environment=environment, label="formal 1M array",
        )
        train_job = _parse_job_id(train_output, "formal training array")
        print(f"submitted three-element 1M TreeWM-v2 array {train_job}")
        report_output = _run_sbatch(
            [f"--dependency=afterok:{train_job}", str(report_slurm)],
            cwd=launch_root, environment=environment, label="strict report",
        )
        report_job = _parse_job_id(report_output, "report")
        print(f"submitted strict afterok report {report_job}")
        return 0
    except (ValidationError, ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"validation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
