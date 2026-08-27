#!/usr/bin/env python3
"""Statically test, dry-run, or explicitly submit the sealed Exp20 pilot DAG."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

from campaign import (
    CAMPAIGN_DIR,
    CONTINUATION_RUNS,
    ContractError,
    PINNED_FORMAL_PYTHON,
    PROTOCOL_FILES,
    REPOSITORY_ROOT,
    RUNS,
    STAGE_TARGETS,
    atomic_json,
    continuation_runs,
    expand_runs,
    load_manifest,
    snapshot_identity_sha256,
    source_contract,
    stable_hash,
    trainer_command,
    verify_all,
    verify_protocol_lock,
    verify_source_snapshot,
)


SBATCH_JOB = re.compile(r"^(?P<job_id>[0-9]+)(?:;[A-Za-z0-9_.-]+)?$")
TRAIN_LINES = (
    "#SBATCH --time=04:00:00",
    "#SBATCH --requeue",
    "#SBATCH --signal=B:USR1@420",
    "#SBATCH --nodes=1",
    "#SBATCH --ntasks-per-node=1",
    "#SBATCH --gpus-per-node=1",
    "#SBATCH --cpus-per-task=12",
    "#SBATCH --mem=64G",
    "#SBATCH --array=0-29%30",
    f"PYTHON_EXECUTABLE={PINNED_FORMAL_PYTHON}",
)
GATE_LINES = (
    "#SBATCH --partition=cpu",
    "#SBATCH --time=04:00:00",
    "#SBATCH --nodes=1",
    "#SBATCH --ntasks-per-node=1",
    "#SBATCH --cpus-per-task=12",
    "#SBATCH --mem=64G",
    f"PYTHON_EXECUTABLE={PINNED_FORMAL_PYTHON}",
)


def _exact_lines(path: Path, required: Sequence[str], label: str) -> str:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for line in required:
        if lines.count(line) != 1:
            raise ContractError(f"{label} must contain exactly one {line!r}")
    return text


def validate_slurms(package: Path) -> None:
    train = _exact_lines(package / "train.slurm", TRAIN_LINES, "training Slurm")
    gate = _exact_lines(package / "gate.slurm", GATE_LINES, "gate Slurm")
    if "TREEWM_PYTHON" in train or "TREEWM_PYTHON" in gate:
        raise ContractError("Slurm scripts permit inherited Python override")
    required_train = (
        'REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"',
        "/cm/shared/apps/slurm/current/bin/srun",
        "/cm/shared/apps/slurm/current/bin/scontrol",
        "--gpus-per-task=1",
        "--gpu-bind=single:1",
        "--cpus-per-task=12",
        "CANCEL_REQUESTED",
        "CANCELLED.json",
        "REQUEUE_REQUESTED.json",
        "READY_FOR_REQUEUE.json",
        "WORKER_COMPLETE.json",
        "REQUEUE_CALLING.json",
        '"$SCONTROL" requeue "$REQUEUE_TARGET"',
        "durable remote-worker CANCELLED.json",
        'campaign.py" snapshot',
        "5000)",
        "25000)",
    )
    for snippet in required_train:
        if snippet not in train:
            raise ContractError(f"training Slurm lacks {snippet!r}")
    forbidden = (
        'kill -TERM "$step_pid"',
        'kill -USR1 "$step_pid"',
        "scancel",
        "TREEWM_EXPECTED_PREREQUISITE",
        "TREEWM_EXPECTED_SELECTED_RECIPE",
    )
    for snippet in forbidden:
        if snippet in train:
            raise ContractError(f"training Slurm reintroduces unsafe/foreign contract {snippet!r}")
    if not (
        train.index('if [[ "$status" -eq 0 ]]')
        < train.index('if [[ -e "$CANCEL_LATCH" || "$status" -eq 143 ]]')
    ):
        raise ContractError("training Slurm does not let durable completion win a cancellation race")
    for snippet in ('campaign.py" snapshot', "stage_gate.py", "--publish", "5000|25000"):
        if snippet not in gate:
            raise ContractError(f"gate Slurm lacks {snippet!r}")


def verify_scheduler_dependency_policy(scontrol: str) -> dict[str, Any]:
    result = subprocess.run(
        [scontrol, "show", "config"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise ContractError(f"cannot inspect scheduler dependency policy: {result.stderr.strip()}")
    matching = [line.strip() for line in result.stdout.splitlines() if "kill_invalid_depend" in line]
    if not matching:
        raise ContractError("scheduler lacks kill_invalid_depend; failed stages could strand dependents")
    return {"status": "verified", "policy": "kill_invalid_depend", "config_lines": matching}


def verify_submit_interpreter(
    manifest: Mapping[str, Any],
    executable: str | Path | None = None,
) -> str:
    expected = Path(manifest["paths"]["python"])
    actual = Path(executable or sys.executable)
    if not expected.is_file() or not os.access(expected, os.X_OK):
        raise ContractError(f"pinned formal Python is unavailable: {expected}")
    if actual.resolve() != expected.resolve():
        raise ContractError(f"submit must run under pinned formal Python {expected}; actual is {actual}")
    return str(expected)


def _snapshot_source_paths(repo_root: Path) -> list[Path]:
    root = repo_root.resolve()
    package = root / "experiments" / "20-treewm-grounded-gauge-pilot-v2"
    candidates = [
        *(root / "treewm").rglob("*.py"),
        *(root / "configs").rglob("*.yaml"),
        root / "scripts" / "train.py",
        package / "protocol.sha256",
        *(package / relative for relative in PROTOCOL_FILES),
    ]
    result: set[Path] = set()
    for candidate in candidates:
        if not candidate.is_file() or candidate.is_symlink():
            raise ContractError(f"snapshot source is missing/symlinked: {candidate}")
        resolved = candidate.resolve()
        if not resolved.is_relative_to(root):
            raise ContractError(f"snapshot source escapes repository: {resolved}")
        result.add(resolved)
    return sorted(result)


def _git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def prepare_source_snapshot(repo_root: Path, manifest: Mapping[str, Any]) -> Path:
    root = repo_root.resolve()
    package = root / "experiments" / "20-treewm-grounded-gauge-pilot-v2"
    protocol = verify_protocol_lock(package)
    source = source_contract(root)
    snapshot_id = snapshot_identity_sha256(source, protocol)
    parent = Path(manifest["paths"]["run_root"]) / "state" / "source-snapshots" / snapshot_id
    destination = parent / "repo"
    marker_path = parent / "SNAPSHOT.json"
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / ".create.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if destination.exists():
            verified = verify_source_snapshot(destination)
            if (
                verified["source_sha256"] != source["source_sha256"]
                or verified["runtime_sha256"] != source["runtime_sha256"]
                or verified["package_protocol_sha256"] != protocol
                or verified["snapshot_identity_sha256"] != snapshot_id
            ):
                raise ContractError("existing snapshot identity differs")
            return destination

        temporary = parent / f".repo.tmp.{os.getpid()}.{time.time_ns()}"
        temporary.mkdir()
        try:
            for source_path in _snapshot_source_paths(root):
                relative = source_path.relative_to(root)
                target = temporary / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, target)
            copied = source_contract(temporary)
            if copied["source_sha256"] != source["source_sha256"]:
                raise ContractError("copied trainer source differs")
            if copied["runtime_sha256"] != source["runtime_sha256"]:
                raise ContractError("copied runtime differs")
            copied_protocol = verify_protocol_lock(
                temporary / "experiments" / "20-treewm-grounded-gauge-pilot-v2"
            )
            if copied_protocol != protocol:
                raise ContractError("copied package protocol differs")
            atomic_json(marker_path, {
                "schema_version": 1,
                "status": "sealed_read_only",
                "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "git_commit": _git_value(root, "rev-parse", "HEAD"),
                "git_remote": _git_value(root, "config", "--get", "remote.origin.url"),
                "trainer_source_sha256": source["source_sha256"],
                "runtime_sha256": source["runtime_sha256"],
                "package_protocol_sha256": protocol,
                "snapshot_identity_sha256": snapshot_id,
                "recipe_files_verified": True,
                "repo_subdirectory": "repo",
                "repo_files_writable": False,
                "formal_validation": False,
            })
            for path in temporary.rglob("*"):
                if path.is_file() and not path.is_symlink():
                    path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
            for directory in sorted(
                (path for path in temporary.rglob("*") if path.is_dir()),
                key=lambda value: len(value.parts),
                reverse=True,
            ):
                directory.chmod(0o555)
            temporary.chmod(0o555)
            os.replace(temporary, destination)
        except BaseException:
            if temporary.exists():
                for path in temporary.rglob("*"):
                    if path.is_dir():
                        path.chmod(0o700)
                    elif path.is_file():
                        path.chmod(0o600)
                temporary.chmod(0o700)
                shutil.rmtree(temporary)
            marker_path.unlink(missing_ok=True)
            raise
    verify_source_snapshot(destination)
    return destination


def namespace_is_fresh(manifest: Mapping[str, Any]) -> bool:
    run_root = Path(manifest["paths"]["run_root"])
    if not run_root.exists():
        return True
    forbidden_names = (
        "GAUGE_PILOT_V2_LAUNCH.json",
        "latest.pt",
        "SUBMISSION_RECEIPT.json",
        "STAGE_GATE_5000.json",
        "STAGE_GATE_25000.json",
    )
    return not any(any(run_root.rglob(name)) for name in forbidden_names)


def static_test(
    manifest: Mapping[str, Any],
    repo_root: Path,
    *,
    verify_files: bool = False,
) -> dict[str, Any]:
    package = repo_root / "experiments" / "20-treewm-grounded-gauge-pilot-v2"
    validate_slurms(package)
    verification = verify_all(manifest, repo_root=repo_root, verify_files=verify_files)
    return {
        "schema_version": 1,
        "status": "static_package_verified",
        "campaign_id": manifest["campaign_id"],
        "formal_validation": False,
        "package_protocol_sha256": verification["package_protocol_sha256"],
        "source_sha256": verification["source_sha256"],
        "runtime_sha256": verification["runtime_sha256"],
        "stage_5000_runs": verification["stage_5000_runs"],
        "stage_25000_slots": verification["stage_25000_slots"],
        "recipe_files_verified": verification["recipe_files_verified"],
        "recipe_coverage_audits": verification["recipe_coverage_audits"],
        "namespace_fresh": namespace_is_fresh(manifest),
        "snapshot_created": False,
        "jobs_submitted": 0,
    }


def launch_plan(
    manifest: Mapping[str, Any],
    repo_root: Path,
    *,
    verify_files: bool,
    inspect_scheduler: bool = True,
) -> dict[str, Any]:
    package = repo_root / "experiments" / "20-treewm-grounded-gauge-pilot-v2"
    validate_slurms(package)
    verification = verify_all(manifest, repo_root=repo_root, verify_files=verify_files)
    scheduler = (
        verify_scheduler_dependency_policy(manifest["execution"]["scontrol"])
        if inspect_scheduler
        else {"status": "not_inspected_in_test"}
    )
    stage_5000 = expand_runs(manifest)
    continuation = continuation_runs(manifest)
    launches_5000 = [trainer_command(manifest, run, repo_root=repo_root) for run in stage_5000]
    launches_25000 = [trainer_command(manifest, run, repo_root=repo_root) for run in continuation]
    if len({launch["launch_sha256"] for launch in launches_5000}) != RUNS:
        raise ContractError("5k launch contracts are not unique across all 30 runs")
    if len(launches_25000) != CONTINUATION_RUNS:
        raise ContractError("25k continuation mapping is not 20 slots")
    stage5_by_index = {int(launch["run"]["index"]): launch for launch in launches_5000}
    for launch in launches_25000:
        original = stage5_by_index[int(launch["run"]["index"])]
        if original != launch:
            raise ContractError("25k continuation launch differs from its fresh 5k identity")

    common: dict[str, str] = {}
    for key in ("source_sha256", "runtime_sha256", "package_protocol_sha256", "actual_evaluation_bank_sha256"):
        values = {launch["hashes"][key] for launch in launches_5000}
        if len(values) != 1:
            raise ContractError(f"launch {key} differs across runs")
        common[key] = next(iter(values))
    plan: dict[str, Any] = {
        "schema_version": 1,
        "status": "sealed_bounded_causal_gauge_pilot_v2_plan",
        "campaign_id": manifest["campaign_id"],
        "formal_validation": False,
        "repo_root": str(repo_root),
        "verification": verification,
        "scheduler_dependency_policy": scheduler,
        "common_hashes": common,
        "stage_5000_runs": [
            {
                "stage_slot": slot,
                "index": run.index,
                "run_name": run.run_name,
                "setting_id": run.setting_id,
                "arm_id": run.arm_id,
                "seed": run.seed,
                "launch_sha256": launch["launch_sha256"],
                "config_sha256": launch["hashes"]["config_sha256"],
            }
            for slot, (run, launch) in enumerate(zip(stage_5000, launches_5000, strict=True))
        ],
        "stage_25000_slots": [
            {
                "stage_slot": slot,
                "index": run.index,
                "run_name": run.run_name,
                "setting_id": run.setting_id,
                "arm_id": run.arm_id,
                "seed": run.seed,
                "launch_sha256": launch["launch_sha256"],
                "action": "resume_if_selected_else_durable_skip",
            }
            for slot, (run, launch) in enumerate(zip(continuation, launches_25000, strict=True))
        ],
        "dag": [
            {"name": "train_5000", "kind": "gpu_array", "elements": RUNS, "array": "0-29%30", "dependency": None},
            {"name": "gate_5000", "kind": "cpu_gate", "dependency": "train_5000"},
            {"name": "train_25000", "kind": "gpu_array", "elements": CONTINUATION_RUNS, "array": "0-19%20", "dependency": "gate_5000"},
            {"name": "gate_25000", "kind": "cpu_gate", "dependency": "train_25000"},
        ],
        "dependency_policy": "Every edge is afterok; kill_invalid_depend is verified fail-closed.",
        "selection_policy": "The immutable 5k gate selects G before GS; N never continues; the nonselected continuation arm durably skips.",
    }
    plan["plan_sha256"] = stable_hash(plan)
    return plan


def create_claim(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _submit_node(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
    script: str,
    exports: Mapping[str, str],
    output: Path,
    dependency: str | None,
    array: str | None,
) -> tuple[str, list[str], str, str]:
    command = [manifest["execution"]["sbatch"], "--parsable"]
    if dependency is not None:
        command.append(f"--dependency=afterok:{dependency}")
    if array is not None:
        command.append(f"--array={array}")
    rendered_exports = ",".join(f"{key}={value}" for key, value in exports.items())
    command.extend([
        f"--export=ALL,{rendered_exports}",
        f"--output={output}",
        str(repo_root / "experiments" / "20-treewm-grounded-gauge-pilot-v2" / script),
    ])
    result = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    response = result.stdout.strip()
    match = SBATCH_JOB.fullmatch(response)
    if result.returncode != 0:
        raise ContractError(f"sbatch failed for {script}: {result.stderr.strip()}")
    if match is None:
        raise ContractError(f"sbatch response for {script} is ambiguous: {response!r}")
    return match.group("job_id"), command, result.stdout, result.stderr


def submit_dag(
    manifest: Mapping[str, Any],
    repo_root: Path,
    plan: Mapping[str, Any],
    state_root: Path,
) -> dict[str, Any]:
    common = plan["common_hashes"]
    base = {
        "TREEWM_GAUGE_PILOT_V2_REPO_ROOT": str(repo_root),
        "TREEWM_EXPECTED_SOURCE_SHA256": common["source_sha256"],
        "TREEWM_EXPECTED_RUNTIME_SHA256": common["runtime_sha256"],
        "TREEWM_EXPECTED_PACKAGE_PROTOCOL_SHA256": common["package_protocol_sha256"],
    }
    logs = Path(manifest["paths"]["run_root"]) / "logs"
    jobs: dict[str, Any] = {}

    def launch(
        name: str,
        script: str,
        extra: Mapping[str, str],
        output: str,
        dependency: str | None,
        array: str | None = None,
    ) -> str:
        job_id, command, stdout, stderr = _submit_node(
            manifest,
            repo_root=repo_root,
            script=script,
            exports={**base, **extra},
            output=logs / output,
            dependency=dependency,
            array=array,
        )
        jobs[name] = {
            "job_id": job_id,
            "dependency": dependency,
            "command": command,
            "stdout": stdout,
            "stderr": stderr,
        }
        atomic_json(state_root / "SUBMISSION_PROGRESS.json", {
            "schema_version": 1,
            "status": f"{name}_submitted",
            "plan_sha256": plan["plan_sha256"],
            "jobs": jobs,
        })
        return job_id

    train_5000 = launch(
        "train_5000", "train.slurm",
        {"TREEWM_EXPECTED_STAGE_TARGET": "5000"},
        "train_5000_%A_%a.out", None, "0-29%30",
    )
    gate_5000 = launch(
        "gate_5000", "gate.slurm",
        {"TREEWM_GATE_TARGET": "5000"},
        "gate_5000_%j.out", train_5000,
    )
    train_25000 = launch(
        "train_25000", "train.slurm",
        {"TREEWM_EXPECTED_STAGE_TARGET": "25000"},
        "train_25000_%A_%a.out", gate_5000, "0-19%20",
    )
    launch(
        "gate_25000", "gate.slurm",
        {"TREEWM_GATE_TARGET": "25000"},
        "gate_25000_%j.out", train_25000,
    )
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "staged_causal_gauge_pilot_v2_submitted",
        "campaign_id": manifest["campaign_id"],
        "formal_validation": False,
        "plan_sha256": plan["plan_sha256"],
        "repo_root": str(repo_root),
        "jobs": jobs,
        "submitted_unix_time": time.time(),
    }
    receipt["receipt_sha256"] = stable_hash(receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--manifest", type=Path, default=CAMPAIGN_DIR / "manifest.json")
    parser.add_argument("--verify-files", action="store_true", help="hash every recipe record during dry-run")
    parser.add_argument("--test-only", action="store_true", help="verify static package; never snapshot or submit")
    parser.add_argument("--submit", action="store_true", help="create a read-only snapshot and submit")
    args = parser.parse_args(argv)
    if args.test_only and args.submit:
        raise ContractError("--test-only and --submit are mutually exclusive")

    live_root = args.repo_root.resolve()
    manifest = load_manifest(args.manifest)
    verify_submit_interpreter(manifest)
    if args.test_only:
        print(json.dumps(static_test(manifest, live_root, verify_files=args.verify_files), sort_keys=True, indent=2))
        return 0
    if not namespace_is_fresh(manifest):
        raise ContractError("fresh Exp20 namespace already contains launch/checkpoint/gate/receipt state")
    dry_plan = launch_plan(
        manifest,
        live_root,
        verify_files=args.verify_files or args.submit,
        inspect_scheduler=True,
    )
    dry_plan["submission_ready"] = namespace_is_fresh(manifest)
    print(json.dumps(dry_plan, sort_keys=True, indent=2))
    if not args.submit:
        print("dry-run only: no snapshot created and no Slurm job submitted", file=sys.stderr)
        return 0

    state_root = Path(manifest["paths"]["run_root"]) / "state" / "submission"
    claim = state_root / "SUBMISSION_CLAIM.json"
    receipt_path = state_root / "SUBMISSION_RECEIPT.json"
    if receipt_path.exists():
        raise ContractError("Exp20 already has a submission receipt")
    create_claim(claim, {
        "schema_version": 1,
        "status": "submission_claimed",
        "dry_plan_sha256": dry_plan["plan_sha256"],
        "pid": os.getpid(),
        "claimed_unix_time": time.time(),
    })
    try:
        snapshot = prepare_source_snapshot(live_root, manifest)
        snapshot_manifest = load_manifest(
            snapshot / "experiments" / "20-treewm-grounded-gauge-pilot-v2" / "manifest.json"
        )
        plan = launch_plan(snapshot_manifest, snapshot, verify_files=True, inspect_scheduler=True)
        Path(manifest["paths"]["run_root"], "logs").mkdir(parents=True, exist_ok=True)
        atomic_json(state_root / "LAUNCH_PLAN.json", plan)
        receipt = submit_dag(snapshot_manifest, snapshot, plan, state_root)
        atomic_json(receipt_path, receipt)
        os.replace(claim, state_root / "SUBMISSION_CLAIM_CONSUMED.json")
    except BaseException as exc:
        atomic_json(state_root / "SUBMISSION_RECONCILIATION_REQUIRED.json", {
            "schema_version": 1,
            "status": "submission_incomplete_or_ambiguous",
            "error": repr(exc),
            "claim": str(claim),
            "unix_time": time.time(),
        })
        raise
    print(json.dumps(receipt, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"Exp20 gauge-pilot submit error: {exc}", file=sys.stderr)
        raise SystemExit(2)
