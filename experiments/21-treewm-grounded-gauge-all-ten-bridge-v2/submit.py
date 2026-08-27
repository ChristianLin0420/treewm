#!/usr/bin/env python3
"""Statically test, dry-run, or explicitly submit the sealed Exp21 bridge DAG."""

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
    BINDING_PATH,
    CAMPAIGN_DIR,
    ContractError,
    PINNED_PYTHON,
    PROTOCOL_FILES,
    REPOSITORY_ROOT,
    RUNS,
    STAGE_TARGET,
    atomic_json,
    expand_runs,
    load_exp20_binding,
    load_compatible_input,
    load_manifest,
    read_json,
    require,
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
    "#SBATCH --array=0-19%20",
    f"PYTHON_EXECUTABLE={PINNED_PYTHON}",
)
GATE_LINES = (
    "#SBATCH --partition=cpu",
    "#SBATCH --time=04:00:00",
    "#SBATCH --nodes=1",
    "#SBATCH --ntasks-per-node=1",
    "#SBATCH --cpus-per-task=12",
    "#SBATCH --mem=64G",
    f"PYTHON_EXECUTABLE={PINNED_PYTHON}",
)


def _exact_lines(path: Path, required: Sequence[str], label: str) -> str:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for line in required:
        require(lines.count(line) == 1, f"{label} must contain exactly one {line!r}")
    return text


def validate_slurms(package: Path = CAMPAIGN_DIR) -> None:
    train = _exact_lines(package / "train.slurm", TRAIN_LINES, "training Slurm")
    gate = _exact_lines(package / "gate.slurm", GATE_LINES, "gate Slurm")
    require("TREEWM_PYTHON" not in train + gate, "Slurm permits inherited Python override")
    for snippet in (
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
        "TREEWM_EXPECTED_EXP20_BINDING_SHA256",
        "TREEWM_EXPECTED_SELECTED_RECIPE_SHA256",
    ):
        require(snippet in train, f"training Slurm lacks {snippet!r}")
    for forbidden in ('kill -TERM "$step_pid"', 'kill -USR1 "$step_pid"', "scancel"):
        require(forbidden not in train, f"training Slurm contains unsafe {forbidden!r}")
    require(
        train.index('if [[ "$status" -eq 0 ]]')
        < train.index('if [[ -e "$CANCEL_LATCH" || "$status" -eq 143 ]]'),
        "training Slurm does not let durable completion win a cancellation race",
    )
    for snippet in ('campaign.py" snapshot', "stage_gate.py", "--publish", "25000"):
        require(snippet in gate, f"gate Slurm lacks {snippet!r}")


def verify_submit_interpreter(manifest: Mapping[str, Any], executable: str | Path | None = None) -> str:
    expected = Path(manifest["paths"]["python"])
    actual = Path(executable or sys.executable)
    require(expected.is_file() and os.access(expected, os.X_OK), f"pinned Python unavailable: {expected}")
    require(actual.resolve() == expected.resolve(), f"submit must use pinned Python {expected}; actual {actual}")
    return str(expected)


def verify_scheduler_dependency_policy(scontrol: str) -> dict[str, Any]:
    result = subprocess.run(
        [scontrol, "show", "config"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(result.returncode == 0, f"cannot inspect scheduler policy: {result.stderr.strip()}")
    matching = [line.strip() for line in result.stdout.splitlines() if "kill_invalid_depend" in line]
    require(bool(matching), "scheduler lacks kill_invalid_depend")
    return {"status": "verified", "policy": "kill_invalid_depend", "config_lines": matching}


def namespace_is_fresh(manifest: Mapping[str, Any]) -> bool:
    root = Path(manifest["paths"]["run_root"])
    if not root.exists():
        return True
    forbidden = (
        "GAUGE_BRIDGE_LAUNCH.json",
        "latest.pt",
        "SUBMISSION_RECEIPT.json",
        "STAGE_COMPLETE_25000.json",
        "acceptance.json",
    )
    return not any(any(root.rglob(name)) for name in forbidden)


def static_test(
    manifest: Mapping[str, Any],
    repo_root: Path,
    *,
    verify_files: bool = False,
) -> dict[str, Any]:
    package = repo_root / "experiments/21-treewm-grounded-gauge-all-ten-bridge-v2"
    validate_slurms(package)
    protocol = verify_protocol_lock(package)
    source = source_contract(repo_root)
    recipe_audits = {
        setting["id"]: load_compatible_input(manifest, setting, verify_files=verify_files)["recipe_coverage_audit"]
        for setting in manifest["settings"]
    }
    raw_binding = read_json(package / "exp20_binding.json")
    sealed = raw_binding.get("status") == "sealed_exp20_acceptance"
    if sealed:
        binding_status = load_exp20_binding(manifest, package / "exp20_binding.json")["status"]
    else:
        require(raw_binding == {
            "schema_version": 1,
            "campaign_id": manifest["campaign_id"],
            "status": "unsealed_waiting_for_exp20_acceptance",
            "launch_allowed": False,
            "exp20": None,
            "selected_arm": None,
            "selected_recipe": None,
            "selected_recipe_sha256": None,
            "binding_sha256": None,
        }, "unsealed binding placeholder differs")
        binding_status = str(raw_binding["status"])
    return {
        "schema_version": 1,
        "status": "static_package_verified" if sealed else "static_package_verified_blocked_on_exp20",
        "campaign_id": manifest["campaign_id"],
        "formal_validation": False,
        "package_protocol_sha256": protocol,
        "source_sha256": source["source_sha256"],
        "runtime_sha256": source["runtime_sha256"],
        "recipe_files_verified": bool(verify_files),
        "recipe_coverage_audits": recipe_audits,
        "binding_status": binding_status,
        "launch_allowed": sealed,
        "namespace_fresh": namespace_is_fresh(manifest),
        "jobs_submitted": 0,
        "snapshot_created": False,
    }


def _snapshot_source_paths(repo_root: Path) -> list[Path]:
    root = repo_root.resolve()
    package = root / "experiments/21-treewm-grounded-gauge-all-ten-bridge-v2"
    candidates = [
        *(root / "treewm").rglob("*.py"),
        *(root / "configs").rglob("*.yaml"),
        root / "scripts/train.py",
        package / "protocol.sha256",
        *(package / relative for relative in PROTOCOL_FILES),
    ]
    result: set[Path] = set()
    for candidate in candidates:
        require(candidate.is_file() and not candidate.is_symlink(), f"snapshot source missing/symlinked: {candidate}")
        resolved = candidate.resolve()
        require(resolved.is_relative_to(root), f"snapshot source escapes repository: {resolved}")
        result.add(resolved)
    return sorted(result)


def _git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=root, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def prepare_source_snapshot(repo_root: Path, manifest: Mapping[str, Any]) -> Path:
    root = repo_root.resolve()
    package = root / "experiments/21-treewm-grounded-gauge-all-ten-bridge-v2"
    protocol = verify_protocol_lock(package)
    source = source_contract(root)
    snapshot_id = snapshot_identity_sha256(source, protocol)
    parent = Path(manifest["paths"]["run_root"]) / "state/source-snapshots" / snapshot_id
    destination = parent / "repo"
    marker = parent / "SNAPSHOT.json"
    parent.mkdir(parents=True, exist_ok=True)
    with (parent / ".create.lock").open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if destination.exists():
            verified = verify_source_snapshot(destination)
            require(verified["snapshot_identity_sha256"] == snapshot_id, "existing snapshot identity differs")
            return destination
        temporary = parent / f".repo.tmp.{os.getpid()}.{time.time_ns()}"
        temporary.mkdir()
        try:
            for source_path in _snapshot_source_paths(root):
                target = temporary / source_path.relative_to(root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source_path, target)
            copied = source_contract(temporary)
            require(copied["source_sha256"] == source["source_sha256"], "copied trainer source differs")
            require(copied["runtime_sha256"] == source["runtime_sha256"], "copied runtime differs")
            require(verify_protocol_lock(temporary / package.relative_to(root)) == protocol, "copied protocol differs")
            atomic_json(marker, {
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
            for directory in sorted((p for p in temporary.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                directory.chmod(0o555)
            temporary.chmod(0o555)
            os.replace(temporary, destination)
        except BaseException:
            if temporary.exists():
                for path in temporary.rglob("*"):
                    path.chmod(0o700 if path.is_dir() else 0o600)
                temporary.chmod(0o700)
                shutil.rmtree(temporary)
            marker.unlink(missing_ok=True)
            raise
    verify_source_snapshot(destination)
    return destination


def launch_plan(
    manifest: Mapping[str, Any],
    repo_root: Path,
    *,
    verify_files: bool,
    inspect_scheduler: bool,
) -> dict[str, Any]:
    validate_slurms(repo_root / "experiments/21-treewm-grounded-gauge-all-ten-bridge-v2")
    verification = verify_all(manifest, repo_root=repo_root, verify_files=verify_files)
    scheduler = verify_scheduler_dependency_policy(manifest["execution"]["scontrol"]) if inspect_scheduler else {"status": "not_inspected_in_test"}
    runs = expand_runs(manifest)
    launches = [trainer_command(manifest, run, repo_root=repo_root) for run in runs]
    require(len(launches) == RUNS and len({row["launch_sha256"] for row in launches}) == RUNS, "launch identities collide")
    common: dict[str, str] = {}
    for key in (
        "source_sha256", "runtime_sha256", "package_protocol_sha256",
        "exp20_binding_sha256", "selected_recipe_sha256", "actual_evaluation_bank_sha256",
    ):
        values = {launch["hashes"][key] for launch in launches}
        require(len(values) == 1, f"fleet {key} differs")
        common[key] = next(iter(values))
    plan: dict[str, Any] = {
        "schema_version": 1,
        "status": "sealed_fresh_bounded_all_ten_gauge_bridge_plan",
        "campaign_id": manifest["campaign_id"],
        "formal_validation": False,
        "repo_root": str(repo_root),
        "verification": verification,
        "scheduler_dependency_policy": scheduler,
        "common_hashes": common,
        "runs": [
            {
                "index": run.index,
                "run_name": run.run_name,
                "setting_id": run.setting_id,
                "seed": run.seed,
                "selected_arm": launch["run"]["selected_arm"],
                "launch_sha256": launch["launch_sha256"],
                "config_sha256": launch["hashes"]["config_sha256"],
            }
            for run, launch in zip(runs, launches, strict=True)
        ],
        "dag": [
            {"name": "train_25000", "kind": "gpu_array", "elements": RUNS, "array": "0-19%20", "dependency": None},
            {"name": "gate_25000", "kind": "cpu_gate", "dependency": "train_25000"},
        ],
        "dependency_policy": "Every edge is afterok; kill_invalid_depend is verified fail-closed.",
        "downstream_policy": "No 1M formal launch is authorized unless this immutable 25k gate accepts all method cells and the prospective outcome quorum.",
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
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    if array:
        command.append(f"--array={array}")
    command.extend([
        f"--export=ALL,{','.join(f'{key}={value}' for key, value in exports.items())}",
        f"--output={output}",
        str(repo_root / "experiments/21-treewm-grounded-gauge-all-ten-bridge-v2" / script),
    ])
    result = subprocess.run(command, cwd=repo_root, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    response = result.stdout.strip()
    match = SBATCH_JOB.fullmatch(response)
    require(result.returncode == 0, f"sbatch failed for {script}: {result.stderr.strip()}")
    require(match is not None, f"ambiguous sbatch response for {script}: {response!r}")
    return match.group("job_id"), command, result.stdout, result.stderr


def submit_dag(manifest: Mapping[str, Any], repo_root: Path, plan: Mapping[str, Any], state_root: Path) -> dict[str, Any]:
    common = plan["common_hashes"]
    base = {
        "TREEWM_GAUGE_BRIDGE_REPO_ROOT": str(repo_root),
        "TREEWM_EXPECTED_SOURCE_SHA256": common["source_sha256"],
        "TREEWM_EXPECTED_RUNTIME_SHA256": common["runtime_sha256"],
        "TREEWM_EXPECTED_PACKAGE_PROTOCOL_SHA256": common["package_protocol_sha256"],
        "TREEWM_EXPECTED_EXP20_BINDING_SHA256": common["exp20_binding_sha256"],
        "TREEWM_EXPECTED_SELECTED_RECIPE_SHA256": common["selected_recipe_sha256"],
    }
    logs = Path(manifest["paths"]["run_root"]) / "logs"
    jobs: dict[str, Any] = {}

    def launch(name: str, script: str, extra: Mapping[str, str], output: str, dependency: str | None, array: str | None = None) -> str:
        job_id, command, stdout, stderr = _submit_node(
            manifest, repo_root=repo_root, script=script, exports={**base, **extra},
            output=logs / output, dependency=dependency, array=array,
        )
        jobs[name] = {"job_id": job_id, "dependency": dependency, "command": command, "stdout": stdout, "stderr": stderr}
        atomic_json(state_root / "SUBMISSION_PROGRESS.json", {
            "schema_version": 1, "status": f"{name}_submitted", "plan_sha256": plan["plan_sha256"], "jobs": jobs,
        })
        return job_id

    train_job = launch(
        "train_25000", "train.slurm", {"TREEWM_EXPECTED_STAGE_TARGET": str(STAGE_TARGET)},
        "train_25000_%A_%a.out", None, "0-19%20",
    )
    launch("gate_25000", "gate.slurm", {"TREEWM_GATE_TARGET": str(STAGE_TARGET)}, "gate_25000_%j.out", train_job)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": "fresh_bounded_all_ten_gauge_bridge_submitted",
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
    parser.add_argument("--verify-files", action="store_true")
    parser.add_argument("--test-only", action="store_true", help="verify static package; never create snapshot or submit")
    parser.add_argument("--submit", action="store_true", help="explicitly snapshot and submit")
    args = parser.parse_args(argv)
    require(not (args.test_only and args.submit), "--test-only and --submit are mutually exclusive")
    live_root = args.repo_root.resolve()
    manifest = load_manifest(args.manifest)
    verify_submit_interpreter(manifest)
    if args.test_only:
        print(json.dumps(static_test(manifest, live_root, verify_files=args.verify_files), sort_keys=True, indent=2))
        return 0
    require(namespace_is_fresh(manifest), "fresh Exp21 namespace already contains launch/checkpoint/gate/receipt state")
    plan = launch_plan(
        manifest, live_root,
        verify_files=args.verify_files or args.submit,
        inspect_scheduler=True,
    )
    print(json.dumps(plan, sort_keys=True, indent=2))
    if not args.submit:
        print("dry-run only: no snapshot created and no Slurm job submitted", file=sys.stderr)
        return 0

    state_root = Path(manifest["paths"]["run_root"]) / "state/submission"
    claim = state_root / "SUBMISSION_CLAIM.json"
    receipt_path = state_root / "SUBMISSION_RECEIPT.json"
    require(not receipt_path.exists(), "Exp21 already has a submission receipt")
    create_claim(claim, {
        "schema_version": 1,
        "status": "submission_claimed",
        "dry_plan_sha256": plan["plan_sha256"],
        "pid": os.getpid(),
        "claimed_unix_time": time.time(),
    })
    try:
        snapshot = prepare_source_snapshot(live_root, manifest)
        snapshot_manifest = load_manifest(snapshot / "experiments/21-treewm-grounded-gauge-all-ten-bridge-v2/manifest.json")
        sealed_plan = launch_plan(snapshot_manifest, snapshot, verify_files=True, inspect_scheduler=True)
        Path(manifest["paths"]["run_root"], "logs").mkdir(parents=True, exist_ok=True)
        atomic_json(state_root / "LAUNCH_PLAN.json", sealed_plan)
        receipt = submit_dag(snapshot_manifest, snapshot, sealed_plan, state_root)
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
        print(f"Exp21 submit error: {exc}", file=sys.stderr)
        raise SystemExit(2)
