#!/usr/bin/env python3
"""Dry-run and explicitly submit the sealed 32-task corrected pilot array."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Sequence

from campaign import (
    CAMPAIGN_DIR,
    ContractError,
    REPOSITORY_ROOT,
    atomic_json,
    expand_runs,
    load_manifest,
    stable_hash,
    trainer_command,
    verify_all,
)


REQUIRED_SLURM_LINES = (
    "#SBATCH --time=04:00:00",
    "#SBATCH --requeue",
    "#SBATCH --signal=B:USR1@420",
    "#SBATCH --gpus-per-node=1",
    "#SBATCH --mem=64G",
    "#SBATCH --array=0-31%32",
)
REQUIRED_SNIPPETS = (
    "/cm/shared/apps/slurm/current/bin/srun",
    "/cm/shared/apps/slurm/current/bin/scontrol",
    'REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"',
    "CANCEL_REQUESTED",
    "READY_FOR_REQUEUE.json",
    "REQUEUE_CALLING.json",
    '"$SCONTROL" requeue "$REQUEUE_TARGET"',
    "worker.py",
    "--gpus-per-task=1",
    "TREEWM_EXPECTED_PACKAGE_PROTOCOL_SHA256",
)

SBATCH_JOB = re.compile(r"^(?P<job_id>[0-9]+)(?:;[A-Za-z0-9_.-]+)?$")


def create_submission_claim(path: Path, payload: dict) -> None:
    """Create the single-submission claim with kernel-enforced exclusivity."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        raise


def validate_slurm(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for line in REQUIRED_SLURM_LINES:
        if lines.count(line) != 1:
            raise ContractError(f"pilot.slurm must contain exactly one {line!r}")
    for snippet in REQUIRED_SNIPPETS:
        if snippet not in text:
            raise ContractError(f"pilot.slurm lacks {snippet!r}")
    if "scripts/train.py" in text:
        raise ContractError("pilot.slurm must launch through the guarded worker")


def launch_plan(manifest, repo_root: Path) -> dict:
    # Seal actual recipe bytes, not only their manifests/size/mtime metadata.
    verification = verify_all(manifest, repo_root=repo_root, verify_files=True)
    runs = expand_runs(manifest)
    launches = [trainer_command(manifest, run, repo_root=repo_root) for run in runs]
    if len({item["launch_sha256"] for item in launches}) != 32:
        raise ContractError("launch contracts are not unique")
    source_hashes = {item["hashes"]["source_sha256"] for item in launches}
    runtime_hashes = {item["hashes"]["runtime_sha256"] for item in launches}
    protocol_hashes = {item["hashes"]["package_protocol_sha256"] for item in launches}
    if not (len(source_hashes) == len(runtime_hashes) == len(protocol_hashes) == 1):
        raise ContractError("launch hashes are inconsistent")
    return {
        "schema_version": 1,
        "status": "sealed_dry_run",
        "formal_validation": False,
        "campaign_id": manifest["campaign_id"],
        "created_unix_time": time.time(),
        "verification": verification,
        "source_sha256": next(iter(source_hashes)),
        "runtime_sha256": next(iter(runtime_hashes)),
        "package_protocol_sha256": next(iter(protocol_hashes)),
        "runs": [
            {"index": run.index, "run_name": run.run_name, "launch_sha256": launch["launch_sha256"], "config_sha256": launch["hashes"]["config_sha256"]}
            for run, launch in zip(runs, launches, strict=True)
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=CAMPAIGN_DIR / "manifest.json")
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--submit", action="store_true", help="actually call sbatch; absent means dry-run only")
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    manifest = load_manifest(args.manifest)
    validate_slurm(repo_root / "experiments" / "13-treewm-corrected-pilot" / "pilot.slurm")
    plan = launch_plan(manifest, repo_root)
    print(json.dumps(plan, sort_keys=True, indent=2))
    if not args.submit:
        print("dry-run only: no Slurm job submitted", file=sys.stderr)
        return 0
    run_root = Path(manifest["paths"]["run_root"])
    run_root.joinpath("logs").mkdir(parents=True, exist_ok=True)
    state_root = run_root / "state"
    audit = state_root / "launch-plan.json"
    if audit.exists():
        existing = json.loads(audit.read_text(encoding="utf-8"))
        comparable = {key: value for key, value in existing.items() if key != "created_unix_time"}
        requested = {key: value for key, value in plan.items() if key != "created_unix_time"}
        if comparable != requested:
            raise ContractError("existing launch plan has a different source/config/protocol")
        plan = existing
    else:
        atomic_json(audit, plan)
    receipt = state_root / "SUBMISSION_RECEIPT.json"
    claim = state_root / "SUBMISSION_CLAIM.json"
    if receipt.exists():
        prior = json.loads(receipt.read_text(encoding="utf-8"))
        raise ContractError(
            f"pilot array was already submitted as job {prior.get('job_id', 'unknown')}"
        )
    create_submission_claim(
        claim,
        {
            "schema_version": 1,
            "status": "submission_claimed",
            "campaign_id": manifest["campaign_id"],
            "plan_sha256": stable_hash(plan),
            "pid": os.getpid(),
            "created_unix_time": time.time(),
        },
    )
    command = [
        manifest["execution"]["sbatch"],
        "--parsable",
        (
            f"--export=ALL,TREEWM_PILOT_REPO_ROOT={repo_root},"
            f"TREEWM_EXPECTED_SOURCE_SHA256={plan['source_sha256']},"
            f"TREEWM_EXPECTED_RUNTIME_SHA256={plan['runtime_sha256']},"
            "TREEWM_EXPECTED_PACKAGE_PROTOCOL_SHA256="
            f"{plan['package_protocol_sha256']}"
        ),
        f"--output={run_root}/logs/%x_%A_%a.out",
        str(repo_root / "experiments" / "13-treewm-corrected-pilot" / "pilot.slurm"),
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    submission = result.stdout.strip()
    match = SBATCH_JOB.fullmatch(submission)
    if result.returncode != 0:
        atomic_json(
            state_root / f"SUBMISSION_FAILED_{time.time_ns()}.json",
            {
                "schema_version": 1,
                "status": "sbatch_failed_before_receipt",
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": command,
                "unix_time": time.time(),
            },
        )
        claim.unlink()
        print(result.stderr, file=sys.stderr, end="")
        return int(result.returncode)
    if match is None:
        # A zero exit may already have created a job. Keep the claim so an operator must
        # reconcile scheduler state instead of risking a duplicate 32-task submission.
        atomic_json(
            state_root / "SUBMISSION_AMBIGUOUS.json",
            {
                "schema_version": 1,
                "status": "sbatch_succeeded_but_job_id_unparseable",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "command": command,
                "unix_time": time.time(),
            },
        )
        raise ContractError("sbatch succeeded but returned no parseable job ID")
    job_id = match.group("job_id")
    atomic_json(
        receipt,
        {
            "schema_version": 1,
            "status": "submitted",
            "campaign_id": manifest["campaign_id"],
            "job_id": job_id,
            "sbatch_response": submission,
            "plan_sha256": stable_hash(plan),
            "source_sha256": plan["source_sha256"],
            "runtime_sha256": plan["runtime_sha256"],
            "package_protocol_sha256": plan["package_protocol_sha256"],
            "repo_root": str(repo_root),
            "command": command,
            "submitted_unix_time": time.time(),
        },
    )
    os.replace(claim, state_root / "SUBMISSION_CLAIM_CONSUMED.json")
    print(job_id)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"corrected-pilot submit error: {exc}", file=sys.stderr)
        raise SystemExit(2)
