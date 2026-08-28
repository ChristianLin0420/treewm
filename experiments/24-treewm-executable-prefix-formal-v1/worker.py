#!/usr/bin/env python3
"""Exp24 staged worker boundary; scientific execution is blocked in M1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

sys.dont_write_bytecode = True
PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import campaign
import runtime


def _execution_block(manifest: dict[str, object]) -> None:
    campaign.assert_launch_authorized(manifest)
    readiness = runtime.execution_readiness(manifest)
    runtime.require(readiness["ready"], "worker execution remains blocked: " + "; ".join(readiness["blockers"]))
    raise runtime.RuntimeContractError("M1 has no sealed scientific worker adapter")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("describe", "run", "record-signal", "requeue"), nargs="?", default="describe")
    parser.add_argument("--submission-root", type=Path)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--submission-sha256")
    parser.add_argument("--node")
    parser.add_argument("--cell-index", type=int)
    parser.add_argument("--restart-count", type=int, default=0)
    parser.add_argument("--array-job-id")
    parser.add_argument("--array-task-id", type=int)
    parser.add_argument("--signal", choices=("USR1", "TERM"))
    args = parser.parse_args(argv)
    if args.command == "describe":
        manifest = campaign.load_manifest()
        print(json.dumps({
            "schema_version": 1,
            "status": "m1_worker_scaffold_execution_blocked",
            "same_cell_requeue_contract_scaffolded": True,
            "same_cell_requeue_execution_implemented": False,
            "cross_stage_promotion": "only_from_immediately_prior_all_40_gate",
            "readiness": runtime.execution_readiness(manifest),
        }, sort_keys=True, indent=2, allow_nan=False))
        return 0
    runtime.require(args.submission_root is not None and args.snapshot_root is not None, "worker roots are required")
    runtime.require(isinstance(args.submission_sha256, str), "worker submission SHA256 is required")
    runtime.bootstrap_queued_entry(
        submission_root=args.submission_root,
        submission_sha256=args.submission_sha256,
        snapshot_root=args.snapshot_root,
        executing_file=Path(__file__),
        expected_relative=runtime.PACKAGE_RELATIVE / "worker.py",
    )
    manifest = campaign.load_manifest()
    _execution_block(manifest)
    return 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (campaign.ContractError, runtime.RuntimeContractError) as exc:
        print(f"EXP24_WORKER_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
