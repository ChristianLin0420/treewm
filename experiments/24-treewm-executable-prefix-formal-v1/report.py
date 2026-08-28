#!/usr/bin/env python3
"""Exp24 immutable report boundary; scientific assembly is blocked in M1."""

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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-only", action="store_true")
    parser.add_argument("--publish", action="store_true")
    parser.add_argument("--submission-root", type=Path)
    parser.add_argument("--submission-sha256")
    args = parser.parse_args(argv)
    if args.test_only and args.publish:
        parser.error("choose one mode")
    if not args.publish:
        manifest = campaign.load_manifest()
        print(json.dumps({
            "schema_version": 1,
            "status": "m1_report_assembly_scaffold_execution_blocked",
            "immutable_artifacts": [
                "REPORT_BUNDLE.json",
                "REPORT_DECISION.json",
                "REPORT_PROVENANCE.json",
                "REPORT_COMMIT.json",
            ],
            "commit_is_final_acceptance_point": True,
            "report_cancel_lock": True,
            "readiness": runtime.execution_readiness(manifest),
        }, sort_keys=True, indent=2, allow_nan=False))
        return 0
    runtime.require(args.submission_root is not None, "submission root is required")
    runtime.require(isinstance(args.submission_sha256, str), "submission SHA256 is required")
    snapshot_root = args.submission_root / "source-snapshot" / "repo"
    runtime.bootstrap_queued_entry(
        submission_root=args.submission_root,
        submission_sha256=args.submission_sha256,
        snapshot_root=snapshot_root,
        executing_file=Path(__file__),
        expected_relative=runtime.PACKAGE_RELATIVE / "report.py",
    )
    manifest = campaign.load_manifest()
    # A scheduled publisher must still authenticate the sealed package before its
    # first report/cancel-lock write.  M1 fails here.
    campaign.assert_launch_authorized(manifest)
    raise runtime.RuntimeContractError("M1 report evidence assembly is not sealed")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (campaign.ContractError, runtime.RuntimeContractError) as exc:
        print(f"EXP24_REPORT_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
