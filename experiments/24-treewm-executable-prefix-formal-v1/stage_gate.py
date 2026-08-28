#!/usr/bin/env python3
"""Exp24 all-forty stage gate boundary; evidence parser is unsealed in M1."""

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
    parser.add_argument("--describe", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--stage-target", type=int)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--submission-root", type=Path)
    parser.add_argument("--submission-sha256")
    parser.add_argument("--node")
    args = parser.parse_args(argv)
    if not args.execute:
        manifest = campaign.load_manifest()
        print(json.dumps({
            "schema_version": 1,
            "status": "m1_stage_gate_scaffold_execution_blocked",
            "targets": list(campaign.STAGE_TARGETS),
            "required_run_count": 40,
            "strict_scalar_identity_parser_required": True,
            "identical_duplicates_inventoried": True,
            "conflicting_duplicates_fatal": True,
        }, sort_keys=True, indent=2, allow_nan=False))
        return 0
    runtime.require(args.snapshot_root is not None and args.submission_root is not None, "gate roots are required")
    runtime.require(isinstance(args.submission_sha256, str), "gate submission SHA256 is required")
    runtime.bootstrap_queued_entry(
        submission_root=args.submission_root,
        submission_sha256=args.submission_sha256,
        snapshot_root=args.snapshot_root,
        executing_file=Path(__file__),
        expected_relative=runtime.PACKAGE_RELATIVE / "stage_gate.py",
    )
    manifest = campaign.load_manifest()
    campaign.assert_launch_authorized(manifest)
    runtime.require(args.node == f"gate_{args.stage_target}", "gate node identity differs")
    runtime.require(args.stage_target in campaign.STAGE_TARGETS, "stage target differs")
    raise runtime.RuntimeContractError("M1 stage evidence parser/all-ten contracts are not sealed")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (campaign.ContractError, runtime.RuntimeContractError) as exc:
        print(f"EXP24_GATE_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
