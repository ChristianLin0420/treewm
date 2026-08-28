#!/usr/bin/env python3
"""Exp24 paired held-out cell boundary; seed/evidence locks are absent in M1."""

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
    parser.add_argument("--cell-index", type=int)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--submission-root", type=Path)
    parser.add_argument("--submission-sha256")
    parser.add_argument("--node")
    args = parser.parse_args(argv)
    if not args.execute:
        manifest = campaign.load_manifest()
        print(json.dumps({
            "schema_version": 1,
            "status": "m1_heldout_scaffold_execution_blocked",
            "cells": campaign.FINAL_EVAL_CELLS,
            "rails": ["learned", "bfs"],
            "episodes_per_cell_per_rail": 50,
            "only_rail_difference": "tree_config.scorer",
        }, sort_keys=True, indent=2, allow_nan=False))
        return 0
    runtime.require(args.snapshot_root is not None and args.submission_root is not None, "held-out roots are required")
    runtime.require(isinstance(args.submission_sha256, str), "held-out submission SHA256 is required")
    runtime.bootstrap_queued_entry(
        submission_root=args.submission_root,
        submission_sha256=args.submission_sha256,
        snapshot_root=args.snapshot_root,
        executing_file=Path(__file__),
        expected_relative=runtime.PACKAGE_RELATIVE / "final_eval.py",
    )
    manifest = campaign.load_manifest()
    campaign.assert_launch_authorized(manifest)
    runtime.require(args.node == "heldout_eval", "held-out node identity differs")
    runtime.require(args.cell_index is not None and 0 <= args.cell_index < 200, "held-out cell differs")
    raise runtime.RuntimeContractError("M1 held-out seed table/parity adapter is not sealed")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (campaign.ContractError, runtime.RuntimeContractError) as exc:
        print(f"EXP24_EVAL_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
