#!/usr/bin/env python3
"""Isolated Exp24 trainer bootstrap; shared formal objective is absent in M1."""

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
    parser.add_argument("--cell-index", type=int)
    args = parser.parse_args(argv)
    if args.describe and args.execute:
        parser.error("choose one mode")
    manifest = campaign.load_manifest()
    if not args.execute:
        print(json.dumps({
            "schema_version": 1,
            "status": "formal_objective_delta_documented_not_applied",
            "objective": manifest["objective"]["objective_version"],
            "allowed_stage_targets": list(campaign.STAGE_TARGETS),
            "trainer_imported": False,
        }, sort_keys=True, indent=2, allow_nan=False))
        return 0
    campaign.assert_launch_authorized(manifest)
    readiness = runtime.execution_readiness(manifest)
    runtime.require(readiness["ready"], "trainer bootstrap remains blocked: " + "; ".join(readiness["blockers"]))
    raise runtime.RuntimeContractError("shared formal objective/config delta has not been applied")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (campaign.ContractError, runtime.RuntimeContractError) as exc:
        print(f"EXP24_TRAINER_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
