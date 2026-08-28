#!/usr/bin/env python3
"""Read-only Exp24 design preflight; executable submission is not yet present."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

sys.dont_write_bytecode = True
PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    # ``-I`` deliberately removes the script directory.  This design-only bootstrap
    # adds exactly the resolved package containing this already-open entry point.
    sys.path.insert(0, str(PACKAGE_DIR))

import campaign


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-only", action="store_true", help="print the read-only blocked preflight (default)")
    parser.add_argument("--submit", action="store_true", help="fail closed; runtime submission is not implemented")
    args = parser.parse_args(argv)
    if args.test_only and args.submit:
        parser.error("--test-only and --submit are mutually exclusive")
    manifest = campaign.load_manifest()
    if args.submit:
        campaign.assert_launch_authorized(manifest)
        raise campaign.ContractError("unreachable: design-only package has no scheduler mutation implementation")
    print(json.dumps(campaign.preflight_report(manifest), sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except campaign.ContractError as exc:
        print(f"EXP24_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
