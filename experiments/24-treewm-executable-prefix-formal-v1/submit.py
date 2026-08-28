#!/usr/bin/env python3
"""Read-only M2A preflight, ephemeral snapshot test, or authority-gated submit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

sys.dont_write_bytecode = True
PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    # ``-I`` deliberately removes the script directory.  This design-only bootstrap
    # adds exactly the resolved package containing this already-open entry point.
    sys.path.insert(0, str(PACKAGE_DIR))
PINNED_PYTHON = (
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/envs/treewm-formal-py311/bin/python"
)


class BootstrapError(RuntimeError):
    pass


def _require_mutation_bootstrap() -> None:
    if not (
        sys.executable == PINNED_PYTHON
        and sys.version_info[:3] == (3, 11, 15)
        and sys.flags.isolated == 1
        and sys.flags.no_site == 1
        and sys.flags.dont_write_bytecode == 1
    ):
        raise BootstrapError(
            "mutation/snapshot test requires exact pinned Python 3.11 with -I -S -B"
        )


def _internal_snapshot_probe(snapshot_root: Path, inventory_file: Path) -> int:
    _require_mutation_bootstrap()
    import campaign
    import runtime
    value, _digest = runtime.authenticated_immutable_json(
        inventory_file,
        "snapshot probe inventory",
        mode=0o400,
    )
    if set(value) != {"schema_version", "files"} or value.get("schema_version") != 1:
        raise campaign.ContractError("snapshot probe inventory schema differs")
    inventory = value.get("files")
    if not isinstance(inventory, dict):
        raise campaign.ContractError("snapshot probe file inventory is malformed")
    runtime.verify_snapshot_files(snapshot_root, inventory)
    manifest = campaign.load_manifest(snapshot_root / runtime.PACKAGE_RELATIVE / "manifest.json")
    result = {
        "schema_version": 1,
        "status": "verified_m2a_snapshot_bootstrap",
        "manifest_sha256": campaign.manifest_sha256(manifest),
        "inventory_sha256": runtime.stable_hash(inventory),
        "file_count": len(inventory),
        "isolated": True,
        "no_site": True,
        "dont_write_bytecode": True,
    }
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--test-only", action="store_true", help="print the read-only blocked preflight (default)")
    modes.add_argument("--snapshot-test", action="store_true", help="build, verify, and remove a private temporary snapshot")
    modes.add_argument("--submit", action="store_true", help="explicit authority-gated submission")
    modes.add_argument("--internal-snapshot-probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--snapshot-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--inventory-file", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.internal_snapshot_probe:
        if args.snapshot_root is None or args.inventory_file is None:
            parser.error("internal snapshot probe requires both private paths")
        return _internal_snapshot_probe(args.snapshot_root, args.inventory_file)
    if args.snapshot_root is not None or args.inventory_file is not None:
        parser.error("private snapshot paths are internal-only")
    if args.submit or args.snapshot_test:
        # This precedes every package import and every local/scheduler mutation.
        _require_mutation_bootstrap()
    import campaign
    import runtime

    manifest = campaign.load_manifest()
    if args.submit:
        # This is deliberately the first action after read-only parsing/manifest
        # authentication.  M2A cannot create a claim, snapshot, output, or scheduler
        # process because future Launch8/all-ten/environment/feasibility authority fails.
        campaign.assert_launch_authorized(manifest)
        result = runtime.authorized_submit(manifest)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    if args.snapshot_test:
        result = runtime.snapshot_test(campaign.REPOSITORY_ROOT)
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    report = campaign.preflight_report(manifest)
    report["runtime"] = runtime.runtime_description(manifest)
    report["working_directory"] = os.getcwd()
    print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"EXP24_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
