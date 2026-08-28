#!/usr/bin/env python3
"""Fail-closed interface for a future accepted Exp23 engineering-pilot binder.

Launch8's reporter and gate protocol is not frozen.  M2A therefore cannot define
an authoritative positive semantic adapter and must not reuse the terminal-negative
Launch7 protocol as a surrogate.  This module freezes the public interface and the
requirements for the later versioned adapter, but every verification attempt fails
before opening a report path or publishing a binding.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True
PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import runtime


EXPECTED_ACCEPTED_CAMPAIGN_ID = runtime.EXPECTED_ACCEPTED_PILOT_CAMPAIGN_ID
FORBIDDEN_POSITIVE_CAMPAIGN_ID = runtime.FORBIDDEN_POSITIVE_PILOT_CAMPAIGN_ID
POSITIVE_ADAPTER_STATE = runtime.ENGINEERING_PILOT_ADAPTER_STATE
POSITIVE_BINDING_STATE = "unbound"
ADAPTER_REQUIREMENTS = runtime.ENGINEERING_PILOT_ADAPTER_REQUIREMENTS


class EngineeringPilotBindingError(runtime.RuntimeContractError):
    """The future positive adapter is unavailable or evidence differs."""


def adapter_description() -> dict[str, Any]:
    return runtime.engineering_pilot_adapter_description()


def verify_engineering_pilot_report_quartet(
    report_root: Path,
    *,
    expected_report_root: Path,
    expected_submission_root: Path,
    expected_submission_sha256: str,
    expected_package_binding: Mapping[str, str],
) -> dict[str, Any]:
    """Reject before inspecting caller-controlled paths until Launch8 freezes."""

    del (
        report_root,
        expected_report_root,
        expected_submission_root,
        expected_submission_sha256,
        expected_package_binding,
    )
    raise EngineeringPilotBindingError(
        "future accepted-pilot adapter is not sealed: " + POSITIVE_ADAPTER_STATE
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args(argv)
    if not args.describe:
        raise EngineeringPilotBindingError(
            "M2A exposes description only; positive binding is disabled"
        )
    print(runtime.canonical_json(adapter_description()))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EngineeringPilotBindingError, runtime.RuntimeContractError) as exc:
        print(f"EXP24_ENGINEERING_PILOT_BINDER_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
