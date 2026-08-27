#!/usr/bin/env python3
"""Fail-closed trainer entry point for one exact Exp20 launch contract."""

from __future__ import annotations

import os
from pathlib import Path
import sys


CAMPAIGN_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = CAMPAIGN_DIR.parents[1]
if str(CAMPAIGN_DIR) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_DIR))
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import campaign


OBJECTIVE = "treewm_v2_grounded_gauge_pilot_v2"


def verify_exact_invocation() -> None:
    """Reject direct config composition and every unsealed Exp20 launch."""
    manifest = campaign.load_manifest()
    campaign.verify_protocol_lock()
    run_name = os.environ.get("TREEWM_RUN_NAME", "")
    matches = [run for run in campaign.expand_runs(manifest) if run.run_name == run_name]
    campaign.require(len(matches) == 1, "TREEWM_RUN_NAME does not identify one sealed Exp20 run")
    expected = campaign.trainer_command(manifest, matches[0], repo_root=REPOSITORY_ROOT)
    campaign.require(
        sys.argv[1:] == expected["argv"][2:],
        "trainer arguments do not exactly match the sealed Exp20 launch",
    )
    for key, value in expected["environment"].items():
        campaign.require(os.environ.get(key) == value, f"trainer environment differs: {key}")


def main() -> int:
    verify_exact_invocation()
    from scripts import train

    # The objective extension is local to this wrapper and cannot affect Exp18/Exp21.
    train.TREEWM_V2_OBJECTIVES = frozenset({*train.TREEWM_V2_OBJECTIVES, OBJECTIVE})
    train.BOUNDED_PILOT_OBJECTIVES = {**train.BOUNDED_PILOT_OBJECTIVES, OBJECTIVE: 25_000}
    train.LATENT_GAUGE_OBJECTIVES = frozenset({*train.LATENT_GAUGE_OBJECTIVES, OBJECTIVE})
    train.main()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except campaign.ContractError as exc:
        print(f"Exp20 trainer entry contract error: {exc}", file=sys.stderr)
        raise SystemExit(2)
