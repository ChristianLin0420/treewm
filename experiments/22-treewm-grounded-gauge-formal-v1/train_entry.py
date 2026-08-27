#!/usr/bin/env python3
"""First fail-closed wrapper for one exact Exp22 launch contract.

After this wrapper execs ``scripts/train.py``, the shared trainer independently
rederives the canonical package launch before constructing data or a model.
"""

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


OBJECTIVE = "treewm_v2_grounded_gauge_formal_v1"


def verify_exact_invocation() -> None:
    """Audit the wrapper invocation against both sealed prerequisite gates."""
    manifest = campaign.load_manifest()
    campaign.verify_protocol_lock()
    # Full raw prerequisite verification is a single pre-submit operation.  Each
    # worker verifies the protocol-bound local receipt plus the exported receipt
    # hash, avoiding repeated reads of historical checkpoints on Lustre.
    campaign.load_prerequisite_bindings(manifest, verify_external_files=False)
    run_name = os.environ.get("TREEWM_RUN_NAME", "")
    matches = [run for run in campaign.expand_runs(manifest) if run.run_name == run_name]
    campaign.require(len(matches) == 1, "TREEWM_RUN_NAME does not identify one sealed Exp22 run")
    expected = campaign.trainer_command(manifest, matches[0], repo_root=REPOSITORY_ROOT)
    campaign.require(
        sys.argv[1:] == expected["argv"][2:],
        "trainer arguments do not exactly match the sealed Exp22 launch",
    )
    for key, value in expected["environment"].items():
        campaign.require(os.environ.get(key) == value, f"trainer environment differs: {key}")


def main() -> int:
    verify_exact_invocation()
    # Replace this process with the direct trainer script so Hydra resolves configs
    # in script mode. The shared trainer then performs the authoritative second,
    # independently rederived canonical-launch authorization check.
    python = campaign.load_manifest()["paths"]["python"]
    trainer = REPOSITORY_ROOT / "scripts" / "train.py"
    os.execve(python, [python, str(trainer), *sys.argv[1:]], os.environ.copy())
    raise AssertionError("os.execve unexpectedly returned")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except campaign.ContractError as exc:
        print(f"Exp22 trainer entry contract error: {exc}", file=sys.stderr)
        raise SystemExit(2)
