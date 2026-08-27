#!/usr/bin/env python3
"""Compose and print the exact direct-Hydra config/argv lock for all Exp23 cells."""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


sys.dont_write_bytecode = True
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
WEIGHT_LEAVES = (
    "executable_prefix_action",
    "executable_prefix_latent",
    "executable_prefix_endpoint",
)


class ConfigAuditError(RuntimeError):
    pass


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _campaign():
    path = PACKAGE_DIR / "campaign.py"
    spec = importlib.util.spec_from_file_location("exp23_campaign_for_config_audit", path)
    if spec is None or spec.loader is None:
        raise ConfigAuditError("cannot load campaign module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _compose(campaign: Any, manifest: dict, lock: dict, cell: Any, protocol: str) -> dict:
    from omegaconf import OmegaConf

    launch = campaign.trainer_command(
        manifest, lock, cell, repo_root=PROJECT_ROOT,
        package_protocol_sha256=protocol,
    )
    trainer_path = Path(launch["argv"][1])
    if trainer_path.resolve() != (PROJECT_ROOT / "scripts/train.py").resolve():
        raise ConfigAuditError(f"cell{cell.index}: trainer path escapes project root")
    normalized_argv = list(launch["argv"])
    normalized_argv[1] = "scripts/train.py"
    environment = os.environ.copy()
    environment.update({str(key): str(value) for key, value in launch["environment"].items()})
    result = subprocess.run(
        [*launch["argv"], "--cfg", "job", "--resolve"],
        cwd=PROJECT_ROOT,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=180,
    )
    if result.returncode != 0:
        raise ConfigAuditError(
            f"cell{cell.index}: direct Hydra composition failed: "
            + (result.stderr or result.stdout)[-4000:]
        )
    config = OmegaConf.to_container(OmegaConf.create(result.stdout), resolve=True)
    if not isinstance(config, dict):
        raise ConfigAuditError(f"cell{cell.index}: resolved config is not an object")
    weights = ((config.get("losses") or {}).get("weights") or {})
    if set(WEIGHT_LEAVES).difference(weights):
        raise ConfigAuditError(f"cell{cell.index}: prefix weights absent")
    if config.get("run_name") is not None or config.get("resume") != "auto":
        raise ConfigAuditError(f"cell{cell.index}: run-name/resume causal parity differs")
    stripped = copy.deepcopy(config)
    for leaf in WEIGHT_LEAVES:
        del stripped["losses"]["weights"][leaf]
    return {
        "index": int(cell.index),
        "setting_id": cell.setting,
        "arm_id": cell.arm,
        "seed": int(cell.seed),
        "resolved_config_sha256": stable_hash(config),
        "resolved_config_without_prefix_weights_sha256": stable_hash(stripped),
        "trainer_argv_repo_relative": normalized_argv,
        "trainer_argv_sha256": stable_hash(normalized_argv),
        "config_override_sha256": launch["hashes"]["config_override_sha256"],
        "resolved_config": config,
    }


def run() -> dict[str, Any]:
    campaign = _campaign()
    manifest = campaign.read_json(campaign.MANIFEST_PATH)
    weight_lock = campaign.read_json(campaign.WEIGHT_LOCK_PATH)
    # The auditor regenerates its own lock, so it validates every upstream
    # contract while deliberately not requiring the superseded output lock.
    campaign.validate_manifest(
        manifest,
        weight_lock,
        PROJECT_ROOT,
        verify_resolved_config_lock=False,
        verify_causal_parity_lock=False,
    )
    protocol = campaign.protocol_sha256(PACKAGE_DIR)
    cells = campaign.expand_matrix(manifest)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(_compose, campaign, manifest, weight_lock, cell, protocol)
            for cell in cells
        ]
        rows = [future.result() for future in futures]
    rows.sort(key=lambda value: value["index"])
    if [row["index"] for row in rows] != list(range(20)):
        raise ConfigAuditError("resolved config matrix is incomplete")
    for setting in campaign.SETTINGS:
        for seed in campaign.SEEDS:
            pair = [
                row for row in rows
                if row["setting_id"] == setting and row["seed"] == seed
            ]
            if [row["arm_id"] for row in pair] != list(campaign.ARMS):
                raise ConfigAuditError(f"{setting}/seed{seed}: pair is missing")
            if pair[0]["resolved_config_without_prefix_weights_sha256"] != pair[1]["resolved_config_without_prefix_weights_sha256"]:
                raise ConfigAuditError(f"{setting}/seed{seed}: configs differ beyond three weights")
    result = {
        "schema_version": 1,
        "status": "frozen_direct_hydra_resolved_config_matrix",
        "audit_id": "treewm_exp23_resolved_config_audit_v1",
        "direct_entrypoint": "scripts/train.py",
        "matrix": rows,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "trainer_code_fingerprint": manifest["core_binding"]["trainer_code_fingerprint"],
        "weight_audit_artifact_sha256": manifest["weight_audit"]["artifact_sha256"],
        "prefix_target_artifact_sha256": manifest["prefix_target_contract"]["artifact_sha256"],
    }
    result["artifact_sha256"] = stable_hash(result)
    return result


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        result = run()
    except Exception as exc:
        print(f"resolved config audit failed: {exc}", file=sys.stderr)
        return 1
    print("EXP23_RESOLVED_CONFIG_AUDIT=" + canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
