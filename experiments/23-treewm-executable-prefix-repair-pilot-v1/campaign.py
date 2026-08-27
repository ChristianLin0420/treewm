#!/usr/bin/env python3
"""Fail-closed scientific and launch identities for the sealed Exp23 pilot."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping


PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]
MANIFEST_PATH = PACKAGE_DIR / "manifest.json"
WEIGHT_LOCK_PATH = PACKAGE_DIR / "weight_audit.lock.json"
PROTOCOL_LOCK_PATH = PACKAGE_DIR / "protocol.sha256"
PREFIX_TARGET_LOCK_PATH = PACKAGE_DIR / "prefix_target.lock.json"
RESOLVED_CONFIG_LOCK_PATH = PACKAGE_DIR / "resolved_config.lock.json"
CAUSAL_PARITY_LOCK_PATH = PACKAGE_DIR / "causal_parity.lock.json"
SETTINGS = (
    "antmaze-large",
    "scene",
    "puzzle-3x3",
    "puzzle-4x4-100m",
    "cube-quadruple-100m",
)
ARMS = ("GS", "GSEP")
SEEDS = (110, 111)
PREFIX_TERMS = (
    "executable_prefix_action",
    "executable_prefix_latent",
    "executable_prefix_endpoint",
)
WEIGHT_KEYS = tuple(f"losses.weights.{name}" for name in PREFIX_TERMS)
CAUSAL_AUDIT_MANIFEST_INPUT_KEYS = (
    "schema_version",
    "campaign_id",
    "method",
    "design",
    "arms",
    "causal_contrast",
    "weight_audit",
    "prefix_target_contract",
    "resolved_config_contract",
    "core_binding",
    "scientific_contract",
    "settings",
    "compatible_v2_recipe_input",
)
PROTOCOL_FILES = (
    "manifest.json",
    "campaign.py",
    "gate.py",
    "weight_audit.py",
    "weight_audit.lock.json",
    "prefix_target_audit.py",
    "prefix_target.lock.json",
    "resolved_config_audit.py",
    "resolved_config.lock.json",
    "causal_parity_audit.py",
    "causal_parity.lock.json",
    "train_entry.py",
    "worker.py",
    "train.slurm",
    "submit.py",
    "cancel.py",
    "report.py",
    "report.slurm",
    "README.md",
    "tests/test_campaign.py",
    "tests/test_gate.py",
    "tests/test_lifecycle.py",
    "tests/test_orchestration.py",
)
SNAPSHOT_IMPORT_FILES = {
    "scripts/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CAMPAIGN_ID = "treewm-executable-prefix-repair-pilot-v1-launch2"
SUPERSEDED_LAUNCH = {
    "campaign_id": "treewm-executable-prefix-repair-pilot-v1",
    "run_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1"
    ),
    "submission_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1/state/submission"
    ),
    "snapshot_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1/state/submission/"
        "source-snapshot/repo"
    ),
    "wandb_project": "treewm-executable-prefix-repair-pilot-v1",
    "status": "aborted_before_submission_contract",
    "source_commit": "85cd77de2d5956944008b4b2b16267858828fa84",
    "source_commit_claimed_by_journal": False,
    "source_commit_evidence": "independent_137_of_137_snapshot_file_byte_match",
    "proof_scope": (
        "The preserved journal does not record git provenance. The source commit is "
        "linked only by an independent comparison proving that all 137 sealed "
        "snapshot files match commit 85cd77de2d5956944008b4b2b16267858828fa84; "
        "the journal proves only its own claim, snapshot, and pre-contract abort records."
    ),
    "package_protocol_sha256": "3e39fb1e6501e3a31e360f569502eb92d1bbb0ad8093c7e747563e50665c2b6e",
    "manifest_canonical_sha256": "25790db3fe7a9a25c6de4f6b8224ccab33751817dc00f1bcf64d25c7fb497e4e",
    "manifest_raw_sha256": "bb841c5a9290465f864407a5d6a8ed927c907e9f4c2b07eeb58d23062a18d0db",
    "snapshot": {
        "inventory_sha256": "6767520819d42ef8866712023211b2f1bc8d236db3ffc836c8dae429b4e5b326",
        "file_count": 137,
        "independently_matched_files": 137,
        "all_files_match": True,
    },
    "claim_token": "0e5e1be5eace176f6a51ec3a3beb7e2579f6914699ac3080f9b2b9d10e4127e9",
    "scientific_output_fingerprint": "786beb527e80f37a8382059309858437df25ec867c5eb3c1e1b1fe1064b62cd4",
    "journal_sha256": {
        "0000_CLAIMED.json": "e9607ea26d07af65b670f2b70abceee9b3f45460159f28a76cd1ec6807a195d4",
        "0001_SNAPSHOT_SEALED.json": "94945e37ad3b363c04ab89c14230c06a62b30067d28f5c49df35854690de1439",
        "9998_OUTER_ABORTED.json": "27316555fd705a63bbd24521cadadf7d6d6b51b177d9c28622d654c99da16f02",
    },
    "submission_sha256": None,
    "known_job_ids": [],
    "submission_receipt_committed": False,
    "scientific_run_started": False,
    "checkpoint_created": False,
    "wandb_run_created": False,
    "optimizer_updates": 0,
    "results_consumed": False,
    "checkpoints_consumed": False,
    "reuse_allowed": False,
    "resume_allowed": False,
}


class ContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class Cell:
    index: int
    setting_index: int
    arm_index: int
    seed_index: int
    setting: str
    env_config: str
    arm: str
    seed: int
    run_name: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_pairs)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return stable_hash(manifest)


def _override(name: str, value: object) -> str:
    if isinstance(value, bool):
        rendered = str(value).lower()
    elif value is None:
        rendered = "null"
    elif isinstance(value, (list, tuple)):
        rendered = "[" + ",".join(
            str(item).lower() if isinstance(item, bool) else str(item)
            for item in value
        ) + "]"
    else:
        rendered = str(value)
    return f"{name}={rendered}"


def run_directory(manifest: Mapping[str, Any], cell: Cell) -> Path:
    return Path(manifest["paths"]["run_root"]) / cell.setting / "treewm" / cell.run_name


def recipe_root(manifest: Mapping[str, Any], setting_id: str) -> Path:
    return Path(manifest["paths"]["compatible_contract_root"]) / "future-recipes" / setting_id


def load_compatible_input(
    manifest: Mapping[str, Any], setting_id: str, *, verify_files: bool = False
) -> dict[str, Any]:
    setting = next(row for row in manifest["settings"] if row["id"] == setting_id)
    path = Path(manifest["paths"]["compatible_contract_root"]) / "data" / f"{setting_id}.json"
    contract = read_json(path)
    claimed = contract.get("contract_sha256")
    body = dict(contract)
    body.pop("contract_sha256", None)
    require(claimed == stable_hash(body) == setting["input_contract_sha256"], f"{setting_id}: input contract differs")
    legacy = manifest["compatible_v2_recipe_input"]
    expected = {
        "campaign_id": legacy["campaign_id"],
        "objective_version": legacy["objective_version"],
        "campaign_protocol_sha256": legacy["campaign_protocol_sha256"],
        "setting_id": setting_id,
        "dataset_kind": setting["dataset_kind"],
        "calibration_sha256": setting["calibration_sha256"],
        "future_recipe_sha256": setting["future_recipe_sha256"],
        "raw_cache_read_only": True,
    }
    for key, value in expected.items():
        require(contract.get(key) == value, f"{setting_id}: compatible {key} differs")
    composite = read_json(recipe_root(manifest, setting_id) / "manifest.json")
    require(composite.get("recipe_sha256") == setting["future_recipe_sha256"], f"{setting_id}: recipe differs")
    if verify_files:
        from treewm.data.future_recipe import validate_recipe_manifest

        validate_recipe_manifest(
            recipe_root(manifest, setting_id),
            composite,
            expected_source_manifest_sha256=contract["data_manifest_sha256"],
            expected_normalizer_sha256=contract["normalizer_sha256"],
            expected_calibration_sha256=contract["calibration_sha256"],
            expected_thresholds=contract["chosen_thresholds"],
            expected_train_manifest_sha256=contract["train_manifest_sha256"],
            expected_validation_manifest_sha256=contract["validation_manifest_sha256"],
            expected_code_sha256=legacy["recipe_code_sha256"],
            expected_runtime_sha256=legacy["recipe_runtime_sha256"],
            verify_file_hash=True,
        )
    return contract


def source_contract(repo_root: str | Path = REPOSITORY_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from treewm.utils.provenance import runtime_fingerprint, trainer_code_fingerprint

    source = trainer_code_fingerprint(root)
    runtime = runtime_fingerprint()
    return {
        "source_sha256": source["manifest_sha256"],
        "source_files": source["files"],
        "runtime_sha256": runtime["sha256"],
        "runtime": runtime,
    }


def protocol_sha256(root: str | Path = PACKAGE_DIR) -> str:
    package = Path(root).resolve()
    require(len(PROTOCOL_FILES) == len(set(PROTOCOL_FILES)), "duplicate protocol file")
    files: dict[str, str] = {}
    for relative in PROTOCOL_FILES:
        path = package / relative
        require(path.is_file() and not path.is_symlink(), f"missing/symlink protocol file: {relative}")
        files[relative] = file_sha256(path)
    return stable_hash({"schema_version": 1, "files": files})


def validate_snapshot_import_files(repo_root: str | Path = REPOSITORY_ROOT) -> None:
    expected = {
        "scripts/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    }
    require(SNAPSHOT_IMPORT_FILES == expected, "snapshot import inventory differs")
    root = Path(repo_root).resolve()
    for relative, expected_sha256 in SNAPSHOT_IMPORT_FILES.items():
        candidate = root / relative
        require(
            candidate.is_file()
            and not candidate.is_symlink()
            and candidate.resolve().is_relative_to(root),
            f"snapshot import is unavailable/symlinked: {relative}",
        )
        require(
            file_sha256(candidate) == expected_sha256,
            f"snapshot import bytes differ: {relative}",
        )


def verify_protocol_lock(root: str | Path = PACKAGE_DIR) -> str:
    package = Path(root).resolve()
    try:
        locked = (package / "protocol.sha256").read_text(encoding="ascii").strip()
    except OSError as exc:
        raise ContractError(f"protocol lock unavailable: {exc}") from exc
    live = protocol_sha256(package)
    require(SHA256.fullmatch(locked) is not None and locked == live, "protocol lock stale")
    return live


def expand_matrix(manifest: Mapping[str, Any]) -> list[Cell]:
    settings = manifest["settings"]
    result: list[Cell] = []
    for setting_index, setting in enumerate(settings):
        for arm_index, arm in enumerate(ARMS):
            for seed_index, seed in enumerate(SEEDS):
                index = ((setting_index * len(ARMS)) + arm_index) * len(SEEDS) + seed_index
                require(index == len(result), "matrix mapping is not contiguous")
                result.append(
                    Cell(
                        index=index,
                        setting_index=setting_index,
                        arm_index=arm_index,
                        seed_index=seed_index,
                        setting=str(setting["id"]),
                        env_config=str(setting["env_config"]),
                        arm=arm,
                        seed=seed,
                        run_name=f"exp23-launch2-{setting['id']}-arm{arm.lower()}-seed{seed}",
                    )
                )
    return result


def cell_overrides(cell: Cell, manifest: Mapping[str, Any], lock: Mapping[str, Any]) -> dict[str, Any]:
    """Return declarative Hydra leaves; never execute them."""
    arm = next(row for row in manifest["arms"] if row["id"] == cell.arm)
    weights = arm["executable_prefix_weights"]
    bounds = lock["action_bounds"][cell.setting]
    return {
        "experiment": manifest["method"]["experiment_config"],
        "env": cell.env_config,
        "arm": "treewm",
        "seed": cell.seed,
        "objective_version": manifest["method"]["objective_version"],
        "train.steps": manifest["scientific_contract"]["optimizer_updates"],
        "future_sets.executable_prefix_steps": 4,
        "losses.enabled.executable_prefix_action": True,
        "losses.enabled.executable_prefix_latent": True,
        "losses.enabled.executable_prefix_endpoint": True,
        "losses.weights.executable_prefix_action": weights["action"],
        "losses.weights.executable_prefix_latent": weights["latent"],
        "losses.weights.executable_prefix_endpoint": weights["endpoint"],
        "losses.executable_action_lower_bound": bounds["lower"],
        "losses.executable_action_upper_bound": bounds["upper"],
        "planner.action_lower_bound": bounds["lower"],
        "planner.action_upper_bound": bounds["upper"],
        "+campaign_id": manifest["campaign_id"],
        "+weight_audit_artifact_sha256": manifest["weight_audit"]["artifact_sha256"],
    }


def scientific_overrides(
    cell: Cell,
    manifest: Mapping[str, Any],
    lock: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> list[str]:
    """Return the complete ordered Hydra scientific config for one cell.

    Unique cell/run labels are deliberately absent.  Within a setting/seed pair the
    only arm-dependent leaves are the three audited executable-prefix weights.
    """
    setting = next(row for row in manifest["settings"] if row["id"] == cell.setting)
    scientific = manifest["scientific_contract"]
    future = scientific["future_sets"]
    chosen = contract["chosen_thresholds"]
    bounds = lock["action_bounds"][cell.setting]
    weights = next(row for row in manifest["arms"] if row["id"] == cell.arm)[
        "executable_prefix_weights"
    ]
    values: list[tuple[str, object]] = [
        ("env", cell.env_config),
        ("experiment", manifest["method"]["experiment_config"]),
        ("arm", "treewm"),
        ("objective_version", manifest["method"]["objective_version"]),
        ("seed", cell.seed),
        ("train.steps", scientific["optimizer_updates"]),
        ("train.scheduler_total_steps", scientific["scheduler_total_steps"]),
        ("train.ckpt_every", scientific["checkpoint_every_updates"]),
        ("train.val_every", scientific["validation_every_updates"]),
        ("train.diag_every", scientific["diagnostics_every_updates"]),
        ("train.eval_every", scientific["periodic_evaluation_every_updates"]),
        ("train.log_every", scientific["training_telemetry_every_updates"]),
        ("train.validation_sample_seed", scientific["validation_sample_seed"]),
        ("train.max_train_anchors", setting["published_union_train_anchors"]),
        ("train.max_val_anchors", setting["published_union_validation_anchors"]),
        ("train.num_workers", scientific["data_loader_workers"]),
        ("train.lr", scientific["world_lr"]),
        ("train.weight_decay", scientific["world_weight_decay"]),
        ("train.gradient_checkpointing", scientific["gradient_checkpointing"]),
        ("train.separate_gain_grad_clip", scientific["separate_gain_grad_clip"]),
        ("train.separate_branch_transformer_grad_clip", scientific["separate_branch_transformer_grad_clip"]),
        ("train.world_grad_clip", scientific["world_grad_clip"]),
        ("train.branch_transformer_grad_clip", scientific["branch_transformer_grad_clip"]),
        ("train.gain_grad_clip", scientific["gain_grad_clip"]),
        ("train.gain_loss_every", scientific["gain_loss_every"]),
        ("train.gain_lr", scientific["gain_lr"]),
        ("train.gain_weight_decay", scientific["gain_weight_decay"]),
        ("train.gain_training_scorers", scientific["gain_training_scorers"]),
        ("train.viz_every", scientific["visualization_every_updates"]),
        ("train.viz_every_early", scientific["visualization_every_early_updates"]),
        ("train.viz_early_until", scientific["visualization_early_until_updates"]),
        ("model.dropout", scientific["model_dropout"]),
        ("model.max_depth", scientific["model_max_depth"]),
        ("tree.max_depth", scientific["tree_max_depth"]),
        ("tree.node_budget", manifest["method"]["node_budget"]),
        ("tree.keep_threshold", scientific["keep_threshold"]),
        ("tree.scorer", scientific["tree_scorer"]),
        ("model.branch_factor", manifest["method"]["branch_factor"]),
        ("planner.decoded_metric", "domain_raw"),
        ("planner.execute_mode", "clipped"),
        ("planner.execute_steps", 4),
        ("planner.max_env_steps", setting["max_episode_steps"]),
        ("planner.require_first_edge_improvement", True),
        ("planner.min_first_edge_improvement", 0.0),
    ]
    values.extend((f"future_sets.{key}", value) for key, value in future.items() if key not in {"recipe_anchor_policy"})
    values.extend(
        [
            ("future_sets.recipe_anchor_policy", "published_union"),
            ("future_sets.relative_endpoints", setting["relative_endpoints"]),
            ("future_sets.retrieval_radius", chosen["retrieval_radius"]),
            ("future_sets.displacement_threshold", chosen["displacement_threshold"]),
            ("future_sets.cluster_threshold", chosen["cluster_threshold"]),
            ("+env.task_metric_dims", setting["task_metric_dims"]),
            ("losses.keep_balance", True),
            ("losses.enabled.multistep", True),
            ("losses.weights.multistep", scientific["multistep_weight"]),
            ("losses.scheduled_sampling_p", scientific["scheduled_sampling_p"]),
            ("losses.scheduled_sampling_warmup", scientific["scheduled_sampling_warmup"]),
            ("losses.scheduled_sampling_granularity", scientific["scheduled_sampling_granularity"]),
            ("losses.multistep_transition_mode", "grounded_execution_v2"),
            ("losses.grounded_select_action_weight", scientific["grounded_select_weights"]["action"]),
            ("losses.grounded_select_endpoint_weight", scientific["grounded_select_weights"]["endpoint"]),
            ("losses.grounded_select_horizon_weight", scientific["grounded_select_weights"]["horizon"]),
            ("losses.grounded_loss_latent_weight", scientific["grounded_loss_weights"]["latent"]),
            ("losses.grounded_loss_action_weight", scientific["grounded_loss_weights"]["action"]),
            ("losses.grounded_loss_horizon_weight", scientific["grounded_loss_weights"]["horizon"]),
            ("losses.grounded_loss_endpoint_weight", scientific["grounded_loss_weights"]["endpoint"]),
            ("losses.grounded_detach_self_fed_parent", scientific["grounded_detach_self_fed_parent"]),
            ("losses.multistep_depth_weights", scientific["multistep_depth_weights"]),
            ("losses.enabled.latent_gauge", scientific["latent_gauge_enabled"]),
            ("losses.weights.latent_gauge", scientific["latent_gauge_weight"]),
            ("losses.latent_gauge_epsilon", scientific["latent_gauge_epsilon"]),
            ("losses.latent_gauge_min_reference_scale", scientific["latent_gauge_min_reference_scale"]),
            ("losses.enabled.executable_prefix_action", True),
            ("losses.enabled.executable_prefix_latent", True),
            ("losses.enabled.executable_prefix_endpoint", True),
            ("losses.weights.executable_prefix_action", weights["action"]),
            ("losses.weights.executable_prefix_latent", weights["latent"]),
            ("losses.weights.executable_prefix_endpoint", weights["endpoint"]),
            ("losses.executable_action_lower_bound", bounds["lower"]),
            ("losses.executable_action_upper_bound", bounds["upper"]),
            ("planner.action_lower_bound", bounds["lower"]),
            ("planner.action_upper_bound", bounds["upper"]),
            ("eval.task_split", scientific["task_split"]),
            ("eval.episodes_per_task", scientific["periodic_episodes_per_task"]),
            ("eval.final_episodes_per_task", scientific["final_episodes_per_task"]),
            ("eval.seed", scientific["evaluation_seed"]),
            ("+campaign_id", manifest["campaign_id"]),
            ("+campaign_input_contract_sha256", contract["contract_sha256"]),
            ("+campaign_calibration_sha256", contract["calibration_sha256"]),
            ("+campaign_future_recipe_sha256", contract["future_recipe_sha256"]),
            ("+campaign_compatible_recipe_code_sha256", manifest["compatible_v2_recipe_input"]["recipe_code_sha256"]),
            ("+weight_audit_artifact_sha256", manifest["weight_audit"]["artifact_sha256"]),
            ("+prefix_target_artifact_sha256", manifest["prefix_target_contract"]["artifact_sha256"]),
            ("run_root", manifest["paths"]["run_root"]),
            ("run_name", None),
            ("resume", "auto"),
        ]
    )
    return [_override(name, value) for name, value in values]


def actual_final_evaluation_rows(manifest: Mapping[str, Any]) -> list[dict[str, int]]:
    scientific = manifest["scientific_contract"]
    rows = [
        {
            "task_index": task_index,
            "task_id": int(task_id),
            "episode_index": episode_index,
            "episode_seed": int(scientific["evaluation_seed"]) + 1000 * task_index + episode_index,
        }
        for task_index, task_id in enumerate(scientific["task_ids"])
        for episode_index in range(int(scientific["final_episodes_per_task"]))
    ]
    require(len(rows) == 25, "terminal evaluation row count differs")
    return rows


def trainer_command(
    manifest: Mapping[str, Any],
    lock: Mapping[str, Any],
    cell: Cell,
    *,
    repo_root: str | Path = REPOSITORY_ROOT,
    package_protocol_sha256: str | None = None,
    verify_recipe_files: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    protocol = package_protocol_sha256 or verify_protocol_lock(root / "experiments/23-treewm-executable-prefix-repair-pilot-v1")
    contract = load_compatible_input(manifest, cell.setting, verify_files=verify_recipe_files)
    source = source_contract(root)
    overrides = scientific_overrides(cell, manifest, lock, contract)
    config_sha = stable_hash({"schema_version": 1, "overrides": overrides})
    output = run_directory(manifest, cell)
    run_protocol = stable_hash(
        {
            "schema_version": 1,
            "campaign_id": manifest["campaign_id"],
            "package_protocol_sha256": protocol,
            "source_sha256": source["source_sha256"],
            "runtime_sha256": source["runtime_sha256"],
            "config_sha256": config_sha,
            "input_contract_sha256": contract["contract_sha256"],
            "data_manifest_sha256": contract["data_manifest_sha256"],
            "validation_manifest_sha256": contract["validation_manifest_sha256"],
            "future_recipe_sha256": contract["future_recipe_sha256"],
            "weight_audit_artifact_sha256": manifest["weight_audit"]["artifact_sha256"],
            "prefix_target_artifact_sha256": manifest["prefix_target_contract"]["artifact_sha256"],
            "resolved_config_artifact_sha256": manifest["resolved_config_contract"]["artifact_sha256"],
            "causal_parity_artifact_sha256": manifest["causal_parity_contract"]["artifact_sha256"],
        }
    )
    argv = [
        manifest["paths"]["python"],
        str(root / "scripts/train.py"),
        *overrides,
        _override("hydra.run.dir", output / "hydra"),
        _override("hydra.job.chdir", False),
    ]
    wandb_id = stable_hash(
        {"campaign_id": manifest["campaign_id"], "cell": asdict(cell)}
    )[:32]
    environment = {
        "TREEWM_PROTOCOL_SHA256": run_protocol,
        "TREEWM_CODE_SHA256": source["source_sha256"],
        "TREEWM_ACTIVE_SOURCE_SHA256": source["source_sha256"],
        "TREEWM_RUNTIME_SHA256": source["runtime_sha256"],
        "TREEWM_RECIPE_CODE_SHA256": manifest["compatible_v2_recipe_input"]["recipe_code_sha256"],
        "TREEWM_RECIPE_RUNTIME_SHA256": manifest["compatible_v2_recipe_input"]["recipe_runtime_sha256"],
        "TREEWM_CONFIG_SHA256": config_sha,
        "TREEWM_DATA_SHA256": contract["data_manifest_sha256"],
        "TREEWM_CALIBRATION_SHA256": contract["calibration_sha256"],
        "TREEWM_FUTURE_RECIPE_SHA256": contract["future_recipe_sha256"],
        "TREEWM_CAUSAL_PARITY_SHA256": manifest["causal_parity_contract"]["artifact_sha256"],
        "TREEWM_RESOLVED_CONFIG_SHA256": manifest["resolved_config_contract"]["artifact_sha256"],
        "TREEWM_DATA_CONTRACT_SHA256": contract["contract_sha256"],
        "TREEWM_DATA_ROOT": manifest["paths"]["data_root"],
        "TREEWM_CACHE": manifest["paths"]["raw_cache_root"],
        "TREEWM_FUTURE_RECIPE_ROOT": str(recipe_root(manifest, cell.setting)),
        "TREEWM_EVALUATION_SEED_PROTOCOL_SHA256": manifest["scientific_contract"]["evaluation_seed_protocol_sha256"],
        "TREEWM_RUN_NAME": cell.run_name,
        "WANDB_PROJECT": manifest["logging"]["wandb_project"],
        "WANDB_RUN_GROUP": manifest["logging"]["wandb_group"],
        "WANDB_RUN_ID": wandb_id,
        "WANDB_MODE": manifest["logging"]["wandb_mode"],
        "OMP_NUM_THREADS": str(manifest["scientific_contract"]["loader_thread_limit"]),
        "MKL_NUM_THREADS": str(manifest["scientific_contract"]["loader_thread_limit"]),
        "OPENBLAS_NUM_THREADS": str(manifest["scientific_contract"]["loader_thread_limit"]),
        "MUJOCO_GL": manifest["execution"]["sealed_trainer_environment"]["MUJOCO_GL"],
        "XLA_PYTHON_CLIENT_PREALLOCATE": manifest["execution"]["sealed_trainer_environment"]["XLA_PYTHON_CLIENT_PREALLOCATE"],
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    launch: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "cell": {**asdict(cell), "run_directory": str(output), "wandb_id": wandb_id},
        "argv": argv,
        "environment": environment,
        "hashes": {
            "manifest_sha256": manifest_sha256(manifest),
            "source_sha256": source["source_sha256"],
            "runtime_sha256": source["runtime_sha256"],
            "package_protocol_sha256": protocol,
            "config_override_sha256": config_sha,
            "run_protocol_sha256": run_protocol,
            "input_contract_sha256": contract["contract_sha256"],
            "data_manifest_sha256": contract["data_manifest_sha256"],
            "normalizer_sha256": contract["normalizer_sha256"],
            "train_manifest_sha256": contract["train_manifest_sha256"],
            "validation_manifest_sha256": contract["validation_manifest_sha256"],
            "calibration_sha256": contract["calibration_sha256"],
            "future_recipe_sha256": contract["future_recipe_sha256"],
            "weight_audit_artifact_sha256": manifest["weight_audit"]["artifact_sha256"],
            "prefix_target_artifact_sha256": manifest["prefix_target_contract"]["artifact_sha256"],
            "resolved_config_artifact_sha256": manifest["resolved_config_contract"]["artifact_sha256"],
            "causal_parity_artifact_sha256": manifest["causal_parity_contract"]["artifact_sha256"],
            "actual_final_evaluation_rows_sha256": stable_hash(actual_final_evaluation_rows(manifest)),
        },
    }
    launch["launch_sha256"] = stable_hash(launch)
    return launch


def _validate_lock(lock: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    require(lock.get("schema_version") == 1 and lock.get("status") == "frozen", "weight lock not frozen")
    identity = lock["result_identity"]
    for key in ("artifact_sha256", "rows_sha256", "summary_sha256"):
        require(SHA256.fullmatch(str(identity.get(key, ""))) is not None, f"invalid audit {key}")
        require(identity[key] == manifest["weight_audit"][key], f"manifest/audit {key} differs")
    require(identity.get("row_count") == 40, "audit row count differs")
    contract = lock["contract"]
    require(contract["settings"] == list(SETTINGS), "audit settings differ")
    require(contract["regimes"] == ["exp20_gs_exact_5000", "scratch_initialization"], "audit regimes differ")
    require(contract["checkpoint_seeds"] == [108, 109] and contract["scratch_seeds"] == [230, 231], "audit seeds differ")
    require(contract["fixed_batches_per_setting_regime"] == 2 and contract["batch_size"] == 16, "audit batch design differs")
    require(contract["groups"] == ["branch_transformer", "world_rest"], "audit groups differ")
    require(contract["per_component_median_base_gradient_fraction_max"] == 0.03, "component gradient budget differs")
    require(contract["aggregate_every_row_group_base_gradient_fraction_max"] == 0.10, "aggregate gradient budget differs")
    require(lock["derived"]["post_scale_max_aggregate_ratio"] <= 0.10, "audited tuple exceeds aggregate budget")
    expected = manifest["arms"][1]["executable_prefix_weights"]
    actual = lock["derived"]["weights"]
    runtime = lock["derived"]["audit_runtime_float_weights"]
    require(
        actual == {
            "executable_prefix_action": expected["action"],
            "executable_prefix_latent": expected["latent"],
            "executable_prefix_endpoint": expected["endpoint"],
        },
        "treatment weights differ from audit",
    )
    require(
        all(0.0 <= runtime[name] - actual[name] <= math.ulp(runtime[name]) for name in actual),
        "canonical audit decimals are not conservative one-ULP renderings",
    )
    require(len(lock["checkpoint_sha256"]) == 10, "checkpoint hash inventory incomplete")
    require(len(lock["batch_sha256"]) == 20, "batch hash inventory incomplete")
    external_keys = {
        "exp20/manifest.json",
        *(
            f"{setting}/seed{seed}/GAUGE_PILOT_V2_LAUNCH.json"
            for setting in SETTINGS
            for seed in (108, 109)
        ),
    }
    require(
        set(lock.get("external_input_sha256") or {}) == external_keys,
        "external input hash inventory differs",
    )
    for inventory in (
        lock["checkpoint_sha256"],
        lock["external_input_sha256"],
        lock["batch_sha256"],
        lock["source_sha256"],
    ):
        require(all(SHA256.fullmatch(str(value)) for value in inventory.values()), "invalid audit hash")
    require(file_sha256(PACKAGE_DIR / "weight_audit.py") == lock["source_sha256"]["audit"], "auditor source differs")
    api = lock["fail_closed_api_binding"]
    require(api["effective_tree_config_required"] is True, "tree config is not fail closed")
    require("tree_config_for" in api["audit_call"], "audit did not bind effective tree config")
    for setting in SETTINGS:
        bounds = lock["action_bounds"][setting]
        require(bounds["action_dim"] > 0 and bounds["lower"] < bounds["upper"], f"{setting}: bounds invalid")
        require(SHA256.fullmatch(bounds["lower_sha256"]) is not None, f"{setting}: lower hash invalid")
        require(SHA256.fullmatch(bounds["upper_sha256"]) is not None, f"{setting}: upper hash invalid")


def _validate_prefix_target_lock(manifest: Mapping[str, Any]) -> dict[str, Any]:
    lock = read_json(PREFIX_TARGET_LOCK_PATH)
    claimed = lock.get("artifact_sha256")
    body = dict(lock)
    body.pop("artifact_sha256", None)
    require(claimed == stable_hash(body), "prefix-target artifact hash differs")
    binding = manifest["prefix_target_contract"]
    require(claimed == binding["artifact_sha256"], "manifest prefix-target artifact differs")
    require(file_sha256(PACKAGE_DIR / binding["audit_source"]) == binding["source_sha256"] == lock["source_sha256"], "prefix-target auditor source differs")
    require(lock["weight_audit_artifact_sha256"] == manifest["weight_audit"]["artifact_sha256"], "prefix-target/weight audit binding differs")
    weight_lock = read_json(WEIGHT_LOCK_PATH)
    require(
        lock.get("external_input_sha256") == weight_lock.get("external_input_sha256"),
        "prefix-target external-input binding differs",
    )
    require(set(lock["settings"]) == set(SETTINGS), "prefix-target setting coverage differs")
    for setting, row in lock["settings"].items():
        require(row["setting_id"] == setting, f"{setting}: prefix-target setting label differs")
        require(row["anchor_count"] == binding["validation_anchor_count_per_setting"] == 5120, f"{setting}: prefix-target anchor count differs")
        require(row["batch_size"] == binding["batch_size"] == 256 and row["num_batches"] == binding["validation_batches"] == 20, f"{setting}: fixed validation shape differs")
        require(row["all_anchors_have_match"] is True and row["matched_branch_count"] >= row["anchor_count"], f"{setting}: incomplete prefix targets")
        require(row["prefix_length_histogram"] == {"1": 0, "2": 0, "3": 0, "4": row["matched_branch_count"]}, f"{setting}: impossible sealed prefix lengths")
        horizon_histogram = row.get("logged_selected_horizon_histogram")
        require(
            isinstance(horizon_histogram, dict)
            and set(horizon_histogram) == {"4", "8", "16", "32", "64"}
            and all(type(value) is int and value >= 0 for value in horizon_histogram.values())
            and sum(horizon_histogram.values()) == row["matched_branch_count"]
            and SHA256.fullmatch(str(row.get("sorted_logged_selected_horizons_sha256", ""))) is not None,
            f"{setting}: logged continuation horizon evidence differs",
        )
        require(row["prefix_action_step_count"] == 4 * row["matched_branch_count"], f"{setting}: action-step denominator differs")
        require(row["prefix_action_scalar_count"] == row["prefix_action_step_count"] * row["action_dim"], f"{setting}: action-scalar denominator differs")
        require(all(SHA256.fullmatch(str(value)) for key, value in row.items() if key.endswith("sha256")), f"{setting}: malformed prefix-target hash")
    return lock


def _validate_resolved_config_lock(manifest: Mapping[str, Any]) -> dict[str, Any]:
    lock = read_json(RESOLVED_CONFIG_LOCK_PATH)
    claimed = lock.get("artifact_sha256")
    body = dict(lock)
    body.pop("artifact_sha256", None)
    require(claimed == stable_hash(body), "resolved-config artifact hash differs")
    binding = manifest["resolved_config_contract"]
    require(claimed == binding["artifact_sha256"], "manifest resolved-config artifact differs")
    require(file_sha256(PACKAGE_DIR / binding["audit_source"]) == binding["source_sha256"] == lock["source_sha256"], "resolved-config auditor source differs")
    require(lock["direct_entrypoint"] == binding["direct_entrypoint"] == "scripts/train.py", "direct trainer entrypoint differs")
    require(lock["trainer_code_fingerprint"] == manifest["core_binding"]["trainer_code_fingerprint"], "resolved config/core differs")
    rows = lock.get("matrix") or []
    require(len(rows) == binding["cell_count"] == 20 and [row.get("index") for row in rows] == list(range(20)), "resolved-config matrix differs")
    cells = expand_matrix(manifest)
    for cell, row in zip(cells, rows, strict=True):
        require((row["setting_id"], row["arm_id"], row["seed"]) == (cell.setting, cell.arm, cell.seed), f"cell{cell.index}: resolved-config identity differs")
        require(stable_hash(row["resolved_config"]) == row["resolved_config_sha256"], f"cell{cell.index}: resolved config hash differs")
        argv = row.get("trainer_argv_repo_relative")
        require(
            isinstance(argv, list)
            and len(argv) >= 3
            and argv[0] == manifest["paths"]["python"]
            and argv[1] == "scripts/train.py"
            and stable_hash(argv) == row["trainer_argv_sha256"],
            f"cell{cell.index}: repo-relative trainer argv differs",
        )
        require(row["resolved_config"].get("run_name") is None and row["resolved_config"].get("resume") == "auto", f"cell{cell.index}: run-name/resume parity differs")
    for setting in SETTINGS:
        for seed in SEEDS:
            pair = [row for row in rows if row["setting_id"] == setting and row["seed"] == seed]
            require(pair[0]["resolved_config_without_prefix_weights_sha256"] == pair[1]["resolved_config_without_prefix_weights_sha256"], f"{setting}/seed{seed}: resolved configs differ beyond weights")
    return lock


def _validate_causal_parity_lock(manifest: Mapping[str, Any]) -> dict[str, Any]:
    lock = read_json(CAUSAL_PARITY_LOCK_PATH)
    claimed = lock.get("artifact_sha256")
    body = dict(lock)
    body.pop("artifact_sha256", None)
    require(claimed == stable_hash(body), "causal-parity artifact hash differs")
    binding = manifest["causal_parity_contract"]
    require(claimed == binding["artifact_sha256"], "manifest causal-parity artifact differs")
    require(
        file_sha256(PACKAGE_DIR / binding["audit_source"])
        == binding["source_sha256"]
        == lock["source_sha256"],
        "causal-parity auditor source differs",
    )
    audit_manifest_input = {
        key: manifest[key] for key in CAUSAL_AUDIT_MANIFEST_INPUT_KEYS
    }
    require(
        stable_hash(audit_manifest_input)
        == binding["audit_manifest_input_sha256"]
        == lock["audit_manifest_input_sha256"],
        "causal-parity manifest input differs",
    )
    require(lock.get("package_protocol_claimed") is False, "causal audit claims a circular protocol")
    require(
        lock["runtime_sha256"] == binding["runtime_sha256"]
        and lock["runtime_sha256"] == manifest["compatible_v2_recipe_input"]["recipe_runtime_sha256"],
        "causal-parity runtime differs",
    )
    require(
        lock["trainer_code_fingerprint"] == manifest["core_binding"]["trainer_code_fingerprint"]
        and lock["weight_audit_artifact_sha256"] == manifest["weight_audit"]["artifact_sha256"]
        and lock["prefix_target_artifact_sha256"] == manifest["prefix_target_contract"]["artifact_sha256"]
        and lock["resolved_config_artifact_sha256"] == manifest["resolved_config_contract"]["artifact_sha256"],
        "causal-parity upstream binding differs",
    )
    require(
        lock["live_output_fingerprint_before"] == lock["live_output_fingerprint_after"],
        "causal audit changed live output metadata",
    )
    expected_fields = [
        "launch_without_allowed_deltas_sha256",
        "resolved_config_without_prefix_weights_sha256",
        "controlled_cpu_scratch_parameters_sha256",
        "data_identity_sha256",
        "sampler_identity_sha256",
        "controlled_cpu_pre_forward_rng_sha256",
        "fixed_validation_batch_sha256",
        "raw_prefix_targets_sha256",
        "raw_prefix_artifacts_sha256",
        "raw_prefix_telemetry_sha256",
        "raw_prefix_values",
    ]
    expected_pairs = [
        (setting, seed) for setting in SETTINGS for seed in SEEDS
    ]
    rows = lock.get("pairs") or []
    require(
        len(rows) == 10
        and [(row.get("setting_id"), row.get("seed")) for row in rows]
        == expected_pairs,
        "causal-parity pair matrix differs",
    )
    expected_weights = {
        "executable_prefix_action": manifest["arms"][1]["executable_prefix_weights"]["action"],
        "executable_prefix_latent": manifest["arms"][1]["executable_prefix_weights"]["latent"],
        "executable_prefix_endpoint": manifest["arms"][1]["executable_prefix_weights"]["endpoint"],
    }
    config_rows = read_json(RESOLVED_CONFIG_LOCK_PATH)["matrix"]
    for row in rows:
        require(row.get("parity_fields") == expected_fields, "causal parity-field set differs")
        require(len(row["parity_fields"]) == len(set(row["parity_fields"])), "causal parity fields duplicate")
        require(
            row.get("allowed_environment_differences")
            == ["TREEWM_CONFIG_SHA256", "TREEWM_PROTOCOL_SHA256", "TREEWM_RUN_NAME", "WANDB_RUN_ID"],
            "causal launch environment deltas differ",
        )
        arms = row.get("arms") or {}
        require(list(arms) == list(ARMS), "causal arm ordering differs")
        require(
            all(arms["GS"][field] == arms["GSEP"][field] for field in expected_fields),
            "causal parity value differs",
        )
        setting, seed = row["setting_id"], row["seed"]
        for arm in ARMS:
            cell = next(
                value for value in expand_matrix(manifest)
                if value.setting == setting and value.seed == seed and value.arm == arm
            )
            require(
                arms[arm]["resolved_config_sha256"]
                == config_rows[cell.index]["resolved_config_sha256"],
                "causal/resolved config identity differs",
            )
            require(arms[arm]["controlled_cpu_parameters_unchanged"] is True, "causal audit mutated parameters")
        require(
            set(arms["GS"]["effective_prefix_weights"].values()) == {0.0}
            and set(arms["GS"]["effective_prefix_values"].values()) == {0.0},
            "causal control is not monitor-only",
        )
        require(arms["GSEP"]["effective_prefix_weights"] == expected_weights, "causal treatment weights differ")
    return lock


def _validate_core(manifest: Mapping[str, Any], lock: Mapping[str, Any], repo: Path) -> None:
    binding = manifest["core_binding"]
    paths = {
        "trainer_sha256": repo / "scripts/train.py",
        "executable_loss_sha256": repo / "treewm/losses/executable_prefix.py",
        "action_projection_sha256": repo / "treewm/planning/action_execution.py",
        "objective_config_sha256": repo / "configs/experiment/treewm_v2_grounded_executable_prefix_pilot_v1.yaml",
    }
    for key, path in paths.items():
        require(file_sha256(path) == binding[key], f"core source drift: {path.relative_to(repo)}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from treewm.utils.provenance import trainer_code_fingerprint

    live = trainer_code_fingerprint(repo)["manifest_sha256"]
    require(live == binding["trainer_code_fingerprint"], "trainer code fingerprint drift")
    require(live == lock["source_sha256"]["trainer_code_fingerprint"], "audit/core fingerprint differs")


def validate_manifest(
    manifest: Mapping[str, Any],
    lock: Mapping[str, Any],
    repo: str | Path = REPOSITORY_ROOT,
    *,
    verify_resolved_config_lock: bool = True,
    verify_causal_parity_lock: bool = True,
) -> None:
    require(manifest.get("schema_version") == 1, "manifest schema differs")
    require(manifest.get("campaign_id") == CAMPAIGN_ID, "campaign ID differs")
    require(manifest.get("status") == "sealed_launch_ready_unsubmitted", "package launch state differs")
    require(manifest.get("formal_validation") is False, "pilot is marked formal")
    require(manifest["package_policy"]["launch_surface"] is True, "launch surface disabled")
    require(manifest.get("superseded_launch") == SUPERSEDED_LAUNCH, "superseded launch identity differs")
    require(
        manifest["paths"]["run_root"] != SUPERSEDED_LAUNCH["run_root"]
        and manifest["paths"]["wandb_project"] != SUPERSEDED_LAUNCH["wandb_project"],
        "superseded namespace was reused",
    )
    require(
        manifest["paths"]["prospective_run_root"]
        == "outputs/treewm-executable-prefix-repair-pilot-v1-launch2"
        and manifest["paths"]["run_root"]
        == (
            "/lustre/fs11/portfolios/edgeai/projects/"
            "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
            "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch2"
        ),
        "launch2 run namespace differs",
    )
    require(
        manifest["paths"]["wandb_project"] == CAMPAIGN_ID
        and manifest["logging"]["wandb_project"] == CAMPAIGN_ID
        and manifest["logging"]["wandb_group"] == CAMPAIGN_ID,
        "launch2 W&B namespace differs",
    )
    require(
        manifest["design"]["fresh_start_policy"].endswith(
            "may be imported, reused, or resumed."
        ),
        "superseded-launch exclusion policy differs",
    )
    require(manifest["design"]["settings"] == list(SETTINGS), "setting order differs")
    require(manifest["design"]["arms"] == list(ARMS), "arm order differs")
    require(manifest["design"]["seeds"] == list(SEEDS), "seed order differs")
    require(manifest["design"]["expected_cells"] == 20, "cell count differs")
    require(manifest["design"]["analysis_boundaries"] == [5000, 25000], "analysis boundaries differ")
    require(manifest["design"]["periodic_evaluation_boundaries"] == [12500, 25000], "eval boundaries differ")
    require([row["id"] for row in manifest["settings"]] == list(SETTINGS), "settings rows differ")
    require([row["id"] for row in manifest["arms"]] == list(ARMS), "arm rows differ")
    require(manifest["arms"][0]["executable_prefix_enabled"] is True, "control graph disabled")
    require(set(manifest["arms"][0]["executable_prefix_weights"].values()) == {0.0}, "control weights nonzero")
    require(all(math.isfinite(value) and value > 0 for value in manifest["arms"][1]["executable_prefix_weights"].values()), "treatment weights invalid")
    require(manifest["causal_contrast"]["sole_resolved_config_difference"] == list(WEIGHT_KEYS), "causal leaves differ")
    require(manifest["scientific_contract"]["optimizer_updates"] == 25000, "train cap differs")
    future = manifest["scientific_contract"]["future_sets"]
    require(future["horizons"] == [4, 8, 16, 32, 64] and future["h_max"] == 64, "horizon contract differs")
    require(future["executable_prefix_steps"] == 4, "prefix cap differs")
    require("min(4" in manifest["scientific_contract"]["prefix_length_rule"], "branchwise prefix rule missing")
    require("never compared to 4" in manifest["acceptance"]["prefix_structural_gates"]["target_rule"], "mean==4 gate reintroduced")
    cells = expand_matrix(manifest)
    require(len(cells) == 20, "matrix expansion differs")
    require(
        all(cell.run_name.startswith("exp23-launch2-") for cell in cells),
        "launch2 run-name namespace differs",
    )
    launch = manifest["launch_contract"]
    require(launch["array"] == "0-19%20" and launch["array_cells"] == 20, "launch array differs")
    require(launch["scratch_to_updates"] == 25_000 and launch["analysis_only_boundary_updates"] == 5_000, "launch boundaries differ")
    require(launch["terminal_final_evaluation"] is True and launch["terminal_final_evaluation_total_episodes"] == 25, "terminal evaluation contract differs")
    require(launch["midpoint_selection"] is False, "midpoint selection enabled")
    require("no TREEWM_STOP_AFTER_UPDATE" in launch["trainer_invocation_policy"], "staged stop reintroduced")
    require(launch["actual_submit_performed"] is False, "manifest claims a submission")
    execution = manifest["execution"]
    require("srun" not in execution, "srun execution path reintroduced")
    require(execution["scontrol"] == "/usr/local/bin/scontrol", "scontrol path differs")
    require(execution["control_python_flags"] == ["-I", "-S", "-B"], "control Python flags differ")
    require(execution["trainer_python_flags"] == ["-P", "-S", "-B"], "trainer Python flags differ")
    require(
        execution["sealed_trainer_environment"]
        == {"MUJOCO_GL": "egl", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        "sealed trainer environment differs",
    )
    require("continuous scratch-to-25000" in execution["training_lifecycle"], "continuous lifecycle differs")
    validate_snapshot_import_files(repo)
    _validate_lock(lock, manifest)
    _validate_prefix_target_lock(manifest)
    if verify_resolved_config_lock:
        _validate_resolved_config_lock(manifest)
    if verify_causal_parity_lock:
        _validate_causal_parity_lock(manifest)
    _validate_core(manifest, lock, Path(repo).resolve())
    for setting in manifest["settings"]:
        audit_data = lock["data_identity"][setting["id"]]
        require(setting["source_manifest_sha256"] == audit_data["source_manifest_sha256"], f"{setting['id']}: source differs")
        require(setting["future_recipe_sha256"] == audit_data["future_recipe_sha256"], f"{setting['id']}: recipe differs")
        require(setting["published_union_train_anchors"] == audit_data["train_population"], f"{setting['id']}: train population differs")
        require(setting["published_union_validation_anchors"] == audit_data["validation_population"], f"{setting['id']}: val population differs")
    for setting in SETTINGS:
        for seed in SEEDS:
            pair = [cell for cell in cells if cell.setting == setting and cell.seed == seed]
            require([cell.arm for cell in pair] == list(ARMS), "matched pair missing")
            resolved = [cell_overrides(cell, manifest, lock) for cell in pair]
            differing = {key for key in resolved[0] if resolved[0][key] != resolved[1][key]}
            require(differing == set(WEIGHT_KEYS), f"{setting}/seed{seed}: arm contrast is not weights-only: {sorted(differing)}")


def load_contract(repo: str | Path = REPOSITORY_ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(MANIFEST_PATH)
    lock = read_json(WEIGHT_LOCK_PATH)
    validate_manifest(manifest, lock, repo)
    return manifest, lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--verify", action="store_true")
    actions.add_argument("--matrix-json", action="store_true")
    actions.add_argument("--cell-json", type=int)
    parser.add_argument("--skip-protocol-lock", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        manifest, lock = load_contract()
        protocol = None if args.skip_protocol_lock else verify_protocol_lock()
        cells = expand_matrix(manifest)
        if args.verify:
            payload = {"status": "verified", "cells": len(cells), "protocol_sha256": protocol}
        elif args.matrix_json:
            payload = {"cells": [asdict(cell) for cell in cells]}
        else:
            require(args.cell_json is not None and 0 <= args.cell_json < len(cells), "cell index out of range")
            cell = cells[args.cell_json]
            payload = {"cell": asdict(cell), "declarative_overrides": cell_overrides(cell, manifest, lock)}
        print(canonical_json(payload))
        return 0
    except Exception as exc:
        print(f"Exp23 contract failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
