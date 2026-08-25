#!/usr/bin/env python3
"""Fail-closed contract and deterministic command builder for corrected pilot 13.

The pilot deliberately consumes the already-published v2 future recipes.  Their
generation-code identity is validated as an approved *input* identity; it is kept
separate from the current trainer source identity recorded for every new run.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


CAMPAIGN_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = CAMPAIGN_DIR.parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

MANIFEST_PATH = CAMPAIGN_DIR / "manifest.json"
PROTOCOL_LOCK_PATH = CAMPAIGN_DIR / "protocol.sha256"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RUNS = 32
SEEDS = (0, 1)
SETTING_IDS = ("antmaze-large", "cube-double", "puzzle-3x3", "scene")
ARM_IDS = ("r0-g0", "r0-g1", "r1-g0", "r1-g1")
PROTOCOL_FILES = (
    "manifest.json",
    "campaign.py",
    "worker.py",
    "report.py",
    "submit.py",
    "pilot.slurm",
    "README.md",
)


class ContractError(RuntimeError):
    """A pilot, input-recipe, source, or launch contract is not exact."""


@dataclass(frozen=True)
class RunSpec:
    index: int
    setting_index: int
    arm_index: int
    seed_index: int
    setting_id: str
    env_config: str
    arm_id: str
    seed: int
    regularization: bool
    grounded_multistep: bool
    run_name: str
    wandb_id: str


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot load pilot manifest: {exc}") from exc
    validate_manifest(value)
    return value


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    require(manifest.get("schema_version") == 1, "manifest schema drifted")
    require(
        manifest.get("campaign_id") == "treewm-v2-corrected-factorial-pilot-v1",
        "campaign ID drifted",
    )
    require(manifest.get("classification") == "bounded_diagnostic_pilot", "not a pilot")
    require("not formal validation" in str(manifest.get("claim_policy", "")), "claim guard missing")
    require(manifest.get("expected_runs") == RUNS, "expected run count drifted")

    method = manifest.get("method") or {}
    require(method.get("arm") == "treewm", "only TreeWM is in scope")
    require(method.get("model_class") == "TreeWM", "model class drifted")
    require(method.get("scorer") == "learned", "scorer drifted")
    require(method.get("experiment_config") == "treewm_v2_grounded_pilot", "config drifted")
    require(
        method.get("objective_version") == "treewm_v2_grounded_pilot_v1",
        "objective drifted",
    )
    require(method.get("node_budget") == 64 and method.get("branch_factor") == 4, "tree size drifted")

    factorial = manifest.get("factorial") or {}
    require(tuple(factorial.get("seeds") or ()) == SEEDS, "seed axis drifted")
    arms = factorial.get("arms") or []
    require(tuple(arm.get("id") for arm in arms) == ARM_IDS, "factorial arm order drifted")
    expected_arms = {
        "r0-g0": (False, False, 3e-4, 1e-4, 0.0, 0.0, 0.0),
        "r0-g1": (False, True, 3e-4, 1e-4, 0.0, 1.0, 0.25),
        "r1-g0": (True, False, 1e-4, 1e-3, 0.1, 0.0, 0.0),
        "r1-g1": (True, True, 1e-4, 1e-3, 0.1, 1.0, 0.25),
    }
    for arm in arms:
        actual = (
            arm.get("regularization"),
            arm.get("grounded_multistep"),
            arm.get("lr"),
            arm.get("weight_decay"),
            arm.get("dropout"),
            arm.get("multistep_weight"),
            arm.get("scheduled_sampling_p"),
        )
        require(actual == expected_arms[arm["id"]], f"{arm['id']}: factorial value drifted")
        require(arm.get("scheduled_sampling_warmup") == 5000, f"{arm['id']}: warmup drifted")

    shared = manifest.get("shared_contract") or {}
    exact_shared = {
        "optimizer_updates": 12_000,
        "scheduler_total_steps": 12_000,
        "checkpoint_every_updates": 2_000,
        "validation_every_updates": 2_000,
        "periodic_evaluation_updates": [6_000, 12_000],
        "periodic_episodes_per_task": 1,
        "final_episodes_per_task": 1,
        "task_ids": [1, 2, 3, 4, 5],
        "task_split": "standard",
        "max_train_anchors": 300_000,
        "max_validation_anchors": 30_000,
        "data_loader_workers": 10,
        "loader_thread_limit": 1,
        "model_max_depth": 3,
        "tree_max_depth": 3,
        "planner_decoded_metric": "domain_raw",
        "planner_execute_mode": "clipped",
        "planner_execute_steps": 4,
        "future_config": {
            "num_neighbors": 24,
            "query_multiplier": 6,
            "time_exclusion": 50,
            "include_self": True,
            "metric_mode": "rms_v2",
            "horizons": [4, 8, 16, 32, 64],
            "h_max": 64,
            "horizon_rule": "displacement",
            "fixed_horizon": 32,
            "cluster_method": "average",
            "max_modes": 4,
            "multi_step_depth": 3,
            "retrieval_pool": 50_000,
            "cache": False,
            "shared_cache": True,
        },
        "multistep_depth_weights": [1.0, 1.0, 1.0],
        "fixed_validation": True,
    }
    require(shared == exact_shared, "shared scientific contract drifted")
    require(shared["optimizer_updates"] <= 20_000, "bounded objective cap exceeded")

    legacy = manifest.get("compatible_v2_recipe_input") or {}
    require(legacy.get("read_only") is True, "compatible input is not read-only")
    require(legacy.get("campaign_id") == "treewm-50task-1m-v2", "input campaign drifted")
    require(legacy.get("objective_version") == "treewm_v2_rms_rank_v1", "input objective drifted")
    for key in ("campaign_protocol_sha256", "recipe_code_sha256", "recipe_runtime_sha256"):
        require(SHA256.fullmatch(str(legacy.get(key, ""))) is not None, f"bad {key}")

    settings = manifest.get("settings") or []
    require(tuple(setting.get("id") for setting in settings) == SETTING_IDS, "setting axis drifted")
    require(len(settings) * len(arms) * len(SEEDS) == RUNS, "factorial does not produce 32 runs")
    for setting in settings:
        require(setting.get("dataset_kind") == "standard", f"{setting['id']}: unexpected dataset")
        require(setting.get("data_subdir") == "standard", f"{setting['id']}: data path drifted")
        require(str(setting.get("env_name", "")).endswith("-v0"), f"{setting['id']}: env drifted")
        dims = setting.get("task_metric_dims") or []
        require(dims and len(set(dims)) == len(dims), f"{setting['id']}: metric dims invalid")
        require(setting.get("max_episode_steps") in {500, 750, 1000}, f"{setting['id']}: episode cap")
        for key in ("input_contract_sha256", "calibration_sha256", "future_recipe_sha256"):
            require(SHA256.fullmatch(str(setting.get(key, ""))) is not None, f"{setting['id']}: bad {key}")

    execution = manifest.get("execution") or {}
    require(execution.get("array") == "0-31%32", "Slurm array drifted")
    require(execution.get("gpus_per_task") == 1, "pilot must use one GPU per task")
    require(execution.get("walltime") == "04:00:00", "walltime drifted")
    require(execution.get("sbatch") == "/usr/local/bin/sbatch", "sbatch is not pinned")
    require(execution.get("srun") == "/cm/shared/apps/slurm/current/bin/srun", "srun is not pinned")
    require(execution.get("scontrol") == "/cm/shared/apps/slurm/current/bin/scontrol", "scontrol is not pinned")
    for key in ("python", "data_root", "raw_cache_root", "compatible_contract_root", "run_root"):
        require(Path((manifest.get("paths") or {}).get(key, "")).is_absolute(), f"{key} must be absolute")
    require("pilot" in manifest["paths"]["run_root"], "run root is not isolated pilot output")
    require(manifest["acceptance"]["candidate_arm"] == "r1-g1", "candidate arm drifted")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    validate_manifest(manifest)
    return stable_hash(manifest)


def protocol_sha256(campaign_dir: str | Path = CAMPAIGN_DIR) -> str:
    root = Path(campaign_dir).resolve()
    files: dict[str, str] = {}
    for relative in PROTOCOL_FILES:
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"protocol source missing/symlinked: {path}")
        files[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return stable_hash({"schema_version": 1, "files": files})


def verify_protocol_lock(campaign_dir: str | Path = CAMPAIGN_DIR) -> str:
    root = Path(campaign_dir).resolve()
    try:
        locked = (root / "protocol.sha256").read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ContractError(f"protocol lock is unavailable: {exc}") from exc
    live = protocol_sha256(root)
    require(SHA256.fullmatch(locked) is not None and locked == live, "protocol.sha256 is stale")
    return live


def source_contract(repo_root: str | Path = REPOSITORY_ROOT) -> dict[str, Any]:
    from treewm.utils.provenance import runtime_fingerprint, trainer_code_fingerprint

    code = trainer_code_fingerprint(Path(repo_root).resolve())
    runtime = runtime_fingerprint()
    return {
        "manifest_sha256": code["manifest_sha256"],
        "files": code["files"],
        "runtime_sha256": runtime["sha256"],
        "runtime": runtime,
    }


def _wandb_id(campaign_id: str, setting: str, arm: str, seed: int) -> str:
    return stable_hash(
        {"campaign_id": campaign_id, "setting_id": setting, "arm_id": arm, "seed": seed}
    )[:32]


def expand_runs(manifest: Mapping[str, Any]) -> list[RunSpec]:
    validate_manifest(manifest)
    result: list[RunSpec] = []
    # This loop order is the public array mapping contract.
    for setting_index, setting in enumerate(manifest["settings"]):
        for arm_index, arm in enumerate(manifest["factorial"]["arms"]):
            for seed_index, seed in enumerate(SEEDS):
                index = ((setting_index * 4) + arm_index) * 2 + seed_index
                require(index == len(result), "array mapping is not contiguous")
                name = f"corrected-{setting['id']}-{arm['id']}-seed{seed}"
                result.append(
                    RunSpec(
                        index=index,
                        setting_index=setting_index,
                        arm_index=arm_index,
                        seed_index=seed_index,
                        setting_id=setting["id"],
                        env_config=setting["env_config"],
                        arm_id=arm["id"],
                        seed=seed,
                        regularization=bool(arm["regularization"]),
                        grounded_multistep=bool(arm["grounded_multistep"]),
                        run_name=name,
                        wandb_id=_wandb_id(manifest["campaign_id"], setting["id"], arm["id"], seed),
                    )
                )
    require(len(result) == RUNS, "run expansion did not produce 32 unique runs")
    require(len({run.run_name for run in result}) == RUNS, "run names collide")
    require(len({run.wandb_id for run in result}) == RUNS, "W&B IDs collide")
    return result


def run_at(manifest: Mapping[str, Any], index: int) -> RunSpec:
    require(0 <= index < RUNS, "array index must be in [0, 32)")
    return expand_runs(manifest)[index]


def setting_for(manifest: Mapping[str, Any], run: RunSpec) -> Mapping[str, Any]:
    setting = manifest["settings"][run.setting_index]
    require(setting["id"] == run.setting_id, "RunSpec setting identity mismatch")
    return setting


def arm_for(manifest: Mapping[str, Any], run: RunSpec) -> Mapping[str, Any]:
    arm = manifest["factorial"]["arms"][run.arm_index]
    require(arm["id"] == run.arm_id, "RunSpec arm identity mismatch")
    return arm


def run_directory(manifest: Mapping[str, Any], run: RunSpec) -> Path:
    # scripts/train.py always places an explicit run beneath <env.short_name>/<arm>.
    # The factorial arm is encoded in run_name and the launch/config identity; Hydra's
    # model arm remains the scientific method `treewm` in every cell.
    return Path(manifest["paths"]["run_root"]) / run.setting_id / "treewm" / run.run_name


def data_contract_path(manifest: Mapping[str, Any], setting_id: str) -> Path:
    return Path(manifest["paths"]["compatible_contract_root"]) / "data" / f"{setting_id}.json"


def recipe_root(manifest: Mapping[str, Any], setting_id: str) -> Path:
    return Path(manifest["paths"]["compatible_contract_root"]) / "future-recipes" / setting_id


def load_compatible_input(
    manifest: Mapping[str, Any], run_or_setting: RunSpec | Mapping[str, Any], *, verify_files: bool = False
) -> dict[str, Any]:
    """Validate the published v2 recipe under its original code/runtime identity."""
    validate_manifest(manifest)
    setting = setting_for(manifest, run_or_setting) if isinstance(run_or_setting, RunSpec) else run_or_setting
    path = data_contract_path(manifest, setting["id"])
    try:
        contract = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"{setting['id']}: invalid compatible data contract: {exc}") from exc
    claimed = contract.get("contract_sha256")
    body = dict(contract)
    body.pop("contract_sha256", None)
    require(claimed == stable_hash(body), f"{setting['id']}: data contract content hash drifted")
    legacy = manifest["compatible_v2_recipe_input"]
    checks = {
        "contract_sha256": setting["input_contract_sha256"],
        "campaign_id": legacy["campaign_id"],
        "objective_version": legacy["objective_version"],
        "campaign_protocol_sha256": legacy["campaign_protocol_sha256"],
        "setting_id": setting["id"],
        "dataset_kind": setting["dataset_kind"],
        "calibration_sha256": setting["calibration_sha256"],
        "future_recipe_sha256": setting["future_recipe_sha256"],
        "raw_cache_read_only": True,
    }
    for key, expected in checks.items():
        require(contract.get(key) == expected, f"{setting['id']}: input {key} drifted")
    root = recipe_root(manifest, setting["id"])
    require(Path(contract.get("future_recipe_manifest", "")).resolve() == (root / "manifest.json").resolve(), f"{setting['id']}: recipe path drifted")
    try:
        recipe = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"{setting['id']}: invalid recipe manifest: {exc}") from exc
    require(recipe.get("recipe_sha256") == setting["future_recipe_sha256"], f"{setting['id']}: recipe digest drifted")
    expected_future_config = {
        key: value
        for key, value in manifest["shared_contract"]["future_config"].items()
        if key not in {"cache", "shared_cache"}
    }
    expected_future_config.update(
        {
            "relative_endpoints": setting["relative_endpoints"],
            "retrieval_radius": contract["chosen_thresholds"]["retrieval_radius"],
            "displacement_threshold": contract["chosen_thresholds"]["displacement_threshold"],
            "cluster_threshold": contract["chosen_thresholds"]["cluster_threshold"],
        }
    )
    for manifest_key in ("train_manifest", "validation_manifest"):
        child_path = root / str(recipe.get(manifest_key, ""))
        try:
            child = json.loads(
                child_path.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise ContractError(
                f"{setting['id']}: invalid {manifest_key}: {exc}"
            ) from exc
        require(
            (child.get("identity") or {}).get("future_config")
            == expected_future_config,
            f"{setting['id']}: compatible recipe future config drifted",
        )
    from treewm.data.future_recipe import validate_recipe_manifest

    validate_recipe_manifest(
        root,
        recipe,
        expected_source_manifest_sha256=contract["data_manifest_sha256"],
        expected_normalizer_sha256=contract["normalizer_sha256"],
        expected_calibration_sha256=contract["calibration_sha256"],
        expected_thresholds=contract["chosen_thresholds"],
        expected_train_manifest_sha256=contract["train_manifest_sha256"],
        expected_validation_manifest_sha256=contract["validation_manifest_sha256"],
        expected_code_sha256=legacy["recipe_code_sha256"],
        expected_runtime_sha256=legacy["recipe_runtime_sha256"],
        verify_file_hash=verify_files,
    )
    return contract


def _override(name: str, value: object) -> str:
    if isinstance(value, bool):
        rendered = str(value).lower()
    elif value is None:
        rendered = "null"
    elif isinstance(value, (list, tuple)):
        rendered = "[" + ",".join(str(item).lower() if isinstance(item, bool) else str(item) for item in value) + "]"
    else:
        rendered = str(value)
    return f"{name}={rendered}"


def scientific_overrides(
    manifest: Mapping[str, Any], run: RunSpec, contract: Mapping[str, Any]
) -> list[str]:
    setting = setting_for(manifest, run)
    arm = arm_for(manifest, run)
    shared = manifest["shared_contract"]
    chosen = contract["chosen_thresholds"]
    future = shared["future_config"]
    grounded = bool(arm["grounded_multistep"])
    return [
        _override("env", run.env_config),
        _override("experiment", manifest["method"]["experiment_config"]),
        _override("arm", "treewm"),
        _override("objective_version", manifest["method"]["objective_version"]),
        _override("seed", run.seed),
        _override("train.steps", shared["optimizer_updates"]),
        _override("train.scheduler_total_steps", shared["scheduler_total_steps"]),
        _override("train.ckpt_every", shared["checkpoint_every_updates"]),
        _override("train.eval_every", 6000),
        _override("train.diag_every", 2000),
        _override("train.max_train_anchors", shared["max_train_anchors"]),
        _override("train.max_val_anchors", shared["max_validation_anchors"]),
        _override("train.num_workers", shared["data_loader_workers"]),
        _override("train.lr", arm["lr"]),
        _override("train.weight_decay", arm["weight_decay"]),
        _override("train.gradient_checkpointing", True),
        _override("train.separate_gain_grad_clip", True),
        _override("train.world_grad_clip", 1.0),
        _override("train.gain_grad_clip", 1.0),
        _override("train.viz_every", 12000),
        _override("train.viz_every_early", 12000),
        _override("train.viz_early_until", 0),
        _override("model.dropout", arm["dropout"]),
        _override("model.max_depth", shared["model_max_depth"]),
        _override("tree.max_depth", shared["tree_max_depth"]),
        _override("tree.node_budget", manifest["method"]["node_budget"]),
        _override("tree.scorer", manifest["method"]["scorer"]),
        _override("model.branch_factor", manifest["method"]["branch_factor"]),
        _override("planner.decoded_metric", shared["planner_decoded_metric"]),
        _override("planner.execute_mode", shared["planner_execute_mode"]),
        _override("planner.execute_steps", shared["planner_execute_steps"]),
        _override("planner.max_env_steps", setting["max_episode_steps"]),
        _override("future_sets.num_neighbors", future["num_neighbors"]),
        _override("future_sets.query_multiplier", future["query_multiplier"]),
        _override("future_sets.time_exclusion", future["time_exclusion"]),
        _override("future_sets.include_self", future["include_self"]),
        _override("future_sets.metric_mode", future["metric_mode"]),
        _override("future_sets.horizons", future["horizons"]),
        _override("future_sets.h_max", future["h_max"]),
        _override("future_sets.horizon_rule", future["horizon_rule"]),
        _override("future_sets.fixed_horizon", future["fixed_horizon"]),
        _override("future_sets.cluster_method", future["cluster_method"]),
        _override("future_sets.max_modes", future["max_modes"]),
        _override("future_sets.multi_step_depth", future["multi_step_depth"]),
        _override("future_sets.retrieval_pool", future["retrieval_pool"]),
        _override("future_sets.cache", future["cache"]),
        _override("future_sets.shared_cache", future["shared_cache"]),
        _override("future_sets.relative_endpoints", setting["relative_endpoints"]),
        _override("future_sets.retrieval_radius", chosen["retrieval_radius"]),
        _override("future_sets.displacement_threshold", chosen["displacement_threshold"]),
        _override("future_sets.cluster_threshold", chosen["cluster_threshold"]),
        _override("+env.task_metric_dims", setting["task_metric_dims"]),
        _override("losses.enabled.multistep", grounded),
        _override("losses.weights.multistep", arm["multistep_weight"]),
        _override("losses.scheduled_sampling_p", arm["scheduled_sampling_p"]),
        _override("losses.scheduled_sampling_warmup", arm["scheduled_sampling_warmup"]),
        _override("losses.multistep_depth_weights", shared["multistep_depth_weights"]),
        _override("eval.task_split", shared["task_split"]),
        _override("eval.episodes_per_task", shared["periodic_episodes_per_task"]),
        _override("eval.final_episodes_per_task", shared["final_episodes_per_task"]),
        _override("eval.seed", run.seed),
        _override("+campaign_input_contract_sha256", contract["contract_sha256"]),
        _override("+campaign_calibration_sha256", contract["calibration_sha256"]),
        _override("+campaign_future_recipe_sha256", contract["future_recipe_sha256"]),
        _override("+campaign_compatible_recipe_code_sha256", manifest["compatible_v2_recipe_input"]["recipe_code_sha256"]),
        _override("+campaign_factorial_arm", run.arm_id),
    ]


def config_sha256(manifest: Mapping[str, Any], run: RunSpec, contract: Mapping[str, Any]) -> str:
    return stable_hash({"schema_version": 1, "overrides": scientific_overrides(manifest, run, contract)})


def run_protocol_sha256(
    manifest: Mapping[str, Any],
    run: RunSpec,
    contract: Mapping[str, Any],
    *,
    source_sha256: str,
    runtime_sha256: str,
    package_protocol_sha256: str,
) -> str:
    return stable_hash(
        {
            "schema_version": 1,
            "campaign_id": manifest["campaign_id"],
            "package_protocol_sha256": package_protocol_sha256,
            "active_trainer_source_sha256": source_sha256,
            "active_trainer_runtime_sha256": runtime_sha256,
            "config_sha256": config_sha256(manifest, run, contract),
            "compatible_input_contract_sha256": contract["contract_sha256"],
            "compatible_recipe_code_sha256": manifest["compatible_v2_recipe_input"]["recipe_code_sha256"],
            "compatible_recipe_runtime_sha256": manifest["compatible_v2_recipe_input"]["recipe_runtime_sha256"],
            "calibration_sha256": contract["calibration_sha256"],
            "future_recipe_sha256": contract["future_recipe_sha256"],
        }
    )


def trainer_command(
    manifest: Mapping[str, Any],
    run: RunSpec,
    *,
    repo_root: str | Path = REPOSITORY_ROOT,
    verify_recipe_files: bool = False,
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    require((repo_root / "scripts" / "train.py").is_file(), "trainer source is missing")
    protocol = verify_protocol_lock(repo_root / "experiments" / "13-treewm-corrected-pilot")
    contract = load_compatible_input(manifest, run, verify_files=verify_recipe_files)
    source = source_contract(repo_root)
    source_sha = source["manifest_sha256"]
    runtime_sha = source["runtime_sha256"]
    config_sha = config_sha256(manifest, run, contract)
    run_protocol = run_protocol_sha256(
        manifest,
        run,
        contract,
        source_sha256=source_sha,
        runtime_sha256=runtime_sha,
        package_protocol_sha256=protocol,
    )
    paths = manifest["paths"]
    python = Path(paths["python"])
    require(python.is_file() and os.access(python, os.X_OK), f"Python is not executable: {python}")
    output = run_directory(manifest, run)
    overrides = scientific_overrides(manifest, run, contract)
    argv = [
        str(python),
        str(repo_root / "scripts" / "train.py"),
        *overrides,
        _override("run_root", paths["run_root"]),
        _override("run_name", run.run_name),
        _override("resume", "auto"),
        _override("+campaign_source_sha256", source_sha),
        _override("+campaign_protocol_sha256", protocol),
        _override("+campaign_config_sha256", config_sha),
        _override("hydra.run.dir", output / "hydra"),
        _override("hydra.job.chdir", False),
    ]
    environment = {
        # Trainer identity and recipe-producer identity are separate contracts. The
        # former must match the code/runtime executing now; the latter deliberately
        # remains the producer of the approved read-only v2 recipes.
        "TREEWM_PROTOCOL_SHA256": run_protocol,
        "TREEWM_CODE_SHA256": source_sha,
        "TREEWM_RUNTIME_SHA256": runtime_sha,
        "TREEWM_RECIPE_CODE_SHA256": manifest["compatible_v2_recipe_input"]["recipe_code_sha256"],
        "TREEWM_RECIPE_RUNTIME_SHA256": manifest["compatible_v2_recipe_input"]["recipe_runtime_sha256"],
        "TREEWM_ACTIVE_SOURCE_SHA256": source_sha,
        "TREEWM_CONFIG_SHA256": config_sha,
        "TREEWM_COMPATIBLE_RECIPE_CODE_SHA256": manifest["compatible_v2_recipe_input"]["recipe_code_sha256"],
        "TREEWM_COMPATIBLE_RECIPE_RUNTIME_SHA256": manifest["compatible_v2_recipe_input"]["recipe_runtime_sha256"],
        "TREEWM_DATA_SHA256": contract["data_manifest_sha256"],
        "TREEWM_CALIBRATION_SHA256": contract["calibration_sha256"],
        "TREEWM_FUTURE_RECIPE_SHA256": contract["future_recipe_sha256"],
        "TREEWM_DATA_CONTRACT_SHA256": contract["contract_sha256"],
        "TREEWM_DATA_ROOT": paths["data_root"],
        "TREEWM_CACHE": paths["raw_cache_root"],
        "TREEWM_FUTURE_RECIPE_ROOT": str(recipe_root(manifest, run.setting_id)),
        "TREEWM_RUN_NAME": run.run_name,
        "WANDB_PROJECT": manifest["logging"]["wandb_project"],
        "WANDB_RUN_GROUP": manifest["logging"]["wandb_group"],
        "WANDB_RUN_ID": run.wandb_id,
        "WANDB_MODE": manifest["logging"]["wandb_mode"],
        "OMP_NUM_THREADS": str(manifest["shared_contract"]["loader_thread_limit"]),
        "MKL_NUM_THREADS": str(manifest["shared_contract"]["loader_thread_limit"]),
        "OPENBLAS_NUM_THREADS": str(manifest["shared_contract"]["loader_thread_limit"]),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    launch = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "classification": manifest["classification"],
        "formal_validation": False,
        "run": {
            "index": run.index,
            "setting_id": run.setting_id,
            "arm_id": run.arm_id,
            "seed": run.seed,
            "run_name": run.run_name,
            "run_directory": str(output),
        },
        "hashes": {
            "manifest_sha256": manifest_sha256(manifest),
            "source_sha256": source_sha,
            "runtime_sha256": runtime_sha,
            "package_protocol_sha256": protocol,
            "config_sha256": config_sha,
            "run_protocol_sha256": run_protocol,
            "compatible_input_contract_sha256": contract["contract_sha256"],
            "compatible_recipe_code_sha256": manifest["compatible_v2_recipe_input"]["recipe_code_sha256"],
            "compatible_recipe_runtime_sha256": manifest["compatible_v2_recipe_input"]["recipe_runtime_sha256"],
            "calibration_sha256": contract["calibration_sha256"],
            "future_recipe_sha256": contract["future_recipe_sha256"],
        },
        "argv": argv,
        "environment": environment,
    }
    launch["launch_sha256"] = stable_hash(launch)
    return launch


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    try:
        descriptor = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def verify_all(
    manifest: Mapping[str, Any], *, repo_root: str | Path = REPOSITORY_ROOT, verify_files: bool = False
) -> dict[str, Any]:
    repo_root = Path(repo_root).resolve()
    protocol = verify_protocol_lock(repo_root / "experiments" / "13-treewm-corrected-pilot")
    runs = expand_runs(manifest)
    contracts: dict[str, str] = {}
    for setting in manifest["settings"]:
        contract = load_compatible_input(manifest, setting, verify_files=verify_files)
        contracts[setting["id"]] = contract["contract_sha256"]
    source = source_contract(repo_root)
    configs = {
        run.run_name: config_sha256(manifest, run, load_compatible_input(manifest, run))
        for run in runs
    }
    return {
        "status": "verified_bounded_pilot",
        "formal_validation": False,
        "runs": len(runs),
        "manifest_sha256": manifest_sha256(manifest),
        "source_sha256": source["manifest_sha256"],
        "runtime_sha256": source["runtime_sha256"],
        "package_protocol_sha256": protocol,
        "input_contracts": contracts,
        "recipe_files_verified": bool(verify_files),
        "unique_config_sha256": sorted(set(configs.values())),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    verify.add_argument("--verify-recipe-files", action="store_true")
    listing = sub.add_parser("list")
    listing.add_argument("--json", action="store_true")
    command = sub.add_parser("command")
    command.add_argument("--index", type=int, required=True)
    command.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    command.add_argument("--output", type=Path)
    sub.add_parser("protocol-hash")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    manifest = load_manifest(args.manifest)
    if args.command == "protocol-hash":
        print(protocol_sha256(CAMPAIGN_DIR))
        return 0
    if args.command == "list":
        rows = [run.__dict__ for run in expand_runs(manifest)]
        if args.json:
            print(json.dumps(rows, sort_keys=True, indent=2))
        else:
            for run in expand_runs(manifest):
                print(f"{run.index:02d} {run.setting_id:15s} {run.arm_id} seed={run.seed} {run.run_name}")
        return 0
    if args.command == "verify":
        result = verify_all(
            manifest, repo_root=args.repo_root, verify_files=args.verify_recipe_files
        )
        print(json.dumps(result, sort_keys=True, indent=2))
        return 0
    run = run_at(manifest, args.index)
    launch = trainer_command(manifest, run, repo_root=args.repo_root)
    if args.output:
        atomic_json(args.output, launch)
    print(json.dumps(launch, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"corrected-pilot contract error: {exc}", file=sys.stderr)
        raise SystemExit(2)
