#!/usr/bin/env python3
"""Immutable, dependency-light contract for the formal TreeWM v2 campaign."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
from typing import Any, Mapping, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CAMPAIGN_ROOT = Path(__file__).resolve().parent
if not (REPOSITORY_ROOT / "treewm" / "__init__.py").is_file():
    raise RuntimeError(f"TreeWM repository package is missing beneath {REPOSITORY_ROOT}")
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))
if str(CAMPAIGN_ROOT) not in sys.path:
    sys.path.insert(0, str(CAMPAIGN_ROOT))


SHA256 = re.compile(r"^[0-9a-f]{64}$")
SETTING_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
FORMAL_UPDATES = 1_000_000
FORMAL_TASK_IDS = (1, 2, 3, 4, 5)
FORMAL_SEEDS = (0, 1, 2, 3)
EXPECTED_WORKERS = 16
EXPECTED_ALLOCATION_SHARDS = 3
FORMAL_PYTHON = (
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/envs/treewm-formal-py311/bin/python"
)
FORMAL_DATA_ROOT = (
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/datasets/ogbench-rql-50task"
)
FORMAL_RAW_CACHE_ROOT = (
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/datasets/treewm-50task-full-cache-v1"
)
FORMAL_CONTRACT_ROOT = (
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/datasets/treewm-50task-formal-v2-contracts-v1"
)
# Output identity is deliberately independent of the read-only protocol source
# snapshot from which a multi-week allocation executes.  The snapshot may live at a
# protocol-keyed path, while checkpoints and final artifacts remain at these one-time
# campaign roots.
FORMAL_RUN_ROOT = (
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/projects/treewm/outputs/treewm-50task-1m-v2"
)
PILOT_RUN_ROOT = (
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/projects/treewm/outputs/treewm-50task-v2-pilot"
)


class ManifestError(ValueError):
    """The campaign manifest or one of its immutable artifacts is invalid."""


@dataclass(frozen=True)
class RunSpec:
    global_index: int
    setting_index: int
    setting_id: str
    env_config: str
    env_name: str
    source_name: str
    dataset_kind: str
    seed: int
    run_name: str
    wandb_id: str

    @property
    def index(self) -> int:
        return self.global_index

    @property
    def run_id(self) -> str:
        return self.run_name


def _canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def _nested_get(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, Mapping) or part not in current:
            raise KeyError(dotted)
        current = current[part]
    return current


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ManifestError(f"duplicate JSON key in campaign manifest: {key}")
        value[key] = item
    return value


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle, object_pairs_hook=_reject_duplicate_json_keys)
    validate_manifest(manifest)
    return manifest


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    validate_manifest(manifest)
    return stable_hash(manifest)


PROTOCOL_SOURCE_FILES = (
    "campaign.py",
    "prepare_cache.py",
    "calibration.py",
    "../../treewm/data/future_recipe.py",
    "../../treewm/data/ogbench_dataset.py",
    "dispatcher.py",
    "gpu_preflight.py",
    "train.slurm",
    "stage_data.slurm",
    "calibration_gate.slurm",
    "pilot.slurm",
    "pilot_gate.slurm",
    "validate_pilot.py",
    "aggregate.py",
    "aggregate.slurm",
    "submit.py",
)


def protocol_sha256(
    manifest: Mapping[str, Any], campaign_dir: str | Path | None = None
) -> str:
    """Hash the exact executable source snapshot, excluding the recursive lock.

    This intentionally reimplements the dependency-light file inventory instead of
    importing trainer modules.  The protocol digest is also the immutable source-
    snapshot directory key, so it must change for any trainer, model, loss, data, or
    Hydra-config edit—not only for orchestration edits.
    """
    validate_manifest(manifest)
    root = Path(campaign_dir or Path(__file__).resolve().parent).resolve()
    repo_root = root.parents[1]
    candidates = {
        root / "manifest.json",
        *(root / relative for relative in PROTOCOL_SOURCE_FILES),
        *(repo_root / "treewm").rglob("*.py"),
        *(repo_root / "configs").rglob("*.yaml"),
        repo_root / "scripts" / "train.py",
        repo_root / "scripts" / "__init__.py",
    }
    sources: dict[str, str] = {}
    for candidate in sorted(candidates):
        path = candidate.resolve()
        if not path.is_relative_to(repo_root) or not path.is_file() or candidate.is_symlink():
            raise ManifestError(f"protocol source is missing, symlinked, or escapes repo: {candidate}")
        relative = path.relative_to(repo_root).as_posix()
        # `protocol.sha256` must never enter its own preimage.
        if relative == "experiments/12-treewm-formal-v2/protocol.sha256":
            raise ManifestError("protocol lock cannot be included in protocol sources")
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return stable_hash({"manifest_sha256": manifest_sha256(manifest), "sources": sources})


EXPECTED_OBJECTIVE = {
    "q_scales": [["mixed", 32, 1.0]],
    "use_depth_embedding": False,
    "keep_threshold": 0.5,
    "keep_balance": False,
    "matching_normalization_version": "rms_v2",
    "matching_num_horizons": 5,
    "future_metric_mode": "rms_v2",
    "max_modes": 4,
    "control_objective": "future_set",
    "control_target_transform": "rms_tanh",
    "control_endpoint_key": "fut_metric_endpoint",
    "control_allow_endpoint_fallback": False,
    "control_require_single_scale": True,
    "control_metric_weight": 1.0,
    "control_rank_weight": 1.0,
    "control_rank_temperature": 0.2,
    "detach_world_targets": True,
    "bind_negative_margin": 0.1,
    "gain_target": "novelty",
    "gain_set_context": True,
    "gain_rank_weight": 1.0,
    "gain_calibration_weight": 0.1,
    "gain_branch_prior_weight": 0.0,
    "mass_enabled": False,
    "mass_weight": 0.0,
    "separate_gain_grad_clip": True,
    "world_grad_clip": 1.0,
    "gain_grad_clip": 1.0,
    "matching_contract": {
        "lambda_z": 1.0,
        "lambda_q": 1.0,
        "lambda_action": 0.5,
        "lambda_horizon": 0.1,
        "method": "hungarian",
    },
    "loss_contract": {
        "scheduled_sampling_p": 0.0,
        "scheduled_sampling_warmup": 2_000,
        "multistep_depth_weights": [],
        "redundancy_temperature": 0.25,
        "contrastive_temperature": 0.1,
        "coverage_space": "q",
        "future_scale": 1.0,
        "control_batch": 64,
        "recursive_batch": 256,
        "warmup": {"redundancy": 5_000, "expand": 2_000, "mass": 1_000},
        "decay": {"redundancy": 0},
        "weights": {
            "state": 1.0,
            "action": 1.0,
            "horizon": 0.5,
            "coverage": 1.0,
            "redundancy": 0.1,
            "mass": 0.0,
            "keep": 0.5,
            "expand": 0.5,
            "bind": 1.0,
            "control": 0.5,
            "reconstruction": 0.1,
            "recursive": 0.2,
            "uncertainty": 0.2,
            "multistep": 0.0,
        },
        "enabled": {
            "state": True,
            "action": True,
            "horizon": True,
            "coverage": True,
            "redundancy": True,
            "mass": False,
            "keep": True,
            "expand": True,
            "bind": True,
            "control": True,
            "reconstruction": True,
            "recursive": True,
            "uncertainty": True,
            "multistep": False,
        },
    },
    "optimizer_contract": {
        "lr": 3e-4,
        "weight_decay": 1e-4,
        "warmup_steps": 1_000,
        "min_lr_scale": 0.1,
        "grad_clip": 1.0,
        "grad_accum": 1,
        "bf16": True,
    },
    "gain_training_contract": {"loss_every": 4, "batch_size": 64, "tree_budget": 64},
    "model_contract": {
        "reconstruction": True,
        "normalize_q": True,
        "novelty_space": "q",
        "use_tree_context": True,
        "horizon_mode": "learned",
        "horizons": [4, 8, 16, 32, 64],
        "h_max": 64,
        "max_depth": 16,
    },
    "tree_contract": {
        "expansion_batch_size": 4,
        "max_depth": 16,
        "branch_factor": 4,
        "context_pooling": "mean",
    },
    "planner_contract": {
        "score_space": "decoded",
        "score_mode": "endpoint",
        "path_cost_weight": 0.02,
        "ancestor_weight": 0.5,
        "execute_mode": "clipped",
        "execute_steps": 16,
        "use_uncertainty": False,
        "uncertainty_weight": 0.0,
        "exclude_root": True,
    },
}


EXPECTED_CALIBRATION = {
    "algorithm_version": "treewm_future_metric_calibration_v1",
    "split": "train",
    "metric_mode": "rms_v2",
    "sample_size": 4096,
    "anchor_sample_seed": 0,
    "retrieval_pool": 50_000,
    "retrieval_pool_seed": 0,
    "radius_quantile": 0.90,
    "fallback_radius_quantile": 0.99,
    "radius_rule": "max_higher_q90_23rd_nonself_higher_q99_nearest_nonself",
    "quantile_method": "higher",
    "horizon_rule": "maximize_five_class_normalized_entropy",
    "cluster_rule": "average_linkage_target_three_modes_max_four",
    "max_insufficient_neighbor_fraction": 0.10,
    "max_truncation_fraction": 0.05,
    "min_mean_retrieved": 18.0,
    "max_retrieval_fallback_fraction": 0.01,
    "min_normalized_horizon_entropy": 0.65,
    "min_occupied_horizon_classes": 4,
    "min_mean_retained_modes": 1.5,
    "max_mean_retained_modes": 3.5,
    "min_multimode_anchor_fraction": 0.40,
}


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed on the complete formal-v2 scientific and execution contract."""
    _require(manifest.get("schema_version") == 2, "schema_version must be 2")
    _require(manifest.get("campaign_id") == "treewm-50task-1m-v2", "campaign ID drifted")
    _require(manifest.get("expected_model_runs") == 40, "model-run count must be 40")
    _require(
        manifest.get("expected_task_seed_evaluations") == 200,
        "task-seed evaluation count must be 200",
    )
    axes = manifest.get("axes") or {}
    _require(tuple(axes.get("seeds", ())) == FORMAL_SEEDS, "seeds must be exactly 0..3")
    _require(tuple(axes.get("task_ids", ())) == FORMAL_TASK_IDS, "tasks must be 1..5")

    method = manifest.get("method") or {}
    expected_method = {
        "arm": "treewm",
        "model_class": "TreeWM",
        "scorer": "learned",
        "objective_version": "treewm_v2_rms_rank_v1",
        "experiment_config": "treewm_v2",
        "node_budget": 64,
        "branch_factor": 4,
        "gradient_checkpointing": True,
    }
    _require(method == expected_method, "formal TreeWM v2 method contract drifted")
    _require(manifest.get("objective") == EXPECTED_OBJECTIVE, "v2 objective contract drifted")

    training = manifest.get("training") or {}
    expected_training = {
        "optimizer_updates": FORMAL_UPDATES,
        "scheduler_total_steps": FORMAL_UPDATES,
        "batch_size": 256,
        "max_train_anchors": 300_000,
        "max_validation_anchors": 30_000,
        "anchor_sampling": "uniform_without_replacement_over_full_valid_transition_universe",
        "anchor_sampling_algorithm": "numpy_choice_le10m_else_uniform_rejection_v1",
        "shared_cache": True,
        "future_set_cache": False,
        "future_horizons": [4, 8, 16, 32, 64],
        "future_h_max": 64,
        "future_horizon_rule": "displacement",
        "future_num_neighbors": 24,
        "future_query_multiplier": 6,
        "future_time_exclusion": 50,
        "future_include_self": True,
        "future_fixed_horizon": 32,
        "future_cluster_method": "average",
        "future_multi_step_depth": 3,
        "future_retrieval_pool": 50_000,
        "latent_retrieval_enabled": False,
        "latent_retrieval_keys": 0,
        "redundancy_decay_updates": 0,
        "checkpoint_every_updates": 2_000,
        "periodic_evaluation_every_updates": 100_000,
        "periodic_episodes_per_task": 1,
        "pilot_updates": 5_000,
        "pilot_checkpoint_every_updates": 500,
        "data_loader_workers": 10,
        "loader_thread_limit": 1,
        "visualization_every_updates": 100_000,
        "early_visualization_every_updates": 10_000,
        "early_visualization_until_update": 50_000,
    }
    _require(training == expected_training, "formal v2 training contract drifted")
    _require(manifest.get("calibration") == EXPECTED_CALIBRATION, "calibration rules drifted")

    evaluation = manifest.get("evaluation") or {}
    _require(
        evaluation
        == {
            "task_split": "standard",
            "task_ids": [1, 2, 3, 4, 5],
            "final_episodes_per_task": 50,
            "pilot_final_episodes_per_task": 1,
            "node_budget": 64,
            "seed_rule": "training_seed",
        },
        "evaluation contract drifted",
    )
    execution = manifest.get("execution") or {}
    _require(execution.get("allocation_shards") == 3, "three allocations are required")
    _require(execution.get("workers_per_allocation") == 16, "16 workers are required")
    _require(execution.get("mapping") == "global_index=16*allocation_shard+rank", "mapping drifted")
    _require(execution.get("gpus_per_worker") == 1, "one GPU/worker is required")
    _require(execution.get("nodes") == 2, "two nodes/allocation are required")
    _require(execution.get("workers_per_node") == 8, "eight workers/node are required")
    _require(execution.get("allocation_deadline_seconds") == 13_200, "deadline drifted")

    logging = manifest.get("logging") or {}
    _require(logging.get("wandb_project") == "treewm-50task-formal-v2", "W&B project drifted")
    _require(logging.get("wandb_group") == manifest["campaign_id"], "W&B group drifted")
    _require(
        logging.get("pilot_wandb_project") == "treewm-50task-formal-v2-pilot",
        "pilot W&B project drifted",
    )
    _require(logging.get("pilot_wandb_group") == "treewm-50task-v2-pilot", "pilot group drifted")
    _require(logging.get("wandb_mode") == "online", "formal W&B must be online")
    _require(
        logging.get("stable_id_rule") == "sha256(namespace,setting_id,seed)[:32]",
        "W&B identity rule drifted",
    )

    paths = manifest.get("paths") or {}
    expected_paths = {
        "python": FORMAL_PYTHON,
        "data_root": FORMAL_DATA_ROOT,
        "raw_cache_root": FORMAL_RAW_CACHE_ROOT,
        "contract_root": FORMAL_CONTRACT_ROOT,
        "run_root": FORMAL_RUN_ROOT,
        "pilot_run_root": PILOT_RUN_ROOT,
    }
    _require(paths == expected_paths, "formal path contract drifted")
    # The approved read-only raw cache is the sole legacy campaign namespace. No v1
    # run, checkpoint, data contract, or W&B identifier may be accepted by exp12.
    for key in ("run_root", "pilot_run_root", "contract_root"):
        _require("treewm-50task-1m-v1" not in paths[key], f"{key} aliases the v1 campaign")
    for key in ("wandb_project", "wandb_group", "pilot_wandb_project", "pilot_wandb_group"):
        _require(not str(logging[key]).endswith("-v1"), f"{key} aliases a v1 namespace")

    settings = manifest.get("settings") or []
    _require(len(settings) == 10, "exactly ten dataset settings are required")
    seen: set[str] = set()
    sharded = 0
    for position, setting in enumerate(settings):
        setting_id = setting.get("id")
        _require(
            isinstance(setting_id, str) and SETTING_ID.fullmatch(setting_id) is not None,
            f"setting {position} has an invalid ID",
        )
        _require(setting_id not in seen, f"duplicate setting {setting_id}")
        seen.add(setting_id)
        _require(str(setting.get("env_name", "")).endswith("-v0"), f"{setting_id}: bad env")
        _require(str(setting.get("source_name", "")).endswith("-v0"), f"{setting_id}: bad source")
        obs_dim = setting.get("obs_dim")
        _require(isinstance(obs_dim, int) and obs_dim > 0, f"{setting_id}: bad obs_dim")
        _require(setting.get("action_dim", 0) > 0, f"{setting_id}: bad action_dim")
        xy_dims = setting.get("xy_dims")
        metric_dims = setting.get("task_metric_dims")
        _require(
            isinstance(xy_dims, list) and xy_dims and len(set(xy_dims)) == len(xy_dims),
            f"{setting_id}: invalid xy_dims",
        )
        _require(metric_dims == xy_dims, f"{setting_id}: formal task metric dims drifted")
        _require(all(isinstance(dim, int) and 0 <= dim < obs_dim for dim in xy_dims),
                 f"{setting_id}: metric dimension outside observation")
        expected_relative = setting_id not in {"scene", "puzzle-3x3", "puzzle-4x4-100m"}
        _require(
            setting.get("relative_endpoints") is expected_relative,
            f"{setting_id}: endpoint semantics drifted",
        )
        _require(setting.get("max_episode_steps") in {500, 750, 1000, 2000},
                 f"{setting_id}: unverified episode limit")
        kind = setting.get("dataset_kind")
        _require(kind in {"standard", "sharded_100m_full"}, f"{setting_id}: bad data kind")
        if kind == "sharded_100m_full":
            sharded += 1
            _require(setting["source_name"].endswith("-100m-v0"), f"{setting_id}: bad 100M source")
            _require(setting["env_name"] != setting["source_name"], f"{setting_id}: simulator/source conflated")
            for key, expected in (
                ("expected_train_shards", 100),
                ("expected_validation_shards", 100),
                ("expected_train_transitions", 100_000_000),
                ("expected_validation_transitions", 10_000_000),
                ("expected_train_trajectories", 100_000),
                ("expected_validation_trajectories", 10_000),
            ):
                _require(setting.get(key) == expected, f"{setting_id}: {key} drifted")
    _require(sharded == 2, "exactly two full-100M sources are required")


def _wandb_id(namespace: str, setting_id: str, seed: int) -> str:
    return stable_hash({"namespace": namespace, "setting_id": setting_id, "seed": seed})[:32]


def expand_runs(manifest: Mapping[str, Any]) -> list[RunSpec]:
    validate_manifest(manifest)
    runs: list[RunSpec] = []
    for setting_index, setting in enumerate(manifest["settings"]):
        for seed in FORMAL_SEEDS:
            run_name = f"treewm-v2-{setting['id']}-seed{seed}"
            runs.append(
                RunSpec(
                    global_index=len(runs),
                    setting_index=setting_index,
                    setting_id=setting["id"],
                    env_config=setting["env_config"],
                    env_name=setting["env_name"],
                    source_name=setting["source_name"],
                    dataset_kind=setting["dataset_kind"],
                    seed=seed,
                    run_name=run_name,
                    wandb_id=_wandb_id(manifest["campaign_id"], setting["id"], seed),
                )
            )
    _require(len(runs) == 40, "run expansion must produce exactly 40 models")
    return runs


def run_for_worker(
    runs_or_manifest: Sequence[RunSpec] | Mapping[str, Any],
    worker_index: int,
    workers: int = EXPECTED_WORKERS,
    *,
    allocation_shard: int = 0,
    allocation_shards: int = EXPECTED_ALLOCATION_SHARDS,
) -> RunSpec | None:
    """Return the unique run at ``16*allocation_shard+worker_index``."""
    if workers != EXPECTED_WORKERS or allocation_shards != EXPECTED_ALLOCATION_SHARDS:
        raise ValueError("formal v2 requires exactly three allocations of 16 workers")
    if not 0 <= worker_index < workers:
        raise ValueError("worker_index must be in [0,16)")
    if not 0 <= allocation_shard < allocation_shards:
        raise ValueError("allocation_shard must be in [0,3)")
    runs = expand_runs(runs_or_manifest) if isinstance(runs_or_manifest, Mapping) else list(runs_or_manifest)
    global_index = workers * allocation_shard + worker_index
    return runs[global_index] if global_index < len(runs) else None


def all_worker_ownership(manifest: Mapping[str, Any]) -> list[int]:
    runs = expand_runs(manifest)
    return [
        run.global_index
        for shard in range(EXPECTED_ALLOCATION_SHARDS)
        for rank in range(EXPECTED_WORKERS)
        if (run := run_for_worker(runs, rank, allocation_shard=shard)) is not None
    ]


def run_directory(run_root: str | Path, run: RunSpec) -> Path:
    return Path(run_root).expanduser().absolute() / run.setting_id / "treewm" / run.run_name


def setting_for_run(manifest: Mapping[str, Any], run: RunSpec) -> Mapping[str, Any]:
    setting = manifest["settings"][run.setting_index]
    if setting["id"] != run.setting_id:
        raise ManifestError("RunSpec setting index/ID mismatch")
    return setting


def required_dataset_files(
    manifest: Mapping[str, Any], data_root: str | Path, setting: Mapping[str, Any]
) -> list[Path]:
    validate_manifest(manifest)
    directory = Path(data_root).expanduser().absolute() / setting["data_subdir"]
    source = setting["source_name"]
    if setting["dataset_kind"] == "standard":
        return [directory / f"{source}.npz", directory / f"{source}-val.npz"]
    stem = source.removesuffix("-100m-v0") + "-v0"
    return [
        path
        for shard in range(100)
        for path in (
            directory / f"{stem}-{shard:03d}.npz",
            directory / f"{stem}-{shard:03d}-val.npz",
        )
    ]


def data_contract_path(contract_root: str | Path, setting_id: str) -> Path:
    return Path(contract_root).expanduser().absolute() / "data" / f"{setting_id}.json"


def calibration_contract_path(contract_root: str | Path, setting_id: str) -> Path:
    return Path(contract_root).expanduser().absolute() / "calibration" / f"{setting_id}.json"


def recipe_root_path(contract_root: str | Path, setting_id: str) -> Path:
    return Path(contract_root).expanduser().absolute() / "future-recipes" / setting_id


def raw_cache_manifest_path(cache_root: str | Path, setting: Mapping[str, Any]) -> Path:
    root = Path(cache_root).expanduser().absolute()
    source = str(setting["source_name"])
    pattern = f"{source}__full__*/manifest.json" if setting["dataset_kind"] == "sharded_100m_full" else f"{source}__*/manifest.json"
    candidates = sorted(root.glob(pattern))
    matches: list[Path] = []
    for candidate in candidates:
        try:
            payload = json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("dataset_name", payload.get("dataset")) == source:
            matches.append(candidate)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one immutable raw cache for {source}, found {matches}")
    return matches[0]


def train_inventory_sha256(source_files: Sequence[Mapping[str, Any]]) -> str:
    train = [
        {
            "split": "train",
            "index": entry.get("index"),
            "path": entry.get("path"),
            "size": entry.get("size"),
            "sha256": entry.get("sha256"),
        }
        for entry in source_files
        if entry.get("split") == "train"
    ]
    if not train or any(SHA256.fullmatch(str(entry["sha256"])) is None for entry in train):
        raise ValueError("raw cache has no complete content-digested train inventory")
    return stable_hash(train)


def normalizer_sha256(norm_stats: Mapping[str, Sequence[float]]) -> str:
    """Hash the exact float32 normalizer state loaded by ``Normalizer``."""
    digest = hashlib.sha256(b"treewm-normalizer-float32-v1\n")
    for key in sorted(norm_stats):
        values = list(norm_stats[key])
        digest.update(key.encode("utf-8") + b"\0" + struct.pack("<Q", len(values)))
        for value in values:
            digest.update(struct.pack("<f", float(value)))
    return digest.hexdigest()


def _validate_current_source_files(
    manifest: Mapping[str, Any],
    setting: Mapping[str, Any],
    data_root: str | Path,
    source_files: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    expected_paths = required_dataset_files(manifest, data_root, setting)
    if len(expected_paths) != len(source_files):
        raise ValueError("raw cache source inventory length drifted")
    validated: list[dict[str, Any]] = []
    root = Path(data_root).expanduser().absolute()
    for expected, entry in zip(expected_paths, source_files, strict=True):
        if expected.name != entry.get("path"):
            raise ValueError(f"raw cache source order/name drifted at {expected}")
        stat = expected.stat()
        if stat.st_size != entry.get("size") or stat.st_mtime_ns != entry.get("mtime_ns"):
            raise ValueError(f"source changed after immutable raw cache build: {expected}")
        if SHA256.fullmatch(str(entry.get("sha256", ""))) is None:
            raise ValueError(f"source content digest missing: {expected}")
        validated.append(
            {
                "split": entry["split"],
                "index": entry.get("index"),
                "path": str(expected.relative_to(root)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": entry["sha256"],
            }
        )
    return validated


def run_protocol_sha256(
    manifest: Mapping[str, Any], contract: Mapping[str, Any], *, namespace: str = "formal"
) -> str:
    if namespace not in {"formal", "pilot"}:
        raise ValueError("run protocol namespace must be formal or pilot")
    return stable_hash(
        {
            "campaign_protocol_sha256": protocol_sha256(manifest),
            "namespace": namespace,
            "data_contract_sha256": contract["contract_sha256"],
            "raw_data_manifest_sha256": contract["data_manifest_sha256"],
            "calibration_sha256": contract["calibration_sha256"],
            "future_recipe_sha256": contract["future_recipe_sha256"],
            "chosen_thresholds": contract["chosen_thresholds"],
        }
    )


def load_data_contract(
    manifest: Mapping[str, Any],
    setting: Mapping[str, Any],
    data_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    *,
    contract_root: str | Path | None = None,
    verify_recipe_files: bool = False,
) -> dict[str, Any]:
    """Load and revalidate raw data, train-only calibration, and future recipes."""
    validate_manifest(manifest)
    data_root = data_root or manifest["paths"]["data_root"]
    cache_root = cache_root or manifest["paths"]["raw_cache_root"]
    contract_root = contract_root or manifest["paths"]["contract_root"]
    path = data_contract_path(contract_root, setting["id"])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing or invalid v2 data contract {path}: {exc}") from exc
    claimed = payload.get("contract_sha256")
    body = dict(payload)
    body.pop("contract_sha256", None)
    if SHA256.fullmatch(str(claimed or "")) is None or stable_hash(body) != claimed:
        raise ValueError(f"v2 data contract content hash mismatch: {path}")
    if (
        payload.get("schema_version") != 2
        or payload.get("status") != "complete"
        or payload.get("campaign_id") != manifest["campaign_id"]
        or payload.get("objective_version") != manifest["method"]["objective_version"]
        or payload.get("campaign_protocol_sha256") != protocol_sha256(manifest)
        or payload.get("setting_id") != setting["id"]
        or payload.get("dataset_kind") != setting["dataset_kind"]
        or payload.get("raw_cache_read_only") is not True
        or SHA256.fullmatch(str(payload.get("data_manifest_sha256", ""))) is None
        or SHA256.fullmatch(str(payload.get("train_manifest_sha256", ""))) is None
        or SHA256.fullmatch(str(payload.get("validation_manifest_sha256", ""))) is None
        or SHA256.fullmatch(str(payload.get("normalizer_sha256", ""))) is None
        or SHA256.fullmatch(str(payload.get("calibration_sha256", ""))) is None
        or SHA256.fullmatch(str(payload.get("future_recipe_sha256", ""))) is None
    ):
        raise ValueError(f"v2 data contract identity drifted: {path}")
    raw_manifest_path = raw_cache_manifest_path(cache_root, setting)
    if str(raw_manifest_path) != payload.get("raw_cache_manifest"):
        raise ValueError(f"raw cache manifest path drifted: {path}")
    raw_bytes = raw_manifest_path.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != payload.get("raw_cache_manifest_file_sha256"):
        raise ValueError(f"raw cache manifest content drifted: {raw_manifest_path}")
    raw = json.loads(raw_bytes)
    source_files = list(raw.get("source_files") or [])
    if raw.get("source_manifest_sha256") != payload["data_manifest_sha256"]:
        raise ValueError("trainer/raw-cache data manifest identity drifted")
    if train_inventory_sha256(source_files) != payload["train_manifest_sha256"]:
        raise ValueError("train-only inventory identity drifted")
    if normalizer_sha256(raw.get("norm_stats") or {}) != payload["normalizer_sha256"]:
        raise ValueError("training normalizer identity drifted")
    current_sources = _validate_current_source_files(
        manifest, setting, data_root, source_files
    )
    if current_sources != payload.get("source_files"):
        raise ValueError("v2 data contract source inventory drifted")

    calibration_path = calibration_contract_path(contract_root, setting["id"])
    try:
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing calibration contract {calibration_path}: {exc}") from exc
    from calibration import CalibrationConfig, validate_contract

    validate_contract(
        calibration,
        expected_config=CalibrationConfig(),
        expected_setting_id=setting["id"],
        expected_train_manifest_sha256=payload["train_manifest_sha256"],
        expected_normalizer_sha256=payload["normalizer_sha256"],
        expected_xy_dims=setting["xy_dims"],
        expected_task_metric_dims=setting["task_metric_dims"],
        expected_relative_endpoints=setting["relative_endpoints"],
    )
    if (
        str(calibration_path) != payload.get("calibration_path")
        or calibration.get("contract_sha256") != payload["calibration_sha256"]
        or calibration.get("chosen") != payload.get("chosen_thresholds")
    ):
        raise ValueError("calibration identity/thresholds drifted")

    recipe_manifest = recipe_root_path(contract_root, setting["id"]) / "manifest.json"
    try:
        recipe = json.loads(recipe_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing future recipe manifest {recipe_manifest}: {exc}") from exc
    from treewm.data.future_recipe import validate_recipe_manifest

    live = live_contract(REPOSITORY_ROOT)
    validate_recipe_manifest(
        recipe_manifest.parent,
        recipe,
        expected_source_manifest_sha256=payload["data_manifest_sha256"],
        expected_normalizer_sha256=payload["normalizer_sha256"],
        expected_calibration_sha256=payload["calibration_sha256"],
        expected_thresholds=payload["chosen_thresholds"],
        expected_train_manifest_sha256=payload["train_manifest_sha256"],
        expected_validation_manifest_sha256=payload["validation_manifest_sha256"],
        expected_code_sha256=live["code_sha256"],
        expected_runtime_sha256=live["runtime_sha256"],
        verify_file_hash=verify_recipe_files,
    )
    if (
        str(recipe_manifest) != payload.get("future_recipe_manifest")
        or recipe.get("recipe_sha256") != payload["future_recipe_sha256"]
    ):
        raise ValueError("future recipe identity drifted")
    return payload


def live_contract(repo_root: str | Path) -> dict[str, str]:
    from treewm.utils.provenance import runtime_fingerprint, trainer_code_fingerprint

    code = trainer_code_fingerprint(repo_root)
    runtime = runtime_fingerprint()
    return {"code_sha256": code["manifest_sha256"], "runtime_sha256": runtime["sha256"]}


def _override(name: str, value: object) -> str:
    if isinstance(value, bool):
        rendered = str(value).lower()
    elif value is None:
        rendered = "null"
    elif isinstance(value, (list, tuple)):
        rendered = "[" + ",".join(
            _override("", item).split("=", 1)[-1] if isinstance(item, (list, tuple)) else str(item)
            for item in value
        ) + "]"
    else:
        rendered = str(value)
    return f"{name}={rendered}"


def trainer_command(
    run_or_manifest: RunSpec | Mapping[str, Any],
    maybe_run: RunSpec | None = None,
    *,
    manifest: Mapping[str, Any] | None = None,
    python_executable: str | Path | None = None,
    repo_root: str | Path,
    run_root: str | Path,
    data_root: str | Path,
    cache_root: str | Path,
    contract_root: str | Path | None = None,
    resume: bool = True,
    wandb_project: str | None = None,
    wandb_mode: str = "online",
) -> tuple[list[str], dict[str, str]]:
    """Build exact v2 trainer argv/env for a formal or isolated pilot run."""
    if isinstance(run_or_manifest, Mapping):
        if maybe_run is None:
            raise TypeError("manifest-first trainer_command requires a RunSpec")
        resolved_manifest = run_or_manifest
        run = maybe_run
    else:
        if manifest is None:
            raise TypeError("trainer_command requires manifest=")
        resolved_manifest = manifest
        run = run_or_manifest
    validate_manifest(resolved_manifest)
    setting = setting_for_run(resolved_manifest, run)
    paths = resolved_manifest["paths"]
    requested_root = str(Path(run_root).expanduser().absolute())
    if requested_root == paths["run_root"]:
        namespace = "formal"
        project = resolved_manifest["logging"]["wandb_project"]
        group = resolved_manifest["logging"]["wandb_group"]
        total_updates = FORMAL_UPDATES
        checkpoint_every = resolved_manifest["training"]["checkpoint_every_updates"]
        final_episodes = resolved_manifest["evaluation"]["final_episodes_per_task"]
        wandb_id = run.wandb_id
    elif requested_root == paths["pilot_run_root"]:
        namespace = "pilot"
        project = resolved_manifest["logging"]["pilot_wandb_project"]
        group = resolved_manifest["logging"]["pilot_wandb_group"]
        total_updates = resolved_manifest["training"]["pilot_updates"]
        checkpoint_every = resolved_manifest["training"]["pilot_checkpoint_every_updates"]
        final_episodes = resolved_manifest["evaluation"]["pilot_final_episodes_per_task"]
        wandb_id = _wandb_id(group, run.setting_id, run.seed)
    else:
        raise ValueError("run_root must be the immutable formal or pilot v2 root")
    if "treewm-50task-1m-v1" in requested_root:
        raise ValueError("v1 run/checkpoint roots are forbidden")
    if wandb_project is not None and wandb_project != project:
        raise ValueError("W&B project is immutable for this run namespace")
    if wandb_mode != "online":
        raise ValueError("formal and pilot v2 runs require online W&B")
    locked_python = os.path.abspath(os.fspath(Path(paths["python"]).expanduser()))
    requested_python = os.path.abspath(
        os.fspath(Path(python_executable or paths["python"]).expanduser())
    )
    if requested_python != locked_python:
        raise ValueError(f"formal Python path is immutable ({locked_python!r})")
    if not Path(locked_python).is_file() or not os.access(locked_python, os.X_OK):
        raise ValueError(f"formal Python is not executable: {locked_python}")

    contract = load_data_contract(
        resolved_manifest,
        setting,
        data_root=data_root,
        cache_root=cache_root,
        contract_root=contract_root,
    )
    live = live_contract(repo_root)
    training = resolved_manifest["training"]
    evaluation = resolved_manifest["evaluation"]
    method = resolved_manifest["method"]
    objective = resolved_manifest["objective"]
    matching_contract = objective["matching_contract"]
    loss_contract = objective["loss_contract"]
    optimizer_contract = objective["optimizer_contract"]
    gain_training_contract = objective["gain_training_contract"]
    model_contract = objective["model_contract"]
    tree_contract = objective["tree_contract"]
    planner_contract = objective["planner_contract"]
    chosen = contract["chosen_thresholds"]
    final_run_dir = run_directory(requested_root, run)
    argv = [
        locked_python,
        str(Path(repo_root).absolute() / "scripts" / "train.py"),
        _override("env", run.env_config),
        _override("experiment", method["experiment_config"]),
        _override("arm", "treewm"),
        _override("objective_version", method["objective_version"]),
        _override("seed", run.seed),
        _override("run_root", requested_root),
        _override("run_name", run.run_name),
        _override("resume", "auto" if resume else None),
        _override("train.steps", total_updates),
        _override("train.scheduler_total_steps", training["scheduler_total_steps"]),
        _override("train.gradient_checkpointing", True),
        _override("train.batch_size", training["batch_size"]),
        _override("train.max_train_anchors", training["max_train_anchors"]),
        _override("train.max_val_anchors", training["max_validation_anchors"]),
        _override("train.num_workers", training["data_loader_workers"]),
        _override("train.ckpt_every", checkpoint_every),
        _override("train.eval_every", training["periodic_evaluation_every_updates"]),
        _override("train.viz_every", training["visualization_every_updates"]),
        _override("train.viz_every_early", training["early_visualization_every_updates"]),
        _override("train.viz_early_until", training["early_visualization_until_update"]),
        _override("train.separate_gain_grad_clip", objective["separate_gain_grad_clip"]),
        _override("train.world_grad_clip", objective["world_grad_clip"]),
        _override("train.gain_grad_clip", objective["gain_grad_clip"]),
        *[
            _override(f"train.{key}", value)
            for key, value in optimizer_contract.items()
        ],
        _override("train.gain_loss_every", gain_training_contract["loss_every"]),
        _override("train.gain_batch_size", gain_training_contract["batch_size"]),
        _override("train.gain_tree_budget", gain_training_contract["tree_budget"]),
        _override("tree.node_budget", method["node_budget"]),
        _override("tree.scorer", method["scorer"]),
        _override("tree.keep_threshold", objective["keep_threshold"]),
        *[
            _override(f"tree.{key}", value)
            for key, value in tree_contract.items()
        ],
        _override("model.branch_factor", method["branch_factor"]),
        _override("model.scales", objective["q_scales"]),
        _override("model.use_depth_embedding", objective["use_depth_embedding"]),
        *[
            _override(f"model.{key}", value)
            for key, value in model_contract.items()
        ],
        _override("future_sets.shared_cache", training["shared_cache"]),
        _override("future_sets.cache", training["future_set_cache"]),
        _override("future_sets.metric_mode", objective["future_metric_mode"]),
        _override("future_sets.num_neighbors", training["future_num_neighbors"]),
        _override("future_sets.query_multiplier", training["future_query_multiplier"]),
        _override("future_sets.time_exclusion", training["future_time_exclusion"]),
        _override("future_sets.include_self", training["future_include_self"]),
        _override("future_sets.horizons", training["future_horizons"]),
        _override("future_sets.h_max", training["future_h_max"]),
        _override("future_sets.horizon_rule", training["future_horizon_rule"]),
        _override("future_sets.fixed_horizon", training["future_fixed_horizon"]),
        _override("future_sets.cluster_method", training["future_cluster_method"]),
        _override("future_sets.multi_step_depth", training["future_multi_step_depth"]),
        _override("future_sets.retrieval_pool", training["future_retrieval_pool"]),
        _override("future_sets.relative_endpoints", setting["relative_endpoints"]),
        _override("future_sets.max_modes", objective["max_modes"]),
        _override("future_sets.retrieval_radius", chosen["retrieval_radius"]),
        _override("future_sets.displacement_threshold", chosen["displacement_threshold"]),
        _override("future_sets.cluster_threshold", chosen["cluster_threshold"]),
        _override("+env.task_metric_dims", setting["task_metric_dims"]),
        _override("matching.normalization_version", objective["matching_normalization_version"]),
        _override("matching.num_horizons", objective["matching_num_horizons"]),
        *[
            _override(f"matching.{key}", value)
            for key, value in matching_contract.items()
        ],
        _override("losses.control_objective", objective["control_objective"]),
        _override("losses.control_target_transform", objective["control_target_transform"]),
        _override("losses.control_endpoint_key", objective["control_endpoint_key"]),
        _override("losses.control_allow_endpoint_fallback", objective["control_allow_endpoint_fallback"]),
        _override("losses.control_require_single_scale", objective["control_require_single_scale"]),
        _override("losses.control_metric_weight", objective["control_metric_weight"]),
        _override("losses.control_rank_weight", objective["control_rank_weight"]),
        _override("losses.control_rank_temperature", objective["control_rank_temperature"]),
        _override("losses.detach_world_targets", objective["detach_world_targets"]),
        _override("losses.bind_negative_margin", objective["bind_negative_margin"]),
        _override("losses.gain_target", objective["gain_target"]),
        _override("losses.gain_set_context", objective["gain_set_context"]),
        _override("losses.gain_rank_weight", objective["gain_rank_weight"]),
        _override("losses.gain_calibration_weight", objective["gain_calibration_weight"]),
        _override("losses.gain_branch_prior_weight", objective["gain_branch_prior_weight"]),
        _override("losses.keep_balance", objective["keep_balance"]),
        *[
            _override(f"losses.{key}", loss_contract[key])
            for key in (
                "scheduled_sampling_p",
                "scheduled_sampling_warmup",
                "multistep_depth_weights",
                "redundancy_temperature",
                "contrastive_temperature",
                "coverage_space",
                "future_scale",
                "control_batch",
                "recursive_batch",
            )
        ],
        *[
            _override(f"losses.warmup.{key}", value)
            for key, value in loss_contract["warmup"].items()
        ],
        *[
            _override(f"losses.decay.{key}", value)
            for key, value in loss_contract["decay"].items()
        ],
        *[
            _override(f"losses.weights.{key}", value)
            for key, value in loss_contract["weights"].items()
        ],
        *[
            _override(f"losses.enabled.{key}", value)
            for key, value in loss_contract["enabled"].items()
        ],
        _override("retrieval.enabled", training["latent_retrieval_enabled"]),
        _override("retrieval.num_keys", training["latent_retrieval_keys"]),
        _override("eval.task_split", evaluation["task_split"]),
        _override("eval.episodes_per_task", training["periodic_episodes_per_task"]),
        _override("eval.final_episodes_per_task", final_episodes),
        _override("eval.seed", run.seed),
        *[
            _override(f"planner.{key}", value)
            for key, value in planner_contract.items()
        ],
        _override("planner.max_env_steps", setting["max_episode_steps"]),
        _override("+campaign_calibration_sha256", contract["calibration_sha256"]),
        _override("+campaign_data_contract_sha256", contract["contract_sha256"]),
        _override("+campaign_future_recipe_sha256", contract["future_recipe_sha256"]),
        _override("hydra.run.dir", final_run_dir / "hydra"),
        _override("hydra.job.chdir", False),
    ]
    env = {
        "TREEWM_PROTOCOL_SHA256": run_protocol_sha256(
            resolved_manifest, contract, namespace=namespace
        ),
        "TREEWM_CODE_SHA256": live["code_sha256"],
        "TREEWM_RUNTIME_SHA256": live["runtime_sha256"],
        # The loader validates this released-source digest directly. Calibration and
        # recipe have their own named immutable identity fields below.
        "TREEWM_DATA_SHA256": contract["data_manifest_sha256"],
        "TREEWM_CALIBRATION_SHA256": contract["calibration_sha256"],
        "TREEWM_FUTURE_RECIPE_SHA256": contract["future_recipe_sha256"],
        "TREEWM_DATA_CONTRACT_SHA256": contract["contract_sha256"],
        "TREEWM_DATA_ROOT": str(Path(data_root).expanduser().absolute()),
        "TREEWM_CACHE": str(Path(cache_root).expanduser().absolute()),
        "TREEWM_FUTURE_RECIPE_ROOT": str(recipe_root_path(contract_root or paths["contract_root"], run.setting_id)),
        "TREEWM_RUN_NAME": run.run_name,
        "WANDB_PROJECT": project,
        "WANDB_RUN_GROUP": group,
        "WANDB_RUN_ID": wandb_id,
        "WANDB_MODE": "online",
        "OMP_NUM_THREADS": str(training["loader_thread_limit"]),
        "MKL_NUM_THREADS": str(training["loader_thread_limit"]),
        "OPENBLAS_NUM_THREADS": str(training["loader_thread_limit"]),
    }
    return argv, env


def _config_contract_is_valid(
    config_path: Path,
    identity_config_sha256: str,
    manifest: Mapping[str, Any],
    run: RunSpec,
    contract: Mapping[str, Any],
) -> bool:
    """Prove the Hydra-resolved config contains every calibrated v2 invariant."""
    try:
        from omegaconf import OmegaConf

        config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
        assert isinstance(config, dict)
        identity_config = copy.deepcopy(config)
        identity_config["resume"] = None
        if stable_hash(identity_config) != identity_config_sha256:
            return False
        objective = manifest["objective"]
        training = manifest["training"]
        matching_contract = objective["matching_contract"]
        loss_contract = objective["loss_contract"]
        optimizer_contract = objective["optimizer_contract"]
        gain_training_contract = objective["gain_training_contract"]
        model_contract = objective["model_contract"]
        tree_contract = objective["tree_contract"]
        planner_contract = objective["planner_contract"]
        chosen = contract["chosen_thresholds"]
        expected = {
            "objective_version": manifest["method"]["objective_version"],
            "arm": "treewm",
            "seed": run.seed,
            "run_name": run.run_name,
            "train.steps": FORMAL_UPDATES,
            "train.scheduler_total_steps": FORMAL_UPDATES,
            "train.gradient_checkpointing": True,
            "train.batch_size": training["batch_size"],
            "train.separate_gain_grad_clip": True,
            "train.world_grad_clip": objective["world_grad_clip"],
            "train.gain_grad_clip": objective["gain_grad_clip"],
            "train.gain_loss_every": gain_training_contract["loss_every"],
            "train.gain_batch_size": gain_training_contract["batch_size"],
            "train.gain_tree_budget": gain_training_contract["tree_budget"],
            "model.branch_factor": 4,
            "model.scales": objective["q_scales"],
            "model.use_depth_embedding": False,
            "tree.node_budget": manifest["method"]["node_budget"],
            "tree.scorer": "learned",
            "tree.keep_threshold": 0.5,
            "tree.expansion_batch_size": tree_contract["expansion_batch_size"],
            "tree.max_depth": tree_contract["max_depth"],
            "tree.branch_factor": tree_contract["branch_factor"],
            "tree.context_pooling": tree_contract["context_pooling"],
            "future_sets.metric_mode": "rms_v2",
            "future_sets.max_modes": 4,
            "future_sets.cache": False,
            "future_sets.shared_cache": True,
            "future_sets.num_neighbors": training["future_num_neighbors"],
            "future_sets.query_multiplier": training["future_query_multiplier"],
            "future_sets.time_exclusion": training["future_time_exclusion"],
            "future_sets.include_self": training["future_include_self"],
            "future_sets.horizons": training["future_horizons"],
            "future_sets.h_max": training["future_h_max"],
            "future_sets.horizon_rule": training["future_horizon_rule"],
            "future_sets.fixed_horizon": training["future_fixed_horizon"],
            "future_sets.cluster_method": training["future_cluster_method"],
            "future_sets.multi_step_depth": training["future_multi_step_depth"],
            "future_sets.retrieval_pool": training["future_retrieval_pool"],
            "future_sets.retrieval_radius": chosen["retrieval_radius"],
            "future_sets.displacement_threshold": chosen["displacement_threshold"],
            "future_sets.cluster_threshold": chosen["cluster_threshold"],
            "env.task_metric_dims": run.setting_id and setting_for_run(manifest, run)["task_metric_dims"],
            "matching.normalization_version": "rms_v2",
            "matching.num_horizons": objective["matching_num_horizons"],
            "retrieval.enabled": False,
            "retrieval.num_keys": 0,
            "losses.control_objective": objective["control_objective"],
            "losses.control_target_transform": "rms_tanh",
            "losses.control_endpoint_key": "fut_metric_endpoint",
            "losses.control_allow_endpoint_fallback": False,
            "losses.control_require_single_scale": True,
            "losses.control_metric_weight": objective["control_metric_weight"],
            "losses.control_rank_weight": objective["control_rank_weight"],
            "losses.control_rank_temperature": objective["control_rank_temperature"],
            "losses.detach_world_targets": True,
            "losses.bind_negative_margin": objective["bind_negative_margin"],
            "losses.gain_target": objective["gain_target"],
            "losses.gain_set_context": True,
            "losses.gain_rank_weight": objective["gain_rank_weight"],
            "losses.gain_calibration_weight": objective["gain_calibration_weight"],
            "losses.gain_branch_prior_weight": 0.0,
            "losses.keep_balance": False,
            "losses.scheduled_sampling_p": loss_contract["scheduled_sampling_p"],
            "losses.scheduled_sampling_warmup": loss_contract["scheduled_sampling_warmup"],
            "losses.multistep_depth_weights": loss_contract["multistep_depth_weights"],
            "losses.redundancy_temperature": loss_contract["redundancy_temperature"],
            "losses.contrastive_temperature": loss_contract["contrastive_temperature"],
            "losses.coverage_space": loss_contract["coverage_space"],
            "losses.future_scale": loss_contract["future_scale"],
            "losses.control_batch": loss_contract["control_batch"],
            "losses.recursive_batch": loss_contract["recursive_batch"],
            "planner.max_env_steps": setting_for_run(manifest, run)["max_episode_steps"],
            "campaign_calibration_sha256": contract["calibration_sha256"],
            "campaign_data_contract_sha256": contract["contract_sha256"],
            "campaign_future_recipe_sha256": contract["future_recipe_sha256"],
        }
        expected.update(
            {f"train.{key}": value for key, value in optimizer_contract.items()}
        )
        expected.update(
            {f"model.{key}": value for key, value in model_contract.items()}
        )
        expected.update(
            {f"matching.{key}": value for key, value in matching_contract.items()}
        )
        for section in ("warmup", "decay", "weights", "enabled"):
            expected.update(
                {
                    f"losses.{section}.{key}": value
                    for key, value in loss_contract[section].items()
                }
            )
        expected.update(
            {f"planner.{key}": value for key, value in planner_contract.items()}
        )
        return all(_nested_get(config, key) == value for key, value in expected.items())
    except (AssertionError, KeyError, OSError, TypeError, ValueError):
        return False


def completion_is_valid(
    run_dir: str | Path,
    run_or_manifest: RunSpec | Mapping[str, Any],
    maybe_run: RunSpec | None = None,
    *,
    manifest: Mapping[str, Any] | None = None,
    repo_root: str | Path,
    data_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    contract_root: str | Path | None = None,
) -> bool:
    """Strict completion validation; v1 paths/IDs and stale calibration fail closed."""
    if isinstance(run_or_manifest, Mapping):
        if maybe_run is None:
            return False
        resolved_manifest = run_or_manifest
        run = maybe_run
    else:
        if manifest is None:
            return False
        resolved_manifest = manifest
        run = run_or_manifest
    try:
        validate_manifest(resolved_manifest)
        expected_dir = run_directory(resolved_manifest["paths"]["run_root"], run)
        candidate_dir = Path(run_dir).expanduser().absolute()
        if candidate_dir != expected_dir or "treewm-50task-1m-v1" in str(candidate_dir):
            return False
        setting = setting_for_run(resolved_manifest, run)
        contract = load_data_contract(
            resolved_manifest,
            setting,
            data_root=data_root,
            cache_root=cache_root,
            contract_root=contract_root,
        )
        live = live_contract(repo_root)
        payload = json.loads((candidate_dir / "COMPLETED.json").read_text(encoding="utf-8"))
        identity = payload["run_identity"]
        metrics = payload["final_evaluation"]
        expected_identity = {
            "schema_version": 1,
            "objective_version": resolved_manifest["method"]["objective_version"],
            "run_dir": str(candidate_dir),
            "run_name": run.run_name,
            "arm": "treewm",
            "env_name": run.env_name,
            "setting": run.setting_id,
            "dataset_kind": run.dataset_kind,
            "source_name": run.source_name,
            "seed": run.seed,
            "total_steps": FORMAL_UPDATES,
            "scheduler_total_steps": FORMAL_UPDATES,
            "world_size": 1,
            "model_class": "TreeWM",
            "scorer": "learned",
            "node_budget": 64,
            "branch_factor": 4,
            "gradient_checkpointing": True,
            "future_set_cache": False,
            "shared_cache": True,
            "task_ids": list(FORMAL_TASK_IDS),
            "final_episodes_per_task": 50,
            "protocol_sha256": run_protocol_sha256(resolved_manifest, contract),
            "code_sha256": live["code_sha256"],
            "runtime_sha256": live["runtime_sha256"],
            "data_manifest_sha256": contract["data_manifest_sha256"],
            "calibration_sha256": contract["calibration_sha256"],
            "future_recipe_sha256": contract["future_recipe_sha256"],
            "retrieval_enabled": False,
            "retrieval_num_keys": 0,
            "wandb_project": resolved_manifest["logging"]["wandb_project"],
            "wandb_group": resolved_manifest["logging"]["wandb_group"],
            "wandb_mode": "online",
            "wandb_id": run.wandb_id,
        }
        if any(identity.get(key) != value for key, value in expected_identity.items()):
            return False
        config_sha = str(identity.get("config_sha256", ""))
        if SHA256.fullmatch(config_sha) is None:
            return False
        if stable_hash(identity) != payload.get("identity_sha256"):
            return False
        if not _config_contract_is_valid(
            candidate_dir / "hydra" / ".hydra" / "config.yaml",
            config_sha,
            resolved_manifest,
            run,
            contract,
        ):
            return False
        if (
            payload.get("schema_version") != 1
            or payload.get("objective_version") != resolved_manifest["method"]["objective_version"]
            or payload.get("status") != "complete"
            or payload.get("protocol_sha256") != expected_identity["protocol_sha256"]
            or payload.get("code_sha256") != live["code_sha256"]
            or payload.get("runtime_sha256") != live["runtime_sha256"]
            or payload.get("data_manifest_sha256") != contract["data_manifest_sha256"]
            or payload.get("calibration_sha256") != contract["calibration_sha256"]
            or payload.get("future_recipe_sha256") != contract["future_recipe_sha256"]
            or payload.get("retrieval_enabled") is not False
            or payload.get("retrieval_num_keys") != 0
            or payload.get("arm") != "treewm"
            or payload.get("model_class") != "TreeWM"
            or payload.get("scorer") != "learned"
            or payload.get("setting") != run.setting_id
            or payload.get("seed") != run.seed
            or payload.get("wandb_id") != run.wandb_id
            or payload.get("completed_updates") != FORMAL_UPDATES
            or payload.get("scheduler_total_steps") != FORMAL_UPDATES
            or payload.get("final_eval_step") != FORMAL_UPDATES
            or payload.get("task_ids") != list(FORMAL_TASK_IDS)
            or payload.get("episodes_per_task") != 50
            or payload.get("node_budget") != 64
            or payload.get("branch_factor") != 4
            or payload.get("gradient_checkpointing") is not True
            or payload.get("future_set_cache") is not False
            or payload.get("shared_cache") is not True
            or metrics.get("eval/num_episodes") != 250
        ):
            return False
        for task_id in FORMAL_TASK_IDS:
            success = metrics.get(f"eval/task{task_id}/success_rate")
            if (
                metrics.get(f"eval/task{task_id}/num_episodes") != 50
                or not isinstance(success, (int, float))
                or not math.isfinite(float(success))
                or not 0 <= float(success) <= 1
            ):
                return False
        progress_path = candidate_dir / str(payload.get("final_eval_progress", ""))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        return bool(
            progress.get("schema_version") == 1
            and progress.get("objective_version") == resolved_manifest["method"]["objective_version"]
            and progress.get("status") == "complete"
            and progress.get("identity_sha256") == payload.get("identity_sha256")
            and progress.get("task_ids") == list(FORMAL_TASK_IDS)
            and progress.get("episodes_per_task") == 50
            and len(progress.get("completed_results", ())) == 250
            and progress.get("metrics") == metrics
        )
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError, ImportError):
        return False
