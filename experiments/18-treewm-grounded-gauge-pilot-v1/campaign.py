#!/usr/bin/env python3
"""Fail-closed contracts and deterministic launch mapping for gauge pilot 18."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
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
PINNED_FORMAL_PYTHON = "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RUNS = 30
CONTINUATION_RUNS = 20
PROMOTED_RUNS = 10
SEEDS = (104, 105)
STAGE_TARGETS = (5_000, 25_000)
TASK_IDS = (1, 2, 3, 4, 5)
SETTING_IDS = (
    "antmaze-large",
    "scene",
    "puzzle-3x3",
    "puzzle-4x4-100m",
    "cube-quadruple-100m",
)
ARM_IDS = ("N", "G", "GS")
CONTINUATION_ARM_IDS = ("G", "GS")
EXPECTED_UNION_COUNTS = {
    "antmaze-large": (758_084, 75_816),
    "scene": (758_084, 75_816),
    "puzzle-3x3": (758_084, 75_816),
    "puzzle-4x4-100m": (1_194_586, 119_473),
    "cube-quadruple-100m": (1_194_586, 119_473),
}
INFERENCE_PROFILES = {
    "learned_guard_on": {"scorer": "learned", "require_first_edge_improvement": True},
    "learned_guard_off": {"scorer": "learned", "require_first_edge_improvement": False},
    "bfs_guard_on": {"scorer": "bfs", "require_first_edge_improvement": True},
    "bfs_guard_off": {"scorer": "bfs", "require_first_edge_improvement": False},
    "novelty_q_guard_on": {"scorer": "novelty_q", "require_first_edge_improvement": True},
    "novelty_q_guard_off": {"scorer": "novelty_q", "require_first_edge_improvement": False},
}
PROTOCOL_FILES = (
    "manifest.json",
    "campaign.py",
    "worker.py",
    "stage_gate.py",
    "submit.py",
    "train.slurm",
    "gate.slurm",
    "README.md",
)


class ContractError(RuntimeError):
    """A scientific, provenance, lifecycle, or reporting contract is not exact."""


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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_reject_duplicate_keys)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read exact JSON object {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON artifact is not an object: {path}")
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
    try:
        descriptor = os.open(destination.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest = read_json(path)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    require(manifest.get("schema_version") == 1, "manifest schema drifted")
    require(manifest.get("campaign_id") == "treewm-grounded-gauge-pilot-v1", "campaign ID drifted")
    require(manifest.get("classification") == "bounded_causal_gauge_pilot", "classification drifted")
    require(manifest.get("formal_validation") is False, "pilot was mislabeled formal")
    require("not formal validation" in str(manifest.get("claim_policy", "")), "claim guard missing")
    require(manifest.get("expected_stage_5000_runs") == RUNS, "5k fleet size drifted")
    require(manifest.get("expected_stage_25000_slots") == CONTINUATION_RUNS, "25k slot count drifted")
    require(manifest.get("expected_promoted_runs") == PROMOTED_RUNS, "promoted fleet size drifted")

    diagnosis = manifest.get("engineering_diagnosis") or {}
    require(diagnosis.get("campaign_id") == "treewm-grounded-repair-pilot-v1", "diagnosis campaign drifted")
    require(diagnosis.get("status") == "engineering_aborted_gauge_collapse", "exp15 status drifted")
    require(diagnosis.get("accepted_prerequisite") is False, "aborted exp15 became a prerequisite")
    require(diagnosis.get("resume_allowed") is False, "exp15 resume became eligible")
    require(diagnosis.get("scheduler_array_job_id") == "33127349", "diagnostic job identity drifted")
    for key in ("package_protocol_sha256", "source_sha256"):
        require(SHA256.fullmatch(str(diagnosis.get(key, ""))) is not None, f"bad diagnosis {key}")

    method = manifest.get("method") or {}
    require(method == {
        "arm": "treewm",
        "model_class": "TreeWM",
        "experiment_config": "treewm_v2_grounded_gauge_pilot",
        "objective_version": "treewm_v2_grounded_gauge_pilot_v1",
        "node_budget": 64,
        "branch_factor": 4,
    }, "method contract drifted")
    inference = manifest.get("inference_choice") or {}
    require(inference.get("profiles") == INFERENCE_PROFILES, "inference profiles drifted")
    require(inference.get("profile") == "learned_guard_on", "inference choice drifted")
    require("cannot change it" in str(inference.get("decision_policy", "")), "prospective inference lock missing")

    design = manifest.get("design") or {}
    require(tuple(design.get("seeds") or ()) == SEEDS, "fresh seeds drifted")
    require(tuple(design.get("task_ids") or ()) == TASK_IDS, "task IDs drifted")
    require(tuple(design.get("stage_5000_arms") or ()) == ARM_IDS, "5k arms drifted")
    require(tuple(design.get("continuation_arms") or ()) == CONTINUATION_ARM_IDS, "continuation arms drifted")
    require(design.get("promotion_precedence") == ["G", "GS"], "promotion precedence drifted")
    require(design.get("nonpromotable_arms") == ["N"], "N promotability drifted")
    require("all ten" in str(design.get("promotion_policy", "")), "universal candidate gate missing")
    require("start from scratch" in str(design.get("fresh_start_policy", "")), "fresh start guard missing")

    arms = manifest.get("arms") or []
    require(tuple(arm.get("id") for arm in arms) == ARM_IDS, "arm order drifted")
    common = (
        3e-5, "grounded_execution_v2",
        (1.0, 1.0, 0.25),
        (0.25, 0.5, 0.25, 0.5),
    )
    expected = {
        "N": (False, False, 0.0, False, 1.0),
        "G": (True, True, 1.0, False, 1.0),
        "GS": (True, True, 1.0, True, 1.0),
    }
    for arm in arms:
        actual_common = (
            arm.get("world_lr"), arm.get("transition_mode"),
            (
                arm.get("grounded_select_action_weight"),
                arm.get("grounded_select_endpoint_weight"),
                arm.get("grounded_select_horizon_weight"),
            ),
            (
                arm.get("grounded_loss_latent_weight"),
                arm.get("grounded_loss_action_weight"),
                arm.get("grounded_loss_horizon_weight"),
                arm.get("grounded_loss_endpoint_weight"),
            ),
        )
        require(actual_common == common, f"arm {arm.get('id')}: historical-C recipe drifted")
        actual = (
            arm.get("promotable"),
            arm.get("latent_gauge_enabled"),
            arm.get("latent_gauge_weight"),
            arm.get("separate_branch_transformer_grad_clip"),
            arm.get("branch_transformer_grad_clip"),
        )
        require(actual == expected[arm["id"]], f"arm {arm['id']}: gauge/clip contract drifted")

    scientific = manifest.get("scientific_contract") or {}
    exact = {
        "optimizer_updates": 25_000,
        "scheduler_total_steps": 1_000_000,
        "checkpoint_every_updates": 1_000,
        "validation_every_updates": 1_000,
        "diagnostics_every_updates": 1_000,
        "periodic_evaluation_every_updates": 12_500,
        "periodic_evaluation_updates": [12_500, 25_000],
        "periodic_episodes_per_task": 1,
        "final_episodes_per_task": 5,
        "validation_sample_seed": 1701,
        "evaluation_seed": 2718,
        "task_split": "standard",
        "data_loader_workers": 10,
        "loader_thread_limit": 1,
        "gradient_checkpointing": True,
        "world_weight_decay": 1e-3,
        "model_dropout": 0.1,
        "model_max_depth": 3,
        "tree_max_depth": 3,
        "keep_threshold": 0.5,
        "keep_balance": True,
        "multistep_enabled": True,
        "multistep_weight": 1.0,
        "multistep_depth_weights": [1.0, 1.0, 1.0],
        "scheduled_sampling_p": 0.25,
        "scheduled_sampling_warmup": 5_000,
        "scheduled_sampling_granularity": "sequence",
        "grounded_detach_self_fed_parent": True,
        "gain_lr": 3e-4,
        "gain_weight_decay": 0.0,
        "gain_loss_every": 1,
        "gain_training_scorers": ["learned", "novelty_q"],
        "planner_decoded_metric": "domain_raw",
        "planner_execute_mode": "clipped",
        "planner_execute_steps": 4,
        "min_first_edge_improvement": 0.0,
        "latent_gauge_epsilon": 1e-8,
        "latent_gauge_min_reference_scale": 1e-4,
    }
    for key, value in exact.items():
        require(scientific.get(key) == value, f"scientific contract {key} drifted")
    require(SHA256.fullmatch(str(scientific.get("evaluation_seed_protocol_sha256", ""))) is not None, "evaluation protocol hash invalid")
    require(scientific.get("future_config") == {
        "recipe_anchor_policy": "published_union", "num_neighbors": 24,
        "query_multiplier": 6, "time_exclusion": 50, "include_self": True,
        "metric_mode": "rms_v2", "horizons": [4, 8, 16, 32, 64],
        "h_max": 64, "horizon_rule": "displacement", "fixed_horizon": 32,
        "cluster_method": "average", "max_modes": 4, "multi_step_depth": 3,
        "retrieval_pool": 50_000, "cache": False, "shared_cache": True,
    }, "future recipe contract drifted")

    legacy = manifest.get("compatible_v2_recipe_input") or {}
    require(legacy.get("read_only") is True, "recipe input is not read-only")
    require(legacy.get("campaign_id") == "treewm-50task-1m-v2", "recipe campaign drifted")
    require(legacy.get("objective_version") == "treewm_v2_rms_rank_v1", "recipe objective drifted")
    for key in ("campaign_protocol_sha256", "recipe_code_sha256", "recipe_runtime_sha256"):
        require(SHA256.fullmatch(str(legacy.get(key, ""))) is not None, f"bad {key}")

    settings = manifest.get("settings") or []
    require(tuple(setting.get("id") for setting in settings) == SETTING_IDS, "setting order drifted")
    require(len(settings) * len(arms) * len(SEEDS) == RUNS, "5k matrix is not 30 runs")
    for setting in settings:
        setting_id = str(setting["id"])
        require(
            (setting.get("published_union_train_anchors"), setting.get("published_union_validation_anchors"))
            == EXPECTED_UNION_COUNTS[setting_id],
            f"{setting_id}: published-union counts drifted",
        )
        require(setting.get("dataset_kind") in {"standard", "sharded_100m_full"}, f"{setting_id}: dataset kind drifted")
        require(setting.get("task_metric_dims") and len(set(setting["task_metric_dims"])) == len(setting["task_metric_dims"]), f"{setting_id}: task metric dims invalid")
        for key in ("input_contract_sha256", "calibration_sha256", "future_recipe_sha256"):
            require(SHA256.fullmatch(str(setting.get(key, ""))) is not None, f"{setting_id}: bad {key}")

    lifecycle = manifest.get("lifecycle") or {}
    require(tuple(lifecycle.get("stage_targets") or ()) == STAGE_TARGETS, "stage targets drifted")
    require(lifecycle.get("stop_environment") == "TREEWM_STOP_AFTER_UPDATE", "stage stop environment drifted")
    require(lifecycle.get("stage_5000_array") == "0-29%30", "5k array drifted")
    require(lifecycle.get("stage_25000_array") == "0-19%20", "25k array drifted")
    require("SKIPPED_BY_SELECTION" in str(lifecycle.get("stage_25000_nonselected_policy", "")), "durable selection skip missing")
    require("exact exp18 5k checkpoint" in str(lifecycle.get("resume_policy", "")), "exact continuation boundary missing")
    cancellation = str(lifecycle.get("cancellation_policy", ""))
    require("publishes CANCELLED.json before the batch exits" in cancellation, "durable cancellation missing")
    require("never signal the local srun client" in cancellation, "srun cancellation bug not excluded")

    gate = manifest.get("stage_acceptance") or {}
    required_tags = tuple(gate.get("required_finite_tags") or ())
    require(len(required_tags) == len(set(required_tags)) and "latent_gauge/min_ratio" in required_tags, "required gauge telemetry inventory invalid")
    training_tags = set(gate.get("training_exact_target_tags") or ())
    require(set((
        "latent_gauge/min_ratio", "latent_gauge/reference_sealed",
        "latent_gauge/reference_update", "train/grad_norm_world",
        "train/grad_norm_gain", "train/grad_clip_coefficient_world",
        "train/grad_clip_coefficient_gain",
    )) <= training_tags, "exact-target telemetry contract incomplete")
    exact_gate = {
        "training_every_updates": 50,
        "validation_diagnostic_every_updates": 1_000,
        "gradient_recent_window_updates": 5_000,
        "gauge_recent_window_updates": 1_000,
        "min_scale_ratio": 0.8,
        "reference_update": 0,
        "reference_sealed": 1.0,
        "min_paired_mean_ratio_delta_vs_n": 0.0,
        "min_gradient_norm": 1e-8,
        "min_clip_coefficient": 0.05,
        "max_clip_fraction_below_threshold": 0.25,
        "max_validation_regret_fraction": 0.1,
        "max_self_fed_multistep_validation_regret_fraction": 0.1,
        "horizon_uniform_cross_entropy": 1.6094379124341003,
        "horizon_label_fraction_sum_tolerance": 0.02,
        "min_q_advantage": 0.0,
        "min_gain_rank_correlation": 0.1,
        "min_gain_pairwise_accuracy": 0.52,
        "min_gain_eligible_decision_fraction": 0.2,
        "min_gain_ordered_pair_count": 1.0,
        "min_gain_pair_coverage_fraction": 0.01,
        "min_support_recall": 0.5,
        "min_support_precision": 0.25,
        "selection_stage": 5_000,
        "outcome_stage": 25_000,
        "outcome_episodes_per_run": 5,
        "min_total_successes_per_seed": 1,
        "min_mean_distance_reduction_per_seed": 0.0,
        "min_settings_with_both_seed_success": 1,
        "min_settings_with_both_seed_positive_progress": 3,
        "universal_candidate_cells_required": 10,
        "selected_terminal_runs_required": 10,
    }
    for key, value in exact_gate.items():
        require(gate.get(key) == value, f"stage gate {key} drifted")
    require("Every promotable cell" in str(gate.get("unchanged_method_gate_policy", "")), "universal method gate missing")

    snapshot = manifest.get("source_snapshot") or {}
    require(snapshot == {
        "required_for_submit": True, "marker": "SNAPSHOT.json",
        "status": "sealed_read_only", "repo_subdirectory": "repo",
        "writable_files_allowed": False,
    }, "snapshot policy drifted")
    execution = manifest.get("execution") or {}
    require(execution.get("stage_5000_array") == "0-29%30", "execution 5k array drifted")
    require(execution.get("stage_25000_array") == "0-19%20", "execution 25k array drifted")
    require(execution.get("gpus_per_task") == 1 and execution.get("cpus_per_task") == 12, "resource contract drifted")
    require(execution.get("memory_per_task") == "64G" and execution.get("walltime") == "04:00:00", "memory/time contract drifted")
    require(execution.get("sbatch") == "/usr/local/bin/sbatch", "sbatch path drifted")
    require(execution.get("srun") == "/cm/shared/apps/slurm/current/bin/srun", "srun path drifted")
    require(execution.get("scontrol") == "/cm/shared/apps/slurm/current/bin/scontrol", "scontrol path drifted")
    for key in ("python", "data_root", "raw_cache_root", "compatible_contract_root", "run_root"):
        require(Path((manifest.get("paths") or {}).get(key, "")).is_absolute(), f"{key} must be absolute")
    require(manifest["paths"]["python"] == PINNED_FORMAL_PYTHON, "Python interpreter drifted")
    require("grounded-gauge-pilot-v1" in manifest["paths"]["run_root"], "output namespace is not isolated")
    require(manifest.get("logging") == {
        "wandb_project": "treewm-grounded-gauge-pilot-v1",
        "wandb_group": "treewm-grounded-gauge-pilot-v1",
        "wandb_mode": "online",
    }, "W&B namespace drifted")

def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    validate_manifest(manifest)
    return stable_hash(manifest)


def protocol_sha256(campaign_dir: str | Path = CAMPAIGN_DIR) -> str:
    root = Path(campaign_dir).resolve()
    require(len(PROTOCOL_FILES) == len(set(PROTOCOL_FILES)), "duplicate protocol inventory")
    files: dict[str, str] = {}
    for relative in PROTOCOL_FILES:
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"protocol file missing/symlinked: {path}")
        files[relative] = file_sha256(path)
    return stable_hash({"schema_version": 1, "files": files})


def verify_protocol_lock(campaign_dir: str | Path = CAMPAIGN_DIR) -> str:
    root = Path(campaign_dir).resolve()
    try:
        locked = (root / "protocol.sha256").read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise ContractError(f"protocol lock unavailable: {exc}") from exc
    live = protocol_sha256(root)
    require(SHA256.fullmatch(locked) is not None and locked == live, "protocol.sha256 is stale")
    return live


def expand_runs(manifest: Mapping[str, Any]) -> list[RunSpec]:
    validate_manifest(manifest)
    result: list[RunSpec] = []
    for setting_index, setting in enumerate(manifest["settings"]):
        for arm_index, arm in enumerate(manifest["arms"]):
            for seed_index, seed in enumerate(SEEDS):
                index = ((setting_index * len(ARM_IDS)) + arm_index) * len(SEEDS) + seed_index
                require(index == len(result), "array mapping is not contiguous")
                name = f"gauge-{setting['id']}-arm{arm['id'].lower()}-seed{seed}"
                wandb_id = stable_hash({
                    "campaign_id": manifest["campaign_id"],
                    "setting_id": setting["id"],
                    "arm_id": arm["id"],
                    "seed": seed,
                })[:32]
                result.append(RunSpec(
                    index,
                    setting_index,
                    arm_index,
                    seed_index,
                    setting["id"],
                    setting["env_config"],
                    arm["id"],
                    seed,
                    name,
                    wandb_id,
                ))
    require(len(result) == RUNS, "run expansion did not produce 30 runs")
    require(len({run.run_name for run in result}) == RUNS, "run names collide")
    require(len({run.wandb_id for run in result}) == RUNS, "W&B IDs collide")
    return result


def run_at(manifest: Mapping[str, Any], index: int) -> RunSpec:
    require(0 <= index < RUNS, "5k array index must be in [0,30)")
    return expand_runs(manifest)[index]


def continuation_runs(manifest: Mapping[str, Any]) -> list[RunSpec]:
    runs = [run for run in expand_runs(manifest) if run.arm_id in CONTINUATION_ARM_IDS]
    require(len(runs) == CONTINUATION_RUNS, "continuation mapping did not produce 20 slots")
    require(tuple(run.arm_id for run in runs[:4]) == ("G", "G", "GS", "GS"), "continuation ordering drifted")
    return runs


def run_at_stage(manifest: Mapping[str, Any], stage_target: int, index: int) -> RunSpec:
    require(stage_target in STAGE_TARGETS, "invalid stage target")
    runs = expand_runs(manifest) if stage_target == STAGE_TARGETS[0] else continuation_runs(manifest)
    require(0 <= index < len(runs), "stage-local array index is out of range")
    return runs[index]


def setting_for(manifest: Mapping[str, Any], run: RunSpec) -> Mapping[str, Any]:
    setting = manifest["settings"][run.setting_index]
    require(setting["id"] == run.setting_id, "run/setting identity differs")
    return setting


def arm_for(manifest: Mapping[str, Any], run: RunSpec) -> Mapping[str, Any]:
    arm = manifest["arms"][run.arm_index]
    require(arm["id"] == run.arm_id, "run/arm identity differs")
    return arm


def inference_profile(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    choice = manifest["inference_choice"]
    return choice["profiles"][choice["profile"]]


def run_directory(manifest: Mapping[str, Any], run: RunSpec) -> Path:
    return Path(manifest["paths"]["run_root"]) / run.setting_id / "treewm" / run.run_name


def recipe_root(manifest: Mapping[str, Any], setting_id: str) -> Path:
    return Path(manifest["paths"]["compatible_contract_root"]) / "future-recipes" / setting_id


def data_contract_path(manifest: Mapping[str, Any], setting_id: str) -> Path:
    return Path(manifest["paths"]["compatible_contract_root"]) / "data" / f"{setting_id}.json"


def recipe_audit_anchors(anchors: Sequence[int], sample_count: int = 257) -> list[int]:
    values = [int(value) for value in anchors]
    require(values and all(a < b for a, b in zip(values, values[1:])), "recipe anchors not ordered/unique")
    count = min(int(sample_count), len(values))
    require(count > 0, "recipe audit sample count must be positive")
    if count == 1:
        return [values[0]]
    positions = [(i * (len(values) - 1)) // (count - 1) for i in range(count)]
    selected = [values[position] for position in positions]
    require(len(selected) == len(set(selected)), "recipe audit selection duplicated anchors")
    return selected


def load_compatible_input(
    manifest: Mapping[str, Any],
    setting_or_run: Mapping[str, Any] | RunSpec,
    *,
    verify_files: bool = False,
) -> dict[str, Any]:
    setting = setting_for(manifest, setting_or_run) if isinstance(setting_or_run, RunSpec) else setting_or_run
    contract = read_json(data_contract_path(manifest, setting["id"]))
    claimed = contract.get("contract_sha256")
    body = dict(contract)
    body.pop("contract_sha256", None)
    require(claimed == stable_hash(body) == setting["input_contract_sha256"], f"{setting['id']}: data contract hash differs")
    legacy = manifest["compatible_v2_recipe_input"]
    expected = {
        "campaign_id": legacy["campaign_id"],
        "objective_version": legacy["objective_version"],
        "campaign_protocol_sha256": legacy["campaign_protocol_sha256"],
        "setting_id": setting["id"],
        "dataset_kind": setting["dataset_kind"],
        "calibration_sha256": setting["calibration_sha256"],
        "future_recipe_sha256": setting["future_recipe_sha256"],
        "raw_cache_read_only": True,
    }
    for key, value in expected.items():
        require(contract.get(key) == value, f"{setting['id']}: compatible input {key} differs")
    require(contract.get("train_manifest_sha256") != contract.get("validation_manifest_sha256"), f"{setting['id']}: train/validation manifest identities overlap")
    source_files = contract.get("source_files") or []
    train_sources = [row for row in source_files if row.get("split") == "train"]
    validation_sources = [row for row in source_files if row.get("split") in {"val", "validation"}]
    require(train_sources and validation_sources, f"{setting['id']}: source split coverage is incomplete")
    require(
        not {str(row.get("path")) for row in train_sources}.intersection(
            str(row.get("path")) for row in validation_sources
        ),
        f"{setting['id']}: train/validation paths overlap",
    )
    require(
        not {str(row.get("sha256")) for row in train_sources}.intersection(
            str(row.get("sha256")) for row in validation_sources
        ),
        f"{setting['id']}: train/validation hashes overlap",
    )
    root = recipe_root(manifest, setting["id"])
    composite = read_json(root / "manifest.json")
    require(composite.get("recipe_sha256") == setting["future_recipe_sha256"], f"{setting['id']}: recipe hash differs")
    from treewm.data.future_recipe import FutureRecipe, validate_recipe_manifest

    validate_recipe_manifest(
        root,
        composite,
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
    audits: dict[str, Any] = {}
    for split, manifest_key, expected_count in (
        ("train", "train_manifest", setting["published_union_train_anchors"]),
        ("validation", "validation_manifest", setting["published_union_validation_anchors"]),
    ):
        split_recipe = FutureRecipe(root / Path(composite[manifest_key]).parent)
        require(len(split_recipe.anchors) == expected_count, f"{setting['id']}: {split} union count differs")
        selected = recipe_audit_anchors(split_recipe.anchors)
        require(split_recipe.contains_all(selected), f"{setting['id']}: {split} audit coverage failed")
        audits[split] = {
            "recipe_anchor_count": len(split_recipe.anchors),
            "audit_anchor_count": len(selected),
            "audit_anchor_sha256": stable_hash(selected),
        }
    result = dict(contract)
    result["recipe_coverage_audit"] = audits
    return result


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


def snapshot_identity_sha256(source: Mapping[str, Any], package_protocol_sha256: str) -> str:
    return stable_hash({
        "source_sha256": source["source_sha256"],
        "runtime_sha256": source["runtime_sha256"],
        "package_protocol_sha256": package_protocol_sha256,
    })


def assert_snapshot_files_read_only(root: str | Path) -> int:
    snapshot_root = Path(root).resolve()
    regular_files = [path for path in snapshot_root.rglob("*") if path.is_file()]
    require(regular_files, "snapshot repository contains no source files")
    require(all(not path.is_symlink() for path in regular_files), "snapshot contains symlinked source")
    require(all(path.stat().st_mode & 0o222 == 0 for path in regular_files), "snapshot has writable source files")
    return len(regular_files)


def verify_source_snapshot(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    marker = read_json(root.parent / "SNAPSHOT.json")
    require(marker.get("schema_version") == 1, "snapshot schema differs")
    require(marker.get("status") == "sealed_read_only", "source snapshot is not sealed")
    require(marker.get("repo_subdirectory") == root.name == "repo", "snapshot repo path differs")
    require(marker.get("repo_files_writable") is False, "snapshot permits writable files")
    source = source_contract(root)
    protocol = verify_protocol_lock(root / "experiments" / "18-treewm-grounded-gauge-pilot-v1")
    identity = snapshot_identity_sha256(source, protocol)
    require(marker.get("trainer_source_sha256") == source["source_sha256"], "snapshot trainer source differs")
    require(marker.get("runtime_sha256") == source["runtime_sha256"], "snapshot runtime differs")
    require(marker.get("package_protocol_sha256") == protocol, "snapshot package protocol differs")
    require(marker.get("snapshot_identity_sha256") == identity, "snapshot identity differs")
    require(root.parent.name == identity, "snapshot directory identity differs")
    assert_snapshot_files_read_only(root)
    return {
        "marker": str(root.parent / "SNAPSHOT.json"),
        "source_sha256": source["source_sha256"],
        "runtime_sha256": source["runtime_sha256"],
        "package_protocol_sha256": protocol,
        "snapshot_identity_sha256": identity,
    }


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


def actual_evaluation_bank(manifest: Mapping[str, Any]) -> dict[str, Any]:
    scientific = manifest["scientific_contract"]
    base = int(scientific["evaluation_seed"])
    episodes = int(scientific["final_episodes_per_task"])
    rows = [
        [base + 1000 * task_index + episode_index for episode_index in range(episodes)]
        for task_index, _task_id in enumerate(TASK_IDS)
    ]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "policy": "fixed_cfg_eval_seed_fallback",
        "task_ids": list(TASK_IDS),
        "episodes_per_task": episodes,
        "seeds": rows,
    }
    payload["sha256"] = stable_hash(payload)
    return payload


def scientific_overrides(
    manifest: Mapping[str, Any], run: RunSpec, contract: Mapping[str, Any]
) -> list[str]:
    setting = setting_for(manifest, run)
    arm = arm_for(manifest, run)
    method = manifest["method"]
    scientific = manifest["scientific_contract"]
    future = scientific["future_config"]
    chosen = contract["chosen_thresholds"]
    profile = inference_profile(manifest)
    return [
        _override("env", run.env_config),
        _override("experiment", method["experiment_config"]),
        _override("arm", method["arm"]),
        _override("objective_version", method["objective_version"]),
        _override("seed", run.seed),
        _override("train.steps", scientific["optimizer_updates"]),
        _override("train.scheduler_total_steps", scientific["scheduler_total_steps"]),
        _override("train.ckpt_every", scientific["checkpoint_every_updates"]),
        _override("train.val_every", scientific["validation_every_updates"]),
        _override("train.diag_every", scientific["diagnostics_every_updates"]),
        _override("train.eval_every", scientific["periodic_evaluation_every_updates"]),
        _override("train.validation_sample_seed", scientific["validation_sample_seed"]),
        _override("train.max_train_anchors", setting["published_union_train_anchors"]),
        _override("train.max_val_anchors", setting["published_union_validation_anchors"]),
        _override("train.num_workers", scientific["data_loader_workers"]),
        _override("train.lr", arm["world_lr"]),
        _override("train.weight_decay", scientific["world_weight_decay"]),
        _override("train.gradient_checkpointing", scientific["gradient_checkpointing"]),
        _override("train.separate_gain_grad_clip", True),
        _override("train.separate_branch_transformer_grad_clip", arm["separate_branch_transformer_grad_clip"]),
        _override("train.world_grad_clip", 1.0),
        _override("train.gain_grad_clip", 1.0),
        _override("train.branch_transformer_grad_clip", arm["branch_transformer_grad_clip"]),
        _override("train.gain_loss_every", scientific["gain_loss_every"]),
        _override("train.gain_lr", scientific["gain_lr"]),
        _override("train.gain_weight_decay", scientific["gain_weight_decay"]),
        _override("train.gain_training_scorers", scientific["gain_training_scorers"]),
        _override("train.viz_every", 25_000),
        _override("train.viz_every_early", 1_000),
        _override("train.viz_early_until", 2_000),
        _override("model.dropout", scientific["model_dropout"]),
        _override("model.max_depth", scientific["model_max_depth"]),
        _override("tree.max_depth", scientific["tree_max_depth"]),
        _override("tree.node_budget", method["node_budget"]),
        _override("tree.keep_threshold", scientific["keep_threshold"]),
        _override("tree.scorer", profile["scorer"]),
        _override("model.branch_factor", method["branch_factor"]),
        _override("planner.decoded_metric", scientific["planner_decoded_metric"]),
        _override("planner.execute_mode", scientific["planner_execute_mode"]),
        _override("planner.execute_steps", scientific["planner_execute_steps"]),
        _override("planner.max_env_steps", setting["max_episode_steps"]),
        _override("planner.require_first_edge_improvement", profile["require_first_edge_improvement"]),
        _override("planner.min_first_edge_improvement", scientific["min_first_edge_improvement"]),
        *[_override(f"future_sets.{key}", value) for key, value in future.items()],
        _override("future_sets.relative_endpoints", setting["relative_endpoints"]),
        _override("future_sets.retrieval_radius", chosen["retrieval_radius"]),
        _override("future_sets.displacement_threshold", chosen["displacement_threshold"]),
        _override("future_sets.cluster_threshold", chosen["cluster_threshold"]),
        _override("+env.task_metric_dims", setting["task_metric_dims"]),
        _override("losses.keep_balance", scientific["keep_balance"]),
        _override("losses.enabled.multistep", scientific["multistep_enabled"]),
        _override("losses.weights.multistep", scientific["multistep_weight"]),
        _override("losses.scheduled_sampling_p", scientific["scheduled_sampling_p"]),
        _override("losses.scheduled_sampling_warmup", scientific["scheduled_sampling_warmup"]),
        _override("losses.scheduled_sampling_granularity", scientific["scheduled_sampling_granularity"]),
        _override("losses.multistep_transition_mode", arm["transition_mode"]),
        _override("losses.grounded_select_action_weight", arm["grounded_select_action_weight"]),
        _override("losses.grounded_select_endpoint_weight", arm["grounded_select_endpoint_weight"]),
        _override("losses.grounded_select_horizon_weight", arm["grounded_select_horizon_weight"]),
        _override("losses.grounded_loss_latent_weight", arm["grounded_loss_latent_weight"]),
        _override("losses.grounded_loss_action_weight", arm["grounded_loss_action_weight"]),
        _override("losses.grounded_loss_horizon_weight", arm["grounded_loss_horizon_weight"]),
        _override("losses.grounded_loss_endpoint_weight", arm["grounded_loss_endpoint_weight"]),
        _override("losses.grounded_detach_self_fed_parent", scientific["grounded_detach_self_fed_parent"]),
        _override("losses.multistep_depth_weights", scientific["multistep_depth_weights"]),
        _override("losses.enabled.latent_gauge", arm["latent_gauge_enabled"]),
        _override("losses.weights.latent_gauge", arm["latent_gauge_weight"]),
        _override("losses.latent_gauge_epsilon", scientific["latent_gauge_epsilon"]),
        _override("losses.latent_gauge_min_reference_scale", scientific["latent_gauge_min_reference_scale"]),
        _override("eval.task_split", scientific["task_split"]),
        _override("eval.episodes_per_task", scientific["periodic_episodes_per_task"]),
        _override("eval.final_episodes_per_task", scientific["final_episodes_per_task"]),
        _override("eval.seed", scientific["evaluation_seed"]),
        _override("+campaign_input_contract_sha256", contract["contract_sha256"]),
        _override("+campaign_calibration_sha256", contract["calibration_sha256"]),
        _override("+campaign_future_recipe_sha256", contract["future_recipe_sha256"]),
        _override("+campaign_compatible_recipe_code_sha256", manifest["compatible_v2_recipe_input"]["recipe_code_sha256"]),
        _override("+campaign_factorial_arm", run.arm_id),
    ]


def trainer_command(
    manifest: Mapping[str, Any],
    run: RunSpec,
    *,
    repo_root: str | Path = REPOSITORY_ROOT,
    verify_recipe_files: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    package = root / "experiments" / "18-treewm-grounded-gauge-pilot-v1"
    protocol = verify_protocol_lock(package)
    contract = load_compatible_input(manifest, run, verify_files=verify_recipe_files)
    source = source_contract(root)
    overrides = scientific_overrides(manifest, run, contract)
    config_sha = stable_hash({"schema_version": 1, "overrides": overrides})
    scientific = manifest["scientific_contract"]
    from treewm.evaluation.rollout import build_evaluation_seed_tables

    seed_tables = build_evaluation_seed_tables(
        scientific["evaluation_seed_protocol_sha256"],
        run.seed,
        TASK_IDS,
        scientific["periodic_episodes_per_task"],
        scientific["final_episodes_per_task"],
    )
    actual_bank = actual_evaluation_bank(manifest)
    run_protocol = stable_hash({
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "package_protocol_sha256": protocol,
        "source_sha256": source["source_sha256"],
        "runtime_sha256": source["runtime_sha256"],
        "config_sha256": config_sha,
        "input_contract_sha256": contract["contract_sha256"],
        "data_manifest_sha256": contract["data_manifest_sha256"],
        "normalizer_sha256": contract["normalizer_sha256"],
        "train_manifest_sha256": contract["train_manifest_sha256"],
        "validation_manifest_sha256": contract["validation_manifest_sha256"],
        "calibration_sha256": contract["calibration_sha256"],
        "future_recipe_sha256": contract["future_recipe_sha256"],
        "evaluation_seed_protocol_sha256": scientific["evaluation_seed_protocol_sha256"],
        "actual_evaluation_bank_sha256": actual_bank["sha256"],
    })
    output = run_directory(manifest, run)
    argv = [
        manifest["paths"]["python"],
        str(root / "scripts" / "train.py"),
        *overrides,
        _override("run_root", manifest["paths"]["run_root"]),
        _override("run_name", run.run_name),
        _override("resume", "auto"),
        _override("+campaign_source_sha256", source["source_sha256"]),
        _override("+campaign_protocol_sha256", protocol),
        _override("+campaign_config_sha256", config_sha),
        _override("hydra.run.dir", output / "hydra"),
        _override("hydra.job.chdir", False),
    ]
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
        "TREEWM_DATA_CONTRACT_SHA256": contract["contract_sha256"],
        "TREEWM_DATA_ROOT": manifest["paths"]["data_root"],
        "TREEWM_CACHE": manifest["paths"]["raw_cache_root"],
        "TREEWM_FUTURE_RECIPE_ROOT": str(recipe_root(manifest, run.setting_id)),
        "TREEWM_EVALUATION_SEED_PROTOCOL_SHA256": scientific["evaluation_seed_protocol_sha256"],
        "TREEWM_EXPECTED_FINAL_SEED_TABLE_SHA256": seed_tables["final"]["sha256"],
        "TREEWM_RUN_NAME": run.run_name,
        "WANDB_PROJECT": manifest["logging"]["wandb_project"],
        "WANDB_RUN_GROUP": manifest["logging"]["wandb_group"],
        "WANDB_RUN_ID": run.wandb_id,
        "WANDB_MODE": manifest["logging"]["wandb_mode"],
        "OMP_NUM_THREADS": str(scientific["loader_thread_limit"]),
        "MKL_NUM_THREADS": str(scientific["loader_thread_limit"]),
        "OPENBLAS_NUM_THREADS": str(scientific["loader_thread_limit"]),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    launch: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "classification": manifest["classification"],
        "formal_validation": False,
        "run": {**asdict(run), "run_directory": str(output)},
        "hashes": {
            "manifest_sha256": manifest_sha256(manifest),
            "source_sha256": source["source_sha256"],
            "runtime_sha256": source["runtime_sha256"],
            "package_protocol_sha256": protocol,
            "config_sha256": config_sha,
            "run_protocol_sha256": run_protocol,
            "input_contract_sha256": contract["contract_sha256"],
            "data_manifest_sha256": contract["data_manifest_sha256"],
            "normalizer_sha256": contract["normalizer_sha256"],
            "train_manifest_sha256": contract["train_manifest_sha256"],
            "validation_manifest_sha256": contract["validation_manifest_sha256"],
            "calibration_sha256": contract["calibration_sha256"],
            "future_recipe_sha256": contract["future_recipe_sha256"],
            "recipe_code_sha256": manifest["compatible_v2_recipe_input"]["recipe_code_sha256"],
            "recipe_runtime_sha256": manifest["compatible_v2_recipe_input"]["recipe_runtime_sha256"],
            "evaluation_seed_tables_sha256": seed_tables["sha256"],
            "final_seed_table_sha256": seed_tables["final"]["sha256"],
            "actual_evaluation_bank_sha256": actual_bank["sha256"],
        },
        "argv": argv,
        "environment": environment,
    }
    launch["launch_sha256"] = stable_hash(launch)
    return launch


def verify_all(
    manifest: Mapping[str, Any],
    *,
    repo_root: str | Path = REPOSITORY_ROOT,
    verify_files: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    protocol = verify_protocol_lock(root / "experiments" / "18-treewm-grounded-gauge-pilot-v1")
    recipe_audits = {
        setting["id"]: load_compatible_input(manifest, setting, verify_files=verify_files)["recipe_coverage_audit"]
        for setting in manifest["settings"]
    }
    runs = expand_runs(manifest)
    source = source_contract(root)
    configs = {
        run.run_name: stable_hash({
            "schema_version": 1,
            "overrides": scientific_overrides(manifest, run, load_compatible_input(manifest, run)),
        })
        for run in runs
    }
    return {
        "schema_version": 1,
        "status": "verified_bounded_causal_gauge_pilot",
        "formal_validation": False,
        "campaign_id": manifest["campaign_id"],
        "stage_5000_runs": len(runs),
        "stage_25000_slots": len(continuation_runs(manifest)),
        "manifest_sha256": manifest_sha256(manifest),
        "package_protocol_sha256": protocol,
        "source_sha256": source["source_sha256"],
        "runtime_sha256": source["runtime_sha256"],
        "recipe_files_verified": bool(verify_files),
        "recipe_coverage_audits": recipe_audits,
        "actual_evaluation_bank": actual_evaluation_bank(manifest),
        "unique_config_sha256": sorted(set(configs.values())),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("protocol-hash", "verify", "snapshot", "runs", "command"))
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--verify-files", action="store_true")
    parser.add_argument("--index", type=int)
    parser.add_argument("--stage-target", type=int, choices=STAGE_TARGETS, default=STAGE_TARGETS[0])
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    if args.command == "protocol-hash":
        print(protocol_sha256(root / "experiments" / "18-treewm-grounded-gauge-pilot-v1"))
        return 0
    manifest = load_manifest(args.manifest)
    if args.command == "verify":
        value = verify_all(manifest, repo_root=root, verify_files=args.verify_files)
    elif args.command == "snapshot":
        value = verify_source_snapshot(root)
    elif args.command == "runs":
        value = {"runs": [asdict(run) for run in expand_runs(manifest)]}
    else:
        require(args.index is not None, "command requires --index")
        value = trainer_command(manifest, run_at_stage(manifest, args.stage_target, args.index), repo_root=root)
        if args.output:
            atomic_json(args.output, value)
    print(json.dumps(value, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"gauge-pilot campaign error: {exc}", file=sys.stderr)
        raise SystemExit(2)
