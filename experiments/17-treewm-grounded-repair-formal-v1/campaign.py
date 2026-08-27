#!/usr/bin/env python3
"""Sealed contracts and deterministic mappings for repaired formal campaign 17."""

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
SEED_TABLE_PATH = CAMPAIGN_DIR / "eval_seed_table.json"
PREREQUISITE_BINDINGS_PATH = CAMPAIGN_DIR / "prerequisite_bindings.json"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
TRAINING_RUNS = 40
FINAL_EVAL_TASKS = 200
PINNED_FORMAL_PYTHON = "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python"
SEEDS = (200, 201, 202, 203)
TASK_IDS = (1, 2, 3, 4, 5)
STAGE_TARGETS = (2_000, 25_000, 100_000, 1_000_000)
SETTING_IDS = (
    "scene",
    "puzzle-3x3",
    "puzzle-4x4-100m",
    "cube-double",
    "cube-triple",
    "cube-quadruple-100m",
    "antmaze-large",
    "antmaze-giant",
    "humanoidmaze-medium",
    "humanoidmaze-large",
)
EXPECTED_UNION_COUNTS = {
    "scene": (758_084, 75_816),
    "puzzle-3x3": (758_084, 75_816),
    "puzzle-4x4-100m": (1_194_586, 119_473),
    "cube-double": (758_084, 75_816),
    "cube-triple": (1_030_685, 102_824),
    "cube-quadruple-100m": (1_194_586, 119_473),
    "antmaze-large": (758_084, 75_816),
    "antmaze-giant": (759_154, 76_196),
    "humanoidmaze-medium": (955_698, 95_746),
    "humanoidmaze-large": (955_698, 95_746),
}
PROTOCOL_FILES = (
    "manifest.json",
    "campaign.py",
    "worker.py",
    "stage_gate.py",
    "submit.py",
    "final_eval.py",
    "aggregate.py",
    "train.slurm",
    "final_eval.slurm",
    "gate.slurm",
    "aggregate.slurm",
    "eval_seed_table.json",
    "prerequisite_bindings.json",
    "bind_prerequisites.py",
    "README.md",
)
EVALUATION_SOURCE_FILES = (
    "scripts/__init__.py",
    "scripts/eval.py",
)


class ContractError(RuntimeError):
    """A scientific, provenance, lifecycle, or evaluation contract is not exact."""


@dataclass(frozen=True)
class RunSpec:
    index: int
    setting_index: int
    seed_index: int
    setting_id: str
    env_config: str
    seed: int
    run_name: str
    wandb_id: str


@dataclass(frozen=True)
class EvalSpec:
    index: int
    training_index: int
    task_id: int
    run: RunSpec


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
    require(manifest.get("campaign_id") == "treewm-grounded-repair-formal-v1", "campaign ID drifted")
    require(manifest.get("classification") == "fresh_formal_validation", "classification drifted")
    require(manifest.get("expected_training_runs") == TRAINING_RUNS, "training count drifted")
    require(manifest.get("expected_final_eval_tasks") == FINAL_EVAL_TASKS, "eval count drifted")
    authority = manifest.get("promotion_authority") or {}
    require(authority.get("pilot_gate") == "accepted_exp15_repair_pilot", "pilot decision drifted")
    require(
        authority.get("recipe_gate") == "deterministic_exp16_all_ten_bridge_selection",
        "recipe decision drifted",
    )
    require(authority.get("bindings_file") == "prerequisite_bindings.json", "binding path drifted")
    require(authority.get("old_checkpoints_allowed") is False, "old checkpoints became eligible")
    require(
        authority.get("no_outcome_based_recipe_selection_within_formal") is True,
        "formal adaptive recipe selection became eligible",
    )

    method = manifest.get("method") or {}
    expected_method = {
        "arm": "treewm",
        "model_class": "TreeWM",
        "experiment_config": "treewm_v2_grounded_repair_formal",
        "objective_version": "treewm_v2_grounded_repair_formal_v1",
        "scorer": "learned",
        "final_eval_rails": ["learned", "bfs"],
        "node_budget": 64,
        "branch_factor": 4,
        "regularization": {"lr": 3e-5, "weight_decay": 1e-3, "dropout": 0.1},
        "grounded_multistep": {
            "enabled": True,
            "weight": 1.0,
            "depth_weights": [1.0, 1.0, 1.0],
            "scheduled_sampling_p": 0.25,
            "scheduled_sampling_warmup": 5000,
            "scheduled_sampling_granularity": "sequence",
            "transition_mode": "grounded_execution_v2",
            "selector_weights": {"action": 1.0, "endpoint": 1.0, "horizon": 0.25},
            "detach_self_fed_parent": True,
            "keep_balance": True,
            "selected_recipe_source": "exp16_SELECTED_RECIPE.json",
        },
    }
    require(method == expected_method, "corrected grounded method recipe drifted")

    prerequisites = manifest.get("prerequisites") or {}
    require(prerequisites.get("bindings_file") == "prerequisite_bindings.json", "prerequisite binding file drifted")
    exp15 = prerequisites.get("exp15") or {}
    require(exp15.get("campaign_id") == "treewm-grounded-repair-pilot-v1", "exp15 identity drifted")
    require(exp15.get("required_status") == "accepted_for_fresh_formal_campaign_design", "exp15 status contract drifted")
    require(exp15.get("required_candidate_arm") == "C", "exp15 candidate drifted")
    require(exp15.get("required_control_arm") == "A", "exp15 control drifted")
    require(exp15.get("required_integrity_runs") == 40, "exp15 integrity quorum drifted")
    require(
        exp15.get("required_aggregate_gates")
        == [
            "integrity_40_of_40",
            "preregistered_candidate_setting_quorum",
            "candidate_not_all_zero_success",
            "candidate_positive_mean_progress",
            "candidate_positive_progress_run_quorum",
            "candidate_success_noninferior_to_control",
            "candidate_distance_reduction_noninferior_to_control",
            "adaptive_selection_disabled",
        ],
        "exp15 aggregate-gate contract drifted",
    )
    require(
        exp15.get("raw_report_recomputation")
        == {
            "settings": [
                "antmaze-large",
                "scene",
                "puzzle-3x3",
                "puzzle-4x4-100m",
                "cube-quadruple-100m",
            ],
            "arms": ["A", "B", "C", "D"],
            "seeds": [100, 101],
            "required_integrity_gates": [
                "exact_launch_completion_and_provenance",
                "identical_terminal_evaluation_bank",
                "fixed_common_validation_sample",
                "exact_1k_validation_axis",
                "finite_terminal_method_telemetry",
                "midpoint_and_terminal_rollouts",
            ],
            "required_scientific_gates": [
                "validation_regret_le_1p10",
                "self_fed_multistep_regret_le_1p10",
                "horizon_below_uniform_and_empirical_prior",
                "q_advantage",
                "gain_rank_pair_eligibility_and_coverage",
                "support_recall_and_precision",
                "aggregate_shared_module_gradients_and_clipping",
            ],
            "terminal_episodes_per_run": 25,
            "candidate_settings_required": 4,
            "candidate_seeds_per_passing_setting": 2,
            "min_candidate_total_successes": 1,
            "min_candidate_runs_with_positive_progress": 6,
            "min_candidate_mean_distance_reduction_exclusive": 0.0,
            "min_paired_success_delta_vs_control": 0.0,
            "min_paired_distance_reduction_delta_vs_control": 0.0,
            "nonpromotable_sensitivity_arms": ["B", "D"],
            "no_adaptive_arm_selection": True,
        },
        "exp15 raw-report recomputation contract drifted",
    )
    require(
        exp15.get("package_protocol_sha256")
        == "ec41f19a97ab0c21d341b00baa69a6f50259408adda2d7bc6428ff46398a4f49"
        and exp15.get("source_sha256")
        == "dc0b5d2c80a25c6ac51495696e83450859de4429fe9e40137572b1e981510d6a"
        and exp15.get("runtime_sha256")
        == "77da91d49a1db99850fbf0632dc02ec58a3209f1a87949d6f5640ae6bf505c6b",
        "exp15 pinned package/source/runtime identity drifted",
    )
    exp16 = prerequisites.get("exp16") or {}
    require(exp16.get("campaign_id") == "treewm-grounded-repair-all-ten-bridge-v1", "exp16 identity drifted")
    require(
        exp16.get("accepted_statuses") == [
            "selected_full_for_fresh_formal_campaign_design",
            "selected_half_for_fresh_formal_campaign_design",
        ],
        "exp16 accepted statuses drifted",
    )
    require(exp16.get("allowed_selected_arms") == ["F", "H"], "exp16 arm set drifted")
    require(exp16.get("selection_precedence") == ["F", "H"], "exp16 precedence drifted")
    require(
        exp16.get("manifest_sha256")
        == "6028a37b1174bafe04dd20a026f58fe7285158eda14615a3430494ffd27e5fa3"
        and exp16.get("package_protocol_sha256")
        == "e53b35b5118b45fa5dec416a8c6b662a94bfd989a1f72a10f756fd908c2b3bbe"
        and exp16.get("source_sha256")
        == "28ebcd66ac25cfb1cbf23359448a065a3f32cd8e5766320e85833eff16f2d74f"
        and exp16.get("runtime_sha256")
        == "77da91d49a1db99850fbf0632dc02ec58a3209f1a87949d6f5640ae6bf505c6b",
        "exp16 pinned manifest/package/source/runtime identity drifted",
    )
    require(
        exp16.get("required_selection_quorum")
        == {
            "bridge_seeds": [102, 103],
            "integrity_runs": 40,
            "runs_per_arm": 20,
            "min_scientific_runs_per_arm": 18,
            "min_settings_with_both_seeds": 8,
            "min_settings_with_at_least_one_seed": 10,
            "min_total_successes_per_arm": 1,
            "min_mean_distance_reduction_exclusive": 0.0,
            "min_positive_progress_runs_per_arm": 12,
            "min_half_minus_full_success_rate": 0.0,
            "min_half_minus_full_distance_reduction": 0.0,
        },
        "exp16 selection quorum drifted",
    )
    require(
        prerequisites.get("allowed_selected_recipes")
        == {
            "F": {
                "grounded_loss_latent_weight": 0.25,
                "grounded_loss_action_weight": 0.5,
                "grounded_loss_horizon_weight": 0.25,
                "grounded_loss_endpoint_weight": 0.5,
            },
            "H": {
                "grounded_loss_latent_weight": 0.125,
                "grounded_loss_action_weight": 0.25,
                "grounded_loss_horizon_weight": 0.125,
                "grounded_loss_endpoint_weight": 0.25,
            },
        },
        "allowed exp16 recipes drifted",
    )
    for dependency in (exp15, exp16):
        for key in ("acceptance_path",):
            require(Path(str(dependency.get(key, ""))).is_absolute(), f"{key} must be absolute")
    require(Path(str(exp15.get("launch_plan_path", ""))).is_absolute(), "exp15 launch plan must be absolute")
    require(Path(str(exp16.get("selected_recipe_path", ""))).is_absolute(), "exp16 recipe path must be absolute")

    design = manifest.get("design") or {}
    require(tuple(design.get("seeds") or ()) == SEEDS, "training seeds drifted")
    require(tuple(design.get("task_ids") or ()) == TASK_IDS, "task IDs drifted")

    scientific = manifest.get("scientific_contract") or {}
    exact_scientific = {
        "optimizer_updates": 1_000_000,
        "scheduler_total_steps": 1_000_000,
        "checkpoint_every_updates": 1_000,
        "training_log_every_updates": 50,
        "validation_every_updates": 2_000,
        "periodic_evaluation_every_updates": 25_000,
        "periodic_episodes_per_task": 1,
        "final_episodes_per_task": 50,
        "validation_sample_seed": 1701,
        "evaluation_seed": 2718,
        "task_split": "standard",
        "data_loader_workers": 10,
        "loader_thread_limit": 1,
        "gradient_checkpointing": True,
        "model_max_depth": 3,
        "tree_max_depth": 3,
        "planner_score_space": "decoded",
        "planner_decoded_metric": "domain_raw",
        "planner_execute_mode": "clipped",
        "planner_execute_steps": 4,
        "require_first_edge_improvement": True,
        "min_first_edge_improvement": 0.0,
        "tree_search_nodes_per_full_budget_replan": 64,
        "first_edge_guard_extra_root_predictions_per_replan": 4,
        "effective_world_predictions_per_full_budget_replan": 68,
        "future_config": {
            "recipe_anchor_policy": "published_union",
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
    }
    require(scientific == exact_scientific, "scientific contract drifted")
    stage_acceptance = manifest.get("stage_acceptance") or {}
    required_finite_tags = stage_acceptance.get("required_finite_tags") or []
    require(
        len(required_finite_tags) == len(set(required_finite_tags))
        and
        "val/loss_multistep" in required_finite_tags
        and "val/loss_multistep_self_fed" in required_finite_tags,
        "required finite telemetry must be unique and include both multistep losses",
    )
    exact_training_tags = [
        "train/grad_norm_world",
        "train/grad_norm_gain",
        "train/grad_clip_coefficient_world",
        "train/grad_clip_coefficient_gain",
    ]
    freshness = stage_acceptance.get("telemetry_freshness") or {}
    require(
        freshness
        == {
            "training_exact_target_tags": exact_training_tags,
            "training_every_updates": 50,
            "validation_diagnostic_every_updates": 2_000,
        },
        "stage telemetry freshness contract drifted",
    )
    require(
        all(tag in required_finite_tags for tag in exact_training_tags),
        "world/gain gradient norms and clip coefficients must all be finite",
    )
    require(
        stage_acceptance.get("gradient_recent_window_updates") == 5_000
        and stage_acceptance.get("min_gradient_norm") == 1e-8
        and stage_acceptance.get("min_clip_coefficient") == 0.05
        and stage_acceptance.get("max_clip_fraction_below_threshold") == 0.25,
        "gradient/clipping numerical-health contract drifted",
    )
    require(
        stage_acceptance.get("max_self_fed_multistep_validation_regret_fraction") == 0.1,
        "self-fed multistep validation nonregression threshold drifted",
    )
    require(stage_acceptance.get("scientific_gate_stage") == 25_000, "25k method gate drifted")
    require(stage_acceptance.get("outcome_sanity_stage") == 100_000, "100k outcome gate drifted")
    require(
        stage_acceptance.get("post_100k_policy") == "integrity_and_numerical_health_only",
        "post-100k continuation policy drifted",
    )
    require(stage_acceptance.get("min_fleet_monitor_successes") == 1, "100k success floor drifted")
    require(stage_acceptance.get("min_fleet_mean_distance_reduction") == 0.0, "100k progress floor drifted")
    require(stage_acceptance.get("min_runs_with_positive_progress") == 1, "100k progress quorum drifted")
    lifecycle = manifest.get("lifecycle") or {}
    require(tuple(lifecycle.get("stage_targets") or ()) == STAGE_TARGETS, "stage targets drifted")
    require(lifecycle.get("stop_environment") == "TREEWM_STOP_AFTER_UPDATE", "stage env drifted")
    require("no external" in str(lifecycle.get("resume_policy", "")), "fresh-resume guard missing")

    legacy = manifest.get("compatible_v2_recipe_input") or {}
    require(legacy.get("read_only") is True, "recipe input is not read-only")
    require(legacy.get("campaign_id") == "treewm-50task-1m-v2", "recipe campaign drifted")
    for key in ("campaign_protocol_sha256", "recipe_code_sha256", "recipe_runtime_sha256"):
        require(SHA256.fullmatch(str(legacy.get(key, ""))) is not None, f"bad {key}")

    settings = manifest.get("settings") or []
    require(tuple(value.get("id") for value in settings) == SETTING_IDS, "setting order drifted")
    require(len(settings) * len(SEEDS) == TRAINING_RUNS, "design does not expand to 40 runs")
    for setting in settings:
        setting_id = str(setting["id"])
        require(
            (setting.get("published_union_train_anchors"), setting.get("published_union_validation_anchors"))
            == EXPECTED_UNION_COUNTS[setting_id],
            f"{setting_id}: published recipe union counts drifted",
        )
        require(setting.get("dataset_kind") in {"standard", "sharded_100m_full"}, f"{setting_id}: dataset kind")
        require(setting.get("task_metric_dims") and len(set(setting["task_metric_dims"])) == len(setting["task_metric_dims"]), f"{setting_id}: task metric dims")
        for key in (
            "input_contract_sha256",
            "calibration_sha256",
            "future_recipe_sha256",
            "evaluation_seed_protocol_sha256",
        ):
            require(SHA256.fullmatch(str(setting.get(key, ""))) is not None, f"{setting_id}: bad {key}")
    require(len({setting["evaluation_seed_protocol_sha256"] for setting in settings}) == 10, "seed protocols collide")

    final = manifest.get("final_evaluation") or {}
    require(final.get("array") == "0-199%40", "final-eval array drifted")
    require(final.get("episodes_per_task_per_rail") == 50, "final episode count drifted")
    require(final.get("rails") == ["learned", "bfs"], "final rails drifted")
    require(final.get("aggregate_requires") == FINAL_EVAL_TASKS, "aggregate count drifted")
    require(final.get("adaptive_selection") is False, "adaptive final selection enabled")
    require(final.get("promotion_criterion") == {
        "min_learned_total_successes": 1,
        "min_paired_seed_delta_ci_lower": 0.0,
        "min_settings_learned_noninferior": 7,
        "primary_inference_unit": "training_seed",
        "training_seed_replicates": 4,
        "t_critical_975_df3": 3.182446,
    }, "final promotion criterion drifted")
    execution = manifest.get("execution") or {}
    require(execution.get("training_array") == "0-39%40", "training array drifted")
    require(execution.get("gpu_partitions") == "polar4,polar3,polar,grizzly", "GPU partitions drifted")
    require(execution.get("cpu_partition") == "cpu", "CPU gate partition drifted")
    require(execution.get("gpus_per_task") == 1, "must be one GPU per job")
    require(execution.get("cpus_per_task") == 12, "CPU contract drifted")
    require(execution.get("memory_per_task") == "64G", "memory contract drifted")
    require(execution.get("sbatch") == "/usr/local/bin/sbatch", "sbatch client drifted")
    require(execution.get("srun") == "/cm/shared/apps/slurm/current/bin/srun", "srun client drifted")
    require(execution.get("scontrol") == "/cm/shared/apps/slurm/current/bin/scontrol", "scontrol client drifted")
    for key in ("python", "data_root", "raw_cache_root", "compatible_contract_root", "run_root", "final_eval_root"):
        require(Path((manifest.get("paths") or {}).get(key, "")).is_absolute(), f"{key} must be absolute")
    require(manifest["paths"]["python"] == PINNED_FORMAL_PYTHON, "formal Python interpreter drifted")
    require("grounded-repair-formal-v1" in manifest["paths"]["run_root"], "run namespace is not fresh")
    require(manifest["logging"] == {
        "wandb_project": "treewm-grounded-repair-formal-v1",
        "wandb_group": "treewm-grounded-repair-formal-v1",
        "wandb_mode": "online",
    }, "W&B namespace drifted")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    validate_manifest(manifest)
    return stable_hash(manifest)


def protocol_sha256(campaign_dir: str | Path = CAMPAIGN_DIR) -> str:
    root = Path(campaign_dir).resolve()
    require(len(PROTOCOL_FILES) == len(set(PROTOCOL_FILES)), "protocol file inventory contains duplicates")
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


def load_seed_table(
    manifest: Mapping[str, Any], path: str | Path = SEED_TABLE_PATH
) -> dict[str, Any]:
    payload = read_json(path)
    claimed = payload.get("sha256")
    body = dict(payload)
    body.pop("sha256", None)
    require(claimed == stable_hash(body), "locked evaluation seed-table hash differs")
    require(payload.get("schema_version") == 1, "seed-table schema differs")
    require(payload.get("campaign_id") == manifest["campaign_id"], "seed-table campaign differs")
    require(payload.get("task_ids") == list(TASK_IDS), "seed-table tasks differ")
    require(payload.get("episodes_per_task") == 50, "seed-table episode count differs")
    tables = payload.get("settings") or {}
    require(set(tables) == set(SETTING_IDS) and len(tables) == len(SETTING_IDS), "seed-table setting coverage differs")
    from treewm.evaluation.rollout import validate_evaluation_seed_table

    for setting in manifest["settings"]:
        table = tables.get(setting["id"]) or {}
        validate_evaluation_seed_table(
            table,
            split="final",
            task_ids=TASK_IDS,
            episodes_per_task=50,
        )
        require(
            table.get("protocol_sha256") == setting["evaluation_seed_protocol_sha256"],
            f"{setting['id']}: evaluation seed protocol differs",
        )
    return payload


def load_prerequisite_bindings(
    manifest: Mapping[str, Any],
    path: str | Path = PREREQUISITE_BINDINGS_PATH,
    *,
    verify_external_files: bool = True,
) -> dict[str, Any]:
    """Load the immutable exp15/exp16 decision binding or fail explicitly as blocked."""
    payload = read_json(path)
    require(
        payload.get("status") == "sealed_accepted_prerequisites",
        "formal campaign is blocked: exp15/exp16 prerequisite binding is not sealed",
    )
    require(payload.get("schema_version") == 1, "prerequisite binding schema differs")
    require(payload.get("campaign_id") == manifest["campaign_id"], "prerequisite binding campaign differs")
    require(payload.get("formal_submission_allowed") is True, "prerequisite binding forbids submission")
    require(
        payload.get("selection_policy")
        == "consume_exp16_deterministic_selection_without_formal_outcome_selection",
        "formal recipe selection policy differs",
    )
    claimed = payload.get("binding_sha256")
    body = dict(payload)
    body.pop("binding_sha256", None)
    require(SHA256.fullmatch(str(claimed or "")) is not None and claimed == stable_hash(body), "prerequisite binding hash differs")
    selected_arm = payload.get("selected_arm")
    require(selected_arm in ("F", "H"), "prerequisite binding selected arm is not F/H")
    selected_recipe = payload.get("selected_recipe")
    require(isinstance(selected_recipe, dict), "prerequisite binding selected recipe is missing")
    require(payload.get("selected_recipe_sha256") == stable_hash(selected_recipe), "selected recipe hash differs")
    expected_losses = manifest["prerequisites"]["allowed_selected_recipes"][selected_arm]
    require(
        all(float(selected_recipe.get(key, -1.0)) == float(value) for key, value in expected_losses.items()),
        "selected recipe loss weights are not the registered F/H arm",
    )
    grounded = manifest["method"]["grounded_multistep"]
    expected_common = {
        "transition_mode": grounded["transition_mode"],
        "grounded_select_action_weight": grounded["selector_weights"]["action"],
        "grounded_select_endpoint_weight": grounded["selector_weights"]["endpoint"],
        "grounded_select_horizon_weight": grounded["selector_weights"]["horizon"],
    }
    require(
        all(selected_recipe.get(key) == value for key, value in expected_common.items()),
        "selected recipe common grounded fields differ",
    )
    exp15 = payload.get("exp15") or {}
    exp16 = payload.get("exp16") or {}
    require(exp15.get("campaign_id") == manifest["prerequisites"]["exp15"]["campaign_id"], "bound exp15 campaign differs")
    require(exp16.get("campaign_id") == manifest["prerequisites"]["exp16"]["campaign_id"], "bound exp16 campaign differs")
    require(exp15.get("accepted_status") == manifest["prerequisites"]["exp15"]["required_status"], "bound exp15 status differs")
    require(exp16.get("accepted_status") in manifest["prerequisites"]["exp16"]["accepted_statuses"], "bound exp16 status differs")
    require(exp16.get("selected_arm") == selected_arm, "bound exp16 arm differs")
    require(exp16.get("selected_recipe") == selected_recipe, "bound exp16 recipe differs")
    require(exp16.get("selected_recipe_sha256") == payload["selected_recipe_sha256"], "bound exp16 recipe hash differs")
    exp15_contract = manifest["prerequisites"]["exp15"]
    require(
        exp15.get("package_protocol_sha256")
        == exp15_contract["package_protocol_sha256"]
        and exp15.get("trainer_source_sha256")
        == exp15_contract["source_sha256"]
        and exp15.get("runtime_sha256") == exp15_contract["runtime_sha256"],
        "bound exp15 pinned package/source/runtime identity differs",
    )
    exp16_contract = manifest["prerequisites"]["exp16"]
    require(
        exp16.get("manifest_sha256") == exp16_contract["manifest_sha256"]
        and exp16.get("package_protocol_sha256")
        == exp16_contract["package_protocol_sha256"]
        and exp16.get("source_sha256") == exp16_contract["source_sha256"]
        and exp16.get("runtime_sha256") == exp16_contract["runtime_sha256"],
        "bound exp16 pinned manifest/package/source/runtime identity differs",
    )
    for dependency, keys in (
        (
            exp15,
            (
                "acceptance_file_sha256",
                "launch_plan_file_sha256",
                "launch_plan_sha256",
                "package_protocol_sha256",
                "trainer_source_sha256",
                "runtime_sha256",
                "actual_evaluation_bank_sha256",
                "exp16_prerequisite_sha256",
            ),
        ),
        (
            exp16,
            (
                "acceptance_file_sha256",
                "acceptance_report_sha256",
                "selected_recipe_file_sha256",
                "selected_recipe_artifact_sha256",
                "selected_recipe_sha256",
                "selection_rule_sha256",
                "manifest_sha256",
                "package_protocol_sha256",
                "source_sha256",
                "runtime_sha256",
                "actual_evaluation_bank_sha256",
                "exp15_prerequisite_sha256",
            ),
        ),
    ):
        for key in keys:
            require(SHA256.fullmatch(str(dependency.get(key, ""))) is not None, f"bound prerequisite has malformed {key}")
    for key, count in (
        ("run_config_sha256", 40),
        ("run_input_contract_sha256", 40),
        ("selected_run_config_sha256", 20),
    ):
        hashes = exp16.get(key)
        require(
            isinstance(hashes, dict)
            and len(hashes) == count
            and all(SHA256.fullmatch(str(value)) is not None for value in hashes.values()),
            f"bound exp16 has malformed {key}",
        )
    exp15_bridge_prerequisite = exp15.get("exp16_prerequisite")
    exp16_bridge_prerequisite = exp16.get("exp15_prerequisite")
    require(
        isinstance(exp15_bridge_prerequisite, dict)
        and isinstance(exp16_bridge_prerequisite, dict)
        and exp15_bridge_prerequisite == exp16_bridge_prerequisite,
        "bound exp15/exp16 prerequisite payloads differ",
    )
    prerequisite_body = dict(exp15_bridge_prerequisite)
    prerequisite_hash = prerequisite_body.pop("prerequisite_sha256", None)
    require(
        prerequisite_hash == stable_hash(prerequisite_body)
        == exp15["exp16_prerequisite_sha256"]
        == exp16["exp15_prerequisite_sha256"],
        "bound exp15/exp16 prerequisite self-hash differs",
    )
    expected_paths = {
        "exp15.acceptance_path": manifest["prerequisites"]["exp15"]["acceptance_path"],
        "exp15.launch_plan_path": manifest["prerequisites"]["exp15"]["launch_plan_path"],
        "exp16.acceptance_path": manifest["prerequisites"]["exp16"]["acceptance_path"],
        "exp16.selected_recipe_path": manifest["prerequisites"]["exp16"]["selected_recipe_path"],
    }
    actual_paths = {
        "exp15.acceptance_path": exp15.get("acceptance_path"),
        "exp15.launch_plan_path": exp15.get("launch_plan_path"),
        "exp16.acceptance_path": exp16.get("acceptance_path"),
        "exp16.selected_recipe_path": exp16.get("selected_recipe_path"),
    }
    for key, expected in expected_paths.items():
        require(Path(str(actual_paths[key])).resolve() == Path(str(expected)).resolve(), f"bound {key} differs")
    if verify_external_files:
        file_checks = (
            (Path(exp15["acceptance_path"]), exp15["acceptance_file_sha256"], "exp15 acceptance"),
            (Path(exp15["launch_plan_path"]), exp15["launch_plan_file_sha256"], "exp15 launch plan"),
            (Path(exp16["acceptance_path"]), exp16["acceptance_file_sha256"], "exp16 acceptance"),
            (Path(exp16["selected_recipe_path"]), exp16["selected_recipe_file_sha256"], "exp16 selected recipe"),
        )
        for artifact, expected_hash, label in file_checks:
            require(artifact.is_file() and not artifact.is_symlink(), f"{label} disappeared or became symlinked")
            require(file_sha256(artifact) == expected_hash, f"{label} bytes changed after binding")
    return payload


def expand_runs(manifest: Mapping[str, Any]) -> list[RunSpec]:
    validate_manifest(manifest)
    runs: list[RunSpec] = []
    for setting_index, setting in enumerate(manifest["settings"]):
        for seed_index, seed in enumerate(SEEDS):
            index = setting_index * len(SEEDS) + seed_index
            name = f"grounded-repair-formal-{setting['id']}-seed{seed}"
            wandb_id = stable_hash({"campaign_id": manifest["campaign_id"], "setting_id": setting["id"], "seed": seed})[:32]
            runs.append(RunSpec(index, setting_index, seed_index, setting["id"], setting["env_config"], seed, name, wandb_id))
    require(len(runs) == TRAINING_RUNS and len({r.run_name for r in runs}) == TRAINING_RUNS, "run expansion differs")
    return runs


def run_at(manifest: Mapping[str, Any], index: int) -> RunSpec:
    require(0 <= index < TRAINING_RUNS, "training index must be in [0,40)")
    return expand_runs(manifest)[index]


def eval_at(manifest: Mapping[str, Any], index: int) -> EvalSpec:
    require(0 <= index < FINAL_EVAL_TASKS, "final-eval index must be in [0,200)")
    training_index, task_offset = divmod(index, len(TASK_IDS))
    return EvalSpec(index, training_index, TASK_IDS[task_offset], run_at(manifest, training_index))


def setting_for(manifest: Mapping[str, Any], run: RunSpec) -> Mapping[str, Any]:
    setting = manifest["settings"][run.setting_index]
    require(setting["id"] == run.setting_id, "run/setting identity differs")
    return setting


def run_directory(manifest: Mapping[str, Any], run: RunSpec) -> Path:
    return Path(manifest["paths"]["run_root"]) / run.setting_id / "treewm" / run.run_name


def recipe_root(manifest: Mapping[str, Any], setting_id: str) -> Path:
    return Path(manifest["paths"]["compatible_contract_root"]) / "future-recipes" / setting_id


def data_contract_path(manifest: Mapping[str, Any], setting_id: str) -> Path:
    return Path(manifest["paths"]["compatible_contract_root"]) / "data" / f"{setting_id}.json"


def recipe_audit_anchors(anchors: Sequence[int], sample_count: int = 257) -> list[int]:
    """Select deterministic audit anchors from the sealed recipe, never a dataset prefix."""
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
    require(
        contract.get("train_manifest_sha256") != contract.get("validation_manifest_sha256"),
        f"{setting['id']}: train/validation split manifest identities overlap",
    )
    source_files = contract.get("source_files") or []
    train_sources = [row for row in source_files if row.get("split") == "train"]
    validation_sources = [row for row in source_files if row.get("split") in {"val", "validation"}]
    require(train_sources and validation_sources, f"{setting['id']}: source split coverage is incomplete")
    train_paths = {str(row.get("path")) for row in train_sources}
    validation_paths = {str(row.get("path")) for row in validation_sources}
    train_hashes = {str(row.get("sha256")) for row in train_sources}
    validation_hashes = {str(row.get("sha256")) for row in validation_sources}
    require(not train_paths.intersection(validation_paths), f"{setting['id']}: train/validation source paths overlap")
    require(not train_hashes.intersection(validation_hashes), f"{setting['id']}: train/validation source hashes overlap")

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
        ("val", "validation_manifest", setting["published_union_validation_anchors"]),
    ):
        split_root = root / Path(composite[manifest_key]).parent
        split_recipe = FutureRecipe(split_root)
        require(len(split_recipe.anchors) == expected_count, f"{setting['id']}: {split} union count differs")
        selected = recipe_audit_anchors(split_recipe.anchors)
        require(split_recipe.contains_all(selected), f"{setting['id']}: recipe-derived audit coverage failed")
        audits[split] = {
            "source": "sealed_recipe_union",
            "recipe_anchor_count": len(split_recipe.anchors),
            "audit_anchor_count": len(selected),
            "audit_anchor_sha256": stable_hash(selected),
        }
    result = dict(contract)
    result["source_split_audit"] = {
        "train_manifest_sha256": contract["train_manifest_sha256"],
        "validation_manifest_sha256": contract["validation_manifest_sha256"],
        "train_source_file_count": len(train_sources),
        "validation_source_file_count": len(validation_sources),
        "path_overlap_count": 0,
        "sha256_overlap_count": 0,
    }
    result["recipe_coverage_audit"] = audits
    return result


def evaluation_source_contract(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    files: dict[str, str] = {}
    for relative in EVALUATION_SOURCE_FILES:
        path = root / relative
        require(path.is_file() and not path.is_symlink(), f"evaluation source missing/symlinked: {path}")
        files[relative] = file_sha256(path)
    return {
        "sha256": stable_hash({"schema_version": 1, "files": files}),
        "files": files,
    }


def source_contract(repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from treewm.utils.provenance import runtime_fingerprint, trainer_code_fingerprint

    source = trainer_code_fingerprint(root)
    runtime = runtime_fingerprint()
    evaluation_source = evaluation_source_contract(root)
    return {
        "source_sha256": source["manifest_sha256"],
        "source_files": source["files"],
        "evaluation_source_sha256": evaluation_source["sha256"],
        "evaluation_source_files": evaluation_source["files"],
        "runtime_sha256": runtime["sha256"],
        "runtime": runtime,
    }


def snapshot_identity_sha256(source: Mapping[str, Any], package_protocol_sha256: str) -> str:
    return stable_hash({
        "source_sha256": source["source_sha256"],
        "evaluation_source_sha256": source["evaluation_source_sha256"],
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
    marker_path = root.parent / "SNAPSHOT.json"
    marker = read_json(marker_path)
    require(marker.get("schema_version") == 1, "snapshot schema differs")
    require(marker.get("status") == "sealed_read_only", "source snapshot is not sealed")
    require(marker.get("repo_subdirectory") == root.name == "repo", "snapshot repo path differs")
    require(marker.get("repo_files_writable") is False, "snapshot permits writable files")
    source = source_contract(root)
    protocol = verify_protocol_lock(root / "experiments" / "17-treewm-grounded-repair-formal-v1")
    snapshot_identity = snapshot_identity_sha256(source, protocol)
    require(marker.get("trainer_source_sha256") == source["source_sha256"], "snapshot trainer source differs")
    require(marker.get("evaluation_source_sha256") == source["evaluation_source_sha256"], "snapshot evaluation source differs")
    require(marker.get("runtime_sha256") == source["runtime_sha256"], "snapshot runtime differs")
    require(marker.get("package_protocol_sha256") == protocol, "snapshot package protocol differs")
    require(marker.get("snapshot_identity_sha256") == snapshot_identity, "snapshot identity differs")
    require(root.parent.name == snapshot_identity, "snapshot directory identity differs")
    assert_snapshot_files_read_only(root)
    return {
        "marker": str(marker_path),
        "source_sha256": source["source_sha256"],
        "evaluation_source_sha256": source["evaluation_source_sha256"],
        "runtime_sha256": source["runtime_sha256"],
        "package_protocol_sha256": protocol,
        "snapshot_identity_sha256": snapshot_identity,
    }


def _override(name: str, value: object) -> str:
    if isinstance(value, bool):
        rendered = str(value).lower()
    elif isinstance(value, (list, tuple)):
        rendered = "[" + ",".join(str(v).lower() if isinstance(v, bool) else str(v) for v in value) + "]"
    else:
        rendered = str(value)
    return f"{name}={rendered}"


def scientific_overrides(
    manifest: Mapping[str, Any],
    run: RunSpec,
    contract: Mapping[str, Any],
    prerequisite_binding: Mapping[str, Any],
) -> list[str]:
    setting = setting_for(manifest, run)
    method = manifest["method"]
    scientific = manifest["scientific_contract"]
    future = scientific["future_config"]
    grounded = method["grounded_multistep"]
    regularization = method["regularization"]
    selected = prerequisite_binding["selected_recipe"]
    chosen = contract["chosen_thresholds"]
    return [
        _override("env", run.env_config),
        _override("experiment", method["experiment_config"]),
        _override("arm", method["arm"]),
        _override("objective_version", method["objective_version"]),
        _override("seed", run.seed),
        _override("train.steps", scientific["optimizer_updates"]),
        _override("train.scheduler_total_steps", scientific["scheduler_total_steps"]),
        _override("train.ckpt_every", scientific["checkpoint_every_updates"]),
        _override("train.log_every", scientific["training_log_every_updates"]),
        _override("train.val_every", scientific["validation_every_updates"]),
        _override("train.diag_every", scientific["validation_every_updates"]),
        _override("train.eval_every", scientific["periodic_evaluation_every_updates"]),
        _override("train.validation_sample_seed", scientific["validation_sample_seed"]),
        _override("train.max_train_anchors", setting["published_union_train_anchors"]),
        _override("train.max_val_anchors", setting["published_union_validation_anchors"]),
        _override("train.num_workers", scientific["data_loader_workers"]),
        _override("train.lr", regularization["lr"]),
        _override("train.weight_decay", regularization["weight_decay"]),
        _override("train.gradient_checkpointing", scientific["gradient_checkpointing"]),
        _override("train.separate_gain_grad_clip", True),
        _override("train.world_grad_clip", 1.0),
        _override("train.gain_grad_clip", 1.0),
        _override("train.gain_loss_every", 1),
        _override("train.gain_lr", 3e-4),
        _override("train.gain_weight_decay", 0.0),
        _override("train.gain_training_scorers", ["learned", "novelty_q"]),
        _override("train.viz_every", 25_000),
        _override("train.viz_every_early", 2_000),
        _override("train.viz_early_until", 25_000),
        _override("model.dropout", regularization["dropout"]),
        _override("model.max_depth", scientific["model_max_depth"]),
        _override("tree.max_depth", scientific["tree_max_depth"]),
        _override("tree.node_budget", method["node_budget"]),
        _override("tree.scorer", method["scorer"]),
        _override("model.branch_factor", method["branch_factor"]),
        _override("planner.score_space", scientific["planner_score_space"]),
        _override("planner.decoded_metric", scientific["planner_decoded_metric"]),
        _override("planner.execute_mode", scientific["planner_execute_mode"]),
        _override("planner.execute_steps", scientific["planner_execute_steps"]),
        _override("planner.max_env_steps", setting["max_episode_steps"]),
        _override("planner.require_first_edge_improvement", scientific["require_first_edge_improvement"]),
        _override("planner.min_first_edge_improvement", scientific["min_first_edge_improvement"]),
        *[_override(f"future_sets.{key}", value) for key, value in future.items()],
        _override("future_sets.relative_endpoints", setting["relative_endpoints"]),
        _override("future_sets.retrieval_radius", chosen["retrieval_radius"]),
        _override("future_sets.displacement_threshold", chosen["displacement_threshold"]),
        _override("future_sets.cluster_threshold", chosen["cluster_threshold"]),
        _override("+env.task_metric_dims", setting["task_metric_dims"]),
        _override("losses.enabled.multistep", grounded["enabled"]),
        _override("losses.weights.multistep", grounded["weight"]),
        _override("losses.scheduled_sampling_p", grounded["scheduled_sampling_p"]),
        _override("losses.scheduled_sampling_warmup", grounded["scheduled_sampling_warmup"]),
        _override("losses.scheduled_sampling_granularity", grounded["scheduled_sampling_granularity"]),
        _override("losses.multistep_transition_mode", grounded["transition_mode"]),
        _override("losses.grounded_select_action_weight", grounded["selector_weights"]["action"]),
        _override("losses.grounded_select_endpoint_weight", grounded["selector_weights"]["endpoint"]),
        _override("losses.grounded_select_horizon_weight", grounded["selector_weights"]["horizon"]),
        _override("losses.grounded_loss_latent_weight", selected["grounded_loss_latent_weight"]),
        _override("losses.grounded_loss_action_weight", selected["grounded_loss_action_weight"]),
        _override("losses.grounded_loss_horizon_weight", selected["grounded_loss_horizon_weight"]),
        _override("losses.grounded_loss_endpoint_weight", selected["grounded_loss_endpoint_weight"]),
        _override("losses.grounded_detach_self_fed_parent", grounded["detach_self_fed_parent"]),
        _override("losses.keep_balance", grounded["keep_balance"]),
        _override("losses.multistep_depth_weights", grounded["depth_weights"]),
        _override("eval.task_split", scientific["task_split"]),
        _override("eval.episodes_per_task", scientific["periodic_episodes_per_task"]),
        _override("eval.final_episodes_per_task", scientific["final_episodes_per_task"]),
        _override("eval.seed", scientific["evaluation_seed"]),
        _override("+campaign_input_contract_sha256", contract["contract_sha256"]),
        _override("+campaign_calibration_sha256", contract["calibration_sha256"]),
        _override("+campaign_future_recipe_sha256", contract["future_recipe_sha256"]),
        _override("+campaign_compatible_recipe_code_sha256", manifest["compatible_v2_recipe_input"]["recipe_code_sha256"]),
        _override("+campaign_factorial_arm", f"exp16-{prerequisite_binding['selected_arm']}"),
        _override("+campaign_prerequisite_binding_sha256", prerequisite_binding["binding_sha256"]),
        _override("+campaign_selected_recipe_sha256", prerequisite_binding["selected_recipe_sha256"]),
    ]


def trainer_command(manifest: Mapping[str, Any], run: RunSpec, *, repo_root: str | Path = REPOSITORY_ROOT, verify_recipe_files: bool = False) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    package = root / "experiments" / "17-treewm-grounded-repair-formal-v1"
    protocol = verify_protocol_lock(package)
    prerequisite_binding = load_prerequisite_bindings(
        manifest,
        package / "prerequisite_bindings.json",
        verify_external_files=True,
    )
    contract = load_compatible_input(manifest, run, verify_files=verify_recipe_files)
    source = source_contract(root)
    overrides = scientific_overrides(manifest, run, contract, prerequisite_binding)
    config_sha = stable_hash({"schema_version": 1, "overrides": overrides})
    setting = setting_for(manifest, run)
    seed_bundle = load_seed_table(manifest, package / "eval_seed_table.json")
    final_seed_table = seed_bundle["settings"][run.setting_id]
    from treewm.evaluation.rollout import build_evaluation_seed_tables

    expected_seed_tables = build_evaluation_seed_tables(
        setting["evaluation_seed_protocol_sha256"],
        run.seed,
        TASK_IDS,
        manifest["scientific_contract"]["periodic_episodes_per_task"],
        manifest["scientific_contract"]["final_episodes_per_task"],
    )
    require(
        expected_seed_tables["final"] == final_seed_table,
        f"{run.setting_id}: locked final seed table differs from trainer generator",
    )
    run_protocol = stable_hash({
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "package_protocol_sha256": protocol,
        "source_sha256": source["source_sha256"],
        "evaluation_source_sha256": source["evaluation_source_sha256"],
        "runtime_sha256": source["runtime_sha256"],
        "config_sha256": config_sha,
        "prerequisite_binding_sha256": prerequisite_binding["binding_sha256"],
        "selected_recipe_sha256": prerequisite_binding["selected_recipe_sha256"],
        "selected_arm": prerequisite_binding["selected_arm"],
        "input_contract_sha256": contract["contract_sha256"],
        "future_recipe_sha256": contract["future_recipe_sha256"],
        "evaluation_seed_protocol_sha256": setting["evaluation_seed_protocol_sha256"],
        "final_seed_table_sha256": final_seed_table["sha256"],
        "monitor_seed_table_sha256": expected_seed_tables["monitor"]["sha256"],
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
        "TREEWM_EVALUATION_SOURCE_SHA256": source["evaluation_source_sha256"],
        "TREEWM_RUNTIME_SHA256": source["runtime_sha256"],
        "TREEWM_RECIPE_CODE_SHA256": manifest["compatible_v2_recipe_input"]["recipe_code_sha256"],
        "TREEWM_RECIPE_RUNTIME_SHA256": manifest["compatible_v2_recipe_input"]["recipe_runtime_sha256"],
        "TREEWM_CONFIG_SHA256": config_sha,
        "TREEWM_PREREQUISITE_BINDING_SHA256": prerequisite_binding["binding_sha256"],
        "TREEWM_SELECTED_RECIPE_SHA256": prerequisite_binding["selected_recipe_sha256"],
        "TREEWM_DATA_SHA256": contract["data_manifest_sha256"],
        "TREEWM_CALIBRATION_SHA256": contract["calibration_sha256"],
        "TREEWM_FUTURE_RECIPE_SHA256": contract["future_recipe_sha256"],
        "TREEWM_DATA_CONTRACT_SHA256": contract["contract_sha256"],
        "TREEWM_DATA_ROOT": manifest["paths"]["data_root"],
        "TREEWM_CACHE": manifest["paths"]["raw_cache_root"],
        "TREEWM_FUTURE_RECIPE_ROOT": str(recipe_root(manifest, run.setting_id)),
        "TREEWM_EVALUATION_SEED_PROTOCOL_SHA256": setting["evaluation_seed_protocol_sha256"],
        "TREEWM_EXPECTED_FINAL_SEED_TABLE_SHA256": final_seed_table["sha256"],
        "TREEWM_RUN_NAME": run.run_name,
        "WANDB_PROJECT": manifest["logging"]["wandb_project"],
        "WANDB_RUN_GROUP": manifest["logging"]["wandb_group"],
        "WANDB_RUN_ID": run.wandb_id,
        "WANDB_MODE": manifest["logging"]["wandb_mode"],
        "OMP_NUM_THREADS": str(manifest["scientific_contract"]["loader_thread_limit"]),
        "MKL_NUM_THREADS": str(manifest["scientific_contract"]["loader_thread_limit"]),
        "OPENBLAS_NUM_THREADS": str(manifest["scientific_contract"]["loader_thread_limit"]),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    launch: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "formal_validation": True,
        "run": {**asdict(run), "run_directory": str(output)},
        "hashes": {
            "manifest_sha256": manifest_sha256(manifest),
            "source_sha256": source["source_sha256"],
            "evaluation_source_sha256": source["evaluation_source_sha256"],
            "runtime_sha256": source["runtime_sha256"],
            "package_protocol_sha256": protocol,
            "config_sha256": config_sha,
            "prerequisite_binding_sha256": prerequisite_binding["binding_sha256"],
            "selected_recipe_sha256": prerequisite_binding["selected_recipe_sha256"],
            "selected_arm": prerequisite_binding["selected_arm"],
            "run_protocol_sha256": run_protocol,
            "input_contract_sha256": contract["contract_sha256"],
            "calibration_sha256": contract["calibration_sha256"],
            "future_recipe_sha256": contract["future_recipe_sha256"],
            "evaluation_seed_protocol_sha256": setting["evaluation_seed_protocol_sha256"],
            "package_seed_table_sha256": seed_bundle["sha256"],
            "final_seed_table_sha256": final_seed_table["sha256"],
            "monitor_seed_table_sha256": expected_seed_tables["monitor"]["sha256"],
        },
        "argv": argv,
        "environment": environment,
    }
    launch["launch_sha256"] = stable_hash(launch)
    return launch


def verify_all(manifest: Mapping[str, Any], *, repo_root: str | Path = REPOSITORY_ROOT, verify_files: bool = False) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    protocol = verify_protocol_lock(root / "experiments" / "17-treewm-grounded-repair-formal-v1")
    prerequisite_binding = load_prerequisite_bindings(
        manifest,
        root / "experiments" / "17-treewm-grounded-repair-formal-v1" / "prerequisite_bindings.json",
        verify_external_files=True,
    )
    seed_table = load_seed_table(
        manifest,
        root / "experiments" / "17-treewm-grounded-repair-formal-v1" / "eval_seed_table.json",
    )
    recipe_audits = {}
    for setting in manifest["settings"]:
        contract = load_compatible_input(manifest, setting, verify_files=verify_files)
        recipe_audits[setting["id"]] = contract["recipe_coverage_audit"]
    runs = expand_runs(manifest)
    source = source_contract(root)
    return {
        "schema_version": 1,
        "status": "verified",
        "campaign_id": manifest["campaign_id"],
        "manifest_sha256": manifest_sha256(manifest),
        "package_protocol_sha256": protocol,
        "prerequisite_binding_sha256": prerequisite_binding["binding_sha256"],
        "selected_recipe_sha256": prerequisite_binding["selected_recipe_sha256"],
        "selected_arm": prerequisite_binding["selected_arm"],
        "source_sha256": source["source_sha256"],
        "evaluation_source_sha256": source["evaluation_source_sha256"],
        "runtime_sha256": source["runtime_sha256"],
        "evaluation_seed_table_sha256": seed_table["sha256"],
        "training_runs": len(runs),
        "final_eval_tasks": len([eval_at(manifest, i) for i in range(FINAL_EVAL_TASKS)]),
        "recipe_anchor_policy": manifest["scientific_contract"]["future_config"]["recipe_anchor_policy"],
        "recipe_coverage_audits": recipe_audits,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("protocol-hash", "verify", "snapshot", "runs", "evals"))
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--verify-files", action="store_true")
    args = parser.parse_args(argv)
    if args.command == "protocol-hash":
        print(protocol_sha256(args.repo_root / "experiments" / "17-treewm-grounded-repair-formal-v1"))
        return 0
    manifest = load_manifest(args.repo_root / "experiments" / "17-treewm-grounded-repair-formal-v1" / "manifest.json")
    if args.command == "snapshot":
        print(json.dumps(verify_source_snapshot(args.repo_root), sort_keys=True, indent=2))
    elif args.command == "verify":
        print(json.dumps(verify_all(manifest, repo_root=args.repo_root, verify_files=args.verify_files), sort_keys=True, indent=2))
    elif args.command == "runs":
        print(json.dumps([asdict(run) for run in expand_runs(manifest)], sort_keys=True, indent=2))
    else:
        print(json.dumps([asdict(eval_at(manifest, i)) for i in range(FINAL_EVAL_TASKS)], sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"grounded-formal campaign error: {exc}", file=sys.stderr)
        raise SystemExit(2)
