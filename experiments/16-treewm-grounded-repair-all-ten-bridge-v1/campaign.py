#!/usr/bin/env python3
"""Fail-closed contracts and deterministic launch mapping for all-ten bridge 16."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
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
RUNS = 40
SEEDS = (102, 103)
TASK_IDS = (1, 2, 3, 4, 5)
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
ARM_IDS = ("F", "H")
EXP15_SETTING_IDS = (
    "antmaze-large",
    "scene",
    "puzzle-3x3",
    "puzzle-4x4-100m",
    "cube-quadruple-100m",
)
EXP15_ARM_IDS = ("A", "B", "C", "D")
EXP15_SEEDS = (100, 101)
EXP15_AGGREGATE_GATES = (
    "integrity_40_of_40",
    "preregistered_candidate_setting_quorum",
    "candidate_not_all_zero_success",
    "candidate_positive_mean_progress",
    "candidate_positive_progress_run_quorum",
    "candidate_success_noninferior_to_control",
    "candidate_distance_reduction_noninferior_to_control",
    "adaptive_selection_disabled",
)
EXP15_INTEGRITY_GATES = (
    "exact_launch_completion_and_provenance",
    "identical_terminal_evaluation_bank",
    "fixed_common_validation_sample",
    "exact_1k_validation_axis",
    "finite_terminal_method_telemetry",
    "midpoint_and_terminal_rollouts",
)
EXP15_SCIENTIFIC_GATES = (
    "validation_regret_le_1p10",
    "self_fed_multistep_regret_le_1p10",
    "horizon_below_uniform_and_empirical_prior",
    "q_advantage",
    "gain_rank_pair_eligibility_and_coverage",
    "support_recall_and_precision",
    "aggregate_shared_module_gradients_and_clipping",
)
EXP15_GAIN_MEAN_TAGS = (
    "expansion/gain_rank_correlation",
    "expansion/gain_pairwise_accuracy",
    "expansion/gain_eligible_decision_fraction",
    "expansion/gain_ordered_pair_count",
    "expansion/gain_pair_coverage_fraction",
)
# These are the acceptance constants sealed by the pinned exp15 package protocol.
# Keeping them here makes prerequisite verification independent of exp15's published
# aggregate booleans and prevents a later, mutable report from redefining the gate.
EXP15_ACCEPTANCE_THRESHOLDS = {
    "candidate_settings_required": 4,
    "candidate_seeds_per_passing_setting": 2,
    "max_validation_regret_fraction": 0.1,
    "max_self_fed_multistep_validation_regret_fraction": 0.1,
    "horizon_uniform_cross_entropy": 1.6094379124341003,
    "min_gain_rank_correlation": 0.1,
    "min_gain_pairwise_accuracy": 0.52,
    "min_gain_eligible_decision_fraction": 0.2,
    "min_gain_ordered_pair_count": 1.0,
    "min_gain_pair_coverage_fraction": 0.01,
    "min_support_recall": 0.5,
    "min_support_precision": 0.25,
    "max_clip_fraction_below_threshold": 0.25,
    "min_candidate_total_successes": 1.0,
    "min_candidate_runs_with_positive_progress": 6,
    "min_candidate_mean_distance_reduction": 0.0,
    "min_paired_success_delta_vs_control": 0.0,
    "min_paired_distance_reduction_delta_vs_control": 0.0,
}
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
    "report.py",
    "submit.py",
    "bridge.slurm",
    "report.slurm",
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
    require(manifest.get("campaign_id") == "treewm-grounded-repair-all-ten-bridge-v1", "campaign ID drifted")
    require(manifest.get("classification") == "bounded_all_ten_bridge", "classification drifted")
    require("not formal validation" in str(manifest.get("claim_policy", "")), "claim guard missing")
    require(manifest.get("expected_runs") == RUNS, "run count drifted")

    prerequisite = manifest.get("prerequisite") or {}
    require(prerequisite.get("required") is True, "exp15 prerequisite is not mandatory")
    require(prerequisite.get("campaign_id") == "treewm-grounded-repair-pilot-v1", "exp15 prerequisite campaign drifted")
    require(Path(str(prerequisite.get("acceptance_path", ""))).is_absolute(), "exp15 acceptance path must be absolute")
    require(Path(str(prerequisite.get("launch_plan_path", ""))).is_absolute(), "exp15 launch-plan path must be absolute")
    require(prerequisite.get("required_status") == "accepted_for_fresh_formal_campaign_design", "exp15 acceptance status drifted")
    require(prerequisite.get("required_candidate_arm") == "C", "exp15 candidate prerequisite drifted")
    require(prerequisite.get("required_control_arm") == "A", "exp15 control prerequisite drifted")
    require(prerequisite.get("required_integrity_runs") == 40, "exp15 integrity prerequisite drifted")
    require(tuple(prerequisite.get("required_aggregate_gates") or ()) == EXP15_AGGREGATE_GATES, "exp15 aggregate-gate inventory drifted")
    require(prerequisite.get("package_protocol_sha256") == "ec41f19a97ab0c21d341b00baa69a6f50259408adda2d7bc6428ff46398a4f49", "exp15 protocol identity drifted")
    require(prerequisite.get("source_sha256") == "dc0b5d2c80a25c6ac51495696e83450859de4429fe9e40137572b1e981510d6a", "exp15 source identity drifted")
    require(prerequisite.get("runtime_sha256") == "77da91d49a1db99850fbf0632dc02ec58a3209f1a87949d6f5640ae6bf505c6b", "exp15 runtime identity drifted")

    method = manifest.get("method") or {}
    require(method == {
        "arm": "treewm",
        "model_class": "TreeWM",
        "experiment_config": "treewm_v2_grounded_repair_pilot",
        "objective_version": "treewm_v2_grounded_repair_pilot_v1",
        "node_budget": 64,
        "branch_factor": 4,
    }, "method contract drifted")

    inference = manifest.get("inference_choice") or {}
    require(inference.get("profiles") == INFERENCE_PROFILES, "inference profile definitions drifted")
    require(inference.get("profile") == "learned_guard_on", "bridge inference profile drifted")
    require("cannot change this bridge" in str(inference.get("decision_policy", "")), "prospective profile lock missing")

    design = manifest.get("design") or {}
    require(tuple(design.get("seeds") or ()) == SEEDS, "fresh seeds drifted")
    require(tuple(design.get("task_ids") or ()) == TASK_IDS, "task IDs drifted")
    require(design.get("preferred_arm") == "F", "preferred arm drifted")
    require(design.get("fallback_arm") == "H", "fallback arm drifted")
    require(design.get("selection_order") == ["F", "H"], "selection order drifted")
    require(design.get("scale_scope") == "one_global_arm_per_campaign_across_all_ten_settings", "global scale scope drifted")
    require(design.get("gradient_warning_settings") == ["cube-triple", "humanoidmaze-medium", "humanoidmaze-large"], "gradient warning scope drifted")
    require("selector weights identical" in str(design.get("gradient_warning_response", "")), "gradient warning response drifted")
    require("domain-specific tuning is forbidden" in str(design.get("gradient_warning_response", "")), "domain-specific tuning guard missing")
    require("F passes at least 18/20 scientific runs" in str(design.get("selection_policy", "")), "selection science quorum drifted")
    require("both seeds in at least 8/10 settings" in str(design.get("selection_policy", "")), "replicated-setting quorum drifted")
    require("at least one seed in every setting" in str(design.get("selection_policy", "")), "setting coverage quorum drifted")
    require("at least 12/20 positive-progress runs" in str(design.get("selection_policy", "")), "progress quorum drifted")
    require("paired H terminal success and distance reduction are each noninferior to F" in str(design.get("selection_policy", "")), "fallback noninferiority policy drifted")

    arms = manifest.get("arms") or []
    require(tuple(arm.get("id") for arm in arms) == ARM_IDS, "arm order drifted")
    expected_arms = {
        "F": (3e-5, "grounded_execution_v2", 1.0, (1.0, 1.0, 0.25), (0.25, 0.5, 0.25, 0.5)),
        "H": (3e-5, "grounded_execution_v2", 0.5, (1.0, 1.0, 0.25), (0.125, 0.25, 0.125, 0.25)),
    }
    for arm in arms:
        actual = (
            arm.get("world_lr"),
            arm.get("transition_mode"),
            arm.get("grounded_loss_scale"),
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
        require(actual == expected_arms[arm["id"]], f"arm {arm['id']} drifted")

    scientific = manifest.get("scientific_contract") or {}
    exact_scalar = {
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
    }
    for key, expected in exact_scalar.items():
        require(scientific.get(key) == expected, f"scientific contract {key} drifted")
    require(
        SHA256.fullmatch(str(scientific.get("evaluation_seed_protocol_sha256", ""))) is not None,
        "evaluation seed protocol is invalid",
    )
    require(scientific.get("future_config") == {
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
    }, "future recipe contract drifted")
    require(len(scientific) == len(exact_scalar) + 2, "unexpected scientific contract fields")

    legacy = manifest.get("compatible_v2_recipe_input") or {}
    require(legacy.get("read_only") is True, "recipe input is not read-only")
    require(legacy.get("campaign_id") == "treewm-50task-1m-v2", "recipe campaign drifted")
    require(legacy.get("objective_version") == "treewm_v2_rms_rank_v1", "recipe objective drifted")
    for key in ("campaign_protocol_sha256", "recipe_code_sha256", "recipe_runtime_sha256"):
        require(SHA256.fullmatch(str(legacy.get(key, ""))) is not None, f"bad {key}")

    settings = manifest.get("settings") or []
    require(tuple(setting.get("id") for setting in settings) == SETTING_IDS, "setting order drifted")
    require(len(settings) * len(arms) * len(SEEDS) == RUNS, "matrix is not exactly 40 runs")
    for setting in settings:
        setting_id = str(setting["id"])
        require(
            (setting.get("published_union_train_anchors"), setting.get("published_union_validation_anchors"))
            == EXPECTED_UNION_COUNTS[setting_id],
            f"{setting_id}: published-union counts drifted",
        )
        require(setting.get("dataset_kind") in {"standard", "sharded_100m_full"}, f"{setting_id}: dataset kind")
        require(setting.get("task_metric_dims") and len(set(setting["task_metric_dims"])) == len(setting["task_metric_dims"]), f"{setting_id}: task metric dims")
        require(setting.get("max_episode_steps") in {500, 750, 1000, 2000}, f"{setting_id}: episode cap")
        for key in ("input_contract_sha256", "calibration_sha256", "future_recipe_sha256"):
            require(SHA256.fullmatch(str(setting.get(key, ""))) is not None, f"{setting_id}: bad {key}")

    acceptance = manifest.get("acceptance") or {}
    require(acceptance.get("integrity_runs_required") == RUNS, "integrity quorum drifted")
    require(acceptance.get("runs_per_arm_required") == 20, "per-arm run quorum drifted")
    require(acceptance.get("settings_per_arm_required") == 10, "per-arm setting quorum drifted")
    require(acceptance.get("seeds_per_setting_required") == 2, "per-setting seed quorum drifted")
    require(acceptance.get("min_arm_scientific_runs_passing") == 18, "scientific run quorum drifted")
    require(acceptance.get("min_arm_settings_with_both_seeds_passing") == 8, "replicated setting quorum drifted")
    require(acceptance.get("min_arm_settings_with_at_least_one_seed_passing") == 10, "setting coverage quorum drifted")
    require(acceptance.get("preferred_arm") == "F", "report preferred arm drifted")
    require(acceptance.get("fallback_arm") == "H", "report fallback arm drifted")
    require(acceptance.get("checkpoint_steps") == list(range(1000, 25_001, 1000)), "checkpoint gate axis drifted")
    require(acceptance.get("max_validation_regret_fraction") == 0.1, "validation regret drifted")
    require(acceptance.get("max_self_fed_multistep_validation_regret_fraction") == 0.1, "self-fed regret drifted")
    require(acceptance.get("min_gain_rank_correlation") == 0.1, "gain rank threshold drifted")
    require(acceptance.get("min_gain_pairwise_accuracy") == 0.52, "gain pair threshold drifted")
    require(acceptance.get("min_gain_eligible_decision_fraction") == 0.2, "gain eligibility drifted")
    require(acceptance.get("min_support_recall") == 0.5, "support recall drifted")
    require(acceptance.get("min_support_precision") == 0.25, "support precision drifted")
    require(
        acceptance.get("gradient_gate_scope")
        == "aggregate_world_and_gain_module_norms_and_clip_coefficients",
        "aggregate shared-module gradient gate drifted",
    )
    require(acceptance.get("min_arm_total_successes") == 1, "all-zero outcome guard drifted")
    require(acceptance.get("min_arm_runs_with_positive_progress") == 12, "positive-progress guard drifted")
    require(acceptance.get("min_arm_mean_distance_reduction") == 0.0, "mean progress threshold drifted")
    require(acceptance.get("min_paired_fallback_success_delta_vs_full") == 0.0, "fallback success noninferiority drifted")
    require(acceptance.get("min_paired_fallback_distance_reduction_delta_vs_full") == 0.0, "fallback progress noninferiority drifted")
    require(acceptance.get("same_scientific_quorum_for_both_arms") is True, "arm-specific science quorum enabled")
    require(acceptance.get("selection_precedence") == ["F", "H"], "selection precedence drifted")
    require(acceptance.get("no_domain_specific_tuning") is True, "domain-specific tuning enabled")
    require(acceptance.get("no_posthoc_arm_selection") is True, "post-hoc selection enabled")

    snapshot = manifest.get("source_snapshot") or {}
    require(snapshot == {
        "required_for_submit": True,
        "marker": "SNAPSHOT.json",
        "status": "sealed_read_only",
        "repo_subdirectory": "repo",
        "writable_files_allowed": False,
    }, "source snapshot policy drifted")
    execution = manifest.get("execution") or {}
    require(execution.get("array") == "0-39%40", "Slurm array drifted")
    require(execution.get("gpus_per_task") == 1, "one GPU per run is required")
    require(execution.get("cpus_per_task") == 12, "CPU request drifted")
    require(execution.get("memory_per_task") == "64G", "memory request drifted")
    require(execution.get("walltime") == "04:00:00", "walltime drifted")
    require(execution.get("sbatch") == "/usr/local/bin/sbatch", "sbatch drifted")
    require(execution.get("srun") == "/cm/shared/apps/slurm/current/bin/srun", "srun drifted")
    require(execution.get("scontrol") == "/cm/shared/apps/slurm/current/bin/scontrol", "scontrol drifted")
    for key in ("python", "data_root", "raw_cache_root", "compatible_contract_root", "run_root"):
        require(Path((manifest.get("paths") or {}).get(key, "")).is_absolute(), f"{key} must be absolute")
    require(manifest["paths"]["python"] == PINNED_FORMAL_PYTHON, "Python interpreter drifted")
    require("grounded-repair-all-ten-bridge-v1" in manifest["paths"]["run_root"], "output namespace is not isolated")
    require(manifest.get("logging") == {
        "wandb_project": "treewm-grounded-repair-all-ten-bridge-v1",
        "wandb_group": "treewm-grounded-repair-all-ten-bridge-v1",
        "wandb_mode": "online",
    }, "W&B namespace drifted")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    validate_manifest(manifest)
    return stable_hash(manifest)


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _exp15_gate_map(
    record: Mapping[str, Any], field: str, expected: Sequence[str], label: str
) -> dict[str, bool]:
    value = record.get(field)
    require(isinstance(value, Mapping), f"{label} {field} is not an object")
    require(set(value) == set(expected), f"{label} {field} keys drifted")
    require(
        all(isinstance(item, bool) for item in value.values()),
        f"{label} {field} contains a non-boolean claim",
    )
    return {name: bool(value[name]) for name in expected}


def _validate_exp15_rollout(
    value: object, *, episodes: int, label: str
) -> dict[str, float]:
    require(isinstance(value, Mapping), f"{label} rollout evidence is not an object")
    fields = {
        name: value.get(name)
        for name in (
            "num_episodes",
            "successes",
            "success_rate",
            "distance_reduction_frac",
        )
    }
    require(
        all(_finite_number(item) for item in fields.values()),
        f"{label} rollout evidence is incomplete or non-finite",
    )
    num_episodes = float(fields["num_episodes"])
    successes = float(fields["successes"])
    success_rate = float(fields["success_rate"])
    require(
        num_episodes.is_integer() and int(num_episodes) == episodes,
        f"{label} rollout episode count differs",
    )
    require(
        successes.is_integer() and 0.0 <= successes <= float(episodes),
        f"{label} rollout success count is invalid",
    )
    require(
        0.0 <= success_rate <= 1.0
        and abs(success_rate - successes / float(episodes)) <= 1e-6,
        f"{label} rollout success rate is inconsistent",
    )
    return {name: float(item) for name, item in fields.items()}


def _validate_exp15_record_evidence(
    record: Mapping[str, Any], *, label: str
) -> tuple[bool, bool, dict[str, float]]:
    """Recompute every acceptance claim recoverable from one published run row."""
    integrity_gates = _exp15_gate_map(
        record, "integrity_gates", EXP15_INTEGRITY_GATES, label
    )
    scientific_gates = _exp15_gate_map(
        record, "scientific_gates", EXP15_SCIENTIFIC_GATES, label
    )
    integrity_pass = all(integrity_gates.values())
    scientific_pass = all(scientific_gates.values())
    require(
        record.get("integrity_pass") is integrity_pass,
        f"{label} integrity_pass differs from its gate evidence",
    )
    require(
        record.get("scientific_pass") is scientific_pass,
        f"{label} scientific_pass differs from its gate evidence",
    )

    metrics = record.get("metrics")
    require(isinstance(metrics, Mapping), f"{label} metrics evidence is not an object")
    midpoint = _validate_exp15_rollout(
        metrics.get("midpoint"), episodes=5, label=f"{label} midpoint"
    )
    final = _validate_exp15_rollout(
        metrics.get("final"), episodes=25, label=f"{label} final"
    )
    require(
        not integrity_gates["midpoint_and_terminal_rollouts"]
        or (midpoint["num_episodes"] == 5.0 and final["num_episodes"] == 25.0),
        f"{label} rollout integrity claim lacks its evidence",
    )

    thresholds = EXP15_ACCEPTANCE_THRESHOLDS
    validation_final = metrics.get("validation_final")
    validation_min = metrics.get("validation_min")
    validation_supported = bool(
        _finite_number(validation_final)
        and _finite_number(validation_min)
        and float(validation_final)
        <= float(validation_min)
        * (1.0 + float(thresholds["max_validation_regret_fraction"]))
    )
    require(
        not scientific_gates["validation_regret_le_1p10"]
        or validation_supported,
        f"{label} validation-regret claim contradicts its metrics",
    )
    self_fed_final = metrics.get("self_fed_final")
    self_fed_min = metrics.get("self_fed_min")
    self_fed_supported = bool(
        _finite_number(self_fed_final)
        and _finite_number(self_fed_min)
        and float(self_fed_final)
        <= float(self_fed_min)
        * (
            1.0
            + float(
                thresholds["max_self_fed_multistep_validation_regret_fraction"]
            )
        )
    )
    require(
        not scientific_gates["self_fed_multistep_regret_le_1p10"]
        or self_fed_supported,
        f"{label} self-fed-regret claim contradicts its metrics",
    )
    horizon_loss = metrics.get("horizon_loss")
    horizon_prior = metrics.get("horizon_empirical_prior")
    horizon_supported = bool(
        _finite_number(horizon_loss)
        and _finite_number(horizon_prior)
        and float(horizon_loss)
        < float(thresholds["horizon_uniform_cross_entropy"])
        and float(horizon_loss) < float(horizon_prior)
    )
    require(
        not scientific_gates["horizon_below_uniform_and_empirical_prior"]
        or horizon_supported,
        f"{label} horizon-loss claim contradicts its metrics",
    )

    gain = metrics.get("gain_recent_mean")
    require(isinstance(gain, Mapping), f"{label} gain evidence is not an object")
    require(set(gain) == set(EXP15_GAIN_MEAN_TAGS), f"{label} gain evidence keys drifted")
    gain_supported = bool(
        all(_finite_number(gain.get(tag)) for tag in EXP15_GAIN_MEAN_TAGS)
        and float(gain["expansion/gain_rank_correlation"])
        >= float(thresholds["min_gain_rank_correlation"])
        and float(gain["expansion/gain_pairwise_accuracy"])
        >= float(thresholds["min_gain_pairwise_accuracy"])
        and float(gain["expansion/gain_eligible_decision_fraction"])
        >= float(thresholds["min_gain_eligible_decision_fraction"])
        and float(gain["expansion/gain_ordered_pair_count"])
        >= float(thresholds["min_gain_ordered_pair_count"])
        and float(gain["expansion/gain_pair_coverage_fraction"])
        >= float(thresholds["min_gain_pair_coverage_fraction"])
    )
    require(
        not scientific_gates["gain_rank_pair_eligibility_and_coverage"]
        or gain_supported,
        f"{label} gain claim contradicts its metrics",
    )
    support_recall = metrics.get("support_recall")
    support_precision = metrics.get("support_precision")
    support_supported = bool(
        _finite_number(support_recall)
        and _finite_number(support_precision)
        and float(support_recall) >= float(thresholds["min_support_recall"])
        and float(support_precision) >= float(thresholds["min_support_precision"])
    )
    require(
        not scientific_gates["support_recall_and_precision"] or support_supported,
        f"{label} support claim contradicts its metrics",
    )
    clip_fraction = metrics.get("clip_fraction_below_threshold")
    clipping_supported = bool(
        _finite_number(clip_fraction)
        and 0.0 <= float(clip_fraction)
        <= float(thresholds["max_clip_fraction_below_threshold"])
    )
    require(
        not scientific_gates["aggregate_shared_module_gradients_and_clipping"]
        or clipping_supported,
        f"{label} gradient/clipping claim contradicts its metrics",
    )
    return integrity_pass, scientific_pass, final


def _verify_exp15_acceptance_evidence(
    acceptance: Mapping[str, Any], records: Sequence[Mapping[str, Any]]
) -> None:
    """Rebuild exp15's quorum and outcome decision from its exact 40 run rows."""
    expected_keys = {
        (setting, arm, seed)
        for setting in EXP15_SETTING_IDS
        for arm in EXP15_ARM_IDS
        for seed in EXP15_SEEDS
    }
    keyed: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    final_by_key: dict[tuple[str, str, int], dict[str, float]] = {}
    integrity_by_key: dict[tuple[str, str, int], bool] = {}
    scientific_by_key: dict[tuple[str, str, int], bool] = {}
    indices: set[int] = set()
    require(len(records) == 40, "exp15 acceptance does not contain exactly 40 run rows")
    for position, record in enumerate(records):
        require(isinstance(record, Mapping), f"exp15 run row {position} is not an object")
        seed = record.get("seed")
        index = record.get("index")
        require(
            isinstance(seed, int) and not isinstance(seed, bool),
            f"exp15 run row {position} seed is invalid",
        )
        require(
            isinstance(index, int) and not isinstance(index, bool),
            f"exp15 run row {position} index is invalid",
        )
        key = (str(record.get("setting_id")), str(record.get("arm_id")), seed)
        require(key in expected_keys, f"exp15 run row {position} key is unexpected")
        require(key not in keyed, f"exp15 run row {position} duplicates a matrix cell")
        require(index not in indices, f"exp15 run row {position} duplicates an index")
        keyed[key] = record
        indices.add(index)
        integrity, scientific, final = _validate_exp15_record_evidence(
            record, label=f"exp15 run {key}"
        )
        integrity_by_key[key] = integrity
        scientific_by_key[key] = scientific
        final_by_key[key] = final
    require(set(keyed) == expected_keys, "exp15 acceptance run matrix is not exact")
    require(indices == set(range(40)), "exp15 acceptance run indices are not exact")

    integrity_count = sum(integrity_by_key.values())
    candidate = "C"
    control = "A"
    setting_pass = {
        setting: sum(
            scientific_by_key[(setting, candidate, seed)] for seed in EXP15_SEEDS
        )
        >= int(EXP15_ACCEPTANCE_THRESHOLDS["candidate_seeds_per_passing_setting"])
        for setting in EXP15_SETTING_IDS
    }
    settings_passing = sum(setting_pass.values())
    paired: list[dict[str, Any]] = []
    for setting in EXP15_SETTING_IDS:
        for seed in EXP15_SEEDS:
            candidate_final = final_by_key[(setting, candidate, seed)]
            control_final = final_by_key[(setting, control, seed)]
            paired.append(
                {
                    "setting_id": setting,
                    "seed": seed,
                    "success_delta_candidate_minus_control": (
                        candidate_final["success_rate"]
                        - control_final["success_rate"]
                    ),
                    "distance_reduction_delta_candidate_minus_control": (
                        candidate_final["distance_reduction_frac"]
                        - control_final["distance_reduction_frac"]
                    ),
                }
            )
    success_delta = statistics.fmean(
        row["success_delta_candidate_minus_control"] for row in paired
    )
    progress_delta = statistics.fmean(
        row["distance_reduction_delta_candidate_minus_control"] for row in paired
    )
    candidate_finals = [
        final_by_key[(setting, candidate, seed)]
        for setting in EXP15_SETTING_IDS
        for seed in EXP15_SEEDS
    ]
    total_successes = sum(row["successes"] for row in candidate_finals)
    mean_progress = statistics.fmean(
        row["distance_reduction_frac"] for row in candidate_finals
    )
    positive_runs = sum(
        row["distance_reduction_frac"] > 0.0 for row in candidate_finals
    )
    thresholds = EXP15_ACCEPTANCE_THRESHOLDS
    aggregate_gates = {
        "integrity_40_of_40": integrity_count == 40,
        "preregistered_candidate_setting_quorum": settings_passing
        >= int(thresholds["candidate_settings_required"]),
        "candidate_not_all_zero_success": total_successes
        >= float(thresholds["min_candidate_total_successes"]),
        "candidate_positive_mean_progress": mean_progress
        > float(thresholds["min_candidate_mean_distance_reduction"]),
        "candidate_positive_progress_run_quorum": positive_runs
        >= int(thresholds["min_candidate_runs_with_positive_progress"]),
        "candidate_success_noninferior_to_control": success_delta
        >= float(thresholds["min_paired_success_delta_vs_control"]),
        "candidate_distance_reduction_noninferior_to_control": progress_delta
        >= float(thresholds["min_paired_distance_reduction_delta_vs_control"]),
        "adaptive_selection_disabled": True,
    }
    aggregate_metrics = {
        "candidate_total_successes": total_successes,
        "candidate_mean_distance_reduction": mean_progress,
        "candidate_runs_with_positive_progress": positive_runs,
        "paired_mean_success_delta_candidate_minus_control": success_delta,
        "paired_mean_distance_reduction_delta_candidate_minus_control": progress_delta,
    }
    reported_setting_pass = acceptance.get("candidate_setting_pass")
    require(
        isinstance(reported_setting_pass, Mapping)
        and set(reported_setting_pass) == set(EXP15_SETTING_IDS)
        and all(isinstance(value, bool) for value in reported_setting_pass.values()),
        "exp15 reported setting quorum has invalid keys or values",
    )
    reported_paired = acceptance.get("paired_comparisons")
    require(
        isinstance(reported_paired, list)
        and all(isinstance(row, Mapping) for row in reported_paired),
        "exp15 reported paired comparisons are malformed",
    )
    reported_metrics = acceptance.get("aggregate_metrics")
    require(
        isinstance(reported_metrics, Mapping)
        and set(reported_metrics) == set(aggregate_metrics)
        and all(_finite_number(value) for value in reported_metrics.values()),
        "exp15 reported aggregate metrics are malformed",
    )
    reported_gates = acceptance.get("aggregate_gates")
    require(
        isinstance(reported_gates, Mapping)
        and set(reported_gates) == set(EXP15_AGGREGATE_GATES)
        and all(isinstance(value, bool) for value in reported_gates.values()),
        "exp15 reported aggregate gates are malformed",
    )
    require(
        acceptance.get("integrity_runs_passing") == integrity_count,
        "exp15 reported integrity count differs from run evidence",
    )
    require(
        reported_setting_pass == setting_pass,
        "exp15 reported setting quorum differs from run evidence",
    )
    require(
        acceptance.get("candidate_settings_passing") == settings_passing,
        "exp15 reported setting-pass count differs from run evidence",
    )
    require(
        reported_paired == paired,
        "exp15 reported paired comparisons differ from run evidence",
    )
    require(
        reported_metrics == aggregate_metrics,
        "exp15 reported aggregate metrics differ from run evidence",
    )
    require(
        reported_gates == aggregate_gates,
        "exp15 reported aggregate gates differ from recomputed evidence",
    )
    require(all(aggregate_gates.values()), "exp15 recomputed aggregate gate failed")
    require(
        acceptance.get("accepted") is True
        and acceptance.get("status")
        == "accepted_for_fresh_formal_campaign_design",
        "exp15 acceptance decision differs from recomputed evidence",
    )


def load_exp15_prerequisite(
    manifest: Mapping[str, Any],
    *,
    acceptance_path: str | Path | None = None,
    launch_plan_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate and bind the exact accepted exp15 report and sealed launch plan."""
    validate_manifest(manifest)
    contract = manifest["prerequisite"]
    acceptance_file = Path(acceptance_path or contract["acceptance_path"])
    plan_file = Path(launch_plan_path or contract["launch_plan_path"])
    for path, label in (
        (acceptance_file, "exp15 acceptance"),
        (plan_file, "exp15 launch plan"),
    ):
        require(path.is_file() and not path.is_symlink(), f"{label} is missing or symlinked: {path}")

    acceptance = read_json(acceptance_file)
    require(acceptance.get("schema_version") == 1, "exp15 acceptance schema drifted")
    require(acceptance.get("campaign_id") == contract["campaign_id"], "exp15 acceptance campaign drifted")
    require(acceptance.get("status") == contract["required_status"], "exp15 prerequisite was not accepted")
    require(acceptance.get("accepted") is True, "exp15 accepted flag is not true")
    require(acceptance.get("formal_validation") is False, "exp15 acceptance incorrectly claims formal validation")
    require(acceptance.get("preregistered_candidate_arm") == contract["required_candidate_arm"], "exp15 candidate arm drifted")
    require(acceptance.get("matched_control_arm") == contract["required_control_arm"], "exp15 control arm drifted")
    require(acceptance.get("integrity_runs_passing") == contract["required_integrity_runs"], "exp15 lacks 40/40 integrity")
    records = acceptance.get("runs") or []
    expected_keys = {
        (setting, arm, seed)
        for setting in EXP15_SETTING_IDS
        for arm in EXP15_ARM_IDS
        for seed in EXP15_SEEDS
    }
    require(acceptance.get("missing_or_extra_keys") == [], "exp15 acceptance reports missing/extra cells")
    _verify_exp15_acceptance_evidence(acceptance, records)

    plan = read_json(plan_file)
    claimed_plan_hash = plan.get("plan_sha256")
    plan_body = dict(plan)
    plan_body.pop("plan_sha256", None)
    require(
        SHA256.fullmatch(str(claimed_plan_hash or "")) is not None
        and claimed_plan_hash == stable_hash(plan_body),
        "exp15 launch-plan self-hash differs",
    )
    require(plan.get("schema_version") == 1, "exp15 launch-plan schema drifted")
    require(plan.get("campaign_id") == contract["campaign_id"], "exp15 launch-plan campaign drifted")
    require(plan.get("status") == "sealed_bounded_repair_pilot_plan", "exp15 launch-plan status drifted")
    require(plan.get("formal_validation") is False, "exp15 launch plan incorrectly claims formal validation")
    plan_runs = plan.get("runs") or []
    plan_keys = {
        (str(record.get("setting_id")), str(record.get("arm_id")), int(record.get("seed", -1)))
        for record in plan_runs
    }
    require(len(plan_runs) == 40 and plan_keys == expected_keys, "exp15 launch-plan run matrix is not exact")
    common = plan.get("common_hashes") or {}
    for key in ("package_protocol_sha256", "source_sha256", "runtime_sha256"):
        require(common.get(key) == contract[key], f"exp15 launch-plan {key} identity drifted")
    require(
        acceptance.get("actual_evaluation_bank_sha256")
        == common.get("actual_evaluation_bank_sha256"),
        "exp15 acceptance/launch-plan evaluation bank differs",
    )

    binding: dict[str, Any] = {
        "schema_version": 1,
        "status": "accepted_exp15_prerequisite",
        "campaign_id": contract["campaign_id"],
        "accepted_status": acceptance["status"],
        "candidate_arm": acceptance["preregistered_candidate_arm"],
        "integrity_runs_passing": acceptance["integrity_runs_passing"],
        "acceptance_path": str(acceptance_file.resolve()),
        "acceptance_sha256": file_sha256(acceptance_file),
        "acceptance_canonical_sha256": stable_hash(acceptance),
        "launch_plan_path": str(plan_file.resolve()),
        "launch_plan_sha256": file_sha256(plan_file),
        "launch_plan_canonical_sha256": str(claimed_plan_hash),
        "package_protocol_sha256": common["package_protocol_sha256"],
        "source_sha256": common["source_sha256"],
        "runtime_sha256": common["runtime_sha256"],
        "actual_evaluation_bank_sha256": common["actual_evaluation_bank_sha256"],
    }
    binding["prerequisite_sha256"] = stable_hash(binding)
    return binding


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
                name = f"bridge-{setting['id']}-arm{arm['id'].lower()}-seed{seed}"
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
    require(len(result) == RUNS, "run expansion did not produce 40 runs")
    require(len({run.run_name for run in result}) == RUNS, "run names collide")
    require(len({run.wandb_id for run in result}) == RUNS, "W&B IDs collide")
    return result


def run_at(manifest: Mapping[str, Any], index: int) -> RunSpec:
    require(0 <= index < RUNS, "array index must be in [0,40)")
    return expand_runs(manifest)[index]


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
    protocol = verify_protocol_lock(root / "experiments" / "16-treewm-grounded-repair-all-ten-bridge-v1")
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
    manifest: Mapping[str, Any],
    run: RunSpec,
    contract: Mapping[str, Any],
    prerequisite: Mapping[str, Any] | None = None,
) -> list[str]:
    bound_prerequisite = prerequisite or load_exp15_prerequisite(manifest)
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
        _override("train.world_grad_clip", 1.0),
        _override("train.gain_grad_clip", 1.0),
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
        _override("eval.task_split", scientific["task_split"]),
        _override("eval.episodes_per_task", scientific["periodic_episodes_per_task"]),
        _override("eval.final_episodes_per_task", scientific["final_episodes_per_task"]),
        _override("eval.seed", scientific["evaluation_seed"]),
        _override("+campaign_input_contract_sha256", contract["contract_sha256"]),
        _override("+campaign_calibration_sha256", contract["calibration_sha256"]),
        _override("+campaign_future_recipe_sha256", contract["future_recipe_sha256"]),
        _override("+campaign_compatible_recipe_code_sha256", manifest["compatible_v2_recipe_input"]["recipe_code_sha256"]),
        _override("+campaign_factorial_arm", run.arm_id),
        _override(
            "+campaign_prerequisite_binding_sha256",
            bound_prerequisite["prerequisite_sha256"],
        ),
    ]


def trainer_command(
    manifest: Mapping[str, Any],
    run: RunSpec,
    *,
    repo_root: str | Path = REPOSITORY_ROOT,
    verify_recipe_files: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    package = root / "experiments" / "16-treewm-grounded-repair-all-ten-bridge-v1"
    protocol = verify_protocol_lock(package)
    prerequisite = load_exp15_prerequisite(manifest)
    contract = load_compatible_input(manifest, run, verify_files=verify_recipe_files)
    source = source_contract(root)
    overrides = scientific_overrides(manifest, run, contract, prerequisite)
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
        "exp15_prerequisite_sha256": prerequisite["prerequisite_sha256"],
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
        "TREEWM_PREREQUISITE_BINDING_SHA256": prerequisite["prerequisite_sha256"],
        "TREEWM_EXP15_PREREQUISITE_SHA256": prerequisite["prerequisite_sha256"],
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
        "exp15_prerequisite": prerequisite,
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
            "exp15_prerequisite_sha256": prerequisite["prerequisite_sha256"],
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
    protocol = verify_protocol_lock(root / "experiments" / "16-treewm-grounded-repair-all-ten-bridge-v1")
    prerequisite = load_exp15_prerequisite(manifest)
    recipe_audits = {
        setting["id"]: load_compatible_input(manifest, setting, verify_files=verify_files)["recipe_coverage_audit"]
        for setting in manifest["settings"]
    }
    runs = expand_runs(manifest)
    source = source_contract(root)
    configs = {
        run.run_name: stable_hash({
            "schema_version": 1,
            "overrides": scientific_overrides(
                manifest,
                run,
                load_compatible_input(manifest, run),
                prerequisite,
            ),
        })
        for run in runs
    }
    return {
        "schema_version": 1,
        "status": "verified_bounded_all_ten_bridge",
        "formal_validation": False,
        "campaign_id": manifest["campaign_id"],
        "runs": len(runs),
        "manifest_sha256": manifest_sha256(manifest),
        "package_protocol_sha256": protocol,
        "source_sha256": source["source_sha256"],
        "runtime_sha256": source["runtime_sha256"],
        "recipe_files_verified": bool(verify_files),
        "recipe_coverage_audits": recipe_audits,
        "actual_evaluation_bank": actual_evaluation_bank(manifest),
        "exp15_prerequisite": prerequisite,
        "unique_config_sha256": sorted(set(configs.values())),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("protocol-hash", "verify", "snapshot", "runs", "command"))
    parser.add_argument("--repo-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--verify-files", action="store_true")
    parser.add_argument("--index", type=int)
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.repo_root.resolve()
    if args.command == "protocol-hash":
        print(protocol_sha256(root / "experiments" / "16-treewm-grounded-repair-all-ten-bridge-v1"))
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
        value = trainer_command(manifest, run_at(manifest, args.index), repo_root=root)
        if args.output:
            atomic_json(args.output, value)
    print(json.dumps(value, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"all-ten bridge campaign error: {exc}", file=sys.stderr)
        raise SystemExit(2)
