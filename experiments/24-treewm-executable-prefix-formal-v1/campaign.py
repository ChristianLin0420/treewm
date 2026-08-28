#!/usr/bin/env python3
"""Outcome-blind design contract for the fresh Exp24 executable-prefix formal.

This module is deliberately stdlib-only and read-only.  It freezes the scientific
matrix and the required execution architecture while the upstream Exp23 Launch7
pilot is still running.  It does not create output directories, snapshots, locks,
or scheduler jobs, and it cannot authorize a launch in its current package phase.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True

PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]
MANIFEST_PATH = PACKAGE_DIR / "manifest.json"
UPSTREAM_BINDING_PATH = PACKAGE_DIR / "launch7_acceptance.binding.json"

CAMPAIGN_ID = "treewm-executable-prefix-formal-v1-launch1"
UPSTREAM_CAMPAIGN_ID = "treewm-executable-prefix-repair-pilot-v1-launch7"
PACKAGE_PHASE = "m1_hardened_runtime_scaffold_execution_blocked"
SELECTED_ARM = "GSEP"
SEEDS = (240, 241, 242, 243)
TASK_IDS = (1, 2, 3, 4, 5)
STAGE_TARGETS = (2_000, 25_000, 100_000, 1_000_000)
PREFIX_WEIGHTS = {
    "action": 0.033368419,
    "latent": 0.027645085,
    "endpoint": 0.011350645,
}
SETTING_SPECS = (
    ("scene", "scene_play"),
    ("puzzle-3x3", "puzzle_3x3_play"),
    ("puzzle-4x4-100m", "puzzle_4x4_play_100m"),
    ("cube-double", "cube_double_play"),
    ("cube-triple", "cube_triple_play"),
    ("cube-quadruple-100m", "cube_quadruple_play_100m"),
    ("antmaze-large", "antmaze_large_navigate"),
    ("antmaze-giant", "antmaze_giant_navigate"),
    ("humanoidmaze-medium", "humanoidmaze_medium_navigate"),
    ("humanoidmaze-large", "humanoidmaze_large_navigate"),
)
PILOT_COVERED_SETTINGS = frozenset(
    {
        "antmaze-large",
        "scene",
        "puzzle-3x3",
        "puzzle-4x4-100m",
        "cube-quadruple-100m",
    }
)
ADDITIONAL_FORMAL_SETTINGS = frozenset(setting for setting, _ in SETTING_SPECS) - PILOT_COVERED_SETTINGS
TRAINING_RUNS = len(SETTING_SPECS) * len(SEEDS)
FINAL_EVAL_CELLS = TRAINING_RUNS * len(TASK_IDS)


class ContractError(RuntimeError):
    """The immutable design, upstream evidence, or launch authority is invalid."""


@dataclass(frozen=True)
class RunSpec:
    index: int
    setting_index: int
    seed_index: int
    setting_id: str
    env_config: str
    arm: str
    seed: int
    run_name: str
    wandb_id: str


@dataclass(frozen=True)
class EvalSpec:
    index: int
    training_index: int
    task_id: int
    run: RunSpec


@dataclass(frozen=True)
class DagNode:
    name: str
    kind: str
    elements: int
    dependency: str | None
    dependency_type: str | None


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


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


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise ContractError(f"non-finite JSON value: {token}")


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=_reject_nonfinite,
            )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read exact JSON object {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _file_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def authenticated_regular_json(path: str | Path) -> tuple[dict[str, Any], str]:
    """Read a stable nonsymlink regular file without following the final entry."""

    source = Path(path).absolute()
    require(
        source.is_absolute()
        and all(part not in ("", ".", "..") for part in source.parts[1:]),
        f"artifact path is not absolute/normalized: {source}",
    )
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    parent_descriptor = os.open(source.anchor, directory_flags)
    try:
        for component in source.parts[1:-1]:
            child = os.open(component, directory_flags, dir_fd=parent_descriptor)
            os.close(parent_descriptor)
            parent_descriptor = child
    except BaseException:
        os.close(parent_descriptor)
        raise
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        listed = os.stat(source.name, dir_fd=parent_descriptor, follow_symlinks=False)
        descriptor = os.open(source.name, flags, dir_fd=parent_descriptor)
    except OSError as exc:
        os.close(parent_descriptor)
        raise ContractError(f"cannot authenticate {source}: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode) and _file_identity(before) == _file_identity(listed),
            f"artifact is not a stable regular file: {source}",
        )
        require(before.st_nlink == 1, f"artifact has an unsafe hard-link count: {source}")
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
        listed_after = os.stat(source.name, dir_fd=parent_descriptor, follow_symlinks=False)
        require(
            _file_identity(before) == _file_identity(after) == _file_identity(listed_after),
            f"artifact changed while reading: {source}",
        )
    finally:
        os.close(descriptor)
        os.close(parent_descriptor)
    try:
        value = json.loads(
            b"".join(chunks).decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot decode authenticated JSON {source}: {exc}") from exc
    require(isinstance(value, dict), f"authenticated JSON root is not an object: {source}")
    return value, digest.hexdigest()


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    manifest, _digest = authenticated_regular_json(path)
    validate_manifest(manifest)
    return manifest


def _exact_set(value: object, expected: set[str], label: str) -> None:
    require(isinstance(value, list), f"{label} is not a list")
    require(len(value) == len(set(value)), f"{label} contains duplicates")
    require(set(value) == expected, f"{label} differs")


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Validate every scientific and architectural decision frozen in phase one."""

    require(manifest.get("schema_version") == 1, "manifest schema differs")
    require(manifest.get("campaign_id") == CAMPAIGN_ID, "campaign ID differs")
    require(
        manifest.get("submission_authority_policy")
        == "Submission authority requires authenticated Exp23 Launch7 acceptance, all-ten outcome-blind audits and per-setting contracts, a sealed hardened runtime and scientific protocol, and verified training/evaluation/requeue feasibility.",
        "submission authority policy differs",
    )
    require(
        manifest.get("scientific_claim_policy")
        == "No formal scientific conclusion or promotion claim exists until all four staged all-40 gates pass, all exact 200 held-out cells complete both paired rails, the seed-level aggregate completes, and the immutable formal report commits.",
        "scientific claim policy differs",
    )
    require("claim_policy" not in manifest, "legacy circular claim policy remains")
    require(manifest.get("package_phase") == PACKAGE_PHASE, "package phase differs")
    require(
        manifest.get("classification") == "fresh_formal_selected_gsep_only",
        "campaign classification differs",
    )
    require(manifest.get("scientific_protocol_sealed") is False, "incomplete scientific protocol is called sealed")
    require(
        manifest.get("per_setting_contract_state") == "unbound_pending_all_ten_audits",
        "per-setting contract state differs",
    )

    design = manifest.get("design") or {}
    require(design.get("selected_arm") == SELECTED_ARM, "formal arm is not selected GSEP")
    require(design.get("candidate_or_control_arms") == [], "formal package contains an adaptive arm")
    require(design.get("seeds") == list(SEEDS), "formal training seeds differ")
    require(design.get("task_ids") == list(TASK_IDS), "held-out task IDs differ")
    require(design.get("expected_training_runs") == TRAINING_RUNS, "training matrix size differs")
    require(design.get("expected_final_eval_cells") == FINAL_EVAL_CELLS, "evaluation matrix size differs")
    require(design.get("scratch_only") is True, "formal training is not scratch-only")
    require(design.get("reuse_upstream_checkpoints") is False, "upstream checkpoints became reusable")
    require(design.get("reuse_upstream_outputs") is False, "upstream outputs became reusable")
    require(design.get("reuse_upstream_wandb") is False, "upstream W&B state became reusable")
    require(
        design.get("seed_bank_provenance") == "new_exp24_only_preregistered_training_bank_240_243",
        "formal seed-bank provenance differs",
    )
    census = design.get("seed_assignment_census") or {}
    require(census.get("git_head") == "bdd5d819d291104d5aa7cbe8934ba935cb76518c", "seed census Git anchor differs")
    require(census.get("prior_training_assignment_matches") == 0, "seed census found a prior training assignment")
    require(census.get("performed_before_exp24_assignment") is True, "seed census timing differs")
    require(
        not set(SEEDS).intersection(set(range(4)) | set(range(100, 112)) | set(range(220, 224))),
        "Exp24 seeds overlap an earlier training or reserved bank",
    )
    settings = design.get("settings") or []
    require(
        [(row.get("id"), row.get("env_config")) for row in settings] == list(SETTING_SPECS),
        "formal setting order or environment map differs",
    )
    require(
        all(row.get("contract_state") == "unbound_pending_all_ten_audits" for row in settings),
        "a per-setting contract is prematurely bound",
    )
    require(
        all(row.get("formal_contract_sha256") is None for row in settings),
        "a per-setting contract hash is present before the all-ten binder",
    )

    objective = manifest.get("objective") or {}
    require(
        objective.get("experiment_config") == "treewm_v2_grounded_executable_prefix_formal_v1",
        "formal experiment config identity differs",
    )
    require(
        objective.get("objective_version") == "treewm_v2_grounded_executable_prefix_formal_v1",
        "formal objective identity differs",
    )
    require(objective.get("executable_prefix_enabled") is True, "executable-prefix objective disabled")
    require(objective.get("executable_prefix_steps") == 4, "executable-prefix horizon differs")
    require(objective.get("weights") == PREFIX_WEIGHTS, "selected executable-prefix weights differ")
    require(objective.get("weight_selection_frozen_before_formal_outcomes") is True, "weights became outcome-adaptive")
    require(objective.get("scheduler_total_steps") == 1_000_000, "scheduler horizon differs")
    require(objective.get("optimizer_updates") == 1_000_000, "optimizer horizon differs")
    require(objective.get("formal_cadence") == {
        "checkpoint_every": 1_000,
        "validation_every": 2_000,
        "diagnostics_every": 2_000,
        "periodic_evaluation_every": 25_000,
        "training_log_every": 50,
        "visualization_every": 25_000,
        "visualization_every_early": 2_000,
        "visualization_early_until": 25_000,
        "periodic_episodes_per_task": 1,
        "final_episodes_per_task": 50,
        "evaluation_seed": 2718,
    }, "formal cadence differs")
    require(objective.get("frozen_base_recipe") == {
        "world_lr": 3e-5,
        "weight_decay": 1e-3,
        "gradient_checkpointing": True,
        "loader_workers": 10,
        "loader_thread_limit": 1,
        "validation_sample_seed": 1701,
        "gain_lr": 3e-4,
        "gain_weight_decay": 0.0,
        "gain_loss_every": 1,
        "gain_training_scorers": ["learned", "novelty_q"],
        "dropout": 0.1,
        "model_max_depth": 3,
        "tree_max_depth": 3,
        "branch_factor": 4,
        "node_budget": 64,
        "keep_threshold": 0.5,
        "tree_scorer": "learned",
        "keep_balance": True,
        "multistep_weight": 1.0,
        "scheduled_sampling_p": 0.25,
        "scheduled_sampling_warmup": 5_000,
        "scheduled_sampling_granularity": "sequence",
        "multistep_transition_mode": "grounded_execution_v2",
        "grounded_select_action_weight": 1.0,
        "grounded_select_endpoint_weight": 1.0,
        "grounded_select_horizon_weight": 0.25,
        "grounded_loss_latent_weight": 0.25,
        "grounded_loss_action_weight": 0.5,
        "grounded_loss_horizon_weight": 0.25,
        "grounded_loss_endpoint_weight": 0.5,
        "grounded_detach_self_fed_parent": True,
        "multistep_depth_weights": [1.0, 1.0, 1.0],
        "latent_gauge_weight": 1.0,
        "latent_gauge_enabled": True,
        "latent_gauge_epsilon": 1e-8,
        "latent_gauge_min_reference_scale": 1e-4,
        "separate_gain_grad_clip": True,
        "separate_branch_transformer_grad_clip": True,
        "world_grad_clip": 1.0,
        "gain_grad_clip": 1.0,
        "branch_transformer_grad_clip": 1.0,
        "planner_decoded_metric": "domain_raw",
        "planner_execute_mode": "clipped",
        "planner_execute_steps": 4,
        "planner_score_mode": "endpoint",
        "planner_score_space": "decoded",
        "planner_require_first_edge_improvement": True,
        "planner_min_first_edge_improvement": 0.0,
        "action_lower_bound": -1.0,
        "action_upper_bound": 1.0,
        "task_split": "standard",
        "recipe_anchor_policy": "published_union",
        "future_num_neighbors": 24,
        "future_query_multiplier": 6,
        "future_time_exclusion": 50,
        "future_include_self": True,
        "future_metric_mode": "rms_v2",
        "future_horizons": [4, 8, 16, 32, 64],
        "future_h_max": 64,
        "future_horizon_rule": "displacement",
        "future_fixed_horizon": 32,
        "future_cluster_method": "average",
        "future_max_modes": 4,
        "future_multi_step_depth": 3,
        "future_retrieval_pool": 50_000,
        "shared_cache": True,
        "future_cache": False,
        "executable_prefix_length_rule": "per_matched_branch_p_equals_min_4_and_logged_selected_continuation_horizon_learned_horizon_never_used",
    }, "formal base recipe differs")

    lifecycle = manifest.get("lifecycle") or {}
    require(lifecycle.get("stage_targets") == list(STAGE_TARGETS), "formal stages differ")
    require(lifecycle.get("all_40_required_at_every_gate") is True, "a stage gate permits partial coverage")
    require(lifecycle.get("dependency_type") == "afterok", "stage dependency is not afterok")
    require(lifecycle.get("post_100k_policy") == "integrity_and_numerical_health_only", "late outcomes can select/stop recipe")
    require(lifecycle.get("resume_scope") == "same_exp24_run_only", "resume scope is not same-run only")
    require(lifecycle.get("cross_stage_lineage") == {
        "stage_2000_source": "generation_zero_scratch_initialization_for_all_40_cells",
        "later_stage_source": "exact_same_cell_checkpoint_accepted_by_immediately_prior_all_40_gate",
        "promotions": [
            {"from": 2_000, "to": 25_000, "required_gate": "gate_2000"},
            {"from": 25_000, "to": 100_000, "required_gate": "gate_25000"},
            {"from": 100_000, "to": 1_000_000, "required_gate": "gate_100000"},
        ],
        "invariant_identity_fields": [
            "campaign_id",
            "setting_id",
            "env_config",
            "arm",
            "seed",
            "run_name",
            "wandb_id",
            "resolved_config_sha256",
            "source_sha256",
            "runtime_sha256",
            "protocol_sha256",
            "formal_contract_sha256",
        ],
        "within_stage_requeue": "same_slurm_array_element_same_run_exact_durable_checkpoint_without_gate_promotion",
        "cross_stage_promotion": "new_afterok_stage_job_authorized_only_by_immediately_prior_all_40_gate_receipt",
        "exp23_checkpoint_or_output_path_allowed": False,
    }, "cross-stage lineage contract differs")

    stages = manifest.get("stage_acceptance") or {}
    require(set(stages) == {
        "common_all_stages",
        "common_numerical_health",
        "stage_2000",
        "stage_25000",
        "stage_100000",
        "stage_1000000",
    }, "stage acceptance schema differs")
    common = stages.get("common_all_stages") or {}
    require(set(common) == {
        "exact_40_run_coverage",
        "exact_checkpoint_update",
        "finite_required_telemetry",
        "strict_scalar_identity_no_conflicts",
        "source_protocol_config_checkpoint_identity",
        "no_missing_or_unexpected_artifacts",
    }, "common stage gate schema differs")
    require(common.get("exact_40_run_coverage") is True, "stage gates permit incomplete run coverage")
    require(common.get("exact_checkpoint_update") is True, "stage gates permit an inexact checkpoint update")
    require(common.get("finite_required_telemetry") is True, "stage gates permit non-finite telemetry")
    require(common.get("strict_scalar_identity_no_conflicts") is True, "stage gates permit conflicting scalar duplicates")
    require(common.get("source_protocol_config_checkpoint_identity") is True, "stage gates do not bind full identity")
    require(common.get("no_missing_or_unexpected_artifacts") is True, "stage gates permit artifact coverage drift")
    require(stages.get("common_numerical_health") == {
        "applies_at_stage_targets": list(STAGE_TARGETS),
        "training_axis_cadence_updates": 50,
        "recent_gradient_window_updates": 5_000,
        "early_gradient_window_policy": "last_min_stage_target_and_5000_positive_updates_on_exact_50_update_axis_ending_at_target",
        "gradient_norm_tags": [
            "train/grad_norm_world",
            "train/grad_norm_gain",
            "train/grad_norm_world_rest",
            "train/grad_norm_branch_transformer",
        ],
        "gradient_norm_operator": "every_value_strictly_greater_than_1e-8",
        "gradient_clip_tags": [
            "train/grad_clip_coefficient_world",
            "train/grad_clip_coefficient_gain",
            "train/grad_clip_coefficient_world_rest",
            "train/grad_clip_coefficient_branch_transformer",
        ],
        "gradient_clip_domain": "every_value_strictly_greater_than_0_and_less_than_or_equal_to_1",
        "candidate_clip_tags": [
            "train/grad_clip_coefficient_branch_transformer",
            "train/grad_clip_coefficient_world_rest",
            "train/grad_clip_coefficient_gain",
        ],
        "candidate_clip_low_threshold": 0.05,
        "candidate_clip_rule": "for_each_tag_fraction_of_recent_values_strictly_below_0.05_less_than_or_equal_to_0.25",
        "recent_gauge_window_updates": 1_000,
        "early_gauge_window_policy": "last_min_stage_target_and_1000_positive_updates_on_exact_50_update_axis_ending_at_target",
        "gauge_min_ratio_operator": "every_recent_and_exact_target_value_greater_than_or_equal_to_0.8",
        "gauge_ratio_consistency_tolerance": 1e-5,
        "gauge_reference_update_exact": 0,
        "gauge_reference_sealed_exact": 1.0,
        "gauge_reference_minimum": 1e-4,
        "validation_axis_cadence_updates": 2_000,
        "validation_axis_policy": "complete_positive_multiples_of_2000_not_exceeding_target_latest_equals_target_minus_target_mod_2000",
        "fixed_validation_sample_count_exact": 5_120,
        "validation_losses_finite_nonnegative": ["val/loss_total", "val/loss_multistep_self_fed"],
        "validation_current_over_axis_minimum_max_ratio": 1.1,
        "prefix_scopes": ["train", "val"],
        "prefix_step_policy": "train_equals_stage_target_validation_equals_target_minus_target_mod_2000",
        "prefix_terms": ["executable_prefix_action", "executable_prefix_latent", "executable_prefix_endpoint"],
        "prefix_term_rule": "alias_equals_raw_finite_nonnegative_weight_equals_frozen_schedule_equals_1_effective_equals_raw_times_weight",
        "prefix_action_finite_fractions_exact": 1.0,
        "prefix_count_denominators_positive_and_exact": True,
        "prefix_fixed_validation_target_contract_exact": True,
        "prefix_gradient_observation_policy": "component_gradients_are_prelaunch_fixed_weight_safety_bound_runtime_uses_exact_world_gain_world_rest_branch_transformer_group_norm_axes_no_invented_component_tag",
    }, "cross-stage numerical health contract differs")
    require(stages.get("stage_2000") == {
        "decision_basis": "integrity_numerical_prefix_structure_only",
        "all_40_common_gates_required": True,
        "all_prefix_count_denominators_complete": True,
        "raw_applied_logged_action_finite_fraction": 1.0,
        "common_gradient_and_gauge_windows_use_early_window_policy": True,
        "prefix_component_gradients_prelaunch_safety_bound": True,
        "outcome_gate": False,
    }, "2k acceptance differs")
    stage_25k = stages.get("stage_25000") or {}
    require(set(stage_25k) == {
        "decision_basis",
        "all_40_common_gates_required",
        "minimum_complete_science_passes_per_setting",
        "training_seed_denominator_per_setting",
        "outcome_gate",
        "fixed_thresholds",
        "operator_semantics",
    }, "25k gate schema differs")
    require(stage_25k.get("decision_basis") == "preregistered_scientific_health", "25k decision basis differs")
    require(stage_25k.get("all_40_common_gates_required") is True, "25k permits incomplete common coverage")
    require(stage_25k.get("minimum_complete_science_passes_per_setting") == 3, "25k per-setting quorum differs")
    require(stage_25k.get("training_seed_denominator_per_setting") == 4, "25k quorum denominator differs")
    require(stage_25k.get("outcome_gate") is False, "25k outcome can stop/select formal recipe")
    thresholds = stage_25k.get("fixed_thresholds") or {}
    require(set(thresholds) == {
        "min_recent_latent_gauge_ratio",
        "reference_update",
        "reference_sealed",
        "min_gradient_norm",
        "gradient_recent_window_updates",
        "gauge_recent_window_updates",
        "gain_recent_window_updates",
        "min_clip_coefficient",
        "max_recent_5k_fraction_below_min_clip",
        "max_validation_regret_fraction",
        "max_self_fed_validation_regret_fraction",
        "max_horizon_cross_entropy",
        "horizon_cross_entropy_below_empirical_prior_entropy",
        "horizon_label_fraction_sum_tolerance",
        "min_q_advantage",
        "min_gain_rank_correlation",
        "min_gain_pairwise_accuracy",
        "min_gain_eligible_decision_fraction",
        "min_gain_ordered_pair_count",
        "min_gain_pair_coverage_fraction",
        "min_support_recall",
        "min_support_precision",
        "max_action_clipped_fraction",
        "action_raw_over_applied_rms_interval",
        "predicted_over_actual_task_displacement_rms_interval",
        "max_endpoint_error_over_actual_task_displacement",
    }, "25k threshold schema differs")
    require(thresholds.get("min_recent_latent_gauge_ratio") == 0.8, "25k gauge threshold differs")
    require(thresholds.get("reference_update") == 0, "25k gauge reference update differs")
    require(thresholds.get("reference_sealed") == 1.0, "25k gauge seal differs")
    require(thresholds.get("min_gradient_norm") == 1e-8, "25k gradient threshold differs")
    require(thresholds.get("gradient_recent_window_updates") == 5_000, "25k gradient window differs")
    require(thresholds.get("gauge_recent_window_updates") == 1_000, "25k gauge window differs")
    require(thresholds.get("gain_recent_window_updates") == 5_000, "25k gain window differs")
    require(thresholds.get("min_clip_coefficient") == 0.05, "25k clipping threshold differs")
    require(thresholds.get("max_recent_5k_fraction_below_min_clip") == 0.25, "25k clipping fraction differs")
    require(thresholds.get("max_validation_regret_fraction") == 0.1, "25k validation regret differs")
    require(thresholds.get("max_self_fed_validation_regret_fraction") == 0.1, "25k self-fed regret differs")
    require(thresholds.get("max_horizon_cross_entropy") == 1.6094379124341003, "25k horizon threshold differs")
    require(thresholds.get("horizon_cross_entropy_below_empirical_prior_entropy") is True, "25k empirical-prior horizon gate differs")
    require(thresholds.get("horizon_label_fraction_sum_tolerance") == 0.02, "25k horizon fraction tolerance differs")
    require(thresholds.get("min_q_advantage") == 0.0, "25k q-advantage threshold differs")
    require(thresholds.get("min_gain_rank_correlation") == 0.1, "25k gain-rank threshold differs")
    require(thresholds.get("min_gain_pairwise_accuracy") == 0.52, "25k gain-pair threshold differs")
    require(thresholds.get("min_gain_eligible_decision_fraction") == 0.2, "25k gain eligibility differs")
    require(thresholds.get("min_gain_ordered_pair_count") == 1.0, "25k ordered-pair threshold differs")
    require(thresholds.get("min_gain_pair_coverage_fraction") == 0.01, "25k pair coverage differs")
    require(thresholds.get("min_support_recall") == 0.5, "25k support-recall threshold differs")
    require(thresholds.get("min_support_precision") == 0.25, "25k support-precision threshold differs")
    require(thresholds.get("max_action_clipped_fraction") == 0.25, "25k prefix clipping threshold differs")
    require(thresholds.get("action_raw_over_applied_rms_interval") == [1.0, 2.0], "25k action ratio differs")
    require(thresholds.get("predicted_over_actual_task_displacement_rms_interval") == [0.5, 2.0], "25k displacement ratio differs")
    require(thresholds.get("max_endpoint_error_over_actual_task_displacement") == 1.0, "25k endpoint threshold differs")
    require(stage_25k.get("operator_semantics") == {
        "finite_and_exact_count_requirements": "must_equal",
        "minimum_floors": "greater_than_or_equal",
        "maximum_ceilings": "less_than_or_equal",
        "ratio_intervals": "closed_inclusive",
        "q_advantage": "strictly_greater_than_zero",
        "horizon_cross_entropy": "strictly_less_than_both_ln5_and_empirical_prior_entropy",
        "gradient_norms": "strictly_greater_than_minimum",
    }, "25k threshold operators differ")
    require(stages.get("stage_100000") == {
        "decision_basis": "integrity_numerical_health_plus_fixed_outcome_sanity",
        "all_40_common_gates_required": True,
        "monitor_episodes_per_run": 5,
        "min_fleet_monitor_successes": 1,
        "fleet_mean_distance_reduction_strictly_greater_than": 0.0,
        "min_runs_with_positive_progress": 1,
        "success_and_run_count_operator": "greater_than_or_equal",
        "monitor_seed_table_protocol_bound_and_heldout_disjoint": True,
        "recipe_selection_allowed": False,
    }, "100k acceptance differs")
    require(stages.get("stage_1000000") == {
        "decision_basis": "exact_completion_integrity_and_numerical_health_only",
        "all_40_common_gates_required": True,
        "exact_optimizer_update": 1000000,
        "outcome_gate": False,
        "recipe_selection_allowed": False,
    }, "1M acceptance differs")

    final = manifest.get("final_evaluation") or {}
    require(final.get("models") == TRAINING_RUNS and final.get("tasks_per_model") == len(TASK_IDS), "final matrix differs")
    require(final.get("cells") == FINAL_EVAL_CELLS, "final cell count differs")
    require(final.get("rails") == ["learned", "bfs"], "paired rails differ")
    require(final.get("episodes_per_cell_per_rail") == 50, "paired episode count differs")
    require(final.get("same_ordered_episode_seeds_across_rails") is True, "rails are not episode-paired")
    require(final.get("adaptive_selection") is False, "held-out outcomes can select the method")
    require(final.get("array") == "0-199%40", "held-out array identity differs")
    require(final.get("heldout_seed_table_common_across_four_model_seeds") is True, "held-out table is not common across seeds")
    require(final.get("heldout_monitor_disjointness_required") is True, "monitor/held-out disjointness is not required")
    require(final.get("pooled_episode_summaries_descriptive_only") is True, "pooled episodes became the inference unit")
    require(final.get("rail_parity_contract") == {
        "same_exact_1m_checkpoint": True,
        "same_resolved_config_except_scorer": True,
        "same_tasks_and_ordered_episode_seeds": True,
        "same_node_budget": 64,
        "same_action_bounds": [-1.0, 1.0],
        "same_per_setting_max_environment_steps": True,
        "only_allowed_difference": "tree_config.scorer=learned|bfs",
        "rail_specific_tuning": False,
        "order_dependent_rng": False,
        "independent_planner_rng_derived_from_same_episode_identity": True,
    }, "learned/BFS rail parity differs")
    seed_policy = final.get("seed_policy") or {}
    require(seed_policy.get("table_status") == "unsealed_blocker", "held-out seed table is prematurely sealed")
    require(seed_policy.get("common_across_model_seeds_and_rails") is True, "held-out bank is not paired/common")
    require(seed_policy.get("unique_across_settings_and_tasks") is True, "held-out bank permits seed reuse")
    require(seed_policy.get("disjoint_from_all_formal_monitor_seeds") is True, "held-out bank can overlap formal monitor seeds")
    require(seed_policy.get("disjoint_from_launch7_evaluation_seeds") is True, "held-out bank can overlap Launch7")
    require(seed_policy.get("disjoint_from_authenticated_prior_consumed_evaluation_seed_inventory") is True, "held-out bank can overlap consumed prior evaluation")
    promotion = final.get("promotion_criterion") or {}
    require(promotion == {
        "learned_total_successes_greater_than_or_equal": 1,
        "paired_seed_delta_ci_lower_greater_than_or_equal": 0.0,
        "settings_learned_noninferior_greater_than_or_equal": 7,
        "all_200_cells_and_raw_episode_pairs_required": True,
        "decision": "promotion_eligible_only_if_all_gates_pass_else_not_eligible",
    }, "terminal promotion criterion differs")
    inference = final.get("primary_inference") or {}
    require(inference.get("unit") == "training_seed", "primary inference unit differs")
    require(inference.get("replicates") == 4, "primary replicate count differs")
    require(inference.get("cells_per_seed") == 50, "primary seed cell denominator differs")
    require(inference.get("episodes_per_seed_per_rail") == 2_500, "primary seed episode denominator differs")
    require(inference.get("paired_quantity") == "learned_minus_bfs_macro_success_rate", "paired estimand differs")
    require(inference.get("two_sided_confidence_level") == 0.95, "confidence level differs")
    require(inference.get("degrees_of_freedom") == 3, "paired t degrees of freedom differs")
    require(inference.get("t_critical") == 3.182446, "paired t critical value differs")
    require(inference.get("seed_replicate_aggregation") == "unweighted_mean_of_exactly_50_cell_success_rates", "seed replicate aggregation differs")
    require(inference.get("missingness_or_cell_selection_allowed") is False, "primary inference permits missingness adjustment")
    require(inference.get("raw_episode_success_accounting_must_match_cell_summary") is True, "raw success accounting is not bound")
    require(inference.get("rail_episode_seed_order_must_match_exactly") is True, "paired rail order is not exact")
    require(final.get("setting_noninferiority") == {
        "denominator_settings": 10,
        "required_noninferior_settings": 7,
        "per_setting_cells_per_rail": 20,
        "per_setting_episodes_per_rail": 1000,
        "estimand": "pooled_success_rate_over_4_seeds_x_5_tasks_x_50_episodes_per_rail",
        "equivalent_equal_cell_macro": True,
        "comparison": "learned_minus_bfs_greater_than_or_equal_to_zero",
    }, "setting noninferiority estimand differs")

    safety = manifest.get("fixed_weight_safety_contract") or {}
    require(safety == {
        "purpose": "verify_frozen_tuple_not_derive_or_retune",
        "settings": [setting for setting, _env in SETTING_SPECS],
        "regimes": ["exp20_gs_exact_5000", "scratch_initialization"],
        "checkpoint_seeds": [108, 109],
        "scratch_seeds": [230, 231],
        "audit_step": 5000,
        "fixed_batches_per_setting_regime": 2,
        "batch_size": 16,
        "expected_rows": 80,
        "batch_selection": "counter_hash_stratified_published_union_train",
        "device": "cpu_fp32",
        "determinism": {
            "torch_deterministic_algorithms": True,
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "components": ["executable_prefix_action", "executable_prefix_latent", "executable_prefix_endpoint"],
        "groups": ["branch_transformer", "world_rest"],
        "required_gradient_state": "every_base_component_group_finite_and_strictly_positive",
        "per_component_median_base_gradient_fraction_max": 0.03,
        "per_component_cap_scope": "each_component_x_regime_x_group_median_over_all_setting_seed_batch_rows",
        "aggregate_every_row_group_base_gradient_fraction_max": 0.10,
        "aggregate_cap_scope": "sum_of_three_weighted_component_norms_divided_by_base_norm_for_every_row_x_group",
        "frozen_weights": PREFIX_WEIGHTS,
        "optimizer_steps": 0,
        "outcomes_or_rollouts_read": False,
        "retuning_allowed": False,
        "failure_decision": "block_formal_and_require_new_engineering_design",
    }, "fixed-weight safety contract differs")

    dependency = manifest.get("launch_dependency") or {}
    require(dependency.get("campaign_id") == UPSTREAM_CAMPAIGN_ID, "upstream Launch7 identity differs")
    require(dependency.get("required_status") == "accepted_engineering_pilot", "upstream acceptance status differs")
    require(dependency.get("binding_file") == UPSTREAM_BINDING_PATH.name, "upstream binding filename differs")
    require(dependency.get("binding_state") == "unbound", "in-flight package claims an upstream binding")
    require(dependency.get("formal_submission_allowed") is False, "design phase permits formal submission")
    _exact_set(
        dependency.get("pilot_covered_settings"),
        set(PILOT_COVERED_SETTINGS),
        "pilot-covered setting set",
    )
    _exact_set(
        dependency.get("additional_formal_settings_requiring_outcome_blind_audits"),
        set(ADDITIONAL_FORMAL_SETTINGS),
        "additional formal setting set",
    )
    require(
        dependency.get("required_all_ten_audits")
        == [
            "input_contract",
            "future_recipe",
            "prefix_target",
            "resolved_config",
            "causal_parity",
            "fixed_weight_safety",
        ],
        "all-ten audit requirements differ",
    )
    require(dependency.get("per_setting_contract_required_fields") == [
        "env_config",
        "env_name",
        "source_name",
        "dataset_kind",
        "data_subdir",
        "expected_shards_or_null",
        "max_episode_steps",
        "task_metric_dims",
        "relative_endpoints",
        "action_dim_and_bounds",
        "published_union_train_population",
        "published_union_validation_population",
        "source_manifest_sha256",
        "cache_manifest_sha256",
        "input_contract_sha256",
        "calibration_sha256",
        "future_recipe_sha256",
        "compatible_recipe_code_sha256",
        "evaluation_seed_protocol_sha256",
        "prefix_target_contract_sha256",
        "resolved_config_sha256",
        "causal_parity_pair_sha256",
        "fixed_weight_safety_rows_sha256",
        "input_contract_audit_root_sha256",
        "future_recipe_audit_root_sha256",
        "prefix_target_audit_root_sha256",
        "resolved_config_audit_root_sha256",
        "causal_parity_audit_root_sha256",
        "fixed_weight_safety_audit_root_sha256",
        "formal_contract_sha256",
    ], "per-setting binding field requirements differ")
    require(
        dependency.get("per_setting_contract_canonical_body_policy")
        == "formal_contract_sha256_hashes_the_exact_no_extra_field_body_and_cross_binds_each_per_setting_row_to_all_six_authenticated_all_ten_audit_roots",
        "per-setting canonical cross-binding policy differs",
    )
    require(
        dependency.get("binding_policy")
        == "Only the authenticated immutable Launch7 report quartet may be read from the Launch7 output tree as prerequisite evidence; bind REPORT_COMMIT plus accepted decision, bundle, provenance, protocol, source, and trainer identities; never read any Launch7 checkpoint, optimizer, run telemetry, W&B, evaluation progress, mutable report, or other output state.",
        "Launch7 evidence/read boundary differs",
    )

    hardening = manifest.get("required_exp23_hardening_port") or {}
    require(hardening.get("source_package") == "experiments/23-treewm-executable-prefix-repair-pilot-v1", "hardening source differs")
    require(hardening.get("port_only_after_upstream_freeze") is True, "runtime can be copied from an in-flight source")
    required_patterns = {
        "immutable_nonsymlink_source_snapshot",
        "isolated_python_I_S_B_bootstrap",
        "exact_snapshot_inventory_and_double_identity_check",
        "exclusive_submission_claim_and_append_only_journals",
        "scheduler_control_plane_observation_binding",
        "exact_id_submit_reconciliation_rollback_and_cancel",
        "whole_array_afterok_dependency_reconciliation",
        "kill_invalid_dependency_validation",
        "same_run_signal_requeue_with_durable_replay_intent",
        "strict_append_only_scalar_identity_and_conflict_rejection",
        "identical_duplicate_inventory_or_upstream_suppression",
        "dense_50_update_gain_and_support_axis",
        "authenticated_wandb_leaf_symlink_policy",
        "immutable_report_triplet_and_commit",
        "test_only_and_snapshot_test_zero_persistent_writes",
        "formal_objective_registered_in_v2_gauge_prefix_formal_staged_authorization_sets",
        "hardened_launch7_report_quartet_binder",
        "execution_ready_manifest_exact_schema_no_unvalidated_fields",
    }
    _exact_set(hardening.get("required_patterns"), required_patterns, "required hardening pattern set")
    require(hardening.get("runtime_files_present") is True, "M1 hardened runtime files are absent")
    require(hardening.get("runtime_execution_ready") is False, "M1 runtime is prematurely execution-ready")
    require(
        hardening.get("runtime_scope")
        == "m1_hardened_control_plane_scaffold_with_execution_adapters_fail_closed",
        "M1 runtime scope differs",
    )
    require(hardening.get("protocol_lock_present") is False, "design package falsely claims a protocol lock")

    execution = manifest.get("execution") or {}
    require(execution.get("training_array") == "0-39%40", "training array differs")
    require(execution.get("heldout_array") == "0-199%40", "held-out array differs")
    require(execution.get("gpus_per_task") == 1 and execution.get("cpus_per_task") == 12, "task resources differ")
    require(execution.get("memory_per_task") == "64G", "task memory differs")
    require(execution.get("walltime") == "04:00:00", "task walltime differs")
    require(execution.get("signal_seconds_before_end") == 420, "preemption signal window differs")
    require(execution.get("queued_receipt_barrier_timeout_seconds") == 900, "queued receipt barrier timeout differs")
    require(execution.get("queued_receipt_barrier_poll_seconds") == 0.25, "queued receipt barrier polling differs")
    require(execution.get("scheduler_client_timeout_seconds") == 30, "scheduler client timeout differs")
    require(execution.get("pre_receipt_transaction_budget_seconds") == 600, "pre-receipt transaction budget differs")
    require(execution.get("receipt_barrier_safety_margin_seconds") == 300, "receipt barrier margin differs")
    require(
        execution["pre_receipt_transaction_budget_seconds"]
        + execution["receipt_barrier_safety_margin_seconds"]
        == execution["queued_receipt_barrier_timeout_seconds"]
        and execution["receipt_barrier_safety_margin_seconds"]
        >= 2 * execution["scheduler_client_timeout_seconds"],
        "queued receipt barrier has insufficient scheduler timeout margin",
    )
    require(execution.get("requeue_required") is True, "formal jobs are not requeueable")
    require(execution.get("feasibility_status") == "unverified_blocker", "resource feasibility is prematurely accepted")
    require(execution.get("requires_measured_updates_per_hour_for_all_ten_settings") is True, "formal feasibility lacks throughput evidence")
    require(execution.get("requires_requeue_budget_and_cluster_limit_check") is True, "formal feasibility lacks requeue-limit evidence")
    require(execution.get("requires_measured_heldout_runtime_for_every_setting_task_and_rail") is True, "held-out runtime feasibility is absent")
    require(execution.get("requires_episode_level_requeue_and_paired_order_probe") is True, "held-out requeue integrity is unproven")
    require(execution.get("requires_final_array_aggregate_report_walltime_check") is True, "terminal DAG feasibility is unproven")
    require(execution.get("requires_clean_committed_source_snapshot") is True, "clean committed source proof is not required")
    require(
        execution.get("requires_held_root_post_receipt_activation_handshake") is True,
        "held-root post-receipt activation handshake is not required",
    )
    require(
        execution.get("requires_pinned_interpreter_environment_provenance") is True,
        "pinned interpreter/environment provenance is not required",
    )
    require(execution.get("control_python_flags") == ["-I", "-S", "-B"], "control Python isolation differs")
    require(execution.get("trainer_python_flags") == ["-P", "-S", "-B"], "trainer Python isolation differs")
    require(execution.get("process_topology") == "Slurm batch shell -> worker.py -> train_entry.py -> in-process scripts.train; no srun", "process topology differs")
    require(execution.get("sbatch") == "/usr/local/bin/sbatch", "sbatch path differs")
    require(execution.get("squeue") == "/usr/local/bin/squeue", "squeue path differs")
    require(execution.get("scancel") == "/usr/local/bin/scancel", "scancel path differs")
    require(execution.get("scontrol") == "/usr/local/bin/scontrol", "scontrol path differs")
    require(execution.get("scheduler_control_plane") == {
        "slurm_conf": "/cm/shared/apps/slurm/var/etc/cs-oci-ord/slurm.conf",
        "cluster_name": "cs-oci-ord",
        "slurmctld_hosts": ["cs-oci-ord-a", "cs-oci-ord-b"],
        "slurmctld_port": 6817,
        "auth_type": "auth/munge",
        "gres_types": ["gpu"],
        "cli_filter_plugins": ["lua"],
        "job_submit_plugins": ["lua"],
        "trust_model": (
            "root-admin mutable scheduler control plane; config and Lua policy bytes are "
            "observation-bound from preclaim through submission; root-owned Slurm clients, "
            "plugin binaries, and shared libraries are trusted mutable external runtime"
        ),
    }, "scheduler control-plane contract differs")

    paths = manifest.get("paths") or {}
    require(
        paths.get("python")
        == "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/envs/treewm-formal-py311/bin/python",
        "formal Python path differs",
    )
    require(str(paths.get("run_root", "")).endswith("/outputs/treewm-executable-prefix-formal-v1-launch1"), "formal run namespace differs")
    require(str(paths.get("eval_root", "")).endswith("/outputs/treewm-executable-prefix-formal-v1-launch1-heldout"), "held-out namespace differs")
    upstream_token = "treewm-executable-prefix-repair-pilot-v1-launch7"
    require(upstream_token not in str(paths.get("run_root", "")), "formal run root reuses upstream namespace")
    require(upstream_token not in str(paths.get("eval_root", "")), "formal eval root reuses upstream namespace")

    milestones = manifest.get("milestones") or []
    require([row.get("id") for row in milestones] == [
        "m0_geometry_and_global_recipe",
        "m1_launch7_binding",
        "m2_all_ten_audits",
        "m3_runtime_port",
        "m4_protocol_and_preflight",
        "m5_submit",
        "m6_training_and_gates",
        "m7_heldout_and_report",
    ], "milestone sequence differs")
    require(milestones[0].get("status") == "complete", "design milestone is not complete")
    require(all(row.get("status") == "blocked" for row in milestones[1:]), "a downstream milestone is prematurely open")


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    validate_manifest(manifest)
    return stable_hash(manifest)


def expand_runs(manifest: Mapping[str, Any]) -> list[RunSpec]:
    validate_manifest(manifest)
    runs: list[RunSpec] = []
    for setting_index, setting in enumerate(manifest["design"]["settings"]):
        for seed_index, seed in enumerate(SEEDS):
            index = setting_index * len(SEEDS) + seed_index
            run_name = f"exp24-{setting['id']}-armgsep-seed{seed}"
            wandb_id = stable_hash(
                {"campaign_id": CAMPAIGN_ID, "setting_id": setting["id"], "arm": SELECTED_ARM, "seed": seed}
            )[:32]
            runs.append(
                RunSpec(
                    index=index,
                    setting_index=setting_index,
                    seed_index=seed_index,
                    setting_id=setting["id"],
                    env_config=setting["env_config"],
                    arm=SELECTED_ARM,
                    seed=seed,
                    run_name=run_name,
                    wandb_id=wandb_id,
                )
            )
    require(len(runs) == TRAINING_RUNS, "training expansion is not 40 runs")
    require(len({row.run_name for row in runs}) == TRAINING_RUNS, "training names collide")
    require(len({row.wandb_id for row in runs}) == TRAINING_RUNS, "W&B IDs collide")
    return runs


def run_at(manifest: Mapping[str, Any], index: int) -> RunSpec:
    require(isinstance(index, int) and not isinstance(index, bool), "training index is not an integer")
    require(0 <= index < TRAINING_RUNS, "training index must be in [0,40)")
    return expand_runs(manifest)[index]


def expand_final_evaluations(manifest: Mapping[str, Any]) -> list[EvalSpec]:
    runs = expand_runs(manifest)
    rows = [
        EvalSpec(
            index=run.index * len(TASK_IDS) + task_offset,
            training_index=run.index,
            task_id=task_id,
            run=run,
        )
        for run in runs
        for task_offset, task_id in enumerate(TASK_IDS)
    ]
    require(len(rows) == FINAL_EVAL_CELLS, "final evaluation expansion is not 200 cells")
    return rows


def eval_at(manifest: Mapping[str, Any], index: int) -> EvalSpec:
    require(isinstance(index, int) and not isinstance(index, bool), "evaluation index is not an integer")
    require(0 <= index < FINAL_EVAL_CELLS, "evaluation index must be in [0,200)")
    return expand_final_evaluations(manifest)[index]


def scheduler_dag(manifest: Mapping[str, Any]) -> list[DagNode]:
    validate_manifest(manifest)
    nodes: list[DagNode] = []
    prior_gate: str | None = None
    for target in STAGE_TARGETS:
        train = f"train_{target}"
        gate = f"gate_{target}"
        nodes.append(DagNode(train, "array", TRAINING_RUNS, prior_gate, "afterok" if prior_gate else None))
        nodes.append(DagNode(gate, "all_40_gate", 1, train, "afterok"))
        prior_gate = gate
    nodes.extend(
        [
            DagNode("heldout_eval", "array", FINAL_EVAL_CELLS, prior_gate, "afterok"),
            DagNode("aggregate", "seed_level_paired_t", 1, "heldout_eval", "afterok"),
            DagNode("formal_report", "immutable_report", 1, "aggregate", "afterok"),
        ]
    )
    return nodes


def paired_t_summary(values: Sequence[float]) -> dict[str, float | int]:
    """The preregistered df=3 summary used only after four seed-level deltas exist."""

    require(len(values) == len(SEEDS), "paired inference requires exactly four seed values")
    floats = [float(value) for value in values]
    require(all(math.isfinite(value) for value in floats), "paired inference contains a non-finite value")
    mean = sum(floats) / len(floats)
    variance = sum((value - mean) ** 2 for value in floats) / (len(floats) - 1)
    standard_error = math.sqrt(variance / len(floats))
    margin = 3.182446 * standard_error
    return {
        "n": len(floats),
        "degrees_of_freedom": len(floats) - 1,
        "mean": mean,
        "standard_error": standard_error,
        "ci95_lower": mean - margin,
        "ci95_upper": mean + margin,
    }


def upstream_binding_status(
    manifest: Mapping[str, Any], path: str | Path = UPSTREAM_BINDING_PATH
) -> dict[str, Any]:
    """Inspect the placeholder or future binder output without following symlinks."""

    validate_manifest(manifest)
    value, file_digest = authenticated_regular_json(path)
    if value.get("status") != "sealed_accepted_engineering_pilot":
        return {
            "accepted": False,
            "status": value.get("status", "invalid"),
            "reason": "formal execution blocked until Launch7 is accepted and byte-bound",
            "binding_file_sha256": file_digest,
        }
    # An accepted-looking object is intentionally *not* interpreted in phase zero.
    # The execution revision must replace this branch with the hardened binder that
    # authenticates every path component, exact report quartet, ownership/modes,
    # commit-to-bundle/decision/provenance cross-hashes, campaign/status semantics,
    # and absence of cancellation.  A self-hash plus arbitrary absolute path is not
    # launch authority.
    return {
        "accepted": False,
        "status": "accepted_looking_binding_unusable_in_design_phase",
        "reason": "formal execution blocked: hardened Launch7 report binder is not implemented",
        "binding_file_sha256": file_digest,
    }


def assert_launch_authorized(
    manifest: Mapping[str, Any], path: str | Path = UPSTREAM_BINDING_PATH
) -> None:
    """No Exp24 launch is possible until a later execution-ready package revision."""

    status = upstream_binding_status(manifest, path)
    require(status["accepted"], status["reason"])
    require(
        manifest["package_phase"] == "execution_ready" and manifest["launch_dependency"]["binding_state"] == "sealed",
        "formal execution remains blocked: Exp24 runtime port, all-ten audits, and protocol seal are incomplete",
    )
    require(manifest["launch_dependency"]["formal_submission_allowed"] is True, "manifest forbids formal submission")


def preflight_report(manifest: Mapping[str, Any]) -> dict[str, Any]:
    dependency = upstream_binding_status(manifest)
    runs = expand_runs(manifest)
    evaluations = expand_final_evaluations(manifest)
    dag = scheduler_dag(manifest)
    blockers = [
        "Launch7 accepted_engineering_pilot binding is absent",
        "outcome-blind input/future/prefix/resolved/causal/fixed-weight-safety audits are not sealed for all ten settings",
        "M1 control-plane runtime is present but scientific worker/gate/eval/report adapters remain fail-closed",
        "new 1M formal objective/config and trainer authorization sets are not yet registered",
        "formal evaluation seed table and protocol lock are not yet sealed",
        "held-out seeds lack authenticated prior-consumption/disjointness evidence",
        "all-ten training plus every held-out setting/task/rail runtime, walltime, resume, and requeue-budget feasibility is not yet verified",
        "independent prelaunch audit, commit, push, and clean-tree verification remain pending",
    ]
    return {
        "schema_version": 1,
        "status": PACKAGE_PHASE,
        "campaign_id": CAMPAIGN_ID,
        "submitted": False,
        "persistent_writes_performed": False,
        "scheduler_calls": [],
        "snapshot_created": False,
        "upstream": dependency,
        "training_runs": len(runs),
        "final_eval_cells": len(evaluations),
        "episodes_per_cell_per_rail": 50,
        "total_heldout_episodes_per_rail": len(evaluations) * 50,
        "dag": [asdict(node) for node in dag],
        "blockers": blockers,
        "manifest_sha256": manifest_sha256(manifest),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs="?", choices=("validate", "runs", "evals", "dag", "preflight"), default="validate")
    args = parser.parse_args(argv)
    manifest = load_manifest()
    if args.command == "validate":
        value: object = {"status": "valid_geometry_and_global_recipe_locked", "manifest_sha256": manifest_sha256(manifest)}
    elif args.command == "runs":
        value = [asdict(row) for row in expand_runs(manifest)]
    elif args.command == "evals":
        value = [asdict(row) for row in expand_final_evaluations(manifest)]
    elif args.command == "dag":
        value = [asdict(row) for row in scheduler_dag(manifest)]
    else:
        value = preflight_report(manifest)
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
