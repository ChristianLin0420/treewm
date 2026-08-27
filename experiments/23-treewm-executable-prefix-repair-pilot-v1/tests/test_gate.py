from __future__ import annotations

import copy
import hashlib
import importlib.util
from pathlib import Path
import statistics

import pytest


PACKAGE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("exp23_gate", PACKAGE / "gate.py")
assert SPEC is not None and SPEC.loader is not None
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _put(scalars: dict[str, list[list[float]]], tag: str, steps, value: float) -> None:
    scalars[tag] = [[int(step), float(value)] for step in steps]


def _replace_at(boundary: dict, tag: str, step: int, value: float) -> None:
    points = boundary["scalars"][tag]
    for point in points:
        if point[0] == step:
            point[1] = float(value)
            return
    raise AssertionError(f"missing {tag}@{step}")


def _target(setting: str) -> dict:
    return gate.load_prefix_target_lock()["settings"][setting]


def _prefix_contract(setting: str) -> dict:
    target = _target(setting)
    lock = gate.load_prefix_target_lock()
    return {
        "setting_id": setting,
        "target_contract_sha256": target["target_contract_sha256"],
        "prefix_target_artifact_sha256": lock["artifact_sha256"],
        "validation_manifest_sha256": target["validation_manifest_sha256"],
        "fixed_validation_sampler": copy.deepcopy(target["fixed_validation_sampler"]),
    }


def _scalars(target: int, arm: str, setting: str) -> dict[str, list[list[float]]]:
    values: dict[str, list[list[float]]] = {}
    gradient_axis = gate._axis(target, 50, 5_000)
    gauge_axis = gate._axis(target, 50, 1_000)
    validation_axis = gate._axis(target, 1_000)
    gain_axis = gate._axis(target, 1_000, 5_000)

    for tag in gate.GRADIENT_NORM_TAGS:
        _put(values, tag, gradient_axis, 0.5)
    for tag in gate.GRADIENT_CLIP_TAGS:
        _put(values, tag, gradient_axis, 0.5)

    _put(values, "latent_gauge/min_ratio", gauge_axis, 0.9)
    gauge_terminal = {
        "latent_gauge/root/scale": 0.9,
        "latent_gauge/root/reference": 1.0,
        "latent_gauge/root/ratio": 0.9,
        "latent_gauge/future/scale": 0.95,
        "latent_gauge/future/reference": 1.0,
        "latent_gauge/future/ratio": 0.95,
        "latent_gauge/loss": 0.01,
        "latent_gauge/reference_sealed": 1.0,
        "latent_gauge/reference_update": 0.0,
    }
    for tag, value in gauge_terminal.items():
        _put(values, tag, [target], value)

    _put(values, "val/loss_total", validation_axis, 1.0)
    _put(values, "val/loss_multistep_self_fed", validation_axis, 1.0)
    _put(values, "data/validation_fixed_sample_count", validation_axis, 5120.0)
    _put(values, "val/loss_horizon", [target], 1.0)
    _put(values, "control/retrieval_uses_task_metric_endpoint", [target], 1.0)
    _put(values, "control/q_advantage_over_z", [target], 0.2)
    _put(values, "control/q_advantage_over_random_proj", [target], 0.2)
    gain_values = {
        "expansion/gain_rank_correlation": 0.2,
        "expansion/gain_pairwise_accuracy": 0.6,
        "expansion/gain_eligible_decision_fraction": 0.3,
        "expansion/gain_ordered_pair_count": 2.0,
        "expansion/gain_pair_coverage_fraction": 0.1,
    }
    for tag, value in gain_values.items():
        _put(values, tag, gain_axis, value)
    _put(values, "tree/support_recall", [target], 0.7)
    _put(values, "tree/support_precision", [target], 0.4)
    for horizon in gate.HORIZONS:
        _put(values, f"data/validation_horizon_label_fraction_h{horizon}", [target], 0.2)

    endpoint_error = 0.6 if arm == "GS" else 0.4
    raw_action = 1.2 if arm == "GS" else 1.1
    clip_fraction = 0.20 if arm == "GS" else 0.10
    target_contract = _target(setting)
    prefix_values = {
        "schema_version": 1.0,
        "loss_action_normalized": 0.1,
        "loss_latent": 0.1,
        "loss_endpoint_normalized_task": 0.1,
        "action_raw_env_abs_mean": 0.8,
        "action_raw_env_rms": raw_action,
        "action_applied_env_abs_mean": 0.7,
        "action_applied_env_rms": 1.0,
        "action_logged_env_abs_mean": 0.6,
        "action_logged_env_rms": 0.9,
        "action_clipped_fraction": clip_fraction,
        "action_finite_fraction": 1.0,
        "action_applied_finite_fraction": 1.0,
        "action_logged_finite_fraction": 1.0,
        "predicted_vs_actual_normalized_task_rms": endpoint_error,
        "predicted_normalized_task_displacement_rms": 1.0,
        "actual_normalized_task_displacement_rms": 1.0,
        "predicted_vs_actual_guard_metric_error": endpoint_error,
        "predicted_guard_metric_displacement": 1.0,
        "actual_guard_metric_displacement": 1.0,
        "prefix_steps_mean": 4.0,
        "valid_anchor_fraction": 1.0,
        "matched_branches_per_anchor": target_contract["matched_branch_count"] / 5120.0,
        "action_scalars_per_anchor": target_contract["prefix_action_scalar_count"] / 5120.0,
        "action_raw_finite_scalars_per_anchor": target_contract["prefix_action_scalar_count"] / 5120.0,
        "action_applied_finite_scalars_per_anchor": target_contract["prefix_action_scalar_count"] / 5120.0,
        "action_logged_finite_scalars_per_anchor": target_contract["prefix_action_scalar_count"] / 5120.0,
        "goal_metric_onehot": 1.0,
        "latent_target_scale": 1.0,
        "predicted_vs_actual_hamming": 1.0,
        "predicted_vs_actual_hamming_fraction": 0.2,
        "predicted_hamming_displacement": 2.0,
        "actual_hamming_displacement": 2.0,
    }
    for metric_prefix in (gate.TRAIN_PREFIX, gate.PREFIX):
        for suffix, value in prefix_values.items():
            _put(values, metric_prefix + suffix, [target], value)

    weights = (
        {
            "executable_prefix_action": 0.0,
            "executable_prefix_latent": 0.0,
            "executable_prefix_endpoint": 0.0,
        }
        if arm == "GS"
        else {
            "executable_prefix_action": 0.033368419,
            "executable_prefix_latent": 0.027645085,
            "executable_prefix_endpoint": 0.011350645,
        }
    )
    raw_terms = {
        "executable_prefix_action": 0.1,
        "executable_prefix_latent": 0.1,
        "executable_prefix_endpoint": 0.1,
    }
    for scope in ("train", "val"):
        for term, raw in raw_terms.items():
            _put(values, f"{scope}/loss_{term}", [target], raw)
            _put(values, f"{scope}/loss_raw/{term}", [target], raw)
            _put(values, f"{scope}/loss_effective/{term}", [target], raw * weights[term])
            _put(values, f"{scope}/loss_weight/{term}", [target], weights[term])
            _put(values, f"{scope}/loss_schedule/{term}", [target], 1.0)
    return values


def _boundary(setting: str, arm: str, target: int) -> dict:
    result = {
        "update": target,
        "scalars": _scalars(target, arm, setting),
        "prefix_contract": _prefix_contract(setting),
    }
    if target == 25_000:
        rows = []
        reduction = 0.1 if arm == "GS" else 0.2
        for task_index, task_id in enumerate(range(1, 6)):
            for episode_index in range(5):
                rows.append({
                    "steps": 10,
                    "replans": 2,
                    "nodes": 8,
                    "best_goal_distance": 0.7,
                    "chunk_lengths": [4, 4, 2],
                    "selected_depths": [1, 2, 1],
                    "displacement": 0.5,
                    "path_length": 1.2,
                    "action_magnitude": 0.4,
                    "no_action_plans": 0,
                    "guard_plans": 2,
                    "guard_rejections": 1,
                    "guard_candidate_count": 8,
                    "guard_accepted_count": 7,
                    "guard_best_predicted_improvements": [0.3, -0.1],
                    "guard_selected_predicted_improvements": [0.2, -0.1],
                    "trajectory": [[0.0, 1.0], [0.1, 0.9]],
                    "progress": {"progress/subgoal_gain": 0.1},
                    "planning_wall_clock_s": 0.25,
                    "task_index": task_index,
                    "task_id": task_id,
                    "episode_index": episode_index,
                    "episode_seed": 2718 + 1000 * task_index + episode_index,
                    "success": episode_index == 0,
                    "initial_goal_distance": 1.0,
                    "final_goal_distance": 1.0 - reduction,
                })
        result["outcome"] = {
            "source": "terminal_final_evaluation",
            "status": "complete",
            "task_ids": [1, 2, 3, 4, 5],
            "episodes_per_task": 5,
            "num_episodes": 25,
            "successes": 5,
            "success_rate": 0.2,
            "distance_reduction_frac": reduction,
            "completed_results": rows,
            "completed_results_sha256": gate.stable_sha256(rows),
            "completion_sha256": _sha(f"completion/{setting}/{arm}"),
            "final_eval_progress_sha256": _sha(f"final-progress/{setting}/{arm}"),
            "checkpoint_sha256": _sha(f"final-checkpoint/{setting}/{arm}"),
        }
    return result


def valid_bundle() -> dict:
    cells = []
    for setting in gate.SETTING_IDS:
        for arm in gate.ARM_IDS:
            for seed in gate.SEEDS:
                cells.append(
                    {
                        "index": gate.expected_index(setting, arm, seed),
                        "setting_id": setting,
                        "arm_id": arm,
                        "seed": seed,
                        "fresh_start": True,
                        "boundaries": {
                            "5000": _boundary(setting, arm, 5_000),
                            "25000": _boundary(setting, arm, 25_000),
                        },
                    }
                )
    return {
        "schema_version": 1,
        "campaign_id": gate.CAMPAIGN_ID,
        **gate.package_binding(gate.load_manifest()),
        "cells": cells,
    }


def _cell(bundle: dict, setting: str, arm: str, seed: int) -> dict:
    return next(
        cell
        for cell in bundle["cells"]
        if (cell["setting_id"], cell["arm_id"], cell["seed"])
        == (setting, arm, seed)
    )


def test_valid_bundle_accepts_exact_source_locked_prefix_targets():
    result = gate.evaluate_bundle(valid_bundle())
    assert result["status"] == "accepted_engineering_pilot"
    assert all(result["gates"].values())
    first = result["cells"][0]["boundaries"]["5000"]["prefix_contract"]
    assert first["anchor_count"] == 5120
    assert first["locked_prefix_steps_mean"] == 4.0


def test_prefix_length_and_mask_contract_is_branchwise_and_fail_closed():
    bundle = valid_bundle()
    prefix = _cell(bundle, "antmaze-large", "GSEP", 110)["boundaries"]["5000"][
        "prefix_contract"
    ]
    prefix["fixed_validation_sampler"]["indices_sha256"] = _sha("subset-of-one-branch")

    result = gate.evaluate_bundle(bundle)
    assert result["status"] == "rejected"
    row = next(row for row in result["cells"] if row["index"] == 2)
    contract = row["boundaries"]["5000"]["prefix_contract"]
    assert not contract["passed"]
    assert not contract["gates"]["actual_sampler_indices_hash_exact"]


def test_control_low_clipping_gauge_and_method_are_observed_not_vetoes():
    bundle = valid_bundle()
    for cell in bundle["cells"]:
        if cell["arm_id"] != "GS":
            continue
        for target in gate.BOUNDARIES:
            boundary = cell["boundaries"][str(target)]
            for point in boundary["scalars"]["latent_gauge/min_ratio"]:
                point[1] = 0.1
            _replace_at(boundary, "latent_gauge/root/ratio", target, 0.1)
            _replace_at(boundary, "latent_gauge/future/ratio", target, 0.1)
            for tag in gate.CANDIDATE_CLIP_TAGS:
                for point in boundary["scalars"][tag]:
                    point[1] = 0.01
            _replace_at(boundary, "val/loss_total", target, 2.0)
            _replace_at(boundary, "control/q_advantage_over_z", target, -0.2)

    result = gate.evaluate_bundle(bundle)
    assert result["status"] == "accepted_engineering_pilot"
    assert result["gates"]["all_gs_cells_structurally_valid_at_both_boundaries"]
    gs = result["cells"][0]["boundaries"]["5000"]
    assert gs["structural_passed"]
    assert not gs["candidate_gates"]["absolute_gauge_retention"]
    assert not gs["candidate_gates"]["bounded_branch_rest_and_gain_clipping"]
    assert not gs["candidate_gates"]["unchanged_method_gates"]


def test_every_candidate_cell_must_pass_at_both_boundaries():
    bundle = valid_bundle()
    boundary = _cell(bundle, "scene", "GSEP", 111)["boundaries"]["5000"]
    _replace_at(boundary, "tree/support_recall", 5_000, 0.49)
    result = gate.evaluate_bundle(bundle)
    assert result["status"] == "rejected"
    assert not result["gates"][
        "all_gsep_cells_pass_unchanged_and_absolute_gates_at_both_boundaries"
    ]


def test_25k_paired_calibration_requires_strict_macro_and_setting_quorum():
    bundle = valid_bundle()
    for cell in bundle["cells"]:
        if cell["arm_id"] == "GSEP":
            _replace_at(
                cell["boundaries"]["25000"],
                gate.PREFIX + "predicted_vs_actual_normalized_task_rms",
                25_000,
                0.8,
            )
    result = gate.evaluate_bundle(bundle)
    assert result["status"] == "rejected"
    paired = result["paired_25k_calibration"]
    assert not paired["passed"]
    assert not paired["gates"]["gsep_macro_endpoint_error_strictly_lower"]
    assert paired["settings_with_strictly_lower_two_seed_endpoint_error"] == 0


def test_final_paired_outcome_requires_noninferiority_and_one_strict_gain():
    bundle = valid_bundle()
    for cell in bundle["cells"]:
        if cell["arm_id"] == "GS":
            cell["boundaries"]["25000"]["outcome"]["distance_reduction_frac"] = 0.3
    result = gate.evaluate_bundle(bundle)
    assert result["status"] == "rejected"
    outcomes = result["outcomes_25000"]
    assert not outcomes["passed"]
    assert not outcomes["paired_gates"]["gsep_macro_distance_reduction_not_worse"]


def test_matrix_indices_and_uniqueness_are_frozen():
    bundle = valid_bundle()
    bundle["cells"][-1]["setting_id"] = bundle["cells"][0]["setting_id"]
    bundle["cells"][-1]["arm_id"] = bundle["cells"][0]["arm_id"]
    bundle["cells"][-1]["seed"] = bundle["cells"][0]["seed"]
    with pytest.raises(gate.GateContractError, match="duplicate cell"):
        gate.evaluate_bundle(bundle)


@pytest.mark.parametrize("bad_count", [1.0, 128.0])
def test_fixed_validation_count_must_be_exact_source_locked_5120(bad_count: float):
    bundle = valid_bundle()
    boundary = _cell(bundle, "scene", "GSEP", 110)["boundaries"]["5000"]
    for point in boundary["scalars"]["data/validation_fixed_sample_count"]:
        point[1] = bad_count
    result = gate.evaluate_bundle(bundle)
    assert result["status"] == "rejected"
    row = next(item for item in result["cells"] if item["index"] == 6)
    assert not row["boundaries"]["5000"]["structural_gates"]["fixed_validation_axis_and_count"]


def test_tiny_valid_anchor_fraction_and_single_branch_subset_reject():
    bundle = valid_bundle()
    boundary = _cell(bundle, "puzzle-3x3", "GSEP", 111)["boundaries"]["25000"]
    for prefix in (gate.TRAIN_PREFIX, gate.PREFIX):
        _replace_at(boundary, prefix + "valid_anchor_fraction", 25_000, 1e-6)
    contract = boundary["prefix_contract"]
    contract["fixed_validation_sampler"]["global_sample_size"] = 1
    result = gate.evaluate_bundle(bundle)
    assert result["status"] == "rejected"
    row = next(item for item in result["cells"] if item["index"] == 11)
    evaluated = row["boundaries"]["25000"]
    assert not evaluated["train_prefix_telemetry"]["gates"]["all_anchor_denominators_complete"]
    assert not evaluated["prefix_contract"]["gates"]["actual_sampler_count_exact"]


def test_periodic_five_episode_monitor_cannot_satisfy_terminal_outcome():
    bundle = valid_bundle()
    outcome = _cell(bundle, "antmaze-large", "GSEP", 110)["boundaries"]["25000"]["outcome"]
    outcome["completed_results"] = outcome["completed_results"][:5]
    outcome["completed_results_sha256"] = gate.stable_sha256(outcome["completed_results"])
    outcome["num_episodes"] = 5
    outcome["successes"] = 1
    result = gate.evaluate_bundle(bundle)
    assert result["status"] == "rejected"
    assert not result["outcomes_25000"]["absolute_gates"]["all_twenty_outcomes_structurally_valid"]


def test_terminal_outcome_requires_exact_fallback_seed_rows_and_aggregates():
    bundle = valid_bundle()
    outcome = _cell(bundle, "scene", "GSEP", 111)["boundaries"]["25000"]["outcome"]
    outcome["completed_results"][7]["episode_seed"] += 1
    outcome["completed_results_sha256"] = gate.stable_sha256(outcome["completed_results"])
    result = gate.evaluate_bundle(bundle)
    assert result["status"] == "rejected"
    row = next(item for item in result["cells"] if item["index"] == 7)
    assert not row["outcome"]["fallback_seed_rows_exact"]


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("task_index", 0.9),
        ("task_id", 1.9),
        ("episode_index", 0.0),
        ("episode_seed", 2718.9),
        ("task_index", True),
    ],
)
def test_terminal_outcome_identity_fields_require_exact_nonbool_ints(field, bad_value):
    bundle = valid_bundle()
    outcome = _cell(bundle, "antmaze-large", "GSEP", 110)["boundaries"]["25000"]["outcome"]
    outcome["completed_results"][0][field] = bad_value
    outcome["completed_results_sha256"] = gate.stable_sha256(outcome["completed_results"])
    result = gate.evaluate_bundle(bundle)
    assert result["status"] == "rejected"


@pytest.mark.parametrize("field", ["initial_goal_distance", "final_goal_distance"])
def test_terminal_outcome_distances_must_be_nonnegative(field):
    bundle = valid_bundle()
    outcome = _cell(bundle, "scene", "GSEP", 111)["boundaries"]["25000"]["outcome"]
    outcome["completed_results"][0][field] = -0.1
    outcome["completed_results_sha256"] = gate.stable_sha256(outcome["completed_results"])
    result = gate.evaluate_bundle(bundle)
    assert result["status"] == "rejected"


@pytest.mark.parametrize("mutation", ["strip", "extra", "nested_nan"])
def test_terminal_outcome_requires_exact_finite_episode_result_schema(mutation):
    bundle = valid_bundle()
    outcome = _cell(bundle, "scene", "GSEP", 111)["boundaries"]["25000"]["outcome"]
    row = outcome["completed_results"][0]
    if mutation == "strip":
        del row["nodes"]
    elif mutation == "extra":
        row["unsealed"] = 1
    else:
        row["progress"]["nested"] = [0.0, float("nan")]
    if mutation != "nested_nan":
        outcome["completed_results_sha256"] = gate.stable_sha256(outcome["completed_results"])
    else:
        with pytest.raises(gate.GateContractError, match="nonfinite"):
            gate.evaluate_bundle(bundle)
        return
    result = gate.evaluate_bundle(bundle)
    assert result["status"] == "rejected"
    evaluated = next(item for item in result["cells"] if item["index"] == 7)
    assert not evaluated["outcome"]["episode_result_schema_exact"]


@pytest.mark.parametrize("field,bad_value", [("index", 0.0), ("seed", 110.0), ("seed", True)])
def test_cell_index_and_seed_require_exact_nonbool_ints(field, bad_value):
    bundle = valid_bundle()
    bundle["cells"][0][field] = bad_value
    with pytest.raises(gate.GateContractError):
        gate.evaluate_bundle(bundle)


def test_report_must_bind_exact_verified_package_and_core():
    bundle = valid_bundle()
    bundle["trainer_code_fingerprint"] = _sha("stale-trainer")
    with pytest.raises(gate.GateContractError, match="trainer_code_fingerprint"):
        gate.evaluate_bundle(bundle)


def test_alternate_manifest_cannot_relax_a_threshold():
    bundle = valid_bundle()
    manifest = gate.load_manifest()
    manifest["acceptance"]["unchanged_exp20_candidate_gates"][
        "min_recent_latent_gauge_ratio"
    ] = 0.0
    with pytest.raises(gate.GateContractError, match="supplied manifest differs"):
        gate.evaluate_bundle(bundle, manifest)
