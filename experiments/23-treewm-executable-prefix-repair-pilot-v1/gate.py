#!/usr/bin/env python3
"""Evaluate the prospective Exp23 result bundle without reading live run trees.

The input is a single, immutable JSON report assembled after all twenty runs finish.
This module deliberately has no launcher, scheduler, checkpoint, or output-directory
integration.  It checks the preregistered scientific contract and prints a
deterministic decision.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
CAMPAIGN_ID = "treewm-executable-prefix-repair-pilot-v1-launch5"
PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MANIFEST = PACKAGE_DIR / "manifest.json"
PREFIX_TARGET_LOCK = PACKAGE_DIR / "prefix_target.lock.json"
RESOLVED_CONFIG_LOCK = PACKAGE_DIR / "resolved_config.lock.json"
CAUSAL_PARITY_LOCK = PACKAGE_DIR / "causal_parity.lock.json"
CAMPAIGN_MODULE_PATH = PACKAGE_DIR / "campaign.py"
SETTING_IDS = (
    "antmaze-large",
    "scene",
    "puzzle-3x3",
    "puzzle-4x4-100m",
    "cube-quadruple-100m",
)
ARM_IDS = ("GS", "GSEP")
SEEDS = (110, 111)
BOUNDARIES = (5_000, 25_000)
HORIZONS = (4, 8, 16, 32, 64)
PREFIX_MAX_STEPS = 4
GRADIENT_WINDOW = 5_000
GAUGE_WINDOW = 1_000
GAIN_WINDOW = 5_000
EPISODE_RESULT_KEYS = frozenset(
    {
        "success", "steps", "replans", "nodes", "final_goal_distance",
        "best_goal_distance", "chunk_lengths", "selected_depths",
        "initial_goal_distance", "displacement", "path_length",
        "action_magnitude", "no_action_plans", "guard_plans",
        "guard_rejections", "guard_candidate_count", "guard_accepted_count",
        "guard_best_predicted_improvements",
        "guard_selected_predicted_improvements", "trajectory", "progress",
        "task_index", "task_id", "episode_index", "episode_seed",
        "planning_wall_clock_s",
    }
)
EPISODE_NONNEGATIVE_INT_FIELDS = (
    "steps", "replans", "nodes", "no_action_plans", "guard_plans",
    "guard_rejections", "guard_candidate_count", "guard_accepted_count",
)
EPISODE_NONNEGATIVE_NUMBER_FIELDS = (
    "final_goal_distance", "best_goal_distance", "initial_goal_distance",
    "displacement", "path_length", "action_magnitude", "planning_wall_clock_s",
)

TRAIN_PREFIX = "train/executable_prefix/"
PREFIX = "val/executable_prefix/"
PREFIX_TERMS = (
    "executable_prefix_action",
    "executable_prefix_latent",
    "executable_prefix_endpoint",
)
GRADIENT_NORM_TAGS = (
    "train/grad_norm_world",
    "train/grad_norm_gain",
    "train/grad_norm_world_rest",
    "train/grad_norm_branch_transformer",
)
GRADIENT_CLIP_TAGS = (
    "train/grad_clip_coefficient_world",
    "train/grad_clip_coefficient_gain",
    "train/grad_clip_coefficient_world_rest",
    "train/grad_clip_coefficient_branch_transformer",
)
# The preregistered clipping decision is branch/rest/gain.  The aggregate world
# coefficient remains structurally required but is not counted a fourth veto.
CANDIDATE_CLIP_TAGS = (
    "train/grad_clip_coefficient_branch_transformer",
    "train/grad_clip_coefficient_world_rest",
    "train/grad_clip_coefficient_gain",
)
GAIN_TAGS = (
    "expansion/gain_rank_correlation",
    "expansion/gain_pairwise_accuracy",
    "expansion/gain_eligible_decision_fraction",
    "expansion/gain_ordered_pair_count",
    "expansion/gain_pair_coverage_fraction",
)
GAUGE_EXACT_TAGS = (
    "latent_gauge/root/scale",
    "latent_gauge/root/reference",
    "latent_gauge/root/ratio",
    "latent_gauge/future/scale",
    "latent_gauge/future/reference",
    "latent_gauge/future/ratio",
    "latent_gauge/min_ratio",
    "latent_gauge/loss",
    "latent_gauge/reference_sealed",
    "latent_gauge/reference_update",
)
METHOD_EXACT_TAGS = (
    "val/loss_total",
    "val/loss_multistep_self_fed",
    "val/loss_horizon",
    "control/retrieval_uses_task_metric_endpoint",
    "control/q_advantage_over_z",
    "control/q_advantage_over_random_proj",
    *GAIN_TAGS,
    "tree/support_recall",
    "tree/support_precision",
    "data/validation_fixed_sample_count",
    *(f"data/validation_horizon_label_fraction_h{h}" for h in HORIZONS),
)
PREFIX_COMMON_SUFFIXES = (
    "schema_version",
    "loss_action_normalized",
    "loss_latent",
    "loss_endpoint_normalized_task",
    "action_raw_env_abs_mean",
    "action_raw_env_rms",
    "action_applied_env_abs_mean",
    "action_applied_env_rms",
    "action_logged_env_abs_mean",
    "action_logged_env_rms",
    "action_clipped_fraction",
    "action_finite_fraction",
    "action_applied_finite_fraction",
    "action_logged_finite_fraction",
    "predicted_vs_actual_normalized_task_rms",
    "predicted_normalized_task_displacement_rms",
    "actual_normalized_task_displacement_rms",
    "predicted_vs_actual_guard_metric_error",
    "predicted_guard_metric_displacement",
    "actual_guard_metric_displacement",
    "prefix_steps_mean",
    "valid_anchor_fraction",
    "matched_branches_per_anchor",
    "action_scalars_per_anchor",
    "action_raw_finite_scalars_per_anchor",
    "action_applied_finite_scalars_per_anchor",
    "action_logged_finite_scalars_per_anchor",
    "goal_metric_onehot",
    "latent_target_scale",
)
PREFIX_HAMMING_SUFFIXES = (
    "predicted_vs_actual_hamming",
    "predicted_vs_actual_hamming_fraction",
    "predicted_hamming_displacement",
    "actual_hamming_displacement",
)


class GateContractError(ValueError):
    """The report is malformed or does not describe the frozen 20-cell design."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GateContractError(message)


def finite(value: object) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _finite_nested(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, list):
        return all(_finite_nested(item) for item in value)
    if isinstance(value, Mapping):
        return all(
            isinstance(key, str) and _finite_nested(item)
            for key, item in value.items()
        )
    return False


def _episode_result_row_valid(row: Mapping[str, Any]) -> bool:
    if set(row) != EPISODE_RESULT_KEYS or type(row.get("success")) is not bool:
        return False
    if any(
        type(row.get(name)) is not int or row[name] < 0
        for name in EPISODE_NONNEGATIVE_INT_FIELDS
    ):
        return False
    if any(
        not finite(row.get(name)) or float(row[name]) < 0.0
        for name in EPISODE_NONNEGATIVE_NUMBER_FIELDS
    ):
        return False
    for name in ("chunk_lengths", "selected_depths"):
        values = row.get(name)
        if not isinstance(values, list) or any(
            type(item) is not int or item < 0 for item in values
        ):
            return False
    for name in (
        "guard_best_predicted_improvements",
        "guard_selected_predicted_improvements",
    ):
        values = row.get(name)
        if not isinstance(values, list) or not all(finite(item) for item in values):
            return False
    if not isinstance(row.get("trajectory"), list) or not _finite_nested(
        row["trajectory"]
    ):
        return False
    if not isinstance(row.get("progress"), Mapping) or not _finite_nested(
        row["progress"]
    ):
        return False
    return True


def sha256_string(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def stable_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_manifest(path: str | Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateContractError(f"cannot read manifest {path}: {exc}") from exc
    _require(isinstance(value, dict), "manifest root must be an object")
    _require(value.get("schema_version") == 1, "manifest schema differs")
    _require(value.get("campaign_id") == CAMPAIGN_ID, "manifest campaign differs")
    _require(value.get("design", {}).get("settings") == list(SETTING_IDS), "manifest settings differ")
    _require(value.get("design", {}).get("arms") == list(ARM_IDS), "manifest arms differ")
    _require(value.get("design", {}).get("seeds") == list(SEEDS), "manifest seeds differ")
    _require(
        value.get("design", {}).get("analysis_boundaries") == list(BOUNDARIES),
        "manifest boundaries differ",
    )
    _require(
        value.get("design", {}).get("expected_cells") == 20,
        "manifest cell count differs",
    )
    _require(
        value.get("scientific_contract", {})
        .get("future_sets", {})
        .get("executable_prefix_steps")
        == PREFIX_MAX_STEPS,
        "manifest executable prefix depth differs",
    )
    _require(isinstance(value.get("acceptance"), Mapping), "manifest acceptance missing")
    return value


def load_prefix_target_lock(path: str | Path = PREFIX_TARGET_LOCK) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateContractError(f"prefix-target lock unavailable: {exc}") from exc
    _require(isinstance(value, dict), "prefix-target lock must be an object")
    body = dict(value)
    claimed = body.pop("artifact_sha256", None)
    _require(
        sha256_string(claimed) and claimed == stable_sha256(body),
        "prefix-target lock artifact hash differs",
    )
    _require(
        value.get("status")
        == "frozen_outcome_blind_fixed_validation_prefix_targets",
        "prefix-target lock is not frozen",
    )
    _require(
        set((value.get("settings") or {}).keys()) == set(SETTING_IDS),
        "prefix-target lock setting coverage differs",
    )
    return value


def load_resolved_config_lock(path: str | Path = RESOLVED_CONFIG_LOCK) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateContractError(f"resolved-config lock unavailable: {exc}") from exc
    _require(isinstance(value, dict), "resolved-config lock must be an object")
    body = dict(value)
    claimed = body.pop("artifact_sha256", None)
    _require(
        sha256_string(claimed) and claimed == stable_sha256(body),
        "resolved-config lock artifact hash differs",
    )
    matrix = value.get("matrix")
    _require(
        isinstance(matrix, list)
        and len(matrix) == 20
        and [row.get("index") for row in matrix if isinstance(row, Mapping)]
        == list(range(20)),
        "resolved-config lock matrix differs",
    )
    return value


def load_causal_parity_lock(path: str | Path = CAUSAL_PARITY_LOCK) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateContractError(f"causal-parity lock unavailable: {exc}") from exc
    _require(isinstance(value, dict), "causal-parity lock must be an object")
    body = dict(value)
    claimed = body.pop("artifact_sha256", None)
    _require(
        sha256_string(claimed) and claimed == stable_sha256(body),
        "causal-parity lock artifact hash differs",
    )
    _require(
        value.get("status") == "frozen_outcome_blind_causal_parity"
        and isinstance(value.get("pairs"), list)
        and len(value["pairs"]) == 10,
        "causal-parity lock coverage differs",
    )
    return value


def package_binding(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Bind a report to this exact verified package, audit, and trainer core."""
    spec = importlib.util.spec_from_file_location(
        "exp23_campaign_for_gate", CAMPAIGN_MODULE_PATH
    )
    _require(spec is not None and spec.loader is not None, "cannot load package verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    frozen_manifest, _lock = module.load_contract()
    _require(
        dict(manifest) == frozen_manifest,
        "supplied manifest differs from frozen package manifest",
    )
    protocol = module.verify_protocol_lock()
    return {
        "package_protocol_sha256": protocol,
        "manifest_sha256": stable_sha256(frozen_manifest),
        "trainer_code_fingerprint": str(
            frozen_manifest["core_binding"]["trainer_code_fingerprint"]
        ),
        "weight_audit_artifact_sha256": str(
            frozen_manifest["weight_audit"]["artifact_sha256"]
        ),
        "prefix_target_artifact_sha256": str(
            frozen_manifest["prefix_target_contract"]["artifact_sha256"]
        ),
        "resolved_config_artifact_sha256": str(
            frozen_manifest["resolved_config_contract"]["artifact_sha256"]
        ),
        "causal_parity_artifact_sha256": str(
            frozen_manifest["causal_parity_contract"]["artifact_sha256"]
        ),
    }


def expected_index(setting_id: str, arm_id: str, seed: int) -> int:
    return (
        (SETTING_IDS.index(setting_id) * len(ARM_IDS) + ARM_IDS.index(arm_id))
        * len(SEEDS)
        + SEEDS.index(seed)
    )


def _axis(target: int, cadence: int, window: int | None = None) -> tuple[int, ...]:
    if window is None:
        lower = cadence
    else:
        lower = max(cadence, target - min(window, target))
    first = ((lower + cadence - 1) // cadence) * cadence
    return tuple(range(first, target + 1, cadence))


def _parse_scalars(value: object, *, cell: str, target: int) -> dict[str, dict[int, float]]:
    _require(isinstance(value, Mapping), f"{cell}@{target}: scalars must be an object")
    result: dict[str, dict[int, float]] = {}
    for raw_tag, raw_points in value.items():
        _require(isinstance(raw_tag, str) and raw_tag, f"{cell}@{target}: invalid scalar tag")
        _require(
            isinstance(raw_points, Sequence) and not isinstance(raw_points, (str, bytes)),
            f"{cell}@{target}:{raw_tag}: scalar series must be a list",
        )
        points: dict[int, float] = {}
        for raw_point in raw_points:
            _require(
                isinstance(raw_point, Sequence)
                and not isinstance(raw_point, (str, bytes))
                and len(raw_point) == 2,
                f"{cell}@{target}:{raw_tag}: each point must be [update, value]",
            )
            raw_step, raw_value = raw_point
            _require(
                isinstance(raw_step, int) and not isinstance(raw_step, bool) and raw_step >= 0,
                f"{cell}@{target}:{raw_tag}: invalid update",
            )
            _require(raw_step not in points, f"{cell}@{target}:{raw_tag}: duplicate update {raw_step}")
            # Store nonfinite numerical values so the scientific/structural gate can
            # reject them without confusing them with malformed JSON structure.
            _require(
                isinstance(raw_value, (int, float)) and not isinstance(raw_value, bool),
                f"{cell}@{target}:{raw_tag}: value is not numeric",
            )
            points[raw_step] = float(raw_value)
        result[raw_tag] = points
    return result


def _at(metrics: Mapping[str, Mapping[int, float]], tag: str, target: int) -> float | None:
    value = metrics.get(tag, {}).get(target)
    return float(value) if finite(value) else None


def _complete(
    metrics: Mapping[str, Mapping[int, float]], tag: str, axis: Sequence[int]
) -> list[float] | None:
    series = metrics.get(tag, {})
    if tuple(sorted(step for step in series if step in set(axis))) != tuple(axis):
        return None
    values = [series[step] for step in axis]
    return [float(value) for value in values] if all(finite(value) for value in values) else None


def _close(left: float, right: float, *, atol: float = 1e-8) -> bool:
    return abs(float(left) - float(right)) <= atol * max(1.0, abs(float(left)), abs(float(right)))


def _normalized_mask(
    value: object, *, label: str, width: int
) -> list[list[int]]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{label} must be a list",
    )
    result: list[list[int]] = []
    for row in value:
        _require(
            isinstance(row, Sequence)
            and not isinstance(row, (str, bytes))
            and len(row) == width,
            f"{label} rows must have {width} entries",
        )
        normalized: list[int] = []
        for item in row:
            _require(
                item in (0, 1, False, True),
                f"{label} entries must be binary",
            )
            normalized.append(int(item))
        result.append(normalized)
    return result


def _normalized_int_list(value: object, *, label: str, positive: bool) -> list[int]:
    _require(
        isinstance(value, Sequence) and not isinstance(value, (str, bytes)),
        f"{label} must be a list",
    )
    result: list[int] = []
    for item in value:
        _require(
            isinstance(item, int)
            and not isinstance(item, bool)
            and (item > 0 if positive else 1 <= item <= PREFIX_MAX_STEPS),
            f"{label} contains an invalid integer",
        )
        result.append(int(item))
    return result


def _length_counts(lengths: Sequence[int]) -> dict[str, int]:
    return {str(length): sum(item == length for item in lengths) for length in range(1, 5)}


def evaluate_prefix_contract(
    value: object,
    manifest: Mapping[str, Any],
    *,
    setting_id: str,
) -> dict[str, Any]:
    """Bind actual sampler metadata to the frozen source-derived target lock.

    Full branch counts, lengths, and 64-wide masks are an audit-level immutable
    target contract.  The trainer publishes only the fixed-validation sampler
    summary plus aggregate scalar telemetry, so no result-time artifact hashes are
    invented or mislabeled as observed model-loss tensors.
    """

    _require(isinstance(value, Mapping), "prefix_contract must be an object")
    lock = load_prefix_target_lock()
    expected = lock["settings"][setting_id]
    sampler = value.get("fixed_validation_sampler")
    _require(
        isinstance(sampler, Mapping),
        "prefix_contract fixed_validation_sampler must be an object",
    )
    expected_sampler = expected["fixed_validation_sampler"]
    gates = {
        "source_lock_setting_exact": value.get("setting_id") == setting_id,
        "source_target_contract_bound": value.get("target_contract_sha256")
        == expected["target_contract_sha256"],
        "source_prefix_artifact_bound": value.get(
            "prefix_target_artifact_sha256"
        )
        == lock["artifact_sha256"],
        "validation_manifest_bound": value.get("validation_manifest_sha256")
        == expected["validation_manifest_sha256"],
        "actual_sampler_summary_exact": dict(sampler) == expected_sampler,
        "actual_sampler_seed_exact": sampler.get("seed") == 1701,
        "actual_sampler_count_exact": sampler.get("global_sample_size") == 5120,
        "actual_sampler_shape_exact": sampler.get("batch_size_per_rank") == 256
        and sampler.get("num_batches") == 20
        and sampler.get("num_replicas") == 1,
        "actual_sampler_indices_hash_exact": sampler.get("indices_sha256")
        == expected_sampler["indices_sha256"],
        "all_source_anchors_have_targets": expected["anchor_count"] == 5120
        and expected["all_anchors_have_match"] is True,
        "source_branch_lengths_are_exact_h4": expected[
            "prefix_length_histogram"
        ]
        == {
            "1": 0,
            "2": 0,
            "3": 0,
            "4": expected["matched_branch_count"],
        },
        "source_hmax64_mask_bound": sha256_string(
            expected["sorted_prefix_masks_hmax64_sha256"]
        ),
    }
    branch_count = int(expected["matched_branch_count"])
    action_steps = int(expected["prefix_action_step_count"])
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "branch_count": branch_count,
        "anchor_count": int(expected["anchor_count"]),
        "action_mask_width": int(
            manifest["scientific_contract"]["future_sets"]["h_max"]
        ),
        "expected_length_counts": dict(expected["prefix_length_histogram"]),
        "locked_prefix_steps_mean": action_steps / branch_count,
        "matched_branches_per_anchor": branch_count / int(expected["anchor_count"]),
        "action_scalars_per_anchor": int(expected["prefix_action_scalar_count"])
        / int(expected["anchor_count"]),
        "target_contract_sha256": expected["target_contract_sha256"],
        "fixed_validation_sampler": dict(sampler),
    }


def _common_prefix_telemetry(
    metrics: Mapping[str, Mapping[int, float]],
    target: int,
    metric_prefix: str,
    manifest: Mapping[str, Any],
    prefix_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    values = {
        suffix: _at(metrics, metric_prefix + suffix, target)
        for suffix in PREFIX_COMMON_SUFFIXES
    }
    onehot = values["goal_metric_onehot"]
    hamming = {
        suffix: _at(metrics, metric_prefix + suffix, target)
        for suffix in PREFIX_HAMMING_SUFFIXES
    }
    required = list(values.values()) + (list(hamming.values()) if onehot == 1.0 else [])
    scalars = values["action_scalars_per_anchor"]
    finite_counts = (
        values["action_raw_finite_scalars_per_anchor"],
        values["action_applied_finite_scalars_per_anchor"],
        values["action_logged_finite_scalars_per_anchor"],
    )
    gates = {
        "all_common_and_conditional_hamming_tags_finite": all(
            finite(item) for item in required
        ),
        "schema_version_matches": values["schema_version"]
        == float(manifest["core_binding"]["executable_prefix_schema_version"]),
        "finite_fractions_complete": all(
            finite(values[key]) and _close(float(values[key]), 1.0)
            for key in (
                "action_finite_fraction",
                "action_applied_finite_fraction",
                "action_logged_finite_fraction",
            )
        ),
        "finite_counts_complete": finite(scalars)
        and float(scalars) > 0.0
        and all(
            finite(item) and _close(float(item), float(scalars), atol=1e-6)
            for item in finite_counts
        ),
        "all_anchor_denominators_complete": finite(
            values["valid_anchor_fraction"]
        )
        and _close(float(values["valid_anchor_fraction"]), 1.0, atol=1e-9)
        and finite(values["matched_branches_per_anchor"])
        and float(values["matched_branches_per_anchor"]) > 0.0
        and finite(values["action_scalars_per_anchor"])
        and float(values["action_scalars_per_anchor"]) > 0.0,
        "losses_and_physical_magnitudes_nonnegative": all(
            finite(values[key]) and float(values[key]) >= 0.0
            for key in (
                "loss_action_normalized",
                "loss_latent",
                "loss_endpoint_normalized_task",
                "action_raw_env_abs_mean",
                "action_raw_env_rms",
                "action_applied_env_abs_mean",
                "action_applied_env_rms",
                "action_logged_env_abs_mean",
                "action_logged_env_rms",
                "predicted_vs_actual_normalized_task_rms",
                "predicted_normalized_task_displacement_rms",
                "actual_normalized_task_displacement_rms",
                "predicted_vs_actual_guard_metric_error",
                "predicted_guard_metric_displacement",
                "actual_guard_metric_displacement",
            )
        ),
        "clipped_fraction_in_unit_interval": finite(
            values["action_clipped_fraction"]
        )
        and 0.0 <= float(values["action_clipped_fraction"]) <= 1.0,
        "prefix_steps_mean_in_descriptive_range": finite(values["prefix_steps_mean"])
        and 0.0 < float(values["prefix_steps_mean"]) <= float(PREFIX_MAX_STEPS),
        "latent_target_scale_positive": finite(values["latent_target_scale"])
        and float(values["latent_target_scale"]) > 0.0,
        "onehot_flag_and_hamming_domain_valid": onehot in (0.0, 1.0)
        and (
            onehot == 0.0
            or (
                all(finite(item) and float(item) >= 0.0 for item in hamming.values())
                and 0.0
                <= float(hamming["predicted_vs_actual_hamming_fraction"] or 0.0)
                <= 1.0
            )
        ),
    }
    if prefix_contract is not None:
        gates["fixed_validation_matched_branches_exact"] = _close(
            float(values["matched_branches_per_anchor"]),
            float(prefix_contract["matched_branches_per_anchor"]),
            atol=1e-6,
        )
        gates["fixed_validation_action_scalars_exact"] = _close(
            float(values["action_scalars_per_anchor"]),
            float(prefix_contract["action_scalars_per_anchor"]),
            atol=1e-6,
        )
        gates["fixed_validation_prefix_mean_exact"] = _close(
            float(values["prefix_steps_mean"]),
            float(prefix_contract["locked_prefix_steps_mean"]),
            atol=1e-6,
        )
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "values": values,
        "hamming_values": hamming if onehot == 1.0 else {},
    }


def _prefix_term_telemetry(
    metrics: Mapping[str, Mapping[int, float]],
    target: int,
    arm_id: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    arm = next(item for item in manifest["arms"] if item["id"] == arm_id)
    short_by_term = {
        "executable_prefix_action": "action",
        "executable_prefix_latent": "latent",
        "executable_prefix_endpoint": "endpoint",
    }
    rows: dict[str, Any] = {}
    all_passed = True
    for scope in ("train", "val"):
        for term in PREFIX_TERMS:
            alias = _at(metrics, f"{scope}/loss_{term}", target)
            raw = _at(metrics, f"{scope}/loss_raw/{term}", target)
            effective = _at(metrics, f"{scope}/loss_effective/{term}", target)
            weight = _at(metrics, f"{scope}/loss_weight/{term}", target)
            schedule = _at(metrics, f"{scope}/loss_schedule/{term}", target)
            expected_weight = float(
                arm["executable_prefix_weights"][short_by_term[term]]
            )
            gates = {
                "all_term_tags_finite": all(
                    finite(item) for item in (alias, raw, effective, weight, schedule)
                ),
                "raw_alias_consistent": finite(alias)
                and finite(raw)
                and _close(float(alias), float(raw), atol=1e-7),
                "frozen_weight_matches": finite(weight)
                and _close(float(weight), expected_weight, atol=1e-7),
                "schedule_exactly_one": finite(schedule) and float(schedule) == 1.0,
                "effective_equals_raw_times_weight_schedule": all(
                    finite(item) for item in (raw, effective, weight, schedule)
                )
                and _close(
                    float(effective),
                    float(raw) * float(weight) * float(schedule),
                    atol=1e-7,
                ),
                "gs_effective_exactly_zero": arm_id != "GS"
                or (finite(effective) and float(effective) == 0.0),
            }
            passed = all(gates.values())
            all_passed &= passed
            rows[f"{scope}/{term}"] = {
                "passed": passed,
                "gates": gates,
                "alias": alias,
                "raw": raw,
                "effective": effective,
                "weight": weight,
                "schedule": schedule,
            }
    return {"passed": all_passed, "terms": rows}


def _calibration(
    metrics: Mapping[str, Mapping[int, float]],
    target: int,
    prefix_contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    h4_gate = manifest["acceptance"]["absolute_h4_calibration_gates"]
    prefix_gate = manifest["acceptance"]["prefix_structural_gates"]
    values = {suffix: _at(metrics, PREFIX + suffix, target) for suffix in PREFIX_COMMON_SUFFIXES}
    onehot = values["goal_metric_onehot"]
    hamming = {
        suffix: _at(metrics, PREFIX + suffix, target) for suffix in PREFIX_HAMMING_SUFFIXES
    }
    required_values = list(values.values()) + (
        list(hamming.values()) if onehot == 1.0 else []
    )
    exact_finite = all(finite(value) for value in required_values)

    raw = values["action_raw_env_rms"]
    applied = values["action_applied_env_rms"]
    predicted = values["predicted_normalized_task_displacement_rms"]
    actual = values["actual_normalized_task_displacement_rms"]
    endpoint_error = values["predicted_vs_actual_normalized_task_rms"]
    raw_applied_ratio = float(raw) / float(applied) if finite(raw) and finite(applied) and applied > 0 else None
    predicted_actual_ratio = (
        float(predicted) / float(actual)
        if finite(predicted) and finite(actual) and actual > 0
        else None
    )
    relative_endpoint_error = (
        float(endpoint_error) / float(actual)
        if finite(endpoint_error) and finite(actual) and actual > 0
        else None
    )
    distortion = (
        (float(raw) - float(applied)) / float(raw)
        if finite(raw) and finite(applied) and raw > 0
        else None
    )

    action_scalars = values["action_scalars_per_anchor"]
    finite_scalar_counts = [
        values["action_raw_finite_scalars_per_anchor"],
        values["action_applied_finite_scalars_per_anchor"],
        values["action_logged_finite_scalars_per_anchor"],
    ]
    structural_gates = {
        "exact_finite_schema": exact_finite
        and values["schema_version"] == float(prefix_gate["schema_version"]),
        "losses_nonnegative": all(
            finite(values[key]) and float(values[key]) >= 0.0
            for key in (
                "loss_action_normalized",
                "loss_latent",
                "loss_endpoint_normalized_task",
            )
        ),
        "finite_action_fractions_complete": all(
            finite(values[key]) and _close(float(values[key]), 1.0)
            for key in (
                "action_finite_fraction",
                "action_applied_finite_fraction",
                "action_logged_finite_fraction",
            )
        ),
        "finite_action_counts_complete": finite(action_scalars)
        and float(action_scalars) > 0.0
        and all(
            finite(count) and _close(float(count), float(action_scalars), atol=1e-6)
            for count in finite_scalar_counts
        ),
        "positive_anchor_branch_and_action_counts": all(
            finite(values[key]) and float(values[key]) > 0.0
            for key in (
                "valid_anchor_fraction",
                "matched_branches_per_anchor",
                "action_scalars_per_anchor",
            )
        )
        and float(values["valid_anchor_fraction"] or 0.0) <= 1.0,
        "positive_action_and_displacement_rms": all(
            finite(value) and float(value) > 0.0
            for value in (raw, applied, predicted, actual)
        ),
        "clip_fraction_in_unit_interval": finite(values["action_clipped_fraction"])
        and 0.0 <= float(values["action_clipped_fraction"]) <= 1.0,
        "guard_domain_telemetry_finite": all(
            finite(values[key]) and float(values[key]) >= 0.0
            for key in (
                "predicted_vs_actual_guard_metric_error",
                "predicted_guard_metric_displacement",
                "actual_guard_metric_displacement",
            )
        ),
        "onehot_flag_and_hamming_telemetry_valid": onehot in (0.0, 1.0)
        and (
            onehot == 0.0
            or (
                all(finite(value) and float(value) >= 0.0 for value in hamming.values())
                and float(hamming["predicted_vs_actual_hamming_fraction"] or -1.0) <= 1.0
            )
        ),
        "positive_latent_target_scale": finite(values["latent_target_scale"])
        and float(values["latent_target_scale"]) > 0.0,
        # This scalar is descriptive only.  Exact acceptance uses the full
        # branchwise vectors/distributions/masks above, never a mean==4 check.
        "reported_prefix_mean_is_descriptive_and_finite": finite(
            values["prefix_steps_mean"]
        )
        and float(values["prefix_steps_mean"]) > 0.0,
    }
    action_interval = h4_gate["action_raw_over_applied_rms_interval"]
    displacement_interval = h4_gate[
        "predicted_over_actual_task_displacement_rms_interval"
    ]
    absolute_gates = {
        "raw_over_applied_action_rms_in_1_to_2": finite(raw_applied_ratio)
        and float(action_interval[0])
        <= float(raw_applied_ratio)
        <= float(action_interval[1]),
        "action_clip_fraction_at_most_0_25": finite(values["action_clipped_fraction"])
        and float(values["action_clipped_fraction"])
        <= float(h4_gate["max_action_clipped_fraction"]),
        "predicted_over_actual_displacement_rms_in_0_5_to_2": finite(predicted_actual_ratio)
        and float(displacement_interval[0])
        <= float(predicted_actual_ratio)
        <= float(displacement_interval[1]),
        "endpoint_error_over_actual_displacement_at_most_1": finite(relative_endpoint_error)
        and 0.0 <= float(relative_endpoint_error)
        <= float(h4_gate["max_endpoint_error_over_actual_task_displacement"]),
    }
    return {
        "structural_passed": all(structural_gates.values()),
        "absolute_passed": all(structural_gates.values()) and all(absolute_gates.values()),
        "structural_gates": structural_gates,
        "absolute_gates": absolute_gates,
        "values": values,
        "hamming_values": hamming if onehot == 1.0 else {},
        "raw_over_applied_action_rms": raw_applied_ratio,
        "predicted_over_actual_displacement_rms": predicted_actual_ratio,
        "relative_normalized_endpoint_error": relative_endpoint_error,
        "action_distortion": distortion,
    }


def evaluate_boundary(
    boundary: Mapping[str, Any],
    *,
    cell_label: str,
    target: int,
    arm_id: str,
    setting_id: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    _require(boundary.get("update") == target, f"{cell_label}@{target}: update differs")
    metrics = _parse_scalars(boundary.get("scalars"), cell=cell_label, target=target)
    prefix_contract = evaluate_prefix_contract(
        boundary.get("prefix_contract"),
        manifest,
        setting_id=setting_id,
    )
    calibration = _calibration(metrics, target, prefix_contract, manifest)
    train_prefix = _common_prefix_telemetry(
        metrics, target, TRAIN_PREFIX, manifest
    )
    validation_prefix = _common_prefix_telemetry(
        metrics, target, PREFIX, manifest, prefix_contract
    )
    prefix_terms = _prefix_term_telemetry(
        metrics, target, arm_id, manifest
    )

    unchanged = manifest["acceptance"]["unchanged_exp20_candidate_gates"]
    scientific = manifest["scientific_contract"]

    exact_tags = (
        *GAUGE_EXACT_TAGS,
        *METHOD_EXACT_TAGS,
        *GRADIENT_NORM_TAGS,
        *GRADIENT_CLIP_TAGS,
        *(TRAIN_PREFIX + suffix for suffix in PREFIX_COMMON_SUFFIXES),
        *(PREFIX + suffix for suffix in PREFIX_COMMON_SUFFIXES),
    )
    required_exact_finite = all(_at(metrics, tag, target) is not None for tag in exact_tags)
    prefix_domain_parity = (
        train_prefix["values"]["goal_metric_onehot"]
        == validation_prefix["values"]["goal_metric_onehot"]
    )

    training_cadence = int(scientific["training_telemetry_every_updates"])
    validation_cadence = int(scientific["validation_every_updates"])
    gradient_axis = _axis(
        target, training_cadence, int(unchanged["gradient_recent_window_updates"])
    )
    gradient_series = {
        tag: _complete(metrics, tag, gradient_axis)
        for tag in (*GRADIENT_NORM_TAGS, *GRADIENT_CLIP_TAGS)
    }
    complete_gradient_axis = all(values is not None for values in gradient_series.values())
    nonzero_gradients = complete_gradient_axis and all(
        value > float(unchanged["min_gradient_norm"])
        for tag in GRADIENT_NORM_TAGS
        for value in (gradient_series[tag] or [])
    )
    valid_clip_coefficients = complete_gradient_axis and all(
        0.0 < value <= 1.0
        for tag in GRADIENT_CLIP_TAGS
        for value in (gradient_series[tag] or [])
    )
    low_clip_fractions = {
        tag: (
            sum(
                value < float(unchanged["min_clip_coefficient"])
                for value in (gradient_series[tag] or [])
            )
            / len(gradient_series[tag] or [])
            if gradient_series[tag]
            else None
        )
        for tag in CANDIDATE_CLIP_TAGS
    }
    bounded_clipping = valid_clip_coefficients and all(
        finite(value)
        and float(value)
        <= float(unchanged["max_recent_5k_fraction_below_min_clip"])
        for value in low_clip_fractions.values()
    )

    gauge_axis = _axis(
        target, training_cadence, int(unchanged["gauge_recent_window_updates"])
    )
    gauge_series = _complete(metrics, "latent_gauge/min_ratio", gauge_axis)
    root_ratio = _at(metrics, "latent_gauge/root/ratio", target)
    future_ratio = _at(metrics, "latent_gauge/future/ratio", target)
    minimum_ratio = _at(metrics, "latent_gauge/min_ratio", target)
    ratio_consistent = all(finite(value) and float(value) > 0.0 for value in (root_ratio, future_ratio, minimum_ratio)) and float(minimum_ratio) <= min(float(root_ratio), float(future_ratio)) + 1e-5
    reference_valid = (
        _at(metrics, "latent_gauge/reference_sealed", target)
        == float(unchanged["reference_sealed"])
        and _at(metrics, "latent_gauge/reference_update", target)
        == float(unchanged["reference_update"])
        and all(
            finite(_at(metrics, tag, target))
            and float(_at(metrics, tag, target) or 0.0)
            >= float(scientific["latent_gauge_min_reference_scale"])
            for tag in ("latent_gauge/root/reference", "latent_gauge/future/reference")
        )
    )
    gauge_absolute = (
        gauge_series is not None
        and ratio_consistent
        and reference_valid
        and min(gauge_series) >= float(unchanged["min_recent_latent_gauge_ratio"])
        and float(minimum_ratio or 0.0)
        >= float(unchanged["min_recent_latent_gauge_ratio"])
    )

    validation_axis = _axis(target, validation_cadence)
    val_total = _complete(metrics, "val/loss_total", validation_axis)
    val_self_fed = _complete(metrics, "val/loss_multistep_self_fed", validation_axis)
    fixed_counts = _complete(metrics, "data/validation_fixed_sample_count", validation_axis)
    fixed_validation = bool(
        fixed_counts
        and all(_close(float(value), 5120.0, atol=1e-9) for value in fixed_counts)
        and len(set(fixed_counts)) == 1
    )
    validation_stable = bool(
        val_total
        and all(value >= 0.0 for value in val_total)
        and val_total[-1]
        <= min(val_total) * (1.0 + float(unchanged["max_validation_regret_fraction"]))
    )
    self_fed_stable = bool(
        val_self_fed
        and all(value >= 0.0 for value in val_self_fed)
        and val_self_fed[-1]
        <= min(val_self_fed)
        * (1.0 + float(unchanged["max_self_fed_validation_regret_fraction"]))
    )

    fractions = [_at(metrics, f"data/validation_horizon_label_fraction_h{h}", target) for h in HORIZONS]
    horizon_distribution = all(
        finite(value) and float(value) >= 0.0 for value in fractions
    ) and abs(sum(float(value) for value in fractions) - 1.0) <= float(
        unchanged["horizon_label_fraction_sum_tolerance"]
    )
    empirical_entropy = (
        -sum(float(value) * math.log(max(float(value), 1e-12)) for value in fractions)
        if horizon_distribution
        else None
    )
    horizon_loss = _at(metrics, "val/loss_horizon", target)
    require_empirical_prior = bool(
        unchanged["horizon_cross_entropy_must_also_be_below_empirical_prior_entropy"]
    )
    horizon_pass = (
        finite(horizon_loss)
        and finite(empirical_entropy)
        and float(horizon_loss) < float(unchanged["max_horizon_cross_entropy"])
        and (
            not require_empirical_prior
            or float(horizon_loss) < float(empirical_entropy)
        )
    )
    q_pass = (
        _at(metrics, "control/retrieval_uses_task_metric_endpoint", target) == 1.0
        and all(
            finite(_at(metrics, tag, target))
            and float(_at(metrics, tag, target) or 0.0)
            > float(unchanged["min_q_advantage"])
            for tag in ("control/q_advantage_over_z", "control/q_advantage_over_random_proj")
        )
    )
    gain_axis = _axis(
        target, validation_cadence, int(unchanged["gain_recent_window_updates"])
    )
    gain_series = {tag: _complete(metrics, tag, gain_axis) for tag in GAIN_TAGS}
    gain_means = {
        tag: statistics.fmean(values) if values else None for tag, values in gain_series.items()
    }
    gain_pass = (
        all(values is not None for values in gain_series.values())
        and float(gain_means[GAIN_TAGS[0]] or -math.inf)
        >= float(unchanged["min_gain_rank_correlation"])
        and float(gain_means[GAIN_TAGS[1]] or -math.inf)
        >= float(unchanged["min_gain_pairwise_accuracy"])
        and float(gain_means[GAIN_TAGS[2]] or -math.inf)
        >= float(unchanged["min_gain_eligible_decision_fraction"])
        and float(gain_means[GAIN_TAGS[3]] or -math.inf)
        >= float(unchanged["min_gain_ordered_pair_count"])
        and float(gain_means[GAIN_TAGS[4]] or -math.inf)
        >= float(unchanged["min_gain_pair_coverage_fraction"])
    )
    support_pass = (
        finite(_at(metrics, "tree/support_recall", target))
        and float(_at(metrics, "tree/support_recall", target) or 0.0)
        >= float(unchanged["min_support_recall"])
        and finite(_at(metrics, "tree/support_precision", target))
        and float(_at(metrics, "tree/support_precision", target) or 0.0)
        >= float(unchanged["min_support_precision"])
    )

    structural_gates = {
        "required_exact_boundary_telemetry_finite": required_exact_finite,
        "fixed_validation_axis_and_count": fixed_validation,
        "complete_recent_gradient_axis": complete_gradient_axis,
        "nonzero_world_gain_and_split_gradients": nonzero_gradients,
        "gradient_clip_coefficients_in_0_to_1": valid_clip_coefficients,
        "complete_recent_gauge_axis": gauge_series is not None,
        "gauge_reference_sealed_at_update_zero": reference_valid,
        "gauge_ratio_consistent": ratio_consistent,
        "prefix_contract_exact": bool(prefix_contract["passed"]),
        "train_prefix_telemetry_structural": bool(train_prefix["passed"]),
        "fixed_validation_prefix_telemetry_structural": bool(
            validation_prefix["passed"]
        ),
        "train_validation_prefix_domain_parity": prefix_domain_parity,
        "generic_prefix_term_telemetry_exact": bool(prefix_terms["passed"]),
        "calibration_telemetry_structural": bool(calibration["structural_passed"]),
    }
    method_gates = {
        "validation_nonregression": validation_stable,
        "self_fed_multistep_validation_nonregression": self_fed_stable,
        "horizon_ce_below_uniform_and_empirical_prior": horizon_pass,
        "q_advantage": q_pass,
        "gain_rank_pair_eligibility_and_coverage": gain_pass,
        "support_recall_and_precision": support_pass,
    }
    candidate_gates = {
        "structural": all(structural_gates.values()),
        "bounded_branch_rest_and_gain_clipping": bounded_clipping,
        "absolute_gauge_retention": gauge_absolute,
        "unchanged_method_gates": all(method_gates.values()),
        "absolute_action_h4_calibration": bool(calibration["absolute_passed"]),
    }
    return {
        "structural_passed": all(structural_gates.values()),
        "candidate_passed": arm_id == "GSEP" and all(candidate_gates.values()),
        "structural_gates": structural_gates,
        "method_gates": method_gates,
        "candidate_gates": candidate_gates,
        "prefix_contract": prefix_contract,
        "train_prefix_telemetry": train_prefix,
        "validation_prefix_telemetry": validation_prefix,
        "prefix_term_telemetry": prefix_terms,
        "calibration": calibration,
        "recent_gauge_min_ratio": min(gauge_series) if gauge_series else None,
        "clip_fraction_below_0_05_by_tag": low_clip_fractions,
        "gain_recent_means": gain_means,
        "horizon_empirical_prior_entropy": empirical_entropy,
    }


def _outcome(
    value: object, *, cell_label: str, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{cell_label}: 25k outcome must be an object")
    task_ids = [int(item) for item in manifest["scientific_contract"]["task_ids"]]
    episodes_per_task = int(
        manifest["scientific_contract"]["final_episodes_per_task"]
    )
    expected_episodes = len(task_ids) * episodes_per_task
    seed_base = int(manifest["scientific_contract"]["evaluation_seed"])
    raw_rows = value.get("completed_results")
    _require(isinstance(raw_rows, list), f"{cell_label}: final completed_results missing")
    expected_keys = [
        (task_index, task_id, episode_index, seed_base + 1000 * task_index + episode_index)
        for task_index, task_id in enumerate(task_ids)
        for episode_index in range(episodes_per_task)
    ]
    projected: list[tuple[int, int, int, int]] = []
    row_values_valid = len(raw_rows) == expected_episodes
    row_schema_valid = len(raw_rows) == expected_episodes
    row_successes: list[float] = []
    row_progress: list[float] = []
    for row in raw_rows:
        if not isinstance(row, Mapping):
            row_values_valid = False
            row_schema_valid = False
            continue
        if not _episode_result_row_valid(row):
            row_values_valid = False
            row_schema_valid = False
            continue
        identity_fields = (
            "task_index",
            "task_id",
            "episode_index",
            "episode_seed",
        )
        if not all(type(row.get(name)) is int for name in identity_fields):
            row_values_valid = False
            continue
        try:
            projected.append(
                (
                    row["task_index"],
                    row["task_id"],
                    row["episode_index"],
                    row["episode_seed"],
                )
            )
        except (KeyError, TypeError, ValueError):
            row_values_valid = False
            continue
        success = row.get("success")
        initial = row.get("initial_goal_distance")
        final = row.get("final_goal_distance")
        if (
            not isinstance(success, bool)
            or not finite(initial)
            or not finite(final)
            or float(initial) < 0.0
            or float(final) < 0.0
        ):
            row_values_valid = False
            continue
        row_successes.append(float(success))
        row_progress.append(
            (float(initial) - float(final)) / max(float(initial), 1e-6)
        )
    episodes = value.get("num_episodes")
    successes = value.get("successes")
    success_rate = value.get("success_rate")
    progress = value.get("distance_reduction_frac")
    recomputed_successes = sum(row_successes)
    recomputed_progress = (
        statistics.fmean(row_progress) if len(row_progress) == expected_episodes else None
    )
    completed_results_sha256 = value.get("completed_results_sha256")
    passed = bool(
        value.get("source") == "terminal_final_evaluation"
        and value.get("status") == "complete"
        and value.get("task_ids") == task_ids
        and value.get("episodes_per_task") == episodes_per_task
        and row_values_valid
        and projected == expected_keys
        and sha256_string(completed_results_sha256)
        and completed_results_sha256 == stable_sha256(raw_rows)
        and all(
            sha256_string(value.get(key))
            for key in (
                "completion_sha256",
                "final_eval_progress_sha256",
                "checkpoint_sha256",
            )
        )
        and
        finite(episodes)
        and float(episodes) == float(expected_episodes)
        and finite(successes)
        and float(successes).is_integer()
        and 0.0 <= float(successes) <= float(expected_episodes)
        and _close(float(successes), recomputed_successes, atol=1e-9)
        and finite(success_rate)
        and _close(
            float(success_rate),
            float(successes) / float(expected_episodes),
            atol=1e-6,
        )
        and finite(progress)
        and recomputed_progress is not None
        and _close(float(progress), float(recomputed_progress), atol=1e-6)
    )
    return {
        "passed": passed,
        "source": value.get("source"),
        "num_episodes": float(episodes) if finite(episodes) else None,
        "successes": float(successes) if finite(successes) else None,
        "success_rate": float(success_rate) if finite(success_rate) else None,
        "distance_reduction_frac": float(progress) if finite(progress) else None,
        "completed_results_sha256": completed_results_sha256,
        "episode_result_schema_exact": row_schema_valid,
        "fallback_seed_rows_exact": projected == expected_keys,
        "task_episode_coverage_exact": len(projected) == len(set(projected)) == expected_episodes,
    }


def _paired_calibration(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    by_key = {
        (row["setting_id"], int(row["seed"]), row["arm_id"]): row for row in rows
    }
    def metric(row: Mapping[str, Any], suffix: str) -> float:
        return float(
            row["boundaries"]["25000"]["calibration"]["values"][suffix]
        )

    def summary(row: Mapping[str, Any], key: str) -> float:
        return float(row["boundaries"]["25000"]["calibration"][key])

    required_suffixes = (
        "predicted_vs_actual_normalized_task_rms",
        "action_clipped_fraction",
    )
    values_finite = all(
        all(
            finite(
                row["boundaries"]["25000"]["calibration"]["values"].get(
                    suffix
                )
            )
            for suffix in required_suffixes
        )
        and finite(
            row["boundaries"]["25000"]["calibration"].get(
                "action_distortion"
            )
        )
        for row in rows
    )
    if not values_finite:
        gates = {
            "all_paired_calibration_values_finite": False,
            "gsep_macro_endpoint_error_strictly_lower": False,
            "at_least_three_settings_lower_endpoint_error": False,
            "gsep_macro_action_distortion_not_higher": False,
            "gsep_macro_action_clip_fraction_not_higher": False,
        }
        return {
            "passed": False,
            "gates": gates,
            "macro_normalized_endpoint_error": {},
            "macro_action_distortion": {},
            "macro_action_clip_fraction": {},
            "settings_with_strictly_lower_two_seed_endpoint_error": 0,
            "per_setting": {},
        }

    endpoint = {
        arm: [
            metric(
                by_key[(setting, seed, arm)],
                "predicted_vs_actual_normalized_task_rms",
            )
            for setting in SETTING_IDS
            for seed in SEEDS
        ]
        for arm in ARM_IDS
    }
    distortion = {
        arm: [
            summary(by_key[(setting, seed, arm)], "action_distortion")
            for setting in SETTING_IDS
            for seed in SEEDS
        ]
        for arm in ARM_IDS
    }
    clipped = {
        arm: [
            metric(by_key[(setting, seed, arm)], "action_clipped_fraction")
            for setting in SETTING_IDS
            for seed in SEEDS
        ]
        for arm in ARM_IDS
    }
    per_setting: dict[str, Any] = {}
    lower_settings = 0
    for setting in SETTING_IDS:
        gs = statistics.fmean(
            metric(
                by_key[(setting, seed, "GS")],
                "predicted_vs_actual_normalized_task_rms",
            )
            for seed in SEEDS
        )
        gsep = statistics.fmean(
            metric(
                by_key[(setting, seed, "GSEP")],
                "predicted_vs_actual_normalized_task_rms",
            )
            for seed in SEEDS
        )
        lower = gsep < gs
        lower_settings += int(lower)
        per_setting[setting] = {
            "gs_two_seed_mean_normalized_endpoint_error": gs,
            "gsep_two_seed_mean_normalized_endpoint_error": gsep,
            "gsep_strictly_lower": lower,
        }
    macro_endpoint = {arm: statistics.fmean(values) for arm, values in endpoint.items()}
    macro_distortion = {arm: statistics.fmean(values) for arm, values in distortion.items()}
    macro_clipped = {arm: statistics.fmean(values) for arm, values in clipped.items()}
    paired_gate = manifest["acceptance"]["paired_h4_calibration_gates_at_25k"]
    gates = {
        "all_paired_calibration_values_finite": True,
        "gsep_macro_endpoint_error_strictly_lower": macro_endpoint["GSEP"]
        < macro_endpoint["GS"],
        "at_least_three_settings_lower_endpoint_error": lower_settings
        >= int(paired_gate["min_settings_candidate_two_seed_endpoint_error_below_control"]),
        "gsep_macro_action_distortion_not_higher": macro_distortion["GSEP"]
        <= macro_distortion["GS"],
        "gsep_macro_action_clip_fraction_not_higher": macro_clipped["GSEP"]
        <= macro_clipped["GS"],
    }
    return {
        "passed": all(gates.values()),
        "gates": gates,
        "macro_normalized_endpoint_error": macro_endpoint,
        "macro_action_distortion": macro_distortion,
        "macro_action_clip_fraction": macro_clipped,
        "settings_with_strictly_lower_two_seed_endpoint_error": lower_settings,
        "per_setting": per_setting,
    }


def _outcome_gates(
    rows: Sequence[Mapping[str, Any]], manifest: Mapping[str, Any]
) -> dict[str, Any]:
    by_key = {
        (row["setting_id"], int(row["seed"]), row["arm_id"]): row for row in rows
    }
    structurally_valid = all(bool(row["outcome"]["passed"]) for row in rows)
    absolute_thresholds = manifest["acceptance"]["absolute_end_to_end_gates_at_25k"]
    paired_thresholds = manifest["acceptance"]["paired_end_to_end_gates_at_25k"]
    if not structurally_valid:
        absolute_gates = {
            "all_twenty_outcomes_structurally_valid": False,
            "each_candidate_seed_has_success_and_positive_mean_progress": False,
            "at_least_one_setting_has_success_both_seeds": False,
            "at_least_three_settings_have_positive_progress_both_seeds": False,
        }
        paired_gates = {
            "gsep_macro_success_not_worse": False,
            "gsep_macro_distance_reduction_not_worse": False,
            "at_least_one_macro_outcome_strictly_better": False,
            "at_least_three_settings_distance_reduction_not_worse": False,
        }
        return {
            "passed": False,
            "absolute_gates": absolute_gates,
            "paired_gates": paired_gates,
            "candidate_per_seed": {},
            "candidate_per_setting": {},
            "macro_success_rate": {},
            "macro_distance_reduction_frac": {},
            "settings_with_gsep_distance_reduction_not_worse": 0,
            "paired_per_setting": {},
        }
    absolute_per_seed: dict[str, Any] = {}
    absolute_seed_pass = True
    for seed in SEEDS:
        candidate = [by_key[(setting, seed, "GSEP")]["outcome"] for setting in SETTING_IDS]
        successes = sum(float(row["successes"] or 0.0) for row in candidate)
        progress = statistics.fmean(float(row["distance_reduction_frac"]) for row in candidate)
        passed = successes >= float(
            absolute_thresholds["min_total_successes_per_seed_across_settings"]
        ) and progress > float(absolute_thresholds["min_mean_distance_reduction_per_seed"])
        absolute_seed_pass &= passed
        absolute_per_seed[str(seed)] = {
            "total_successes": successes,
            "mean_distance_reduction_frac": progress,
            "passed": passed,
        }

    both_success = 0
    both_progress = 0
    absolute_per_setting: dict[str, Any] = {}
    paired_progress_settings = 0
    paired_per_setting: dict[str, Any] = {}
    for setting in SETTING_IDS:
        candidate = [by_key[(setting, seed, "GSEP")]["outcome"] for seed in SEEDS]
        replicated_success = all(float(row["successes"] or 0.0) > 0.0 for row in candidate)
        replicated_progress = all(float(row["distance_reduction_frac"]) > 0.0 for row in candidate)
        both_success += int(replicated_success)
        both_progress += int(replicated_progress)
        absolute_per_setting[setting] = {
            "both_seed_nonzero_success": replicated_success,
            "both_seed_positive_progress": replicated_progress,
        }
        means = {
            arm: statistics.fmean(
                float(by_key[(setting, seed, arm)]["outcome"]["distance_reduction_frac"])
                for seed in SEEDS
            )
            for arm in ARM_IDS
        }
        not_worse = means["GSEP"] >= means["GS"]
        paired_progress_settings += int(not_worse)
        paired_per_setting[setting] = {
            "two_seed_mean_distance_reduction_frac": means,
            "gsep_not_worse": not_worse,
        }

    macro_success = {
        arm: statistics.fmean(
            float(by_key[(setting, seed, arm)]["outcome"]["success_rate"])
            for setting in SETTING_IDS
            for seed in SEEDS
        )
        for arm in ARM_IDS
    }
    macro_progress = {
        arm: statistics.fmean(
            float(by_key[(setting, seed, arm)]["outcome"]["distance_reduction_frac"])
            for setting in SETTING_IDS
            for seed in SEEDS
        )
        for arm in ARM_IDS
    }
    absolute_gates = {
        "all_twenty_outcomes_structurally_valid": structurally_valid,
        "each_candidate_seed_has_success_and_positive_mean_progress": absolute_seed_pass,
        "at_least_one_setting_has_success_both_seeds": both_success
        >= int(absolute_thresholds["min_settings_with_success_for_both_seeds"]),
        "at_least_three_settings_have_positive_progress_both_seeds": both_progress
        >= int(
            absolute_thresholds[
                "min_settings_with_positive_progress_for_both_seeds"
            ]
        ),
    }
    paired_gates = {
        "gsep_macro_success_not_worse": macro_success["GSEP"] >= macro_success["GS"],
        "gsep_macro_distance_reduction_not_worse": macro_progress["GSEP"]
        >= macro_progress["GS"],
        "at_least_one_macro_outcome_strictly_better": macro_success["GSEP"]
        > macro_success["GS"]
        or macro_progress["GSEP"] > macro_progress["GS"],
        "at_least_three_settings_distance_reduction_not_worse": paired_progress_settings
        >= int(
            paired_thresholds[
                "min_settings_candidate_two_seed_distance_reduction_not_below_control"
            ]
        ),
    }
    return {
        "passed": all(absolute_gates.values()) and all(paired_gates.values()),
        "absolute_gates": absolute_gates,
        "paired_gates": paired_gates,
        "candidate_per_seed": absolute_per_seed,
        "candidate_per_setting": absolute_per_setting,
        "macro_success_rate": macro_success,
        "macro_distance_reduction_frac": macro_progress,
        "settings_with_gsep_distance_reduction_not_worse": paired_progress_settings,
        "paired_per_setting": paired_per_setting,
    }


def evaluate_bundle(
    bundle: Mapping[str, Any], manifest: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    try:
        report_bundle_sha256 = stable_sha256(bundle)
    except (TypeError, ValueError) as exc:
        raise GateContractError("report bundle contains non-JSON or nonfinite values") from exc
    manifest = load_manifest() if manifest is None else dict(manifest)
    # Validate a caller-supplied manifest through the same frozen checks without
    # silently falling back to package defaults.
    _require(manifest.get("schema_version") == 1, "manifest schema differs")
    _require(manifest.get("campaign_id") == CAMPAIGN_ID, "manifest campaign differs")
    _require(manifest.get("design", {}).get("settings") == list(SETTING_IDS), "manifest settings differ")
    _require(manifest.get("design", {}).get("arms") == list(ARM_IDS), "manifest arms differ")
    _require(manifest.get("design", {}).get("seeds") == list(SEEDS), "manifest seeds differ")
    _require(
        manifest.get("design", {}).get("analysis_boundaries") == list(BOUNDARIES),
        "manifest boundaries differ",
    )
    _require(isinstance(bundle, Mapping), "report bundle must be an object")
    binding = package_binding(manifest)
    _require(bundle.get("schema_version") == SCHEMA_VERSION, "report schema_version differs")
    _require(bundle.get("campaign_id") == CAMPAIGN_ID, "campaign_id differs")
    for name, expected in binding.items():
        _require(
            bundle.get(name) == expected,
            f"report {name} differs from frozen package",
        )
    cells = bundle.get("cells")
    _require(isinstance(cells, list), "cells must be a list")
    _require(len(cells) == 20, "report must contain exactly 20 cells")

    expected_keys = {
        (setting, arm, seed)
        for setting in SETTING_IDS
        for arm in ARM_IDS
        for seed in SEEDS
    }
    seen: set[tuple[str, str, int]] = set()
    rows: list[dict[str, Any]] = []
    source_cells: dict[tuple[str, str, int], Mapping[str, Any]] = {}
    for raw_cell in cells:
        _require(isinstance(raw_cell, Mapping), "each cell must be an object")
        setting = raw_cell.get("setting_id")
        arm = raw_cell.get("arm_id")
        seed = raw_cell.get("seed")
        _require(type(setting) is str and setting in SETTING_IDS, f"invalid setting_id: {setting!r}")
        _require(type(arm) is str and arm in ARM_IDS, f"invalid arm_id: {arm!r}")
        _require(type(seed) is int and seed in SEEDS, f"invalid seed: {seed!r}")
        key = (setting, arm, seed)
        _require(key not in seen, f"duplicate cell: {key}")
        seen.add(key)
        source_cells[key] = raw_cell
        index = expected_index(*key)
        _require(type(raw_cell.get("index")) is int and raw_cell.get("index") == index, f"{key}: index differs from frozen mapping")
        _require(raw_cell.get("fresh_start") is True, f"{key}: fresh_start must be true")
        boundaries = raw_cell.get("boundaries")
        _require(isinstance(boundaries, Mapping), f"{key}: boundaries missing")
        _require(set(boundaries) == {str(item) for item in BOUNDARIES}, f"{key}: boundaries must be exactly 5000 and 25000")
        label = f"{setting}/{arm}/seed{seed}"
        boundary_results = {
            str(target): evaluate_boundary(
                boundaries[str(target)],
                cell_label=label,
                target=target,
                arm_id=str(arm),
                setting_id=str(setting),
                manifest=manifest,
            )
            for target in BOUNDARIES
        }
        outcome = _outcome(
            boundaries["25000"].get("outcome"),
            cell_label=label,
            manifest=manifest,
        )
        rows.append(
            {
                "index": index,
                "setting_id": setting,
                "arm_id": arm,
                "seed": seed,
                "boundaries": boundary_results,
                "outcome": outcome,
            }
        )
    _require(seen == expected_keys, "20-cell setting/arm/seed matrix differs")
    rows.sort(key=lambda row: int(row["index"]))

    parity_lock = load_causal_parity_lock()
    common_validation = True
    for setting in SETTING_IDS:
        hashes = {
            stable_sha256(
                source_cells[(setting, arm, seed)]["boundaries"][str(target)][
                    "prefix_contract"
                ]["fixed_validation_sampler"]
            )
            for arm in ARM_IDS
            for seed in SEEDS
            for target in BOUNDARIES
        }
        common_validation &= len(hashes) == 1

    control_structural = all(
        row["boundaries"][str(target)]["structural_passed"]
        for row in rows
        if row["arm_id"] == "GS"
        for target in BOUNDARIES
    )
    candidate_universal = all(
        row["boundaries"][str(target)]["candidate_passed"]
        for row in rows
        if row["arm_id"] == "GSEP"
        for target in BOUNDARIES
    )
    paired_calibration = _paired_calibration(rows, manifest)
    outcomes = _outcome_gates(rows, manifest)
    top_gates = {
        "frozen_outcome_blind_causal_parity_audit_bound": parity_lock[
            "artifact_sha256"
        ]
        == manifest["causal_parity_contract"]["artifact_sha256"],
        "fixed_common_validation_samples": common_validation,
        "all_gs_cells_structurally_valid_at_both_boundaries": control_structural,
        "all_gsep_cells_pass_unchanged_and_absolute_gates_at_both_boundaries": candidate_universal,
        "paired_25k_calibration": bool(paired_calibration["passed"]),
        "absolute_and_paired_25k_outcomes": bool(outcomes["passed"]),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "campaign_id": CAMPAIGN_ID,
        "status": "accepted_engineering_pilot" if all(top_gates.values()) else "rejected",
        "formal_validation": False,
        "supports_1m_claim": False,
        "report_bundle_sha256": report_bundle_sha256,
        "package_binding": binding,
        "manifest_sha256": binding["manifest_sha256"],
        "matrix": {
            "settings": list(SETTING_IDS),
            "arms": list(ARM_IDS),
            "seeds": list(SEEDS),
            "boundaries": list(BOUNDARIES),
            "cells": len(rows),
        },
        "gates": top_gates,
        "causal_parity_audit": {
            "artifact_sha256": parity_lock["artifact_sha256"],
            "pairs": len(parity_lock["pairs"]),
            "status": parity_lock["status"],
        },
        "paired_25k_calibration": paired_calibration,
        "outcomes_25000": outcomes,
        "cells": rows,
    }
    result["gate_sha256"] = stable_sha256(result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="immutable report-bundle JSON")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        bundle = json.loads(args.report.read_text(encoding="utf-8"))
        result = evaluate_bundle(bundle, load_manifest(args.manifest))
    except (OSError, json.JSONDecodeError, GateContractError, ValueError) as exc:
        print(f"exp23 gate contract error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0 if result["status"] == "accepted_engineering_pilot" else 1


if __name__ == "__main__":
    raise SystemExit(main())
