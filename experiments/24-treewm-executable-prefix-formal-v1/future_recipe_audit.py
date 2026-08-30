#!/usr/bin/env python3
"""Audit all-ten published future recipes and every immutable dependency byte.

The authority covers each composite and child manifest, both complete records.npy
files, the ten calibration contracts and their all-settings summary, plus the declared
historical recipe code/runtime identities.  It performs no recipe build and no write.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import data_authority_common as common
import input_contract_audit


AUDIT_ID = "treewm_exp24_future_recipe_authority_v1"
STATUS = "sealed_outcome_blind_all_ten_future_recipe_authority"
COMPOSITE_KEYS = {
    "calibration_sha256", "chosen_thresholds", "code_sha256", "normalizer_sha256",
    "recipe_sha256", "recipe_version", "runtime_sha256", "schema_version",
    "source_manifest_sha256", "status", "train_manifest", "train_manifest_sha256",
    "train_recipe_sha256", "validation_manifest", "validation_manifest_sha256",
    "validation_recipe_sha256",
}
CHILD_KEYS = {
    "identity", "identity_sha256", "recipe_sha256", "record_count", "record_dtype",
    "records_file", "records_mtime_ns", "records_sha256", "records_size",
    "schema_version", "status",
}
IDENTITY_KEYS = {
    "anchor_sets", "anchor_union_count", "anchor_union_sha256", "calibration_sha256",
    "chosen_thresholds", "code_sha256", "future_config", "normalizer_sha256",
    "numpy_version", "recipe_version", "runtime_sha256", "schema_version",
    "scipy_version", "source_manifest_sha256", "split", "split_manifest_sha256",
    "task_metric_dims", "xy_dims",
}
FUTURE_CONFIG_KEYS = {
    "cluster_method", "cluster_threshold", "displacement_threshold", "fixed_horizon",
    "h_max", "horizon_rule", "horizons", "include_self", "max_modes",
    "metric_mode", "multi_step_depth", "num_neighbors", "query_multiplier",
    "relative_endpoints", "retrieval_pool", "retrieval_radius", "time_exclusion",
}
CALIBRATION_KEYS = {
    "algorithm_version", "anchor_population", "chosen", "cluster", "config",
    "contract_sha256", "eligible_query_count_histogram", "gates", "horizon",
    "implementation_sha256", "input_units", "metric_mode", "normalizer_sha256",
    "numpy_version", "relative_endpoints", "retrieval", "retrieval_pool_indices",
    "retrieval_pool_indices_sha256", "sample_indices", "sample_indices_sha256",
    "sample_seed", "schema_version", "scipy_version", "selected_continuation_count",
    "selected_continuation_indices", "selected_continuation_indices_sha256",
    "selected_continuation_offsets", "setting_id", "split", "status",
    "task_metric_dims", "train_manifest_sha256", "xy_dims",
}
CALIBRATION_CONFIG_KEYS = {
    "anchor_sample_seed", "cluster_candidate_quantiles", "cluster_method",
    "fallback_radius_quantile", "horizon_candidate_quantiles", "horizons",
    "max_insufficient_neighbor_fraction", "max_mean_retained_modes", "max_modes",
    "max_retrieval_fallback_fraction", "max_truncation_fraction",
    "min_mean_retained_modes", "min_mean_retrieved", "min_multimode_anchor_fraction",
    "min_normalized_horizon_entropy", "min_occupied_horizon_classes", "num_neighbors",
    "query_multiplier", "radius_quantile", "radius_report_quantiles", "retrieval_pool",
    "retrieval_pool_seed", "sample_size", "time_exclusion",
}
GATE_KEYS = {
    "eligible_23rd_neighbor", "mean_retained_modes", "mean_retrieved",
    "mode_truncation_fraction", "multimode_anchor_fraction",
    "normalized_horizon_entropy", "occupied_horizon_classes",
    "retrieval_fallback_fraction",
}
SUMMARY_ROW_KEYS = {
    "calibration_sha256", "chosen", "gates", "horizon_histogram",
    "raw_mode_histogram", "retained_mode_histogram", "retrieval", "setting_id",
}
SUMMARY_RETRIEVAL_KEYS = {
    "fallback_fraction", "max_retrieved", "mean_retrieved", "min_retrieved",
    "radius", "retrieved_histogram",
}
CLUSTER_KEYS = {
    "candidate_quantiles", "candidates", "chosen_index", "chosen_raw_mode_histogram",
    "chosen_retained_mode_histogram", "chosen_threshold",
    "had_truncation_feasible_candidate", "mean_retained_modes", "median_retained_modes",
    "multimode_anchor_fraction", "quantile_method", "rule", "truncation_fraction",
}
CLUSTER_CANDIDATE_KEYS = {
    "mean_raw_modes", "mean_retained_modes", "median_retained_modes",
    "multimode_anchor_fraction", "raw_mode_histogram", "retained_mode_histogram",
    "threshold", "truncation_fraction",
}
HORIZON_KEYS = {
    "candidate_quantiles", "candidates", "chosen_histogram", "chosen_index",
    "chosen_threshold", "normalized_entropy", "occupied_classes", "quantile_method",
    "rule", "threshold_boundary",
}
HORIZON_CANDIDATE_KEYS = {
    "histogram", "max_class_fraction", "normalized_entropy", "occupied_classes",
    "threshold",
}
RETRIEVAL_KEYS = {
    "chosen", "chosen_radius", "fallback_component", "insufficient_anchor_count",
    "insufficient_anchor_fraction", "kth_distance_count", "kth_distance_sha256",
    "maximum_insufficient_anchor_fraction", "quantile_method", "query_k",
    "report_candidates", "required_nonself_neighbors", "rule", "sufficient_anchor_count",
    "support_count_component",
}
RETRIEVAL_CHOSEN_KEYS = SUMMARY_RETRIEVAL_KEYS
RADIUS_COMPONENT_KEYS = {
    "distance_count", "distances_sha256", "order_statistic", "quantile", "radius",
}
RETRIEVAL_REPORT_KEYS = SUMMARY_RETRIEVAL_KEYS | {"quantile"}

# The finalized formal-v2 calibration protocol is data authority, not advisory
# metadata.  Preserve the JSON scalar types as well as the values: in particular,
# counts/seeds are integers and the registered numeric gates are floats.
FORMAL_CALIBRATION_CONFIG = {
    "anchor_sample_seed": 0,
    "cluster_candidate_quantiles": [i / 100.0 for i in range(1, 100)],
    "cluster_method": "average",
    "fallback_radius_quantile": 0.99,
    "horizon_candidate_quantiles": [i / 100.0 for i in range(101)],
    "horizons": [4, 8, 16, 32, 64],
    "max_insufficient_neighbor_fraction": 0.10,
    "max_mean_retained_modes": 3.5,
    "max_modes": 4,
    "max_retrieval_fallback_fraction": 0.01,
    "max_truncation_fraction": 0.05,
    "min_mean_retained_modes": 1.5,
    "min_mean_retrieved": 18.0,
    "min_multimode_anchor_fraction": 0.40,
    "min_normalized_horizon_entropy": 0.65,
    "min_occupied_horizon_classes": 4,
    "num_neighbors": 24,
    "query_multiplier": 6,
    "radius_quantile": 0.90,
    "radius_report_quantiles": [0.50, 0.75, 0.90, 0.95],
    "retrieval_pool": 50_000,
    "retrieval_pool_seed": 0,
    "sample_size": 4096,
    "time_exclusion": 50,
}
CLUSTER_RULE = (
    "if any candidate has raw_modes>max_modes fraction <=0.05, minimize "
    "|mean retained-3|, truncation, |median retained-3|, then prefer larger "
    "threshold; otherwise minimize truncation before the same tie-breaks"
)
HORIZON_RULE = (
    "maximize normalized Shannon entropy; tie: occupied classes descending, "
    "max-class fraction ascending, threshold ascending"
)
HORIZON_BOUNDARY = "select first feasible horizon with displacement > threshold"


def expected_record_dtype() -> list[list[Any]]:
    return [
        ["anchor", "<i8"],
        ["neighbors", "<i8", [24]],
        ["horizon_idx", "|u1", [24]],
        ["fut_valid", "|u1", [24]],
        ["cluster", "|i1", [24]],
        ["mode_rep", "|i1", [4]],
        ["mode_mass", "<f4", [4]],
        ["mode_valid", "|u1", [4]],
        ["ms_horizon_idx", "|u1", [3]],
        ["ms_valid", "|u1", [3]],
        ["future_diversity", "<f4"],
        ["num_retrieved", "|u1"],
        ["retrieval_num_candidates", "<u2"],
        ["retrieval_num_valid", "|u1"],
        ["retrieval_mean_distance", "<f4"],
        ["retrieval_fallback", "|u1"],
        ["retrieval_truncated", "|u1"],
        ["retrieval_query_saturated", "|u1"],
        ["modes_raw", "|u1"],
        ["modes_retained", "|u1"],
        ["modes_truncated", "|u1"],
    ]


def _indices_sha256(values: object, label: str) -> str:
    import numpy as np

    integers = common.require_int_list(values, label)
    return hashlib.sha256(np.asarray(integers, dtype="<i8").tobytes(order="C")).hexdigest()


def _require_number_range(
    value: object,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    number = common.require_number(value, label)
    if minimum is not None:
        common.require(number >= minimum, f"{label} is below {minimum}")
    if maximum is not None:
        common.require(number <= maximum, f"{label} exceeds {maximum}")
    return number


def _validate_histogram(
    value: object,
    label: str,
    *,
    key_minimum: int | None = None,
    key_maximum: int | None = None,
    total: int | None = None,
) -> dict[str, int]:
    common.require(isinstance(value, dict) and bool(value), f"{label} is empty or not an object")
    observed_total = 0
    for key, count in value.items():
        common.require(isinstance(key, str) and key.isdigit() and str(int(key)) == key,
                       f"{label} has a non-canonical integer key")
        numeric_key = int(key)
        if key_minimum is not None:
            common.require(numeric_key >= key_minimum, f"{label} key {key} is below range")
        if key_maximum is not None:
            common.require(numeric_key <= key_maximum, f"{label} key {key} exceeds range")
        observed_total += common.require_int(count, f"{label}.{key}", minimum=0)
    if total is not None:
        common.require(observed_total == total, f"{label} count total differs")
    return value


def _validate_number_list(
    value: object,
    label: str,
    *,
    length: int | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
) -> list[Any]:
    common.require(isinstance(value, list), f"{label} is not an array")
    if length is not None:
        common.require(len(value) == length, f"{label} length differs")
    for index, item in enumerate(value):
        _require_number_range(
            item, f"{label}[{index}]", minimum=minimum, maximum=maximum
        )
    return value


def _validate_histogram_list(
    value: object,
    label: str,
    *,
    length: int,
    total: int,
) -> list[int]:
    values = common.require_int_list(value, label, minimum=0)
    common.require(len(values) == length, f"{label} length differs")
    common.require(sum(values) == total, f"{label} count total differs")
    return values


def _validate_retrieval_result(
    value: object,
    label: str,
    *,
    sample_size: int,
    num_neighbors: int,
    has_quantile: bool,
) -> Mapping[str, Any]:
    keys = RETRIEVAL_REPORT_KEYS if has_quantile else RETRIEVAL_CHOSEN_KEYS
    row = common.require_exact_keys(value, keys, label)
    minimum = common.require_int(row["min_retrieved"], f"{label}.min_retrieved",
                                 minimum=1, maximum=num_neighbors)
    maximum = common.require_int(row["max_retrieved"], f"{label}.max_retrieved",
                                 minimum=1, maximum=num_neighbors)
    common.require(minimum <= maximum, f"{label} retrieval extrema are reversed")
    mean = _require_number_range(row["mean_retrieved"], f"{label}.mean_retrieved",
                                 minimum=float(minimum), maximum=float(maximum))
    fallback = _require_number_range(row["fallback_fraction"], f"{label}.fallback_fraction",
                                     minimum=0.0, maximum=1.0)
    _require_number_range(row["radius"], f"{label}.radius", minimum=0.0)
    histogram = _validate_histogram(
        row["retrieved_histogram"], f"{label}.retrieved_histogram",
        key_minimum=1, key_maximum=num_neighbors, total=sample_size,
    )
    weighted = sum(int(key) * count for key, count in histogram.items()) / sample_size
    common.require(mean == weighted, f"{label}.mean_retrieved disagrees with histogram")
    common.require(minimum == min(int(key) for key in histogram),
                   f"{label}.min_retrieved disagrees with histogram")
    common.require(maximum == max(int(key) for key in histogram),
                   f"{label}.max_retrieved disagrees with histogram")
    common.require(fallback == histogram.get("1", 0) / sample_size,
                   f"{label}.fallback_fraction disagrees with histogram")
    if has_quantile:
        _require_number_range(row["quantile"], f"{label}.quantile",
                              minimum=0.0, maximum=1.0)
    return row


def _validate_candidate_structures(
    calibration: Mapping[str, Any],
    label: str,
    *,
    config: Mapping[str, Any],
    sample_size: int,
    selected_count: int,
) -> None:
    num_neighbors = config["num_neighbors"]
    max_modes = config["max_modes"]
    cluster = common.require_exact_keys(calibration["cluster"], CLUSTER_KEYS, f"{label}.cluster")
    _validate_number_list(cluster["candidate_quantiles"], f"{label}.cluster.candidate_quantiles",
                          minimum=0.0, maximum=1.0)
    common.require(common.canonical_json(cluster["candidate_quantiles"])
                   == common.canonical_json(config["cluster_candidate_quantiles"]),
                   f"{label}.cluster.candidate_quantiles differs from config")
    common.require(isinstance(cluster["candidates"], list) and cluster["candidates"],
                   f"{label}.cluster.candidates is empty")
    common.require(len(cluster["candidates"]) <= len(cluster["candidate_quantiles"]),
                   f"{label}.cluster has too many candidates")
    for index, value in enumerate(cluster["candidates"]):
        row = common.require_exact_keys(
            value, CLUSTER_CANDIDATE_KEYS, f"{label}.cluster.candidates[{index}]"
        )
        candidate_label = f"{label}.cluster.candidates[{index}]"
        _require_number_range(row["mean_raw_modes"], f"{candidate_label}.mean_raw_modes",
                              minimum=1.0, maximum=float(num_neighbors))
        _require_number_range(row["mean_retained_modes"], f"{candidate_label}.mean_retained_modes",
                              minimum=1.0, maximum=float(max_modes))
        _require_number_range(row["median_retained_modes"],
                              f"{candidate_label}.median_retained_modes",
                              minimum=1.0, maximum=float(max_modes))
        _require_number_range(row["multimode_anchor_fraction"],
                              f"{candidate_label}.multimode_anchor_fraction",
                              minimum=0.0, maximum=1.0)
        _require_number_range(row["threshold"], f"{candidate_label}.threshold", minimum=0.0)
        _require_number_range(row["truncation_fraction"],
                              f"{candidate_label}.truncation_fraction",
                              minimum=0.0, maximum=1.0)
        _validate_histogram(row["raw_mode_histogram"], f"{candidate_label}.raw_mode_histogram",
                            key_minimum=1, key_maximum=num_neighbors, total=sample_size)
        _validate_histogram(row["retained_mode_histogram"],
                            f"{candidate_label}.retained_mode_histogram",
                            key_minimum=1, key_maximum=max_modes, total=sample_size)
    chosen_index = common.require_int(
        cluster["chosen_index"], f"{label}.cluster.chosen_index",
        minimum=0, maximum=len(cluster["candidates"]) - 1,
    )
    _validate_histogram(cluster["chosen_raw_mode_histogram"],
                        f"{label}.cluster.chosen_raw_mode_histogram",
                        key_minimum=1, key_maximum=num_neighbors, total=sample_size)
    _validate_histogram(cluster["chosen_retained_mode_histogram"],
                        f"{label}.cluster.chosen_retained_mode_histogram",
                        key_minimum=1, key_maximum=max_modes, total=sample_size)
    common.require_bool(cluster["had_truncation_feasible_candidate"],
                        f"{label}.cluster.had_truncation_feasible_candidate")
    common.require(cluster["had_truncation_feasible_candidate"] is True,
                   f"{label}.cluster has no truncation-feasible candidate")
    common.require(cluster["quantile_method"] == "higher",
                   f"{label}.cluster.quantile_method differs")
    common.require(cluster["rule"] == CLUSTER_RULE, f"{label}.cluster.rule differs")
    selected_cluster = cluster["candidates"][chosen_index]
    expected_cluster_projection = {
        "chosen_threshold": selected_cluster["threshold"],
        "chosen_raw_mode_histogram": selected_cluster["raw_mode_histogram"],
        "chosen_retained_mode_histogram": selected_cluster["retained_mode_histogram"],
        "mean_retained_modes": selected_cluster["mean_retained_modes"],
        "median_retained_modes": selected_cluster["median_retained_modes"],
        "multimode_anchor_fraction": selected_cluster["multimode_anchor_fraction"],
        "truncation_fraction": selected_cluster["truncation_fraction"],
    }
    for key, expected in expected_cluster_projection.items():
        common.require(type(cluster[key]) is type(expected) and cluster[key] == expected,
                       f"{label}.cluster.{key} disagrees with chosen candidate")

    horizon = common.require_exact_keys(calibration["horizon"], HORIZON_KEYS,
                                        f"{label}.horizon")
    _validate_number_list(horizon["candidate_quantiles"], f"{label}.horizon.candidate_quantiles",
                          minimum=0.0, maximum=1.0)
    common.require(common.canonical_json(horizon["candidate_quantiles"])
                   == common.canonical_json(config["horizon_candidate_quantiles"]),
                   f"{label}.horizon.candidate_quantiles differs from config")
    common.require(isinstance(horizon["candidates"], list) and horizon["candidates"],
                   f"{label}.horizon.candidates is empty")
    common.require(len(horizon["candidates"]) <= len(horizon["candidate_quantiles"]),
                   f"{label}.horizon has too many candidates")
    for index, value in enumerate(horizon["candidates"]):
        row = common.require_exact_keys(
            value, HORIZON_CANDIDATE_KEYS, f"{label}.horizon.candidates[{index}]"
        )
        candidate_label = f"{label}.horizon.candidates[{index}]"
        histogram = _validate_histogram_list(
            row["histogram"], f"{candidate_label}.histogram", length=5, total=selected_count
        )
        occupied = common.require_int(row["occupied_classes"],
                                      f"{candidate_label}.occupied_classes", minimum=1, maximum=5)
        common.require(occupied == sum(count > 0 for count in histogram),
                       f"{candidate_label}.occupied_classes disagrees with histogram")
        fraction = _require_number_range(row["max_class_fraction"],
                                         f"{candidate_label}.max_class_fraction",
                                         minimum=0.0, maximum=1.0)
        common.require(fraction == max(histogram) / selected_count,
                       f"{candidate_label}.max_class_fraction disagrees with histogram")
        _require_number_range(row["normalized_entropy"], f"{candidate_label}.normalized_entropy",
                              minimum=0.0, maximum=1.0)
        _require_number_range(row["threshold"], f"{candidate_label}.threshold", minimum=0.0)
    chosen_index = common.require_int(
        horizon["chosen_index"], f"{label}.horizon.chosen_index",
        minimum=0, maximum=len(horizon["candidates"]) - 1,
    )
    _validate_histogram_list(horizon["chosen_histogram"],
                             f"{label}.horizon.chosen_histogram",
                             length=5, total=selected_count)
    common.require(horizon["quantile_method"] == "higher",
                   f"{label}.horizon.quantile_method differs")
    common.require(horizon["rule"] == HORIZON_RULE, f"{label}.horizon.rule differs")
    common.require(horizon["threshold_boundary"] == HORIZON_BOUNDARY,
                   f"{label}.horizon.threshold_boundary differs")
    selected_horizon = horizon["candidates"][chosen_index]
    expected_horizon_projection = {
        "chosen_threshold": selected_horizon["threshold"],
        "chosen_histogram": selected_horizon["histogram"],
        "normalized_entropy": selected_horizon["normalized_entropy"],
        "occupied_classes": selected_horizon["occupied_classes"],
    }
    for key, expected in expected_horizon_projection.items():
        common.require(type(horizon[key]) is type(expected) and horizon[key] == expected,
                       f"{label}.horizon.{key} disagrees with chosen candidate")

    retrieval = common.require_exact_keys(calibration["retrieval"], RETRIEVAL_KEYS,
                                          f"{label}.retrieval")
    required = config["num_neighbors"] - 1
    for component, expected_order, expected_quantile in (
        ("fallback_component", 1, config["fallback_radius_quantile"]),
        ("support_count_component", required, config["radius_quantile"]),
    ):
        row = common.require_exact_keys(retrieval[component], RADIUS_COMPONENT_KEYS,
                                        f"{label}.retrieval.{component}")
        common.require_int(row["distance_count"], f"{label}.retrieval.{component}.distance_count",
                           minimum=1, maximum=sample_size)
        common.require_int(row["order_statistic"],
                           f"{label}.retrieval.{component}.order_statistic",
                           minimum=1, maximum=required)
        common.require(row["order_statistic"] == expected_order,
                       f"{label}.retrieval.{component}.order_statistic differs")
        _require_number_range(row["quantile"], f"{label}.retrieval.{component}.quantile",
                              minimum=0.0, maximum=1.0)
        common.require(type(row["quantile"]) is type(expected_quantile)
                       and row["quantile"] == expected_quantile,
                       f"{label}.retrieval.{component}.quantile differs")
        _require_number_range(row["radius"], f"{label}.retrieval.{component}.radius",
                              minimum=0.0)
        common.require_sha256(row["distances_sha256"],
                              f"{label}.retrieval.{component}.distances_sha256")
    _validate_retrieval_result(
        retrieval["chosen"], f"{label}.retrieval.chosen",
        sample_size=sample_size, num_neighbors=num_neighbors, has_quantile=False,
    )
    common.require(isinstance(retrieval["report_candidates"], list)
                   and retrieval["report_candidates"],
                   f"{label}.retrieval.report_candidates is empty")
    common.require(len(retrieval["report_candidates"]) == len(config["radius_report_quantiles"]),
                   f"{label}.retrieval.report_candidates length differs")
    for index, value in enumerate(retrieval["report_candidates"]):
        row = _validate_retrieval_result(
            value, f"{label}.retrieval.report_candidates[{index}]",
            sample_size=sample_size, num_neighbors=num_neighbors, has_quantile=True,
        )
        expected_quantile = config["radius_report_quantiles"][index]
        common.require(type(row["quantile"]) is type(expected_quantile)
                       and row["quantile"] == expected_quantile,
                       f"{label}.retrieval.report_candidates[{index}].quantile differs")
    sufficient = common.require_int(retrieval["sufficient_anchor_count"],
                                    f"{label}.retrieval.sufficient_anchor_count",
                                    minimum=0, maximum=sample_size)
    insufficient = common.require_int(retrieval["insufficient_anchor_count"],
                                      f"{label}.retrieval.insufficient_anchor_count",
                                      minimum=0, maximum=sample_size)
    common.require(sufficient + insufficient == sample_size,
                   f"{label}.retrieval anchor counts differ")
    common.require_int(retrieval["kth_distance_count"],
                       f"{label}.retrieval.kth_distance_count",
                       minimum=1, maximum=sample_size)
    common.require(retrieval["kth_distance_count"] == sufficient
                   == retrieval["support_count_component"]["distance_count"],
                   f"{label}.retrieval support distance counts differ")
    common.require_int(retrieval["query_k"], f"{label}.retrieval.query_k", minimum=1)
    common.require(retrieval["query_k"] == num_neighbors * config["query_multiplier"] + 1,
                   f"{label}.retrieval.query_k differs")
    common.require_int(retrieval["required_nonself_neighbors"],
                       f"{label}.retrieval.required_nonself_neighbors", minimum=1)
    common.require(retrieval["required_nonself_neighbors"] == required,
                   f"{label}.retrieval.required_nonself_neighbors differs")
    insufficient_fraction = _require_number_range(
        retrieval["insufficient_anchor_fraction"],
        f"{label}.retrieval.insufficient_anchor_fraction", minimum=0.0, maximum=1.0,
    )
    common.require(abs(insufficient_fraction - (insufficient / sample_size)) <= 1e-15,
                   f"{label}.retrieval.insufficient_anchor_fraction disagrees with counts")
    maximum_fraction = _require_number_range(
        retrieval["maximum_insufficient_anchor_fraction"],
        f"{label}.retrieval.maximum_insufficient_anchor_fraction",
        minimum=0.0, maximum=1.0,
    )
    common.require(type(retrieval["maximum_insufficient_anchor_fraction"])
                   is type(config["max_insufficient_neighbor_fraction"])
                   and maximum_fraction == config["max_insufficient_neighbor_fraction"],
                   f"{label}.retrieval.maximum_insufficient_anchor_fraction differs")
    chosen_radius = _require_number_range(retrieval["chosen_radius"],
                                          f"{label}.retrieval.chosen_radius", minimum=0.0)
    common.require(chosen_radius == max(retrieval["fallback_component"]["radius"],
                                        retrieval["support_count_component"]["radius"]),
                   f"{label}.retrieval.chosen_radius disagrees with components")
    common.require(retrieval["chosen"]["radius"] == chosen_radius,
                   f"{label}.retrieval chosen radius differs")
    common.require_sha256(retrieval["kth_distance_sha256"],
                          f"{label}.retrieval.kth_distance_sha256")
    common.require(retrieval["quantile_method"] == "higher",
                   f"{label}.retrieval.quantile_method differs")
    expected_rule = (
        f"max(higher q={config['radius_quantile']:.2f} of the {required}th non-self "
        f"eligible neighbor RMS distance, higher q={config['fallback_radius_quantile']:.2f} "
        "of nearest eligible non-self RMS distance)"
    )
    common.require(retrieval["rule"] == expected_rule, f"{label}.retrieval.rule differs")


def _validate_calibration(
    value: object, setting: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    label = f"{setting['id']} calibration"
    calibration = dict(common.require_exact_keys(value, CALIBRATION_KEYS, label))
    common.require_int(calibration["schema_version"], f"{label}.schema_version")
    common.require(calibration["schema_version"] == 1, f"{label} schema differs")
    expected = {
        "status": "complete", "setting_id": setting["id"], "split": "train",
        "metric_mode": "rms_v2", "relative_endpoints": setting["relative_endpoints"],
        "task_metric_dims": setting["task_metric_dims"],
        "train_manifest_sha256": contract["train_manifest_sha256"],
        "normalizer_sha256": contract["normalizer_sha256"],
    }
    for key, expected_value in expected.items():
        common.require(type(calibration[key]) is type(expected_value)
                       and calibration[key] == expected_value,
                       f"{label}.{key} differs")
    common.require_string(calibration["algorithm_version"], f"{label}.algorithm_version")
    common.require_sha256(calibration["implementation_sha256"], f"{label}.implementation_sha256")
    common.require_string(calibration["numpy_version"], f"{label}.numpy_version")
    common.require_string(calibration["scipy_version"], f"{label}.scipy_version")
    common.require(calibration["input_units"] == "training-normalizer standardized observations",
                   f"{label}.input_units differs")
    xy_dims = common.require_int_list(calibration["xy_dims"], f"{label}.xy_dims",
                                      nonempty=True, minimum=0,
                                      maximum=contract["obs_dim"] - 1)
    common.require(len(set(xy_dims)) == len(xy_dims), f"{label}.xy_dims repeats an index")
    metric_dims = common.require_int_list(calibration["task_metric_dims"],
                                          f"{label}.task_metric_dims",
                                          nonempty=True, minimum=0,
                                          maximum=contract["obs_dim"] - 1)
    common.require(len(set(metric_dims)) == len(metric_dims),
                   f"{label}.task_metric_dims repeats an index")
    common.require(calibration["contract_sha256"] == setting["calibration_sha256"],
                   f"{label} finalized identity differs")
    body = dict(calibration)
    claimed = body.pop("contract_sha256")
    common.require(common.stable_hash(body) == claimed, f"{label} canonical self-hash differs")
    common.require(
        common.canonical_json(calibration["chosen"]).encode("ascii")
        == common.canonical_json(contract["chosen_thresholds"]).encode("ascii"),
                   f"{label} chosen thresholds differ from input contract")
    input_contract_audit._validate_thresholds(calibration["chosen"], f"{label}.chosen")
    config = common.require_exact_keys(
        calibration["config"], CALIBRATION_CONFIG_KEYS, f"{label}.config"
    )
    int_config_ranges = {
        "anchor_sample_seed": (0, None),
        "max_modes": (1, None),
        "min_occupied_horizon_classes": (1, len(FORMAL_CALIBRATION_CONFIG["horizons"])),
        "num_neighbors": (2, None),
        "query_multiplier": (1, None),
        "retrieval_pool": (1, None),
        "retrieval_pool_seed": (0, None),
        "sample_size": (1, None),
        "time_exclusion": (0, None),
    }
    for key, (minimum, maximum) in int_config_ranges.items():
        common.require_int(config[key], f"{label}.config.{key}",
                           minimum=minimum, maximum=maximum)
    horizons = common.require_int_list(config["horizons"], f"{label}.config.horizons",
                                       nonempty=True, minimum=1)
    common.require(horizons == sorted(set(horizons)),
                   f"{label}.config.horizons is not strictly increasing")
    for key in (
        "fallback_radius_quantile", "max_insufficient_neighbor_fraction",
        "max_retrieval_fallback_fraction", "max_truncation_fraction",
        "min_multimode_anchor_fraction", "min_normalized_horizon_entropy",
        "radius_quantile",
    ):
        _require_number_range(config[key], f"{label}.config.{key}",
                              minimum=0.0, maximum=1.0)
    for key in ("min_mean_retained_modes", "max_mean_retained_modes",
                "min_mean_retrieved"):
        _require_number_range(config[key], f"{label}.config.{key}", minimum=0.0)
    for key in ("cluster_candidate_quantiles", "horizon_candidate_quantiles",
                "radius_report_quantiles"):
        values = _validate_number_list(config[key], f"{label}.config.{key}",
                                       minimum=0.0, maximum=1.0)
        common.require(bool(values), f"{label}.config.{key} is empty")
    common.require(config["cluster_method"] == "average",
                   f"{label}.config.cluster_method differs")
    common.require(common.canonical_json(config)
                   == common.canonical_json(FORMAL_CALIBRATION_CONFIG),
                   f"{label}.config differs from the finalized calibration protocol")

    train_transitions = contract["train_transitions"]
    sample_size = config["sample_size"]
    common.require_int(
        calibration["anchor_population"], f"{label}.anchor_population",
        minimum=sample_size, maximum=train_transitions,
    )
    common.require_int(calibration["sample_seed"], f"{label}.sample_seed", minimum=0)
    common.require(calibration["sample_seed"] == config["anchor_sample_seed"],
                   f"{label}.sample_seed differs from config")
    selected_count = common.require_int(
        calibration["selected_continuation_count"],
        f"{label}.selected_continuation_count",
        minimum=sample_size,
        maximum=sample_size * config["num_neighbors"],
    )
    sample_indices = common.require_int_list(
        calibration["sample_indices"], f"{label}.sample_indices",
        minimum=0, maximum=train_transitions - 1,
    )
    common.require(len(sample_indices) == sample_size, f"{label} sample length differs")
    common.require(all(left < right for left, right in zip(sample_indices, sample_indices[1:])),
                   f"{label}.sample_indices is not strictly increasing and unique")
    pool_indices = common.require_int_list(
        calibration["retrieval_pool_indices"], f"{label}.retrieval_pool_indices",
        minimum=0, maximum=train_transitions - 1,
    )
    expected_pool_size = min(config["retrieval_pool"], train_transitions)
    common.require(len(pool_indices) == expected_pool_size,
                   f"{label} retrieval pool length differs")
    common.require(all(left < right for left, right in zip(pool_indices, pool_indices[1:])),
                   f"{label}.retrieval_pool_indices is not strictly increasing and unique")
    selected_indices = common.require_int_list(
        calibration["selected_continuation_indices"],
        f"{label}.selected_continuation_indices",
        minimum=0, maximum=train_transitions - 1,
    )
    common.require(len(selected_indices) == selected_count,
                   f"{label} selected continuation count differs")
    for key, values in (
        ("sample", sample_indices),
        ("retrieval_pool", pool_indices),
        ("selected_continuation", selected_indices),
    ):
        common.require(_indices_sha256(values, f"{label}.{key}_indices")
                       == calibration[f"{key}_indices_sha256"],
                       f"{label} {key} byte hash differs")
    offsets = common.require_int_list(
        calibration["selected_continuation_offsets"],
        f"{label}.selected_continuation_offsets",
    )
    common.require(len(offsets) == sample_size + 1 and offsets[0] == 0
                   and offsets[-1] == selected_count
                   and all(left < right for left, right in zip(offsets, offsets[1:])),
                   f"{label} continuation offsets differ")
    common.require(all(1 <= right - left <= config["num_neighbors"]
                       for left, right in zip(offsets, offsets[1:])),
                   f"{label} continuation group sizes differ")
    for owner, (left, right) in enumerate(zip(offsets, offsets[1:])):
        group = selected_indices[left:right]
        common.require(group[0] == sample_indices[owner],
                       f"{label} continuation group does not begin with its sampled anchor")
        common.require(len(set(group)) == len(group),
                       f"{label} continuation group repeats an index")
    gates = common.require_exact_keys(calibration["gates"], GATE_KEYS, f"{label}.gates")
    for key, gate_value in gates.items():
        gate = common.require_exact_keys(gate_value, {"passed", "rule", "value"},
                                         f"{label}.gates.{key}")
        common.require_bool(gate["passed"], f"{label}.gates.{key}.passed")
        common.require(gate["passed"] is True, f"{label}.gates.{key} did not pass")
        common.require_string(gate["rule"], f"{label}.gates.{key}.rule")
        common.require(type(gate["value"]) is float,
                       f"{label}.gates.{key}.value is not a floating-point number")
        common.require_number(gate["value"], f"{label}.gates.{key}.value")
    for object_key in ("cluster", "horizon", "retrieval", "eligible_query_count_histogram"):
        common.require(isinstance(calibration[object_key], dict),
                       f"{label}.{object_key} is not an object")
    _validate_candidate_structures(
        calibration, label, config=config, sample_size=sample_size,
        selected_count=selected_count,
    )
    _validate_histogram(
        calibration["eligible_query_count_histogram"],
        f"{label}.eligible_query_count_histogram",
        key_minimum=0, key_maximum=calibration["retrieval"]["query_k"], total=sample_size,
    )
    common.require(calibration["chosen"]["retrieval_radius"]
                   == calibration["retrieval"]["chosen_radius"],
                   f"{label} chosen retrieval radius disagrees")
    common.require(calibration["chosen"]["displacement_threshold"]
                   == calibration["horizon"]["chosen_threshold"],
                   f"{label} chosen horizon threshold disagrees")
    common.require(calibration["chosen"]["cluster_threshold"]
                   == calibration["cluster"]["chosen_threshold"],
                   f"{label} chosen cluster threshold disagrees")
    expected_gate_values = {
        "eligible_23rd_neighbor": (
            calibration["retrieval"]["insufficient_anchor_fraction"],
            f"<= {config['max_insufficient_neighbor_fraction']}",
        ),
        "mean_retrieved": (
            calibration["retrieval"]["chosen"]["mean_retrieved"],
            f">= {config['min_mean_retrieved']}",
        ),
        "retrieval_fallback_fraction": (
            calibration["retrieval"]["chosen"]["fallback_fraction"],
            f"<= {config['max_retrieval_fallback_fraction']}",
        ),
        "normalized_horizon_entropy": (
            calibration["horizon"]["normalized_entropy"],
            f">= {config['min_normalized_horizon_entropy']}",
        ),
        "occupied_horizon_classes": (
            calibration["horizon"]["occupied_classes"],
            f">= {config['min_occupied_horizon_classes']}",
        ),
        "mean_retained_modes": (
            calibration["cluster"]["mean_retained_modes"],
            f"in [{config['min_mean_retained_modes']}, {config['max_mean_retained_modes']}]",
        ),
        "multimode_anchor_fraction": (
            calibration["cluster"]["multimode_anchor_fraction"],
            f">= {config['min_multimode_anchor_fraction']}",
        ),
        "mode_truncation_fraction": (
            calibration["cluster"]["truncation_fraction"],
            f"<= {config['max_truncation_fraction']}",
        ),
    }
    for gate_name, (expected_value, expected_rule) in expected_gate_values.items():
        gate = gates[gate_name]
        common.require(gate["value"] == float(expected_value),
                       f"{label}.gates.{gate_name}.value disagrees with calibration")
        common.require(gate["rule"] == expected_rule,
                       f"{label}.gates.{gate_name}.rule differs")
    return calibration


def _validate_composite(
    value: object, setting: Mapping[str, Any], contract: Mapping[str, Any]
) -> dict[str, Any]:
    label = f"{setting['id']} composite recipe"
    composite = dict(common.require_exact_keys(value, COMPOSITE_KEYS, label))
    common.require_int(composite["schema_version"], f"{label}.schema_version")
    expected = {
        "schema_version": 1, "status": "complete",
        "recipe_version": "treewm_compact_future_recipe_v1",
        "recipe_sha256": setting["future_recipe_sha256"],
        "source_manifest_sha256": contract["data_manifest_sha256"],
        "normalizer_sha256": contract["normalizer_sha256"],
        "calibration_sha256": setting["calibration_sha256"],
        "chosen_thresholds": contract["chosen_thresholds"],
        "train_manifest_sha256": contract["train_manifest_sha256"],
        "validation_manifest_sha256": contract["validation_manifest_sha256"],
        "code_sha256": common.RECIPE_CODE_SHA256,
        "runtime_sha256": common.RECIPE_RUNTIME_SHA256,
        "train_manifest": "train/manifest.json",
        "validation_manifest": "val/manifest.json",
    }
    for key, expected_value in expected.items():
        common.require(common.canonical_json(composite[key]).encode("ascii")
                       == common.canonical_json(expected_value).encode("ascii"),
                       f"{label}.{key} differs")
    for key in (
        "recipe_sha256", "source_manifest_sha256", "normalizer_sha256",
        "calibration_sha256", "train_manifest_sha256", "validation_manifest_sha256",
        "train_recipe_sha256", "validation_recipe_sha256", "code_sha256", "runtime_sha256",
    ):
        common.require_sha256(composite[key], f"{label}.{key}")
    body = dict(composite)
    claimed = body.pop("recipe_sha256")
    common.require(common.stable_hash(body) == claimed, f"{label} canonical self-hash differs")
    return composite


def _validate_child(
    value: object,
    *,
    setting: Mapping[str, Any],
    contract: Mapping[str, Any],
    composite: Mapping[str, Any],
    calibration: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    label = f"{setting['id']} {split} recipe"
    child = dict(common.require_exact_keys(value, CHILD_KEYS, label))
    common.require_int(child["schema_version"], f"{label}.schema_version")
    common.require(child["schema_version"] == 1 and child["status"] == "complete",
                   f"{label} is incomplete")
    for key in ("identity_sha256", "recipe_sha256", "records_sha256"):
        common.require_sha256(child[key], f"{label}.{key}")
    body = dict(child)
    claimed = body.pop("recipe_sha256")
    common.require(common.stable_hash(body) == claimed, f"{label} canonical self-hash differs")
    expected_recipe = composite["train_recipe_sha256" if split == "train" else "validation_recipe_sha256"]
    common.require(claimed == expected_recipe, f"{label} does not match composite")
    common.require(child["records_file"] == "records.npy", f"{label}.records_file differs")
    common.require_int(child["records_size"], f"{label}.records_size", minimum=1)
    common.require_int(child["records_mtime_ns"], f"{label}.records_mtime_ns", minimum=1)
    common.require_int(child["record_count"], f"{label}.record_count", minimum=1)
    expected_count = setting[
        "published_union_train_anchors" if split == "train" else "published_union_validation_anchors"
    ]
    common.require(child["record_count"] == expected_count, f"{label} record count differs")
    common.require(
        common.canonical_json(child["record_dtype"]).encode("ascii")
        == common.canonical_json(expected_record_dtype()).encode("ascii"),
        f"{label} record dtype differs",
    )
    identity = dict(common.require_exact_keys(child["identity"], IDENTITY_KEYS, f"{label}.identity"))
    common.require(common.stable_hash(identity) == child["identity_sha256"],
                   f"{label} identity canonical hash differs")
    expected_identity = {
        "schema_version": 1, "recipe_version": "treewm_compact_future_recipe_v1",
        "split": split, "anchor_union_count": expected_count,
        "source_manifest_sha256": contract["data_manifest_sha256"],
        "split_manifest_sha256": contract[
            "train_manifest_sha256" if split == "train" else "validation_manifest_sha256"
        ],
        "normalizer_sha256": contract["normalizer_sha256"],
        "calibration_sha256": setting["calibration_sha256"],
        "chosen_thresholds": contract["chosen_thresholds"],
        "code_sha256": common.RECIPE_CODE_SHA256,
        "runtime_sha256": common.RECIPE_RUNTIME_SHA256,
        "task_metric_dims": setting["task_metric_dims"],
        "xy_dims": calibration["xy_dims"],
    }
    for key, expected_value in expected_identity.items():
        common.require(common.canonical_json(identity[key]).encode("ascii")
                       == common.canonical_json(expected_value).encode("ascii"),
                       f"{label}.identity.{key} differs")
    common.require_int_list(
        identity["task_metric_dims"], f"{label}.identity.task_metric_dims",
        nonempty=True, minimum=0, maximum=contract["obs_dim"] - 1,
    )
    common.require_int_list(
        identity["xy_dims"], f"{label}.identity.xy_dims",
        nonempty=True, minimum=0, maximum=contract["obs_dim"] - 1,
    )
    common.require_sha256(identity["anchor_union_sha256"], f"{label}.identity.anchor_union_sha256")
    common.require_string(identity["numpy_version"], f"{label}.identity.numpy_version")
    common.require_string(identity["scipy_version"], f"{label}.identity.scipy_version")
    anchor_sets = common.require_exact_keys(
        identity["anchor_sets"], {"seed0", "seed1", "seed2", "seed3"},
        f"{label}.identity.anchor_sets",
    )
    expected_seed_count = 300000 if split == "train" else 30000
    for seed, value in anchor_sets.items():
        row = common.require_exact_keys(value, {"count", "sha256"},
                                        f"{label}.identity.anchor_sets.{seed}")
        common.require_int(row["count"], f"{label}.identity.anchor_sets.{seed}.count", minimum=1)
        common.require(row["count"] <= expected_seed_count,
                       f"{label}.identity.anchor_sets.{seed}.count is too large")
        common.require_sha256(row["sha256"], f"{label}.identity.anchor_sets.{seed}.sha256")
    config = common.require_exact_keys(identity["future_config"], FUTURE_CONFIG_KEYS,
                                       f"{label}.identity.future_config")
    expected_config = {
        "cluster_method": "average", "fixed_horizon": 32, "h_max": 64,
        "horizon_rule": "displacement", "horizons": [4, 8, 16, 32, 64],
        "include_self": True, "max_modes": 4, "metric_mode": "rms_v2",
        "multi_step_depth": 3, "num_neighbors": 24, "query_multiplier": 6,
        "relative_endpoints": setting["relative_endpoints"], "retrieval_pool": 50000,
        "time_exclusion": 50,
        "cluster_threshold": contract["chosen_thresholds"]["cluster_threshold"],
        "displacement_threshold": contract["chosen_thresholds"]["displacement_threshold"],
        "retrieval_radius": contract["chosen_thresholds"]["retrieval_radius"],
    }
    for key, expected_value in expected_config.items():
        common.require(common.canonical_json(config[key]).encode("ascii")
                       == common.canonical_json(expected_value).encode("ascii"),
                       f"{label}.identity.future_config.{key} differs")
    common.require_int_list(
        config["horizons"], f"{label}.identity.future_config.horizons",
        nonempty=True, minimum=1,
    )
    return child


def _validate_record_content(array: Any, *, child: Mapping[str, Any], contract: Mapping[str, Any], label: str) -> dict[str, Any]:
    import numpy as np

    common.require(list(array.dtype.descr) == [tuple([row[0], row[1], tuple(row[2])])
                                               if len(row) == 3 else tuple(row)
                                               for row in expected_record_dtype()],
                   f"{label} loaded dtype differs")
    common.require(array.shape == (child["record_count"],), f"{label} loaded shape differs")
    anchors = np.asarray(array["anchor"], dtype=np.int64)
    common.require(len(anchors) > 0 and np.all(anchors[1:] > anchors[:-1]),
                   f"{label} anchors are not strictly increasing")
    common.require(hashlib.sha256(anchors.astype("<i8", copy=False).tobytes()).hexdigest()
                   == child["identity"]["anchor_union_sha256"],
                   f"{label} anchor union hash differs")
    split = child["identity"]["split"]
    transition_count = contract[
        "train_transitions" if split == "train" else "validation_transitions"
    ]
    common.require(int(anchors[0]) >= 0 and int(anchors[-1]) < transition_count,
                   f"{label} anchor range escapes its split")
    for section in common.chunks(len(array), 250_000):
        rows = array[section]
        neighbors = rows["neighbors"]
        present = neighbors >= 0
        counts = present.sum(axis=1)
        common.require(np.all((neighbors == -1) | ((neighbors >= 0) & (neighbors < transition_count))),
                       f"{label} neighbor range differs")
        expected_present = np.arange(neighbors.shape[1], dtype=np.int64)[None, :] < counts[:, None]
        common.require(np.array_equal(present, expected_present),
                       f"{label} present neighbors are not a contiguous prefix")
        sorted_neighbors = np.sort(neighbors, axis=1)
        repeated_neighbors = (
            (sorted_neighbors[:, 1:] == sorted_neighbors[:, :-1])
            & (sorted_neighbors[:, 1:] >= 0)
        )
        common.require(not bool(np.any(repeated_neighbors)),
                       f"{label} repeats a present neighbor")
        common.require(np.array_equal(counts.astype(np.uint8), rows["num_retrieved"]),
                       f"{label} retrieved counts differ")
        common.require(np.array_equal(rows["num_retrieved"], rows["retrieval_num_valid"]),
                       f"{label} valid retrieval counts differ")
        maximum_candidates = (
            child["identity"]["future_config"]["num_neighbors"]
            * child["identity"]["future_config"]["query_multiplier"]
            + 1
            + (1 if child["identity"]["future_config"]["include_self"] else 0)
        )
        common.require(np.all(
            (rows["retrieval_num_candidates"] >= rows["num_retrieved"])
            & (rows["retrieval_num_candidates"] <= maximum_candidates)
        ), f"{label} candidate counts are outside the retrieval query range")
        for name in ("fut_valid", "mode_valid", "ms_valid", "retrieval_fallback",
                     "retrieval_truncated", "retrieval_query_saturated"):
            common.require(np.all((rows[name] == 0) | (rows[name] == 1)),
                           f"{label} {name} is not binary")
        future_valid = rows["fut_valid"].astype(bool)
        common.require(np.all(~future_valid | present),
                       f"{label} future-valid slot has no present neighbor")
        common.require(np.all(rows["horizon_idx"][present] < 5),
                       f"{label} continuation horizon index differs")
        common.require(np.all(rows["ms_horizon_idx"][rows["ms_valid"].astype(bool)] < 5),
                       f"{label} multistep horizon index differs")
        common.require(np.all((rows["cluster"] == -1) | ((rows["cluster"] >= 0) & (rows["cluster"] < 4))),
                       f"{label} cluster labels differ")
        common.require(np.all(rows["cluster"][~present] == -1),
                       f"{label} absent-neighbor cluster labels differ")
        common.require(np.array_equal(rows["fut_valid"].astype(bool), rows["cluster"] >= 0),
                       f"{label} future validity and cluster labels disagree")
        retained = rows["mode_valid"].sum(axis=1)
        common.require(np.array_equal(retained.astype(np.uint8), rows["modes_retained"]),
                       f"{label} retained-mode counts differ")
        common.require(np.all((retained >= 1) & (retained <= 4)),
                       f"{label} retained-mode range differs")
        common.require(np.all(
            (rows["modes_raw"] >= rows["modes_retained"])
            & (rows["modes_raw"] <= rows["num_retrieved"])
        ), f"{label} raw-mode counts differ")
        common.require(np.array_equal(
            (rows["modes_raw"].astype(np.int16) - rows["modes_retained"].astype(np.int16)),
            rows["modes_truncated"].astype(np.int16),
        ), f"{label} truncated-mode counts differ")
        valid_rep = rows["mode_valid"].astype(bool)
        common.require(np.all(
            (~valid_rep)
            | ((rows["mode_rep"] >= 0) & (rows["mode_rep"] < neighbors.shape[1]))
        ),
                       f"{label} mode representatives differ")
        safe_representatives = np.where(valid_rep, rows["mode_rep"], 0).astype(np.intp)
        representative_present = np.take_along_axis(present, safe_representatives, axis=1)
        common.require(np.all(~valid_rep | representative_present),
                       f"{label} valid mode representative does not gather a present neighbor")
        representative_future_valid = np.take_along_axis(
            future_valid, safe_representatives, axis=1
        )
        common.require(np.all(~valid_rep | representative_future_valid),
                       f"{label} valid mode representative is not future-valid")
        common.require(np.all(rows["mode_rep"][~valid_rep] == -1),
                       f"{label} invalid mode representatives are not sentinel -1")
        common.require(np.all(rows["mode_valid"] ==
                              (np.arange(4, dtype=np.int64)[None, :] < retained[:, None])),
                       f"{label} valid modes are not a contiguous prefix")
        sorted_reps = np.sort(np.where(valid_rep, rows["mode_rep"], 127), axis=1)
        common.require(np.all(np.diff(sorted_reps, axis=1)[valid_rep[:, 1:]] > 0),
                       f"{label} repeats a valid mode representative")
        common.require(np.all(np.isfinite(rows["mode_mass"]))
                       and np.all(rows["mode_mass"] >= 0),
                       f"{label} mode masses are invalid")
        common.require(np.all(rows["mode_mass"][~valid_rep] == 0),
                       f"{label} invalid mode masses are nonzero")
        common.require(np.allclose(rows["mode_mass"].sum(axis=1), 1.0,
                                   rtol=1e-5, atol=1e-6),
                       f"{label} valid mode masses do not sum to one")
        ms_valid = rows["ms_valid"].astype(bool)
        ms_count = ms_valid.sum(axis=1)
        common.require(np.all(rows["ms_valid"] ==
                              (np.arange(3, dtype=np.int64)[None, :] < ms_count[:, None])),
                       f"{label} multistep validity is not a contiguous prefix")
        common.require(np.all(np.isfinite(rows["future_diversity"]))
                       and np.all(rows["future_diversity"] >= 0)
                       and np.all(np.isfinite(rows["retrieval_mean_distance"]))
                       and np.all(rows["retrieval_mean_distance"] >= 0),
                       f"{label} floating telemetry is invalid")
    return {
        "anchor_count": len(anchors),
        "anchor_min": int(anchors[0]),
        "anchor_max": int(anchors[-1]),
        "anchor_union_sha256": child["identity"]["anchor_union_sha256"],
    }


def _load_input_contract(contract_root: common.SecureRoot, setting: Mapping[str, Any]) -> dict[str, Any]:
    value, _row = contract_root.read_json(PurePosixPath("data") / f"{setting['id']}.json")
    return input_contract_audit.validate_contract(value, setting)


def _calibration_summary_row(calibration: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the only summary row authorized by a validated calibration."""
    return {
        "setting_id": calibration["setting_id"],
        "calibration_sha256": calibration["contract_sha256"],
        "chosen": calibration["chosen"],
        "gates": calibration["gates"],
        "horizon_histogram": calibration["horizon"]["chosen_histogram"],
        "raw_mode_histogram": calibration["cluster"]["chosen_raw_mode_histogram"],
        "retained_mode_histogram": calibration["cluster"]["chosen_retained_mode_histogram"],
        "retrieval": calibration["retrieval"]["chosen"],
    }


def _validate_summary(
    value: object,
    settings: Sequence[Mapping[str, Any]],
    expected_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    summary = dict(common.require_exact_keys(
        value,
        {"campaign_id", "campaign_protocol_sha256", "schema_version", "setting_count",
         "settings", "status", "summary_sha256"},
        "all-settings calibration summary",
    ))
    expected = {
        "campaign_id": common.CAMPAIGN_ID,
        "campaign_protocol_sha256": common.CAMPAIGN_PROTOCOL_SHA256,
        "schema_version": 1, "setting_count": len(settings), "status": "complete",
    }
    for key, expected_value in expected.items():
        common.require(type(summary[key]) is type(expected_value) and summary[key] == expected_value,
                       f"all-settings calibration summary {key} differs")
    common.require(isinstance(summary["settings"], list) and len(summary["settings"]) == len(settings),
                   "all-settings calibration summary rows differ")
    common.require(set(expected_rows) == {setting["id"] for setting in settings},
                   "validated calibration summary-row key set differs")
    for expected_setting, row in zip(settings, summary["settings"], strict=True):
        row = common.require_exact_keys(
            row, SUMMARY_ROW_KEYS,
            f"all-settings calibration row {expected_setting['id']}",
        )
        common.require(row["setting_id"] == expected_setting["id"],
                       "all-settings calibration order differs")
        common.require(row["calibration_sha256"] == expected_setting["calibration_sha256"],
                       "all-settings calibration identity differs")
        common.require_sha256(row["calibration_sha256"],
                              f"all-settings calibration {expected_setting['id']} identity")
        input_contract_audit._validate_thresholds(
            row["chosen"], f"all-settings calibration {expected_setting['id']}.chosen"
        )
        gates = common.require_exact_keys(
            row["gates"], GATE_KEYS, f"all-settings calibration {expected_setting['id']}.gates"
        )
        for gate_name, gate_value in gates.items():
            gate = common.require_exact_keys(
                gate_value, {"passed", "rule", "value"},
                f"all-settings calibration {expected_setting['id']}.gates.{gate_name}",
            )
            common.require_bool(gate["passed"],
                                f"all-settings calibration {expected_setting['id']}.gates.{gate_name}.passed")
            common.require(gate["passed"] is True,
                           f"all-settings calibration {expected_setting['id']}.gates.{gate_name} did not pass")
            common.require_string(gate["rule"],
                                  f"all-settings calibration {expected_setting['id']}.gates.{gate_name}.rule")
            common.require_number(gate["value"],
                                  f"all-settings calibration {expected_setting['id']}.gates.{gate_name}.value")
        _validate_number_list(row["horizon_histogram"],
                              f"all-settings calibration {expected_setting['id']}.horizon_histogram",
                              length=5)
        _validate_histogram(row["raw_mode_histogram"],
                            f"all-settings calibration {expected_setting['id']}.raw_mode_histogram")
        _validate_histogram(row["retained_mode_histogram"],
                            f"all-settings calibration {expected_setting['id']}.retained_mode_histogram")
        retrieval = common.require_exact_keys(
            row["retrieval"], SUMMARY_RETRIEVAL_KEYS,
            f"all-settings calibration {expected_setting['id']}.retrieval",
        )
        _validate_histogram(retrieval["retrieved_histogram"],
                            f"all-settings calibration {expected_setting['id']}.retrieval.retrieved_histogram")
        for key in SUMMARY_RETRIEVAL_KEYS - {"retrieved_histogram"}:
            common.require_number(retrieval[key],
                                  f"all-settings calibration {expected_setting['id']}.retrieval.{key}")
        expected_row = expected_rows[expected_setting["id"]]
        common.require(
            common.canonical_json(dict(row)).encode("ascii")
            == common.canonical_json(dict(expected_row)).encode("ascii"),
            f"all-settings calibration {expected_setting['id']} row differs from validated calibration",
        )
    claimed = summary["summary_sha256"]
    common.require_sha256(claimed, "all-settings calibration summary self-hash")
    body = dict(summary)
    body.pop("summary_sha256")
    common.require(common.stable_hash(body) == claimed,
                   "all-settings calibration summary canonical self-hash differs")
    return summary


def audit_setting(
    *,
    contract_root: common.SecureRoot,
    calibration_root: common.SecureRoot,
    future_root: common.SecureRoot,
    setting: Mapping[str, Any],
) -> dict[str, Any]:
    setting_id = setting["id"]
    contract = _load_input_contract(contract_root, setting)
    calibration, calibration_inventory = calibration_root.read_json(
        f"{setting_id}.json"
    )
    calibration = _validate_calibration(calibration, setting, contract)
    with future_root.subroot(
        setting_id, f"{setting_id} future recipe"
    ) as recipe_root:
        expected_files = (
            "manifest.json", "train/.build.lock", "train/manifest.json", "train/records.npy",
            "val/.build.lock", "val/manifest.json", "val/records.npy",
        )
        recipe_root.require_exact_tree(files=expected_files, directories=("train", "val"))
        composite, composite_inventory = recipe_root.read_json("manifest.json")
        composite = _validate_composite(composite, setting, contract)
        inventory = {
            "calibration.json": calibration_inventory,
            "manifest.json": composite_inventory,
        }
        splits: dict[str, Any] = {}
        for split in ("train", "val"):
            lock_relative = f"{split}/.build.lock"
            with recipe_root.open_regular(lock_relative, f"{setting_id} {split} build lock") as lock_source:
                common.require(lock_source.size == 0, f"{setting_id} {split} build lock is not empty")
                inventory[lock_relative] = common.inventory_row(lock_source)
            manifest_relative = f"{split}/manifest.json"
            child, child_inventory = recipe_root.read_json(manifest_relative)
            child = _validate_child(
                child,
                setting=setting,
                contract=contract,
                composite=composite,
                calibration=calibration,
                split=split,
            )
            inventory[manifest_relative] = child_inventory
            records_relative = f"{split}/records.npy"
            with recipe_root.open_regular(records_relative, f"{setting_id} {split} records") as records_source:
                common.require(records_source.size == child["records_size"],
                               f"{setting_id} {split} records size differs")
                common.require(records_source.mtime_ns == child["records_mtime_ns"],
                               f"{setting_id} {split} records mtime differs")
                records_digest = records_source.sha256()
                common.require(records_digest == child["records_sha256"],
                               f"{setting_id} {split} records byte hash differs")
                inventory[records_relative] = common.inventory_row(records_source, digest=records_digest)
                with common.StableNpy(records_source, f"{setting_id} {split} records") as records:
                    content = _validate_record_content(
                        records.array, child=child, contract=contract,
                        label=f"{setting_id} {split} records",
                    )
            splits[split] = {
                "recipe_sha256": child["recipe_sha256"],
                "identity_sha256": child["identity_sha256"],
                "records_sha256": child["records_sha256"],
                **content,
            }
    return {
        "setting_id": setting_id,
        "input_contract_sha256": setting["input_contract_sha256"],
        "calibration_sha256": setting["calibration_sha256"],
        "future_recipe_sha256": setting["future_recipe_sha256"],
        "source_manifest_sha256": contract["data_manifest_sha256"],
        "normalizer_sha256": contract["normalizer_sha256"],
        "code_sha256": common.RECIPE_CODE_SHA256,
        "runtime_sha256": common.RECIPE_RUNTIME_SHA256,
        "_calibration_summary_row": _calibration_summary_row(calibration),
        "splits": splits,
        "inventory": inventory,
        "inventory_sha256": common.stable_hash(inventory),
    }


def run(
    *,
    contract_root_path: Path,
    input_lock: Mapping[str, Any] | None,
    ledger: Sequence[Mapping[str, Any]] = common.SETTINGS,
    require_all_ten: bool = True,
) -> dict[str, Any]:
    settings = tuple(dict(row) for row in ledger)
    if require_all_ten:
        common.require(
            common.canonical_json(list(settings)).encode("ascii")
            == common.canonical_json(list(common.SETTINGS)).encode("ascii"),
            "future audit is not using finalized all-ten ledger",
        )
    input_artifact: str | None = None
    if input_lock is not None:
        validated = common.validate_sealed_result(
            input_lock, audit_id=input_contract_audit.AUDIT_ID, status=input_contract_audit.STATUS
        )
        common.require(validated.get("setting_order") == [row["id"] for row in settings],
                       "input lock setting order differs")
        input_artifact = validated["artifact_sha256"]
    elif require_all_ten:
        raise common.DataAuthorityError("sealed input-contract lock is required")
    with common.SecureRoot(contract_root_path, "registered compatible-contract root") as contract_root:
        with contract_root.subroot(
            "future-recipes", "all-ten future-recipe root"
        ) as future_root, contract_root.subroot(
            "calibration", "all-ten calibration root"
        ) as calibration_root:
            files: list[str] = []
            directories: list[str] = []
            for setting in settings:
                setting_id = setting["id"]
                directories.extend((setting_id, f"{setting_id}/train", f"{setting_id}/val"))
                files.extend((
                    f"{setting_id}/manifest.json",
                    f"{setting_id}/train/.build.lock",
                    f"{setting_id}/train/manifest.json",
                    f"{setting_id}/train/records.npy",
                    f"{setting_id}/val/.build.lock",
                    f"{setting_id}/val/manifest.json",
                    f"{setting_id}/val/records.npy",
                ))
            future_root.require_exact_tree(files=files, directories=directories)
            calibration_root.require_exact_tree(
                files=("ALL_SETTINGS.json", *(f"{row['id']}.json" for row in settings)),
                directories=(),
            )
            summary, summary_inventory = calibration_root.read_json(
                "ALL_SETTINGS.json"
            )
            rows: dict[str, Any] = {}
            expected_summary_rows: dict[str, Any] = {}
            for setting in settings:
                row = audit_setting(
                    contract_root=contract_root,
                    calibration_root=calibration_root,
                    future_root=future_root,
                    setting=setting,
                )
                expected_summary_rows[setting["id"]] = row.pop(
                    "_calibration_summary_row"
                )
                rows[setting["id"]] = row
            summary = _validate_summary(summary, settings, expected_summary_rows)
    inventories = {
        "calibration/ALL_SETTINGS.json": summary_inventory,
        **{setting_id: row["inventory_sha256"] for setting_id, row in rows.items()},
    }
    body = {
        "classification": "outcome_blind_read_only_full_future_recipe_calibration_code_runtime_authority",
        "setting_order": [row["id"] for row in settings],
        "settings": rows,
        "all_settings_count": len(rows),
        "input_contract_audit_artifact_sha256": input_artifact,
        "calibration_summary_sha256": summary["summary_sha256"],
        "recipe_code_sha256": common.RECIPE_CODE_SHA256,
        "recipe_runtime_sha256": common.RECIPE_RUNTIME_SHA256,
        "ledger_sha256": common.stable_hash(list(settings)),
        "inventory_root_sha256": common.stable_hash(inventories),
        "source_sha256": common.source_file_sha256(Path(__file__)),
    }
    return common.seal_result(body, audit_id=AUDIT_ID, status=STATUS)


def _read_lock(path: Path, label: str) -> dict[str, Any]:
    with common.SecureRoot(path.absolute().parent, f"{label} parent") as root:
        value, _row = root.read_json(path.name, label)
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-root", type=Path, required=True)
    parser.add_argument("--input-lock", type=Path, required=True)
    parser.add_argument("--expected-lock", type=Path)
    args = parser.parse_args()
    try:
        input_lock = _read_lock(args.input_lock, "input-contract audit lock")
        result = run(
            contract_root_path=args.contract_root.absolute(),
            input_lock=input_lock,
        )
        if args.expected_lock is not None:
            lock = _read_lock(args.expected_lock, "future-recipe audit lock")
            common.validate_or_compare_lock(result, lock, audit_id=AUDIT_ID, status=STATUS)
    except Exception as exc:
        print(f"future recipe audit failed: {exc}", file=sys.stderr)
        return 1
    print("EXP24_FUTURE_RECIPE_AUDIT=" + common.canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
