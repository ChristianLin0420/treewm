from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CAMPAIGN_DIR.parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))
sys.path.insert(0, str(REPO_ROOT))

import calibration  # noqa: E402
from treewm.data.ogbench_dataset import TrajectoryIndex  # noqa: E402


def _synthetic_training_split(num_trajectories: int = 40, length: int = 70):
    observations = np.zeros((num_trajectories * length, 4), dtype=np.float32)
    terminals = np.zeros(num_trajectories * length, dtype=np.float32)
    for trajectory in range(num_trajectories):
        start = trajectory * length
        time = np.arange(length, dtype=np.float32)
        # Nearby trajectories share local states but have different speeds/directions,
        # producing alternative continuations and multiple horizon classes.
        observations[start : start + length, 0] = (
            np.sin(time / (3.0 + trajectory % 5)) + 0.01 * (trajectory % 4)
        )
        observations[start : start + length, 1] = (
            np.cos(time / (5.0 + trajectory % 7)) + 0.01 * (trajectory % 3)
        )
        observations[start : start + length, 2] = trajectory / num_trajectories
        observations[start : start + length, 3] = time / length
        terminals[start + length - 1] = 1.0
    return observations, TrajectoryIndex.from_terminals(terminals)


def _small_config(**overrides) -> calibration.CalibrationConfig:
    values = {
        "sample_size": 32,
        "num_neighbors": 4,
        "query_multiplier": 8,
        "time_exclusion": 0,
        "retrieval_pool": 2_000,
        "radius_report_quantiles": (0.5, 0.9),
        "horizon_candidate_quantiles": tuple(i / 10 for i in range(11)),
        "cluster_candidate_quantiles": tuple(i / 20 for i in range(1, 20)),
        "max_insufficient_neighbor_fraction": 1.0,
        "max_truncation_fraction": 1.0,
        "min_mean_retrieved": 1.0,
        "max_retrieval_fallback_fraction": 1.0,
        "min_normalized_horizon_entropy": 0.0,
        "min_occupied_horizon_classes": 1,
        "min_mean_retained_modes": 1.0,
        "max_mean_retained_modes": 4.0,
        "min_multimode_anchor_fraction": 0.0,
    }
    values.update(overrides)
    return calibration.CalibrationConfig(**values)


def test_horizon_threshold_maximizes_five_class_entropy_with_strict_boundary():
    displacement = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0],
            [0.1, 1.0, 2.0, 3.0, 4.0],
            [0.1, 0.2, 1.0, 2.0, 3.0],
            [0.1, 0.2, 0.3, 1.0, 2.0],
            [0.1, 0.2, 0.3, 0.4, 1.0],
        ]
    )
    result = calibration.select_horizon_threshold(
        displacement,
        candidate_quantiles=tuple(i / 100 for i in range(101)),
    )
    assert result["quantile_method"] == "higher"
    assert result["threshold_boundary"].endswith("> threshold")
    assert result["normalized_entropy"] == pytest.approx(1.0)
    assert result["occupied_classes"] == 5
    assert result["chosen_histogram"] == [1, 1, 1, 1, 1]


def test_cluster_threshold_targets_three_modes_under_truncation_constraint():
    # At the smallest observed threshold, only the closest pair merges: four raw
    # endpoints become exactly three modes without exceeding max_modes=4.
    endpoint_sets = [
        np.asarray([[0.0], [1.0], [3.0], [7.0]], dtype=np.float64) for _ in range(20)
    ]
    result = calibration.select_cluster_threshold(
        endpoint_sets,
        candidate_quantiles=tuple(i / 100 for i in range(1, 100)),
        max_modes=4,
        max_truncation_fraction=0.05,
    )
    assert result["quantile_method"] == "higher"
    assert result["had_truncation_feasible_candidate"] is True
    assert result["mean_retained_modes"] == pytest.approx(3.0)
    assert result["truncation_fraction"] == 0.0
    assert result["chosen_retained_mode_histogram"] == {"3": 20}


def test_radius_is_higher_q90_of_kth_neighbor_and_reports_insufficient_fraction():
    cfg = _small_config(num_neighbors=4, radius_report_quantiles=(0.5, 0.9))
    anchors = np.arange(10, dtype=np.int64)
    eligible_indices = [np.asarray([100, 101, 102], dtype=np.int64) for _ in range(9)] + [
        np.asarray([100, 101], dtype=np.int64)
    ]
    eligible_distances = [
        np.asarray([0.1, 0.2, float(row + 1)], dtype=np.float64) for row in range(9)
    ] + [np.asarray([0.1, 0.2], dtype=np.float64)]
    result, selected = calibration.select_retrieval_radius(
        anchors, eligible_indices, eligible_distances, cfg
    )
    assert result["required_nonself_neighbors"] == 3
    assert result["insufficient_anchor_fraction"] == pytest.approx(0.1)
    assert result["quantile_method"] == "higher"
    assert result["support_count_component"]["radius"] == pytest.approx(9.0)
    assert result["fallback_component"]["radius"] == pytest.approx(0.1)
    assert result["chosen_radius"] == pytest.approx(9.0)
    assert len(selected) == len(anchors)
    assert all(values[0] == anchor for anchor, values in zip(anchors, selected, strict=True))


def test_radius_uses_nearest_q99_to_enforce_fallback_without_relaxing_gate():
    cfg = _small_config(
        num_neighbors=4,
        radius_quantile=0.5,
        fallback_radius_quantile=0.99,
    )
    anchors = np.arange(101, dtype=np.int64)
    # The 3rd neighbour is usually close, but two anchors have a distant nearest
    # neighbour. q99(method=higher) selects the larger of those exact observed values.
    eligible_indices = []
    eligible_distances = []
    for row in range(101):
        base = 5.0 if row == 100 else (4.0 if row == 99 else 0.1)
        eligible_indices.append(np.asarray([1000 + 3 * row + i for i in range(3)]))
        eligible_distances.append(np.asarray([base, base + 0.1, base + 0.2]))
    result, _ = calibration.select_retrieval_radius(
        anchors, eligible_indices, eligible_distances, cfg
    )
    assert result["support_count_component"]["radius"] < 1.0
    assert result["fallback_component"]["radius"] == pytest.approx(4.0)
    assert result["chosen_radius"] == pytest.approx(4.0)
    assert result["chosen"]["fallback_fraction"] == pytest.approx(1 / 101)


def test_full_calibration_is_deterministic_train_only_and_content_hashed(tmp_path):
    observations, index = _synthetic_training_split()
    output = tmp_path / "calibration.json"
    kwargs = {
        "setting_id": "synthetic-setting",
        "train_manifest_sha256": "a" * 64,
        "normalizer_sha256": "c" * 64,
        "xy_dims": (0, 1),
        "task_metric_dims": (0, 1),
        "relative_endpoints": True,
        "config": _small_config(),
        "enforce_gates": True,
    }
    first = calibration.calibrate_future_metrics(
        observations, index, output_path=output, **kwargs
    )
    second = calibration.calibrate_future_metrics(observations, index, **kwargs)

    assert first == second
    assert first["status"] == "complete"
    assert first["split"] == "train"
    assert first["metric_mode"] == "rms_v2"
    assert first["selected_continuation_count"] > len(first["sample_indices"])
    assert first["horizon"]["quantile_method"] == "higher"
    assert first["cluster"]["quantile_method"] == "higher"
    assert first["numpy_version"] == np.__version__
    assert first["sample_seed"] == 0
    assert json.loads(output.read_text()) == first
    calibration.validate_contract(
        first,
        expected_config=kwargs["config"],
        expected_setting_id="synthetic-setting",
        expected_train_manifest_sha256="a" * 64,
        expected_normalizer_sha256="c" * 64,
        expected_xy_dims=(0, 1),
        expected_task_metric_dims=(0, 1),
        expected_relative_endpoints=True,
    )
    with pytest.raises(calibration.CalibrationError, match="config/rules"):
        calibration.validate_contract(first)  # relaxed fixtures cannot masquerade as formal

    tampered = json.loads(json.dumps(first))
    tampered["chosen"]["retrieval_radius"] += 1e-3
    with pytest.raises(calibration.CalibrationError, match="hash mismatch"):
        calibration.validate_contract(tampered, expected_config=kwargs["config"])


def test_failed_hard_gate_is_persisted_then_raises(tmp_path):
    observations, index = _synthetic_training_split()
    output = tmp_path / "failed.json"
    cfg = _small_config(min_mean_retrieved=1000.0)
    with pytest.raises(calibration.CalibrationGateError) as error:
        calibration.calibrate_future_metrics(
            observations,
            index,
            setting_id="synthetic-failure",
            train_manifest_sha256="b" * 64,
            normalizer_sha256="d" * 64,
            xy_dims=(0, 1),
            task_metric_dims=(0, 1),
            relative_endpoints=False,
            config=cfg,
            output_path=output,
            enforce_gates=True,
        )
    payload = error.value.payload
    assert payload["status"] == "failed"
    assert payload["gates"]["mean_retrieved"]["passed"] is False
    assert json.loads(output.read_text()) == payload
    with pytest.raises(calibration.CalibrationError, match="did not pass"):
        calibration.validate_contract(payload, expected_config=cfg)


def test_calibration_stop_callback_interrupts_without_publishing(tmp_path):
    observations, index = _synthetic_training_split()
    output = tmp_path / "must-not-publish.json"
    calls = 0

    class RequestedStop(RuntimeError):
        pass

    def stop_callback():
        nonlocal calls
        calls += 1
        if calls == 4:
            raise RequestedStop

    with pytest.raises(RequestedStop):
        calibration.calibrate_future_metrics(
            observations,
            index,
            setting_id="synthetic-stop",
            train_manifest_sha256="a" * 64,
            normalizer_sha256="c" * 64,
            xy_dims=(0, 1),
            task_metric_dims=(0, 1),
            relative_endpoints=True,
            config=_small_config(),
            output_path=output,
            enforce_gates=True,
            stop_callback=stop_callback,
        )
    assert calls == 4
    assert not output.exists()


def test_sample_valid_anchors_is_uniform_bounded_and_trajectory_safe():
    _, index = _synthetic_training_split(num_trajectories=4, length=70)
    anchors, population = calibration.sample_valid_anchors(
        index, min_horizon=64, sample_size=10, seed=9
    )
    assert population == 4 * (70 - 64)
    assert len(anchors) == 10
    assert len(np.unique(anchors)) == 10
    assert (_remaining(index, anchors) >= 64).all()

    with pytest.raises(calibration.CalibrationError, match="smaller than the locked sample"):
        calibration.sample_valid_anchors(
            index, min_horizon=64, sample_size=population + 1, seed=9
        )


def _remaining(index, indices):
    return index.remaining_at(np.asarray(indices, dtype=np.int64))
