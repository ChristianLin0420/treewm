#!/usr/bin/env python3
"""Deterministic train-only calibration of TreeWM v2 future-set metric thresholds.

The three thresholds share units but answer different questions:

* retrieval radius: full-standardized-state RMS distance;
* horizon displacement: task-coordinate standardized RMS displacement;
* endpoint clustering: task-coordinate standardized RMS endpoint distance.

No validation observations, rewards, goals, task successes, or final evaluations enter
the calibration.  Every candidate and selection rule is persisted so a chosen scalar is
an auditable consequence of the immutable training cache rather than a hand-tuned value.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from treewm.data.future_sets import bounded_uniform_indices  # noqa: E402


SCHEMA_VERSION = 1
ALGORITHM_VERSION = "treewm_future_metric_calibration_v1"
HORIZONS = (4, 8, 16, 32, 64)
StopCallback = Callable[[], None] | None


def _check_stop(stop_callback: StopCallback) -> None:
    if stop_callback is not None:
        stop_callback()


def implementation_sha256() -> str:
    """Bind calibration plus the builder whose metric semantics it mirrors."""
    sources = (
        Path(__file__).resolve(),
        REPOSITORY_ROOT / "treewm" / "data" / "future_sets.py",
        REPOSITORY_ROOT / "treewm" / "data" / "ogbench_dataset.py",
    )
    digest = hashlib.sha256()
    for path in sources:
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        content = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(relative.encode("utf-8") + b"\0" + content.encode("ascii") + b"\n")
    return digest.hexdigest()


class CalibrationError(ValueError):
    """The calibration input or candidate population is invalid."""


class CalibrationGateError(CalibrationError):
    """The deterministic calibration completed but failed a preregistered data gate."""

    def __init__(self, payload: dict[str, Any]):
        failed = [name for name, result in payload["gates"].items() if not result["passed"]]
        super().__init__(f"future-set calibration failed gates: {failed}")
        self.payload = payload


@dataclass(frozen=True)
class CalibrationConfig:
    """Preregistered calibration rules; formal v2 records every value verbatim."""

    sample_size: int = 4096
    num_neighbors: int = 24
    query_multiplier: int = 6
    time_exclusion: int = 50
    retrieval_pool: int = 50_000
    retrieval_pool_seed: int = 0
    anchor_sample_seed: int = 0
    radius_quantile: float = 0.90
    fallback_radius_quantile: float = 0.99
    radius_report_quantiles: tuple[float, ...] = (0.50, 0.75, 0.90, 0.95)
    horizon_candidate_quantiles: tuple[float, ...] = tuple(
        float(i) / 100.0 for i in range(101)
    )
    cluster_candidate_quantiles: tuple[float, ...] = tuple(
        float(i) / 100.0 for i in range(1, 100)
    )
    horizons: tuple[int, ...] = HORIZONS
    max_modes: int = 4
    cluster_method: str = "average"
    max_insufficient_neighbor_fraction: float = 0.10
    max_truncation_fraction: float = 0.05
    min_mean_retrieved: float = 18.0
    max_retrieval_fallback_fraction: float = 0.01
    min_normalized_horizon_entropy: float = 0.65
    min_occupied_horizon_classes: int = 4
    # These are non-degeneracy gates, not a demand that every continuous-control
    # setting fabricate three modes. The selector still targets three subject to the
    # <=5% raw-mode truncation constraint.
    min_mean_retained_modes: float = 1.5
    max_mean_retained_modes: float = 3.5
    min_multimode_anchor_fraction: float = 0.40

    def __post_init__(self) -> None:
        if self.sample_size < 1:
            raise CalibrationError("sample_size must be positive")
        if self.num_neighbors < 2:
            raise CalibrationError("num_neighbors must include self plus an alternative")
        if self.query_multiplier < 1:
            raise CalibrationError("query_multiplier must be positive")
        if self.retrieval_pool < self.num_neighbors:
            raise CalibrationError("retrieval_pool must cover num_neighbors")
        if not 0 < self.radius_quantile < 1:
            raise CalibrationError("radius_quantile must lie strictly inside (0, 1)")
        if not 0 < self.fallback_radius_quantile < 1:
            raise CalibrationError("fallback_radius_quantile must lie strictly inside (0, 1)")
        if tuple(sorted(self.horizons)) != self.horizons or len(self.horizons) != 5:
            raise CalibrationError("formal horizon grid must be five sorted lengths")
        if self.max_modes != 4:
            raise CalibrationError("formal v2 max_modes must be four")
        if self.cluster_method != "average":
            raise CalibrationError("formal v2 calibration uses average linkage")
        for name, values in (
            ("radius_report_quantiles", self.radius_report_quantiles),
            ("horizon_candidate_quantiles", self.horizon_candidate_quantiles),
            ("cluster_candidate_quantiles", self.cluster_candidate_quantiles),
        ):
            if not values or any(not 0 <= float(q) <= 1 for q in values):
                raise CalibrationError(f"{name} must contain quantiles in [0, 1]")

    @property
    def nonself_neighbors(self) -> int:
        return self.num_neighbors - 1

    @property
    def query_k(self) -> int:
        return self.num_neighbors * self.query_multiplier + 1


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _indices_sha256(indices: np.ndarray) -> str:
    canonical = np.asarray(indices, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _remaining_at(index: Any, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if hasattr(index, "remaining_at"):
        return np.asarray(index.remaining_at(indices), dtype=np.int64)
    return np.asarray(index.steps_remaining[indices], dtype=np.int64)


def sample_valid_anchors(
    index: Any,
    *,
    min_horizon: int,
    sample_size: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    """Uniformly sample the exact valid-transition universe without flattening it."""
    starts = np.asarray(index.starts, dtype=np.int64)
    lengths = np.asarray(index.lengths, dtype=np.int64)
    counts = np.maximum(lengths - int(min_horizon), 0)
    population = int(counts.sum(dtype=np.int64))
    if population == 0:
        raise CalibrationError("training split has no anchor with the minimum horizon")
    if population < int(sample_size):
        raise CalibrationError(
            f"valid training-anchor population {population} is smaller than the locked "
            f"sample size {sample_size}"
        )
    size = int(sample_size)
    ranks = bounded_uniform_indices(population, size, seed=seed)
    cumulative = np.cumsum(counts, dtype=np.int64)
    trajectories = np.searchsorted(cumulative, ranks, side="right")
    previous = np.where(trajectories == 0, 0, cumulative[trajectories - 1])
    anchors = starts[trajectories] + ranks - previous
    return anchors.astype(np.int64, copy=False), population


def _higher_quantile(values: np.ndarray, quantile: float) -> float:
    return float(np.quantile(values, float(quantile), method="higher"))


def _higher_quantile_grid(values: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        raise CalibrationError("cannot create a threshold grid from no finite values")
    grid = np.quantile(values, np.asarray(quantiles, dtype=np.float64), method="higher")
    return np.unique(np.asarray(grid, dtype=np.float64))


def _histogram(labels: np.ndarray, size: int | None = None) -> dict[str, int] | list[int]:
    labels = np.asarray(labels, dtype=np.int64)
    if size is not None:
        return np.bincount(labels, minlength=size).astype(np.int64).tolist()
    values, counts = np.unique(labels, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts, strict=True)}


def _normalized_entropy(histogram: Sequence[int]) -> float:
    counts = np.asarray(histogram, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return 0.0
    probabilities = counts[counts > 0] / total
    return float(-(probabilities * np.log(probabilities)).sum() / math.log(len(counts)))


def select_horizon_threshold(
    displacements: np.ndarray,
    *,
    candidate_quantiles: Sequence[float],
    num_horizons: int = 5,
    stop_callback: StopCallback = None,
) -> dict[str, Any]:
    """Maximize five-class horizon entropy with an explicit deterministic tie-break."""
    values = np.asarray(displacements, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != num_horizons:
        raise CalibrationError(
            f"displacements must be [continuations,{num_horizons}], got {values.shape}"
        )
    feasible = np.isfinite(values)
    if not feasible[:, 0].all():
        raise CalibrationError("every selected continuation must support the shortest horizon")
    grid = _higher_quantile_grid(values[feasible], candidate_quantiles)
    results: list[dict[str, Any]] = []
    labels_by_threshold: list[np.ndarray] = []
    last_feasible = feasible.sum(1) - 1
    for threshold in grid:
        _check_stop(stop_callback)
        crosses = feasible & (values > threshold)
        any_cross = crosses.any(1)
        first_cross = crosses.argmax(1)
        labels = np.where(any_cross, first_cross, last_feasible).astype(np.int64)
        hist = _histogram(labels, size=num_horizons)
        assert isinstance(hist, list)
        entropy = _normalized_entropy(hist)
        occupied = int(np.count_nonzero(hist))
        max_fraction = float(max(hist) / max(sum(hist), 1))
        results.append(
            {
                "threshold": float(threshold),
                "histogram": hist,
                "normalized_entropy": entropy,
                "occupied_classes": occupied,
                "max_class_fraction": max_fraction,
            }
        )
        labels_by_threshold.append(labels)

    # Highest entropy, then more occupied classes, lower largest-class fraction, then
    # the smaller threshold (conservative shorter-duration tie-break).
    chosen_index = min(
        range(len(results)),
        key=lambda i: (
            -results[i]["normalized_entropy"],
            -results[i]["occupied_classes"],
            results[i]["max_class_fraction"],
            results[i]["threshold"],
        ),
    )
    return {
        "rule": (
            "maximize normalized Shannon entropy; tie: occupied classes descending, "
            "max-class fraction ascending, threshold ascending"
        ),
        "candidate_quantiles": [float(q) for q in candidate_quantiles],
        "quantile_method": "higher",
        "threshold_boundary": "select first feasible horizon with displacement > threshold",
        "candidates": results,
        "chosen_index": int(chosen_index),
        "chosen_threshold": results[chosen_index]["threshold"],
        "chosen_histogram": results[chosen_index]["histogram"],
        "normalized_entropy": results[chosen_index]["normalized_entropy"],
        "occupied_classes": results[chosen_index]["occupied_classes"],
        "labels": labels_by_threshold[chosen_index],
    }


def _endpoint_linkage_distances(
    endpoint_sets: Sequence[np.ndarray], *, stop_callback: StopCallback = None
) -> tuple[list[np.ndarray], np.ndarray]:
    """Return average-linkage merge distances and pooled task-RMS pair distances."""
    from scipy.cluster.hierarchy import linkage
    from scipy.spatial.distance import pdist

    merges: list[np.ndarray] = []
    pooled: list[np.ndarray] = []
    for row, endpoints in enumerate(endpoint_sets):
        if row % 32 == 0:
            _check_stop(stop_callback)
        endpoints = np.asarray(endpoints, dtype=np.float64)
        if endpoints.ndim != 2 or endpoints.shape[1] < 1:
            raise CalibrationError("each endpoint set must be [continuations,metric_dim>=1]")
        if len(endpoints) <= 1:
            merges.append(np.empty(0, dtype=np.float64))
            continue
        distances = pdist(endpoints) / math.sqrt(endpoints.shape[1])
        if not np.isfinite(distances).all():
            raise CalibrationError("endpoint metric contains non-finite pair distances")
        pooled.append(distances)
        merges.append(linkage(distances, method="average")[:, 2])
    if not pooled:
        raise CalibrationError("no anchor has two endpoints for cluster calibration")
    return merges, np.concatenate(pooled)


def select_cluster_threshold(
    endpoint_sets: Sequence[np.ndarray],
    *,
    candidate_quantiles: Sequence[float],
    max_modes: int = 4,
    max_truncation_fraction: float = 0.05,
    target_mean_modes: float = 3.0,
    stop_callback: StopCallback = None,
) -> dict[str, Any]:
    """Select an average-linkage threshold under the preregistered mode-count rule."""
    merges, pooled = _endpoint_linkage_distances(
        endpoint_sets, stop_callback=stop_callback
    )
    grid = _higher_quantile_grid(pooled, candidate_quantiles)
    results: list[dict[str, Any]] = []
    raw_by_threshold: list[np.ndarray] = []
    sizes = np.asarray([len(endpoints) for endpoints in endpoint_sets], dtype=np.int64)
    for threshold in grid:
        _check_stop(stop_callback)
        raw = np.asarray(
            [
                int(size - np.searchsorted(distance, threshold, side="right"))
                for size, distance in zip(sizes, merges, strict=True)
            ],
            dtype=np.int64,
        )
        retained = np.minimum(raw, int(max_modes))
        truncation = float(np.mean(raw > max_modes))
        result = {
            "threshold": float(threshold),
            "raw_mode_histogram": _histogram(raw),
            "retained_mode_histogram": _histogram(retained),
            "mean_raw_modes": float(raw.mean()),
            "mean_retained_modes": float(retained.mean()),
            "median_retained_modes": float(np.median(retained)),
            "multimode_anchor_fraction": float(np.mean(retained >= 2)),
            "truncation_fraction": truncation,
        }
        results.append(result)
        raw_by_threshold.append(raw)

    feasible = [
        i for i, result in enumerate(results)
        if result["truncation_fraction"] <= max_truncation_fraction
    ]
    if feasible:
        chosen_index = min(
            feasible,
            key=lambda i: (
                abs(results[i]["mean_retained_modes"] - target_mean_modes),
                results[i]["truncation_fraction"],
                abs(results[i]["median_retained_modes"] - target_mean_modes),
                -results[i]["threshold"],
            ),
        )
        feasible_rule = True
    else:
        chosen_index = min(
            range(len(results)),
            key=lambda i: (
                results[i]["truncation_fraction"],
                abs(results[i]["mean_retained_modes"] - target_mean_modes),
                abs(results[i]["median_retained_modes"] - target_mean_modes),
                -results[i]["threshold"],
            ),
        )
        feasible_rule = False
    chosen = results[chosen_index]
    return {
        "rule": (
            "if any candidate has raw_modes>max_modes fraction <=0.05, minimize "
            "|mean retained-3|, truncation, |median retained-3|, then prefer larger "
            "threshold; otherwise minimize truncation before the same tie-breaks"
        ),
        "candidate_quantiles": [float(q) for q in candidate_quantiles],
        "quantile_method": "higher",
        "candidates": results,
        "had_truncation_feasible_candidate": feasible_rule,
        "chosen_index": int(chosen_index),
        "chosen_threshold": chosen["threshold"],
        "chosen_raw_mode_histogram": chosen["raw_mode_histogram"],
        "chosen_retained_mode_histogram": chosen["retained_mode_histogram"],
        "mean_retained_modes": chosen["mean_retained_modes"],
        "median_retained_modes": chosen["median_retained_modes"],
        "multimode_anchor_fraction": chosen["multimode_anchor_fraction"],
        "truncation_fraction": chosen["truncation_fraction"],
        "raw_modes": raw_by_threshold[chosen_index],
    }


def _eligible_retrievals(
    obs_norm: np.ndarray,
    index: Any,
    anchors: np.ndarray,
    pool_indices: np.ndarray,
    cfg: CalibrationConfig,
    stop_callback: StopCallback = None,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Query the bounded pool and apply the exact future-set eligibility filters."""
    from scipy.spatial import cKDTree

    _check_stop(stop_callback)
    tree = cKDTree(np.asarray(obs_norm[pool_indices]))
    _check_stop(stop_callback)
    query_k = min(cfg.query_k, len(pool_indices))
    distances, positions = tree.query(
        np.asarray(obs_norm[anchors]), k=query_k, workers=1
    )
    _check_stop(stop_callback)
    if query_k == 1:
        distances = np.asarray(distances)[:, None]
        positions = np.asarray(positions)[:, None]
    else:
        distances = np.asarray(distances)
        positions = np.asarray(positions)
    distances = distances / math.sqrt(obs_norm.shape[1])
    candidates = pool_indices[positions]
    min_horizon = min(cfg.horizons)
    candidate_remaining = _remaining_at(index, candidates.reshape(-1)).reshape(candidates.shape)
    candidate_traj = np.asarray(index.traj_id[candidates])
    anchor_traj = np.asarray(index.traj_id[anchors])[:, None]
    keep = (candidate_remaining >= min_horizon) & (candidates != anchors[:, None])
    same_trajectory = candidate_traj == anchor_traj
    too_close = np.abs(candidates.astype(np.int64) - anchors[:, None]) < cfg.time_exclusion
    keep &= ~(same_trajectory & too_close)

    eligible_indices: list[np.ndarray] = []
    eligible_distances: list[np.ndarray] = []
    counts = keep.sum(1).astype(np.int64)
    for row in range(len(anchors)):
        if row % 128 == 0:
            _check_stop(stop_callback)
        eligible_indices.append(candidates[row, keep[row]].astype(np.int64, copy=False))
        eligible_distances.append(distances[row, keep[row]].astype(np.float64, copy=False))
    return eligible_indices, eligible_distances, counts


def _retrieval_result_for_radius(
    anchors: np.ndarray,
    eligible_indices: Sequence[np.ndarray],
    eligible_distances: Sequence[np.ndarray],
    radius: float,
    nonself_limit: int,
    stop_callback: StopCallback = None,
) -> tuple[list[np.ndarray], dict[str, float]]:
    selected: list[np.ndarray] = []
    counts: list[int] = []
    fallbacks = 0
    for row, (anchor, indices, distances) in enumerate(
        zip(anchors, eligible_indices, eligible_distances, strict=True)
    ):
        if row % 128 == 0:
            _check_stop(stop_callback)
        alternatives = indices[distances <= radius][:nonself_limit]
        if len(alternatives) == 0:
            fallbacks += 1
        continuations = np.concatenate(
            [np.asarray([anchor], dtype=np.int64), alternatives.astype(np.int64, copy=False)]
        )
        selected.append(continuations)
        counts.append(len(continuations))
    count_array = np.asarray(counts, dtype=np.int64)
    return selected, {
        "radius": float(radius),
        "mean_retrieved": float(count_array.mean()),
        "min_retrieved": int(count_array.min()),
        "max_retrieved": int(count_array.max()),
        "retrieved_histogram": _histogram(count_array),
        "fallback_fraction": float(fallbacks / len(anchors)),
    }


def select_retrieval_radius(
    anchors: np.ndarray,
    eligible_indices: Sequence[np.ndarray],
    eligible_distances: Sequence[np.ndarray],
    cfg: CalibrationConfig,
    stop_callback: StopCallback = None,
) -> tuple[dict[str, Any], list[np.ndarray]]:
    """Select the preregistered support-count/fallback constrained RMS radius."""
    required = cfg.nonself_neighbors
    sufficient = np.asarray([len(values) >= required for values in eligible_distances])
    insufficient_fraction = float(1.0 - sufficient.mean())
    kth = np.asarray(
        [distances[required - 1] for distances, ok in zip(eligible_distances, sufficient) if ok],
        dtype=np.float64,
    )
    if len(kth) == 0:
        raise CalibrationError("no anchor has enough eligible retrieval candidates")
    nearest = np.asarray(
        [distances[0] for distances in eligible_distances if len(distances)],
        dtype=np.float64,
    )
    if len(nearest) == 0:
        raise CalibrationError("no anchor has an eligible non-self retrieval candidate")
    reports: list[dict[str, Any]] = []
    for quantile in cfg.radius_report_quantiles:
        radius = _higher_quantile(kth, quantile)
        _, result = _retrieval_result_for_radius(
            anchors,
            eligible_indices,
            eligible_distances,
            radius,
            required,
            stop_callback=stop_callback,
        )
        result["quantile"] = float(quantile)
        reports.append(result)
    support_radius = _higher_quantile(kth, cfg.radius_quantile)
    fallback_radius = _higher_quantile(nearest, cfg.fallback_radius_quantile)
    chosen_radius = max(support_radius, fallback_radius)
    selected, chosen = _retrieval_result_for_radius(
        anchors,
        eligible_indices,
        eligible_distances,
        chosen_radius,
        required,
        stop_callback=stop_callback,
    )
    return (
        {
            "rule": (
                f"max(higher q={cfg.radius_quantile:.2f} of the {required}th non-self "
                f"eligible neighbor RMS distance, higher q={cfg.fallback_radius_quantile:.2f} "
                "of nearest eligible non-self RMS distance)"
            ),
            "quantile_method": "higher",
            "support_count_component": {
                "quantile": float(cfg.radius_quantile),
                "order_statistic": int(required),
                "radius": float(support_radius),
                "distance_count": int(len(kth)),
                "distances_sha256": hashlib.sha256(
                    np.asarray(kth, dtype="<f8").tobytes(order="C")
                ).hexdigest(),
            },
            "fallback_component": {
                "quantile": float(cfg.fallback_radius_quantile),
                "order_statistic": 1,
                "radius": float(fallback_radius),
                "distance_count": int(len(nearest)),
                "distances_sha256": hashlib.sha256(
                    np.asarray(nearest, dtype="<f8").tobytes(order="C")
                ).hexdigest(),
            },
            "required_nonself_neighbors": required,
            "query_k": cfg.query_k,
            "sufficient_anchor_count": int(sufficient.sum()),
            "insufficient_anchor_count": int((~sufficient).sum()),
            "insufficient_anchor_fraction": insufficient_fraction,
            "maximum_insufficient_anchor_fraction": cfg.max_insufficient_neighbor_fraction,
            "kth_distance_count": int(len(kth)),
            "kth_distance_sha256": hashlib.sha256(
                np.asarray(kth, dtype="<f8").tobytes(order="C")
            ).hexdigest(),
            "report_candidates": reports,
            "chosen_radius": float(chosen_radius),
            "chosen": chosen,
        },
        selected,
    )


def _continuation_displacements(
    obs_norm: np.ndarray,
    index: Any,
    selected: Sequence[np.ndarray],
    task_metric_dims: np.ndarray,
    horizons: Sequence[int],
    stop_callback: StopCallback = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _check_stop(stop_callback)
    owners = np.concatenate(
        [np.full(len(values), row, dtype=np.int64) for row, values in enumerate(selected)]
    )
    continuations = np.concatenate(selected).astype(np.int64, copy=False)
    remaining = _remaining_at(index, continuations)
    values = np.full((len(continuations), len(horizons)), np.nan, dtype=np.float64)
    base = np.asarray(obs_norm[continuations])[:, task_metric_dims]
    scale = math.sqrt(len(task_metric_dims))
    for column, horizon in enumerate(horizons):
        _check_stop(stop_callback)
        valid = remaining >= int(horizon)
        endpoints = np.asarray(obs_norm[continuations[valid] + int(horizon)])[:, task_metric_dims]
        values[valid, column] = np.linalg.norm(endpoints - base[valid], axis=-1) / scale
    expected_feasible = remaining[:, None] >= np.asarray(horizons, dtype=np.int64)[None, :]
    if not np.isfinite(values[expected_feasible]).all():
        raise CalibrationError(
            "a trajectory-feasible task displacement is non-finite; refusing to "
            "reinterpret corrupt data as a shorter feasible horizon set"
        )
    return continuations, owners, values


def _metric_endpoints(
    obs_norm: np.ndarray,
    continuations: np.ndarray,
    owners: np.ndarray,
    anchors: np.ndarray,
    horizon_labels: np.ndarray,
    horizons: Sequence[int],
    task_metric_dims: np.ndarray,
    xy_dims: np.ndarray,
    relative_endpoints: bool,
) -> np.ndarray:
    lengths = np.asarray(horizons, dtype=np.int64)[horizon_labels]
    endpoints = np.asarray(obs_norm[continuations + lengths])[:, task_metric_dims].copy()
    if relative_endpoints:
        metric_positions = {int(dim): position for position, dim in enumerate(task_metric_dims)}
        overlap = [int(dim) for dim in xy_dims if int(dim) in metric_positions]
        if overlap:
            anchor_obs = np.asarray(obs_norm[anchors[owners]])
            continuation_obs = np.asarray(obs_norm[continuations])
            endpoint_obs = np.asarray(obs_norm[continuations + lengths])
            for dim in overlap:
                endpoints[:, metric_positions[dim]] = (
                    anchor_obs[:, dim] + endpoint_obs[:, dim] - continuation_obs[:, dim]
                )
    return endpoints.astype(np.float64, copy=False)


def _gate(value: float, *, rule: str, passed: bool) -> dict[str, Any]:
    return {"value": float(value), "rule": rule, "passed": bool(passed)}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def calibrate_future_metrics(
    obs_norm: np.ndarray,
    index: Any,
    *,
    setting_id: str,
    train_manifest_sha256: str,
    normalizer_sha256: str,
    xy_dims: Sequence[int],
    task_metric_dims: Sequence[int] | None = None,
    relative_endpoints: bool,
    config: CalibrationConfig | None = None,
    output_path: str | Path | None = None,
    enforce_gates: bool = True,
    stop_callback: StopCallback = None,
) -> dict[str, Any]:
    """Calibrate and optionally atomically publish one immutable setting contract."""
    cfg = config or CalibrationConfig()
    for name, digest in (
        ("train_manifest_sha256", train_manifest_sha256),
        ("normalizer_sha256", normalizer_sha256),
    ):
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise CalibrationError(f"{name} must be 64 lowercase hex characters")
    obs_norm = np.asarray(obs_norm) if not hasattr(obs_norm, "shape") else obs_norm
    if len(obs_norm.shape) != 2 or obs_norm.shape[0] < 1 or obs_norm.shape[1] < 1:
        raise CalibrationError("obs_norm must be a non-empty [transitions,obs_dim] array")
    xy = np.asarray(tuple(int(dim) for dim in xy_dims), dtype=np.int64)
    metric = np.asarray(
        tuple(int(dim) for dim in (xy_dims if task_metric_dims is None else task_metric_dims)),
        dtype=np.int64,
    )
    if len(metric) == 0 or len(np.unique(metric)) != len(metric):
        raise CalibrationError("task_metric_dims must be non-empty and unique")
    if metric.min() < 0 or metric.max() >= obs_norm.shape[1]:
        raise CalibrationError("task_metric_dims lie outside obs_norm")
    if len(xy) and (xy.min() < 0 or xy.max() >= obs_norm.shape[1]):
        raise CalibrationError("xy_dims lie outside obs_norm")

    # The fixed seed is deliberately independent of the combined cache/source manifest,
    # whose inventory also contains validation files. Validation-only drift must never
    # alter selected train anchors or any calibrated threshold.
    _check_stop(stop_callback)
    sample_seed = int(cfg.anchor_sample_seed)
    anchors, anchor_population = sample_valid_anchors(
        index,
        min_horizon=min(cfg.horizons),
        sample_size=cfg.sample_size,
        seed=sample_seed,
    )
    pool_size = min(int(cfg.retrieval_pool), int(obs_norm.shape[0]))
    pool_indices = bounded_uniform_indices(
        int(obs_norm.shape[0]), pool_size, seed=cfg.retrieval_pool_seed
    )
    eligible_indices, eligible_distances, eligible_counts = _eligible_retrievals(
        obs_norm, index, anchors, pool_indices, cfg, stop_callback=stop_callback
    )
    radius, selected = select_retrieval_radius(
        anchors,
        eligible_indices,
        eligible_distances,
        cfg,
        stop_callback=stop_callback,
    )

    continuations, owners, displacements = _continuation_displacements(
        obs_norm,
        index,
        selected,
        metric,
        cfg.horizons,
        stop_callback=stop_callback,
    )
    horizon = select_horizon_threshold(
        displacements,
        candidate_quantiles=cfg.horizon_candidate_quantiles,
        num_horizons=len(cfg.horizons),
        stop_callback=stop_callback,
    )
    horizon_labels = horizon.pop("labels")
    endpoints = _metric_endpoints(
        obs_norm,
        continuations,
        owners,
        anchors,
        horizon_labels,
        cfg.horizons,
        metric,
        xy,
        relative_endpoints,
    )
    # Selected continuations were concatenated anchor-by-anchor, so slice their
    # contiguous ranges. Repeated boolean scans would be O(num_anchors*num_continuations)
    # at the formal 4096 x ~98k calibration size.
    selected_counts = np.asarray([len(values) for values in selected], dtype=np.int64)
    selected_offsets = np.concatenate([[0], np.cumsum(selected_counts, dtype=np.int64)])
    endpoint_sets = [
        endpoints[int(selected_offsets[row]) : int(selected_offsets[row + 1])]
        for row in range(len(anchors))
    ]
    cluster = select_cluster_threshold(
        endpoint_sets,
        candidate_quantiles=cfg.cluster_candidate_quantiles,
        max_modes=cfg.max_modes,
        max_truncation_fraction=cfg.max_truncation_fraction,
        stop_callback=stop_callback,
    )
    cluster.pop("raw_modes")

    mean_retrieved = radius["chosen"]["mean_retrieved"]
    fallback_fraction = radius["chosen"]["fallback_fraction"]
    gates = {
        "eligible_23rd_neighbor": _gate(
            radius["insufficient_anchor_fraction"],
            rule=f"<= {cfg.max_insufficient_neighbor_fraction}",
            passed=radius["insufficient_anchor_fraction"] <= cfg.max_insufficient_neighbor_fraction,
        ),
        "mean_retrieved": _gate(
            mean_retrieved,
            rule=f">= {cfg.min_mean_retrieved}",
            passed=mean_retrieved >= cfg.min_mean_retrieved,
        ),
        "retrieval_fallback_fraction": _gate(
            fallback_fraction,
            rule=f"<= {cfg.max_retrieval_fallback_fraction}",
            passed=fallback_fraction <= cfg.max_retrieval_fallback_fraction,
        ),
        "normalized_horizon_entropy": _gate(
            horizon["normalized_entropy"],
            rule=f">= {cfg.min_normalized_horizon_entropy}",
            passed=horizon["normalized_entropy"] >= cfg.min_normalized_horizon_entropy,
        ),
        "occupied_horizon_classes": _gate(
            horizon["occupied_classes"],
            rule=f">= {cfg.min_occupied_horizon_classes}",
            passed=horizon["occupied_classes"] >= cfg.min_occupied_horizon_classes,
        ),
        "mean_retained_modes": _gate(
            cluster["mean_retained_modes"],
            rule=(
                f"in [{cfg.min_mean_retained_modes}, {cfg.max_mean_retained_modes}]"
            ),
            passed=(
                cfg.min_mean_retained_modes
                <= cluster["mean_retained_modes"]
                <= cfg.max_mean_retained_modes
            ),
        ),
        "multimode_anchor_fraction": _gate(
            cluster["multimode_anchor_fraction"],
            rule=f">= {cfg.min_multimode_anchor_fraction}",
            passed=cluster["multimode_anchor_fraction"] >= cfg.min_multimode_anchor_fraction,
        ),
        "mode_truncation_fraction": _gate(
            cluster["truncation_fraction"],
            rule=f"<= {cfg.max_truncation_fraction}",
            passed=cluster["truncation_fraction"] <= cfg.max_truncation_fraction,
        ),
    }

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "implementation_sha256": implementation_sha256(),
        "numpy_version": np.__version__,
        "scipy_version": __import__("scipy").__version__,
        "status": "complete" if all(result["passed"] for result in gates.values()) else "failed",
        "setting_id": setting_id,
        "split": "train",
        "train_manifest_sha256": train_manifest_sha256,
        "normalizer_sha256": normalizer_sha256,
        "input_units": "training-normalizer standardized observations",
        "metric_mode": "rms_v2",
        "relative_endpoints": bool(relative_endpoints),
        "xy_dims": xy.astype(int).tolist(),
        "task_metric_dims": metric.astype(int).tolist(),
        "config": asdict(cfg),
        "anchor_population": int(anchor_population),
        "sample_seed": int(sample_seed),
        "sample_indices": anchors.astype(int).tolist(),
        "sample_indices_sha256": _indices_sha256(anchors),
        "retrieval_pool_indices": pool_indices.astype(int).tolist(),
        "retrieval_pool_indices_sha256": _indices_sha256(pool_indices),
        "eligible_query_count_histogram": _histogram(eligible_counts),
        "selected_continuation_count": int(len(continuations)),
        "selected_continuation_indices": continuations.astype(int).tolist(),
        "selected_continuation_indices_sha256": _indices_sha256(continuations),
        "selected_continuation_offsets": selected_offsets.astype(int).tolist(),
        "retrieval": radius,
        "horizon": horizon,
        "cluster": cluster,
        "chosen": {
            "retrieval_radius": radius["chosen_radius"],
            "displacement_threshold": horizon["chosen_threshold"],
            "cluster_threshold": cluster["chosen_threshold"],
        },
        "gates": gates,
    }
    # Return exactly the JSON representation that is hashed and persisted (not an
    # in-memory mixture of tuples/lists whose equality changes after a round trip).
    payload = json.loads(_canonical_json(payload))
    payload["contract_sha256"] = stable_hash(payload)
    _check_stop(stop_callback)
    if output_path is not None:
        _atomic_json(Path(output_path), payload)
    if enforce_gates and payload["status"] != "complete":
        raise CalibrationGateError(payload)
    return payload


def validate_contract(
    payload: dict[str, Any],
    *,
    require_complete: bool = True,
    expected_config: CalibrationConfig | None = None,
    expected_setting_id: str | None = None,
    expected_train_manifest_sha256: str | None = None,
    expected_normalizer_sha256: str | None = None,
    expected_xy_dims: Sequence[int] | None = None,
    expected_task_metric_dims: Sequence[int] | None = None,
    expected_relative_endpoints: bool | None = None,
) -> None:
    """Dependency-light fail-closed validation for campaign/dispatcher consumers."""
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise CalibrationError("unsupported calibration schema")
    if payload.get("algorithm_version") != ALGORITHM_VERSION:
        raise CalibrationError("calibration algorithm version drifted")
    if payload.get("implementation_sha256") != implementation_sha256():
        raise CalibrationError("calibration implementation/source fingerprint drifted")
    claimed = payload.get("contract_sha256")
    if not isinstance(claimed, str) or len(claimed) != 64:
        raise CalibrationError("calibration contract SHA256 is missing")
    body = dict(payload)
    body.pop("contract_sha256", None)
    if stable_hash(body) != claimed:
        raise CalibrationError("calibration contract content hash mismatch")
    if require_complete and payload.get("status") != "complete":
        raise CalibrationError("calibration contract did not pass all hard gates")
    if payload.get("metric_mode") != "rms_v2" or payload.get("split") != "train":
        raise CalibrationError("calibration metric/split contract drifted")
    # Default validation is the exact formal protocol, not merely any self-consistent
    # relaxed test configuration. Callers using a synthetic config must provide it.
    locked_config = json.loads(_canonical_json(asdict(expected_config or CalibrationConfig())))
    if payload.get("config") != locked_config:
        raise CalibrationError("calibration config/rules do not match the locked protocol")
    sample_indices = np.asarray(payload.get("sample_indices", ()), dtype=np.int64)
    pool_indices = np.asarray(payload.get("retrieval_pool_indices", ()), dtype=np.int64)
    continuation_indices = np.asarray(
        payload.get("selected_continuation_indices", ()), dtype=np.int64
    )
    if len(sample_indices) != int(locked_config["sample_size"]):
        raise CalibrationError("calibration sample does not have the locked size")
    for name, values in (
        ("sample", sample_indices),
        ("retrieval_pool", pool_indices),
        ("selected_continuation", continuation_indices),
    ):
        if _indices_sha256(values) != payload.get(f"{name}_indices_sha256"):
            raise CalibrationError(f"calibration {name} index hash mismatch")
    offsets = np.asarray(payload.get("selected_continuation_offsets", ()), dtype=np.int64)
    if (
        len(offsets) != len(sample_indices) + 1
        or len(offsets) == 0
        or int(offsets[0]) != 0
        or int(offsets[-1]) != len(continuation_indices)
        or np.any(np.diff(offsets) < 1)
    ):
        raise CalibrationError("calibration continuation offsets are inconsistent")
    expected_fields = {
        "setting_id": expected_setting_id,
        "train_manifest_sha256": expected_train_manifest_sha256,
        "normalizer_sha256": expected_normalizer_sha256,
        "xy_dims": None if expected_xy_dims is None else [int(x) for x in expected_xy_dims],
        "task_metric_dims": (
            None if expected_task_metric_dims is None
            else [int(x) for x in expected_task_metric_dims]
        ),
        "relative_endpoints": expected_relative_endpoints,
    }
    for key, expected in expected_fields.items():
        if expected is not None and payload.get(key) != expected:
            raise CalibrationError(f"calibration {key} does not match the live setting contract")
    gates = payload.get("gates") or {}
    if require_complete and (not gates or not all(item.get("passed") is True for item in gates.values())):
        raise CalibrationError("one or more calibration gates are not satisfied")
    chosen = payload.get("chosen") or {}
    if (
        chosen.get("retrieval_radius") != (payload.get("retrieval") or {}).get("chosen_radius")
        or chosen.get("displacement_threshold")
        != (payload.get("horizon") or {}).get("chosen_threshold")
        or chosen.get("cluster_threshold")
        != (payload.get("cluster") or {}).get("chosen_threshold")
    ):
        raise CalibrationError("calibration chosen thresholds disagree with candidate records")
