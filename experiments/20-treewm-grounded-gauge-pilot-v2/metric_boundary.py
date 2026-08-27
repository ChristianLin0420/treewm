#!/usr/bin/env python3
"""Recover a committed-but-unflushed Exp20 MetricTracker log boundary.

The shared trainer checks a lifecycle signal immediately after committing an optimizer
update and immediately before its 50-update logging block.  If those events coincide,
the exact-resume checkpoint correctly contains the complete MetricTracker window, but
the TensorBoard point for that boundary has not yet been published.  This module makes
that narrow case transactional without changing model, optimizer, scheduler, loader,
or RNG state.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from campaign import ContractError, atomic_json, file_sha256, read_json, require, stable_hash


LOG_EVERY = 50


def _artifact_body(path: Path, hash_key: str) -> dict[str, Any]:
    value = read_json(path)
    claimed = value.get(hash_key)
    body = dict(value)
    body.pop(hash_key, None)
    require(claimed == stable_hash(body), f"metric-boundary artifact hash differs: {path}")
    return value


def _histogram_summary(histograms: Mapping[str, np.ndarray]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, values in sorted(histograms.items()):
        array = np.asarray(values, dtype=np.float32).ravel()
        require(array.size > 0 and np.isfinite(array).all(), f"{name}: invalid saved histogram")
        result[name] = {
            "samples": int(array.size),
            "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        }
    return result


def _load_checkpoint(path: Path) -> dict[str, Any]:
    import torch

    require(path.is_file() and not path.is_symlink(), f"checkpoint missing/symlinked: {path}")
    try:
        try:
            value = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
        except TypeError:
            value = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise ContractError(f"metric-boundary checkpoint cannot be loaded: {exc}") from exc
    require(isinstance(value, dict), "metric-boundary checkpoint is not a mapping")
    return value


def _tracker(payload: Mapping[str, Any]):
    from treewm.logging.metrics import MetricTracker

    rank_states = payload.get("rank_states") or []
    require(
        int((payload.get("run_identity") or {}).get("world_size", -1)) == 1
        and len(rank_states) == 1
        and isinstance(rank_states[0], Mapping),
        "metric-boundary recovery requires one complete rank state",
    )
    tracker = MetricTracker()
    try:
        tracker.load_state_dict(rank_states[0].get("metric_tracker"))
    except (TypeError, ValueError) as exc:
        raise ContractError(f"metric-boundary tracker state is invalid: {exc}") from exc
    return tracker


def _tracker_is_empty(tracker: Any) -> bool:
    state = tracker.state_dict()
    return not state["sums"] and not state["counts"] and not state["hists"]


def _write_event(run_dir: Path, step: int, metrics: Mapping[str, float], histograms: Mapping[str, np.ndarray]) -> Path:
    from torch.utils.tensorboard import SummaryWriter

    before = set(run_dir.glob("events.out.tfevents.*"))
    writer = SummaryWriter(
        log_dir=str(run_dir),
        filename_suffix=f".exp20-metric-boundary-{step}",
        flush_secs=1,
    )
    try:
        for tag, value in sorted(metrics.items()):
            require(math.isfinite(float(value)), f"{tag}: recovered scalar is nonfinite")
            writer.add_scalar(tag, float(value), step)
        for tag, values in sorted(histograms.items()):
            writer.add_histogram(tag, np.asarray(values, dtype=np.float32), step)
        writer.flush()
    finally:
        writer.close()
    created = set(run_dir.glob("events.out.tfevents.*")) - before
    require(len(created) == 1, "metric-boundary recovery did not create exactly one event file")
    event_path = next(iter(created))
    require(event_path.is_file() and not event_path.is_symlink(), "recovery event is missing/symlinked")
    return event_path


def _save_checkpoint(path: Path, payload: Mapping[str, Any]) -> None:
    import torch

    temporary = path.with_name(f".{path.name}.metric-boundary.tmp.{os.getpid()}")
    require(not temporary.exists(), f"stale metric-boundary checkpoint temporary exists: {temporary}")
    try:
        torch.save(dict(payload), temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def recover_metric_boundary(
    checkpoint_path: Path,
    run_dir: Path,
    launch: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish/reset a complete tracker window iff requeue hit an exact log boundary."""
    payload = _load_checkpoint(checkpoint_path)
    completed = int(payload.get("completed_updates", -1))
    reason = str(payload.get("reason", ""))
    tracker = _tracker(payload)
    plan_path = run_dir / "stage-gates" / f"METRIC_BOUNDARY_EVENT_{completed}.json"
    complete_path = run_dir / "stage-gates" / f"METRIC_BOUNDARY_RECOVERED_{completed}.json"

    if complete_path.exists():
        complete = _artifact_body(complete_path, "recovery_sha256")
        require(_tracker_is_empty(tracker), "completed recovery retained a nonempty tracker")
        require(complete.get("launch_sha256") == launch["launch_sha256"], "recovery launch differs")
        return {
            **complete,
            "status": "metric_boundary_previously_recovered",
            "current_checkpoint_sha256": file_sha256(checkpoint_path),
        }

    if completed <= 0 or completed % LOG_EVERY != 0 or not reason.startswith("graceful-stop:"):
        require(not plan_path.exists(), "metric-boundary plan exists for an ineligible checkpoint")
        return {
            "schema_version": 1,
            "status": "no_metric_boundary_recovery_needed",
            "completed_updates": completed,
            "launch_sha256": launch["launch_sha256"],
        }

    if _tracker_is_empty(tracker):
        if plan_path.exists():
            plan = _artifact_body(plan_path, "event_plan_sha256")
            require(plan.get("launch_sha256") == launch["launch_sha256"], "planned recovery launch differs")
            event_path = run_dir / str(plan.get("event_file", ""))
            require(event_path.is_file() and not event_path.is_symlink(), "planned recovery event is missing/symlinked")
            require(plan.get("event_file_sha256") == file_sha256(event_path), "planned recovery event bytes differ")
            complete = {
                "schema_version": 1,
                "status": "metric_boundary_recovered",
                "campaign_id": launch["campaign_id"],
                "run_name": launch["run"]["run_name"],
                "completed_updates": completed,
                "launch_sha256": launch["launch_sha256"],
                "event_plan_sha256": plan["event_plan_sha256"],
                "checkpoint_before_sha256": plan["checkpoint_before_sha256"],
                "checkpoint_after_sha256": file_sha256(checkpoint_path),
            }
            complete["recovery_sha256"] = stable_hash(complete)
            atomic_json(complete_path, complete)
            return complete
        return {
            "schema_version": 1,
            "status": "exact_boundary_already_flushed",
            "completed_updates": completed,
            "launch_sha256": launch["launch_sha256"],
        }

    metrics = tracker.compute(reduce=False)
    histograms = tracker.histograms()
    required = set(launch["metric_boundary_required_tags"])
    require(required <= set(metrics), "saved boundary tracker lacks required gate scalars")
    require(all(math.isfinite(float(value)) for value in metrics.values()), "saved boundary scalar is nonfinite")
    histogram_summary = _histogram_summary(histograms)
    checkpoint_before = file_sha256(checkpoint_path)

    if plan_path.exists():
        plan = _artifact_body(plan_path, "event_plan_sha256")
        require(plan.get("checkpoint_before_sha256") == checkpoint_before, "planned recovery checkpoint bytes differ")
        require(plan.get("metrics") == metrics, "planned recovery scalars differ")
        require(plan.get("histograms") == histogram_summary, "planned recovery histograms differ")
        event_path = run_dir / str(plan.get("event_file", ""))
        require(event_path.is_file() and not event_path.is_symlink(), "planned recovery event is missing/symlinked")
        require(plan.get("event_file_sha256") == file_sha256(event_path), "planned recovery event bytes differ")
    else:
        event_path = _write_event(run_dir, completed, metrics, histograms)
        plan = {
            "schema_version": 1,
            "status": "metric_boundary_event_published",
            "campaign_id": launch["campaign_id"],
            "run_name": launch["run"]["run_name"],
            "completed_updates": completed,
            "launch_sha256": launch["launch_sha256"],
            "checkpoint_before_sha256": checkpoint_before,
            "metrics": metrics,
            "histograms": histogram_summary,
            "event_file": event_path.relative_to(run_dir).as_posix(),
            "event_file_sha256": file_sha256(event_path),
        }
        plan["event_plan_sha256"] = stable_hash(plan)
        atomic_json(plan_path, plan)

    from treewm.logging.metrics import MetricTracker

    rank_states = payload["rank_states"]
    rank_states[0]["metric_tracker"] = MetricTracker().state_dict()
    _save_checkpoint(checkpoint_path, payload)
    repaired = _load_checkpoint(checkpoint_path)
    require(_tracker_is_empty(_tracker(repaired)), "repaired checkpoint tracker is not empty")
    complete = {
        "schema_version": 1,
        "status": "metric_boundary_recovered",
        "campaign_id": launch["campaign_id"],
        "run_name": launch["run"]["run_name"],
        "completed_updates": completed,
        "launch_sha256": launch["launch_sha256"],
        "event_plan_sha256": plan["event_plan_sha256"],
        "checkpoint_before_sha256": checkpoint_before,
        "checkpoint_after_sha256": file_sha256(checkpoint_path),
    }
    complete["recovery_sha256"] = stable_hash(complete)
    atomic_json(complete_path, complete)
    return complete
