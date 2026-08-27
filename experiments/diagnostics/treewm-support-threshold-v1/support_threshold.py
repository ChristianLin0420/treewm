#!/usr/bin/env python3
"""Read-only KEEP-threshold diagnostic for paused grounded-formal checkpoints.

This program deliberately reconstructs the model and data through the immutable
source snapshot named by ``FORMAL_LAUNCH.json``.  It reads a fixed stratified sample
from the already-published future recipe, computes the exact level-one Hungarian KEEP
targets, and writes a content-addressed JSON result outside the formal run root.

It never calls a cache/recipe builder and refuses any output path inside the formal
campaign tree.  The protected run tree is stat-fingerprinted before and after scoring
so an accidental run-directory mutation fails closed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence


SCHEMA_VERSION = 1
DIAGNOSTIC_ID = "treewm_support_threshold_diagnostic_v1"
CAMPAIGN_ID = "treewm-grounded-formal-v1"
ALLOWED_SETTINGS = ("puzzle-3x3", "puzzle-4x4-100m")
DEFAULT_SAMPLE_SIZE = 4096
DEFAULT_BATCH_SIZE = 256
DEFAULT_STAGE_TARGET = 25_000
THRESHOLD_START_MILLI = 250
THRESHOLD_STOP_MILLI = 550
THRESHOLD_STEP_MILLI = 25


class DiagnosticError(RuntimeError):
    """The read-only diagnostic contract is incomplete or inconsistent."""


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


def read_json(path: str | Path) -> dict[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DiagnosticError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=reject_duplicates)
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"cannot read exact JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DiagnosticError(f"JSON artifact is not an object: {path}")
    return value


def threshold_grid() -> tuple[float, ...]:
    """The preregistered inclusive 0.25--0.55 grid, without float-step drift."""
    return tuple(
        milli / 1000.0
        for milli in range(
            THRESHOLD_START_MILLI,
            THRESHOLD_STOP_MILLI + 1,
            THRESHOLD_STEP_MILLI,
        )
    )


def _u64_hash(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("ascii")).digest()[:8], "little")


def fixed_stratified_positions(population: int, sample_size: int) -> "Any":
    """Select exactly one deterministic element from each equal-width stratum.

    Counter hashing, rather than a library RNG, fixes the sample across NumPy/PyTorch
    releases.  The returned little-endian int64 bytes are part of every artifact.
    """
    import numpy as np

    population = int(population)
    sample_size = int(sample_size)
    if population <= 0 or sample_size <= 0:
        raise ValueError("population and sample_size must be positive")
    if sample_size > population:
        raise ValueError("sample_size cannot exceed the recipe population")
    positions = np.empty(sample_size, dtype=np.int64)
    for index in range(sample_size):
        start = (index * population) // sample_size
        stop = ((index + 1) * population) // sample_size
        width = stop - start
        if width <= 0:
            raise AssertionError("empty representative-sample stratum")
        offset = _u64_hash(f"{DIAGNOSTIC_ID}:stratum:{index}") % width
        positions[index] = start + offset
    if sample_size > 1 and not bool((positions[1:] > positions[:-1]).all()):
        raise AssertionError("representative positions are not strictly increasing")
    return positions


def array_sha256(array: "Any", dtype: str) -> str:
    import numpy as np

    canonical = np.asarray(array, dtype=np.dtype(dtype)).astype(dtype, copy=False)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def homogeneous_nodes_proxy(admitted_children: "Any", max_depth: int, node_budget: int) -> "Any":
    """Static node proxy using each root's admitted width at every later depth.

    This is intentionally named a proxy: child-state KEEP scores are not available in
    a level-one sweep.  Top-one fallback is applied before the homogeneous expansion,
    and the result is capped at the checkpoint's node budget.
    """
    import numpy as np

    width = np.maximum(np.asarray(admitted_children, dtype=np.int64), 1)
    total = np.ones_like(width)
    level = np.ones_like(width)
    for _ in range(int(max_depth)):
        level = level * width
        total = total + level
    return np.minimum(total, int(node_budget))


def threshold_metrics(
    scores: "Any",
    labels: "Any",
    threshold: float,
    *,
    max_depth: int,
    node_budget: int,
) -> dict[str, Any]:
    import numpy as np

    score = np.asarray(scores, dtype=np.float32)
    target = np.asarray(labels, dtype=np.bool_)
    if score.shape != target.shape or score.ndim != 2:
        raise ValueError("scores and labels must have identical [anchors, branches] shape")
    if not math.isfinite(float(threshold)) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("threshold must be finite and within [0, 1]")
    predicted = score >= float(threshold)
    tp = int(np.logical_and(predicted, target).sum(dtype=np.int64))
    fp = int(np.logical_and(predicted, ~target).sum(dtype=np.int64))
    fn = int(np.logical_and(~predicted, target).sum(dtype=np.int64))
    tn = int(np.logical_and(~predicted, ~target).sum(dtype=np.int64))
    raw_children = predicted.sum(axis=1, dtype=np.int64)
    admitted = np.maximum(raw_children, 1)
    nodes = homogeneous_nodes_proxy(admitted, int(max_depth), int(node_budget))
    return {
        "threshold": float(threshold),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
        "support_recall": _safe_ratio(tp, tp + fn),
        "support_precision": _safe_ratio(tp, tp + fp),
        "raw_keep_rate": float(predicted.mean(dtype=np.float64)),
        "raw_kept_children_per_anchor": float(raw_children.mean(dtype=np.float64)),
        "root_children_after_top1_fallback_mean": float(admitted.mean(dtype=np.float64)),
        "root_top1_fallback_fraction": float((raw_children == 0).mean(dtype=np.float64)),
        "root_multi_child_fraction": float((admitted >= 2).mean(dtype=np.float64)),
        "homogeneous_full_depth_nodes_proxy_mean": float(nodes.mean(dtype=np.float64)),
        "homogeneous_full_depth_node_budget_fraction": float(
            (nodes >= int(node_budget)).mean(dtype=np.float64)
        ),
    }


def protected_tree_fingerprint(root: str | Path) -> str:
    """Hash metadata for every run-tree entry without reading or modifying content."""
    base = Path(root).resolve()
    if not base.is_dir() or base.is_symlink():
        raise DiagnosticError(f"formal run directory is unavailable or symlinked: {base}")
    entries: list[dict[str, Any]] = []
    for path in sorted(base.rglob("*"), key=lambda value: str(value.relative_to(base))):
        stat = path.lstat()
        entries.append(
            {
                "path": str(path.relative_to(base)),
                "kind": "symlink" if path.is_symlink() else "directory" if path.is_dir() else "file",
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "mode": int(stat.st_mode),
            }
        )
    return stable_hash(entries)


def validate_output_root(output_root: str | Path, formal_root: str | Path) -> Path:
    destination = Path(output_root).expanduser().resolve()
    protected = Path(formal_root).expanduser().resolve()
    if destination == protected or destination.is_relative_to(protected):
        raise DiagnosticError(
            f"diagnostic output must be outside the formal campaign tree: {protected}"
        )
    return destination


def write_immutable_json(output_root: str | Path, body: Mapping[str, Any]) -> tuple[Path, bool]:
    """Exclusively publish a content-addressed, self-hashed, read-only JSON object."""
    destination = Path(output_root)
    payload = dict(body)
    payload["artifact_sha256"] = stable_hash(payload)
    setting = str(payload["run"]["setting_id"])
    seed = int(payload["run"]["seed"])
    step = int(payload["checkpoint"]["completed_updates"])
    filename = (
        f"support-threshold__{setting}__seed-{seed}__step-{step}__"
        f"{payload['artifact_sha256'][:16]}.json"
    )
    path = destination / filename
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    destination.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        if path.read_bytes() != encoded:
            raise DiagnosticError(f"existing content-addressed artifact differs: {path}")
        return path, True
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    try:
        directory_fd = os.open(destination, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        pass
    return path, False


@contextlib.contextmanager
def patched_environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update({key: str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _snapshot_root_from_launch(launch: Mapping[str, Any]) -> Path:
    argv = launch.get("argv") or []
    if len(argv) < 2:
        raise DiagnosticError("formal launch lacks the pinned trainer command")
    trainer = Path(str(argv[1])).expanduser().resolve()
    if trainer.name != "train.py" or trainer.parent.name != "scripts" or not trainer.is_file():
        raise DiagnosticError(f"formal launch trainer is unavailable: {trainer}")
    root = trainer.parents[1]
    if root.name != "repo":
        raise DiagnosticError(f"formal launch does not name a sealed snapshot repo: {root}")
    return root


def _activate_snapshot(snapshot_root: Path) -> None:
    """Make all scientific imports resolve only from the sealed source snapshot."""
    sys.dont_write_bytecode = True
    snapshot_text = str(snapshot_root)
    while snapshot_text in sys.path:
        sys.path.remove(snapshot_text)
    sys.path.insert(0, snapshot_text)
    package = snapshot_root / "experiments" / "14-treewm-grounded-formal-v1"
    package_text = str(package)
    while package_text in sys.path:
        sys.path.remove(package_text)
    sys.path.insert(0, package_text)
    for name in tuple(sys.modules):
        if name == "treewm" or name.startswith("treewm.") or name in {"campaign", "worker"}:
            del sys.modules[name]
    importlib.invalidate_caches()


def _find_exact_cache_manifest(
    cache_root: Path,
    *,
    source_manifest_sha256: str,
    source_name: str,
) -> tuple[Path, dict[str, Any]]:
    matches: list[tuple[Path, dict[str, Any]]] = []
    if not cache_root.is_dir():
        raise DiagnosticError(f"formal cache root is unavailable: {cache_root}")
    for path in sorted(cache_root.glob("*/manifest.json"), key=str):
        payload = read_json(path)
        named = payload.get("dataset") or payload.get("dataset_name")
        if (
            named == source_name
            and payload.get("source_manifest_sha256") == source_manifest_sha256
        ):
            matches.append((path, payload))
    if len(matches) != 1:
        raise DiagnosticError(
            f"expected one sealed cache manifest for {source_name}/{source_manifest_sha256}, "
            f"found {len(matches)}"
        )
    manifest_path, manifest = matches[0]
    required = ("train_obs.npy", "train_act.npy", "train_obs_norm.npy", "train_act_norm.npy")
    missing = [name for name in required if not (manifest_path.parent / name).is_file()]
    if missing:
        raise DiagnosticError(f"sealed cache is incomplete ({', '.join(missing)}): {manifest_path}")
    return manifest_path, manifest


def _load_dataset_read_only(cfg: Any, launch: Mapping[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Open existing mmap cache and recipe directly; never enter builder code paths."""
    import numpy as np
    from dataclasses import replace
    from treewm.data.ogbench_dataset import (
        ChunkDataset,
        Normalizer,
        _attach_future_recipes_if_requested,
    )
    from treewm.data.future_recipe import normalizer_state_sha256
    from treewm.utils import config as cfg_utils

    environment = launch.get("environment") or {}
    source_sha = str(environment.get("TREEWM_DATA_SHA256", ""))
    cache_root = Path(str(environment.get("TREEWM_CACHE", ""))).expanduser().resolve()
    source_name = str(cfg.env.get("source_name", cfg.env.name))
    manifest_path, cache_manifest = _find_exact_cache_manifest(
        cache_root,
        source_manifest_sha256=source_sha,
        source_name=source_name,
    )
    if str(cfg.env.get("dataset_kind", "standard")) == "sharded_100m_full":
        from treewm.data.sharded_ogbench import _load_cache

        cache = _load_cache(manifest_path.parent, cache_manifest, was_hit=True)
    else:
        from treewm.data.shared_cache import _cache_from_manifest

        cache = _cache_from_manifest(manifest_path.parent, cache_manifest, was_hit=True)
    if cache.source_manifest_sha256 != source_sha:
        raise DiagnosticError("opened cache identity differs from FORMAL_LAUNCH.json")
    normalizer = Normalizer.from_state_dict(cache.norm_stats)
    future_cfg = cfg_utils.future_set_config(cfg)
    if cfg.env.get("relative_endpoints") is not None:
        future_cfg = replace(
            future_cfg, relative_endpoints=bool(cfg.env.get("relative_endpoints"))
        )
    common = {
        "xy_dims": tuple(cfg.env.xy_dims),
        "cache_future_sets": False,
        "task_metric_dims": tuple(cfg.env.get("task_metric_dims") or cfg.env.xy_dims),
    }
    # published_union replaces anchors from the sealed recipes below.  Starting from
    # zero anchors avoids allocating/sampling a million-entry temporary selection.
    train_ds = ChunkDataset(
        {}, normalizer, future_cfg, max_anchors=0, seed=int(cfg.seed), shared=cache.train, **common
    )
    val_ds = ChunkDataset(
        {}, normalizer, future_cfg, max_anchors=0, seed=int(cfg.seed) + 1, shared=cache.val, **common
    )
    train_ds.cache_metrics = cache.assert_consumed_by(train_ds, val_ds)
    _attach_future_recipes_if_requested(
        train_ds,
        val_ds,
        normalizer,
        future_cfg,
        source_sha,
        anchor_policy=str(cfg.future_sets.get("recipe_anchor_policy", "selected_seed")),
    )
    if str(getattr(train_ds, "future_recipe_sha256", "")) != str(
        environment.get("TREEWM_FUTURE_RECIPE_SHA256", "")
    ):
        raise DiagnosticError("loaded train recipe identity differs from formal launch")
    records_path = (
        train_ds.future_recipe.root
        / str(train_ds.future_recipe.manifest["records_file"])
    )
    records_sha256 = file_sha256(records_path)
    if records_sha256 != str(train_ds.future_recipe.manifest["records_sha256"]):
        raise DiagnosticError("train future-recipe records content hash differs")
    anchor_array = np.asarray(train_ds.anchors, dtype="<i8")
    return train_ds, {
        "split": "train",
        "population": int(len(train_ds)),
        "source_manifest_sha256": source_sha,
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": file_sha256(manifest_path),
        "future_recipe_sha256": str(train_ds.future_recipe_sha256),
        "future_recipe_split_sha256": str(train_ds.future_recipe.recipe_sha256),
        "future_recipe_records_sha256": records_sha256,
        "normalizer_sha256": normalizer_state_sha256(normalizer.state_dict()),
        "anchor_population_sha256": array_sha256(anchor_array, "<i8"),
    }


def _score_sample(
    model: Any,
    dataset: Any,
    positions: "Any",
    cfg: Any,
    *,
    device: Any,
    batch_size: int,
) -> tuple[Any, Any, Any]:
    import numpy as np
    import torch
    from torch.utils.data import default_collate
    from treewm.data.future_sets import gather_mode_targets
    from treewm.tree.matching import branch_mode_cost, match
    from treewm.utils import config as cfg_utils

    match_cfg = cfg_utils.matching_config(cfg)
    score_chunks: list[Any] = []
    label_chunks: list[Any] = []
    mode_chunks: list[Any] = []
    with torch.inference_mode():
        for start in range(0, len(positions), int(batch_size)):
            selected = positions[start : start + int(batch_size)]
            batch = default_collate([dataset[int(position)] for position in selected])
            batch = {key: value.to(device, non_blocking=False) for key, value in batch.items()}
            z = model.encode(batch["obs"])
            child = model.predict_children(z)
            modes = gather_mode_targets(batch)
            target_z = model.encode(modes["endpoint"])
            target_q = model.q_of(target_z)
            cost = branch_mode_cost(
                pred_z=child["latent"],
                pred_q=child["q"],
                pred_action=child["branch"].action,
                pred_horizon_idx=child["horizon_idx"],
                tgt_z=target_z,
                tgt_q=target_q,
                tgt_action=modes["actions"],
                tgt_action_mask=modes["action_mask"],
                tgt_horizon_idx=modes["horizon_idx"],
                tgt_valid=modes["valid"],
                cfg=match_cfg,
                q_cdist=model.q_cdist,
            )
            branch_to_mode, _ = match(cost, modes["valid"], match_cfg)
            score_chunks.append(child["branch"].keep.float().cpu().numpy())
            label_chunks.append((branch_to_mode >= 0).cpu().numpy())
            mode_chunks.append(modes["valid"].sum(1).to(torch.int64).cpu().numpy())
    return (
        np.concatenate(score_chunks, axis=0).astype(np.float32, copy=False),
        np.concatenate(label_chunks, axis=0).astype(np.bool_, copy=False),
        np.concatenate(mode_chunks, axis=0).astype(np.int64, copy=False),
    )


def run_diagnostic(
    checkpoint: str | Path,
    output_root: str | Path,
    *,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    stage_target: int = DEFAULT_STAGE_TARGET,
    device_name: str = "cuda",
) -> tuple[Path, bool]:
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if checkpoint_path.name != "latest.pt" or checkpoint_path.parent.name != "checkpoints":
        raise DiagnosticError("checkpoint must be a formal run's checkpoints/latest.pt")
    run_dir = checkpoint_path.parents[1]
    launch_path = run_dir / "FORMAL_LAUNCH.json"
    launch = read_json(launch_path)
    if launch.get("campaign_id") != CAMPAIGN_ID or launch.get("formal_validation") is not True:
        raise DiagnosticError("checkpoint is not from grounded formal campaign 14")
    run = launch.get("run") or {}
    if run.get("setting_id") not in ALLOWED_SETTINGS:
        raise DiagnosticError(f"diagnostic is restricted to puzzle settings: {run.get('setting_id')}")
    if Path(str(run.get("run_directory", ""))).resolve() != run_dir:
        raise DiagnosticError("formal launch run directory differs from checkpoint path")
    formal_root = run_dir.parents[2]
    destination = validate_output_root(output_root, formal_root)
    before = protected_tree_fingerprint(run_dir)

    snapshot_root = _snapshot_root_from_launch(launch)
    _activate_snapshot(snapshot_root)
    campaign = importlib.import_module("campaign")
    worker = importlib.import_module("worker")
    claimed_launch_sha = launch.get("launch_sha256")
    launch_body = dict(launch)
    launch_body.pop("launch_sha256", None)
    if claimed_launch_sha != campaign.stable_hash(launch_body):
        raise DiagnosticError("FORMAL_LAUNCH.json content hash differs")
    snapshot = campaign.verify_source_snapshot(snapshot_root)
    if snapshot["source_sha256"] != launch["hashes"]["source_sha256"]:
        raise DiagnosticError("sealed source snapshot differs from launch identity")
    verified_checkpoint = worker.verify_stage_marker(run_dir, int(stage_target), launch)

    import numpy as np
    from omegaconf import OmegaConf
    import torch
    from treewm.data.future_recipe import normalizer_state_sha256
    from treewm.data.ogbench_dataset import Normalizer
    from treewm.models.baselines import build_model
    from treewm.utils import config as cfg_utils

    if device_name != "cuda":
        raise DiagnosticError("formal support sweeps require --device cuda")
    if not torch.cuda.is_available():
        raise DiagnosticError("CUDA is unavailable; launch through the GPU Slurm wrapper")
    device = torch.device("cuda")
    environment = {
        str(key): str(value) for key, value in (launch.get("environment") or {}).items()
    }
    environment.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "WANDB_MODE": "disabled",
        }
    )
    with patched_environment(environment):
        try:
            payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=False, mmap=True
            )
        except TypeError:
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg = OmegaConf.create(payload["config"])
        if int(payload.get("completed_updates", -1)) != int(stage_target):
            raise DiagnosticError("checkpoint is not at the requested paused stage")
        if str(cfg.env.short_name) != str(run["setting_id"]) or int(cfg.seed) != int(run["seed"]):
            raise DiagnosticError("checkpoint config differs from formal launch run")
        if Path(str(cfg.run_root)).expanduser().resolve() != formal_root:
            raise DiagnosticError("checkpoint config formal root differs from checkpoint path")
        dataset, data_identity = _load_dataset_read_only(cfg, launch)
        checkpoint_normalizer_sha256 = normalizer_state_sha256(
            Normalizer.from_state_dict(payload["normalizer"]).state_dict()
        )
        if checkpoint_normalizer_sha256 != data_identity["normalizer_sha256"]:
            raise DiagnosticError("checkpoint and sealed-data normalizers differ")
        positions = fixed_stratified_positions(len(dataset), int(sample_size))
        anchors = np.asarray(dataset.anchors, dtype=np.int64)[positions]
        model = build_model(
            str(cfg.arm), cfg_utils.model_config(cfg), k_max=int(cfg.model.flatk_max)
        ).to(device)
        model.gain_head.set_set_aware(bool(cfg.losses.get("gain_set_context", False)))
        model.load_state_dict(payload["model"], strict=True)
        model.eval()
        scores, labels, mode_counts = _score_sample(
            model,
            dataset,
            positions,
            cfg,
            device=device,
            batch_size=int(batch_size),
        )
        assigned_counts = labels.sum(axis=1, dtype=np.int64)
        if not np.array_equal(assigned_counts, mode_counts):
            raise DiagnosticError(
                "level-one assignment did not cover every valid mode; support recall "
                "cannot be reduced to matched-branch recall"
            )
        del model, payload, dataset
        torch.cuda.empty_cache()

    branch_factor = int(scores.shape[1])
    max_depth = int(cfg.tree.max_depth)
    node_budget = int(cfg.tree.node_budget)
    rows = [
        threshold_metrics(
            scores,
            labels,
            threshold,
            max_depth=max_depth,
            node_budget=node_budget,
        )
        for threshold in threshold_grid()
    ]
    positive_prior = float(labels.mean(dtype=np.float64))
    if not 0.0 < positive_prior < 1.0:
        raise DiagnosticError("KEEP target prior is degenerate")
    balanced_proxy = threshold_metrics(
        scores,
        labels,
        positive_prior,
        max_depth=max_depth,
        node_budget=node_budget,
    )
    score_quantiles = np.quantile(scores.astype(np.float64), [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    before_publish = protected_tree_fingerprint(run_dir)
    if before_publish != before:
        raise DiagnosticError("formal run tree changed while the read-only diagnostic ran")
    source_path = Path(__file__).resolve()
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "diagnostic_id": DIAGNOSTIC_ID,
        "method": {
            "target": "exact_level1_hungarian_branch_assignment",
            "score": "sigmoid_keep_logit",
            "comparison": "score_greater_than_or_equal_to_threshold",
            "threshold_grid": list(threshold_grid()),
            "sample": "one_counter_hashed_anchor_per_equal_width_recipe_rank_stratum",
            "sample_size": int(sample_size),
            "inference_batch_size": int(batch_size),
            "inference_precision": "float32",
            "split": "train",
            "nodes_proxy": (
                "top1-fallback root width repeated homogeneously through checkpoint "
                "max_depth and capped at node_budget; not an executed recursive tree"
            ),
            "diagnostic_source_sha256": file_sha256(source_path),
        },
        "run": {
            "campaign_id": CAMPAIGN_ID,
            "setting_id": str(run["setting_id"]),
            "seed": int(run["seed"]),
            "run_name": str(run["run_name"]),
            "launch_sha256": str(claimed_launch_sha),
            "objective_version": str(cfg.objective_version),
        },
        "protocol": {
            "diagnostic_source_sha256": file_sha256(source_path),
            "formal_trainer_source_sha256": str(launch["hashes"]["source_sha256"]),
            "formal_package_protocol_sha256": str(
                launch["hashes"]["package_protocol_sha256"]
            ),
            "formal_run_protocol_sha256": str(
                launch["hashes"]["run_protocol_sha256"]
            ),
            "formal_runtime_sha256": str(launch["hashes"]["runtime_sha256"]),
            "source_snapshot_identity_sha256": str(
                snapshot["snapshot_identity_sha256"]
            ),
        },
        "checkpoint": {
            "completed_updates": int(verified_checkpoint["completed_updates"]),
            "checkpoint_sha256": str(verified_checkpoint["checkpoint_sha256"]),
            "identity_sha256": str(verified_checkpoint["identity_sha256"]),
            "normalizer_sha256": checkpoint_normalizer_sha256,
            "stage_marker": str(verified_checkpoint["marker"]),
        },
        "source_snapshot": snapshot,
        "data": data_identity,
        "sample": {
            "size": int(len(positions)),
            "positions_sha256": array_sha256(positions, "<i8"),
            "anchors_sha256": array_sha256(anchors, "<i8"),
            "rank_fraction_quantiles": {
                key: float(value)
                for key, value in zip(
                    ("q00", "q25", "q50", "q75", "q100"),
                    np.quantile(
                        positions.astype(np.float64)
                        / max(int(data_identity["population"]) - 1, 1),
                        [0.0, 0.25, 0.5, 0.75, 1.0],
                    ),
                    strict=True,
                )
            },
        },
        "predictions": {
            "anchors": int(scores.shape[0]),
            "branch_factor": branch_factor,
            "score_dtype": "float32",
            "scores_sha256": array_sha256(scores, "<f4"),
            "labels_sha256": array_sha256(labels, "|u1"),
            "mode_counts_sha256": array_sha256(mode_counts, "<i8"),
            "positive_branch_prior": positive_prior,
            "mean_modes_per_anchor": float(mode_counts.mean(dtype=np.float64)),
            "assignment_capacity_recall": 1.0,
            "score_quantiles": {
                key: float(value)
                for key, value in zip(
                    ("q00", "q10", "q25", "q50", "q75", "q90", "q100"),
                    score_quantiles,
                    strict=True,
                )
            },
        },
        "balanced_bce_intercept_only_proxy": {
            "interpretation": (
                "If balanced BCE only shifted the learned log-odds intercept, its 0.5 "
                "decision is equivalent to the present score threshold equal to the "
                "observed positive prior; retraining may also change ranking."
            ),
            "positive_weight": float((1.0 - positive_prior) / positive_prior),
            "logit_shift": float(math.log((1.0 - positive_prior) / positive_prior)),
            "equivalent_current_threshold": positive_prior,
            "metrics": balanced_proxy,
        },
        "thresholds": rows,
        "read_only_proof": {
            "formal_run_tree_before_sha256": before,
            "formal_run_tree_after_sha256": before_publish,
            "unchanged": True,
        },
    }
    body["input_identity_sha256"] = stable_hash(
        {
            "diagnostic_id": DIAGNOSTIC_ID,
            "diagnostic_source_sha256": body["method"]["diagnostic_source_sha256"],
            "launch_sha256": claimed_launch_sha,
            "checkpoint_sha256": verified_checkpoint["checkpoint_sha256"],
            "future_recipe_split_sha256": data_identity["future_recipe_split_sha256"],
            "positions_sha256": body["sample"]["positions_sha256"],
            "threshold_grid": body["method"]["threshold_grid"],
        }
    )
    artifact_path, reused = write_immutable_json(destination, body)
    return artifact_path, reused


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--stage-target", type=int, default=DEFAULT_STAGE_TARGET)
    parser.add_argument("--device", choices=("cuda",), default="cuda")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_size <= 0 or args.batch_size <= 0:
        raise SystemExit("--sample-size and --batch-size must be positive")
    try:
        path, reused = run_diagnostic(
            args.checkpoint,
            args.output_root,
            sample_size=args.sample_size,
            batch_size=args.batch_size,
            stage_target=args.stage_target,
            device_name=args.device,
        )
    except (DiagnosticError, OSError, RuntimeError, ValueError) as exc:
        print(f"support-threshold diagnostic failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"artifact": str(path), "reused": reused}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
