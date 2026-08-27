#!/usr/bin/env python3
"""Read-only effective-gradient audit for paused grounded-formal checkpoints.

The checkpoint's sealed experiment-14 snapshot supplies the exact model, branch
objective, matching, and dataset implementation. Candidate recipes opt into only the
new grounded-execution multistep API from this diagnostic's immutable source root.
Every recipe sees the same three fixed representative train batches and three fixed
representative validation batches. No optimizer, ``Parameter.grad``, checkpoint tensor,
formal config, run-directory file, cache builder, or recipe builder is mutated.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence


SCHEMA_VERSION = 1
DIAGNOSTIC_ID = "treewm_effective_gradient_audit_v1"
CAMPAIGN_ID = "treewm-grounded-formal-v1"
STAGE_TARGET = 25_000
AUDIT_BATCHES = 3
AUDIT_BATCH_SIZE = 16
GRADIENT_SHARE_BOUND = 0.80
SHARED_NORM_RATIO_BOUND = 1.50
MODULE_NORM_RATIO_BOUND = 2.00
INTENDED_PATH_NORM_FLOOR = 1.0e-8
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
MODULE_NAMES = (
    "encoder",
    "branch_transformer",
    "dynamics",
    "controllability",
    "action_head",
    "horizon_head",
    "keep_head",
    "decoder",
)
INTENDED_PATHS = ("action_head", "horizon_head", "keep_head", "decoder")
INTENDED_PATH_TERMS = {
    "action_head": "multistep",
    "horizon_head": "multistep",
    "keep_head": "keep",
    "decoder": "multistep",
}
EXPECTED_TERMS = frozenset(
    {
        "state",
        "action",
        "horizon",
        "bind",
        "coverage",
        "redundancy",
        "keep",
        "uncertainty",
        "recursive",
        "reconstruction",
        "control",
        "multistep",
    }
)
GROUNDED_KWARGS = (
    "grounded_select_action_weight",
    "grounded_select_endpoint_weight",
    "grounded_select_horizon_weight",
    "grounded_loss_latent_weight",
    "grounded_loss_action_weight",
    "grounded_loss_horizon_weight",
    "grounded_loss_endpoint_weight",
    "grounded_detach_self_fed_parent",
)


LOCKED_RECIPES: tuple[dict[str, Any], ...] = (
    {
        "recipe_id": "baseline-exact",
        "transition_mode": "teacher_action",
        "scheduled_sampling_granularity": "step",
        "scheduled_sampling_p": 0.25,
        "scheduled_sampling_warmup": 5000,
        "multistep_depth_weights": [1.0, 1.0, 1.0],
        "outer_multistep_weight": 1.0,
        "keep_balance": False,
        "grounded_select_action_weight": 0.0,
        "grounded_select_endpoint_weight": 0.0,
        "grounded_select_horizon_weight": 0.0,
        "grounded_loss_latent_weight": 0.0,
        "grounded_loss_action_weight": 0.0,
        "grounded_loss_horizon_weight": 0.0,
        "grounded_loss_endpoint_weight": 0.0,
        "grounded_detach_self_fed_parent": True,
    },
    {
        "recipe_id": "candidate-conservative",
        "transition_mode": "grounded_execution_v2",
        "scheduled_sampling_granularity": "sequence",
        "scheduled_sampling_p": 0.25,
        "scheduled_sampling_warmup": 5000,
        "multistep_depth_weights": [1.0, 1.0, 1.0],
        "outer_multistep_weight": 1.0,
        "keep_balance": True,
        "grounded_select_action_weight": 1.0,
        "grounded_select_endpoint_weight": 1.0,
        "grounded_select_horizon_weight": 0.25,
        "grounded_loss_latent_weight": 0.25,
        "grounded_loss_action_weight": 0.5,
        "grounded_loss_horizon_weight": 0.25,
        "grounded_loss_endpoint_weight": 0.5,
        "grounded_detach_self_fed_parent": True,
    },
    {
        "recipe_id": "candidate-control",
        "transition_mode": "grounded_execution_v2",
        "scheduled_sampling_granularity": "sequence",
        "scheduled_sampling_p": 0.25,
        "scheduled_sampling_warmup": 5000,
        "multistep_depth_weights": [1.0, 1.0, 1.0],
        "outer_multistep_weight": 1.0,
        "keep_balance": True,
        "grounded_select_action_weight": 1.0,
        "grounded_select_endpoint_weight": 2.0,
        "grounded_select_horizon_weight": 0.25,
        "grounded_loss_latent_weight": 0.25,
        "grounded_loss_action_weight": 1.0,
        "grounded_loss_horizon_weight": 0.25,
        "grounded_loss_endpoint_weight": 1.0,
        "grounded_detach_self_fed_parent": True,
    },
)


class DiagnosticError(RuntimeError):
    """The audit cannot establish its read-only scientific contract."""


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
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise DiagnosticError(f"duplicate JSON key {key!r} in {path}")
            value[key] = item
        return value

    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=reject_duplicates)
    except (OSError, json.JSONDecodeError) as exc:
        raise DiagnosticError(f"cannot read exact JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise DiagnosticError(f"JSON artifact is not an object: {path}")
    return value


def load_locked_recipes(path: str | Path) -> dict[str, Any]:
    payload = read_json(path)
    claimed = payload.get("recipe_set_sha256")
    body = dict(payload)
    body.pop("recipe_set_sha256", None)
    if not isinstance(claimed, str) or stable_hash(body) != claimed:
        raise DiagnosticError("audit recipe-set content hash differs")
    expected = {
        "schema_version": 1,
        "diagnostic_id": DIAGNOSTIC_ID,
        "audit_step": STAGE_TARGET,
        "gradient_share_bound": GRADIENT_SHARE_BOUND,
        "shared_norm_ratio_bound": SHARED_NORM_RATIO_BOUND,
        "module_norm_ratio_bound": MODULE_NORM_RATIO_BOUND,
        "intended_path_norm_floor": INTENDED_PATH_NORM_FLOOR,
        "recipes": list(LOCKED_RECIPES),
    }
    if body != expected:
        raise DiagnosticError("audit recipes or scale-only guardrails drifted")
    return payload


def recipe_sha256(recipe: Mapping[str, Any]) -> str:
    return stable_hash(dict(recipe))


def _u64_hash(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("ascii")).digest()[:8], "little")


def fixed_representative_batches(
    population: int,
    *,
    split: str,
    batches: int = AUDIT_BATCHES,
    batch_size: int = AUDIT_BATCH_SIZE,
) -> list[list[int]]:
    """Return disjoint fixed batches that each span the complete recipe rank range."""
    population = int(population)
    batches = int(batches)
    batch_size = int(batch_size)
    count = batches * batch_size
    if population <= 0 or batches <= 0 or batch_size <= 0 or count > population:
        raise ValueError("invalid representative gradient-audit sample dimensions")
    positions: list[int] = []
    for index in range(count):
        start = (index * population) // count
        stop = ((index + 1) * population) // count
        width = stop - start
        offset = _u64_hash(f"{DIAGNOSTIC_ID}:{split}:stratum:{index}") % width
        positions.append(start + offset)
    if any(right <= left for left, right in zip(positions, positions[1:])):
        raise AssertionError("representative positions are not strictly increasing")
    # Interleave global strata so every individual batch spans the population.
    return [positions[batch_index::batches] for batch_index in range(batches)]


def array_sha256(array: Any, dtype: str) -> str:
    import numpy as np

    canonical = np.asarray(array, dtype=np.dtype(dtype)).astype(dtype, copy=False)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def tensor_mapping_sha256(values: Mapping[str, Any]) -> str:
    """Hash tensor names, shapes, dtypes, and exact CPU bytes in stable order."""
    import torch

    digest = hashlib.sha256()
    for name in sorted(values):
        tensor = values[name]
        if not torch.is_tensor(tensor):
            raise TypeError(f"{name} is not a tensor")
        cpu = tensor.detach().cpu().contiguous()
        header = canonical_json(
            {"name": str(name), "shape": list(cpu.shape), "dtype": str(cpu.dtype)}
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        raw = cpu.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def tensors_finite(values: Mapping[str, Any]) -> bool:
    import torch

    for tensor in values.values():
        if torch.is_tensor(tensor) and (tensor.is_floating_point() or tensor.is_complex()):
            if not bool(torch.isfinite(tensor).all()):
                return False
    return True


def protected_tree_fingerprint(root: str | Path) -> str:
    base = Path(root).resolve()
    if not base.is_dir() or base.is_symlink():
        raise DiagnosticError(f"formal run directory is unavailable or symlinked: {base}")
    entries: list[dict[str, Any]] = []
    for path in sorted(base.rglob("*"), key=lambda item: str(item.relative_to(base))):
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
    output = Path(output_root).expanduser().resolve()
    protected = Path(formal_root).expanduser().resolve()
    if output == protected or output.is_relative_to(protected):
        raise DiagnosticError(
            f"gradient-audit output must be outside the formal campaign tree: {protected}"
        )
    return output


def write_immutable_json(output_root: str | Path, body: Mapping[str, Any]) -> tuple[Path, bool]:
    payload = dict(body)
    payload["artifact_sha256"] = stable_hash(payload)
    run = payload["run"]
    checkpoint = payload["checkpoint"]
    filename = (
        f"gradient-audit__{run['setting_id']}__seed-{run['seed']}__"
        f"step-{checkpoint['completed_updates']}__{payload['artifact_sha256'][:16]}.json"
    )
    root = Path(output_root)
    path = root / filename
    encoded = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    root.mkdir(parents=True, exist_ok=True)
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
        directory_fd = os.open(root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
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
        raise DiagnosticError("formal launch lacks its pinned trainer command")
    trainer = Path(str(argv[1])).expanduser().resolve()
    if trainer.name != "train.py" or trainer.parent.name != "scripts" or not trainer.is_file():
        raise DiagnosticError(f"formal launch trainer is unavailable: {trainer}")
    root = trainer.parents[1]
    if root.name != "repo":
        raise DiagnosticError("formal launch trainer is not inside a sealed snapshot repo")
    return root


def _activate_snapshot(snapshot_root: Path) -> None:
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


def _load_candidate_api(source_root: Path) -> tuple[Any, dict[str, str]]:
    recursive_path = source_root / "treewm" / "losses" / "recursive_losses.py"
    trainer_path = source_root / "scripts" / "train.py"
    if not recursive_path.is_file() or not trainer_path.is_file():
        raise DiagnosticError("candidate grounded-execution sources are unavailable")
    name = "treewm_gradient_audit_grounded_execution_api_v1"
    spec = importlib.util.spec_from_file_location(name, recursive_path)
    if spec is None or spec.loader is None:
        raise DiagnosticError("cannot load grounded-execution API source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    function = getattr(module, "multi_step_recursive_loss", None)
    if function is None:
        raise DiagnosticError("candidate source lacks multi_step_recursive_loss")
    required = {
        "scheduled_sampling_granularity",
        "transition_mode",
        *GROUNDED_KWARGS,
    }
    if not required.issubset(inspect.signature(function).parameters):
        raise DiagnosticError("candidate grounded-execution API signature is incomplete")
    trainer_text = trainer_path.read_text(encoding="utf-8")
    if "def multistep_transition_kwargs(" not in trainer_text:
        raise DiagnosticError("candidate trainer lacks its canonical transition bundle")
    recursive_text = recursive_path.read_text(encoding="utf-8")
    if (
        "horizon_error" not in recursive_text
        or "1.0 - horizon_target_probability" not in recursive_text
    ):
        raise DiagnosticError(
            "candidate selector lacks bounded 1-softmax(target-horizon) semantics"
        )
    return function, {
        "recursive_losses_path": str(recursive_path.resolve()),
        "recursive_losses_sha256": file_sha256(recursive_path),
        "trainer_path": str(trainer_path.resolve()),
        "trainer_sha256": file_sha256(trainer_path),
    }


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
        name = payload.get("dataset") or payload.get("dataset_name")
        if name == source_name and payload.get("source_manifest_sha256") == source_manifest_sha256:
            matches.append((path, payload))
    if len(matches) != 1:
        raise DiagnosticError(
            f"expected one sealed cache for {source_name}/{source_manifest_sha256}, "
            f"found {len(matches)}"
        )
    path, payload = matches[0]
    required = ("train_obs.npy", "train_act.npy", "train_obs_norm.npy", "train_act_norm.npy")
    missing = [name for name in required if not (path.parent / name).is_file()]
    if missing:
        raise DiagnosticError(f"sealed cache is incomplete ({', '.join(missing)}): {path}")
    return path, payload


def _load_datasets_read_only(cfg: Any, launch: Mapping[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    import numpy as np
    from dataclasses import replace
    from treewm.data.future_recipe import normalizer_state_sha256
    from treewm.data.ogbench_dataset import (
        ChunkDataset,
        Normalizer,
        _attach_future_recipes_if_requested,
    )
    from treewm.utils import config as cfg_utils

    environment = launch.get("environment") or {}
    source_sha256 = str(environment.get("TREEWM_DATA_SHA256", ""))
    cache_root = Path(str(environment.get("TREEWM_CACHE", ""))).expanduser().resolve()
    source_name = str(cfg.env.get("source_name", cfg.env.name))
    manifest_path, cache_manifest = _find_exact_cache_manifest(
        cache_root,
        source_manifest_sha256=source_sha256,
        source_name=source_name,
    )
    if str(cfg.env.get("dataset_kind", "standard")) == "sharded_100m_full":
        from treewm.data.sharded_ogbench import _load_cache

        cache = _load_cache(manifest_path.parent, cache_manifest, was_hit=True)
    else:
        from treewm.data.shared_cache import _cache_from_manifest

        cache = _cache_from_manifest(manifest_path.parent, cache_manifest, was_hit=True)
    if cache.source_manifest_sha256 != source_sha256:
        raise DiagnosticError("opened cache identity differs from formal launch")
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
    train_ds = ChunkDataset(
        {}, normalizer, future_cfg, max_anchors=0, seed=int(cfg.seed), shared=cache.train, **common
    )
    val_ds = ChunkDataset(
        {}, normalizer, future_cfg, max_anchors=0, seed=int(cfg.seed) + 1, shared=cache.val, **common
    )
    cache.assert_consumed_by(train_ds, val_ds)
    _attach_future_recipes_if_requested(
        train_ds,
        val_ds,
        normalizer,
        future_cfg,
        source_sha256,
        anchor_policy=str(cfg.future_sets.get("recipe_anchor_policy", "selected_seed")),
    )
    expected_recipe = str(environment.get("TREEWM_FUTURE_RECIPE_SHA256", ""))
    if any(
        str(getattr(dataset, "future_recipe_sha256", "")) != expected_recipe
        for dataset in (train_ds, val_ds)
    ):
        raise DiagnosticError("loaded future recipe differs from formal launch")
    split_identity: dict[str, Any] = {}
    for split, dataset in (("train", train_ds), ("validation", val_ds)):
        recipe = dataset.future_recipe
        records_path = recipe.root / str(recipe.manifest["records_file"])
        records_sha256 = file_sha256(records_path)
        if records_sha256 != str(recipe.manifest["records_sha256"]):
            raise DiagnosticError(f"{split} future-recipe records content hash differs")
        anchors = np.asarray(dataset.anchors, dtype="<i8")
        split_identity[split] = {
            "population": int(len(dataset)),
            "recipe_split_sha256": str(recipe.recipe_sha256),
            "recipe_records_sha256": records_sha256,
            "anchor_population_sha256": array_sha256(anchors, "<i8"),
        }
    return train_ds, val_ds, {
        "source_manifest_sha256": source_sha256,
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": file_sha256(manifest_path),
        "future_recipe_sha256": expected_recipe,
        "normalizer_sha256": normalizer_state_sha256(normalizer.state_dict()),
        "splits": split_identity,
    }


def _materialize_fixed_batches(train_ds: Any, val_ds: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    from torch.utils.data import default_collate

    fixed: list[dict[str, Any]] = []
    identity: dict[str, Any] = {}
    for split, dataset in (("train", train_ds), ("validation", val_ds)):
        position_batches = fixed_representative_batches(len(dataset), split=split)
        split_rows: list[dict[str, Any]] = []
        for batch_index, positions in enumerate(position_batches):
            anchors = [int(dataset.anchors[position]) for position in positions]
            batch = default_collate([dataset[position] for position in positions])
            if not tensors_finite(batch):
                raise DiagnosticError(f"{split} batch {batch_index} contains non-finite input")
            row = {
                "split": split,
                "batch_index": batch_index,
                "positions": positions,
                "positions_sha256": array_sha256(np.asarray(positions), "<i8"),
                "anchors": anchors,
                "anchors_sha256": array_sha256(np.asarray(anchors), "<i8"),
                "batch_tensors_sha256": tensor_mapping_sha256(batch),
            }
            split_rows.append(row)
            fixed.append({**row, "batch": batch})
        identity[split] = split_rows
    return fixed, identity


def _freeze_as_training(model: Any, loss_cfg: Any, objective_version: str) -> None:
    if objective_version.startswith("treewm_v2") and str(loss_cfg.control_objective) != "bootstrap":
        model.tree_signature.requires_grad_(False)
    if bool(loss_cfg.gain_set_context) and float(loss_cfg.gain_branch_prior_weight) == 0.0:
        model.heads.gain_head.requires_grad_(False)
    if objective_version.startswith("treewm_v2") and not loss_cfg.on("mass"):
        model.heads.mass_head.requires_grad_(False)


def _recipe_loss_config(base: Any, recipe: Mapping[str, Any]) -> Any:
    value = copy.deepcopy(base)
    value.keep_balance = bool(recipe["keep_balance"])
    value.scheduled_sampling_p = float(recipe["scheduled_sampling_p"])
    value.scheduled_sampling_warmup = int(recipe["scheduled_sampling_warmup"])
    value.scheduled_sampling_granularity = str(recipe["scheduled_sampling_granularity"])
    value.multistep_depth_weights = tuple(float(x) for x in recipe["multistep_depth_weights"])
    value.weights.multistep = float(recipe["outer_multistep_weight"])
    return value


def _audit_seed(setting_id: str, split: str, batch_index: int) -> int:
    value = _u64_hash(f"{DIAGNOSTIC_ID}:{setting_id}:{split}:batch:{batch_index}")
    return int(value % (2**63 - 1))


def _to_device(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {key: value.to(device, non_blocking=False) for key, value in batch.items()}


def _assert_finite_mapping(values: Mapping[str, Any], context: str) -> None:
    bad: list[str] = []
    for key, value in values.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and not math.isfinite(float(value)):
            bad.append(str(key))
    if bad:
        raise DiagnosticError(f"{context} contains non-finite values: {', '.join(sorted(bad))}")


def _structured_gradient_result(result: Any, terms: Any) -> dict[str, Any]:
    active = tuple(str(name) for name in result.active_terms)
    if set(active) != EXPECTED_TERMS:
        raise DiagnosticError(f"active loss terms differ: {sorted(active)}")
    _assert_finite_mapping(result.metrics, "effective gradient audit")
    modules: dict[str, Any] = {}
    for module_name in (*MODULE_NAMES, "shared"):
        objective = float(result.metrics[f"gradient_audit/objective_norm/{module_name}"])
        module_terms: dict[str, Any] = {}
        for term_name in active:
            module_terms[term_name] = {
                "effective_norm": float(
                    result.metrics[
                        f"gradient_audit/effective_norm/{module_name}/{term_name}"
                    ]
                ),
                "share": float(
                    result.metrics[f"gradient_audit/share/{module_name}/{term_name}"]
                ),
                "cosine_total": float(
                    result.metrics[
                        f"gradient_audit/cosine_total/{module_name}/{term_name}"
                    ]
                ),
            }
        modules[module_name] = {
            "objective_norm": objective,
            "max_term_share": max(
                (row["share"] for row in module_terms.values()), default=0.0
            ),
            "terms": module_terms,
        }
    term_scalars: dict[str, Any] = {}
    for term_name in active:
        raw = float(terms.raw[term_name].detach().float().item())
        effective = float(terms.effective[term_name].detach().float().item())
        if not math.isfinite(raw) or not math.isfinite(effective):
            raise DiagnosticError(f"loss term {term_name} is non-finite")
        term_scalars[term_name] = {
            "raw": raw,
            "effective": effective,
            "weight": float(terms.weights[term_name]),
            "schedule": float(terms.schedules[term_name]),
        }
    return {
        "active_terms": list(active),
        "term_scalars": term_scalars,
        "modules": modules,
    }


def _run_one_recipe_batch(
    *,
    model: Any,
    cpu_batch: Mapping[str, Any],
    cfg: Any,
    base_loss_cfg: Any,
    match_cfg: Any,
    recipe: Mapping[str, Any],
    current_multistep: Any,
    candidate_multistep: Any,
    audit_function: Any,
    assemble_function: Any,
    branch_function: Any,
    schedule_function: Any,
    device: Any,
    seed: int,
) -> dict[str, Any]:
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    batch = _to_device(cpu_batch, device)
    loss_cfg = _recipe_loss_config(base_loss_cfg, recipe)
    use_bf16 = bool(cfg.train.bf16)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
        _, branch_metrics, _, branch_terms = branch_function(
            model,
            batch,
            loss_cfg,
            match_cfg,
            step=STAGE_TARGET,
            return_loss_terms=True,
        )
        raw_terms = dict(branch_terms.raw)
        scheduled_p = schedule_function(
            STAGE_TARGET,
            float(recipe["scheduled_sampling_p"]),
            int(recipe["scheduled_sampling_warmup"]),
        )
        common = {
            "scheduled_sampling_p": scheduled_p,
            "depth_weights": tuple(float(x) for x in recipe["multistep_depth_weights"]),
        }
        if recipe["transition_mode"] == "teacher_action":
            multistep_loss, multistep_metrics = current_multistep(model, batch, **common)
        else:
            multistep_loss, multistep_metrics = candidate_multistep(
                model,
                batch,
                scheduled_sampling_granularity=str(
                    recipe["scheduled_sampling_granularity"]
                ),
                transition_mode=str(recipe["transition_mode"]),
                **{key: recipe[key] for key in GROUNDED_KWARGS},
                **common,
            )
        raw_terms["multistep"] = multistep_loss
        terms = assemble_function(raw_terms, loss_cfg, STAGE_TARGET)
    if not bool(torch.isfinite(terms.total).all()):
        raise DiagnosticError("effective objective is non-finite")
    _assert_finite_mapping(branch_metrics, "branch metrics")
    _assert_finite_mapping(multistep_metrics, "multistep metrics")
    audit = audit_function(
        terms,
        {
            "encoder": model.encoder,
            "branch_transformer": model.branch_transformer,
            "dynamics": model.dynamics,
            "controllability": model.controllability,
            "action_head": model.heads.action_head,
            "horizon_head": model.heads.horizon_head,
            "keep_head": model.heads.keep_head,
            "decoder": model.decoder,
        },
        max_terms=32,
        max_gradient_share=GRADIENT_SHARE_BOUND,
        fail_on_excess=False,
        preserve_graph=False,
    )
    structured = _structured_gradient_result(audit, terms)
    structured.update(
        {
            "rng_seed": seed,
            "objective": float(terms.total.detach().float().item()),
            "multistep_metrics": {
                str(key): float(value)
                for key, value in multistep_metrics.items()
                if isinstance(value, (int, float))
            },
            "branch_metrics": {
                key: float(branch_metrics[key])
                for key in (
                    "tree/support_recall",
                    "tree/support_precision",
                    "tree/keep_rate",
                    "tree/effective_branching_factor",
                )
                if key in branch_metrics
            },
        }
    )
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise DiagnosticError("autograd audit mutated Parameter.grad")
    del batch, terms, branch_terms, multistep_loss
    return structured


def _aggregate_batches(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise DiagnosticError("cannot aggregate an empty gradient audit")

    def stats(values: Sequence[float]) -> dict[str, float]:
        if not values or not all(math.isfinite(float(value)) for value in values):
            raise DiagnosticError("gradient aggregate received non-finite values")
        return {
            "mean": float(sum(values) / len(values)),
            "min": float(min(values)),
            "max": float(max(values)),
        }

    modules: dict[str, Any] = {}
    active = rows[0]["active_terms"]
    for module_name in (*MODULE_NAMES, "shared"):
        module: dict[str, Any] = {
            "objective_norm": stats(
                [float(row["modules"][module_name]["objective_norm"]) for row in rows]
            ),
            "max_term_share": stats(
                [float(row["modules"][module_name]["max_term_share"]) for row in rows]
            ),
            "terms": {},
        }
        for term_name in active:
            module["terms"][term_name] = {
                field: stats(
                    [
                        float(row["modules"][module_name]["terms"][term_name][field])
                        for row in rows
                    ]
                )
                for field in ("effective_norm", "share", "cosine_total")
            }
        modules[module_name] = module
    scalar_keys = sorted(
        set.intersection(
            *(set(row["multistep_metrics"]) for row in rows)
        )
    )
    return {
        "batches": len(rows),
        "objective": stats([float(row["objective"]) for row in rows]),
        "modules": modules,
        "multistep_metrics": {
            key: stats([float(row["multistep_metrics"][key]) for row in rows])
            for key in scalar_keys
        },
    }


def _candidate_scale_gate(
    baseline: Mapping[tuple[str, int], Mapping[str, Any]],
    candidate: Mapping[tuple[str, int], Mapping[str, Any]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for key in sorted(baseline):
        if key not in candidate:
            raise DiagnosticError("candidate does not cover every baseline batch")
        base = baseline[key]
        value = candidate[key]
        for module_name in (*MODULE_NAMES, "shared"):
            base_norm = float(base["modules"][module_name]["objective_norm"])
            candidate_norm = float(value["modules"][module_name]["objective_norm"])
            ratio = candidate_norm / base_norm if base_norm > 0.0 else None
            bound = (
                SHARED_NORM_RATIO_BOUND
                if module_name == "shared"
                else MODULE_NORM_RATIO_BOUND
            )
            checks.append(
                {
                    "split": key[0],
                    "batch_index": key[1],
                    "check": "candidate_to_baseline_objective_norm_ratio",
                    "module": module_name,
                    "value": ratio,
                    "baseline_value": base_norm,
                    "candidate_value": candidate_norm,
                    "bound": bound,
                    "passed": bool(
                        ratio is not None and math.isfinite(ratio) and ratio <= bound
                    ),
                }
            )
            maximum_share = float(value["modules"][module_name]["max_term_share"])
            checks.append(
                {
                    "split": key[0],
                    "batch_index": key[1],
                    "check": "maximum_active_term_share",
                    "module": module_name,
                    "value": maximum_share,
                    "bound": GRADIENT_SHARE_BOUND,
                    "passed": bool(
                        math.isfinite(maximum_share)
                        and maximum_share <= GRADIENT_SHARE_BOUND
                    ),
                }
            )
        for module_name in INTENDED_PATHS:
            term_name = INTENDED_PATH_TERMS[module_name]
            norm = float(
                value["modules"][module_name]["terms"][term_name]["effective_norm"]
            )
            checks.append(
                {
                    "split": key[0],
                    "batch_index": key[1],
                    "check": "intended_effective_term_path_positive_finite",
                    "module": module_name,
                    "term": term_name,
                    "value": norm,
                    "lower_bound_exclusive": INTENDED_PATH_NORM_FLOOR,
                    "passed": bool(
                        math.isfinite(norm) and norm > INTENDED_PATH_NORM_FLOOR
                    ),
                }
            )
    failures = [row for row in checks if not row["passed"]]
    return {
        "classification": "scale_safety_only_not_efficacy",
        "passed": not failures,
        "checks": checks,
        "failure_count": len(failures),
        "failures": failures,
    }


def run_diagnostic(checkpoint: str | Path, output_root: str | Path) -> tuple[Path, bool]:
    # Loading the candidate module must not create bytecode in the immutable source.
    sys.dont_write_bytecode = True
    checkpoint_path = Path(checkpoint).expanduser().resolve()
    if checkpoint_path.name != "latest.pt" or checkpoint_path.parent.name != "checkpoints":
        raise DiagnosticError("checkpoint must be a formal checkpoints/latest.pt")
    run_dir = checkpoint_path.parents[1]
    launch = read_json(run_dir / "FORMAL_LAUNCH.json")
    run = launch.get("run") or {}
    if launch.get("campaign_id") != CAMPAIGN_ID or launch.get("formal_validation") is not True:
        raise DiagnosticError("checkpoint is not from grounded formal campaign 14")
    if run.get("setting_id") not in SETTING_IDS or int(run.get("seed", -1)) != 0:
        raise DiagnosticError("gradient audit is locked to the ten seed-zero formal runs")
    if Path(str(run.get("run_directory", ""))).resolve() != run_dir:
        raise DiagnosticError("formal launch run directory differs from checkpoint path")
    formal_root = run_dir.parents[2]
    destination = validate_output_root(output_root, formal_root)
    run_fingerprint = protected_tree_fingerprint(run_dir)
    diagnostic_source_root = Path(__file__).resolve().parents[3]
    package = Path(__file__).resolve().parent
    recipes_path = package / "recipes.json"
    recipe_set = load_locked_recipes(recipes_path)
    package_files_before = {
        name: file_sha256(package / name)
        for name in ("gradient_audit.py", "recipes.json", "gradient_audit.slurm")
    }
    snapshot_root = _snapshot_root_from_launch(launch)
    _activate_snapshot(snapshot_root)
    # Load only the candidate recursive implementation after snapshot activation. Its
    # unchanged world-loss imports therefore resolve to the verified formal snapshot,
    # rather than to mutable working-tree dependencies.
    candidate_multistep, candidate_api = _load_candidate_api(diagnostic_source_root)
    candidate_api.update(
        {
            "dependency_contract": "sealed_formal_snapshot_world_losses",
            "world_losses_path": str(
                (snapshot_root / "treewm" / "losses" / "world_losses.py").resolve()
            ),
            "world_losses_sha256": file_sha256(
                snapshot_root / "treewm" / "losses" / "world_losses.py"
            ),
        }
    )
    campaign = importlib.import_module("campaign")
    worker = importlib.import_module("worker")
    claimed_launch_sha256 = launch.get("launch_sha256")
    launch_body = dict(launch)
    launch_body.pop("launch_sha256", None)
    if claimed_launch_sha256 != campaign.stable_hash(launch_body):
        raise DiagnosticError("FORMAL_LAUNCH.json content hash differs")
    snapshot = campaign.verify_source_snapshot(snapshot_root)
    if snapshot["source_sha256"] != launch["hashes"]["source_sha256"]:
        raise DiagnosticError("sealed source snapshot differs from formal launch")
    verified = worker.verify_stage_marker(run_dir, STAGE_TARGET, launch)

    import numpy as np
    from omegaconf import OmegaConf
    import torch
    from treewm.data.future_recipe import normalizer_state_sha256
    from treewm.data.ogbench_dataset import Normalizer
    from treewm.losses.recursive_losses import (
        multi_step_recursive_loss as current_multistep,
        scheduled_sampling_schedule,
    )
    from treewm.losses.total import (
        assemble_loss_terms,
        audit_effective_loss_gradients,
        compute_branch_losses,
    )
    from treewm.models.baselines import build_model
    from treewm.utils import config as cfg_utils

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise DiagnosticError("gradient audit requires exactly one visible CUDA GPU")
    device = torch.device("cuda:0")
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
        if int(payload.get("completed_updates", -1)) != STAGE_TARGET:
            raise DiagnosticError("checkpoint is not at the locked 25k boundary")
        cfg = OmegaConf.create(payload["config"])
        if str(cfg.env.short_name) != str(run["setting_id"]) or int(cfg.seed) != 0:
            raise DiagnosticError("checkpoint config differs from formal launch run")
        if Path(str(cfg.run_root)).expanduser().resolve() != formal_root:
            raise DiagnosticError("checkpoint config formal root differs from checkpoint path")
        config_before = stable_hash(OmegaConf.to_container(cfg, resolve=True))
        train_ds, val_ds, data_identity = _load_datasets_read_only(cfg, launch)
        checkpoint_normalizer = normalizer_state_sha256(
            Normalizer.from_state_dict(payload["normalizer"]).state_dict()
        )
        if checkpoint_normalizer != data_identity["normalizer_sha256"]:
            raise DiagnosticError("checkpoint and sealed-data normalizers differ")
        fixed_batches, sample_identity = _materialize_fixed_batches(train_ds, val_ds)
        del train_ds, val_ds

        base_loss_cfg = cfg_utils.loss_config(cfg)
        baseline = LOCKED_RECIPES[0]
        baseline_checks = {
            "keep_balance": bool(base_loss_cfg.keep_balance)
            == bool(baseline["keep_balance"]),
            "scheduled_sampling_p": float(base_loss_cfg.scheduled_sampling_p)
            == float(baseline["scheduled_sampling_p"]),
            "scheduled_sampling_warmup": int(base_loss_cfg.scheduled_sampling_warmup)
            == int(baseline["scheduled_sampling_warmup"]),
            "multistep_depth_weights": tuple(base_loss_cfg.multistep_depth_weights)
            == tuple(baseline["multistep_depth_weights"]),
            "outer_multistep_weight": float(base_loss_cfg.weights.multistep)
            == float(baseline["outer_multistep_weight"]),
            "historical_transition_fields_absent": all(
                key not in cfg.losses
                for key in (
                    "scheduled_sampling_granularity",
                    "multistep_transition_mode",
                    *GROUNDED_KWARGS,
                )
            ),
            "historical_multistep_signature": tuple(
                inspect.signature(current_multistep).parameters
            )
            == (
                "model",
                "batch",
                "scheduled_sampling_p",
                "depth_weights",
                "generator",
            ),
        }
        if not all(baseline_checks.values()):
            raise DiagnosticError(f"checkpoint baseline recipe differs: {baseline_checks}")

        model = build_model(
            str(cfg.arm), cfg_utils.model_config(cfg), k_max=int(cfg.model.flatk_max)
        )
        model.gain_head.set_set_aware(bool(base_loss_cfg.gain_set_context))
        _freeze_as_training(model, base_loss_cfg, str(cfg.objective_version))
        model.load_state_dict(payload["model"], strict=True)
        if model.decoder is None:
            raise DiagnosticError("grounded gradient audit requires the formal decoder")
        checkpoint_model_sha256 = tensor_mapping_sha256(payload["model"])
        model = model.to(device)
        model.set_gradient_checkpointing(bool(cfg.train.gradient_checkpointing))
        model.train()
        match_cfg = cfg_utils.matching_config(cfg)
        trainable_names = sorted(
            name for name, parameter in model.named_parameters() if parameter.requires_grad
        )
        trainable_names_sha256 = stable_hash(trainable_names)
        del payload

        recipe_results: dict[str, Any] = {}
        indexed_results: dict[str, dict[tuple[str, int], Any]] = {}
        for recipe in LOCKED_RECIPES:
            recipe_id = str(recipe["recipe_id"])
            batch_rows: list[dict[str, Any]] = []
            indexed: dict[tuple[str, int], Any] = {}
            for fixed in fixed_batches:
                audit = _run_one_recipe_batch(
                    model=model,
                    cpu_batch=fixed["batch"],
                    cfg=cfg,
                    base_loss_cfg=base_loss_cfg,
                    match_cfg=match_cfg,
                    recipe=recipe,
                    current_multistep=current_multistep,
                    candidate_multistep=candidate_multistep,
                    audit_function=audit_effective_loss_gradients,
                    assemble_function=assemble_loss_terms,
                    branch_function=compute_branch_losses,
                    schedule_function=scheduled_sampling_schedule,
                    device=device,
                    seed=_audit_seed(
                        str(run["setting_id"]),
                        str(fixed["split"]),
                        int(fixed["batch_index"]),
                    ),
                )
                row = {
                    "split": fixed["split"],
                    "batch_index": fixed["batch_index"],
                    "positions_sha256": fixed["positions_sha256"],
                    "anchors_sha256": fixed["anchors_sha256"],
                    "batch_tensors_sha256": fixed["batch_tensors_sha256"],
                    **audit,
                }
                batch_rows.append(row)
                indexed[(str(fixed["split"]), int(fixed["batch_index"]))] = audit
                torch.cuda.empty_cache()
            split_aggregates = {
                split: _aggregate_batches(
                    [row for row in batch_rows if row["split"] == split]
                )
                for split in ("train", "validation")
            }
            recipe_results[recipe_id] = {
                "recipe": dict(recipe),
                "recipe_sha256": recipe_sha256(recipe),
                "batches": batch_rows,
                "aggregates": {
                    **split_aggregates,
                    "combined": _aggregate_batches(batch_rows),
                },
            }
            indexed_results[recipe_id] = indexed

        baseline_indexed = indexed_results["baseline-exact"]
        scale_gates = {
            recipe_id: _candidate_scale_gate(baseline_indexed, indexed_results[recipe_id])
            for recipe_id in ("candidate-conservative", "candidate-control")
        }
        model_after_sha256 = tensor_mapping_sha256(model.state_dict())
        if model_after_sha256 != checkpoint_model_sha256:
            raise DiagnosticError("gradient audit changed checkpoint model tensors")
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise DiagnosticError("gradient audit left Parameter.grad populated")
        config_after = stable_hash(OmegaConf.to_container(cfg, resolve=True))
        if config_after != config_before:
            raise DiagnosticError("gradient audit changed checkpoint config")
        del model
        torch.cuda.empty_cache()

    after = protected_tree_fingerprint(run_dir)
    if after != run_fingerprint:
        raise DiagnosticError("formal run tree changed during read-only gradient audit")
    package_files_after = {
        name: file_sha256(package / name)
        for name in ("gradient_audit.py", "recipes.json", "gradient_audit.slurm")
    }
    if package_files_after != package_files_before:
        raise DiagnosticError("diagnostic source package changed during the audit")
    snapshot_after = campaign.verify_source_snapshot(snapshot_root)
    if snapshot_after != snapshot:
        raise DiagnosticError("sealed formal source snapshot changed during the audit")
    candidate_api_after = {
        "recursive_losses_sha256": file_sha256(
            diagnostic_source_root / "treewm" / "losses" / "recursive_losses.py"
        ),
        "trainer_sha256": file_sha256(diagnostic_source_root / "scripts" / "train.py"),
    }
    if any(candidate_api_after[key] != candidate_api[key] for key in candidate_api_after):
        raise DiagnosticError("candidate grounded-execution source changed during the audit")
    package_sha256 = stable_hash(package_files_before)
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "diagnostic_id": DIAGNOSTIC_ID,
        "classification": "read_only_scale_audit_not_efficacy_evidence",
        "run": {
            "campaign_id": CAMPAIGN_ID,
            "setting_id": str(run["setting_id"]),
            "seed": 0,
            "run_name": str(run["run_name"]),
            "launch_sha256": str(claimed_launch_sha256),
            "objective_version": str(cfg.objective_version),
        },
        "checkpoint": {
            "completed_updates": int(verified["completed_updates"]),
            "checkpoint_sha256": str(verified["checkpoint_sha256"]),
            "identity_sha256": str(verified["identity_sha256"]),
            "normalizer_sha256": checkpoint_normalizer,
            "model_tensors_sha256": checkpoint_model_sha256,
            "model_tensors_after_sha256": model_after_sha256,
            "trainable_parameter_names_sha256": trainable_names_sha256,
            "stage_marker": str(verified["marker"]),
        },
        "source_snapshot": snapshot,
        "protocol": {
            "diagnostic_package_files": package_files_before,
            "diagnostic_package_sha256": package_sha256,
            "recipe_set_sha256": recipe_set["recipe_set_sha256"],
            "candidate_grounded_execution_api": candidate_api,
            "formal_trainer_source_sha256": str(launch["hashes"]["source_sha256"]),
            "formal_package_protocol_sha256": str(
                launch["hashes"]["package_protocol_sha256"]
            ),
            "formal_run_protocol_sha256": str(
                launch["hashes"]["run_protocol_sha256"]
            ),
            "formal_runtime_sha256": str(launch["hashes"]["runtime_sha256"]),
            "source_snapshot_identity_sha256": str(snapshot["snapshot_identity_sha256"]),
        },
        "method": {
            "audit_step": STAGE_TARGET,
            "splits": ["train", "validation"],
            "batches_per_split": AUDIT_BATCHES,
            "batch_size": AUDIT_BATCH_SIZE,
            "sampling": "counter_hashed_equal_rank_strata_interleaved_per_batch_v1",
            "precision": "formal_train_bfloat16_autocast",
            "model_mode": "train_with_fixed_rng_per_matching_batch",
            "terms": sorted(EXPECTED_TERMS),
            "excluded_effective_term": "expand (not a branch or multistep term)",
            "modules": [*MODULE_NAMES, "shared"],
            "shared_definition": "deduplicated union of the eight named module parameters",
            "grounded_selector_horizon_metric": "1-softmax(target_horizon)",
            "grounded_supervised_horizon_metric": "cross_entropy",
            "nonfinite_policy": "fail_without_publishing",
        },
        "guardrails": {
            "classification": "scale_safety_only_not_efficacy",
            "gradient_share_bound": GRADIENT_SHARE_BOUND,
            "shared_candidate_to_baseline_norm_ratio_bound": SHARED_NORM_RATIO_BOUND,
            "module_candidate_to_baseline_norm_ratio_bound": MODULE_NORM_RATIO_BOUND,
            "intended_path_norm_floor_exclusive": INTENDED_PATH_NORM_FLOOR,
            "candidate_results": scale_gates,
        },
        "data": data_identity,
        "sample": sample_identity,
        "baseline_checkpoint_recipe_checks": baseline_checks,
        "recipes": recipe_results,
        "read_only_proof": {
            "formal_run_tree_before_sha256": run_fingerprint,
            "formal_run_tree_after_sha256": after,
            "config_before_sha256": config_before,
            "config_after_sha256": config_after,
            "model_before_sha256": checkpoint_model_sha256,
            "model_after_sha256": model_after_sha256,
            "parameter_grad_all_none": True,
            "unchanged": True,
        },
    }
    body["input_identity_sha256"] = stable_hash(
        {
            "diagnostic_package_sha256": package_sha256,
            "recipe_set_sha256": recipe_set["recipe_set_sha256"],
            "candidate_api_sha256": candidate_api["recursive_losses_sha256"],
            "launch_sha256": claimed_launch_sha256,
            "checkpoint_sha256": verified["checkpoint_sha256"],
            "future_recipe_sha256": data_identity["future_recipe_sha256"],
            "sample": sample_identity,
        }
    )
    return write_immutable_json(destination, body)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        path, reused = run_diagnostic(args.checkpoint, args.output_root)
    except (DiagnosticError, OSError, RuntimeError, ValueError) as exc:
        print(f"gradient audit failed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"artifact": str(path), "reused": reused}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
