#!/usr/bin/env python3
"""Outcome-blind, stdout-only gradient-scale audit for prospective Exp23.

This is deliberately not a launcher.  It reads the ten immutable Exp20 GS update-5000
checkpoints and the already-sealed published-union recipes, constructs a second regime
from scratch at seeds 230/231, and prints one self-hashed JSON result.  It performs no
optimizer step, rollout, evaluation, cache build, checkpoint write, or artifact write.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any, Iterator, Mapping


sys.dont_write_bytecode = True
# These variables must be forced before the first NumPy or torch import.  The blind
# audit compares very small gradient-norm differences, so allowing a caller's BLAS or
# OpenMP thread defaults to select a reduction tree makes its final decimal floor
# nondeterministic.  A direct script invocation is the only supported execution mode;
# fail closed below if either numerical library was imported before this module.
DETERMINISM_ENVIRONMENT = {
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
NUMERICAL_LIBRARY_PREIMPORTED = any(
    name == "torch" or name.startswith("torch.") or name == "numpy" or name.startswith("numpy.")
    for name in sys.modules
)
os.environ.update(DETERMINISM_ENVIRONMENT)
PACKAGE_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PROJECT_ROOT))

AUDIT_ID = "treewm_executable_prefix_weight_audit_v1"
AUDIT_STEP = 5_000
AUDIT_BATCHES = 2
AUDIT_BATCH_SIZE = 16
AUDIT_SEEDS = (230, 231)
CHECKPOINT_SEEDS = (108, 109)
COMPONENTS = (
    "executable_prefix_action",
    "executable_prefix_latent",
    "executable_prefix_endpoint",
)
GROUPS = ("branch_transformer", "world_rest")
REGIMES = ("exp20_gs_exact_5000", "scratch_initialization")
PER_COMPONENT_MEDIAN_FRACTION = 0.03
AGGREGATE_WORST_CASE_FRACTION = 0.10
NONZERO_FLOOR = 0.0
SETTINGS = (
    "antmaze-large",
    "scene",
    "puzzle-3x3",
    "puzzle-4x4-100m",
    "cube-quadruple-100m",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CHECKPOINT_LOCK_PATH = Path(__file__).resolve().parent / "weight_audit.lock.json"


class AuditError(RuntimeError):
    pass


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def tensor_mapping_sha256(values: Mapping[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for name in sorted(values):
        value = values[name]
        if not torch.is_tensor(value):
            continue
        cpu = value.detach().cpu().contiguous()
        header = canonical_json(
            {"name": str(name), "shape": list(cpu.shape), "dtype": str(cpu.dtype)}
        ).encode("ascii")
        digest.update(len(header).to_bytes(8, "little"))
        digest.update(header)
        raw = cpu.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(len(raw).to_bytes(8, "little"))
        digest.update(raw)
    return digest.hexdigest()


def parameter_mapping_sha256(model: Any) -> str:
    return tensor_mapping_sha256(dict(model.named_parameters()))


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


def _json_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise AuditError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _parse_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_json_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                AuditError(f"non-finite JSON value in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError(f"invalid JSON in {label}: {exc}") from exc
    if not isinstance(value, dict):
        raise AuditError(f"expected JSON object: {label}")
    return value


def _open_regular(path: Path, label: str) -> tuple[int, os.stat_result]:
    parent = _open_directory_components(path.parent, f"{label} parent")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent,
        )
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise AuditError(f"{label} is not a single-link regular file")
        return descriptor, info
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        raise
    finally:
        os.close(parent)


def _authenticated_regular_bytes(
    path: Path,
    expected_sha256: str | None,
    label: str,
) -> tuple[bytes, str]:
    """Read exact bytes from one stable O_NOFOLLOW inode and authenticate them."""

    if expected_sha256 is not None and SHA256.fullmatch(str(expected_sha256)) is None:
        raise AuditError(f"{label} expected SHA256 is malformed")
    descriptor, before = _open_regular(path, label)
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    copied = 0
    try:
        while block := os.read(descriptor, 16 * 1024 * 1024):
            digest.update(block)
            chunks.append(block)
            copied += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _file_identity(after) != _file_identity(before) or copied != before.st_size:
        raise AuditError(f"{label} changed while being authenticated")
    actual = digest.hexdigest()
    if expected_sha256 is not None and actual != expected_sha256:
        raise AuditError(f"{label} SHA256 differs from weight-audit lock")
    return b"".join(chunks), actual


def read_json(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    label: str | None = None,
) -> dict[str, Any]:
    source = Path(path)
    payload, _digest = _authenticated_regular_bytes(
        source, expected_sha256, label or str(source)
    )
    return _parse_json_bytes(payload, label or str(source))


def load_weight_lock(expected_sha256: str) -> dict[str, Any]:
    return read_json(
        CHECKPOINT_LOCK_PATH,
        expected_sha256=expected_sha256,
        label="adjacent weight-audit lock",
    )


def frozen_checkpoint_sha256(lock: Mapping[str, Any]) -> dict[str, str]:
    mapping = lock.get("checkpoint_sha256")
    expected_keys = {
        f"{setting}/seed{seed}" for setting in SETTINGS for seed in CHECKPOINT_SEEDS
    }
    if not isinstance(mapping, dict) or set(mapping) != expected_keys:
        raise AuditError("frozen checkpoint hash map is missing or has extra entries")
    result = {str(key): str(value) for key, value in mapping.items()}
    if not all(SHA256.fullmatch(value) is not None for value in result.values()):
        raise AuditError("frozen checkpoint hash map contains malformed SHA256")
    return result


def _external_input_keys() -> set[str]:
    return {
        "exp20/manifest.json",
        *(
            f"{setting}/seed{seed}/GAUGE_PILOT_V2_LAUNCH.json"
            for setting in SETTINGS
            for seed in CHECKPOINT_SEEDS
        ),
    }


def frozen_external_input_sha256(lock: Mapping[str, Any]) -> dict[str, str]:
    mapping = lock.get("external_input_sha256")
    expected_keys = _external_input_keys()
    if not isinstance(mapping, dict) or set(mapping) != expected_keys:
        raise AuditError(
            "frozen external-input hash map is missing or has extra entries"
        )
    result = {str(key): str(value) for key, value in mapping.items()}
    if not all(SHA256.fullmatch(value) is not None for value in result.values()):
        raise AuditError("frozen external-input hash map contains malformed SHA256")
    return dict(sorted(result.items()))


def _open_directory_components(path: Path, label: str) -> int:
    absolute = path.absolute()
    if not absolute.is_absolute() or any(part in ("", ".", "..") for part in absolute.parts[1:]):
        raise AuditError(f"{label} is not an absolute normalized path")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(absolute.anchor, flags)
    try:
        for part in absolute.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        info = os.fstat(descriptor)
        if not stat.S_ISDIR(info.st_mode):
            raise AuditError(f"{label} is not a nonsymlink directory")
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def exact_checkpoint_run(output_root: Path, setting: str, seed: int) -> Path:
    tree_root = output_root / setting / "treewm"
    descriptor = _open_directory_components(tree_root, f"{setting} checkpoint run root")
    try:
        suffix = f"armgs-seed{seed}"
        candidates = sorted(name for name in os.listdir(descriptor) if name.endswith(suffix))
        if len(candidates) != 1:
            raise AuditError(
                f"{setting}/seed{seed}: expected exactly one frozen GS run, found {len(candidates)}"
            )
        info = os.stat(candidates[0], dir_fd=descriptor, follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode):
            raise AuditError(f"{setting}/seed{seed}: frozen GS run is not a nonsymlink directory")
        return tree_root / candidates[0]
    finally:
        os.close(descriptor)


def load_frozen_external_inputs(
    project_root: Path,
    lock: Mapping[str, Any],
) -> tuple[
    dict[str, Any],
    dict[str, dict[int, Path]],
    dict[str, dict[int, dict[str, Any]]],
    dict[str, str],
]:
    """Authenticate every external control JSON before any checkpoint is loaded."""

    expected = frozen_external_input_sha256(lock)
    manifest_path = (
        project_root
        / "experiments"
        / "20-treewm-grounded-gauge-pilot-v2"
        / "manifest.json"
    )
    output_root = project_root / "outputs" / "treewm-grounded-gauge-pilot-v2-launch2"
    manifest = read_json(
        manifest_path,
        expected_sha256=expected["exp20/manifest.json"],
        label="frozen Exp20 manifest",
    )
    if tuple(setting.get("id") for setting in manifest.get("settings", ())) != SETTINGS:
        raise AuditError("Exp20 five-setting order differs")

    run_dirs: dict[str, dict[int, Path]] = {}
    launches: dict[str, dict[int, dict[str, Any]]] = {}
    for setting in SETTINGS:
        run_dirs[setting] = {
            seed: exact_checkpoint_run(output_root, setting, seed)
            for seed in CHECKPOINT_SEEDS
        }
        launches[setting] = {}
        for seed in CHECKPOINT_SEEDS:
            key = f"{setting}/seed{seed}/GAUGE_PILOT_V2_LAUNCH.json"
            launches[setting][seed] = read_json(
                run_dirs[setting][seed] / "GAUGE_PILOT_V2_LAUNCH.json",
                expected_sha256=expected[key],
                label=f"{setting}/seed{seed} frozen Exp20 launch",
            )
    return manifest, run_dirs, launches, expected


def _open_checkpoint(run_dir: Path) -> tuple[int, os.stat_result]:
    run_descriptor = _open_directory_components(run_dir, "frozen checkpoint run")
    checkpoint_descriptor: int | None = None
    try:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        checkpoints = os.open("checkpoints", directory_flags, dir_fd=run_descriptor)
        try:
            checkpoint_descriptor = os.open(
                "latest.pt",
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=checkpoints,
            )
        finally:
            os.close(checkpoints)
        info = os.fstat(checkpoint_descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise AuditError("frozen checkpoint is not a single-link regular file")
        return checkpoint_descriptor, info
    except BaseException:
        if checkpoint_descriptor is not None:
            os.close(checkpoint_descriptor)
        raise
    finally:
        os.close(run_descriptor)


def load_frozen_checkpoint(
    run_dir: Path,
    expected_sha256: str,
    torch_module: Any,
) -> tuple[Any, str]:
    """Authenticate one source inode into a private copy before pickle executes."""

    if SHA256.fullmatch(str(expected_sha256)) is None:
        raise AuditError("expected checkpoint SHA256 is malformed")
    source_descriptor, before = _open_checkpoint(run_dir)
    digest = hashlib.sha256()
    copied = 0
    temporary_directory = os.environ.get("TMPDIR")
    try:
        with tempfile.TemporaryFile(mode="w+b", dir=temporary_directory) as verified:
            private_info = os.fstat(verified.fileno())
            if (
                not stat.S_ISREG(private_info.st_mode)
                or private_info.st_uid != os.getuid()
                or private_info.st_mode & 0o077
            ):
                raise AuditError("private checkpoint copy is not a private regular file")
            while block := os.read(source_descriptor, 16 * 1024 * 1024):
                digest.update(block)
                verified.write(block)
                copied += len(block)
            after = os.fstat(source_descriptor)
            if _file_identity(after) != _file_identity(before) or copied != before.st_size:
                raise AuditError("frozen checkpoint changed while being authenticated")
            actual = digest.hexdigest()
            if actual != expected_sha256:
                raise AuditError("frozen checkpoint SHA256 differs from weight-audit lock")
            verified.flush()
            verified.seek(0)
            payload = torch_module.load(
                verified, map_location="cpu", weights_only=False
            )
            if os.fstat(verified.fileno()).st_size != copied:
                raise AuditError("private checkpoint copy changed during torch.load")
            return payload, actual
    finally:
        os.close(source_descriptor)


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


def _u64(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "little")


def fixed_positions(population: int, setting: str, regime: str) -> list[list[int]]:
    count = AUDIT_BATCHES * AUDIT_BATCH_SIZE
    if population < count:
        raise AuditError(f"{setting}/{regime}: population {population} < {count}")
    positions: list[int] = []
    for index in range(count):
        start = (index * population) // count
        stop = ((index + 1) * population) // count
        offset = _u64(f"{AUDIT_ID}:{setting}:{regime}:stratum:{index}") % (stop - start)
        positions.append(start + offset)
    if len(set(positions)) != count:
        raise AuditError("counter-hash strata produced duplicate positions")
    return [positions[batch::AUDIT_BATCHES] for batch in range(AUDIT_BATCHES)]


def batch_sha256(batch: Mapping[str, Any]) -> str:
    return tensor_mapping_sha256(batch)


def _find_cache_manifest(cache_root: Path, source_name: str, source_sha: str):
    matches = []
    for path in sorted(cache_root.glob("*/manifest.json"), key=str):
        payload = read_json(path)
        name = payload.get("dataset") or payload.get("dataset_name")
        if name == source_name and payload.get("source_manifest_sha256") == source_sha:
            matches.append((path, payload))
    if len(matches) != 1:
        raise AuditError(
            f"expected one cache for {source_name}/{source_sha}, found {len(matches)}"
        )
    return matches[0]


def load_read_only_datasets(cfg: Any, launch: Mapping[str, Any]):
    from dataclasses import replace

    from treewm.data.ogbench_dataset import (
        ChunkDataset,
        Normalizer,
        _attach_future_recipes_if_requested,
    )
    from treewm.utils import config as cfg_utils

    environment = launch["environment"]
    source_sha = str(environment["TREEWM_DATA_SHA256"])
    cache_root = Path(environment["TREEWM_CACHE"]).resolve()
    source_name = str(cfg.env.get("source_name", cfg.env.name))
    manifest_path, manifest = _find_cache_manifest(cache_root, source_name, source_sha)
    if str(cfg.env.get("dataset_kind", "standard")) == "sharded_100m_full":
        from treewm.data.sharded_ogbench import _load_cache

        cache = _load_cache(manifest_path.parent, manifest, was_hit=True)
    else:
        from treewm.data.shared_cache import _cache_from_manifest

        cache = _cache_from_manifest(manifest_path.parent, manifest, was_hit=True)
    if str(cache.source_manifest_sha256) != source_sha:
        raise AuditError("cache source identity differs from launch")
    normalizer = Normalizer.from_state_dict(cache.norm_stats)
    future_cfg = cfg_utils.future_set_config(cfg)
    if cfg.env.get("relative_endpoints") is not None:
        future_cfg = replace(
            future_cfg, relative_endpoints=bool(cfg.env.get("relative_endpoints"))
        )
    from treewm.evaluation.domains import get_domain

    domain = get_domain(str(cfg.env.name))
    common = {
        "xy_dims": tuple(cfg.env.xy_dims),
        "cache_future_sets": False,
        "task_metric_dims": tuple(cfg.env.get("task_metric_dims") or cfg.env.xy_dims),
        "task_goal_metric": str(domain.goal_metric),
        "task_subgoals": tuple(domain.subgoals),
    }
    train_ds = ChunkDataset(
        {}, normalizer, future_cfg, max_anchors=0, seed=int(cfg.seed),
        shared=cache.train, **common,
    )
    val_ds = ChunkDataset(
        {}, normalizer, future_cfg, max_anchors=0, seed=int(cfg.seed) + 1,
        shared=cache.val, **common,
    )
    cache.assert_consumed_by(train_ds, val_ds)
    _attach_future_recipes_if_requested(
        train_ds,
        val_ds,
        normalizer,
        future_cfg,
        source_sha,
        anchor_policy="published_union",
    )
    expected_recipe = str(environment["TREEWM_FUTURE_RECIPE_SHA256"])
    if str(train_ds.future_recipe_sha256) != expected_recipe:
        raise AuditError("future recipe identity differs from launch")
    identity = {
        "cache_manifest": str(manifest_path),
        "cache_manifest_sha256": file_sha256(manifest_path),
        "source_manifest_sha256": source_sha,
        "future_recipe_sha256": expected_recipe,
        "train_population": len(train_ds),
        "validation_population": len(val_ds),
    }
    return train_ds, val_ds, normalizer, domain, identity


def materialize_batches(dataset: Any, setting: str, regime: str):
    import numpy as np
    from torch.utils.data import default_collate

    rows = []
    for index, positions in enumerate(fixed_positions(len(dataset), setting, regime)):
        batch = default_collate([dataset[position] for position in positions])
        anchors = [int(dataset.anchors[position]) for position in positions]
        for name, value in batch.items():
            if value.is_floating_point() and not bool(value.isfinite().all()):
                raise AuditError(f"{setting}/{regime}/batch{index}: non-finite {name}")
        rows.append(
            {
                "batch_index": index,
                "positions": positions,
                "positions_sha256": hashlib.sha256(
                    np.asarray(positions, dtype="<i8").tobytes()
                ).hexdigest(),
                "anchors_sha256": hashlib.sha256(
                    np.asarray(anchors, dtype="<i8").tobytes()
                ).hexdigest(),
                "batch_sha256": batch_sha256(batch),
                "batch": batch,
            }
        )
    return rows


def prepare_cfg(payload_cfg: Mapping[str, Any], *, seed: int, lower: float, upper: float):
    from omegaconf import OmegaConf, open_dict

    cfg = OmegaConf.create(payload_cfg)
    with open_dict(cfg):
        cfg.seed = int(seed)
        cfg.objective_version = "treewm_v2_grounded_executable_prefix_pilot_v1"
        cfg.device = "cpu"
        cfg.future_sets.executable_prefix_steps = 4
        cfg.losses.executable_action_lower_bound = float(lower)
        cfg.losses.executable_action_upper_bound = float(upper)
        cfg.planner.action_lower_bound = float(lower)
        cfg.planner.action_upper_bound = float(upper)
        for name in COMPONENTS:
            cfg.losses.enabled[name] = True
            cfg.losses.weights[name] = 1.0
        # The audit evaluates all terms at the exact 5k campaign boundary.
        for name in COMPONENTS:
            cfg.losses.warmup[name] = 0
            cfg.losses.decay[name] = 0
    return cfg


def build_model_for_audit(cfg: Any, *, checkpoint: Mapping[str, Any] | None, seed: int):
    import torch

    from treewm.losses.latent_gauge import LatentGauge
    from treewm.models.baselines import build_model
    from treewm.utils import config as cfg_utils

    torch.manual_seed(int(seed))
    loss_cfg = cfg_utils.loss_config(cfg)
    model = build_model(
        str(cfg.arm), cfg_utils.model_config(cfg), k_max=int(cfg.model.flatk_max)
    )
    model.add_module(
        "latent_gauge",
        LatentGauge(
            epsilon=float(loss_cfg.latent_gauge_epsilon),
            min_reference_scale=float(loss_cfg.latent_gauge_min_reference_scale),
        ),
    )
    model.gain_head.set_set_aware(bool(loss_cfg.gain_set_context))
    model.tree_signature.requires_grad_(False)
    if loss_cfg.gain_set_context and float(loss_cfg.gain_branch_prior_weight) == 0.0:
        model.heads.gain_head.requires_grad_(False)
    if not loss_cfg.on("mass"):
        model.heads.mass_head.requires_grad_(False)
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"], strict=True)
    model.set_gradient_checkpointing(False)
    model.train()
    return model, loss_cfg


def _to_cpu(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {name: value.to("cpu") for name, value in batch.items()}


def _norm_for_group(grads: tuple[Any, ...], indices: list[int], like: Any) -> float:
    import torch

    values = [
        grads[index].detach().float().square().sum()
        for index in indices
        if grads[index] is not None
    ]
    value = torch.stack(values).sum().sqrt() if values else like.new_zeros(())
    result = float(value.item())
    if not math.isfinite(result):
        raise AuditError("gradient norm is non-finite")
    return result


def audit_batch(model: Any, loss_cfg: Any, match_cfg: Any, batch: Mapping[str, Any], seed: int):
    import torch

    from scripts.train import (
        gradient_parameter_groups,
        multistep_transition_kwargs,
        split_branch_transformer_parameters,
    )
    from treewm.losses.recursive_losses import (
        multi_step_recursive_loss,
        scheduled_sampling_schedule,
    )
    from treewm.losses.total import assemble_loss_terms, compute_branch_losses

    torch.manual_seed(int(seed))
    model._horizon_gen = torch.Generator(device="cpu")
    model._horizon_gen.manual_seed(int(seed) ^ 0x5A17)
    batch = _to_cpu(batch)
    _, metrics, _, branch_terms = compute_branch_losses(
        model, batch, loss_cfg, match_cfg, step=AUDIT_STEP, return_loss_terms=True
    )
    raw = dict(branch_terms.raw)
    p_ss = scheduled_sampling_schedule(
        AUDIT_STEP,
        float(loss_cfg.scheduled_sampling_p),
        int(loss_cfg.scheduled_sampling_warmup),
    )
    multistep, _ = multi_step_recursive_loss(
        model,
        batch,
        scheduled_sampling_p=p_ss,
        scheduled_sampling_granularity=str(loss_cfg.scheduled_sampling_granularity),
        depth_weights=loss_cfg.multistep_depth_weights or None,
        **multistep_transition_kwargs(loss_cfg),
    )
    raw["multistep"] = multistep
    terms = assemble_loss_terms(raw, loss_cfg, AUDIT_STEP)
    base_names = [name for name in terms.effective if name not in COMPONENTS]
    base = sum(terms.effective[name] for name in base_names)
    if not bool(torch.isfinite(base)):
        raise AuditError("base objective is non-finite")

    world, _ = gradient_parameter_groups(model, include_branch_prior=False)
    rest, branch = split_branch_transformer_parameters(model, world)
    parameters = [*branch, *rest]
    branch_indices = list(range(len(branch)))
    rest_indices = list(range(len(branch), len(parameters)))
    group_indices = {
        "branch_transformer": branch_indices,
        "world_rest": rest_indices,
    }
    objectives = [("base", base), *[(name, raw[name]) for name in COMPONENTS]]
    norms: dict[str, dict[str, float]] = {}
    for offset, (name, objective) in enumerate(objectives):
        grads = torch.autograd.grad(
            objective,
            parameters,
            retain_graph=offset < len(objectives) - 1,
            allow_unused=True,
        )
        norms[name] = {
            group: _norm_for_group(grads, indices, base)
            for group, indices in group_indices.items()
        }
    for component in COMPONENTS:
        for group in GROUPS:
            if norms[component][group] <= NONZERO_FLOOR:
                raise AuditError(f"zero {component}/{group} gradient")
    for group in GROUPS:
        if norms["base"][group] <= NONZERO_FLOOR:
            raise AuditError(f"zero base/{group} gradient")
    if any(parameter.grad is not None for parameter in model.parameters()):
        raise AuditError("audit populated Parameter.grad")
    return {
        "norms": norms,
        "base_terms": sorted(base_names),
        "prefix_steps_mean": float(
            metrics["train/executable_prefix/prefix_steps_mean"]
        ),
        "valid_anchor_fraction": float(
            metrics["train/executable_prefix/valid_anchor_fraction"]
        ),
        "schema_version": float(metrics["train/executable_prefix/schema_version"]),
    }


def seal_scratch_gauge(model: Any, loss_cfg: Any, match_cfg: Any, batch: Mapping[str, Any], seed: int):
    import torch

    from treewm.losses.total import compute_branch_losses

    if model.latent_gauge.is_sealed:
        raise AuditError("scratch gauge unexpectedly starts sealed")
    torch.manual_seed(int(seed))
    model._horizon_gen = torch.Generator(device="cpu")
    model._horizon_gen.manual_seed(int(seed) ^ 0x71A5)
    with torch.no_grad():
        compute_branch_losses(
            model,
            _to_cpu(batch),
            loss_cfg,
            match_cfg,
            step=0,
            return_loss_terms=False,
        )
    if not model.latent_gauge.is_sealed:
        raise AuditError("scratch gauge failed to seal at update zero")


def median(values: list[float]) -> float:
    import statistics

    if not values or not all(math.isfinite(value) and value > 0.0 for value in values):
        raise AuditError("median input is empty, non-finite, or nonpositive")
    return float(statistics.median(values))


def floor_significant(value: float, digits: int = 8) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise AuditError("cannot freeze nonpositive weight")
    power = math.floor(math.log10(value)) - digits + 1
    quantum = 10.0 ** power
    return math.floor(value / quantum) * quantum


def derive_weights(rows: list[dict[str, Any]]):
    candidates: dict[str, list[dict[str, Any]]] = {name: [] for name in COMPONENTS}
    weights: dict[str, float] = {}
    for component in COMPONENTS:
        for regime in REGIMES:
            subset = [row for row in rows if row["regime"] == regime]
            for group in GROUPS:
                base_median = median([row["norms"]["base"][group] for row in subset])
                component_median = median(
                    [row["norms"][component][group] for row in subset]
                )
                candidate = (
                    PER_COMPONENT_MEDIAN_FRACTION * base_median / component_median
                )
                candidates[component].append(
                    {
                        "regime": regime,
                        "group": group,
                        "base_median": base_median,
                        "component_raw_median": component_median,
                        "candidate_weight": candidate,
                    }
                )
        weights[component] = min(row["candidate_weight"] for row in candidates[component])

    def maximum_ratio(current: Mapping[str, float]) -> float:
        return max(
            sum(current[name] * row["norms"][name][group] for name in COMPONENTS)
            / row["norms"]["base"][group]
            for row in rows
            for group in GROUPS
        )

    pre_scale_max = maximum_ratio(weights)
    common_scale = min(1.0, AGGREGATE_WORST_CASE_FRACTION / pre_scale_max)
    frozen = {
        name: floor_significant(weights[name] * common_scale, 8)
        for name in COMPONENTS
    }
    post_scale_max = maximum_ratio(frozen)
    if post_scale_max > AGGREGATE_WORST_CASE_FRACTION + 1e-12:
        raise AuditError("frozen weights violate aggregate gradient budget")
    return {
        "weights": frozen,
        "component_candidates": candidates,
        "pre_scale_max_aggregate_ratio": pre_scale_max,
        "common_scale": common_scale,
        "post_scale_max_aggregate_ratio": post_scale_max,
    }


def _run_fingerprint(run_dir: Path) -> str:
    rows = []
    for path in sorted(run_dir.rglob("*"), key=str):
        stat = path.lstat()
        rows.append(
            [str(path.relative_to(run_dir)), stat.st_size, stat.st_mtime_ns, stat.st_mode]
        )
    return stable_hash(rows)


def run(project_root: Path, expected_weight_lock_sha256: str) -> dict[str, Any]:
    if NUMERICAL_LIBRARY_PREIMPORTED:
        raise AuditError(
            "numpy/torch was imported before the audit pinned its thread environment"
        )
    import numpy as np
    import torch
    from omegaconf import OmegaConf

    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        raise AuditError(f"cannot seal torch thread pools: {exc}") from exc
    torch.use_deterministic_algorithms(True)
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise AuditError("torch thread pools differ from the sealed single-thread audit")
    if any(os.environ.get(name) != value for name, value in DETERMINISM_ENVIRONMENT.items()):
        raise AuditError("numerical thread environment differs after import")

    from scripts.train import validate_executable_prefix_configuration
    from treewm.data.future_recipe import normalizer_state_sha256
    from treewm.data.ogbench_dataset import load_ogbench
    from treewm.models.baselines import tree_config_for
    from treewm.utils import config as cfg_utils

    weight_lock = load_weight_lock(expected_weight_lock_sha256)
    (
        _exp20_manifest,
        run_dirs_by_setting,
        launches_by_setting,
        locked_external_inputs,
    ) = load_frozen_external_inputs(project_root, weight_lock)
    locked_checkpoint_hashes = frozen_checkpoint_sha256(weight_lock)

    protected_before: dict[str, str] = {}
    checkpoint_hashes: dict[str, str] = {}
    rows: list[dict[str, Any]] = []
    data_identities: dict[str, Any] = {}
    bounds_by_setting: dict[str, Any] = {}
    batch_identities: dict[str, Any] = {}

    for setting in SETTINGS:
        run_dirs = run_dirs_by_setting[setting]
        launches = launches_by_setting[setting]
        checkpoints: dict[int, Any] = {}
        for seed, run_dir in run_dirs.items():
            protected_before[str(run_dir)] = _run_fingerprint(run_dir)
            checkpoint_key = f"{setting}/seed{seed}"
            payload, checkpoint_digest = load_frozen_checkpoint(
                run_dir,
                locked_checkpoint_hashes[checkpoint_key],
                torch,
            )
            checkpoint_hashes[checkpoint_key] = checkpoint_digest
            if int(payload.get("completed_updates", -1)) != AUDIT_STEP:
                raise AuditError(f"{setting}/seed{seed}: checkpoint is not exact 5k")
            checkpoints[seed] = payload

        base_payload = checkpoints[CHECKPOINT_SEEDS[0]]
        base_launch = launches[CHECKPOINT_SEEDS[0]]
        env_values = {
            str(key): str(value) for key, value in base_launch["environment"].items()
        }
        env_values.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONNOUSERSITE": "1",
                "WANDB_MODE": "disabled",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
            }
        )
        with patched_environment(env_values):
            cfg_for_env = OmegaConf.create(base_payload["config"])
            env = load_ogbench(
                str(cfg_for_env.env.name),
                dataset_dir=str(cfg_for_env.env.dataset_dir),
                env_only=True,
            )
            low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
            if (
                low.size == 0
                or low.shape != high.shape
                or not np.isfinite(low).all()
                or not np.isfinite(high).all()
                or not np.all(low < high)
                or not np.all(low == low[0])
                or not np.all(high == high[0])
            ):
                raise AuditError(f"{setting}: action bounds are not finite uniform Box")
            lower, upper = float(low[0]), float(high[0])
            bounds_by_setting[setting] = {
                "lower": lower,
                "upper": upper,
                "action_dim": int(low.size),
                "lower_sha256": hashlib.sha256(low.tobytes()).hexdigest(),
                "upper_sha256": hashlib.sha256(high.tobytes()).hexdigest(),
            }
            env.close()

            data_cfg = prepare_cfg(
                base_payload["config"], seed=CHECKPOINT_SEEDS[0], lower=lower, upper=upper
            )
            train_ds, val_ds, normalizer, domain, data_identity = load_read_only_datasets(
                data_cfg, base_launch
            )
            data_identities[setting] = data_identity
            checkpoint_normalizer = normalizer_state_sha256(base_payload["normalizer"])
            if checkpoint_normalizer != normalizer_state_sha256(normalizer.state_dict()):
                raise AuditError(f"{setting}: checkpoint/cache normalizer mismatch")
            batches_by_regime = {
                regime: materialize_batches(train_ds, setting, regime)
                for regime in REGIMES
            }
            batch_identities[setting] = {
                regime: [
                    {key: value for key, value in row.items() if key != "batch"}
                    for row in batches
                ]
                for regime, batches in batches_by_regime.items()
            }
            del train_ds, val_ds

            for regime, seeds in (
                ("exp20_gs_exact_5000", CHECKPOINT_SEEDS),
                ("scratch_initialization", AUDIT_SEEDS),
            ):
                for seed in seeds:
                    payload = checkpoints[seed] if regime == REGIMES[0] else None
                    source_cfg = (
                        checkpoints[seed]["config"] if payload is not None else base_payload["config"]
                    )
                    cfg = prepare_cfg(source_cfg, seed=seed, lower=lower, upper=upper)
                    model, loss_cfg = build_model_for_audit(
                        cfg, checkpoint=payload, seed=seed
                    )
                    match_cfg = cfg_utils.matching_config(cfg)
                    validate_executable_prefix_configuration(
                        str(cfg.objective_version),
                        loss_cfg,
                        cfg_utils.future_set_config(cfg),
                        cfg_utils.planner_config(cfg),
                        tree_cfg=tree_config_for(
                            str(cfg.arm), cfg_utils.tree_config(cfg), model
                        ),
                        action_space=type(
                            "SealedBox",
                            (),
                            {"low": low.copy(), "high": high.copy()},
                        )(),
                        model=model,
                    )
                    before_parameters = parameter_mapping_sha256(model)
                    if regime == REGIMES[0]:
                        if not model.latent_gauge.is_sealed:
                            raise AuditError(f"{setting}/seed{seed}: checkpoint gauge unsealed")
                    else:
                        seal_scratch_gauge(
                            model,
                            loss_cfg,
                            match_cfg,
                            batches_by_regime[regime][0]["batch"],
                            _u64(f"{AUDIT_ID}:{setting}:{regime}:{seed}:seal") % (2**63 - 1),
                        )
                    for fixed in batches_by_regime[regime]:
                        audit_seed = _u64(
                            f"{AUDIT_ID}:{setting}:{regime}:{seed}:batch:{fixed['batch_index']}"
                        ) % (2**63 - 1)
                        result = audit_batch(
                            model,
                            loss_cfg,
                            match_cfg,
                            fixed["batch"],
                            audit_seed,
                        )
                        rows.append(
                            {
                                "setting": setting,
                                "regime": regime,
                                "model_seed": int(seed),
                                "batch_index": int(fixed["batch_index"]),
                                "batch_sha256": fixed["batch_sha256"],
                                "rng_seed": int(audit_seed),
                                **result,
                            }
                        )
                    if parameter_mapping_sha256(model) != before_parameters:
                        raise AuditError(f"{setting}/{regime}/seed{seed}: parameters mutated")
                    del model
        del checkpoints

    if len(rows) != len(SETTINGS) * len(REGIMES) * 2 * AUDIT_BATCHES:
        raise AuditError(f"unexpected audit row count: {len(rows)}")
    for run_dir, fingerprint in protected_before.items():
        if _run_fingerprint(Path(run_dir)) != fingerprint:
            raise AuditError(f"protected Exp20 run changed during audit: {run_dir}")
    for setting in SETTINGS:
        for seed in CHECKPOINT_SEEDS:
            run_dir = exact_checkpoint_run(
                project_root
                / "outputs"
                / "treewm-grounded-gauge-pilot-v2-launch2",
                setting,
                seed,
            )
            descriptor, before = _open_checkpoint(run_dir)
            digest = hashlib.sha256()
            copied = 0
            try:
                while block := os.read(descriptor, 16 * 1024 * 1024):
                    digest.update(block)
                    copied += len(block)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            if (
                _file_identity(after) != _file_identity(before)
                or copied != before.st_size
                or digest.hexdigest() != checkpoint_hashes[f"{setting}/seed{seed}"]
            ):
                raise AuditError(f"checkpoint changed during audit: {run_dir}")

    source_files = {
        "trainer": project_root / "scripts" / "train.py",
        "executable_loss": project_root / "treewm" / "losses" / "executable_prefix.py",
        "action_projection": project_root / "treewm" / "planning" / "action_execution.py",
        "future_recipe": project_root / "treewm" / "data" / "future_recipe.py",
        "future_sets": project_root / "treewm" / "data" / "future_sets.py",
        "dataset": project_root / "treewm" / "data" / "ogbench_dataset.py",
        "total_loss": project_root / "treewm" / "losses" / "total.py",
        "objective_config": project_root / "configs" / "experiment" / "treewm_v2_grounded_executable_prefix_pilot_v1.yaml",
        "audit": Path(__file__).resolve(),
    }
    result = {
        "schema_version": 1,
        "status": "complete",
        "audit_id": AUDIT_ID,
        "classification": "outcome_blind_scale_only_no_optimizer_no_rollout",
        "contract": {
            "audit_step": AUDIT_STEP,
            "batch_count_per_setting_regime": AUDIT_BATCHES,
            "batch_size": AUDIT_BATCH_SIZE,
            "checkpoint_seeds": list(CHECKPOINT_SEEDS),
            "scratch_seeds": list(AUDIT_SEEDS),
            "settings": list(SETTINGS),
            "groups": list(GROUPS),
            "components": list(COMPONENTS),
            "per_component_median_fraction": PER_COMPONENT_MEDIAN_FRACTION,
            "aggregate_worst_case_fraction": AGGREGATE_WORST_CASE_FRACTION,
            "device": "cpu",
            "autocast": False,
            "gradient_checkpointing": False,
            "determinism": {
                "environment": dict(DETERMINISM_ENVIRONMENT),
                "torch_num_threads": int(torch.get_num_threads()),
                "torch_num_interop_threads": int(torch.get_num_interop_threads()),
                "torch_deterministic_algorithms": bool(
                    torch.are_deterministic_algorithms_enabled()
                ),
                "torch_version": str(torch.__version__),
                "numpy_version": str(np.__version__),
                "torch_mkldnn_enabled": bool(torch.backends.mkldnn.enabled),
                "torch_float32_matmul_precision": str(
                    torch.get_float32_matmul_precision()
                ),
            },
        },
        "derived": derive_weights(rows),
        "rows": rows,
        "checkpoint_sha256": checkpoint_hashes,
        "external_input_sha256": locked_external_inputs,
        "batch_identities": batch_identities,
        "data_identities": data_identities,
        "action_bounds": bounds_by_setting,
        "source_sha256": {
            name: file_sha256(path) for name, path in source_files.items()
        },
    }
    result["artifact_sha256"] = stable_hash(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--weight-lock-sha256", required=True)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    try:
        result = run(
            args.project_root.expanduser().resolve(),
            args.weight_lock_sha256,
        )
    except Exception as exc:
        print(f"weight audit failed: {exc}", file=sys.stderr)
        return 1
    if args.summary_only:
        summary = {
            key: result[key]
            for key in (
                "schema_version",
                "status",
                "audit_id",
                "classification",
                "artifact_sha256",
                "contract",
                "derived",
                "checkpoint_sha256",
                "external_input_sha256",
                "batch_identities",
                "data_identities",
                "action_bounds",
                "source_sha256",
            )
        }
        summary["row_count"] = len(result["rows"])
        summary["rows_sha256"] = stable_hash(result["rows"])
        summary["summary_sha256"] = stable_hash(summary)
        print("EXP23_WEIGHT_AUDIT_SUMMARY=" + canonical_json(summary), flush=True)
    else:
        print("EXP23_WEIGHT_AUDIT_JSON=" + canonical_json(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
