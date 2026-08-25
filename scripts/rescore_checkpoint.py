#!/usr/bin/env python3
"""Read-only held-out validation rescore for paused TreeWM checkpoints.

The scorer reconstructs the checkpoint's immutable formal dataset and compact future
recipe, evaluates a fixed representative validation sample, and writes one exclusive,
content-addressed JSON artifact under a caller-owned output root. It never writes in a
training run directory and never restores optimizer/RNG state into a trainer.

Examples::

    python scripts/rescore_checkpoint.py \
      --checkpoint outputs/.../checkpoints/latest.pt --device cuda

    python scripts/rescore_checkpoint.py \
      --checkpoint-glob 'outputs/treewm-50task-1m-v2/*/treewm/*/checkpoints/latest.pt' \
      --work-index 0 --resume
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import glob
import hashlib
import json
import math
import os
from dataclasses import replace
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import default_collate

from scripts.checkpoint_ablation import (
    _json_safe,
    _stat_identity,
    checkpoint_provenance,
    git_provenance,
    sha256_file,
    stable_hash,
    write_or_validate,
)
from scripts.train import (
    add_validation_label_metrics,
    fixed_validation_rng,
)
from treewm.data.future_recipe import normalizer_state_sha256
from treewm.data.ogbench_dataset import Normalizer, build_datasets
from treewm.data.samplers import (
    build_fixed_validation_dataloader,
    to_device,
)
from treewm.evaluation import diagnostics as diag
from treewm.logging.metrics import MetricTracker
from treewm.losses.total import compute_branch_losses
from treewm.models.baselines import build_model
from treewm.utils import config as cfg_utils
from treewm.utils.provenance import runtime_fingerprint, trainer_code_fingerprint


SCHEMA_VERSION = 1
DEFAULT_CHECKPOINT_GLOB = (
    "outputs/treewm-50task-1m-v2/*/treewm/*/checkpoints/latest.pt"
)
DEFAULT_CONTRACT_ROOT = (
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/datasets/treewm-50task-formal-v2-contracts-v1"
)
DEFAULT_CACHE_ROOT = (
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/datasets/treewm-50task-full-cache-v1"
)
DEFAULT_OUTPUT_ROOT = "outputs/treewm-v2-validation-rescore"
SHA_FIELDS = {
    "TREEWM_PROTOCOL_SHA256": "protocol_sha256",
    "TREEWM_CODE_SHA256": "code_sha256",
    "TREEWM_RUNTIME_SHA256": "runtime_sha256",
    "TREEWM_DATA_SHA256": "data_manifest_sha256",
    "TREEWM_CALIBRATION_SHA256": "calibration_sha256",
    "TREEWM_FUTURE_RECIPE_SHA256": "future_recipe_sha256",
}


def discover_checkpoints(
    checkpoint_values: Sequence[str],
    patterns: Sequence[str],
    *,
    repo_root: Path = REPOSITORY_ROOT,
) -> list[Path]:
    """Resolve direct checkpoint paths and globs into one deterministic work list."""
    found: set[Path] = set()
    for value in checkpoint_values:
        path = Path(value).expanduser()
        path = path if path.is_absolute() else repo_root / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {path}")
        found.add(path)
    for pattern in patterns:
        expanded = Path(pattern).expanduser()
        query = str(expanded if expanded.is_absolute() else repo_root / expanded)
        found.update(
            Path(value).resolve()
            for value in glob.glob(query, recursive=True)
            if Path(value).is_file()
        )
    return sorted(found, key=str)


def select_work(checkpoints: Sequence[Path], work_index: int | None) -> list[Path]:
    if work_index is None:
        return list(checkpoints)
    if not 0 <= int(work_index) < len(checkpoints):
        raise IndexError(f"work index {work_index} outside [0, {len(checkpoints)})")
    return [checkpoints[int(work_index)]]


def evaluation_code_fingerprint(repo_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Hash every live source/config input capable of changing this measurement."""
    root = repo_root.resolve()
    candidates = {
        Path(__file__).resolve(),
        root / "scripts" / "train.py",
        root / "scripts" / "checkpoint_ablation.py",  # artifact helpers imported above
        *(root / "treewm").rglob("*.py"),
        *(root / "configs").rglob("*.yaml"),
    }
    files: dict[str, str] = {}
    manifest = hashlib.sha256()
    for path in sorted(candidate.resolve() for candidate in candidates):
        if not path.is_file() or not path.is_relative_to(root):
            raise RuntimeError(f"evaluation source missing or outside repository: {path}")
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        files[relative] = digest
        manifest.update(relative.encode() + b"\0" + digest.encode("ascii") + b"\n")
    return {"manifest_sha256": manifest.hexdigest(), "files": files}


def _require_sha256(value: Any, label: str) -> str:
    value = str(value or "")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"checkpoint has invalid {label}: {value!r}")
    return value


def validate_checkpoint_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Fail closed on checkpoint/config identity drift before reading formal data."""
    identity = dict(payload.get("run_identity") or {})
    if not identity:
        raise ValueError("checkpoint has no run_identity")
    expected_identity_sha = _require_sha256(
        payload.get("identity_sha256"), "identity_sha256"
    )
    if stable_hash(identity) != expected_identity_sha:
        raise ValueError("checkpoint run_identity does not match identity_sha256")

    config = copy.deepcopy(payload.get("config") or {})
    if not config:
        raise ValueError("checkpoint has no resolved config")
    config["resume"] = None
    expected_config_sha = _require_sha256(
        identity.get("config_sha256"), "run_identity.config_sha256"
    )
    if stable_hash(config) != expected_config_sha:
        raise ValueError("checkpoint config does not match run_identity.config_sha256")
    for identity_name in SHA_FIELDS.values():
        _require_sha256(identity.get(identity_name), f"run_identity.{identity_name}")
    setting = str(identity.get("setting") or "")
    if not setting or setting in {".", ".."} or Path(setting).name != setting:
        raise ValueError(f"unsafe or missing checkpoint setting: {setting!r}")
    return identity


def dataset_environment(
    payload: Mapping[str, Any],
    contract_root: Path,
    cache_root: Path,
    *,
    live_code_sha256: str,
    live_runtime_sha256: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Derive historical formal-loader environment solely from checkpoint identity."""
    identity = validate_checkpoint_identity(payload)
    setting = str(identity["setting"])
    contract_path = contract_root / "data" / f"{setting}.json"
    recipe_root = contract_root / "future-recipes" / setting
    recipe_manifest_path = recipe_root / "manifest.json"
    if not contract_path.is_file() or not recipe_manifest_path.is_file():
        raise FileNotFoundError(
            f"formal contract/recipe missing for {setting}: "
            f"{contract_path}, {recipe_manifest_path}"
        )
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    recipe_manifest = json.loads(recipe_manifest_path.read_text(encoding="utf-8"))
    expected_contract_sha = str(payload["config"].get("campaign_data_contract_sha256", ""))
    if expected_contract_sha and contract.get("contract_sha256") != expected_contract_sha:
        raise ValueError("formal data contract identity differs from checkpoint config")
    expected = {
        "setting_id": setting,
        "data_manifest_sha256": identity["data_manifest_sha256"],
        "calibration_sha256": identity["calibration_sha256"],
        "future_recipe_sha256": identity["future_recipe_sha256"],
        "objective_version": identity["objective_version"],
    }
    for key, value in expected.items():
        if contract.get(key) != value:
            raise ValueError(f"formal data contract {key} differs from checkpoint identity")
    recipe_expected = {
        "source_manifest_sha256": identity["data_manifest_sha256"],
        "calibration_sha256": identity["calibration_sha256"],
        "recipe_sha256": identity["future_recipe_sha256"],
        "code_sha256": identity["code_sha256"],
        "runtime_sha256": identity["runtime_sha256"],
    }
    for key, value in recipe_expected.items():
        if recipe_manifest.get(key) != value:
            raise ValueError(f"formal recipe {key} differs from checkpoint identity")

    # Keep the code executing this rescore distinct from the historical producer of
    # the immutable compact recipe. Mislabeling the old hash as current trainer code
    # would make provenance internally false; the explicit RECIPE_* channel exists for
    # exactly this compatibility case.
    environment = {
        "TREEWM_PROTOCOL_SHA256": _require_sha256(
            identity["protocol_sha256"], "protocol_sha256"
        ),
        "TREEWM_CODE_SHA256": _require_sha256(live_code_sha256, "live_code_sha256"),
        "TREEWM_RUNTIME_SHA256": _require_sha256(
            live_runtime_sha256, "live_runtime_sha256"
        ),
        "TREEWM_DATA_SHA256": _require_sha256(
            identity["data_manifest_sha256"], "data_manifest_sha256"
        ),
        "TREEWM_CALIBRATION_SHA256": _require_sha256(
            identity["calibration_sha256"], "calibration_sha256"
        ),
        "TREEWM_FUTURE_RECIPE_SHA256": _require_sha256(
            identity["future_recipe_sha256"], "future_recipe_sha256"
        ),
        "TREEWM_RECIPE_CODE_SHA256": _require_sha256(
            identity["code_sha256"], "recipe_code_sha256"
        ),
        "TREEWM_RECIPE_RUNTIME_SHA256": _require_sha256(
            identity["runtime_sha256"], "recipe_runtime_sha256"
        ),
    }
    environment.update(
        {
            "TREEWM_CACHE": str(cache_root.resolve()),
            "TREEWM_FUTURE_RECIPE_ROOT": str(recipe_root.resolve()),
        }
    )
    data_contract_sha = str(payload["config"].get("campaign_data_contract_sha256", ""))
    if data_contract_sha:
        environment["TREEWM_DATA_CONTRACT_SHA256"] = _require_sha256(
            data_contract_sha, "campaign_data_contract_sha256"
        )
    return environment, {
        "setting": setting,
        "contract_path": str(contract_path.resolve()),
        "contract_sha256": contract.get("contract_sha256"),
        "recipe_root": str(recipe_root.resolve()),
        "recipe_manifest": str(recipe_manifest_path.resolve()),
        "cache_root": str(cache_root.resolve()),
        "live_trainer_code_sha256": live_code_sha256,
        "live_runtime_sha256": live_runtime_sha256,
        "recipe_code_sha256": identity["code_sha256"],
        "recipe_runtime_sha256": identity["runtime_sha256"],
        **{
            field: identity[field]
            for field in (
                "protocol_sha256",
                "data_manifest_sha256",
                "calibration_sha256",
                "future_recipe_sha256",
            )
        },
        "normalizer_sha256": contract.get("normalizer_sha256"),
    }


@contextlib.contextmanager
def patched_environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update({name: str(value) for name, value in values.items()})
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def rebuild_validation_data(
    cfg,
    payload: Mapping[str, Any],
    *,
    contract_root: Path,
    cache_root: Path,
    live_code_sha256: str,
    live_runtime_sha256: str,
):
    environment, contract = dataset_environment(
        payload,
        contract_root,
        cache_root,
        live_code_sha256=live_code_sha256,
        live_runtime_sha256=live_runtime_sha256,
    )
    future_cfg = cfg_utils.future_set_config(cfg)
    if cfg.env.get("relative_endpoints") is not None:
        future_cfg = replace(
            future_cfg, relative_endpoints=bool(cfg.env.get("relative_endpoints"))
        )
    with patched_environment(environment):
        env, train_ds, val_ds, rebuilt_normalizer = build_datasets(
            str(cfg.env.name),
            future_cfg,
            dataset_dir=cfg.env.dataset_dir,
            xy_dims=tuple(cfg.env.xy_dims),
            max_train_anchors=int(cfg.train.max_train_anchors),
            max_val_anchors=int(cfg.train.max_val_anchors),
            seed=int(cfg.seed),
            cache_future_sets=bool(cfg.future_sets.get("cache", False)),
            shared_cache=bool(cfg.future_sets.get("shared_cache", False)),
            dataset_kind=str(cfg.env.get("dataset_kind", "standard")),
            source_name=str(cfg.env.get("source_name", cfg.env.name)),
            expected_shards=int(cfg.env.get("expected_shards", 1)),
            cache_root=str(cache_root),
            data_manifest_sha256=environment["TREEWM_DATA_SHA256"],
            task_metric_dims=tuple(cfg.env.get("task_metric_dims") or cfg.env.xy_dims),
        )
    # Building both splits is part of the loader's formal identity contract, but only
    # val_ds is iterated below. Keep the train summary for provenance and release it.
    data_summary = {"train": train_ds.summary(), "validation": val_ds.summary()}
    del train_ds
    checkpoint_normalizer = Normalizer.from_state_dict(payload["normalizer"])
    checkpoint_normalizer_sha = normalizer_state_sha256(
        checkpoint_normalizer.state_dict()
    )
    rebuilt_normalizer_sha = normalizer_state_sha256(rebuilt_normalizer.state_dict())
    if checkpoint_normalizer_sha != rebuilt_normalizer_sha:
        raise ValueError("rebuilt dataset normalizer differs from checkpoint normalizer")
    if checkpoint_normalizer_sha != contract.get("normalizer_sha256", checkpoint_normalizer_sha):
        raise ValueError("checkpoint normalizer differs from formal data contract")
    contract["normalizer_sha256"] = checkpoint_normalizer_sha
    return env, val_ds, checkpoint_normalizer, future_cfg, contract, data_summary


def load_model(payload: Mapping[str, Any], cfg, device: torch.device):
    model = build_model(
        str(cfg.arm), cfg_utils.model_config(cfg), k_max=int(cfg.model.flatk_max)
    ).to(device)
    model.gain_head.set_set_aware(bool(cfg.losses.get("gain_set_context", False)))
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    return model


def _normalized_horizon_entropy(metrics: Mapping[str, float], horizons: Sequence[int]) -> float:
    probabilities = np.asarray(
        [
            metrics.get(f"data/validation_horizon_label_fraction_h{int(value)}", 0.0)
            for value in horizons
        ],
        dtype=np.float64,
    )
    nonzero = probabilities[probabilities > 0]
    entropy = float(-(nonzero * np.log(nonzero)).sum()) if len(nonzero) else 0.0
    return entropy / (math.log(len(horizons)) if len(horizons) > 1 else 1.0)


def evaluate_fixed_validation(
    model,
    val_ds,
    cfg,
    future_cfg,
    *,
    device: torch.device,
    val_batches: int,
    num_workers: int,
    step: int,
) -> dict[str, Any]:
    batch_size = int(cfg.train.batch_size)
    loader, sampler = build_fixed_validation_dataloader(
        val_ds,
        batch_size=batch_size,
        num_batches=int(val_batches),
        num_workers=int(num_workers),
        seed=int(cfg.seed),
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(int(cfg.seed) * 1_000_003 + 30_013),
    )
    diagnostic_positions = sampler.local_indices[:batch_size]
    diagnostic_batch_cpu = default_collate(
        [val_ds[int(position)] for position in diagnostic_positions.tolist()]
    )
    loss_cfg = cfg_utils.loss_config(cfg)
    match_cfg = cfg_utils.matching_config(cfg)
    metric_tracker = MetricTracker(device)
    label_tracker = MetricTracker(device)
    examples = 0
    batches = 0
    with fixed_validation_rng(int(cfg.seed), rank=0):
        with torch.no_grad():
            for batch in loader:
                batch = to_device(batch, device)
                _, metrics, _ = compute_branch_losses(
                    model, batch, loss_cfg, match_cfg, step=int(step)
                )
                heldout = {
                    (f"val/{key[6:]}" if key.startswith("train/") else key): value
                    for key, value in metrics.items()
                }
                count = int(batch["obs"].shape[0])
                metric_tracker.add_many(heldout, count=count)
                add_validation_label_metrics(
                    label_tracker,
                    batch,
                    tuple(int(value) for value in future_cfg.horizons),
                    int(future_cfg.max_modes),
                )
                batches += 1
                examples += count
            diagnostic_batch = to_device(diagnostic_batch_cpu, device)
            diagnostic_metrics = {
                **diag.q_vs_z_retrieval(model, diagnostic_batch),
                **diag.branching_diversity_correlation(model, diagnostic_batch),
            }
    metrics = metric_tracker.compute(reduce=False)
    labels = label_tracker.compute(reduce=False)
    labels["data/validation_horizon_label_normalized_entropy"] = (
        _normalized_horizon_entropy(labels, future_cfg.horizons)
    )
    sample = sampler.summary()
    positions = sampler.global_indices.detach().cpu().numpy().astype(np.int64)
    anchor_indices = np.asarray(val_ds.anchors, dtype=np.int64)[positions]
    sample["selected_anchor_index_quantiles"] = {
        name: float(value)
        for name, value in zip(
            ("q00", "q25", "q50", "q75", "q100"),
            np.quantile(anchor_indices, [0.0, 0.25, 0.5, 0.75, 1.0]),
            strict=True,
        )
    }
    sample["diagnostic_dataset_positions"] = diagnostic_positions.tolist()
    sample["diagnostic_anchor_indices"] = (
        diagnostic_batch_cpu["anchor_index"].long().tolist()
    )
    sample["diagnostic_anchor_indices_sha256"] = hashlib.sha256(
        diagnostic_batch_cpu["anchor_index"].long().numpy().astype("<i8").tobytes()
    ).hexdigest()
    return {
        "evaluated_batches": batches,
        "evaluated_examples": examples,
        "sample": sample,
        "branch_loss_components": {
            key: value for key, value in metrics.items() if key.startswith("val/loss_")
        },
        "heldout_metrics": metrics,
        "diagnostics": diagnostic_metrics,
        "label_distribution": labels,
    }


def result_path(
    output_root: Path,
    checkpoint: Path,
    step: int,
    artifact_id: str,
) -> Path:
    run_name = checkpoint.parents[1].name
    safe_name = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in run_name
    ).strip("._") or "checkpoint"
    return (
        output_root / "results" / safe_name
        / f"step{int(step)}-{artifact_id}.json"
    )


def assert_output_outside_runs(output_root: Path, checkpoints: Sequence[Path]) -> None:
    output_root = output_root.resolve()
    for checkpoint in checkpoints:
        run_dir = checkpoint.parents[1].resolve()
        if output_root == run_dir or run_dir in output_root.parents:
            raise ValueError(
                f"output root must be outside checkpoint run directory {run_dir}"
            )


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", default=[])
    parser.add_argument(
        "--checkpoint-glob", action="append", default=[],
        help=f"repeatable glob (wrapper default: {DEFAULT_CHECKPOINT_GLOB})",
    )
    parser.add_argument("--work-index", type=int)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--val-batches", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--contract-root", default=DEFAULT_CONTRACT_ROOT)
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--resume", action="store_true",
        help="verify and retain an identical existing artifact; never overwrite",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.val_batches <= 0 or args.num_workers < 0:
        raise ValueError("val_batches must be positive and num_workers non-negative")
    patterns = tuple(args.checkpoint_glob)
    if not args.checkpoint and not patterns:
        patterns = (DEFAULT_CHECKPOINT_GLOB,)
    checkpoints = discover_checkpoints(args.checkpoint, patterns)
    if not checkpoints:
        raise FileNotFoundError("no checkpoints matched the requested paths/globs")
    work = select_work(checkpoints, args.work_index)
    contract_root = Path(args.contract_root).expanduser().resolve()
    cache_root = Path(args.cache_root).expanduser().resolve()
    output_root = Path(args.output_root).expanduser()
    output_root = (
        output_root if output_root.is_absolute() else REPOSITORY_ROOT / output_root
    ).resolve()
    assert_output_outside_runs(output_root, checkpoints)
    code = evaluation_code_fingerprint()
    print(
        f"[rescore] checkpoints={len(checkpoints)} selected={len(work)} "
        f"code={code['manifest_sha256']}",
        flush=True,
    )
    for index, checkpoint in enumerate(checkpoints):
        marker = "*" if checkpoint in work else " "
        print(f"  {marker} work={index:03d} {checkpoint}")
    if args.dry_run:
        return 0
    if not contract_root.is_dir() or not cache_root.is_dir():
        raise FileNotFoundError(
            f"contract/cache root unavailable: {contract_root}, {cache_root}"
        )

    device = _resolve_device(args.device)
    runtime = runtime_fingerprint()
    live_trainer_code = trainer_code_fingerprint(REPOSITORY_ROOT)
    git = git_provenance(REPOSITORY_ROOT)
    for checkpoint in work:
        checkpoint_before = _stat_identity(checkpoint)
        print(f"[rescore] hashing {checkpoint}", flush=True)
        checkpoint_sha = sha256_file(checkpoint)
        if _stat_identity(checkpoint) != checkpoint_before:
            raise RuntimeError(f"checkpoint changed while hashing: {checkpoint}")
        payload = torch.load(
            str(checkpoint), map_location="cpu", weights_only=False, mmap=True
        )
        identity = validate_checkpoint_identity(payload)
        cfg = OmegaConf.create(payload["config"])
        completed_updates = int(payload.get("completed_updates", payload.get("step", -1)))
        protocol = {
            "schema_version": SCHEMA_VERSION,
            "measurement": "fixed_representative_heldout_branch_rescore_v1",
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_identity_sha256": payload["identity_sha256"],
            "evaluation_code_sha256": code["manifest_sha256"],
            "val_batches": int(args.val_batches),
            "batch_size": int(cfg.train.batch_size),
            "seed": int(cfg.seed),
            "sampler": "fixed_representative_stratified_permutation_v1",
            "primary_q_target": "fut_metric_endpoint",
            "live_trainer_code_sha256": live_trainer_code["manifest_sha256"],
            "live_runtime_sha256": runtime["sha256"],
            "recipe_code_sha256": identity["code_sha256"],
            "recipe_runtime_sha256": identity["runtime_sha256"],
        }
        artifact_id = stable_hash(protocol)
        path = result_path(output_root, checkpoint, completed_updates, artifact_id)
        if path.exists() and args.resume:
            existing = json.loads(path.read_text(encoding="utf-8"))
            if existing.get("artifact_id") != artifact_id:
                raise RuntimeError(f"existing artifact identity differs: {path}")
            print(f"[rescore] exact artifact already complete; retained {path}")
            continue
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing artifact {path}")

        env = None
        try:
            env, val_ds, normalizer, future_cfg, contract, data_summary = (
                rebuild_validation_data(
                    cfg,
                    payload,
                    contract_root=contract_root,
                    cache_root=cache_root,
                    live_code_sha256=live_trainer_code["manifest_sha256"],
                    live_runtime_sha256=runtime["sha256"],
                )
            )
            model = load_model(payload, cfg, device)
            measurement = evaluate_fixed_validation(
                model,
                val_ds,
                cfg,
                future_cfg,
                device=device,
                val_batches=int(args.val_batches),
                num_workers=int(args.num_workers),
                step=completed_updates,
            )
            if _stat_identity(checkpoint) != checkpoint_before:
                raise RuntimeError(f"checkpoint changed during rescore: {checkpoint}")
            checkpoint_info = checkpoint_provenance(
                checkpoint, checkpoint_sha, payload, cfg
            )
            checkpoint_info["normalizer_sha256"] = normalizer_state_sha256(
                normalizer.state_dict()
            )
            result = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "checkpoint_validation_rescore",
                "artifact_id": artifact_id,
                "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "protocol": protocol,
                "checkpoint": checkpoint_info,
                "formal_data_contract": contract,
                "dataset": data_summary,
                "evaluation_code": code,
                "runtime": runtime,
                "git": git,
                "measurement": _json_safe(measurement),
            }
            created = write_or_validate(
                path, result, identity_key="artifact_id", resume=args.resume
            )
            print(f"[rescore] {'wrote' if created else 'retained'} {path}", flush=True)
        finally:
            if env is not None and hasattr(env, "close"):
                env.close()
            if device.type == "cuda":
                torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
