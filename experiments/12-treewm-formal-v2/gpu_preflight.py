#!/usr/bin/env python3
"""Topology and fingerprinted real-data TreeWM-v2 objective preflight."""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import socket
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_WORKERS = 16
MAX_SHARED_GRADIENT_SHARE = 0.80
AUDIT_STEP = 5000
AUDIT_BATCHES = 3
FORMAL_ACTIVE_TERMS = frozenset(
    {
        "state", "action", "horizon", "bind", "coverage", "redundancy",
        "keep", "uncertainty", "recursive", "reconstruction", "control", "expand",
    }
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def representative_dataset_positions(
    dataset_size: int, *, batches: int = AUDIT_BATCHES,
    batch_size: int = 16, seed: int = 0,
) -> list[list[int]]:
    """Return deterministic, non-prefix audit samples split into fixed batches."""
    import numpy as np

    sample_count = int(batches) * int(batch_size)
    if dataset_size < sample_count:
        raise ValueError(
            f"representative audit needs {sample_count} examples, got {dataset_size}"
        )
    chosen = np.random.default_rng(seed).choice(
        int(dataset_size), sample_count, replace=False
    )
    return [
        sorted(int(value) for value in chosen[start:start + batch_size])
        for start in range(0, sample_count, batch_size)
    ]


def nested_state_equal(left: Any, right: Any) -> bool:
    """Exact comparison for checkpoint state containing tensors and containers."""
    try:
        import torch
    except ImportError:
        torch = None
    if torch is not None and torch.is_tensor(left) and torch.is_tensor(right):
        return bool(torch.equal(left.detach().cpu(), right.detach().cpu()))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            nested_state_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            nested_state_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    try:
        fd = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _load_contract(repo_root: Path) -> dict[str, Any]:
    sys.path.insert(0, str(repo_root))
    from treewm.utils.provenance import runtime_fingerprint, trainer_code_fingerprint

    return {"code": trainer_code_fingerprint(repo_root), "runtime": runtime_fingerprint()}


def cache_key(protocol_lock: Path, repo_root: Path, cache_root: Path | None = None) -> str:
    campaign_dir = Path(__file__).resolve().parent
    files = [
        Path(__file__).resolve(), campaign_dir / "dispatcher.py",
        campaign_dir / "train.slurm", campaign_dir / "pilot.slurm",
        campaign_dir / "validate_pilot.py", protocol_lock.resolve(),
    ]
    manifest = json.loads((campaign_dir / "manifest.json").read_text(encoding="utf-8"))
    contract_root = Path(manifest["paths"]["contract_root"])
    contract_files = sorted(contract_root.rglob("*.json")) if contract_root.is_dir() else []
    expected_data_hashes: dict[str, str] = {}
    expected_calibration_hashes: dict[str, str] = {}
    expected_recipe_hashes: dict[str, str] = {}
    data_contracts: dict[str, dict[str, Any]] = {}
    try:
        for setting in manifest["settings"]:
            contract = json.loads(
                (contract_root / "data" / f"{setting['id']}.json").read_text(
                    encoding="utf-8"
                )
            )
            setting_id = str(setting["id"])
            data_contracts[setting_id] = contract
            expected_data_hashes[setting_id] = str(contract["data_manifest_sha256"])
            expected_calibration_hashes[setting_id] = str(
                contract["calibration_sha256"]
            )
            expected_recipe_hashes[setting_id] = str(contract["future_recipe_sha256"])
    except (FileNotFoundError, OSError, KeyError, json.JSONDecodeError) as exc:
        raise RuntimeError("formal heavy preflight requires all v2 data contracts") from exc
    pilot_acceptance = Path(manifest["paths"]["pilot_run_root"]) / "state" / "PILOT_ACCEPTED.json"
    try:
        accepted = json.loads(pilot_acceptance.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("formal heavy preflight requires PILOT_ACCEPTED.json") from exc
    gradient_hashes = accepted.get("gradient_audit_sha256_by_setting")
    sys.path.insert(0, str(campaign_dir))
    from campaign import run_protocol_sha256

    expected_pilot_protocols = {
        setting_id: run_protocol_sha256(manifest, contract, namespace="pilot")
        for setting_id, contract in data_contracts.items()
    }
    live_contract = _load_contract(repo_root.resolve())
    locked_protocol = protocol_lock.read_text(encoding="utf-8").strip()
    try:
        accepted_maximum_share = float(accepted.get("max_shared_module_gradient_share"))
    except (TypeError, ValueError) as exc:
        raise RuntimeError("pilot acceptance has an invalid gradient-share bound") from exc
    if (
        accepted.get("schema_version") != 2
        or accepted.get("status") != "passed"
        or accepted.get("settings") != [str(setting["id"]) for setting in manifest["settings"]]
        or accepted.get("campaign_protocol_sha256") != locked_protocol
        or accepted.get("pilot_protocol_sha256_by_setting") != expected_pilot_protocols
        or accepted.get("code_sha256") != live_contract["code"]["manifest_sha256"]
        or accepted.get("runtime_sha256") != live_contract["runtime"]["sha256"]
        or not math.isfinite(accepted_maximum_share)
        or not 0.0 <= accepted_maximum_share <= MAX_SHARED_GRADIENT_SHARE
        or accepted.get("data_manifest_sha256_by_setting") != expected_data_hashes
        or accepted.get("calibration_sha256_by_setting") != expected_calibration_hashes
        or accepted.get("future_recipe_sha256_by_setting") != expected_recipe_hashes
        or not isinstance(gradient_hashes, dict)
        or set(gradient_hashes) != set(expected_recipe_hashes)
        or any(
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
            for value in gradient_hashes.values()
        )
    ):
        raise RuntimeError("pilot acceptance does not satisfy the v2 gradient gate")
    contract_files.append(pilot_acceptance)
    payload = {
        "contract": live_contract,
        "cache_root": str(cache_root.resolve()) if cache_root is not None else None,
        "files": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [*files, *contract_files]
        },
    }
    return canonical_hash(payload)


def _quick_torch_smoke() -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected one visible CUDA device, available={torch.cuda.is_available()} "
            f"count={torch.cuda.device_count()}"
        )
    device = torch.device("cuda:0")
    value = (torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4) @ torch.eye(4, device=device)).sum()
    torch.cuda.synchronize(device)
    if float(value.item()) != 120.0:
        raise RuntimeError("CUDA arithmetic probe returned the wrong result")
    return {
        "torch_cuda_device_count": 1,
        "torch_cuda_device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "preflight_level": "quick",
    }


@contextlib.contextmanager
def temporary_environment(values: Mapping[str, str]):
    old = {key: os.environ.get(key) for key in values}
    os.environ.update({str(key): str(value) for key, value in values.items()})
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _compose_audit_config(command: list[str], repo_root: Path):
    from hydra import compose, initialize_config_dir

    overrides = [value for value in command[2:] if "=" in value]
    replacements = {
        "train.batch_size": "16",
        "train.num_workers": "0",
        "train.gain_batch_size": "4",
        "train.steps": "5000",
        "train.scheduler_total_steps": "1000000",
        "eval.final_episodes_per_task": "1",
        "resume": "null",
    }
    for name, value in replacements.items():
        prefix = f"{name}="
        matches = [index for index, token in enumerate(overrides) if token.startswith(prefix)]
        if len(matches) != 1:
            raise RuntimeError(f"audit command has {len(matches)} {name} overrides")
        overrides[matches[0]] = f"{name}={value}"
    with initialize_config_dir(version_base=None, config_dir=str(repo_root / "configs")):
        return compose(config_name="base", overrides=overrides)


def _configured_model(cfg, device):
    from treewm.models.baselines import build_model
    from treewm.utils import config as cfg_utils

    model_cfg = cfg_utils.model_config(cfg)
    model = build_model("treewm", model_cfg, k_max=int(cfg.model.flatk_max)).to(device).train()
    model.set_gradient_checkpointing(True)
    loss_cfg = cfg_utils.loss_config(cfg)
    model.gain_head.set_set_aware(bool(loss_cfg.gain_set_context))
    if str(loss_cfg.control_objective) != "bootstrap":
        model.tree_signature.requires_grad_(False)
    if loss_cfg.gain_set_context and float(loss_cfg.gain_branch_prior_weight) == 0.0:
        model.heads.gain_head.requires_grad_(False)
    if not loss_cfg.on("mass"):
        model.heads.mass_head.requires_grad_(False)
    return model, model_cfg, loss_cfg


def real_v2_gradient_audit(
    *, setting_index: int, output_dir: Path, manifest_path: Path,
    repo_root: Path, data_root: Path, cache_root: Path,
    post_training_checkpoint: bool = True,
) -> dict[str, Any]:
    """Restore the 5k pilot and audit every loss on three fixed real train batches."""

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    sys.path.insert(0, str(repo_root))
    import numpy as np
    import torch
    from omegaconf import open_dict

    from campaign import (
        expand_runs,
        load_data_contract,
        load_manifest,
        run_directory,
        trainer_command,
    )
    from scripts.train import (
        TrainingStepModule,
        formal_v2_objective_contract,
        gradient_parameter_groups,
    )
    from treewm.data.ogbench_dataset import build_datasets
    from treewm.data.samplers import to_device
    from treewm.evaluation.coverage import StateQuantizer
    from treewm.losses.total import audit_effective_loss_gradients
    from treewm.models.baselines import tree_config_for
    from treewm.utils import config as cfg_utils
    from treewm.utils.checkpoint import load_checkpoint, save_checkpoint
    from treewm.utils.rng import RngStreams, make_generator
    from torch.utils.data import default_collate

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("real v2 audit requires exactly one visible GPU")
    manifest = load_manifest(manifest_path)
    if not 0 <= setting_index < len(manifest["settings"]):
        raise RuntimeError("audit setting index is out of range")
    setting = manifest["settings"][setting_index]
    candidates = [
        run for run in expand_runs(manifest)
        if run.setting_id == setting["id"] and int(run.seed) == 0
    ]
    if len(candidates) != 1:
        raise RuntimeError("audit cannot resolve the seed-zero pilot run")
    run = candidates[0]
    pilot_root = Path(manifest["paths"]["pilot_run_root"])
    if not post_training_checkpoint:
        raise RuntimeError("formal v2 gradient audits must use the post-training checkpoint")
    data_contract = load_data_contract(
        manifest, setting, data_root=data_root, cache_root=cache_root,
        verify_recipe_files=True,
    )
    command, environment = trainer_command(
        manifest, run, python_executable=sys.executable, repo_root=repo_root,
        run_root=pilot_root, data_root=data_root, cache_root=cache_root,
        wandb_project=manifest["logging"]["pilot_wandb_project"], wandb_mode="online",
    )
    command = [str(value) for value in command]
    cfg = _compose_audit_config(command, repo_root)
    device = torch.device("cuda:0")
    with temporary_environment(environment):
        future_cfg = cfg_utils.future_set_config(cfg)
        if cfg.env.get("relative_endpoints") is not None:
            from dataclasses import replace
            future_cfg = replace(future_cfg, relative_endpoints=bool(cfg.env.relative_endpoints))
        env, train_ds, _val_ds, normalizer = build_datasets(
            cfg.env.name, future_cfg, dataset_dir=cfg.env.dataset_dir,
            xy_dims=tuple(cfg.env.xy_dims), max_train_anchors=int(cfg.train.max_train_anchors),
            # Dataset construction attaches and verifies both immutable recipes. Use the
            # preregistered validation selection here; independently resampling only 16
            # anchors selects a different set that the sealed recipe never claimed to
            # cover and made every historical pilot audit fail before gradients ran.
            max_val_anchors=int(cfg.train.max_val_anchors), seed=0,
            cache_future_sets=False, shared_cache=True,
            dataset_kind=str(cfg.env.dataset_kind), source_name=str(cfg.env.source_name),
            expected_shards=int(cfg.env.get("expected_shards", 1)), cache_root=str(cache_root),
            data_manifest_sha256=environment["TREEWM_DATA_SHA256"],
            task_metric_dims=tuple(cfg.env.task_metric_dims),
        )
        try:
            env.reset(seed=0)
            frame = env.render()
            if frame is None or np.asarray(frame).size == 0:
                raise RuntimeError("OGBench EGL render returned no pixels")
        finally:
            env.close()
        if cfg.env.obs_dim is None or cfg.env.action_dim is None:
            with open_dict(cfg):
                cfg.env.obs_dim = int(train_ds.obs_dim)
                cfg.env.action_dim = int(train_ds.act_dim)
        audit_batch_size = 16
        if len(train_ds) < AUDIT_BATCHES * audit_batch_size:
            raise RuntimeError("real-data audit requires at least 48 train anchors")
        sampled_positions = representative_dataset_positions(
            len(train_ds), batches=AUDIT_BATCHES,
            batch_size=audit_batch_size, seed=0,
        )
        fixed_batches: list[dict[str, Any]] = []
        for batch_index, positions in enumerate(sampled_positions):
            anchor_indices = [int(train_ds.anchors[index]) for index in positions]
            selection_identity = {
                "dataset_positions": positions,
                "anchor_indices": anchor_indices,
            }
            fixed_batches.append(
                {
                    "batch_index": batch_index,
                    "dataset_positions": positions,
                    "anchor_indices": anchor_indices,
                    "batch_sha256": canonical_hash(selection_identity),
                    "batch": to_device(
                        default_collate([train_ds[index] for index in positions]), device
                    ),
                }
            )
        model, model_cfg, loss_cfg = _configured_model(cfg, device)
        base_tree_cfg = cfg_utils.tree_config(cfg)
        tree_cfg = tree_config_for("treewm", base_tree_cfg, model)
        from dataclasses import replace
        gain_tree_cfg = tree_config_for(
            "treewm", replace(base_tree_cfg, node_budget=int(cfg.train.gain_tree_budget)), model
        )
        match_cfg = cfg_utils.matching_config(cfg)
        contract = formal_v2_objective_contract(
            model, loss_cfg, match_cfg, future_cfg, tree_cfg,
            separate_gain_clip=bool(cfg.train.separate_gain_grad_clip),
        )
        failed = [name for name, passed in contract.items() if not passed]
        if failed:
            raise RuntimeError(f"v2 objective contract failed: {failed}")
        rng = RngStreams(seed=0, device=device)
        model._horizon_gen = make_generator(0, "train", device)
        step_module = TrainingStepModule(
            model, loss_cfg, match_cfg, gain_tree_cfg, None,
            StateQuantizer(resolution=float(cfg.retrieval.grid_resolution), dims=tuple(cfg.env.xy_dims)),
            cfg.train, cfg.model, cfg.losses,
        )
        world_parameters, gain_parameters = gradient_parameter_groups(
            model, include_branch_prior=float(loss_cfg.gain_branch_prior_weight) > 0
        )
        optimizer = torch.optim.AdamW(
            [{"params": world_parameters}, {"params": gain_parameters}],
            lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay),
        )
        pilot_checkpoint = run_directory(pilot_root, run) / "checkpoints" / "latest.pt"
        if not pilot_checkpoint.is_file():
            raise RuntimeError(f"post-training pilot checkpoint is missing: {pilot_checkpoint}")
        pilot_checkpoint_sha256 = hashlib.sha256(pilot_checkpoint.read_bytes()).hexdigest()
        pilot_payload = load_checkpoint(
            pilot_checkpoint, model, optimizer, map_location="cuda:0", restore_rng=False
        )
        pilot_identity = pilot_payload.get("run_identity") or {}
        if (
            int(pilot_payload.get("completed_updates", -1)) != AUDIT_STEP
            or int(pilot_payload.get("step", -1)) != AUDIT_STEP
            or pilot_identity.get("objective_version") != "treewm_v2_rms_rank_v1"
            or int(pilot_identity.get("total_steps", -1)) != AUDIT_STEP
            or int(pilot_identity.get("scheduler_total_steps", -1)) != 1_000_000
            or pilot_identity.get("protocol_sha256") != environment["TREEWM_PROTOCOL_SHA256"]
            or pilot_identity.get("code_sha256") != environment["TREEWM_CODE_SHA256"]
            or pilot_identity.get("runtime_sha256") != environment["TREEWM_RUNTIME_SHA256"]
            or pilot_identity.get("wandb_id") != environment["WANDB_RUN_ID"]
            or pilot_identity.get("wandb_project") != environment["WANDB_PROJECT"]
            or pilot_identity.get("wandb_group") != environment["WANDB_RUN_GROUP"]
            or pilot_identity.get("data_manifest_sha256") != environment["TREEWM_DATA_SHA256"]
            or pilot_identity.get("calibration_sha256") != environment["TREEWM_CALIBRATION_SHA256"]
            or pilot_identity.get("future_recipe_sha256")
            != environment["TREEWM_FUTURE_RECIPE_SHA256"]
            or data_contract["data_manifest_sha256"] != environment["TREEWM_DATA_SHA256"]
            or data_contract["calibration_sha256"]
            != environment["TREEWM_CALIBRATION_SHA256"]
            or data_contract["future_recipe_sha256"]
            != environment["TREEWM_FUTURE_RECIPE_SHA256"]
        ):
            raise RuntimeError("post-training checkpoint identity does not match the v2 pilot")
        rank_states = pilot_payload.get("rank_states") or []
        if len(rank_states) != 1 or not isinstance(rank_states[0].get("loader"), Mapping) or not isinstance(rank_states[0].get("rng_streams"), Mapping):
            raise RuntimeError("pilot checkpoint lacks exact loader/RNG resume state")
        modules = {
            "encoder": model.encoder,
            "branch_transformer": model.branch_transformer,
            "dynamics": model.dynamics,
            "controllability": model.controllability,
            "contextual_gain": model.gain_head,
        }
        shared_modules = ("encoder", "branch_transformer", "dynamics", "controllability")
        batch_audits: list[dict[str, Any]] = []
        active_term_union: set[str] = set()
        for fixed in fixed_batches:
            # Dropout/frontier randomness is deterministic per fixed batch as well as
            # the data itself; hashes and seeds are persisted in the artifact.
            batch_seed = 50_000 + int(fixed["batch_index"])
            torch.manual_seed(batch_seed)
            torch.cuda.manual_seed_all(batch_seed)
            rng.reset("planner")
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                batch_loss, batch_metrics, _artifacts, terms = step_module(
                    fixed["batch"], AUDIT_STEP, rng.planner, return_loss_terms=True
                )
            if not torch.isfinite(batch_loss):
                raise RuntimeError("active-gain v2 audit objective is non-finite")
            audit = audit_effective_loss_gradients(
                terms, modules, max_terms=32,
                max_gradient_share=MAX_SHARED_GRADIENT_SHARE,
                fail_on_excess=False, preserve_graph=False,
            )
            if set(audit.active_terms) != FORMAL_ACTIVE_TERMS:
                raise RuntimeError(
                    "audit active terms differ from the exact formal v2 objective: "
                    f"{sorted(audit.active_terms)}"
                )
            for module_name in modules:
                norm = float(audit.metrics[f"gradient_audit/objective_norm/{module_name}"])
                if not math.isfinite(norm) or norm <= 0:
                    raise RuntimeError(
                        f"batch {fixed['batch_index']} {module_name} gradient is not positive finite"
                    )
            for module_name in shared_modules:
                shares = [
                    float(value) for key, value in audit.metrics.items()
                    if key.startswith(f"gradient_audit/share/{module_name}/")
                ]
                maximum = max(shares, default=1.0)
                if not shares or maximum > MAX_SHARED_GRADIENT_SHARE:
                    raise RuntimeError(
                        f"batch {fixed['batch_index']} {module_name} gradient dominance "
                        f"{maximum:.6f} exceeds {MAX_SHARED_GRADIENT_SHARE:.2f}"
                    )
            active_term_union.update(audit.active_terms)
            batch_audits.append(
                {
                    "batch_index": fixed["batch_index"],
                    "dataset_positions": fixed["dataset_positions"],
                    "anchor_indices": fixed["anchor_indices"],
                    "batch_sha256": fixed["batch_sha256"],
                    "torch_seed": batch_seed,
                    "active_terms": list(audit.active_terms),
                    "loss_total": float(batch_loss.detach().item()),
                    "gain_active": float(batch_metrics.get("train/gain_active", 0.0)),
                    "metrics": audit.metrics,
                }
            )

        # Build a fresh fourth traversal for a conventional rematerialized backward;
        # the audit helper deliberately released each bounded diagnostic graph.
        optimizer.zero_grad(set_to_none=True)
        backward_seed = 60_000
        torch.manual_seed(backward_seed)
        torch.cuda.manual_seed_all(backward_seed)
        rng.reset("planner")
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss, metrics, _artifacts = step_module(
                fixed_batches[0]["batch"], AUDIT_STEP, rng.planner
            )
        if not torch.isfinite(loss):
            raise RuntimeError("rematerialized v2 backward objective is non-finite")
        loss.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        if not gradients or not all(bool(torch.isfinite(gradient).all()) for gradient in gradients):
            raise RuntimeError("v2 rematerialized backward produced non-finite gradients")
        optimizer.step()
        torch.cuda.synchronize(device)

        identity = {
            "kind": "treewm-v2-real-data-preflight", "setting_id": setting["id"],
            "objective_version": "treewm_v2_rms_rank_v1",
            "data_manifest_sha256": environment["TREEWM_DATA_SHA256"],
            "calibration_sha256": environment["TREEWM_CALIBRATION_SHA256"],
            "future_recipe_sha256": environment["TREEWM_FUTURE_RECIPE_SHA256"],
        }
        checkpoint_dir = output_dir / "checkpoint-resume" / setting["id"]
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = checkpoint_dir / "latest.pt"
        save_checkpoint(
            checkpoint, model=model, optimizer=optimizer, step=1,
            extra={"run_identity": identity, "completed_updates": 1,
                   "loader_state": rank_states[0]["loader"],
                   "rng_streams": rank_states[0]["rng_streams"]},
        )
        restored, _, _ = _configured_model(cfg, device)
        restored_world, restored_gain = gradient_parameter_groups(
            restored, include_branch_prior=float(loss_cfg.gain_branch_prior_weight) > 0
        )
        restored_optimizer = torch.optim.AdamW(
            [{"params": restored_world}, {"params": restored_gain}],
            lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay),
        )
        payload = load_checkpoint(
            checkpoint, restored, restored_optimizer, map_location="cuda:0",
            restore_rng=False, expected_identity=identity,
        )
        if (
            int(payload.get("completed_updates", -1)) != 1
            or not nested_state_equal(payload.get("loader_state"), rank_states[0]["loader"])
            or not nested_state_equal(payload.get("rng_streams"), rank_states[0]["rng_streams"])
            or not restored_optimizer.state
        ):
            raise RuntimeError("exact model/optimizer/loader/RNG checkpoint resume failed")

    if active_term_union != FORMAL_ACTIVE_TERMS:
        raise RuntimeError(
            "audit artifact union differs from the exact formal v2 objective: "
            f"{sorted(active_term_union)}"
        )

    audit_payload: dict[str, Any] = {
        "schema_version": 2, "status": "passed", "setting_id": setting["id"],
        "setting_index": setting_index, "seed": 0, "audit_step": AUDIT_STEP,
        "scheduler_total_steps": 1_000_000,
        "objective_version": "treewm_v2_rms_rank_v1",
        "active_terms": sorted(active_term_union),
        "batch_audits": batch_audits,
        "objective_contract": contract,
        "loss_total": float(loss.detach().item()),
        "protocol_sha256": environment["TREEWM_PROTOCOL_SHA256"],
        "code_sha256": environment["TREEWM_CODE_SHA256"],
        "runtime_sha256": environment["TREEWM_RUNTIME_SHA256"],
        "data_manifest_sha256": environment["TREEWM_DATA_SHA256"],
        "calibration_sha256": environment["TREEWM_CALIBRATION_SHA256"],
        "future_recipe_sha256": environment["TREEWM_FUTURE_RECIPE_SHA256"],
        "gradient_checkpointing": True, "real_data_batch": True,
        "gain_active": float(metrics.get("train/gain_active", 0.0)),
        "checkpoint_exact_resume": True, "ogbench_egl_render": True,
        "checkpoint_completed_updates": AUDIT_STEP,
        "pilot_checkpoint_sha256": pilot_checkpoint_sha256,
        "fixed_train_batch_count": len(batch_audits),
        "dataset_size": len(train_ds),
        "dataset_selection_seed": 0,
        "dataset_selection_sha256": canonical_hash(
            [batch["dataset_positions"] for batch in batch_audits]
        ),
        "backward_torch_seed": backward_seed,
        "generated_unix_time": time.time(),
    }
    audit_payload["artifact_sha256"] = canonical_hash(audit_payload)
    output_path = output_dir / f"{setting['id']}.json"
    atomic_json(output_path, audit_payload)
    atomic_json(
        output_dir / f"setting-{setting_index}.passed",
        {"schema_version": 2, "status": "passed", "setting_id": setting["id"],
         "artifact_sha256": audit_payload["artifact_sha256"]},
    )
    return audit_payload


def _wandb_auth_readonly() -> bool:
    import wandb

    return bool(wandb.Api(timeout=30).viewer)


def probe(
    output_dir: Path, repo_root: Path, *, manifest: Path, data_root: Path,
    cache_root: Path, quick: bool = False, skip_gpu: bool = False,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    rank_text = os.environ.get("SLURM_PROCID")
    if rank_text is None or not rank_text.isdigit():
        print("SLURM_PROCID is missing or invalid", file=sys.stderr)
        return 2
    rank = int(rank_text)
    visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
    if len(visible) != 1:
        print(f"rank {rank}: expected one visible GPU, got {visible}", file=sys.stderr)
        return 2
    payload: dict[str, Any] = {
        "schema_version": 2, "status": "ok", "rank": rank,
        "local_rank": int(os.environ.get("SLURM_LOCALID", "-1")),
        "job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
        "cuda_visible_devices": visible, "unix_time": time.time(),
    }
    try:
        if skip_gpu:
            payload.update({"torch_cuda_device_count": 1, "torch_cuda_device": "test-only", "torch_version": "test-only", "preflight_level": "quick" if quick else "full"})
            if not quick:
                payload.update({"treewm_v2_real_data_update": True, "objective_version": "treewm_v2_rms_rank_v1", "gradient_checkpointing": True, "active_gain_gradient_audit": True, "gradient_audit_batches": AUDIT_BATCHES, "gradient_share_bound": MAX_SHARED_GRADIENT_SHARE, "checkpoint_exact_resume": True, "ogbench_egl_render": True, "wandb_auth_readonly": True if rank == 0 else None})
        elif quick:
            payload.update(_quick_torch_smoke())
        else:
            payload.update(_quick_torch_smoke())
            audit = real_v2_gradient_audit(
                setting_index=rank % 10, output_dir=output_dir / f"audit-rank-{rank}",
                manifest_path=manifest, repo_root=repo_root, data_root=data_root,
                cache_root=cache_root,
            )
            payload.update({
                "preflight_level": "full", "treewm_v2_real_data_update": True,
                "objective_version": audit["objective_version"],
                "gradient_checkpointing": audit["gradient_checkpointing"],
                "active_gain_gradient_audit": audit["gain_active"] == 1.0,
                "gradient_audit_batches": audit["fixed_train_batch_count"],
                "gradient_share_bound": MAX_SHARED_GRADIENT_SHARE,
                "checkpoint_exact_resume": audit["checkpoint_exact_resume"],
                "ogbench_egl_render": audit["ogbench_egl_render"],
                "gradient_audit_sha256": audit["artifact_sha256"],
                "wandb_auth_readonly": _wandb_auth_readonly() if rank == 0 else None,
            })
    except Exception as exc:
        print(f"rank {rank}: TreeWM-v2 GPU preflight failed: {exc}", file=sys.stderr)
        return 2
    atomic_json(output_dir / f"rank.{rank}.json", payload)
    return 0


def verify(output_dir: Path, workers: int, *, level: str) -> int:
    failures: list[str] = []
    payloads: list[dict[str, Any]] = []
    for rank in range(workers):
        try:
            payload = json.loads((output_dir / f"rank.{rank}.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, json.JSONDecodeError) as exc:
            failures.append(f"rank {rank}: {exc}")
            continue
        if payload.get("status") != "ok" or payload.get("rank") != rank:
            failures.append(f"rank {rank}: invalid payload")
        if len(payload.get("cuda_visible_devices", [])) != 1 or payload.get("torch_cuda_device_count") != 1:
            failures.append(f"rank {rank}: not bound to one GPU")
        if payload.get("preflight_level") != level:
            failures.append(f"rank {rank}: wrong preflight level")
        if level == "full":
            expected = {
                "treewm_v2_real_data_update": True,
                "objective_version": "treewm_v2_rms_rank_v1",
                "gradient_checkpointing": True,
                "active_gain_gradient_audit": True,
                "gradient_audit_batches": AUDIT_BATCHES,
                "gradient_share_bound": MAX_SHARED_GRADIENT_SHARE,
                "checkpoint_exact_resume": True,
                "ogbench_egl_render": True,
            }
            for key, value in expected.items():
                if payload.get(key) != value:
                    failures.append(f"rank {rank}: {key} != {value!r}")
            if not isinstance(payload.get("gradient_audit_sha256"), str) or len(payload["gradient_audit_sha256"]) != 64:
                failures.append(f"rank {rank}: missing gradient audit hash")
            if rank == 0 and payload.get("wandb_auth_readonly") is not True:
                failures.append("rank 0: W&B authentication failed")
        payloads.append(payload)
    hosts: dict[str, int] = {}
    local: dict[str, set[int]] = {}
    for payload in payloads:
        host = str(payload.get("hostname"))
        hosts[host] = hosts.get(host, 0) + 1
        local.setdefault(host, set()).add(int(payload.get("local_rank", -1)))
    if sorted(hosts.values()) != [8, 8]:
        failures.append(f"expected two hosts with eight ranks, got {hosts}")
    for host, ranks in local.items():
        if ranks != set(range(8)):
            failures.append(f"host {host}: local ranks are {sorted(ranks)}")
    if failures:
        print("GPU preflight verification failed:\n  " + "\n  ".join(failures), file=sys.stderr)
        return 2
    return 0


def validate_success_sentinel(path: Path, cache_key_value: str, workers: int) -> bool:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    return bool(
        payload.get("schema_version") == 2
        and payload.get("status") == "full_gpu_preflight_complete"
        and payload.get("cache_key") == cache_key_value
        and payload.get("workers") == workers == EXPECTED_WORKERS
        and payload.get("objective_version") == "treewm_v2_rms_rank_v1"
        and payload.get("gradient_audit_batches") == AUDIT_BATCHES
        and payload.get("gradient_share_bound") == MAX_SHARED_GRADIENT_SHARE
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--repo-root", type=Path, default=_repo_root())
    parser.add_argument("--manifest", type=Path, default=here / "manifest.json")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("TREEWM_DATA_ROOT", here / "data")))
    parser.add_argument("--cache-root", type=Path, default=Path(os.environ.get("TREEWM_CACHE", here / "cache")))
    parser.add_argument("--workers", type=int, default=EXPECTED_WORKERS)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-level", choices=("quick", "full"), default="full")
    parser.add_argument("--print-cache-key", action="store_true")
    parser.add_argument("--print-contract", action="store_true")
    parser.add_argument("--protocol-lock", type=Path, default=here / "protocol.sha256")
    parser.add_argument("--success-sentinel", type=Path)
    parser.add_argument("--check-success-sentinel", action="store_true")
    parser.add_argument("--cache-key-value")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--audit-setting-index", type=int)
    parser.add_argument("--post-training-checkpoint", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    if args.print_cache_key:
        print(cache_key(args.protocol_lock, repo_root, args.cache_root))
        return 0
    if args.print_contract:
        print(json.dumps(_load_contract(repo_root), sort_keys=True))
        return 0
    if args.check_success_sentinel:
        if args.success_sentinel is None or not args.cache_key_value:
            return 2
        return 0 if validate_success_sentinel(args.success_sentinel, args.cache_key_value, args.workers) else 2
    if args.output_dir is None:
        return 2
    output_dir = args.output_dir.resolve()
    if args.audit_setting_index is not None:
        if not args.post_training_checkpoint:
            print("gradient audit requires --post-training-checkpoint", file=sys.stderr)
            return 2
        try:
            real_v2_gradient_audit(
                setting_index=args.audit_setting_index, output_dir=output_dir,
                manifest_path=args.manifest.resolve(), repo_root=repo_root,
                data_root=args.data_root.resolve(), cache_root=args.cache_root.resolve(),
                post_training_checkpoint=True,
            )
            return 0
        except Exception as exc:
            print(f"TreeWM-v2 gradient audit failed: {exc}", file=sys.stderr)
            return 2
    if args.verify:
        status = verify(output_dir, args.workers, level=args.verify_level)
        if status == 0 and args.success_sentinel is not None:
            if not args.cache_key_value or len(args.cache_key_value) != 64:
                return 2
            atomic_json(
                args.success_sentinel.resolve(),
                {"schema_version": 2, "status": "full_gpu_preflight_complete",
                 "cache_key": args.cache_key_value, "workers": args.workers,
                 "objective_version": "treewm_v2_rms_rank_v1",
                 "gradient_audit_batches": AUDIT_BATCHES,
                 "gradient_share_bound": MAX_SHARED_GRADIENT_SHARE,
                 "unix_time": time.time()},
            )
        return status
    return probe(
        output_dir, repo_root, manifest=args.manifest.resolve(),
        data_root=args.data_root.resolve(), cache_root=args.cache_root.resolve(),
        quick=args.quick, skip_gpu=args.skip_gpu,
    )


if __name__ == "__main__":
    raise SystemExit(main())
