#!/usr/bin/env python3
"""Outcome-blind all-pair causal-parity audit for the frozen Exp23 matrix.

This audit runs before submission.  It composes the exact locked Hydra configs and,
for each setting/seed pair, reconstructs both arms on a fixed published-union batch.
It performs no optimizer step, rollout, checkpoint write, or result read.  The two
arms must match in initial parameter bytes, data/sampler/RNG/fixed-batch identities,
raw executable-prefix targets/artifacts/telemetry, while only the three prescribed
effective weighted values may differ.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
from typing import Any, Mapping


sys.dont_write_bytecode = True
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
FIXED_BATCH_SIZE = 16
VALIDATION_SAMPLE_SEED = 1701
PREFIX_TERMS = (
    "executable_prefix_action",
    "executable_prefix_latent",
    "executable_prefix_endpoint",
)
PREFIX_ARTIFACTS = (
    "raw_action_env",
    "applied_action_env",
    "applied_action_normalized",
    "target_prefix_action_normalized",
    "target_prefix_action_env",
    "predicted_prefix_latent",
    "predicted_prefix_endpoint",
    "target_prefix_endpoint",
    "predicted_prefix_metric_endpoint",
    "target_prefix_metric_endpoint",
    "predicted_vs_actual_guard_metric_error",
    "predicted_guard_metric_displacement",
    "actual_guard_metric_displacement",
    "predicted_normalized_task_displacement_rms",
    "actual_normalized_task_displacement_rms",
    "prefix_length",
    "prefix_action_mask",
    "matched",
)
OPTIONAL_HAMMING_ARTIFACTS = (
    "predicted_vs_actual_hamming",
    "predicted_hamming_displacement",
    "actual_hamming_displacement",
)
ALLOWED_PAIR_ENVIRONMENT_DELTAS = frozenset(
    {
        "TREEWM_CONFIG_SHA256",
        "TREEWM_PROTOCOL_SHA256",
        "TREEWM_RUN_NAME",
        "WANDB_RUN_ID",
    }
)
AUDIT_PROTOCOL_PLACEHOLDER = hashlib.sha256(
    b"exp23-controlled-causal-parity-audit-no-package-protocol-claim"
).hexdigest()
POST_AUDIT_LAUNCH_BINDINGS = frozenset({"TREEWM_CAUSAL_PARITY_SHA256"})
AUDIT_MANIFEST_INPUT_KEYS = (
    "schema_version",
    "campaign_id",
    "method",
    "design",
    "arms",
    "causal_contrast",
    "weight_audit",
    "prefix_target_contract",
    "resolved_config_contract",
    "core_binding",
    "scientific_contract",
    "settings",
    "compatible_v2_recipe_input",
)


class ParityAuditError(RuntimeError):
    pass


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ParityAuditError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_project_module(module: Any, project_root: Path) -> None:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise ParityAuditError(f"module has no concrete source file: {module!r}")
    path = Path(module_file)
    if (
        path.is_symlink()
        or not path.is_file()
        or not path.resolve().is_relative_to(project_root)
    ):
        raise ParityAuditError(f"module is not a regular project-root file: {path}")


def _rng_sha256(torch: Any, np: Any) -> str:
    numpy_state = np.random.get_state()
    digest = hashlib.sha256()
    python_state = canonical_json(random.getstate()).encode("ascii")
    digest.update(len(python_state).to_bytes(8, "little"))
    digest.update(python_state)
    numpy_header = canonical_json(
        {
            "algorithm": numpy_state[0],
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        }
    ).encode("ascii")
    digest.update(len(numpy_header).to_bytes(8, "little"))
    digest.update(numpy_header)
    numpy_keys = np.asarray(numpy_state[1], dtype="<u4").tobytes(order="C")
    digest.update(len(numpy_keys).to_bytes(8, "little"))
    digest.update(numpy_keys)
    torch_state = torch.get_rng_state().cpu().numpy().tobytes(order="C")
    digest.update(len(torch_state).to_bytes(8, "little"))
    digest.update(torch_state)
    return digest.hexdigest()


def _strip_weights(config: Mapping[str, Any]) -> dict[str, Any]:
    value = copy.deepcopy(dict(config))
    for name in PREFIX_TERMS:
        del value["losses"]["weights"][name]
    return value


def _output_tree_fingerprint(path: Path) -> str:
    """Hash run-output metadata without reading or mutating live result bytes."""

    if not path.exists():
        return stable_hash({"path": str(path), "exists": False})
    rows: list[dict[str, Any]] = []
    for candidate in sorted(path.rglob("*"), key=lambda item: str(item)):
        stat = candidate.lstat()
        rows.append(
            {
                "relative": str(candidate.relative_to(path)),
                "mode": int(stat.st_mode),
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
                "symlink_target": os.readlink(candidate) if candidate.is_symlink() else None,
            }
        )
    return stable_hash({"path": str(path), "exists": True, "entries": rows})


def _launch_pair_identity(
    launch: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    argv = list(launch["argv"])
    if len(argv) < 3 or Path(argv[1]).resolve() != (project_root / "scripts/train.py"):
        raise ParityAuditError("controlled launch does not use the exact trainer module")
    if Path(argv[1]).is_symlink():
        raise ParityAuditError("controlled trainer entrypoint is a symlink")
    normalized: list[str] = [str(argv[0]), "scripts/train.py"]
    seen: set[str] = set()
    removed_weights: dict[str, str] = {}
    for argument in argv[2:]:
        if "=" not in argument:
            raise ParityAuditError(f"non-Hydra trainer argument: {argument}")
        key, rendered = argument.split("=", 1)
        normalized_key = key.lstrip("+")
        if normalized_key in seen:
            raise ParityAuditError(f"duplicate controlled launch override: {normalized_key}")
        seen.add(normalized_key)
        if normalized_key in {
            "losses.weights.executable_prefix_action",
            "losses.weights.executable_prefix_latent",
            "losses.weights.executable_prefix_endpoint",
        }:
            removed_weights[normalized_key] = rendered
            continue
        if normalized_key == "hydra.run.dir":
            normalized.append("hydra.run.dir=<cell-run-directory>/hydra")
        else:
            normalized.append(argument)
    if len(removed_weights) != 3:
        raise ParityAuditError("controlled launch does not expose exactly three prefix weights")
    environment = {str(key): str(value) for key, value in launch["environment"].items()}
    stripped_environment = {
        key: value
        for key, value in sorted(environment.items())
        if key not in ALLOWED_PAIR_ENVIRONMENT_DELTAS
        and key not in POST_AUDIT_LAUNCH_BINDINGS
    }
    return {
        "normalized_argv_without_prefix_weights_sha256": stable_hash(normalized),
        "environment_without_allowed_deltas_sha256": stable_hash(stripped_environment),
        "removed_prefix_weights": removed_weights,
        "environment": environment,
    }


def _arm_audit(
    *,
    audit: Any,
    campaign: Any,
    cfg: Any,
    batch: Mapping[str, Any],
    seed: int,
    data_identity_sha256: str,
    sampler_identity_sha256: str,
    fixed_batch_sha256: str,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from scripts.train import validate_executable_prefix_configuration
    from treewm.models.baselines import tree_config_for
    from treewm.utils import config as cfg_utils

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model, loss_cfg = audit.build_model_for_audit(cfg, checkpoint=None, seed=seed)
    match_cfg = cfg_utils.matching_config(cfg)
    tree_cfg = tree_config_for(str(cfg.arm), cfg_utils.tree_config(cfg), model)
    action_dim = int(batch["executable_action_mean"].shape[-1])
    lower = torch.full((action_dim,), float(loss_cfg.executable_action_lower_bound))
    upper = torch.full((action_dim,), float(loss_cfg.executable_action_upper_bound))
    validate_executable_prefix_configuration(
        str(cfg.objective_version),
        loss_cfg,
        cfg_utils.future_set_config(cfg),
        cfg_utils.planner_config(cfg),
        tree_cfg=tree_cfg,
        action_space=type("SealedBox", (), {"low": lower.numpy(), "high": upper.numpy()})(),
        model=model,
    )
    controlled_parameters_sha256 = audit.parameter_mapping_sha256(model)
    controlled_pre_forward_rng_sha256 = _rng_sha256(torch, np)
    model.eval()
    with torch.no_grad():
        _loss, metrics, artifacts, terms = __import__(
            "treewm.losses.total", fromlist=["compute_branch_losses"]
        ).compute_branch_losses(
            model,
            {name: value.to("cpu") for name, value in batch.items()},
            loss_cfg,
            match_cfg,
            step=0,
            return_loss_terms=True,
        )
    target_names = tuple(name for name in PREFIX_ARTIFACTS if name.startswith("target_") or name in {"prefix_length", "prefix_action_mask", "matched"})
    artifact_names = tuple(name for name in PREFIX_ARTIFACTS if name in artifacts)
    if set(PREFIX_ARTIFACTS).difference(artifact_names):
        raise ParityAuditError("prefix artifact schema is incomplete")
    optional_hamming = tuple(
        name for name in OPTIONAL_HAMMING_ARTIFACTS if name in artifacts
    )
    if optional_hamming and optional_hamming != OPTIONAL_HAMMING_ARTIFACTS:
        raise ParityAuditError("optional Hamming artifact schema is incomplete")
    artifact_names = (*artifact_names, *optional_hamming)
    if any(
        not torch.is_tensor(artifacts[name])
        or not bool(torch.isfinite(artifacts[name].detach().float()).all())
        for name in artifact_names
    ):
        raise ParityAuditError("prefix artifact schema contains a nonfinite/nontensor value")
    raw_telemetry = {
        key: float(value)
        for key, value in sorted(metrics.items())
        if key.startswith("train/executable_prefix/")
        or key in {f"train/loss_{name}" for name in PREFIX_TERMS}
        or key in {f"train/loss_raw/{name}" for name in PREFIX_TERMS}
    }
    if not raw_telemetry or not all(np.isfinite(value) for value in raw_telemetry.values()):
        raise ParityAuditError("raw prefix telemetry is empty/nonfinite")
    effective = {
        name: float(terms.effective[name].detach().item()) for name in PREFIX_TERMS
    }
    raw = {name: float(terms.raw[name].detach().item()) for name in PREFIX_TERMS}
    weights = {name: float(terms.weights[name]) for name in PREFIX_TERMS}
    if any(
        not np.isfinite(raw[name])
        or raw[name] <= 0.0
        or not np.isfinite(effective[name])
        or not np.isfinite(weights[name])
        or abs(effective[name] - raw[name] * weights[name])
        > 1e-6 * max(1.0, abs(effective[name]))
        for name in PREFIX_TERMS
    ):
        raise ParityAuditError("raw/effective prefix terms are invalid")
    return {
        "resolved_config_sha256": stable_hash(
            __import__("omegaconf", fromlist=["OmegaConf"]).OmegaConf.to_container(cfg, resolve=True)
        ),
        "resolved_config_without_prefix_weights_sha256": stable_hash(
            _strip_weights(
                __import__("omegaconf", fromlist=["OmegaConf"]).OmegaConf.to_container(cfg, resolve=True)
            )
        ),
        "controlled_cpu_scratch_parameters_sha256": controlled_parameters_sha256,
        "data_identity_sha256": data_identity_sha256,
        "sampler_identity_sha256": sampler_identity_sha256,
        "controlled_cpu_pre_forward_rng_sha256": controlled_pre_forward_rng_sha256,
        "fixed_validation_batch_sha256": fixed_batch_sha256,
        "raw_prefix_targets_sha256": audit.tensor_mapping_sha256(
            {name: artifacts[name] for name in target_names}
        ),
        "raw_prefix_artifacts_sha256": audit.tensor_mapping_sha256(
            {name: artifacts[name] for name in artifact_names}
        ),
        "raw_prefix_telemetry": raw_telemetry,
        "raw_prefix_telemetry_sha256": stable_hash(raw_telemetry),
        "raw_prefix_values": raw,
        "effective_prefix_weights": weights,
        "effective_prefix_values": effective,
        "controlled_cpu_parameters_unchanged": audit.parameter_mapping_sha256(model)
        == controlled_parameters_sha256,
    }


def run(project_root: Path) -> dict[str, Any]:
    # Weight-audit import is first and pins NumPy/torch reduction threads before import.
    if project_root != PROJECT_ROOT or project_root.is_symlink():
        raise ParityAuditError("audit must use the exact nonsymlink package project root")
    audit = _load("exp23_weight_helpers_for_parity", PACKAGE_DIR / "weight_audit.py")
    campaign = _load("exp23_campaign_for_parity", PACKAGE_DIR / "campaign.py")
    for module in (audit, campaign):
        module_path = Path(module.__file__)
        if module_path.is_symlink() or not module_path.resolve().is_relative_to(project_root):
            raise ParityAuditError("audit helper module escapes the exact project root")

    import numpy as np
    import torch
    import scripts.train as trainer_module
    import treewm as treewm_module
    import treewm.data.ogbench_dataset as dataset_module
    import treewm.data.samplers as samplers_module
    import treewm.losses.total as total_loss_module
    import treewm.models.baselines as model_module
    import treewm.utils.config as config_module
    from omegaconf import OmegaConf
    from torch.utils.data import default_collate
    from treewm.data.samplers import FixedRepresentativeSampler

    for module in (
        trainer_module,
        treewm_module,
        dataset_module,
        samplers_module,
        total_loss_module,
        model_module,
        config_module,
    ):
        _assert_project_module(module, project_root)

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        if torch.get_num_interop_threads() != 1:
            raise
    torch.use_deterministic_algorithms(True)
    manifest = campaign.read_json(PACKAGE_DIR / "manifest.json")
    weight_lock = campaign.read_json(PACKAGE_DIR / "weight_audit.lock.json")
    # This program regenerates its own lock.  Validate every upstream contract,
    # but do not require a superseded causal-parity output to match this source.
    campaign.validate_manifest(
        manifest,
        weight_lock,
        project_root,
        verify_causal_parity_lock=False,
    )
    config_lock = campaign.read_json(PACKAGE_DIR / "resolved_config.lock.json")
    source = campaign.source_contract(project_root)
    cells = campaign.expand_matrix(manifest)
    output_root = Path(manifest["paths"]["run_root"])
    output_before = _output_tree_fingerprint(output_root)
    expected_gsep = {
        "executable_prefix_action": float(
            manifest["arms"][1]["executable_prefix_weights"]["action"]
        ),
        "executable_prefix_latent": float(
            manifest["arms"][1]["executable_prefix_weights"]["latent"]
        ),
        "executable_prefix_endpoint": float(
            manifest["arms"][1]["executable_prefix_weights"]["endpoint"]
        ),
    }
    equality_names = (
        "launch_without_allowed_deltas_sha256",
        "resolved_config_without_prefix_weights_sha256",
        "controlled_cpu_scratch_parameters_sha256",
        "data_identity_sha256",
        "sampler_identity_sha256",
        "controlled_cpu_pre_forward_rng_sha256",
        "fixed_validation_batch_sha256",
        "raw_prefix_targets_sha256",
        "raw_prefix_artifacts_sha256",
        "raw_prefix_telemetry_sha256",
        "raw_prefix_values",
    )
    if len(equality_names) != len(set(equality_names)):
        raise ParityAuditError("parity field set contains duplicates")

    rows: list[dict[str, Any]] = []
    for setting in campaign.SETTINGS:
        for seed in campaign.SEEDS:
            seed_cells = [
                cell
                for cell in cells
                if cell.setting == setting and cell.seed == seed
            ]
            pair: dict[str, Any] = {}
            launch_inputs: dict[str, dict[str, Any]] = {}
            fixed_positions_by_arm: dict[str, list[int]] = {}
            for arm in campaign.ARMS:
                cell = next(value for value in seed_cells if value.arm == arm)
                lock_row = config_lock["matrix"][cell.index]
                if (
                    lock_row["index"] != cell.index
                    or lock_row["setting_id"] != setting
                    or lock_row["arm_id"] != arm
                    or lock_row["seed"] != seed
                ):
                    raise ParityAuditError(f"cell{cell.index}: resolved-lock identity differs")
                launch = campaign.trainer_command(
                    manifest,
                    weight_lock,
                    cell,
                    repo_root=project_root,
                    package_protocol_sha256=AUDIT_PROTOCOL_PLACEHOLDER,
                )
                if launch["hashes"]["config_override_sha256"] != lock_row["config_override_sha256"]:
                    raise ParityAuditError(f"cell{cell.index}: launch/config lock differs")
                launch_input = _launch_pair_identity(launch, project_root=project_root)
                launch_inputs[arm] = launch_input
                environment = {
                    **launch_input["environment"],
                    "WANDB_MODE": "disabled",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONNOUSERSITE": "1",
                }
                with audit.patched_environment(environment):
                    cfg = OmegaConf.create(lock_row["resolved_config"])
                    resolved = OmegaConf.to_container(cfg, resolve=True)
                    if stable_hash(resolved) != lock_row["resolved_config_sha256"]:
                        raise ParityAuditError(f"cell{cell.index}: resolved config bytes differ")
                    train_ds, val_ds, _normalizer, _domain, data_identity = (
                        audit.load_read_only_datasets(cfg, launch)
                    )
                    val_sampler = FixedRepresentativeSampler(
                        val_ds,
                        batch_size=int(cfg.train.batch_size),
                        num_batches=int(cfg.train.val_batches),
                        seed=VALIDATION_SAMPLE_SEED,
                    )
                    fixed_positions = [
                        int(value)
                        for value in val_sampler.local_indices[:FIXED_BATCH_SIZE].tolist()
                    ]
                    fixed_positions_by_arm[arm] = fixed_positions
                    fixed_batch = default_collate(
                        [val_ds[position] for position in fixed_positions]
                    )
                    sampler_identity_sha256 = stable_hash(
                        {
                            "train": {
                                "class": "DistributedSampler",
                                "dataset_size": len(train_ds),
                                "seed": seed,
                                "shuffle": True,
                                "drop_last": True,
                                "epoch": 0,
                            },
                            "fixed_validation": val_sampler.summary(),
                            "controlled_fixed_validation_positions": fixed_positions,
                        }
                    )
                    arm_row = _arm_audit(
                        audit=audit,
                        campaign=campaign,
                        cfg=cfg,
                        batch=fixed_batch,
                        seed=seed,
                        data_identity_sha256=stable_hash(data_identity),
                        sampler_identity_sha256=sampler_identity_sha256,
                        fixed_batch_sha256=audit.batch_sha256(fixed_batch),
                    )
                    arm_row["launch_without_allowed_deltas_sha256"] = stable_hash(
                        {
                            "argv": launch_input[
                                "normalized_argv_without_prefix_weights_sha256"
                            ],
                            "environment": launch_input[
                                "environment_without_allowed_deltas_sha256"
                            ],
                        }
                    )
                    arm_row["controlled_launch_config_override_sha256"] = launch[
                        "hashes"
                    ]["config_override_sha256"]
                    pair[arm] = arm_row
                    del train_ds, val_ds, fixed_batch

            environment_differences = {
                name
                for name in set(launch_inputs["GS"]["environment"])
                | set(launch_inputs["GSEP"]["environment"])
                if launch_inputs["GS"]["environment"].get(name)
                != launch_inputs["GSEP"]["environment"].get(name)
            }
            if environment_differences != ALLOWED_PAIR_ENVIRONMENT_DELTAS:
                raise ParityAuditError(
                    f"{setting}/seed{seed}: unexpected launch environment deltas: "
                    f"{sorted(environment_differences)}"
                )
            differing = [
                name for name in equality_names if pair["GS"][name] != pair["GSEP"][name]
            ]
            if differing:
                raise ParityAuditError(
                    f"{setting}/seed{seed}: causal parity differs: {differing}"
                )
            if fixed_positions_by_arm["GS"] != fixed_positions_by_arm["GSEP"]:
                raise ParityAuditError(f"{setting}/seed{seed}: fixed samples differ")
            if any(
                not pair[arm]["controlled_cpu_parameters_unchanged"]
                for arm in campaign.ARMS
            ):
                raise ParityAuditError(f"{setting}/seed{seed}: audit mutated parameters")
            if any(
                pair["GS"]["effective_prefix_weights"][name] != 0.0
                or pair["GS"]["effective_prefix_values"][name] != 0.0
                for name in PREFIX_TERMS
            ):
                raise ParityAuditError(f"{setting}/seed{seed}: GS is not monitor-only")
            if pair["GSEP"]["effective_prefix_weights"] != expected_gsep:
                raise ParityAuditError(f"{setting}/seed{seed}: GSEP weights differ")
            rows.append(
                {
                    "setting_id": setting,
                    "seed": seed,
                    "controlled_fixed_validation_positions": fixed_positions_by_arm["GS"],
                    "allowed_environment_differences": sorted(environment_differences),
                    "arms": pair,
                    "parity_fields": list(equality_names),
                }
            )

    if len(rows) != 10:
        raise ParityAuditError("causal parity matrix is incomplete")
    output_after = _output_tree_fingerprint(output_root)
    if output_after != output_before:
        raise ParityAuditError("causal audit changed the Exp23 live-output tree")
    audit_manifest_input = {
        key: manifest[key] for key in AUDIT_MANIFEST_INPUT_KEYS
    }
    result = {
        "schema_version": 1,
        "status": "frozen_outcome_blind_causal_parity",
        "audit_id": "treewm_exp23_causal_parity_audit_v1",
        "classification": "controlled_cpu_scratch_fixed_validation_reconstruction_no_optimizer_no_eval_no_rollout_no_outcome_no_live_run_mutation",
        "fixed_validation_batch_size": FIXED_BATCH_SIZE,
        "pairs": rows,
        "source_sha256": file_sha256(Path(__file__)),
        "audit_manifest_input_sha256": stable_hash(audit_manifest_input),
        "package_protocol_claimed": False,
        "trainer_code_fingerprint": manifest["core_binding"]["trainer_code_fingerprint"],
        "runtime_sha256": source["runtime_sha256"],
        "resolved_config_artifact_sha256": manifest["resolved_config_contract"]["artifact_sha256"],
        "weight_audit_artifact_sha256": manifest["weight_audit"]["artifact_sha256"],
        "prefix_target_artifact_sha256": manifest["prefix_target_contract"]["artifact_sha256"],
        "live_output_fingerprint_before": output_before,
        "live_output_fingerprint_after": output_after,
        "determinism": {
            "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
            "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS"),
            "OPENBLAS_NUM_THREADS": os.environ.get("OPENBLAS_NUM_THREADS"),
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
    }
    result["artifact_sha256"] = stable_hash(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    args = parser.parse_args()
    try:
        result = run(args.project_root.resolve())
    except Exception as exc:
        print(f"causal parity audit failed: {exc}", file=sys.stderr)
        return 1
    print("EXP23_CAUSAL_PARITY_AUDIT=" + canonical_json(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
