from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
from typing import Any, Callable, Mapping
import zipfile

import pytest

import calibration_contract as contract
import calibration_controller as controller
import calibration_seal as seal
import calibration_worker as worker


PACKAGE_DIR = Path(__file__).resolve().parents[1]


def _malicious_checkpoint_side_effect(path: str) -> None:
    Path(path).write_text("REDUCER EXECUTED", encoding="utf-8")


def _malicious_checkpoint_unlock(descriptor: int) -> None:
    fcntl.flock(descriptor, fcntl.LOCK_UN)


class _MaliciousCheckpointReducer:
    def __init__(self, path: Path) -> None:
        self.path = str(path)

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return (_malicious_checkpoint_side_effect, (self.path,))


class _InheritedUnlockReducer:
    def __init__(self, descriptor: int) -> None:
        self.descriptor = descriptor

    def __reduce__(self) -> tuple[object, tuple[int]]:
        return (_malicious_checkpoint_unlock, (self.descriptor,))


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


_RUNTIME_LOCKS: dict[str, dict[str, Any]] = {}
_MODEL_AUTHORITIES: dict[str, dict[str, Any]] = {}
_CREATION_RECEIPTS: dict[str, dict[str, Any]] = {}
_TERMINAL_CENSUSES: dict[str, dict[str, Any]] = {}


def _authority_key(authority: Mapping[str, Any]) -> str:
    return authority["environment"]["scratch_root"]


def _runtime_lock(authority: Mapping[str, Any]) -> dict[str, Any]:
    return _RUNTIME_LOCKS[_authority_key(authority)]


def _model_authority(authority: Mapping[str, Any]) -> dict[str, Any]:
    return _MODEL_AUTHORITIES[_authority_key(authority)]


def _terminal_census(authority: Mapping[str, Any]) -> dict[str, Any]:
    return _TERMINAL_CENSUSES[_authority_key(authority)]


def _synthetic_treewm_parameter_schema() -> dict[str, Any]:
    rows = [
        ("decoder.head.bias", [3]),
        ("decoder.head.weight", [3, 128, 1, 1]),
        ("dynamics.action_proj.bias", [128]),
        ("dynamics.action_proj.weight", [128, 6]),
        ("dynamics.transition.bias", [128]),
        ("dynamics.transition.weight", [128, 128]),
        ("encoder.block0.norm.bias", [32]),
        ("encoder.block0.norm.weight", [32]),
        ("encoder.stem.bias", [32]),
        ("encoder.stem.weight", [32, 3, 3, 3]),
        ("scorer.mlp.0.bias", [64]),
        ("scorer.mlp.0.weight", [64, 128]),
        ("scorer.mlp.2.bias", [1]),
        ("scorer.mlp.2.weight", [1, 64]),
    ]
    parameters = []
    for name, shape in rows:
        numel = 1
        for size in shape:
            numel *= size
        parameters.append({
            "name": name,
            "shape": shape,
            "dtype": "torch.float32",
            "numel": numel,
            "storage_bytes": numel * 4,
            "device": "cpu",
            "layout": "strided",
            "storage_alias_policy": "unique_exact_storage",
        })
    schema: dict[str, Any] = {
        "schema_version": 1,
        "model_class": "TreeWM",
        "parameters": parameters,
        "parameter_count": len(parameters),
        "total_numel": sum(row["numel"] for row in parameters),
        "total_storage_bytes": sum(row["storage_bytes"] for row in parameters),
    }
    schema["schema_sha256"] = contract.stable_hash(schema)
    return schema


def _model_hook_source() -> bytes:
    """Synthetic reviewed-hook fixture; it proves mechanics, not production readiness."""
    schema_literal = repr(_synthetic_treewm_parameter_schema())
    source = f'''# synthetic production-shaped Exp24 hook fixture only
s = __import__("sys")
schema = {schema_literal}
def quote(value):
    return '"' + value.replace('\\\\', '\\\\\\\\').replace('"', '\\\\"') + '"'
def encode(value):
    if value is None:
        return "null"
    if type(value) is int:
        return str(value)
    if type(value) is str:
        return quote(value)
    if type(value) is list:
        return "[" + ",".join(encode(item) for item in value) + "]"
    if type(value) is dict:
        return "{{" + ",".join(quote(key) + ":" + encode(value[key]) for key in sorted(value)) + "}}"
    raise TypeError(type(value).__name__)
request_sha = [arg.split("=", 1)[1] for arg in s.argv if arg.startswith("--request-sha256=")][0]
settings = []
for arg in s.argv:
    if arg.startswith("--setting-contract="):
        setting, env_config, seed244, seed245 = arg.split("=", 1)[1].split(":")
        settings.append({{
            "setting_id": setting,
            "env_config": env_config,
            "resolved_config_contract_sha256_by_seed": {{"244": seed244, "245": seed245}},
            "parameter_schema": schema,
        }})
output = {{
    "schema_version": 1,
    "interface": "{contract.MODEL_AUTHORITY_HOOK_INTERFACE}",
    "request_sha256": request_sha,
    "settings": settings,
}}
print(encode(output))
'''
    return source.encode("utf-8")


def _build_runtime_lock(root: Path) -> dict[str, Any]:
    runtime = root / "sealed-runtime"
    (runtime / "bin").mkdir(parents=True)
    (runtime / "stdlib").mkdir()
    (runtime / "site-packages").mkdir()
    (runtime / "lib").mkdir()
    (runtime / "libexec").mkdir()
    python = runtime / "bin/python"
    # The fixture hook is deliberately import-free.  A small system interpreter
    # keeps the many independent retained-tree scans affordable while still
    # exercising a real ELF interpreter capability.
    fixture_python = Path("/usr/bin/python3").resolve()
    if not fixture_python.is_file():
        fixture_python = Path(sys.executable).resolve()
    shutil.copyfile(fixture_python, python)
    decoder_source = runtime / "libexec/calibration_contract.py"
    shutil.copyfile(contract.__file__, decoder_source)
    (runtime / "pyvenv.cfg").write_text("exp24 synthetic runtime fixture\n", encoding="utf-8")
    (runtime / "stdlib/BOUND.txt").write_text("stdlib fixture\n", encoding="utf-8")
    (runtime / "site-packages/treewm_native.so").write_bytes(b"synthetic-native-extension")
    (runtime / "lib/libtreewm.so").write_bytes(b"synthetic-shared-library")
    for path in runtime.rglob("*"):
        if path.is_file():
            path.chmod(0o555 if path == python else 0o444)
    for directory in sorted(
        [path for path in runtime.rglob("*") if path.is_dir()],
        key=lambda path: len(path.parts), reverse=True,
    ):
        directory.chmod(0o555)
    runtime.chmod(0o555)
    with contract.RetainedTree(
        runtime, "synthetic runtime fixture", directory_mode=0o555,
        file_mode=(0o444, 0o555), lock_exclusive=False,
    ) as tree:
        inventory = tree.inventory
        root_identity = contract.directory_identity(tree.root.before)

    def file_row(relative: str, mode: int) -> dict[str, Any]:
        return {"relative_path": relative, "sha256": inventory[relative]["sha256"],
                "mode": mode}

    def root_row(relative: str) -> dict[str, Any]:
        subtree = {
            key: item for key, item in inventory.items()
            if key == relative or key.startswith(relative + "/")
        }
        return {"relative_path": relative, "subtree_sha256": contract.stable_hash(subtree)}

    serialization = {
        "torch_load_weights_only": True,
        "safe_globals": list(contract.SAFE_CHECKPOINT_ALLOWED_GLOBALS),
        "decoder_isolation": "clean_exec_close_fds_hard_rlimit_canonical_json_only",
        "storage_alias_policy": "unique_exact_storage",
        "max_checkpoint_bytes": contract.SAFE_CHECKPOINT_MAX_BYTES,
        "max_archive_entries": contract.SAFE_CHECKPOINT_MAX_ARCHIVE_ENTRIES,
        "max_archive_uncompressed_bytes": (
            contract.SAFE_CHECKPOINT_MAX_ARCHIVE_UNCOMPRESSED_BYTES
        ),
        "max_archive_depth": contract.SAFE_CHECKPOINT_MAX_ARCHIVE_DEPTH,
        "max_central_directory_bytes": (
            contract.SAFE_CHECKPOINT_MAX_CENTRAL_DIRECTORY_BYTES
        ),
        "max_graph_nodes": contract.SAFE_CHECKPOINT_MAX_GRAPH_NODES,
        "max_graph_depth": contract.SAFE_CHECKPOINT_MAX_GRAPH_DEPTH,
        "max_tensors": contract.SAFE_CHECKPOINT_MAX_TENSORS,
        "max_tensor_numel": contract.SAFE_CHECKPOINT_MAX_TENSOR_NUMEL,
        "max_tensor_bytes": contract.SAFE_CHECKPOINT_MAX_TENSOR_BYTES,
        "max_report_bytes": contract.SAFE_CHECKPOINT_MAX_REPORT_BYTES,
        "rlimit_address_space": contract.SAFE_CHECKPOINT_MAX_ADDRESS_SPACE,
        "rlimit_cpu_seconds": contract.SAFE_CHECKPOINT_MAX_CPU_SECONDS,
        "rlimit_nofile": contract.SAFE_CHECKPOINT_MAX_OPEN_FILES,
        "wall_clock_seconds": contract.SAFE_CHECKPOINT_MAX_WALL_SECONDS,
    }
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": contract.RUNTIME_LOCK_STATUS,
        "campaign_id": contract.CAMPAIGN_ID,
        "manifest_file_sha256": contract.file_sha256(contract.MANIFEST_PATH),
        "runtime_root": str(runtime),
        "runtime_root_identity": root_identity,
        "runtime_inventory": inventory,
        "runtime_inventory_sha256": contract.stable_hash(inventory),
        "interpreter": file_row("bin/python", 0o555),
        "decoder_source": file_row("libexec/calibration_contract.py", 0o444),
        "pyvenv_cfg": file_row("pyvenv.cfg", 0o444),
        "stdlib_roots": [root_row("stdlib")],
        "site_package_roots": [root_row("lib"), root_row("site-packages")],
        "native_extensions": [file_row("site-packages/treewm_native.so", 0o444)],
        "shared_libraries": [file_row("lib/libtreewm.so", 0o444)],
        "sys_path": ["stdlib", "site-packages"],
        "loader_paths": ["lib"],
        "symlink_policy": "forbid_all_components_and_entries",
        "serialization_policy": serialization,
        "execution_profile": contract.RUNTIME_SYNTHETIC_PROFILE,
        "production_ready": False,
        "closure_attestation": {
            "schema_version": 1,
            "status": contract.RUNTIME_SYNTHETIC_CLOSURE_STATUS,
            "probe_source_sha256": None,
            "probe_stdout_sha256": None,
            "sys_executable": None,
            "sys_prefix": None,
            "sys_path": [],
            "imported_module_files": [],
            "elf_interpreter": None,
            "loaded_shared_libraries": [],
            "outside_runtime_paths": [],
        },
    }
    content_body = {
        "schema_version": 1,
        "runtime_root_identity": root_identity,
        "runtime_inventory_sha256": value["runtime_inventory_sha256"],
        "interpreter": value["interpreter"],
        "decoder_source": value["decoder_source"],
        "pyvenv_cfg": value["pyvenv_cfg"],
        "stdlib_roots": value["stdlib_roots"],
        "site_package_roots": value["site_package_roots"],
        "native_extensions": value["native_extensions"],
        "shared_libraries": value["shared_libraries"],
        "sys_path": value["sys_path"],
        "loader_paths": value["loader_paths"],
        "symlink_policy": value["symlink_policy"],
        "serialization_policy": serialization,
        "execution_profile": value["execution_profile"],
        "production_ready": value["production_ready"],
        "closure_attestation": value["closure_attestation"],
    }
    value["runtime_content_sha256"] = contract.stable_hash(content_body)
    value["runtime_lock_sha256"] = contract.stable_hash(value)
    contract.validate_runtime_lock(value, contract.load_manifest())
    return value


def _write_json(path: Path, value: object, mode: int = 0o644) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    path.chmod(mode)


def _safe_load_path(path: Path) -> tuple[Mapping[str, Any], dict[str, Any]]:
    with contract.DirectoryCapability(path.parent.absolute(), "safe-loader fixture") as root:
        descriptor, before = root.open_regular(path.name, "safe-loader fixture checkpoint")
        try:
            return contract.safe_load_checkpoint_fd(
                descriptor,
                before,
                "safe-loader fixture checkpoint",
                verify=lambda: root.require_named_identity(
                    path.name, before, "safe-loader fixture checkpoint"
                ),
            )
        finally:
            os.close(descriptor)


def _paths(authority: dict[str, Any]) -> tuple[Path, Path]:
    environment = authority["environment"]
    scratch = Path(environment["scratch_root"])
    return (
        scratch / environment["snapshot_relative_path"],
        scratch / environment["result_relative_path"],
    )


def _config(
    authority: dict[str, Any],
    cell: contract.CalibrationCell | dict[str, Any],
    result_root: Path,
) -> dict[str, Any]:
    setting_id = cell.setting_id if isinstance(cell, contract.CalibrationCell) else cell["setting_id"]
    seed = cell.seed if isinstance(cell, contract.CalibrationCell) else cell["seed"]
    setting = contract.setting_authority(authority, setting_id)
    return {
        "seed": seed,
        "arm": "treewm",
        "objective_version": contract.OBJECTIVE,
        "run_root": str(result_root / "live-runs"),
        "run_name": None,
        "resume": "auto",
        "campaign_id": contract.CAMPAIGN_ID,
        "campaign_source_sha256": authority["roots"]["source_sha256"],
        "campaign_protocol_sha256": authority["roots"]["protocol_sha256"],
        "campaign_config_sha256": authority["roots"]["config_sha256"],
        "campaign_input_contract_sha256": setting["input_contract_sha256"],
        "campaign_factorial_arm": "zero_prefix_calibration",
        "campaign_prerequisite_binding_sha256": authority.get(
            "authority_sha256", "0" * 64
        ),
        "campaign_selected_recipe_sha256": setting["future_recipe_sha256"],
        "train": {
            "steps": 25_000,
            "scheduler_total_steps": 1_000_000,
            "ckpt_every": 5_000,
            "val_every": 25_000,
            "diag_every": 5_000,
            "log_every": 50,
            "eval_every": 25_000,
            "viz_every": 25_000,
            "viz_every_early": 25_000,
            "viz_early_until": 0,
            "max_train_anchors": setting["published_union_train_anchors"],
            "max_val_anchors": setting["published_union_validation_anchors"],
            "validation_sample_seed": 1701,
            "gradient_checkpointing": True,
            "fixture_full_config_leaf": "authenticated",
        },
        "losses": {
            "enabled": {
                "executable_prefix_action": True,
                "executable_prefix_latent": True,
                "executable_prefix_endpoint": True,
            },
            "weights": {
                "executable_prefix_action": 0.0,
                "executable_prefix_latent": 0.0,
                "executable_prefix_endpoint": 0.0,
            },
            "executable_action_lower_bound": setting["action_lower_bound"],
            "executable_action_upper_bound": setting["action_upper_bound"],
        },
        "future_sets": {
            "executable_prefix_steps": 4,
            "recipe_anchor_policy": "published_union",
            "cache": False,
            "shared_cache": True,
        },
        "planner": {
            "action_lower_bound": setting["action_lower_bound"],
            "action_upper_bound": setting["action_upper_bound"],
            "max_env_steps": setting["max_environment_steps"],
            "execute_steps": 4,
            "execute_mode": "clipped",
            "decoded_metric": "domain_raw",
            "require_first_edge_improvement": True,
        },
        "eval": {
            "task_split": "standard",
            "episodes_per_task": 1,
            "final_episodes_per_task": 1,
        },
        "env": {
            "short_name": setting_id,
            "name": setting["env_name"],
            "source_name": setting["source_name"],
            "dataset_kind": setting["dataset_kind"],
        },
        "model": {"branch_factor": 4, "max_depth": 3},
        "tree": {"node_budget": 64, "max_depth": 3},
        "retrieval": {"enabled": False, "num_keys": 0},
    }


def _authority(root: Path) -> dict[str, Any]:
    root = root.absolute()
    runtime_lock = _build_runtime_lock(root)
    scratch = root / "authorized-scratch"
    formal = root / "formal-output"
    scratch.mkdir(parents=True)
    formal.mkdir(parents=True)
    scratch.chmod(0o700)
    formal.chmod(0o700)
    roots = {name: _digest(name) for name in contract.ROOT_NAMES}
    roots["runtime_sha256"] = runtime_lock["runtime_content_sha256"]
    roots["config_sha256"] = "0" * 64
    snapshot_relative = f"snapshots/{roots['source_sha256']}"
    result_relative = f"results/{contract.CAMPAIGN_ID}"
    snapshot = scratch / snapshot_relative
    result = scratch / result_relative
    (snapshot / "scripts").mkdir(parents=True)
    trainer = snapshot / "scripts/train.py"
    trainer.write_bytes(b"import treewm_import_probe\n# pinned synthetic trainer\n")
    imported_module = snapshot / "treewm_import_probe.py"
    imported_module.write_bytes(b"AUTHENTICATED_IMPORT = True\n")
    model_hook = snapshot / contract.MODEL_AUTHORITY_HOOK_PATH
    model_hook.write_bytes(_model_hook_source())
    trainer.chmod(0o444)
    imported_module.chmod(0o444)
    model_hook.chmod(0o444)
    (snapshot / "scripts").chmod(0o555)
    snapshot.chmod(0o555)
    python = Path(runtime_lock["runtime_root"]) / runtime_lock["interpreter"][
        "relative_path"
    ]
    environment = {
        "python": str(python),
        "python_sha256": contract.file_sha256(python),
        "data_root": str(root / "data"),
        "cache_root": str(root / "cache"),
        "future_recipe_root": str(root / "future"),
        "scratch_root": str(scratch),
        "scratch_root_identity": contract.nofollow_directory_identity(
            scratch, "fixture scratch"
        ),
        "formal_output_root": str(formal),
        "formal_output_root_identity": contract.nofollow_directory_identity(
            formal, "fixture formal"
        ),
        "snapshot_relative_path": snapshot_relative,
        "result_relative_path": result_relative,
        "trainer_sha256": contract.file_sha256(trainer),
        "WANDB_MODE": "disabled",
        "MUJOCO_GL": "egl",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    roots["environment_sha256"] = contract.stable_hash(environment)
    census: dict[str, Any] = {
        "schema_version": 1,
        "status": "sealed_exact_zero_prior_assignment_collision_census",
        "campaign_id": contract.CAMPAIGN_ID,
        "seeds": [244, 245],
        "scope": contract.SEED_CENSUS_SCOPE,
        "repository_head": "a" * 40,
        "git_ref_inventory_sha256": _digest("git-ref-inventory"),
        "worktree_inventory_sha256": _digest("worktree-inventory"),
        "reachable_history_inventory_sha256": _digest("reachable-history"),
        "prior_assignment_inventory_sha256": _digest("prior-assignments"),
        "prior_assignment_matches": 0,
        "performed_before_first_calibration_run": True,
        "scope_sha256": contract.seed_census_scope_sha256(),
    }
    census["evidence_sha256"] = contract.seed_census_evidence_sha256(census)
    census["census_sha256"] = contract.stable_hash(census)
    settings: list[dict[str, Any]] = []
    for index, (setting_id, env_config) in enumerate(contract.SETTINGS):
        metadata = contract.SETTING_METADATA[setting_id]
        settings.append({
            "setting_id": setting_id,
            "env_config": env_config,
            **metadata,
            "input_contract_sha256": _digest(f"input-{index}"),
            "data_manifest_sha256": _digest(f"data-{index}"),
            "calibration_sha256": _digest(f"future-calibration-{index}"),
            "future_recipe_sha256": _digest(f"future-{index}"),
            "recipe_code_sha256": _digest(f"recipe-code-{index}"),
            "recipe_runtime_sha256": _digest(f"recipe-runtime-{index}"),
            "published_union_train_anchors": 10_000 + index,
            "published_union_validation_anchors": 2_000 + index,
            "action_lower_bound": -1.0,
            "action_upper_bound": 1.0,
            "max_environment_steps": 500 + index,
            "resolved_config_contract_sha256_by_seed": {
                "244": "0" * 64,
                "245": "0" * 64,
            },
        })
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "sealed_zero_prefix_calibration_authorities",
        "campaign_id": contract.CAMPAIGN_ID,
        "manifest_file_sha256": contract.file_sha256(contract.MANIFEST_PATH),
        "roots": roots,
        "seed_collision_census": census,
        "environment": environment,
        "settings": settings,
        "authority_sha256": "0" * 64,
    }
    for cell in contract.expand_cells(contract.load_manifest()):
        config_contract = worker._resolved_config_contract_sha256(
            _config(value, cell, result)
        )
        settings[cell.setting_index]["resolved_config_contract_sha256_by_seed"][
            str(cell.seed)
        ] = config_contract
    value["roots"]["config_sha256"] = contract.logical_config_sha256(
        contract.load_manifest(), value
    )
    value.pop("authority_sha256")
    value["authority_sha256"] = contract.stable_hash(value)
    contract.validate_authority(value, contract.load_manifest(), runtime_lock)
    model_authority = contract.build_model_state_authority(
        contract.load_manifest(), value, runtime_lock, snapshot
    )
    creation_receipt = controller.create_result_root_exclusive(
        contract.load_manifest(), value, runtime_lock, model_authority
    )
    key = _authority_key(value)
    _RUNTIME_LOCKS[key] = runtime_lock
    _MODEL_AUTHORITIES[key] = model_authority
    _CREATION_RECEIPTS[key] = creation_receipt
    return value


def _launch(authority: dict[str, Any], index: int = 0) -> dict[str, Any]:
    snapshot, result = _paths(authority)
    return controller.build_launch(
        contract.load_manifest(), authority, _runtime_lock(authority),
        _model_authority(authority), index, snapshot, result
    )


def _identity_config_sha256(config: dict[str, Any]) -> str:
    value = copy.deepcopy(config)
    value["resume"] = None
    return contract.stable_hash(value)


def _checkpoint_payload(
    authority: dict[str, Any], launch: dict[str, Any], torch: Any
) -> dict[str, Any]:
    config = _config(authority, launch["cell"], Path(launch["result_root"]))
    identity = worker._expected_run_identity(
        authority,
        launch["cell"],
        launch["result_root"],
        _identity_config_sha256(config),
    )
    return {
        "step": 5_000,
        "completed_updates": 5_000,
        "next_step": 5_000,
        "reason": "awaiting-external-stage-gate",
        "phase": "train",
        "pending_eval_step": None,
        "final_eval": None,
        "post_update_cadence": {
            "schema_version": 1,
            "committed_update": 5_000,
            "completed_update": 5_000,
            "replay_action": None,
        },
        "scheduler": {"last_epoch": 5_000},
        "config": config,
        "run_identity": identity,
        "identity_sha256": contract.stable_hash(identity),
        "evaluation_seed_tables_sha256": identity[
            "evaluation_seed_tables_sha256"
        ],
        "model": {
            row["name"]: torch.zeros(tuple(row["shape"]), dtype=torch.float32)
            for row in launch["expected_model_parameter_schema"]["parameters"]
        },
    }


def _receipt(
    authority: dict[str, Any],
    cell: contract.CalibrationCell,
    checkpoint: bytes,
) -> dict[str, Any]:
    _snapshot, result = _paths(authority)
    config = _config(authority, cell, result)
    identity = worker._expected_run_identity(
        authority, cell, result, _identity_config_sha256(config)
    )
    launch = _launch(authority, cell.index)
    schema = launch["expected_model_parameter_schema"]
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": worker.COMPLETE_STATUS,
        "campaign_id": contract.CAMPAIGN_ID,
        "authority_sha256": authority["authority_sha256"],
        "runtime_lock_sha256": launch["runtime_lock_sha256"],
        "model_state_authority_sha256": launch["model_state_authority_sha256"],
        "result_creation_receipt_sha256": launch[
            "result_creation_receipt_sha256"
        ],
        "roots": authority["roots"],
        "cell_index": cell.index,
        "setting_id": cell.setting_id,
        "env_config": cell.env_config,
        "seed": cell.seed,
        "run_name": cell.run_name,
        "wave_index": 0,
        "run_identity": identity,
        "run_identity_sha256": contract.stable_hash(identity),
        "completed_updates": 5_000,
        "nominal_optimizer_updates": 25_000,
        "scheduler_total_steps": 1_000_000,
        "checkpoint_relative_path": f"sealed-checkpoints/cell-{cell.index:03d}.pt",
        "checkpoint_raw_sha256": hashlib.sha256(checkpoint).hexdigest(),
        "checkpoint_size": len(checkpoint),
        "model_parameter_schema": schema,
        "model_parameter_schema_sha256": schema["schema_sha256"],
        "model_parameter_tensor_count": schema["parameter_count"],
        "model_parameter_total_numel": schema["total_numel"],
        "resolved_config": config,
        "resolved_config_contract_sha256": worker._resolved_config_contract_sha256(
            config
        ),
        "recipe": worker._receipt_recipe(),
        "outcome_observations": worker._receipt_outcome(),
        "reuse_policy": worker._receipt_reuse_policy(),
        "launch_sha256": launch["launch_sha256"],
    }
    value["receipt_sha256"] = contract.stable_hash(value)
    worker.validate_receipt(
        value, contract.load_manifest(), authority, cell, result, launch
    )
    return value


def _marker_launch(
    authority: dict[str, Any], cell: contract.CalibrationCell, launch_sha256: str
) -> dict[str, Any]:
    return {
        "authority_sha256": authority["authority_sha256"],
        "runtime_lock_sha256": _runtime_lock(authority)["runtime_lock_sha256"],
        "model_state_authority_sha256": _model_authority(authority)[
            "model_state_authority_sha256"
        ],
        "result_creation_receipt_sha256": _CREATION_RECEIPTS[
            _authority_key(authority)
        ]["result_creation_receipt_sha256"],
        "roots": authority["roots"],
        "launch_sha256": launch_sha256,
        "cell": {
            "index": cell.index,
            "setting_id": cell.setting_id,
            "env_config": cell.env_config,
            "seed": cell.seed,
            "run_name": cell.run_name,
        },
    }


def _freeze_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            path.chmod(0o444)
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for path in sorted(directories, key=lambda value: len(value.parts), reverse=True):
        path.chmod(0o555)
    root.chmod(0o555)


def _build_terminal_census(authority: dict[str, Any]) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    manifest = contract.load_manifest()
    for wave_index, job_id in ((0, "910001"), (1, "910002")):
        for cell in contract.expand_cells(manifest):
            row: dict[str, Any] = {
                "wave_index": wave_index,
                "cell_index": cell.index,
                "setting_id": cell.setting_id,
                "seed": cell.seed,
                "array_job_id": job_id,
                "array_task_id": cell.index,
                "state": "COMPLETED",
                "exit_code": "0:0",
            }
            row["sacct_row_sha256"] = contract.stable_hash(row)
            tasks.append(row)
    scratch = Path(authority["environment"]["scratch_root"])
    submission_receipt = {
        "schema_version": 1,
        "status": contract.TERMINAL_SUBMISSION_STATUS,
        "campaign_id": contract.CAMPAIGN_ID,
        "authority_sha256": authority["authority_sha256"],
        "scheduler_protocol_sha256": authority["roots"]["protocol_sha256"],
        "wave0": {
            "array_job_id": "910001",
            "array_spec": "0-19%20",
            "task_count": 20,
        },
        "wave1": {
            "array_job_id": "910002",
            "array_spec": "0-19%20",
            "task_count": 20,
            "dependency": "afterok:910001",
            "kill_on_invalid_dependency": True,
        },
        "report": {"dependency": "afterok:910002"},
        "external_submission_evidence_sha256": _digest(
            "synthetic-scheduler-submission-evidence"
        ),
    }
    evidence_directory = scratch / "scheduler"
    evidence_directory.mkdir(mode=0o700)
    evidence_path = evidence_directory / "terminal-census.raw.json"
    evidence_bytes = contract.terminal_census_evidence_bytes(
        tasks, submission_receipt
    )
    evidence_path.write_bytes(evidence_bytes)
    evidence_path.chmod(0o444)
    evidence_info = evidence_path.stat(follow_symlinks=False)
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": contract.TERMINAL_CENSUS_STATUS,
        "campaign_id": contract.CAMPAIGN_ID,
        "manifest_file_sha256": contract.file_sha256(contract.MANIFEST_PATH),
        "authority_sha256": authority["authority_sha256"],
        "runtime_lock_sha256": _runtime_lock(authority)["runtime_lock_sha256"],
        "model_state_authority_sha256": _model_authority(authority)[
            "model_state_authority_sha256"
        ],
        "result_creation_receipt_sha256": _CREATION_RECEIPTS[
            _authority_key(authority)
        ]["result_creation_receipt_sha256"],
        "scheduler_protocol_sha256": authority["roots"]["protocol_sha256"],
        "evidence_relative_path": "scheduler/terminal-census.raw.json",
        "evidence_file_identity": {
            "device": evidence_info.st_dev,
            "inode": evidence_info.st_ino,
            "mode": stat.S_IMODE(evidence_info.st_mode),
            "uid": evidence_info.st_uid,
            "gid": evidence_info.st_gid,
            "nlink": evidence_info.st_nlink,
            "size": evidence_info.st_size,
        },
        "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        "submission_receipt": submission_receipt,
        "submission_receipt_sha256": contract.stable_hash(submission_receipt),
        "production_ready": False,
        "external_anchor_sha256": None,
        "tasks": tasks,
        "tasks_sha256": contract.stable_hash(tasks),
    }
    value["terminal_census_sha256"] = contract.stable_hash(value)
    contract.validate_terminal_census(
        value, manifest, authority, _runtime_lock(authority),
        _model_authority(authority), value["result_creation_receipt_sha256"],
    )
    return value


def _result_fixture(root: Path, authority: dict[str, Any]) -> Path:
    del root
    torch = pytest.importorskip("torch")
    _snapshot, result = _paths(authority)
    assert [path.name for path in result.iterdir()] == [
        controller.RESULT_CREATION_RECEIPT_PATH
    ]
    launches = [_launch(authority, index) for index in range(20)]
    (result / "sealed-cells").mkdir()
    (result / "sealed-checkpoints").mkdir()
    (result / "control").mkdir()
    (result / "live-runs").mkdir()
    for cell, launch in zip(
        contract.expand_cells(contract.load_manifest()), launches, strict=True
    ):
        checkpoint_path = result / "sealed-checkpoints" / f"cell-{cell.index:03d}.pt"
        torch.save(_checkpoint_payload(authority, launch, torch), checkpoint_path)
        checkpoint_path.chmod(0o644)
        inspected = worker.inspect_checkpoint(checkpoint_path, launch, authority)
        receipt = worker._build_receipt(
            launch, inspected, f"sealed-checkpoints/cell-{cell.index:03d}.pt", 0
        )
        _write_json(
            result / "sealed-cells" / f"cell-{cell.index:03d}.json", receipt
        )
        control = result / "control" / f"cell-{cell.index:03d}"
        control.mkdir()
        _write_json(
            control / "wave-0.json",
            worker._completion_marker(launch, 0, receipt["receipt_sha256"]),
        )
        _write_json(
            control / "wave-1.json",
            worker._completion_marker(
                launch, 1, receipt["receipt_sha256"], noop=True
            ),
        )
    _freeze_tree(result)
    _LAUNCH_AUTHORIZATIONS[str(result)] = controller.build_launch_authorization(
        contract.load_manifest(), authority, _runtime_lock(authority),
        _model_authority(authority), launches
    )
    terminal = _build_terminal_census(authority)
    _TERMINAL_CENSUSES[_authority_key(authority)] = terminal
    return result


_LAUNCH_AUTHORIZATIONS: dict[str, dict[str, Any]] = {}


def _launch_authorization(result: Path) -> dict[str, Any]:
    return _LAUNCH_AUTHORIZATIONS[str(result)]


def _validate_result(
    result: Path,
    manifest: dict[str, Any],
    authority: dict[str, Any],
) -> dict[str, Any]:
    # The runtime fixture is intentionally marked synthetic/non-production.  It
    # proves validator mechanics but can never exercise the publication-ready API.
    return seal.validate_result_root(
        result, manifest, authority, _runtime_lock(authority),
        _model_authority(authority), _terminal_census(authority),
        _launch_authorization(result), _regeneration_only=True,
    )


def _validate_lock(
    value: dict[str, Any],
    manifest: dict[str, Any],
    authority: dict[str, Any],
) -> dict[str, Any]:
    result = Path(value["result_root"])
    return seal.validate_lock(
        value, manifest, authority, _runtime_lock(authority),
        _model_authority(authority), _terminal_census(authority),
        _launch_authorization(result),
    )


def _rehash_receipt(value: dict[str, Any]) -> None:
    value["run_identity_sha256"] = contract.stable_hash(value["run_identity"])
    value.pop("receipt_sha256", None)
    value["receipt_sha256"] = contract.stable_hash(value)


def _edit_receipt(
    result: Path,
    index: int,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    cells = result / "sealed-cells"
    path = cells / f"cell-{index:03d}.json"
    cells.chmod(0o755)
    path.chmod(0o644)
    value = json.loads(path.read_text(encoding="utf-8"))
    mutate(value)
    _rehash_receipt(value)
    _write_json(path, value, 0o444)
    cells.chmod(0o555)
    control = result / "control" / f"cell-{index:03d}"
    control.chmod(0o755)
    for marker_path in sorted(control.glob("wave-*.json")):
        marker_path.chmod(0o644)
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if "receipt_sha256" in marker:
            marker["receipt_sha256"] = value["receipt_sha256"]
            marker.pop("marker_sha256", None)
            marker["marker_sha256"] = contract.stable_hash(marker)
        _write_json(marker_path, marker, 0o444)
    control.chmod(0o555)


def _rehash_authority(authority: dict[str, Any]) -> None:
    census = authority["seed_collision_census"]
    census["evidence_sha256"] = contract.seed_census_evidence_sha256(census)
    census.pop("census_sha256", None)
    census["census_sha256"] = contract.stable_hash(census)
    authority.pop("authority_sha256", None)
    authority["authority_sha256"] = contract.stable_hash(authority)


def _rehash_lock(lock: dict[str, Any]) -> None:
    lock["inventory_sha256"] = contract.stable_hash(lock["inventory"])
    lock.pop("seal_sha256", None)
    lock["seal_sha256"] = contract.stable_hash(lock)


def test_calibration_manifest_preregisters_honest_all_ten_recipe() -> None:
    manifest = contract.load_manifest()
    cells = contract.expand_cells(manifest)
    assert len(cells) == 20
    assert [cell.seed for cell in cells[:4]] == [244, 245, 244, 245]
    assert {cell.setting_id for cell in cells} == {row[0] for row in contract.SETTINGS}
    assert manifest["seed_collision_census"]["scope"] == contract.SEED_CENSUS_SCOPE
    assert manifest["seed_collision_census"]["scope_sha256"] == (
        contract.seed_census_scope_sha256()
    )
    recipe = manifest["recipe"]
    assert recipe["nominal_optimizer_updates"] == 25_000
    assert recipe["stop_environment"] == {"TREEWM_STOP_AFTER_UPDATE": "5000"}
    assert recipe["scheduler_total_steps"] == 1_000_000
    assert recipe["validation_due_at_export"] is False
    assert recipe["evaluation_due_at_export"] is False
    assert recipe["visualization_due_at_export"] is False
    assert recipe["executable_prefix_enabled"] == {
        "action": True, "latent": True, "endpoint": True
    }
    assert recipe["executable_prefix_weights"] == {
        "action": 0.0, "latent": 0.0, "endpoint": 0.0
    }
    assert recipe["outcome_measurement_allowed"] is False
    prerequisites = manifest["external_prerequisites"]
    assert prerequisites["runtime_content_lock"]["production_lock_sha256"] is None
    assert prerequisites["runtime_content_lock"]["availability"] == (
        "required_sealed_external_artifact_absent"
    )
    assert prerequisites["model_state_authority"]["hook_relative_path"] == (
        contract.MODEL_AUTHORITY_HOOK_PATH
    )
    assert prerequisites["model_state_authority"]["hook_interface"] == (
        contract.MODEL_AUTHORITY_HOOK_INTERFACE
    )
    assert prerequisites["model_state_authority"]["hook_raw_sha256"] is None
    assert prerequisites["model_state_authority"][
        "synthetic_fixture_authorizes_production"
    ] is False
    assert prerequisites["result_creation_receipt"]["production_receipt_sha256"] is None
    assert prerequisites["scheduler_terminal_census"]["production_lock_sha256"] is None
    assert prerequisites["scheduler_terminal_census"]["leaf_scheduler_query"] == "forbidden"
    publication_boundary = prerequisites["lock_publication_boundary"]
    assert publication_boundary["production_lock_sha256"] is None
    assert publication_boundary["advisory_flock_alone_is_immutability_proof"] is False
    assert publication_boundary["transient_detached_write_prevention_claimed"] is False
    assert manifest["fixed_weight_audit_downstream"] == {
        "regimes": ["exp24_zero_prefix_exact_5000", "scratch_initialization"],
        "checkpoint_seeds": [244, 245],
        "scratch_seeds": [230, 231],
        "settings": 10,
        "batches_per_setting_regime": 2,
        "expected_rows": 80,
        "device": "cpu_fp32",
        "optimizer_steps": 0,
        "weight_tuple_source": "frozen_exp24_formal_manifest",
        "retuning_allowed": False,
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["seed_collision_census"].update(scope="anything"),
        lambda value: value["fixed_weight_audit_downstream"].update(
            retuning_allowed=0
        ),
        lambda value: value["recipe"].update(validation_due_at_export=0),
    ],
)
def test_manifest_scope_and_exact_boolean_aliases_fail_closed(
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    value = copy.deepcopy(contract.load_manifest())
    mutate(value)
    with pytest.raises(contract.CalibrationContractError):
        contract.validate_manifest(value)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value["seed_collision_census"].update(scope="anything"), "scope"),
        (
            lambda value: value["seed_collision_census"].update(
                prior_assignment_matches=1
            ),
            "collision",
        ),
        (
            lambda value: value["seed_collision_census"].update(
                performed_before_first_calibration_run=1
            ),
            "before the first run|timing",
        ),
        (
            lambda value: value["seed_collision_census"].update(
                reachable_history_inventory_sha256="bad"
            ),
            "reachable_history",
        ),
        (
            lambda value: value["settings"][0][
                "resolved_config_contract_sha256_by_seed"
            ].update({"244": "bad"}),
            "resolved config contract",
        ),
    ],
)
def test_rehashed_authority_census_and_config_contract_drift_fails_closed(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    authority = _authority(tmp_path)
    mutation(authority)
    if authority["seed_collision_census"].get(
        "reachable_history_inventory_sha256"
    ) == "bad":
        authority["seed_collision_census"]["evidence_sha256"] = "0" * 64
    else:
        _rehash_authority(authority)
    with pytest.raises(contract.CalibrationContractError, match=match):
        contract.validate_authority(
            authority, contract.load_manifest(), _runtime_lock(authority)
        )


def test_controller_launch_and_two_wave_topology_are_exact(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    manifest = contract.load_manifest()
    launch = _launch(authority, 19)
    assert controller.validate_launch(
        launch, manifest, authority, _runtime_lock(authority),
        _model_authority(authority),
    ) == launch
    assert launch["cell"]["setting_id"] == "humanoidmaze-large"
    assert launch["cell"]["seed"] == 245
    assert launch["environment"]["TREEWM_STOP_AFTER_UPDATE"] == "5000"
    assert launch["environment"]["WANDB_MODE"] == "disabled"
    assert launch["path_authority"]["formal_output_membership_allowed"] is False
    assert "train.steps=25000" in launch["argv"]
    assert "train.scheduler_total_steps=1000000" in launch["argv"]
    for override in (
        "losses.enabled.executable_prefix_action=true",
        "losses.enabled.executable_prefix_latent=true",
        "losses.enabled.executable_prefix_endpoint=true",
        "losses.weights.executable_prefix_action=0.0",
        "losses.weights.executable_prefix_latent=0.0",
        "losses.weights.executable_prefix_endpoint=0.0",
    ):
        assert override in launch["argv"]
    topology = controller.topology_description()
    assert topology["wave0"]["array"] == topology["wave1"]["array"] == "0-19%20"
    assert topology["wave1"]["dependency"].startswith("afterok:")
    assert topology["adaptive_or_third_wave"] is False
    assert topology["exclusive_result_initialization_implemented_here"] is True
    with pytest.raises(
        contract.CalibrationContractError, match="result root already exists"
    ):
        controller.create_result_root_exclusive(
            manifest, authority, _runtime_lock(authority),
            _model_authority(authority),
        )


def test_controller_rejects_unsealed_path_escape_alias_and_trainer_drift(
    tmp_path: Path,
) -> None:
    manifest = contract.load_manifest()
    authority = _authority(tmp_path / "escape")
    snapshot, _result = _paths(authority)
    arbitrary = Path(authority["environment"]["scratch_root"]) / "arbitrary"
    arbitrary.mkdir()
    with pytest.raises(contract.CalibrationContractError, match="sealed scratch-relative"):
        controller.build_launch(
            manifest, authority, _runtime_lock(authority),
            _model_authority(authority), 0, snapshot, arbitrary,
        )
    with pytest.raises(contract.CalibrationContractError):
        controller.build_launch(
            manifest, authority, _runtime_lock(authority),
            _model_authority(authority), 0,
            snapshot,
            Path(authority["environment"]["formal_output_root"]),
        )

    authority = _authority(tmp_path / "trainer")
    snapshot, result = _paths(authority)
    snapshot.chmod(0o755)
    (snapshot / "scripts").chmod(0o755)
    trainer = snapshot / "scripts/train.py"
    trainer.chmod(0o644)
    trainer.write_bytes(b"drift")
    trainer.chmod(0o444)
    (snapshot / "scripts").chmod(0o555)
    snapshot.chmod(0o555)
    with pytest.raises(
        contract.CalibrationContractError,
        match="trainer bytes|snapshot_inventory_sha256",
    ):
        controller.build_launch(
            manifest, authority, _runtime_lock(authority),
            _model_authority(authority), 0, snapshot, result,
        )

    authority = _authority(tmp_path / "symlink")
    snapshot, result = _paths(authority)
    alternate = tmp_path / "symlink/alternate-result"
    alternate.mkdir()
    (result / controller.RESULT_CREATION_RECEIPT_PATH).unlink()
    result.rmdir()
    result.symlink_to(alternate, target_is_directory=True)
    with pytest.raises((contract.CalibrationContractError, OSError)):
        controller.build_launch(
            manifest, authority, _runtime_lock(authority),
            _model_authority(authority), 0, snapshot, result,
        )


def test_launch_rehashed_int_float_bool_alias_fails_closed(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    launch = _launch(authority)
    launch["path_authority"]["schema_version"] = 1.0
    launch.pop("launch_sha256")
    launch["launch_sha256"] = contract.stable_hash(launch)
    with pytest.raises(contract.CalibrationContractError, match="schema"):
        controller.validate_launch(
            launch, contract.load_manifest(), authority, _runtime_lock(authority),
            _model_authority(authority),
        )


@pytest.mark.parametrize(
    "relative",
    ["bin/python", "stdlib/BOUND.txt", "site-packages/treewm_native.so", "lib/libtreewm.so"],
)
def test_runtime_content_lock_detects_interpreter_module_native_and_library_substitution(
    tmp_path: Path, relative: str
) -> None:
    authority = _authority(tmp_path)
    runtime_lock = _runtime_lock(authority)
    target = Path(runtime_lock["runtime_root"]) / relative
    replacement = tmp_path / (target.name + ".replacement")
    replacement.write_bytes(target.read_bytes())
    replacement.chmod(stat.S_IMODE(target.stat().st_mode))
    target.parent.chmod(0o755)
    os.replace(replacement, target)
    target.parent.chmod(0o555)
    with pytest.raises(contract.CalibrationContractError, match="inventory|changed|identity"):
        contract.validate_runtime_lock(runtime_lock, contract.load_manifest())


def test_model_authority_requires_exact_reviewed_hook_and_rejects_coordinated_schema(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "schema")
    runtime_lock = _runtime_lock(authority)
    model_authority = copy.deepcopy(_model_authority(authority))
    snapshot, _result = _paths(authority)
    with contract.RetainedTree(
        snapshot, "model hostile fixture", directory_mode=0o555, file_mode=0o444,
        lock_exclusive=False,
    ) as tree:
        inventory = copy.deepcopy(tree.inventory)
    schema = model_authority["settings"][0]["parameter_schema"]
    schema["parameters"][0]["name"] = "decoder.head.bias_forged"
    schema.pop("schema_sha256")
    schema["schema_sha256"] = contract.stable_hash(schema)
    stdout = {
        "schema_version": 1,
        "interface": contract.MODEL_AUTHORITY_HOOK_INTERFACE,
        "request_sha256": model_authority["derivation_request_sha256"],
        "settings": model_authority["settings"],
    }
    model_authority["derivation_stdout_sha256"] = contract.stable_hash(stdout)
    model_authority.pop("model_state_authority_sha256")
    model_authority["model_state_authority_sha256"] = contract.stable_hash(
        model_authority
    )
    with pytest.raises(contract.CalibrationContractError, match="hook output differs"):
        contract.validate_model_state_authority(
            model_authority, contract.load_manifest(), authority, runtime_lock, inventory
        )

    authority = _authority(tmp_path / "missing-hook")
    runtime_lock = _runtime_lock(authority)
    snapshot, _result = _paths(authority)
    hook = snapshot / contract.MODEL_AUTHORITY_HOOK_PATH
    snapshot.chmod(0o755)
    hook.parent.chmod(0o755)
    hook.unlink()
    hook.parent.chmod(0o555)
    snapshot.chmod(0o555)
    with pytest.raises(contract.CalibrationContractError, match="lacks the exact"):
        contract.build_model_state_authority(
            contract.load_manifest(), authority, runtime_lock, snapshot
        )


def test_synthetic_runtime_and_model_authorities_never_become_publication_ready(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path)
    runtime_lock = _runtime_lock(authority)
    model_authority = _model_authority(authority)
    assert runtime_lock["execution_profile"] == contract.RUNTIME_SYNTHETIC_PROFILE
    assert runtime_lock["production_ready"] is False
    assert runtime_lock["closure_attestation"]["outside_runtime_paths"] == []
    assert model_authority["execution_profile"] == contract.MODEL_UNSEALED_PROFILE
    assert model_authority["production_ready"] is False
    assert model_authority["external_hook_source_sha256"] is None
    with pytest.raises(contract.CalibrationContractError, match="production runtime"):
        contract.require_production_runtime(runtime_lock, "synthetic fixture")
    with pytest.raises(contract.CalibrationContractError, match="production model-state"):
        contract.require_production_model_authority(model_authority, "synthetic fixture")

    forged = copy.deepcopy(model_authority)
    forged["execution_profile"] = contract.MODEL_PRODUCTION_PROFILE
    forged["production_ready"] = True
    forged["external_hook_source_sha256"] = forged["hook_source_sha256"]
    forged.pop("model_state_authority_sha256")
    forged["model_state_authority_sha256"] = contract.stable_hash(forged)
    snapshot, _result = _paths(authority)
    with contract.RetainedTree(
        snapshot, "synthetic model readiness hostile", directory_mode=0o555,
        file_mode=0o444, lock_exclusive=False,
    ) as tree:
        with pytest.raises(contract.CalibrationContractError, match="production runtime"):
            contract.validate_model_state_authority(
                forged, contract.load_manifest(), authority, runtime_lock,
                tree.inventory,
            )


@pytest.mark.parametrize("mutation", ["same-byte-replace", "rehashed-wrong-root"])
def test_exclusive_result_creation_receipt_binds_its_created_inode_and_root(
    tmp_path: Path, mutation: str
) -> None:
    authority = _authority(tmp_path)
    snapshot, result = _paths(authority)
    receipt_path = result / controller.RESULT_CREATION_RECEIPT_PATH
    if mutation == "same-byte-replace":
        replacement = tmp_path / "replacement-receipt"
        replacement.write_bytes(receipt_path.read_bytes())
        replacement.chmod(0o444)
        os.replace(replacement, receipt_path)
    else:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["result_root"] = str(tmp_path / "forged-result")
        receipt.pop("result_creation_receipt_sha256")
        receipt["result_creation_receipt_sha256"] = contract.stable_hash(receipt)
        receipt_path.chmod(0o644)
        _write_json(receipt_path, receipt, 0o444)
    with pytest.raises(contract.CalibrationContractError, match="receipt"):
        controller.build_launch(
            contract.load_manifest(), authority, _runtime_lock(authority),
            _model_authority(authority), 0, snapshot, result,
        )


@pytest.mark.parametrize(
    "mutation", [
        "missing-evidence", "same-byte-replace", "task-state", "duplicate-task",
        "forty-job-ids", "wrong-wave-dependency", "path",
    ]
)
def test_external_scheduler_terminal_census_and_evidence_fail_closed(
    tmp_path: Path, mutation: str
) -> None:
    authority = _authority(tmp_path)
    terminal = _build_terminal_census(authority)
    evidence_path = (
        Path(authority["environment"]["scratch_root"])
        / terminal["evidence_relative_path"]
    )
    if mutation == "missing-evidence":
        evidence_path.unlink()
    elif mutation == "same-byte-replace":
        replacement = tmp_path / "terminal-evidence-replacement"
        replacement.write_bytes(evidence_path.read_bytes())
        replacement.chmod(0o444)
        os.replace(replacement, evidence_path)
    elif mutation == "path":
        terminal["evidence_relative_path"] = "../terminal-census.raw.json"
        terminal.pop("terminal_census_sha256")
        terminal["terminal_census_sha256"] = contract.stable_hash(terminal)
    else:
        if mutation == "task-state":
            terminal["tasks"][0]["state"] = "RUNNING"
            terminal["tasks"][0].pop("sacct_row_sha256")
            terminal["tasks"][0]["sacct_row_sha256"] = contract.stable_hash(
                terminal["tasks"][0]
            )
        elif mutation == "duplicate-task":
            terminal["tasks"][1] = copy.deepcopy(terminal["tasks"][0])
        elif mutation == "forty-job-ids":
            for index, row in enumerate(terminal["tasks"]):
                row["array_job_id"] = str(920000 + index)
                row.pop("sacct_row_sha256")
                row["sacct_row_sha256"] = contract.stable_hash(row)
        else:
            terminal["submission_receipt"]["wave1"]["dependency"] = "afterok:999999"
            terminal["submission_receipt_sha256"] = contract.stable_hash(
                terminal["submission_receipt"]
            )
        raw = contract.terminal_census_evidence_bytes(
            terminal["tasks"], terminal["submission_receipt"]
        )
        evidence_path.chmod(0o644)
        evidence_path.write_bytes(raw)
        evidence_path.chmod(0o444)
        info = evidence_path.stat(follow_symlinks=False)
        terminal["evidence_file_identity"] = {
            "device": info.st_dev, "inode": info.st_ino,
            "mode": stat.S_IMODE(info.st_mode), "uid": info.st_uid,
            "gid": info.st_gid, "nlink": info.st_nlink, "size": info.st_size,
        }
        terminal["evidence_sha256"] = hashlib.sha256(raw).hexdigest()
        terminal["tasks_sha256"] = contract.stable_hash(terminal["tasks"])
        terminal.pop("terminal_census_sha256")
        terminal["terminal_census_sha256"] = contract.stable_hash(terminal)
    with pytest.raises((contract.CalibrationContractError, OSError)):
        contract.validate_terminal_census(
            terminal, contract.load_manifest(), authority, _runtime_lock(authority),
            _model_authority(authority),
            _CREATION_RECEIPTS[_authority_key(authority)][
                "result_creation_receipt_sha256"
            ],
        )


def test_missing_terminal_census_fails_before_any_lock_publication(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    snapshot, result = _paths(authority)
    launches = [_launch(authority, index) for index in range(20)]
    _freeze_tree(result)
    authorization = controller.build_launch_authorization(
        contract.load_manifest(), authority, _runtime_lock(authority),
        _model_authority(authority), launches,
    )
    publication = (
        Path(authority["environment"]["scratch_root"])
        / seal.CHECKPOINT_LOCK_RELATIVE_PATH
    )
    publication.parent.mkdir(parents=True)
    publication.parent.chmod(0o700)
    with pytest.raises(contract.CalibrationContractError, match="terminal census"):
        seal.validate_result_root(
            result, contract.load_manifest(), authority, _runtime_lock(authority),
            _model_authority(authority), None, authorization,
            publication_path=publication,
        )
    assert not publication.exists()


def test_worker_inspects_exact_real_checkpoint_schema(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    authority = _authority(tmp_path)
    launch = _launch(authority)
    payload = _checkpoint_payload(authority, launch, torch)
    checkpoint = tmp_path / "latest.pt"
    torch.save(payload, checkpoint)
    inspected = worker.inspect_checkpoint(checkpoint, launch, authority)
    assert inspected["checkpoint_size"] == checkpoint.stat().st_size
    expected_schema = launch["expected_model_parameter_schema"]
    assert inspected["model_parameter_tensor_count"] == expected_schema["parameter_count"]
    assert inspected["model_parameter_total_numel"] == expected_schema["total_numel"]
    assert inspected["model_parameter_schema"] == expected_schema
    assert inspected["resolved_config"] == payload["config"]
    assert inspected["resolved_config_contract_sha256"] == (
        contract.setting_authority(authority, "scene")[
            "resolved_config_contract_sha256_by_seed"
        ]["244"]
    )


def test_safe_loader_numpy_globals_are_scoped_and_resource_limits_are_bound(
    tmp_path: Path,
) -> None:
    numpy = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "numpy-checkpoint.pt"
    torch.save({
        "array": numpy.arange(8, dtype=numpy.float32),
        "dtype": numpy.dtype(numpy.uint32),
    }, checkpoint)
    before = list(torch.serialization.get_safe_globals())
    payload, evidence = _safe_load_path(checkpoint)
    after = list(torch.serialization.get_safe_globals())
    assert before == after
    assert payload["array"][contract.SAFE_NUMPY_MARKER] == 1
    assert payload["dtype"] == {"__exp24_numpy_dtype_v1__": "uint32"}
    assert evidence["safe_globals"] == list(contract.SAFE_CHECKPOINT_ALLOWED_GLOBALS)
    assert evidence["weights_only"] is True
    limits = evidence["decoder_limits"]
    assert 0 < limits["RLIMIT_AS"][0] == limits["RLIMIT_AS"][1] <= (
        contract.SAFE_CHECKPOINT_MAX_ADDRESS_SPACE
    )
    assert 0 < limits["RLIMIT_CPU"][0] == limits["RLIMIT_CPU"][1] <= (
        contract.SAFE_CHECKPOINT_MAX_CPU_SECONDS
    )
    assert limits["RLIMIT_FSIZE"] == limits["RLIMIT_CORE"] == [0, 0]
    assert 0 < limits["RLIMIT_NOFILE"][0] == limits["RLIMIT_NOFILE"][1] <= (
        contract.SAFE_CHECKPOINT_MAX_OPEN_FILES
    )
    assert evidence["decoder_isolation"] == (
        "clean_exec_close_fds_hard_rlimit_canonical_json_only"
    )
    assert evidence["decoder_production_ready"] is False


def test_safe_loader_clean_exec_does_not_inherit_registered_safe_global(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    marker = tmp_path / "INHERITED_GLOBAL_EXECUTED"
    checkpoint = tmp_path / "inherited-safe-global.pt"
    torch.save({"payload": _MaliciousCheckpointReducer(marker)}, checkpoint)
    previous = list(torch.serialization.get_safe_globals())
    try:
        torch.serialization.add_safe_globals([_malicious_checkpoint_side_effect])
        with pytest.raises(contract.CalibrationContractError, match="safely inspect"):
            _safe_load_path(checkpoint)
    finally:
        torch.serialization.clear_safe_globals()
        torch.serialization.add_safe_globals(previous)
    assert not marker.exists()


def test_safe_loader_clean_exec_cannot_release_parent_result_flock(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    result = tmp_path / "result-lock"
    result.mkdir()
    locked = os.open(result, contract.DIRECTORY_FLAGS)
    contender: int | None = None
    try:
        fcntl.flock(locked, fcntl.LOCK_EX | fcntl.LOCK_NB)
        checkpoint = tmp_path / "unlock-parent-flock.pt"
        torch.save({"payload": _InheritedUnlockReducer(locked)}, checkpoint)
        previous = list(torch.serialization.get_safe_globals())
        try:
            torch.serialization.add_safe_globals([_malicious_checkpoint_unlock])
            with pytest.raises(contract.CalibrationContractError, match="safely inspect"):
                _safe_load_path(checkpoint)
        finally:
            torch.serialization.clear_safe_globals()
            torch.serialization.add_safe_globals(previous)
        contender = os.open(result, contract.DIRECTORY_FLAGS)
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        if contender is not None:
            os.close(contender)
        fcntl.flock(locked, fcntl.LOCK_UN)
        os.close(locked)


def test_safe_loader_rejects_zero_tensor_reserved_marker_model_forgery(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    authority = _authority(tmp_path)
    launch = _launch(authority)
    payload = _checkpoint_payload(authority, launch, torch)
    payload["model"] = {
        row["name"]: {
            contract.SAFE_TENSOR_MARKER: 1,
            "shape": row["shape"],
            "dtype": row["dtype"],
            "numel": row["numel"],
            "storage_bytes": row["storage_bytes"],
            "content_sha256": "0" * 64,
            "device": "cpu",
            "layout": "strided",
            "storage_alias_policy": "unique_exact_storage",
        }
        for row in launch["expected_model_parameter_schema"]["parameters"]
    }
    checkpoint = tmp_path / "zero-tensor-marker-forgery.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(contract.CalibrationContractError, match="reserved evidence marker"):
        worker.inspect_checkpoint(checkpoint, launch, authority)


@pytest.mark.parametrize("nested", [False, True])
def test_safe_loader_rejects_malicious_reducers_without_side_effect(
    tmp_path: Path, nested: bool
) -> None:
    torch = pytest.importorskip("torch")
    marker = tmp_path / "MALICIOUS_SIDE_EFFECT"
    hostile: object = _MaliciousCheckpointReducer(marker)
    if nested:
        hostile = {"outer": [{"inner": hostile}]}
    checkpoint = tmp_path / "malicious.pt"
    torch.save({"payload": hostile}, checkpoint)
    with pytest.raises(contract.CalibrationContractError, match="safely inspect"):
        _safe_load_path(checkpoint)
    assert not marker.exists()


@pytest.mark.parametrize(
    "hostile",
    ["compression", "traversal", "duplicate", "archive-depth"],
)
def test_safe_loader_rejects_hostile_archive_container(
    tmp_path: Path, hostile: str
) -> None:
    checkpoint = tmp_path / f"{hostile}.pt"
    compression = zipfile.ZIP_DEFLATED if hostile == "compression" else zipfile.ZIP_STORED
    with zipfile.ZipFile(checkpoint, "w", compression=compression) as archive:
        name = "root/data.pkl"
        if hostile == "traversal":
            name = "root/../data.pkl"
        elif hostile == "archive-depth":
            name = "a/b/c/d/e/f/g/h/data.pkl"
        archive.writestr(name, b"not decoded")
        if hostile == "duplicate":
            archive.writestr(name, b"duplicate")
    with pytest.raises(contract.CalibrationContractError):
        _safe_load_path(checkpoint)


def test_safe_loader_archive_storage_size_and_graph_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "bounded.pt"
    torch.save({"tensor": torch.zeros(2)}, checkpoint)
    monkeypatch.setattr(contract, "SAFE_CHECKPOINT_MAX_TENSORS", 0)
    with pytest.raises(contract.CalibrationContractError, match="storage count"):
        _safe_load_path(checkpoint)
    monkeypatch.setattr(contract, "SAFE_CHECKPOINT_MAX_TENSORS", 65_536)
    monkeypatch.setattr(contract, "SAFE_CHECKPOINT_MAX_BYTES", checkpoint.stat().st_size - 1)
    with pytest.raises(contract.CalibrationContractError, match="size"):
        _safe_load_path(checkpoint)
    monkeypatch.setattr(contract, "SAFE_CHECKPOINT_MAX_BYTES", 16 * 1024**3)
    monkeypatch.setattr(contract, "SAFE_CHECKPOINT_MAX_CENTRAL_DIRECTORY_BYTES", 1)
    with pytest.raises(contract.CalibrationContractError, match="central directory"):
        _safe_load_path(checkpoint)
    monkeypatch.setattr(
        contract, "SAFE_CHECKPOINT_MAX_CENTRAL_DIRECTORY_BYTES", 64 * 1024**2
    )
    nested: object = "leaf"
    for _index in range(contract.SAFE_CHECKPOINT_MAX_GRAPH_DEPTH + 2):
        nested = [nested]
    torch.save({"nested": nested}, checkpoint)
    with pytest.raises(contract.CalibrationContractError, match="nesting"):
        _safe_load_path(checkpoint)


@pytest.mark.parametrize(
    "hostile",
    ["meta-trillion", "sparse", "quantized", "subclass", "shared-storage", "nan", "inf"],
)
def test_safe_loader_rejects_non_plain_or_nonfinite_model_tensors(
    tmp_path: Path, hostile: str
) -> None:
    torch = pytest.importorskip("torch")
    authority = _authority(tmp_path)
    launch = _launch(authority)
    payload = _checkpoint_payload(authority, launch, torch)
    model = payload["model"]
    first_name = next(iter(model))
    if hostile == "meta-trillion":
        model[first_name] = torch.empty((10**12,), device="meta")
    elif hostile == "sparse":
        model[first_name] = torch.sparse_coo_tensor(
            torch.tensor([[0]]), torch.tensor([1.0]), (3,)
        )
    elif hostile == "quantized":
        model[first_name] = torch.quantize_per_tensor(
            torch.zeros(3), scale=0.1, zero_point=0, dtype=torch.qint8
        )
    elif hostile == "subclass":
        model[first_name] = torch.nn.Parameter(torch.zeros(3))
    elif hostile == "shared-storage":
        payload["aliased_tensor"] = model[first_name]
    elif hostile == "nan":
        model[first_name].reshape(-1)[0] = float("nan")
    else:
        model[first_name].reshape(-1)[0] = float("inf")
    checkpoint = tmp_path / f"{hostile}.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(contract.CalibrationContractError):
        worker.inspect_checkpoint(checkpoint, launch, authority)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda payload: payload.update(step=4_999), "update 5000"),
        (lambda payload: payload.update(step=5_000.0), "update 5000"),
        (
            lambda payload: payload["config"]["losses"]["weights"].update(
                executable_prefix_action=0
            ),
            "exact float zero",
        ),
        (
            lambda payload: payload["config"]["losses"]["enabled"].update(
                executable_prefix_latent=False
            ),
            "graph is disabled",
        ),
        (
            lambda payload: payload["config"]["train"].update(eval_every=5_000),
            "eval_every",
        ),
        (
            lambda payload: payload["config"]["train"].update(viz_every=5_000),
            "viz_every",
        ),
        (
            lambda payload: payload["config"].update(arm="formal"),
            "config arm",
        ),
        (
            lambda payload: payload["config"]["train"].update(
                fixture_full_config_leaf="drift"
            ),
            "resolved config contract",
        ),
        (
            lambda payload: payload["run_identity"].update(arm="formal"),
            "run identity differs",
        ),
        (
            lambda payload: payload["run_identity"].update(unexpected="formal"),
            "fields differ",
        ),
        (
            lambda payload: payload["run_identity"].update(
                gradient_checkpointing=1
            ),
            "run identity differs",
        ),
        (
            lambda payload: payload.update(evaluation_seed_tables_sha256="0" * 64),
            "seed-table",
        ),
    ],
)
def test_worker_checkpoint_update_weight_eval_viz_identity_and_config_hostiles(
    tmp_path: Path,
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    torch = pytest.importorskip("torch")
    authority = _authority(tmp_path)
    launch = _launch(authority)
    payload = _checkpoint_payload(authority, launch, torch)
    mutate(payload)
    payload["identity_sha256"] = contract.stable_hash(payload["run_identity"])
    checkpoint = tmp_path / "latest.pt"
    torch.save(payload, checkpoint)
    with pytest.raises(contract.CalibrationContractError, match=match):
        worker.inspect_checkpoint(checkpoint, launch, authority)


def test_checkpoint_load_hash_path_swap_is_detected(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    authority = _authority(tmp_path)
    launch = _launch(authority)
    checkpoint = tmp_path / "latest.pt"
    replacement = tmp_path / "replacement.pt"
    torch.save(_checkpoint_payload(authority, launch, torch), checkpoint)
    torch.save({"hostile": True}, replacement)
    replacement_sha256 = hashlib.sha256(replacement.read_bytes()).hexdigest()

    class _TestSwapAfterCleanDecoder:
        """Inject a persistent rebind only after the authenticated child exits."""

        def __init__(self, base: Any) -> None:
            self.base = base
            self.profile = base.profile
            self.production_ready = base.production_ready
            self.source_sha256 = base.source_sha256
            self.python_sha256 = base.python_sha256
            self.verify_calls = 0

        def command(
            self, checkpoint_fd: int
        ) -> tuple[list[str], tuple[int, ...], dict[str, str]]:
            return self.base.command(checkpoint_fd)

        def verify(self) -> None:
            self.base.verify()
            self.verify_calls += 1
            if self.verify_calls == 2:
                os.replace(replacement, checkpoint)

    decoder: _TestSwapAfterCleanDecoder | None = None
    with pytest.raises(
        contract.CalibrationContractError,
        match="identity changed|pathname changed",
    ):
        with contract._AmbientCheckpointDecoder() as base:
            decoder = _TestSwapAfterCleanDecoder(base)
            with worker.RetainedCheckpoint(
                checkpoint.absolute(), decoder=decoder
            ) as retained:
                worker._inspect_retained_checkpoint(retained, launch, authority)
    assert decoder is not None
    assert decoder.verify_calls == 2
    assert not replacement.exists()
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == replacement_sha256


def test_retained_checkpoint_publish_is_exclusive_and_same_descriptor(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"checkpoint-source")
    result = tmp_path / "result"
    result.mkdir()
    with worker.RetainedCheckpoint(source) as retained:
        digest, size = retained.publish_exclusive(result, "sealed/cell.pt")
        assert source.exists()
        retained.unlink_verified()
    target = result / "sealed/cell.pt"
    assert not source.exists()
    assert target.read_bytes() == b"checkpoint-source"
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    assert size == len(target.read_bytes())
    assert stat.S_IMODE(target.stat().st_mode) == 0o444

    second = tmp_path / "second.pt"
    second.write_bytes(b"different")
    with worker.RetainedCheckpoint(second) as retained:
        with pytest.raises(FileExistsError):
            retained.publish_exclusive(result, "sealed/cell.pt")
    assert second.exists()
    assert target.read_bytes() == b"checkpoint-source"


def test_retained_checkpoint_publish_path_swap_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.pt"
    source.write_bytes(b"x" * 8192)
    result = tmp_path / "result"
    result.mkdir()
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"hostile")
    target = result / "sealed/cell.pt"
    real_write = os.write
    swapped = False

    def hostile_write(descriptor: int, data: bytes) -> int:
        nonlocal swapped
        written = real_write(descriptor, data)
        if not swapped:
            swapped = True
            target.parent.mkdir(parents=True, exist_ok=True)
            os.replace(replacement, target)
        return written

    monkeypatch.setattr(os, "write", hostile_write)
    with worker.RetainedCheckpoint(source) as retained:
        with pytest.raises(contract.CalibrationContractError, match="pathname changed"):
            retained.publish_exclusive(result, "sealed/cell.pt")
    assert source.exists()
    assert target.read_bytes() == b"hostile"


@pytest.mark.parametrize("wave_index", [0, 1])
def test_complete_export_binds_checkpoint_receipt_then_retires_source(
    tmp_path: Path, wave_index: int
) -> None:
    torch = pytest.importorskip("torch")
    authority = _authority(tmp_path)
    launch = _launch(authority)
    run_directory = Path(launch["run_directory"])
    source = run_directory / "checkpoints/latest.pt"
    source.parent.mkdir(parents=True)
    torch.save(_checkpoint_payload(authority, launch, torch), source)
    inspected = worker.inspect_checkpoint(source, launch, authority)
    stage = {
        "schema_version": 1,
        "status": "awaiting_external_stage_gate",
        "objective_version": contract.OBJECTIVE,
        "completed_updates": 5_000,
        "step": 5_000,
        "total_steps": 25_000,
        "scheduler_total_steps": 1_000_000,
        "identity_sha256": inspected["run_identity_sha256"],
        "checkpoint": "checkpoints/latest.pt",
        "checkpoint_sha256": inspected["checkpoint_raw_sha256"],
        "evaluation_seed_tables_sha256": inspected["run_identity"][
            "evaluation_seed_tables_sha256"
        ],
    }
    _write_json(run_directory / "stage-gates/AWAITING_GATE_5000.json", stage)
    if wave_index == 1:
        _write_json(
            Path(launch["result_root"]) / "control/cell-000/wave-0.json",
            worker._continuation_marker(launch, inspected),
            0o444,
        )

    receipt = worker._export_complete(launch, authority, wave_index)
    result = Path(launch["result_root"])
    published = result / receipt["checkpoint_relative_path"]
    receipt_path = result / "sealed-cells/cell-000.json"
    assert not source.exists()
    assert published.exists() and receipt_path.exists()
    assert stat.S_IMODE(published.stat().st_mode) == 0o444
    assert stat.S_IMODE(receipt_path.stat().st_mode) == 0o444
    assert hashlib.sha256(published.read_bytes()).hexdigest() == (
        receipt["checkpoint_raw_sha256"]
    )
    assert json.loads(receipt_path.read_text(encoding="utf-8")) == receipt


def test_contract_read_json_rejects_parent_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    _write_json(real / "value.json", {"status": "hostile"})
    link = tmp_path / "link"
    link.symlink_to(real, target_is_directory=True)
    with pytest.raises((contract.CalibrationContractError, OSError)):
        contract.read_json(link / "value.json", "symlinked JSON")


@pytest.mark.parametrize(
    "script",
    ["calibration_controller.py", "calibration_worker.py", "calibration_seal.py"],
)
@pytest.mark.parametrize("mode", ["--describe", "--test-only"])
def test_describe_and_test_only_are_write_free(
    tmp_path: Path, script: str, mode: str
) -> None:
    before = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    result = subprocess.run(
        [sys.executable, "-B", str(PACKAGE_DIR / script), mode],
        cwd=tmp_path,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["schema_version"] == 1
    after = sorted(path.relative_to(tmp_path) for path in tmp_path.rglob("*"))
    assert before == after == []


def test_unsealed_placeholder_is_not_accepted_as_checkpoint_authority(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path)
    result = _result_fixture(tmp_path, authority)
    placeholder = contract.read_json(contract.PLACEHOLDER_PATH, "placeholder")
    assert placeholder["status"] == (
        "unsealed_exp24_all_ten_zero_prefix_checkpoint_source"
    )
    assert placeholder["inventory"] == []
    assert placeholder["fixed_weight_audit"]["settings"] == 10
    seal.validate_unsealed_placeholder(placeholder)
    assert placeholder["production_readiness"] == {
        "executable": False,
        "sealable": False,
        "blocked_on": [
            "sealed_runtime_content_lock",
            "reviewed_production_model_authority_hook_and_sealed_model_state_authority",
            "sealed_result_creation_receipt",
            "sealed_external_scheduler_terminal_census",
            "sealed_external_lock_publication_boundary",
        ],
        "synthetic_fixture_is_production_authority": False,
    }
    assert placeholder["unsealed_prerequisites"]["runtime_content_lock"]["status"] == (
        "unsealed_exp24_calibration_runtime_content_v1"
    )
    model_prerequisite = placeholder["unsealed_prerequisites"][
        "model_state_authority"
    ]
    assert model_prerequisite["hook_relative_path"] == (
        contract.MODEL_AUTHORITY_HOOK_PATH
    )
    assert model_prerequisite["hook_interface"] == (
        contract.MODEL_AUTHORITY_HOOK_INTERFACE
    )
    assert model_prerequisite["hook_raw_sha256"] is None
    assert placeholder["unsealed_prerequisites"]["scheduler_terminal_census"][
        "terminal_census_sha256"
    ] is None
    boundary = placeholder["unsealed_prerequisites"]["lock_publication_boundary"]
    assert boundary["production_ready"] is False
    assert boundary["production_lock_sha256"] is None
    assert boundary["writer_closure_status"] == (
        "outer_transaction_terminal_no_owner_writer_capabilities_v1"
    )
    hostile = copy.deepcopy(placeholder)
    hostile["production_readiness"]["executable"] = True
    with pytest.raises(contract.CalibrationContractError, match="dependency ledger"):
        seal.validate_unsealed_placeholder(hostile)
    with pytest.raises(contract.CalibrationContractError, match="status is invalid"):
        seal.validate_lock(
            placeholder, contract.load_manifest(), authority,
            _runtime_lock(authority), _model_authority(authority),
            _terminal_census(authority), _launch_authorization(result),
        )
    census = contract.read_json(
        contract.SEED_CENSUS_PLACEHOLDER_PATH, "seed census placeholder"
    )
    assert census["status"] == "unsealed_exact_prior_assignment_collision_census"
    assert census["seeds"] == [244, 245]
    assert census["scope"] == contract.SEED_CENSUS_SCOPE
    assert census["evidence_sha256"] is None


def test_complete_real_torch_twenty_checkpoint_seal_passes(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    result = _result_fixture(tmp_path, authority)
    lock = _validate_result(result, contract.load_manifest(), authority)
    assert lock["status"] == seal.LOCK_SYNTHETIC_STATUS
    assert lock["validation_profile"] == seal.LOCK_SYNTHETIC_PROFILE
    assert lock["production_ready"] is False
    assert len(lock["inventory"]) == 20
    assert len({row["checkpoint_raw_sha256"] for row in lock["inventory"]}) == 20
    assert len({row["run_identity_sha256"] for row in lock["inventory"]}) == 20
    assert set(lock["model_parameter_schema_by_setting"]) == {
        row[0] for row in contract.SETTINGS
    }
    assert lock["fixed_weight_audit"] == {
        "regimes": ["exp24_zero_prefix_exact_5000", "scratch_initialization"],
        "checkpoint_seeds": [244, 245],
        "scratch_seeds": [230, 231],
        "settings": 10,
        "batches_per_setting_regime": 2,
        "expected_rows": 80,
        "device": "cpu_fp32",
        "optimizer_steps": 0,
        "weight_tuple_source": "frozen_exp24_formal_manifest",
        "retuning_allowed": False,
    }
    assert lock["seed_collision_census_sha256"] == authority[
        "seed_collision_census"
    ]["census_sha256"]
    assert _validate_lock(lock, contract.load_manifest(), authority) == lock


def test_result_exclusive_lock_spans_authorization_seal_and_durable_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority(tmp_path)
    result = _result_fixture(tmp_path, authority)
    observed: list[str] = []

    def require_result_lock(phase: str) -> None:
        descriptor = os.open(result, contract.DIRECTORY_FLAGS)
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            os.close(descriptor)
        observed.append(phase)

    real_authorization = controller.validate_launch_authorization
    real_read = seal._read_json
    real_publish = seal._publish_lock_exclusive

    def guarded_authorization(*args: Any, **kwargs: Any) -> Any:
        require_result_lock("authorization")
        return real_authorization(*args, **kwargs)

    def guarded_read(*args: Any, **kwargs: Any) -> Any:
        if "construction" not in observed:
            require_result_lock("construction")
        return real_read(*args, **kwargs)

    monkeypatch.setattr(controller, "validate_launch_authorization", guarded_authorization)
    monkeypatch.setattr(seal, "_read_json", guarded_read)
    lock = _validate_result(result, contract.load_manifest(), authority)
    assert set(observed) >= {"authorization", "construction"}
    descriptor = os.open(result, contract.DIRECTORY_FLAGS)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)

    publication = (
        Path(authority["environment"]["scratch_root"])
        / seal.CHECKPOINT_LOCK_RELATIVE_PATH
    )
    publication.parent.mkdir(parents=True)
    publication.parent.chmod(0o700)
    secure = seal.SecureRoot(result)
    try:
        require_result_lock("before-publication")
        real_publish(publication, lock, authority, secure)
        require_result_lock("after-file-and-parent-fsync")
    finally:
        secure.close()
    before = publication.read_bytes()
    before_mode = stat.S_IMODE(publication.stat().st_mode)
    secure = seal.SecureRoot(result)
    try:
        with pytest.raises(FileExistsError):
            real_publish(publication, lock, authority, secure)
    finally:
        secure.close()
    assert publication.read_bytes() == before
    assert stat.S_IMODE(publication.stat().st_mode) == before_mode == 0o444


def test_lock_publication_rejects_parent_rebind_and_fsyncs_after_fchmod(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority(tmp_path)
    result = _result_fixture(tmp_path, authority)
    lock = _validate_result(result, contract.load_manifest(), authority)
    publication = (
        Path(authority["environment"]["scratch_root"])
        / seal.CHECKPOINT_LOCK_RELATIVE_PATH
    )
    publication.parent.mkdir(parents=True)
    publication.parent.chmod(0o700)
    real_fchmod = seal.os.fchmod
    real_fsync = seal.os.fsync
    published_fd: int | None = None
    fsync_after_fchmod = False

    def record_fchmod(descriptor: int, mode: int) -> None:
        nonlocal published_fd
        real_fchmod(descriptor, mode)
        if mode == 0o444:
            published_fd = descriptor

    def record_fsync(descriptor: int) -> None:
        nonlocal fsync_after_fchmod
        real_fsync(descriptor)
        if descriptor == published_fd:
            fsync_after_fchmod = True

    monkeypatch.setattr(seal.os, "fchmod", record_fchmod)
    monkeypatch.setattr(seal.os, "fsync", record_fsync)
    secure = seal.SecureRoot(result)
    try:
        seal._publish_lock_exclusive(publication, lock, authority, secure)
    finally:
        secure.close()
    assert fsync_after_fchmod is True

    authority = _authority(tmp_path / "rebind")
    result = _result_fixture(tmp_path / "rebind", authority)
    lock = _validate_result(result, contract.load_manifest(), authority)
    assert lock["production_ready"] is False
    assert lock["publication_boundary_authority_sha256"] is None
    publication = (
        Path(authority["environment"]["scratch_root"])
        / seal.CHECKPOINT_LOCK_RELATIVE_PATH
    )
    publication.parent.mkdir(parents=True)
    publication.parent.chmod(0o700)
    detached = publication.parent.with_name(publication.parent.name + "-detached")
    moved = False

    def rebind_after_fchmod(descriptor: int, mode: int) -> None:
        nonlocal moved
        real_fchmod(descriptor, mode)
        if mode == 0o444 and not moved:
            moved = True
            os.rename(publication.parent, detached)
            publication.parent.mkdir(mode=0o700)

    monkeypatch.setattr(seal.os, "fchmod", rebind_after_fchmod)
    monkeypatch.setattr(seal.os, "fsync", real_fsync)
    secure = seal.SecureRoot(result)
    try:
        with pytest.raises(contract.CalibrationContractError, match="publication parent"):
            seal._publish_lock_exclusive(publication, lock, authority, secure)
    finally:
        secure.close()
    assert moved is True
    assert not publication.exists()
    assert not (detached / publication.name).exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "launch-body", "launch-path", "launch-source-sha",
        "path-authority", "creation-receipt",
    ],
)
def test_outer_launch_authorization_rejects_coordinated_rehash(
    tmp_path: Path, mutation: str
) -> None:
    authority = _authority(tmp_path)
    result = _result_fixture(tmp_path, authority)
    bad = copy.deepcopy(_launch_authorization(result))
    row = bad["inventory"][0]
    if mutation == "launch-body":
        row["launch"]["environment"]["TREEWM_STOP_AFTER_UPDATE"] = "5001"
        row["launch"].pop("launch_sha256")
        row["launch"]["launch_sha256"] = contract.stable_hash(row["launch"])
        row["launch_sha256"] = row["launch"]["launch_sha256"]
        row["launch_source_sha256"] = hashlib.sha256(
            controller.launch_source_bytes(row["launch"])
        ).hexdigest()
    elif mutation == "launch-path":
        row["launch_relative_path"] = "launches/../formal.json"
    elif mutation == "launch-source-sha":
        row["launch_source_sha256"] = _digest("forged-source")
    elif mutation == "path-authority":
        forged = _digest("forged-python-authority")
        bad["path_authority"]["python_sha256"] = forged
        for candidate in bad["inventory"]:
            candidate["launch"]["path_authority"]["python_sha256"] = forged
            candidate["launch"].pop("launch_sha256")
            candidate["launch"]["launch_sha256"] = contract.stable_hash(
                candidate["launch"]
            )
            candidate["launch_sha256"] = candidate["launch"]["launch_sha256"]
            candidate["launch_source_sha256"] = hashlib.sha256(
                controller.launch_source_bytes(candidate["launch"])
            ).hexdigest()
    else:
        bad["result_creation_receipt_sha256"] = _digest("forged-creation")
    bad["inventory_sha256"] = contract.stable_hash(bad["inventory"])
    bad.pop("authorization_sha256")
    bad["authorization_sha256"] = contract.stable_hash(bad)
    with pytest.raises(contract.CalibrationContractError):
        controller.validate_launch_authorization(
            bad, contract.load_manifest(), authority, _runtime_lock(authority),
            _model_authority(authority),
        )


def test_seal_rejects_unloadable_checkpoint_and_coordinated_schema_forgery(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path / "unloadable")
    result = _result_fixture(tmp_path / "unloadable", authority)
    checkpoints = result / "sealed-checkpoints"
    checkpoint = checkpoints / "cell-000.pt"
    checkpoints.chmod(0o755)
    checkpoint.chmod(0o644)
    checkpoint.write_bytes(b"not a torch checkpoint")
    checkpoint.chmod(0o444)
    checkpoints.chmod(0o555)
    _edit_receipt(
        result,
        0,
        lambda value: value.update(
            checkpoint_raw_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            checkpoint_size=checkpoint.stat().st_size,
        ),
    )
    with pytest.raises(
        contract.CalibrationContractError,
        match=(
            "cannot inspect sealed|bounded Torch ZIP archive|"
            "no ZIP end-of-directory record"
        ),
    ):
        _validate_result(result, contract.load_manifest(), authority)

    authority = _authority(tmp_path / "schema")
    result = _result_fixture(tmp_path / "schema", authority)
    _edit_receipt(
        result,
        0,
        lambda value: value.update(
            model_parameter_schema_sha256=_digest("forged-schema"),
            model_parameter_tensor_count=999,
            model_parameter_total_numel=999_999,
        ),
    )
    with pytest.raises(
        contract.CalibrationContractError,
        match="checkpoint-derived|pre-output authority",
    ):
        _validate_result(result, contract.load_manifest(), authority)


def test_missing_and_extra_cells_fail_closed(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    result = _result_fixture(tmp_path, authority)
    cells = result / "sealed-cells"
    cells.chmod(0o755)
    (cells / "cell-019.json").unlink()
    cells.chmod(0o555)
    with pytest.raises(contract.CalibrationContractError, match="inventory"):
        _validate_result(result, contract.load_manifest(), authority)

    result.chmod(0o755)
    cells.chmod(0o755)
    _write_json(cells / "cell-999.json", {"status": "extra"}, 0o444)
    cells.chmod(0o555)
    result.chmod(0o555)
    with pytest.raises(contract.CalibrationContractError, match="inventory"):
        _validate_result(result, contract.load_manifest(), authority)


def test_duplicate_cell_and_checkpoint_fail_closed(tmp_path: Path) -> None:
    authority = _authority(tmp_path / "identity")
    result = _result_fixture(tmp_path / "identity", authority)
    receipt0 = json.loads((result / "sealed-cells/cell-000.json").read_text())
    _edit_receipt(
        result,
        1,
        lambda value: value.update(
            cell_index=receipt0["cell_index"],
            setting_id=receipt0["setting_id"],
            env_config=receipt0["env_config"],
            seed=receipt0["seed"],
            run_name=receipt0["run_name"],
            run_identity=receipt0["run_identity"],
            resolved_config=receipt0["resolved_config"],
            resolved_config_contract_sha256=receipt0[
                "resolved_config_contract_sha256"
            ],
        ),
    )
    with pytest.raises(contract.CalibrationContractError, match="differs"):
        _validate_result(result, contract.load_manifest(), authority)

    authority = _authority(tmp_path / "checkpoint")
    result = _result_fixture(tmp_path / "checkpoint", authority)
    checkpoint0 = (result / "sealed-checkpoints/cell-000.pt").read_bytes()
    checkpoints = result / "sealed-checkpoints"
    checkpoints.chmod(0o755)
    target = checkpoints / "cell-001.pt"
    target.chmod(0o644)
    target.write_bytes(checkpoint0)
    target.chmod(0o444)
    checkpoints.chmod(0o555)
    _edit_receipt(
        result,
        1,
        lambda value: value.update(
            checkpoint_raw_sha256=hashlib.sha256(checkpoint0).hexdigest(),
            checkpoint_size=len(checkpoint0),
        ),
    )
    with pytest.raises(contract.CalibrationContractError, match="duplicated|seed differs"):
        _validate_result(result, contract.load_manifest(), authority)


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda value: value.update(completed_updates=4_999), "completed_updates"),
        (lambda value: value.update(completed_updates=5_000.0), "completed_updates"),
        (lambda value: value["recipe"]["prefix_weights"].update(action=0.1), "recipe"),
        (lambda value: value["recipe"]["prefix_weights"].update(action=0), "recipe"),
        (lambda value: value["recipe"]["prefix_graph_enabled"].update(latent=False), "recipe"),
        (
            lambda value: value["outcome_observations"].update(
                periodic_evaluation_count=1
            ),
            "outcome",
        ),
        (
            lambda value: value["outcome_observations"].update(
                visualization_count=1
            ),
            "outcome",
        ),
        (
            lambda value: value["outcome_observations"].update(wandb_mode="online"),
            "outcome",
        ),
        (lambda value: value.update(model_parameter_schema_sha256="bad"), "SHA256"),
        (lambda value: value.update(model_parameter_tensor_count=1.0), "tensor_count"),
        (
            lambda value: value["reuse_policy"].update(formal_resume_allowed=True),
            "formal-arm reuse",
        ),
        (
            lambda value: value["run_identity"].update(arm="formal"),
            "run identity differs",
        ),
        (
            lambda value: value["run_identity"].update(
                run_dir="/formal/hostile", arm="formal"
            ),
            "run identity differs",
        ),
        (
            lambda value: value["run_identity"].update(unexpected="formal"),
            "fields differ",
        ),
        (
            lambda value: value["resolved_config"].update(arm="formal"),
            "config arm",
        ),
        (
            lambda value: value["resolved_config"]["train"].update(
                fixture_full_config_leaf="drift"
            ),
            "config contract",
        ),
    ],
)
def test_receipt_recipe_outcome_schema_type_config_and_formal_reuse_hostiles(
    tmp_path: Path,
    mutation: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    authority = _authority(tmp_path)
    result = _result_fixture(tmp_path, authority)

    def mutate_and_rehash_config(value: dict[str, Any]) -> None:
        mutation(value)
        value["run_identity"]["config_sha256"] = _identity_config_sha256(
            value["resolved_config"]
        )

    _edit_receipt(result, 0, mutate_and_rehash_config)
    with pytest.raises(contract.CalibrationContractError, match=match):
        _validate_result(result, contract.load_manifest(), authority)


def test_per_setting_parameter_schema_must_match_across_seeds(tmp_path: Path) -> None:
    authority = _authority(tmp_path)
    result = _result_fixture(tmp_path, authority)
    _edit_receipt(
        result,
        1,
        lambda value: value.update(
            model_parameter_schema_sha256=_digest("different-schema")
        ),
    )
    with pytest.raises(
        contract.CalibrationContractError,
        match=(
            "schema differs across seeds|checkpoint-derived model_parameter_schema|"
            "pre-output authority"
        ),
    ):
        _validate_result(result, contract.load_manifest(), authority)


def test_path_link_mode_hash_and_source_drift_fail_closed(tmp_path: Path) -> None:
    manifest = contract.load_manifest()
    authority = _authority(tmp_path / "path")
    result = _result_fixture(tmp_path / "path", authority)
    _edit_receipt(
        result, 0, lambda value: value.update(checkpoint_relative_path="../escape.pt")
    )
    with pytest.raises(contract.CalibrationContractError, match="checkpoint_relative_path"):
        _validate_result(result, manifest, authority)

    authority = _authority(tmp_path / "link")
    result = _result_fixture(tmp_path / "link", authority)
    directory = result / "sealed-checkpoints"
    directory.chmod(0o755)
    target = directory / "cell-000.pt"
    target.unlink()
    target.symlink_to("cell-001.pt")
    directory.chmod(0o555)
    with pytest.raises((contract.CalibrationContractError, OSError)):
        _validate_result(result, manifest, authority)

    authority = _authority(tmp_path / "hardlink")
    result = _result_fixture(tmp_path / "hardlink", authority)
    directory = result / "sealed-checkpoints"
    directory.chmod(0o755)
    target = directory / "cell-000.pt"
    target.unlink()
    os.link(directory / "cell-001.pt", target)
    directory.chmod(0o555)
    with pytest.raises(contract.CalibrationContractError, match="link"):
        _validate_result(result, manifest, authority)

    authority = _authority(tmp_path / "mode")
    result = _result_fixture(tmp_path / "mode", authority)
    (result / "sealed-checkpoints/cell-000.pt").chmod(0o644)
    with pytest.raises(contract.CalibrationContractError, match="mode"):
        _validate_result(result, manifest, authority)

    authority = _authority(tmp_path / "hash")
    result = _result_fixture(tmp_path / "hash", authority)
    _edit_receipt(
        result, 0, lambda value: value.update(checkpoint_raw_sha256="0" * 64)
    )
    with pytest.raises(contract.CalibrationContractError, match="bytes/size"):
        _validate_result(result, manifest, authority)

    authority = _authority(tmp_path / "source")
    result = _result_fixture(tmp_path / "source", authority)
    _edit_receipt(
        result,
        0,
        lambda value: value["run_identity"].update(code_sha256="f" * 64),
    )
    with pytest.raises(contract.CalibrationContractError, match="run identity differs"):
        _validate_result(result, manifest, authority)


def test_result_root_formal_escape_and_parent_symlink_fail_closed(tmp_path: Path) -> None:
    authority = _authority(tmp_path / "formal")
    authorized_result = _result_fixture(tmp_path / "formal", authority)
    with pytest.raises(contract.CalibrationContractError, match="authorized"):
        seal.validate_result_root(
            authority["environment"]["formal_output_root"],
            contract.load_manifest(), authority, _runtime_lock(authority),
            _model_authority(authority), _terminal_census(authority),
            _launch_authorization(authorized_result), _regeneration_only=True,
        )

    authority = _authority(tmp_path / "symlink")
    result = _result_fixture(tmp_path / "symlink", authority)
    result.chmod(0o755)
    parent = result.parent
    parent.chmod(0o755)
    real = parent / "real-result"
    os.replace(result, real)
    result.symlink_to(real, target_is_directory=True)
    with pytest.raises((contract.CalibrationContractError, OSError)):
        _validate_result(result, contract.load_manifest(), authority)


def test_terminal_and_adverse_markers_fail_closed(tmp_path: Path) -> None:
    authority = _authority(tmp_path / "terminal")
    result = _result_fixture(tmp_path / "terminal", authority)
    live = result / "live-runs"
    live.chmod(0o755)
    _write_json(live / "COMPLETED.json", {"status": "complete"}, 0o444)
    live.chmod(0o555)
    with pytest.raises(contract.CalibrationContractError, match="terminal/adverse"):
        _validate_result(result, contract.load_manifest(), authority)

    authority = _authority(tmp_path / "adverse")
    result = _result_fixture(tmp_path / "adverse", authority)
    control = result / "control/cell-000"
    control.chmod(0o755)
    _write_json(control / "CANCELLED.json", {"status": "cancelled"}, 0o444)
    control.chmod(0o555)
    with pytest.raises(contract.CalibrationContractError, match="lifecycle inventory"):
        _validate_result(result, contract.load_manifest(), authority)


def _convert_cell_zero_to_continuation(
    result: Path, authority: dict[str, Any]
) -> None:
    cell = contract.expand_cells(contract.load_manifest())[0]
    receipt_path = result / "sealed-cells/cell-000.json"
    receipt_path.chmod(0o644)
    receipt = json.loads(receipt_path.read_text())
    receipt["wave_index"] = 1
    _rehash_receipt(receipt)
    _write_json(receipt_path, receipt, 0o444)
    control = result / "control/cell-000"
    control.chmod(0o755)
    inspected = {
        "checkpoint_raw_sha256": receipt["checkpoint_raw_sha256"],
        "checkpoint_size": receipt["checkpoint_size"],
        "run_identity_sha256": receipt["run_identity_sha256"],
        "model_parameter_schema_sha256": receipt[
            "model_parameter_schema_sha256"
        ],
        "model_parameter_tensor_count": receipt["model_parameter_tensor_count"],
        "model_parameter_total_numel": receipt["model_parameter_total_numel"],
    }
    launch = _marker_launch(authority, cell, receipt["launch_sha256"])
    (control / "wave-0.json").chmod(0o644)
    (control / "wave-1.json").chmod(0o644)
    _write_json(
        control / "wave-0.json", worker._continuation_marker(launch, inspected), 0o444
    )
    _write_json(
        control / "wave-1.json",
        worker._completion_marker(launch, 1, receipt["receipt_sha256"]),
        0o444,
    )
    control.chmod(0o555)


def test_wave_zero_continuation_then_wave_one_completion_is_accepted(
    tmp_path: Path,
) -> None:
    authority = _authority(tmp_path)
    result = _result_fixture(tmp_path, authority)
    _convert_cell_zero_to_continuation(result, authority)
    lock = _validate_result(result, contract.load_manifest(), authority)
    assert lock["inventory"][0]["completion_wave"] == 1


@pytest.mark.parametrize(
    ("field", "hostile"),
    [
        ("checkpoint_raw_sha256", "BAD"),
        ("checkpoint_size", True),
        ("run_identity_sha256", "BAD"),
        ("model_parameter_schema_sha256", "BAD"),
        ("model_parameter_tensor_count", 1.5),
        ("model_parameter_total_numel", False),
    ],
)
def test_continuation_marker_cannot_self_supply_bad_historical_evidence(
    tmp_path: Path, field: str, hostile: object
) -> None:
    authority = _authority(tmp_path)
    result = _result_fixture(tmp_path, authority)
    _convert_cell_zero_to_continuation(result, authority)
    path = result / "control/cell-000/wave-0.json"
    directory = path.parent
    directory.chmod(0o755)
    path.chmod(0o644)
    marker = json.loads(path.read_text(encoding="utf-8"))
    marker[field] = hostile
    marker.pop("marker_sha256")
    marker["marker_sha256"] = contract.stable_hash(marker)
    _write_json(path, marker, 0o444)
    directory.chmod(0o555)
    with pytest.raises(contract.CalibrationContractError):
        _validate_result(result, contract.load_manifest(), authority)


@pytest.mark.parametrize(
    ("field", "hostile"),
    [
        ("checkpoint_raw_sha256", _digest("forged-wave-zero-checkpoint")),
        ("checkpoint_size", 999_999),
        ("run_identity_sha256", _digest("forged-wave-zero-identity")),
        ("model_parameter_schema_sha256", _digest("forged-wave-zero-schema")),
        ("model_parameter_tensor_count", 999),
        ("model_parameter_total_numel", 999_999),
    ],
)
def test_continuation_marker_well_formed_claims_bind_to_final_checkpoint_receipt(
    tmp_path: Path, field: str, hostile: object
) -> None:
    authority = _authority(tmp_path)
    result = _result_fixture(tmp_path, authority)
    _convert_cell_zero_to_continuation(result, authority)
    path = result / "control/cell-000/wave-0.json"
    directory = path.parent
    directory.chmod(0o755)
    path.chmod(0o644)
    marker = json.loads(path.read_text(encoding="utf-8"))
    marker[field] = hostile
    marker.pop("marker_sha256")
    marker["marker_sha256"] = contract.stable_hash(marker)
    _write_json(path, marker, 0o444)
    directory.chmod(0o555)
    with pytest.raises(contract.CalibrationContractError, match="continuation marker"):
        _validate_result(result, contract.load_manifest(), authority)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(completed_updates=4_999),
        lambda value: value.update(completed_updates=5_000.0),
        lambda value: value.update(expected_checkpoints=True),
        lambda value: value["inventory"].pop(),
        lambda value: value["inventory"].append(copy.deepcopy(value["inventory"][0])),
        lambda value: value["inventory"][1].update(
            checkpoint_raw_sha256=value["inventory"][0]["checkpoint_raw_sha256"]
        ),
        lambda value: value["inventory"][1].update(
            run_identity_sha256=value["inventory"][0]["run_identity_sha256"]
        ),
        lambda value: value["inventory"][0].update(cell_index=0.0),
        lambda value: value["inventory"][0].update(env_config="wrong"),
        lambda value: value["inventory"][0].update(
            checkpoint_relative_path="../escape.pt"
        ),
        lambda value: value["inventory"][0].update(checkpoint_raw_sha256="BAD"),
        lambda value: value["inventory"][0].update(completion_wave=True),
        lambda value: value["inventory"][0].update(
            resolved_config_contract_sha256="f" * 64
        ),
        lambda value: value["model_parameter_schema_by_setting"].update(
            scene="f" * 64
        ),
        lambda value: value["fixed_weight_audit"].update(retuning_allowed=0),
        lambda value: value.update(formal_training_or_resume_reuse_allowed=0),
        lambda value: value.update(seed_collision_census_sha256="f" * 64),
        lambda value: value.update(result_root="/formal/escape"),
        lambda value: value["roots"].update(source_sha256="f" * 64),
    ],
)
def test_checkpoint_lock_every_field_type_path_hash_and_authority_mutation_fails(
    tmp_path: Path, mutation: Callable[[dict[str, Any]], None]
) -> None:
    authority = _authority(tmp_path)
    result = _result_fixture(tmp_path, authority)
    authorization = _launch_authorization(result)
    lock = _validate_result(result, contract.load_manifest(), authority)
    bad = copy.deepcopy(lock)
    mutation(bad)
    _rehash_lock(bad)
    with pytest.raises((contract.CalibrationContractError, OSError)):
        seal.validate_lock(
            bad, contract.load_manifest(), authority, _runtime_lock(authority),
            _model_authority(authority), _terminal_census(authority), authorization,
        )


def test_checkpoint_lock_rejects_rehashed_authority_substitution(tmp_path: Path) -> None:
    authority = _authority(tmp_path / "one")
    result = _result_fixture(tmp_path / "one", authority)
    lock = _validate_result(result, contract.load_manifest(), authority)
    other = _authority(tmp_path / "two")
    with pytest.raises(
        contract.CalibrationContractError,
        match="authority|authorized scratch-relative root",
    ):
        seal.validate_lock(
            lock, contract.load_manifest(), other, _runtime_lock(other),
            _model_authority(other), _terminal_census(authority),
            _launch_authorization(result),
        )


@pytest.mark.parametrize(
    "mutation",
    [
        "coordinated-schema",
        "duplicate-receipts",
        "arbitrary-size",
        "arbitrary-tensor-counts",
        "completion-wave",
        "arbitrary-unique-checkpoint-hashes",
        "arbitrary-unique-run-identity-hashes",
    ],
)
def test_rehashed_lock_claims_must_match_regenerated_frozen_evidence(
    tmp_path: Path, mutation: str
) -> None:
    authority = _authority(tmp_path)
    result = _result_fixture(tmp_path, authority)
    lock = _validate_result(result, contract.load_manifest(), authority)
    bad = copy.deepcopy(lock)
    rows = bad["inventory"]
    if mutation == "coordinated-schema":
        hostile = _digest("coordinated-hostile-schema")
        rows[0]["model_parameter_schema_sha256"] = hostile
        rows[1]["model_parameter_schema_sha256"] = hostile
        bad["model_parameter_schema_by_setting"]["scene"] = hostile
    elif mutation == "duplicate-receipts":
        for row in rows:
            row["receipt_sha256"] = _digest("duplicated-hostile-receipt")
    elif mutation == "arbitrary-size":
        rows[0]["checkpoint_size"] += 1
    elif mutation == "arbitrary-tensor-counts":
        rows[0]["model_parameter_tensor_count"] += 1
        rows[0]["model_parameter_total_numel"] += 1
    elif mutation == "completion-wave":
        rows[0]["completion_wave"] = 1
    elif mutation == "arbitrary-unique-checkpoint-hashes":
        for index, row in enumerate(rows):
            row["checkpoint_raw_sha256"] = _digest(f"hostile-checkpoint-{index}")
    else:
        for index, row in enumerate(rows):
            row["run_identity_sha256"] = _digest(f"hostile-identity-{index}")
    _rehash_lock(bad)
    with pytest.raises(
        contract.CalibrationContractError, match="regenerated frozen result evidence"
    ):
        _validate_lock(bad, contract.load_manifest(), authority)


@pytest.mark.parametrize(
    "mutation",
    [
        "top-add", "top-delete", "top-replace",
        "nested-add", "nested-delete", "nested-replace",
    ],
)
def test_secure_root_same_tick_recursive_tree_mutation_fails_closed(
    tmp_path: Path, mutation: str
) -> None:
    root = tmp_path / "root"
    nested = root / "top/nested"
    nested.mkdir(parents=True)
    top_target = root / "top-value.json"
    nested_target = nested / "value.json"
    top_target.write_bytes(b"same bytes")
    nested_target.write_bytes(b"same bytes")
    _freeze_tree(root)
    secure = seal.SecureRoot(root)
    parent = root if mutation.startswith("top-") else nested
    target = top_target if mutation.startswith("top-") else nested_target
    parent.chmod(0o755)
    operation = mutation.split("-", 1)[1]
    if operation == "add":
        (parent / "added").write_bytes(b"added")
        (parent / "added").chmod(0o444)
    elif operation == "delete":
        target.unlink()
    else:
        replacement = tmp_path / "replacement"
        replacement.write_bytes(b"same bytes")
        replacement.chmod(0o444)
        os.replace(replacement, target)
    parent.chmod(0o555)
    with pytest.raises(contract.CalibrationContractError, match="changed"):
        secure.close()


@pytest.mark.parametrize("mutation", ["top-file", "nested-file", "directory-path"])
def test_secure_root_detects_already_scanned_mutation_while_later_file_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root = tmp_path / "root"
    (root / "a").mkdir(parents=True)
    (root / "a/value").write_bytes(b"same bytes")
    (root / "a-top").write_bytes(b"same bytes")
    (root / "z").mkdir()
    (root / "z/trigger").write_bytes(b"later hash trigger")
    _freeze_tree(root)
    secure = seal.SecureRoot(root)
    real_hash = contract.hash_descriptor
    mutated = False

    def hostile_hash(
        descriptor: int, before: os.stat_result, label: str
    ) -> tuple[str, int]:
        nonlocal mutated
        if not mutated and "z/trigger" in label:
            mutated = True
            if mutation == "directory-path":
                replacement = tmp_path / "replacement-a"
                (replacement / "value").parent.mkdir(parents=True)
                (replacement / "value").write_bytes(b"same bytes")
                _freeze_tree(replacement)
                root.chmod(0o755)
                os.rename(root / "a", tmp_path / "parked-a")
                os.rename(replacement, root / "a")
                root.chmod(0o555)
            else:
                target = root / ("a-top" if mutation == "top-file" else "a/value")
                replacement = tmp_path / f"replacement-{mutation}"
                replacement.write_bytes(target.read_bytes())
                replacement.chmod(0o444)
                parent = target.parent
                parent.chmod(0o755)
                os.replace(replacement, target)
                parent.chmod(0o555)
        return real_hash(descriptor, before, label)

    monkeypatch.setattr(contract, "hash_descriptor", hostile_hash)
    with pytest.raises(contract.CalibrationContractError, match="changed"):
        secure.close()
    assert mutated


def _mutate_execution_boundary(
    target: str,
    authority: dict[str, Any],
    launch: dict[str, Any],
    tmp_path: Path,
) -> None:
    snapshot = Path(launch["snapshot_root"])
    result = Path(launch["result_root"])
    if target == "python":
        path = Path(authority["environment"]["python"])
        replacement = tmp_path / "replacement-python"
        replacement.write_bytes(path.read_bytes())
        replacement.chmod(0o755)
        path.parent.chmod(0o755)
        try:
            os.replace(replacement, path)
        finally:
            path.parent.chmod(0o555)
    elif target in ("trainer", "imported-module"):
        path = snapshot / (
            "scripts/train.py" if target == "trainer" else "treewm_import_probe.py"
        )
        replacement = tmp_path / f"replacement-{target}"
        replacement.write_bytes(path.read_bytes())
        replacement.chmod(0o444)
        parent = path.parent
        parent.chmod(0o755)
        os.replace(replacement, path)
        parent.chmod(0o555)
    elif target == "snapshot-root":
        replacement = tmp_path / "replacement-snapshot"
        shutil.copytree(snapshot, replacement, copy_function=shutil.copy2)
        replacement.chmod(0o755)
        snapshot.parent.chmod(0o700)
        os.rename(snapshot, snapshot.parent / "parked-snapshot")
        os.rename(replacement, snapshot)
        snapshot.chmod(0o555)
    else:
        replacement = tmp_path / "replacement-result"
        replacement.mkdir(mode=0o700)
        result.parent.chmod(0o700)
        os.rename(result, result.parent / "parked-result")
        os.rename(replacement, result)


def test_trainer_spawn_uses_retained_procfd_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority(tmp_path)
    launch = _launch(authority)
    observed: dict[str, Any] = {}

    class FinishedProcess:
        def poll(self) -> int:
            return 0

        def send_signal(self, _signum: int) -> None:
            pytest.fail("finished synthetic trainer received a signal")

    def capture(argv: list[str], **kwargs: Any) -> FinishedProcess:
        observed["argv"] = argv
        observed.update(kwargs)
        return FinishedProcess()

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(subprocess, "Popen", capture)
    monkeypatch.setattr(contract, "require_production_runtime", lambda *_args: None)
    monkeypatch.setattr(
        contract, "require_production_model_authority", lambda *_args: None
    )
    with worker.ExecutionAuthority(
        launch, authority, _runtime_lock(authority), _model_authority(authority)
    ) as execution:
        retained = {
            execution.python_fd,
            execution.trainer_fd,
            execution.snapshot.root.fd,
            execution.runtime.root.fd,
            execution.result.fd,
        }
        assert worker._run_trainer(launch, execution) == 0
        assert observed["argv"][0] == execution.python_procfd
        assert observed["argv"][3] == execution.trainer_procfd
        assert observed["cwd"] == execution.snapshot_procfd
        assert set(observed["pass_fds"]) == retained
        assert observed["env"]["PYTHONPATH"].split(os.pathsep)[0] == (
            execution.snapshot_procfd
        )
        assert authority["environment"]["python"] not in observed["argv"]
        assert str(Path(launch["snapshot_root"]) / "scripts/train.py") not in observed["argv"]


@pytest.mark.parametrize(
    "target", ["python", "trainer", "imported-module", "snapshot-root", "result-root"]
)
def test_execution_authority_rejects_pre_spawn_path_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    authority = _authority(tmp_path)
    launch = _launch(authority)
    monkeypatch.setattr(
        subprocess,
        "Popen",
        lambda *_args, **_kwargs: pytest.fail("drift reached child spawn"),
    )
    monkeypatch.setattr(contract, "require_production_runtime", lambda *_args: None)
    monkeypatch.setattr(
        contract, "require_production_model_authority", lambda *_args: None
    )
    with pytest.raises(contract.CalibrationContractError, match="changed|differs"):
        with worker.ExecutionAuthority(
            launch, authority, _runtime_lock(authority), _model_authority(authority)
        ) as execution:
            _mutate_execution_boundary(target, authority, launch, tmp_path)
            worker._run_trainer(launch, execution)
    result = Path(launch["result_root"])
    assert not (result / "sealed-cells").exists()
    assert not (result / "control").exists()


@pytest.mark.parametrize(
    "target", ["python", "trainer", "imported-module", "snapshot-root", "result-root"]
)
def test_execution_authority_rejects_post_exit_replacement_before_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, target: str
) -> None:
    authority = _authority(tmp_path)
    launch = _launch(authority)

    def hostile_return(
        _launch_value: Mapping[str, Any], _execution: worker.ExecutionAuthority
    ) -> int:
        _mutate_execution_boundary(target, authority, launch, tmp_path)
        return worker.GRACEFUL_EXIT_CODE

    monkeypatch.setattr(worker, "_run_trainer", hostile_return)
    monkeypatch.setattr(contract, "require_production_runtime", lambda *_args: None)
    monkeypatch.setattr(
        contract, "require_production_model_authority", lambda *_args: None
    )
    monkeypatch.setattr(
        worker,
        "_exclusive_json",
        lambda *_args, **_kwargs: pytest.fail("drift published receipt/control bytes"),
    )
    with pytest.raises(contract.CalibrationContractError, match="changed|differs"):
        worker.execute_wave(
            launch, authority, _runtime_lock(authority),
            _model_authority(authority), 0,
        )
    result = Path(launch["result_root"])
    assert not (result / "sealed-cells").exists()
    assert not (result / "sealed-checkpoints").exists()
    assert not (result / "control").exists()


def test_seal_detects_post_consumption_receipt_replace_and_adverse_add(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority(tmp_path / "replace")
    result = _result_fixture(tmp_path / "replace", authority)
    real_read = seal._read_json
    replaced = False

    def replace_after_read(
        root: seal.SecureRoot, relative: str, label: str
    ) -> dict[str, Any]:
        nonlocal replaced
        value = real_read(root, relative, label)
        if relative == "sealed-cells/cell-000.json" and not replaced:
            replaced = True
            directory = result / "sealed-cells"
            target = directory / "cell-000.json"
            replacement = tmp_path / "same-receipt"
            replacement.write_bytes(target.read_bytes())
            replacement.chmod(0o444)
            directory.chmod(0o755)
            os.replace(replacement, target)
            directory.chmod(0o555)
        return value

    monkeypatch.setattr(seal, "_read_json", replace_after_read)
    with pytest.raises(contract.CalibrationContractError, match="changed"):
        _validate_result(result, contract.load_manifest(), authority)

    monkeypatch.setattr(seal, "_read_json", real_read)
    authority = _authority(tmp_path / "adverse")
    result = _result_fixture(tmp_path / "adverse", authority)
    real_scan = seal._scan_live_names

    def add_after_scan(root: seal.SecureRoot) -> None:
        real_scan(root)
        live = result / "live-runs"
        live.chmod(0o755)
        _write_json(live / "ADVERSE.json", {"status": "adverse"}, 0o444)
        live.chmod(0o555)

    monkeypatch.setattr(seal, "_scan_live_names", add_after_scan)
    with pytest.raises(contract.CalibrationContractError, match="changed"):
        _validate_result(result, contract.load_manifest(), authority)


def _tree_bytes(root: Path) -> dict[str, tuple[str, bytes | None]]:
    rows: dict[str, tuple[str, bytes | None]] = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            rows[relative] = ("link", os.readlink(path).encode())
        elif path.is_file():
            rows[relative] = ("file", path.read_bytes())
        else:
            rows[relative] = ("dir", None)
    return rows


def test_wave_one_forged_noop_fails_before_trainer_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    authority = _authority(tmp_path)
    launch = _launch(authority)
    result = Path(launch["result_root"])
    cell = contract.expand_cells(contract.load_manifest())[0]
    checkpoint = b"checkpoint"
    receipt = _receipt(authority, cell, checkpoint)
    receipt["launch_sha256"] = launch["launch_sha256"]
    receipt["reuse_policy"]["formal_resume_allowed"] = True
    _rehash_receipt(receipt)
    _write_json(result / "sealed-cells/cell-000.json", receipt, 0o444)
    (result / "sealed-checkpoints").mkdir()
    (result / "sealed-checkpoints/cell-000.pt").write_bytes(checkpoint)
    (result / "sealed-checkpoints/cell-000.pt").chmod(0o444)
    marker_launch = _marker_launch(authority, cell, launch["launch_sha256"])
    previous = worker._completion_marker(
        marker_launch, 0, receipt["receipt_sha256"]
    )
    _write_json(result / "control/cell-000/wave-0.json", previous, 0o444)
    before = _tree_bytes(result)
    monkeypatch.setattr(
        worker,
        "_run_trainer",
        lambda _launch, _execution: pytest.fail("forged no-op reached trainer"),
    )
    monkeypatch.setattr(
        worker,
        "_exclusive_json",
        lambda *_args, **_kwargs: pytest.fail("forged no-op performed a write"),
    )
    monkeypatch.setattr(contract, "require_production_runtime", lambda *_args: None)
    monkeypatch.setattr(
        contract, "require_production_model_authority", lambda *_args: None
    )
    with pytest.raises(contract.CalibrationContractError, match="formal-arm reuse"):
        worker.execute_wave(
            launch, authority, _runtime_lock(authority),
            _model_authority(authority), 1,
        )
    assert _tree_bytes(result) == before


def test_wave_one_forged_continuation_fails_before_trainer_or_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    torch = pytest.importorskip("torch")
    authority = _authority(tmp_path)
    launch = _launch(authority)
    result = Path(launch["result_root"])
    run = Path(launch["run_directory"])
    (run / "checkpoints").mkdir(parents=True)
    checkpoint = run / "checkpoints/latest.pt"
    torch.save(_checkpoint_payload(authority, launch, torch), checkpoint)
    inspected = worker.inspect_checkpoint(checkpoint, launch, authority)
    marker = worker._continuation_marker(launch, inspected)
    marker["checkpoint_relative_path"] = "live-runs/formal/escape/latest.pt"
    marker.pop("marker_sha256")
    marker["marker_sha256"] = contract.stable_hash(marker)
    _write_json(result / "control/cell-000/wave-0.json", marker, 0o444)
    before = _tree_bytes(result)
    monkeypatch.setattr(
        worker,
        "_run_trainer",
        lambda _launch, _execution: pytest.fail("forged continuation reached trainer"),
    )
    monkeypatch.setattr(
        worker,
        "_exclusive_json",
        lambda *_args, **_kwargs: pytest.fail("forged continuation performed a write"),
    )
    monkeypatch.setattr(contract, "require_production_runtime", lambda *_args: None)
    monkeypatch.setattr(
        contract, "require_production_model_authority", lambda *_args: None
    )
    with pytest.raises(contract.CalibrationContractError, match="continuation marker"):
        worker.execute_wave(
            launch, authority, _runtime_lock(authority),
            _model_authority(authority), 1,
        )
    assert _tree_bytes(result) == before
