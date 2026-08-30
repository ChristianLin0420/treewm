#!/usr/bin/env python3
"""Read-only controller and launch constructor for the Exp24 calibration leaf.

The controller emits authenticated per-cell launch JSON and the exact two-wave Slurm
shape.  It never invokes ``sbatch`` itself: the parent Exp24 transaction must snapshot
these bytes, persist a receipt, and release wave zero under its own authority.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import calibration_contract as contract


LAUNCH_FIELDS = {
    "schema_version", "status", "campaign_id", "manifest_file_sha256",
    "authority_sha256", "runtime_lock_sha256", "model_state_authority_sha256",
    "result_creation_receipt_sha256", "roots", "cell", "snapshot_root", "result_root",
    "run_directory", "trainer_repo_relative", "argv", "environment",
    "path_authority", "expected_model_parameter_schema",
    "expected_model_parameter_schema_sha256", "config_override_sha256", "launch_sha256",
}
LAUNCH_AUTHORIZATION_FIELDS = {
    "schema_version", "status", "campaign_id", "manifest_file_sha256",
    "authority_sha256", "runtime_lock_sha256", "model_state_authority_sha256",
    "result_creation_receipt_sha256", "roots", "snapshot_root", "result_root",
    "path_authority", "inventory", "inventory_sha256", "authorization_sha256",
}
LAUNCH_AUTHORIZATION_ROW_FIELDS = {
    "cell_index", "setting_id", "env_config", "seed", "run_name",
    "launch_relative_path", "launch_source_sha256", "launch_sha256",
    "expected_model_parameter_schema_sha256", "launch",
}
PATH_AUTHORITY_FIELDS = {
    "schema_version", "scratch_root", "scratch_root_identity",
    "formal_output_root", "formal_output_root_identity", "snapshot_root",
    "snapshot_root_identity", "snapshot_inventory", "snapshot_inventory_sha256",
    "python_sha256", "trainer_sha256", "result_root", "result_root_identity",
    "result_creation_receipt_sha256", "runtime_lock_sha256",
    "model_state_authority_sha256",
    "snapshot_result_overlap_allowed", "formal_output_membership_allowed",
}
RESULT_CREATION_RECEIPT_PATH = "RESULT_CREATION.json"
RESULT_CREATION_RECEIPT_FIELDS = {
    "schema_version", "status", "campaign_id", "manifest_file_sha256",
    "authority_sha256", "runtime_lock_sha256", "model_state_authority_sha256",
    "scratch_root", "scratch_root_identity", "result_root", "result_relative_path",
    "result_root_initial_identity", "creation_protocol_sha256",
    "receipt_relative_path", "receipt_file_identity",
    "result_creation_receipt_sha256",
}
RESULT_CREATION_PROTOCOL_SHA256 = contract.stable_hash({
    "schema_version": 1,
    "operation": "mkdirat_final_component_exclusive",
    "scratch_boundary": "retained_owner_mode_0700",
    "result_initial_mode": 0o700,
    "receipt_publish": "openat_nofollow_create_exclusive_fsync_fchmod0444_fsync",
    "receipt_identity": "exclusive_created_inode_bound_inside_receipt",
    "overwrite": "forbidden",
})


def _absolute_lexical(path: str | Path, label: str) -> Path:
    value = Path(path)
    contract.require(value.is_absolute(), f"{label} is not absolute")
    normalized = Path(value.as_posix())
    contract.require(".." not in value.parts, f"{label} contains traversal")
    return normalized


def create_result_root_exclusive(
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    model_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the sole writable result boundary without accepting prior contents."""
    contract.validate_manifest(manifest)
    contract.validate_authority(authority, manifest, runtime_lock)
    environment = authority["environment"]
    scratch = _absolute_lexical(environment["scratch_root"], "authorized scratch root")
    relative = contract.safe_relative(
        environment["result_relative_path"], "exclusive result relative path"
    )
    scratch_cap = contract.DirectoryCapability(scratch, "exclusive result scratch root")
    parent = os.dup(scratch_cap.fd)
    parent_components: list[str] = []
    created_final = False
    created_identity: tuple[int, ...] | None = None
    receipt_created = False
    try:
        def require_parent_binding() -> None:
            scratch_cap.require_directory_identity(
                "/".join(parent_components) or ".",
                parent,
                "exclusive result creation parent",
            )

        contract.require(
            contract.directory_identity(scratch_cap.before)
            == environment["scratch_root_identity"],
            "exclusive result scratch identity differs",
        )
        snapshot = scratch / environment["snapshot_relative_path"]
        with contract.RetainedTree(
            snapshot,
            "exclusive result model-authority snapshot",
            directory_mode=0o555,
            file_mode=0o444,
            lock_exclusive=False,
        ) as snapshot_tree:
            contract.validate_model_state_authority(
                model_authority, manifest, authority, runtime_lock,
                snapshot_tree.inventory,
            )
        for component in relative.parts[:-1]:
            require_parent_binding()
            try:
                os.mkdir(component, mode=0o700, dir_fd=parent)
                os.fsync(parent)
            except FileExistsError:
                pass
            child = os.open(component, contract.DIRECTORY_FLAGS, dir_fd=parent)
            child_info = os.fstat(child)
            contract.require(
                stat.S_ISDIR(child_info.st_mode)
                and stat.S_IMODE(child_info.st_mode) == 0o700
                and child_info.st_uid == os.getuid(),
                f"exclusive result parent is not owner-only: {component}",
            )
            os.close(parent)
            parent = child
            parent_components.append(component)
            require_parent_binding()
        require_parent_binding()
        try:
            os.mkdir(relative.name, mode=0o700, dir_fd=parent)
            created_final = True
            created_identity = contract.stat_identity(os.stat(
                relative.name, dir_fd=parent, follow_symlinks=False
            ))
            os.fsync(parent)
            require_parent_binding()
        except FileExistsError as exc:
            raise contract.CalibrationContractError(
                "exclusive calibration result root already exists"
            ) from exc
        result = os.open(relative.name, contract.DIRECTORY_FLAGS, dir_fd=parent)
        try:
            info = os.fstat(result)
            contract.require(
                stat.S_ISDIR(info.st_mode)
                and stat.S_IMODE(info.st_mode) == 0o700
                and info.st_uid == os.getuid()
                and not os.listdir(result),
                "exclusive calibration result root is not an empty owner-only boundary",
            )
            named = os.stat(relative.name, dir_fd=parent, follow_symlinks=False)
            contract.require(contract.stat_identity(named) == contract.stat_identity(info),
                             "exclusive calibration result root raced")
            scratch_cap.require_directory_identity(
                str(relative), result, "exclusive calibration result root"
            )
            result_path = scratch / relative
            receipt_fd = os.open(
                RESULT_CREATION_RECEIPT_PATH,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=result,
            )
            try:
                receipt_created = True
                initial_receipt_info = os.fstat(receipt_fd)
                contract.require(stat.S_ISREG(initial_receipt_info.st_mode)
                                 and initial_receipt_info.st_nlink == 1
                                 and initial_receipt_info.st_uid == os.getuid(),
                                 "exclusive result creation receipt inode differs")
                receipt: dict[str, Any] = {
                    "schema_version": 1,
                    "status": "sealed_exclusive_calibration_result_creation_v1",
                    "campaign_id": contract.CAMPAIGN_ID,
                    "manifest_file_sha256": contract.file_sha256(contract.MANIFEST_PATH),
                    "authority_sha256": authority["authority_sha256"],
                    "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
                    "model_state_authority_sha256": model_authority[
                        "model_state_authority_sha256"
                    ],
                    "scratch_root": str(scratch),
                    "scratch_root_identity": environment["scratch_root_identity"],
                    "result_root": str(result_path),
                    "result_relative_path": str(relative),
                    "result_root_initial_identity": contract.directory_identity(info),
                    "creation_protocol_sha256": RESULT_CREATION_PROTOCOL_SHA256,
                    "receipt_relative_path": RESULT_CREATION_RECEIPT_PATH,
                    "receipt_file_identity": {
                        "device": int(initial_receipt_info.st_dev),
                        "inode": int(initial_receipt_info.st_ino),
                        "mode": 0o444,
                        "uid": int(initial_receipt_info.st_uid),
                        "gid": int(initial_receipt_info.st_gid),
                        "nlink": 1,
                    },
                }
                receipt["result_creation_receipt_sha256"] = contract.stable_hash(receipt)
                payload = json.dumps(
                    receipt, sort_keys=True, indent=2, allow_nan=False
                ).encode("utf-8") + b"\n"
                offset = 0
                while offset < len(payload):
                    count = os.write(receipt_fd, payload[offset:])
                    contract.require(count > 0, "result creation receipt write stopped")
                    offset += count
                os.fsync(receipt_fd)
                scratch_cap.require_directory_identity(
                    str(relative), result, "exclusive calibration result root"
                )
                os.fchmod(receipt_fd, 0o444)
                os.fsync(receipt_fd)
                receipt_info = os.fstat(receipt_fd)
                contract.require(stat.S_IMODE(receipt_info.st_mode) == 0o444
                                 and receipt_info.st_nlink == 1,
                                 "result creation receipt mode/link differs")
                observed_receipt_identity = {
                    "device": int(receipt_info.st_dev),
                    "inode": int(receipt_info.st_ino),
                    "mode": int(stat.S_IMODE(receipt_info.st_mode)),
                    "uid": int(receipt_info.st_uid),
                    "gid": int(receipt_info.st_gid),
                    "nlink": int(receipt_info.st_nlink),
                }
                contract.require(
                    contract.canonical_json(receipt["receipt_file_identity"])
                    == contract.canonical_json(observed_receipt_identity),
                    "result creation receipt inode binding differs",
                )
                contract.require(os.pread(receipt_fd, len(payload) + 1, 0) == payload,
                                 "result creation receipt bytes differ")
                named_receipt = os.stat(
                    RESULT_CREATION_RECEIPT_PATH,
                    dir_fd=result,
                    follow_symlinks=False,
                )
                contract.require(contract.stat_identity(named_receipt)
                                 == contract.stat_identity(receipt_info),
                                 "result creation receipt pathname raced")
                os.fsync(result)
                scratch_cap.require_directory_identity(
                    str(relative), result, "exclusive calibration result root"
                )
                os.fsync(parent)
                require_parent_binding()
                scratch_cap.require_directory_identity(
                    str(relative), result, "exclusive calibration result root"
                )
            finally:
                os.close(receipt_fd)
            return receipt
        finally:
            os.close(result)
    except BaseException:
        if created_final:
            try:
                result_fd = os.open(relative.name, contract.DIRECTORY_FLAGS, dir_fd=parent)
                try:
                    if receipt_created:
                        os.unlink(RESULT_CREATION_RECEIPT_PATH, dir_fd=result_fd)
                        os.fsync(result_fd)
                finally:
                    os.close(result_fd)
                named = os.stat(relative.name, dir_fd=parent, follow_symlinks=False)
                if contract.stat_identity(named) == created_identity:
                    os.rmdir(relative.name, dir_fd=parent)
                    os.fsync(parent)
            except OSError:
                pass
        raise
    finally:
        os.close(parent)
        scratch_cap.close()


def validate_result_creation_receipt(
    result_root: str | Path | contract.DirectoryCapability,
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    model_authority: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind the persistent receipt emitted inside the exclusive result mkdir."""
    owned = not isinstance(result_root, contract.DirectoryCapability)
    root = (contract.DirectoryCapability(result_root, "created calibration result root")
            if owned else result_root)
    try:
        descriptor, before = root.open_regular(
            RESULT_CREATION_RECEIPT_PATH,
            "exclusive result creation receipt",
            mode=0o444,
        )
        try:
            contract.require(before.st_nlink == 1 and before.st_size <= 1024 * 1024,
                             "result creation receipt file authority differs")
            payload = os.pread(descriptor, before.st_size + 1, 0)
            contract.require(len(payload) == before.st_size,
                             "result creation receipt length differs")
            contract.hash_descriptor(
                descriptor, before, "exclusive result creation receipt"
            )
            root.require_named_identity(
                RESULT_CREATION_RECEIPT_PATH, before,
                "exclusive result creation receipt",
            )
        finally:
            os.close(descriptor)
        receipt = contract.parse_json_bytes(payload, "exclusive result creation receipt")
        contract.require_exact_keys(
            receipt, RESULT_CREATION_RECEIPT_FIELDS,
            "exclusive result creation receipt",
        )
        body = dict(receipt)
        claimed = body.pop("result_creation_receipt_sha256", None)
        contract.require_sha256(claimed, "result creation receipt SHA256")
        contract.require(claimed == contract.stable_hash(body),
                         "result creation receipt self-hash differs")
        receipt_identity = contract.require_exact_keys(
            receipt["receipt_file_identity"],
            {"device", "inode", "mode", "uid", "gid", "nlink"},
            "result creation receipt file identity",
        )
        observed_receipt_identity = {
            "device": int(before.st_dev),
            "inode": int(before.st_ino),
            "mode": int(stat.S_IMODE(before.st_mode)),
            "uid": int(before.st_uid),
            "gid": int(before.st_gid),
            "nlink": int(before.st_nlink),
        }
        for name in observed_receipt_identity:
            contract.require_int(
                receipt_identity[name], f"result creation receipt identity {name}",
                minimum=0,
            )
        contract.require(
            contract.canonical_json(receipt_identity)
            == contract.canonical_json(observed_receipt_identity),
            "result creation receipt inode differs",
        )
        result_identity = contract.directory_identity(root.before)
        environment = authority["environment"]
        expected = {
            "schema_version": 1,
            "status": "sealed_exclusive_calibration_result_creation_v1",
            "campaign_id": contract.CAMPAIGN_ID,
            "manifest_file_sha256": contract.file_sha256(contract.MANIFEST_PATH),
            "authority_sha256": authority["authority_sha256"],
            "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
            "model_state_authority_sha256": model_authority[
                "model_state_authority_sha256"
            ],
            "scratch_root": environment["scratch_root"],
            "scratch_root_identity": environment["scratch_root_identity"],
            "result_root": str(root.path),
            "result_relative_path": environment["result_relative_path"],
            "result_root_initial_identity": {
                **result_identity,
                "mode": 0o700,
            },
            "creation_protocol_sha256": RESULT_CREATION_PROTOCOL_SHA256,
            "receipt_relative_path": RESULT_CREATION_RECEIPT_PATH,
            "receipt_file_identity": observed_receipt_identity,
            "result_creation_receipt_sha256": claimed,
        }
        contract.require(contract.canonical_json(receipt)
                         == contract.canonical_json(expected),
                         "result creation receipt differs from live authority")
        contract.require(result_identity["uid"] == os.getuid()
                         and result_identity["mode"] in (0o700, 0o555),
                         "created result root ownership/mode differs")
        return receipt
    finally:
        if owned:
            root.close()


def authorized_path_contract(
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    model_authority: Mapping[str, Any],
    snapshot_root: str | Path,
    result_root: str | Path,
) -> dict[str, Any]:
    """Bind existing snapshot/result directories below the one authorized scratch root."""
    environment = authority["environment"]
    scratch = _absolute_lexical(environment["scratch_root"], "authorized scratch root")
    formal = _absolute_lexical(environment["formal_output_root"], "formal output root")
    snapshot = contract.lexical_descendant(snapshot_root, scratch, "snapshot root")
    result = contract.lexical_descendant(result_root, scratch, "result root")
    contract.require(
        snapshot == scratch / environment["snapshot_relative_path"],
        "snapshot root differs from sealed scratch-relative path",
    )
    contract.require(
        result == scratch / environment["result_relative_path"],
        "result root differs from sealed scratch-relative path",
    )
    contract.require(
        snapshot != result
        and not snapshot.is_relative_to(result)
        and not result.is_relative_to(snapshot),
        "snapshot and result roots overlap",
    )
    for path, label in ((snapshot, "snapshot root"), (result, "result root")):
        contract.require(
            path != formal and not path.is_relative_to(formal),
            f"{label} enters formal output root",
        )
    observed_scratch = contract.nofollow_directory_identity(scratch, "authorized scratch root")
    observed_formal = contract.nofollow_directory_identity(formal, "formal output root")
    contract.require(observed_scratch == environment["scratch_root_identity"],
                     "authorized scratch identity drifted")
    contract.require(observed_scratch["mode"] == 0o700
                     and observed_scratch["uid"] == os.getuid(),
                     "authorized scratch is not an owner-only mode-0700 boundary")
    contract.require(observed_formal == environment["formal_output_root_identity"],
                     "formal output identity drifted")
    snapshot_identity = contract.nofollow_directory_identity(snapshot, "snapshot root")
    result_identity = contract.nofollow_directory_identity(result, "result root")
    contract.require(snapshot_identity != result_identity,
                     "snapshot and result directories alias")
    contract.require(snapshot_identity["mode"] == 0o555
                     and snapshot_identity["uid"] == os.getuid(),
                     "snapshot root is not an owner-controlled frozen boundary")
    contract.require(result_identity["mode"] == 0o700
                     and result_identity["uid"] == os.getuid(),
                     "result root is not an owner-only mode-0700 boundary")
    with contract.RetainedTree(
        snapshot,
        "calibration source snapshot",
        directory_mode=0o555,
        file_mode=0o444,
        lock_exclusive=False,
    ) as snapshot_tree:
        snapshot_inventory = snapshot_tree.inventory
        trainer_entry = snapshot_inventory.get("scripts/train.py")
        contract.require(isinstance(trainer_entry, Mapping)
                         and trainer_entry.get("kind") == "file",
                         "snapshot trainer is absent from frozen inventory")
        trainer_sha256 = trainer_entry["sha256"]
        contract.validate_model_state_authority(
            model_authority, manifest, authority, runtime_lock, snapshot_inventory
        )
    contract.require(trainer_sha256 == environment["trainer_sha256"],
                     "snapshot trainer bytes differ from authority")
    python_sha256 = contract.file_sha256(environment["python"])
    contract.require(python_sha256 == environment["python_sha256"],
                     "pinned calibration Python bytes differ from authority")
    creation_receipt = validate_result_creation_receipt(
        result, manifest, authority, runtime_lock, model_authority
    )
    return {
        "schema_version": 1,
        "scratch_root": str(scratch),
        "scratch_root_identity": observed_scratch,
        "formal_output_root": str(formal),
        "formal_output_root_identity": observed_formal,
        "snapshot_root": str(snapshot),
        "snapshot_root_identity": snapshot_identity,
        "snapshot_inventory": snapshot_inventory,
        "snapshot_inventory_sha256": contract.stable_hash(snapshot_inventory),
        "python_sha256": python_sha256,
        "trainer_sha256": trainer_sha256,
        "result_root": str(result),
        "result_root_identity": result_identity,
        "result_creation_receipt_sha256": creation_receipt[
            "result_creation_receipt_sha256"
        ],
        "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
        "model_state_authority_sha256": model_authority[
            "model_state_authority_sha256"
        ],
        "snapshot_result_overlap_allowed": False,
        "formal_output_membership_allowed": False,
    }


def _construct_launch(
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    model_authority: Mapping[str, Any],
    cell_index: int,
    snapshot_root: str | Path,
    result_root: str | Path,
    path_authority: Mapping[str, Any],
) -> dict[str, Any]:
    snapshot = _absolute_lexical(snapshot_root, "snapshot root")
    result = _absolute_lexical(result_root, "result root")
    cell = contract.expand_cells(manifest)[cell_index]
    run_directory = (
        result / "live-runs" / cell.setting_id / "treewm" / cell.run_name
    )
    overrides = contract.scientific_overrides(
        manifest, authority, cell, result / "live-runs"
    )
    overrides.extend([
        f"hydra.run.dir={run_directory / 'hydra'}",
        "hydra.job.chdir=false",
    ])
    setting = contract.setting_authority(authority, cell.setting_id)
    parameter_schema = contract.setting_model_schema(model_authority, cell.setting_id)
    environment_authority = authority["environment"]
    run_protocol = contract.stable_hash({
        "schema_version": 1,
        "campaign_id": contract.CAMPAIGN_ID,
        "authority_sha256": authority["authority_sha256"],
        "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
        "model_state_authority_sha256": model_authority[
            "model_state_authority_sha256"
        ],
        "result_creation_receipt_sha256": path_authority[
            "result_creation_receipt_sha256"
        ],
        "cell": asdict(cell),
        "roots": authority["roots"],
        "config_overrides": overrides,
    })
    environment = {
        "TREEWM_STOP_AFTER_UPDATE": "5000",
        "TREEWM_PROTOCOL_SHA256": authority["roots"]["protocol_sha256"],
        "TREEWM_CALIBRATION_RUN_PROTOCOL_SHA256": run_protocol,
        "TREEWM_CODE_SHA256": authority["roots"]["source_sha256"],
        "TREEWM_ACTIVE_SOURCE_SHA256": authority["roots"]["source_sha256"],
        "TREEWM_RUNTIME_SHA256": authority["roots"]["runtime_sha256"],
        "TREEWM_RUNTIME_LOCK_SHA256": runtime_lock["runtime_lock_sha256"],
        "TREEWM_MODEL_STATE_AUTHORITY_SHA256": model_authority[
            "model_state_authority_sha256"
        ],
        "TREEWM_EXPECTED_MODEL_PARAMETER_SCHEMA_SHA256": parameter_schema[
            "schema_sha256"
        ],
        "TREEWM_CONFIG_SHA256": authority["roots"]["config_sha256"],
        "TREEWM_DATA_SHA256": setting["data_manifest_sha256"],
        "TREEWM_DATA_CONTRACT_SHA256": setting["input_contract_sha256"],
        "TREEWM_CALIBRATION_SHA256": setting["calibration_sha256"],
        "TREEWM_FUTURE_RECIPE_SHA256": setting["future_recipe_sha256"],
        "TREEWM_RECIPE_CODE_SHA256": setting["recipe_code_sha256"],
        "TREEWM_RECIPE_RUNTIME_SHA256": setting["recipe_runtime_sha256"],
        "TREEWM_DATA_ROOT": environment_authority["data_root"],
        "TREEWM_CACHE": environment_authority["cache_root"],
        "TREEWM_FUTURE_RECIPE_ROOT": str(
            Path(environment_authority["future_recipe_root"]) / cell.setting_id
        ),
        "TREEWM_AUTHORIZED_SCRATCH_ROOT": path_authority["scratch_root"],
        "TREEWM_FORMAL_OUTPUT_ROOT": path_authority["formal_output_root"],
        "TREEWM_EVALUATION_SEED_PROTOCOL_SHA256": authority["roots"]["protocol_sha256"],
        "TREEWM_RUN_NAME": cell.run_name,
        "WANDB_PROJECT": contract.CAMPAIGN_ID,
        "WANDB_RUN_GROUP": "exp24-zero-prefix-calibration",
        "WANDB_RUN_ID": contract.stable_hash({
            "campaign_id": contract.CAMPAIGN_ID,
            "cell_index": cell.index,
            "authority_sha256": authority["authority_sha256"],
        })[:32],
        "WANDB_MODE": "disabled",
        "MUJOCO_GL": environment_authority["MUJOCO_GL"],
        "XLA_PYTHON_CLIENT_PREALLOCATE": environment_authority[
            "XLA_PYTHON_CLIENT_PREALLOCATE"
        ],
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    launch: dict[str, Any] = {
        "schema_version": 1,
        "status": "sealed_zero_prefix_calibration_cell_launch",
        "campaign_id": contract.CAMPAIGN_ID,
        "manifest_file_sha256": contract.file_sha256(contract.MANIFEST_PATH),
        "authority_sha256": authority["authority_sha256"],
        "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
        "model_state_authority_sha256": model_authority[
            "model_state_authority_sha256"
        ],
        "result_creation_receipt_sha256": path_authority[
            "result_creation_receipt_sha256"
        ],
        "roots": dict(authority["roots"]),
        "cell": {**asdict(cell), "run_directory": str(run_directory)},
        "snapshot_root": str(snapshot),
        "result_root": str(result),
        "run_directory": str(run_directory),
        "trainer_repo_relative": "scripts/train.py",
        "argv": [
            environment_authority["python"],
            str(snapshot / "scripts/train.py"),
            *overrides,
        ],
        "environment": environment,
        "path_authority": path_authority,
        "expected_model_parameter_schema": dict(parameter_schema),
        "expected_model_parameter_schema_sha256": parameter_schema["schema_sha256"],
        "config_override_sha256": contract.stable_hash({
            "schema_version": 1, "overrides": overrides
        }),
    }
    launch["launch_sha256"] = contract.stable_hash(launch)
    return launch


def build_launch(
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    model_authority: Mapping[str, Any],
    cell_index: int,
    snapshot_root: str | Path,
    result_root: str | Path,
) -> dict[str, Any]:
    contract.validate_manifest(manifest)
    contract.validate_authority(authority, manifest, runtime_lock)
    contract.require(type(cell_index) is int and 0 <= cell_index < 20,
                     "cell index is outside 0..19")
    snapshot = _absolute_lexical(snapshot_root, "snapshot root")
    result = _absolute_lexical(result_root, "result root")
    path_authority = authorized_path_contract(
        manifest, authority, runtime_lock, model_authority, snapshot, result
    )
    return _construct_launch(
        manifest, authority, runtime_lock, model_authority,
        cell_index, snapshot, result, path_authority
    )


def launch_source_bytes(launch: Mapping[str, Any]) -> bytes:
    """The one exact byte encoding authorized for an outer launch source file."""
    return json.dumps(
        dict(launch), sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"


def validate_launch(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    model_authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    contract.require_exact_keys(value, LAUNCH_FIELDS, "calibration launch")
    contract.require(type(value["schema_version"]) is int and value["schema_version"] == 1,
                     "calibration launch schema differs")
    contract.require(value["status"] == "sealed_zero_prefix_calibration_cell_launch",
                     "calibration launch status differs")
    contract.require(value["campaign_id"] == contract.CAMPAIGN_ID,
                     "calibration launch campaign differs")
    contract.require(value["authority_sha256"] == authority["authority_sha256"],
                     "calibration launch authority differs")
    contract.require(value["runtime_lock_sha256"] == runtime_lock["runtime_lock_sha256"]
                     and value["model_state_authority_sha256"]
                     == model_authority["model_state_authority_sha256"],
                     "calibration launch prerequisite authority differs")
    contract.require(value["roots"] == authority["roots"],
                     "calibration launch authority roots differ")
    contract.require_exact_keys(value["path_authority"], PATH_AUTHORITY_FIELDS,
                                "calibration launch path authority")
    contract.require_int(value["path_authority"]["schema_version"],
                         "calibration path authority schema")
    contract.require_bool(value["path_authority"]["snapshot_result_overlap_allowed"],
                          "calibration path overlap policy")
    contract.require_bool(value["path_authority"]["formal_output_membership_allowed"],
                          "calibration formal-output policy")
    contract.require(value["path_authority"]["snapshot_result_overlap_allowed"] is False
                     and value["path_authority"]["formal_output_membership_allowed"] is False,
                     "calibration launch path policy permits overlap")
    contract.require_sha256(
        value["path_authority"]["snapshot_inventory_sha256"],
        "calibration snapshot inventory SHA256",
    )
    contract.require(
        value["path_authority"]["snapshot_inventory_sha256"]
        == contract.stable_hash(value["path_authority"]["snapshot_inventory"]),
        "calibration snapshot inventory hash differs",
    )
    for name in ("python_sha256", "trainer_sha256"):
        contract.require_sha256(
            value["path_authority"][name], f"calibration path authority {name}"
        )
    body = dict(value)
    claimed = body.pop("launch_sha256", None)
    contract.require_sha256(claimed, "calibration launch SHA256")
    contract.require(claimed == contract.stable_hash(body),
                     "calibration launch self-hash differs")
    cell_value = value["cell"]
    contract.require(isinstance(cell_value, Mapping), "calibration launch cell is invalid")
    index = contract.require_int(cell_value.get("index"), "calibration launch cell index")
    expected = build_launch(
        manifest,
        authority,
        runtime_lock,
        model_authority,
        index,
        value["snapshot_root"],
        value["result_root"],
    )
    contract.require(contract.canonical_json(value) == contract.canonical_json(expected),
                     "calibration launch differs from reconstruction")
    return value


def _validate_frozen_path_authority(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    model_authority: Mapping[str, Any],
) -> None:
    """Re-derive every path claim after the result boundary has been frozen."""
    path_authority = contract.require_exact_keys(
        value, PATH_AUTHORITY_FIELDS, "frozen calibration path authority"
    )
    contract.require_int(path_authority["schema_version"], "frozen path schema")
    contract.require(path_authority["schema_version"] == 1,
                     "frozen path schema differs")
    for name, expected in (
        ("snapshot_result_overlap_allowed", False),
        ("formal_output_membership_allowed", False),
    ):
        contract.require_bool(path_authority[name], f"frozen path policy {name}")
        contract.require(path_authority[name] is expected,
                         f"frozen path policy {name} differs")

    environment = authority["environment"]
    scratch = _absolute_lexical(environment["scratch_root"], "frozen scratch root")
    formal = _absolute_lexical(environment["formal_output_root"], "frozen formal root")
    snapshot = scratch / environment["snapshot_relative_path"]
    result = scratch / environment["result_relative_path"]
    for name, expected in (
        ("scratch_root", str(scratch)),
        ("formal_output_root", str(formal)),
        ("snapshot_root", str(snapshot)),
        ("result_root", str(result)),
    ):
        contract.require(type(path_authority[name]) is str
                         and path_authority[name] == expected,
                         f"frozen path authority {name} differs")
    contract.require(
        snapshot != result
        and not snapshot.is_relative_to(result)
        and not result.is_relative_to(snapshot)
        and result != formal
        and not result.is_relative_to(formal),
        "frozen path authority enters an unauthorized namespace",
    )

    observed_scratch = contract.nofollow_directory_identity(
        scratch, "frozen launch scratch root"
    )
    observed_formal = contract.nofollow_directory_identity(
        formal, "frozen launch formal root"
    )
    contract.require(
        contract.canonical_json(path_authority["scratch_root_identity"])
        == contract.canonical_json(environment["scratch_root_identity"])
        == contract.canonical_json(observed_scratch),
        "frozen launch scratch identity differs",
    )
    contract.require(
        contract.canonical_json(path_authority["formal_output_root_identity"])
        == contract.canonical_json(environment["formal_output_root_identity"])
        == contract.canonical_json(observed_formal),
        "frozen launch formal identity differs",
    )
    contract.require(observed_scratch["mode"] == 0o700
                     and observed_scratch["uid"] == os.getuid(),
                     "frozen launch scratch boundary is not owner-only")

    with contract.RetainedTree(
        snapshot,
        "frozen calibration source snapshot",
        directory_mode=0o555,
        file_mode=0o444,
        lock_exclusive=False,
        expected_inventory=path_authority["snapshot_inventory"],
    ) as snapshot_tree:
        observed_snapshot_identity = contract.directory_identity(
            snapshot_tree.root.before
        )
        observed_inventory = snapshot_tree.inventory
        trainer = observed_inventory.get("scripts/train.py")
        contract.require(isinstance(trainer, Mapping)
                         and trainer.get("kind") == "file",
                         "frozen snapshot trainer is absent")
        trainer_sha256 = trainer["sha256"]
        contract.validate_model_state_authority(
            model_authority, manifest, authority, runtime_lock, observed_inventory
        )
    contract.require(
        contract.canonical_json(path_authority["snapshot_root_identity"])
        == contract.canonical_json(observed_snapshot_identity),
        "frozen launch snapshot identity differs",
    )
    contract.require_sha256(
        path_authority["snapshot_inventory_sha256"],
        "frozen launch snapshot inventory SHA256",
    )
    contract.require(
        path_authority["snapshot_inventory_sha256"]
        == contract.stable_hash(observed_inventory),
        "frozen launch snapshot inventory hash differs",
    )
    python_sha256 = contract.file_sha256(environment["python"])
    for name, observed, expected in (
        ("python_sha256", python_sha256, environment["python_sha256"]),
        ("trainer_sha256", trainer_sha256, environment["trainer_sha256"]),
    ):
        contract.require_sha256(path_authority[name], f"frozen launch {name}")
        contract.require(path_authority[name] == observed == expected,
                         f"frozen launch {name} differs")

    initial_result = contract.require_exact_keys(
        path_authority["result_root_identity"],
        {"device", "inode", "mode", "uid", "gid"},
        "frozen launch initial result identity",
    )
    for name in ("device", "inode", "mode", "uid", "gid"):
        contract.require_int(initial_result[name],
                             f"frozen launch result identity {name}", minimum=0)
    current_result = contract.nofollow_directory_identity(
        result, "frozen launch result root"
    )
    contract.require(initial_result["mode"] == 0o700
                     and initial_result["uid"] == os.getuid(),
                     "launch result was not initially owner-only mode 0700")
    contract.require(current_result["mode"] == 0o555
                     and current_result["uid"] == os.getuid(),
                     "launch result is not frozen mode 0555")
    contract.require(
        all(initial_result[name] == current_result[name]
            for name in ("device", "inode", "uid", "gid")),
        "frozen launch result root identity differs",
    )
    creation_receipt = validate_result_creation_receipt(
        result, manifest, authority, runtime_lock, model_authority
    )
    contract.require(
        path_authority["result_creation_receipt_sha256"]
        == creation_receipt["result_creation_receipt_sha256"]
        and path_authority["runtime_lock_sha256"] == runtime_lock["runtime_lock_sha256"]
        and path_authority["model_state_authority_sha256"]
        == model_authority["model_state_authority_sha256"],
        "frozen launch prerequisite bindings differ",
    )


def _validate_static_launch(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    model_authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate launch semantics without reopening a now-frozen result pathname."""
    contract.require_exact_keys(value, LAUNCH_FIELDS, "authorized calibration launch")
    path_authority = contract.require_exact_keys(
        value["path_authority"], PATH_AUTHORITY_FIELDS,
        "authorized calibration launch path authority",
    )
    contract.require_int(value["schema_version"], "authorized launch schema")
    contract.require(value["schema_version"] == 1
                     and value["status"] == "sealed_zero_prefix_calibration_cell_launch"
                     and value["campaign_id"] == contract.CAMPAIGN_ID,
                     "authorized launch header differs")
    contract.require(value["manifest_file_sha256"] == contract.file_sha256(contract.MANIFEST_PATH),
                     "authorized launch manifest differs")
    contract.require(value["authority_sha256"] == authority["authority_sha256"]
                     and contract.canonical_json(value["roots"])
                     == contract.canonical_json(authority["roots"]),
                     "authorized launch authority differs")
    body = dict(value)
    claimed = body.pop("launch_sha256", None)
    contract.require_sha256(claimed, "authorized launch SHA256")
    contract.require(claimed == contract.stable_hash(body),
                     "authorized launch self-hash differs")
    cell_value = contract.require_exact_keys(
        value["cell"],
        {"index", "setting_index", "seed_index", "setting_id", "env_config", "seed",
         "run_name", "run_directory"},
        "authorized launch cell",
    )
    index = contract.require_int(cell_value["index"], "authorized launch cell index")
    contract.require(0 <= index < 20, "authorized launch cell index differs")
    _validate_frozen_path_authority(
        path_authority, manifest, authority, runtime_lock, model_authority
    )
    contract.require(
        type(value["snapshot_root"]) is str
        and value["snapshot_root"] == path_authority["snapshot_root"]
        and type(value["result_root"]) is str
        and value["result_root"] == path_authority["result_root"],
        "authorized launch roots differ from frozen path authority",
    )
    expected = _construct_launch(
        manifest,
        authority,
        runtime_lock,
        model_authority,
        index,
        value["snapshot_root"],
        value["result_root"],
        path_authority,
    )
    contract.require(contract.canonical_json(value) == contract.canonical_json(expected),
                     "authorized launch differs from static reconstruction")
    return value


def build_launch_authorization(
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    model_authority: Mapping[str, Any],
    launches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Seal all exact launch sources after workers are terminal and result is frozen."""
    contract.validate_manifest(manifest)
    contract.validate_authority(authority, manifest, runtime_lock)
    contract.require(type(launches) in (list, tuple) and len(launches) == 20,
                     "launch authorization does not contain exactly 20 launches")
    rows: list[dict[str, Any]] = []
    common_path_authority: Mapping[str, Any] | None = None
    for cell, launch in zip(contract.expand_cells(manifest), launches, strict=True):
        _validate_static_launch(
            launch, manifest, authority, runtime_lock, model_authority
        )
        contract.require(launch["cell"]["index"] == cell.index,
                         f"launch authorization cell order differs: {cell.index}")
        if common_path_authority is None:
            common_path_authority = launch["path_authority"]
        contract.require(
            contract.canonical_json(launch["path_authority"])
            == contract.canonical_json(common_path_authority),
            f"launch authorization path authority differs: {cell.index}",
        )
        source = launch_source_bytes(launch)
        rows.append({
            "cell_index": cell.index,
            "setting_id": cell.setting_id,
            "env_config": cell.env_config,
            "seed": cell.seed,
            "run_name": cell.run_name,
            "launch_relative_path": f"launches/cell-{cell.index:03d}.json",
            "launch_source_sha256": hashlib.sha256(source).hexdigest(),
            "launch_sha256": launch["launch_sha256"],
            "expected_model_parameter_schema_sha256": launch[
                "expected_model_parameter_schema_sha256"
            ],
            "launch": dict(launch),
        })
    assert common_path_authority is not None
    environment = authority["environment"]
    scratch_identity = contract.nofollow_directory_identity(
        environment["scratch_root"], "launch authorization scratch root"
    )
    result_identity = contract.nofollow_directory_identity(
        common_path_authority["result_root"], "launch authorization frozen result root"
    )
    contract.require(scratch_identity["mode"] == 0o700
                     and scratch_identity["uid"] == os.getuid(),
                     "launch authorization scratch boundary is not owner-only")
    contract.require(result_identity["mode"] == 0o555
                     and result_identity["uid"] == os.getuid(),
                     "launch authorization result boundary is not frozen")
    initial_result = common_path_authority["result_root_identity"]
    contract.require(
        all(result_identity[name] == initial_result[name]
            for name in ("device", "inode", "uid", "gid"))
        and initial_result["mode"] == 0o700,
        "launch authorization result root is not the exclusively created launch root",
    )
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": "sealed_outer_exact_calibration_launch_authorization",
        "campaign_id": contract.CAMPAIGN_ID,
        "manifest_file_sha256": contract.file_sha256(contract.MANIFEST_PATH),
        "authority_sha256": authority["authority_sha256"],
        "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
        "model_state_authority_sha256": model_authority[
            "model_state_authority_sha256"
        ],
        "result_creation_receipt_sha256": common_path_authority[
            "result_creation_receipt_sha256"
        ],
        "roots": dict(authority["roots"]),
        "snapshot_root": common_path_authority["snapshot_root"],
        "result_root": common_path_authority["result_root"],
        "path_authority": dict(common_path_authority),
        "inventory": rows,
        "inventory_sha256": contract.stable_hash(rows),
    }
    value["authorization_sha256"] = contract.stable_hash(value)
    return value


def validate_launch_authorization(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    model_authority: Mapping[str, Any],
) -> list[Mapping[str, Any]]:
    contract.require_exact_keys(value, LAUNCH_AUTHORIZATION_FIELDS,
                                "launch authorization")
    body = dict(value)
    claimed = body.pop("authorization_sha256", None)
    contract.require_sha256(claimed, "launch authorization SHA256")
    contract.require(claimed == contract.stable_hash(body),
                     "launch authorization self-hash differs")
    inventory = value["inventory"]
    contract.require(isinstance(inventory, list) and len(inventory) == 20,
                     "launch authorization inventory differs")
    launches: list[Mapping[str, Any]] = []
    for row, cell in zip(inventory, contract.expand_cells(manifest), strict=True):
        contract.require_exact_keys(row, LAUNCH_AUTHORIZATION_ROW_FIELDS,
                                    f"launch authorization cell {cell.index}")
        expected_scalars = {
            "cell_index": cell.index, "setting_id": cell.setting_id,
            "env_config": cell.env_config, "seed": cell.seed,
            "run_name": cell.run_name,
            "launch_relative_path": f"launches/cell-{cell.index:03d}.json",
        }
        for name, expected in expected_scalars.items():
            contract.require(type(row[name]) is type(expected) and row[name] == expected,
                             f"launch authorization cell {cell.index} {name} differs")
        _validate_static_launch(
            row["launch"], manifest, authority, runtime_lock, model_authority
        )
        source_sha = hashlib.sha256(launch_source_bytes(row["launch"])).hexdigest()
        contract.require_sha256(row["launch_source_sha256"],
                                f"launch authorization cell {cell.index} source SHA256")
        contract.require_sha256(row["launch_sha256"],
                                f"launch authorization cell {cell.index} launch SHA256")
        contract.require(row["launch_source_sha256"] == source_sha
                         and row["launch_sha256"] == row["launch"]["launch_sha256"],
                         f"launch authorization cell {cell.index} source/hash differs")
        expected_schema = contract.setting_model_schema(
            model_authority, cell.setting_id
        )["schema_sha256"]
        contract.require_sha256(
            row["expected_model_parameter_schema_sha256"],
            f"launch authorization cell {cell.index} model schema SHA256",
        )
        contract.require(
            row["expected_model_parameter_schema_sha256"] == expected_schema
            == row["launch"]["expected_model_parameter_schema_sha256"],
            f"launch authorization cell {cell.index} model schema differs",
        )
        launches.append(row["launch"])
    contract.require(value["inventory_sha256"] == contract.stable_hash(inventory),
                     "launch authorization inventory hash differs")
    expected = build_launch_authorization(
        manifest, authority, runtime_lock, model_authority, launches
    )
    contract.require(contract.canonical_json(value) == contract.canonical_json(expected),
                     "launch authorization differs from frozen reconstruction")
    return list(inventory)


def validate_authorized_launch(
    launch: Mapping[str, Any],
    launch_relative_path: str,
    launch_source_sha256: str,
    authorization: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    model_authority: Mapping[str, Any],
) -> Mapping[str, Any]:
    rows = validate_launch_authorization(
        authorization, manifest, authority, runtime_lock, model_authority
    )
    validated = _validate_static_launch(
        launch, manifest, authority, runtime_lock, model_authority
    )
    index = launch["cell"]["index"]
    row = rows[index]
    contract.require(launch_relative_path == row["launch_relative_path"],
                     "launch source path differs from outer authorization")
    contract.require_sha256(launch_source_sha256, "launch source file SHA256")
    contract.require(launch_source_sha256 == row["launch_source_sha256"],
                     "launch source bytes differ from outer authorization")
    contract.require(contract.canonical_json(validated)
                     == contract.canonical_json(row["launch"]),
                     "launch differs from outer authorization")
    return validated


def topology_description() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "authenticated_outer_submission_adapter_required",
        "wave0": {
            "array": "0-19%20",
            "held_until_durable_receipt_and_authorization": True,
            "signal": "B:USR1@420",
            "walltime": "04:00:00",
            "requeue": False,
        },
        "wave1": {
            "array": "0-19%20",
            "dependency": "afterok:<exact-whole-wave0-array-job-id>",
            "kill_on_invalid_dependency": True,
            "requeue": False,
        },
        "report": {
            "dependency": "afterok:<exact-whole-wave1-array-job-id>",
            "action": "read_only_calibration_seal_validation",
        },
        "adaptive_or_third_wave": False,
        "scheduler_invocation_implemented_here": False,
        "exclusive_result_initialization_implemented_here": True,
        "reason": "parent_Exp24_transaction_must_snapshot_and_authenticate_before_release",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--describe", action="store_true")
    modes.add_argument("--test-only", action="store_true")
    modes.add_argument("--validate-authority", action="store_true")
    modes.add_argument("--initialize-result-root", action="store_true")
    modes.add_argument("--emit-launch", action="store_true")
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--runtime-lock", type=Path)
    parser.add_argument("--model-state-authority", type=Path)
    parser.add_argument("--cell-index", type=int)
    parser.add_argument("--snapshot-root", type=Path)
    parser.add_argument("--result-root", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = contract.load_manifest()
    if args.describe:
        print(json.dumps({
            **contract.describe(manifest),
            "submission_topology": topology_description(),
        }, sort_keys=True, indent=2, allow_nan=False))
        return 0
    if args.test_only:
        print(json.dumps({
            "schema_version": 1,
            "status": "test_only_passed_no_persistent_writes",
            "cells": [contract.cell_description(cell) for cell in contract.expand_cells(manifest)],
            "submission_topology": topology_description(),
            "scheduler_calls_performed": False,
            "source_or_output_scan_performed": False,
        }, sort_keys=True, indent=2, allow_nan=False))
        return 0
    contract.require(args.authority is not None and args.runtime_lock is not None,
                     "--authority and --runtime-lock are required")
    authority = contract.read_json(args.authority, "calibration authority")
    runtime_lock = contract.read_json(args.runtime_lock, "calibration runtime lock")
    contract.validate_authority(authority, manifest, runtime_lock)
    if args.validate_authority:
        print(json.dumps({
            "schema_version": 1,
            "status": "calibration_authority_valid",
            "authority_sha256": authority["authority_sha256"],
            "persistent_writes_performed": False,
        }, sort_keys=True, indent=2, allow_nan=False))
        return 0
    if args.initialize_result_root:
        contract.require(args.model_state_authority is not None,
                         "--model-state-authority is required")
        model_authority = contract.read_json(
            args.model_state_authority, "calibration model-state authority"
        )
        receipt = create_result_root_exclusive(
            manifest, authority, runtime_lock, model_authority
        )
        print(json.dumps({
            "schema_version": 1,
            "status": "exclusive_calibration_result_root_initialized",
            "result_creation_receipt_sha256": receipt[
                "result_creation_receipt_sha256"
            ],
        }, sort_keys=True, indent=2, allow_nan=False))
        return 0
    contract.require(args.model_state_authority is not None,
                     "--model-state-authority is required")
    model_authority = contract.read_json(
        args.model_state_authority, "calibration model-state authority"
    )
    contract.require(args.cell_index is not None, "--cell-index is required")
    contract.require(args.snapshot_root is not None, "--snapshot-root is required")
    contract.require(args.result_root is not None, "--result-root is required")
    launch = build_launch(
        manifest, authority, runtime_lock, model_authority,
        args.cell_index, args.snapshot_root, args.result_root
    )
    print(json.dumps(launch, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except contract.CalibrationContractError as exc:
        print(f"EXP24_CALIBRATION_CONTROLLER_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
