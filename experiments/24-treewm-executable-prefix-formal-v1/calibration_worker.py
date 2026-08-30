#!/usr/bin/env python3
"""Two-wave worker for the scratch-only Exp24 zero-prefix calibration.

Wave zero either exports an exact update-5000 checkpoint or publishes a continuation
receipt after the trainer's graceful exit 75.  Wave one authenticates that receipt and
resumes the same cell, while already-complete cells perform an authenticated no-op.
"""

from __future__ import annotations

import argparse
import copy
from contextlib import AbstractContextManager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import select
import signal
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import calibration_contract as contract
import calibration_controller as controller


GRACEFUL_EXIT_CODE = 75
COMPLETE_STATUS = "calibration_checkpoint_complete"
CONTINUATION_STATUS = "continuation_ready"
NOOP_STATUS = "authenticated_completion_noop"
RECEIPT_FIELDS = {
    "schema_version", "status", "campaign_id", "authority_sha256",
    "runtime_lock_sha256", "model_state_authority_sha256",
    "result_creation_receipt_sha256", "roots",
    "cell_index", "setting_id", "env_config", "seed", "run_name", "wave_index",
    "run_identity", "run_identity_sha256", "completed_updates",
    "nominal_optimizer_updates", "scheduler_total_steps", "checkpoint_relative_path",
    "checkpoint_raw_sha256", "checkpoint_size", "model_parameter_schema",
    "model_parameter_schema_sha256",
    "model_parameter_tensor_count", "model_parameter_total_numel", "recipe",
    "resolved_config", "resolved_config_contract_sha256",
    "outcome_observations", "reuse_policy", "launch_sha256", "receipt_sha256",
}
SOURCE_CHECKPOINT_MODE = 0o644
RUN_IDENTITY_FIELDS = {
    "schema_version", "objective_version", "run_dir", "run_name", "arm",
    "env_name", "setting", "dataset_kind", "source_name", "seed", "total_steps",
    "scheduler_total_steps", "world_size", "model_class", "scorer", "node_budget",
    "branch_factor", "gradient_checkpointing", "future_set_cache", "shared_cache",
    "retrieval_enabled", "retrieval_num_keys", "task_ids", "final_episodes_per_task",
    "evaluation_seed_protocol_sha256", "evaluation_seed_tables_sha256",
    "monitor_seed_table_sha256", "final_seed_table_sha256", "config_sha256",
    "protocol_sha256", "code_sha256", "runtime_sha256", "data_manifest_sha256",
    "calibration_sha256", "future_recipe_sha256", "recipe_anchor_policy",
    "train_anchor_count", "validation_anchor_count", "recipe_code_sha256",
    "recipe_runtime_sha256", "campaign_source_sha256", "campaign_protocol_sha256",
    "campaign_config_sha256", "campaign_input_contract_sha256",
    "campaign_factorial_arm", "campaign_prerequisite_binding_sha256",
    "campaign_selected_recipe_sha256", "wandb_project", "wandb_entity",
    "wandb_group", "wandb_mode", "wandb_id",
}
MARKER_PROVENANCE_FIELDS = {
    "schema_version", "status", "campaign_id", "authority_sha256",
    "runtime_lock_sha256", "model_state_authority_sha256",
    "result_creation_receipt_sha256", "roots",
    "cell_index", "setting_id", "env_config", "seed", "run_name", "wave_index",
    "launch_sha256",
}
COMPLETION_MARKER_FIELDS = MARKER_PROVENANCE_FIELDS | {"receipt_sha256", "marker_sha256"}
CONTINUATION_MARKER_FIELDS = MARKER_PROVENANCE_FIELDS | {
    "checkpoint_relative_path", "checkpoint_raw_sha256", "checkpoint_size",
    "run_identity_sha256", "model_parameter_schema_sha256",
    "model_parameter_tensor_count", "model_parameter_total_numel", "marker_sha256",
}


class RetainedRegularFile(AbstractContextManager["RetainedRegularFile"]):
    """A no-follow executable capability retained through child termination."""

    def __init__(self, path: Path, label: str, expected_sha256: str, *, mode: int) -> None:
        self.path = Path(os.path.normpath(str(path.absolute())))
        self.label = label
        self.parent = contract.DirectoryCapability(self.path.parent, f"{label} parent")
        self.fd: int | None = None
        self._closed = False
        try:
            self.fd, self.before = self.parent.open_regular(
                self.path.name, label, mode=mode
            )
            self.expected_sha256 = contract.require_sha256(
                expected_sha256, f"{label} expected SHA256"
            )
            self.verify()
        except BaseException:
            if self.fd is not None:
                os.close(self.fd)
            self.parent.close()
            raise

    @property
    def procfd(self) -> str:
        assert self.fd is not None
        return f"/proc/self/fd/{self.fd}"

    def verify(self) -> None:
        assert self.fd is not None
        digest, _size = contract.hash_descriptor(self.fd, self.before, self.label)
        contract.require(digest == self.expected_sha256, f"{self.label} bytes differ")
        self.parent.require_named_identity(self.path.name, self.before, self.label)

    def close(self) -> None:
        if not self._closed:
            try:
                self.verify()
            finally:
                assert self.fd is not None
                os.close(self.fd)
                self.parent.close()
                self._closed = True

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        elif not self._closed:
            os.close(self.fd)
            self.parent.close()
            self._closed = True


def _scan_mutable_result(root: contract.DirectoryCapability) -> dict[str, Any]:
    """Take one exact no-follow inventory of the live scratch result tree."""
    inventory: dict[str, Any] = {}

    def scan(descriptor: int, relative: PurePosixPath) -> None:
        before = os.fstat(descriptor)
        contract.require(stat.S_ISDIR(before.st_mode) and before.st_uid == os.getuid(),
                         f"result path is not an owned directory: {relative}")
        contract.require(stat.S_IMODE(before.st_mode) in (0o700, 0o755, 0o555),
                         f"result directory mode differs: {relative}")
        names = tuple(sorted(os.listdir(descriptor)))
        key = "." if str(relative) == "." else str(relative)
        inventory[key] = {
            "kind": "directory",
            "identity": list(contract.stat_identity(before)),
            "names": list(names),
        }
        for name in names:
            named = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            child_relative = PurePosixPath(name) if key == "." else relative / name
            child_key = str(child_relative)
            if stat.S_ISDIR(named.st_mode):
                child = os.open(name, contract.DIRECTORY_FLAGS, dir_fd=descriptor)
                try:
                    opened = os.fstat(child)
                    contract.require(contract.stat_identity(opened)
                                     == contract.stat_identity(named),
                                     f"result directory raced: {child_relative}")
                    scan(child, child_relative)
                    contract.require(
                        contract.stat_identity(os.stat(
                            name, dir_fd=descriptor, follow_symlinks=False
                        )) == contract.stat_identity(opened),
                        f"result directory changed: {child_relative}",
                    )
                finally:
                    os.close(child)
            else:
                contract.require(
                    stat.S_ISREG(named.st_mode) and named.st_nlink == 1
                    and named.st_uid == os.getuid()
                    and stat.S_IMODE(named.st_mode) in (0o600, 0o644, 0o444),
                    f"result file type/mode/link/owner differs: {child_relative}",
                )
                source = os.open(name, contract.READ_FLAGS, dir_fd=descriptor)
                try:
                    opened = os.fstat(source)
                    contract.require(contract.stat_identity(opened)
                                     == contract.stat_identity(named),
                                     f"result file raced: {child_relative}")
                    digest, size = contract.hash_descriptor(
                        source, opened, f"result file {child_relative}"
                    )
                    contract.require(
                        contract.stat_identity(os.stat(
                            name, dir_fd=descriptor, follow_symlinks=False
                        )) == contract.stat_identity(opened),
                        f"result file changed: {child_relative}",
                    )
                    inventory[child_key] = {
                        "kind": "file",
                        "identity": list(contract.stat_identity(opened)),
                        "sha256": digest,
                        "size": size,
                    }
                finally:
                    os.close(source)
        contract.require(tuple(sorted(os.listdir(descriptor))) == names
                         and contract.stat_identity(os.fstat(descriptor))
                         == contract.stat_identity(before),
                         f"result directory changed during scan: {relative}")

    descriptor = os.dup(root.fd)
    try:
        scan(descriptor, PurePosixPath("."))
    finally:
        os.close(descriptor)
    root.require_path_identity()
    return inventory


class ExecutionAuthority(AbstractContextManager["ExecutionAuthority"]):
    """Retain all executable/source/result authority until publication completes."""

    def __init__(
        self,
        launch: Mapping[str, Any],
        authority: Mapping[str, Any],
        runtime_lock: Mapping[str, Any],
        model_authority: Mapping[str, Any],
    ) -> None:
        self.launch = launch
        self.authority = authority
        self.runtime_lock = runtime_lock
        self.model_authority = model_authority
        path_authority = launch["path_authority"]
        try:
            self.runtime = contract.RetainedTree(
                runtime_lock["runtime_root"],
                "authenticated calibration runtime",
                directory_mode=0o555,
                file_mode=(0o444, 0o555),
                lock_exclusive=False,
                expected_inventory=runtime_lock["runtime_inventory"],
            )
            self.python_fd, self.python_before = self.runtime.duplicate_file(
                runtime_lock["interpreter"]["relative_path"]
            )
            python_sha, _python_size = contract.hash_descriptor(
                self.python_fd, self.python_before, "pinned calibration Python"
            )
            contract.require(python_sha == path_authority["python_sha256"]
                             == runtime_lock["interpreter"]["sha256"],
                             "pinned calibration Python differs from runtime lock")
            self.snapshot = contract.RetainedTree(
                launch["snapshot_root"],
                "authenticated calibration source snapshot",
                directory_mode=0o555,
                file_mode=0o444,
                lock_exclusive=False,
                expected_inventory=path_authority["snapshot_inventory"],
            )
            self.trainer_fd, self.trainer_before = self.snapshot.duplicate_file(
                launch["trainer_repo_relative"]
            )
            self.result = contract.DirectoryCapability(
                launch["result_root"], "retained calibration result root"
            )
            fcntl.flock(self.result.fd, fcntl.LOCK_SH | fcntl.LOCK_NB)
            contract.require(
                contract.directory_identity(self.result.before)
                == path_authority["result_root_identity"],
                "retained result root identity differs from launch",
            )
            contract.require(stat.S_IMODE(self.result.before.st_mode) == 0o700
                             and self.result.before.st_uid == os.getuid(),
                             "retained result root is not owner-only mode 0700")
            self.decoder = contract.RuntimeCheckpointDecoder(
                runtime_lock, contract.load_manifest(), allow_synthetic=True
            )
            self._closed = False
            self.verify_boundary("pre-spawn")
        except BaseException:
            if hasattr(self, "result"):
                self.result.close()
            if hasattr(self, "decoder"):
                self.decoder.__exit__(Exception, Exception(), None)
            if hasattr(self, "trainer_fd"):
                os.close(self.trainer_fd)
            if hasattr(self, "snapshot"):
                self.snapshot._release(verify=False)
            if hasattr(self, "python_fd"):
                os.close(self.python_fd)
            if hasattr(self, "runtime"):
                self.runtime._release(verify=False)
            raise

    @property
    def python_procfd(self) -> str:
        return f"/proc/self/fd/{self.python_fd}"

    @property
    def trainer_procfd(self) -> str:
        return f"/proc/self/fd/{self.trainer_fd}"

    @property
    def snapshot_procfd(self) -> str:
        return f"/proc/self/fd/{self.snapshot.root.fd}"

    def verify_boundary(self, phase: str) -> None:
        python_sha, _python_size = contract.hash_descriptor(
            self.python_fd, self.python_before, "pinned calibration Python"
        )
        contract.require(python_sha == self.runtime_lock["interpreter"]["sha256"],
                         f"runtime interpreter bytes changed at {phase}")
        self.runtime.verify_two_scans()
        contract.require(
            contract.stat_identity(os.fstat(self.trainer_fd))
            == contract.stat_identity(self.trainer_before),
            f"snapshot trainer descriptor changed at {phase}",
        )
        self.snapshot.verify_two_scans()
        self.result.require_path_identity()
        first = _scan_mutable_result(self.result)
        second = _scan_mutable_result(self.result)
        contract.require(contract.canonical_json(first) == contract.canonical_json(second),
                         f"result tree is not quiescent at {phase}")

    def close(self) -> None:
        if self._closed:
            return
        error: BaseException | None = None
        try:
            self.verify_boundary("authority-release")
        except BaseException as exc:
            error = exc
        try:
            fcntl.flock(self.result.fd, fcntl.LOCK_UN)
        finally:
            self.result.close()
            self.decoder.close()
            os.close(self.trainer_fd)
            os.close(self.python_fd)
            try:
                self.snapshot.close()
            finally:
                self.runtime.close()
            self._closed = True
        if error is not None:
            raise error

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            try:
                fcntl.flock(self.result.fd, fcntl.LOCK_UN)
            finally:
                self.result.close()
                self.decoder.__exit__(exc_type, exc, traceback)
                os.close(self.trainer_fd)
                os.close(self.python_fd)
                self.snapshot._release(verify=False)
                self.runtime._release(verify=False)
                self._closed = True


def _open_or_create_directory(parent: int, name: str, mode: int = 0o755) -> int:
    try:
        os.mkdir(name, mode=mode, dir_fd=parent)
    except FileExistsError:
        pass
    named = os.stat(name, dir_fd=parent, follow_symlinks=False)
    contract.require(
        stat.S_ISDIR(named.st_mode)
        and named.st_uid == os.getuid()
        and stat.S_IMODE(named.st_mode) in (0o700, 0o755),
        f"calibration publication directory authority differs: {name}",
    )
    descriptor = os.open(name, contract.DIRECTORY_FLAGS, dir_fd=parent)
    opened = os.fstat(descriptor)
    contract.require(
        contract.stat_identity(opened) == contract.stat_identity(named),
        f"calibration publication directory raced: {name}",
    )
    return descriptor


def _directory_chain(root: contract.DirectoryCapability, relative: PurePosixPath) -> int:
    descriptor = os.dup(root.fd)
    try:
        for component in (() if str(relative) in ("", ".") else relative.parts):
            next_descriptor = _open_or_create_directory(descriptor, component)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _exclusive_json(
    result_root: Path | contract.DirectoryCapability,
    relative: PurePosixPath | str,
    value: Mapping[str, Any],
) -> None:
    path = contract.safe_relative(str(relative), "exclusive JSON path")
    payload = json.dumps(
        dict(value), sort_keys=True, indent=2, allow_nan=False
    ).encode("utf-8") + b"\n"
    owned = not isinstance(result_root, contract.DirectoryCapability)
    root = (contract.DirectoryCapability(result_root, "calibration result root")
            if owned else result_root)
    try:
        parent = _directory_chain(root, path.parent)
        descriptor: int | None = None
        created = False
        try:
            root.require_directory_identity(
                path.parent, parent, f"exclusive JSON parent {path.parent}"
            )
            descriptor = os.open(
                path.name,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent,
            )
            created = True
            offset = 0
            while offset < len(payload):
                written = os.write(descriptor, payload[offset:])
                contract.require(written > 0, f"exclusive JSON write stopped: {path}")
                offset += written
            os.fsync(descriptor)
            root.require_directory_identity(
                path.parent, parent, f"exclusive JSON parent {path.parent}"
            )
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            written_info = os.fstat(descriptor)
            contract.require(os.pread(descriptor, len(payload) + 1, 0) == payload,
                             f"exclusive JSON bytes differ after write: {path}")
            named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            contract.require(
                contract.stat_identity(written_info) == contract.stat_identity(named),
                f"exclusive JSON pathname changed: {path}",
            )
            root.require_directory_identity(
                path.parent, parent, f"exclusive JSON parent {path.parent}"
            )
            os.fsync(parent)
            root.require_directory_identity(
                path.parent, parent, f"exclusive JSON parent {path.parent}"
            )
        except BaseException:
            if created and descriptor is not None:
                try:
                    named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                    if (contract.stat_identity(named)
                            == contract.stat_identity(os.fstat(descriptor))):
                        os.unlink(path.name, dir_fd=parent)
                        os.fsync(parent)
                except OSError:
                    pass
            raise
        finally:
            if descriptor is not None:
                os.close(descriptor)
            os.close(parent)
    finally:
        if owned:
            root.close()


class RetainedCheckpoint(AbstractContextManager["RetainedCheckpoint"]):
    """One no-follow checkpoint descriptor retained through load, hash, and publish."""

    def __init__(
        self,
        path: Path,
        *,
        mode: int = SOURCE_CHECKPOINT_MODE,
        decoder: Any | None = None,
    ) -> None:
        contract.require(path.is_absolute(), "checkpoint path is not absolute")
        self.path = Path(os.path.normpath(str(path)))
        self.parent = contract.DirectoryCapability(self.path.parent, "checkpoint parent")
        try:
            self.fd, self.before = self.parent.open_regular(
                self.path.name, "calibration checkpoint", mode=mode
            )
        except BaseException:
            self.parent.close()
            raise
        contract.require(0 < self.before.st_size <= 16 * 1024**3,
                         "checkpoint size is outside the sealed bound")
        self._closed = False
        self._unlinked = False
        self.safe_load_evidence: dict[str, Any] | None = None
        self.decoder = decoder

    def verify(self, *, require_name: bool = True) -> None:
        contract.require(
            contract.stat_identity(os.fstat(self.fd)) == contract.stat_identity(self.before),
            "checkpoint descriptor identity changed",
        )
        if require_name and not self._unlinked:
            self.parent.require_named_identity(
                self.path.name, self.before, "calibration checkpoint"
            )

    def load(self) -> Mapping[str, Any]:
        payload, evidence = contract.safe_load_checkpoint_fd(
            self.fd,
            self.before,
            "calibration checkpoint",
            verify=self.verify,
            decoder=self.decoder,
        )
        self.safe_load_evidence = evidence
        return payload

    def sha256(self) -> tuple[str, int]:
        digest, size = contract.hash_descriptor(
            self.fd, self.before, "calibration checkpoint"
        )
        self.verify()
        return digest, size

    def publish_exclusive(
        self,
        result_root: Path | contract.DirectoryCapability,
        relative: PurePosixPath | str,
    ) -> tuple[str, int]:
        path = contract.safe_relative(str(relative), "sealed checkpoint path")
        self.verify()
        owned = not isinstance(result_root, contract.DirectoryCapability)
        root = (contract.DirectoryCapability(result_root, "calibration result root")
                if owned else result_root)
        try:
            parent = _directory_chain(root, path.parent)
            destination: int | None = None
            created = False
            try:
                root.require_directory_identity(
                    path.parent, parent, f"checkpoint publication parent {path.parent}"
                )
                destination = os.open(
                    path.name,
                    os.O_RDWR | os.O_CREAT | os.O_EXCL
                    | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=parent,
                )
                created = True
                offset = 0
                while offset < self.before.st_size:
                    block = os.pread(
                        self.fd, min(4 * 1024 * 1024, self.before.st_size - offset), offset
                    )
                    contract.require(bool(block), "checkpoint ended during exclusive publish")
                    written = 0
                    while written < len(block):
                        count = os.write(destination, block[written:])
                        contract.require(count > 0, "checkpoint publish write stopped")
                        written += count
                    offset += len(block)
                os.fsync(destination)
                root.require_directory_identity(
                    path.parent, parent, f"checkpoint publication parent {path.parent}"
                )
                os.fchmod(destination, 0o444)
                os.fsync(destination)
                destination_before = os.fstat(destination)
                digest, size = contract.hash_descriptor(
                    destination, destination_before, "published calibration checkpoint"
                )
                named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                contract.require(
                    contract.stat_identity(named) == contract.stat_identity(destination_before),
                    "published checkpoint pathname changed",
                )
                source_digest, source_size = self.sha256()
                contract.require(
                    (digest, size) == (source_digest, source_size),
                    "published checkpoint differs from retained source descriptor",
                )
                self.verify()
                final_named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
                contract.require(
                    contract.stat_identity(final_named)
                    == contract.stat_identity(destination_before),
                    "published checkpoint changed after postcheck",
                )
                root.require_directory_identity(
                    path.parent, parent, f"checkpoint publication parent {path.parent}"
                )
                os.fsync(parent)
                root.require_directory_identity(
                    path.parent, parent, f"checkpoint publication parent {path.parent}"
                )
                return digest, size
            except BaseException:
                if created and destination is not None:
                    try:
                        named = os.stat(
                            path.name, dir_fd=parent, follow_symlinks=False
                        )
                        if (contract.stat_identity(named)
                                == contract.stat_identity(os.fstat(destination))):
                            os.unlink(path.name, dir_fd=parent)
                            os.fsync(parent)
                    except OSError:
                        pass
                raise
            finally:
                if destination is not None:
                    os.close(destination)
                os.close(parent)
        finally:
            if owned:
                root.close()

    def unlink_verified(self) -> None:
        """Retire the authenticated source only after export receipts are durable."""
        self.verify()
        os.unlink(self.path.name, dir_fd=self.parent.fd)
        self._unlinked = True
        self.before = os.fstat(self.fd)
        contract.require(self.before.st_nlink == 0,
                         "exported source checkpoint remains linked")

    def close(self) -> None:
        if not self._closed:
            try:
                self.verify(require_name=not self._unlinked)
            finally:
                os.close(self.fd)
                self.parent.close()
                self._closed = True

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _hash_open_file(path: Path, *, mode: int = SOURCE_CHECKPOINT_MODE) -> tuple[str, int]:
    with RetainedCheckpoint(path, mode=mode) as checkpoint:
        return checkpoint.sha256()


def _hash_result_file(
    root: contract.DirectoryCapability,
    relative: PurePosixPath | str,
    *,
    mode: int,
) -> tuple[str, int]:
    path = contract.safe_relative(str(relative), "retained result file path")
    descriptor, before = root.open_regular(
        path, f"retained result file {path}", mode=mode
    )
    try:
        digest, size = contract.hash_descriptor(
            descriptor, before, f"retained result file {path}"
        )
        root.require_named_identity(path, before, f"retained result file {path}")
        return digest, size
    finally:
        os.close(descriptor)


def _read_result_json(
    root: contract.DirectoryCapability,
    relative: PurePosixPath | str,
    label: str,
    *,
    mode: int = 0o444,
) -> dict[str, Any]:
    path = contract.safe_relative(str(relative), f"{label} path")
    descriptor, before = root.open_regular(path, label, mode=mode)
    try:
        contract.require(before.st_size <= 16 * 1024 * 1024,
                         f"{label} exceeds the read bound")
        payload = bytearray()
        offset = 0
        while offset < before.st_size:
            block = os.pread(
                descriptor, min(1024 * 1024, before.st_size - offset), offset
            )
            contract.require(bool(block), f"{label} ended during read")
            payload.extend(block)
            offset += len(block)
        contract.require(not os.pread(descriptor, 1, before.st_size),
                         f"{label} grew during read")
        contract.require(
            contract.stat_identity(os.fstat(descriptor))
            == contract.stat_identity(before),
            f"{label} changed during read",
        )
        root.require_named_identity(path, before, label)
        return contract.parse_json_bytes(bytes(payload), label)
    finally:
        os.close(descriptor)


def _tensor_schema_evidence(model: object) -> dict[str, Any]:
    contract.require(type(model) is dict and bool(model),
                     "checkpoint model state is absent")
    rows: list[dict[str, Any]] = []
    total = 0
    for name, tensor in sorted(model.items()):
        contract.require(type(name) is str and bool(name),
                         "checkpoint model parameter name is invalid")
        descriptor = contract.require_exact_keys(
            tensor,
            {contract.SAFE_TENSOR_MARKER, "shape", "dtype", "numel",
             "storage_bytes", "content_sha256", "device", "layout",
             "storage_alias_policy"},
            f"checkpoint model parameter {name}",
        )
        contract.require_int(descriptor[contract.SAFE_TENSOR_MARKER],
                             f"checkpoint model parameter marker {name}")
        contract.require(descriptor[contract.SAFE_TENSOR_MARKER] == 1,
                         f"checkpoint model parameter marker differs: {name}")
        shape = descriptor["shape"]
        contract.require(type(shape) is list and bool(shape)
                         and all(type(value) is int and value > 0 for value in shape),
                         f"checkpoint model parameter shape is invalid: {name}")
        dtype = contract.require_string(
            descriptor["dtype"], f"checkpoint model parameter dtype {name}"
        )
        numel = contract.require_int(
            descriptor["numel"], f"checkpoint model parameter numel {name}", minimum=1
        )
        storage_bytes = contract.require_int(
            descriptor["storage_bytes"],
            f"checkpoint model parameter storage bytes {name}", minimum=1,
        )
        contract.require_sha256(
            descriptor["content_sha256"],
            f"checkpoint model parameter content SHA256 {name}",
        )
        contract.require(descriptor["device"] == "cpu"
                         and descriptor["layout"] == "strided"
                         and descriptor["storage_alias_policy"] == "unique_exact_storage",
                         f"checkpoint model parameter materialization differs: {name}")
        product = 1
        for size in shape:
            product *= size
        contract.require(product == numel,
                         f"checkpoint model parameter shape/numel differs: {name}")
        total += numel
        rows.append({
            "name": name,
            "shape": list(shape),
            "dtype": dtype,
            "numel": numel,
            "storage_bytes": storage_bytes,
            "device": "cpu",
            "layout": "strided",
            "storage_alias_policy": "unique_exact_storage",
        })
    contract.require(rows and total > 0, "checkpoint model parameter schema is empty")
    schema: dict[str, Any] = {
        "schema_version": 1,
        "model_class": "TreeWM",
        "parameters": rows,
        "parameter_count": len(rows),
        "total_numel": total,
        "total_storage_bytes": sum(row["storage_bytes"] for row in rows),
    }
    schema["schema_sha256"] = contract.stable_hash(schema)
    contract.validate_model_parameter_schema(schema, "checkpoint model parameter schema")
    return schema


def _tensor_schema(model: object) -> tuple[str, int, int]:
    schema = _tensor_schema_evidence(model)
    return (
        schema["schema_sha256"],
        schema["parameter_count"],
        schema["total_numel"],
    )


def _protocol_seed(
    protocol_sha256: str, split: str, task_id: int, episode_index: int, nonce: int
) -> int:
    digest = hashlib.sha256(contract.canonical_json({
        "protocol_sha256": protocol_sha256,
        "split": split,
        "task_id": task_id,
        "episode_index": episode_index,
        "nonce": nonce,
    })).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def _evaluation_seed_tables(protocol_sha256: str, training_seed: int) -> dict[str, Any]:
    contract.require_sha256(protocol_sha256, "evaluation seed protocol SHA256")
    contract.require_int(training_seed, "evaluation seed training seed", minimum=0)
    used: set[int] = set()
    tables: dict[str, Any] = {}
    for split in ("monitor", "final"):
        rows: list[list[int]] = []
        for task_id in contract.EVALUATION_TASK_IDS:
            nonce = 0
            value = _protocol_seed(protocol_sha256, split, task_id, 0, nonce)
            while value in used:
                nonce += 1
                value = _protocol_seed(protocol_sha256, split, task_id, 0, nonce)
            used.add(value)
            rows.append([value])
        table: dict[str, Any] = {
            "schema_version": 1,
            "split": split,
            "protocol_sha256": protocol_sha256,
            "task_ids": list(contract.EVALUATION_TASK_IDS),
            "episodes_per_task": 1,
            "seeds": rows,
        }
        table["sha256"] = contract.stable_hash(table)
        tables[split] = table
    bundle: dict[str, Any] = {
        "schema_version": 1,
        "protocol_sha256": protocol_sha256,
        "training_seed": training_seed,
        "monitor": tables["monitor"],
        "final": tables["final"],
    }
    bundle["sha256"] = contract.stable_hash(bundle)
    return bundle


def _expected_run_identity(
    authority: Mapping[str, Any],
    cell: Mapping[str, Any] | contract.CalibrationCell,
    result_root: str | Path,
    config_sha256: str,
) -> dict[str, Any]:
    index = cell.index if isinstance(cell, contract.CalibrationCell) else cell["index"]
    setting_id = cell.setting_id if isinstance(cell, contract.CalibrationCell) else cell["setting_id"]
    seed = cell.seed if isinstance(cell, contract.CalibrationCell) else cell["seed"]
    run_name = cell.run_name if isinstance(cell, contract.CalibrationCell) else cell["run_name"]
    setting = contract.setting_authority(authority, setting_id)
    tables = _evaluation_seed_tables(authority["roots"]["protocol_sha256"], seed)
    run_dir = Path(result_root) / "live-runs" / setting_id / "treewm" / run_name
    wandb_id = contract.stable_hash({
        "campaign_id": contract.CAMPAIGN_ID,
        "cell_index": index,
        "authority_sha256": authority["authority_sha256"],
    })[:32]
    return {
        "schema_version": 1,
        "objective_version": contract.OBJECTIVE,
        "run_dir": str(run_dir),
        "run_name": run_name,
        "arm": "treewm",
        "env_name": setting["env_name"],
        "setting": setting_id,
        "dataset_kind": setting["dataset_kind"],
        "source_name": setting["source_name"],
        "seed": seed,
        "total_steps": 25_000,
        "scheduler_total_steps": 1_000_000,
        "world_size": 1,
        "model_class": "TreeWM",
        "scorer": "learned",
        "node_budget": 64,
        "branch_factor": 4,
        "gradient_checkpointing": True,
        "future_set_cache": False,
        "shared_cache": True,
        "retrieval_enabled": False,
        "retrieval_num_keys": 0,
        "task_ids": list(contract.EVALUATION_TASK_IDS),
        "final_episodes_per_task": 1,
        "evaluation_seed_protocol_sha256": authority["roots"]["protocol_sha256"],
        "evaluation_seed_tables_sha256": tables["sha256"],
        "monitor_seed_table_sha256": tables["monitor"]["sha256"],
        "final_seed_table_sha256": tables["final"]["sha256"],
        "config_sha256": config_sha256,
        "protocol_sha256": authority["roots"]["protocol_sha256"],
        "code_sha256": authority["roots"]["source_sha256"],
        "runtime_sha256": authority["roots"]["runtime_sha256"],
        "data_manifest_sha256": setting["data_manifest_sha256"],
        "calibration_sha256": setting["calibration_sha256"],
        "future_recipe_sha256": setting["future_recipe_sha256"],
        "recipe_anchor_policy": "published_union",
        "train_anchor_count": setting["published_union_train_anchors"],
        "validation_anchor_count": setting["published_union_validation_anchors"],
        "recipe_code_sha256": setting["recipe_code_sha256"],
        "recipe_runtime_sha256": setting["recipe_runtime_sha256"],
        "campaign_source_sha256": authority["roots"]["source_sha256"],
        "campaign_protocol_sha256": authority["roots"]["protocol_sha256"],
        "campaign_config_sha256": authority["roots"]["config_sha256"],
        "campaign_input_contract_sha256": setting["input_contract_sha256"],
        "campaign_factorial_arm": "zero_prefix_calibration",
        "campaign_prerequisite_binding_sha256": authority["authority_sha256"],
        "campaign_selected_recipe_sha256": setting["future_recipe_sha256"],
        "wandb_project": contract.CAMPAIGN_ID,
        "wandb_entity": "",
        "wandb_group": "exp24-zero-prefix-calibration",
        "wandb_mode": "disabled",
        "wandb_id": wandb_id,
    }


def _resolved_config_contract_sha256(config: Mapping[str, Any]) -> str:
    """Hash the complete config while normalizing its two authority self-references."""
    identity_config = copy.deepcopy(dict(config))
    identity_config["resume"] = None
    identity_config["campaign_config_sha256"] = "<CONFIG_ROOT>"
    identity_config["campaign_prerequisite_binding_sha256"] = "<AUTHORITY>"
    return contract.stable_hash(identity_config)


def _validate_checkpoint_config(
    config: Mapping[str, Any], launch: Mapping[str, Any], authority: Mapping[str, Any]
) -> tuple[str, str]:
    cell = launch["cell"]
    setting = contract.setting_authority(authority, cell["setting_id"])
    exact_top = {
        "seed": cell["seed"],
        "arm": "treewm",
        "objective_version": contract.OBJECTIVE,
        "run_root": str(Path(launch["result_root"]) / "live-runs"),
        "run_name": None,
        "resume": "auto",
        "campaign_id": contract.CAMPAIGN_ID,
        "campaign_source_sha256": authority["roots"]["source_sha256"],
        "campaign_protocol_sha256": authority["roots"]["protocol_sha256"],
        "campaign_config_sha256": authority["roots"]["config_sha256"],
        "campaign_input_contract_sha256": setting["input_contract_sha256"],
        "campaign_factorial_arm": "zero_prefix_calibration",
        "campaign_prerequisite_binding_sha256": authority["authority_sha256"],
        "campaign_selected_recipe_sha256": setting["future_recipe_sha256"],
    }
    for name, expected in exact_top.items():
        observed = config.get(name)
        contract.require(type(observed) is type(expected) and observed == expected,
                         f"checkpoint config {name} differs")
    sections = {
        name: config.get(name)
        for name in ("train", "losses", "future_sets", "planner", "eval", "env", "model", "tree", "retrieval")
    }
    contract.require(all(isinstance(value, Mapping) for value in sections.values()),
                     "checkpoint config sections are absent")
    train = sections["train"]
    expected_train = {
        "steps": 25_000, "scheduler_total_steps": 1_000_000,
        "ckpt_every": 5_000, "val_every": 25_000, "diag_every": 5_000,
        "log_every": 50, "eval_every": 25_000, "viz_every": 25_000,
        "viz_every_early": 25_000, "viz_early_until": 0,
        "max_train_anchors": setting["published_union_train_anchors"],
        "max_val_anchors": setting["published_union_validation_anchors"],
        "validation_sample_seed": 1701, "gradient_checkpointing": True,
    }
    for name, expected in expected_train.items():
        observed = train.get(name)
        contract.require(type(observed) is type(expected) and observed == expected,
                         f"checkpoint train.{name} differs")
    env = sections["env"]
    for name, expected in {
        "short_name": cell["setting_id"], "name": setting["env_name"],
        "source_name": setting["source_name"], "dataset_kind": setting["dataset_kind"],
    }.items():
        contract.require(type(env.get(name)) is str and env[name] == expected,
                         f"checkpoint env.{name} differs")
    losses = sections["losses"]
    enabled = losses.get("enabled")
    weights = losses.get("weights")
    contract.require(isinstance(enabled, Mapping) and isinstance(weights, Mapping),
                     "checkpoint executable-prefix config is absent")
    for name in ("executable_prefix_action", "executable_prefix_latent", "executable_prefix_endpoint"):
        contract.require(type(enabled.get(name)) is bool and enabled[name] is True,
                         f"checkpoint {name} graph is disabled")
        contract.require(type(weights.get(name)) is float and weights[name] == 0.0,
                         f"checkpoint {name} weight is not exact float zero")
    for name in ("executable_action_lower_bound", "executable_action_upper_bound"):
        expected = setting["action_lower_bound" if name.endswith("lower_bound") else "action_upper_bound"]
        contract.require(type(losses.get(name)) is float and losses[name] == expected,
                         f"checkpoint losses.{name} differs")
    future = sections["future_sets"]
    for name, expected in {
        "executable_prefix_steps": 4, "recipe_anchor_policy": "published_union",
        "cache": False, "shared_cache": True,
    }.items():
        observed = future.get(name)
        contract.require(type(observed) is type(expected) and observed == expected,
                         f"checkpoint future_sets.{name} differs")
    planner = sections["planner"]
    for name, expected in {
        "action_lower_bound": setting["action_lower_bound"],
        "action_upper_bound": setting["action_upper_bound"],
        "max_env_steps": setting["max_environment_steps"],
        "execute_steps": 4, "execute_mode": "clipped", "decoded_metric": "domain_raw",
        "require_first_edge_improvement": True,
    }.items():
        observed = planner.get(name)
        contract.require(type(observed) is type(expected) and observed == expected,
                         f"checkpoint planner.{name} differs")
    evaluation = sections["eval"]
    for name, expected in {
        "task_split": "standard", "episodes_per_task": 1,
        "final_episodes_per_task": 1,
    }.items():
        observed = evaluation.get(name)
        contract.require(type(observed) is type(expected) and observed == expected,
                         f"checkpoint eval.{name} differs")
    for section, name, expected in (
        (sections["model"], "branch_factor", 4),
        (sections["model"], "max_depth", 3),
        (sections["tree"], "node_budget", 64),
        (sections["tree"], "max_depth", 3),
        (sections["retrieval"], "enabled", False),
        (sections["retrieval"], "num_keys", 0),
    ):
        observed = section.get(name)
        contract.require(type(observed) is type(expected) and observed == expected,
                         f"checkpoint resolved config {name} differs")
    identity_config = copy.deepcopy(dict(config))
    identity_config["resume"] = None
    config_sha256 = contract.stable_hash(identity_config)
    config_contract_sha256 = _resolved_config_contract_sha256(config)
    expected_contract = setting["resolved_config_contract_sha256_by_seed"][
        str(cell["seed"])
    ]
    contract.require(config_contract_sha256 == expected_contract,
                     "checkpoint resolved config contract differs")
    return config_sha256, config_contract_sha256


def inspect_checkpoint(
    checkpoint: Path,
    launch: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_lock: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if runtime_lock is None:
        with RetainedCheckpoint(checkpoint) as source:
            return _inspect_retained_checkpoint(source, launch, authority)
    with contract.RuntimeCheckpointDecoder(
        runtime_lock, contract.load_manifest(), allow_synthetic=True
    ) as decoder, RetainedCheckpoint(checkpoint, decoder=decoder) as source:
        return _inspect_retained_checkpoint(source, launch, authority)


def _inspect_retained_checkpoint(
    source: RetainedCheckpoint,
    launch: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> dict[str, Any]:
    payload = source.load()
    contract.require(isinstance(payload, Mapping), "checkpoint root is not an object")
    for name in ("step", "completed_updates", "next_step"):
        contract.require(type(payload.get(name)) is int and payload[name] == 5_000,
                         f"checkpoint {name} is not exact update 5000")
    contract.require(payload.get("reason") == "awaiting-external-stage-gate",
                     "checkpoint reason differs")
    contract.require(payload.get("phase") == "train", "checkpoint phase differs")
    contract.require(payload.get("pending_eval_step") is None,
                     "checkpoint retains evaluation intent")
    contract.require(payload.get("final_eval") is None,
                     "checkpoint contains outcome evaluation")
    cadence = contract.require_exact_keys(
        payload.get("post_update_cadence"),
        {"schema_version", "committed_update", "completed_update", "replay_action"},
        "checkpoint post-update cadence",
    )
    contract.require(type(cadence["schema_version"]) is int and cadence["schema_version"] == 1
                     and type(cadence["committed_update"]) is int
                     and type(cadence["completed_update"]) is int
                     and cadence["committed_update"] == cadence["completed_update"] == 5_000
                     and cadence["replay_action"] is None,
                     "checkpoint post-update cadence is incomplete")
    scheduler = payload.get("scheduler")
    contract.require(isinstance(scheduler, Mapping)
                     and type(scheduler.get("last_epoch")) is int
                     and scheduler["last_epoch"] == 5_000,
                     "checkpoint scheduler is not at update 5000")
    config = payload.get("config")
    contract.require(isinstance(config, Mapping), "checkpoint resolved config is absent")
    config_sha256, config_contract_sha256 = _validate_checkpoint_config(
        config, launch, authority
    )

    identity = payload.get("run_identity")
    contract.require(isinstance(identity, Mapping), "checkpoint run identity is absent")
    cell = launch["cell"]
    setting = contract.setting_authority(authority, cell["setting_id"])
    contract.require_exact_keys(identity, RUN_IDENTITY_FIELDS, "checkpoint run identity")
    expected_identity = _expected_run_identity(
        authority, cell, launch["result_root"], config_sha256
    )
    contract.require(contract.canonical_json(identity) == contract.canonical_json(expected_identity),
                     "checkpoint run identity differs")
    identity_sha = payload.get("identity_sha256")
    contract.require_sha256(identity_sha, "checkpoint run identity SHA256")
    contract.require(identity_sha == contract.stable_hash(identity),
                     "checkpoint run identity self-hash differs")
    parameter_schema = _tensor_schema_evidence(payload.get("model"))
    schema_sha = parameter_schema["schema_sha256"]
    tensor_count = parameter_schema["parameter_count"]
    total_numel = parameter_schema["total_numel"]
    decoder_evidence = source.safe_load_evidence
    contract.require(type(decoder_evidence) is dict,
                     "checkpoint decoder graph evidence is absent")
    contract.require(
        decoder_evidence["tensor_count"] == tensor_count
        and decoder_evidence["tensor_numel"] == total_numel
        and decoder_evidence["tensor_bytes"]
        == parameter_schema["total_storage_bytes"],
        "checkpoint model schema is not bound to the actual decoded tensor graph",
    )
    expected_schema = launch.get("expected_model_parameter_schema")
    contract.validate_model_parameter_schema(
        expected_schema, "launch expected model parameter schema"
    )
    contract.require(
        contract.canonical_json(parameter_schema) == contract.canonical_json(expected_schema)
        and schema_sha == launch.get("expected_model_parameter_schema_sha256"),
        "checkpoint model parameter schema differs from pre-output authority",
    )
    contract.require(payload.get("evaluation_seed_tables_sha256")
                     == expected_identity["evaluation_seed_tables_sha256"],
                     "checkpoint evaluation seed-table provenance differs")
    raw_sha, size = source.sha256()
    return {
        "run_identity": dict(identity),
        "run_identity_sha256": identity_sha,
        "model_parameter_schema_sha256": schema_sha,
        "model_parameter_schema": parameter_schema,
        "model_parameter_tensor_count": tensor_count,
        "model_parameter_total_numel": total_numel,
        "checkpoint_raw_sha256": raw_sha,
        "checkpoint_size": size,
        "resolved_config": copy.deepcopy(dict(config)),
        "resolved_config_contract_sha256": config_contract_sha256,
    }


def _receipt_recipe() -> dict[str, Any]:
    return {
        "objective_version": contract.OBJECTIVE,
        "prefix_graph_enabled": {"action": True, "latent": True, "endpoint": True},
        "prefix_weights": {"action": 0.0, "latent": 0.0, "endpoint": 0.0},
        "stop_after_update": 5_000,
    }


def _receipt_outcome() -> dict[str, Any]:
    return {
        "periodic_evaluation_count": 0,
        "terminal_evaluation_count": 0,
        "visualization_count": 0,
        "outcome_rows": 0,
        "wandb_mode": "disabled",
        "wandb_network_calls_allowed": False,
    }


def _receipt_reuse_policy() -> dict[str, Any]:
    return {
        "scope": "fixed_weight_audit_checkpoint_source_only",
        "formal_training_checkpoint_source_allowed": False,
        "formal_resume_allowed": False,
        "formal_arm_membership": "none",
        "fixed_weight_audit_optimizer_steps": 0,
    }


def validate_receipt(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    cell: contract.CalibrationCell,
    result_root: str | Path,
    expected_launch: Mapping[str, Any],
) -> Mapping[str, Any]:
    contract.require_exact_keys(value, RECEIPT_FIELDS, "calibration completion receipt")
    body = dict(value)
    claimed = body.pop("receipt_sha256", None)
    contract.require_sha256(claimed, "calibration completion receipt SHA256")
    contract.require(claimed == contract.stable_hash(body),
                     "calibration completion receipt self-hash differs")
    expected_outer = {
        "schema_version": 1,
        "status": COMPLETE_STATUS,
        "campaign_id": contract.CAMPAIGN_ID,
        "authority_sha256": authority["authority_sha256"],
        "runtime_lock_sha256": expected_launch["runtime_lock_sha256"],
        "model_state_authority_sha256": expected_launch[
            "model_state_authority_sha256"
        ],
        "result_creation_receipt_sha256": expected_launch[
            "result_creation_receipt_sha256"
        ],
        "roots": authority["roots"],
        "cell_index": cell.index,
        "setting_id": cell.setting_id,
        "env_config": cell.env_config,
        "seed": cell.seed,
        "run_name": cell.run_name,
        "completed_updates": 5_000,
        "nominal_optimizer_updates": 25_000,
        "scheduler_total_steps": 1_000_000,
        "checkpoint_relative_path": f"sealed-checkpoints/cell-{cell.index:03d}.pt",
    }
    for name, expected in expected_outer.items():
        contract.require(
            contract.canonical_json(value.get(name)) == contract.canonical_json(expected),
            f"calibration receipt differs: {name}",
        )
    for name in (
        "schema_version", "cell_index", "seed", "wave_index", "completed_updates",
        "nominal_optimizer_updates", "scheduler_total_steps", "checkpoint_size",
        "model_parameter_tensor_count", "model_parameter_total_numel",
    ):
        contract.require_int(value[name], f"calibration receipt {name}", minimum=0)
    contract.require(value["wave_index"] in (0, 1), "calibration receipt wave differs")
    for name in (
        "launch_sha256", "checkpoint_raw_sha256", "run_identity_sha256",
        "model_parameter_schema_sha256", "resolved_config_contract_sha256",
    ):
        contract.require_sha256(value[name], f"calibration receipt {name}")
    contract.require(value["checkpoint_size"] > 0
                     and value["model_parameter_tensor_count"] > 0
                     and value["model_parameter_total_numel"] > 0,
                     "calibration receipt checkpoint/schema counts are empty")
    contract.require(value["launch_sha256"] == expected_launch["launch_sha256"],
                     "calibration receipt launch differs")
    contract.validate_model_parameter_schema(
        value["model_parameter_schema"], "calibration receipt model schema"
    )
    contract.require(
        contract.canonical_json(value["model_parameter_schema"])
        == contract.canonical_json(expected_launch["expected_model_parameter_schema"])
        and value["model_parameter_schema_sha256"]
        == expected_launch["expected_model_parameter_schema_sha256"],
        "calibration receipt model schema differs from pre-output authority",
    )
    recipe = value["recipe"]
    outcome = value["outcome_observations"]
    reuse = value["reuse_policy"]
    contract.require(contract.canonical_json(recipe) == contract.canonical_json(_receipt_recipe()),
                     "calibration receipt recipe differs")
    contract.require(contract.canonical_json(outcome) == contract.canonical_json(_receipt_outcome()),
                     "calibration receipt outcome state differs")
    contract.require(contract.canonical_json(reuse) == contract.canonical_json(_receipt_reuse_policy()),
                     "calibration receipt permits formal-arm reuse")
    contract.require(all(type(recipe["prefix_graph_enabled"][name]) is bool
                         for name in ("action", "latent", "endpoint"))
                     and all(type(recipe["prefix_weights"][name]) is float
                             for name in ("action", "latent", "endpoint"))
                     and type(recipe["stop_after_update"]) is int,
                     "calibration receipt recipe scalar types differ")
    for name in (
        "periodic_evaluation_count", "terminal_evaluation_count",
        "visualization_count", "outcome_rows",
    ):
        contract.require(type(outcome[name]) is int,
                         f"calibration receipt outcome type differs: {name}")
    contract.require(type(outcome["wandb_network_calls_allowed"]) is bool,
                     "calibration receipt W&B policy type differs")
    contract.require(type(reuse["formal_training_checkpoint_source_allowed"]) is bool
                     and type(reuse["formal_resume_allowed"]) is bool
                     and type(reuse["fixed_weight_audit_optimizer_steps"]) is int,
                     "calibration receipt reuse scalar types differ")
    identity = value["run_identity"]
    contract.require_exact_keys(identity, RUN_IDENTITY_FIELDS,
                                "calibration receipt run identity")
    contract.require_sha256(identity["config_sha256"],
                            "calibration receipt config SHA256")
    resolved_config = value["resolved_config"]
    contract.require(type(resolved_config) is dict,
                     "calibration receipt resolved config is not an object")
    synthetic_launch = {
        "cell": {
            "index": cell.index, "setting_id": cell.setting_id,
            "env_config": cell.env_config, "seed": cell.seed,
            "run_name": cell.run_name,
        },
        "result_root": str(result_root),
    }
    config_sha256, config_contract_sha256 = _validate_checkpoint_config(
        resolved_config, synthetic_launch, authority
    )
    contract.require(
        value["resolved_config_contract_sha256"] == config_contract_sha256,
        "calibration receipt resolved config contract differs",
    )
    contract.require(identity["config_sha256"] == config_sha256,
                     "calibration receipt resolved config identity differs")
    expected_identity = _expected_run_identity(authority, cell, result_root, config_sha256)
    contract.require(contract.canonical_json(identity)
                     == contract.canonical_json(expected_identity),
                     "calibration receipt run identity differs")
    contract.require(value["run_identity_sha256"] == contract.stable_hash(identity),
                     "calibration receipt run identity self-hash differs")
    return value


def _marker_provenance(
    launch: Mapping[str, Any], status: str, wave_index: int
) -> dict[str, Any]:
    cell = launch["cell"]
    return {
        "schema_version": 1,
        "status": status,
        "campaign_id": contract.CAMPAIGN_ID,
        "authority_sha256": launch["authority_sha256"],
        "runtime_lock_sha256": launch["runtime_lock_sha256"],
        "model_state_authority_sha256": launch["model_state_authority_sha256"],
        "result_creation_receipt_sha256": launch[
            "result_creation_receipt_sha256"
        ],
        "roots": launch["roots"],
        "cell_index": cell["index"],
        "setting_id": cell["setting_id"],
        "env_config": cell["env_config"],
        "seed": cell["seed"],
        "run_name": cell["run_name"],
        "wave_index": wave_index,
        "launch_sha256": launch["launch_sha256"],
    }


def _completion_marker(
    launch: Mapping[str, Any], wave_index: int, receipt_sha256: str, *, noop: bool = False
) -> dict[str, Any]:
    marker = {
        **_marker_provenance(launch, NOOP_STATUS if noop else COMPLETE_STATUS, wave_index),
        "receipt_sha256": receipt_sha256,
    }
    marker["marker_sha256"] = contract.stable_hash(marker)
    return marker


def _continuation_marker(
    launch: Mapping[str, Any], inspected: Mapping[str, Any]
) -> dict[str, Any]:
    cell = launch["cell"]
    marker = {
        **_marker_provenance(launch, CONTINUATION_STATUS, 0),
        "checkpoint_relative_path": (
            f"live-runs/{cell['setting_id']}/treewm/{cell['run_name']}/checkpoints/latest.pt"
        ),
        "checkpoint_raw_sha256": inspected["checkpoint_raw_sha256"],
        "checkpoint_size": inspected["checkpoint_size"],
        "run_identity_sha256": inspected["run_identity_sha256"],
        "model_parameter_schema_sha256": inspected["model_parameter_schema_sha256"],
        "model_parameter_tensor_count": inspected["model_parameter_tensor_count"],
        "model_parameter_total_numel": inspected["model_parameter_total_numel"],
    }
    marker["marker_sha256"] = contract.stable_hash(marker)
    return marker


def validate_marker(
    value: Mapping[str, Any], launch: Mapping[str, Any], expected_status: str,
    expected_wave: int, *, receipt_sha256: str | None = None,
    inspected: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    fields = CONTINUATION_MARKER_FIELDS if expected_status == CONTINUATION_STATUS \
        else COMPLETION_MARKER_FIELDS
    contract.require_exact_keys(value, fields, "calibration lifecycle marker")
    body = dict(value)
    claimed = body.pop("marker_sha256", None)
    contract.require_sha256(claimed, "calibration lifecycle marker SHA256")
    contract.require(claimed == contract.stable_hash(body),
                     "calibration lifecycle marker self-hash differs")
    expected_provenance = _marker_provenance(launch, expected_status, expected_wave)
    for name, expected in expected_provenance.items():
        contract.require(contract.canonical_json(value.get(name))
                         == contract.canonical_json(expected),
                         f"calibration lifecycle marker differs: {name}")
    for name in ("schema_version", "cell_index", "seed", "wave_index"):
        contract.require_int(value[name], f"calibration lifecycle marker {name}", minimum=0)
    if expected_status == CONTINUATION_STATUS:
        contract.safe_relative(
            value["checkpoint_relative_path"],
            "calibration continuation checkpoint path",
        )
        for name in (
            "checkpoint_raw_sha256", "run_identity_sha256",
            "model_parameter_schema_sha256",
        ):
            contract.require_sha256(
                value[name], f"calibration continuation marker {name}"
            )
        for name in (
            "checkpoint_size", "model_parameter_tensor_count",
            "model_parameter_total_numel",
        ):
            contract.require_int(
                value[name], f"calibration continuation marker {name}", minimum=1
            )
        expected = _continuation_marker(launch, inspected or value)
        contract.require(contract.canonical_json(value) == contract.canonical_json(expected),
                         "calibration continuation marker differs from checkpoint")
    else:
        contract.require_sha256(receipt_sha256, "expected completion receipt SHA256")
        contract.require(value["receipt_sha256"] == receipt_sha256,
                         "calibration lifecycle receipt differs")
    return value


def _controlled_environment(
    launch: Mapping[str, Any], execution: ExecutionAuthority | None = None
) -> dict[str, str]:
    allowed = {str(name): str(value) for name, value in launch["environment"].items()}
    python = Path(launch["argv"][0])
    environment = {
        **allowed,
        "PATH": f"/proc/self/fd/{execution.runtime.root.fd}/{python.parent.name}"
        if execution is not None else str(python.parent),
        "HOME": launch["path_authority"]["scratch_root"],
        "PYTHONHASHSEED": str(launch["cell"]["seed"]),
    }
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    contract.require(isinstance(visible, str) and visible and "," not in visible,
                     "exactly one CUDA device is required")
    environment["CUDA_VISIBLE_DEVICES"] = visible
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if execution is not None:
        runtime_paths = [
            f"/proc/self/fd/{execution.runtime.root.fd}/{path}"
            for path in execution.runtime_lock["sys_path"]
        ]
        environment["PYTHONPATH"] = os.pathsep.join(
            [execution.snapshot_procfd, *runtime_paths]
        )
        environment["LD_LIBRARY_PATH"] = os.pathsep.join(
            f"/proc/self/fd/{execution.runtime.root.fd}/{path}"
            for path in execution.runtime_lock["loader_paths"]
        )
    for name in (
        "SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID", "SLURM_ARRAY_TASK_ID",
        "SLURM_RESTART_COUNT", "SLURM_JOB_GPUS",
    ):
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


class SignalRelay:
    def __init__(self) -> None:
        self.pending: int | None = None
        self.forwarded: int | None = None

    def install(self) -> None:
        signal.signal(signal.SIGUSR1, self._latch)
        signal.signal(signal.SIGTERM, self._latch)

    def _latch(self, signum: int, _frame: object) -> None:
        if self.pending is None or signum == signal.SIGTERM:
            self.pending = signum

    def service(self, process: subprocess.Popen[bytes]) -> None:
        if self.pending is not None and self.forwarded != self.pending:
            try:
                process.send_signal(self.pending)
                self.forwarded = self.pending
            except ProcessLookupError:
                pass


def _run_trainer(launch: Mapping[str, Any], execution: ExecutionAuthority) -> int:
    contract.require_production_runtime(
        execution.runtime_lock, "calibration trainer execution"
    )
    contract.require_production_model_authority(
        execution.model_authority, "calibration trainer execution"
    )
    relay = SignalRelay()
    relay.install()
    execution.verify_boundary("immediately-before-spawn")
    pass_fds = (
        execution.python_fd,
        execution.trainer_fd,
        execution.snapshot.root.fd,
        execution.runtime.root.fd,
        execution.result.fd,
    )
    process = subprocess.Popen(
        [execution.python_procfd, "-P", "-B", execution.trainer_procfd,
         *launch["argv"][2:]],
        cwd=execution.snapshot_procfd,
        env=_controlled_environment(launch, execution),
        stdin=subprocess.DEVNULL,
        pass_fds=pass_fds,
    )
    while True:
        relay.service(process)
        status = process.poll()
        if status is not None:
            execution.verify_boundary("immediately-after-child-exit")
            return int(status)
        select.select([], [], [], 0.25)


def _control_path(result_root: Path, cell_index: int, wave_index: int) -> Path:
    return result_root / "control" / f"cell-{cell_index:03d}" / f"wave-{wave_index}.json"


def _receipt_path(result_root: Path, cell_index: int) -> Path:
    return result_root / "sealed-cells" / f"cell-{cell_index:03d}.json"


def _build_receipt(
    launch: Mapping[str, Any],
    inspected: Mapping[str, Any],
    checkpoint_relative: str,
    wave_index: int,
) -> dict[str, Any]:
    cell = launch["cell"]
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "status": COMPLETE_STATUS,
        "campaign_id": contract.CAMPAIGN_ID,
        "authority_sha256": launch["authority_sha256"],
        "runtime_lock_sha256": launch["runtime_lock_sha256"],
        "model_state_authority_sha256": launch["model_state_authority_sha256"],
        "result_creation_receipt_sha256": launch[
            "result_creation_receipt_sha256"
        ],
        "roots": launch["roots"],
        "cell_index": cell["index"],
        "setting_id": cell["setting_id"],
        "env_config": cell["env_config"],
        "seed": cell["seed"],
        "run_name": cell["run_name"],
        "wave_index": wave_index,
        "run_identity": inspected["run_identity"],
        "run_identity_sha256": inspected["run_identity_sha256"],
        "completed_updates": 5_000,
        "nominal_optimizer_updates": 25_000,
        "scheduler_total_steps": 1_000_000,
        "checkpoint_relative_path": checkpoint_relative,
        "checkpoint_raw_sha256": inspected["checkpoint_raw_sha256"],
        "checkpoint_size": inspected["checkpoint_size"],
        "model_parameter_schema_sha256": inspected["model_parameter_schema_sha256"],
        "model_parameter_schema": inspected["model_parameter_schema"],
        "model_parameter_tensor_count": inspected["model_parameter_tensor_count"],
        "model_parameter_total_numel": inspected["model_parameter_total_numel"],
        "resolved_config": inspected["resolved_config"],
        "resolved_config_contract_sha256": inspected[
            "resolved_config_contract_sha256"
        ],
        "recipe": _receipt_recipe(),
        "outcome_observations": _receipt_outcome(),
        "reuse_policy": _receipt_reuse_policy(),
        "launch_sha256": launch["launch_sha256"],
    }
    receipt["receipt_sha256"] = contract.stable_hash(receipt)
    return receipt


def _export_complete(
    launch: Mapping[str, Any],
    authority: Mapping[str, Any],
    wave_index: int,
    execution: ExecutionAuthority | None = None,
) -> dict[str, Any]:
    contract.require(type(wave_index) is int and wave_index in (0, 1),
                     "calibration export wave differs")
    result_root = Path(launch["result_root"])
    run_directory = Path(launch["run_directory"])
    contract.require(not os.path.lexists(run_directory / "COMPLETED.json"),
                     "calibration reached terminal evaluation completion")
    contract.require(not os.path.lexists(run_directory / "final_eval_progress.json"),
                     "calibration produced terminal outcome rows")
    source_checkpoint = run_directory / "checkpoints" / "latest.pt"
    checkpoint_directory = source_checkpoint.parent
    checkpoint_relative = f"sealed-checkpoints/cell-{launch['cell']['index']:03d}.pt"
    cell_index = launch["cell"]["index"]
    publication_root: Path | contract.DirectoryCapability = (
        execution.result if execution is not None else result_root
    )
    if execution is not None:
        execution.verify_boundary("before-completion-export")
    for target, label in (
        (result_root / checkpoint_relative, "sealed calibration checkpoint"),
        (_receipt_path(result_root, cell_index), "calibration completion receipt"),
        (_control_path(result_root, cell_index, wave_index), "calibration completion marker"),
    ):
        contract.require(not os.path.lexists(target), f"{label} already exists")
    with contract.DirectoryCapability(checkpoint_directory, "checkpoint source directory") as directory:
        contract.require(sorted(os.listdir(directory.fd)) == ["latest.pt"],
                         "calibration run contains an extra checkpoint source")
    with RetainedCheckpoint(
        source_checkpoint,
        decoder=execution.decoder if execution is not None else None,
    ) as source:
        inspected = _inspect_retained_checkpoint(source, launch, authority)
        if wave_index == 1:
            if execution is not None:
                previous = _read_result_json(
                    execution.result,
                    f"control/cell-{cell_index:03d}/wave-0.json",
                    "wave-zero continuation marker",
                )
            else:
                previous = contract.read_json(
                    _control_path(result_root, cell_index, 0),
                    "wave-zero continuation marker",
                )
            validate_marker(
                previous,
                launch,
                CONTINUATION_STATUS,
                0,
                inspected=inspected,
            )
        stage = run_directory / "stage-gates" / "AWAITING_GATE_5000.json"
        if execution is not None:
            stage_value = _read_result_json(
                execution.result,
                str(stage.relative_to(result_root)),
                "calibration stage marker",
                mode=0o644,
            )
        else:
            stage_value = contract.read_json(stage, "calibration stage marker")
        contract.require_exact_keys(stage_value, {
            "schema_version", "status", "objective_version", "completed_updates",
            "step", "total_steps", "scheduler_total_steps", "identity_sha256",
            "checkpoint", "checkpoint_sha256", "evaluation_seed_tables_sha256",
        }, "calibration stage marker")
        expected_stage = {
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
        contract.require(contract.canonical_json(stage_value)
                         == contract.canonical_json(expected_stage),
                         "calibration stage marker differs")
        receipt = _build_receipt(launch, inspected, checkpoint_relative, wave_index)
        validate_receipt(
            receipt, contract.load_manifest(), authority,
            contract.expand_cells(contract.load_manifest())[cell_index], result_root,
            launch,
        )
        published_sha, published_size = source.publish_exclusive(
            publication_root, checkpoint_relative
        )
        contract.require(
            published_sha == inspected["checkpoint_raw_sha256"]
            and published_size == inspected["checkpoint_size"],
            "checkpoint changed during calibration export",
        )
        _exclusive_json(
            publication_root, f"sealed-cells/cell-{cell_index:03d}.json", receipt
        )
        if execution is not None:
            execution.verify_boundary("after-checkpoint-and-receipt-publication")
        if execution is not None:
            final_sha, final_size = _hash_result_file(
                execution.result, checkpoint_relative, mode=0o444
            )
        else:
            final_sha, final_size = _hash_open_file(
                result_root / checkpoint_relative, mode=0o444
            )
        contract.require(
            (final_sha, final_size) == (published_sha, published_size),
            "published checkpoint changed before source retirement",
        )
        source.unlink_verified()
        if execution is not None:
            execution.verify_boundary("after-source-retirement")
    return receipt


def _execute_wave_authorized(
    launch: Mapping[str, Any],
    authority: Mapping[str, Any],
    wave_index: int,
    execution: ExecutionAuthority,
) -> int:
    contract.require(type(wave_index) is int and wave_index in (0, 1),
                     "calibration wave index must be zero or one")
    result_root = Path(launch["result_root"])
    run_directory = Path(launch["run_directory"])
    cell_index = launch["cell"]["index"]
    contract.require(type(cell_index) is int and 0 <= cell_index < 20,
                     "calibration launch cell index differs")
    execution.verify_boundary("before-lifecycle-consumption")
    destination_marker = _control_path(result_root, cell_index, wave_index)
    contract.require(not os.path.lexists(destination_marker),
                     "calibration lifecycle marker already exists")
    previous_path = _control_path(result_root, cell_index, 0)
    if wave_index == 0:
        contract.require(not os.path.lexists(run_directory),
                         "wave zero is not a fresh scratch run")
        contract.require(not os.path.lexists(previous_path), "wave-zero marker already exists")
        contract.require(not os.path.lexists(_receipt_path(result_root, cell_index)),
                         "wave zero found a prior completion receipt")
        contract.require(not os.path.lexists(
            result_root / f"sealed-checkpoints/cell-{cell_index:03d}.pt"
        ), "wave zero found a prior sealed checkpoint")
    else:
        previous = _read_result_json(
            execution.result,
            f"control/cell-{cell_index:03d}/wave-0.json",
            "wave-zero lifecycle marker",
        )
        if previous.get("status") == COMPLETE_STATUS:
            receipt = _read_result_json(
                execution.result,
                f"sealed-cells/cell-{cell_index:03d}.json",
                "calibration completion receipt",
            )
            cell = contract.expand_cells(contract.load_manifest())[cell_index]
            validate_receipt(
                receipt, contract.load_manifest(), authority, cell, result_root,
                launch,
            )
            validate_marker(
                previous, launch, COMPLETE_STATUS, 0,
                receipt_sha256=receipt["receipt_sha256"],
            )
            checkpoint = result_root / receipt["checkpoint_relative_path"]
            raw_sha, size = _hash_result_file(
                execution.result, receipt["checkpoint_relative_path"], mode=0o444
            )
            contract.require((raw_sha, size)
                             == (receipt["checkpoint_raw_sha256"], receipt["checkpoint_size"]),
                             "completed checkpoint differs before authenticated no-op")
            marker = _completion_marker(
                launch, 1, receipt["receipt_sha256"], noop=True
            )
            validate_marker(
                marker, launch, NOOP_STATUS, 1,
                receipt_sha256=receipt["receipt_sha256"],
            )
            execution.verify_boundary("before-authenticated-noop-publication")
            _exclusive_json(
                execution.result, f"control/cell-{cell_index:03d}/wave-1.json", marker
            )
            execution.verify_boundary("after-authenticated-noop-publication")
            return 0
        contract.require(previous.get("status") == CONTINUATION_STATUS,
                         "wave zero is neither complete nor continuation-ready")
        checkpoint = run_directory / "checkpoints" / "latest.pt"
        with RetainedCheckpoint(checkpoint, decoder=execution.decoder) as retained:
            inspected = _inspect_retained_checkpoint(retained, launch, authority)
        validate_marker(previous, launch, CONTINUATION_STATUS, 0, inspected=inspected)

    # Materialize only the authorized scratch ancestry, descriptor-relative and
    # no-follow, immediately before the trainer is allowed to create its run leaf.
    parent_relative = PurePosixPath("live-runs") / launch["cell"]["setting_id"] / "treewm"
    parent = _directory_chain(execution.result, parent_relative)
    os.close(parent)
    execution.verify_boundary("after-run-ancestry-creation")

    status = _run_trainer(launch, execution)
    execution.verify_boundary("post-trainer-return")
    if status == GRACEFUL_EXIT_CODE:
        contract.require(wave_index == 0,
                         "wave one requires completion; a third wave is forbidden")
        checkpoint = run_directory / "checkpoints" / "latest.pt"
        with RetainedCheckpoint(checkpoint, decoder=execution.decoder) as retained:
            inspected = _inspect_retained_checkpoint(retained, launch, authority)
        marker = _continuation_marker(launch, inspected)
        validate_marker(marker, launch, CONTINUATION_STATUS, 0, inspected=inspected)
        execution.verify_boundary("before-continuation-publication")
        _exclusive_json(
            execution.result, f"control/cell-{cell_index:03d}/wave-0.json", marker
        )
        execution.verify_boundary("after-continuation-publication")
        return 0
    contract.require(status == 0, f"calibration trainer exited {status}")
    execution.verify_boundary("before-complete-export")
    receipt = _export_complete(launch, authority, wave_index, execution)
    marker = _completion_marker(launch, wave_index, receipt["receipt_sha256"])
    validate_marker(
        marker, launch, COMPLETE_STATUS, wave_index,
        receipt_sha256=receipt["receipt_sha256"],
    )
    execution.verify_boundary("before-completion-marker-publication")
    _exclusive_json(
        execution.result, f"control/cell-{cell_index:03d}/wave-{wave_index}.json", marker
    )
    execution.verify_boundary("after-completion-marker-publication")
    return 0


def execute_wave(
    launch: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    model_authority: Mapping[str, Any],
    wave_index: int,
) -> int:
    contract.require_production_runtime(runtime_lock, "calibration worker execution")
    contract.require_production_model_authority(
        model_authority, "calibration worker execution"
    )
    controller.validate_launch(
        launch, contract.load_manifest(), authority, runtime_lock, model_authority
    )
    with ExecutionAuthority(
        launch, authority, runtime_lock, model_authority
    ) as execution:
        return _execute_wave_authorized(launch, authority, wave_index, execution)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--describe", action="store_true")
    modes.add_argument("--test-only", action="store_true")
    modes.add_argument("--run", action="store_true")
    parser.add_argument("--authority", type=Path)
    parser.add_argument("--runtime-lock", type=Path)
    parser.add_argument("--model-state-authority", type=Path)
    parser.add_argument("--launch", type=Path)
    parser.add_argument("--cell-index", type=int)
    parser.add_argument("--wave-index", type=int)
    parser.add_argument("--array-task-id", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = contract.load_manifest()
    if args.describe:
        print(json.dumps({
            "schema_version": 1,
            "status": "two_wave_worker_source_ready_execution_blocked_on_external_authorities",
            "campaign_id": contract.CAMPAIGN_ID,
            "production_readiness": False,
            "blocked_on": [
                "sealed_runtime_content_lock",
                "reviewed_production_model_authority_hook_and_sealed_model_state_authority",
                "sealed_result_creation_receipt",
                "outer_launch_authorization",
            ],
            "synthetic_fixture_is_production_authority": False,
            "wave0": "fresh_or_continuation_ready",
            "wave1": "same_cell_resume_or_authenticated_completion_noop",
            "third_wave_allowed": False,
            "persistent_writes_performed": False,
        }, sort_keys=True, indent=2, allow_nan=False))
        return 0
    if args.test_only:
        print(json.dumps({
            "schema_version": 1,
            "status": "test_only_passed_no_persistent_writes",
            "production_readiness": False,
            "cell_count": len(contract.expand_cells(manifest)),
            "checkpoint_inspection_performed": False,
            "trainer_spawned": False,
        }, sort_keys=True, indent=2, allow_nan=False))
        return 0
    contract.require(args.authority is not None and args.launch is not None
                     and args.runtime_lock is not None
                     and args.model_state_authority is not None,
                     "--authority, --runtime-lock, --model-state-authority, and --launch are required")
    contract.require(args.cell_index is not None and args.wave_index is not None,
                     "--cell-index and --wave-index are required")
    contract.require(args.array_task_id is not None,
                     "--array-task-id is required")
    contract.require(args.array_task_id == args.cell_index,
                     "Slurm array task/cell mapping differs")
    authority = contract.read_json(args.authority, "calibration authority")
    runtime_lock = contract.read_json(args.runtime_lock, "calibration runtime lock")
    contract.validate_authority(authority, manifest, runtime_lock)
    model_authority = contract.read_json(
        args.model_state_authority, "calibration model-state authority"
    )
    launch = contract.read_json(args.launch, "calibration launch")
    controller.validate_launch(
        launch, manifest, authority, runtime_lock, model_authority
    )
    contract.require(launch["cell"]["index"] == args.cell_index,
                     "requested cell differs from launch")
    python = Path(authority["environment"]["python"])
    contract.require(python.is_file() and not python.is_symlink(),
                     "pinned calibration Python is unavailable or symlinked")
    contract.require(contract.file_sha256(python) == authority["environment"]["python_sha256"],
                     "pinned calibration Python bytes drifted")
    trainer = Path(launch["snapshot_root"]) / "scripts/train.py"
    contract.require(trainer.is_file() and not trainer.is_symlink(),
                     "snapshot trainer is unavailable or symlinked")
    return execute_wave(
        launch, authority, runtime_lock, model_authority, args.wave_index
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except contract.CalibrationContractError as exc:
        print(f"EXP24_CALIBRATION_WORKER_BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(2)
