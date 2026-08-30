#!/usr/bin/env python3
"""Fail-closed identities for the Exp24 all-ten zero-prefix calibration leaf.

This module is deliberately independent of the still-changing Exp24 formal runtime.
It only describes and authenticates calibration cells.  It does not scan datasets,
submit jobs, or consume a formal outcome.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from contextlib import AbstractContextManager
from collections import OrderedDict
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import resource
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence
import zipfile


sys.dont_write_bytecode = True
PACKAGE_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = PACKAGE_DIR / "calibration_manifest.json"
PLACEHOLDER_PATH = PACKAGE_DIR / "checkpoint_source.lock.unsealed.json"
SEED_CENSUS_PLACEHOLDER_PATH = PACKAGE_DIR / "calibration_seed_census.lock.unsealed.json"
CAMPAIGN_ID = "treewm-executable-prefix-formal-v1-zero-prefix-calibration-v1"
PARENT_CAMPAIGN_ID = "treewm-executable-prefix-formal-v1-launch1"
OBJECTIVE = "treewm_v2_grounded_executable_prefix_pilot_v1"
SETTINGS = (
    ("scene", "scene_play"),
    ("puzzle-3x3", "puzzle_3x3_play"),
    ("puzzle-4x4-100m", "puzzle_4x4_play_100m"),
    ("cube-double", "cube_double_play"),
    ("cube-triple", "cube_triple_play"),
    ("cube-quadruple-100m", "cube_quadruple_play_100m"),
    ("antmaze-large", "antmaze_large_navigate"),
    ("antmaze-giant", "antmaze_giant_navigate"),
    ("humanoidmaze-medium", "humanoidmaze_medium_navigate"),
    ("humanoidmaze-large", "humanoidmaze_large_navigate"),
)
SETTING_METADATA: Mapping[str, Mapping[str, str]] = {
    "scene": {"env_name": "scene-play-v0", "source_name": "scene-play-v0", "dataset_kind": "standard"},
    "puzzle-3x3": {"env_name": "puzzle-3x3-play-v0", "source_name": "puzzle-3x3-play-v0", "dataset_kind": "standard"},
    "puzzle-4x4-100m": {"env_name": "puzzle-4x4-play-v0", "source_name": "puzzle-4x4-play-100m-v0", "dataset_kind": "sharded_100m_full"},
    "cube-double": {"env_name": "cube-double-play-v0", "source_name": "cube-double-play-v0", "dataset_kind": "standard"},
    "cube-triple": {"env_name": "cube-triple-play-v0", "source_name": "cube-triple-play-v0", "dataset_kind": "standard"},
    "cube-quadruple-100m": {"env_name": "cube-quadruple-play-v0", "source_name": "cube-quadruple-play-100m-v0", "dataset_kind": "sharded_100m_full"},
    "antmaze-large": {"env_name": "antmaze-large-navigate-v0", "source_name": "antmaze-large-navigate-v0", "dataset_kind": "standard"},
    "antmaze-giant": {"env_name": "antmaze-giant-navigate-v0", "source_name": "antmaze-giant-navigate-v0", "dataset_kind": "standard"},
    "humanoidmaze-medium": {"env_name": "humanoidmaze-medium-navigate-v0", "source_name": "humanoidmaze-medium-navigate-v0", "dataset_kind": "standard"},
    "humanoidmaze-large": {"env_name": "humanoidmaze-large-navigate-v0", "source_name": "humanoidmaze-large-navigate-v0", "dataset_kind": "standard"},
}
SEEDS = (244, 245)
EVALUATION_TASK_IDS = (1, 2, 3, 4, 5)
SEED_CENSUS_SCOPE = (
    "all_git_refs_and_history_plus_prelaunch_worktree_explicit_training_model_seed_"
    "assignments_excluding_this_preregistration_and_evaluation_seed_tables"
)
ROOT_NAMES = (
    "campaign_sha256",
    "source_sha256",
    "protocol_sha256",
    "config_sha256",
    "input_sha256",
    "future_sha256",
    "environment_sha256",
    "runtime_sha256",
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
if not hasattr(os, "O_NOFOLLOW") or not hasattr(os, "O_DIRECTORY"):
    raise RuntimeError("Exp24 calibration requires O_NOFOLLOW and O_DIRECTORY")
READ_FLAGS = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
DIRECTORY_FLAGS = READ_FLAGS | os.O_DIRECTORY
SAFE_CHECKPOINT_MAX_BYTES = 16 * 1024**3
SAFE_CHECKPOINT_MAX_ARCHIVE_ENTRIES = 65_536
SAFE_CHECKPOINT_MAX_ARCHIVE_UNCOMPRESSED_BYTES = 24 * 1024**3
SAFE_CHECKPOINT_MAX_GRAPH_NODES = 1_000_000
SAFE_CHECKPOINT_MAX_GRAPH_DEPTH = 64
SAFE_CHECKPOINT_MAX_TENSORS = 65_536
SAFE_CHECKPOINT_MAX_TENSOR_NUMEL = 1_000_000_000
SAFE_CHECKPOINT_MAX_TENSOR_BYTES = 8 * 1024**3
SAFE_CHECKPOINT_MAX_REPORT_BYTES = 128 * 1024**2
SAFE_CHECKPOINT_MAX_ARCHIVE_DEPTH = 8
SAFE_CHECKPOINT_MAX_CENTRAL_DIRECTORY_BYTES = 64 * 1024**2
SAFE_CHECKPOINT_MAX_ADDRESS_SPACE = 12 * 1024**3
SAFE_CHECKPOINT_MAX_CPU_SECONDS = 120
SAFE_CHECKPOINT_MAX_WALL_SECONDS = 180
SAFE_CHECKPOINT_MAX_OPEN_FILES = 64
RUNTIME_LOCK_STATUS = "sealed_exp24_calibration_runtime_content_v1"
MODEL_AUTHORITY_STATUS = "sealed_exp24_calibration_model_state_authority_v1"
TERMINAL_CENSUS_STATUS = "sealed_exp24_calibration_scheduler_terminal_census_v1"
MODEL_AUTHORITY_HOOK_PATH = "scripts/exp24_calibration_model_authority.py"
MODEL_AUTHORITY_HOOK_INTERFACE = "exp24_model_state_authority_canonical_stdio_v1"
MODEL_UNSEALED_PROFILE = (
    "production_shaped_validator_fixture_external_hook_unsealed"
)
MODEL_PRODUCTION_PROFILE = "externally_reviewed_authenticated_model_hook_v1"
MODEL_MAX_PARAMETER_COUNT = 65_536
MODEL_MAX_TOTAL_NUMEL = 1_000_000_000
MODEL_MAX_TOTAL_STORAGE_BYTES = 8 * 1024**3
_MODEL_HOOK_VALIDATION_CACHE: dict[tuple[str, str, str], bytes] = {}
SAFE_CHECKPOINT_ALLOWED_GLOBALS = (
    "numpy._core.multiarray._reconstruct",
    "numpy.ndarray",
    "numpy.dtype",
    "numpy.dtypes.UInt32DType",
    "numpy.dtypes.Float32DType",
)
SAFE_TENSOR_MARKER = "__exp24_safe_tensor_v1__"
SAFE_NUMPY_MARKER = "__exp24_safe_numpy_v1__"
SAFE_BYTES_MARKER = "__exp24_bytes_v1__"
SAFE_NUMPY_DTYPE_MARKER = "__exp24_numpy_dtype_v1__"
SAFE_EVIDENCE_MARKERS = frozenset({
    SAFE_TENSOR_MARKER,
    SAFE_NUMPY_MARKER,
    SAFE_BYTES_MARKER,
    SAFE_NUMPY_DTYPE_MARKER,
})


class CalibrationContractError(RuntimeError):
    """A calibration authority or lifecycle invariant was not exact."""


def require(condition: object, message: str) -> None:
    if not condition:
        raise CalibrationContractError(message)


def require_int(value: object, label: str, *, minimum: int | None = None) -> int:
    require(type(value) is int, f"{label} is not an integer")
    result = int(value)
    if minimum is not None:
        require(result >= minimum, f"{label} is below {minimum}")
    return result


def require_bool(value: object, label: str) -> bool:
    require(type(value) is bool, f"{label} is not a boolean")
    return bool(value)


def require_string(value: object, label: str, *, nonempty: bool = True) -> str:
    require(type(value) is str, f"{label} is not a string")
    if nonempty:
        require(bool(value), f"{label} is empty")
    return str(value)


def require_sha256(value: object, label: str) -> str:
    require(type(value) is str and SHA256.fullmatch(value) is not None,
            f"{label} is not a lowercase SHA256")
    return str(value)


def canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise CalibrationContractError(f"value is not canonical finite JSON: {exc}") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def seed_census_scope_sha256() -> str:
    return stable_hash({
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "scope": SEED_CENSUS_SCOPE,
        "seeds": list(SEEDS),
    })


def seed_census_evidence_sha256(census: Mapping[str, Any]) -> str:
    return stable_hash({
        "schema_version": 1,
        "repository_head": census["repository_head"],
        "git_ref_inventory_sha256": census["git_ref_inventory_sha256"],
        "worktree_inventory_sha256": census["worktree_inventory_sha256"],
        "reachable_history_inventory_sha256": census[
            "reachable_history_inventory_sha256"
        ],
        "prior_assignment_inventory_sha256": census[
            "prior_assignment_inventory_sha256"
        ],
    })


def file_sha256(path: str | Path) -> str:
    candidate = Path(path).absolute()
    parent = DirectoryCapability(candidate.parent, f"parent for {candidate.name}")
    try:
        descriptor, before = parent.open_regular(candidate.name, f"source {candidate.name}")
        try:
            digest, _size = hash_descriptor(descriptor, before, f"source {candidate.name}")
            parent.require_named_identity(candidate.name, before, f"source {candidate.name}")
            return digest
        finally:
            os.close(descriptor)
    finally:
        parent.close()


def stat_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid,
        info.st_nlink, info.st_size, info.st_mtime_ns, info.st_ctime_ns,
    )


def directory_identity(info: os.stat_result) -> dict[str, int]:
    require(stat.S_ISDIR(info.st_mode), "directory identity source is not a directory")
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "mode": int(stat.S_IMODE(info.st_mode)),
        "uid": int(info.st_uid),
        "gid": int(info.st_gid),
    }


def hash_descriptor(
    descriptor: int, before: os.stat_result, label: str
) -> tuple[str, int]:
    digest = hashlib.sha256()
    offset = 0
    while offset < before.st_size:
        block = os.pread(descriptor, min(4 * 1024 * 1024, before.st_size - offset), offset)
        require(bool(block), f"{label} ended during hash")
        digest.update(block)
        offset += len(block)
    require(not os.pread(descriptor, 1, before.st_size), f"{label} grew during hash")
    after = os.fstat(descriptor)
    require(stat_identity(after) == stat_identity(before), f"{label} changed during hash")
    return digest.hexdigest(), offset


def _bounded_zip_directory_metadata(
    descriptor: int, before: os.stat_result, label: str
) -> tuple[int, int]:
    """Read only the bounded EOCD/ZIP64 tail before ZipFile allocates member rows."""
    duplicate = os.dup(descriptor)
    try:
        with os.fdopen(duplicate, "rb", closefd=True) as handle:
            duplicate = -1
            try:
                end = zipfile._EndRecData(handle)  # type: ignore[attr-defined]
            except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
                raise CalibrationContractError(
                    f"{label} ZIP directory tail is invalid: {type(exc).__name__}: {exc}"
                ) from exc
    finally:
        if duplicate >= 0:
            os.close(duplicate)
    require(end is not None, f"{label} has no ZIP end-of-directory record")
    entries = require_int(
        end[zipfile._ECD_ENTRIES_TOTAL],  # type: ignore[attr-defined]
        f"{label} ZIP central-directory entry count",
        minimum=1,
    )
    entries_this_disk = require_int(
        end[zipfile._ECD_ENTRIES_THIS_DISK],  # type: ignore[attr-defined]
        f"{label} ZIP per-disk entry count",
        minimum=1,
    )
    directory_size = require_int(
        end[zipfile._ECD_SIZE],  # type: ignore[attr-defined]
        f"{label} ZIP central-directory size",
        minimum=1,
    )
    directory_offset = require_int(
        end[zipfile._ECD_OFFSET],  # type: ignore[attr-defined]
        f"{label} ZIP central-directory offset",
        minimum=0,
    )
    require(end[zipfile._ECD_DISK_NUMBER] == 0  # type: ignore[attr-defined]
            and end[zipfile._ECD_DISK_START] == 0,  # type: ignore[attr-defined]
            f"{label} is a multi-disk ZIP archive")
    require(entries == entries_this_disk <= SAFE_CHECKPOINT_MAX_ARCHIVE_ENTRIES,
            f"{label} archive entry count is outside the safe bound")
    require(directory_size <= SAFE_CHECKPOINT_MAX_CENTRAL_DIRECTORY_BYTES,
            f"{label} ZIP central directory exceeds the safe byte bound")
    require(directory_offset + directory_size <= before.st_size,
            f"{label} ZIP central directory lies outside the retained file")
    return entries, directory_size


def _audit_torch_archive(descriptor: int, before: os.stat_result, label: str) -> dict[str, Any]:
    """Prebound the central directory, then authenticate every Torch ZIP member."""
    require(0 < before.st_size <= SAFE_CHECKPOINT_MAX_BYTES,
            f"{label} size is outside the safe checkpoint bound")
    expected_entries, directory_size = _bounded_zip_directory_metadata(
        descriptor, before, label
    )
    duplicate = os.dup(descriptor)
    try:
        with os.fdopen(duplicate, "rb", closefd=True) as handle:
            duplicate = -1
            try:
                with zipfile.ZipFile(handle, "r") as archive:
                    members = archive.infolist()
            except (OSError, zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
                raise CalibrationContractError(
                    f"{label} is not a bounded Torch ZIP archive: {type(exc).__name__}: {exc}"
                ) from exc
    finally:
        if duplicate >= 0:
            os.close(duplicate)
    require(len(members) == expected_entries,
            f"{label} archive entry count changed after bounded preflight")
    names: set[str] = set()
    total_uncompressed = 0
    storage_entries = 0
    data_pickles = 0
    for member in members:
        name = require_string(member.filename, f"{label} archive member")
        require("\\" not in name and not name.startswith("/"),
                f"{label} archive member path is unsafe")
        path = PurePosixPath(name)
        require(all(part not in ("", ".", "..") for part in path.parts),
                f"{label} archive member path contains traversal")
        require(len(path.parts) <= SAFE_CHECKPOINT_MAX_ARCHIVE_DEPTH,
                f"{label} archive nesting exceeds the safe bound")
        require(name not in names, f"{label} archive contains duplicate members")
        names.add(name)
        require(member.flag_bits & 0x1 == 0, f"{label} archive member is encrypted")
        require(member.compress_type == zipfile.ZIP_STORED,
                f"{label} archive compression is not permitted")
        require(type(member.file_size) is int and member.file_size >= 0,
                f"{label} archive member size is invalid")
        require(member.file_size <= SAFE_CHECKPOINT_MAX_ARCHIVE_UNCOMPRESSED_BYTES,
                f"{label} archive member exceeds the safe size bound")
        total_uncompressed += member.file_size
        require(total_uncompressed <= SAFE_CHECKPOINT_MAX_ARCHIVE_UNCOMPRESSED_BYTES,
                f"{label} archive expansion exceeds the safe bound")
        if len(path.parts) >= 2 and path.parts[-1] == "data.pkl":
            data_pickles += 1
        if len(path.parts) >= 3 and path.parts[-2] == "data":
            require(path.parts[-1].isdigit(), f"{label} storage member name is invalid")
            storage_entries += 1
    require(data_pickles == 1, f"{label} archive data pickle count differs")
    require(storage_entries <= SAFE_CHECKPOINT_MAX_TENSORS,
            f"{label} archive storage count exceeds the safe bound")
    return {
        "archive_entries": len(members),
        "archive_central_directory_bytes": directory_size,
        "archive_uncompressed_bytes": total_uncompressed,
        "archive_storage_entries": storage_entries,
    }


def _apply_checkpoint_child_limits() -> dict[str, list[int]]:
    """Irreversibly lower soft and hard limits in the clean-exec decoder."""
    requested = {
        "RLIMIT_AS": SAFE_CHECKPOINT_MAX_ADDRESS_SPACE,
        "RLIMIT_CPU": SAFE_CHECKPOINT_MAX_CPU_SECONDS,
        "RLIMIT_FSIZE": 0,
        "RLIMIT_CORE": 0,
        "RLIMIT_NOFILE": SAFE_CHECKPOINT_MAX_OPEN_FILES,
    }
    observed: dict[str, list[int]] = {}
    for name, limit in requested.items():
        kind = getattr(resource, name)
        _soft, hard = resource.getrlimit(kind)
        new_limit = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
        resource.setrlimit(kind, (new_limit, new_limit))
        final_soft, final_hard = resource.getrlimit(kind)
        require(final_soft == final_hard == new_limit,
                f"safe checkpoint decoder {name} was not irreversibly lowered")
        observed[name] = [int(final_soft), int(final_hard)]
    return observed


def _safe_checkpoint_globals() -> list[object]:
    """Resolve exactly the preregistered NumPy globals; never mutate global policy."""
    import numpy

    values = [
        numpy._core.multiarray._reconstruct,
        numpy.ndarray,
        numpy.dtype,
        numpy.dtypes.UInt32DType,
        numpy.dtypes.Float32DType,
    ]
    observed = tuple(f"{value.__module__}.{value.__qualname__}" for value in values)
    require(observed == SAFE_CHECKPOINT_ALLOWED_GLOBALS,
            "checkpoint safe-global identities differ from the pinned interface")
    return values


def _sanitize_checkpoint_payload(value: object) -> tuple[object, dict[str, Any]]:
    """Reduce a weights-only object graph to finite JSON and strict tensor evidence."""
    import numpy
    import torch

    nodes = 0
    tensor_count = 0
    tensor_numel = 0
    tensor_bytes = 0
    storages: set[tuple[int, int]] = set()

    def visit(item: object, depth: int) -> object:
        nonlocal nodes, tensor_count, tensor_numel, tensor_bytes
        nodes += 1
        require(nodes <= SAFE_CHECKPOINT_MAX_GRAPH_NODES,
                "checkpoint object graph exceeds the safe node bound")
        require(depth <= SAFE_CHECKPOINT_MAX_GRAPH_DEPTH,
                "checkpoint object graph exceeds the safe nesting bound")
        if item is None or type(item) in (bool, int, str):
            return item
        if type(item) is float:
            require(math.isfinite(item), "checkpoint contains a non-finite scalar")
            return item
        if type(item) is bytes:
            require(len(item) <= 1024 * 1024, "checkpoint byte scalar exceeds the safe bound")
            return {SAFE_BYTES_MARKER: item.hex()}
        if type(item) in (dict, OrderedDict):
            result: dict[str, object] = {}
            for key, child in item.items():
                require(type(key) is str and key not in result,
                        "checkpoint mapping keys are not unique strings")
                require(key not in SAFE_EVIDENCE_MARKERS,
                        "checkpoint input mapping forges a reserved evidence marker")
                result[key] = visit(child, depth + 1)
            return result
        if type(item) in (list, tuple):
            return [visit(child, depth + 1) for child in item]
        if type(item) is torch.Tensor:
            require(item.device.type == "cpu" and item.device.index is None,
                    "checkpoint tensor is not materialized on CPU")
            require(item.layout == torch.strided and not item.is_sparse
                    and not item.is_quantized and not item.is_meta,
                    "checkpoint tensor layout is not plain dense-strided")
            require(item.is_contiguous() and item.storage_offset() == 0,
                    "checkpoint tensor is a view or is non-contiguous")
            shape = list(item.shape)
            require(len(shape) <= 16 and all(type(size) is int and 0 < size <= 1_000_000
                                             for size in shape),
                    "checkpoint tensor shape is outside the safe bound")
            numel = int(item.numel())
            item_bytes = numel * int(item.element_size())
            require(0 < numel <= SAFE_CHECKPOINT_MAX_TENSOR_NUMEL,
                    "checkpoint tensor numel is outside the safe bound")
            require(0 < item_bytes <= SAFE_CHECKPOINT_MAX_TENSOR_BYTES,
                    "checkpoint tensor bytes are outside the safe bound")
            storage = item.untyped_storage()
            require(int(storage.nbytes()) == item_bytes,
                    "checkpoint tensor storage has aliasing or excess bytes")
            storage_key = (int(storage.data_ptr()), int(storage.nbytes()))
            require(storage_key not in storages,
                    "checkpoint tensors share or alias storage")
            storages.add(storage_key)
            if item.is_floating_point() or item.is_complex():
                require(bool(torch.isfinite(item).all().item()),
                        "checkpoint tensor contains NaN or infinity")
            tensor_count += 1
            tensor_numel += numel
            tensor_bytes += item_bytes
            require(tensor_count <= SAFE_CHECKPOINT_MAX_TENSORS,
                    "checkpoint tensor count exceeds the safe bound")
            require(tensor_numel <= SAFE_CHECKPOINT_MAX_TENSOR_NUMEL,
                    "checkpoint total tensor numel exceeds the safe bound")
            require(tensor_bytes <= SAFE_CHECKPOINT_MAX_TENSOR_BYTES,
                    "checkpoint total tensor bytes exceeds the safe bound")
            raw = bytes(storage)
            require(len(raw) == item_bytes, "checkpoint tensor storage byte count differs")
            return {
                SAFE_TENSOR_MARKER: 1,
                "shape": shape,
                "dtype": str(item.dtype),
                "numel": numel,
                "storage_bytes": item_bytes,
                "content_sha256": hashlib.sha256(raw).hexdigest(),
                "device": "cpu",
                "layout": "strided",
                "storage_alias_policy": "unique_exact_storage",
            }
        if type(item) is numpy.ndarray:
            require(item.flags.c_contiguous and item.dtype.hasobject is False,
                    "checkpoint NumPy array is not plain C-contiguous numeric data")
            require(item.size <= SAFE_CHECKPOINT_MAX_TENSOR_NUMEL
                    and item.nbytes <= SAFE_CHECKPOINT_MAX_TENSOR_BYTES,
                    "checkpoint NumPy array exceeds the safe bound")
            if numpy.issubdtype(item.dtype, numpy.floating) or numpy.issubdtype(
                item.dtype, numpy.complexfloating
            ):
                require(bool(numpy.isfinite(item).all()),
                        "checkpoint NumPy array contains NaN or infinity")
            return {
                SAFE_NUMPY_MARKER: 1,
                "shape": list(item.shape),
                "dtype": str(item.dtype),
                "nbytes": int(item.nbytes),
                "content_sha256": hashlib.sha256(item.tobytes(order="C")).hexdigest(),
            }
        if isinstance(item, numpy.dtype):
            require(type(item) in (numpy.dtype, numpy.dtypes.UInt32DType,
                                   numpy.dtypes.Float32DType),
                    "checkpoint NumPy dtype class is not pinned")
            return {SAFE_NUMPY_DTYPE_MARKER: str(item)}
        raise CalibrationContractError(
            f"checkpoint contains unsupported type {type(item).__module__}.{type(item).__qualname__}"
        )

    sanitized = visit(value, 0)
    require(type(sanitized) is dict, "checkpoint root is not a plain mapping")
    return sanitized, {
        "graph_nodes": nodes,
        "tensor_count": tensor_count,
        "tensor_numel": tensor_numel,
        "tensor_bytes": tensor_bytes,
        "storage_alias_policy": "unique_exact_storage",
    }


class _AmbientCheckpointDecoder(AbstractContextManager["_AmbientCheckpointDecoder"]):
    """Explicitly non-production clean-exec decoder used by synthetic tests only."""

    profile = "synthetic_ambient_clean_exec_test_only"
    production_ready = False

    def __init__(self) -> None:
        self.python_path = Path(sys.executable).resolve(strict=True)
        self.source_path = Path(__file__).resolve(strict=True)
        self.python_parent = DirectoryCapability(
            self.python_path.parent, "ambient decoder interpreter parent"
        )
        self.source_parent = DirectoryCapability(
            self.source_path.parent, "ambient decoder source parent"
        )
        try:
            self.python_fd, self.python_before = self.python_parent.open_regular(
                self.python_path.name,
                "ambient decoder interpreter",
                mode=stat.S_IMODE(self.python_path.stat(follow_symlinks=False).st_mode),
            )
            self.source_fd, self.source_before = self.source_parent.open_regular(
                self.source_path.name, "ambient decoder source", mode=0o644
            )
            self.python_sha256, _ = hash_descriptor(
                self.python_fd, self.python_before, "ambient decoder interpreter"
            )
            self.source_sha256, _ = hash_descriptor(
                self.source_fd, self.source_before, "ambient decoder source"
            )
        except BaseException:
            if hasattr(self, "python_fd"):
                os.close(self.python_fd)
            self.source_parent.close()
            self.python_parent.close()
            raise

    def verify(self) -> None:
        python_sha, _ = hash_descriptor(
            self.python_fd, self.python_before, "ambient decoder interpreter"
        )
        source_sha, _ = hash_descriptor(
            self.source_fd, self.source_before, "ambient decoder source"
        )
        require(python_sha == self.python_sha256 and source_sha == self.source_sha256,
                "ambient decoder executable/source bytes changed")
        self.python_parent.require_named_identity(
            self.python_path.name,
            self.python_before,
            "ambient decoder interpreter",
        )
        self.source_parent.require_named_identity(
            self.source_path.name, self.source_before, "ambient decoder source"
        )

    def command(self, checkpoint_fd: int) -> tuple[list[str], tuple[int, ...], dict[str, str]]:
        retained = (checkpoint_fd, self.python_fd, self.source_fd)
        allowed = ",".join(str(fd) for fd in sorted({0, 1, 2, *retained}))
        argv = [
            f"/proc/self/fd/{self.python_fd}", "-I", "-B",
            f"/proc/self/fd/{self.source_fd}",
            "--exp24-safe-checkpoint-decoder",
            str(checkpoint_fd), allowed,
        ]
        environment = {
            "PATH": "",
            "HOME": "/nonexistent-exp24-decoder-home",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        return argv, retained, environment

    def close(self) -> None:
        os.close(self.source_fd)
        os.close(self.python_fd)
        self.source_parent.close()
        self.python_parent.close()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _checkpoint_decoder_main(checkpoint_fd: int, allowed_fds: set[int]) -> int:
    """Clean-exec child entry: no parent Python state or flock description survives."""
    report: dict[str, Any]
    try:
        limits = _apply_checkpoint_child_limits()
        observed_fds: set[int] = set()
        try:
            candidates = {
                int(name) for name in os.listdir("/proc/self/fd")
                if name.isdigit()
            }
        except OSError as exc:
            raise CalibrationContractError(
                "safe checkpoint decoder cannot enumerate its complete descriptor table"
            ) from exc
        for candidate in candidates:
            try:
                os.fstat(candidate)
            except OSError:
                continue
            observed_fds.add(candidate)
        require(observed_fds == allowed_fds,
                "safe checkpoint decoder inherited an undeclared descriptor")
        import torch

        torch.serialization.clear_safe_globals()
        require(torch.serialization.get_safe_globals() == [],
                "safe checkpoint decoder global registry did not clear")
        allowed = _safe_checkpoint_globals()
        expected_names = sorted(SAFE_CHECKPOINT_ALLOWED_GLOBALS)
        duplicate = os.dup(checkpoint_fd)
        with os.fdopen(duplicate, "rb", closefd=True) as handle:
            handle.seek(0)
            with torch.serialization.safe_globals(allowed):
                effective = sorted(
                    f"{value.__module__}.{value.__qualname__}"
                    for value in torch.serialization.get_safe_globals()
                )
                require(effective == expected_names,
                        "safe checkpoint decoder effective global registry differs")
                payload = torch.load(handle, map_location="cpu", weights_only=True)
                effective_after = sorted(
                    f"{value.__module__}.{value.__qualname__}"
                    for value in torch.serialization.get_safe_globals()
                )
                require(effective_after == expected_names,
                        "safe checkpoint decoder global registry changed during load")
        require(torch.serialization.get_safe_globals() == [],
                "safe checkpoint decoder global registry survived scoped load")
        sanitized, graph = _sanitize_checkpoint_payload(payload)
        report = {
            "ok": True,
            "payload": sanitized,
            "graph": graph,
            "safe_globals": list(SAFE_CHECKPOINT_ALLOWED_GLOBALS),
            "limits": limits,
            "inherited_fds": sorted(observed_fds),
        }
    except BaseException as exc:
        report = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc)[:4096],
        }
    encoded = canonical_json(report)
    if len(encoded) > SAFE_CHECKPOINT_MAX_REPORT_BYTES:
        encoded = canonical_json({
            "ok": False,
            "error_type": "CalibrationContractError",
            "error": "safe checkpoint report exceeds bound",
        })
    offset = 0
    while offset < len(encoded):
        count = os.write(1, encoded[offset:])
        require(count > 0, "safe checkpoint decoder report write stopped")
        offset += count
    return 0


def safe_load_checkpoint_fd(
    descriptor: int,
    before: os.stat_result,
    label: str,
    *,
    verify: Any,
    decoder: Any | None = None,
) -> tuple[Mapping[str, Any], dict[str, Any]]:
    """Inspect one retained checkpoint through a clean-exec, bounded decoder."""
    require(callable(verify), f"{label} verifier is not callable")
    verify()
    pre_sha256, pre_size = hash_descriptor(descriptor, before, label)
    archive = _audit_torch_archive(descriptor, before, label)
    verify()
    owned_decoder = decoder is None
    authority = _AmbientCheckpointDecoder() if owned_decoder else decoder
    process: subprocess.Popen[bytes] | None = None
    started = time.monotonic()
    try:
        require(callable(getattr(authority, "verify", None)),
                f"{label} decoder authority is invalid")
        authority.verify()
        argv, pass_fds, environment = authority.command(descriptor)
        expected_inherited_fds = sorted({0, 1, 2, *pass_fds})
        process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            close_fds=True,
            pass_fds=pass_fds,
            env=environment,
            start_new_session=True,
        )
        try:
            stdout, _stderr = process.communicate(timeout=SAFE_CHECKPOINT_MAX_WALL_SECONDS)
        except subprocess.TimeoutExpired as exc:
            process.kill()
            process.communicate()
            raise CalibrationContractError(
                f"{label} safe-loader exceeded the parent wall-clock deadline"
            ) from exc
        elapsed = time.monotonic() - started
        require(elapsed <= SAFE_CHECKPOINT_MAX_WALL_SECONDS + 5,
                f"{label} safe-loader exceeded the wall-clock bound")
        require(len(stdout) <= SAFE_CHECKPOINT_MAX_REPORT_BYTES,
                f"{label} safe-loader report exceeds bound")
        require(process.returncode == 0,
                f"{label} safe-loader clean-exec child failed")
        authority.verify()
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait()
        if owned_decoder:
            authority.close()
    verify()
    post_sha256, post_size = hash_descriptor(descriptor, before, label)
    verify()
    try:
        report = json.loads(stdout.decode("ascii"))
    except (UnboundLocalError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationContractError(f"{label} safe-loader report is invalid") from exc
    require(type(report) is dict and type(report.get("ok")) is bool,
            f"{label} safe-loader report schema differs")
    if report["ok"] is not True:
        raise CalibrationContractError(
            f"cannot safely inspect {label}: {report.get('error_type')}: {report.get('error')}"
        )
    require((pre_sha256, pre_size) == (post_sha256, post_size),
            f"{label} bytes changed across safe load")
    require(report.get("safe_globals") == list(SAFE_CHECKPOINT_ALLOWED_GLOBALS),
            f"{label} safe-global policy differs")
    graph = require_exact_keys(
        report.get("graph"),
        {"graph_nodes", "tensor_count", "tensor_numel", "tensor_bytes",
         "storage_alias_policy"},
        f"{label} safe-loader graph evidence",
    )
    for name in ("graph_nodes", "tensor_count", "tensor_numel", "tensor_bytes"):
        require_int(graph[name], f"{label} safe-loader {name}", minimum=0)
    require(graph["storage_alias_policy"] == "unique_exact_storage",
            f"{label} storage alias policy differs")
    limits = require_exact_keys(
        report.get("limits"),
        {"RLIMIT_AS", "RLIMIT_CPU", "RLIMIT_FSIZE", "RLIMIT_CORE", "RLIMIT_NOFILE"},
        f"{label} safe-loader resource limits",
    )
    for name, maximum in (
        ("RLIMIT_AS", SAFE_CHECKPOINT_MAX_ADDRESS_SPACE),
        ("RLIMIT_CPU", SAFE_CHECKPOINT_MAX_CPU_SECONDS),
        ("RLIMIT_NOFILE", SAFE_CHECKPOINT_MAX_OPEN_FILES),
    ):
        pair = limits[name]
        require(type(pair) is list and len(pair) == 2,
                f"{label} safe-loader {name} shape differs")
        soft = require_int(pair[0], f"{label} safe-loader {name} soft limit")
        hard = require_int(pair[1], f"{label} safe-loader {name} hard limit")
        require(0 < soft == hard <= maximum,
                f"{label} safe-loader {name} hard/soft limits differ")
    for name in ("RLIMIT_FSIZE", "RLIMIT_CORE"):
        pair = limits[name]
        require(type(pair) is list and len(pair) == 2,
                f"{label} safe-loader {name} shape differs")
        require(pair == [0, 0], f"{label} safe-loader {name} is not irreversibly disabled")
    inherited = report.get("inherited_fds")
    require(type(inherited) is list
            and all(type(value) is int for value in inherited)
            and inherited == expected_inherited_fds
            and len(inherited) <= 7,
            f"{label} safe-loader descriptor evidence differs")
    payload = report.get("payload")
    require(type(payload) is dict, f"{label} safe payload root differs")
    evidence = {
        **archive,
        **dict(graph),
        "raw_sha256": pre_sha256,
        "raw_size": pre_size,
        "safe_globals": list(SAFE_CHECKPOINT_ALLOWED_GLOBALS),
        "decoder_isolation": "clean_exec_close_fds_hard_rlimit_canonical_json_only",
        "decoder_limits": dict(limits),
        "decoder_wall_seconds": elapsed,
        "decoder_profile": authority.profile,
        "decoder_production_ready": authority.production_ready,
        "decoder_source_sha256": authority.source_sha256,
        "decoder_interpreter_sha256": authority.python_sha256,
        "weights_only": True,
    }
    return payload, evidence


class DirectoryCapability(AbstractContextManager["DirectoryCapability"]):
    """A retained absolute directory reached without following any symlink."""

    def __init__(self, path: str | Path, label: str) -> None:
        lexical = Path(path)
        require(lexical.is_absolute(), f"{label} is not absolute")
        require(".." not in lexical.parts, f"{label} contains traversal")
        self.path = Path(os.path.normpath(str(lexical)))
        self.label = label
        descriptor = os.open("/", DIRECTORY_FLAGS)
        try:
            for component in self.path.parts[1:]:
                next_descriptor = os.open(component, DIRECTORY_FLAGS, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
        except BaseException:
            os.close(descriptor)
            raise
        self.fd = descriptor
        self.before = os.fstat(descriptor)
        require(stat.S_ISDIR(self.before.st_mode), f"{label} is not a directory")
        self._closed = False

    def open_directory(self, relative: PurePosixPath | str = ".") -> int:
        descriptor = os.dup(self.fd)
        path = PurePosixPath(str(relative))
        if str(path) in ("", "."):
            return descriptor
        path = safe_relative(str(path), f"directory below {self.label}")
        try:
            for component in path.parts:
                next_descriptor = os.open(component, DIRECTORY_FLAGS, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def open_regular(
        self, relative: PurePosixPath | str, label: str, *, mode: int | None = None
    ) -> tuple[int, os.stat_result]:
        path = safe_relative(str(relative), f"file below {self.label}")
        parent = self.open_directory(path.parent)
        descriptor: int | None = None
        try:
            named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            require(stat.S_ISREG(named.st_mode), f"{label} is not a regular file")
            require(named.st_nlink == 1, f"{label} is hard-linked")
            if mode is not None:
                require(stat.S_IMODE(named.st_mode) == mode, f"{label} mode differs")
            descriptor = os.open(path.name, READ_FLAGS, dir_fd=parent)
            opened = os.fstat(descriptor)
            require(stat_identity(opened) == stat_identity(named), f"{label} raced before open")
            return descriptor, opened
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            raise
        finally:
            os.close(parent)

    def require_named_identity(
        self, relative: PurePosixPath | str, expected: os.stat_result, label: str
    ) -> None:
        path = safe_relative(str(relative), f"file below {self.label}")
        parent = self.open_directory(path.parent)
        try:
            named = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
            require(stat_identity(named) == stat_identity(expected), f"{label} pathname changed")
        finally:
            os.close(parent)

    def require_path_identity(self) -> None:
        """Reopen the absolute no-follow path and compare it with the retained root."""
        with DirectoryCapability(self.path, f"path recheck for {self.label}") as reopened:
            require(
                directory_identity(reopened.before) == directory_identity(self.before),
                f"{self.label} pathname changed",
            )

    def require_directory_identity(
        self,
        relative: PurePosixPath | str,
        retained_descriptor: int,
        label: str,
    ) -> None:
        """Rebind a retained descendant directory to the current lexical path."""
        self.require_path_identity()
        reopened = self.open_directory(relative)
        try:
            require(
                directory_identity(os.fstat(reopened))
                == directory_identity(os.fstat(retained_descriptor)),
                f"{label} lexical directory binding changed",
            )
        finally:
            os.close(reopened)

    def close(self) -> None:
        if not self._closed:
            try:
                after = os.fstat(self.fd)
                require(
                    directory_identity(after) == directory_identity(self.before),
                    f"{self.label} identity changed",
                )
            finally:
                os.close(self.fd)
                self._closed = True

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def _tree_record(
    info: os.stat_result,
    kind: str,
    *,
    names: Sequence[str] | None = None,
    sha256: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "kind": kind,
        "identity": list(stat_identity(info)),
        "mode": int(stat.S_IMODE(info.st_mode)),
        "uid": int(info.st_uid),
        "gid": int(info.st_gid),
        "nlink": int(info.st_nlink),
        "size": int(info.st_size),
    }
    if names is not None:
        record["names"] = list(names)
    if sha256 is not None:
        record["sha256"] = sha256
    return record


class RetainedTree(AbstractContextManager["RetainedTree"]):
    """Immutable-tree authority with retained capabilities and independent scans.

    The filesystem lock coordinates only participants in this calibration protocol.
    Immutability additionally requires an owner-only boundary and exact read-only
    modes.  Persistent writes that ignore the protocol are detected by the two
    independent scans and the final recheck of every retained descriptor and name.
    """

    def __init__(
        self,
        path: str | Path,
        label: str,
        *,
        directory_mode: int = 0o555,
        file_mode: int | Sequence[int] = 0o444,
        lock_exclusive: bool,
        expected_inventory: Mapping[str, Any] | None = None,
    ) -> None:
        self.root = DirectoryCapability(path, label)
        self.path = self.root.path
        self.label = label
        self.directory_mode = directory_mode
        self.file_modes = ((file_mode,) if type(file_mode) is int else tuple(file_mode))
        require(bool(self.file_modes)
                and all(type(mode) is int for mode in self.file_modes),
                f"{label} allowed file modes are invalid")
        self._directories: dict[str, tuple[int, os.stat_result, tuple[str, ...]]] = {}
        self._files: dict[str, tuple[int, os.stat_result, str]] = {}
        self._closed = False
        mode = fcntl.LOCK_EX if lock_exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(self.root.fd, mode | fcntl.LOCK_NB)
        except OSError as exc:
            self.root.close()
            raise CalibrationContractError(
                f"{label} is not quiescent under the calibration writer protocol: {exc}"
            ) from exc
        try:
            self.inventory = self._retain_scan()
            if expected_inventory is not None:
                require(
                    canonical_json(self.inventory) == canonical_json(expected_inventory),
                    f"{label} inventory differs from authority",
                )
            self.verify_two_scans()
        except BaseException:
            self._release(verify=False)
            raise

    @staticmethod
    def _key(relative: PurePosixPath) -> str:
        return "." if str(relative) in ("", ".") else str(relative)

    def _validate_directory(self, info: os.stat_result, relative: PurePosixPath) -> None:
        require(stat.S_ISDIR(info.st_mode), f"{self.label} path is not a directory: {relative}")
        require(stat.S_IMODE(info.st_mode) == self.directory_mode,
                f"{self.label} directory mode differs: {relative}")
        require(info.st_uid == os.getuid(),
                f"{self.label} directory owner differs: {relative}")

    def _validate_file(self, info: os.stat_result, relative: PurePosixPath) -> None:
        require(stat.S_ISREG(info.st_mode), f"{self.label} path is not regular: {relative}")
        require(info.st_nlink == 1, f"{self.label} file is hard-linked: {relative}")
        require(stat.S_IMODE(info.st_mode) in self.file_modes,
                f"{self.label} file mode differs: {relative}")
        require(info.st_uid == os.getuid(), f"{self.label} file owner differs: {relative}")

    def _retain_scan(self) -> dict[str, Any]:
        inventory: dict[str, Any] = {}

        def scan(descriptor: int, relative: PurePosixPath) -> None:
            before = os.fstat(descriptor)
            self._validate_directory(before, relative)
            names = tuple(sorted(os.listdir(descriptor)))
            require(len(names) == len(set(names)), f"duplicate names in {self.label}: {relative}")
            key = self._key(relative)
            self._directories[key] = (descriptor, before, names)
            inventory[key] = _tree_record(before, "directory", names=names)
            for name in names:
                named = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                child_relative = PurePosixPath(name) if key == "." else relative / name
                child_key = self._key(child_relative)
                if stat.S_ISDIR(named.st_mode):
                    child = os.open(name, DIRECTORY_FLAGS, dir_fd=descriptor)
                    opened = os.fstat(child)
                    require(stat_identity(opened) == stat_identity(named),
                            f"{self.label} directory raced: {child_relative}")
                    scan(child, child_relative)
                else:
                    self._validate_file(named, child_relative)
                    source = os.open(name, READ_FLAGS, dir_fd=descriptor)
                    opened = os.fstat(source)
                    require(stat_identity(opened) == stat_identity(named),
                            f"{self.label} file raced: {child_relative}")
                    digest, _size = hash_descriptor(
                        source, opened, f"{self.label} file {child_relative}"
                    )
                    self._files[child_key] = (source, opened, digest)
                    inventory[child_key] = _tree_record(opened, "file", sha256=digest)
            require(tuple(sorted(os.listdir(descriptor))) == names
                    and stat_identity(os.fstat(descriptor)) == stat_identity(before),
                    f"{self.label} directory changed while retaining: {relative}")

        scan(self.root.fd, PurePosixPath("."))
        self._verify_retained()
        return inventory

    def _fresh_scan(self) -> dict[str, Any]:
        inventory: dict[str, Any] = {}

        def scan(descriptor: int, relative: PurePosixPath) -> None:
            before = os.fstat(descriptor)
            self._validate_directory(before, relative)
            names = tuple(sorted(os.listdir(descriptor)))
            key = self._key(relative)
            inventory[key] = _tree_record(before, "directory", names=names)
            for name in names:
                named = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                child_relative = PurePosixPath(name) if key == "." else relative / name
                child_key = self._key(child_relative)
                if stat.S_ISDIR(named.st_mode):
                    child = os.open(name, DIRECTORY_FLAGS, dir_fd=descriptor)
                    try:
                        opened = os.fstat(child)
                        require(stat_identity(opened) == stat_identity(named),
                                f"{self.label} directory raced: {child_relative}")
                        scan(child, child_relative)
                        require(
                            stat_identity(os.stat(name, dir_fd=descriptor, follow_symlinks=False))
                            == stat_identity(opened),
                            f"{self.label} directory changed: {child_relative}",
                        )
                    finally:
                        os.close(child)
                else:
                    self._validate_file(named, child_relative)
                    source = os.open(name, READ_FLAGS, dir_fd=descriptor)
                    try:
                        opened = os.fstat(source)
                        require(stat_identity(opened) == stat_identity(named),
                                f"{self.label} file raced: {child_relative}")
                        digest, _size = hash_descriptor(
                            source, opened, f"{self.label} file {child_relative}"
                        )
                        require(
                            stat_identity(os.stat(name, dir_fd=descriptor, follow_symlinks=False))
                            == stat_identity(opened),
                            f"{self.label} file changed: {child_relative}",
                        )
                        inventory[child_key] = _tree_record(opened, "file", sha256=digest)
                    finally:
                        os.close(source)
            require(tuple(sorted(os.listdir(descriptor))) == names
                    and stat_identity(os.fstat(descriptor)) == stat_identity(before),
                    f"{self.label} directory changed while scanning: {relative}")

        root = os.dup(self.root.fd)
        try:
            scan(root, PurePosixPath("."))
        finally:
            os.close(root)
        return inventory

    def _verify_retained(self) -> None:
        self.root.require_path_identity()
        for key, (descriptor, before, names) in self._directories.items():
            require(stat_identity(os.fstat(descriptor)) == stat_identity(before),
                    f"{self.label} retained directory changed: {key}")
            require(tuple(sorted(os.listdir(descriptor))) == names,
                    f"{self.label} retained directory names changed: {key}")
            if key != ".":
                path = PurePosixPath(key)
                parent_key = self._key(path.parent)
                parent = self._directories[parent_key][0]
                require(
                    stat_identity(os.stat(path.name, dir_fd=parent, follow_symlinks=False))
                    == stat_identity(before),
                    f"{self.label} retained directory pathname changed: {key}",
                )
        for key, (descriptor, before, digest) in self._files.items():
            observed_digest, _size = hash_descriptor(
                descriptor, before, f"{self.label} retained file {key}"
            )
            require(observed_digest == digest, f"{self.label} retained file bytes changed: {key}")
            path = PurePosixPath(key)
            parent = self._directories[self._key(path.parent)][0]
            require(
                stat_identity(os.stat(path.name, dir_fd=parent, follow_symlinks=False))
                == stat_identity(before),
                f"{self.label} retained file pathname changed: {key}",
            )

    def verify_two_scans(self) -> None:
        first = self._fresh_scan()
        self._verify_retained()
        second = self._fresh_scan()
        self._verify_retained()
        expected = canonical_json(self.inventory)
        require(canonical_json(first) == expected,
                f"{self.label} first independent inventory changed")
        require(canonical_json(second) == expected,
                f"{self.label} second independent inventory changed")
        require(canonical_json(first) == canonical_json(second),
                f"{self.label} independent inventories differ")

    def list_directory(self, relative: PurePosixPath | str = ".") -> list[str]:
        path = PurePosixPath(str(relative))
        if str(path) not in ("", "."):
            path = safe_relative(str(path), f"directory below {self.label}")
        key = self._key(path)
        require(key in self._directories, f"directory is absent from {self.label}: {path}")
        return list(self._directories[key][2])

    def read_regular(
        self, relative: PurePosixPath | str, *, max_bytes: int = 16 * 1024 * 1024
    ) -> tuple[bytes, os.stat_result]:
        path = safe_relative(str(relative), f"file below {self.label}")
        key = self._key(path)
        require(key in self._files, f"file is absent from {self.label}: {path}")
        descriptor, before, digest = self._files[key]
        require(before.st_size <= max_bytes, f"file exceeds bound in {self.label}: {path}")
        payload = bytearray()
        offset = 0
        while offset < before.st_size:
            block = os.pread(descriptor, min(1024 * 1024, before.st_size - offset), offset)
            require(bool(block), f"file ended in {self.label}: {path}")
            payload.extend(block)
            offset += len(block)
        require(hashlib.sha256(payload).hexdigest() == digest,
                f"file bytes changed in {self.label}: {path}")
        self._verify_retained()
        return bytes(payload), before

    def duplicate_file(self, relative: PurePosixPath | str) -> tuple[int, os.stat_result]:
        path = safe_relative(str(relative), f"file below {self.label}")
        key = self._key(path)
        require(key in self._files, f"file is absent from {self.label}: {path}")
        descriptor, before, _digest = self._files[key]
        self._verify_retained()
        return os.dup(descriptor), before

    def descriptor_for_directory(self, relative: PurePosixPath | str = ".") -> int:
        path = PurePosixPath(str(relative))
        if str(path) not in ("", "."):
            path = safe_relative(str(path), f"directory below {self.label}")
        key = self._key(path)
        require(key in self._directories, f"directory is absent from {self.label}: {path}")
        return self._directories[key][0]

    def _release(self, *, verify: bool) -> None:
        if self._closed:
            return
        error: BaseException | None = None
        if verify:
            try:
                self.verify_two_scans()
            except BaseException as exc:
                error = exc
        for key, (descriptor, _before, _digest) in list(self._files.items()):
            del key
            os.close(descriptor)
        for key, (descriptor, _before, _names) in sorted(
            self._directories.items(), key=lambda item: item[0].count("/"), reverse=True
        ):
            if key != ".":
                os.close(descriptor)
        try:
            fcntl.flock(self.root.fd, fcntl.LOCK_UN)
        finally:
            self.root.close()
            self._closed = True
        if error is not None:
            raise error

    def close(self) -> None:
        self._release(verify=True)

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            self._release(verify=False)


def nofollow_directory_identity(path: str | Path, label: str) -> dict[str, int]:
    with DirectoryCapability(Path(path), label) as root:
        return directory_identity(root.before)


def lexical_descendant(path: str | Path, root: str | Path, label: str) -> Path:
    candidate = Path(os.path.normpath(str(Path(path))))
    anchor = Path(os.path.normpath(str(Path(root))))
    require(candidate.is_absolute() and anchor.is_absolute(), f"{label} is not absolute")
    try:
        relative = candidate.relative_to(anchor)
    except ValueError as exc:
        raise CalibrationContractError(f"{label} is outside authorized scratch") from exc
    require(str(relative) not in ("", "."), f"{label} equals authorized scratch root")
    return candidate


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CalibrationContractError(f"non-finite JSON token in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CalibrationContractError(f"invalid JSON in {label}: {exc}") from exc
    require(isinstance(value, dict), f"{label} root is not an object")
    canonical_json(value)
    return value


def read_json(path: str | Path, label: str = "JSON") -> dict[str, Any]:
    candidate = Path(path).absolute()
    parent = DirectoryCapability(candidate.parent, f"{label} parent")
    descriptor: int | None = None
    try:
        descriptor, opened = parent.open_regular(candidate.name, label)
        chunks: list[bytes] = []
        offset = 0
        while offset < opened.st_size:
            block = os.pread(descriptor, min(1024 * 1024, opened.st_size - offset), offset)
            require(bool(block), f"{label} ended during read")
            chunks.append(block)
            offset += len(block)
        require(not os.pread(descriptor, 1, opened.st_size), f"{label} grew during read")
        after = os.fstat(descriptor)
        require(stat_identity(after) == stat_identity(opened), f"{label} changed during read")
        parent.require_named_identity(candidate.name, opened, label)
        return parse_json_bytes(b"".join(chunks), label)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        parent.close()


def require_exact_keys(value: object, expected: set[str], label: str) -> Mapping[str, Any]:
    require(type(value) is dict, f"{label} is not an exact object")
    observed = set(value)
    require(observed == expected,
            f"{label} fields differ: missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}")
    return value


@dataclass(frozen=True)
class CalibrationCell:
    index: int
    setting_index: int
    seed_index: int
    setting_id: str
    env_config: str
    seed: int
    run_name: str


def expand_cells(manifest: Mapping[str, Any]) -> list[CalibrationCell]:
    matrix = manifest["matrix"]
    rows: list[CalibrationCell] = []
    for setting_index, setting in enumerate(matrix["settings"]):
        for seed_index, seed in enumerate(matrix["seeds"]):
            index = setting_index * 2 + seed_index
            require(index == len(rows), "calibration cell mapping is not contiguous")
            rows.append(CalibrationCell(
                index=index,
                setting_index=setting_index,
                seed_index=seed_index,
                setting_id=setting["id"],
                env_config=setting["env_config"],
                seed=seed,
                run_name=f"exp24-calibration-{setting['id']}-zero-prefix-seed{seed}",
            ))
    require(len(rows) == 20, "calibration matrix does not contain exactly 20 cells")
    require(len({row.run_name for row in rows}) == 20, "calibration run names collide")
    return rows


def validate_manifest(value: Mapping[str, Any]) -> Mapping[str, Any]:
    require_exact_keys(value, {
        "schema_version", "status", "campaign_id", "parent_campaign_id", "purpose",
        "matrix", "seed_collision_census", "recipe", "authority",
        "external_prerequisites", "execution", "seal", "fixed_weight_audit_downstream",
    }, "calibration manifest")
    require_int(value["schema_version"], "manifest schema")
    require(value["schema_version"] == 1, "manifest schema differs")
    require(value["status"] == "preregistered_execution_blocked_on_unsealed_authorities",
            "manifest status differs")
    require(value["campaign_id"] == CAMPAIGN_ID, "calibration campaign differs")
    require(value["parent_campaign_id"] == PARENT_CAMPAIGN_ID, "parent campaign differs")
    require(value["purpose"] == "outcome_blind_all_ten_zero_prefix_checkpoint_calibration",
            "calibration purpose differs")

    matrix = require_exact_keys(value["matrix"], {
        "settings", "seeds", "cell_count", "index_rule", "scratch_only", "formal_arm",
        "reuse_prior_checkpoint", "reuse_prior_output", "reuse_prior_wandb",
    }, "calibration matrix")
    require(matrix["settings"] == [
        {"id": setting, "env_config": env} for setting, env in SETTINGS
    ], "all-ten calibration setting order differs")
    require(all(
        type(row.get("id")) is str and type(row.get("env_config")) is str
        for row in matrix["settings"]
    ), "calibration setting scalar types differ")
    require(matrix["seeds"] == list(SEEDS), "calibration seed bank differs")
    require(all(type(seed) is int for seed in matrix["seeds"]),
            "calibration seeds are not integers")
    require_int(matrix["cell_count"], "calibration cell count")
    require(matrix["cell_count"] == 20, "calibration cell count differs")
    require(matrix["index_rule"] == "setting_index_times_2_plus_seed_index",
            "calibration index rule differs")
    require(matrix["scratch_only"] is True and matrix["formal_arm"] == "none",
            "calibration is not scratch-only/non-formal")
    require(all(matrix[name] is False for name in (
        "reuse_prior_checkpoint", "reuse_prior_output", "reuse_prior_wandb"
    )), "calibration permits prior-state reuse")

    census = require_exact_keys(value["seed_collision_census"], {
        "status", "seeds", "required_prior_assignment_matches",
        "performed_before_first_calibration_run", "scope", "scope_sha256",
        "required_evidence_fields", "sealed_census_sha256",
    }, "seed collision census registration")
    require(census["status"] == "required_unsealed_before_first_calibration_run",
            "seed census is not preregistered unsealed")
    require(census["seeds"] == list(SEEDS), "seed census bank differs")
    require(all(type(seed) is int for seed in census["seeds"]),
            "seed census bank types differ")
    require_int(census["required_prior_assignment_matches"], "prior assignment matches")
    require(census["required_prior_assignment_matches"] == 0,
            "seed census permits prior assignments")
    require(census["performed_before_first_calibration_run"] is True,
            "seed census timing is not preregistered")
    require(census["scope"] == SEED_CENSUS_SCOPE,
            "seed census scope differs from preregistration")
    require(type(census["scope"]) is str, "seed census scope type differs")
    require(census["scope_sha256"] == seed_census_scope_sha256(),
            "preregistered seed census scope SHA256 differs")
    require(census["required_evidence_fields"] == [
        "repository_head", "git_ref_inventory_sha256", "worktree_inventory_sha256",
        "reachable_history_inventory_sha256", "prior_assignment_inventory_sha256",
    ], "seed census evidence field registration differs")
    require(census["sealed_census_sha256"] is None,
            "source manifest claims an already sealed seed census")

    recipe = value["recipe"]
    require(recipe == {
        "experiment_config": OBJECTIVE,
        "objective_version": OBJECTIVE,
        "nominal_optimizer_updates": 25_000,
        "stop_environment": {"TREEWM_STOP_AFTER_UPDATE": "5000"},
        "export_update": 5_000,
        "scheduler_total_steps": 1_000_000,
        "checkpoint_every": 5_000,
        "validation_every": 25_000,
        "diagnostics_every": 5_000,
        "training_log_every": 50,
        "periodic_evaluation_every": 25_000,
        "visualization_every": 25_000,
        "visualization_every_early": 25_000,
        "visualization_early_until": 0,
        "validation_due_at_export": False,
        "evaluation_due_at_export": False,
        "visualization_due_at_export": False,
        "terminal_evaluation_reached": False,
        "wandb_mode": "disabled",
        "future_recipe_anchor_policy": "published_union",
        "future_cache": False,
        "shared_cache": True,
        "executable_prefix_steps": 4,
        "executable_prefix_enabled": {"action": True, "latent": True, "endpoint": True},
        "executable_prefix_weights": {"action": 0.0, "latent": 0.0, "endpoint": 0.0},
        "checkpoint_reason": "awaiting-external-stage-gate",
        "checkpoint_phase": "train",
        "pending_eval_step": None,
        "outcome_measurement_allowed": False,
    }, "calibration recipe differs")
    for name in (
        "nominal_optimizer_updates", "export_update", "scheduler_total_steps",
        "checkpoint_every", "validation_every", "diagnostics_every",
        "training_log_every", "periodic_evaluation_every", "visualization_every",
        "visualization_every_early", "visualization_early_until",
        "executable_prefix_steps",
    ):
        require_int(recipe[name], f"calibration recipe {name}")
    for name in (
        "validation_due_at_export", "evaluation_due_at_export", "visualization_due_at_export",
        "terminal_evaluation_reached", "future_cache", "shared_cache",
        "outcome_measurement_allowed",
    ):
        require(type(recipe[name]) is bool, f"calibration recipe {name} is not boolean")
    require(all(type(value) is bool for value in recipe["executable_prefix_enabled"].values()),
            "calibration prefix enabled types differ")
    require(all(type(value) is float for value in recipe["executable_prefix_weights"].values()),
            "calibration prefix weight types differ")

    authority = require_exact_keys(value["authority"], {
        "required_status", "required_roots", "all_roots_lowercase_sha256",
        "all_ten_setting_rows_required", "seed_collision_census_required",
    }, "manifest authority")
    require(authority["required_status"] == "sealed_zero_prefix_calibration_authorities",
            "authority status differs")
    require(authority["required_roots"] == list(ROOT_NAMES), "authority roots differ")
    require(all(authority[name] is True for name in (
        "all_roots_lowercase_sha256", "all_ten_setting_rows_required",
        "seed_collision_census_required",
    )), "authority strictness differs")

    prerequisites = value["external_prerequisites"]
    require(prerequisites == {
        "runtime_content_lock": {
            "required_status": RUNTIME_LOCK_STATUS,
            "availability": "required_sealed_external_artifact_absent",
            "production_lock_sha256": None,
            "runtime_sha256_derivation": "validated_runtime_content_sha256",
            "symlink_policy": "forbid_all_components_and_entries",
            "required_content": [
                "interpreter", "pyvenv_cfg", "stdlib_roots", "site_package_roots",
                "native_extensions", "shared_libraries", "sys_path", "loader_paths",
                "serialization_policy", "decoder_source", "closure_attestation",
                "execution_profile",
            ],
        },
        "model_state_authority": {
            "required_status": MODEL_AUTHORITY_STATUS,
            "availability": "required_sealed_external_artifact_absent",
            "production_authority_sha256": None,
            "hook_relative_path": MODEL_AUTHORITY_HOOK_PATH,
            "hook_raw_sha256": None,
            "hook_interface": MODEL_AUTHORITY_HOOK_INTERFACE,
            "hook_sha256_derivation": "authenticated_snapshot_inventory_raw_bytes",
            "construction": (
                "authenticated_source_resolved_config_deterministic_zero_initialization"
            ),
            "schema_contract": (
                "exact_names_shapes_dtypes_counts_numel_storage_and_root_per_setting"
            ),
            "execution_profile": MODEL_UNSEALED_PROFILE,
            "production_ready": False,
            "synthetic_fixture_authorizes_production": False,
        },
        "result_creation_receipt": {
            "required_status": "sealed_exclusive_calibration_result_creation_v1",
            "production_receipt_sha256": None,
            "receipt_relative_path": "RESULT_CREATION.json",
            "creation_operation": "mkdirat_final_component_exclusive",
            "initial_result_mode": "0700",
            "receipt_inode_binding": "exclusive_created_inode_bound_inside_receipt",
            "overwrite": "forbidden",
        },
        "scheduler_terminal_census": {
            "required_status": TERMINAL_CENSUS_STATUS,
            "availability": "required_sealed_external_artifact_absent",
            "production_lock_sha256": None,
            "required_task_rows": 40,
            "waves": [0, 1],
            "required_state": "COMPLETED",
            "required_exit_code": "0:0",
            "required_whole_array_topology": {
                "wave0": "0-19%20",
                "wave1": "0-19%20",
                "wave1_dependency": "afterok_exact_wave0_array_job_id",
                "report_dependency": "afterok_exact_wave1_array_job_id",
                "distinct_array_job_ids": 2,
            },
            "evidence_source": "external_scheduler_terminal_census",
            "evidence_encoding": (
                "canonical_json_newline_slurm_sacct_parsable2_exact_external_authority"
            ),
            "leaf_scheduler_query": "forbidden",
        },
        "lock_publication_boundary": {
            "required_status": (
                "sealed_exp24_checkpoint_lock_publication_boundary_v1"
            ),
            "availability": "required_sealed_external_artifact_absent",
            "production_lock_sha256": None,
            "publication_parent_relative_path": (
                "locks/treewm-executable-prefix-formal-v1-zero-prefix-calibration-v1"
            ),
            "writer_closure_status": (
                "outer_transaction_terminal_no_owner_writer_capabilities_v1"
            ),
            "advisory_flock_alone_is_immutability_proof": False,
            "persistent_rebind_detection_and_cleanup_required": True,
            "transient_detached_write_prevention_claimed": False,
        },
    }, "external prerequisite registration differs")

    execution = value["execution"]
    require(execution == {
        "topology": "two_whole_arrays_afterok",
        "wave_count": 2,
        "wave0_array": "0-19%20",
        "wave1_array": "0-19%20",
        "wave1_dependency": "afterok_exact_whole_wave0_array",
        "kill_on_invalid_dependency": True,
        "scheduler_requeue": False,
        "same_cell_resume_only": True,
        "wave1_completed_cell_action": "authenticated_noop",
        "signal": "B:USR1@420",
        "walltime": "04:00:00",
        "gpus_per_cell": 1,
        "cpus_per_cell": 12,
        "fresh_wave0_required": True,
        "third_or_adaptive_wave_allowed": False,
        "owner_only_scratch_and_live_result_mode": "0700",
        "frozen_snapshot_directory_mode": "0555",
        "frozen_snapshot_file_mode": "0444",
        "python_trainer_and_cwd_descriptor_execution": True,
        "retained_source_and_result_capabilities_through_publication": True,
        "exact_post_exit_source_and_result_rescan": True,
    }, "two-wave calibration topology differs")
    for name in ("wave_count", "gpus_per_cell", "cpus_per_cell"):
        require_int(execution[name], f"calibration execution {name}")
    for name in (
        "kill_on_invalid_dependency", "scheduler_requeue", "same_cell_resume_only",
        "fresh_wave0_required", "third_or_adaptive_wave_allowed",
        "python_trainer_and_cwd_descriptor_execution",
        "retained_source_and_result_capabilities_through_publication",
        "exact_post_exit_source_and_result_rescan",
    ):
        require(type(execution[name]) is bool,
                f"calibration execution {name} is not boolean")

    seal = value["seal"]
    require(seal == {
        "status": "sealed_exp24_all_ten_zero_prefix_checkpoint_source",
        "expected_checkpoints": 20,
        "completed_updates": 5_000,
        "unique_checkpoint_raw_sha256": True,
        "unique_run_identity_sha256": True,
        "per_setting_uniform_model_parameter_schema_sha256": True,
        "checkpoint_mode": "0444",
        "checkpoint_nlink": 1,
        "cell_receipt_mode": "0444",
        "sealed_directory_mode": "0555",
        "outer_exact_launch_authorization_required": True,
        "launch_path_source_and_sha_bound_per_row": True,
        "checkpoint_torch_load_and_schema_derivation_required": True,
        "retained_all_entry_capabilities": True,
        "two_independent_complete_final_scans": True,
        "scheduler_terminal_census_prerequisite": TERMINAL_CENSUS_STATUS,
        "result_creation_receipt_prerequisite": (
            "sealed_exclusive_calibration_result_creation_v1"
        ),
        "exclusive_publication_lock_protocol": (
            "retained_result_root_exclusive_nonblocking_from_prevalidation_"
            "through_file_and_parent_fsync"
        ),
        "uncooperative_persistent_writer_must_be_detected": True,
        "transient_uncooperative_writer_detection_claimed": False,
        "terminal_or_adverse_markers_allowed": False,
        "formal_training_or_resume_reuse_allowed": False,
    }, "calibration seal policy differs")
    for name in ("expected_checkpoints", "completed_updates", "checkpoint_nlink"):
        require_int(seal[name], f"calibration seal {name}")
    for name in (
        "unique_checkpoint_raw_sha256", "unique_run_identity_sha256",
        "per_setting_uniform_model_parameter_schema_sha256",
        "outer_exact_launch_authorization_required",
        "launch_path_source_and_sha_bound_per_row",
        "checkpoint_torch_load_and_schema_derivation_required",
        "retained_all_entry_capabilities", "two_independent_complete_final_scans",
        "uncooperative_persistent_writer_must_be_detected",
        "transient_uncooperative_writer_detection_claimed",
        "terminal_or_adverse_markers_allowed",
        "formal_training_or_resume_reuse_allowed",
    ):
        require(type(seal[name]) is bool, f"calibration seal {name} is not boolean")
    fixed = value["fixed_weight_audit_downstream"]
    require(fixed == {
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
    }, "fixed-weight downstream contract differs")
    for name in ("settings", "batches_per_setting_regime", "expected_rows", "optimizer_steps"):
        require_int(fixed[name], f"fixed-weight audit {name}")
    require(all(type(seed) is int for seed in fixed["checkpoint_seeds"] + fixed["scratch_seeds"]),
            "fixed-weight audit seed types differ")
    require(type(fixed["retuning_allowed"]) is bool
            and fixed["retuning_allowed"] is False,
            "fixed-weight audit retuning policy type differs")
    expand_cells(value)
    return value


def load_manifest(path: str | Path = MANIFEST_PATH) -> dict[str, Any]:
    value = read_json(path, "calibration manifest")
    validate_manifest(value)
    return value


RUNTIME_LOCK_FIELDS = {
    "schema_version", "status", "campaign_id", "manifest_file_sha256",
    "runtime_root", "runtime_root_identity", "runtime_inventory",
    "runtime_inventory_sha256", "interpreter", "pyvenv_cfg",
    "stdlib_roots", "site_package_roots", "native_extensions",
    "shared_libraries", "sys_path", "loader_paths", "symlink_policy",
    "serialization_policy", "execution_profile", "production_ready",
    "decoder_source", "closure_attestation",
    "runtime_content_sha256", "runtime_lock_sha256",
}
RUNTIME_FILE_BINDING_FIELDS = {"relative_path", "sha256", "mode"}
RUNTIME_ROOT_BINDING_FIELDS = {"relative_path", "subtree_sha256"}
RUNTIME_SYNTHETIC_PROFILE = "synthetic_fixture_only_never_publication_ready"
RUNTIME_PRODUCTION_PROFILE = "complete_relocatable_authenticated_runtime_v1"
RUNTIME_SYNTHETIC_CLOSURE_STATUS = "unverified_synthetic_fixture_runtime_closure"
RUNTIME_PRODUCTION_CLOSURE_STATUS = "verified_complete_runtime_closure_v1"
RUNTIME_CLOSURE_FIELDS = {
    "schema_version", "status", "probe_source_sha256", "probe_stdout_sha256",
    "sys_executable", "sys_prefix", "sys_path", "imported_module_files",
    "elf_interpreter", "loaded_shared_libraries", "outside_runtime_paths",
}


def _validate_identity_object(value: object, label: str) -> Mapping[str, Any]:
    identity = require_exact_keys(
        value, {"device", "inode", "mode", "uid", "gid"}, label
    )
    for name in ("device", "inode", "mode", "uid", "gid"):
        require_int(identity[name], f"{label} {name}", minimum=0)
    return identity


def _inventory_subtree(
    inventory: Mapping[str, Any], relative: PurePosixPath
) -> dict[str, Any]:
    prefix = str(relative)
    result = {
        key: value for key, value in inventory.items()
        if key == prefix or key.startswith(prefix + "/")
    }
    require(bool(result), f"runtime inventory subtree is absent: {relative}")
    return result


def _validate_runtime_file_binding(
    value: object,
    inventory: Mapping[str, Any],
    label: str,
    *,
    expected_mode: int | None = None,
) -> Mapping[str, Any]:
    row = require_exact_keys(value, RUNTIME_FILE_BINDING_FIELDS, label)
    relative = safe_relative(row["relative_path"], f"{label} relative path")
    require_sha256(row["sha256"], f"{label} SHA256")
    mode = require_int(row["mode"], f"{label} mode", minimum=0)
    if expected_mode is not None:
        require(mode == expected_mode, f"{label} mode differs")
    observed = inventory.get(str(relative))
    require(type(observed) is dict and observed.get("kind") == "file",
            f"{label} is absent from runtime inventory")
    require(observed.get("sha256") == row["sha256"]
            and observed.get("mode") == mode,
            f"{label} differs from runtime inventory")
    return row


def _validate_runtime_root_bindings(
    value: object,
    inventory: Mapping[str, Any],
    label: str,
    *,
    allow_empty: bool = False,
) -> list[Mapping[str, Any]]:
    require(type(value) is list and (allow_empty or bool(value)),
            f"{label} inventory is invalid")
    rows: list[Mapping[str, Any]] = []
    paths: list[str] = []
    for index, item in enumerate(value):
        row = require_exact_keys(
            item, RUNTIME_ROOT_BINDING_FIELDS, f"{label} {index}"
        )
        relative = safe_relative(row["relative_path"], f"{label} {index} path")
        require_sha256(row["subtree_sha256"], f"{label} {index} subtree SHA256")
        observed = inventory.get(str(relative))
        require(type(observed) is dict and observed.get("kind") == "directory",
                f"{label} {index} root is absent from runtime inventory")
        require(row["subtree_sha256"] == stable_hash(_inventory_subtree(inventory, relative)),
                f"{label} {index} subtree hash differs")
        paths.append(str(relative))
        rows.append(row)
    require(paths == sorted(set(paths)), f"{label} paths are not sorted and unique")
    return rows


def _validate_runtime_closure_schema(
    value: object,
    inventory: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
) -> Mapping[str, Any]:
    closure = require_exact_keys(
        value, RUNTIME_CLOSURE_FIELDS, "runtime execution closure attestation"
    )
    require_int(closure["schema_version"], "runtime closure schema")
    require(closure["schema_version"] == 1, "runtime closure schema differs")
    profile = runtime_lock["execution_profile"]
    if profile == RUNTIME_SYNTHETIC_PROFILE:
        require(closure["status"] == RUNTIME_SYNTHETIC_CLOSURE_STATUS,
                "synthetic runtime closure status differs")
        require(closure["probe_source_sha256"] is None
                and closure["probe_stdout_sha256"] is None
                and closure["sys_executable"] is None
                and closure["sys_prefix"] is None,
                "synthetic runtime claims production closure evidence")
        for name in (
            "sys_path", "imported_module_files", "loaded_shared_libraries",
            "outside_runtime_paths",
        ):
            require(closure[name] == [],
                    f"synthetic runtime closure {name} must remain empty")
        require(closure["elf_interpreter"] is None,
                "synthetic runtime claims an authenticated ELF interpreter")
        return closure

    require(profile == RUNTIME_PRODUCTION_PROFILE,
            "runtime execution profile differs")
    require(closure["status"] == RUNTIME_PRODUCTION_CLOSURE_STATUS,
            "production runtime closure is not verified")
    require_sha256(closure["probe_source_sha256"], "runtime closure probe source SHA256")
    require(closure["probe_source_sha256"] == runtime_lock["decoder_source"]["sha256"],
            "runtime closure probe source differs from decoder source")
    require_sha256(closure["probe_stdout_sha256"], "runtime closure probe stdout SHA256")
    body = dict(closure)
    claimed = body.pop("probe_stdout_sha256")
    require(claimed == stable_hash(body), "runtime closure probe output hash differs")
    require(closure["sys_executable"] == runtime_lock["interpreter"]["relative_path"],
            "runtime closure interpreter path differs")
    require(closure["sys_prefix"] == ".", "runtime closure prefix escaped runtime root")
    require(canonical_json(closure["sys_path"]) == canonical_json(runtime_lock["sys_path"]),
            "runtime closure sys.path differs")
    require(closure["outside_runtime_paths"] == [],
            "runtime execution closure contains paths outside the sealed runtime")
    for name in ("imported_module_files", "loaded_shared_libraries"):
        paths = closure[name]
        require(type(paths) is list and bool(paths)
                and paths == sorted(set(paths))
                and all(type(path) is str for path in paths),
                f"runtime closure {name} is not sorted, unique, and nonempty")
        for path_text in paths:
            relative = safe_relative(path_text, f"runtime closure {name} path")
            row = inventory.get(str(relative))
            require(type(row) is dict and row.get("kind") == "file",
                    f"runtime closure {name} file is outside inventory: {relative}")
    elf = safe_relative(closure["elf_interpreter"], "runtime closure ELF interpreter")
    require(str(elf) in closure["loaded_shared_libraries"],
            "runtime ELF interpreter is absent from loaded library closure")
    require(any(str(elf) == row["relative_path"]
                for row in runtime_lock["shared_libraries"]),
            "runtime ELF interpreter is not explicitly content-addressed")
    native_paths = {row["relative_path"] for row in runtime_lock["native_extensions"]}
    require(native_paths.issubset(set(closure["imported_module_files"])),
            "runtime native extensions are absent from imported module closure")
    shared_paths = {row["relative_path"] for row in runtime_lock["shared_libraries"]}
    require(shared_paths | native_paths == set(closure["loaded_shared_libraries"]),
            "runtime loaded shared-library closure differs from sealed bindings")
    return closure


def validate_runtime_lock(
    value: Mapping[str, Any], manifest: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Validate a separately sealed, symlink-free, content-addressed runtime tree."""
    validate_manifest(manifest)
    require_exact_keys(value, RUNTIME_LOCK_FIELDS, "calibration runtime-content lock")
    require_int(value["schema_version"], "runtime-content lock schema")
    require(value["schema_version"] == 1, "runtime-content lock schema differs")
    require(value["status"] == RUNTIME_LOCK_STATUS,
            "calibration runtime-content lock is not sealed")
    require(value["campaign_id"] == CAMPAIGN_ID, "runtime-content campaign differs")
    require_sha256(value["manifest_file_sha256"], "runtime-content manifest SHA256")
    require(value["manifest_file_sha256"] == file_sha256(MANIFEST_PATH),
            "runtime-content manifest bytes drifted")
    runtime_root = Path(require_string(value["runtime_root"], "runtime root"))
    require(runtime_root.is_absolute() and ".." not in runtime_root.parts,
            "runtime root is not an absolute lexical path")
    inventory = value["runtime_inventory"]
    require(type(inventory) is dict and bool(inventory),
            "runtime inventory is not a nonempty object")
    require_sha256(value["runtime_inventory_sha256"], "runtime inventory SHA256")
    require(value["runtime_inventory_sha256"] == stable_hash(inventory),
            "runtime inventory hash differs")
    with RetainedTree(
        runtime_root,
        "sealed calibration runtime tree",
        directory_mode=0o555,
        file_mode=(0o444, 0o555),
        lock_exclusive=False,
        expected_inventory=inventory,
    ) as tree:
        identity = directory_identity(tree.root.before)
        require(canonical_json(value["runtime_root_identity"]) == canonical_json(identity),
                "runtime root identity differs")
        _validate_identity_object(value["runtime_root_identity"], "runtime root identity")
        interpreter = _validate_runtime_file_binding(
            value["interpreter"], inventory, "runtime interpreter", expected_mode=0o555
        )
        decoder_source = _validate_runtime_file_binding(
            value["decoder_source"], inventory, "runtime checkpoint decoder source",
            expected_mode=0o444,
        )
        _validate_runtime_file_binding(
            value["pyvenv_cfg"], inventory, "runtime pyvenv configuration",
            expected_mode=0o444,
        )
        stdlib = _validate_runtime_root_bindings(
            value["stdlib_roots"], inventory, "runtime stdlib roots"
        )
        site = _validate_runtime_root_bindings(
            value["site_package_roots"], inventory, "runtime site-package roots"
        )
        native = value["native_extensions"]
        shared = value["shared_libraries"]
        require(type(native) is list and type(shared) is list,
                "runtime native/shared inventories are invalid")
        native_rows = [
            _validate_runtime_file_binding(row, inventory, f"runtime native extension {index}")
            for index, row in enumerate(native)
        ]
        shared_rows = [
            _validate_runtime_file_binding(row, inventory, f"runtime shared library {index}")
            for index, row in enumerate(shared)
        ]
        for label, rows in (("native extension", native_rows), ("shared library", shared_rows)):
            paths = [row["relative_path"] for row in rows]
            require(paths == sorted(set(paths)), f"runtime {label} paths differ")
        governed_dirs = {
            row["relative_path"] for row in [*stdlib, *site]
        }
        for name in ("sys_path", "loader_paths"):
            paths = value[name]
            require(type(paths) is list and all(type(path) is str for path in paths),
                    f"runtime {name} is invalid")
            require(paths == list(dict.fromkeys(paths)), f"runtime {name} is duplicated")
            for index, path_text in enumerate(paths):
                relative = safe_relative(path_text, f"runtime {name} {index}")
                observed = inventory.get(str(relative))
                require(type(observed) is dict and observed.get("kind") == "directory",
                        f"runtime {name} path is not governed")
                require(str(relative) in governed_dirs,
                        f"runtime {name} path is not an exact declared root")
        require(value["symlink_policy"] == "forbid_all_components_and_entries",
                "runtime symlink policy differs")
        profile = require_string(value["execution_profile"], "runtime execution profile")
        production_ready = require_bool(
            value["production_ready"], "runtime production readiness"
        )
        require(profile in (RUNTIME_SYNTHETIC_PROFILE, RUNTIME_PRODUCTION_PROFILE),
                "runtime execution profile differs")
        require(production_ready is (profile == RUNTIME_PRODUCTION_PROFILE),
                "runtime production readiness/profile differs")
        serialization = require_exact_keys(
            value["serialization_policy"],
            {"torch_load_weights_only", "safe_globals", "decoder_isolation",
             "storage_alias_policy", "max_checkpoint_bytes", "max_archive_entries",
             "max_archive_uncompressed_bytes", "max_archive_depth",
             "max_central_directory_bytes",
             "max_graph_nodes", "max_graph_depth", "max_tensors",
             "max_tensor_numel", "max_tensor_bytes", "max_report_bytes",
             "rlimit_address_space", "rlimit_cpu_seconds", "rlimit_nofile",
             "wall_clock_seconds"},
            "runtime serialization policy",
        )
        require_bool(serialization["torch_load_weights_only"],
                     "runtime weights-only policy")
        require(serialization["torch_load_weights_only"] is True
                and serialization["safe_globals"] == list(SAFE_CHECKPOINT_ALLOWED_GLOBALS)
                and serialization["decoder_isolation"]
                == "clean_exec_close_fds_hard_rlimit_canonical_json_only"
                and serialization["storage_alias_policy"] == "unique_exact_storage",
                "runtime serialization policy differs")
        for name, expected in (
            ("max_checkpoint_bytes", SAFE_CHECKPOINT_MAX_BYTES),
            ("max_archive_entries", SAFE_CHECKPOINT_MAX_ARCHIVE_ENTRIES),
            ("max_archive_uncompressed_bytes",
             SAFE_CHECKPOINT_MAX_ARCHIVE_UNCOMPRESSED_BYTES),
            ("max_archive_depth", SAFE_CHECKPOINT_MAX_ARCHIVE_DEPTH),
            ("max_central_directory_bytes", SAFE_CHECKPOINT_MAX_CENTRAL_DIRECTORY_BYTES),
            ("max_graph_nodes", SAFE_CHECKPOINT_MAX_GRAPH_NODES),
            ("max_graph_depth", SAFE_CHECKPOINT_MAX_GRAPH_DEPTH),
            ("max_tensors", SAFE_CHECKPOINT_MAX_TENSORS),
            ("max_tensor_numel", SAFE_CHECKPOINT_MAX_TENSOR_NUMEL),
            ("max_tensor_bytes", SAFE_CHECKPOINT_MAX_TENSOR_BYTES),
            ("max_report_bytes", SAFE_CHECKPOINT_MAX_REPORT_BYTES),
            ("rlimit_address_space", SAFE_CHECKPOINT_MAX_ADDRESS_SPACE),
            ("rlimit_cpu_seconds", SAFE_CHECKPOINT_MAX_CPU_SECONDS),
            ("rlimit_nofile", SAFE_CHECKPOINT_MAX_OPEN_FILES),
            ("wall_clock_seconds", SAFE_CHECKPOINT_MAX_WALL_SECONDS),
        ):
            require_int(serialization[name], f"runtime serialization {name}")
            require(serialization[name] == expected,
                    f"runtime serialization {name} differs")
        closure = _validate_runtime_closure_schema(
            value["closure_attestation"], inventory, value
        )
        if production_ready:
            observed_closure = _probe_production_runtime_closure(value, tree)
            require(canonical_json(observed_closure) == canonical_json(closure),
                    "runtime closure probe differs from sealed attestation")
        tree.verify_two_scans()
    content_body = {
        "schema_version": 1,
        "runtime_root_identity": value["runtime_root_identity"],
        "runtime_inventory_sha256": value["runtime_inventory_sha256"],
        "interpreter": interpreter,
        "decoder_source": decoder_source,
        "pyvenv_cfg": value["pyvenv_cfg"],
        "stdlib_roots": value["stdlib_roots"],
        "site_package_roots": value["site_package_roots"],
        "native_extensions": value["native_extensions"],
        "shared_libraries": value["shared_libraries"],
        "sys_path": value["sys_path"],
        "loader_paths": value["loader_paths"],
        "symlink_policy": value["symlink_policy"],
        "serialization_policy": value["serialization_policy"],
        "execution_profile": value["execution_profile"],
        "production_ready": value["production_ready"],
        "closure_attestation": value["closure_attestation"],
    }
    require_sha256(value["runtime_content_sha256"], "runtime content SHA256")
    require(value["runtime_content_sha256"] == stable_hash(content_body),
            "runtime content hash differs")
    body = dict(value)
    claimed = body.pop("runtime_lock_sha256", None)
    require_sha256(claimed, "runtime lock SHA256")
    require(claimed == stable_hash(body), "runtime lock self-hash differs")
    return value


def require_production_runtime(
    runtime_lock: Mapping[str, Any], label: str
) -> None:
    require(runtime_lock.get("execution_profile") == RUNTIME_PRODUCTION_PROFILE
            and runtime_lock.get("production_ready") is True,
            f"{label} requires a complete externally sealed production runtime closure")


class RuntimeCheckpointDecoder(AbstractContextManager["RuntimeCheckpointDecoder"]):
    """Retain the runtime and exact decoder source across every clean-exec load."""

    def __init__(
        self,
        runtime_lock: Mapping[str, Any],
        manifest: Mapping[str, Any],
        *,
        allow_synthetic: bool = False,
    ) -> None:
        validate_runtime_lock(runtime_lock, manifest)
        self.runtime_lock = runtime_lock
        self.profile = runtime_lock["execution_profile"]
        self.production_ready = runtime_lock["production_ready"]
        require(self.production_ready or allow_synthetic,
                "synthetic runtime decoder is forbidden outside validator-only tests")
        self.runtime = RetainedTree(
            runtime_lock["runtime_root"],
            "checkpoint decoder runtime",
            directory_mode=0o555,
            file_mode=(0o444, 0o555),
            lock_exclusive=False,
            expected_inventory=runtime_lock["runtime_inventory"],
        )
        try:
            self.source_fd, self.source_before = self.runtime.duplicate_file(
                runtime_lock["decoder_source"]["relative_path"]
            )
            self.source_sha256, _ = hash_descriptor(
                self.source_fd, self.source_before, "runtime checkpoint decoder source"
            )
            require(self.source_sha256 == runtime_lock["decoder_source"]["sha256"]
                    == file_sha256(Path(__file__).resolve(strict=True)),
                    "runtime decoder source is not the exact reviewed contract source")
            if self.production_ready:
                self.python_fd, self.python_before = self.runtime.duplicate_file(
                    runtime_lock["interpreter"]["relative_path"]
                )
                self.python_parent = None
                self.python_path = None
            else:
                self.python_path = Path(sys.executable).resolve(strict=True)
                self.python_parent = DirectoryCapability(
                    self.python_path.parent,
                    "synthetic decoder ambient interpreter parent",
                )
                self.python_fd, self.python_before = self.python_parent.open_regular(
                    self.python_path.name,
                    "synthetic decoder ambient interpreter",
                    mode=stat.S_IMODE(self.python_path.stat(follow_symlinks=False).st_mode),
                )
            self.python_sha256, _ = hash_descriptor(
                self.python_fd, self.python_before, "checkpoint decoder interpreter"
            )
        except BaseException:
            if hasattr(self, "python_fd"):
                os.close(self.python_fd)
            if getattr(self, "python_parent", None) is not None:
                self.python_parent.close()
            if hasattr(self, "source_fd"):
                os.close(self.source_fd)
            self.runtime._release(verify=False)
            raise
        self._closed = False

    def verify(self) -> None:
        require(stat_identity(os.fstat(self.source_fd)) == stat_identity(self.source_before),
                "runtime checkpoint decoder source descriptor changed")
        require(stat_identity(os.fstat(self.python_fd)) == stat_identity(self.python_before),
                "runtime checkpoint decoder interpreter descriptor changed")
        source_sha, _ = hash_descriptor(
            self.source_fd, self.source_before, "runtime checkpoint decoder source"
        )
        python_sha, _ = hash_descriptor(
            self.python_fd, self.python_before, "checkpoint decoder interpreter"
        )
        require(source_sha == self.source_sha256 and python_sha == self.python_sha256,
                "checkpoint decoder executable/source bytes changed")
        self.runtime._verify_retained()
        self.runtime.root.require_path_identity()
        if self.python_parent is not None:
            assert self.python_path is not None
            self.python_parent.require_named_identity(
                self.python_path.name,
                self.python_before,
                "synthetic decoder ambient interpreter",
            )

    def command(self, checkpoint_fd: int) -> tuple[list[str], tuple[int, ...], dict[str, str]]:
        retained = (
            checkpoint_fd, self.python_fd, self.source_fd, self.runtime.root.fd
        )
        allowed = ",".join(str(fd) for fd in sorted({0, 1, 2, *retained}))
        common = [
            f"/proc/self/fd/{self.python_fd}",
            *(["-s", "-S", "-P", "-B"] if self.production_ready else ["-I", "-B"]),
            f"/proc/self/fd/{self.source_fd}",
            "--exp24-safe-checkpoint-decoder",
            str(checkpoint_fd), allowed,
        ]
        environment = {
            "PATH": "",
            "HOME": "/nonexistent-exp24-decoder-home",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        if self.production_ready:
            environment["PYTHONHOME"] = f"/proc/self/fd/{self.runtime.root.fd}"
            environment["PYTHONPATH"] = os.pathsep.join(
                f"/proc/self/fd/{self.runtime.root.fd}/{path}"
                for path in self.runtime_lock["sys_path"]
            )
            environment["LD_LIBRARY_PATH"] = os.pathsep.join(
                f"/proc/self/fd/{self.runtime.root.fd}/{path}"
                for path in self.runtime_lock["loader_paths"]
            )
        return common, retained, environment

    def close(self) -> None:
        if self._closed:
            return
        error: BaseException | None = None
        try:
            self.verify()
            self.runtime.verify_two_scans()
        except BaseException as exc:
            error = exc
        os.close(self.python_fd)
        if self.python_parent is not None:
            self.python_parent.close()
        os.close(self.source_fd)
        self.runtime._release(verify=False)
        self._closed = True
        if error is not None:
            raise error

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.close()
        else:
            os.close(self.python_fd)
            if self.python_parent is not None:
                self.python_parent.close()
            os.close(self.source_fd)
            self.runtime._release(verify=False)
            self._closed = True


def _runtime_closure_probe_main(
    runtime_root_fd: int,
    source_sha256: str,
    declared_sys_path: Sequence[str],
) -> int:
    """Emit the actual import/native/loader closure of a clean production exec."""
    root_proc = f"/proc/self/fd/{runtime_root_fd}"
    root_real = os.path.realpath(root_proc)
    sys.path[:] = [f"{root_proc}/{path}" for path in declared_sys_path]
    import numpy  # noqa: F401 - the production closure must include it
    import torch  # noqa: F401 - the production closure must include it

    outside: set[str] = set()

    def relative_path(raw: str) -> str | None:
        candidate = os.path.realpath(raw)
        try:
            if os.path.commonpath((root_real, candidate)) != root_real:
                outside.add(candidate)
                return None
        except ValueError:
            outside.add(candidate)
            return None
        relative = os.path.relpath(candidate, root_real)
        if relative == "." or relative.startswith("../"):
            outside.add(candidate)
            return None
        return PurePosixPath(relative).as_posix()

    imported: set[str] = set()
    for module in tuple(sys.modules.values()):
        source = getattr(module, "__file__", None)
        if type(source) is str and source:
            relative = relative_path(source)
            if relative is not None:
                imported.add(relative)
    mapped: set[str] = set()
    with open("/proc/self/maps", "r", encoding="ascii", errors="strict") as handle:
        for line in handle:
            fields = line.rstrip("\n").split(maxsplit=5)
            if len(fields) != 6 or not fields[5].startswith("/"):
                continue
            raw = fields[5]
            basename = os.path.basename(raw)
            if not (".so" in basename or basename.startswith("ld-")):
                continue
            relative = relative_path(raw)
            if relative is not None:
                mapped.add(relative)
    loaders = sorted(
        path for path in mapped
        if os.path.basename(path).startswith("ld-")
        or os.path.basename(path).startswith("ld-linux")
    )
    report = {
        "schema_version": 1,
        "status": RUNTIME_PRODUCTION_CLOSURE_STATUS,
        "probe_source_sha256": source_sha256,
        "sys_executable": relative_path(sys.executable),
        "sys_prefix": "." if os.path.realpath(sys.prefix) == root_real
        else relative_path(sys.prefix),
        "sys_path": list(declared_sys_path),
        "imported_module_files": sorted(imported),
        "elf_interpreter": loaders[0] if len(loaders) == 1 else None,
        "loaded_shared_libraries": sorted(mapped),
        "outside_runtime_paths": sorted(outside),
    }
    encoded = canonical_json(report)
    require(len(encoded) <= 64 * 1024**2,
            "runtime closure probe report exceeds bound")
    offset = 0
    while offset < len(encoded):
        count = os.write(1, encoded[offset:])
        require(count > 0, "runtime closure probe report write stopped")
        offset += count
    return 0


def _probe_production_runtime_closure(
    runtime_lock: Mapping[str, Any], tree: RetainedTree
) -> dict[str, Any]:
    """Clean-exec the sealed interpreter and compare its actual complete closure."""
    python_fd, python_before = tree.duplicate_file(
        runtime_lock["interpreter"]["relative_path"]
    )
    source_fd, source_before = tree.duplicate_file(
        runtime_lock["decoder_source"]["relative_path"]
    )
    try:
        python_sha, _ = hash_descriptor(
            python_fd, python_before, "runtime closure interpreter"
        )
        source_sha, _ = hash_descriptor(
            source_fd, source_before, "runtime closure probe source"
        )
        require(python_sha == runtime_lock["interpreter"]["sha256"]
                and source_sha == runtime_lock["decoder_source"]["sha256"],
                "runtime closure executable/source hash differs")
        argv = [
            f"/proc/self/fd/{python_fd}", "-s", "-S", "-P", "-B",
            f"/proc/self/fd/{source_fd}", "--exp24-runtime-closure-probe",
            str(tree.root.fd), source_sha,
            json.dumps(runtime_lock["sys_path"], separators=(",", ":")),
        ]
        environment = {
            "PATH": "",
            "HOME": "/nonexistent-exp24-runtime-probe-home",
            "PYTHONHOME": f"/proc/self/fd/{tree.root.fd}",
            "PYTHONPATH": os.pathsep.join(
                f"/proc/self/fd/{tree.root.fd}/{path}"
                for path in runtime_lock["sys_path"]
            ),
            "LD_LIBRARY_PATH": os.pathsep.join(
                f"/proc/self/fd/{tree.root.fd}/{path}"
                for path in runtime_lock["loader_paths"]
            ),
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONHASHSEED": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
        }
        try:
            completed = subprocess.run(
                argv,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                close_fds=True,
                pass_fds=(python_fd, source_fd, tree.root.fd),
                env=environment,
                timeout=SAFE_CHECKPOINT_MAX_WALL_SECONDS,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise CalibrationContractError(
                f"production runtime closure probe failed: {type(exc).__name__}: {exc}"
            ) from exc
        require(completed.returncode == 0
                and 0 < len(completed.stdout) <= 64 * 1024**2,
                "production runtime closure probe did not return bounded evidence")
        try:
            report = json.loads(completed.stdout.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CalibrationContractError(
                "production runtime closure probe output is invalid"
            ) from exc
        require(type(report) is dict and canonical_json(report) == completed.stdout,
                "production runtime closure probe output is not canonical")
        report["probe_stdout_sha256"] = stable_hash(report)
        tree._verify_retained()
        return report
    finally:
        os.close(source_fd)
        os.close(python_fd)


MODEL_PARAMETER_DTYPES = {
    "torch.float16": 2,
    "torch.bfloat16": 2,
    "torch.float32": 4,
    "torch.float64": 8,
    "torch.int8": 1,
    "torch.uint8": 1,
    "torch.int16": 2,
    "torch.int32": 4,
    "torch.int64": 8,
    "torch.bool": 1,
}
MODEL_PARAMETER_FIELDS = {
    "name", "shape", "dtype", "numel", "storage_bytes", "device", "layout",
    "storage_alias_policy",
}
MODEL_SCHEMA_FIELDS = {
    "schema_version", "model_class", "parameters", "parameter_count",
    "total_numel", "total_storage_bytes", "schema_sha256",
}


def validate_model_parameter_schema(value: object, label: str) -> Mapping[str, Any]:
    schema = require_exact_keys(value, MODEL_SCHEMA_FIELDS, label)
    require_int(schema["schema_version"], f"{label} schema")
    require(schema["schema_version"] == 1 and schema["model_class"] == "TreeWM",
            f"{label} header differs")
    parameters = schema["parameters"]
    require(type(parameters) is list and bool(parameters), f"{label} parameters are absent")
    names: list[str] = []
    total_numel = 0
    total_bytes = 0
    for index, item in enumerate(parameters):
        row = require_exact_keys(item, MODEL_PARAMETER_FIELDS, f"{label} parameter {index}")
        name = require_string(row["name"], f"{label} parameter {index} name")
        require(name.strip() == name and ".." not in name.split("."),
                f"{label} parameter name is invalid")
        shape = row["shape"]
        require(type(shape) is list and 0 < len(shape) <= 16,
                f"{label} parameter shape is invalid")
        product = 1
        for dimension in shape:
            product *= require_int(
                dimension, f"{label} parameter dimension", minimum=1
            )
        dtype = require_string(row["dtype"], f"{label} parameter dtype")
        require(dtype in MODEL_PARAMETER_DTYPES, f"{label} parameter dtype differs")
        numel = require_int(row["numel"], f"{label} parameter numel", minimum=1)
        storage_bytes = require_int(
            row["storage_bytes"], f"{label} parameter storage bytes", minimum=1
        )
        require(product == numel
                and storage_bytes == numel * MODEL_PARAMETER_DTYPES[dtype],
                f"{label} parameter shape/storage differs")
        require(row["device"] == "cpu" and row["layout"] == "strided"
                and row["storage_alias_policy"] == "unique_exact_storage",
                f"{label} parameter materialization differs")
        names.append(name)
        total_numel += numel
        total_bytes += storage_bytes
    require(names == sorted(set(names)), f"{label} parameter names are not sorted/unique")
    count = require_int(schema["parameter_count"], f"{label} parameter count", minimum=1)
    claimed_numel = require_int(schema["total_numel"], f"{label} total numel", minimum=1)
    claimed_bytes = require_int(
        schema["total_storage_bytes"], f"{label} total storage bytes", minimum=1
    )
    require(count == len(parameters) <= MODEL_MAX_PARAMETER_COUNT,
            f"{label} parameter count differs")
    require(claimed_numel == total_numel <= MODEL_MAX_TOTAL_NUMEL,
            f"{label} total numel differs")
    require(claimed_bytes == total_bytes <= MODEL_MAX_TOTAL_STORAGE_BYTES,
            f"{label} total storage bytes differs")
    require_sha256(schema["schema_sha256"], f"{label} SHA256")
    body = dict(schema)
    claimed = body.pop("schema_sha256")
    require(claimed == stable_hash(body), f"{label} self-hash differs")
    return schema


MODEL_AUTHORITY_FIELDS = {
    "schema_version", "status", "campaign_id", "manifest_file_sha256",
    "source_sha256", "config_sha256", "runtime_content_sha256",
    "runtime_lock_sha256", "snapshot_root", "snapshot_root_identity",
    "snapshot_inventory_sha256", "hook_relative_path", "hook_source_sha256",
    "hook_interface", "derivation_request_sha256", "derivation_stdout_sha256",
    "execution_profile", "production_ready", "external_hook_source_sha256",
    "deterministic_construction", "settings", "model_state_authority_sha256",
}


def model_authority_request(
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    snapshot_inventory_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "interface": MODEL_AUTHORITY_HOOK_INTERFACE,
        "campaign_id": CAMPAIGN_ID,
        "manifest_file_sha256": file_sha256(MANIFEST_PATH),
        "source_sha256": authority["roots"]["source_sha256"],
        "config_sha256": authority["roots"]["config_sha256"],
        "runtime_content_sha256": runtime_lock["runtime_content_sha256"],
        "snapshot_inventory_sha256": require_sha256(
            snapshot_inventory_sha256, "model authority snapshot inventory SHA256"
        ),
        "settings": [
            {
                "setting_id": row["setting_id"],
                "env_config": row["env_config"],
                "resolved_config_contract_sha256_by_seed": dict(
                    row["resolved_config_contract_sha256_by_seed"]
                ),
            }
            for row in authority["settings"]
        ],
        "construction": {
            "model_class": "TreeWM",
            "initialization": "deterministic_zero_init_before_optimizer_or_data",
            "device": "cpu",
            "layout": "strided",
            "storage_alias_policy": "unique_exact_storage",
        },
    }


def execute_model_authority_hook(
    runtime_lock: Mapping[str, Any],
    snapshot_root: str | Path,
    snapshot_inventory: Mapping[str, Any],
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Run the authenticated, separately reviewed hook through retained procfd paths."""
    runtime_root = Path(runtime_lock["runtime_root"])
    snapshot_path = Path(snapshot_root)
    with RetainedTree(
        runtime_root,
        "model-authority runtime",
        directory_mode=0o555,
        file_mode=(0o444, 0o555),
        lock_exclusive=False,
        expected_inventory=runtime_lock["runtime_inventory"],
    ) as runtime_tree, RetainedTree(
        snapshot_path,
        "model-authority source snapshot",
        directory_mode=0o555,
        file_mode=0o444,
        lock_exclusive=False,
        expected_inventory=snapshot_inventory,
    ) as snapshot_tree:
        python_fd, python_before = runtime_tree.duplicate_file(
            runtime_lock["interpreter"]["relative_path"]
        )
        hook_fd, hook_before = snapshot_tree.duplicate_file(MODEL_AUTHORITY_HOOK_PATH)
        try:
            python_procfd = f"/proc/self/fd/{python_fd}"
            hook_procfd = f"/proc/self/fd/{hook_fd}"
            snapshot_procfd = f"/proc/self/fd/{snapshot_tree.root.fd}"
            bootstrap = (
                "s=__import__('sys');p=s.argv[1];"
                "s.path[:]=s.argv[2].split(':');"
                "b=open(p,'rb').read();"
                "exec(compile(b,p,'exec'),{'__name__':'__main__'})"
            )
            governed_sys_path = ":".join(
                f"/proc/self/fd/{runtime_tree.root.fd}/{path}"
                for path in runtime_lock["sys_path"]
            )
            controlled_env = {
                "PATH": "",
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONHASHSEED": "0",
                "OMP_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "LD_LIBRARY_PATH": ":".join(
                    f"/proc/self/fd/{runtime_tree.root.fd}/{path}"
                    for path in runtime_lock["loader_paths"]
                ),
            }
            try:
                completed = subprocess.run(
                    [python_procfd, "-I", "-S", "-B", "-c", bootstrap, hook_procfd,
                     governed_sys_path,
                     f"--interface={MODEL_AUTHORITY_HOOK_INTERFACE}",
                     f"--request-sha256={stable_hash(request)}",
                     *[
                         "--setting-contract=" + ":".join((
                             row["setting_id"], row["env_config"],
                             row["resolved_config_contract_sha256_by_seed"]["244"],
                             row["resolved_config_contract_sha256_by_seed"]["245"],
                         ))
                         for row in request["settings"]
                     ]],
                    input=canonical_json(request) + b"\n",
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=snapshot_procfd,
                    env=controlled_env,
                    pass_fds=(python_fd, hook_fd, snapshot_tree.root.fd,
                              runtime_tree.root.fd),
                    check=False,
                    timeout=SAFE_CHECKPOINT_MAX_CPU_SECONDS,
                )
            except (OSError, subprocess.SubprocessError) as exc:
                raise CalibrationContractError(
                    f"model-authority hook could not run: {type(exc).__name__}: {exc}"
                ) from exc
            require(completed.returncode == 0,
                    "model-authority hook did not exit successfully")
            require(not completed.stderr,
                    "model-authority hook emitted unauthenticated stderr")
            require(0 < len(completed.stdout) <= 64 * 1024 * 1024,
                    "model-authority hook stdout is outside the bound")
            try:
                output = json.loads(completed.stdout.decode("ascii"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise CalibrationContractError(
                    "model-authority hook stdout is not canonical JSON"
                ) from exc
            require(type(output) is dict
                    and canonical_json(output) + b"\n" == completed.stdout,
                    "model-authority hook stdout encoding is not canonical")
            require(stat_identity(os.fstat(python_fd)) == stat_identity(python_before),
                    "model-authority interpreter changed during execution")
            require(stat_identity(os.fstat(hook_fd)) == stat_identity(hook_before),
                    "model-authority hook changed during execution")
            runtime_tree.verify_two_scans()
            snapshot_tree.verify_two_scans()
            return output
        finally:
            os.close(hook_fd)
            os.close(python_fd)


def build_model_state_authority(
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    snapshot_root: str | Path,
    *,
    expected_external_hook_source_sha256: str | None = None,
) -> dict[str, Any]:
    """Derive exact parameter schemas before any calibration output is permitted."""
    validate_runtime_lock(runtime_lock, manifest)
    snapshot_path = Path(snapshot_root)
    with RetainedTree(
        snapshot_path,
        "model-authority source snapshot",
        directory_mode=0o555,
        file_mode=0o444,
        lock_exclusive=False,
    ) as snapshot_tree:
        inventory = snapshot_tree.inventory
        identity = directory_identity(snapshot_tree.root.before)
    hook = inventory.get(MODEL_AUTHORITY_HOOK_PATH)
    require(type(hook) is dict and hook.get("kind") == "file"
            and hook.get("mode") == 0o444,
            "authenticated source snapshot lacks the exact model-authority hook")
    inventory_sha256 = stable_hash(inventory)
    request = model_authority_request(
        manifest, authority, runtime_lock, inventory_sha256
    )
    request_sha256 = stable_hash(request)
    output = execute_model_authority_hook(
        runtime_lock, snapshot_path, inventory, request
    )
    require_exact_keys(
        output, {"schema_version", "interface", "request_sha256", "settings"},
        "model-authority hook stdout",
    )
    require(type(output["schema_version"]) is int and output["schema_version"] == 1
            and output["interface"] == MODEL_AUTHORITY_HOOK_INTERFACE
            and output["request_sha256"] == request_sha256,
            "model-authority hook stdout header differs")
    production_ready = expected_external_hook_source_sha256 is not None
    if production_ready:
        require_sha256(
            expected_external_hook_source_sha256,
            "externally reviewed model-authority hook SHA256",
        )
        require(runtime_lock.get("production_ready") is True
                and expected_external_hook_source_sha256 == hook["sha256"],
                "production model authority lacks the exact external hook pin")
    value: dict[str, Any] = {
        "schema_version": 1,
        "status": MODEL_AUTHORITY_STATUS,
        "campaign_id": CAMPAIGN_ID,
        "manifest_file_sha256": file_sha256(MANIFEST_PATH),
        "source_sha256": authority["roots"]["source_sha256"],
        "config_sha256": authority["roots"]["config_sha256"],
        "runtime_content_sha256": runtime_lock["runtime_content_sha256"],
        "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
        "snapshot_root": str(snapshot_path),
        "snapshot_root_identity": identity,
        "snapshot_inventory_sha256": inventory_sha256,
        "hook_relative_path": MODEL_AUTHORITY_HOOK_PATH,
        "hook_source_sha256": hook["sha256"],
        "hook_interface": MODEL_AUTHORITY_HOOK_INTERFACE,
        "execution_profile": (
            MODEL_PRODUCTION_PROFILE if production_ready else MODEL_UNSEALED_PROFILE
        ),
        "production_ready": production_ready,
        "external_hook_source_sha256": (
            expected_external_hook_source_sha256 if production_ready else None
        ),
        "derivation_request_sha256": request_sha256,
        "derivation_stdout_sha256": stable_hash(output),
        "deterministic_construction": dict(request["construction"]),
        "settings": output["settings"],
    }
    value["model_state_authority_sha256"] = stable_hash(value)
    cache_key = (
        value["model_state_authority_sha256"],
        inventory_sha256,
        runtime_lock["runtime_lock_sha256"],
    )
    _MODEL_HOOK_VALIDATION_CACHE[cache_key] = canonical_json(output)
    validate_model_state_authority(
        value, manifest, authority, runtime_lock, inventory
    )
    return value


def validate_model_state_authority(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    snapshot_inventory: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validate output from the separately reviewed source-snapshot model hook."""
    validate_runtime_lock(runtime_lock, manifest)
    require_exact_keys(value, MODEL_AUTHORITY_FIELDS, "model-state authority")
    require_int(value["schema_version"], "model-state authority schema")
    require(value["schema_version"] == 1 and value["status"] == MODEL_AUTHORITY_STATUS,
            "model-state authority is not sealed")
    require(value["campaign_id"] == CAMPAIGN_ID, "model-state authority campaign differs")
    expected_scalars = {
        "manifest_file_sha256": file_sha256(MANIFEST_PATH),
        "source_sha256": authority["roots"]["source_sha256"],
        "config_sha256": authority["roots"]["config_sha256"],
        "runtime_content_sha256": runtime_lock["runtime_content_sha256"],
        "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
        "snapshot_inventory_sha256": stable_hash(snapshot_inventory),
        "hook_relative_path": MODEL_AUTHORITY_HOOK_PATH,
        "hook_interface": MODEL_AUTHORITY_HOOK_INTERFACE,
    }
    for name, expected in expected_scalars.items():
        require(type(value[name]) is type(expected) and value[name] == expected,
                f"model-state authority {name} differs")
    hook = snapshot_inventory.get(MODEL_AUTHORITY_HOOK_PATH)
    require(type(hook) is dict and hook.get("kind") == "file"
            and hook.get("mode") == 0o444,
            "authenticated source snapshot lacks the exact model-authority hook")
    require_sha256(value["hook_source_sha256"], "model-authority hook SHA256")
    require(value["hook_source_sha256"] == hook.get("sha256"),
            "model-authority hook raw bytes differ")
    profile = require_string(
        value["execution_profile"], "model-state authority execution profile"
    )
    production_ready = require_bool(
        value["production_ready"], "model-state authority production readiness"
    )
    if profile == MODEL_PRODUCTION_PROFILE:
        require(production_ready is True
                and runtime_lock.get("production_ready") is True,
                "production model authority lacks a production runtime")
        require_sha256(
            value["external_hook_source_sha256"],
            "external model-authority hook SHA256",
        )
        require(value["external_hook_source_sha256"] == value["hook_source_sha256"],
                "external model-authority hook pin differs")
    else:
        require(profile == MODEL_UNSEALED_PROFILE and production_ready is False
                and value["external_hook_source_sha256"] is None,
                "unsealed model authority claims production readiness")
    snapshot_root = Path(require_string(value["snapshot_root"], "model snapshot root"))
    require(snapshot_root.is_absolute() and ".." not in snapshot_root.parts,
            "model snapshot root path differs")
    observed_snapshot_identity = nofollow_directory_identity(
        snapshot_root, "model-authority source snapshot"
    )
    _validate_identity_object(value["snapshot_root_identity"], "model snapshot identity")
    require(canonical_json(value["snapshot_root_identity"])
            == canonical_json(observed_snapshot_identity),
            "model snapshot identity differs")
    construction = require_exact_keys(
        value["deterministic_construction"],
        {"model_class", "initialization", "device", "layout", "storage_alias_policy"},
        "model deterministic construction",
    )
    require(construction == {
        "model_class": "TreeWM",
        "initialization": "deterministic_zero_init_before_optimizer_or_data",
        "device": "cpu",
        "layout": "strided",
        "storage_alias_policy": "unique_exact_storage",
    }, "model deterministic construction differs")
    request = model_authority_request(
        manifest, authority, runtime_lock, value["snapshot_inventory_sha256"]
    )
    require_sha256(value["derivation_request_sha256"],
                   "model derivation request SHA256")
    require(value["derivation_request_sha256"] == stable_hash(request),
            "model derivation request hash differs")
    require_sha256(value["derivation_stdout_sha256"],
                   "model derivation stdout SHA256")
    settings = value["settings"]
    require(type(settings) is list and len(settings) == len(SETTINGS),
            "model-state authority settings differ")
    for index, (row, expected) in enumerate(zip(settings, SETTINGS, strict=True)):
        item = require_exact_keys(
            row,
            {"setting_id", "env_config", "resolved_config_contract_sha256_by_seed",
             "parameter_schema"},
            f"model-state setting {index}",
        )
        require((item["setting_id"], item["env_config"]) == expected,
                f"model-state setting {index} identity differs")
        expected_contracts = authority["settings"][index][
            "resolved_config_contract_sha256_by_seed"
        ]
        require(canonical_json(item["resolved_config_contract_sha256_by_seed"])
                == canonical_json(expected_contracts),
                f"model-state setting {index} config contracts differ")
        validate_model_parameter_schema(
            item["parameter_schema"], f"model-state setting {index} schema"
        )
    stdout_body = {
        "schema_version": 1,
        "interface": MODEL_AUTHORITY_HOOK_INTERFACE,
        "request_sha256": value["derivation_request_sha256"],
        "settings": settings,
    }
    require(value["derivation_stdout_sha256"] == stable_hash(stdout_body),
            "model derivation stdout hash differs")
    cache_key = (
        value.get("model_state_authority_sha256"),
        value["snapshot_inventory_sha256"],
        runtime_lock["runtime_lock_sha256"],
    )
    observed_bytes = _MODEL_HOOK_VALIDATION_CACHE.get(cache_key)
    if observed_bytes is None:
        observed_stdout = execute_model_authority_hook(
            runtime_lock, snapshot_root, snapshot_inventory, request
        )
        observed_bytes = canonical_json(observed_stdout)
        _MODEL_HOOK_VALIDATION_CACHE[cache_key] = observed_bytes
    require(observed_bytes == canonical_json(stdout_body),
            "model-authority hook output differs from the sealed derivation")
    body = dict(value)
    claimed = body.pop("model_state_authority_sha256", None)
    require_sha256(claimed, "model-state authority SHA256")
    require(claimed == stable_hash(body), "model-state authority self-hash differs")
    return value


def require_production_model_authority(
    model_authority: Mapping[str, Any], label: str
) -> None:
    require(
        model_authority.get("execution_profile") == MODEL_PRODUCTION_PROFILE
        and model_authority.get("production_ready") is True
        and model_authority.get("external_hook_source_sha256")
        == model_authority.get("hook_source_sha256"),
        f"{label} requires an externally reviewed production model-state authority",
    )


def setting_model_schema(
    model_authority: Mapping[str, Any], setting_id: str
) -> Mapping[str, Any]:
    rows = [row for row in model_authority["settings"] if row["setting_id"] == setting_id]
    require(len(rows) == 1, f"model-state authority row count differs for {setting_id}")
    return rows[0]["parameter_schema"]


TERMINAL_CENSUS_FIELDS = {
    "schema_version", "status", "campaign_id", "manifest_file_sha256",
    "authority_sha256", "runtime_lock_sha256", "model_state_authority_sha256",
    "result_creation_receipt_sha256", "scheduler_protocol_sha256",
    "evidence_relative_path", "evidence_file_identity", "evidence_sha256",
    "submission_receipt", "submission_receipt_sha256",
    "production_ready", "external_anchor_sha256",
    "tasks", "tasks_sha256",
    "terminal_census_sha256",
}
TERMINAL_SUBMISSION_STATUS = "sealed_exp24_two_whole_array_submission_receipt_v1"
TERMINAL_SUBMISSION_FIELDS = {
    "schema_version", "status", "campaign_id", "authority_sha256",
    "scheduler_protocol_sha256", "wave0", "wave1", "report",
    "external_submission_evidence_sha256",
}


def terminal_census_evidence_bytes(
    tasks: Sequence[Mapping[str, Any]],
    submission_receipt: Mapping[str, Any],
) -> bytes:
    return canonical_json({
        "schema_version": 1,
        "source": "slurm_sacct_parsable2_exact_external_authority",
        "campaign_id": CAMPAIGN_ID,
        "submission_receipt": dict(submission_receipt),
        "tasks": list(tasks),
    }) + b"\n"


def validate_terminal_census(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    runtime_lock: Mapping[str, Any],
    model_authority: Mapping[str, Any],
    result_creation_receipt_sha256: str,
) -> Mapping[str, Any]:
    """Consume outer scheduler evidence; this leaf never infers scheduler state."""
    require_exact_keys(value, TERMINAL_CENSUS_FIELDS, "scheduler terminal census")
    require_int(value["schema_version"], "scheduler terminal census schema")
    require(value["schema_version"] == 1 and value["status"] == TERMINAL_CENSUS_STATUS,
            "scheduler terminal census is unavailable or unsealed")
    expected = {
        "campaign_id": CAMPAIGN_ID,
        "manifest_file_sha256": file_sha256(MANIFEST_PATH),
        "authority_sha256": authority["authority_sha256"],
        "runtime_lock_sha256": runtime_lock["runtime_lock_sha256"],
        "model_state_authority_sha256": model_authority[
            "model_state_authority_sha256"
        ],
        "result_creation_receipt_sha256": require_sha256(
            result_creation_receipt_sha256, "result creation receipt SHA256"
        ),
        "scheduler_protocol_sha256": authority["roots"]["protocol_sha256"],
    }
    for name, expected_value in expected.items():
        require(type(value[name]) is type(expected_value) and value[name] == expected_value,
                f"scheduler terminal census {name} differs")
    evidence_relative = safe_relative(
        value["evidence_relative_path"], "scheduler evidence relative path"
    )
    require(str(evidence_relative) == "scheduler/terminal-census.raw.json",
            "scheduler evidence relative path differs")
    evidence_identity = require_exact_keys(
        value["evidence_file_identity"],
        {"device", "inode", "mode", "uid", "gid", "nlink", "size"},
        "scheduler evidence file identity",
    )
    for name in ("device", "inode", "mode", "uid", "gid", "nlink", "size"):
        require_int(evidence_identity[name], f"scheduler evidence identity {name}", minimum=0)
    require(evidence_identity["mode"] == 0o444
            and evidence_identity["uid"] == os.getuid()
            and evidence_identity["nlink"] == 1
            and evidence_identity["size"] > 0,
            "scheduler evidence file authority differs")
    require_sha256(value["evidence_sha256"], "scheduler evidence SHA256")
    production_ready = require_bool(
        value["production_ready"], "scheduler terminal production readiness"
    )
    if production_ready:
        require_sha256(
            value["external_anchor_sha256"],
            "scheduler terminal external authority anchor SHA256",
        )
    else:
        require(value["external_anchor_sha256"] is None,
                "synthetic scheduler terminal census claims an external anchor")
    submission = require_exact_keys(
        value["submission_receipt"],
        TERMINAL_SUBMISSION_FIELDS,
        "scheduler two-array submission receipt",
    )
    require_int(submission["schema_version"], "scheduler submission schema")
    require(submission["schema_version"] == 1
            and submission["status"] == TERMINAL_SUBMISSION_STATUS
            and submission["campaign_id"] == CAMPAIGN_ID
            and submission["authority_sha256"] == authority["authority_sha256"]
            and submission["scheduler_protocol_sha256"]
            == authority["roots"]["protocol_sha256"],
            "scheduler submission receipt authority differs")
    wave0 = require_exact_keys(
        submission["wave0"],
        {"array_job_id", "array_spec", "task_count"},
        "scheduler wave-zero submission",
    )
    wave1 = require_exact_keys(
        submission["wave1"],
        {"array_job_id", "array_spec", "task_count", "dependency",
         "kill_on_invalid_dependency"},
        "scheduler wave-one submission",
    )
    report_submission = require_exact_keys(
        submission["report"], {"dependency"}, "scheduler report submission"
    )
    wave0_job = require_string(wave0["array_job_id"], "wave-zero array job ID")
    wave1_job = require_string(wave1["array_job_id"], "wave-one array job ID")
    require(re.fullmatch(r"[1-9][0-9]*", wave0_job) is not None
            and re.fullmatch(r"[1-9][0-9]*", wave1_job) is not None
            and wave0_job != wave1_job,
            "scheduler wave array job IDs are invalid or not distinct")
    require(wave0["array_spec"] == wave1["array_spec"] == "0-19%20"
            and type(wave0["task_count"]) is int and wave0["task_count"] == 20
            and type(wave1["task_count"]) is int and wave1["task_count"] == 20,
            "scheduler whole-array topology differs")
    require(wave1["dependency"] == f"afterok:{wave0_job}"
            and type(wave1["kill_on_invalid_dependency"]) is bool
            and wave1["kill_on_invalid_dependency"] is True,
            "scheduler wave-one exact afterok dependency differs")
    require(report_submission["dependency"] == f"afterok:{wave1_job}",
            "scheduler report exact afterok dependency differs")
    require_sha256(
        submission["external_submission_evidence_sha256"],
        "scheduler external submission evidence SHA256",
    )
    require_sha256(value["submission_receipt_sha256"],
                   "scheduler submission receipt SHA256")
    require(value["submission_receipt_sha256"] == stable_hash(submission),
            "scheduler submission receipt hash differs")
    tasks = value["tasks"]
    require(type(tasks) is list and len(tasks) == 40,
            "scheduler terminal census does not contain exactly 40 tasks")
    seen: set[tuple[int, int]] = set()
    job_tasks: set[str] = set()
    wave_jobs: dict[int, set[str]] = {0: set(), 1: set()}
    for row, expected_cell in zip(tasks, [cell for wave in (0, 1)
                                           for cell in expand_cells(manifest)], strict=True):
        item = require_exact_keys(
            row,
            {"wave_index", "cell_index", "setting_id", "seed", "array_job_id",
             "array_task_id", "state", "exit_code", "sacct_row_sha256"},
            "scheduler terminal task",
        )
        wave = require_int(item["wave_index"], "scheduler task wave")
        cell_index = require_int(item["cell_index"], "scheduler task cell")
        task_index = require_int(item["array_task_id"], "scheduler array task")
        require(wave in (0, 1) and cell_index == expected_cell.index == task_index,
                "scheduler terminal task topology differs")
        require(item["setting_id"] == expected_cell.setting_id
                and type(item["seed"]) is int and item["seed"] == expected_cell.seed,
                "scheduler terminal task cell identity differs")
        job_id = require_string(item["array_job_id"], "scheduler array job ID")
        require(re.fullmatch(r"[1-9][0-9]*", job_id) is not None,
                "scheduler array job ID is invalid")
        require(item["state"] == "COMPLETED" and item["exit_code"] == "0:0",
                "scheduler task is not terminal-successful")
        require_sha256(item["sacct_row_sha256"], "scheduler sacct row SHA256")
        require(item["sacct_row_sha256"] == stable_hash({
            key: item[key] for key in (
                "wave_index", "cell_index", "setting_id", "seed", "array_job_id",
                "array_task_id", "state", "exit_code"
            )
        }), "scheduler task evidence hash differs")
        require((wave, cell_index) not in seen, "scheduler terminal task is duplicated")
        seen.add((wave, cell_index))
        job_tasks.add(f"{job_id}_{task_index}")
        wave_jobs[wave].add(job_id)
    require(len(seen) == len(job_tasks) == 40,
            "scheduler terminal census task identities are not unique")
    require(wave_jobs == {0: {wave0_job}, 1: {wave1_job}},
            "scheduler terminal census is not exactly two whole arrays")
    require_sha256(value["tasks_sha256"], "scheduler terminal tasks SHA256")
    require(value["tasks_sha256"] == stable_hash(tasks),
            "scheduler terminal tasks hash differs")
    scratch = DirectoryCapability(
        authority["environment"]["scratch_root"], "scheduler evidence scratch root"
    )
    evidence_fd: int | None = None
    try:
        evidence_fd, before = scratch.open_regular(
            evidence_relative, "scheduler terminal evidence", mode=0o444
        )
        observed_identity = {
            "device": int(before.st_dev),
            "inode": int(before.st_ino),
            "mode": int(stat.S_IMODE(before.st_mode)),
            "uid": int(before.st_uid),
            "gid": int(before.st_gid),
            "nlink": int(before.st_nlink),
            "size": int(before.st_size),
        }
        require(canonical_json(evidence_identity) == canonical_json(observed_identity),
                "scheduler evidence file identity differs")
        observed_sha, observed_size = hash_descriptor(
            evidence_fd, before, "scheduler terminal evidence"
        )
        require(observed_sha == value["evidence_sha256"]
                and observed_size == evidence_identity["size"],
                "scheduler evidence bytes differ")
        raw = os.pread(evidence_fd, before.st_size, 0)
        require(len(raw) == before.st_size
                and raw == terminal_census_evidence_bytes(tasks, submission),
                "scheduler evidence encoding or contents differ")
        scratch.require_named_identity(
            evidence_relative, before, "scheduler terminal evidence"
        )
    finally:
        if evidence_fd is not None:
            os.close(evidence_fd)
        scratch.close()
    body = dict(value)
    claimed = body.pop("terminal_census_sha256", None)
    require_sha256(claimed, "scheduler terminal census SHA256")
    require(claimed == stable_hash(body), "scheduler terminal census self-hash differs")
    return value


AUTHORITY_SETTING_KEYS = {
    "setting_id", "env_config", "env_name", "source_name", "dataset_kind",
    "input_contract_sha256", "data_manifest_sha256",
    "calibration_sha256", "future_recipe_sha256", "recipe_code_sha256",
    "recipe_runtime_sha256", "published_union_train_anchors",
    "published_union_validation_anchors", "action_lower_bound", "action_upper_bound",
    "max_environment_steps", "resolved_config_contract_sha256_by_seed",
}


def validate_authority(
    value: Mapping[str, Any],
    manifest: Mapping[str, Any],
    runtime_lock: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    require(runtime_lock is not None,
            "sealed runtime-content lock prerequisite is required")
    validate_runtime_lock(runtime_lock, manifest)
    require_exact_keys(value, {
        "schema_version", "status", "campaign_id", "manifest_file_sha256", "roots",
        "seed_collision_census", "environment", "settings", "authority_sha256",
    }, "calibration authority")
    require_int(value["schema_version"], "authority schema")
    require(value["schema_version"] == 1, "authority schema differs")
    require(value["status"] == "sealed_zero_prefix_calibration_authorities",
            "calibration authority is not sealed")
    require(value["campaign_id"] == CAMPAIGN_ID, "authority campaign differs")
    require_sha256(value["manifest_file_sha256"], "authority manifest file SHA256")
    require(value["manifest_file_sha256"] == file_sha256(MANIFEST_PATH),
            "authority calibration manifest bytes drifted")
    roots = require_exact_keys(value["roots"], set(ROOT_NAMES), "authority roots")
    for name in ROOT_NAMES:
        require_sha256(roots[name], f"authority root {name}")
    require(roots["runtime_sha256"] == runtime_lock["runtime_content_sha256"],
            "authority runtime root is not derived from the sealed runtime lock")

    environment = require_exact_keys(value["environment"], {
        "python", "python_sha256", "data_root", "cache_root", "future_recipe_root",
        "scratch_root", "scratch_root_identity", "formal_output_root",
        "formal_output_root_identity", "snapshot_relative_path",
        "result_relative_path", "trainer_sha256",
        "WANDB_MODE", "MUJOCO_GL", "XLA_PYTHON_CLIENT_PREALLOCATE",
        "PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS",
        "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    }, "calibration environment")
    for name in (
        "python", "data_root", "cache_root", "future_recipe_root",
        "scratch_root", "formal_output_root",
    ):
        require(isinstance(environment[name], str) and Path(environment[name]).is_absolute(),
                f"calibration environment {name} is not absolute")
        require(".." not in Path(environment[name]).parts,
                f"calibration environment {name} contains traversal")
    scratch_identity = require_exact_keys(
        environment["scratch_root_identity"],
        {"device", "inode", "mode", "uid", "gid"},
        "calibration scratch root identity",
    )
    formal_identity = require_exact_keys(
        environment["formal_output_root_identity"],
        {"device", "inode", "mode", "uid", "gid"},
        "formal output root identity",
    )
    for label, identity in (("scratch", scratch_identity), ("formal", formal_identity)):
        for name in ("device", "inode", "mode", "uid", "gid"):
            require_int(identity[name], f"{label} root identity {name}", minimum=0)
    require(scratch_identity["mode"] == 0o700
            and scratch_identity["uid"] == os.getuid(),
            "authorized scratch root is not an owner-only mode-0700 boundary")
    require(
        dict(scratch_identity)
        == nofollow_directory_identity(environment["scratch_root"], "authorized scratch root"),
        "authorized scratch root identity differs",
    )
    require(
        dict(formal_identity)
        == nofollow_directory_identity(environment["formal_output_root"], "formal output root"),
        "formal output root identity differs",
    )
    require(dict(scratch_identity) != dict(formal_identity),
            "authorized scratch aliases formal output root")
    scratch_path = Path(os.path.normpath(environment["scratch_root"]))
    formal_path = Path(os.path.normpath(environment["formal_output_root"]))
    require(not (scratch_path == formal_path or scratch_path.is_relative_to(formal_path)
                 or formal_path.is_relative_to(scratch_path)),
            "authorized scratch and formal output roots overlap")
    require_sha256(environment["python_sha256"], "calibration Python SHA256")
    expected_python = (
        Path(runtime_lock["runtime_root"])
        / runtime_lock["interpreter"]["relative_path"]
    )
    require(Path(environment["python"]) == expected_python
            and environment["python_sha256"] == runtime_lock["interpreter"]["sha256"],
            "calibration Python is not derived from the runtime-content lock")
    require_sha256(environment["trainer_sha256"], "calibration trainer SHA256")
    expected_snapshot_relative = f"snapshots/{roots['source_sha256']}"
    require(environment["snapshot_relative_path"] == expected_snapshot_relative,
            "calibration snapshot relative path differs")
    require(environment["result_relative_path"]
            == f"results/{CAMPAIGN_ID}",
            "calibration result relative path differs")
    safe_relative(environment["snapshot_relative_path"], "snapshot relative path")
    safe_relative(environment["result_relative_path"], "result relative path")
    require(environment["WANDB_MODE"] == "disabled", "calibration W&B is not disabled")
    require(environment["MUJOCO_GL"] == "egl", "calibration MUJOCO_GL differs")
    require(environment["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false",
            "calibration XLA preallocation differs")
    for name in ("PYTHONNOUSERSITE", "PYTHONDONTWRITEBYTECODE", "OMP_NUM_THREADS",
                 "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        require(environment[name] == "1", f"calibration environment {name} differs")
    require(roots["environment_sha256"] == stable_hash(environment),
            "calibration environment root differs")

    census = require_exact_keys(value["seed_collision_census"], {
        "schema_version", "status", "campaign_id", "seeds", "scope",
        "repository_head", "git_ref_inventory_sha256", "worktree_inventory_sha256",
        "reachable_history_inventory_sha256", "prior_assignment_inventory_sha256",
        "prior_assignment_matches", "performed_before_first_calibration_run",
        "scope_sha256", "evidence_sha256", "census_sha256",
    }, "seed collision census")
    require_int(census["schema_version"], "seed census schema")
    require(census["schema_version"] == 1, "seed census schema differs")
    require(census["status"] == "sealed_exact_zero_prior_assignment_collision_census",
            "seed collision census is not sealed")
    require(census["campaign_id"] == CAMPAIGN_ID, "seed census campaign differs")
    require(census["seeds"] == list(SEEDS), "seed collision census bank differs")
    require(all(type(seed) is int for seed in census["seeds"]),
            "seed collision census seed types differ")
    require_int(census["prior_assignment_matches"], "seed prior assignment count")
    require(census["prior_assignment_matches"] == 0, "calibration seed collision detected")
    require(census["performed_before_first_calibration_run"] is True,
            "seed census was not performed before the first run")
    require_bool(census["performed_before_first_calibration_run"],
                 "seed census timing")
    require(census["scope"] == SEED_CENSUS_SCOPE,
            "seed census scope differs from preregistration")
    require_string(census["repository_head"], "seed census repository head")
    require(re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", census["repository_head"]) is not None,
            "seed census repository head is not a Git object ID")
    for name in (
        "git_ref_inventory_sha256", "worktree_inventory_sha256",
        "reachable_history_inventory_sha256", "prior_assignment_inventory_sha256",
    ):
        require_sha256(census[name], f"seed census {name}")
    require_sha256(census["scope_sha256"], "seed census scope SHA256")
    expected_scope_sha = seed_census_scope_sha256()
    require(census["scope_sha256"] == expected_scope_sha,
            "seed census scope digest differs")
    require_sha256(census["evidence_sha256"], "seed census evidence SHA256")
    expected_evidence_sha = seed_census_evidence_sha256(census)
    require(census["evidence_sha256"] == expected_evidence_sha,
            "seed census evidence digest differs")
    census_body = dict(census)
    claimed_census = census_body.pop("census_sha256", None)
    require_sha256(claimed_census, "seed census SHA256")
    require(claimed_census == stable_hash(census_body), "seed census self-hash differs")

    settings = value["settings"]
    require(isinstance(settings, list) and len(settings) == 10,
            "authority does not contain exactly ten setting rows")
    for index, (row, expected) in enumerate(zip(settings, SETTINGS, strict=True)):
        item = require_exact_keys(row, AUTHORITY_SETTING_KEYS, f"authority setting {index}")
        require((item["setting_id"], item["env_config"]) == expected,
                f"authority setting {index} identity differs")
        metadata = SETTING_METADATA[item["setting_id"]]
        for name in ("env_name", "source_name", "dataset_kind"):
            require(type(item[name]) is str and item[name] == metadata[name],
                    f"authority setting {index} {name} differs")
        for name in (
            "input_contract_sha256", "data_manifest_sha256", "calibration_sha256",
            "future_recipe_sha256", "recipe_code_sha256", "recipe_runtime_sha256",
        ):
            require_sha256(item[name], f"authority setting {index} {name}")
        require_int(item["published_union_train_anchors"],
                    f"authority setting {index} train anchors", minimum=1)
        require_int(item["published_union_validation_anchors"],
                    f"authority setting {index} validation anchors", minimum=1)
        require_int(item["max_environment_steps"],
                    f"authority setting {index} max environment steps", minimum=1)
        config_contracts = require_exact_keys(
            item["resolved_config_contract_sha256_by_seed"],
            {str(seed) for seed in SEEDS},
            f"authority setting {index} resolved config contracts",
        )
        for seed in SEEDS:
            require_sha256(
                config_contracts[str(seed)],
                f"authority setting {index} seed {seed} resolved config contract",
            )
        for name in ("action_lower_bound", "action_upper_bound"):
            number = item[name]
            require(type(number) is float and math.isfinite(number),
                    f"authority setting {index} {name} is invalid")
        require(float(item["action_lower_bound"]) < float(item["action_upper_bound"]),
                f"authority setting {index} action bounds are reversed")
    require(len({row["input_contract_sha256"] for row in settings}) == 10,
            "authority input contracts are not per-setting unique")
    require(len({row["future_recipe_sha256"] for row in settings}) == 10,
            "authority future recipes are not per-setting unique")
    require(roots["config_sha256"] == logical_config_sha256(manifest, value),
            "calibration logical config root differs")

    body = dict(value)
    claimed = body.pop("authority_sha256", None)
    require_sha256(claimed, "calibration authority SHA256")
    require(claimed == stable_hash(body), "calibration authority self-hash differs")
    return value


def setting_authority(authority: Mapping[str, Any], setting_id: str) -> Mapping[str, Any]:
    matches = [row for row in authority["settings"] if row["setting_id"] == setting_id]
    require(len(matches) == 1, f"authority row count differs for {setting_id}")
    return matches[0]


def _override(name: str, value: object) -> str:
    if value is True:
        rendered = "true"
    elif value is False:
        rendered = "false"
    elif value is None:
        rendered = "null"
    elif isinstance(value, list):
        rendered = "[" + ",".join(str(item) for item in value) + "]"
    else:
        rendered = str(value)
    return f"{name}={rendered}"


def scientific_overrides(
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
    cell: CalibrationCell,
    run_root: str | Path,
) -> list[str]:
    recipe = manifest["recipe"]
    setting = setting_authority(authority, cell.setting_id)
    values: list[tuple[str, object]] = [
        ("env", cell.env_config),
        ("experiment", recipe["experiment_config"]),
        ("arm", "treewm"),
        ("objective_version", OBJECTIVE),
        ("seed", cell.seed),
        ("train.steps", 25_000),
        ("train.scheduler_total_steps", 1_000_000),
        ("train.ckpt_every", 5_000),
        ("train.val_every", 25_000),
        ("train.diag_every", 5_000),
        ("train.log_every", 50),
        ("train.eval_every", 25_000),
        ("train.viz_every", 25_000),
        ("train.viz_every_early", 25_000),
        ("train.viz_early_until", 0),
        ("train.max_train_anchors", setting["published_union_train_anchors"]),
        ("train.max_val_anchors", setting["published_union_validation_anchors"]),
        ("future_sets.recipe_anchor_policy", "published_union"),
        ("future_sets.cache", False),
        ("future_sets.shared_cache", True),
        ("future_sets.executable_prefix_steps", 4),
        ("losses.enabled.executable_prefix_action", True),
        ("losses.enabled.executable_prefix_latent", True),
        ("losses.enabled.executable_prefix_endpoint", True),
        ("losses.weights.executable_prefix_action", 0.0),
        ("losses.weights.executable_prefix_latent", 0.0),
        ("losses.weights.executable_prefix_endpoint", 0.0),
        ("losses.executable_action_lower_bound", setting["action_lower_bound"]),
        ("losses.executable_action_upper_bound", setting["action_upper_bound"]),
        ("planner.action_lower_bound", setting["action_lower_bound"]),
        ("planner.action_upper_bound", setting["action_upper_bound"]),
        ("planner.max_env_steps", setting["max_environment_steps"]),
        ("eval.episodes_per_task", 1),
        ("eval.final_episodes_per_task", 1),
        ("+campaign_id", CAMPAIGN_ID),
        ("+campaign_source_sha256", authority["roots"]["source_sha256"]),
        ("+campaign_protocol_sha256", authority["roots"]["protocol_sha256"]),
        ("+campaign_config_sha256", authority["roots"]["config_sha256"]),
        ("+campaign_input_contract_sha256", setting["input_contract_sha256"]),
        ("+campaign_factorial_arm", "zero_prefix_calibration"),
        ("+campaign_prerequisite_binding_sha256", authority["authority_sha256"]),
        ("+campaign_selected_recipe_sha256", setting["future_recipe_sha256"]),
        ("run_root", Path(run_root).as_posix()),
        ("run_name", None),
        ("resume", "auto"),
    ]
    return [_override(name, value) for name, value in values]


def logical_config_sha256(
    manifest: Mapping[str, Any], authority: Mapping[str, Any]
) -> str:
    # The config root authenticates all scientific leaves while excluding the two
    # values that necessarily refer back to the root/authority being constructed.
    shadow = dict(authority)
    shadow["roots"] = {**authority["roots"], "config_sha256": "0" * 64}
    shadow["authority_sha256"] = "0" * 64
    rows = []
    for cell in expand_cells(manifest):
        overrides = scientific_overrides(manifest, shadow, cell, "/SCRATCH_ROOT")
        normalized = []
        for item in overrides:
            if item.startswith("+campaign_config_sha256="):
                item = "+campaign_config_sha256=<CONFIG_ROOT>"
            elif item.startswith("+campaign_prerequisite_binding_sha256="):
                item = "+campaign_prerequisite_binding_sha256=<AUTHORITY>"
            normalized.append(item)
        rows.append({"cell": asdict(cell), "overrides": normalized})
    return stable_hash({"schema_version": 1, "cells": rows})


def safe_relative(value: object, label: str) -> PurePosixPath:
    require(isinstance(value, str), f"{label} is not a string")
    path = PurePosixPath(value)
    require(not path.is_absolute() and str(path) not in ("", "."),
            f"{label} is not a nonempty relative path")
    require(all(part not in ("", ".", "..") for part in path.parts),
            f"{label} contains traversal")
    return path


def cell_description(cell: CalibrationCell) -> dict[str, Any]:
    return asdict(cell)


def describe(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "calibration_source_ready_execution_blocked_on_unsealed_authorities",
        "production_readiness": False,
        "blocked_on": [
            "sealed_runtime_content_lock",
            "reviewed_production_model_authority_hook_and_sealed_model_state_authority",
            "sealed_result_creation_receipt",
            "sealed_external_scheduler_terminal_census",
        ],
        "synthetic_fixture_is_production_authority": False,
        "campaign_id": CAMPAIGN_ID,
        "manifest_file_sha256": file_sha256(MANIFEST_PATH),
        "cell_count": len(expand_cells(manifest)),
        "seeds": list(SEEDS),
        "settings": [setting for setting, _ in SETTINGS],
        "nominal_updates": 25_000,
        "export_update": 5_000,
        "scheduler_horizon": 1_000_000,
        "zero_prefix_weights_with_graph_enabled": True,
        "outcome_measurement_allowed": False,
        "wandb_mode": "disabled",
        "topology": {
            "wave0": "0-19%20",
            "wave1": "0-19%20",
            "dependency": "afterok_exact_whole_wave0_array",
            "third_wave_allowed": False,
        },
        "checkpoint_lock": PLACEHOLDER_PATH.name,
        "checkpoint_lock_state": "unsealed",
        "seed_census_lock": SEED_CENSUS_PLACEHOLDER_PATH.name,
        "seed_census_lock_state": "unsealed",
        "persistent_writes_performed": False,
        "scheduler_calls_performed": False,
        "source_or_output_scan_performed": False,
    }


def _internal_decoder_entry(argv: Sequence[str]) -> int:
    if len(argv) == 4 and argv[0] == "--exp24-runtime-closure-probe":
        require(re.fullmatch(r"[0-9]+", argv[1]) is not None,
                "runtime closure root descriptor is invalid")
        require_sha256(argv[2], "runtime closure probe source SHA256")
        try:
            sys_paths = json.loads(argv[3])
        except json.JSONDecodeError as exc:
            raise CalibrationContractError(
                "runtime closure declared sys.path is invalid"
            ) from exc
        require(type(sys_paths) is list
                and all(type(path) is str for path in sys_paths),
                "runtime closure declared sys.path differs")
        return _runtime_closure_probe_main(int(argv[1]), argv[2], sys_paths)
    require(len(argv) == 3 and argv[0] == "--exp24-safe-checkpoint-decoder",
            "invalid internal calibration-contract invocation")
    require(re.fullmatch(r"[0-9]+", argv[1]) is not None,
            "safe checkpoint descriptor argument is invalid")
    checkpoint_fd = int(argv[1])
    fields = argv[2].split(",")
    require(fields and all(re.fullmatch(r"[0-9]+", field) is not None for field in fields),
            "safe checkpoint descriptor allowlist is invalid")
    allowed = {int(field) for field in fields}
    require(checkpoint_fd in allowed and {0, 1, 2}.issubset(allowed),
            "safe checkpoint descriptor allowlist is incomplete")
    return _checkpoint_decoder_main(checkpoint_fd, allowed)


if __name__ == "__main__":  # pragma: no cover - exercised through clean exec
    raise SystemExit(_internal_decoder_entry(sys.argv[1:]))
