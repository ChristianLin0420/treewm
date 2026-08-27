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
import stat
import sys
import tempfile
from typing import Any, Mapping


sys.dont_write_bytecode = True
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]
AUDIT_SOURCE_RELATIVE = PACKAGE_DIR.relative_to(PROJECT_ROOT) / Path(__file__).name
BOOTSTRAP_EXECUTION_ROOT_FD = globals().get("__treewm_execution_root_fd__")
BOOTSTRAP_EXECUTION_ROOT_IDENTITY = globals().get(
    "__treewm_execution_root_identity__"
)
BOOTSTRAP_EXECUTION_ROOT_SEALED = globals().get("__treewm_execution_root_sealed__")
BOOTSTRAP_EXECUTION_SCRIPT_RELATIVE = globals().get(
    "__treewm_execution_script_relative__"
)
BOOTSTRAP_EXECUTION_SCRIPT_SHA256 = globals().get(
    "__treewm_execution_script_sha256__"
)
BOOTSTRAP_EXECUTION_DIRECTORY_IDENTITIES = globals().get(
    "__treewm_execution_directory_identities__"
)
BOOTSTRAP_EXECUTION_SCRIPT_IDENTITY = globals().get(
    "__treewm_execution_script_identity__"
)
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


def _lstat_if_present(path: str | Path, label: str) -> os.stat_result | None:
    """Return one lexical entry, treating only ENOENT as absence."""

    try:
        return Path(path).lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ParityAuditError(f"cannot determine whether {label} exists: {exc}") from exc


def _lexical_exists(path: str | Path, label: str) -> bool:
    return _lstat_if_present(path, label) is not None


def _resolve_strict(path: str | Path, label: str) -> Path:
    try:
        return Path(path).resolve(strict=True)
    except OSError as exc:
        raise ParityAuditError(f"cannot resolve {label}: {exc}") from exc


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def _open_absolute_directory(path: Path, label: str) -> tuple[int, os.stat_result]:
    absolute = path.absolute()
    if (
        not absolute.is_absolute()
        or any(part in ("", ".", "..") for part in absolute.parts[1:])
        or os.fspath(absolute) != os.path.normpath(os.fspath(absolute))
    ):
        raise ParityAuditError(f"{label} is not an absolute normalized path")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor: int | None = None
    try:
        descriptor = os.open(absolute.anchor, flags)
        for part in absolute.parts[1:]:
            listed = os.stat(part, dir_fd=descriptor, follow_symlinks=False)
            child = os.open(part, flags, dir_fd=descriptor)
            try:
                opened = os.fstat(child)
                if (
                    not stat.S_ISDIR(listed.st_mode)
                    or _tree_stat_identity(opened) != _tree_stat_identity(listed)
                ):
                    raise ParityAuditError(
                        f"{label} component raced or is not a directory"
                    )
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        info = os.fstat(descriptor)
        result = descriptor
        descriptor = None
        return result, info
    except OSError as exc:
        raise ParityAuditError(f"cannot open {label} without symlinks: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _open_snapshot_fd_source(
    raw: str,
) -> tuple[
    int,
    os.stat_result,
    int,
    os.stat_result,
    tuple[tuple[int, str, int, os.stat_result], ...],
]:
    """Open the exact causal source below a retained sealed snapshot-root fd."""

    if (
        not raw.startswith("/proc/self/fd/")
        or raw.startswith("//")
        or "//" in raw
        or raw != os.path.normpath(raw)
    ):
        raise ParityAuditError("snapshot-fd hash source has an unsafe /proc spelling")
    parts = raw.split("/")
    if len(parts) < 6 or parts[:4] != ["", "proc", "self", "fd"]:
        raise ParityAuditError("snapshot-fd hash source has an invalid prefix")
    fd_token = parts[4]
    if (
        not fd_token
        or not fd_token.isascii()
        or not fd_token.isdecimal()
        or (fd_token != "0" and fd_token.startswith("0"))
    ):
        raise ParityAuditError("snapshot-fd hash source has a noncanonical descriptor")
    relative_parts = parts[5:]
    if not relative_parts or any(part in ("", ".", "..") for part in relative_parts):
        raise ParityAuditError("snapshot-fd hash source has an unsafe relative path")
    relative = Path(*relative_parts)
    if relative != AUDIT_SOURCE_RELATIVE:
        raise ParityAuditError("snapshot-fd hash source is not the causal auditor")
    expected_root_identity = BOOTSTRAP_EXECUTION_ROOT_IDENTITY
    expected_directory_identities = BOOTSTRAP_EXECUTION_DIRECTORY_IDENTITIES
    expected_script_identity = BOOTSTRAP_EXECUTION_SCRIPT_IDENTITY
    expected_directory_paths = tuple(
        Path(*relative.parts[:index]).as_posix()
        for index in range(1, len(relative.parts))
    )
    if (
        type(BOOTSTRAP_EXECUTION_ROOT_FD) is not int
        or BOOTSTRAP_EXECUTION_ROOT_FD < 0
        or int(fd_token, 10) != BOOTSTRAP_EXECUTION_ROOT_FD
        or not isinstance(expected_root_identity, tuple)
        or len(expected_root_identity) != 9
        or not all(type(value) is int for value in expected_root_identity)
        or type(BOOTSTRAP_EXECUTION_ROOT_SEALED) is not bool
        or BOOTSTRAP_EXECUTION_SCRIPT_RELATIVE != relative.as_posix()
        or not isinstance(BOOTSTRAP_EXECUTION_SCRIPT_SHA256, str)
        or len(BOOTSTRAP_EXECUTION_SCRIPT_SHA256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in BOOTSTRAP_EXECUTION_SCRIPT_SHA256
        )
        or not isinstance(expected_directory_identities, tuple)
        or len(expected_directory_identities) != len(expected_directory_paths)
        or any(
            not isinstance(record, tuple)
            or len(record) != 2
            or record[0] != expected_directory_paths[index]
            or not isinstance(record[1], tuple)
            or len(record[1]) != 9
            or not all(type(value) is int for value in record[1])
            for index, record in enumerate(expected_directory_identities)
        )
        or not isinstance(expected_script_identity, tuple)
        or len(expected_script_identity) != 9
        or not all(type(value) is int for value in expected_script_identity)
    ):
        raise ParityAuditError("snapshot-fd execution context is absent or malformed")

    try:
        root_fd = os.dup(int(fd_token, 10))
    except (OSError, ValueError) as exc:
        raise ParityAuditError(f"snapshot-root descriptor is unavailable: {exc}") from exc
    expected_fd: int | None = None
    descriptor: int | None = None
    directory_chain: list[tuple[int, str, int, os.stat_result]] = []
    root_transferred = False
    try:
        root_before = os.fstat(root_fd)
        root_mode = stat.S_IMODE(root_before.st_mode)
        sealed_context = BOOTSTRAP_EXECUTION_ROOT_SEALED
        if (
            not stat.S_ISDIR(root_before.st_mode)
            or root_before.st_uid != os.getuid()
            or root_mode != (0o555 if sealed_context else 0o755)
        ):
            raise ParityAuditError("snapshot-root descriptor type, owner, or mode differs")
        if _tree_stat_identity(root_before) != expected_root_identity:
            raise ParityAuditError("snapshot-root descriptor identity differs")
        expected_fd, expected = _open_absolute_directory(
            PROJECT_ROOT, "expected causal snapshot root"
        )
        if _tree_stat_identity(expected) != _tree_stat_identity(root_before):
            raise ParityAuditError("snapshot-root descriptor identity differs")
        os.close(expected_fd)
        expected_fd = None

        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        current_fd = root_fd
        relative_directories: list[str] = []
        for index, part in enumerate(relative.parts[:-1]):
            listed = os.stat(part, dir_fd=current_fd, follow_symlinks=False)
            child = os.open(part, directory_flags, dir_fd=current_fd)
            try:
                opened = os.fstat(child)
                opened_mode = stat.S_IMODE(opened.st_mode)
                if (
                    not stat.S_ISDIR(opened.st_mode)
                    or opened.st_uid != os.getuid()
                    or (
                        opened_mode != 0o555
                        if sealed_context
                        else (
                            opened_mode & 0o500 != 0o500
                            or opened_mode & 0o022 != 0
                            or opened_mode & 0o7000 != 0
                        )
                    )
                    or _tree_stat_identity(opened) != _tree_stat_identity(listed)
                ):
                    raise ParityAuditError(
                        "snapshot-fd source directory raced or is unsafe"
                    )
                relative_directories.append(part)
                if expected_directory_identities[index] != (
                    "/".join(relative_directories),
                    _tree_stat_identity(opened),
                ):
                    raise ParityAuditError(
                        "snapshot-fd source directory identity differs from bootstrap"
                    )
            except BaseException:
                os.close(child)
                raise
            directory_chain.append((current_fd, part, child, opened))
            current_fd = child
        listed_file = os.stat(
            relative.name, dir_fd=current_fd, follow_symlinks=False
        )
        descriptor = os.open(
            relative.name,
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=current_fd,
        )
        opened_file = os.fstat(descriptor)
        opened_file_mode = stat.S_IMODE(opened_file.st_mode)
        if (
            not stat.S_ISREG(opened_file.st_mode)
            or opened_file.st_uid != os.getuid()
            or opened_file.st_nlink != 1
            or (
                opened_file_mode != 0o444
                if sealed_context
                else (
                    opened_file_mode & 0o400 == 0
                    or opened_file_mode & 0o022 != 0
                    or opened_file_mode & 0o7000 != 0
                )
            )
            or _tree_stat_identity(opened_file) != _tree_stat_identity(listed_file)
            or _tree_stat_identity(opened_file) != expected_script_identity
        ):
            raise ParityAuditError("snapshot-fd causal source raced or is unsafe")
        result = descriptor
        descriptor = None
        root_transferred = True
        return result, opened_file, root_fd, root_before, tuple(directory_chain)
    except OSError as exc:
        raise ParityAuditError(f"cannot open snapshot-fd causal source: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if expected_fd is not None:
            os.close(expected_fd)
        if not root_transferred:
            for _parent, _name, child, _before in reversed(directory_chain):
                os.close(child)
            os.close(root_fd)


def file_sha256(path: str | Path) -> str:
    raw = os.fspath(path)
    snapshot_root_fd: int | None = None
    snapshot_root_before: os.stat_result | None = None
    snapshot_directory_chain: tuple[
        tuple[int, str, int, os.stat_result], ...
    ] = ()
    opened: os.stat_result | None = None
    if raw.startswith("/proc/") or raw.startswith("//proc/"):
        (
            descriptor,
            opened,
            snapshot_root_fd,
            snapshot_root_before,
            snapshot_directory_chain,
        ) = _open_snapshot_fd_source(raw)
        source = Path(raw)
    else:
        source = Path(path).absolute()
        descriptor = None
    if not source.is_absolute() or any(part in ("", ".", "..") for part in source.parts[1:]):
        raise ParityAuditError(f"hash source is not an absolute normalized path: {source}")
    if descriptor is None:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        parent_fd: int | None = None
        try:
            parent_fd = os.open(source.anchor, directory_flags)
            for part in source.parts[1:-1]:
                child = os.open(part, directory_flags, dir_fd=parent_fd)
                os.close(parent_fd)
                parent_fd = child
            descriptor = os.open(
                source.name,
                os.O_RDONLY
                | getattr(os, "O_NONBLOCK", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=parent_fd,
            )
        except OSError as exc:
            if descriptor is not None:
                os.close(descriptor)
            if parent_fd is not None:
                os.close(parent_fd)
            raise ParityAuditError(f"cannot open regular file for hashing {source}: {exc}") from exc
        assert descriptor is not None and parent_fd is not None
        os.close(parent_fd)
    digest = hashlib.sha256()
    try:
        current = os.fstat(descriptor)
        if opened is None:
            opened = current
        elif _tree_stat_identity(current) != _tree_stat_identity(opened):
            raise ParityAuditError(f"snapshot-fd source changed before hashing: {source}")
        if (
            not stat.S_ISREG(opened.st_mode)
            or stat.S_IMODE(opened.st_mode) & 0o444 == 0
        ):
            raise ParityAuditError(
                f"hash source is not a readable regular nonsymlink file: {source}"
            )
        while block := os.read(descriptor, 16 * 1024 * 1024):
            digest.update(block)
        after = os.fstat(descriptor)
        if _tree_stat_identity(after) != _tree_stat_identity(opened):
            raise ParityAuditError(f"regular file changed while hashing: {source}")
        if snapshot_root_fd is not None and snapshot_root_before is not None:
            leaf_parent_fd = (
                snapshot_directory_chain[-1][2]
                if snapshot_directory_chain
                else snapshot_root_fd
            )
            named_source = os.stat(
                AUDIT_SOURCE_RELATIVE.name,
                dir_fd=leaf_parent_fd,
                follow_symlinks=False,
            )
            if _tree_stat_identity(named_source) != _tree_stat_identity(opened):
                raise ParityAuditError("snapshot-fd causal source name changed while hashing")
            for parent_fd, name, child_fd, directory_before in reversed(
                snapshot_directory_chain
            ):
                if _tree_stat_identity(os.fstat(child_fd)) != _tree_stat_identity(
                    directory_before
                ):
                    raise ParityAuditError(
                        "snapshot-fd source directory changed while hashing"
                    )
                named_directory = os.stat(
                    name, dir_fd=parent_fd, follow_symlinks=False
                )
                if _tree_stat_identity(named_directory) != _tree_stat_identity(
                    directory_before
                ):
                    raise ParityAuditError(
                        "snapshot-fd source directory name changed while hashing"
                    )
            if _tree_stat_identity(os.fstat(snapshot_root_fd)) != _tree_stat_identity(
                snapshot_root_before
            ):
                raise ParityAuditError("snapshot root changed while hashing causal source")
            expected_after_fd, expected_after = _open_absolute_directory(
                PROJECT_ROOT, "expected causal snapshot root after hashing"
            )
            try:
                if _tree_stat_identity(expected_after) != _tree_stat_identity(
                    snapshot_root_before
                ):
                    raise ParityAuditError(
                        "named causal snapshot root changed while hashing"
                    )
            finally:
                os.close(expected_after_fd)
    except OSError as exc:
        raise ParityAuditError(f"cannot hash regular file {source}: {exc}") from exc
    finally:
        os.close(descriptor)
        for _parent, _name, child, _before in reversed(snapshot_directory_chain):
            os.close(child)
        if snapshot_root_fd is not None:
            os.close(snapshot_root_fd)
    result = digest.hexdigest()
    if (
        snapshot_root_before is not None
        and result != BOOTSTRAP_EXECUTION_SCRIPT_SHA256
    ):
        raise ParityAuditError("snapshot-fd causal source differs from compiled script")
    return result


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
    info = _lstat_if_present(path, f"project module {path}")
    if (
        info is None
        or not stat.S_ISREG(info.st_mode)
        or not _resolve_strict(path, f"project module {path}").is_relative_to(project_root)
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


def _tree_stat_identity(value: os.stat_result) -> tuple[int, ...]:
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


def _tree_error(message: str, exc: OSError | None = None) -> None:
    if exc is None:
        raise ParityAuditError(message)
    raise ParityAuditError(f"{message}: {exc}") from exc


def _secure_output_rows(root: Path) -> list[dict[str, Any]]:
    """Return a complete, fd-stable metadata/content inventory of ``root``."""

    try:
        named_root = root.lstat()
    except OSError as exc:
        _tree_error("Exp23 live-output root is unavailable", exc)
    if stat.S_ISLNK(named_root.st_mode) or not stat.S_ISDIR(named_root.st_mode):
        _tree_error("Exp23 live-output root is not a nonsymlink directory")
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    cloexec = getattr(os, "O_CLOEXEC", 0)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | nofollow | cloexec
    file_flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | nofollow | cloexec
    absolute = root.absolute()
    if not absolute.is_absolute() or any(
        part in ("", ".", "..") for part in absolute.parts[1:]
    ):
        _tree_error("Exp23 live-output root is not an absolute normalized path")
    root_fd: int | None = None
    try:
        root_fd = os.open(absolute.anchor, directory_flags)
        for part in absolute.parts[1:]:
            child_fd = os.open(part, directory_flags, dir_fd=root_fd)
            os.close(root_fd)
            root_fd = child_fd
    except OSError as exc:
        if root_fd is not None:
            os.close(root_fd)
        _tree_error("Exp23 live-output root cannot be opened without symlinks", exc)
    assert root_fd is not None
    rows: list[dict[str, Any]] = []

    def require_directory(info: os.stat_result, relative: Path) -> None:
        if not stat.S_ISDIR(info.st_mode):
            _tree_error(f"Exp23 live-output entry is not a directory: {relative}")
        permissions = stat.S_IMODE(info.st_mode)
        if permissions & 0o444 == 0 or permissions & 0o111 == 0:
            _tree_error(f"Exp23 live-output directory is not traversable: {relative}")

    def walk(directory_fd: int, parent: Path, before: os.stat_result) -> None:
        require_directory(before, parent)
        try:
            with os.scandir(directory_fd) as iterator:
                children = []
                for entry in iterator:
                    try:
                        child_info = entry.stat(follow_symlinks=False)
                    except OSError as exc:
                        _tree_error(
                            f"Exp23 live-output entry cannot be stated: {parent / entry.name}",
                            exc,
                        )
                    children.append((entry.name, child_info))
        except OSError as exc:
            _tree_error(f"Exp23 live-output directory cannot be enumerated: {parent}", exc)
        if len({name for name, _info in children}) != len(children):
            _tree_error(f"Exp23 live-output directory has duplicate entries: {parent}")
        for name, listed in sorted(children, key=lambda value: value[0]):
            if not isinstance(name, str) or name in ("", ".", "..") or "/" in name:
                _tree_error("Exp23 live-output entry name is unsafe")
            relative = parent / name
            if stat.S_ISLNK(listed.st_mode):
                _tree_error(f"Exp23 live-output tree contains symlink: {relative}")
            if stat.S_ISDIR(listed.st_mode):
                require_directory(listed, relative)
                try:
                    child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                except OSError as exc:
                    _tree_error(f"Exp23 live-output directory cannot be opened: {relative}", exc)
                try:
                    opened = os.fstat(child_fd)
                    if _tree_stat_identity(opened) != _tree_stat_identity(listed):
                        _tree_error(f"Exp23 live-output directory raced before open: {relative}")
                    if opened.st_uid != os.getuid():
                        _tree_error(f"Exp23 live-output directory owner differs: {relative}")
                    rows.append(
                        {
                            "relative": str(relative),
                            "kind": "directory",
                            "mode": int(opened.st_mode),
                            "size": int(opened.st_size),
                            "mtime_ns": int(opened.st_mtime_ns),
                        }
                    )
                    walk(child_fd, relative, opened)
                    after = os.fstat(child_fd)
                    if _tree_stat_identity(after) != _tree_stat_identity(opened):
                        _tree_error(
                            f"Exp23 live-output directory changed while traversing: {relative}"
                        )
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(listed.st_mode):
                if stat.S_IMODE(listed.st_mode) & 0o444 == 0:
                    _tree_error(f"Exp23 live-output file is not readable: {relative}")
                try:
                    child_fd = os.open(name, file_flags, dir_fd=directory_fd)
                except OSError as exc:
                    _tree_error(f"Exp23 live-output file cannot be opened: {relative}", exc)
                digest = hashlib.sha256()
                try:
                    opened = os.fstat(child_fd)
                    if (
                        not stat.S_ISREG(opened.st_mode)
                        or opened.st_uid != os.getuid()
                        or opened.st_nlink != 1
                        or _tree_stat_identity(opened) != _tree_stat_identity(listed)
                    ):
                        _tree_error(f"Exp23 live-output file raced before open: {relative}")
                    try:
                        while block := os.read(child_fd, 16 * 1024 * 1024):
                            digest.update(block)
                        after = os.fstat(child_fd)
                    except OSError as exc:
                        _tree_error(f"Exp23 live-output file cannot be read: {relative}", exc)
                    if _tree_stat_identity(after) != _tree_stat_identity(opened):
                        _tree_error(f"Exp23 live-output file changed while hashing: {relative}")
                finally:
                    os.close(child_fd)
                rows.append(
                    {
                        "relative": str(relative),
                        "kind": "file",
                        "mode": int(opened.st_mode),
                        "size": int(opened.st_size),
                        "mtime_ns": int(opened.st_mtime_ns),
                        "sha256": digest.hexdigest(),
                    }
                )
            else:
                _tree_error(f"Exp23 live-output tree contains special file: {relative}")
        try:
            after = os.fstat(directory_fd)
        except OSError as exc:
            _tree_error(f"Exp23 live-output directory cannot be restated: {parent}", exc)
        if _tree_stat_identity(after) != _tree_stat_identity(before):
            _tree_error(f"Exp23 live-output directory changed while enumerating: {parent}")

    try:
        opened_root = os.fstat(root_fd)
        if _tree_stat_identity(opened_root) != _tree_stat_identity(named_root):
            _tree_error("Exp23 live-output root raced before open")
        if opened_root.st_uid != os.getuid():
            _tree_error("Exp23 live-output root owner differs")
        walk(root_fd, Path(), opened_root)
        after_root = os.fstat(root_fd)
        if _tree_stat_identity(after_root) != _tree_stat_identity(opened_root):
            _tree_error("Exp23 live-output root changed while traversing")
    finally:
        os.close(root_fd)
    rows.sort(key=lambda row: str(row["relative"]))
    return rows


def _output_tree_fingerprint(path: Path) -> str:
    """Hash output metadata outside the intentional submission transaction tree.

    The copied-tree audit is replayed once before submission and again after the
    submitter has durably created ``state/submission`` and sealed its source copy.
    Those transaction bytes are orchestration metadata, not scientific output.
    Project them out (along with the now-empty logical ``state`` container) so the
    two legitimate phases have one identity.  Every other output entry remains in
    the projection, and symlinks/special files fail closed.

    The enclosing snapshot preflight separately fingerprints the complete output
    tree before and after all audit subprocesses, so a replay still cannot mutate
    even the intentionally projected transaction subtree without detection.
    """

    if not _lexical_exists(path, f"Exp23 live-output root {path}"):
        return stable_hash({"ignored_subtree": "state/submission", "entries": []})
    all_rows = _secure_output_rows(path)
    rows: list[dict[str, Any]] = []
    for row in all_rows:
        relative = Path(str(row["relative"]))
        parts = relative.parts
        if relative == Path("state"):
            if row["kind"] != "directory":
                raise ParityAuditError("Exp23 submission-state parent is not a directory")
            continue
        if len(parts) >= 2 and parts[:2] == ("state", "submission"):
            if len(parts) == 2 and row["kind"] != "directory":
                raise ParityAuditError("Exp23 submission root is not a directory")
            continue
        rows.append(row)
    return stable_hash({"ignored_subtree": "state/submission", "entries": rows})


def _verify_output_projection_regression() -> None:
    """Exercise the absent-versus-post-claim projection in every real audit run."""

    temporary_parent = Path(tempfile.gettempdir())
    parent_stat = temporary_parent.lstat()
    if stat.S_ISLNK(parent_stat.st_mode) or not stat.S_ISDIR(parent_stat.st_mode):
        raise ParityAuditError("causal-audit temporary parent is not a directory")
    with tempfile.TemporaryDirectory(
        prefix=f"treewm-exp23-causal-projection-{os.getuid()}-",
        dir=temporary_parent,
    ) as raw_root:
        probe = Path(raw_root)
        baseline = _output_tree_fingerprint(probe / "absent")
        claimed = probe / "claimed"
        snapshot = claimed / "state" / "submission" / "source-snapshot" / "repo"
        journal = claimed / "state" / "submission" / "journal"
        snapshot.mkdir(parents=True)
        journal.mkdir()
        (snapshot / "sealed.py").write_text("VALUE = 1\n", encoding="utf-8")
        (journal / "0000_CLAIMED.json").write_text("{}\n", encoding="utf-8")
        if _output_tree_fingerprint(claimed) != baseline:
            raise ParityAuditError(
                "submission-state projection differs between absent and post-claim roots"
            )
        scientific_probe = claimed / "scientific-write.json"
        scientific_probe.write_text("{}\n", encoding="utf-8")
        if _output_tree_fingerprint(claimed) == baseline:
            raise ParityAuditError("submission-state projection hid a scientific write")
        hostile = claimed / "state" / "submission" / "hostile-symlink"
        hostile.symlink_to(probe / "outside")
        try:
            _output_tree_fingerprint(claimed)
        except ParityAuditError as exc:
            if "symlink" not in str(exc):
                raise
        else:
            raise ParityAuditError("submission-state projection accepted a symlink")


def _launch_pair_identity(
    launch: Mapping[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    argv = list(launch["argv"])
    if len(argv) < 3:
        raise ParityAuditError("controlled launch does not use the exact trainer module")
    trainer_path = Path(argv[1])
    trainer_info = _lstat_if_present(trainer_path, "controlled trainer entrypoint")
    if (
        trainer_info is None
        or not stat.S_ISREG(trainer_info.st_mode)
        or _resolve_strict(trainer_path, "controlled trainer entrypoint")
        != (project_root / "scripts/train.py")
    ):
        raise ParityAuditError("controlled launch does not use the exact regular trainer module")
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
    project_info = _lstat_if_present(project_root, "causal-audit project root")
    if (
        project_root != PROJECT_ROOT
        or project_info is None
        or not stat.S_ISDIR(project_info.st_mode)
    ):
        raise ParityAuditError("audit must use the exact nonsymlink package project root")
    _verify_output_projection_regression()
    audit = _load("exp23_weight_helpers_for_parity", PACKAGE_DIR / "weight_audit.py")
    campaign = _load("exp23_campaign_for_parity", PACKAGE_DIR / "campaign.py")
    for module in (audit, campaign):
        module_path = Path(module.__file__)
        module_info = _lstat_if_present(module_path, f"audit helper module {module_path}")
        if (
            module_info is None
            or not stat.S_ISREG(module_info.st_mode)
            or not _resolve_strict(
                module_path, f"audit helper module {module_path}"
            ).is_relative_to(project_root)
        ):
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
