#!/usr/bin/env python3
"""Fail-closed primitives for the Exp24 outcome-blind data authorities.

This module deliberately has no dependency on an experiment controller.  The three
auditors use descriptor-relative opens so a checked pathname cannot be exchanged for
a symlink (or another inode) between validation and use.  Exact-tree checks bind the
persistent entry set, identities, and bytes at both the beginning and end of a
governed read.  A transient add/remove or replace/restore that restores the same final
identities and bytes without changing any consumed descriptor is outside this bounded
persistent-state claim; detecting that would require a filesystem snapshot, lease, or
event monitor.  No helper in this file writes to a dataset, cache, recipe, lock, or
output directory.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, ExitStack
import hashlib
import io
import json
import mmap
import os
from pathlib import Path, PurePosixPath
import re
import stat
from typing import Any, Iterable, Mapping, Sequence
import zipfile


SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_CLOEXEC", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
)
DIRECTORY_FLAGS = READ_FLAGS | getattr(os, "O_DIRECTORY", 0)
READ_CHUNK = 16 * 1024 * 1024

# Outcome-blind prerequisite identities copied from the finalized Exp22 all-ten
# publication ledger.  They are values published before Exp24 and contain no Exp23
# terminal result or Launch8 outcome.  Keeping them in executable source makes a
# contract unable to nominate its own expected digest.
CAMPAIGN_ID = "treewm-50task-1m-v2"
OBJECTIVE_VERSION = "treewm_v2_rms_rank_v1"
CAMPAIGN_PROTOCOL_SHA256 = "1ee9dd8a65ab5b01def3d55826efe34212628e9c1393259ef17e5d12665afc82"
RECIPE_CODE_SHA256 = "4cb70b4421d3eae1a6e947e3b0359336bd64248897ffcf52d4a672ca4adcd30c"
RECIPE_RUNTIME_SHA256 = "77da91d49a1db99850fbf0632dc02ec58a3209f1a87949d6f5640ae6bf505c6b"
SETTINGS: tuple[dict[str, Any], ...] = (
    {"id": "scene", "env_name": "scene-play-v0", "source_name": "scene-play-v0", "dataset_kind": "standard", "data_subdir": "standard", "expected_shards": None, "task_metric_dims": [19,20,21,28,29,32,33,36,38], "relative_endpoints": False, "input_contract_sha256": "0fa8d6c363a88503133669d5715a549a9e4c2b70148d486bc25bda9090958f89", "calibration_sha256": "b2935ffac00eeb409e48066625760b8243d5a6f34fc83e18e1cea46b5c7a051e", "future_recipe_sha256": "8c345a6b855ab1bf378dc84839b0ce8728284cd3620e05336ec6e527d0e8af09", "published_union_train_anchors": 758084, "published_union_validation_anchors": 75816},
    {"id": "puzzle-3x3", "env_name": "puzzle-3x3-play-v0", "source_name": "puzzle-3x3-play-v0", "dataset_kind": "standard", "data_subdir": "standard", "expected_shards": None, "task_metric_dims": [19,20,23,24,27,28,31,32,35,36,39,40,43,44,47,48,51,52], "relative_endpoints": False, "input_contract_sha256": "39f7ca2799cf13f876f7bbe4991589f8bf967e4372d9484d21f3bd6dee778e6c", "calibration_sha256": "58263941d8902dfee9d4171ea13930ee8ca3f0ee5f1e316c4151ee28283ed4c7", "future_recipe_sha256": "58c0d8dc7ca10be57d971e1d30031736c972521c0e735cb01b0c2938185adae9", "published_union_train_anchors": 758084, "published_union_validation_anchors": 75816},
    {"id": "puzzle-4x4-100m", "env_name": "puzzle-4x4-play-v0", "source_name": "puzzle-4x4-play-100m-v0", "dataset_kind": "sharded_100m_full", "data_subdir": "100m/puzzle-4x4-play-100m-v0", "expected_shards": 100, "task_metric_dims": [19,20,23,24,27,28,31,32,35,36,39,40,43,44,47,48,51,52,55,56,59,60,63,64,67,68,71,72,75,76,79,80], "relative_endpoints": False, "input_contract_sha256": "7ba892f3e4b7eb11b1b1c9712bbf9d9237c66910b13c034f4fbe01c1c34e4883", "calibration_sha256": "03bc78c2b7948093dd752e01ec512338da83f2cf677b6c4fc5e7e900ab51ba18", "future_recipe_sha256": "819a452dc22c3d80fe44aad3848b1e6a11295863b80fe87ba0510f5a2f9a6ff3", "published_union_train_anchors": 1194586, "published_union_validation_anchors": 119473},
    {"id": "cube-double", "env_name": "cube-double-play-v0", "source_name": "cube-double-play-v0", "dataset_kind": "standard", "data_subdir": "standard", "expected_shards": None, "task_metric_dims": [19,20,21,28,29,30], "relative_endpoints": True, "input_contract_sha256": "0e493d41ca2ab31e0cc2f446bc02acfa083e5a9d93327471678eb762e679355d", "calibration_sha256": "b148fe9ae0e3964a526deb3fd6fb1c17ff87475bc5497e7104dccb73688f9005", "future_recipe_sha256": "edbedf726f447f8213d9b848c31b07ba67976a39febdac9047a14df2d47cda0b", "published_union_train_anchors": 758084, "published_union_validation_anchors": 75816},
    {"id": "cube-triple", "env_name": "cube-triple-play-v0", "source_name": "cube-triple-play-v0", "dataset_kind": "standard", "data_subdir": "standard", "expected_shards": None, "task_metric_dims": [19,20,21,28,29,30,37,38,39], "relative_endpoints": True, "input_contract_sha256": "7f9bdfa9810d14bc9eb072bfb945aa3db4b58302946bada0ac4159786f434284", "calibration_sha256": "b8d82d722bd678aa280dccd71be8919089d12b84b824557b440524cbdab7951c", "future_recipe_sha256": "546fbe6584827fb96a9aa2d1a38b780bbeff6655aceb73da0f02b9b747f2aa5a", "published_union_train_anchors": 1030685, "published_union_validation_anchors": 102824},
    {"id": "cube-quadruple-100m", "env_name": "cube-quadruple-play-v0", "source_name": "cube-quadruple-play-100m-v0", "dataset_kind": "sharded_100m_full", "data_subdir": "100m/cube-quadruple-play-100m-v0", "expected_shards": 100, "task_metric_dims": [19,20,21,28,29,30,37,38,39,46,47,48], "relative_endpoints": True, "input_contract_sha256": "cfd70e8f04c396e9ea88720d483e5190bed4ff1ae36b27e2da81b5324542e9a6", "calibration_sha256": "3240b1462484bca29e10e8d3c8a59a4df12c2989ab74066bd6484b038bba4928", "future_recipe_sha256": "42aec1a4b9de2b4c98efd0bb7b51edad0330f7241adf5ad148dfbbded0eeca5f", "published_union_train_anchors": 1194586, "published_union_validation_anchors": 119473},
    {"id": "antmaze-large", "env_name": "antmaze-large-navigate-v0", "source_name": "antmaze-large-navigate-v0", "dataset_kind": "standard", "data_subdir": "standard", "expected_shards": None, "task_metric_dims": [0,1], "relative_endpoints": True, "input_contract_sha256": "9009f22bcf2007341a6e3040b5f8a325fdd542520e852c372b90cdd50cad9d6e", "calibration_sha256": "77d33a07fa5862150472b6bf8d97e3ed7e7c3a4c803fb0c3c121ebe7cb234a62", "future_recipe_sha256": "34087d2bab3b42f5fe96a2148a96975a757def976ec69bab2d9b206ffe953598", "published_union_train_anchors": 758084, "published_union_validation_anchors": 75816},
    {"id": "antmaze-giant", "env_name": "antmaze-giant-navigate-v0", "source_name": "antmaze-giant-navigate-v0", "dataset_kind": "standard", "data_subdir": "standard", "expected_shards": None, "task_metric_dims": [0,1], "relative_endpoints": True, "input_contract_sha256": "4894df8940189d2d16e463b9c42d0b3e3fbc7dfc82ada803095737de2b207e93", "calibration_sha256": "1a0902b21d32a424cb6dd404a223b3b0aa2d9d29471ece9337f5a934ccec87f1", "future_recipe_sha256": "003529d2da3672d8bc51e3c55aa676db4e09bdcb1f2dc70d695397a2c2bb7715", "published_union_train_anchors": 759154, "published_union_validation_anchors": 76196},
    {"id": "humanoidmaze-medium", "env_name": "humanoidmaze-medium-navigate-v0", "source_name": "humanoidmaze-medium-navigate-v0", "dataset_kind": "standard", "data_subdir": "standard", "expected_shards": None, "task_metric_dims": [0,1], "relative_endpoints": True, "input_contract_sha256": "f2614d844fea24878e59fcbc8e2512eac6d4e32cb88437e57d26e7f1161a5575", "calibration_sha256": "9c98a93fa0b2434057d05e2c3b08284ac15fd463ab684fb00b3477eeb1dc316c", "future_recipe_sha256": "9aa71c7c792f553ec7c20397d401efbd1fec208b69b774e8fc8771382b132a59", "published_union_train_anchors": 955698, "published_union_validation_anchors": 95746},
    {"id": "humanoidmaze-large", "env_name": "humanoidmaze-large-navigate-v0", "source_name": "humanoidmaze-large-navigate-v0", "dataset_kind": "standard", "data_subdir": "standard", "expected_shards": None, "task_metric_dims": [0,1], "relative_endpoints": True, "input_contract_sha256": "5dbf92877f842ff2673ee30eef6796d482502a063e24967a5843371b6c81d87a", "calibration_sha256": "2cd142dfa3d07ec315733da4b0953a519df02de52181f2b60a8339c8f075bf50", "future_recipe_sha256": "430c2a2cf18233de8071b39ed17d95f93f203b76531e2c413fa76dfc66423995", "published_union_train_anchors": 955698, "published_union_validation_anchors": 95746},
)


class DataAuthorityError(RuntimeError):
    """An authority input is missing, mutable, aliased, or structurally invalid."""


def canonical_json(value: object) -> str:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DataAuthorityError(f"value is not canonical JSON: {exc}") from exc


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise DataAuthorityError(message)


def require_sha256(value: object, label: str) -> str:
    require(isinstance(value, str) and SHA256_RE.fullmatch(value) is not None,
            f"{label} is not a lowercase SHA256")
    return value


def require_exact_keys(value: object, keys: Iterable[str], label: str) -> Mapping[str, Any]:
    require(isinstance(value, dict), f"{label} is not an object")
    expected = set(keys)
    actual = set(value)
    require(actual == expected,
            f"{label} key set differs (missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)})")
    require(all(isinstance(key, str) for key in value), f"{label} has a non-string key")
    return value


def require_int(
    value: object,
    label: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    require(type(value) is int, f"{label} is not an integer")
    if minimum is not None:
        require(value >= minimum, f"{label} is below {minimum}")
    if maximum is not None:
        require(value <= maximum, f"{label} exceeds {maximum}")
    return value


def require_number(value: object, label: str) -> float:
    require(type(value) in (int, float), f"{label} is not a JSON number")
    # canonical_json rejects non-finite values, but callers also use this independently.
    require(value == value and value not in (float("inf"), float("-inf")),
            f"{label} is non-finite")
    return float(value)


def require_bool(value: object, label: str) -> bool:
    require(type(value) is bool, f"{label} is not a boolean")
    return value


def require_string(value: object, label: str, *, nonempty: bool = True) -> str:
    require(isinstance(value, str), f"{label} is not a string")
    if nonempty:
        require(bool(value), f"{label} is empty")
    return value


def require_int_list(
    value: object,
    label: str,
    *,
    nonempty: bool = False,
    minimum: int | None = None,
    maximum: int | None = None,
) -> list[int]:
    require(isinstance(value, list), f"{label} is not an array")
    if nonempty:
        require(bool(value), f"{label} is empty")
    for index, item in enumerate(value):
        require_int(item, f"{label}[{index}]", minimum=minimum, maximum=maximum)
    return value


def _json_object_pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise DataAuthorityError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_json_object_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                DataAuthorityError(f"non-finite JSON token in {label}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DataAuthorityError(f"invalid UTF-8 JSON in {label}: {exc}") from exc
    require(isinstance(value, dict), f"{label} JSON root is not an object")
    # This also catches Python float infinities produced by enormous numeric literals.
    canonical_json(value)
    return value


def file_identity(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def safe_relative(value: str | PurePosixPath, label: str = "relative path") -> PurePosixPath:
    path = PurePosixPath(str(value))
    require(not path.is_absolute(), f"{label} is absolute")
    require(str(path) not in ("", "."), f"{label} is empty")
    require(all(part not in ("", ".", "..") for part in path.parts),
            f"{label} contains traversal")
    return path


def lexical_relative_to(path: str | Path, root: str | Path, label: str) -> PurePosixPath:
    candidate = Path(path)
    anchor = Path(root)
    require(candidate.is_absolute() and anchor.is_absolute(), f"{label} is not absolute")
    # Do not call resolve(): following a symlink before the descriptor walk would turn
    # the very condition being rejected into an apparently safe path.
    normalized_candidate = Path(os.path.normpath(str(candidate)))
    normalized_anchor = Path(os.path.normpath(str(anchor)))
    try:
        relative = normalized_candidate.relative_to(normalized_anchor)
    except ValueError as exc:
        raise DataAuthorityError(f"{label} escapes its registered root") from exc
    return safe_relative(PurePosixPath(relative.as_posix()), label)


class StableRegular(AbstractContextManager["StableRegular"]):
    """One held, single-link regular-file descriptor with end-of-use mutation check."""

    def __init__(self, descriptor: int, before: os.stat_result, label: str) -> None:
        self.fd = descriptor
        self.before = before
        self.label = label
        self._closed = False
        self._observed_digest: str | None = None

    @property
    def size(self) -> int:
        return int(self.before.st_size)

    @property
    def mtime_ns(self) -> int:
        return int(self.before.st_mtime_ns)

    def read_bytes(self) -> bytes:
        require(self.size <= 2**31, f"{self.label} is too large for an in-memory read")
        chunks: list[bytes] = []
        offset = 0
        while offset < self.size:
            block = os.pread(self.fd, min(READ_CHUNK, self.size - offset), offset)
            require(bool(block), f"{self.label} ended before its stable stat size")
            chunks.append(block)
            offset += len(block)
        require(not os.pread(self.fd, 1, self.size), f"{self.label} grew while being read")
        payload = b"".join(chunks)
        self._observed_digest = hashlib.sha256(payload).hexdigest()
        return payload

    def _compute_sha256(self) -> str:
        digest = hashlib.sha256()
        offset = 0
        while offset < self.size:
            block = os.pread(self.fd, min(READ_CHUNK, self.size - offset), offset)
            require(bool(block), f"{self.label} ended before its stable stat size")
            digest.update(block)
            offset += len(block)
        require(not os.pread(self.fd, 1, self.size), f"{self.label} grew while being hashed")
        return digest.hexdigest()

    def sha256(self) -> str:
        digest = self._compute_sha256()
        self._observed_digest = digest
        return digest

    def duplicate_file(self):
        descriptor = os.dup(self.fd)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return os.fdopen(descriptor, "rb", closefd=True)

    def verify_stable(self) -> None:
        after = os.fstat(self.fd)
        require(file_identity(after) == file_identity(self.before),
                f"{self.label} mutated while open")
        if self._observed_digest is not None:
            require(self._compute_sha256() == self._observed_digest,
                    f"{self.label} content mutated while open")

    def close(self) -> None:
        if not self._closed:
            try:
                self.verify_stable()
            finally:
                os.close(self.fd)
                self._closed = True

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class SecureRoot(AbstractContextManager["SecureRoot"]):
    """A directory capability reached without following any symlink component."""

    def __init__(self, path: str | Path, label: str) -> None:
        lexical = Path(path)
        require(lexical.is_absolute(), f"{label} must be an absolute path")
        self.path = Path(os.path.normpath(str(lexical)))
        self.label = label
        descriptor = os.open("/", DIRECTORY_FLAGS)
        try:
            for component in self.path.parts[1:]:
                next_descriptor = os.open(component, DIRECTORY_FLAGS, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
                info = os.fstat(descriptor)
                require(stat.S_ISDIR(info.st_mode), f"{label} component is not a directory")
        except BaseException:
            os.close(descriptor)
            raise
        self.fd = descriptor
        self.before = os.fstat(descriptor)
        self._closed = False
        self._exact_tree_contract: tuple[
            frozenset[PurePosixPath], frozenset[PurePosixPath]
        ] | None = None
        self._exact_tree_snapshot: dict[str, Any] | None = None
        self._derived_directories: dict[PurePosixPath, tuple[int, ...]] = {}

    @classmethod
    def _from_descriptor(
        cls, descriptor: int, path: Path, label: str
    ) -> "SecureRoot":
        root = cls.__new__(cls)
        root.path = path
        root.label = label
        root.fd = descriptor
        root.before = os.fstat(descriptor)
        root._closed = False
        root._exact_tree_contract = None
        root._exact_tree_snapshot = None
        root._derived_directories = {}
        return root

    def _open_directory(self, relative: PurePosixPath | str) -> int:
        if str(relative) in ("", "."):
            return os.dup(self.fd)
        path = safe_relative(relative, f"path below {self.label}")
        descriptor = os.dup(self.fd)
        try:
            for component in path.parts:
                next_descriptor = os.open(component, DIRECTORY_FLAGS, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = next_descriptor
                require(stat.S_ISDIR(os.fstat(descriptor).st_mode),
                        f"{self.label}/{path} has a non-directory component")
            return descriptor
        except BaseException:
            os.close(descriptor)
            raise

    def subroot(
        self, relative: PurePosixPath | str, label: str | None = None
    ) -> "SecureRoot":
        """Derive a child capability from this retained directory descriptor."""
        path = safe_relative(relative, f"directory below {self.label}")
        descriptor = self._open_directory(path)
        try:
            identity = file_identity(os.fstat(descriptor))
            prior = self._derived_directories.get(path)
            require(
                prior is None or prior == identity,
                f"{self.label}/{path} directory identity changed between uses",
            )
            self._derived_directories[path] = identity
            return SecureRoot._from_descriptor(
                descriptor,
                self.path / Path(path.as_posix()),
                label or f"{self.label}/{path}",
            )
        except BaseException:
            os.close(descriptor)
            raise

    def open_regular(self, relative: PurePosixPath | str, label: str | None = None) -> StableRegular:
        path = safe_relative(relative, f"file below {self.label}")
        parent = self._open_directory(path.parent)
        descriptor: int | None = None
        try:
            descriptor = os.open(path.name, READ_FLAGS, dir_fd=parent)
            info = os.fstat(descriptor)
            require(stat.S_ISREG(info.st_mode), f"{label or path} is not a regular file")
            require(info.st_nlink == 1, f"{label or path} is hard-linked")
            require(stat.S_IMODE(info.st_mode) & 0o444 != 0,
                    f"{label or path} has no read permission bits")
            return StableRegular(descriptor, info, label or f"{self.label}/{path}")
        except BaseException:
            if descriptor is not None:
                os.close(descriptor)
            raise
        finally:
            os.close(parent)

    def directory_entries(self, relative: PurePosixPath | str = ".") -> dict[str, os.stat_result]:
        descriptor = self._open_directory(relative)
        try:
            names = os.listdir(descriptor)
            require(len(names) == len(set(names)), f"duplicate names below {self.label}")
            return {
                name: os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                for name in names
            }
        finally:
            os.close(descriptor)

    def require_exact_tree(
        self,
        *,
        files: Iterable[str],
        directories: Iterable[str],
    ) -> None:
        require(
            self._exact_tree_contract is None,
            f"{self.label} already has an exact-tree contract",
        )
        expected_files = {safe_relative(path) for path in files}
        expected_directories = {PurePosixPath(".")} | {
            safe_relative(path) for path in directories
        }
        for file_path in expected_files:
            expected_directories.update(
                parent for parent in file_path.parents if parent != PurePosixPath(".")
            )
        for directory_path in tuple(expected_directories):
            expected_directories.update(
                parent
                for parent in directory_path.parents
                if parent != PurePosixPath(".")
            )
        overlap = expected_files & expected_directories
        require(
            not overlap,
            f"{self.label} exact tree names both files and directories: "
            f"{sorted(str(path) for path in overlap)}",
        )
        contract = (frozenset(expected_files), frozenset(expected_directories))
        snapshot = self._scan_exact_tree(*contract)
        self._exact_tree_contract = contract
        self._exact_tree_snapshot = snapshot

    def _scan_exact_tree(
        self,
        expected_files: frozenset[PurePosixPath],
        expected_directories: frozenset[PurePosixPath],
    ) -> dict[str, Any]:
        """Return a stable no-follow identity/content snapshot of an exact tree."""
        children: dict[PurePosixPath, set[str]] = {path: set() for path in expected_directories}
        kinds: dict[tuple[PurePosixPath, str], str] = {}
        for directory_path in expected_directories:
            if directory_path != PurePosixPath("."):
                children[directory_path.parent].add(directory_path.name)
                kinds[(directory_path.parent, directory_path.name)] = "directory"
        for file_path in expected_files:
            children[file_path.parent].add(file_path.name)
            kinds[(file_path.parent, file_path.name)] = "file"
        ordered_directories = sorted(
            expected_directories, key=lambda item: (len(item.parts), str(item))
        )
        ordered_files = sorted(expected_files, key=str)
        directory_descriptors: dict[PurePosixPath, int] = {}
        directory_identities: dict[PurePosixPath, tuple[int, ...]] = {}
        file_sources: dict[PurePosixPath, StableRegular] = {}
        file_digests: dict[PurePosixPath, str] = {}
        with ExitStack() as stack:
            root_descriptor = os.dup(self.fd)
            stack.callback(os.close, root_descriptor)
            root_path = PurePosixPath(".")
            directory_descriptors[root_path] = root_descriptor
            directory_identities[root_path] = file_identity(
                os.fstat(root_descriptor)
            )
            for directory_path in ordered_directories:
                if directory_path == root_path:
                    continue
                parent_descriptor = directory_descriptors[directory_path.parent]
                descriptor = os.open(
                    directory_path.name,
                    DIRECTORY_FLAGS,
                    dir_fd=parent_descriptor,
                )
                stack.callback(os.close, descriptor)
                info = os.fstat(descriptor)
                require(
                    stat.S_ISDIR(info.st_mode),
                    f"{self.label}/{directory_path} is not a directory",
                )
                directory_descriptors[directory_path] = descriptor
                directory_identities[directory_path] = file_identity(info)

            for directory_path in ordered_directories:
                descriptor = directory_descriptors[directory_path]
                names = os.listdir(descriptor)
                require(
                    len(names) == len(set(names)),
                    f"duplicate names below {self.label}/{directory_path}",
                )
                expected = children[directory_path]
                actual = set(names)
                require(
                    actual == expected,
                    f"{self.label}/{directory_path} inventory differs "
                    f"(missing={sorted(expected - actual)}, "
                    f"extra={sorted(actual - expected)})",
                )
                for name in sorted(names):
                    info = os.stat(
                        name, dir_fd=descriptor, follow_symlinks=False
                    )
                    entry_path = (
                        PurePosixPath(name)
                        if directory_path == root_path
                        else directory_path / name
                    )
                    kind = kinds[(directory_path, name)]
                    if kind == "directory":
                        require(
                            stat.S_ISDIR(info.st_mode),
                            f"{self.label}/{entry_path} is not a directory",
                        )
                        require(
                            file_identity(info)
                            == directory_identities[entry_path],
                            f"{self.label}/{entry_path} directory identity raced",
                        )
                        continue
                    require(
                        stat.S_ISREG(info.st_mode),
                        f"{self.label}/{entry_path} is special or a symlink",
                    )
                    require(
                        info.st_nlink == 1,
                        f"{self.label}/{entry_path} is hard-linked",
                    )
                    file_descriptor = os.open(
                        name, READ_FLAGS, dir_fd=descriptor
                    )
                    file_info = os.fstat(file_descriptor)
                    try:
                        require(
                            file_identity(file_info) == file_identity(info),
                            f"{self.label}/{entry_path} file identity raced",
                        )
                        require(
                            stat.S_ISREG(file_info.st_mode),
                            f"{self.label}/{entry_path} is not a regular file",
                        )
                        require(
                            file_info.st_nlink == 1,
                            f"{self.label}/{entry_path} is hard-linked",
                        )
                        require(
                            stat.S_IMODE(file_info.st_mode) & 0o444 != 0,
                            f"{self.label}/{entry_path} has no read permission bits",
                        )
                        source = stack.enter_context(
                            StableRegular(
                                file_descriptor,
                                file_info,
                                f"{self.label}/{entry_path}",
                            )
                        )
                        file_descriptor = -1
                    finally:
                        if file_descriptor >= 0:
                            os.close(file_descriptor)
                    file_sources[entry_path] = source
                    file_digests[entry_path] = source.sha256()

            # Re-enumerate every retained directory after all file hashes.  This
            # catches persistent entry changes even when the filesystem coalesces
            # directory mtime/ctime updates into a coarse timestamp tick.
            for directory_path in ordered_directories:
                descriptor = directory_descriptors[directory_path]
                names = os.listdir(descriptor)
                expected = children[directory_path]
                actual = set(names)
                require(
                    len(names) == len(actual) and actual == expected,
                    f"{self.label}/{directory_path} inventory changed while scanning",
                )
                require(
                    file_identity(os.fstat(descriptor))
                    == directory_identities[directory_path],
                    f"{self.label}/{directory_path} directory mutated while scanning",
                )
                for name in sorted(names):
                    info = os.stat(
                        name, dir_fd=descriptor, follow_symlinks=False
                    )
                    entry_path = (
                        PurePosixPath(name)
                        if directory_path == root_path
                        else directory_path / name
                    )
                    if kinds[(directory_path, name)] == "directory":
                        expected_identity = directory_identities[entry_path]
                    else:
                        expected_identity = file_identity(
                            file_sources[entry_path].before
                        )
                    require(
                        file_identity(info) == expected_identity,
                        f"{self.label}/{entry_path} identity changed while scanning",
                    )

            snapshot = {
                "directories": {
                    str(path): list(directory_identities[path])
                    for path in ordered_directories
                },
                "files": {
                    str(path): {
                        "identity": list(file_identity(file_sources[path].before)),
                        "sha256": file_digests[path],
                    }
                    for path in ordered_files
                },
            }
        return snapshot

    def read_json(self, relative: PurePosixPath | str, label: str | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        with self.open_regular(relative, label) as source:
            payload = source.read_bytes()
            digest = hashlib.sha256(payload).hexdigest()
            value = parse_json_bytes(payload, label or str(relative))
            row = {
                "sha256": digest,
                "size": source.size,
                "mtime_ns": source.mtime_ns,
            }
        return value, row

    def close(self) -> None:
        if not self._closed:
            try:
                if self._exact_tree_contract is not None:
                    final_snapshot = self._scan_exact_tree(
                        *self._exact_tree_contract
                    )
                    require(
                        final_snapshot == self._exact_tree_snapshot,
                        f"{self.label} governed exact tree changed during audit",
                    )
                for path, expected_identity in self._derived_directories.items():
                    descriptor = self._open_directory(path)
                    try:
                        require(
                            file_identity(os.fstat(descriptor))
                            == expected_identity,
                            f"{self.label}/{path} derived directory changed "
                            "during audit",
                        )
                    finally:
                        os.close(descriptor)
                require(file_identity(os.fstat(self.fd)) == file_identity(self.before),
                        f"{self.label} root directory mutated during audit")
            finally:
                os.close(self.fd)
                self._closed = True

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


class StableNpy(AbstractContextManager["StableNpy"]):
    """A NumPy array mapped from the already-authenticated open descriptor."""

    def __init__(self, source: StableRegular, label: str) -> None:
        # NumPy is intentionally imported lazily so metadata-only CLIs stay lightweight.
        import numpy as np

        self.source = source
        self.label = label
        with source.duplicate_file() as handle:
            version = np.lib.format.read_magic(handle)
            if version == (1, 0):
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(handle)
            elif version in ((2, 0), (3, 0)):
                shape, fortran, dtype = np.lib.format.read_array_header_2_0(handle)
            else:
                raise DataAuthorityError(f"{label} has unsupported NPY version {version}")
            offset = handle.tell()
        dtype = np.dtype(dtype)
        require(not dtype.hasobject, f"{label} has an object dtype")
        require(all(type(dimension) is int and dimension >= 0 for dimension in shape),
                f"{label} has an invalid shape")
        count = 1
        for dimension in shape:
            count *= dimension
        require(offset + count * dtype.itemsize == source.size,
                f"{label} NPY payload size does not match its header")
        self._mapping = mmap.mmap(source.fd, 0, access=mmap.ACCESS_READ)
        self.array = np.ndarray(
            shape=shape,
            dtype=dtype,
            buffer=self._mapping,
            offset=offset,
            order="F" if fortran else "C",
        )
        self.fortran_order = bool(fortran)

    def close(self) -> None:
        if hasattr(self, "array"):
            del self.array
        if hasattr(self, "_mapping"):
            self._mapping.close()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def open_stable_npy(root: SecureRoot, relative: str, label: str) -> tuple[StableRegular, StableNpy]:
    source = root.open_regular(relative, label)
    try:
        return source, StableNpy(source, label)
    except BaseException:
        source.close()
        raise


def validate_npz_members(source: StableRegular, label: str) -> tuple[zipfile.ZipFile, Any]:
    """Open one stable NPZ and reject duplicate, nested, missing, or unsafe members."""
    handle = source.duplicate_file()
    try:
        archive = zipfile.ZipFile(handle, mode="r")
        infos = archive.infolist()
        names = [info.filename for info in infos]
        require(len(names) == len(set(names)), f"{label} has duplicate ZIP members")
        require(bool(names), f"{label} has no NPY members")
        for info in infos:
            member = PurePosixPath(info.filename)
            require(
                len(member.parts) == 1
                and member.name.endswith(".npy")
                and member.name not in ("", ".npy")
                and not info.is_dir(),
                f"{label} has an unsafe or non-NPY ZIP member: {info.filename}",
            )
            require(info.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                    f"{label} uses an unsupported ZIP compression method")
        return archive, handle
    except BaseException:
        handle.close()
        raise


def load_npz_array(archive: zipfile.ZipFile, key: str, label: str):
    import numpy as np

    member = f"{key}.npy"
    require(member in archive.namelist(), f"{label} is missing {member}")
    try:
        with archive.open(member, "r") as stream:
            value = np.load(stream, allow_pickle=False)
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise DataAuthorityError(f"cannot decode {label}/{member}: {exc}") from exc
    require(isinstance(value, np.ndarray) and not value.dtype.hasobject,
            f"{label}/{member} is not a non-object ndarray")
    return value


def inventory_row(source: StableRegular, *, digest: str | None = None) -> dict[str, Any]:
    return {
        "sha256": digest if digest is not None else source.sha256(),
        "size": source.size,
        "mtime_ns": source.mtime_ns,
    }


def seal_result(body: Mapping[str, Any], *, audit_id: str, status: str) -> dict[str, Any]:
    require("artifact_sha256" not in body, "audit body already has artifact_sha256")
    result = {"schema_version": 1, "status": status, "audit_id": audit_id, **dict(body)}
    result["artifact_sha256"] = stable_hash(result)
    return result


def validate_sealed_result(value: object, *, audit_id: str, status: str) -> dict[str, Any]:
    require(isinstance(value, dict), "sealed lock is not an object")
    schema_version = require_int(
        value.get("schema_version"), "sealed lock schema_version", minimum=1, maximum=1
    )
    require(schema_version == 1, "sealed lock schema differs")
    require(value.get("audit_id") == audit_id, "sealed lock audit_id differs")
    require(value.get("status") == status, "sealed lock status differs")
    claimed = require_sha256(value.get("artifact_sha256"), "sealed lock artifact_sha256")
    body = dict(value)
    body.pop("artifact_sha256")
    require(stable_hash(body) == claimed, "sealed lock canonical self-hash differs")
    return value


def validate_or_compare_lock(
    result: Mapping[str, Any], lock: object, *, audit_id: str, status: str
) -> None:
    sealed = validate_sealed_result(lock, audit_id=audit_id, status=status)
    require(
        canonical_json(dict(result)).encode("ascii")
        == canonical_json(dict(sealed)).encode("ascii"),
        "computed audit result differs from sealed lock",
    )


def array_sha256(array: Any) -> str:
    """Domain-separated shape/dtype/content hash used by prefix contracts."""
    import numpy as np

    value = np.ascontiguousarray(array)
    header = canonical_json({"dtype": value.dtype.str, "shape": list(value.shape)}).encode("ascii")
    digest = hashlib.sha256()
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def chunks(length: int, size: int = 1_000_000) -> Iterable[slice]:
    for start in range(0, length, size):
        yield slice(start, min(start + size, length))


def arrays_equal_chunked(left: Any, right: Any, *, label: str, chunk_rows: int = 1_000_000) -> None:
    import numpy as np

    require(tuple(left.shape) == tuple(right.shape), f"{label} shape differs")
    require(np.dtype(left.dtype) == np.dtype(right.dtype), f"{label} dtype differs")
    for section in chunks(len(left), chunk_rows):
        require(np.array_equal(left[section], right[section], equal_nan=True),
                f"{label} content differs at rows {section.start}:{section.stop}")


def source_file_sha256(path: Path) -> str:
    """Hash an auditor source file without accepting a symlink or hard link."""
    root = SecureRoot(path.parent.absolute(), f"source parent for {path.name}")
    try:
        with root.open_regular(path.name, f"auditor source {path.name}") as source:
            return source.sha256()
    finally:
        root.close()
