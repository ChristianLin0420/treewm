#!/usr/bin/env python3
"""Read-only cancellation plan, or explicitly cancel one sealed Exp23 receipt."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
PACKAGE_RELATIVE = Path("experiments/23-treewm-executable-prefix-repair-pilot-v1")
PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]
JOB_ID = re.compile(r"^[1-9][0-9]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_root",
        "snapshot_root",
        "submission_sha256",
        "train_array_job_id",
        "report_job_id",
        "array",
        "dependency",
    }
)
SCHEDULER_CONTROL_PLANE = {
    "slurm_conf": "/cm/shared/apps/slurm/var/etc/cs-oci-ord/slurm.conf",
    "cluster_name": "cs-oci-ord",
    "slurmctld_hosts": ["cs-oci-ord-a", "cs-oci-ord-b"],
    "slurmctld_port": 6817,
    "auth_type": "auth/munge",
    "gres_types": ["gpu"],
    "cli_filter_plugins": ["lua"],
    "job_submit_plugins": ["lua"],
    "trust_model": (
        "root-admin mutable scheduler control plane; config and Lua policy bytes are "
        "observation-bound from preclaim through submission; root-owned Slurm clients, "
        "plugin binaries, and shared libraries are trusted mutable external runtime"
    ),
}
REPORT_DEPENDENCY_TEST_REQUIREMENT = {
    "phase": "after_train_reconciliation_before_report_submission",
    "dependency": "afterok:<accepted_train_array_job_id>",
    "kill_on_invalid_dep": "yes",
    "required": True,
}


class CancellationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CancellationError(message)


def _lstat_if_present(
    path: str | Path, label: str | None = None
) -> os.stat_result | None:
    """Return one lexical entry, treating only ENOENT as absence."""

    source = Path(path)
    try:
        return source.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CancellationError(
            f"cannot determine whether {label or source} exists: {exc}"
        ) from exc


def _lexical_exists(path: str | Path, label: str | None = None) -> bool:
    return _lstat_if_present(path, label) is not None


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    payload, _digest = _authenticated_regular_bytes(path, f"JSON artifact {path}")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                CancellationError(f"non-finite JSON value in {path}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CancellationError(f"cannot read {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def file_sha256(path: Path) -> str:
    _payload, digest = _authenticated_regular_bytes(path, f"SHA256 source {path}")
    return digest


def _regular(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise CancellationError(f"{label} is unavailable: {exc}") from exc
    require(stat.S_ISREG(info.st_mode), f"{label} is not a regular nonsymlink file")


def _directory(path: Path, label: str) -> None:
    lexical = path.absolute()
    require(
        lexical.is_absolute()
        and all(part not in ("", ".", "..") for part in lexical.parts[1:]),
        f"{label} is not an absolute normalized path",
    )
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        try:
            component = current.lstat()
        except OSError as exc:
            raise CancellationError(f"{label} path component is unavailable: {current}: {exc}") from exc
        require(not stat.S_ISLNK(component.st_mode), f"{label} has a symlink path component: {current}")
    try:
        info = path.lstat()
    except OSError as exc:
        raise CancellationError(f"{label} is unavailable: {exc}") from exc
    require(stat.S_ISDIR(info.st_mode), f"{label} is not a nonsymlink directory")


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


def _open_directory_components(path: Path, label: str) -> int:
    absolute = path.absolute()
    require(
        absolute.is_absolute()
        and all(part not in ("", ".", "..") for part in absolute.parts[1:]),
        f"{label} is not an absolute normalized path",
    )
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(absolute.anchor, flags)
    except OSError as exc:
        raise CancellationError(f"{label} root cannot be opened: {exc}") from exc
    try:
        for part in absolute.parts[1:]:
            child = os.open(part, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        info = os.fstat(descriptor)
        permissions = stat.S_IMODE(info.st_mode)
        require(
            stat.S_ISDIR(info.st_mode)
            and permissions & 0o444 != 0
            and permissions & 0o111 != 0,
            f"{label} is not a traversable nonsymlink directory",
        )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _authenticated_regular_bytes(path: Path, label: str) -> tuple[bytes, str]:
    parent_fd = _open_directory_components(path.parent, f"{label} parent")
    descriptor: int | None = None
    try:
        descriptor = os.open(
            path.name,
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        before = os.fstat(descriptor)
        require(
            stat.S_ISREG(before.st_mode) and stat.S_IMODE(before.st_mode) & 0o444 != 0,
            f"{label} is not a readable regular file",
        )
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while block := os.read(descriptor, 16 * 1024 * 1024):
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
        require(_file_identity(after) == _file_identity(before), f"{label} changed while reading")
        return b"".join(chunks), digest.hexdigest()
    except OSError as exc:
        raise CancellationError(f"{label} cannot be read without symlinks: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)


def _secure_tree_rows(root: Path, label: str) -> list[dict[str, Any]]:
    _directory(root, f"{label} root")
    root_fd = _open_directory_components(root, f"{label} root")
    rows: list[dict[str, Any]] = []
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )

    def walk(directory_fd: int, parent: Path, before: os.stat_result) -> None:
        try:
            with os.scandir(directory_fd) as iterator:
                children = []
                for entry in iterator:
                    try:
                        children.append((entry.name, entry.stat(follow_symlinks=False)))
                    except OSError as exc:
                        raise CancellationError(f"cannot stat {label} entry {parent / entry.name}: {exc}") from exc
        except OSError as exc:
            raise CancellationError(f"cannot enumerate {label} directory {parent}: {exc}") from exc
        for name, listed in sorted(children, key=lambda value: value[0]):
            relative = parent / name
            require(not stat.S_ISLNK(listed.st_mode), f"{label} contains symlink: {relative}")
            if stat.S_ISDIR(listed.st_mode):
                permissions = stat.S_IMODE(listed.st_mode)
                require(
                    permissions & 0o444 != 0 and permissions & 0o111 != 0,
                    f"{label} directory is not traversable: {relative}",
                )
                try:
                    child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise CancellationError(f"cannot open {label} directory {relative}: {exc}") from exc
                try:
                    opened = os.fstat(child_fd)
                    require(_file_identity(opened) == _file_identity(listed), f"{label} directory raced: {relative}")
                    require(opened.st_uid == os.getuid(), f"{label} directory owner differs: {relative}")
                    rows.append({"path": str(relative), "kind": "directory", "mode": stat.S_IMODE(opened.st_mode)})
                    walk(child_fd, relative, opened)
                    require(_file_identity(os.fstat(child_fd)) == _file_identity(opened), f"{label} directory changed: {relative}")
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(listed.st_mode):
                require(stat.S_IMODE(listed.st_mode) & 0o444 != 0, f"{label} file is unreadable: {relative}")
                try:
                    child_fd = os.open(name, file_flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise CancellationError(f"cannot open {label} file {relative}: {exc}") from exc
                digest = hashlib.sha256()
                try:
                    opened = os.fstat(child_fd)
                    require(_file_identity(opened) == _file_identity(listed), f"{label} file raced: {relative}")
                    require(
                        opened.st_uid == os.getuid() and opened.st_nlink == 1,
                        f"{label} file ownership/link count differs: {relative}",
                    )
                    while block := os.read(child_fd, 16 * 1024 * 1024):
                        digest.update(block)
                    require(_file_identity(os.fstat(child_fd)) == _file_identity(opened), f"{label} file changed: {relative}")
                except OSError as exc:
                    raise CancellationError(f"cannot read {label} file {relative}: {exc}") from exc
                finally:
                    os.close(child_fd)
                rows.append({"path": str(relative), "kind": "file", "mode": stat.S_IMODE(opened.st_mode), "sha256": digest.hexdigest()})
            else:
                raise CancellationError(f"{label} contains special file: {relative}")
        require(_file_identity(os.fstat(directory_fd)) == _file_identity(before), f"{label} directory changed: {parent}")

    try:
        opened_root = os.fstat(root_fd)
        require(opened_root.st_uid == os.getuid(), f"{label} root owner differs")
        rows.append(
            {
                "path": "",
                "kind": "root",
                "mode": stat.S_IMODE(opened_root.st_mode),
            }
        )
        walk(root_fd, Path(), opened_root)
        require(_file_identity(os.fstat(root_fd)) == _file_identity(opened_root), f"{label} root changed")
    finally:
        os.close(root_fd)
    rows.sort(key=lambda row: str(row["path"]))
    return rows


class _ReportCancelLock:
    """Linearize the durable cancellation latch against report publication."""

    def __init__(self, submission_root: Path) -> None:
        self.root = submission_root
        self.path = submission_root / ".REPORT_CANCEL.lock"
        self.descriptor: int | None = None

    def __enter__(self) -> "_ReportCancelLock":
        _directory(self.root, "report/cancel lock root")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise CancellationError(f"cannot open report/cancel lock: {exc}") from exc
        try:
            opened = os.fstat(descriptor)
            named = self.path.lstat()
            require(stat.S_ISREG(opened.st_mode), "report/cancel lock is not regular")
            require((opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino), "report/cancel lock path raced")
            require(opened.st_uid == os.getuid() and opened.st_nlink == 1, "report/cancel lock ownership differs")
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            _fsync_directory(self.root)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        except BaseException:
            os.close(descriptor)
            raise
        self.descriptor = descriptor
        return self

    def __exit__(self, _kind: object, _value: object, _traceback: object) -> None:
        assert self.descriptor is not None
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self.descriptor)
            self.descriptor = None


def _safe_relative(value: object, label: str) -> Path:
    relative = Path(str(value))
    require(
        not relative.is_absolute()
        and relative.parts
        and all(part not in ("", ".", "..") for part in relative.parts),
        f"{label} is not a safe relative path",
    )
    return relative


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def seal_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = (json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if not _lexical_exists(path.parent):
        path.parent.mkdir(mode=0o700)
        _fsync_directory(path.parent.parent)
    else:
        _directory(path.parent, f"parent of {path}")
    temporary = path.parent / f".{path.name}.tmp.{os.getpid()}.{time.time_ns()}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, f"short immutable write: {temporary}")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        temporary.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    try:
        os.link(temporary, path, follow_symlinks=False)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)
    return hashlib.sha256(payload).hexdigest()


def activate_isolated_runtime(manifest: Mapping[str, Any]) -> dict[str, Any]:
    require(bool(sys.flags.isolated) and bool(sys.flags.no_site), "cancellation requires Python -I -S")
    expected = Path(str(manifest["paths"]["python"]))
    require(expected.is_absolute(), "pinned Python path is not absolute")
    try:
        info = expected.lstat()
    except OSError as exc:
        raise CancellationError(f"pinned Python is unavailable: {exc}") from exc
    require(stat.S_ISLNK(info.st_mode) or stat.S_ISREG(info.st_mode), "pinned Python has invalid type")
    require(os.path.normpath(os.path.abspath(sys.executable)) == str(expected), "cancellation interpreter lexical path differs")
    target = expected.resolve(strict=True)
    _regular(target, "resolved pinned Python")
    venv_root = expected.parent.parent
    pyvenv = venv_root / "pyvenv.cfg"
    _regular(pyvenv, "pinned pyvenv.cfg")
    pyvenv_payload, _pyvenv_digest = _authenticated_regular_bytes(
        pyvenv, "pinned pyvenv.cfg"
    )
    values: dict[str, str] = {}
    try:
        pyvenv_text = pyvenv_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise CancellationError(f"pinned pyvenv.cfg is not UTF-8: {exc}") from exc
    for line in pyvenv_text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    require("home" in values and Path(values["home"]).is_absolute(), "pyvenv home is absent")
    base_root = Path(values["home"]).parent
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    sites = (
        venv_root / "lib" / version / "site-packages",
        base_root / "lib" / version / "site-packages",
    )
    for path in sites:
        try:
            site_info = path.lstat()
        except OSError as exc:
            raise CancellationError(f"bound site-packages is unavailable: {exc}") from exc
        require(stat.S_ISDIR(site_info.st_mode), "bound site-packages is not a nonsymlink directory")
    existing = [value for value in sys.path if "site-packages" in value]
    require(not existing or existing == [str(value) for value in sites], "unexpected site-package bootstrap path")
    for path in sites:
        if str(path) not in sys.path:
            sys.path.append(str(path))
    return {
        "lexical_executable": str(expected),
        "lexical_symlink_target": os.readlink(expected) if stat.S_ISLNK(info.st_mode) else None,
        "resolved_executable": str(target),
        "resolved_executable_sha256": file_sha256(target),
        "resolved_executable_size": target.stat().st_size,
        "base_executable": str(Path(str(getattr(sys, "_base_executable", target)))),
        "venv_site_packages": str(sites[0]),
        "base_site_packages": str(sites[1]),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }


def _fsync_directory(path: Path) -> None:
    descriptor = _open_directory_components(path, f"fsync directory {path}")
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def reject_environment(environ: Mapping[str, str] | None = None) -> None:
    environment = os.environ if environ is None else environ
    failures = sorted(
        key
        for key in environment
        if key.startswith("TREEWM_") or key in {"RANK", "WORLD_SIZE", "LOCAL_RANK"}
    )
    require(not failures, "forbidden inherited environment: " + ", ".join(failures))


def scheduler_environment(control_plane: object) -> dict[str, str]:
    """Exact local-cluster environment shared with submit/recovery."""

    require(
        isinstance(control_plane, Mapping)
        and dict(control_plane) == SCHEDULER_CONTROL_PLANE,
        "scheduler control-plane contract differs",
    )

    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "SLURM_CONF": SCHEDULER_CONTROL_PLANE["slurm_conf"],
    }


def scheduler_control_plane_observation(
    snapshot_root: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    """Run the exact snapshot-bound root-admin scheduler authenticator."""

    control_plane = manifest["execution"].get("scheduler_control_plane")
    require(
        isinstance(control_plane, Mapping)
        and dict(control_plane) == SCHEDULER_CONTROL_PLANE,
        "snapshot scheduler control-plane contract differs",
    )
    path = snapshot_root / PACKAGE_RELATIVE / "submit.py"
    _regular(path, "snapshot scheduler verifier")
    name = f"_treewm_exp23_cancel_scheduler_{os.getpid()}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot load snapshot scheduler verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
        observe = getattr(module, "_scheduler_control_plane_observation", None)
        require(callable(observe), "snapshot scheduler verifier API differs")
        value = observe(control_plane)
    finally:
        sys.modules.pop(name, None)
    require(
        isinstance(value, Mapping)
        and value.get("schema_version") == 1
        and isinstance(value.get("config"), Mapping)
        and SHA256.fullmatch(str(value["config"].get("sha256", ""))) is not None,
        "scheduler control-plane observation differs",
    )
    return dict(value)


def scheduler_fallback_config(
    snapshot_root: Path,
    manifest: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> tuple[dict[str, Any], bytes]:
    """Validate the sealed original config used only for exact-ID cleanup."""

    control_plane = manifest["execution"].get("scheduler_control_plane")
    require(
        isinstance(control_plane, Mapping)
        and dict(control_plane) == SCHEDULER_CONTROL_PLANE,
        "snapshot scheduler control-plane contract differs",
    )
    path = snapshot_root / PACKAGE_RELATIVE / "submit.py"
    _regular(path, "snapshot scheduler fallback verifier")
    name = f"_treewm_exp23_cancel_fallback_{os.getpid()}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot load snapshot fallback verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
        validate = getattr(module, "_validated_scheduler_fallback", None)
        require(callable(validate), "snapshot scheduler fallback verifier API differs")
        binding, payload = validate(
            contract.get("scheduler_fallback_config"),
            control_plane,
            (contract.get("scheduler_preclaim") or {}).get(
                "scheduler_control_plane"
            ),
        )
    finally:
        sys.modules.pop(name, None)
    require(
        isinstance(binding, Mapping)
        and isinstance(payload, bytes)
        and hashlib.sha256(payload).hexdigest() == binding.get("sha256"),
        "scheduler fallback config differs",
    )
    return dict(binding), payload


def _scheduler_fallback_descriptor(payload: bytes) -> int:
    require(hasattr(os, "memfd_create"), "scheduler fallback requires memfd_create")
    descriptor = os.memfd_create(
        "treewm-exp23-slurm-conf",
        getattr(os, "MFD_CLOEXEC", 0) | getattr(os, "MFD_ALLOW_SEALING", 0),
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, "scheduler fallback config write was short")
            view = view[written:]
        os.fchmod(descriptor, 0o400)
        os.lseek(descriptor, 0, os.SEEK_SET)
        seals = (
            getattr(fcntl, "F_SEAL_SEAL", 0)
            | getattr(fcntl, "F_SEAL_SHRINK", 0)
            | getattr(fcntl, "F_SEAL_GROW", 0)
            | getattr(fcntl, "F_SEAL_WRITE", 0)
        )
        require(seals != 0, "scheduler fallback seals are unavailable")
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        require(
            fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & seals == seals,
            "scheduler fallback config is not immutable",
        )
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def validate_receipt(submission_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _directory(submission_root, "submission root")
    submission_root = submission_root.absolute()
    receipt_path = submission_root / "SUBMISSION_RECEIPT.json"
    contract_path = submission_root / "SUBMISSION_CONTRACT.json"
    _regular(receipt_path, "submission receipt")
    _regular(contract_path, "submission contract")
    require(stat.S_IMODE(receipt_path.lstat().st_mode) == 0o444, "submission receipt mode differs")
    require(stat.S_IMODE(contract_path.lstat().st_mode) == 0o444, "submission contract mode differs")
    receipt = read_json(receipt_path)
    require(set(receipt) == RECEIPT_KEYS, "submission receipt schema differs")
    require(receipt["schema_version"] == 1 and receipt["status"] == "submitted", "submission receipt is not committed")
    require(receipt["campaign_id"] == "treewm-executable-prefix-repair-pilot-v1-launch5", "submission receipt campaign differs")
    require(Path(str(receipt["submission_root"])) == submission_root.absolute(), "receipt submission root differs")
    require(Path(str(receipt["snapshot_root"])).is_absolute(), "receipt snapshot root is not absolute")
    claimed = str(receipt["submission_sha256"])
    require(SHA256.fullmatch(claimed) is not None and file_sha256(contract_path) == claimed, "submission contract hash differs")
    seal_path = submission_root / "journal" / "0002_CONTRACT_SEALED.json"
    _regular(seal_path, "contract-seal journal")
    require(stat.S_IMODE(seal_path.lstat().st_mode) == 0o444, "contract-seal journal mode differs")
    require(
        read_json(seal_path)
        == {
            "schema_version": 1,
            "record": "contract_sealed",
            "submission_sha256": claimed,
            "launch_count": 20,
        },
        "contract-seal journal differs",
    )
    train_id = str(receipt["train_array_job_id"])
    report_id = str(receipt["report_job_id"])
    require(JOB_ID.fullmatch(train_id) is not None, "training receipt job ID is malformed")
    require(JOB_ID.fullmatch(report_id) is not None, "report receipt job ID is malformed")
    require(train_id != report_id, "receipt job IDs are not distinct")
    require(receipt["array"] == "0-19%20", "receipt array differs")
    require(receipt["dependency"] == f"afterok:{train_id}", "receipt dependency differs")
    for role, ordinal, expected_id in (("train", 3, train_id), ("report", 4, report_id)):
        submitted_path = submission_root / "journal" / f"{ordinal:04d}_{role.upper()}_SUBMITTED.json"
        _regular(submitted_path, f"{role} submission journal")
        require(stat.S_IMODE(submitted_path.lstat().st_mode) == 0o444, f"{role} submission journal mode differs")
        submitted = read_json(submitted_path)
        require(
            submitted.get("record") == f"{role}_submitted"
            and str(submitted.get("job_id")) == expected_id,
            f"receipt {role} job differs from durable journal",
        )
    contract = read_json(contract_path)
    require(contract.get("status") == "sealed_for_submission", "submission contract is not sealed")
    require(contract.get("submission_root") == str(submission_root.absolute()), "contract submission root differs")
    require(contract.get("snapshot_root") == receipt["snapshot_root"], "contract snapshot root differs")
    require(len(contract.get("launches") or []) == 20, "submission contract launch coverage differs")
    snapshot_root = Path(str(receipt["snapshot_root"]))
    _directory(snapshot_root, "snapshot root")
    snapshot = snapshot_root.absolute()
    require(snapshot.is_relative_to(submission_root), "snapshot root escapes submission root")
    require(str(snapshot) == receipt["snapshot_root"], "snapshot root is not canonical")
    require(
        snapshot == submission_root / "source-snapshot" / "repo",
        "snapshot root differs from exact source-snapshot namespace",
    )
    for path, mode, label in (
        (submission_root, 0o700, "submission root"),
        (submission_root / "source-snapshot", 0o555, "source-snapshot parent"),
        (snapshot, 0o555, "snapshot root"),
    ):
        descriptor = _open_directory_components(path, label)
        try:
            info = os.fstat(descriptor)
            require(
                info.st_uid == os.getuid()
                and info.st_gid == os.getgid()
                and stat.S_IMODE(info.st_mode) == mode,
                f"{label} ownership/mode differs",
            )
        finally:
            os.close(descriptor)
    inventory = contract.get("snapshot_inventory")
    require(isinstance(inventory, Mapping) and inventory, "snapshot inventory is absent")
    normalized: dict[str, str] = {}
    for raw_relative, digest in inventory.items():
        relative = str(_safe_relative(raw_relative, "snapshot inventory path"))
        require(SHA256.fullmatch(str(digest)) is not None, f"snapshot digest is malformed: {relative}")
        require(relative not in normalized, f"duplicate normalized snapshot path: {relative}")
        normalized[relative] = str(digest)
    require(stable_hash(normalized) == contract.get("snapshot_inventory_sha256"), "snapshot inventory hash differs")
    expected_dirs = {
        str(parent)
        for relative in normalized
        for parent in list(Path(relative).parents)[:-1]
    }
    actual_files: set[str] = set()
    actual_dirs: set[str] = set()
    for row in _secure_tree_rows(snapshot, "sealed snapshot"):
        relative = str(row["path"])
        if row["kind"] == "root":
            require(int(row["mode"]) == 0o555, "snapshot root mode differs")
            continue
        if row["kind"] == "file":
            actual_files.add(relative)
            require(relative in normalized, f"snapshot contains unclaimed file: {relative}")
            require(int(row["mode"]) & 0o222 == 0, f"snapshot file is writable: {relative}")
            require(row["sha256"] == normalized[relative], f"snapshot hash differs: {relative}")
        else:
            actual_dirs.add(relative)
            require(int(row["mode"]) & 0o222 == 0, f"snapshot directory is writable: {relative}")
    require(actual_files == set(normalized), "snapshot file coverage differs")
    require(actual_dirs == expected_dirs, "snapshot directory coverage differs")
    manifest_path = snapshot / PACKAGE_RELATIVE / "manifest.json"
    _regular(manifest_path, "snapshot manifest")
    require(normalized.get(str(PACKAGE_RELATIVE / "manifest.json")) == file_sha256(manifest_path), "snapshot manifest inventory binding differs")
    manifest = read_json(manifest_path)
    require(contract.get("manifest_sha256") == stable_hash(manifest), "snapshot manifest contract hash differs")
    require(
        manifest["execution"].get("scheduler_control_plane")
        == SCHEDULER_CONTROL_PLANE
        and contract.get("scheduler_control_plane_contract")
        == SCHEDULER_CONTROL_PLANE,
        "scheduler control-plane contract differs",
    )
    scheduler_preclaim = contract.get("scheduler_preclaim")
    require(
        isinstance(scheduler_preclaim, Mapping)
        and scheduler_preclaim.get("schema_version") == 1
        and scheduler_preclaim.get("status") == "scheduler_preclaim_verified"
        and scheduler_preclaim.get("campaign_id") == receipt["campaign_id"]
        and scheduler_preclaim.get("scheduler_calls") == 7
        and scheduler_preclaim.get("scheduler_mutation_calls") == 0,
        "scheduler preclaim contract differs",
    )
    require(
        isinstance(scheduler_preclaim.get("scheduler_probe_commands"), list)
        and len(scheduler_preclaim["scheduler_probe_commands"]) == 7
        and scheduler_preclaim.get("report_dependency_test")
        == REPORT_DEPENDENCY_TEST_REQUIREMENT
        and scheduler_preclaim.get("zero_job_proof")
        == {
            "job_names": {
                "train": "exp23-launch5-scheduler-test-train",
                "report": "exp23-launch5-scheduler-test-report",
            },
            "pre_queries": 2,
            "post_queries": 2,
            "matching_jobs_before": 0,
            "matching_jobs_after": 0,
        },
        "scheduler preclaim call ledger differs",
    )
    protocol_path = snapshot / PACKAGE_RELATIVE / "protocol.sha256"
    protocol_payload, _protocol_digest = _authenticated_regular_bytes(
        protocol_path, "snapshot protocol lock"
    )
    try:
        protocol = protocol_payload.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise CancellationError(f"snapshot protocol lock is not ASCII: {exc}") from exc
    require(SHA256.fullmatch(protocol) is not None and protocol == contract.get("package_protocol_sha256"), "snapshot protocol contract differs")
    interpreter = activate_isolated_runtime(manifest)
    require(contract.get("orchestration_interpreter") == interpreter, "cancellation interpreter contract differs")
    scheduler_fallback_config(snapshot, manifest, contract)
    return receipt, contract, manifest


def cancellation_plan(submission_root: Path) -> dict[str, Any]:
    reject_environment()
    receipt, _contract, _manifest = validate_receipt(submission_root)
    ids = [str(receipt["train_array_job_id"]), str(receipt["report_job_id"])]
    return {
        "schema_version": 1,
        "status": "read_only_cancellation_plan",
        "campaign_id": receipt["campaign_id"],
        "submission_root": str(submission_root.absolute()),
        "submission_sha256": receipt["submission_sha256"],
        "job_ids": ids,
        "latch_path": str(submission_root / "CANCEL_REQUESTED.json"),
        "writes_performed": 0,
        "scheduler_calls": 0,
    }


def seal_latch(submission_root: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "status": "cancel_requested",
        "campaign_id": receipt["campaign_id"],
        "submission_sha256": receipt["submission_sha256"],
        "train_array_job_id": str(receipt["train_array_job_id"]),
        "report_job_id": str(receipt["report_job_id"]),
    }
    path = submission_root / "CANCEL_REQUESTED.json"
    if _lexical_exists(path):
        _regular(path, "cancellation latch")
        require(read_json(path) == value, "existing cancellation latch differs")
        return value
    try:
        seal_json(path, value)
    except FileExistsError:
        _regular(path, "concurrent cancellation latch")
        require(read_json(path) == value, "concurrent cancellation latch differs")
    require(read_json(path) == value, "cancellation latch verification failed")
    return value


def explicit_cancel(submission_root: Path) -> dict[str, Any]:
    reject_environment()
    receipt, contract, manifest = validate_receipt(submission_root)
    with _ReportCancelLock(submission_root):
        latch = seal_latch(submission_root, receipt)
    # Resolve the executable only after the durable latch exists.  A missing scheduler
    # client therefore still leaves every worker with an authoritative stop request.
    snapshot_root = Path(str(receipt["snapshot_root"]))
    scancel = Path(str(manifest["execution"]["scancel"]))
    _regular(scancel, "scancel")
    require(os.access(scancel, os.X_OK), "scancel is not executable")
    ids = [str(receipt["train_array_job_id"]), str(receipt["report_job_id"])]
    require(all(JOB_ID.fullmatch(value) for value in ids), "refusing non-exact cancellation target")
    control_plane = manifest["execution"].get("scheduler_control_plane")
    fallback_binding, fallback_payload = scheduler_fallback_config(
        snapshot_root, manifest, contract
    )
    canonical_boundary_error: str | None = None
    try:
        observation_before = scheduler_control_plane_observation(snapshot_root, manifest)
        environment = scheduler_environment(control_plane)
        inherited_fds: tuple[int, ...] = ()
        scheduler_mode = "canonical_root_admin_config"
    except BaseException as exc:
        canonical_boundary_error = repr(exc)
        scheduler_mode = "sealed_original_config_fallback"
        observation_before = {
            "schema_version": 1,
            "mode": scheduler_mode,
            "sha256": fallback_binding["sha256"],
            "size": fallback_binding["size"],
        }
        environment = {}
        inherited_fds = ()
    command = [str(scancel), *ids]
    token = f"{time.time_ns()}-{os.getpid()}"
    evidence_root = submission_root / "cancellation"
    scheduler_attempts: list[dict[str, Any]] = []

    def call_exact(
        *,
        call_token: str,
        mode: str,
        observation: Mapping[str, Any],
        call_environment: Mapping[str, str],
        pass_descriptors: Sequence[int],
        boundary_error: str | None,
    ) -> subprocess.CompletedProcess[str]:
        intent = {
            "schema_version": 1,
            "status": "exact_cancel_call_intent",
            "campaign_id": receipt["campaign_id"],
            "submission_sha256": receipt["submission_sha256"],
            "call_token": call_token,
            "job_ids": ids,
            "command": command,
            "scheduler_control_plane": dict(observation),
            "scheduler_mode": mode,
            "canonical_boundary_error": boundary_error,
        }
        seal_json(evidence_root / f"CANCEL_CALL.{call_token}.json", intent)
        value = subprocess.run(
            command,
            cwd=snapshot_root,
            env=dict(call_environment),
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=tuple(pass_descriptors),
        )
        scheduler_attempts.append(
            {
                "call_token": call_token,
                "scheduler_mode": mode,
                "returncode": value.returncode,
                "stdout": value.stdout,
                "stderr": value.stderr,
                "scheduler_control_plane": dict(observation),
                "canonical_boundary_error": boundary_error,
            }
        )
        return value

    if scheduler_mode == "canonical_root_admin_config":
        completed = call_exact(
            call_token=token,
            mode=scheduler_mode,
            observation=observation_before,
            call_environment=environment,
            pass_descriptors=inherited_fds,
            boundary_error=canonical_boundary_error,
        )
    else:
        fallback_descriptor = _scheduler_fallback_descriptor(fallback_payload)
        try:
            environment = {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "LANG": "C",
                "LC_ALL": "C",
                "SLURM_CONF": f"/proc/self/fd/{fallback_descriptor}",
            }
            completed = call_exact(
                call_token=token,
                mode=scheduler_mode,
                observation=observation_before,
                call_environment=environment,
                pass_descriptors=(fallback_descriptor,),
                boundary_error=canonical_boundary_error,
            )
        finally:
            os.close(fallback_descriptor)
    if scheduler_mode == "canonical_root_admin_config":
        try:
            observation_after = scheduler_control_plane_observation(snapshot_root, manifest)
            if observation_after != observation_before:
                raise CancellationError("scheduler control plane changed during scancel")
        except BaseException as exc:
            canonical_boundary_error = repr(exc)
            scheduler_attempts[-1]["canonical_boundary_error"] = canonical_boundary_error
            if completed.returncode != 0:
                fallback_descriptor = _scheduler_fallback_descriptor(fallback_payload)
                try:
                    scheduler_mode = "sealed_original_config_fallback_after_canonical_failure"
                    observation_before = {
                        "schema_version": 1,
                        "mode": scheduler_mode,
                        "sha256": fallback_binding["sha256"],
                        "size": fallback_binding["size"],
                    }
                    environment = {
                        "PATH": "/usr/local/bin:/usr/bin:/bin",
                        "LANG": "C",
                        "LC_ALL": "C",
                        "SLURM_CONF": f"/proc/self/fd/{fallback_descriptor}",
                    }
                    completed = call_exact(
                        call_token=f"{token}.fallback",
                        mode=scheduler_mode,
                        observation=observation_before,
                        call_environment=environment,
                        pass_descriptors=(fallback_descriptor,),
                        boundary_error=canonical_boundary_error,
                    )
                finally:
                    os.close(fallback_descriptor)
    result = {
        **latch,
        "status": (
            "cancel_requested_and_exact_jobs_signalled"
            if completed.returncode == 0
            else "cancel_requested_scheduler_call_failed"
        ),
        "call_token": token,
        "job_ids": ids,
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "scheduler_control_plane": observation_before,
        "scheduler_mode": scheduler_mode,
        "canonical_boundary_error": canonical_boundary_error,
        "scheduler_attempts": scheduler_attempts,
        "scheduler_calls": len(scheduler_attempts),
    }
    result_path = evidence_root / f"CANCEL_RESULT.{token}.json"
    result_sha256 = seal_json(result_path, result)
    require(completed.returncode == 0, f"scancel failed after durable result seal: {completed.stderr.strip()}")
    return {**result, "cancel_result_path": str(result_path), "cancel_result_sha256": result_sha256}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--test-only", action="store_true", help="read-only plan (default)")
    actions.add_argument("--cancel", action="store_true", help="seal the latch, then call scancel")
    parser.add_argument("--submission-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = args.submission_root.absolute()
        value = explicit_cancel(root) if args.cancel else cancellation_plan(root)
        print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"Exp23 cancellation error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
