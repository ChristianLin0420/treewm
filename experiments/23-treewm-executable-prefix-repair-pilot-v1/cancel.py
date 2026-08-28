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
import pwd
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
CAMPAIGN_ID = "treewm-executable-prefix-repair-pilot-v1-launch8"
JOB_ID = re.compile(r"^[1-9][0-9]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "submission_authorization_sha256",
        "wave0_array_job_id",
        "wave1_array_job_id",
        "report_job_id",
        "array",
        "wave1_dependency",
        "report_dependency",
        "kill_on_invalid_dependency",
        "within_wave_requeue",
        "wave0_submitted_held",
    }
)
AUTHORIZATION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "array",
        "job_ids",
        "dependencies",
        "kill_on_invalid_dependency",
        "within_wave_requeue",
        "wave0_submitted_held",
        "accepted_job_evidence_sha256",
        "authorized_at_utc",
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
DEPENDENCY_TEST_REQUIREMENT = {
    "phases": [
        "after_wave0_reconciliation_before_wave1_submission",
        "after_wave1_reconciliation_before_report_submission",
    ],
    "dependencies": [
        "afterok:<accepted_wave0_array_job_id>",
        "afterok:<accepted_wave1_array_job_id>",
    ],
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


def _transaction_lock_path(submission_root: Path) -> Path:
    """Derive the exact persistent controller lock used by ``submit.py``."""

    absolute = submission_root.absolute()
    require(
        absolute.name == "submission" and absolute.parent.name == "state",
        "transaction lock requires the sealed <run_root>/state/submission layout",
    )
    token = hashlib.sha256(str(absolute).encode("utf-8")).hexdigest()[:16]
    return absolute.parents[2] / f".exp23-{token}.transaction.lock"


class _CancellationTransactionLock:
    """Block on the submit/recovery inode before publishing a cancel latch.

    Lock ordering is always external transaction lock, then ``_ReportCancelLock``.
    The persistent inode must already have been created by the controller; cancel
    never substitutes a new inode for the one which serialized submission.
    """

    def __init__(self, submission_root: Path) -> None:
        self.path = _transaction_lock_path(submission_root)
        self.descriptor: int | None = None

    def __enter__(self) -> "_CancellationTransactionLock":
        _directory(self.path.parent, "transaction-lock parent")
        flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
        try:
            descriptor = os.open(self.path, flags)
        except OSError as exc:
            raise CancellationError(f"cannot open existing transaction lock: {exc}") from exc
        try:
            opened = os.fstat(descriptor)
            named = self.path.lstat()
            require(
                stat.S_ISREG(opened.st_mode)
                and (opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino)
                and opened.st_uid == os.getuid()
                and opened.st_nlink == 1
                and stat.S_IMODE(opened.st_mode) == 0o600,
                "transaction lock identity/mode differs",
            )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            after = os.fstat(descriptor)
            current = self.path.lstat()
            require(
                (after.st_dev, after.st_ino) == (current.st_dev, current.st_ino),
                "transaction lock path changed while waiting",
            )
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


def _run_scheduler_client_with_lock_supervisor(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    inherited_fds: Sequence[int],
    transaction_lock_descriptor: int,
) -> subprocess.CompletedProcess[str]:
    """Retain the transaction flock in a supervisor until the client exits."""

    require(transaction_lock_descriptor >= 0, "cancellation transaction-lock descriptor differs")
    lock_info = os.fstat(transaction_lock_descriptor)
    require(
        stat.S_ISREG(lock_info.st_mode)
        and lock_info.st_uid == os.getuid()
        and stat.S_IMODE(lock_info.st_mode) == 0o600,
        "cancellation transaction-lock descriptor identity differs",
    )
    descriptors = tuple(
        dict.fromkeys([*map(int, inherited_fds), transaction_lock_descriptor])
    )
    supervisor = (
        "import os,subprocess,sys;"
        "fds=tuple(int(v) for v in sys.argv[1].split(',') if v);"
        "p=subprocess.run(sys.argv[2:],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,"
        "stderr=subprocess.PIPE,pass_fds=fds,close_fds=True);"
        "os.write(1,p.stdout);os.write(2,p.stderr);raise SystemExit(p.returncode)"
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            "-c",
            supervisor,
            ",".join(map(str, descriptors)),
            *map(str, command),
        ],
        cwd=cwd,
        env=dict(environment),
        check=False,
        text=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=descriptors,
    )
    return subprocess.CompletedProcess(
        list(map(str, command)),
        completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


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


def exact_json_equal(left: object, right: object) -> bool:
    """Compare JSON values recursively without Python's bool/int coercion."""

    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(exact_json_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(exact_json_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def _validated_published_report(
    submission_root: Path, receipt: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return an exact terminal report commit, or fail closed on any report entry."""

    report_root = submission_root / "report"
    if not _lexical_exists(report_root):
        return None
    _directory(report_root, "published report root")
    rows = _secure_tree_rows(report_root, "published report tree")
    roots = [row for row in rows if row["kind"] == "root"]
    files = {str(row["path"]): row for row in rows if row["kind"] == "file"}
    require(
        len(roots) == 1
        and roots[0]["mode"] == 0o555
        and len(files) == len(rows) - 1,
        "published report tree shape/mode differs",
    )
    commit_path = report_root / "REPORT_COMMIT.json"
    _regular(commit_path, "published report commit")
    commit = read_json(commit_path)
    require(
        set(commit)
        == {
            "schema_version",
            "status",
            "scientific_rejection",
            "campaign_id",
            "submission_sha256",
            "report_bundle",
            "report_bundle_sha256",
            "report_bundle_file_sha256",
            "gate_decision",
            "gate_sha256",
            "gate_decision_file_sha256",
            "provenance",
            "provenance_sha256",
            "provenance_file_sha256",
        }
        and type(commit.get("schema_version")) is int
        and commit.get("schema_version") == 1
        and commit.get("status") in {"accepted_engineering_pilot", "rejected"}
        and commit.get("scientific_rejection")
        is (commit.get("status") == "rejected")
        and commit.get("campaign_id") == receipt["campaign_id"]
        and commit.get("submission_sha256") == receipt["submission_sha256"],
        "published report commit differs",
    )
    for key in (
        "report_bundle_sha256",
        "report_bundle_file_sha256",
        "gate_sha256",
        "gate_decision_file_sha256",
        "provenance_sha256",
        "provenance_file_sha256",
    ):
        require(SHA256.fullmatch(str(commit.get(key, ""))) is not None, f"published report {key} differs")
    expected = {
        "REPORT_COMMIT.json",
        f"REPORT_BUNDLE.{commit['report_bundle_sha256']}.json",
        f"GATE_DECISION.{commit['gate_sha256']}.json",
        f"REPORT_PROVENANCE.{commit['provenance_sha256']}.json",
    }
    require(
        commit["report_bundle"] == f"REPORT_BUNDLE.{commit['report_bundle_sha256']}.json"
        and commit["gate_decision"] == f"GATE_DECISION.{commit['gate_sha256']}.json"
        and commit["provenance"] == f"REPORT_PROVENANCE.{commit['provenance_sha256']}.json"
        and set(files) == expected,
        "published report file coverage/names differ",
    )
    raw_bindings = {
        str(commit["report_bundle"]): str(commit["report_bundle_file_sha256"]),
        str(commit["gate_decision"]): str(commit["gate_decision_file_sha256"]),
        str(commit["provenance"]): str(commit["provenance_file_sha256"]),
    }
    require(
        all(row["mode"] == 0o444 for row in files.values())
        and all(files[name]["sha256"] == digest for name, digest in raw_bindings.items()),
        "published report file mode/hash differs",
    )
    bundle = read_json(report_root / str(commit["report_bundle"]))
    decision = read_json(report_root / str(commit["gate_decision"]))
    provenance = read_json(report_root / str(commit["provenance"]))
    decision_body = dict(decision)
    decision_hash = decision_body.pop("gate_sha256", None)
    require(
        stable_hash(bundle) == commit["report_bundle_sha256"]
        and stable_hash(provenance) == commit["provenance_sha256"]
        and decision_hash == commit["gate_sha256"]
        and stable_hash(decision_body) == commit["gate_sha256"]
        and decision.get("status") == commit["status"],
        "published report logical hashes differ",
    )
    return commit


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
        and exact_json_equal(control_plane, SCHEDULER_CONTROL_PLANE),
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
        and exact_json_equal(control_plane, SCHEDULER_CONTROL_PLANE),
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
        and type(value.get("schema_version")) is int
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
        and exact_json_equal(control_plane, SCHEDULER_CONTROL_PLANE),
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


def _reconcile_exact_cancel_ids(
    *,
    snapshot_root: Path,
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    fallback_binding: Mapping[str, Any],
    fallback_payload: bytes,
    expected_control_plane: Mapping[str, Any],
    transaction_lock_descriptor: int,
    reconciled_call_records: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[str], dict[str, Any]]:
    """Resolve an intent-without-result using one authenticated exact-ID query."""

    squeue = Path(str(manifest["execution"].get("squeue")))
    _regular(squeue, "cancellation reconciliation squeue")
    require(os.access(squeue, os.X_OK), "cancellation reconciliation squeue is not executable")
    ids = [
        receipt["wave0_array_job_id"],
        receipt["wave1_array_job_id"],
        receipt["report_job_id"],
    ]
    token = str(receipt["submission_sha256"])[:16]
    names = {
        ids[0]: f"exp23-launch8-{token}-wave0",
        ids[1]: f"exp23-launch8-{token}-wave1",
        ids[2]: f"exp23-launch8-{token}-report",
    }
    comment = f"treewm-exp23:{receipt['submission_sha256']}"
    command = [
        str(squeue),
        "--noheader",
        f"--jobs={','.join(ids)}",
        "--format=%A|%j|%u|%T|%k",
    ]
    control_plane = manifest["execution"].get("scheduler_control_plane")
    boundary_error: str | None = None
    descriptor: int | None = None
    try:
        try:
            before = scheduler_control_plane_observation(snapshot_root, manifest)
            require(
                exact_json_equal(before, expected_control_plane),
                "scheduler control plane differs from the committed cancellation preclaim",
            )
            environment = scheduler_environment(control_plane)
            pass_fds: tuple[int, ...] = ()
            mode = "canonical_root_admin_config"
        except BaseException as exc:
            boundary_error = repr(exc)
            mode = "sealed_original_config_fallback"
            before = {
                "schema_version": 1,
                "mode": mode,
                "sha256": fallback_binding["sha256"],
                "size": fallback_binding["size"],
            }
            descriptor = _scheduler_fallback_descriptor(fallback_payload)
            environment = {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "LANG": "C",
                "LC_ALL": "C",
                "SLURM_CONF": f"/proc/self/fd/{descriptor}",
            }
            pass_fds = (descriptor,)
        completed = _run_scheduler_client_with_lock_supervisor(
            command,
            cwd=snapshot_root,
            environment=environment,
            inherited_fds=pass_fds,
            transaction_lock_descriptor=transaction_lock_descriptor,
        )
        require(
            completed.returncode == 0
            and len(completed.stdout) <= 1024 * 1024
            and len(completed.stderr) <= 1024 * 1024,
            "exact-ID cancellation reconciliation query failed",
        )
        if mode == "canonical_root_admin_config":
            after = scheduler_control_plane_observation(snapshot_root, manifest)
            require(
                exact_json_equal(after, before)
                and exact_json_equal(before, expected_control_plane),
                "scheduler control plane changed during cancellation reconciliation",
            )
        else:
            after = before
    finally:
        if descriptor is not None:
            os.close(descriptor)
    active: set[str] = set()
    expected_user = pwd.getpwuid(os.getuid()).pw_name
    for raw in completed.stdout.splitlines():
        fields = raw.split("|", 4)
        require(len(fields) == 5, "cancellation reconciliation row differs")
        job_id, job_name, user, state, observed_comment = fields
        require(
            job_id in names
            and job_name == names[job_id]
            and user == expected_user
            and bool(state)
            and observed_comment == comment,
            "cancellation reconciliation identity differs",
        )
        active.add(job_id)
    evidence = {
        "kind": "intent_without_result_exact_id_reconciliation",
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "active_job_ids": sorted(active, key=int),
        "scheduler_mode": mode,
        "scheduler_control_plane_before": before,
        "scheduler_control_plane_after": after,
        "canonical_boundary_error": boundary_error,
        "reconciled_call_records": [dict(item) for item in reconciled_call_records],
    }
    return sorted(active, key=int), evidence


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
    require(
        type(receipt["schema_version"]) is int
        and receipt["schema_version"] == 1
        and receipt["status"] == "committed_two_wave_dag",
        "submission receipt is not committed",
    )
    require(receipt["campaign_id"] == CAMPAIGN_ID, "submission receipt campaign differs")
    claimed = str(receipt["submission_sha256"])
    require(SHA256.fullmatch(claimed) is not None and file_sha256(contract_path) == claimed, "submission contract hash differs")
    seal_path = submission_root / "journal" / "0002_CONTRACT_SEALED.json"
    _regular(seal_path, "contract-seal journal")
    require(stat.S_IMODE(seal_path.lstat().st_mode) == 0o444, "contract-seal journal mode differs")
    require(
        exact_json_equal(
            read_json(seal_path),
            {
            "schema_version": 1,
            "record": "contract_sealed",
            "submission_sha256": claimed,
            "launch_count": 20,
            },
        ),
        "contract-seal journal differs",
    )
    wave0_id = receipt["wave0_array_job_id"]
    wave1_id = receipt["wave1_array_job_id"]
    report_id = receipt["report_job_id"]
    require(
        all(
            isinstance(value, str) and JOB_ID.fullmatch(value) is not None
            for value in (wave0_id, wave1_id, report_id)
        ),
        "receipt job ID is malformed",
    )
    require(len({wave0_id, wave1_id, report_id}) == 3, "receipt job IDs are not distinct")
    require(receipt["array"] == "0-19%20", "receipt array differs")
    require(
        receipt["wave1_dependency"] == f"afterok:{wave0_id}"
        and receipt["report_dependency"] == f"afterok:{wave1_id}"
        and exact_json_equal(
            receipt["kill_on_invalid_dependency"],
            {"wave1": "yes", "report": "yes"},
        )
        and receipt["within_wave_requeue"] is False
        and receipt["wave0_submitted_held"] is True,
        "receipt DAG differs",
    )
    authorization_path = submission_root / "SUBMISSION_AUTHORIZATION.json"
    _regular(authorization_path, "submission authorization")
    require(
        stat.S_IMODE(authorization_path.lstat().st_mode) == 0o444
        and file_sha256(authorization_path)
        == receipt["submission_authorization_sha256"],
        "submission authorization bytes differ",
    )
    authorization = read_json(authorization_path)
    job_ids = {"wave0": wave0_id, "wave1": wave1_id, "report": report_id}
    dependencies = {
        "wave0": "none",
        "wave1": f"afterok:{wave0_id}",
        "report": f"afterok:{wave1_id}",
    }
    require(
        set(authorization) == AUTHORIZATION_KEYS
        and type(authorization.get("schema_version")) is int
        and authorization.get("schema_version") == 1
        and authorization.get("status") == "authorized_two_wave_dag"
        and authorization.get("campaign_id") == receipt["campaign_id"]
        and authorization.get("submission_sha256") == claimed
        and authorization.get("array") == "0-19%20"
        and exact_json_equal(authorization.get("job_ids"), job_ids)
        and exact_json_equal(authorization.get("dependencies"), dependencies)
        and exact_json_equal(
            authorization.get("kill_on_invalid_dependency"),
            {"wave1": "yes", "report": "yes"},
        )
        and authorization.get("within_wave_requeue") is False
        and authorization.get("wave0_submitted_held") is True
        and SHA256.fullmatch(str(authorization.get("accepted_job_evidence_sha256", "")))
        is not None
        and isinstance(authorization.get("authorized_at_utc"), str)
        and bool(authorization["authorized_at_utc"]),
        "submission authorization DAG differs",
    )
    records: dict[str, Any] = {}
    for role, ordinal, expected_id in (
        ("wave0", 3, wave0_id),
        ("wave1", 4, wave1_id),
        ("report", 5, report_id),
    ):
        submitted_path = submission_root / "journal" / f"{ordinal:04d}_{role.upper()}_SUBMITTED.json"
        _regular(submitted_path, f"{role} submission journal")
        require(stat.S_IMODE(submitted_path.lstat().st_mode) == 0o444, f"{role} submission journal mode differs")
        submitted = read_json(submitted_path)
        require(
            type(submitted.get("schema_version")) is int
            and submitted.get("schema_version") == 1
            and submitted.get("record") == f"{role}_submitted"
            and isinstance(submitted.get("job_id"), str)
            and submitted.get("job_id") == expected_id,
            f"receipt {role} job differs from durable journal",
        )
        records[role] = {
            key: value
            for key, value in submitted.items()
            if key not in {"schema_version", "record", "job_id"}
        }
    require(
        stable_hash(records) == authorization["accepted_job_evidence_sha256"],
        "DAG accepted-job evidence hash differs",
    )
    authorized_path = submission_root / "journal" / "0006_DAG_AUTHORIZED.json"
    _regular(authorized_path, "durable DAG authorization journal")
    require(stat.S_IMODE(authorized_path.lstat().st_mode) == 0o444, "DAG authorization journal mode differs")
    require(
        exact_json_equal(
            read_json(authorized_path),
            {
            "schema_version": 1,
            "record": "dag_authorized",
            "submission_authorization_sha256": receipt[
                "submission_authorization_sha256"
            ],
            "accepted_job_evidence_sha256": authorization[
                "accepted_job_evidence_sha256"
            ],
            "job_ids": job_ids,
            "dependencies": dependencies,
            },
        ),
        "durable DAG authorization journal differs",
    )
    ready_path = submission_root / "journal" / "0007_READY_TO_COMMIT.json"
    _regular(ready_path, "durable ready-to-commit journal")
    require(stat.S_IMODE(ready_path.lstat().st_mode) == 0o444, "ready-to-commit journal mode differs")
    require(
        exact_json_equal(
            read_json(ready_path),
            {"schema_version": 1, "record": "ready_to_commit", **receipt},
        ),
        "durable ready-to-commit journal differs",
    )
    contract = read_json(contract_path)
    require(contract.get("status") == "sealed_for_submission", "submission contract is not sealed")
    require(contract.get("submission_root") == str(submission_root.absolute()), "contract submission root differs")
    require(Path(str(contract.get("snapshot_root", ""))).is_absolute(), "contract snapshot root is not absolute")
    require(len(contract.get("launches") or []) == 20, "submission contract launch coverage differs")
    snapshot_root = Path(str(contract["snapshot_root"]))
    _directory(snapshot_root, "snapshot root")
    snapshot = snapshot_root.absolute()
    require(snapshot.is_relative_to(submission_root), "snapshot root escapes submission root")
    require(str(snapshot) == contract["snapshot_root"], "snapshot root is not canonical")
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
        exact_json_equal(
            manifest["execution"].get("scheduler_control_plane"),
            SCHEDULER_CONTROL_PLANE,
        )
        and exact_json_equal(
            contract.get("scheduler_control_plane_contract"),
            SCHEDULER_CONTROL_PLANE,
        ),
        "scheduler control-plane contract differs",
    )
    scheduler_preclaim = contract.get("scheduler_preclaim")
    require(
        isinstance(scheduler_preclaim, Mapping)
        and type(scheduler_preclaim.get("schema_version")) is int
        and scheduler_preclaim.get("schema_version") == 1
        and scheduler_preclaim.get("status") == "scheduler_preclaim_verified"
        and scheduler_preclaim.get("campaign_id") == receipt["campaign_id"]
        and type(scheduler_preclaim.get("scheduler_calls")) is int
        and scheduler_preclaim.get("scheduler_calls") == 10
        and type(scheduler_preclaim.get("scheduler_mutation_calls")) is int
        and scheduler_preclaim.get("scheduler_mutation_calls") == 0
        and type(scheduler_preclaim.get("persistent_writes_performed")) is int
        and scheduler_preclaim.get("persistent_writes_performed") == 0,
        "scheduler preclaim contract differs",
    )
    require(
        isinstance(scheduler_preclaim.get("scheduler_probe_commands"), list)
        and len(scheduler_preclaim["scheduler_probe_commands"]) == 10
        and exact_json_equal(
            scheduler_preclaim.get("dependency_tests"),
            DEPENDENCY_TEST_REQUIREMENT,
        )
        and exact_json_equal(
            scheduler_preclaim.get("zero_job_proof"),
            {
                "job_names": {
                    "wave0": "exp23-launch8-scheduler-test-wave0",
                    "wave1": "exp23-launch8-scheduler-test-wave1",
                    "report": "exp23-launch8-scheduler-test-report",
                },
                "pre_queries": 3,
                "post_queries": 3,
                "matching_jobs_before": 0,
                "matching_jobs_after": 0,
            },
        ),
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
    ids = [
        receipt["wave0_array_job_id"],
        receipt["wave1_array_job_id"],
        receipt["report_job_id"],
    ]
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
        "wave0_array_job_id": receipt["wave0_array_job_id"],
        "wave1_array_job_id": receipt["wave1_array_job_id"],
        "report_job_id": receipt["report_job_id"],
    }
    path = submission_root / "CANCEL_REQUESTED.json"

    def validate_existing(label: str) -> None:
        _regular(path, label)
        existing = read_json(path)
        require(
            stat.S_IMODE(path.lstat().st_mode) == 0o444
            and type(existing.get("schema_version")) is int
            and existing == value,
            f"{label} differs",
        )

    if _lexical_exists(path):
        validate_existing("existing cancellation latch")
        return value
    try:
        seal_json(path, value)
    except FileExistsError:
        validate_existing("concurrent cancellation latch")
    validate_existing("sealed cancellation latch")
    return value


CALL_TOKEN = re.compile(r"^[1-9][0-9]*-[1-9][0-9]*(?:\.(?:fallback|reconciled))?$")


def _fallback_attempt_observation(
    fallback_binding: Mapping[str, Any], mode: str
) -> dict[str, Any]:
    require(
        mode
        in {
            "sealed_original_config_fallback",
            "sealed_original_config_fallback_after_canonical_failure",
            "sealed_original_config_fallback_after_unknown_response",
            "sealed_original_config_fallback_residual",
        }
        and SHA256.fullmatch(str(fallback_binding.get("sha256", ""))) is not None
        and type(fallback_binding.get("size")) is int
        and fallback_binding["size"] > 0,
        "cancellation fallback attempt binding differs",
    )
    return {
        "schema_version": 1,
        "mode": mode,
        "sha256": fallback_binding["sha256"],
        "size": fallback_binding["size"],
    }


def _cancel_call_inventory(
    evidence_root: Path,
    *,
    receipt: Mapping[str, Any],
    ids: Sequence[str],
    scancel: str,
    expected_control_plane: Mapping[str, Any],
    fallback_binding: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    _directory(evidence_root, "cancellation evidence root")
    result: dict[str, dict[str, Any]] = {}
    for entry in os.scandir(evidence_root):
        require(
            not entry.name.startswith("CANCEL_FAILURE."),
            "unsupported durable cancellation failure artifact exists",
        )
        if not entry.name.startswith("CANCEL_CALL.") or not entry.name.endswith(".json"):
            continue
        token = entry.name[len("CANCEL_CALL.") : -len(".json")]
        require(CALL_TOKEN.fullmatch(token) is not None, "cancel call token differs")
        path = evidence_root / entry.name
        _regular(path, f"cancel call {token}")
        require(
            stat.S_IMODE(path.lstat().st_mode) == 0o444,
            f"cancel call {token} mode differs",
        )
        value = read_json(path)
        require(
            set(value)
            == {
                "schema_version",
                "status",
                "campaign_id",
                "submission_sha256",
                "call_token",
                "job_ids",
                "command",
                "scheduler_control_plane",
                "scheduler_mode",
                "canonical_boundary_error",
            }
            and type(value.get("schema_version")) is int
            and value["schema_version"] == 1
            and value.get("status") == "exact_cancel_call_intent"
            and value.get("campaign_id") == receipt["campaign_id"]
            and value.get("submission_sha256") == receipt["submission_sha256"]
            and value.get("call_token") == token
            and value.get("job_ids") == list(ids)
            and isinstance(value.get("command"), list)
            and len(value["command"]) >= 2
            and all(isinstance(item, str) for item in value["command"])
            and value["command"][0] == scancel
            and all(JOB_ID.fullmatch(item) is not None for item in value["command"][1:])
            and value["command"][1:]
            == sorted(set(map(str, value["command"][1:])), key=int)
            and set(value["command"][1:]).issubset(ids)
            and value.get("scheduler_mode")
            in {
                "canonical_root_admin_config",
                "sealed_original_config_fallback",
                "sealed_original_config_fallback_after_canonical_failure",
                "sealed_original_config_fallback_after_unknown_response",
                "sealed_original_config_fallback_residual",
            },
            f"cancel call {token} identity differs",
        )
        mode = str(value["scheduler_mode"])
        if mode == "canonical_root_admin_config":
            require(
                exact_json_equal(
                    value.get("scheduler_control_plane"), expected_control_plane
                )
                and value.get("canonical_boundary_error") is None,
                f"cancel call {token} canonical binding differs",
            )
        else:
            require(
                exact_json_equal(
                    value.get("scheduler_control_plane"),
                    _fallback_attempt_observation(fallback_binding, mode),
                )
                and isinstance(value.get("canonical_boundary_error"), str)
                and bool(value["canonical_boundary_error"]),
                f"cancel call {token} fallback binding differs",
            )
        result[token] = {
            "name": entry.name,
            "sha256": file_sha256(path),
            "call_token": token,
            "value": value,
        }
    return result


def _validated_prior_cancel_result(
    submission_root: Path,
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    command: Sequence[str],
    fallback_binding: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Purely authenticate the immutable initial cancellation result and commit."""

    evidence_root = submission_root / "cancellation"
    result_path = evidence_root / "CANCEL_RESULT.json"
    commit_path = evidence_root / "CANCEL_COMMIT.json"
    result_present = _lexical_exists(result_path)
    commit_present = _lexical_exists(commit_path)
    require(not commit_present or result_present, "cancel commit exists without its result")
    if not result_present:
        return None
    _regular(result_path, "durable cancellation result")
    require(stat.S_IMODE(result_path.lstat().st_mode) == 0o444, "cancellation result mode differs")
    result = read_json(result_path)
    ids = [
        receipt["wave0_array_job_id"],
        receipt["wave1_array_job_id"],
        receipt["report_job_id"],
    ]
    latch_path = submission_root / "CANCEL_REQUESTED.json"
    _regular(latch_path, "durable cancellation latch")
    latch_value = read_json(latch_path)
    require(
        stat.S_IMODE(latch_path.lstat().st_mode) == 0o444
        and type(latch_value.get("schema_version")) is int
        and exact_json_equal(
            latch_value,
            {
            "schema_version": 1,
            "status": "cancel_requested",
            "campaign_id": receipt["campaign_id"],
            "submission_sha256": receipt["submission_sha256"],
            "wave0_array_job_id": ids[0],
            "wave1_array_job_id": ids[1],
            "report_job_id": ids[2],
            },
        ),
        "durable cancellation latch differs",
    )
    cancel_latch_sha256 = file_sha256(latch_path)
    expected_control_plane = (contract.get("scheduler_preclaim") or {}).get(
        "scheduler_control_plane"
    )
    require(
        isinstance(expected_control_plane, Mapping),
        "cancellation result lacks its preclaim control-plane binding",
    )
    intent_path = evidence_root / "CANCEL_INTENT.json"
    _regular(intent_path, "durable cancellation intent")
    require(
        stat.S_IMODE(intent_path.lstat().st_mode) == 0o444,
        "durable cancellation intent mode differs",
    )
    durable_intent = {
        "schema_version": 1,
        "status": "exact_cancel_intent",
        "campaign_id": receipt["campaign_id"],
        "submission_sha256": receipt["submission_sha256"],
        "submission_authorization_sha256": receipt[
            "submission_authorization_sha256"
        ],
        "job_ids": ids,
        "command": list(command),
    }
    intent_value = read_json(intent_path)
    require(
        type(intent_value.get("schema_version")) is int
        and exact_json_equal(intent_value, durable_intent),
        "durable cancellation intent differs",
    )
    durable_intent_sha256 = file_sha256(intent_path)
    call_inventory = _cancel_call_inventory(
        evidence_root,
        receipt=receipt,
        ids=ids,
        scancel=str(command[0]),
        expected_control_plane=expected_control_plane,
        fallback_binding=fallback_binding,
    )
    require(
        set(result)
        == {
            "schema_version",
            "status",
            "campaign_id",
            "submission_sha256",
            "wave0_array_job_id",
            "wave1_array_job_id",
            "report_job_id",
            "call_token",
            "job_ids",
            "cancel_latch_sha256",
            "durable_intent_sha256",
            "requested_command",
            "executed_cancel_command",
            "reconciled_active_job_ids",
            "terminal_or_absent_job_ids",
            "returncode",
            "stdout",
            "stderr",
            "scheduler_control_plane",
            "scheduler_mode",
            "canonical_boundary_error",
            "scheduler_attempts",
            "scheduler_calls",
        }
        and type(result.get("schema_version")) is int
        and result.get("schema_version") == 1
        and result.get("status")
        in {
            "cancel_requested_and_all_exact_jobs_signalled",
            "cancel_reconciled_active_exact_jobs_signalled",
            "cancel_reconciled_all_exact_jobs_terminal_or_absent",
        }
        and result.get("campaign_id") == receipt["campaign_id"]
        and result.get("submission_sha256") == receipt["submission_sha256"]
        and result.get("wave0_array_job_id") == ids[0]
        and result.get("wave1_array_job_id") == ids[1]
        and result.get("report_job_id") == ids[2]
        and result.get("job_ids") == ids
        and result.get("cancel_latch_sha256") == cancel_latch_sha256
        and result.get("durable_intent_sha256") == durable_intent_sha256
        and result.get("requested_command") == list(command)
        and (
            result.get("reconciled_active_job_ids") is None
            or (
                isinstance(result["reconciled_active_job_ids"], list)
                and all(
                    isinstance(item, str) and JOB_ID.fullmatch(item) is not None
                    for item in result["reconciled_active_job_ids"]
                )
                and result["reconciled_active_job_ids"]
                == sorted(set(result["reconciled_active_job_ids"]), key=int)
                and set(result["reconciled_active_job_ids"]).issubset(ids)
            )
        )
        and isinstance(result.get("terminal_or_absent_job_ids"), list)
        and all(
            isinstance(item, str) and JOB_ID.fullmatch(item) is not None
            for item in result["terminal_or_absent_job_ids"]
        )
        and result["terminal_or_absent_job_ids"]
        == sorted(set(result["terminal_or_absent_job_ids"]), key=int)
        and set(result["terminal_or_absent_job_ids"]).issubset(ids)
        and (
            result.get("executed_cancel_command") == list(command)
            if result.get("reconciled_active_job_ids") is None
            else (
                result.get("executed_cancel_command")
                == (
                    [str(command[0]), *result["reconciled_active_job_ids"]]
                    if result["reconciled_active_job_ids"]
                    else None
                )
                and set(result["terminal_or_absent_job_ids"])
                == set(ids) - set(result["reconciled_active_job_ids"])
            )
        )
        and result.get("status")
        == (
            "cancel_requested_and_all_exact_jobs_signalled"
            if result.get("reconciled_active_job_ids") is None
            else (
                "cancel_reconciled_active_exact_jobs_signalled"
                if result["reconciled_active_job_ids"]
                else "cancel_reconciled_all_exact_jobs_terminal_or_absent"
            )
        )
        and type(result.get("returncode")) is int
        and result.get("returncode") == 0
        and isinstance(result.get("stdout"), str)
        and isinstance(result.get("stderr"), str)
        and (
            result.get("call_token") is None
            or (
                isinstance(result["call_token"], str)
                and CALL_TOKEN.fullmatch(result["call_token"]) is not None
            )
        )
        and isinstance(result.get("scheduler_attempts"), list)
        and type(result.get("scheduler_calls")) is int
        and result.get("scheduler_calls") == len(result["scheduler_attempts"])
        and result["scheduler_calls"] >= 1,
        "durable cancellation result differs",
    )
    matched_call_tokens: set[str] = set()
    reconciled_call_tokens: set[str] = set()
    unknown_call_tokens: set[str] = set()
    terminal_call: Mapping[str, Any] | None = None
    terminal_reconciliation: Mapping[str, Any] | None = None
    expected_squeue = str(manifest["execution"].get("squeue"))
    expected_user = pwd.getpwuid(os.getuid()).pw_name
    token_prefix = str(receipt["submission_sha256"])[:16]
    names = {
        ids[0]: f"exp23-launch8-{token_prefix}-wave0",
        ids[1]: f"exp23-launch8-{token_prefix}-wave1",
        ids[2]: f"exp23-launch8-{token_prefix}-report",
    }
    expected_comment = f"treewm-exp23:{receipt['submission_sha256']}"
    for index, attempt in enumerate(result["scheduler_attempts"]):
        require(isinstance(attempt, Mapping), f"cancellation attempt {index} differs")
        if attempt.get("kind") == "intent_without_result_exact_id_reconciliation":
            require(
                set(attempt)
                == {
                    "kind",
                    "command",
                    "returncode",
                    "stdout",
                    "stderr",
                    "active_job_ids",
                    "scheduler_mode",
                    "scheduler_control_plane_before",
                    "scheduler_control_plane_after",
                    "canonical_boundary_error",
                    "reconciled_call_records",
                }
                and attempt.get("command")
                == [
                    expected_squeue,
                    "--noheader",
                    f"--jobs={','.join(ids)}",
                    "--format=%A|%j|%u|%T|%k",
                ]
                and type(attempt.get("returncode")) is int
                and attempt["returncode"] == 0
                and isinstance(attempt.get("stdout"), str)
                and isinstance(attempt.get("stderr"), str)
                and isinstance(attempt.get("active_job_ids"), list)
                and all(
                    isinstance(item, str) and JOB_ID.fullmatch(item) is not None
                    for item in attempt["active_job_ids"]
                )
                and attempt["active_job_ids"]
                == sorted(set(attempt["active_job_ids"]), key=int)
                and set(attempt["active_job_ids"]).issubset(ids)
                and isinstance(attempt.get("reconciled_call_records"), list),
                f"cancellation reconciliation attempt {index} differs",
            )
            parsed: set[str] = set()
            for raw_line in attempt["stdout"].splitlines():
                fields = raw_line.split("|", 4)
                require(len(fields) == 5, "cancellation reconciliation output differs")
                job_id, job_name, user, state, comment = fields
                require(
                    job_id in names
                    and job_name == names[job_id]
                    and user == expected_user
                    and bool(state)
                    and comment == expected_comment,
                    "cancellation reconciliation identity differs",
                )
                parsed.add(job_id)
            require(
                attempt["active_job_ids"] == sorted(parsed, key=int),
                "cancellation reconciliation active IDs were not derived from stdout",
            )
            mode = attempt.get("scheduler_mode")
            if mode == "canonical_root_admin_config":
                require(
                    exact_json_equal(
                        attempt.get("scheduler_control_plane_before"),
                        expected_control_plane,
                    )
                    and exact_json_equal(
                        attempt.get("scheduler_control_plane_after"),
                        expected_control_plane,
                    )
                    and attempt.get("canonical_boundary_error") is None,
                    "cancellation reconciliation canonical binding differs",
                )
            else:
                require(
                    mode == "sealed_original_config_fallback"
                    and exact_json_equal(
                        attempt.get("scheduler_control_plane_before"),
                        _fallback_attempt_observation(fallback_binding, mode),
                    )
                    and exact_json_equal(
                        attempt.get("scheduler_control_plane_after"),
                        attempt.get("scheduler_control_plane_before"),
                    )
                    and isinstance(attempt.get("canonical_boundary_error"), str)
                    and bool(attempt["canonical_boundary_error"]),
                    "cancellation reconciliation fallback binding differs",
                )
            for raw_record in attempt["reconciled_call_records"]:
                require(
                    isinstance(raw_record, Mapping)
                    and set(raw_record) == {"name", "sha256", "call_token"}
                    and raw_record.get("call_token") in call_inventory
                    and raw_record.get("call_token") not in reconciled_call_tokens
                    and raw_record
                    == {
                        key: call_inventory[str(raw_record["call_token"])][key]
                        for key in ("name", "sha256", "call_token")
                    },
                    "cancellation reconciled call binding differs",
                )
                reconciled_call_tokens.add(str(raw_record["call_token"]))
            terminal_reconciliation = attempt
            continue
        require(
            set(attempt)
            == {
                "kind",
                "call_token",
                "call_intent_name",
                "call_intent_sha256",
                "command",
                "scheduler_mode",
                "returncode",
                "stdout",
                "stderr",
                "scheduler_control_plane_before",
                "scheduler_control_plane_after",
                "postcondition_error",
                "canonical_boundary_error",
            }
            and attempt.get("kind") == "exact_cancel_call"
            and isinstance(attempt.get("call_token"), str)
            and attempt["call_token"] in call_inventory
            and type(attempt.get("returncode")) is int
            and isinstance(attempt.get("stdout"), str)
            and isinstance(attempt.get("stderr"), str),
            f"cancellation call attempt {index} differs",
        )
        call = call_inventory[attempt["call_token"]]["value"]
        require(
            attempt["command"] == call["command"]
            and attempt["call_intent_name"]
            == call_inventory[attempt["call_token"]]["name"]
            and attempt["call_intent_sha256"]
            == call_inventory[attempt["call_token"]]["sha256"]
            and attempt["scheduler_mode"] == call["scheduler_mode"]
            and exact_json_equal(
                attempt["scheduler_control_plane_before"],
                call["scheduler_control_plane"],
            )
            and attempt["canonical_boundary_error"]
            == call["canonical_boundary_error"],
            f"cancellation call attempt {index} is detached from its intent",
        )
        if attempt["scheduler_mode"] == "canonical_root_admin_config":
            require(
                (
                    attempt["postcondition_error"] is None
                    and exact_json_equal(
                        attempt["scheduler_control_plane_after"],
                        expected_control_plane,
                    )
                )
                or (
                    isinstance(attempt["postcondition_error"], str)
                    and bool(attempt["postcondition_error"])
                    and attempt["scheduler_control_plane_after"] is None
                ),
                f"cancellation call attempt {index} postcondition differs",
            )
        else:
            require(
                exact_json_equal(
                    attempt["scheduler_control_plane_after"],
                    attempt["scheduler_control_plane_before"],
                )
                and attempt["postcondition_error"] is None,
                f"cancellation fallback attempt {index} postcondition differs",
            )
        matched_call_tokens.add(str(attempt["call_token"]))
        if attempt["postcondition_error"] is not None:
            unknown_call_tokens.add(str(attempt["call_token"]))
        terminal_call = attempt

    attempts = result["scheduler_attempts"]

    def consume_census(start: int) -> tuple[int, list[str], set[str]]:
        stop = start + 3
        require(stop <= len(attempts), "cancellation census is truncated")
        group = attempts[start:stop]
        require(
            all(
                isinstance(row, Mapping)
                and row.get("kind")
                == "intent_without_result_exact_id_reconciliation"
                for row in group
            )
            and group[0]["command"] == group[1]["command"] == group[2]["command"]
            and group[1]["active_job_ids"] == group[2]["active_job_ids"]
            and group[1]["reconciled_call_records"] == []
            and group[2]["reconciled_call_records"] == [],
            "cancellation census is not an exact settled three-round group",
        )
        round_zero_tokens = {
            str(record["call_token"])
            for record in group[0]["reconciled_call_records"]
        }
        return stop, list(group[2]["active_job_ids"]), round_zero_tokens

    cursor, mutation_authority, pre_census_tokens = consume_census(0)
    require(
        pre_census_tokens.isdisjoint(matched_call_tokens),
        "initial census reconciles a call that occurs later in the result",
    )
    if cursor < len(attempts):
        first_call = attempts[cursor]
        require(
            isinstance(first_call, Mapping)
            and first_call.get("kind") == "exact_cancel_call"
            and mutation_authority
            and first_call.get("command")
            == [str(command[0]), *mutation_authority]
            and first_call.get("scheduler_mode")
            in {"canonical_root_admin_config", "sealed_original_config_fallback"},
            "cancellation call was not authorized by the settled exact census",
        )
        cursor += 1
        if first_call.get("postcondition_error") is not None:
            cursor, mutation_authority, post_census_tokens = consume_census(cursor)
            require(
                post_census_tokens == {str(first_call["call_token"])},
                "post-call census does not reconcile the unknown response",
            )
            if cursor < len(attempts):
                retry_call = attempts[cursor]
                require(
                    isinstance(retry_call, Mapping)
                    and retry_call.get("kind") == "exact_cancel_call"
                    and mutation_authority
                    and retry_call.get("command")
                    == [str(command[0]), *mutation_authority]
                    and retry_call.get("scheduler_mode")
                    == "sealed_original_config_fallback_after_unknown_response"
                    and retry_call.get("returncode") == 0
                    and retry_call.get("postcondition_error") is None,
                    "cancellation retry was not authorized by fresh exact census",
                )
                cursor += 1
            else:
                require(
                    mutation_authority == [],
                    "unverifiable cancellation response left active IDs unhandled",
                )
        else:
            require(
                first_call.get("returncode") == 0,
                "terminal cancellation result contains a failed scheduler call",
            )
    require(
        cursor == len(attempts)
        and result["reconciled_active_job_ids"] == mutation_authority,
        "cancellation scheduler-attempt lineage differs",
    )
    require(
        matched_call_tokens.intersection(reconciled_call_tokens)
        == unknown_call_tokens
        and matched_call_tokens | reconciled_call_tokens <= set(call_inventory),
        "durable cancellation call intents are not fully accounted",
    )
    if result["reconciled_active_job_ids"] == []:
        require(
            terminal_reconciliation is not None
            and result["executed_cancel_command"] is None
            and result["call_token"] is None
            and result["returncode"] == terminal_reconciliation["returncode"]
            and result["stdout"] == terminal_reconciliation["stdout"]
            and result["stderr"] == terminal_reconciliation["stderr"]
            and result["scheduler_mode"]
            == terminal_reconciliation["scheduler_mode"]
            and exact_json_equal(
                result["scheduler_control_plane"],
                terminal_reconciliation["scheduler_control_plane_before"],
            )
            and result["canonical_boundary_error"]
            == terminal_reconciliation["canonical_boundary_error"],
            "reconciled terminal cancellation summary differs",
        )
    else:
        require(
            terminal_call is not None
            and terminal_call["returncode"] == 0
            and terminal_call["postcondition_error"] is None
            and result["call_token"] == terminal_call["call_token"]
            and result["executed_cancel_command"] == terminal_call["command"]
            and result["returncode"] == terminal_call["returncode"]
            and result["stdout"] == terminal_call["stdout"]
            and result["stderr"] == terminal_call["stderr"]
            and result["scheduler_mode"] == terminal_call["scheduler_mode"]
            and exact_json_equal(
                result["scheduler_control_plane"],
                terminal_call["scheduler_control_plane_before"],
            )
            and result["canonical_boundary_error"]
            == terminal_call["canonical_boundary_error"],
            "cancellation terminal call summary differs",
        )
    result_sha256 = file_sha256(result_path)
    commit = {
        "schema_version": 1,
        "status": "exact_cancel_signal_evidence_committed",
        "campaign_id": receipt["campaign_id"],
        "submission_sha256": receipt["submission_sha256"],
        "job_ids": ids,
        "requested_command": list(command),
        "result": "CANCEL_RESULT.json",
        "result_sha256": result_sha256,
    }
    if commit_present:
        _regular(commit_path, "durable cancellation commit")
        require(stat.S_IMODE(commit_path.lstat().st_mode) == 0o444, "cancellation commit mode differs")
        commit_value = read_json(commit_path)
        require(
            type(commit_value.get("schema_version")) is int
            and exact_json_equal(commit_value, commit),
            "durable cancellation commit differs",
        )
    else:
        seal_json(commit_path, commit)
    return {
        **result,
        "cancel_result_path": str(result_path),
        "cancel_result_sha256": result_sha256,
        "cancel_commit_path": str(commit_path),
        "cancel_commit_sha256": file_sha256(commit_path),
        "reused_durable_cancel_result": True,
        "scheduler_calls": 0,
    }


def _validated_cancel_census_evidence(
    value: Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    manifest: Mapping[str, Any],
    expected_control_plane: Mapping[str, Any],
    fallback_binding: Mapping[str, Any],
    expected_reconciled_call_records: Sequence[Mapping[str, Any]] = (),
) -> list[str]:
    ids = [
        receipt["wave0_array_job_id"],
        receipt["wave1_array_job_id"],
        receipt["report_job_id"],
    ]
    expected_squeue = str(manifest["execution"].get("squeue"))
    require(
        set(value)
        == {
            "kind",
            "command",
            "returncode",
            "stdout",
            "stderr",
            "active_job_ids",
            "scheduler_mode",
            "scheduler_control_plane_before",
            "scheduler_control_plane_after",
            "canonical_boundary_error",
            "reconciled_call_records",
        }
        and value.get("kind") == "intent_without_result_exact_id_reconciliation"
        and value.get("command")
        == [
            expected_squeue,
            "--noheader",
            f"--jobs={','.join(ids)}",
            "--format=%A|%j|%u|%T|%k",
        ]
        and type(value.get("returncode")) is int
        and value["returncode"] == 0
        and isinstance(value.get("stdout"), str)
        and isinstance(value.get("stderr"), str)
        and value.get("reconciled_call_records")
        == [dict(item) for item in expected_reconciled_call_records],
        "fresh cancellation census evidence differs",
    )
    token = str(receipt["submission_sha256"])[:16]
    names = {
        ids[0]: f"exp23-launch8-{token}-wave0",
        ids[1]: f"exp23-launch8-{token}-wave1",
        ids[2]: f"exp23-launch8-{token}-report",
    }
    expected_user = pwd.getpwuid(os.getuid()).pw_name
    comment = f"treewm-exp23:{receipt['submission_sha256']}"
    parsed: set[str] = set()
    for line in value["stdout"].splitlines():
        fields = line.split("|", 4)
        require(len(fields) == 5, "fresh cancellation census row differs")
        job_id, name, user, state, observed_comment = fields
        require(
            job_id in names
            and name == names[job_id]
            and user == expected_user
            and bool(state)
            and observed_comment == comment,
            "fresh cancellation census identity differs",
        )
        parsed.add(job_id)
    active = sorted(parsed, key=int)
    require(
        value.get("active_job_ids") == active,
        "fresh cancellation census active IDs were not derived from stdout",
    )
    if value.get("scheduler_mode") == "canonical_root_admin_config":
        require(
            exact_json_equal(
                value.get("scheduler_control_plane_before"), expected_control_plane
            )
            and exact_json_equal(
                value.get("scheduler_control_plane_after"), expected_control_plane
            )
            and value.get("canonical_boundary_error") is None,
            "fresh cancellation census canonical binding differs",
        )
    else:
        require(
            value.get("scheduler_mode") == "sealed_original_config_fallback"
            and exact_json_equal(
                value.get("scheduler_control_plane_before"),
                _fallback_attempt_observation(
                    fallback_binding, "sealed_original_config_fallback"
                ),
            )
            and exact_json_equal(
                value.get("scheduler_control_plane_after"),
                value.get("scheduler_control_plane_before"),
            )
            and isinstance(value.get("canonical_boundary_error"), str)
            and bool(value["canonical_boundary_error"]),
            "fresh cancellation census fallback binding differs",
        )
    return active


def _cancel_census_rounds(
    *,
    snapshot_root: Path,
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    fallback_binding: Mapping[str, Any],
    fallback_payload: bytes,
    transaction_lock_descriptor: int,
    reconciled_call_records: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], list[str]]:
    expected = contract["scheduler_preclaim"]["scheduler_control_plane"]
    rounds: list[dict[str, Any]] = []
    active: list[str] = []
    for index in range(3):
        active, evidence = _reconcile_exact_cancel_ids(
            snapshot_root=snapshot_root,
            receipt=receipt,
            manifest=manifest,
            fallback_binding=fallback_binding,
            fallback_payload=fallback_payload,
            expected_control_plane=expected,
            transaction_lock_descriptor=transaction_lock_descriptor,
            reconciled_call_records=(
                reconciled_call_records if index == 0 else ()
            ),
        )
        require(
            _validated_cancel_census_evidence(
                evidence,
                receipt=receipt,
                manifest=manifest,
                expected_control_plane=expected,
                fallback_binding=fallback_binding,
                expected_reconciled_call_records=(
                    reconciled_call_records if index == 0 else ()
                ),
            )
            == active,
            "fresh cancellation census self-validation differs",
        )
        rounds.append({"round": index, "evidence": evidence})
        if index < 2:
            time.sleep(0.25)
    require(
        rounds[-2]["evidence"]["active_job_ids"]
        == rounds[-1]["evidence"]["active_job_ids"],
        "fresh cancellation census did not settle",
    )
    return rounds, active


def _validated_cancel_continuation_chain(
    submission_root: Path,
    *,
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    fallback_binding: Mapping[str, Any],
) -> list[dict[str, Any]]:
    evidence_root = submission_root / "cancellation"
    _directory(evidence_root, "cancellation evidence root")
    pattern = re.compile(r"^CANCEL_CONTINUATION\.([0-9]{4})\.json$")
    indices = {
        int(match.group(1))
        for entry in os.scandir(evidence_root)
        if (match := pattern.fullmatch(entry.name)) is not None
    }
    require(
        indices == set(range(len(indices))),
        "cancel continuation records are not an append-only prefix",
    )
    ids = [
        receipt["wave0_array_job_id"],
        receipt["wave1_array_job_id"],
        receipt["report_job_id"],
    ]
    expected = contract["scheduler_preclaim"]["scheduler_control_plane"]
    call_inventory = _cancel_call_inventory(
        evidence_root,
        receipt=receipt,
        ids=ids,
        scancel=str(manifest["execution"]["scancel"]),
        expected_control_plane=expected,
        fallback_binding=fallback_binding,
    )
    previous_name = "CANCEL_RESULT.json"
    previous_sha256 = file_sha256(evidence_root / previous_name)
    initial_result = read_json(evidence_root / previous_name)
    initial_tokens: set[str] = set()
    for attempt in initial_result.get("scheduler_attempts", []):
        if isinstance(attempt, Mapping) and isinstance(
            attempt.get("call_token"), str
        ):
            initial_tokens.add(str(attempt["call_token"]))
        if isinstance(attempt, Mapping) and isinstance(
            attempt.get("reconciled_call_records"), list
        ):
            initial_tokens.update(
                str(item.get("call_token"))
                for item in attempt["reconciled_call_records"]
                if isinstance(item, Mapping)
            )
    chain: list[dict[str, Any]] = []
    seen_tokens: set[str] = set()
    for generation in range(len(indices)):
        name = f"CANCEL_CONTINUATION.{generation:04d}.json"
        path = evidence_root / name
        _regular(path, f"cancel continuation {generation}")
        require(
            stat.S_IMODE(path.lstat().st_mode) == 0o444,
            f"cancel continuation {generation} mode differs",
        )
        value = read_json(path)
        require(
            set(value)
            == {
                "schema_version",
                "status",
                "campaign_id",
                "submission_sha256",
                "generation",
                "previous_terminal_name",
                "previous_terminal_sha256",
                "job_ids",
                "pre_cancel_census_rounds",
                "active_job_ids",
                "reconciled_call_records",
                "cancel_call",
                "cancel_call_sha256",
                "post_cancel_census_rounds",
                "post_cancel_active_job_ids",
                "scheduler_calls",
            }
            and type(value.get("schema_version")) is int
            and value["schema_version"] == 1
            and value.get("campaign_id") == receipt["campaign_id"]
            and value.get("submission_sha256") == receipt["submission_sha256"]
            and type(value.get("generation")) is int
            and value.get("generation") == generation
            and value.get("previous_terminal_name") == previous_name
            and value.get("previous_terminal_sha256") == previous_sha256
            and value.get("job_ids") == ids
            and isinstance(value.get("pre_cancel_census_rounds"), list)
            and len(value["pre_cancel_census_rounds"]) == 3,
            f"cancel continuation {generation} identity differs",
        )
        raw_reconciled = value.get("reconciled_call_records")
        require(
            isinstance(raw_reconciled, list),
            f"cancel continuation {generation} reconciled-call ledger differs",
        )
        reconciled_tokens: set[str] = set()
        for raw_record in raw_reconciled:
            require(
                isinstance(raw_record, Mapping)
                and set(raw_record) == {"name", "sha256", "call_token"}
                and isinstance(raw_record.get("call_token"), str)
                and raw_record["call_token"] in call_inventory
                and raw_record
                == {
                    key: call_inventory[str(raw_record["call_token"])][key]
                    for key in ("name", "sha256", "call_token")
                }
                and raw_record["call_token"] not in reconciled_tokens,
                f"cancel continuation {generation} reconciled call differs",
            )
            reconciled_tokens.add(str(raw_record["call_token"]))
        require(
            not reconciled_tokens.intersection(initial_tokens | seen_tokens),
            f"cancel continuation {generation} reconciles an accounted call",
        )
        pre_active: list[str] = []
        pre_active_rounds: list[list[str]] = []
        for round_index, row in enumerate(value["pre_cancel_census_rounds"]):
            require(
                isinstance(row, Mapping)
                and set(row) == {"round", "evidence"}
                and type(row.get("round")) is int
                and row.get("round") == round_index
                and isinstance(row.get("evidence"), Mapping),
                f"cancel continuation {generation} pre-census differs",
            )
            pre_active = _validated_cancel_census_evidence(
                row["evidence"],
                receipt=receipt,
                manifest=manifest,
                expected_control_plane=expected,
                fallback_binding=fallback_binding,
                expected_reconciled_call_records=(
                    raw_reconciled if round_index == 0 else ()
                ),
            )
            pre_active_rounds.append(pre_active)
        require(
            pre_active_rounds[-2] == pre_active_rounds[-1]
            and value.get("active_job_ids") == pre_active,
            f"cancel continuation {generation} active IDs differ",
        )
        call = value.get("cancel_call")
        post_rounds = value.get("post_cancel_census_rounds")
        if pre_active:
            require(
                isinstance(call, Mapping)
                and set(call)
                == {
                    "kind",
                    "call_token",
                    "call_intent_name",
                    "call_intent_sha256",
                    "command",
                    "scheduler_mode",
                    "returncode",
                    "stdout",
                    "stderr",
                    "scheduler_control_plane_before",
                    "scheduler_control_plane_after",
                    "postcondition_error",
                    "canonical_boundary_error",
                }
                and call.get("kind") == "exact_cancel_call"
                and call.get("call_token") in call_inventory
                and call.get("call_intent_name")
                == call_inventory[str(call.get("call_token"))]["name"]
                and call.get("call_intent_sha256")
                == call_inventory[str(call.get("call_token"))]["sha256"]
                and call["call_token"]
                not in (initial_tokens | seen_tokens | reconciled_tokens)
                and call.get("command")
                == [str(manifest["execution"]["scancel"]), *pre_active]
                and call.get("scheduler_mode")
                == "sealed_original_config_fallback_residual"
                and exact_json_equal(
                    call.get("scheduler_control_plane_before"),
                    _fallback_attempt_observation(
                        fallback_binding,
                        "sealed_original_config_fallback_residual",
                    ),
                )
                and exact_json_equal(
                    call.get("scheduler_control_plane_after"),
                    call.get("scheduler_control_plane_before"),
                )
                and call.get("postcondition_error") is None
                and isinstance(call.get("canonical_boundary_error"), str)
                and bool(call["canonical_boundary_error"])
                and type(call.get("returncode")) is int
                and call["returncode"] == 0,
                f"cancel continuation {generation} residual call differs",
            )
            inventory = call_inventory[str(call["call_token"])]
            intent = inventory["value"]
            require(
                value.get("cancel_call_sha256") == inventory["sha256"]
                and call["command"] == intent["command"]
                and call["scheduler_mode"] == intent["scheduler_mode"]
                and exact_json_equal(
                    call["scheduler_control_plane_before"],
                    intent["scheduler_control_plane"],
                )
                and call["canonical_boundary_error"]
                == intent["canonical_boundary_error"]
                and isinstance(call.get("stdout"), str)
                and isinstance(call.get("stderr"), str),
                f"cancel continuation {generation} call intent differs",
            )
            seen_tokens.add(str(call["call_token"]))
            require(
                isinstance(post_rounds, list) and len(post_rounds) == 3,
                f"cancel continuation {generation} post-census differs",
            )
            post_active: list[str] = []
            post_active_rounds: list[list[str]] = []
            for round_index, row in enumerate(post_rounds):
                require(
                    isinstance(row, Mapping)
                    and set(row) == {"round", "evidence"}
                    and type(row.get("round")) is int
                    and row.get("round") == round_index
                    and isinstance(row.get("evidence"), Mapping),
                    f"cancel continuation {generation} post-census differs",
                )
                post_active = _validated_cancel_census_evidence(
                    row["evidence"],
                    receipt=receipt,
                    manifest=manifest,
                    expected_control_plane=expected,
                    fallback_binding=fallback_binding,
                )
                post_active_rounds.append(post_active)
            require(
                post_active_rounds[-2] == post_active_rounds[-1]
                and value.get("post_cancel_active_job_ids") == post_active
                and value.get("status")
                == (
                    "cancel_residual_signalled_pending_terminal"
                    if post_active
                    else "cancel_residual_reconciled_terminal"
                )
                and type(value.get("scheduler_calls")) is int
                and value.get("scheduler_calls") == 7,
                f"cancel continuation {generation} terminal status differs",
            )
        else:
            require(
                call is None
                and value.get("cancel_call_sha256") is None
                and post_rounds == []
                and value.get("post_cancel_active_job_ids") == []
                and value.get("status") == "cancel_residual_reconciled_terminal"
                and type(value.get("scheduler_calls")) is int
                and value.get("scheduler_calls") == 3,
                f"cancel continuation {generation} no-active status differs",
            )
        chain.append(value)
        seen_tokens.update(reconciled_tokens)
        previous_name = name
        previous_sha256 = file_sha256(path)
    require(
        initial_tokens | seen_tokens <= set(call_inventory),
        "cancel continuation chain references an unknown call intent",
    )
    return chain


def _cancel_accounted_call_tokens(
    prior: Mapping[str, Any], chain: Sequence[Mapping[str, Any]]
) -> set[str]:
    result: set[str] = set()
    for attempt in prior.get("scheduler_attempts", []):
        if not isinstance(attempt, Mapping):
            continue
        if isinstance(attempt.get("call_token"), str):
            result.add(str(attempt["call_token"]))
        for record in attempt.get("reconciled_call_records", []):
            if isinstance(record, Mapping) and isinstance(
                record.get("call_token"), str
            ):
                result.add(str(record["call_token"]))
    for generation in chain:
        call = generation.get("cancel_call")
        if isinstance(call, Mapping) and isinstance(call.get("call_token"), str):
            result.add(str(call["call_token"]))
        for record in generation.get("reconciled_call_records", []):
            if isinstance(record, Mapping) and isinstance(
                record.get("call_token"), str
            ):
                result.add(str(record["call_token"]))
    return result


def _revalidate_or_continue_prior_cancel(
    *,
    submission_root: Path,
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    latch: Mapping[str, Any],
    prior: Mapping[str, Any],
    fallback_binding: Mapping[str, Any],
    fallback_payload: bytes,
    transaction_lock_descriptor: int,
) -> dict[str, Any]:
    del latch
    snapshot_root = Path(str(contract["snapshot_root"]))
    chain = _validated_cancel_continuation_chain(
        submission_root,
        receipt=receipt,
        contract=contract,
        manifest=manifest,
        fallback_binding=fallback_binding,
    )
    ids = [
        receipt["wave0_array_job_id"],
        receipt["wave1_array_job_id"],
        receipt["report_job_id"],
    ]
    inventory = _cancel_call_inventory(
        submission_root / "cancellation",
        receipt=receipt,
        ids=ids,
        scancel=str(manifest["execution"]["scancel"]),
        expected_control_plane=contract["scheduler_preclaim"][
            "scheduler_control_plane"
        ],
        fallback_binding=fallback_binding,
    )
    accounted = _cancel_accounted_call_tokens(prior, chain)
    require(
        accounted <= set(inventory),
        "cancel terminal chain references an unknown call intent",
    )
    pending_records = [
        {
            key: inventory[token][key]
            for key in ("name", "sha256", "call_token")
        }
        for token in sorted(set(inventory) - accounted)
    ]
    pre_rounds, active = _cancel_census_rounds(
        snapshot_root=snapshot_root,
        receipt=receipt,
        contract=contract,
        manifest=manifest,
        fallback_binding=fallback_binding,
        fallback_payload=fallback_payload,
        transaction_lock_descriptor=transaction_lock_descriptor,
        reconciled_call_records=pending_records,
    )
    append_generation = bool(active) or bool(pending_records) or (
        bool(chain)
        and chain[-1]["status"] == "cancel_residual_signalled_pending_terminal"
    )
    if append_generation:
        cancel_call: dict[str, Any] | None = None
        cancel_call_sha256: str | None = None
        post_rounds: list[dict[str, Any]] = []
        post_active: list[str] = []
        if active:
            scancel = str(manifest["execution"]["scancel"])
            token = f"{time.time_ns()}-{os.getpid()}.reconciled"
            mode = "sealed_original_config_fallback_residual"
            observation = _fallback_attempt_observation(fallback_binding, mode)
            boundary_error = "residual exact-ID cancellation after fresh terminal revalidation"
            exact_command = [scancel, *active]
            intent = {
                "schema_version": 1,
                "status": "exact_cancel_call_intent",
                "campaign_id": receipt["campaign_id"],
                "submission_sha256": receipt["submission_sha256"],
                "call_token": token,
                "job_ids": [
                    receipt["wave0_array_job_id"],
                    receipt["wave1_array_job_id"],
                    receipt["report_job_id"],
                ],
                "command": exact_command,
                "scheduler_control_plane": observation,
                "scheduler_mode": mode,
                "canonical_boundary_error": boundary_error,
            }
            call_path = submission_root / "cancellation" / f"CANCEL_CALL.{token}.json"
            cancel_call_sha256 = seal_json(call_path, intent)
            descriptor = _scheduler_fallback_descriptor(fallback_payload)
            try:
                completed = _run_scheduler_client_with_lock_supervisor(
                    exact_command,
                    cwd=snapshot_root,
                    environment={
                        "PATH": "/usr/local/bin:/usr/bin:/bin",
                        "LANG": "C",
                        "LC_ALL": "C",
                        "SLURM_CONF": f"/proc/self/fd/{descriptor}",
                    },
                    inherited_fds=(descriptor,),
                    transaction_lock_descriptor=transaction_lock_descriptor,
                )
            finally:
                os.close(descriptor)
            require(
                completed.returncode == 0,
                "residual exact cancellation scheduler call failed",
            )
            cancel_call = {
                "kind": "exact_cancel_call",
                "call_token": token,
                "call_intent_name": call_path.name,
                "call_intent_sha256": cancel_call_sha256,
                "command": exact_command,
                "scheduler_mode": mode,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "scheduler_control_plane_before": observation,
                "scheduler_control_plane_after": observation,
                "postcondition_error": None,
                "canonical_boundary_error": boundary_error,
            }
            post_rounds, post_active = _cancel_census_rounds(
                snapshot_root=snapshot_root,
                receipt=receipt,
                contract=contract,
                manifest=manifest,
                fallback_binding=fallback_binding,
                fallback_payload=fallback_payload,
                transaction_lock_descriptor=transaction_lock_descriptor,
            )
        generation = len(chain)
        previous_path = (
            submission_root
            / "cancellation"
            / f"CANCEL_CONTINUATION.{generation - 1:04d}.json"
            if generation
            else submission_root / "cancellation" / "CANCEL_RESULT.json"
        )
        seal_json(
            submission_root
            / "cancellation"
            / f"CANCEL_CONTINUATION.{generation:04d}.json",
            {
                "schema_version": 1,
                "status": (
                    "cancel_residual_signalled_pending_terminal"
                    if post_active
                    else "cancel_residual_reconciled_terminal"
                ),
                "campaign_id": receipt["campaign_id"],
                "submission_sha256": receipt["submission_sha256"],
                "generation": generation,
                "previous_terminal_name": previous_path.name,
                "previous_terminal_sha256": file_sha256(previous_path),
                "job_ids": [
                    receipt["wave0_array_job_id"],
                    receipt["wave1_array_job_id"],
                    receipt["report_job_id"],
                ],
                "pre_cancel_census_rounds": pre_rounds,
                "active_job_ids": active,
                "reconciled_call_records": pending_records,
                "cancel_call": cancel_call,
                "cancel_call_sha256": cancel_call_sha256,
                "post_cancel_census_rounds": post_rounds,
                "post_cancel_active_job_ids": post_active,
                "scheduler_calls": 3 + (4 if cancel_call is not None else 0),
            },
        )
        chain = _validated_cancel_continuation_chain(
            submission_root,
            receipt=receipt,
            contract=contract,
            manifest=manifest,
            fallback_binding=fallback_binding,
        )
        inventory = _cancel_call_inventory(
            submission_root / "cancellation",
            receipt=receipt,
            ids=ids,
            scancel=str(manifest["execution"]["scancel"]),
            expected_control_plane=contract["scheduler_preclaim"][
                "scheduler_control_plane"
            ],
            fallback_binding=fallback_binding,
        )
        require(
            _cancel_accounted_call_tokens(prior, chain) == set(inventory),
            "cancel continuation did not account for every durable call intent",
        )
    return {
        **prior,
        "reused_durable_cancel_result": True,
        "residual_continuation_chain": chain,
        "fresh_revalidation_census_rounds": pre_rounds,
        "fresh_active_job_ids": active,
        "scheduler_calls": 3 + (4 if append_generation and active else 0),
    }


def explicit_cancel(submission_root: Path) -> dict[str, Any]:
    reject_environment()
    submission_root = submission_root.absolute()
    # A committed receipt is published while submit still owns the external lock.
    # Wait for that owner (or recovery) to finish, then make cancellation durable
    # under the same inode before any future recovery can consider a release.
    with _CancellationTransactionLock(submission_root) as transaction_lock:
        receipt, contract, manifest = validate_receipt(submission_root)
        with _ReportCancelLock(submission_root):
            committed_report = _validated_published_report(submission_root, receipt)
            if committed_report is not None:
                require(
                    not _lexical_exists(submission_root / "CANCEL_REQUESTED.json"),
                    "published report conflicts with a cancellation latch",
                )
                return {
                    "schema_version": 1,
                    "status": "report_already_committed_cancel_noop",
                    "campaign_id": receipt["campaign_id"],
                    "submission_sha256": receipt["submission_sha256"],
                    "report_commit": committed_report,
                    "writes_performed": 0,
                    "scheduler_calls": 0,
                }
            latch = seal_latch(submission_root, receipt)
        return _explicit_cancel_locked(
            submission_root,
            receipt,
            contract,
            manifest,
            latch,
            transaction_lock_descriptor=transaction_lock.descriptor,
        )


def _explicit_cancel_locked(
    submission_root: Path,
    receipt: Mapping[str, Any],
    contract: Mapping[str, Any],
    manifest: Mapping[str, Any],
    latch: Mapping[str, Any],
    *,
    transaction_lock_descriptor: int | None,
) -> dict[str, Any]:
    """Signal and durably commit cancellation while the transaction lock is held."""

    require(
        transaction_lock_descriptor is not None,
        "cancellation transaction lock is not held",
    )
    latch_path = submission_root / "CANCEL_REQUESTED.json"
    _regular(latch_path, "durable cancellation latch")
    require(
        stat.S_IMODE(latch_path.lstat().st_mode) == 0o444
        and exact_json_equal(read_json(latch_path), latch),
        "durable cancellation latch differs",
    )
    cancel_latch_sha256 = file_sha256(latch_path)

    # Resolve the executable only after the durable latch exists.  A missing scheduler
    # client therefore still leaves every worker with an authoritative stop request.
    snapshot_root = Path(str(contract["snapshot_root"]))
    scancel = Path(str(manifest["execution"]["scancel"]))
    _regular(scancel, "scancel")
    require(os.access(scancel, os.X_OK), "scancel is not executable")
    ids = [
        receipt["wave0_array_job_id"],
        receipt["wave1_array_job_id"],
        receipt["report_job_id"],
    ]
    require(all(JOB_ID.fullmatch(value) for value in ids), "refusing non-exact cancellation target")
    command = [str(scancel), *ids]
    evidence_root = submission_root / "cancellation"
    durable_intent = {
        "schema_version": 1,
        "status": "exact_cancel_intent",
        "campaign_id": receipt["campaign_id"],
        "submission_sha256": receipt["submission_sha256"],
        "submission_authorization_sha256": receipt[
            "submission_authorization_sha256"
        ],
        "job_ids": ids,
        "command": command,
    }
    durable_intent_path = evidence_root / "CANCEL_INTENT.json"
    intent_preexisting = _lexical_exists(durable_intent_path)
    if intent_preexisting:
        _regular(durable_intent_path, "durable cancellation intent")
        durable_intent_value = read_json(durable_intent_path)
        require(
            stat.S_IMODE(durable_intent_path.lstat().st_mode) == 0o444
            and type(durable_intent_value.get("schema_version")) is int
            and exact_json_equal(durable_intent_value, durable_intent),
            "durable cancellation intent differs",
        )
        durable_intent_sha256 = file_sha256(durable_intent_path)
    else:
        durable_intent_sha256 = seal_json(durable_intent_path, durable_intent)
    control_plane = manifest["execution"].get("scheduler_control_plane")
    fallback_binding, fallback_payload = scheduler_fallback_config(
        snapshot_root, manifest, contract
    )
    prior = _validated_prior_cancel_result(
        submission_root,
        receipt,
        contract,
        manifest,
        command,
        fallback_binding,
    )
    if prior is not None:
        return _revalidate_or_continue_prior_cancel(
            submission_root=submission_root,
            receipt=receipt,
            contract=contract,
            manifest=manifest,
            latch=latch,
            prior=prior,
            fallback_binding=fallback_binding,
            fallback_payload=fallback_payload,
            transaction_lock_descriptor=transaction_lock_descriptor,
        )
    expected_control_plane = contract["scheduler_preclaim"][
        "scheduler_control_plane"
    ]
    require(
        isinstance(expected_control_plane, Mapping),
        "cancellation scheduler preclaim is absent",
    )
    existing_calls = _cancel_call_inventory(
        evidence_root,
        receipt=receipt,
        ids=ids,
        scancel=str(scancel),
        expected_control_plane=expected_control_plane,
        fallback_binding=fallback_binding,
    )
    pre_cancel_rounds, reconciled_active_ids = _cancel_census_rounds(
        snapshot_root=snapshot_root,
        receipt=receipt,
        contract=contract,
        manifest=manifest,
        fallback_binding=fallback_binding,
        fallback_payload=fallback_payload,
        transaction_lock_descriptor=transaction_lock_descriptor,
        reconciled_call_records=[
            {
                key: value[key]
                for key in ("name", "sha256", "call_token")
            }
            for value in existing_calls.values()
        ],
    )
    canonical_boundary_error: str | None = None
    try:
        observation_before = scheduler_control_plane_observation(snapshot_root, manifest)
        require(
            exact_json_equal(observation_before, expected_control_plane),
            "scheduler control plane differs from the committed cancellation preclaim",
        )
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
    token = f"{time.time_ns()}-{os.getpid()}"
    scheduler_attempts: list[dict[str, Any]] = [
        dict(row["evidence"]) for row in pre_cancel_rounds
    ]
    exact_cancel_command = [
        str(scancel),
        *reconciled_active_ids,
    ]

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
            "command": exact_cancel_command,
            "scheduler_control_plane": dict(observation),
            "scheduler_mode": mode,
            "canonical_boundary_error": boundary_error,
        }
        call_intent_name = f"CANCEL_CALL.{call_token}.json"
        call_intent_sha256 = seal_json(
            evidence_root / call_intent_name, intent
        )
        value = _run_scheduler_client_with_lock_supervisor(
            exact_cancel_command,
            cwd=snapshot_root,
            environment=dict(call_environment),
            inherited_fds=tuple(pass_descriptors),
            transaction_lock_descriptor=transaction_lock_descriptor,
        )
        scheduler_attempts.append(
            {
                "kind": "exact_cancel_call",
                "call_token": call_token,
                "call_intent_name": call_intent_name,
                "call_intent_sha256": call_intent_sha256,
                "command": list(exact_cancel_command),
                "scheduler_mode": mode,
                "returncode": value.returncode,
                "stdout": value.stdout,
                "stderr": value.stderr,
                "scheduler_control_plane_before": dict(observation),
                "scheduler_control_plane_after": (
                    None
                    if mode == "canonical_root_admin_config"
                    else dict(observation)
                ),
                "postcondition_error": None,
                "canonical_boundary_error": boundary_error,
            }
        )
        return value

    if reconciled_active_ids == []:
        terminal_reconciliation = scheduler_attempts[-1]
        scheduler_mode = terminal_reconciliation["scheduler_mode"]
        observation_before = terminal_reconciliation[
            "scheduler_control_plane_before"
        ]
        canonical_boundary_error = terminal_reconciliation[
            "canonical_boundary_error"
        ]
        completed = subprocess.CompletedProcess(
            terminal_reconciliation["command"],
            terminal_reconciliation["returncode"],
            stdout=terminal_reconciliation["stdout"],
            stderr=terminal_reconciliation["stderr"],
        )
    elif scheduler_mode == "canonical_root_admin_config":
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
            if not exact_json_equal(observation_after, observation_before):
                raise CancellationError("scheduler control plane changed during scancel")
            scheduler_attempts[-1]["scheduler_control_plane_after"] = observation_after
        except BaseException as exc:
            canonical_boundary_error = repr(exc)
            scheduler_attempts[-1]["postcondition_error"] = canonical_boundary_error
            current_calls = _cancel_call_inventory(
                evidence_root,
                receipt=receipt,
                ids=ids,
                scancel=str(scancel),
                expected_control_plane=contract["scheduler_preclaim"][
                    "scheduler_control_plane"
                ],
                fallback_binding=fallback_binding,
            )
            reconciliation_rounds, reconciled_active_ids = _cancel_census_rounds(
                snapshot_root=snapshot_root,
                receipt=receipt,
                contract=contract,
                manifest=manifest,
                fallback_binding=fallback_binding,
                fallback_payload=fallback_payload,
                transaction_lock_descriptor=transaction_lock_descriptor,
                reconciled_call_records=[
                    {
                        key: current_calls[token][key]
                        for key in ("name", "sha256", "call_token")
                    }
                ],
            )
            scheduler_attempts.extend(
                dict(row["evidence"]) for row in reconciliation_rounds
            )
            reconciliation_evidence = scheduler_attempts[-1]
            if not reconciled_active_ids:
                scheduler_mode = reconciliation_evidence["scheduler_mode"]
                observation_before = reconciliation_evidence[
                    "scheduler_control_plane_before"
                ]
                canonical_boundary_error = reconciliation_evidence[
                    "canonical_boundary_error"
                ]
                completed = subprocess.CompletedProcess(
                    reconciliation_evidence["command"],
                    reconciliation_evidence["returncode"],
                    stdout=reconciliation_evidence["stdout"],
                    stderr=reconciliation_evidence["stderr"],
                )
                exact_cancel_command = []
            else:
                fallback_descriptor = _scheduler_fallback_descriptor(fallback_payload)
                try:
                    scheduler_mode = (
                        "sealed_original_config_fallback_after_unknown_response"
                    )
                    observation_before = _fallback_attempt_observation(
                        fallback_binding, scheduler_mode
                    )
                    environment = {
                        "PATH": "/usr/local/bin:/usr/bin:/bin",
                        "LANG": "C",
                        "LC_ALL": "C",
                        "SLURM_CONF": f"/proc/self/fd/{fallback_descriptor}",
                    }
                    exact_cancel_command = [str(scancel), *reconciled_active_ids]
                    completed = call_exact(
                        call_token=f"{token}.reconciled",
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
            "cancel_requested_and_all_exact_jobs_signalled"
            if reconciled_active_ids is None and completed.returncode == 0
            else (
                "cancel_reconciled_all_exact_jobs_terminal_or_absent"
                if reconciled_active_ids == [] and completed.returncode == 0
                else (
                    "cancel_reconciled_active_exact_jobs_signalled"
                    if completed.returncode == 0
                    else "cancel_requested_scheduler_call_failed"
                )
            )
        ),
        "call_token": (
            None
            if reconciled_active_ids == []
            else str(scheduler_attempts[-1]["call_token"])
        ),
        "job_ids": ids,
        "cancel_latch_sha256": cancel_latch_sha256,
        "durable_intent_sha256": durable_intent_sha256,
        "requested_command": command,
        "executed_cancel_command": (
            None if reconciled_active_ids == [] else exact_cancel_command
        ),
        "reconciled_active_job_ids": reconciled_active_ids,
        "terminal_or_absent_job_ids": (
            []
            if reconciled_active_ids is None
            else sorted(set(ids) - set(reconciled_active_ids), key=int)
        ),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "scheduler_control_plane": observation_before,
        "scheduler_mode": scheduler_mode,
        "canonical_boundary_error": canonical_boundary_error,
        "scheduler_attempts": scheduler_attempts,
        "scheduler_calls": len(scheduler_attempts),
    }
    if completed.returncode != 0:
        raise CancellationError(
            "scancel returned nonzero; its durable CALL intent remains pending and "
            f"the next invocation must reconcile it: {completed.stderr.strip()}"
        )
    result_path = evidence_root / "CANCEL_RESULT.json"
    result_sha256 = seal_json(result_path, result)
    committed = _validated_prior_cancel_result(
        submission_root,
        receipt,
        contract,
        manifest,
        command,
        fallback_binding,
    )
    assert committed is not None
    return {
        **committed,
        "cancel_result_path": str(result_path),
        "cancel_result_sha256": result_sha256,
        "reused_durable_cancel_result": False,
        "scheduler_calls": len(scheduler_attempts),
    }


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
