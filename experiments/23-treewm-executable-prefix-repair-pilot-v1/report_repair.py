#!/usr/bin/env python3
"""One-generation, append-only repair for the failed Launch8 report publisher.

This controller is deliberately narrower than the scientific submitter.  It can
authorize exactly one replacement *publisher* for the already-completed Launch8
submission; it cannot launch training, alter the report inputs/gate, or retry a
terminal repair failure.  The default action is read-only.  The explicit submit
action holds both the production transaction lock and the shared report/cancel
lock across every scheduler mutation.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pwd
import re
import shutil
import stat
import subprocess
import sys
import time
from types import ModuleType
from typing import Any, Callable, Mapping, NamedTuple, Sequence


sys.dont_write_bytecode = True
PACKAGE_RELATIVE = Path("experiments/23-treewm-executable-prefix-repair-pilot-v1")
PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]
CAMPAIGN_ID = "treewm-executable-prefix-repair-pilot-v1-launch8"
ATTEMPT = 1
CONFIRMATION = "SUBMIT_EXP23_LAUNCH8_REPORT_REPAIR_0001"
CANONICAL_PRODUCTION_SUBMISSION_ROOT = Path(
    "/lustre/fs11/portfolios/edgeai/projects/"
    "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
    "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch8/"
    "state/submission"
)
EXPECTED_SUBMISSION_SHA256 = (
    "bbeaa71f8f37f22cbe74c16c68b733742e8a4366838812832180257d145f5418"
)
EXPECTED_ORIGINAL_SOURCE_COMMIT = "33122e15d0aaf3661893a4c853fd5ac49173c685"
EXPECTED_ORIGINAL_PROTOCOL = (
    "2c0231b61197fe67790432c78a896272a55c3497a777d490598b53a6be67342f"
)
EXPECTED_SNAPSHOT_INVENTORY_SHA256 = (
    "9bff89010f792d1aed8b3b691567655daab8f83135d6421798b5efea29a2f2c5"
)
EXPECTED_AUTHORIZATION_RAW_SHA256 = (
    "371ae8df4add6338b98469eca6a287902cb69325dfda9d5be6ce5b1600e6fd55"
)
EXPECTED_RECEIPT_RAW_SHA256 = (
    "58d1fd0f004efae049afd51e9592a79e963ba3fc8c2d3aae8a4af0bb7791a6a7"
)
EXPECTED_ORIGINAL_REPORT_CALLING_SHA256 = (
    "e0fd250dcd21fc7a0a62b5da0fe2c3d95401a24e6ad92c4e806082867e623047"
)
EXPECTED_ORIGINAL_REPORT_SUBMITTED_SHA256 = (
    "923f49755df3fcab99a547e0347b158ed42daef20cd640e24c605848b0769e57"
)
EXPECTED_ORIGINAL_REPORT_JOB_ID = "33311218"
EXPECTED_ORIGINAL_REPORT_JOB_NAME = (
    f"exp23-launch8-{EXPECTED_SUBMISSION_SHA256[:16]}-report"
)
EXPECTED_ORIGINAL_SCHEDULER_COMMENT = f"treewm-exp23:{EXPECTED_SUBMISSION_SHA256}"
EXPECTED_ORIGINAL_REPORT_LOG_SHA256 = (
    "2c5a23103e00fc07196886c62e7c9d069ed1b011fb9f44095a4242cc926e43a6"
)
EXPECTED_ORIGINAL_REPORT_LOG_SIZE = 384
EXPECTED_BUNDLE_SHA256 = (
    "b9102090021c103fa2362663d1a51310d239d50223108dba0106758b199d9b83"
)
EXPECTED_GATE_SHA256 = (
    "d41b37f6806c77f15557ecd0329596da8385c02db5b06cecfb29247bb5f4682a"
)
EXPECTED_BUNDLE_FILE_SHA256 = (
    "1a72e7968c5bc1639845eb18a64584db2204310c70c6301cdcccf804f576f139"
)
EXPECTED_DECISION_FILE_SHA256 = (
    "53a7af1c91e4b09b8a04fdab7c1c0192d2076a88eb495855d9eafe39601f64b6"
)
EXPECTED_PROVENANCE_V1_FILE_SHA256 = (
    "3e99d102d6f5faa92699fb9bed4e1607e00a08349f03107048153c8d0764e858"
)
EXPECTED_PROVENANCE_V1_SHA256 = (
    "3fca5a3893cfd2e948f922438ee57bcc03e7763cfdb615500429700153820f77"
)
EXPECTED_WORKER_MARKER_AGGREGATE_SHA256 = (
    "ab1ced2e9b736edede8e1353297682feb800865f03da0c25b681208ce7d8cfc8"
)
EXPECTED_BUNDLE_FILE_SIZE = 424_013_704
EXPECTED_DECISION_FILE_SIZE = 704_147
EXPECTED_PROVENANCE_V1_FILE_SIZE = 236_577
EXPECTED_REPORT_STATUS = "rejected"
SOURCE_NAMES = (
    "report.py",
    "report_repair.py",
    "report_repair.slurm",
    "protocol.sha256",
)
SOURCE_AUTHORITY_NAME = "SOURCE_AUTHORITY.json"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_RE = re.compile(r"[0-9a-f]{40}\Z")
JOB_ID_RE = re.compile(r"[1-9][0-9]*\Z")
SACCT_FIELDS = (
    "JobIDRaw",
    "JobName",
    "State",
    "ExitCode",
    "ElapsedRaw",
    "AllocNodes",
    "NodeList",
    "Submit",
    "Eligible",
    "Start",
    "End",
    "Comment",
)
REPAIR_SACCT_FIELDS = (
    "JobIDRaw",
    "JobName",
    "User",
    "State",
    "ExitCode",
    "ElapsedRaw",
    "AllocNodes",
    "NodeList",
    "Submit",
    "Eligible",
    "Start",
    "End",
    "Comment",
    "Reason",
)
SQUEUE_FORMAT = "%A|%j|%u|%T|%k|%r"
REPAIR_WALLTIME_SECONDS = 14_400
REPAIR_RELEASE_EVIDENCE_WAIT_SECONDS = 10_800
REPAIR_ASSEMBLY_BUDGET_SECONDS = 3_600
REPAIR_RELEASE_POLL_SECONDS = 0.25
REPAIR_WORKER_HANDOFF = {
    "schema_version": 1,
    "slurm_walltime_seconds": REPAIR_WALLTIME_SECONDS,
    "release_evidence_wait_seconds": REPAIR_RELEASE_EVIDENCE_WAIT_SECONDS,
    "minimum_assembly_budget_seconds": REPAIR_ASSEMBLY_BUDGET_SECONDS,
    "clock": "time.monotonic",
    "poll_interval_seconds": REPAIR_RELEASE_POLL_SECONDS,
}

FAILURE_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "attempt",
        "original_report_job_id",
        "original_report_job_name",
        "scheduler_comment",
        "original_report_calling_sha256",
        "original_report_submitted_sha256",
        "submission_authorization_sha256",
        "submission_receipt_sha256",
        "snapshot_root",
        "snapshot_inventory_sha256",
        "original_source_commit",
        "original_package_protocol_sha256",
        "report_log",
        "terminal_scheduler_observation",
        "pre_submit_active_census",
        "worker_receipt_map",
        "worker_receipt_map_sha256",
        "expected_reassembly",
        "publication_state",
        "observed_at_utc",
    }
)
SUBMIT_CALLING_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "attempt",
        "original_failure_evidence",
        "original_failure_evidence_sha256",
        "repair_source_root",
        "repair_source_commit",
        "repair_package_protocol_sha256",
        "repair_source_files",
        "repair_source_files_sha256",
        "scheduler_pre_submit_census",
        "scheduler_pre_submit_census_sha256",
        "command",
        "scheduler_environment",
        "transaction_lock",
        "report_cancel_lock",
        "called_at_utc",
    }
)
SUBMITTED_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "attempt",
        "submit_calling_sha256",
        "repair_report_job_id",
        "submission_evidence",
        "accepted_at_utc",
    }
)
SUBMIT_FAILURE_TERMINAL_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "attempt",
        "submit_calling_sha256",
        "scheduler_evidence",
        "post_failure_census",
        "publication_allowed",
        "retry_allowed",
        "sealed_at_utc",
    }
)
RELEASED_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "attempt",
        "repair_report_job_id",
        "authorization_sha256",
        "release_attempts",
        "release_attempts_sha256",
        "post_release_census",
        "post_release_census_sha256",
        "worker_liveness_observation",
        "worker_liveness_observation_sha256",
        "released_at_utc",
    }
)
RELEASE_DENIED_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "attempt",
        "repair_report_job_id",
        "authorization_sha256",
        "reason",
        "pre_release_census",
        "pre_release_census_sha256",
        "publication_allowed",
        "retry_allowed",
        "sealed_at_utc",
    }
)
WORKER_FAILURE_TERMINAL_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "attempt",
        "repair_report_job_id",
        "authorization_sha256",
        "reason",
        "release_attempts",
        "release_attempts_sha256",
        "released_evidence",
        "released_evidence_sha256",
        "post_release_census",
        "post_release_census_sha256",
        "terminal_scheduler_observation",
        "terminal_scheduler_observation_sha256",
        "publication_allowed",
        "retry_allowed",
        "sealed_at_utc",
    }
)
HISTORICAL_JOB_IDS = frozenset(
    {
        "33285485",
        "33285486",
        "33295657",
        "33295659",
        "33295661",
        "33311213",
        "33311216",
        "33311218",
    }
)
RELEASE_CALLING_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "attempt",
        "release_attempt",
        "repair_report_job_id",
        "authorization_sha256",
        "command",
        "scheduler_environment",
        "transaction_lock",
        "report_cancel_lock",
        "called_at_utc",
    }
)
RELEASE_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "attempt",
        "release_attempt",
        "repair_report_job_id",
        "authorization_sha256",
        "release_calling_sha256",
        "mode",
        "scheduler_evidence",
        "observed_at_utc",
    }
)
CANCEL_AUTHORIZATION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "attempt",
        "cancel_generation",
        "reason",
        "job_ids",
        "pre_cancel_census",
        "pre_cancel_census_sha256",
        "transaction_lock",
        "report_cancel_lock",
        "authorized_at_utc",
    }
)
CANCEL_CALLING_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "attempt",
        "cancel_generation",
        "cancel_attempt",
        "authorization_sha256",
        "job_ids",
        "command",
        "scheduler_environment",
        "transaction_lock",
        "report_cancel_lock",
        "called_at_utc",
    }
)
CANCEL_RESULT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "attempt",
        "cancel_generation",
        "cancel_attempt",
        "authorization_sha256",
        "calling_sha256",
        "job_ids",
        "mode",
        "scheduler_evidence",
        "observed_at_utc",
    }
)
CANCEL_TERMINAL_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "attempt",
        "cancel_generation",
        "reason",
        "authorization_sha256",
        "cancel_attempts",
        "cancel_attempts_sha256",
        "post_cancel_census",
        "post_cancel_census_sha256",
        "remaining_job_ids",
        "publication_allowed",
        "retry_allowed",
        "sealed_at_utc",
    }
)
AUTHORIZATION_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "attempt",
        "original_report_job_id",
        "repair_report_job_id",
        "repair_job_name",
        "scheduler_comment",
        "snapshot_root",
        "snapshot_inventory_sha256",
        "original_package_protocol_sha256",
        "original_failure_evidence",
        "original_failure_evidence_sha256",
        "worker_receipt_map",
        "worker_receipt_map_sha256",
        "repair_source_root",
        "repair_source_commit",
        "repair_package_protocol_sha256",
        "repair_source_files",
        "repair_source_files_sha256",
        "submit_calling_sha256",
        "submitted_evidence",
        "submitted_evidence_sha256",
        "scheduler_authority_census",
        "scheduler_authority_census_sha256",
        "worker_handoff",
        "expected_reassembly",
        "publication_allowed",
        "deterministic_reassembly_allowed",
        "scientific_input_change_allowed",
        "gate_change_allowed",
        "scheduler_submission_allowed",
        "authorized_at_utc",
    }
)


class RepairError(RuntimeError):
    """The one-generation report-repair state is unsafe or ambiguous."""


class CommandResult(NamedTuple):
    returncode: int
    stdout: bytes
    stderr: bytes


Runner = Callable[[Sequence[str], Path, Mapping[str, str]], CommandResult]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RepairError(message)


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("ascii")).hexdigest()


def exact_json_equal(left: object, right: object) -> bool:
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


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _file_identity(info: os.stat_result) -> tuple[int, int, int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_uid,
        info.st_nlink,
        stat.S_IMODE(info.st_mode),
        info.st_size,
    )


def _directory(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RepairError(f"{label} is unavailable: {exc}") from exc
    require(stat.S_ISDIR(info.st_mode), f"{label} is not a nonsymlink directory")
    return path.absolute()


def _canonical_existing_directory(path: Path, label: str) -> Path:
    """Require one lexical, physical, nonsymlink spelling of a directory."""

    raw = os.fspath(path)
    require(
        path.is_absolute()
        and not raw.startswith("//")
        and all(part not in {"", ".", ".."} for part in raw.split("/")[1:]),
        f"{label} is not a canonical absolute path",
    )
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise RepairError(f"{label} is unavailable: {exc}") from exc
    require(
        resolved == path and path.is_dir(),
        f"{label} is symlinked or noncanonical",
    )
    return path


def _canonical_cli_path(raw: str, label: str) -> Path:
    """Reject noncanonical CLI spellings before ``Path`` normalizes them."""

    require(type(raw) is str, f"{label} is not a path string")
    parts = raw.split("/")
    require(
        raw.startswith("/")
        and not raw.startswith("//")
        and len(parts) > 1
        and all(part not in {"", ".", ".."} for part in parts[1:]),
        f"{label} is not a canonical absolute path",
    )
    return Path(raw)


def _regular_bytes(path: Path, label: str, *, max_size: int = 1 << 30) -> tuple[bytes, str, os.stat_result]:
    try:
        listed = path.lstat()
        require(stat.S_ISREG(listed.st_mode), f"{label} is not a regular nonsymlink file")
        require(0 <= listed.st_size <= max_size, f"{label} size differs")
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
    except OSError as exc:
        raise RepairError(f"{label} cannot be opened: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        require(_file_identity(opened) == _file_identity(listed), f"{label} identity raced")
        payload = bytearray()
        while len(payload) <= max_size:
            chunk = os.read(descriptor, min(1024 * 1024, max_size + 1 - len(payload)))
            if not chunk:
                break
            payload.extend(chunk)
        require(len(payload) == listed.st_size, f"{label} read size differs")
        after = os.fstat(descriptor)
        require(_file_identity(after) == _file_identity(opened), f"{label} changed while reading")
    finally:
        os.close(descriptor)
    value = bytes(payload)
    return value, hashlib.sha256(value).hexdigest(), listed


def _pairs(path: Path):
    def hook(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    return hook


def _decode_json(path: Path, payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs(path),
            parse_constant=lambda token: (_ for _ in ()).throw(
                RepairError(f"non-finite JSON value in {path}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairError(f"cannot decode {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def read_json(path: Path, label: str) -> tuple[dict[str, Any], str, os.stat_result]:
    payload, digest, info = _regular_bytes(path, label)
    return _decode_json(path, payload), digest, info


def _fsync_directory(path: Path) -> None:
    root = _directory(path, f"fsync directory {path}")
    descriptor = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically install a sealed repair directory without replacing a peer."""

    require(source.parent == target.parent, "repair snapshot rename parents differ")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2 is unavailable for repair snapshot")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), str(target))
    raise OSError(error, os.strerror(error), f"{source} -> {target}")


def seal_json(path: Path, value: Mapping[str, Any]) -> str:
    """Append one immutable JSON object, or exact-validate an existing copy."""

    payload = (json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    try:
        descriptor = os.open(
            path,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
    except FileExistsError:
        existing, existing_digest, info = _regular_bytes(path, f"immutable repair artifact {path.name}")
        require(
            stat.S_IMODE(info.st_mode) == 0o444
            and info.st_uid == os.getuid()
            and info.st_nlink == 1
            and existing_digest == digest
            and existing == payload,
            f"immutable repair artifact differs: {path}",
        )
        return digest
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, f"short repair artifact write: {path}")
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        readback = bytearray()
        while len(readback) < len(payload):
            chunk = os.read(descriptor, len(payload) - len(readback))
            require(bool(chunk), f"short repair artifact readback: {path}")
            readback.extend(chunk)
        opened = os.fstat(descriptor)
        named = path.lstat()
        require(
            os.read(descriptor, 1) == b""
            and bytes(readback) == payload
            and hashlib.sha256(readback).hexdigest() == digest
            and _file_identity(opened) == _file_identity(named)
            and opened.st_uid == os.getuid()
            and opened.st_nlink == 1
            and stat.S_IMODE(opened.st_mode) == 0o444,
            f"sealed repair artifact identity/readback differs: {path}",
        )
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return digest


def _transaction_lock_path(submission_root: Path) -> Path:
    absolute = submission_root.absolute()
    require(
        absolute.name == "submission" and absolute.parent.name == "state",
        "report repair requires the sealed <run_root>/state/submission layout",
    )
    token = hashlib.sha256(str(absolute).encode("utf-8")).hexdigest()[:16]
    return absolute.parents[2] / f".exp23-{token}.transaction.lock"


class _ExistingLock:
    def __init__(self, path: Path, label: str) -> None:
        self.path = path
        self.label = label
        self.descriptor: int | None = None

    def __enter__(self) -> "_ExistingLock":
        _directory(self.path.parent, f"{self.label} parent")
        try:
            descriptor = os.open(
                self.path,
                os.O_RDWR
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
        except OSError as exc:
            raise RepairError(f"cannot open existing {self.label}: {exc}") from exc
        try:
            opened = os.fstat(descriptor)
            named = self.path.lstat()
            require(
                stat.S_ISREG(opened.st_mode)
                and _file_identity(opened) == _file_identity(named)
                and opened.st_uid == os.getuid()
                and opened.st_nlink == 1
                and stat.S_IMODE(opened.st_mode) == 0o600,
                f"{self.label} identity/mode differs",
            )
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            named_after_lock = self.path.lstat()
            opened_after_lock = os.fstat(descriptor)
            require(
                stat.S_ISREG(opened_after_lock.st_mode)
                and _file_identity(opened_after_lock)
                == _file_identity(named_after_lock)
                and opened_after_lock.st_uid == os.getuid()
                and opened_after_lock.st_nlink == 1
                and stat.S_IMODE(opened_after_lock.st_mode) == 0o600,
                f"{self.label} binding changed while waiting for flock",
            )
        except BaseException:
            os.close(descriptor)
            raise
        self.descriptor = descriptor
        return self

    def binding(self) -> dict[str, Any]:
        require(self.descriptor is not None, f"{self.label} is not held")
        opened = os.fstat(self.descriptor)
        named = self.path.lstat()
        require(
            _file_identity(opened) == _file_identity(named)
            and stat.S_IMODE(opened.st_mode) == 0o600,
            f"{self.label} binding changed",
        )
        return {
            "path": str(self.path),
            "device": opened.st_dev,
            "inode": opened.st_ino,
            "uid": opened.st_uid,
            "mode": stat.S_IMODE(opened.st_mode),
        }

    def __exit__(self, _kind: object, _value: object, _traceback: object) -> None:
        assert self.descriptor is not None
        try:
            fcntl.flock(self.descriptor, fcntl.LOCK_UN)
        finally:
            os.close(self.descriptor)
            self.descriptor = None


class _RepairLocks:
    """Hold transaction then shared report/cancel lock in the production order."""

    def __init__(self, submission_root: Path) -> None:
        self.transaction = _ExistingLock(
            _transaction_lock_path(submission_root), "production transaction lock"
        )
        self.report_cancel = _ExistingLock(
            submission_root / ".REPORT_CANCEL.lock", "report/cancel lock"
        )

    def __enter__(self) -> "_RepairLocks":
        self.transaction.__enter__()
        try:
            self.report_cancel.__enter__()
        except BaseException:
            self.transaction.__exit__(None, None, None)
            raise
        return self

    def bindings(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self.transaction.binding(), self.report_cancel.binding()

    def __exit__(self, kind: object, value: object, traceback: object) -> None:
        try:
            self.report_cancel.__exit__(kind, value, traceback)
        finally:
            self.transaction.__exit__(kind, value, traceback)


def _default_runner(
    argv: Sequence[str], cwd: Path, environment: Mapping[str, str]
) -> CommandResult:
    completed = subprocess.run(
        list(argv),
        cwd=cwd,
        env=dict(environment),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _scheduler_environment(slurm_conf: str) -> dict[str, str]:
    username = pwd.getpwuid(os.getuid()).pw_name
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "USER": username,
        "LOGNAME": username,
        "SLURM_CONF": slurm_conf,
    }


def _command_evidence(
    argv: Sequence[str], environment: Mapping[str, str], result: CommandResult
) -> dict[str, Any]:
    return {
        "argv": list(argv),
        "environment": dict(environment),
        "returncode": result.returncode,
        "stdout": {
            "encoding": "base64",
            "size": len(result.stdout),
            "sha256": hashlib.sha256(result.stdout).hexdigest(),
            "data": base64.b64encode(result.stdout).decode("ascii"),
        },
        "stderr": {
            "encoding": "base64",
            "size": len(result.stderr),
            "sha256": hashlib.sha256(result.stderr).hexdigest(),
            "data": base64.b64encode(result.stderr).decode("ascii"),
        },
    }


def _decoded_command_stream(value: object, label: str) -> bytes:
    require(
        isinstance(value, Mapping)
        and set(value) == {"encoding", "size", "sha256", "data"}
        and value.get("encoding") == "base64"
        and type(value.get("size")) is int
        and 0 <= value["size"] <= 16 * 1024 * 1024
        and SHA256_RE.fullmatch(str(value.get("sha256", ""))) is not None
        and isinstance(value.get("data"), str),
        f"{label} stream envelope differs",
    )
    try:
        payload = base64.b64decode(value["data"].encode("ascii"), validate=True)
    except (UnicodeEncodeError, ValueError) as exc:
        raise RepairError(f"{label} stream base64 differs: {exc}") from exc
    require(
        len(payload) == value["size"]
        and hashlib.sha256(payload).hexdigest() == value["sha256"],
        f"{label} stream bytes differ",
    )
    return payload


def _validated_command_evidence(
    value: object,
    *,
    label: str,
    expected_argv: Sequence[str] | None = None,
    expected_environment: Mapping[str, str] | None = None,
) -> CommandResult:
    require(
        isinstance(value, Mapping)
        and set(value)
        == {"argv", "environment", "returncode", "stdout", "stderr"}
        and isinstance(value.get("argv"), list)
        and all(isinstance(item, str) for item in value["argv"])
        and isinstance(value.get("environment"), Mapping)
        and all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in value["environment"].items()
        )
        and type(value.get("returncode")) is int,
        f"{label} command evidence differs",
    )
    if expected_argv is not None:
        require(value["argv"] == list(expected_argv), f"{label} argv differs")
    if expected_environment is not None:
        require(
            exact_json_equal(value["environment"], expected_environment),
            f"{label} environment differs",
        )
    stdout = _decoded_command_stream(value["stdout"], f"{label} stdout")
    stderr = _decoded_command_stream(value["stderr"], f"{label} stderr")
    return CommandResult(value["returncode"], stdout, stderr)


def _run(
    runner: Runner,
    argv: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    locks: _RepairLocks,
) -> tuple[CommandResult, dict[str, Any]]:
    before = locks.bindings()
    result = runner(tuple(argv), cwd, environment)
    require(locks.bindings() == before, "repair scheduler-call lock lease changed")
    require(
        type(result.returncode) is int
        and isinstance(result.stdout, bytes)
        and isinstance(result.stderr, bytes),
        "repair scheduler result types differ",
    )
    return result, _command_evidence(argv, environment, result)


def _load_module(name: str, path: Path) -> ModuleType:
    payload, digest, info = _regular_bytes(path, f"module {name}")
    require(
        stat.S_IMODE(info.st_mode) in {0o444, 0o644}
        and bool(payload)
        and SHA256_RE.fullmatch(digest) is not None,
        f"module {name} identity differs",
    )
    unique = f"_exp23_report_repair_{name}_{os.getpid()}_{time.time_ns()}"
    module = ModuleType(unique)
    module.__file__ = str(path)
    module.__package__ = ""
    sys.modules[unique] = module
    try:
        code = compile(payload, str(path), "exec", dont_inherit=True)
        exec(code, module.__dict__)
    except BaseException:
        sys.modules.pop(unique, None)
        raise
    return module


def _expected_reassembly() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": EXPECTED_REPORT_STATUS,
        "report_bundle_sha256": EXPECTED_BUNDLE_SHA256,
        "gate_sha256": EXPECTED_GATE_SHA256,
        "report_bundle_file_sha256": EXPECTED_BUNDLE_FILE_SHA256,
        "gate_decision_file_sha256": EXPECTED_DECISION_FILE_SHA256,
        "original_provenance_v1_file_sha256": EXPECTED_PROVENANCE_V1_FILE_SHA256,
        "original_provenance_v1_sha256": EXPECTED_PROVENANCE_V1_SHA256,
        "report_bundle_file_size": EXPECTED_BUNDLE_FILE_SIZE,
        "gate_decision_file_size": EXPECTED_DECISION_FILE_SIZE,
        "original_provenance_v1_file_size": EXPECTED_PROVENANCE_V1_FILE_SIZE,
        "worker_marker_aggregate_sha256": EXPECTED_WORKER_MARKER_AGGREGATE_SHA256,
        "deterministic_reassembly_allowed": True,
        "scientific_input_change_allowed": False,
        "gate_change_allowed": False,
    }


def _journal_path(submission_root: Path, name: str) -> Path:
    return submission_root / "journal" / name


def _repair_root(submission_root: Path) -> Path:
    return submission_root / "report-repair" / "attempt-0001"


def _repair_source_root(submission_root: Path) -> Path:
    return _repair_root(submission_root) / "source"


def _repair_name(submission_sha256: str) -> str:
    return f"exp23-launch8-{submission_sha256[:16]}-report-repair-0001"


def _repair_comment(submission_sha256: str) -> str:
    return f"treewm-exp23-report-repair:{submission_sha256}:0001"


def _json_file_row(path: Path, label: str, *, expected_mode: int = 0o444) -> dict[str, Any]:
    _value, digest, info = read_json(path, label)
    require(
        stat.S_IMODE(info.st_mode) == expected_mode
        and info.st_uid == os.getuid()
        and info.st_nlink == 1,
        f"{label} identity/mode differs",
    )
    return {"mode": expected_mode, "size": info.st_size, "sha256": digest}


def _worker_receipt_map(submission_root: Path) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    aggregate = hashlib.sha256()
    for index in range(20):
        relative = Path("tasks") / f"cell-{index:02d}" / "WORKER_COMPLETE.json"
        path = submission_root / relative
        payload, digest, info = _regular_bytes(path, f"cell{index} worker receipt")
        value = _decode_json(path, payload)
        require(
            stat.S_IMODE(info.st_mode) == 0o444
            and info.st_uid == os.getuid()
            and info.st_nlink == 1
            and value.get("campaign_id") == CAMPAIGN_ID
            and value.get("submission_sha256") == EXPECTED_SUBMISSION_SHA256
            and value.get("cell_index") == index
            and value.get("status") == "worker_complete",
            f"cell{index} worker receipt differs",
        )
        rows[relative.as_posix()] = {
            "mode": 0o444,
            "size": info.st_size,
            "sha256": digest,
        }
        encoded_path = relative.as_posix().encode("utf-8")
        aggregate.update(len(encoded_path).to_bytes(8, "big"))
        aggregate.update(encoded_path)
        aggregate.update(len(payload).to_bytes(8, "big"))
        aggregate.update(payload)
    require(
        aggregate.hexdigest() == EXPECTED_WORKER_MARKER_AGGREGATE_SHA256,
        "worker receipt aggregate differs",
    )
    return {"schema_version": 1, "files": rows}


def _pretty_json_sha(value: Mapping[str, Any]) -> str:
    payload = (json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _pretty_json_size(value: Mapping[str, Any]) -> int:
    return len(
        (json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
            "utf-8"
        )
    )


def _validate_original_submission(
    submission_root: Path,
    submission_sha256: str,
    *,
    report_program: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate original snapshot/receipt/all cells and exact deterministic assembly."""

    require(
        submission_root == CANONICAL_PRODUCTION_SUBMISSION_ROOT,
        "report repair submission root differs",
    )
    require(submission_sha256 == EXPECTED_SUBMISSION_SHA256, "report repair submission SHA differs")
    _directory(submission_root, "original submission root")
    contract, contract_digest, contract_info = read_json(
        submission_root / "SUBMISSION_CONTRACT.json", "original submission contract"
    )
    require(
        contract_digest == submission_sha256
        and stat.S_IMODE(contract_info.st_mode) == 0o444
        and contract_info.st_uid == os.getuid()
        and contract_info.st_nlink == 1
        and contract.get("campaign_id") == CAMPAIGN_ID
        and contract.get("submission_root") == str(submission_root)
        and contract.get("snapshot_inventory_sha256")
        == EXPECTED_SNAPSHOT_INVENTORY_SHA256
        and contract.get("package_protocol_sha256") == EXPECTED_ORIGINAL_PROTOCOL,
        "original submission contract differs",
    )
    git_provenance = contract.get("git_provenance")
    require(
        isinstance(git_provenance, Mapping)
        and git_provenance.get("commit") == EXPECTED_ORIGINAL_SOURCE_COMMIT,
        "original submission source commit differs",
    )
    snapshot_root = Path(str(contract.get("snapshot_root", "")))
    require(
        snapshot_root == submission_root / "source-snapshot" / "repo",
        "original snapshot root differs",
    )
    _directory(snapshot_root, "original source snapshot")
    receipt, receipt_digest, receipt_info = read_json(
        submission_root / "SUBMISSION_RECEIPT.json", "original submission receipt"
    )
    require(
        receipt_digest == EXPECTED_RECEIPT_RAW_SHA256
        and stat.S_IMODE(receipt_info.st_mode) == 0o444
        and receipt_info.st_uid == os.getuid()
        and receipt_info.st_nlink == 1
        and receipt.get("submission_sha256") == submission_sha256
        and receipt.get("report_job_id") == EXPECTED_ORIGINAL_REPORT_JOB_ID,
        "original submission receipt differs",
    )
    authorization_row = _json_file_row(
        submission_root / "SUBMISSION_AUTHORIZATION.json",
        "original submission authorization",
    )
    require(
        authorization_row["sha256"] == EXPECTED_AUTHORIZATION_RAW_SHA256,
        "original submission authorization hash differs",
    )
    receipt_map = _worker_receipt_map(submission_root)
    repaired_report = _load_module("sealed_repaired_report", report_program)
    bundle, decision, provenance = repaired_report.assemble_report(
        snapshot_root,
        submission_root,
        submission_sha256,
        allow_repair_cleanup_for_audit=True,
    )
    require(
        repaired_report.stable_hash(bundle) == EXPECTED_BUNDLE_SHA256
        and decision.get("status") == EXPECTED_REPORT_STATUS
        and decision.get("gate_sha256") == EXPECTED_GATE_SHA256
        and _pretty_json_sha(bundle) == EXPECTED_BUNDLE_FILE_SHA256
        and _pretty_json_sha(decision) == EXPECTED_DECISION_FILE_SHA256
        and repaired_report.stable_hash(provenance) == EXPECTED_PROVENANCE_V1_SHA256
        and _pretty_json_sha(provenance) == EXPECTED_PROVENANCE_V1_FILE_SHA256
        and _pretty_json_size(bundle) == EXPECTED_BUNDLE_FILE_SIZE
        and _pretty_json_size(decision) == EXPECTED_DECISION_FILE_SIZE
        and _pretty_json_size(provenance) == EXPECTED_PROVENANCE_V1_FILE_SIZE,
        "original deterministic report reassembly differs",
    )
    return contract, receipt, receipt_map, _expected_reassembly()


def _publication_state(submission_root: Path) -> dict[str, Any]:
    journal = submission_root / "journal"
    cleanup = [
        "CANCEL_REQUESTED.json",
        "journal/PREREQUISITE_MISSING.json",
        "journal/9000_RECOVERY_CANCELLED.json",
        "journal/9001_PRODUCTION_PREREQUISITE_MISSING.json",
    ]
    present_cleanup = [name for name in cleanup if os.path.lexists(submission_root / name)]
    staging = sorted(path.name for path in submission_root.glob(".report.tmp.*"))
    require(
        not os.path.lexists(submission_root / "report")
        and not present_cleanup
        and not staging,
        "report publication/cancellation/staging state is not fresh for repair",
    )
    return {
        "report_absent": True,
        "staging_entries": [],
        "cleanup_prefixes": [],
        "journal_directory": str(journal),
    }


def _original_report_timeline_is_ordered(parsed: Mapping[str, Any]) -> bool:
    return (
        isinstance(parsed.get("Submit"), str)
        and bool(parsed["Submit"])
        and isinstance(parsed.get("Eligible"), str)
        and bool(parsed["Eligible"])
        and isinstance(parsed.get("Start"), str)
        and bool(parsed["Start"])
        and isinstance(parsed.get("End"), str)
        and bool(parsed["End"])
        and parsed["Submit"]
        <= parsed["Eligible"]
        <= parsed["Start"]
        < parsed["End"]
    )


def _terminal_scheduler_observation(
    submission_root: Path,
    contract: Mapping[str, Any],
    runner: Runner,
    locks: _RepairLocks,
) -> dict[str, Any]:
    control_plane = contract.get("scheduler_control_plane_contract")
    require(isinstance(control_plane, Mapping), "original scheduler control-plane contract differs")
    slurm_conf = str(control_plane.get("slurm_conf", ""))
    require(Path(slurm_conf).is_absolute(), "original Slurm configuration path differs")
    environment = _scheduler_environment(slurm_conf)
    argv = [
        "/usr/local/bin/sacct",
        "-X",
        "-n",
        "-j",
        EXPECTED_ORIGINAL_REPORT_JOB_ID,
        "-o",
        ",".join(SACCT_FIELDS),
        "-P",
    ]
    result, raw = _run(runner, argv, submission_root, environment, locks)
    require(result.returncode == 0 and result.stderr == b"", "original report sacct query failed")
    try:
        stdout = result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RepairError(f"original report sacct stdout is not UTF-8: {exc}") from exc
    lines = [line for line in stdout.splitlines() if line]
    require(len(lines) == 1, "original report sacct row count differs")
    row = lines[0].split("|")
    require(len(row) == len(SACCT_FIELDS), "original report sacct field count differs")
    parsed = dict(zip(SACCT_FIELDS, row, strict=True))
    require(
        parsed["JobIDRaw"] == EXPECTED_ORIGINAL_REPORT_JOB_ID
        and parsed["JobName"] == EXPECTED_ORIGINAL_REPORT_JOB_NAME
        and parsed["State"] == "FAILED"
        and parsed["ExitCode"] == "2:0"
        and parsed["ElapsedRaw"] == "355"
        and parsed["AllocNodes"] == "1"
        and parsed["NodeList"] == "cpu-00090"
        and parsed["Start"] == "2026-08-29T08:28:49"
        and parsed["End"] == "2026-08-29T08:34:44"
        and _original_report_timeline_is_ordered(parsed)
        and parsed["Comment"] == EXPECTED_ORIGINAL_SCHEDULER_COMMENT,
        "original report terminal scheduler row differs",
    )
    canonical = {
        "schema_version": 1,
        "fields": list(SACCT_FIELDS),
        "rows": [row],
    }
    return {
        "schema_version": 1,
        "captured_at_utc": _utc_now(),
        "scheduler_control_plane": dict(control_plane),
        "raw": raw,
        "canonical": canonical,
        "canonical_sha256": stable_hash(canonical),
        "parsed_row": parsed,
    }


def _parse_squeue_rows(result: CommandResult) -> list[dict[str, str]]:
    require(result.returncode == 0 and result.stderr == b"", "repair squeue query failed")
    try:
        stdout = result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RepairError(f"repair squeue stdout is not UTF-8: {exc}") from exc
    rows: list[dict[str, str]] = []
    for raw_line in stdout.splitlines():
        if not raw_line:
            continue
        fields = raw_line.split("|", 5)
        require(len(fields) == 6, "repair squeue field count differs")
        job_id, name, owner, state, comment, reason = fields
        require(JOB_ID_RE.fullmatch(job_id) is not None, "repair squeue job ID differs")
        rows.append(
            {
                "job_id": job_id,
                "job_name": name,
                "owner": owner,
                "state": state,
                "comment": comment,
                "reason": reason,
            }
        )
    return rows


def _scheduler_census(
    submission_root: Path,
    contract: Mapping[str, Any],
    runner: Runner,
    locks: _RepairLocks,
    *,
    rounds: int = 3,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    require(rounds == 3, "repair scheduler census round count differs")
    control_plane = contract.get("scheduler_control_plane_contract")
    require(isinstance(control_plane, Mapping), "repair scheduler control plane differs")
    environment = _scheduler_environment(str(control_plane.get("slurm_conf", "")))
    owner = environment["USER"]
    argv = [
        "/usr/local/bin/squeue",
        "--noheader",
        f"--user={owner}",
        f"--format={SQUEUE_FORMAT}",
    ]
    observations: list[dict[str, Any]] = []
    relevant_rows: list[list[dict[str, str]]] = []
    for index in range(rounds):
        result, raw = _run(runner, argv, submission_root, environment, locks)
        rows = _parse_squeue_rows(result)
        relevant = [
            row
            for row in rows
            if row["job_name"].startswith("exp23-launch8-")
            or row["comment"].startswith("treewm-exp23")
        ]
        require(
            all(row["owner"] == owner for row in relevant),
            "repair scheduler census owner differs",
        )
        observations.append(
            {
                "round": index,
                "raw": raw,
                "relevant_rows": relevant,
            }
        )
        relevant_rows.append(relevant)
        if index + 1 < rounds:
            sleep(0.25)
    require(
        exact_json_equal(relevant_rows[-2], relevant_rows[-1]),
        "repair scheduler census did not settle",
    )
    return {
        "schema_version": 1,
        "rounds": observations,
        "settled_rows": relevant_rows[-1],
        "captured_at_utc": _utc_now(),
    }


def _validated_scheduler_census(census: Mapping[str, Any]) -> dict[str, Any]:
    require(
        set(census) == {"schema_version", "rounds", "settled_rows", "captured_at_utc"}
        and type(census.get("schema_version")) is int
        and census.get("schema_version") == 1
        and isinstance(census.get("rounds"), list)
        and len(census["rounds"]) == 3
        and isinstance(census.get("settled_rows"), list)
        and isinstance(census.get("captured_at_utc"), str)
        and bool(census["captured_at_utc"]),
        "report repair scheduler census envelope differs",
    )
    reconstructed: list[list[dict[str, str]]] = []
    environment: dict[str, str] | None = None
    for index, observation in enumerate(census["rounds"]):
        require(
            isinstance(observation, Mapping)
            and set(observation) == {"round", "raw", "relevant_rows"}
            and type(observation.get("round")) is int
            and observation.get("round") == index
            and isinstance(observation.get("raw"), Mapping)
            and isinstance(observation.get("relevant_rows"), list),
            f"report repair scheduler census round {index} differs",
        )
        raw = observation["raw"]
        raw_environment = raw.get("environment")
        require(
            isinstance(raw_environment, Mapping)
            and isinstance(raw_environment.get("USER"), str)
            and bool(raw_environment["USER"]),
            f"report repair scheduler census environment {index} differs",
        )
        expected_argv = [
            "/usr/local/bin/squeue",
            "--noheader",
            f"--user={raw_environment['USER']}",
            f"--format={SQUEUE_FORMAT}",
        ]
        result = _validated_command_evidence(
            raw,
            label=f"report repair scheduler census round {index}",
            expected_argv=expected_argv,
            expected_environment=raw_environment,
        )
        rows = _parse_squeue_rows(result)
        relevant = [
            row
            for row in rows
            if row["job_name"].startswith("exp23-launch8-")
            or row["comment"].startswith("treewm-exp23")
        ]
        require(
            all(row["owner"] == raw_environment["USER"] for row in relevant)
            and exact_json_equal(observation["relevant_rows"], relevant),
            f"report repair scheduler census parsed/raw rows differ at round {index}",
        )
        if environment is None:
            environment = dict(raw_environment)
        else:
            require(
                exact_json_equal(environment, raw_environment),
                "report repair scheduler census environment changed",
            )
        reconstructed.append(relevant)
    require(
        exact_json_equal(reconstructed[-2], reconstructed[-1])
        and exact_json_equal(census["settled_rows"], reconstructed[-1]),
        "report repair scheduler census did not settle exactly",
    )
    return dict(census)


def _repair_rows(census: Mapping[str, Any], submission_sha256: str) -> list[dict[str, str]]:
    _validated_scheduler_census(census)
    rows = census.get("settled_rows")
    require(isinstance(rows, list), "repair settled scheduler rows differ")
    expected_name = _repair_name(submission_sha256)
    expected_comment = _repair_comment(submission_sha256)
    return [
        dict(row)
        for row in rows
        if isinstance(row, Mapping)
        and row.get("job_name") == expected_name
        and row.get("comment") == expected_comment
    ]


def _repair_accounting_argv(job_id: str) -> list[str]:
    return [
        "/usr/local/bin/sacct",
        "-X",
        "-n",
        "-j",
        job_id,
        "-o",
        ",".join(REPAIR_SACCT_FIELDS),
        "-P",
    ]


def _repair_job_accounting_observation(
    submission_root: Path,
    contract: Mapping[str, Any],
    job_id: str,
    submission_sha256: str,
    runner: Runner,
    locks: _RepairLocks,
) -> dict[str, Any]:
    require(JOB_ID_RE.fullmatch(job_id) is not None, "repair accounting job ID differs")
    control_plane = contract.get("scheduler_control_plane_contract")
    require(isinstance(control_plane, Mapping), "repair accounting control plane differs")
    environment = _scheduler_environment(str(control_plane.get("slurm_conf", "")))
    argv = _repair_accounting_argv(job_id)
    result, raw = _run(runner, argv, submission_root, environment, locks)
    require(
        result.returncode == 0 and result.stderr == b"",
        "repair worker sacct query failed",
    )
    try:
        text = result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RepairError(f"repair worker sacct stdout is not UTF-8: {exc}") from exc
    lines = [line for line in text.splitlines() if line]
    require(len(lines) <= 1, "repair worker sacct row count differs")
    rows: list[list[str]] = []
    parsed_row: dict[str, str] | None = None
    if lines:
        row = lines[0].split("|")
        require(
            len(row) == len(REPAIR_SACCT_FIELDS),
            "repair worker sacct field count differs",
        )
        parsed_row = dict(zip(REPAIR_SACCT_FIELDS, row, strict=True))
        require(
            parsed_row["JobIDRaw"] == job_id
            and parsed_row["JobName"] == _repair_name(submission_sha256)
            and parsed_row["User"] == environment["USER"]
            and parsed_row["Comment"] == _repair_comment(submission_sha256)
            and bool(parsed_row["State"]),
            "repair worker sacct identity differs",
        )
        rows.append(row)
    canonical = {
        "schema_version": 1,
        "fields": list(REPAIR_SACCT_FIELDS),
        "rows": rows,
    }
    return {
        "schema_version": 1,
        "captured_at_utc": _utc_now(),
        "scheduler_control_plane": dict(control_plane),
        "raw": raw,
        "canonical": canonical,
        "canonical_sha256": stable_hash(canonical),
        "parsed_row": parsed_row,
    }


def _validated_repair_job_accounting_observation(
    value: Mapping[str, Any],
    *,
    contract: Mapping[str, Any],
    job_id: str,
    submission_sha256: str,
) -> dict[str, Any]:
    require(
        set(value)
        == {
            "schema_version",
            "captured_at_utc",
            "scheduler_control_plane",
            "raw",
            "canonical",
            "canonical_sha256",
            "parsed_row",
        }
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and isinstance(value.get("captured_at_utc"), str)
        and bool(value["captured_at_utc"])
        and exact_json_equal(
            value.get("scheduler_control_plane"),
            contract.get("scheduler_control_plane_contract"),
        )
        and isinstance(value.get("canonical"), Mapping)
        and value.get("canonical_sha256") == stable_hash(value["canonical"]),
        "repair worker accounting observation differs",
    )
    control_plane = contract.get("scheduler_control_plane_contract")
    assert isinstance(control_plane, Mapping)
    environment = _scheduler_environment(str(control_plane.get("slurm_conf", "")))
    result = _validated_command_evidence(
        value.get("raw"),
        label="repair worker accounting",
        expected_argv=_repair_accounting_argv(job_id),
        expected_environment=environment,
    )
    require(
        result.returncode == 0 and result.stderr == b"",
        "repair worker accounting command differs",
    )
    try:
        text = result.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RepairError(f"repair worker accounting stdout is not UTF-8: {exc}") from exc
    lines = [line for line in text.splitlines() if line]
    require(len(lines) <= 1, "repair worker accounting row count differs")
    rows: list[list[str]] = []
    parsed: dict[str, str] | None = None
    if lines:
        row = lines[0].split("|")
        require(
            len(row) == len(REPAIR_SACCT_FIELDS),
            "repair worker accounting field count differs",
        )
        parsed = dict(zip(REPAIR_SACCT_FIELDS, row, strict=True))
        require(
            parsed["JobIDRaw"] == job_id
            and parsed["JobName"] == _repair_name(submission_sha256)
            and parsed["User"] == environment["USER"]
            and parsed["Comment"] == _repair_comment(submission_sha256)
            and bool(parsed["State"]),
            "repair worker accounting identity differs",
        )
        rows.append(row)
    require(
        exact_json_equal(
            value["canonical"],
            {
                "schema_version": 1,
                "fields": list(REPAIR_SACCT_FIELDS),
                "rows": rows,
            },
        )
        and exact_json_equal(value.get("parsed_row"), parsed),
        "repair worker accounting raw/canonical rows differ",
    )
    return dict(value)


def _base_slurm_state(value: str) -> str:
    return value.split(maxsplit=1)[0].removesuffix("+")


def _repair_accounting_classification(observation: Mapping[str, Any]) -> str:
    row = observation.get("parsed_row")
    if row is None:
        return "unavailable"
    require(isinstance(row, Mapping), "repair worker accounting row differs")
    state = _base_slurm_state(str(row.get("State", "")))
    if state == "PENDING" and row.get("Reason") in {
        "JobHeldUser",
        "JobHeldAdmin",
    }:
        return "held"
    if state in {
        "PENDING",
        "RUNNING",
        "CONFIGURING",
        "COMPLETING",
        "SUSPENDED",
        "RESIZING",
        "STAGE_OUT",
    }:
        return "active"
    if state in {
        "BOOT_FAIL",
        "CANCELLED",
        "COMPLETED",
        "DEADLINE",
        "FAILED",
        "NODE_FAIL",
        "OUT_OF_MEMORY",
        "PREEMPTED",
        "REVOKED",
        "SPECIAL_EXIT",
        "STOPPED",
        "TIMEOUT",
    }:
        start = row.get("Start")
        no_start = start in {"", "Unknown", "None", "N/A"}
        require(
            isinstance(row.get("Submit"), str)
            and bool(row["Submit"])
            and isinstance(row.get("Eligible"), str)
            and bool(row["Eligible"])
            and isinstance(row.get("End"), str)
            and bool(row["End"])
            and isinstance(row.get("ExitCode"), str)
            and bool(row["ExitCode"])
            and row["Submit"] <= row["Eligible"] <= row["End"]
            and (
                (
                    isinstance(start, str)
                    and not no_start
                    and row["Eligible"] <= start <= row["End"]
                )
                or (
                    no_start
                    and state
                    in {
                        "BOOT_FAIL",
                        "CANCELLED",
                        "DEADLINE",
                        "PREEMPTED",
                        "REVOKED",
                        "TIMEOUT",
                    }
                )
            ),
            "repair worker terminal accounting timeline differs",
        )
        return "terminal"
    return "unknown"


def _capture_report_log(submission_root: Path) -> dict[str, Any]:
    relative = Path("logs") / f"report_{EXPECTED_ORIGINAL_REPORT_JOB_ID}.out"
    path = submission_root / relative
    payload, digest, info = _regular_bytes(path, "original failed report log", max_size=1 << 20)
    require(
        stat.S_IMODE(info.st_mode) == 0o600
        and info.st_uid == os.getuid()
        and info.st_nlink == 1
        and info.st_size == EXPECTED_ORIGINAL_REPORT_LOG_SIZE
        and digest == EXPECTED_ORIGINAL_REPORT_LOG_SHA256,
        "original failed report log identity/hash differs",
    )
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RepairError(f"original failed report log is not UTF-8: {exc}") from exc
    require(
        "Exp23 report engineering error: staged report artifact identity differs:" in text
        and f"REPORT_BUNDLE.{EXPECTED_BUNDLE_SHA256}.json" in text,
        "original failed report log content differs",
    )
    return {
        "path": relative.as_posix(),
        "mode": 0o600,
        "size": len(payload),
        "uid": info.st_uid,
        "nlink": info.st_nlink,
        "sha256": digest,
        "encoding": "base64",
        "data": base64.b64encode(payload).decode("ascii"),
    }


def _build_failure_evidence(
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    receipt_map: Mapping[str, Any],
    expected_reassembly: Mapping[str, Any],
    runner: Runner,
    locks: _RepairLocks,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    publication_state = _publication_state(submission_root)
    calling_row = _json_file_row(
        _journal_path(submission_root, "CALLING_REPORT.json"),
        "original report scheduler calling record",
    )
    submitted_row = _json_file_row(
        _journal_path(submission_root, "0005_REPORT_SUBMITTED.json"),
        "original report submitted record",
    )
    require(
        calling_row["sha256"] == EXPECTED_ORIGINAL_REPORT_CALLING_SHA256
        and submitted_row["sha256"] == EXPECTED_ORIGINAL_REPORT_SUBMITTED_SHA256,
        "original report submission journal hashes differ",
    )
    census = _scheduler_census(
        submission_root, contract, runner, locks, sleep=sleep
    )
    require(
        not _repair_rows(census, submission_sha256)
        and not any(
            row["job_id"]
            in {
                str(EXPECTED_ORIGINAL_REPORT_JOB_ID),
                "33311213",
                "33311216",
            }
            for row in census["settled_rows"]
        ),
        "original/repair scheduler identities remain active before repair",
    )
    terminal = _terminal_scheduler_observation(
        submission_root, contract, runner, locks
    )
    receipt_sha = _json_file_row(
        submission_root / "SUBMISSION_RECEIPT.json", "original submission receipt"
    )["sha256"]
    authorization_sha = _json_file_row(
        submission_root / "SUBMISSION_AUTHORIZATION.json",
        "original submission authorization",
    )["sha256"]
    return {
        "schema_version": 1,
        "status": "original_report_terminal_failure_authenticated",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "attempt": ATTEMPT,
        "original_report_job_id": EXPECTED_ORIGINAL_REPORT_JOB_ID,
        "original_report_job_name": EXPECTED_ORIGINAL_REPORT_JOB_NAME,
        "scheduler_comment": EXPECTED_ORIGINAL_SCHEDULER_COMMENT,
        "original_report_calling_sha256": calling_row["sha256"],
        "original_report_submitted_sha256": submitted_row["sha256"],
        "submission_authorization_sha256": authorization_sha,
        "submission_receipt_sha256": receipt_sha,
        "snapshot_root": str(contract["snapshot_root"]),
        "snapshot_inventory_sha256": contract["snapshot_inventory_sha256"],
        "original_source_commit": EXPECTED_ORIGINAL_SOURCE_COMMIT,
        "original_package_protocol_sha256": contract["package_protocol_sha256"],
        "report_log": _capture_report_log(submission_root),
        "terminal_scheduler_observation": terminal,
        "pre_submit_active_census": census,
        "worker_receipt_map": dict(receipt_map),
        "worker_receipt_map_sha256": stable_hash(receipt_map),
        "expected_reassembly": dict(expected_reassembly),
        "publication_state": publication_state,
        "observed_at_utc": _utc_now(),
    }


def _validate_failure_evidence(
    value: Mapping[str, Any],
    *,
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    receipt_map: Mapping[str, Any],
    expected_reassembly: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        set(value) == FAILURE_KEYS
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("status") == "original_report_terminal_failure_authenticated"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and type(value.get("attempt")) is int
        and value.get("attempt") == ATTEMPT
        and value.get("original_report_job_id") == EXPECTED_ORIGINAL_REPORT_JOB_ID
        and value.get("original_report_job_name") == EXPECTED_ORIGINAL_REPORT_JOB_NAME
        and value.get("scheduler_comment") == EXPECTED_ORIGINAL_SCHEDULER_COMMENT
        and value.get("original_report_calling_sha256")
        == EXPECTED_ORIGINAL_REPORT_CALLING_SHA256
        and value.get("original_report_submitted_sha256")
        == EXPECTED_ORIGINAL_REPORT_SUBMITTED_SHA256
        and value.get("submission_authorization_sha256")
        == EXPECTED_AUTHORIZATION_RAW_SHA256
        and value.get("submission_receipt_sha256") == EXPECTED_RECEIPT_RAW_SHA256
        and value.get("snapshot_root") == contract.get("snapshot_root")
        and value.get("snapshot_inventory_sha256")
        == EXPECTED_SNAPSHOT_INVENTORY_SHA256
        and value.get("original_source_commit") == EXPECTED_ORIGINAL_SOURCE_COMMIT
        and value.get("original_package_protocol_sha256")
        == EXPECTED_ORIGINAL_PROTOCOL
        and exact_json_equal(value.get("worker_receipt_map"), receipt_map)
        and value.get("worker_receipt_map_sha256") == stable_hash(receipt_map)
        and exact_json_equal(value.get("expected_reassembly"), expected_reassembly)
        and isinstance(value.get("observed_at_utc"), str)
        and bool(value["observed_at_utc"]),
        "original report failure evidence fields differ",
    )
    log = value.get("report_log")
    require(
        isinstance(log, Mapping)
        and set(log)
        == {"path", "mode", "size", "uid", "nlink", "sha256", "encoding", "data"}
        and log.get("path") == f"logs/report_{EXPECTED_ORIGINAL_REPORT_JOB_ID}.out"
        and type(log.get("mode")) is int
        and log.get("mode") == 0o600
        and type(log.get("size")) is int
        and log.get("size") == EXPECTED_ORIGINAL_REPORT_LOG_SIZE
        and type(log.get("uid")) is int
        and log.get("uid") == os.getuid()
        and type(log.get("nlink")) is int
        and log.get("nlink") == 1
        and log.get("sha256") == EXPECTED_ORIGINAL_REPORT_LOG_SHA256
        and log.get("encoding") == "base64"
        and isinstance(log.get("data"), str),
        "original report failure log evidence differs",
    )
    try:
        log_payload = base64.b64decode(log["data"], validate=True)
    except (ValueError, TypeError) as exc:
        raise RepairError(f"original report failure log base64 differs: {exc}") from exc
    require(
        len(log_payload) == log["size"]
        and hashlib.sha256(log_payload).hexdigest() == log["sha256"],
        "original report failure log payload differs",
    )
    require(
        exact_json_equal(log, _capture_report_log(submission_root)),
        "original report failure log no longer matches durable evidence",
    )
    require(
        _json_file_row(
            _journal_path(submission_root, "CALLING_REPORT.json"),
            "original report scheduler calling record",
        )["sha256"]
        == EXPECTED_ORIGINAL_REPORT_CALLING_SHA256
        and _json_file_row(
            _journal_path(submission_root, "0005_REPORT_SUBMITTED.json"),
            "original report submitted record",
        )["sha256"]
        == EXPECTED_ORIGINAL_REPORT_SUBMITTED_SHA256,
        "original report submission journals no longer match failure evidence",
    )
    terminal = value.get("terminal_scheduler_observation")
    require(
        isinstance(terminal, Mapping)
        and set(terminal)
        == {
            "schema_version",
            "captured_at_utc",
            "scheduler_control_plane",
            "raw",
            "canonical",
            "canonical_sha256",
            "parsed_row",
        }
        and type(terminal.get("schema_version")) is int
        and terminal.get("schema_version") == 1
        and isinstance(terminal.get("captured_at_utc"), str)
        and isinstance(terminal.get("scheduler_control_plane"), Mapping)
        and exact_json_equal(
            terminal.get("scheduler_control_plane"),
            contract.get("scheduler_control_plane_contract"),
        )
        and isinstance(terminal.get("raw"), Mapping)
        and isinstance(terminal.get("canonical"), Mapping)
        and terminal.get("canonical_sha256") == stable_hash(terminal["canonical"])
        and isinstance(terminal.get("parsed_row"), Mapping),
        "original report terminal scheduler observation differs",
    )
    parsed = terminal["parsed_row"]
    require(
        set(parsed) == set(SACCT_FIELDS)
        and parsed.get("JobIDRaw") == EXPECTED_ORIGINAL_REPORT_JOB_ID
        and parsed.get("JobName") == EXPECTED_ORIGINAL_REPORT_JOB_NAME
        and parsed.get("State") == "FAILED"
        and parsed.get("ExitCode") == "2:0"
        and parsed.get("ElapsedRaw") == "355"
        and parsed.get("AllocNodes") == "1"
        and parsed.get("NodeList") == "cpu-00090"
        and parsed.get("Start") == "2026-08-29T08:28:49"
        and parsed.get("End") == "2026-08-29T08:34:44"
        and _original_report_timeline_is_ordered(parsed)
        and parsed.get("Comment") == EXPECTED_ORIGINAL_SCHEDULER_COMMENT,
        "original report terminal scheduler semantics differ",
    )
    canonical = terminal["canonical"]
    require(
        canonical.get("schema_version") == 1
        and canonical.get("fields") == list(SACCT_FIELDS)
        and canonical.get("rows")
        == [[parsed[field] for field in SACCT_FIELDS]],
        "original report terminal canonical row differs",
    )
    raw = terminal["raw"]
    terminal_argv = [
        "/usr/local/bin/sacct",
        "-X",
        "-n",
        "-j",
        EXPECTED_ORIGINAL_REPORT_JOB_ID,
        "-o",
        ",".join(SACCT_FIELDS),
        "-P",
    ]
    terminal_result = _validated_command_evidence(
        raw,
        label="original report terminal command",
        expected_argv=terminal_argv,
        expected_environment=_scheduler_environment(
            str(contract["scheduler_control_plane_contract"]["slurm_conf"])
        ),
    )
    require(
        terminal_result.returncode == 0
        and terminal_result.stderr == b""
        and terminal_result.stdout
        == ("|".join(parsed[field] for field in SACCT_FIELDS) + "\n").encode(
            "utf-8"
        ),
        "original report terminal raw/canonical row differs",
    )
    publication = value.get("publication_state")
    require(
        exact_json_equal(
            publication,
            {
                "report_absent": True,
                "staging_entries": [],
                "cleanup_prefixes": [],
                "journal_directory": str(submission_root / "journal"),
            },
        ),
        "original report publication state evidence differs",
    )
    pre = value.get("pre_submit_active_census")
    require(isinstance(pre, Mapping), "original report pre-submit census differs")
    _validated_scheduler_census(pre)
    require(
        len(pre["rounds"]) == 3 and pre.get("settled_rows") == [],
        "original report pre-submit active census differs",
    )
    return dict(value)


def _git_output(argv: Sequence[str], repo_root: Path) -> str:
    completed = subprocess.run(
        list(argv),
        cwd=repo_root,
        env={
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
        },
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    require(
        completed.returncode == 0 and completed.stderr == b"",
        f"repair source git command failed: {' '.join(argv)}",
    )
    try:
        return completed.stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RepairError(f"repair source git output is not UTF-8: {exc}") from exc


def _verified_live_repair_source(repo_root: Path) -> dict[str, Any]:
    root = _canonical_existing_directory(repo_root, "repair source repository")
    require(root == REPOSITORY_ROOT, "repair source repository root differs")
    campaign = _load_module("campaign", PACKAGE_DIR / "campaign.py")
    campaign.load_contract(root)
    protocol = campaign.verify_protocol_lock(PACKAGE_DIR)
    require(SHA256_RE.fullmatch(protocol) is not None, "repair package protocol differs")
    commit = _git_output(["/usr/bin/git", "rev-parse", "HEAD"], root).strip()
    origin = _git_output(["/usr/bin/git", "rev-parse", "origin/main"], root).strip()
    status = _git_output(
        ["/usr/bin/git", "status", "--porcelain", "--untracked-files=all"], root
    )
    require(
        GIT_RE.fullmatch(commit) is not None
        and commit == origin
        and status == "",
        "repair source git state is not a clean origin/main commit",
    )
    files: dict[str, Any] = {}
    for name in SOURCE_NAMES:
        path = PACKAGE_DIR / name
        payload, digest, info = _regular_bytes(path, f"live repair source {name}")
        require(
            stat.S_IMODE(info.st_mode) in {0o444, 0o644, 0o755}
            and info.st_uid == os.getuid()
            and info.st_nlink == 1
            and bool(payload),
            f"live repair source identity differs: {name}",
        )
        files[name] = {"mode": 0o444, "size": len(payload), "sha256": digest}
    return {
        "schema_version": 1,
        "repair_source_commit": commit,
        "repair_package_protocol_sha256": protocol,
        "repair_source_files": files,
        "repair_source_files_sha256": stable_hash(files),
    }


def _write_sealed_file(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_RDWR
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
        0o600,
    )
    try:
        view = memoryview(payload)
        while view:
            count = os.write(descriptor, view)
            require(count > 0, f"short repair source write: {path}")
            view = view[count:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        os.lseek(descriptor, 0, os.SEEK_SET)
        readback = bytearray()
        while len(readback) < len(payload):
            chunk = os.read(descriptor, len(payload) - len(readback))
            require(bool(chunk), f"short repair source readback: {path}")
            readback.extend(chunk)
        info = os.fstat(descriptor)
        named = path.lstat()
        require(
            bytes(readback) == payload
            and os.read(descriptor, 1) == b""
            and _file_identity(info) == _file_identity(named)
            and info.st_uid == os.getuid()
            and info.st_nlink == 1
            and stat.S_IMODE(info.st_mode) == 0o444,
            f"repair source seal differs: {path}",
        )
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)


def _validate_sealed_repair_source(
    source_root: Path, source: Mapping[str, Any]
) -> None:
    root = _directory(source_root, "sealed repair source root")
    require(
        set(source)
        == {
            "schema_version",
            "repair_source_commit",
            "repair_package_protocol_sha256",
            "repair_source_files",
            "repair_source_files_sha256",
        }
        and type(source.get("schema_version")) is int
        and source.get("schema_version") == 1
        and isinstance(source.get("repair_source_commit"), str)
        and GIT_RE.fullmatch(source["repair_source_commit"]) is not None
        and isinstance(source.get("repair_package_protocol_sha256"), str)
        and SHA256_RE.fullmatch(source["repair_package_protocol_sha256"])
        is not None,
        "sealed repair source identity differs",
    )
    require(
        stat.S_IMODE(root.lstat().st_mode) == 0o555,
        "sealed repair source root mode differs",
    )
    expected_files = source.get("repair_source_files")
    require(
        isinstance(expected_files, Mapping)
        and set(expected_files) == set(SOURCE_NAMES)
        and source.get("repair_source_files_sha256") == stable_hash(expected_files),
        "sealed repair source inventory differs",
    )
    actual_names = {entry.name for entry in os.scandir(root)}
    require(
        actual_names == {*SOURCE_NAMES, SOURCE_AUTHORITY_NAME},
        "sealed repair source coverage differs",
    )
    authority, _authority_sha256, authority_info = read_json(
        root / SOURCE_AUTHORITY_NAME, "sealed repair source authority"
    )
    require(
        stat.S_IMODE(authority_info.st_mode) == 0o444
        and authority_info.st_uid == os.getuid()
        and authority_info.st_nlink == 1
        and exact_json_equal(authority, source),
        "sealed repair source authority differs",
    )
    for name in SOURCE_NAMES:
        expected = expected_files[name]
        payload, digest, info = _regular_bytes(root / name, f"sealed repair source {name}")
        require(
            isinstance(expected, Mapping)
            and set(expected) == {"mode", "size", "sha256"}
            and expected.get("mode") == 0o444
            and expected.get("size") == len(payload)
            and expected.get("sha256") == digest
            and stat.S_IMODE(info.st_mode) == 0o444
            and info.st_uid == os.getuid()
            and info.st_nlink == 1,
            f"sealed repair source differs: {name}",
        )
    protocol_payload, _digest, _info = _regular_bytes(
        root / "protocol.sha256", "sealed repair protocol"
    )
    try:
        protocol = protocol_payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RepairError(f"sealed repair protocol is not ASCII: {exc}") from exc
    require(
        protocol == f"{source['repair_package_protocol_sha256']}\n",
        "sealed repair protocol value differs",
    )


def _load_sealed_repair_source(source_root: Path) -> dict[str, Any]:
    source, _source_sha256, source_info = read_json(
        source_root / SOURCE_AUTHORITY_NAME,
        "sealed repair source authority",
    )
    require(
        stat.S_IMODE(source_info.st_mode) == 0o444
        and source_info.st_uid == os.getuid()
        and source_info.st_nlink == 1,
        "sealed repair source authority identity differs",
    )
    _validate_sealed_repair_source(source_root, source)
    return source


def _remove_repair_source_staging(
    staging: Path, source: Mapping[str, Any], attempt_root: Path
) -> None:
    info = staging.lstat()
    mode = stat.S_IMODE(info.st_mode)
    require(
        stat.S_ISDIR(info.st_mode)
        and info.st_uid == os.getuid()
        and info.st_nlink == 2
        and mode in {0o700, 0o555},
        "repair source staging identity differs",
    )
    entries = list(os.scandir(staging))
    entry_names = {entry.name for entry in entries}
    allowed_names = {*SOURCE_NAMES, SOURCE_AUTHORITY_NAME}
    require(
        len(entries) == len(entry_names) and entry_names <= allowed_names,
        "repair source staging coverage differs",
    )
    if mode == 0o555:
        _validate_sealed_repair_source(staging, source)
    else:
        for entry in entries:
            listed = entry.stat(follow_symlinks=False)
            entry_mode = stat.S_IMODE(listed.st_mode)
            require(
                stat.S_ISREG(listed.st_mode)
                and listed.st_uid == os.getuid()
                and listed.st_nlink == 1
                and entry_mode in {0o600, 0o444},
                "repair source staging contains an unsafe entry",
            )
            if entry_mode == 0o600:
                continue
            payload, digest, _opened = _regular_bytes(
                Path(entry.path), f"repair source staging {entry.name}"
            )
            if entry.name == SOURCE_AUTHORITY_NAME:
                require(
                    entry_names == allowed_names,
                    "repair source staging authority coverage differs",
                )
                authority = _decode_json(Path(entry.path), payload)
                require(
                    exact_json_equal(authority, source),
                    "repair source staging authority differs",
                )
            else:
                expected = source["repair_source_files"][entry.name]
                require(
                    expected["mode"] == 0o444
                    and expected["size"] == len(payload)
                    and expected["sha256"] == digest,
                    f"repair source staging differs: {entry.name}",
                )
    os.chmod(staging, 0o700, follow_symlinks=False)
    authority_entries = [
        entry for entry in entries if entry.name == SOURCE_AUTHORITY_NAME
    ]
    other_entries = sorted(
        (entry for entry in entries if entry.name != SOURCE_AUTHORITY_NAME),
        key=lambda entry: entry.name,
    )
    invalidation_order = [*authority_entries, *other_entries]
    for entry in invalidation_order:
        listed = entry.stat(follow_symlinks=False)
        require(
            stat.S_ISREG(listed.st_mode)
            and listed.st_uid == os.getuid()
            and listed.st_nlink == 1
            and stat.S_IMODE(listed.st_mode) in {0o600, 0o444},
            "repair source staging contains an unsafe entry",
        )
        descriptor = os.open(
            entry.path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
        )
        try:
            opened = os.fstat(descriptor)
            require(
                _file_identity(opened) == _file_identity(listed),
                "repair source staging entry identity raced",
            )
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    _fsync_directory(staging)
    removal_order = [*other_entries, *authority_entries]
    for entry in removal_order:
        listed = os.stat(entry.path, follow_symlinks=False)
        require(
            stat.S_ISREG(listed.st_mode)
            and listed.st_uid == os.getuid()
            and listed.st_nlink == 1
            and stat.S_IMODE(listed.st_mode) == 0o600,
            "repair source staging invalidation differs",
        )
        os.unlink(entry.path)
    _fsync_directory(staging)
    staging.rmdir()
    _fsync_directory(attempt_root)


def _seal_repair_source_snapshot(
    submission_root: Path, source: Mapping[str, Any]
) -> Path:
    repair_parent = submission_root / "report-repair"
    attempt_root = _repair_root(submission_root)
    for path in (repair_parent, attempt_root):
        if not os.path.lexists(path):
            path.mkdir(mode=0o700)
            _fsync_directory(path.parent)
        else:
            info = _directory(path, f"repair state directory {path.name}").lstat()
            require(
                info.st_uid == os.getuid()
                and stat.S_IMODE(info.st_mode) == 0o700,
                f"repair state directory identity differs: {path}",
            )
    source_root = _repair_source_root(submission_root)
    if os.path.lexists(source_root):
        sealed_source = _load_sealed_repair_source(source_root)
        require(
            exact_json_equal(sealed_source, source),
            "sealed repair source differs from requested source identity",
        )
        return source_root
    leftovers = sorted(attempt_root.glob(".source.tmp.*"))
    for leftover in leftovers:
        _remove_repair_source_staging(leftover, source, attempt_root)
    staging = attempt_root / f".source.tmp.{os.getpid()}.{time.time_ns()}"
    staging.mkdir(mode=0o700)
    _fsync_directory(attempt_root)
    try:
        expected_files = source["repair_source_files"]
        for name in SOURCE_NAMES:
            payload, digest, _info = _regular_bytes(
                PACKAGE_DIR / name, f"live repair source snapshot {name}"
            )
            require(
                expected_files[name]["size"] == len(payload)
                and expected_files[name]["sha256"] == digest,
                f"live repair source changed before snapshot: {name}",
            )
            _write_sealed_file(staging / name, payload)
        authority_payload = (
            json.dumps(dict(source), sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        _write_sealed_file(staging / SOURCE_AUTHORITY_NAME, authority_payload)
        _fsync_directory(staging)
        descriptor = os.open(
            staging,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fchmod(descriptor, 0o555)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        require(not os.path.lexists(source_root), "repair source target appeared concurrently")
        _rename_directory_noreplace(staging, source_root)
        _fsync_directory(attempt_root)
    except BaseException:
        if os.path.lexists(staging):
            _remove_repair_source_staging(staging, source, attempt_root)
        raise
    _validate_sealed_repair_source(source_root, source)
    return source_root


def _failure_path(submission_root: Path) -> Path:
    return _journal_path(
        submission_root, "REPORT_REPAIR_0001_ORIGINAL_FAILURE.json"
    )


def _submit_calling_path(submission_root: Path) -> Path:
    return _journal_path(
        submission_root, "CALLING_REPORT_REPAIR_0001_SUBMIT.json"
    )


def _submitted_path(submission_root: Path) -> Path:
    return _journal_path(submission_root, "REPORT_REPAIR_0001_SUBMITTED.json")


def _submit_failure_terminal_path(submission_root: Path) -> Path:
    return _journal_path(
        submission_root, "REPORT_REPAIR_0001_TERMINAL_SUBMIT_FAILURE.json"
    )


def _authorization_path(submission_root: Path) -> Path:
    return _journal_path(submission_root, "REPORT_REPAIR_0001_AUTHORIZED.json")


def _released_path(submission_root: Path) -> Path:
    return _journal_path(submission_root, "REPORT_REPAIR_0001_RELEASED.json")


def _release_denied_path(submission_root: Path) -> Path:
    return _journal_path(
        submission_root, "REPORT_REPAIR_0001_TERMINAL_RELEASE_DENIED.json"
    )


def _worker_failure_terminal_path(submission_root: Path) -> Path:
    return _journal_path(
        submission_root, "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json"
    )


def _repair_log_path(submission_root: Path) -> Path:
    return submission_root / "logs" / "report-repair-0001-%j.out"


def _repair_journal_namespace_names(submission_root: Path) -> list[str]:
    return sorted(
        entry.name
        for entry in os.scandir(submission_root / "journal")
        if entry.name.startswith("REPORT_REPAIR_")
        or entry.name.startswith("CALLING_REPORT_REPAIR_")
    )


def _require_known_single_generation_namespace(submission_root: Path) -> list[str]:
    names = _repair_journal_namespace_names(submission_root)
    static = {
        "REPORT_REPAIR_0001_ORIGINAL_FAILURE.json",
        "CALLING_REPORT_REPAIR_0001_SUBMIT.json",
        "REPORT_REPAIR_0001_SUBMITTED.json",
        "REPORT_REPAIR_0001_TERMINAL_SUBMIT_FAILURE.json",
        "REPORT_REPAIR_0001_AUTHORIZED.json",
        "REPORT_REPAIR_0001_RELEASED.json",
        "REPORT_REPAIR_0001_TERMINAL_RELEASE_DENIED.json",
        "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json",
    }
    patterns = (
        re.compile(r"CALLING_REPORT_REPAIR_0001_RELEASE_[0-9]{4}\.json\Z"),
        re.compile(r"REPORT_REPAIR_0001_RELEASE_RESULT_[0-9]{4}\.json\Z"),
        re.compile(r"REPORT_REPAIR_0001_CANCEL_AUTHORIZED_[0-9]{4}\.json\Z"),
        re.compile(
            r"CALLING_REPORT_REPAIR_0001_SCANCEL_[0-9]{4}_[0-9]{4}\.json\Z"
        ),
        re.compile(
            r"REPORT_REPAIR_0001_SCANCEL_RESULT_[0-9]{4}_[0-9]{4}\.json\Z"
        ),
        re.compile(r"REPORT_REPAIR_0001_CANCEL_TERMINAL_[0-9]{4}\.json\Z"),
    )
    require(
        all(
            name in static or any(pattern.fullmatch(name) for pattern in patterns)
            for name in names
        ),
        "report repair namespace contains a forbidden generation or artifact",
    )
    return names


def _sbatch_command(
    source_root: Path,
    submission_root: Path,
    submission_sha256: str,
    source: Mapping[str, Any],
) -> list[str]:
    snapshot_root = submission_root / "source-snapshot" / "repo"
    report_source = source["repair_source_files"]["report.py"]
    return [
        "/usr/local/bin/sbatch",
        "--parsable",
        "--hold",
        "--no-requeue",
        "--export=NONE",
        f"--job-name={_repair_name(submission_sha256)}",
        f"--comment={_repair_comment(submission_sha256)}",
        f"--output={_repair_log_path(submission_root)}",
        str(source_root / "report_repair.slurm"),
        str(snapshot_root),
        str(submission_root),
        submission_sha256,
        str(ATTEMPT),
        str(source_root),
        str(source_root.joinpath("report.py")),
        str(report_source["sha256"]),
        str(report_source["size"]),
    ]


def _parse_sbatch_job_id(result: CommandResult) -> str:
    require(result.returncode == 0 and result.stderr == b"", "report repair sbatch failed")
    try:
        line = result.stdout.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise RepairError(f"report repair sbatch stdout is not ASCII: {exc}") from exc
    require(line and "\n" not in line and "\r" not in line, "report repair sbatch stdout differs")
    job_id = line.split(";", 1)[0]
    require(JOB_ID_RE.fullmatch(job_id) is not None, "report repair sbatch job ID differs")
    return job_id


def _validated_submit_failure_terminal(
    submission_root: Path,
    submission_sha256: str,
    calling: Mapping[str, Any],
    calling_sha256: str,
) -> dict[str, Any] | None:
    path = _submit_failure_terminal_path(submission_root)
    if not os.path.lexists(path):
        return None
    value, _digest, info = read_json(path, "report repair terminal submit failure")
    require(
        stat.S_IMODE(info.st_mode) == 0o444
        and info.st_uid == os.getuid()
        and info.st_nlink == 1
        and set(value) == SUBMIT_FAILURE_TERMINAL_KEYS
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("status") == "report_repair_terminal_submit_failure"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and type(value.get("attempt")) is int
        and value.get("attempt") == ATTEMPT
        and value.get("submit_calling_sha256") == calling_sha256
        and isinstance(value.get("scheduler_evidence"), Mapping)
        and isinstance(value.get("post_failure_census"), Mapping)
        and value.get("publication_allowed") is False
        and value.get("retry_allowed") is False
        and isinstance(value.get("sealed_at_utc"), str)
        and bool(value["sealed_at_utc"]),
        "report repair terminal submit failure differs",
    )
    result = _validated_command_evidence(
        value["scheduler_evidence"],
        label="report repair failed sbatch result",
        expected_argv=calling["command"],
        expected_environment=calling["scheduler_environment"],
    )
    _validated_scheduler_census(value["post_failure_census"])
    require(
        result.returncode != 0
        and not _repair_rows(value["post_failure_census"], submission_sha256),
        "report repair terminal submit failure outcome differs",
    )
    return dict(value)


def _validate_submit_calling(
    value: Mapping[str, Any],
    *,
    submission_root: Path,
    submission_sha256: str,
    failure_sha256: str,
    source: Mapping[str, Any],
    locks: _RepairLocks,
) -> dict[str, Any]:
    source_root = _repair_source_root(submission_root)
    expected_command = _sbatch_command(
        source_root, submission_root, submission_sha256, source
    )
    transaction, report_cancel = locks.bindings()
    require(
        set(value) == SUBMIT_CALLING_KEYS
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("status") == "calling_held_report_repair_submission"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and type(value.get("attempt")) is int
        and value.get("attempt") == ATTEMPT
        and value.get("original_failure_evidence")
        == "journal/REPORT_REPAIR_0001_ORIGINAL_FAILURE.json"
        and value.get("original_failure_evidence_sha256") == failure_sha256
        and value.get("repair_source_root") == str(source_root)
        and value.get("repair_source_commit") == source.get("repair_source_commit")
        and value.get("repair_package_protocol_sha256")
        == source.get("repair_package_protocol_sha256")
        and exact_json_equal(
            value.get("repair_source_files"), source.get("repair_source_files")
        )
        and value.get("repair_source_files_sha256")
        == source.get("repair_source_files_sha256")
        and isinstance(value.get("scheduler_pre_submit_census"), Mapping)
        and value.get("scheduler_pre_submit_census_sha256")
        == stable_hash(value["scheduler_pre_submit_census"])
        and value.get("command") == expected_command
        and exact_json_equal(
            value.get("scheduler_environment"),
            _scheduler_environment(
                str(
                    read_json(
                        submission_root / "SUBMISSION_CONTRACT.json",
                        "submission contract for repair calling",
                    )[0]["scheduler_control_plane_contract"]["slurm_conf"]
                )
            ),
        )
        and exact_json_equal(value.get("transaction_lock"), transaction)
        and exact_json_equal(value.get("report_cancel_lock"), report_cancel)
        and isinstance(value.get("called_at_utc"), str)
        and bool(value["called_at_utc"]),
        "report repair submit-calling evidence differs",
    )
    pre_submit = _validated_scheduler_census(
        value["scheduler_pre_submit_census"]
    )
    require(
        pre_submit["settled_rows"] == []
        and pre_submit["captured_at_utc"] <= value["called_at_utc"]
        and all(
            exact_json_equal(
                observation["raw"]["environment"],
                value["scheduler_environment"],
            )
            for observation in pre_submit["rounds"]
        ),
        "report repair submit-calling fresh scheduler authority differs",
    )
    return dict(value)


def _validate_submitted(
    value: Mapping[str, Any],
    *,
    submission_sha256: str,
    calling_sha256: str,
    calling: Mapping[str, Any],
) -> dict[str, Any]:
    job_id = value.get("repair_report_job_id")
    evidence = value.get("submission_evidence")
    require(
        set(value) == SUBMITTED_KEYS
        and value.get("schema_version") == 1
        and value.get("status") == "held_report_repair_submitted"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and value.get("attempt") == ATTEMPT
        and value.get("submit_calling_sha256") == calling_sha256
        and isinstance(job_id, str)
        and JOB_ID_RE.fullmatch(job_id) is not None
        and isinstance(evidence, Mapping)
        and evidence.get("mode") in {"direct_sbatch_response", "lost_response_census_adoption"}
        and isinstance(value.get("accepted_at_utc"), str),
        "report repair submitted evidence differs",
    )
    if evidence["mode"] == "direct_sbatch_response":
        require(
            set(evidence) == {"mode", "raw"},
            "report repair direct submission evidence shape differs",
        )
        raw = evidence.get("raw")
        result = _validated_command_evidence(
            raw,
            label="report repair direct submission",
            expected_argv=calling["command"],
            expected_environment=calling["scheduler_environment"],
        )
        require(
            result.returncode == 0 and result.stderr == b"",
            "report repair direct submission evidence differs",
        )
        require(
            _parse_sbatch_job_id(result) == job_id,
            "report repair direct submission ID differs",
        )
    else:
        require(
            set(evidence) == {"mode", "census", "census_sha256"}
            and isinstance(evidence.get("census"), Mapping)
            and evidence.get("census_sha256") == stable_hash(evidence["census"]),
            "report repair lost-response adoption evidence differs",
        )
        _validated_scheduler_census(evidence["census"])
        rows = _repair_rows(evidence["census"], submission_sha256)
        require(
            len(rows) == 1
            and len(evidence["census"]["settled_rows"]) == 1
            and rows[0]["job_id"] == job_id
            and job_id not in HISTORICAL_JOB_IDS
            and rows[0]["state"] == "PENDING"
            and rows[0]["reason"] in {"JobHeldUser", "JobHeldAdmin"},
            "report repair lost-response adopted job was not exact and held",
        )
    return dict(value)


def _authorization_value(
    *,
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_map: Mapping[str, Any],
    expected_reassembly: Mapping[str, Any],
    source: Mapping[str, Any],
    failure_sha256: str,
    calling_sha256: str,
    submitted_sha256: str,
    job_id: str,
    census: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "authorized_terminal_report_repair",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "attempt": ATTEMPT,
        "original_report_job_id": receipt["report_job_id"],
        "repair_report_job_id": job_id,
        "repair_job_name": _repair_name(submission_sha256),
        "scheduler_comment": _repair_comment(submission_sha256),
        "snapshot_root": contract["snapshot_root"],
        "snapshot_inventory_sha256": contract["snapshot_inventory_sha256"],
        "original_package_protocol_sha256": contract["package_protocol_sha256"],
        "original_failure_evidence": "journal/REPORT_REPAIR_0001_ORIGINAL_FAILURE.json",
        "original_failure_evidence_sha256": failure_sha256,
        "worker_receipt_map": dict(receipt_map),
        "worker_receipt_map_sha256": stable_hash(receipt_map),
        "repair_source_root": str(_repair_source_root(submission_root)),
        "repair_source_commit": source["repair_source_commit"],
        "repair_package_protocol_sha256": source["repair_package_protocol_sha256"],
        "repair_source_files": dict(source["repair_source_files"]),
        "repair_source_files_sha256": source["repair_source_files_sha256"],
        "submit_calling_sha256": calling_sha256,
        "submitted_evidence": "journal/REPORT_REPAIR_0001_SUBMITTED.json",
        "submitted_evidence_sha256": submitted_sha256,
        "scheduler_authority_census": dict(census),
        "scheduler_authority_census_sha256": stable_hash(census),
        "worker_handoff": dict(REPAIR_WORKER_HANDOFF),
        "expected_reassembly": dict(expected_reassembly),
        "publication_allowed": True,
        "deterministic_reassembly_allowed": True,
        "scientific_input_change_allowed": False,
        "gate_change_allowed": False,
        "scheduler_submission_allowed": False,
        "authorized_at_utc": _utc_now(),
    }


def _validate_authorization(
    value: Mapping[str, Any],
    *,
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_map: Mapping[str, Any],
    expected_reassembly: Mapping[str, Any],
    source: Mapping[str, Any],
    failure_sha256: str,
    calling_sha256: str,
    submitted_sha256: str,
) -> dict[str, Any]:
    require(
        set(value) == AUTHORIZATION_KEYS
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("status") == "authorized_terminal_report_repair"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and type(value.get("attempt")) is int
        and value.get("attempt") == ATTEMPT
        and value.get("original_report_job_id") == receipt.get("report_job_id")
        and isinstance(value.get("repair_report_job_id"), str)
        and JOB_ID_RE.fullmatch(value["repair_report_job_id"]) is not None
        and value["repair_report_job_id"] not in HISTORICAL_JOB_IDS
        and value.get("repair_job_name") == _repair_name(submission_sha256)
        and value.get("scheduler_comment") == _repair_comment(submission_sha256)
        and value.get("snapshot_root") == contract.get("snapshot_root")
        and value.get("snapshot_inventory_sha256")
        == contract.get("snapshot_inventory_sha256")
        and value.get("original_package_protocol_sha256")
        == contract.get("package_protocol_sha256")
        and value.get("original_failure_evidence")
        == "journal/REPORT_REPAIR_0001_ORIGINAL_FAILURE.json"
        and value.get("original_failure_evidence_sha256") == failure_sha256
        and exact_json_equal(value.get("worker_receipt_map"), receipt_map)
        and value.get("worker_receipt_map_sha256") == stable_hash(receipt_map)
        and value.get("repair_source_root") == str(_repair_source_root(submission_root))
        and value.get("repair_source_commit") == source.get("repair_source_commit")
        and value.get("repair_package_protocol_sha256")
        == source.get("repair_package_protocol_sha256")
        and exact_json_equal(
            value.get("repair_source_files"), source.get("repair_source_files")
        )
        and value.get("repair_source_files_sha256")
        == source.get("repair_source_files_sha256")
        and value.get("submit_calling_sha256") == calling_sha256
        and value.get("submitted_evidence")
        == "journal/REPORT_REPAIR_0001_SUBMITTED.json"
        and value.get("submitted_evidence_sha256") == submitted_sha256
        and exact_json_equal(value.get("worker_handoff"), REPAIR_WORKER_HANDOFF)
        and exact_json_equal(value.get("expected_reassembly"), expected_reassembly)
        and value.get("publication_allowed") is True
        and value.get("deterministic_reassembly_allowed") is True
        and value.get("scientific_input_change_allowed") is False
        and value.get("gate_change_allowed") is False
        and value.get("scheduler_submission_allowed") is False
        and isinstance(value.get("authorized_at_utc"), str)
        and bool(value["authorized_at_utc"]),
        "report repair authorization differs",
    )
    census = value.get("scheduler_authority_census")
    require(
        isinstance(census, Mapping)
        and value.get("scheduler_authority_census_sha256") == stable_hash(census),
        "report repair authorization census binding differs",
    )
    rows = _repair_rows(census, submission_sha256)
    require(
        len(rows) == 1
        and len(census["settled_rows"]) == 1
        and rows[0]["job_id"] == value["repair_report_job_id"]
        and value["repair_report_job_id"] not in HISTORICAL_JOB_IDS
        and rows[0]["state"] == "PENDING"
        and rows[0]["reason"] in {"JobHeldUser", "JobHeldAdmin"},
        "report repair authorization is not bound to one exact held job",
    )
    return dict(value)


def _release_calling_path(submission_root: Path, index: int) -> Path:
    return _journal_path(
        submission_root,
        f"CALLING_REPORT_REPAIR_0001_RELEASE_{index:04d}.json",
    )


def _release_result_path(submission_root: Path, index: int) -> Path:
    return _journal_path(
        submission_root,
        f"REPORT_REPAIR_0001_RELEASE_RESULT_{index:04d}.json",
    )


def _validate_release_calling(
    value: Mapping[str, Any],
    *,
    submission_root: Path,
    submission_sha256: str,
    index: int,
    job_id: str,
    authorization_sha256: str,
    contract: Mapping[str, Any],
    locks: _RepairLocks,
) -> dict[str, Any]:
    transaction, report_cancel = locks.bindings()
    require(
        set(value) == RELEASE_CALLING_KEYS
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("status") == "calling_report_repair_release"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and type(value.get("attempt")) is int
        and value.get("attempt") == ATTEMPT
        and type(value.get("release_attempt")) is int
        and value.get("release_attempt") == index
        and value.get("repair_report_job_id") == job_id
        and value.get("authorization_sha256") == authorization_sha256
        and value.get("command") == ["/usr/local/bin/scontrol", "release", job_id]
        and exact_json_equal(
            value.get("scheduler_environment"),
            _scheduler_environment(
                str(contract["scheduler_control_plane_contract"]["slurm_conf"])
            ),
        )
        and exact_json_equal(value.get("transaction_lock"), transaction)
        and exact_json_equal(value.get("report_cancel_lock"), report_cancel)
        and isinstance(value.get("called_at_utc"), str)
        and bool(value["called_at_utc"]),
        "report repair release-calling evidence differs",
    )
    return dict(value)


def _validate_release_result(
    value: Mapping[str, Any],
    *,
    submission_sha256: str,
    index: int,
    job_id: str,
    authorization_sha256: str,
    calling_sha256: str,
    scheduler_environment: Mapping[str, str],
) -> dict[str, Any]:
    require(
        set(value) == RELEASE_RESULT_KEYS
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("status") == "report_repair_release_attempt_observed"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and type(value.get("attempt")) is int
        and value.get("attempt") == ATTEMPT
        and type(value.get("release_attempt")) is int
        and value.get("release_attempt") == index
        and value.get("repair_report_job_id") == job_id
        and value.get("authorization_sha256") == authorization_sha256
        and value.get("release_calling_sha256") == calling_sha256
        and value.get("mode")
        in {
            "direct_release_response",
            "lost_response_reconciled_still_held",
            "lost_response_reconciled_release_effect",
            "lost_response_reconciled_ambiguous_identity",
        }
        and isinstance(value.get("scheduler_evidence"), Mapping)
        and isinstance(value.get("observed_at_utc"), str)
        and bool(value["observed_at_utc"]),
        "report repair release-result evidence differs",
    )
    evidence = value["scheduler_evidence"]
    if value["mode"] == "direct_release_response":
        observed = _validated_command_evidence(
            evidence,
            label=f"report repair release result {index}",
            expected_argv=["/usr/local/bin/scontrol", "release", job_id],
            expected_environment=scheduler_environment,
        )
        require(
            observed.returncode == 0 and observed.stderr == b"",
            "report repair direct release evidence differs",
        )
    else:
        require(
            set(evidence) == {"census", "census_sha256"}
            and isinstance(evidence.get("census"), Mapping)
            and evidence.get("census_sha256") == stable_hash(evidence["census"]),
            "report repair reconciled release evidence differs",
        )
        _validated_scheduler_census(evidence["census"])
        rows = _repair_rows(evidence["census"], submission_sha256)
        ambiguous = _release_census_is_ambiguous(
            evidence["census"], submission_sha256, job_id
        )
        if value["mode"] == "lost_response_reconciled_ambiguous_identity":
            require(ambiguous, "report repair reconciled ambiguity differs")
        else:
            require(not ambiguous, "report repair reconciled release is ambiguous")
            released = _job_is_released_or_absent(
                evidence["census"], submission_sha256, job_id
            )
            require(
                (value["mode"] == "lost_response_reconciled_release_effect")
                == released,
                "report repair reconciled release effect differs",
            )
    return dict(value)


def _release_attempt_prefix(
    submission_root: Path,
    submission_sha256: str,
    job_id: str,
    authorization_sha256: str,
    contract: Mapping[str, Any],
    locks: _RepairLocks,
) -> tuple[
    list[dict[str, Any]],
    int,
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    journal = submission_root / "journal"
    calling_pattern = re.compile(
        r"CALLING_REPORT_REPAIR_0001_RELEASE_([0-9]{4})\.json\Z"
    )
    result_pattern = re.compile(
        r"REPORT_REPAIR_0001_RELEASE_RESULT_([0-9]{4})\.json\Z"
    )
    calling_indices: list[int] = []
    result_indices: list[int] = []
    for path in journal.glob("CALLING_REPORT_REPAIR_0001_RELEASE_*.json"):
        match = calling_pattern.fullmatch(path.name)
        require(match is not None, "report repair release-calling name differs")
        calling_indices.append(int(match.group(1)))
    for path in journal.glob("REPORT_REPAIR_0001_RELEASE_RESULT_*.json"):
        match = result_pattern.fullmatch(path.name)
        require(match is not None, "report repair release-result name differs")
        result_indices.append(int(match.group(1)))
    require(
        sorted(calling_indices) == list(range(len(calling_indices)))
        and set(result_indices) <= set(calling_indices)
        and sorted(result_indices) == list(range(len(result_indices)))
        and len(calling_indices) - len(result_indices) in {0, 1}
        and len(calling_indices) <= 3,
        "report repair release attempt prefix differs",
    )
    records: list[dict[str, Any]] = []
    index = 0
    unpaired: dict[str, Any] | None = None
    ambiguous_result: dict[str, Any] | None = None
    release_effect_result: dict[str, Any] | None = None
    while os.path.lexists(_release_calling_path(submission_root, index)):
        calling, calling_sha, calling_info = read_json(
            _release_calling_path(submission_root, index),
            f"report repair release calling {index}",
        )
        require(
            stat.S_IMODE(calling_info.st_mode) == 0o444
            and calling_info.st_uid == os.getuid()
            and calling_info.st_nlink == 1,
            "report repair release calling identity differs",
        )
        _validate_release_calling(
            calling,
            submission_root=submission_root,
            submission_sha256=submission_sha256,
            index=index,
            job_id=job_id,
            authorization_sha256=authorization_sha256,
            contract=contract,
            locks=locks,
        )
        result_path = _release_result_path(submission_root, index)
        if not os.path.lexists(result_path):
            unpaired = {
                "index": index,
                "calling": calling,
                "calling_sha256": calling_sha,
            }
            index += 1
            break
        result, result_sha, result_info = read_json(
            result_path, f"report repair release result {index}"
        )
        require(
            stat.S_IMODE(result_info.st_mode) == 0o444
            and result_info.st_uid == os.getuid()
            and result_info.st_nlink == 1,
            "report repair release result identity differs",
        )
        validated_result = _validate_release_result(
            result,
            submission_sha256=submission_sha256,
            index=index,
            job_id=job_id,
            authorization_sha256=authorization_sha256,
            calling_sha256=calling_sha,
            scheduler_environment=calling["scheduler_environment"],
        )
        if validated_result["mode"] == "lost_response_reconciled_ambiguous_identity":
            require(
                ambiguous_result is None,
                "report repair has multiple ambiguous release results",
            )
            ambiguous_result = {
                "release_attempt": index,
                "result": result_path.name,
                "result_sha256": result_sha,
                "census": dict(validated_result["scheduler_evidence"]["census"]),
            }
        if validated_result["mode"] == "lost_response_reconciled_release_effect":
            require(
                release_effect_result is None,
                "report repair has multiple terminal release-effect results",
            )
            release_effect_result = {
                "release_attempt": index,
                "result": result_path.name,
                "result_sha256": result_sha,
                "census": dict(validated_result["scheduler_evidence"]["census"]),
            }
        records.append(
            {
                "release_attempt": index,
                "calling": _release_calling_path(submission_root, index).name,
                "calling_sha256": calling_sha,
                "result": result_path.name,
                "result_sha256": result_sha,
            }
        )
        index += 1
    require(
        not os.path.lexists(_release_result_path(submission_root, index)),
        "report repair release result exists without calling evidence",
    )
    require(
        ambiguous_result is None
        or (
            unpaired is None
            and ambiguous_result["release_attempt"] == len(calling_indices) - 1
        ),
        "report repair ambiguous release result has successor attempts",
    )
    require(
        release_effect_result is None
        or (
            unpaired is None
            and release_effect_result["release_attempt"]
            == len(calling_indices) - 1
        ),
        "report repair terminal release-effect result has successor attempts",
    )
    require(index <= 3, "report repair release-attempt limit exceeded")
    return records, index, unpaired, ambiguous_result, release_effect_result


def _job_is_released_or_absent(
    census: Mapping[str, Any], submission_sha256: str, job_id: str
) -> bool:
    rows = _repair_rows(census, submission_sha256)
    require(
        len(rows) <= 1 and (not rows or rows[0]["job_id"] == job_id),
        "report repair release census identity differs",
    )
    if not rows:
        return True
    return not (
        rows[0]["state"] == "PENDING"
        and rows[0]["reason"] in {"JobHeldUser", "JobHeldAdmin"}
    )


def _release_census_is_ambiguous(
    census: Mapping[str, Any], submission_sha256: str, job_id: str
) -> bool:
    rows = _repair_rows(census, submission_sha256)
    settled = census.get("settled_rows")
    require(isinstance(settled, list), "report repair release census rows differ")
    return (
        len(settled) != len(rows)
        or len(rows) > 1
        or (len(rows) == 1 and rows[0]["job_id"] != job_id)
    )


def _active_squeue_worker_liveness(
    census: Mapping[str, Any], submission_sha256: str, job_id: str
) -> dict[str, Any] | None:
    rows = _repair_rows(census, submission_sha256)
    require(
        len(census["settled_rows"]) == 1
        and len(rows) == 1
        and rows[0]["job_id"] == job_id,
        "report repair worker liveness census identity differs",
    )
    row = rows[0]
    require(
        row["state"]
        in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
        and not (
            row["state"] == "PENDING"
            and row["reason"] in {"JobHeldUser", "JobHeldAdmin"}
        ),
        "report repair worker is not active after release",
    )
    return {
        "schema_version": 1,
        "mode": "active_squeue_identity",
        "repair_report_job_id": job_id,
        "state": row["state"],
        "reason": row["reason"],
        "scheduler_census_sha256": stable_hash(census),
    }


def _accounting_worker_liveness(
    observation: Mapping[str, Any], job_id: str
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "mode": "active_accounting_identity",
        "repair_report_job_id": job_id,
        "accounting_observation": dict(observation),
        "accounting_observation_sha256": stable_hash(observation),
    }


def _validated_worker_liveness(
    value: Mapping[str, Any],
    *,
    census: Mapping[str, Any],
    contract: Mapping[str, Any],
    job_id: str,
    submission_sha256: str,
) -> dict[str, Any]:
    mode = value.get("mode")
    if mode == "active_squeue_identity":
        expected = _active_squeue_worker_liveness(census, submission_sha256, job_id)
        require(
            expected is not None and exact_json_equal(value, expected),
            "report repair squeue worker liveness differs",
        )
    elif mode == "active_accounting_identity":
        require(
            set(value)
            == {
                "schema_version",
                "mode",
                "repair_report_job_id",
                "accounting_observation",
                "accounting_observation_sha256",
            }
            and type(value.get("schema_version")) is int
            and value.get("schema_version") == 1
            and value.get("repair_report_job_id") == job_id
            and isinstance(value.get("accounting_observation"), Mapping)
            and value.get("accounting_observation_sha256")
            == stable_hash(value["accounting_observation"]),
            "report repair accounting worker liveness differs",
        )
        observation = _validated_repair_job_accounting_observation(
            value["accounting_observation"],
            contract=contract,
            job_id=job_id,
            submission_sha256=submission_sha256,
        )
        require(
            not census.get("settled_rows")
            and not _repair_rows(census, submission_sha256)
            and _repair_accounting_classification(observation) == "active",
            "report repair accounting worker is not active",
        )
    else:
        raise RepairError("report repair worker liveness mode differs")
    return dict(value)


def _validated_release_denied_terminal(
    submission_root: Path,
    submission_sha256: str,
    job_id: str,
    authorization_sha256: str,
) -> dict[str, Any] | None:
    path = _release_denied_path(submission_root)
    if not os.path.lexists(path):
        return None
    value, _digest, info = read_json(path, "report repair release-denied terminal")
    require(
        stat.S_IMODE(info.st_mode) == 0o444
        and info.st_uid == os.getuid()
        and info.st_nlink == 1
        and set(value) == RELEASE_DENIED_KEYS
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("status") == "report_repair_terminal_release_denied"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and type(value.get("attempt")) is int
        and value.get("attempt") == ATTEMPT
        and value.get("repair_report_job_id") == job_id
        and value.get("authorization_sha256") == authorization_sha256
        and value.get("reason") == "authorized_repair_job_absent_before_release"
        and isinstance(value.get("pre_release_census"), Mapping)
        and value.get("pre_release_census_sha256")
        == stable_hash(value["pre_release_census"])
        and value.get("publication_allowed") is False
        and value.get("retry_allowed") is False
        and isinstance(value.get("sealed_at_utc"), str)
        and bool(value["sealed_at_utc"]),
        "report repair release-denied terminal differs",
    )
    _validated_scheduler_census(value["pre_release_census"])
    require(
        _repair_rows(value["pre_release_census"], submission_sha256) == [],
        "report repair release-denied terminal has a live repair identity",
    )
    require(
        not os.path.lexists(_released_path(submission_root))
        and not list(
            (submission_root / "journal").glob(
                "CALLING_REPORT_REPAIR_0001_RELEASE_*.json"
            )
        )
        and not list(
            (submission_root / "journal").glob(
                "REPORT_REPAIR_0001_RELEASE_RESULT_*.json"
            )
        ),
        "report repair release-denied terminal has release successor state",
    )
    return dict(value)


def _seal_release_denied_terminal(
    submission_root: Path,
    submission_sha256: str,
    job_id: str,
    authorization_sha256: str,
    census: Mapping[str, Any],
) -> dict[str, Any]:
    value = {
        "schema_version": 1,
        "status": "report_repair_terminal_release_denied",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "attempt": ATTEMPT,
        "repair_report_job_id": job_id,
        "authorization_sha256": authorization_sha256,
        "reason": "authorized_repair_job_absent_before_release",
        "pre_release_census": dict(census),
        "pre_release_census_sha256": stable_hash(census),
        "publication_allowed": False,
        "retry_allowed": False,
        "sealed_at_utc": _utc_now(),
    }
    seal_json(_release_denied_path(submission_root), value)
    validated = _validated_release_denied_terminal(
        submission_root, submission_sha256, job_id, authorization_sha256
    )
    assert validated is not None
    return validated


def _validated_worker_failure_terminal(
    submission_root: Path,
    submission_sha256: str,
    job_id: str,
    authorization_sha256: str,
    contract: Mapping[str, Any],
    locks: _RepairLocks,
) -> dict[str, Any] | None:
    path = _worker_failure_terminal_path(submission_root)
    if not os.path.lexists(path):
        return None
    value, _digest, info = read_json(path, "report repair worker-failure terminal")
    require(
        stat.S_IMODE(info.st_mode) == 0o444
        and info.st_uid == os.getuid()
        and info.st_nlink == 1
        and set(value) == WORKER_FAILURE_TERMINAL_KEYS
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("status") == "report_repair_terminal_worker_failure"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and type(value.get("attempt")) is int
        and value.get("attempt") == ATTEMPT
        and value.get("repair_report_job_id") == job_id
        and value.get("authorization_sha256") == authorization_sha256
        and value.get("reason")
        in {
            "repair_worker_terminal_before_release_evidence",
            "repair_worker_terminal_after_release_evidence",
        }
        and isinstance(value.get("release_attempts"), list)
        and bool(value["release_attempts"])
        and value.get("release_attempts_sha256")
        == stable_hash(value["release_attempts"])
        and isinstance(value.get("post_release_census"), Mapping)
        and value.get("post_release_census_sha256")
        == stable_hash(value["post_release_census"])
        and isinstance(value.get("terminal_scheduler_observation"), Mapping)
        and value.get("terminal_scheduler_observation_sha256")
        == stable_hash(value["terminal_scheduler_observation"])
        and value.get("publication_allowed") is False
        and value.get("retry_allowed") is False
        and isinstance(value.get("sealed_at_utc"), str)
        and bool(value["sealed_at_utc"]),
        "report repair worker-failure terminal differs",
    )
    _validated_scheduler_census(value["post_release_census"])
    require(
        not _repair_rows(value["post_release_census"], submission_sha256),
        "report repair worker-failure terminal has a live squeue identity",
    )
    observation = _validated_repair_job_accounting_observation(
        value["terminal_scheduler_observation"],
        contract=contract,
        job_id=job_id,
        submission_sha256=submission_sha256,
    )
    require(
        _repair_accounting_classification(observation) == "terminal",
        "report repair worker-failure accounting is not terminal",
    )
    records, _next, unpaired, ambiguous, _effect = _release_attempt_prefix(
        submission_root,
        submission_sha256,
        job_id,
        authorization_sha256,
        contract,
        locks,
    )
    require(
        unpaired is None
        and ambiguous is None
        and exact_json_equal(value["release_attempts"], records),
        "report repair worker-failure release prefix differs",
    )
    release_path = _released_path(submission_root)
    if value.get("released_evidence") is None:
        require(
            value.get("released_evidence_sha256") is None
            and not os.path.lexists(release_path)
            and value["reason"] == "repair_worker_terminal_before_release_evidence",
            "report repair worker-failure release predecessor differs",
        )
    else:
        require(
            value.get("released_evidence")
            == "journal/REPORT_REPAIR_0001_RELEASED.json"
            and SHA256_RE.fullmatch(
                str(value.get("released_evidence_sha256", ""))
            )
            is not None
            and os.path.lexists(release_path)
            and value["reason"] == "repair_worker_terminal_after_release_evidence",
            "report repair worker-failure released predecessor differs",
        )
        _released, release_sha, release_info = read_json(
            release_path, "report repair worker-failure released predecessor"
        )
        require(
            stat.S_IMODE(release_info.st_mode) == 0o444
            and release_info.st_uid == os.getuid()
            and release_info.st_nlink == 1
            and release_sha == value["released_evidence_sha256"],
            "report repair worker-failure released predecessor identity differs",
        )
    require(
        not os.path.lexists(submission_root / "report"),
        "report repair worker-failure conflicts with a published report",
    )
    require(
        not os.path.lexists(_submit_failure_terminal_path(submission_root))
        and not os.path.lexists(_release_denied_path(submission_root)),
        "report repair worker-failure conflicts with another terminal state",
    )
    return dict(value)


def _seal_worker_failure_terminal(
    submission_root: Path,
    submission_sha256: str,
    job_id: str,
    authorization_sha256: str,
    records: Sequence[Mapping[str, Any]],
    census: Mapping[str, Any],
    accounting: Mapping[str, Any],
    contract: Mapping[str, Any],
    locks: _RepairLocks,
) -> dict[str, Any]:
    release_path = _released_path(submission_root)
    released_evidence: str | None = None
    released_evidence_sha256: str | None = None
    if os.path.lexists(release_path):
        _released, released_evidence_sha256, release_info = read_json(
            release_path, "report repair released predecessor"
        )
        require(
            stat.S_IMODE(release_info.st_mode) == 0o444
            and release_info.st_uid == os.getuid()
            and release_info.st_nlink == 1,
            "report repair released predecessor identity differs",
        )
        released_evidence = "journal/REPORT_REPAIR_0001_RELEASED.json"
    value = {
        "schema_version": 1,
        "status": "report_repair_terminal_worker_failure",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "attempt": ATTEMPT,
        "repair_report_job_id": job_id,
        "authorization_sha256": authorization_sha256,
        "reason": (
            "repair_worker_terminal_after_release_evidence"
            if released_evidence is not None
            else "repair_worker_terminal_before_release_evidence"
        ),
        "release_attempts": [dict(item) for item in records],
        "release_attempts_sha256": stable_hash(records),
        "released_evidence": released_evidence,
        "released_evidence_sha256": released_evidence_sha256,
        "post_release_census": dict(census),
        "post_release_census_sha256": stable_hash(census),
        "terminal_scheduler_observation": dict(accounting),
        "terminal_scheduler_observation_sha256": stable_hash(accounting),
        "publication_allowed": False,
        "retry_allowed": False,
        "sealed_at_utc": _utc_now(),
    }
    seal_json(_worker_failure_terminal_path(submission_root), value)
    validated = _validated_worker_failure_terminal(
        submission_root,
        submission_sha256,
        job_id,
        authorization_sha256,
        contract,
        locks,
    )
    assert validated is not None
    return validated


def _seal_released(
    submission_root: Path,
    submission_sha256: str,
    job_id: str,
    authorization_sha256: str,
    records: Sequence[Mapping[str, Any]],
    census: Mapping[str, Any],
    worker_liveness: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    value = {
        "schema_version": 1,
        "status": "report_repair_released",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "attempt": ATTEMPT,
        "repair_report_job_id": job_id,
        "authorization_sha256": authorization_sha256,
        "release_attempts": [dict(item) for item in records],
        "release_attempts_sha256": stable_hash(records),
        "post_release_census": dict(census),
        "post_release_census_sha256": stable_hash(census),
        "worker_liveness_observation": dict(worker_liveness),
        "worker_liveness_observation_sha256": stable_hash(worker_liveness),
        "released_at_utc": _utc_now(),
    }
    return value, seal_json(_released_path(submission_root), value)


def _absent_worker_release_disposition(
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    job_id: str,
    authorization_sha256: str,
    records: Sequence[Mapping[str, Any]],
    census: Mapping[str, Any],
    runner: Runner,
    locks: _RepairLocks,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    require(
        not census.get("settled_rows")
        and not _repair_rows(census, submission_sha256),
        "repair worker absence disposition has a live squeue identity",
    )
    accounting = _repair_job_accounting_observation(
        submission_root,
        contract,
        job_id,
        submission_sha256,
        runner,
        locks,
    )
    _validated_repair_job_accounting_observation(
        accounting,
        contract=contract,
        job_id=job_id,
        submission_sha256=submission_sha256,
    )
    classification = _repair_accounting_classification(accounting)
    if classification == "active":
        return _accounting_worker_liveness(accounting, job_id), None
    if classification == "terminal":
        return None, _seal_worker_failure_terminal(
            submission_root,
            submission_sha256,
            job_id,
            authorization_sha256,
            records,
            census,
            accounting,
            contract,
            locks,
        )
    return None, {
        "schema_version": 1,
        "status": "report_repair_release_effect_awaiting_accounting",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "attempt": ATTEMPT,
        "repair_report_job_id": job_id,
        "accounting_classification": classification,
        "scheduler_calls": 1,
        "publication_allowed": False,
    }


def _broad_only_release_disposition(
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    job_id: str,
    authorization_sha256: str,
    records: Sequence[Mapping[str, Any]],
    census: Mapping[str, Any],
    runner: Runner,
    locks: _RepairLocks,
) -> dict[str, Any]:
    require(
        bool(census.get("settled_rows"))
        and not _repair_rows(census, submission_sha256),
        "repair broad-only release disposition differs",
    )
    accounting = _repair_job_accounting_observation(
        submission_root,
        contract,
        job_id,
        submission_sha256,
        runner,
        locks,
    )
    _validated_repair_job_accounting_observation(
        accounting,
        contract=contract,
        job_id=job_id,
        submission_sha256=submission_sha256,
    )
    classification = _repair_accounting_classification(accounting)
    if classification == "terminal":
        return _seal_worker_failure_terminal(
            submission_root,
            submission_sha256,
            job_id,
            authorization_sha256,
            records,
            census,
            accounting,
            contract,
            locks,
        )
    return {
        "schema_version": 1,
        "status": "report_repair_release_effect_awaiting_unambiguous_namespace",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "attempt": ATTEMPT,
        "repair_report_job_id": job_id,
        "accounting_classification": classification,
        "publication_allowed": False,
        "scheduler_calls": 1,
    }


def _release_authorized_job(
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    authorization: Mapping[str, Any],
    authorization_sha256: str,
    runner: Runner,
    locks: _RepairLocks,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    job_id = str(authorization["repair_report_job_id"])
    worker_failure = _validated_worker_failure_terminal(
        submission_root,
        submission_sha256,
        job_id,
        authorization_sha256,
        contract,
        locks,
    )
    if worker_failure is not None:
        delayed_census = _scheduler_census(
            submission_root, contract, runner, locks, sleep=sleep
        )
        if _repair_rows(delayed_census, submission_sha256):
            return _cleanup_repair_rows(
                submission_root,
                submission_sha256,
                contract,
                delayed_census,
                "identity_visible_after_terminal_worker_failure",
                runner,
                locks,
                sleep=sleep,
            )
        return worker_failure
    existing_cancel_generations = _cancel_generation_count(submission_root)
    if existing_cancel_generations:
        cleanup_census = _scheduler_census(
            submission_root, contract, runner, locks, sleep=sleep
        )
        latest_generation = existing_cancel_generations - 1
        latest_terminal = _cancel_terminal_path(
            submission_root, latest_generation
        )
        if os.path.lexists(latest_terminal) and not _repair_rows(
            cleanup_census, submission_sha256
        ):
            return _validated_cleanup_terminal(
                submission_root,
                submission_sha256,
                latest_generation,
                contract,
                locks,
            )
        return _cleanup_repair_rows(
            submission_root,
            submission_sha256,
            contract,
            cleanup_census,
            "residual_exact_repair_jobs_after_release_stop",
            runner,
            locks,
            sleep=sleep,
        )
    denied = _validated_release_denied_terminal(
        submission_root, submission_sha256, job_id, authorization_sha256
    )
    if denied is not None:
        delayed_census = _scheduler_census(
            submission_root, contract, runner, locks, sleep=sleep
        )
        if _repair_rows(delayed_census, submission_sha256):
            return _cleanup_repair_rows(
                submission_root,
                submission_sha256,
                contract,
                delayed_census,
                "delayed_identity_after_terminal_release_denied",
                runner,
                locks,
                sleep=sleep,
            )
        return denied
    (
        records,
        next_index,
        unpaired,
        ambiguous_result,
        release_effect_result,
    ) = _release_attempt_prefix(
        submission_root,
        submission_sha256,
        job_id,
        authorization_sha256,
        contract,
        locks,
    )
    if ambiguous_result is not None:
        cleanup_census = _scheduler_census(
            submission_root, contract, runner, locks, sleep=sleep
        )
        if _repair_rows(cleanup_census, submission_sha256):
            return _cleanup_repair_rows(
                submission_root,
                submission_sha256,
                contract,
                cleanup_census,
                "durable_ambiguous_release_identity",
                runner,
                locks,
                sleep=sleep,
            )
        return {
            "schema_version": 1,
            "status": "report_repair_terminal_ambiguous_release_identity",
            "campaign_id": CAMPAIGN_ID,
            "submission_sha256": submission_sha256,
            "attempt": ATTEMPT,
            "repair_report_job_id": job_id,
            "ambiguous_release_result": ambiguous_result["result"],
            "ambiguous_release_result_sha256": ambiguous_result["result_sha256"],
            "publication_allowed": False,
            "retry_allowed": False,
            "scheduler_calls": 3,
        }
    if release_effect_result is not None:
        effect_recheck = _scheduler_census(
            submission_root, contract, runner, locks, sleep=sleep
        )
        effect_rows = _repair_rows(effect_recheck, submission_sha256)
        effect_ambiguous = _release_census_is_ambiguous(
            effect_recheck, submission_sha256, job_id
        )
        effect_held = bool(
            len(effect_rows) == 1
            and effect_rows[0]["state"] == "PENDING"
            and effect_rows[0]["reason"] in {"JobHeldUser", "JobHeldAdmin"}
        )
        if effect_ambiguous or effect_held:
            if effect_ambiguous and not effect_rows:
                return _broad_only_release_disposition(
                    submission_root,
                    submission_sha256,
                    contract,
                    job_id,
                    authorization_sha256,
                    records,
                    effect_recheck,
                    runner,
                    locks,
                )
            return _cleanup_repair_rows(
                submission_root,
                submission_sha256,
                contract,
                effect_recheck,
                "identity_visible_after_terminal_release_effect",
                runner,
                locks,
                sleep=sleep,
            )
        if effect_rows:
            worker_liveness = _active_squeue_worker_liveness(
                effect_recheck, submission_sha256, job_id
            )
            assert worker_liveness is not None
        else:
            worker_liveness, terminal = _absent_worker_release_disposition(
                submission_root,
                submission_sha256,
                contract,
                job_id,
                authorization_sha256,
                records,
                effect_recheck,
                runner,
                locks,
            )
            if terminal is not None:
                return terminal
            assert worker_liveness is not None
        _final, _sha = _seal_released(
            submission_root,
            submission_sha256,
            job_id,
            authorization_sha256,
            records,
            effect_recheck,
            worker_liveness,
        )
        validated = _validate_existing_release(
            submission_root,
            submission_sha256,
            authorization_sha256,
            job_id,
            contract,
            locks,
        )
        assert validated is not None
        return validated
    if unpaired is not None:
        census = _scheduler_census(
            submission_root, contract, runner, locks, sleep=sleep
        )
        observed_rows = _repair_rows(census, submission_sha256)
        ambiguous = _release_census_is_ambiguous(
            census, submission_sha256, job_id
        )
        released = (
            False
            if ambiguous
            else _job_is_released_or_absent(census, submission_sha256, job_id)
        )
        result = {
            "schema_version": 1,
            "status": "report_repair_release_attempt_observed",
            "campaign_id": CAMPAIGN_ID,
            "submission_sha256": submission_sha256,
            "attempt": ATTEMPT,
            "release_attempt": unpaired["index"],
            "repair_report_job_id": job_id,
            "authorization_sha256": authorization_sha256,
            "release_calling_sha256": unpaired["calling_sha256"],
            "mode": (
                "lost_response_reconciled_ambiguous_identity"
                if ambiguous
                else (
                    "lost_response_reconciled_release_effect"
                    if released
                    else "lost_response_reconciled_still_held"
                )
            ),
            "scheduler_evidence": {
                "census": census,
                "census_sha256": stable_hash(census),
            },
            "observed_at_utc": _utc_now(),
        }
        if ambiguous and observed_rows:
            _ensure_cleanup_authority(
                submission_root,
                submission_sha256,
                census,
                "ambiguous_identity_after_release_calling",
                contract,
                locks,
            )
        result_sha = seal_json(
            _release_result_path(submission_root, unpaired["index"]), result
        )
        records.append(
            {
                "release_attempt": unpaired["index"],
                "calling": _release_calling_path(
                    submission_root, unpaired["index"]
                ).name,
                "calling_sha256": unpaired["calling_sha256"],
                "result": _release_result_path(
                    submission_root, unpaired["index"]
                ).name,
                "result_sha256": result_sha,
            }
        )
        if ambiguous:
            if not observed_rows:
                return _broad_only_release_disposition(
                    submission_root,
                    submission_sha256,
                    contract,
                    job_id,
                    authorization_sha256,
                    records,
                    census,
                    runner,
                    locks,
                )
            return _cleanup_repair_rows(
                submission_root,
                submission_sha256,
                contract,
                census,
                "ambiguous_identity_after_release_calling",
                runner,
                locks,
                sleep=sleep,
            )
        if released:
            if observed_rows:
                worker_liveness = _active_squeue_worker_liveness(
                    census, submission_sha256, job_id
                )
                assert worker_liveness is not None
            else:
                worker_liveness, terminal = _absent_worker_release_disposition(
                    submission_root,
                    submission_sha256,
                    contract,
                    job_id,
                    authorization_sha256,
                    records,
                    census,
                    runner,
                    locks,
                )
                if terminal is not None:
                    return terminal
                assert worker_liveness is not None
            _final, _sha = _seal_released(
                submission_root,
                submission_sha256,
                job_id,
                authorization_sha256,
                records,
                census,
                worker_liveness,
            )
            validated = _validate_existing_release(
                submission_root,
                submission_sha256,
                authorization_sha256,
                job_id,
                contract,
                locks,
            )
            assert validated is not None
            return validated
        next_index = unpaired["index"] + 1
    pre_release_census = _scheduler_census(
        submission_root, contract, runner, locks, sleep=sleep
    )
    pre_release_rows = _repair_rows(pre_release_census, submission_sha256)
    if records:
        completed_attempt_ambiguous = _release_census_is_ambiguous(
            pre_release_census, submission_sha256, job_id
        )
        if completed_attempt_ambiguous:
            if not pre_release_rows:
                return _broad_only_release_disposition(
                    submission_root,
                    submission_sha256,
                    contract,
                    job_id,
                    authorization_sha256,
                    records,
                    pre_release_census,
                    runner,
                    locks,
                )
            return _cleanup_repair_rows(
                submission_root,
                submission_sha256,
                contract,
                pre_release_census,
                "ambiguous_identity_after_completed_release_attempt",
                runner,
                locks,
                sleep=sleep,
            )
        if _job_is_released_or_absent(
            pre_release_census, submission_sha256, job_id
        ):
            if pre_release_rows:
                worker_liveness = _active_squeue_worker_liveness(
                    pre_release_census, submission_sha256, job_id
                )
                assert worker_liveness is not None
            else:
                worker_liveness, terminal = _absent_worker_release_disposition(
                    submission_root,
                    submission_sha256,
                    contract,
                    job_id,
                    authorization_sha256,
                    records,
                    pre_release_census,
                    runner,
                    locks,
                )
                if terminal is not None:
                    return terminal
                assert worker_liveness is not None
            _final, _sha = _seal_released(
                submission_root,
                submission_sha256,
                job_id,
                authorization_sha256,
                records,
                pre_release_census,
                worker_liveness,
            )
            validated = _validate_existing_release(
                submission_root,
                submission_sha256,
                authorization_sha256,
                job_id,
                contract,
                locks,
            )
            assert validated is not None
            return validated
    exact_held_authority = (
        len(pre_release_rows) == 1
        and len(pre_release_census["settled_rows"]) == 1
        and pre_release_rows[0]["job_id"] == job_id
        and job_id not in HISTORICAL_JOB_IDS
        and pre_release_rows[0]["state"] == "PENDING"
        and pre_release_rows[0]["reason"] in {"JobHeldUser", "JobHeldAdmin"}
    )
    if not exact_held_authority:
        if pre_release_rows:
            return _cleanup_repair_rows(
                submission_root,
                submission_sha256,
                contract,
                pre_release_census,
                "pre_release_scheduler_authority_ambiguous",
                runner,
                locks,
                sleep=sleep,
            )
        return _seal_release_denied_terminal(
            submission_root,
            submission_sha256,
            job_id,
            authorization_sha256,
            pre_release_census,
        )
    if next_index >= 3:
        return _cleanup_repair_rows(
            submission_root,
            submission_sha256,
            contract,
            pre_release_census,
            "release_attempt_limit_survived",
            runner,
            locks,
            sleep=sleep,
        )
    environment = _scheduler_environment(
        str(contract["scheduler_control_plane_contract"]["slurm_conf"])
    )
    command = ["/usr/local/bin/scontrol", "release", job_id]
    transaction, report_cancel = locks.bindings()
    calling = {
        "schema_version": 1,
        "status": "calling_report_repair_release",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "attempt": ATTEMPT,
        "release_attempt": next_index,
        "repair_report_job_id": job_id,
        "authorization_sha256": authorization_sha256,
        "command": command,
        "scheduler_environment": environment,
        "transaction_lock": transaction,
        "report_cancel_lock": report_cancel,
        "called_at_utc": _utc_now(),
    }
    calling_sha = seal_json(
        _release_calling_path(submission_root, next_index), calling
    )
    result, evidence = _run(
        runner, command, submission_root, environment, locks
    )
    require(
        result.returncode == 0 and result.stderr == b"",
        "report repair scontrol release failed",
    )
    result_value = {
        "schema_version": 1,
        "status": "report_repair_release_attempt_observed",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "attempt": ATTEMPT,
        "release_attempt": next_index,
        "repair_report_job_id": job_id,
        "authorization_sha256": authorization_sha256,
        "release_calling_sha256": calling_sha,
        "mode": "direct_release_response",
        "scheduler_evidence": evidence,
        "observed_at_utc": _utc_now(),
    }
    result_sha = seal_json(
        _release_result_path(submission_root, next_index), result_value
    )
    records.append(
        {
            "release_attempt": next_index,
            "calling": _release_calling_path(submission_root, next_index).name,
            "calling_sha256": calling_sha,
            "result": _release_result_path(submission_root, next_index).name,
            "result_sha256": result_sha,
        }
    )
    census = _scheduler_census(
        submission_root, contract, runner, locks, sleep=sleep
    )
    direct_rows = _repair_rows(census, submission_sha256)
    direct_ambiguous = _release_census_is_ambiguous(
        census, submission_sha256, job_id
    )
    if direct_ambiguous:
        if not direct_rows:
            return _broad_only_release_disposition(
                submission_root,
                submission_sha256,
                contract,
                job_id,
                authorization_sha256,
                records,
                census,
                runner,
                locks,
            )
        return _cleanup_repair_rows(
            submission_root,
            submission_sha256,
            contract,
            census,
            "ambiguous_identity_after_direct_release_result",
            runner,
            locks,
            sleep=sleep,
        )
    require(
        _job_is_released_or_absent(census, submission_sha256, job_id),
        "report repair job remains held after successful release",
    )
    if direct_rows:
        worker_liveness = _active_squeue_worker_liveness(
            census, submission_sha256, job_id
        )
        assert worker_liveness is not None
    else:
        worker_liveness, terminal = _absent_worker_release_disposition(
            submission_root,
            submission_sha256,
            contract,
            job_id,
            authorization_sha256,
            records,
            census,
            runner,
            locks,
        )
        if terminal is not None:
            return terminal
        assert worker_liveness is not None
    _final, _sha = _seal_released(
        submission_root,
        submission_sha256,
        job_id,
        authorization_sha256,
        records,
        census,
        worker_liveness,
    )
    validated = _validate_existing_release(
        submission_root,
        submission_sha256,
        authorization_sha256,
        job_id,
        contract,
        locks,
    )
    assert validated is not None
    return validated


def _cancel_generation_count(submission_root: Path) -> int:
    existing: list[int] = []
    pattern = re.compile(r"REPORT_REPAIR_0001_CANCEL_AUTHORIZED_([0-9]{4})\.json\Z")
    for path in (submission_root / "journal").glob(
        "REPORT_REPAIR_0001_CANCEL_AUTHORIZED_*.json"
    ):
        match = pattern.fullmatch(path.name)
        require(match is not None, "report repair cancel-authorization name differs")
        existing.append(int(match.group(1)))
    require(
        sorted(existing) == list(range(len(existing))),
        "report repair cancel generations are not contiguous",
    )
    count = len(existing)
    dependent_patterns = (
        (
            "CALLING_REPORT_REPAIR_0001_SCANCEL_*.json",
            re.compile(
                r"CALLING_REPORT_REPAIR_0001_SCANCEL_([0-9]{4})_([0-9]{4})\.json\Z"
            ),
        ),
        (
            "REPORT_REPAIR_0001_SCANCEL_RESULT_*.json",
            re.compile(
                r"REPORT_REPAIR_0001_SCANCEL_RESULT_([0-9]{4})_([0-9]{4})\.json\Z"
            ),
        ),
        (
            "REPORT_REPAIR_0001_CANCEL_TERMINAL_*.json",
            re.compile(r"REPORT_REPAIR_0001_CANCEL_TERMINAL_([0-9]{4})\.json\Z"),
        ),
    )
    for glob_pattern, dependent_pattern in dependent_patterns:
        for path in (submission_root / "journal").glob(glob_pattern):
            match = dependent_pattern.fullmatch(path.name)
            require(
                match is not None and int(match.group(1)) < count,
                "report repair cleanup evidence exists without authorization",
            )
    return count


def _cancel_authority_path(submission_root: Path, generation: int) -> Path:
    return _journal_path(
        submission_root,
        f"REPORT_REPAIR_0001_CANCEL_AUTHORIZED_{generation:04d}.json",
    )


def _cancel_calling_path(
    submission_root: Path, generation: int, cancel_attempt: int
) -> Path:
    return _journal_path(
        submission_root,
        f"CALLING_REPORT_REPAIR_0001_SCANCEL_{generation:04d}_{cancel_attempt:04d}.json",
    )


def _cancel_result_path(
    submission_root: Path, generation: int, cancel_attempt: int
) -> Path:
    return _journal_path(
        submission_root,
        f"REPORT_REPAIR_0001_SCANCEL_RESULT_{generation:04d}_{cancel_attempt:04d}.json",
    )


def _cancel_terminal_path(submission_root: Path, generation: int) -> Path:
    return _journal_path(
        submission_root,
        f"REPORT_REPAIR_0001_CANCEL_TERMINAL_{generation:04d}.json",
    )


def _validated_cancel_authority(
    value: Mapping[str, Any],
    *,
    submission_sha256: str,
    generation: int,
    locks: _RepairLocks,
) -> dict[str, Any]:
    transaction, report_cancel = locks.bindings()
    job_ids = value.get("job_ids")
    require(
        set(value) == CANCEL_AUTHORIZATION_KEYS
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("status") == "report_repair_cleanup_authorized"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and type(value.get("attempt")) is int
        and value.get("attempt") == ATTEMPT
        and type(value.get("cancel_generation")) is int
        and value.get("cancel_generation") == generation
        and isinstance(value.get("reason"), str)
        and bool(value["reason"])
        and isinstance(job_ids, list)
        and bool(job_ids)
        and all(
            isinstance(item, str) and JOB_ID_RE.fullmatch(item) is not None
            for item in job_ids
        )
        and job_ids == sorted(set(job_ids), key=int)
        and isinstance(value.get("pre_cancel_census"), Mapping)
        and value.get("pre_cancel_census_sha256")
        == stable_hash(value["pre_cancel_census"])
        and exact_json_equal(value.get("transaction_lock"), transaction)
        and exact_json_equal(value.get("report_cancel_lock"), report_cancel)
        and isinstance(value.get("authorized_at_utc"), str),
        "report repair cleanup authorization differs",
    )
    _validated_scheduler_census(value["pre_cancel_census"])
    bound_ids = sorted(
        {row["job_id"] for row in _repair_rows(value["pre_cancel_census"], submission_sha256)},
        key=int,
    )
    require(bound_ids == value["job_ids"], "report repair cleanup authority targets differ")
    return dict(value)


def _cancel_attempt_prefix(
    submission_root: Path,
    submission_sha256: str,
    generation: int,
    authority_sha256: str,
    job_ids: Sequence[str],
    contract: Mapping[str, Any],
    locks: _RepairLocks,
) -> tuple[list[dict[str, Any]], int, dict[str, Any] | None]:
    records: list[dict[str, Any]] = []
    index = 0
    unpaired: dict[str, Any] | None = None
    while os.path.lexists(_cancel_calling_path(submission_root, generation, index)):
        calling, calling_sha, calling_info = read_json(
            _cancel_calling_path(submission_root, generation, index),
            f"report repair cleanup calling {generation}/{index}",
        )
        require(
            stat.S_IMODE(calling_info.st_mode) == 0o444
            and calling_info.st_uid == os.getuid()
            and calling_info.st_nlink == 1
            and set(calling) == CANCEL_CALLING_KEYS
            and type(calling.get("schema_version")) is int
            and calling.get("schema_version") == 1
            and calling.get("status") == "calling_report_repair_cleanup"
            and calling.get("campaign_id") == CAMPAIGN_ID
            and calling.get("submission_sha256") == submission_sha256
            and type(calling.get("attempt")) is int
            and calling.get("attempt") == ATTEMPT
            and type(calling.get("cancel_generation")) is int
            and calling.get("cancel_generation") == generation
            and type(calling.get("cancel_attempt")) is int
            and calling.get("cancel_attempt") == index
            and calling.get("authorization_sha256") == authority_sha256
            and calling.get("job_ids") == list(job_ids)
            and calling.get("command") == ["/usr/local/bin/scancel", *job_ids]
            and exact_json_equal(
                calling.get("scheduler_environment"),
                _scheduler_environment(
                    str(contract["scheduler_control_plane_contract"]["slurm_conf"])
                ),
            )
            and exact_json_equal(calling.get("transaction_lock"), locks.bindings()[0])
            and exact_json_equal(calling.get("report_cancel_lock"), locks.bindings()[1])
            and isinstance(calling.get("called_at_utc"), str)
            and bool(calling["called_at_utc"]),
            "report repair cleanup calling differs",
        )
        result_path = _cancel_result_path(submission_root, generation, index)
        if not os.path.lexists(result_path):
            unpaired = {
                "cancel_attempt": index,
                "calling_sha256": calling_sha,
            }
            index += 1
            break
        result, result_sha, result_info = read_json(
            result_path, f"report repair cleanup result {generation}/{index}"
        )
        require(
            stat.S_IMODE(result_info.st_mode) == 0o444
            and result_info.st_uid == os.getuid()
            and result_info.st_nlink == 1
            and set(result) == CANCEL_RESULT_KEYS
            and type(result.get("schema_version")) is int
            and result.get("schema_version") == 1
            and result.get("status") == "report_repair_cleanup_attempt_observed"
            and result.get("campaign_id") == CAMPAIGN_ID
            and result.get("submission_sha256") == submission_sha256
            and type(result.get("attempt")) is int
            and result.get("attempt") == ATTEMPT
            and type(result.get("cancel_generation")) is int
            and result.get("cancel_generation") == generation
            and type(result.get("cancel_attempt")) is int
            and result.get("cancel_attempt") == index
            and result.get("authorization_sha256") == authority_sha256
            and result.get("calling_sha256") == calling_sha
            and result.get("job_ids") == list(job_ids)
            and result.get("mode")
            in {
                "direct_scancel_response",
                "lost_response_reconciled_still_live",
                "lost_response_reconciled_cancel_effect",
            }
            and isinstance(result.get("scheduler_evidence"), Mapping)
            and isinstance(result.get("observed_at_utc"), str)
            and bool(result["observed_at_utc"]),
            "report repair cleanup result differs",
        )
        evidence = result["scheduler_evidence"]
        if result["mode"] == "direct_scancel_response":
            observed = _validated_command_evidence(
                evidence,
                label=f"report repair cleanup result {generation}/{index}",
                expected_argv=["/usr/local/bin/scancel", *job_ids],
                expected_environment=calling["scheduler_environment"],
            )
            require(
                observed.returncode == 0 and observed.stderr == b"",
                "report repair direct cleanup scheduler result differs",
            )
        else:
            require(
                set(evidence) == {"census", "census_sha256"}
                and isinstance(evidence.get("census"), Mapping)
                and evidence.get("census_sha256") == stable_hash(evidence["census"]),
                "report repair reconciled cleanup census binding differs",
            )
            _validated_scheduler_census(evidence["census"])
            live_ids = {
                row["job_id"]
                for row in _repair_rows(evidence["census"], submission_sha256)
            }
            require(
                (result["mode"] == "lost_response_reconciled_still_live")
                == bool(set(job_ids) & live_ids),
                "report repair reconciled cleanup effect differs",
            )
        records.append(
            {
                "cancel_attempt": index,
                "calling": _cancel_calling_path(
                    submission_root, generation, index
                ).name,
                "calling_sha256": calling_sha,
                "result": result_path.name,
                "result_sha256": result_sha,
            }
        )
        index += 1
    require(
        not os.path.lexists(_cancel_result_path(submission_root, generation, index)),
        "report repair cleanup result exists without calling",
    )
    require(index <= 3, "report repair cleanup-attempt limit exceeded")
    return records, index, unpaired


def _seal_cleanup_terminal(
    submission_root: Path,
    submission_sha256: str,
    generation: int,
    authority: Mapping[str, Any],
    authority_sha256: str,
    records: Sequence[Mapping[str, Any]],
    post: Mapping[str, Any],
    contract: Mapping[str, Any],
    locks: _RepairLocks,
) -> dict[str, Any]:
    remaining = _repair_rows(post, submission_sha256)
    terminal = {
        "schema_version": 1,
        "status": (
            "report_repair_terminal_cleanup_complete"
            if not remaining
            else "report_repair_cleanup_residual_jobs"
        ),
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "attempt": ATTEMPT,
        "cancel_generation": generation,
        "reason": authority["reason"],
        "authorization_sha256": authority_sha256,
        "cancel_attempts": [dict(item) for item in records],
        "cancel_attempts_sha256": stable_hash(records),
        "post_cancel_census": dict(post),
        "post_cancel_census_sha256": stable_hash(post),
        "remaining_job_ids": sorted(
            {row["job_id"] for row in remaining}, key=int
        ),
        "publication_allowed": False,
        "retry_allowed": False,
        "sealed_at_utc": _utc_now(),
    }
    seal_json(_cancel_terminal_path(submission_root, generation), terminal)
    return _validated_cleanup_terminal(
        submission_root,
        submission_sha256,
        generation,
        contract,
        locks,
    )


def _validated_cleanup_terminal(
    submission_root: Path,
    submission_sha256: str,
    generation: int,
    contract: Mapping[str, Any],
    locks: _RepairLocks,
) -> dict[str, Any]:
    authority, authority_sha, authority_info = read_json(
        _cancel_authority_path(submission_root, generation),
        f"report repair cleanup authorization {generation}",
    )
    require(
        stat.S_IMODE(authority_info.st_mode) == 0o444
        and authority_info.st_uid == os.getuid()
        and authority_info.st_nlink == 1,
        "report repair cleanup authorization identity differs",
    )
    authority = _validated_cancel_authority(
        authority,
        submission_sha256=submission_sha256,
        generation=generation,
        locks=locks,
    )
    records, _next_attempt, unpaired = _cancel_attempt_prefix(
        submission_root,
        submission_sha256,
        generation,
        authority_sha,
        authority["job_ids"],
        contract,
        locks,
    )
    require(unpaired is None, "report repair cleanup terminal has an unpaired call")
    terminal, _terminal_sha, terminal_info = read_json(
        _cancel_terminal_path(submission_root, generation),
        f"report repair cleanup terminal {generation}",
    )
    require(
        stat.S_IMODE(terminal_info.st_mode) == 0o444
        and terminal_info.st_uid == os.getuid()
        and terminal_info.st_nlink == 1
        and set(terminal) == CANCEL_TERMINAL_KEYS
        and type(terminal.get("schema_version")) is int
        and terminal.get("schema_version") == 1
        and terminal.get("campaign_id") == CAMPAIGN_ID
        and terminal.get("submission_sha256") == submission_sha256
        and type(terminal.get("attempt")) is int
        and terminal.get("attempt") == ATTEMPT
        and type(terminal.get("cancel_generation")) is int
        and terminal.get("cancel_generation") == generation
        and terminal.get("reason") == authority["reason"]
        and terminal.get("authorization_sha256") == authority_sha
        and exact_json_equal(terminal.get("cancel_attempts"), records)
        and terminal.get("cancel_attempts_sha256") == stable_hash(records)
        and isinstance(terminal.get("post_cancel_census"), Mapping)
        and terminal.get("post_cancel_census_sha256")
        == stable_hash(terminal["post_cancel_census"])
        and isinstance(terminal.get("remaining_job_ids"), list)
        and terminal.get("publication_allowed") is False
        and terminal.get("retry_allowed") is False
        and isinstance(terminal.get("sealed_at_utc"), str)
        and bool(terminal["sealed_at_utc"]),
        "report repair cleanup terminal differs",
    )
    _validated_scheduler_census(terminal["post_cancel_census"])
    remaining = sorted(
        {
            row["job_id"]
            for row in _repair_rows(
                terminal["post_cancel_census"], submission_sha256
            )
        },
        key=int,
    )
    require(
        terminal["remaining_job_ids"] == remaining
        and terminal.get("status")
        == (
            "report_repair_cleanup_residual_jobs"
            if remaining
            else "report_repair_terminal_cleanup_complete"
        ),
        "report repair cleanup terminal outcome differs",
    )
    return dict(terminal)


def _ensure_cleanup_authority(
    submission_root: Path,
    submission_sha256: str,
    census: Mapping[str, Any],
    reason: str,
    contract: Mapping[str, Any],
    locks: _RepairLocks,
) -> tuple[int, dict[str, Any], str]:
    """Validate or durably seal the sole next cleanup authority generation."""

    count = _cancel_generation_count(submission_root)
    for completed_generation in range(count):
        if not os.path.lexists(
            _cancel_terminal_path(submission_root, completed_generation)
        ):
            require(
                completed_generation == count - 1,
                "report repair cleanup terminal gap differs",
            )
            break
        _validated_cleanup_terminal(
            submission_root,
            submission_sha256,
            completed_generation,
            contract,
            locks,
        )
    if count and not os.path.lexists(
        _cancel_terminal_path(submission_root, count - 1)
    ):
        generation = count - 1
        authority, authority_sha, authority_info = read_json(
            _cancel_authority_path(submission_root, generation),
            f"report repair cleanup authorization {generation}",
        )
        require(
            stat.S_IMODE(authority_info.st_mode) == 0o444
            and authority_info.st_uid == os.getuid()
            and authority_info.st_nlink == 1,
            "report repair cleanup authorization identity differs",
        )
        return (
            generation,
            _validated_cancel_authority(
                authority,
                submission_sha256=submission_sha256,
                generation=generation,
                locks=locks,
            ),
            authority_sha,
        )

    job_ids = sorted(
        {row["job_id"] for row in _repair_rows(census, submission_sha256)},
        key=int,
    )
    require(job_ids, "report repair cleanup has no exact jobs")
    generation = count
    transaction, report_cancel = locks.bindings()
    authority = {
        "schema_version": 1,
        "status": "report_repair_cleanup_authorized",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "attempt": ATTEMPT,
        "cancel_generation": generation,
        "reason": reason,
        "job_ids": job_ids,
        "pre_cancel_census": dict(census),
        "pre_cancel_census_sha256": stable_hash(census),
        "transaction_lock": transaction,
        "report_cancel_lock": report_cancel,
        "authorized_at_utc": _utc_now(),
    }
    authority_sha = seal_json(
        _cancel_authority_path(submission_root, generation), authority
    )
    return generation, authority, authority_sha


def _cleanup_repair_rows(
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    census: Mapping[str, Any],
    reason: str,
    runner: Runner,
    locks: _RepairLocks,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    generation, authority, authority_sha = _ensure_cleanup_authority(
        submission_root,
        submission_sha256,
        census,
        reason,
        contract,
        locks,
    )

    job_ids = list(authority["job_ids"])
    records, next_attempt, unpaired = _cancel_attempt_prefix(
        submission_root,
        submission_sha256,
        generation,
        authority_sha,
        job_ids,
        contract,
        locks,
    )
    current_rows = _repair_rows(census, submission_sha256)
    current_ids = {row["job_id"] for row in current_rows}
    targets_still_live = bool(set(job_ids) & current_ids)
    if unpaired is not None:
        result_value = {
            "schema_version": 1,
            "status": "report_repair_cleanup_attempt_observed",
            "campaign_id": CAMPAIGN_ID,
            "submission_sha256": submission_sha256,
            "attempt": ATTEMPT,
            "cancel_generation": generation,
            "cancel_attempt": unpaired["cancel_attempt"],
            "authorization_sha256": authority_sha,
            "calling_sha256": unpaired["calling_sha256"],
            "job_ids": job_ids,
            "mode": (
                "lost_response_reconciled_still_live"
                if targets_still_live
                else "lost_response_reconciled_cancel_effect"
            ),
            "scheduler_evidence": {
                "census": dict(census),
                "census_sha256": stable_hash(census),
            },
            "observed_at_utc": _utc_now(),
        }
        result_sha = seal_json(
            _cancel_result_path(
                submission_root, generation, unpaired["cancel_attempt"]
            ),
            result_value,
        )
        records.append(
            {
                "cancel_attempt": unpaired["cancel_attempt"],
                "calling": _cancel_calling_path(
                    submission_root, generation, unpaired["cancel_attempt"]
                ).name,
                "calling_sha256": unpaired["calling_sha256"],
                "result": _cancel_result_path(
                    submission_root, generation, unpaired["cancel_attempt"]
                ).name,
                "result_sha256": result_sha,
            }
        )
        next_attempt = unpaired["cancel_attempt"] + 1
    if targets_still_live:
        require(next_attempt < 3, "report repair cleanup target survived attempt limit")
        environment = _scheduler_environment(
            str(contract["scheduler_control_plane_contract"]["slurm_conf"])
        )
        command = ["/usr/local/bin/scancel", *job_ids]
        transaction, report_cancel = locks.bindings()
        calling = {
            "schema_version": 1,
            "status": "calling_report_repair_cleanup",
            "campaign_id": CAMPAIGN_ID,
            "submission_sha256": submission_sha256,
            "attempt": ATTEMPT,
            "cancel_generation": generation,
            "cancel_attempt": next_attempt,
            "authorization_sha256": authority_sha,
            "job_ids": job_ids,
            "command": command,
            "scheduler_environment": environment,
            "transaction_lock": transaction,
            "report_cancel_lock": report_cancel,
            "called_at_utc": _utc_now(),
        }
        calling_sha = seal_json(
            _cancel_calling_path(submission_root, generation, next_attempt), calling
        )
        result, scheduler_evidence = _run(
            runner, command, submission_root, environment, locks
        )
        require(
            result.returncode == 0 and result.stderr == b"",
            "report repair cleanup scancel failed",
        )
        result_value = {
            "schema_version": 1,
            "status": "report_repair_cleanup_attempt_observed",
            "campaign_id": CAMPAIGN_ID,
            "submission_sha256": submission_sha256,
            "attempt": ATTEMPT,
            "cancel_generation": generation,
            "cancel_attempt": next_attempt,
            "authorization_sha256": authority_sha,
            "calling_sha256": calling_sha,
            "job_ids": job_ids,
            "mode": "direct_scancel_response",
            "scheduler_evidence": scheduler_evidence,
            "observed_at_utc": _utc_now(),
        }
        result_sha = seal_json(
            _cancel_result_path(submission_root, generation, next_attempt),
            result_value,
        )
        records.append(
            {
                "cancel_attempt": next_attempt,
                "calling": _cancel_calling_path(
                    submission_root, generation, next_attempt
                ).name,
                "calling_sha256": calling_sha,
                "result": _cancel_result_path(
                    submission_root, generation, next_attempt
                ).name,
                "result_sha256": result_sha,
            }
        )
        census = _scheduler_census(
            submission_root, contract, runner, locks, sleep=sleep
        )
    return _seal_cleanup_terminal(
        submission_root,
        submission_sha256,
        generation,
        authority,
        authority_sha,
        records,
        census,
        contract,
        locks,
    )


def _validate_existing_release(
    submission_root: Path,
    submission_sha256: str,
    authorization_sha256: str,
    job_id: str,
    contract: Mapping[str, Any],
    locks: _RepairLocks,
) -> dict[str, Any] | None:
    path = _released_path(submission_root)
    if not os.path.lexists(path):
        return None
    require(
        not os.path.lexists(_release_denied_path(submission_root)),
        "report repair released and release-denied states conflict",
    )
    value, digest, info = read_json(path, "report repair released result")
    require(
        stat.S_IMODE(info.st_mode) == 0o444
        and info.st_uid == os.getuid()
        and info.st_nlink == 1
        and set(value) == RELEASED_KEYS
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("status") == "report_repair_released"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and type(value.get("attempt")) is int
        and value.get("attempt") == ATTEMPT
        and value.get("repair_report_job_id") == job_id
        and value.get("authorization_sha256") == authorization_sha256
        and isinstance(value.get("release_attempts"), list)
        and bool(value["release_attempts"])
        and value.get("release_attempts_sha256")
        == stable_hash(value["release_attempts"])
        and isinstance(value.get("post_release_census"), Mapping)
        and value.get("post_release_census_sha256")
        == stable_hash(value["post_release_census"])
        and isinstance(value.get("worker_liveness_observation"), Mapping)
        and value.get("worker_liveness_observation_sha256")
        == stable_hash(value["worker_liveness_observation"])
        and isinstance(value.get("released_at_utc"), str)
        and bool(value["released_at_utc"])
        and SHA256_RE.fullmatch(digest) is not None,
        "report repair released result differs",
    )
    (
        records,
        _next_index,
        unpaired,
        ambiguous_result,
        _release_effect_result,
    ) = _release_attempt_prefix(
        submission_root,
        submission_sha256,
        job_id,
        authorization_sha256,
        contract,
        locks,
    )
    require(
        unpaired is None
        and ambiguous_result is None
        and exact_json_equal(value["release_attempts"], records),
        "report repair released attempt prefix differs",
    )
    _validated_scheduler_census(value["post_release_census"])
    require(
        _job_is_released_or_absent(
            value["post_release_census"], submission_sha256, job_id
        ),
        "report repair released evidence still shows a held job",
    )
    _validated_worker_liveness(
        value["worker_liveness_observation"],
        census=value["post_release_census"],
        contract=contract,
        job_id=job_id,
        submission_sha256=submission_sha256,
    )
    return dict(value)


def _reconcile_released_worker(
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    authorization_sha256: str,
    released: Mapping[str, Any],
    runner: Runner,
    locks: _RepairLocks,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    job_id = str(released["repair_report_job_id"])
    terminal = _validated_worker_failure_terminal(
        submission_root,
        submission_sha256,
        job_id,
        authorization_sha256,
        contract,
        locks,
    )
    census = _scheduler_census(
        submission_root, contract, runner, locks, sleep=sleep
    )
    rows = _repair_rows(census, submission_sha256)
    existing_cancel_generations = _cancel_generation_count(submission_root)
    if existing_cancel_generations:
        latest_generation = existing_cancel_generations - 1
        latest_terminal = _cancel_terminal_path(
            submission_root, latest_generation
        )
        if os.path.lexists(latest_terminal) and not rows:
            return _validated_cleanup_terminal(
                submission_root,
                submission_sha256,
                latest_generation,
                contract,
                locks,
            )
        return _cleanup_repair_rows(
            submission_root,
            submission_sha256,
            contract,
            census,
            "residual_exact_repair_jobs_after_released_cleanup",
            runner,
            locks,
            sleep=sleep,
        )
    ambiguous = _release_census_is_ambiguous(
        census, submission_sha256, job_id
    )
    held = bool(
        len(rows) == 1
        and rows[0]["state"] == "PENDING"
        and rows[0]["reason"] in {"JobHeldUser", "JobHeldAdmin"}
    )
    if rows and (ambiguous or held or terminal is not None):
        return _cleanup_repair_rows(
            submission_root,
            submission_sha256,
            contract,
            census,
            (
                "identity_visible_after_terminal_worker_failure"
                if terminal is not None
                else "identity_ambiguous_after_repair_release"
            ),
            runner,
            locks,
            sleep=sleep,
        )
    if terminal is not None:
        return terminal
    if ambiguous and not rows:
        return _broad_only_release_disposition(
            submission_root,
            submission_sha256,
            contract,
            job_id,
            authorization_sha256,
            released["release_attempts"],
            census,
            runner,
            locks,
        )
    if rows:
        _active_squeue_worker_liveness(census, submission_sha256, job_id)
        return dict(released)
    accounting = _repair_job_accounting_observation(
        submission_root,
        contract,
        job_id,
        submission_sha256,
        runner,
        locks,
    )
    _validated_repair_job_accounting_observation(
        accounting,
        contract=contract,
        job_id=job_id,
        submission_sha256=submission_sha256,
    )
    classification = _repair_accounting_classification(accounting)
    if classification == "active":
        return dict(released)
    if classification == "terminal":
        return _seal_worker_failure_terminal(
            submission_root,
            submission_sha256,
            job_id,
            authorization_sha256,
            released["release_attempts"],
            census,
            accounting,
            contract,
            locks,
        )
    return {
        "schema_version": 1,
        "status": "report_repair_released_worker_awaiting_accounting",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "attempt": ATTEMPT,
        "repair_report_job_id": job_id,
        "accounting_classification": classification,
        "publication_allowed": False,
        "scheduler_calls": 4,
    }


def _validated_repaired_report_tree(
    submission_root: Path,
    submission_sha256: str,
) -> dict[str, Any] | None:
    report_root = submission_root / "report"
    if not os.path.lexists(report_root):
        return None
    root = _directory(report_root, "published repaired report root")
    require(
        stat.S_IMODE(root.lstat().st_mode) == 0o555,
        "published repaired report root mode differs",
    )
    commit, _commit_digest, commit_info = read_json(
        root / "REPORT_COMMIT.json", "published repaired report commit"
    )
    require(
        stat.S_IMODE(commit_info.st_mode) == 0o444
        and commit_info.st_uid == os.getuid()
        and commit_info.st_nlink == 1
        and set(commit)
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
        and commit.get("schema_version") == 1
        and commit.get("status") == EXPECTED_REPORT_STATUS
        and commit.get("scientific_rejection") is True
        and commit.get("campaign_id") == CAMPAIGN_ID
        and commit.get("submission_sha256") == submission_sha256
        and commit.get("report_bundle_sha256") == EXPECTED_BUNDLE_SHA256
        and commit.get("report_bundle_file_sha256") == EXPECTED_BUNDLE_FILE_SHA256
        and commit.get("gate_sha256") == EXPECTED_GATE_SHA256
        and commit.get("gate_decision_file_sha256") == EXPECTED_DECISION_FILE_SHA256,
        "published repaired report commit differs",
    )
    names = {
        "REPORT_COMMIT.json",
        str(commit["report_bundle"]),
        str(commit["gate_decision"]),
        str(commit["provenance"]),
    }
    entries = list(os.scandir(root))
    require(
        {entry.name for entry in entries} == names,
        "published repaired report file coverage differs",
    )
    for entry in entries:
        info = entry.stat(follow_symlinks=False)
        require(
            stat.S_ISREG(info.st_mode)
            and stat.S_IMODE(info.st_mode) == 0o444
            and info.st_uid == os.getuid()
            and info.st_nlink == 1,
            f"published repaired report file identity differs: {entry.name}",
        )
    require(
        _regular_bytes(
            root / str(commit["report_bundle"]),
            "published repaired report bundle",
        )[1]
        == EXPECTED_BUNDLE_FILE_SHA256
        and _regular_bytes(
            root / str(commit["gate_decision"]),
            "published repaired gate decision",
        )[1]
        == EXPECTED_DECISION_FILE_SHA256,
        "published repaired scientific bundle/gate bytes differ",
    )
    provenance, provenance_digest, _info = read_json(
        root / str(commit["provenance"]), "published repaired report provenance"
    )
    require(
        provenance_digest == commit.get("provenance_file_sha256")
        and stable_hash(provenance) == commit.get("provenance_sha256")
        and provenance.get("schema_version") == 2
        and provenance.get("submission_sha256") == submission_sha256
        and isinstance(provenance.get("publication_authority"), Mapping),
        "published repaired report provenance differs",
    )
    authority = provenance["publication_authority"]
    require(
        set(authority)
        == {
            "schema_version",
            "status",
            "attempt",
            "authorization",
            "authorization_sha256",
            "release",
            "release_sha256",
            "original_report_job_id",
            "repair_report_job_id",
            "original_failure_evidence",
            "original_failure_evidence_sha256",
            "worker_receipt_map_sha256",
            "original_snapshot_root",
            "original_snapshot_inventory_sha256",
            "original_package_protocol_sha256",
            "repair_source_root",
            "repair_source_commit",
            "repair_package_protocol_sha256",
            "repair_source_files_sha256",
            "expected_report_bundle_sha256",
            "expected_report_bundle_file_sha256",
            "expected_gate_sha256",
            "expected_gate_decision_file_sha256",
            "deterministic_reassembly_allowed",
            "scientific_input_change_allowed",
            "gate_change_allowed",
        }
        and authority.get("schema_version") == 1
        and authority.get("status") == "authorized_terminal_report_repair"
        and authority.get("attempt") == ATTEMPT
        and authority.get("original_report_job_id") == EXPECTED_ORIGINAL_REPORT_JOB_ID
        and isinstance(authority.get("repair_report_job_id"), str)
        and JOB_ID_RE.fullmatch(authority["repair_report_job_id"]) is not None
        and authority.get("authorization")
        == "journal/REPORT_REPAIR_0001_AUTHORIZED.json"
        and SHA256_RE.fullmatch(str(authority.get("authorization_sha256", "")))
        is not None
        and authority.get("release") == "journal/REPORT_REPAIR_0001_RELEASED.json"
        and SHA256_RE.fullmatch(str(authority.get("release_sha256", ""))) is not None
        and authority.get("original_failure_evidence")
        == "journal/REPORT_REPAIR_0001_ORIGINAL_FAILURE.json"
        and SHA256_RE.fullmatch(
            str(authority.get("original_failure_evidence_sha256", ""))
        )
        is not None
        and authority.get("original_snapshot_root")
        == str(submission_root / "source-snapshot/repo")
        and authority.get("original_snapshot_inventory_sha256")
        == EXPECTED_SNAPSHOT_INVENTORY_SHA256
        and authority.get("original_package_protocol_sha256")
        == EXPECTED_ORIGINAL_PROTOCOL
        and authority.get("expected_report_bundle_sha256")
        == EXPECTED_BUNDLE_SHA256
        and authority.get("expected_report_bundle_file_sha256")
        == EXPECTED_BUNDLE_FILE_SHA256
        and authority.get("expected_gate_sha256") == EXPECTED_GATE_SHA256
        and authority.get("expected_gate_decision_file_sha256")
        == EXPECTED_DECISION_FILE_SHA256
        and authority.get("deterministic_reassembly_allowed") is True
        and authority.get("scientific_input_change_allowed") is False
        and authority.get("gate_change_allowed") is False,
        "published repaired report publication authority differs",
    )
    for relative_key, sha_key, label in (
        ("authorization", "authorization_sha256", "repair authorization"),
        ("release", "release_sha256", "repair release"),
        (
            "original_failure_evidence",
            "original_failure_evidence_sha256",
            "original failure evidence",
        ),
    ):
        relative = Path(str(authority[relative_key]))
        require(
            not relative.is_absolute()
            and relative.parts
            and all(part not in {"", ".", ".."} for part in relative.parts),
            f"published repaired {label} path differs",
        )
        payload, digest, info = _regular_bytes(
            submission_root / relative, f"published repaired {label}"
        )
        require(
            bool(payload)
            and digest == authority[sha_key]
            and stat.S_IMODE(info.st_mode) == 0o444
            and info.st_uid == os.getuid()
            and info.st_nlink == 1,
            f"published repaired {label} bytes differ",
        )
    return dict(commit)


def _source_from_calling(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "repair_source_commit": value.get("repair_source_commit"),
        "repair_package_protocol_sha256": value.get(
            "repair_package_protocol_sha256"
        ),
        "repair_source_files": value.get("repair_source_files"),
        "repair_source_files_sha256": value.get("repair_source_files_sha256"),
    }


def execute_report_repair(
    repo_root: Path,
    submission_root: Path,
    submission_sha256: str,
    *,
    allow_initial_submission: bool,
    runner: Runner = _default_runner,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Advance or reconcile the one repair generation under both locks."""

    root = _canonical_existing_directory(
        repo_root, "report repair repository root"
    )
    submission = _canonical_existing_directory(
        submission_root, "report repair submission root"
    )
    require(root == REPOSITORY_ROOT, "report repair repository root differs")
    require(
        submission == CANONICAL_PRODUCTION_SUBMISSION_ROOT,
        "report repair submission root differs",
    )
    require(submission_sha256 == EXPECTED_SUBMISSION_SHA256, "report repair submission SHA differs")
    with _RepairLocks(submission) as locks:
        repair_namespace_names = _require_known_single_generation_namespace(
            submission
        )
        report_present = os.path.lexists(submission / "report")
        calling_path = _submit_calling_path(submission)
        submitted_path = _submitted_path(submission)
        authorization_path = _authorization_path(submission)
        release_successor_present = any(
            name
            in {
                "REPORT_REPAIR_0001_RELEASED.json",
                "REPORT_REPAIR_0001_TERMINAL_RELEASE_DENIED.json",
                "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json",
            }
            or name.startswith("CALLING_REPORT_REPAIR_0001_RELEASE_")
            or name.startswith("REPORT_REPAIR_0001_RELEASE_RESULT_")
            for name in repair_namespace_names
        )
        if os.path.lexists(calling_path) and not os.path.lexists(submitted_path):
            require(
                not os.path.lexists(authorization_path)
                and not release_successor_present,
                "report repair positive successor lacks submitted evidence",
            )
        if os.path.lexists(submitted_path) and not os.path.lexists(
            authorization_path
        ):
            require(
                not release_successor_present,
                "report repair release successor lacks authorization evidence",
            )
        worker_terminal_present = os.path.lexists(
            _worker_failure_terminal_path(submission)
        )
        if worker_terminal_present:
            require(
                os.path.lexists(calling_path)
                and os.path.lexists(_failure_path(submission))
                and os.path.lexists(_submitted_path(submission))
                and os.path.lexists(_authorization_path(submission))
                and not os.path.lexists(_submit_failure_terminal_path(submission))
                and not os.path.lexists(_release_denied_path(submission)),
                "report repair worker-failure lacks its mandatory predecessor chain",
            )
        require(
            not report_present or os.path.lexists(calling_path),
            "published repaired report lacks its submit-calling prefix",
        )
        if os.path.lexists(calling_path):
            recovery_calling, _recovery_calling_sha, recovery_calling_info = read_json(
                calling_path, "report repair submit calling"
            )
            require(
                stat.S_IMODE(recovery_calling_info.st_mode) == 0o444
                and recovery_calling_info.st_uid == os.getuid()
                and recovery_calling_info.st_nlink == 1
                and set(recovery_calling) == SUBMIT_CALLING_KEYS
                and recovery_calling.get("status")
                == "calling_held_report_repair_submission"
                and recovery_calling.get("campaign_id") == CAMPAIGN_ID
                and recovery_calling.get("submission_sha256")
                == submission_sha256
                and recovery_calling.get("attempt") == ATTEMPT
                and recovery_calling.get("repair_source_root")
                == str(_repair_source_root(submission)),
                "report repair submit calling identity differs",
            )
            recovery_source = _source_from_calling(recovery_calling)
            _validate_sealed_repair_source(
                _repair_source_root(submission), recovery_source
            )
            report_program = _repair_source_root(submission) / "report.py"
        elif os.path.lexists(_repair_source_root(submission)):
            _load_sealed_repair_source(_repair_source_root(submission))
            report_program = _repair_source_root(submission) / "report.py"
        else:
            report_program = PACKAGE_DIR / "report.py"

        contract, receipt, receipt_map, expected_reassembly = (
            _validate_original_submission(
                submission,
                submission_sha256,
                report_program=report_program,
            )
        )
        if report_present:
            require(
                not os.path.lexists(_submit_failure_terminal_path(submission))
                and not os.path.lexists(_release_denied_path(submission))
                and not os.path.lexists(_worker_failure_terminal_path(submission))
                and _cancel_generation_count(submission) == 0,
                "published repaired report conflicts with terminal cleanup state",
            )
            published = _validated_repaired_report_tree(
                submission, submission_sha256
            )
            require(published is not None, "published repaired report is absent")
            calling, calling_sha, _calling_info = read_json(
                calling_path, "report repair submit calling"
            )
            source = _source_from_calling(calling)
            failure, failure_sha, failure_info = read_json(
                _failure_path(submission),
                "report repair original failure evidence",
            )
            require(
                stat.S_IMODE(failure_info.st_mode) == 0o444
                and failure_info.st_uid == os.getuid()
                and failure_info.st_nlink == 1,
                "report repair original failure evidence identity differs",
            )
            _validate_failure_evidence(
                failure,
                submission_root=submission,
                submission_sha256=submission_sha256,
                contract=contract,
                receipt_map=receipt_map,
                expected_reassembly=expected_reassembly,
            )
            _validate_submit_calling(
                calling,
                submission_root=submission,
                submission_sha256=submission_sha256,
                failure_sha256=failure_sha,
                source=source,
                locks=locks,
            )
            submitted, submitted_sha, submitted_info = read_json(
                _submitted_path(submission), "report repair submitted evidence"
            )
            require(
                stat.S_IMODE(submitted_info.st_mode) == 0o444
                and submitted_info.st_uid == os.getuid()
                and submitted_info.st_nlink == 1,
                "report repair submitted evidence identity differs",
            )
            _validate_submitted(
                submitted,
                submission_sha256=submission_sha256,
                calling_sha256=calling_sha,
                calling=calling,
            )
            authorization, authorization_sha, authorization_info = read_json(
                _authorization_path(submission), "report repair authorization"
            )
            require(
                stat.S_IMODE(authorization_info.st_mode) == 0o444
                and authorization_info.st_uid == os.getuid()
                and authorization_info.st_nlink == 1,
                "report repair authorization identity differs",
            )
            _validate_authorization(
                authorization,
                submission_root=submission,
                submission_sha256=submission_sha256,
                contract=contract,
                receipt=receipt,
                receipt_map=receipt_map,
                expected_reassembly=expected_reassembly,
                source=source,
                failure_sha256=failure_sha,
                calling_sha256=calling_sha,
                submitted_sha256=submitted_sha,
            )
            released = _validate_existing_release(
                submission,
                submission_sha256,
                authorization_sha,
                str(authorization["repair_report_job_id"]),
                contract,
                locks,
            )
            require(released is not None, "published repaired report lacks release evidence")
            return {
                "schema_version": 1,
                "status": "report_repair_already_published",
                "commit": published,
                "scheduler_calls": 0,
            }
        publication_state = _publication_state(submission)
        require(publication_state["report_absent"] is True, "repair publication state differs")

        failure_path = _failure_path(submission)
        if not os.path.lexists(calling_path):
            no_calling_repair_names = sorted(
                name
                for name in repair_namespace_names
                if name != "REPORT_REPAIR_0001_ORIGINAL_FAILURE.json"
            )
            require(
                not no_calling_repair_names,
                "report repair successor/stop state exists without submit calling",
            )
            source_was_sealed = os.path.lexists(
                _repair_source_root(submission)
            )
            failure_was_sealed = os.path.lexists(failure_path)
            require(
                allow_initial_submission
                or source_was_sealed
                or failure_was_sealed,
                "report repair recovery cannot create the initial scheduler call",
            )
            require(
                not failure_was_sealed or source_was_sealed,
                "report repair failure evidence lacks its sealed source predecessor",
            )
            if source_was_sealed:
                source_root = _repair_source_root(submission)
                source = _load_sealed_repair_source(source_root)
            else:
                source = _verified_live_repair_source(root)
                source_root = _seal_repair_source_snapshot(submission, source)
            if failure_was_sealed:
                failure, failure_sha, failure_info = read_json(
                    failure_path, "report repair original failure evidence"
                )
                require(
                    stat.S_IMODE(failure_info.st_mode) == 0o444
                    and failure_info.st_uid == os.getuid()
                    and failure_info.st_nlink == 1,
                    "report repair original failure evidence identity differs",
                )
                _validate_failure_evidence(
                    failure,
                    submission_root=submission,
                    submission_sha256=submission_sha256,
                    contract=contract,
                    receipt_map=receipt_map,
                    expected_reassembly=expected_reassembly,
                )
            else:
                failure = _build_failure_evidence(
                    submission,
                    submission_sha256,
                    contract,
                    receipt_map,
                    expected_reassembly,
                    runner,
                    locks,
                    sleep=sleep,
                )
                _validate_failure_evidence(
                    failure,
                    submission_root=submission,
                    submission_sha256=submission_sha256,
                    contract=contract,
                    receipt_map=receipt_map,
                    expected_reassembly=expected_reassembly,
                )
                failure_sha = seal_json(failure_path, failure)

            # The failure artifact's census is historical evidence about the
            # original report failure, not authority for this new scheduler
            # mutation.  Every path that has not yet sealed SUBMIT CALLING
            # therefore takes a new owner-wide settled census immediately at
            # the mutation boundary.  In particular, this covers recovery
            # from a kill after ORIGINAL_FAILURE was sealed.
            scheduler_pre_submit_census = _scheduler_census(
                submission,
                contract,
                runner,
                locks,
                sleep=sleep,
            )
            require(
                scheduler_pre_submit_census["settled_rows"] == [],
                "report repair fresh pre-submit scheduler census is not empty",
            )

            # Rebind every local authority used below after the final census.
            # The report/cancel lock lives inside the submission tree, so its
            # named-inode revalidation also detects replacement of that tree
            # while the census was running.
            require(
                _directory(submission, "report repair submission root")
                == submission
                and _directory(
                    submission / "journal", "report repair journal directory"
                )
                == submission / "journal",
                "report repair state path binding differs",
            )
            _validate_sealed_repair_source(source_root, source)
            rebound_failure, rebound_failure_sha, rebound_failure_info = read_json(
                failure_path, "report repair original failure evidence"
            )
            require(
                stat.S_IMODE(rebound_failure_info.st_mode) == 0o444
                and rebound_failure_info.st_uid == os.getuid()
                and rebound_failure_info.st_nlink == 1
                and rebound_failure_sha == failure_sha
                and exact_json_equal(rebound_failure, failure),
                "report repair original failure evidence changed before submission",
            )
            _validate_failure_evidence(
                rebound_failure,
                submission_root=submission,
                submission_sha256=submission_sha256,
                contract=contract,
                receipt_map=receipt_map,
                expected_reassembly=expected_reassembly,
            )
            require(
                _publication_state(submission)["report_absent"] is True
                and _require_known_single_generation_namespace(submission)
                == ["REPORT_REPAIR_0001_ORIGINAL_FAILURE.json"],
                "report repair pre-submit append-only state differs",
            )
            environment = _scheduler_environment(
                str(contract["scheduler_control_plane_contract"]["slurm_conf"])
            )
            transaction, report_cancel = locks.bindings()
            command = _sbatch_command(
                source_root, submission, submission_sha256, source
            )
            calling = {
                "schema_version": 1,
                "status": "calling_held_report_repair_submission",
                "campaign_id": CAMPAIGN_ID,
                "submission_sha256": submission_sha256,
                "attempt": ATTEMPT,
                "original_failure_evidence": "journal/REPORT_REPAIR_0001_ORIGINAL_FAILURE.json",
                "original_failure_evidence_sha256": failure_sha,
                "repair_source_root": str(source_root),
                "repair_source_commit": source["repair_source_commit"],
                "repair_package_protocol_sha256": source[
                    "repair_package_protocol_sha256"
                ],
                "repair_source_files": dict(source["repair_source_files"]),
                "repair_source_files_sha256": source[
                    "repair_source_files_sha256"
                ],
                "scheduler_pre_submit_census": dict(
                    scheduler_pre_submit_census
                ),
                "scheduler_pre_submit_census_sha256": stable_hash(
                    scheduler_pre_submit_census
                ),
                "command": command,
                "scheduler_environment": environment,
                "transaction_lock": transaction,
                "report_cancel_lock": report_cancel,
                "called_at_utc": _utc_now(),
            }
            _validate_submit_calling(
                calling,
                submission_root=submission,
                submission_sha256=submission_sha256,
                failure_sha256=failure_sha,
                source=source,
                locks=locks,
            )
            calling_sha = seal_json(calling_path, calling)
            result, evidence = _run(
                runner, command, source_root, environment, locks
            )
            if result.returncode == 0:
                job_id = _parse_sbatch_job_id(result)
                submitted = {
                    "schema_version": 1,
                    "status": "held_report_repair_submitted",
                    "campaign_id": CAMPAIGN_ID,
                    "submission_sha256": submission_sha256,
                    "attempt": ATTEMPT,
                    "submit_calling_sha256": calling_sha,
                    "repair_report_job_id": job_id,
                    "submission_evidence": {
                        "mode": "direct_sbatch_response",
                        "raw": evidence,
                    },
                    "accepted_at_utc": _utc_now(),
                }
                seal_json(_submitted_path(submission), submitted)
            else:
                census = _scheduler_census(
                    submission, contract, runner, locks, sleep=sleep
                )
                rows = _repair_rows(census, submission_sha256)
                if not rows:
                    terminal = {
                        "schema_version": 1,
                        "status": "report_repair_terminal_submit_failure",
                        "campaign_id": CAMPAIGN_ID,
                        "submission_sha256": submission_sha256,
                        "attempt": ATTEMPT,
                        "submit_calling_sha256": calling_sha,
                        "scheduler_evidence": evidence,
                        "post_failure_census": census,
                        "publication_allowed": False,
                        "retry_allowed": False,
                        "sealed_at_utc": _utc_now(),
                    }
                    seal_json(
                        _submit_failure_terminal_path(submission),
                        terminal,
                    )
                    validated_terminal = _validated_submit_failure_terminal(
                        submission,
                        submission_sha256,
                        calling,
                        calling_sha,
                    )
                    assert validated_terminal is not None
                    return validated_terminal
                return _cleanup_repair_rows(
                    submission,
                    submission_sha256,
                    contract,
                    census,
                    "nonzero_sbatch_with_live_exact_repair_identity",
                    runner,
                    locks,
                    sleep=sleep,
                )
        else:
            calling, calling_sha, calling_info = read_json(
                calling_path, "report repair submit calling"
            )
            require(
                stat.S_IMODE(calling_info.st_mode) == 0o444
                and calling_info.st_uid == os.getuid()
                and calling_info.st_nlink == 1,
                "report repair submit calling identity differs",
            )
            source = _source_from_calling(calling)
            _validate_sealed_repair_source(
                _repair_source_root(submission), source
            )
            require(os.path.lexists(failure_path), "report repair failure evidence is absent")
            failure, failure_sha, failure_info = read_json(
                failure_path, "report repair original failure evidence"
            )
            require(
                stat.S_IMODE(failure_info.st_mode) == 0o444
                and failure_info.st_uid == os.getuid()
                and failure_info.st_nlink == 1,
                "report repair original failure evidence identity differs",
            )
            _validate_failure_evidence(
                failure,
                submission_root=submission,
                submission_sha256=submission_sha256,
                contract=contract,
                receipt_map=receipt_map,
                expected_reassembly=expected_reassembly,
            )
            _validate_submit_calling(
                calling,
                submission_root=submission,
                submission_sha256=submission_sha256,
                failure_sha256=failure_sha,
                source=source,
                locks=locks,
            )

        terminal_submit_failure = _validated_submit_failure_terminal(
            submission,
            submission_sha256,
            calling,
            calling_sha,
        )
        if terminal_submit_failure is not None:
            require(
                not os.path.lexists(_submitted_path(submission))
                and not os.path.lexists(_authorization_path(submission))
                and not os.path.lexists(_released_path(submission)),
                "terminal report repair submit failure has successor state",
            )
            if _cancel_generation_count(submission) == 0:
                delayed_census = _scheduler_census(
                    submission, contract, runner, locks, sleep=sleep
                )
                if _repair_rows(delayed_census, submission_sha256):
                    return _cleanup_repair_rows(
                        submission,
                        submission_sha256,
                        contract,
                        delayed_census,
                        "delayed_identity_after_terminal_submit_failure",
                        runner,
                        locks,
                        sleep=sleep,
                    )
                return terminal_submit_failure

        # A durable cleanup generation permanently wins over authorization, even
        # if the live package/authority later becomes available again.
        existing_cancel_generation = _cancel_generation_count(submission)
        if existing_cancel_generation:
            census = _scheduler_census(
                submission, contract, runner, locks, sleep=sleep
            )
            latest_terminal = _cancel_terminal_path(
                submission, existing_cancel_generation - 1
            )
            if os.path.lexists(latest_terminal) and not _repair_rows(
                census, submission_sha256
            ):
                terminal = _validated_cleanup_terminal(
                    submission,
                    submission_sha256,
                    existing_cancel_generation - 1,
                    contract,
                    locks,
                )
                require(
                    terminal.get("status")
                    == "report_repair_terminal_cleanup_complete",
                    "report repair cleanup completion differs",
                )
                return terminal
            return _cleanup_repair_rows(
                submission,
                submission_sha256,
                contract,
                census,
                "residual_exact_repair_jobs_after_cleanup",
                runner,
                locks,
                sleep=sleep,
            )

        calling, calling_sha, _info = read_json(
            calling_path, "report repair submit calling"
        )
        source = _source_from_calling(calling)
        failure, failure_sha, _failure_info = read_json(
            failure_path, "report repair original failure evidence"
        )
        if not os.path.lexists(submitted_path):
            census = _scheduler_census(
                submission, contract, runner, locks, sleep=sleep
            )
            rows = _repair_rows(census, submission_sha256)
            if not rows:
                return {
                    "schema_version": 1,
                    "status": "report_repair_lost_submit_response_awaiting_identity",
                    "attempt": ATTEMPT,
                    "scheduler_calls": 3,
                }
            exact_held_lost_response = (
                len(rows) == 1
                and len(census["settled_rows"]) == 1
                and rows[0]["job_id"] not in HISTORICAL_JOB_IDS
                and rows[0]["state"] == "PENDING"
                and rows[0]["reason"] in {"JobHeldUser", "JobHeldAdmin"}
            )
            if not exact_held_lost_response:
                return _cleanup_repair_rows(
                    submission,
                    submission_sha256,
                    contract,
                    census,
                    (
                        "historical_numeric_id_recycled"
                        if any(
                            row["job_id"] in HISTORICAL_JOB_IDS for row in rows
                        )
                        else "ambiguous_lost_submit_response"
                    ),
                    runner,
                    locks,
                    sleep=sleep,
                )
            job_id = rows[0]["job_id"]
            submitted = {
                "schema_version": 1,
                "status": "held_report_repair_submitted",
                "campaign_id": CAMPAIGN_ID,
                "submission_sha256": submission_sha256,
                "attempt": ATTEMPT,
                "submit_calling_sha256": calling_sha,
                "repair_report_job_id": job_id,
                "submission_evidence": {
                    "mode": "lost_response_census_adoption",
                    "census": census,
                    "census_sha256": stable_hash(census),
                },
                "accepted_at_utc": _utc_now(),
            }
            seal_json(submitted_path, submitted)

        submitted, submitted_sha, submitted_info = read_json(
            submitted_path, "report repair submitted evidence"
        )
        require(
            stat.S_IMODE(submitted_info.st_mode) == 0o444
            and submitted_info.st_uid == os.getuid()
            and submitted_info.st_nlink == 1,
            "report repair submitted evidence identity differs",
        )
        _validate_submitted(
            submitted,
            submission_sha256=submission_sha256,
            calling_sha256=calling_sha,
            calling=calling,
        )
        job_id = str(submitted["repair_report_job_id"])

        if not os.path.lexists(authorization_path):
            census = _scheduler_census(
                submission, contract, runner, locks, sleep=sleep
            )
            rows = _repair_rows(census, submission_sha256)
            if (
                job_id in HISTORICAL_JOB_IDS
                or len(rows) != 1
                or rows[0]["job_id"] != job_id
                or rows[0]["state"] != "PENDING"
                or rows[0]["reason"] not in {"JobHeldUser", "JobHeldAdmin"}
                or len(census["settled_rows"]) != 1
            ):
                if rows:
                    return _cleanup_repair_rows(
                        submission,
                        submission_sha256,
                        contract,
                        census,
                        (
                            "historical_numeric_id_recycled"
                            if job_id in HISTORICAL_JOB_IDS
                            else "repair_scheduler_authority_ambiguous"
                        ),
                        runner,
                        locks,
                        sleep=sleep,
                    )
                return {
                    "schema_version": 1,
                    "status": "report_repair_submitted_identity_not_yet_visible",
                    "attempt": ATTEMPT,
                    "repair_report_job_id": job_id,
                    "scheduler_calls": 3,
                }
            authorization = _authorization_value(
                submission_root=submission,
                submission_sha256=submission_sha256,
                contract=contract,
                receipt=receipt,
                receipt_map=receipt_map,
                expected_reassembly=expected_reassembly,
                source=source,
                failure_sha256=failure_sha,
                calling_sha256=calling_sha,
                submitted_sha256=submitted_sha,
                job_id=job_id,
                census=census,
            )
            authorization_sha = seal_json(authorization_path, authorization)
        else:
            authorization, authorization_sha, authorization_info = read_json(
                authorization_path, "report repair authorization"
            )
            require(
                stat.S_IMODE(authorization_info.st_mode) == 0o444
                and authorization_info.st_uid == os.getuid()
                and authorization_info.st_nlink == 1,
                "report repair authorization identity differs",
            )
            _validate_authorization(
                authorization,
                submission_root=submission,
                submission_sha256=submission_sha256,
                contract=contract,
                receipt=receipt,
                receipt_map=receipt_map,
                expected_reassembly=expected_reassembly,
                source=source,
                failure_sha256=failure_sha,
                calling_sha256=calling_sha,
                submitted_sha256=submitted_sha,
            )

        released = _validate_existing_release(
            submission,
            submission_sha256,
            authorization_sha,
            job_id,
            contract,
            locks,
        )
        if released is None:
            released = _release_authorized_job(
                submission,
                submission_sha256,
                contract,
                authorization,
                authorization_sha,
                runner,
                locks,
                sleep=sleep,
            )
        else:
            released = _reconcile_released_worker(
                submission,
                submission_sha256,
                contract,
                authorization_sha,
                released,
                runner,
                locks,
                sleep=sleep,
            )
        if released.get("status") != "report_repair_released":
            return released
        return {
            "schema_version": 1,
            "status": "report_repair_released_for_publication",
            "attempt": ATTEMPT,
            "repair_report_job_id": job_id,
            "authorization_sha256": authorization_sha,
            "release": released,
        }


def describe() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "one_generation_report_repair_source_available",
        "campaign_id": CAMPAIGN_ID,
        "attempt": ATTEMPT,
        "submission_root": str(CANONICAL_PRODUCTION_SUBMISSION_ROOT),
        "submission_sha256": EXPECTED_SUBMISSION_SHA256,
        "original_report_job_id": EXPECTED_ORIGINAL_REPORT_JOB_ID,
        "expected_report_bundle_sha256": EXPECTED_BUNDLE_SHA256,
        "expected_gate_sha256": EXPECTED_GATE_SHA256,
        "scheduler_calls": 0,
        "writes_performed": 0,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--describe", action="store_true")
    actions.add_argument("--test-only", action="store_true")
    actions.add_argument("--submit-real-report-repair", action="store_true")
    actions.add_argument("--recover-or-cancel-report-repair", action="store_true")
    parser.add_argument("--repo-root", default=str(REPOSITORY_ROOT))
    parser.add_argument(
        "--submission-root",
        default=str(CANONICAL_PRODUCTION_SUBMISSION_ROOT),
    )
    parser.add_argument("--submission-sha256", default=EXPECTED_SUBMISSION_SHA256)
    parser.add_argument("--confirmation")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        repo_root = _canonical_cli_path(
            args.repo_root, "report repair CLI repository root"
        )
        submission_root = _canonical_cli_path(
            args.submission_root, "report repair CLI submission root"
        )
        mutating = args.submit_real_report_repair or args.recover_or_cancel_report_repair
        if mutating:
            require(args.confirmation == CONFIRMATION, "report repair confirmation differs")
            os.umask(0o077)
            result = execute_report_repair(
                repo_root,
                submission_root,
                args.submission_sha256,
                allow_initial_submission=args.submit_real_report_repair,
            )
        elif args.test_only:
            source = _verified_live_repair_source(repo_root)
            result = {
                **describe(),
                "status": "read_only_report_repair_source_verified",
                "repair_source_commit": source["repair_source_commit"],
                "repair_package_protocol_sha256": source[
                    "repair_package_protocol_sha256"
                ],
                "repair_source_files_sha256": source[
                    "repair_source_files_sha256"
                ],
            }
        else:
            result = describe()
    except (RepairError, OSError, ValueError) as exc:
        print(f"Exp23 report repair engineering error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
