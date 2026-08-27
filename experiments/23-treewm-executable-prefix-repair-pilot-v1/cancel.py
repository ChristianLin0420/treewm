#!/usr/bin/env python3
"""Read-only cancellation plan, or explicitly cancel one sealed Exp23 receipt."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
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


class CancellationError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CancellationError(message)


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_pairs,
                parse_constant=lambda token: (_ for _ in ()).throw(
                    CancellationError(f"non-finite JSON value in {path}: {token}")
                ),
            )
    except (OSError, json.JSONDecodeError) as exc:
        raise CancellationError(f"cannot read {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _regular(path: Path, label: str) -> None:
    try:
        info = path.lstat()
    except OSError as exc:
        raise CancellationError(f"{label} is unavailable: {exc}") from exc
    require(stat.S_ISREG(info.st_mode), f"{label} is not a regular nonsymlink file")


def _directory(path: Path, label: str) -> None:
    lexical = path.absolute()
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
    if not path.parent.exists():
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
    values: dict[str, str] = {}
    for line in pyvenv.read_text(encoding="utf-8").splitlines():
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
        "lexical_symlink_target": os.readlink(expected) if expected.is_symlink() else None,
        "resolved_executable": str(target),
        "resolved_executable_sha256": file_sha256(target),
        "resolved_executable_size": target.stat().st_size,
        "base_executable": str(Path(str(getattr(sys, "_base_executable", target)))),
        "venv_site_packages": str(sites[0]),
        "base_site_packages": str(sites[1]),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
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


def scheduler_environment() -> dict[str, str]:
    """Exact local-cluster environment shared with submit/recovery."""

    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
    }


def validate_receipt(submission_root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _directory(submission_root, "submission root")
    receipt_path = submission_root / "SUBMISSION_RECEIPT.json"
    contract_path = submission_root / "SUBMISSION_CONTRACT.json"
    _regular(receipt_path, "submission receipt")
    _regular(contract_path, "submission contract")
    require(stat.S_IMODE(receipt_path.lstat().st_mode) == 0o444, "submission receipt mode differs")
    require(stat.S_IMODE(contract_path.lstat().st_mode) == 0o444, "submission contract mode differs")
    receipt = read_json(receipt_path)
    require(set(receipt) == RECEIPT_KEYS, "submission receipt schema differs")
    require(receipt["schema_version"] == 1 and receipt["status"] == "submitted", "submission receipt is not committed")
    require(receipt["campaign_id"] == "treewm-executable-prefix-repair-pilot-v1", "submission receipt campaign differs")
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
    snapshot = snapshot_root.resolve(strict=True)
    require(snapshot.is_relative_to(submission_root.resolve(strict=True)), "snapshot root escapes submission root")
    require(str(snapshot) == receipt["snapshot_root"], "snapshot root is not canonical")
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
    require(snapshot.lstat().st_mode & 0o222 == 0, "snapshot root is writable")
    actual_files: set[str] = set()
    actual_dirs: set[str] = set()
    for path in snapshot.rglob("*"):
        info = path.lstat()
        relative = str(path.relative_to(snapshot))
        require(not stat.S_ISLNK(info.st_mode), f"snapshot contains symlink: {relative}")
        if stat.S_ISREG(info.st_mode):
            actual_files.add(relative)
            require(relative in normalized, f"snapshot contains unclaimed file: {relative}")
            require(info.st_mode & 0o222 == 0, f"snapshot file is writable: {relative}")
            require(file_sha256(path) == normalized[relative], f"snapshot hash differs: {relative}")
        elif stat.S_ISDIR(info.st_mode):
            actual_dirs.add(relative)
            require(info.st_mode & 0o222 == 0, f"snapshot directory is writable: {relative}")
        else:
            raise CancellationError(f"snapshot contains special entry: {relative}")
    require(actual_files == set(normalized), "snapshot file coverage differs")
    require(actual_dirs == expected_dirs, "snapshot directory coverage differs")
    manifest_path = snapshot / PACKAGE_RELATIVE / "manifest.json"
    _regular(manifest_path, "snapshot manifest")
    require(normalized.get(str(PACKAGE_RELATIVE / "manifest.json")) == file_sha256(manifest_path), "snapshot manifest inventory binding differs")
    manifest = read_json(manifest_path)
    require(contract.get("manifest_sha256") == stable_hash(manifest), "snapshot manifest contract hash differs")
    protocol_path = snapshot / PACKAGE_RELATIVE / "protocol.sha256"
    _regular(protocol_path, "snapshot protocol lock")
    protocol = protocol_path.read_text(encoding="ascii").strip()
    require(SHA256.fullmatch(protocol) is not None and protocol == contract.get("package_protocol_sha256"), "snapshot protocol contract differs")
    interpreter = activate_isolated_runtime(manifest)
    require(contract.get("orchestration_interpreter") == interpreter, "cancellation interpreter contract differs")
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
    if os.path.lexists(path):
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
    receipt, _contract, manifest = validate_receipt(submission_root)
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
    environment = scheduler_environment()
    command = [str(scancel), *ids]
    token = f"{time.time_ns()}-{os.getpid()}"
    intent = {
        "schema_version": 1,
        "status": "exact_cancel_call_intent",
        "campaign_id": receipt["campaign_id"],
        "submission_sha256": receipt["submission_sha256"],
        "call_token": token,
        "job_ids": ids,
        "command": command,
    }
    evidence_root = submission_root / "cancellation"
    seal_json(evidence_root / f"CANCEL_CALL.{token}.json", intent)
    completed = subprocess.run(
        command,
        cwd=snapshot_root,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
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
        "scheduler_calls": 1,
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
