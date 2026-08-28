#!/usr/bin/env python3
"""Tiny opt-in Launch8 two-wave GPU topology canary worker.

This file intentionally contains no scheduler client.  The login-side canary
controller predeclares both waves and the reporter before releasing wave zero.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


CAMPAIGN_ID = "treewm-executable-prefix-repair-pilot-v1-launch8"
AUTH_NAME = "CANARY_AUTHORIZATION.json"
RECEIPT_NAME = "CANARY_SUBMISSION_RECEIPT.json"
READY_TO_RELEASE_NAME = "CANARY_READY_TO_RELEASE.json"
READY_NAME = "WAVE0_READY.json"
COMPLETE_NAME = "WAVE1_COMPLETE.json"
REPORT_NAME = "CANARY_REPORT.json"
SHA256 = frozenset("0123456789abcdef")
PINNED_PYTHON = Path(
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/envs/treewm-formal-py311/bin/python"
)
_BOUND_SITES: tuple[Path, Path] | None = None


class CanaryError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise CanaryError(message)


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


def _is_job_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value[0] in "123456789"
        and all(character in "0123456789" for character in value)
    )


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA256


def _bootstrap_runtime() -> tuple[Path, Path]:
    """Bind the two exact package roots under pinned isolated Python."""

    global _BOUND_SITES
    if _BOUND_SITES is not None:
        return _BOUND_SITES
    require(
        bool(sys.flags.isolated)
        and bool(sys.flags.no_site)
        and bool(sys.flags.dont_write_bytecode)
        and bool(sys.dont_write_bytecode)
        and bool(sys.flags.safe_path),
        "canary worker requires pinned Python -I -S -B",
    )
    require(
        Path(sys.executable).absolute() == PINNED_PYTHON,
        "canary worker lexical interpreter differs",
    )
    executable = PINNED_PYTHON.resolve(strict=True)
    require(stat.S_ISREG(executable.lstat().st_mode), "canary resolved interpreter differs")
    pyvenv = PINNED_PYTHON.parent.parent / "pyvenv.cfg"
    require(stat.S_ISREG(pyvenv.lstat().st_mode) and not pyvenv.is_symlink(), "canary pyvenv differs")
    values: dict[str, str] = {}
    for line in pyvenv.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    require("home" in values and Path(values["home"]).is_absolute(), "canary pyvenv home differs")
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    sites = (
        PINNED_PYTHON.parent.parent / "lib" / version / "site-packages",
        Path(values["home"]).parent / "lib" / version / "site-packages",
    )
    require(
        not any("site-packages" in item for item in sys.path),
        "canary package roots were ambiently preloaded",
    )
    for site in sites:
        info = site.lstat()
        require(stat.S_ISDIR(info.st_mode) and not site.is_symlink(), "canary package root differs")
        sys.path.append(str(site))
    _BOUND_SITES = sites
    return sites


def _verified_torch():
    sites = _bootstrap_runtime()
    require("torch" not in sys.modules, "torch was imported before canary provenance validation")
    torch = importlib.import_module("torch")
    source = Path(str(torch.__file__)).absolute()
    require(
        source.is_file()
        and not source.is_symlink()
        and any(source.is_relative_to(site) for site in sites),
        "canary torch imported outside bound package roots",
    )
    return torch


def sha256_file(path: Path) -> str:
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode), f"canary artifact is not a regular file: {path}")
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        opened = os.fstat(descriptor)
        require(
            (opened.st_dev, opened.st_ino, opened.st_size)
            == (info.st_dev, info.st_ino, info.st_size),
            f"canary artifact raced: {path}",
        )
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        require(os.fstat(descriptor).st_size == opened.st_size, f"canary artifact changed: {path}")
    finally:
        os.close(descriptor)
    return digest.hexdigest()


def read_json(path: Path, label: str) -> dict[str, Any]:
    info = path.lstat()
    require(
        stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o444,
        f"{label} is not an immutable regular file",
    )
    try:
        value = json.loads(path.read_text(encoding="ascii"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise CanaryError(f"cannot read {label}: {exc}") from exc
    require(isinstance(value, dict), f"{label} is not a JSON object")
    return value


def seal_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = (canonical_json(dict(value)) + "\n").encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(
        path,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)
    return hashlib.sha256(payload).hexdigest()


def _one_line_field(stdout: object, name: str, label: str) -> str:
    require(isinstance(stdout, str), f"{label} stdout differs")
    lines = [line for line in stdout.splitlines() if line.strip()]
    require(len(lines) == 1, f"{label} stdout is ambiguous")
    prefix = f"{name}="
    values = [
        token[len(prefix) :]
        for token in lines[0].split()
        if token.startswith(prefix)
    ]
    require(len(values) == 1 and bool(values[0]), f"{label} {name} differs")
    return values[0]


def _scheduler_observation_is_type_strict(value: object) -> bool:
    """Scheduler observations contain no booleans/floats or coerced integers."""

    def valid(item: object) -> bool:
        if isinstance(item, Mapping):
            return all(
                isinstance(key, str) and valid(child)
                for key, child in item.items()
            )
        if isinstance(item, list):
            return all(valid(child) for child in item)
        if isinstance(item, bool) or isinstance(item, float):
            return False
        return item is None or isinstance(item, (str, int))

    return (
        isinstance(value, Mapping)
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and valid(value)
    )


def _scheduler_observation_matches(value: object, authorization: Mapping[str, Any]) -> bool:
    expected = authorization.get("scheduler_control_plane")
    return (
        isinstance(value, Mapping)
        and isinstance(expected, Mapping)
        and _scheduler_observation_is_type_strict(expected)
        and _scheduler_observation_is_type_strict(value)
        and exact_json_equal(value, expected)
        and stable_hash(expected)
        == authorization["scheduler_control_plane_sha256"]
        and stable_hash(value) == authorization["scheduler_control_plane_sha256"]
    )


def _expected_submission_commands(
    root: Path, authorization: Mapping[str, Any]
) -> dict[str, list[str]]:
    jobs = authorization["job_ids"]
    names = authorization["job_names"]
    clients = authorization["scheduler_executables"]
    comment = authorization["scheduler_comment"]
    source = root / "source"
    return {
        "wave0": [
            clients["submit"],
            "--parsable",
            "--export=NONE",
            "--hold",
            f"--job-name={names['wave0']}",
            f"--comment={comment}",
            f"--output={root / 'logs/wave0_%j.out'}",
            str(source / "canary_gpu.slurm"),
            str(source / "canary_worker.py"),
            str(root),
            "wave0",
        ],
        "wave1": [
            clients["submit"],
            "--parsable",
            "--export=NONE",
            f"--dependency=afterok:{jobs['wave0']}",
            "--kill-on-invalid-dep=yes",
            f"--job-name={names['wave1']}",
            f"--comment={comment}",
            f"--output={root / 'logs/wave1_%j.out'}",
            str(source / "canary_gpu.slurm"),
            str(source / "canary_worker.py"),
            str(root),
            "wave1",
        ],
        "report": [
            clients["submit"],
            "--parsable",
            "--export=NONE",
            f"--dependency=afterok:{jobs['wave1']}",
            "--kill-on-invalid-dep=yes",
            f"--job-name={names['report']}",
            f"--comment={comment}",
            f"--output={root / 'logs/report_%j.out'}",
            str(source / "canary_report.slurm"),
            str(source / "canary_worker.py"),
            str(root),
        ],
    }


def _validate_submission_record(
    value: object,
    *,
    role: str,
    root: Path,
    authorization: Mapping[str, Any],
) -> None:
    require(isinstance(value, Mapping), f"canary {role} submission record differs")
    require(
        set(value)
        == {
            "command",
            "returncode",
            "stdout",
            "stderr",
            "reconciled_job_ids",
            "scheduler_control_plane",
            "calling_sha256",
        },
        f"canary {role} submission record fields differ",
    )
    job_id = authorization["job_ids"][role]
    response = value.get("stdout")
    require(
        value.get("command") == _expected_submission_commands(root, authorization)[role]
        and type(value.get("returncode")) is int
        and value.get("returncode") == 0
        and isinstance(response, str)
        and isinstance(value.get("stderr"), str)
        and len(response) <= 1024 * 1024
        and len(value["stderr"]) <= 1024 * 1024
        and value.get("reconciled_job_ids") == [job_id]
        and _is_sha256(value.get("calling_sha256"))
        and _scheduler_observation_matches(
            value.get("scheduler_control_plane"), authorization
        ),
        f"canary {role} submission record identity differs",
    )
    parsed = response.strip().split(";", 1)[0]
    require(
        bool(response.strip()) and parsed == job_id,
        f"canary {role} accepted response differs",
    )
    calling_names = {
        "wave0": "CANARY_WAVE0_CALLING.json",
        "wave1": "CANARY_WAVE1_CALLING.json",
        "report": "CANARY_REPORT_CALLING.json",
    }
    calling_path = root / calling_names[role]
    require(
        sha256_file(calling_path) == value["calling_sha256"],
        f"canary {role} calling record hash differs",
    )
    calling = read_json(calling_path, f"canary {role} calling record")
    require(
        exact_json_equal(
            calling,
            {
            "schema_version": 1,
            "status": "canary_scheduler_calling",
            "campaign_id": CAMPAIGN_ID,
            "state_root": str(root),
            "canary_token": authorization["canary_token"],
            "role": role,
            "job_name": authorization["job_names"][role],
            "scheduler_comment": authorization["scheduler_comment"],
            "command": _expected_submission_commands(root, authorization)[role],
            "controller_lock": authorization["controller_lock"],
            },
        ),
        f"canary {role} calling record differs",
    )


def _validate_scheduler_evidence(
    value: object,
    *,
    role: str,
    root: Path,
    authorization: Mapping[str, Any],
) -> None:
    require(isinstance(value, Mapping), f"canary {role} scheduler evidence differs")
    jobs = authorization["job_ids"]
    names = authorization["job_names"]
    control = authorization["scheduler_executables"]["control"]
    comment = authorization["scheduler_comment"]
    job_id = jobs[role]
    common = {
        "command",
        "returncode",
        "stdout",
        "stderr",
        "scheduler_control_plane",
    }
    if role == "wave0":
        require(
            set(value) == common | {"state", "reason"},
            "canary held-wave scheduler evidence fields differ",
        )
    else:
        require(
            set(value)
            == common
            | {"dependency", "role", "kill_on_invalid_dependency"},
            f"canary {role} dependency evidence fields differ",
        )
    require(
        value.get("command") == [control, "show", "job", job_id, "--oneliner"]
        and type(value.get("returncode")) is int
        and value.get("returncode") == 0
        and isinstance(value.get("stdout"), str)
        and isinstance(value.get("stderr"), str)
        and len(value["stdout"]) <= 1024 * 1024
        and len(value["stderr"]) <= 1024 * 1024
        and _scheduler_observation_matches(
            value.get("scheduler_control_plane"), authorization
        )
        and _one_line_field(value["stdout"], "JobId", role) == job_id
        and _one_line_field(value["stdout"], "JobName", role) == names[role]
        and _one_line_field(value["stdout"], "Comment", role) == comment
        and _one_line_field(value["stdout"], "JobState", role) == "PENDING",
        f"canary {role} scheduler identity differs",
    )
    if role == "wave0":
        require(
            value.get("state") == "PENDING"
            and value.get("reason") == "JobHeldUser"
            and _one_line_field(value["stdout"], "Reason", role)
            == "JobHeldUser",
            "canary wave zero was not accepted held",
        )
        return
    predecessor = jobs["wave0" if role == "wave1" else "wave1"]
    dependency = f"afterok:{predecessor}_*(unfulfilled)"
    require(
        value.get("role") == role
        and value.get("dependency") == dependency
        and _one_line_field(value["stdout"], "Dependency", role) == dependency
        and value.get("kill_on_invalid_dependency") == "Yes"
        and _one_line_field(
            value["stdout"], "KillOInInvalidDependent", role
        )
        == "Yes",
        f"canary {role} accepted dependency differs",
    )


def _validate_release_record(
    root: Path, authorization: Mapping[str, Any]
) -> tuple[dict[str, Any], str]:
    authorization_sha256 = sha256_file(root / AUTH_NAME)
    receipt_sha256 = sha256_file(root / RECEIPT_NAME)
    ready_sha256 = sha256_file(root / READY_TO_RELEASE_NAME)
    record = read_json(root / "CANARY_WAVE0_RELEASED.json", "canary wave-zero release")
    require(
        set(record)
        == {
            "schema_version",
            "status",
            "campaign_id",
            "authorization_sha256",
            "receipt_sha256",
            "ready_to_release_sha256",
            "calling_sha256",
            "wave0_job_id",
            "wave0_release",
        }
        and type(record.get("schema_version")) is int
        and record.get("schema_version") == 1
        and record.get("status") == "canary_wave0_released"
        and record.get("campaign_id") == CAMPAIGN_ID
        and record.get("authorization_sha256") == authorization_sha256
        and record.get("receipt_sha256") == receipt_sha256
        and record.get("ready_to_release_sha256") == ready_sha256
        and _is_sha256(record.get("calling_sha256"))
        and record.get("wave0_job_id") == authorization["job_ids"]["wave0"],
        "canary release record identity differs",
    )
    release = record.get("wave0_release")
    calling_path = root / "CANARY_WAVE0_RELEASE_CALLING.json"
    require(
        sha256_file(calling_path) == record["calling_sha256"],
        "canary release calling hash differs",
    )
    require(
        exact_json_equal(
            read_json(calling_path, "canary release calling record"),
            {
            "schema_version": 1,
            "status": "canary_scheduler_calling",
            "campaign_id": CAMPAIGN_ID,
            "state_root": str(root),
            "canary_token": authorization["canary_token"],
            "role": "wave0_release",
            "job_name": authorization["job_names"]["wave0"],
            "scheduler_comment": authorization["scheduler_comment"],
            "command": [
                authorization["scheduler_executables"]["control"],
                "release",
                authorization["job_ids"]["wave0"],
            ],
            "controller_lock": authorization["controller_lock"],
            "authorization_sha256": authorization_sha256,
            "receipt_sha256": receipt_sha256,
            },
        ),
        "canary release calling record differs",
    )
    require(isinstance(release, Mapping), "canary release evidence differs")
    require(
        set(release)
        == {
            "release_command",
            "release_returncode",
            "release_stdout",
            "release_stderr",
            "show_command",
            "show_returncode",
            "show_stdout",
            "show_stderr",
            "state",
            "reason",
            "release_scheduler_control_plane",
            "show_scheduler_control_plane",
        },
        "canary release evidence fields differ",
    )
    control = authorization["scheduler_executables"]["control"]
    job_id = authorization["job_ids"]["wave0"]
    stdout = release.get("show_stdout")
    require(
        release.get("release_command") == [control, "release", job_id]
        and type(release.get("release_returncode")) is int
        and release.get("release_returncode") == 0
        and isinstance(release.get("release_stdout"), str)
        and isinstance(release.get("release_stderr"), str)
        and release.get("show_command")
        == [control, "show", "job", job_id, "--oneliner"]
        and type(release.get("show_returncode")) is int
        and release.get("show_returncode") == 0
        and isinstance(stdout, str)
        and isinstance(release.get("show_stderr"), str)
        and all(
            len(str(release[key])) <= 1024 * 1024
            for key in (
                "release_stdout",
                "release_stderr",
                "show_stdout",
                "show_stderr",
            )
        )
        and _scheduler_observation_matches(
            release.get("release_scheduler_control_plane"), authorization
        )
        and _scheduler_observation_matches(
            release.get("show_scheduler_control_plane"), authorization
        )
        and _one_line_field(stdout, "JobId", "released wave zero") == job_id
        and _one_line_field(stdout, "JobName", "released wave zero")
        == authorization["job_names"]["wave0"]
        and _one_line_field(stdout, "Comment", "released wave zero")
        == authorization["scheduler_comment"],
        "canary released wave-zero scheduler identity differs",
    )
    state = _one_line_field(stdout, "JobState", "released wave zero")
    reason = _one_line_field(stdout, "Reason", "released wave zero")
    require(
        release.get("state") == state
        and state in {"PENDING", "RUNNING", "COMPLETING", "COMPLETED"}
        and release.get("reason") == reason
        and reason != "JobHeldUser",
        "canary wave zero remained held or has invalid release state",
    )
    return record, sha256_file(root / "CANARY_WAVE0_RELEASED.json")


def _authorization(state_root: Path) -> dict[str, Any]:
    root = state_root.absolute()
    require(root == state_root and root.is_dir() and not root.is_symlink(), "canary state root differs")
    executing_worker = Path(__file__).absolute()
    expected_worker = root / "source" / "canary_worker.py"
    require(
        executing_worker == expected_worker
        and stat.S_ISREG(executing_worker.lstat().st_mode)
        and not executing_worker.is_symlink(),
        "canary worker is not executing from the authorized state snapshot",
    )
    value = read_json(root / AUTH_NAME, "canary authorization")
    require(
        set(value)
        == {
            "schema_version",
            "status",
            "campaign_id",
            "canary_token",
            "state_root",
            "controller_identity_sha256",
            "controller_lock",
            "worker_sha256",
            "source_sha256",
            "package_protocol_sha256",
            "job_ids",
            "job_names",
            "dependencies",
            "scheduler_comment",
            "scheduler_executables",
            "scheduler_control_plane",
            "scheduler_control_plane_sha256",
            "accepted_submission_records_sha256",
            "accepted_scheduler_evidence_sha256",
            "within_wave_requeue",
        },
        "canary authorization fields differ",
    )
    jobs = value.get("job_ids")
    token = value.get("canary_token")
    expected_names = (
        {
            "wave0": f"exp23-launch8-canary-{token}-wave0",
            "wave1": f"exp23-launch8-canary-{token}-wave1",
            "report": f"exp23-launch8-canary-{token}-report",
        }
        if isinstance(token, str)
        else {}
    )
    clients = value.get("scheduler_executables")
    controller_lock = value.get("controller_lock")
    require(
        type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("status") == "authorized_two_wave_gpu_canary"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("state_root") == str(root)
        and isinstance(token, str)
        and len(token) == 16
        and set(token) <= SHA256
        and _is_sha256(value.get("worker_sha256"))
        and isinstance(value.get("source_sha256"), dict)
        and set(value["source_sha256"])
        == {
            "two_wave_canary.py",
            "canary_worker.py",
            "canary_gpu.slurm",
            "canary_report.slurm",
        }
        and all(_is_sha256(digest) for digest in value["source_sha256"].values())
        and value["source_sha256"]["canary_worker.py"] == value["worker_sha256"]
        and _is_sha256(value.get("package_protocol_sha256"))
        and _is_sha256(value.get("controller_identity_sha256"))
        and _scheduler_observation_is_type_strict(
            value.get("scheduler_control_plane")
        )
        and _is_sha256(value.get("scheduler_control_plane_sha256"))
        and stable_hash(value["scheduler_control_plane"])
        == value["scheduler_control_plane_sha256"]
        and _is_sha256(value.get("accepted_submission_records_sha256"))
        and _is_sha256(value.get("accepted_scheduler_evidence_sha256"))
        and value.get("within_wave_requeue") is False
        and isinstance(jobs, dict)
        and set(jobs) == {"wave0", "wave1", "report"}
        and all(_is_job_id(item) for item in jobs.values())
        and len(set(jobs.values())) == 3,
        "canary authorization identity differs",
    )
    require(
        exact_json_equal(value.get("job_names"), expected_names)
        and value.get("scheduler_comment") == f"treewm-exp23-canary:{token}"
        and isinstance(clients, dict)
        and set(clients) == {"submit", "control"}
        and all(
            isinstance(item, str)
            and Path(item).is_absolute()
            and Path(item).absolute() == Path(item)
            for item in clients.values()
        ),
        "canary scheduler authority differs",
    )
    require(
        isinstance(controller_lock, dict)
        and set(controller_lock) == {"path", "device", "inode", "uid", "mode"}
        and controller_lock.get("path") == str(root / ".CANARY_CONTROLLER.lock")
        and type(controller_lock.get("uid")) is int
        and controller_lock.get("uid") == os.getuid()
        and type(controller_lock.get("mode")) is int
        and controller_lock.get("mode") == 0o600
        and all(
            type(controller_lock.get(key)) is int
            and controller_lock[key] >= 0
            for key in ("device", "inode")
        ),
        "canary controller-lock authority differs",
    )
    require(
        exact_json_equal(
            value.get("dependencies"),
            {
            "wave0": "none",
            "wave1": f"afterok:{jobs['wave0']}",
            "report": f"afterok:{jobs['wave1']}",
            },
        ),
        "canary authorization graph differs",
    )
    source_root = Path(__file__).absolute().parent
    require(
        all(
            sha256_file(source_root / name) == digest
            for name, digest in value["source_sha256"].items()
        ),
        "canary source bytes differ",
    )
    require(
        sha256_file(executing_worker) == value["worker_sha256"],
        "executing canary worker bytes differ",
    )
    authorization_sha256 = sha256_file(root / AUTH_NAME)
    identity_path = root / "CANARY_CONTROLLER_IDENTITY.json"
    require(
        sha256_file(identity_path) == value["controller_identity_sha256"],
        "canary controller identity hash differs",
    )
    identity = read_json(identity_path, "canary controller identity")
    require(
        exact_json_equal(
            identity,
            {
            "schema_version": 1,
            "status": "canary_controller_claimed",
            "campaign_id": CAMPAIGN_ID,
            "scientific": False,
            "state_root": str(root),
            "canary_token": value["canary_token"],
            "job_names": value["job_names"],
            "scheduler_comment": value["scheduler_comment"],
            "controller_lock": controller_lock,
            "source_sha256": value["source_sha256"],
            "package_protocol_sha256": value["package_protocol_sha256"],
            "scheduler_control_plane": value["scheduler_control_plane"],
            "scheduler_control_plane_sha256": value[
                "scheduler_control_plane_sha256"
            ],
            },
        ),
        "canary controller identity differs",
    )
    receipt = read_json(root / RECEIPT_NAME, "canary submission receipt")
    require(
        set(receipt)
        == {
            "schema_version",
            "status",
            "campaign_id",
            "scientific",
            "state_root",
            "canary_token",
            "controller_identity_sha256",
            "controller_lock",
            "authorization_sha256",
            "job_ids",
            "job_names",
            "dependencies",
            "scheduler_comment",
            "scheduler_executables",
            "scheduler_control_plane",
            "scheduler_control_plane_sha256",
            "source_sha256",
            "package_protocol_sha256",
            "accepted_submission_records_sha256",
            "accepted_scheduler_evidence_sha256",
            "wave0_accepted_hold",
            "wave1_accepted_dependency",
            "report_accepted_dependency",
            "accepted_submission_records",
        }
        and type(receipt.get("schema_version")) is int
        and receipt.get("schema_version") == 1
        and receipt.get("status") == "two_wave_gpu_canary_ready_to_release"
        and receipt.get("campaign_id") == CAMPAIGN_ID
        and receipt.get("scientific") is False
        and receipt.get("state_root") == str(root)
        and receipt.get("canary_token") == value["canary_token"]
        and receipt.get("controller_identity_sha256")
        == value["controller_identity_sha256"]
        and exact_json_equal(receipt.get("controller_lock"), controller_lock)
        and receipt.get("authorization_sha256") == authorization_sha256
        and exact_json_equal(receipt.get("job_ids"), jobs)
        and exact_json_equal(receipt.get("job_names"), value["job_names"])
        and exact_json_equal(
            receipt.get("dependencies"), value["dependencies"]
        )
        and receipt.get("scheduler_comment") == value["scheduler_comment"]
        and exact_json_equal(
            receipt.get("scheduler_executables"), value["scheduler_executables"]
        )
        and exact_json_equal(
            receipt.get("scheduler_control_plane"),
            value["scheduler_control_plane"],
        )
        and receipt.get("scheduler_control_plane_sha256")
        == value["scheduler_control_plane_sha256"]
        and exact_json_equal(
            receipt.get("source_sha256"), value["source_sha256"]
        )
        and receipt.get("package_protocol_sha256")
        == value["package_protocol_sha256"]
        and receipt.get("accepted_submission_records_sha256")
        == value["accepted_submission_records_sha256"]
        and receipt.get("accepted_scheduler_evidence_sha256")
        == value["accepted_scheduler_evidence_sha256"],
        "canary durable receipt differs",
    )
    submission_records = receipt.get("accepted_submission_records")
    scheduler_evidence = {
        "wave0_accepted_hold": receipt.get("wave0_accepted_hold"),
        "wave1_accepted_dependency": receipt.get("wave1_accepted_dependency"),
        "report_accepted_dependency": receipt.get("report_accepted_dependency"),
    }
    require(
        isinstance(submission_records, Mapping)
        and set(submission_records) == {"wave0", "wave1", "report"}
        and stable_hash(submission_records)
        == value["accepted_submission_records_sha256"]
        and stable_hash(scheduler_evidence)
        == value["accepted_scheduler_evidence_sha256"],
        "canary accepted scheduler evidence hashes differ",
    )
    for role in ("wave0", "wave1", "report"):
        _validate_submission_record(
            submission_records[role], role=role, root=root, authorization=value
        )
        _validate_scheduler_evidence(
            scheduler_evidence[
                "wave0_accepted_hold"
                if role == "wave0"
                else f"{role}_accepted_dependency"
            ],
            role=role,
            root=root,
            authorization=value,
        )
    ready = read_json(root / READY_TO_RELEASE_NAME, "canary ready-to-release record")
    require(
        exact_json_equal(
            ready,
            {
            "schema_version": 1,
            "status": "canary_ready_to_release",
            "campaign_id": CAMPAIGN_ID,
            "authorization_sha256": authorization_sha256,
            "receipt_sha256": sha256_file(root / RECEIPT_NAME),
            "job_ids": jobs,
            "dependencies": value["dependencies"],
            "accepted_submission_records_sha256": value[
                "accepted_submission_records_sha256"
            ],
            "accepted_scheduler_evidence_sha256": value[
                "accepted_scheduler_evidence_sha256"
            ],
            },
        ),
        "canary ready-to-release record differs",
    )
    return value


def _slurm_job_id() -> str:
    value = os.environ.get("SLURM_JOB_ID", "")
    require(_is_job_id(value), "canary requires a numeric SLURM_JOB_ID")
    require(os.environ.get("SLURM_RESTART_COUNT", "0") == "0", "canary within-wave requeue is forbidden")
    return value


def _wave0(state_root: Path, authorization: Mapping[str, Any]) -> int:
    require(_slurm_job_id() == authorization["job_ids"]["wave0"], "wave-zero job identity differs")
    checkpoint = state_root / "wave0_checkpoint.pt"
    require(not os.path.lexists(checkpoint), "wave-zero checkpoint already exists")
    torch = _verified_torch()

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "canary requires exactly one visible CUDA device")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(bool(visible) and "," not in visible and torch.cuda.current_device() == 0, "canary selected GPU differs")
    device = torch.device("cuda:0")
    source = torch.arange(256, dtype=torch.float32, device=device).reshape(16, 16)
    product = source @ source.transpose(0, 1)
    torch.cuda.synchronize(device)
    payload = {
        "schema_version": 1,
        "canary_token": authorization["canary_token"],
        "tensor": product.cpu(),
        "checksum": float(product.sum().item()),
    }
    temporary = state_root / f".{checkpoint.name}.{os.getpid()}.tmp"
    torch.save(payload, temporary)
    os.chmod(temporary, 0o444)
    os.replace(temporary, checkpoint)
    ready = {
        "schema_version": 1,
        "status": "wave0_ready",
        "campaign_id": CAMPAIGN_ID,
        "canary_token": authorization["canary_token"],
        "wave0_job_id": authorization["job_ids"]["wave0"],
        "checkpoint_sha256": sha256_file(checkpoint),
        "expected_resumed_result": float(payload["checksum"]) + float(product.numel()),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "within_wave_requeue": False,
    }
    seal_json(state_root / READY_NAME, ready)
    return 0


def _wave1(state_root: Path, authorization: Mapping[str, Any]) -> int:
    require(_slurm_job_id() == authorization["job_ids"]["wave1"], "wave-one job identity differs")
    ready = read_json(state_root / READY_NAME, "wave-zero READY")
    checkpoint = state_root / "wave0_checkpoint.pt"
    require(
        set(ready)
        == {
            "schema_version",
            "status",
            "campaign_id",
            "canary_token",
            "wave0_job_id",
            "checkpoint_sha256",
            "expected_resumed_result",
            "cuda_device_name",
            "within_wave_requeue",
        }
        and type(ready.get("schema_version")) is int
        and ready.get("schema_version") == 1
        and ready.get("status") == "wave0_ready"
        and ready.get("campaign_id") == CAMPAIGN_ID
        and ready.get("canary_token") == authorization["canary_token"]
        and ready.get("wave0_job_id") == authorization["job_ids"]["wave0"]
        and ready.get("within_wave_requeue") is False
        and ready.get("checkpoint_sha256") == sha256_file(checkpoint),
        "wave-zero READY/checkpoint binding differs",
    )
    require(
        isinstance(ready.get("expected_resumed_result"), (int, float))
        and not isinstance(ready["expected_resumed_result"], bool)
        and math.isfinite(float(ready["expected_resumed_result"])),
        "wave-zero READY expected result differs",
    )
    torch = _verified_torch()

    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "canary requires exactly one visible CUDA device")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(bool(visible) and "," not in visible and torch.cuda.current_device() == 0, "canary selected GPU differs")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(
        isinstance(saved, dict)
        and type(saved.get("schema_version")) is int
        and saved.get("schema_version") == 1
        and saved.get("canary_token") == authorization["canary_token"],
        "canary checkpoint identity differs",
    )
    tensor = saved.get("tensor")
    require(isinstance(tensor, torch.Tensor) and tuple(tensor.shape) == (16, 16), "canary tensor differs")
    device = torch.device("cuda:0")
    resumed = (tensor.to(device) + 1.0).sum()
    torch.cuda.synchronize(device)
    observed = float(resumed.item())
    expected = float(saved["checksum"]) + float(tensor.numel())
    require(
        math.isfinite(observed)
        and math.isclose(observed, expected, rel_tol=1e-6, abs_tol=1.0),
        "canary resumed GPU result differs",
    )
    require(
        math.isclose(
            expected,
            float(ready["expected_resumed_result"]),
            rel_tol=1e-12,
            abs_tol=0.0,
        ),
        "canary checkpoint/READY expected result differs",
    )
    seal_json(
        state_root / COMPLETE_NAME,
        {
            "schema_version": 1,
            "status": "wave1_complete",
            "campaign_id": CAMPAIGN_ID,
            "canary_token": authorization["canary_token"],
            "wave0_job_id": authorization["job_ids"]["wave0"],
            "wave1_job_id": authorization["job_ids"]["wave1"],
            "ready_sha256": sha256_file(state_root / READY_NAME),
            "checkpoint_sha256": sha256_file(checkpoint),
            "resumed_result": observed,
            "within_wave_requeue": False,
        },
    )
    return 0


def _report(state_root: Path, authorization: Mapping[str, Any]) -> int:
    require(_slurm_job_id() == authorization["job_ids"]["report"], "canary report job identity differs")
    _release_record, release_sha256 = _validate_release_record(
        state_root, authorization
    )
    complete = read_json(state_root / COMPLETE_NAME, "wave-one completion")
    ready = read_json(state_root / READY_NAME, "wave-zero READY")
    checkpoint = state_root / "wave0_checkpoint.pt"
    require(
        set(ready)
        == {
            "schema_version",
            "status",
            "campaign_id",
            "canary_token",
            "wave0_job_id",
            "checkpoint_sha256",
            "expected_resumed_result",
            "cuda_device_name",
            "within_wave_requeue",
        }
        and type(ready.get("schema_version")) is int
        and ready.get("schema_version") == 1
        and ready.get("status") == "wave0_ready"
        and ready.get("campaign_id") == CAMPAIGN_ID
        and ready.get("canary_token") == authorization["canary_token"]
        and ready.get("wave0_job_id") == authorization["job_ids"]["wave0"]
        and ready.get("checkpoint_sha256") == sha256_file(checkpoint)
        and ready.get("within_wave_requeue") is False
        and isinstance(ready.get("expected_resumed_result"), (int, float))
        and not isinstance(ready["expected_resumed_result"], bool)
        and math.isfinite(float(ready["expected_resumed_result"])),
        "canary READY/checkpoint identity differs",
    )
    require(
        set(complete)
        == {
            "schema_version",
            "status",
            "campaign_id",
            "canary_token",
            "wave0_job_id",
            "wave1_job_id",
            "ready_sha256",
            "checkpoint_sha256",
            "resumed_result",
            "within_wave_requeue",
        }
        and type(complete.get("schema_version")) is int
        and complete.get("schema_version") == 1
        and complete.get("status") == "wave1_complete"
        and complete.get("campaign_id") == CAMPAIGN_ID
        and complete.get("canary_token") == authorization["canary_token"]
        and complete.get("wave0_job_id") == authorization["job_ids"]["wave0"]
        and complete.get("wave1_job_id") == authorization["job_ids"]["wave1"]
        and complete.get("ready_sha256") == sha256_file(state_root / READY_NAME)
        and complete.get("checkpoint_sha256") == ready["checkpoint_sha256"]
        and complete.get("within_wave_requeue") is False
        and isinstance(complete.get("resumed_result"), (int, float))
        and not isinstance(complete["resumed_result"], bool)
        and math.isfinite(float(complete["resumed_result"]))
        and math.isclose(
            float(complete["resumed_result"]),
            float(ready["expected_resumed_result"]),
            rel_tol=1e-6,
            abs_tol=1.0,
        ),
        "canary completion identity differs",
    )
    seal_json(
        state_root / REPORT_NAME,
        {
            "schema_version": 1,
            "status": "two_wave_gpu_canary_passed",
            "campaign_id": CAMPAIGN_ID,
            "canary_token": authorization["canary_token"],
            "job_ids": authorization["job_ids"],
            "dependencies": authorization["dependencies"],
            "authorization_sha256": sha256_file(state_root / AUTH_NAME),
            "receipt_sha256": sha256_file(state_root / RECEIPT_NAME),
            "ready_to_release_sha256": sha256_file(
                state_root / READY_TO_RELEASE_NAME
            ),
            "wave0_release_sha256": release_sha256,
            "accepted_submission_records_sha256": authorization[
                "accepted_submission_records_sha256"
            ],
            "accepted_scheduler_evidence_sha256": authorization[
                "accepted_scheduler_evidence_sha256"
            ],
            "wave0_ready_sha256": sha256_file(state_root / READY_NAME),
            "checkpoint_sha256": sha256_file(checkpoint),
            "wave1_complete_sha256": sha256_file(state_root / COMPLETE_NAME),
        },
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--role", choices=("wave0", "wave1", "report"), required=True)
    args = parser.parse_args(argv)
    try:
        _bootstrap_runtime()
        authorization = _authorization(args.state_root)
        if args.role == "wave0":
            return _wave0(args.state_root, authorization)
        if args.role == "wave1":
            return _wave1(args.state_root, authorization)
        return _report(args.state_root, authorization)
    except Exception as exc:
        print(f"Exp23 two-wave GPU canary error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
