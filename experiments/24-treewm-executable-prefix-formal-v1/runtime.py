#!/usr/bin/env python3
"""Hardened, stdlib-only Exp24 control-plane primitives.

This is an M2A runtime-authority scaffold, not launch authority.  The public mutation entry
point first calls :func:`campaign.assert_launch_authorized`; the checked-in M2A
manifest cannot satisfy that call.  The lower-level functions are deliberately
dependency-injected so their crash, reconciliation, and rollback behavior can be
tested without contacting Slurm or creating an Exp24 output namespace.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import base64
import ctypes
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import pwd
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence


sys.dont_write_bytecode = True

PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]
PACKAGE_RELATIVE = Path("experiments/24-treewm-executable-prefix-formal-v1")
CAMPAIGN_ID = "treewm-executable-prefix-formal-v1-launch1"
JOB_ID = re.compile(r"^[1-9][0-9]*$")
SBATCH_RESPONSE = re.compile(r"^(?P<job_id>[1-9][0-9]*)$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PINNED_PYTHON = (
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/envs/treewm-formal-py311/bin/python"
)
SCHEDULER_CLIENT_TIMEOUT_SECONDS = 30
PRE_RECEIPT_TRANSACTION_BUDGET_SECONDS = 600
EXPECTED_ACCEPTED_PILOT_CAMPAIGN_ID = "treewm-executable-prefix-repair-pilot-v1-launch8"
FORBIDDEN_POSITIVE_PILOT_CAMPAIGN_ID = "treewm-executable-prefix-repair-pilot-v1-launch7"
ENGINEERING_PILOT_ADAPTER_STATE = "sealed_versioned_adapter"
FROZEN_LAUNCH8_SOURCE_COMMIT = "33122e15d0aaf3661893a4c853fd5ac49173c685"
FROZEN_LAUNCH8_PACKAGE_RELATIVE = Path(
    "experiments/23-treewm-executable-prefix-repair-pilot-v1"
)
FROZEN_LAUNCH8_PROTOCOL_SHA256 = (
    "2c0231b61197fe67790432c78a896272a55c3497a777d490598b53a6be67342f"
)
FROZEN_LAUNCH8_VERIFIER_SOURCE_FILE_COUNT = 147
FROZEN_LAUNCH8_TRAINER_SOURCE_FILE_COUNT = 114
FROZEN_LAUNCH8_VERIFIER_SOURCE_INVENTORY_SHA256 = (
    "9bff89010f792d1aed8b3b691567655daab8f83135d6421798b5efea29a2f2c5"
)
FROZEN_LAUNCH8_ENTRYPOINT_SHA256 = {
    "campaign.py": "9df19ac344d5bab82557633c8c679c5664f80ddeed03204ebd203b1a307dec99",
    "gate.py": "15b7afaa7b0bb83dac067f15284ef56786134ba6aeedf5e9474bc45b32b49189",
    "report.py": "510b7178f0cfdc7ba62d837dcf64bb20001f3b1828aed286829603742df63057",
    "manifest.json": "c0365b1bb44f36f128fbe969e66b83a929ba40bf653b311c02b2c79f9caf9c95",
    "protocol.sha256": "b12cd1db90dc81be419407eb28e75bbf4e76d5507cff743a84390c3f5e174767",
}
FROZEN_LAUNCH8_PACKAGE_BINDING = {
    "package_protocol_sha256": FROZEN_LAUNCH8_PROTOCOL_SHA256,
    "manifest_sha256": "bc8ec56aa0ac4d786be6d059a334ed506055732b897d1fee34b054d8c67cd9ec",
    "trainer_code_fingerprint": "a547375cb37e9daa431a56ba72cbd6e993a9fb89a8c00aa03ec400bade448a59",
    "weight_audit_artifact_sha256": "34f6aa6cc8a8bfe6aeca4fd716d0497d778464e1d6f308fdd25d462b101f5fcc",
    "prefix_target_artifact_sha256": "256cf920458c75455910906f6aa3d13b2dcae143b27adab337840dd655afbb48",
    "resolved_config_artifact_sha256": "ff3e11ddac225d6acfb570c532830d103d8a917c78e4d4f072758b995b7216f4",
    "causal_parity_artifact_sha256": "f74d2cf9c9f07f01c58979825c8bd8cdc358027d26bcd16e98c1985af404a4b1",
}
FROZEN_LAUNCH8_PROTOCOL_FILES = (
    "manifest.json",
    "campaign.py",
    "gate.py",
    "weight_audit.py",
    "weight_audit.lock.json",
    "prefix_target_audit.py",
    "prefix_target.lock.json",
    "resolved_config_audit.py",
    "resolved_config.lock.json",
    "causal_parity_audit.py",
    "causal_parity.lock.json",
    "train_entry.py",
    "worker.py",
    "train.slurm",
    "submit.py",
    "cancel.py",
    "report.py",
    "report.slurm",
    "dag_evidence.py",
    "two_wave_canary.py",
    "canary_worker.py",
    "canary_gpu.slurm",
    "canary_report.slurm",
    "canary1_negative_provenance.json",
    "canary2_acceptance_provenance.json",
    "launch7_negative_provenance.json",
    "README.md",
    "tests/test_campaign.py",
    "tests/test_gate.py",
    "tests/test_lifecycle.py",
    "tests/test_orchestration.py",
    "tests/test_two_wave_canary.py",
)
ENGINEERING_PILOT_ADAPTER_REQUIREMENTS = (
    "exact immutable report quartet schemas, ownership, modes, inventory, and cross-hashes",
    "exact independently anchored campaign, submission, package, and reporter/gate protocol identities",
    "exact per-cell bundle identity, raw scalar axes, prefix source, terminal outcome rows, and metrics",
    "deterministic raw bundle telemetry to decision-boundary derivation under the frozen Launch8 gate",
    "exact structural, method, candidate, calibration, and outcome gate key sets and nested schemas",
    "exact bundle-decision-provenance per-index identity join and all four terminal artifact hash joins",
    "exact outcome-blind boundary-evaluation and paired-calibration provenance hashes",
    "recomputed per-seed, per-setting, macro, paired, strict, and not-worse acceptance predicates",
    "authenticated report/cancel serialization and terminal non-canceled submission state",
    "retained no-follow fd trust chain and procfd replay over the exact closed 33122e15 source/report trees and full verifier closure",
    "independent audit of positive adapter bytes before any binding publication",
)
CONTROL_ENVIRONMENT = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "LANG": "C",
    "LC_ALL": "C",
    "SLURM_CONF": "/cm/shared/apps/slurm/var/etc/cs-oci-ord/slurm.conf",
}
SCHEDULER_TRUST_MODEL = (
    "root-admin mutable scheduler control plane; config and Lua policy bytes are "
    "observation-bound from preclaim through submission; root-owned Slurm clients, "
    "plugin binaries, and shared libraries are trusted mutable external runtime"
)
CONTROL_PLANE_CONTRACT = {
    "slurm_conf": "/cm/shared/apps/slurm/var/etc/cs-oci-ord/slurm.conf",
    "cluster_name": "cs-oci-ord",
    "slurmctld_hosts": ["cs-oci-ord-a", "cs-oci-ord-b"],
    "slurmctld_port": 6817,
    "auth_type": "auth/munge",
    "gres_types": ["gpu"],
    "cli_filter_plugins": ["lua"],
    "job_submit_plugins": ["lua"],
    "trust_model": SCHEDULER_TRUST_MODEL,
}
SCHEDULER_POLICY_FILES = (
    "cli_filter.lua",
    "cli_filter_config.lua",
    "cli_filter_config_defaults.lua",
)
SCHEDULER_POLICY_DIRECTORY = "cli_filters"
SCHEDULER_REQUIRED_POLICY_MODULES = frozenset(
    {
        "util.lua",
        "cli_filter_checks_nvl72.lua",
        "cli_filter_checks_qos.lua",
        "cli_filter_checks_stale_data.lua",
    }
)
PACKAGE_SNAPSHOT_FILES = (
    "README.md",
    "aggregate.py",
    "campaign.py",
    "cancel.py",
    "final_eval.py",
    "formal_objective_delta.json",
    "gate.slurm",
    "accepted_engineering_pilot.binding.json",
    "engineering_pilot_binder.py",
    "launch7_negative.binding.json",
    "m2a_schema.json",
    "manifest.json",
    "report.py",
    "report.slurm",
    "runtime.py",
    "stage_gate.py",
    "submit.py",
    "train.slurm",
    "train_entry.py",
    "worker.py",
    "aggregate.slurm",
    "final_eval.slurm",
)
EMERGENCY_DISPATCH_FILES = (
    "campaign.py",
    "cancel.py",
    "manifest.json",
    "runtime.py",
)
SUBMISSION_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_root",
        "snapshot_root",
        "manifest_sha256",
        "m2a_schema",
        "interpreter_provenance",
        "launch7_negative_binding",
        "engineering_pilot_adapter_interface",
        "accepted_engineering_pilot_binding",
        "snapshot_inventory",
        "snapshot_inventory_sha256",
        "emergency_dispatch",
        "dag",
        "scheduler_control_plane",
        "scheduler_preclaim",
        "scheduler_fallback",
        "contract_body_sha256",
    }
)
SUBMISSION_RECEIPT_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_root",
        "snapshot_root",
        "submission_sha256",
        "manifest_sha256",
        "jobs",
        "dag_names",
        "training_array",
        "heldout_array",
    }
)
INTERPRETER_PROVENANCE_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "source_manifest_sha256",
        "lexical_executable",
        "lexical_kind",
        "lexical_symlink_target",
        "resolved_executable",
        "resolved_executable_sha256",
        "resolved_executable_size",
        "base_executable",
        "pyvenv_cfg",
        "pyvenv_cfg_sha256",
        "pyvenv_cfg_size",
        "venv_site_packages",
        "base_site_packages",
        "python_version",
        "implementation",
        "cache_tag",
        "provenance_sha256",
    }
)


class RuntimeContractError(RuntimeError):
    """A local byte, transaction, scheduler, or immutable-artifact contract failed."""


class SchedulerTransactionError(RuntimeContractError):
    """Scheduler mutation failed; ``job_ids`` are exact accepted/reconciled IDs."""

    def __init__(self, message: str, job_ids: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.job_ids = tuple(sorted(set(job_ids), key=int))


class CommitRecoveryRequired(RuntimeContractError):
    """Durable READY exists; recovery, never rollback, must resolve commit."""


class ActivationRecoveryRequired(RuntimeContractError):
    """Receipt/authorization is durable; recovery must finish root activation."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeContractError(message)


def canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def launch8_verifier_dependency_relatives(repo_root: Path) -> tuple[Path, ...]:
    """Return the exact frozen Launch8 semantic-verifier dependency closure.

    The protocol inventory is explicit in the frozen Exp23 campaign.  Its campaign
    validator additionally hashes every trainer Python/config source, so those files
    are part of the adapter's executable dependency closure rather than merely
    descriptive provenance.
    """

    root = repo_root.absolute()
    trainer = {
        Path("scripts/__init__.py"),
        Path("scripts/train.py"),
        Path("configs/__init__.py"),
        *(path.relative_to(root) for path in (root / "treewm").rglob("*.py")),
        *(path.relative_to(root) for path in (root / "configs").rglob("*.yaml")),
    }
    require(
        len(trainer) == FROZEN_LAUNCH8_TRAINER_SOURCE_FILE_COUNT,
        "frozen Launch8 trainer source path coverage differs",
    )
    protocol = {
        FROZEN_LAUNCH8_PACKAGE_RELATIVE / name
        for name in FROZEN_LAUNCH8_PROTOCOL_FILES
    }
    closure = protocol | trainer | {
        FROZEN_LAUNCH8_PACKAGE_RELATIVE / "protocol.sha256"
    }
    require(
        len(closure) == FROZEN_LAUNCH8_VERIFIER_SOURCE_FILE_COUNT,
        "frozen Launch8 verifier dependency path coverage differs",
    )
    return tuple(sorted(closure, key=str))


def launch8_verifier_source_inventory(
    repo_root: Path,
    *,
    immutable: bool = False,
) -> dict[str, str]:
    """Authenticate the exact frozen Launch8 source closure without using Git."""

    root = repo_root.absolute()
    inventory: dict[str, str] = {}
    for relative in launch8_verifier_dependency_relatives(root):
        _payload, digest, info = authenticated_regular_bytes(
            root / relative,
            f"frozen Launch8 verifier source {relative}",
        )
        if immutable:
            require(
                info.st_uid == os.getuid()
                and info.st_gid == os.getgid()
                and info.st_nlink == 1
                and stat.S_IMODE(info.st_mode) == 0o444,
                f"frozen Launch8 verifier source is not immutable: {relative}",
            )
        inventory[str(relative)] = digest
    require(
        stable_hash(inventory)
        == FROZEN_LAUNCH8_VERIFIER_SOURCE_INVENTORY_SHA256,
        "frozen Launch8 verifier source inventory differs from 33122e15",
    )
    for name, expected in FROZEN_LAUNCH8_ENTRYPOINT_SHA256.items():
        require(
            inventory[str(FROZEN_LAUNCH8_PACKAGE_RELATIVE / name)] == expected,
            f"frozen Launch8 verifier entry point differs: {name}",
        )
    protocol_payload, _protocol_file_sha, _protocol_info = authenticated_regular_bytes(
        root / FROZEN_LAUNCH8_PACKAGE_RELATIVE / "protocol.sha256",
        "frozen Launch8 protocol lock",
    )
    try:
        locked_protocol = protocol_payload.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise RuntimeContractError("frozen Launch8 protocol lock is not ASCII") from exc
    require(
        locked_protocol == FROZEN_LAUNCH8_PROTOCOL_SHA256,
        "frozen Launch8 protocol semantic lock differs",
    )
    return inventory


def engineering_pilot_adapter_description() -> dict[str, Any]:
    """Canonical sealed-adapter/unbound-positive description shared by runtime and CLI."""

    return {
        "schema_version": 1,
        "status": "sealed_launch8_semantic_adapter_real_binding_unbound",
        "expected_campaign_id": EXPECTED_ACCEPTED_PILOT_CAMPAIGN_ID,
        "forbidden_positive_campaign_id": FORBIDDEN_POSITIVE_PILOT_CAMPAIGN_ID,
        "required_status": "accepted_engineering_pilot",
        "adapter_state": ENGINEERING_PILOT_ADAPTER_STATE,
        "binding_state": "unbound",
        "implementation_dependency_files": [
            str(PACKAGE_RELATIVE / "engineering_pilot_binder.py"),
            str(PACKAGE_RELATIVE / "runtime.py"),
        ],
        "frozen_source_commit": FROZEN_LAUNCH8_SOURCE_COMMIT,
        "frozen_package_relative": str(FROZEN_LAUNCH8_PACKAGE_RELATIVE),
        "frozen_protocol_sha256": FROZEN_LAUNCH8_PROTOCOL_SHA256,
        "frozen_source_inventory_sha256": (
            FROZEN_LAUNCH8_VERIFIER_SOURCE_INVENTORY_SHA256
        ),
        "frozen_source_file_count": FROZEN_LAUNCH8_VERIFIER_SOURCE_FILE_COUNT,
        "frozen_entrypoint_sha256": dict(FROZEN_LAUNCH8_ENTRYPOINT_SHA256),
        "frozen_package_binding": dict(FROZEN_LAUNCH8_PACKAGE_BINDING),
        "semantic_adapter_implemented": True,
        "requirements": list(ENGINEERING_PILOT_ADAPTER_REQUIREMENTS),
        "persistent_writes_performed": False,
        "real_report_opened": False,
    }


def accepted_engineering_pilot_placeholder() -> dict[str, Any]:
    """Exact M2A future-positive placeholder; no report bytes are trusted yet."""

    return {
        "schema_version": 1,
        "status": "awaiting_launch8_accepted_engineering_pilot",
        "campaign_id": EXPECTED_ACCEPTED_PILOT_CAMPAIGN_ID,
        "formal_submission_allowed": False,
        "adapter_file_sha256": None,
        "adapter_runtime_file_sha256": None,
        "adapter_description_sha256": None,
        "report_commit_file_sha256": None,
        "binding_sha256": None,
    }


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in items:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = child
    return value


def _nonfinite(token: str) -> None:
    raise RuntimeContractError(f"non-finite JSON value: {token}")


def parse_json_bytes(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_nonfinite,
        )
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeContractError(f"cannot decode {label}: {exc}") from exc
    require(isinstance(value, dict), f"{label} root is not an object")
    return value


def _identity(value: os.stat_result) -> tuple[int, ...]:
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


def safe_relative(value: str | Path, label: str) -> Path:
    raw = str(value)
    posix = PurePosixPath(raw)
    require(raw and not posix.is_absolute(), f"{label} must be relative")
    require("\\" not in raw, f"{label} contains a non-POSIX separator")
    require(all(part not in ("", ".", "..") for part in posix.parts), f"{label} is not normalized")
    normalized = Path(*posix.parts)
    require(str(normalized) == raw, f"{label} is not canonical")
    return normalized


def _absolute_parts(path: Path, label: str) -> tuple[str, ...]:
    raw = str(path)
    require(path.is_absolute(), f"{label} must be absolute")
    require(os.path.normpath(raw) == raw, f"{label} must be normalized")
    parts = tuple(part for part in raw.split(os.sep) if part)
    require(all(part not in (".", "..") for part in parts), f"{label} contains traversal")
    return parts


def open_absolute_directory(path: Path, label: str) -> int:
    """Open every directory component with ``O_NOFOLLOW``."""

    parts = _absolute_parts(path, label)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(os.sep, flags)
    try:
        for part in parts:
            child = os.open(part, flags, dir_fd=descriptor)
            info = os.fstat(child)
            require(stat.S_ISDIR(info.st_mode), f"{label} component is not a directory: {part}")
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def lexical_entry_exists(parent: Path, name: str, label: str) -> bool:
    """Test one directory entry without following its final component."""

    require(name not in {"", ".", ".."} and "/" not in name, f"{label} name differs")
    parent_fd = open_absolute_directory(parent, f"{label} parent")
    try:
        try:
            os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return True
    finally:
        os.close(parent_fd)


def authenticated_regular_bytes(path: Path, label: str) -> tuple[bytes, str, os.stat_result]:
    """Read one stable regular file through a no-follow parent descriptor."""

    parent_fd = open_absolute_directory(path.parent, f"{label} parent")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path.name, flags, dir_fd=parent_fd)
    except BaseException:
        os.close(parent_fd)
        raise
    try:
        before = os.fstat(descriptor)
        require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
        require(before.st_nlink == 1, f"{label} has an unsafe hard-link count")
        digest = hashlib.sha256()
        chunks: list[bytes] = []
        while True:
            block = os.read(descriptor, 16 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
            chunks.append(block)
        after = os.fstat(descriptor)
        require(_identity(before) == _identity(after), f"{label} changed while reading")
        return b"".join(chunks), digest.hexdigest(), before
    finally:
        os.close(descriptor)
        os.close(parent_fd)


def authenticated_json(path: Path, label: str) -> tuple[dict[str, Any], str]:
    payload, digest, _info = authenticated_regular_bytes(path, label)
    return parse_json_bytes(payload, label), digest


def authenticated_immutable_json(
    path: Path,
    label: str,
    *,
    mode: int = 0o444,
) -> tuple[dict[str, Any], str]:
    payload, digest, info = authenticated_regular_bytes(path, label)
    require(
        info.st_uid == os.getuid()
        and info.st_gid == os.getgid()
        and info.st_nlink == 1
        and stat.S_IMODE(info.st_mode) == mode,
        f"{label} ownership/mode differs",
    )
    return parse_json_bytes(payload, label), digest


def _verified_directory(path: Path, label: str) -> None:
    descriptor = open_absolute_directory(path, label)
    try:
        info = os.fstat(descriptor)
        require(
            stat.S_ISDIR(info.st_mode)
            and info.st_uid == os.getuid()
            and info.st_gid == os.getgid()
            and stat.S_IMODE(info.st_mode) == 0o755
            and info.st_nlink >= 1,
            f"{label} is not a stable directory",
        )
    finally:
        os.close(descriptor)


def capture_interpreter_provenance(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Capture the exact venv entry, resolved binary, and environment roots.

    The lexical entry matters: this environment is a venv symlink whose target is
    also the base interpreter.  Invoking only the resolved target would silently
    change package-root semantics even though its binary digest is identical.
    """

    lexical = Path(str(manifest["paths"]["python"]))
    require(lexical.is_absolute() and str(lexical) == PINNED_PYTHON, "pinned interpreter path differs")
    require(
        os.path.normpath(os.path.abspath(sys.executable)) == str(lexical),
        f"running interpreter is not the exact lexical venv entry: {sys.executable}",
    )
    require(sys.version_info[:3] == (3, 11, 15), "interpreter version is not Python 3.11.15")
    try:
        lexical_info = lexical.lstat()
    except OSError as exc:
        raise RuntimeContractError(f"pinned interpreter entry is unavailable: {exc}") from exc
    require(
        stat.S_ISLNK(lexical_info.st_mode)
        and lexical_info.st_uid == os.getuid()
        and lexical_info.st_gid == os.getgid()
        and lexical_info.st_nlink == 1,
        "pinned interpreter lexical entry is not the exact user-owned venv symlink",
    )
    lexical_target = os.readlink(lexical)
    require(lexical_target != "" and "\x00" not in lexical_target, "pinned interpreter symlink target differs")
    resolved = lexical.resolve(strict=True)
    _binary, binary_sha, binary_info = authenticated_regular_bytes(
        resolved,
        "resolved pinned interpreter",
    )
    require(
        binary_info.st_size > 0
        and bool(binary_info.st_mode & stat.S_IXUSR)
        and binary_info.st_uid == os.getuid()
        and binary_info.st_gid == os.getgid()
        and stat.S_IMODE(binary_info.st_mode) == 0o755
        and os.access(lexical, os.X_OK),
        "resolved pinned interpreter is not executable",
    )
    base_executable = Path(str(getattr(sys, "_base_executable", "")))
    require(
        base_executable.is_absolute()
        and base_executable.resolve(strict=True) == resolved,
        "base interpreter target differs from the pinned venv target",
    )
    venv_root = lexical.parent.parent
    pyvenv = venv_root / "pyvenv.cfg"
    pyvenv_payload, pyvenv_sha, pyvenv_info = authenticated_regular_bytes(
        pyvenv,
        "pinned pyvenv.cfg",
    )
    require(
        pyvenv_info.st_uid == os.getuid()
        and pyvenv_info.st_gid == os.getgid()
        and stat.S_IMODE(pyvenv_info.st_mode) == 0o644,
        "pinned pyvenv.cfg ownership/mode differs",
    )
    try:
        pyvenv_text = pyvenv_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeContractError(f"pinned pyvenv.cfg is not UTF-8: {exc}") from exc
    values: dict[str, str] = {}
    for raw_line in pyvenv_text.splitlines():
        if "=" not in raw_line:
            continue
        key, child = (item.strip() for item in raw_line.split("=", 1))
        require(key and key not in values, f"pinned pyvenv.cfg duplicates {key!r}")
        values[key] = child
    require(
        values.get("version") == "3.11.15"
        and values.get("include-system-site-packages") == "true"
        and isinstance(values.get("home"), str)
        and Path(values["home"]).is_absolute(),
        "pinned pyvenv.cfg environment identity differs",
    )
    version_directory = "python3.11"
    venv_site = (venv_root / "lib" / version_directory / "site-packages").resolve(strict=True)
    base_site = (
        Path(values["home"]).parent / "lib" / version_directory / "site-packages"
    ).resolve(strict=True)
    _verified_directory(venv_site, "pinned venv site-packages")
    _verified_directory(base_site, "pinned base site-packages")
    seed = {
        "schema_version": 1,
        "status": "captured_exact_pinned_python_3_11_15",
        "source_manifest_sha256": stable_hash(manifest),
        "lexical_executable": str(lexical),
        "lexical_kind": "symlink",
        "lexical_symlink_target": lexical_target,
        "resolved_executable": str(resolved),
        "resolved_executable_sha256": binary_sha,
        "resolved_executable_size": binary_info.st_size,
        "base_executable": str(base_executable),
        "pyvenv_cfg": str(pyvenv),
        "pyvenv_cfg_sha256": pyvenv_sha,
        "pyvenv_cfg_size": pyvenv_info.st_size,
        "venv_site_packages": str(venv_site),
        "base_site_packages": str(base_site),
        "python_version": "3.11.15",
        "implementation": sys.implementation.name,
        "cache_tag": sys.implementation.cache_tag,
    }
    seed["provenance_sha256"] = stable_hash(seed)
    return seed


def validate_interpreter_provenance(
    manifest: Mapping[str, Any],
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-capture and exact-match interpreter provenance before execution."""

    require(set(value) == INTERPRETER_PROVENANCE_KEYS, "interpreter provenance schema differs")
    body = dict(value)
    claimed = body.pop("provenance_sha256", None)
    require(SHA256.fullmatch(str(claimed or "")) is not None, "interpreter provenance hash is malformed")
    require(claimed == stable_hash(body), "interpreter provenance self-hash differs")
    current = capture_interpreter_provenance(manifest)
    require(dict(value) == current, "pinned interpreter provenance drifted")
    return current


def _fsync_directory(path: Path) -> None:
    descriptor = open_absolute_directory(path, f"fsync directory {path}")
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_json(path: Path, value: Mapping[str, Any], *, mode: int = 0o444) -> str:
    """Crash-safely publish one immutable JSON record without replacing a name.

    Bytes are completed and fsynced under a deterministic private sibling name.
    ``link(2)`` is the no-replace publication point.  The deterministic sibling
    also lets a retry finish or discard a dead writer's pre-publication state.
    """

    payload = (canonical_json(value) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    require(path.name not in {"", ".", ".."} and "/" not in path.name, "immutable record name differs")
    require(mode in {0o400, 0o440, 0o444}, "immutable record mode differs")
    parent_fd = open_absolute_directory(path.parent, f"immutable record parent {path.name}")
    temp_name = f".{path.name}.PUBLISHING"
    open_flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)

    def validate_final() -> None:
        existing, existing_digest, info = authenticated_regular_bytes(path, f"existing {path.name}")
        require(
            existing == payload
            and existing_digest == digest
            and info.st_uid == os.getuid()
            and info.st_gid == os.getgid()
            and stat.S_IMODE(info.st_mode) == mode,
            f"immutable record differs: {path}",
        )

    try:
        for _attempt in range(2):
            try:
                descriptor = os.open(
                    temp_name,
                    open_flags | os.O_CREAT | os.O_EXCL,
                    0o600,
                    dir_fd=parent_fd,
                )
                created = True
            except FileExistsError:
                descriptor = os.open(
                    temp_name,
                    os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=parent_fd,
                )
                created = False
            try:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeContractError(f"immutable publication is already active: {path}") from exc
                if not created:
                    before = os.fstat(descriptor)
                    require(
                        stat.S_ISREG(before.st_mode)
                        and before.st_uid == os.getuid()
                        and before.st_gid == os.getgid()
                        and before.st_nlink in {1, 2},
                        f"immutable publication residue differs: {path}",
                    )
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    chunks: list[bytes] = []
                    while True:
                        block = os.read(descriptor, 1024 * 1024)
                        if not block:
                            break
                        chunks.append(block)
                    after = os.fstat(descriptor)
                    require(_identity(before) == _identity(after), f"immutable publication residue raced: {path}")
                    residue_payload = b"".join(chunks)
                    final_exists = False
                    try:
                        os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
                        final_exists = True
                    except FileNotFoundError:
                        pass
                    if final_exists:
                        final_info = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
                        if before.st_nlink == 2:
                            require(
                                final_info.st_dev == before.st_dev
                                and final_info.st_ino == before.st_ino
                                and final_info.st_nlink == 2
                                and stat.S_ISREG(final_info.st_mode)
                                and final_info.st_uid == os.getuid()
                                and final_info.st_gid == os.getgid()
                                and stat.S_IMODE(final_info.st_mode) == mode
                                and residue_payload == payload,
                                f"recoverable immutable publication link differs: {path}",
                            )
                        else:
                            validate_final()
                        os.unlink(temp_name, dir_fd=parent_fd)
                        os.fsync(parent_fd)
                        validate_final()
                        return digest
                    if (
                        before.st_nlink == 1
                        and stat.S_IMODE(before.st_mode) == mode
                        and residue_payload == payload
                    ):
                        os.link(
                            temp_name,
                            path.name,
                            src_dir_fd=parent_fd,
                            dst_dir_fd=parent_fd,
                            follow_symlinks=False,
                        )
                        os.unlink(temp_name, dir_fd=parent_fd)
                        os.fsync(parent_fd)
                        return digest
                    # A lockable 0600/incomplete sibling has no publication point;
                    # it is safe to discard and recreate after a dead writer.
                    require(before.st_nlink == 1, f"immutable publication link state differs: {path}")
                    os.unlink(temp_name, dir_fd=parent_fd)
                    os.fsync(parent_fd)
                    continue
                offset = 0
                while offset < len(payload):
                    written = os.write(descriptor, payload[offset:])
                    require(written > 0, f"short write for {path}")
                    offset += written
                os.fsync(descriptor)
                os.fchmod(descriptor, mode)
                os.fsync(descriptor)
                try:
                    os.link(
                        temp_name,
                        path.name,
                        src_dir_fd=parent_fd,
                        dst_dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    validate_final()
                os.unlink(temp_name, dir_fd=parent_fd)
                os.fsync(parent_fd)
                return digest
            finally:
                os.close(descriptor)
        raise RuntimeContractError(f"cannot recover immutable publication residue: {path}")
    finally:
        os.close(parent_fd)


def repair_publication_residues(
    directory: Path,
    *,
    allowed_final: re.Pattern[str],
) -> list[str]:
    """Repair only authenticated dead-writer siblings while holding a caller lock."""

    descriptor = open_absolute_directory(directory, "publication-residue directory")
    repaired: list[str] = []
    try:
        with os.scandir(descriptor) as iterator:
            names = sorted(entry.name for entry in iterator)
        for name in names:
            match = re.fullmatch(r"\.(.+)\.PUBLISHING", name)
            if match is None:
                continue
            final_name = match.group(1)
            require(allowed_final.fullmatch(final_name) is not None, f"unexpected publication residue: {name}")
            temp_fd = os.open(
                name,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
                dir_fd=descriptor,
            )
            try:
                try:
                    fcntl.flock(temp_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise RuntimeContractError(f"publication remains owned by a live writer: {name}") from exc
                temp_info = os.fstat(temp_fd)
                require(
                    stat.S_ISREG(temp_info.st_mode)
                    and temp_info.st_uid == os.getuid()
                    and temp_info.st_gid == os.getgid()
                    and temp_info.st_nlink in {1, 2}
                    and stat.S_IMODE(temp_info.st_mode) in {0o600, 0o400, 0o440, 0o444},
                    f"publication residue identity/mode differs: {name}",
                )
                try:
                    final_info = os.stat(final_name, dir_fd=descriptor, follow_symlinks=False)
                except FileNotFoundError:
                    final_info = None
                if final_info is not None:
                    require(
                        temp_info.st_nlink == 2
                        and stat.S_IMODE(temp_info.st_mode) == 0o444
                        and final_info.st_dev == temp_info.st_dev
                        and final_info.st_ino == temp_info.st_ino
                        and final_info.st_nlink == 2,
                        f"publication residue does not bind its final record: {name}",
                    )
                else:
                    require(temp_info.st_nlink == 1, f"unpublished residue link count differs: {name}")
                os.unlink(name, dir_fd=descriptor)
                os.fsync(descriptor)
                repaired.append(name)
                if final_info is not None:
                    after = os.stat(final_name, dir_fd=descriptor, follow_symlinks=False)
                    require(after.st_nlink == 1, f"repaired final record link count differs: {final_name}")
            finally:
                os.close(temp_fd)
    finally:
        os.close(descriptor)
    return repaired


def _mkdir_exact(path: Path, mode: int, label: str) -> None:
    parent_fd = open_absolute_directory(path.parent, f"{label} parent")
    try:
        os.mkdir(path.name, mode=mode, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _rename_noreplace(parent_fd: int, source: str, destination: str) -> None:
    """Linux atomic directory publication with explicit no-replace semantics."""

    require(
        all(value not in {"", ".", ".."} and "/" not in value for value in (source, destination)),
        "rename publication name differs",
    )
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2 is unavailable for atomic transaction claim")
    renameat2.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_fd,
        source.encode("utf-8"),
        parent_fd,
        destination.encode("utf-8"),
        1,  # RENAME_NOREPLACE
    )
    if result != 0:
        number = ctypes.get_errno()
        if number == errno.EEXIST:
            raise FileExistsError(number, os.strerror(number), destination)
        raise OSError(number, os.strerror(number), destination)
    os.fsync(parent_fd)


def _tree_rows(root: Path, *, sealed: bool, owner: int | None = None) -> list[dict[str, Any]]:
    """Inventory a tree without following any descendant entry."""

    root_fd = open_absolute_directory(root, "tree root")
    rows: list[dict[str, Any]] = []
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)

    def visit(descriptor: int, prefix: str) -> None:
        before = os.fstat(descriptor)
        require(stat.S_ISDIR(before.st_mode), f"tree entry is not a directory: {prefix or '.'}")
        if owner is not None:
            require(before.st_uid == owner, f"tree directory owner differs: {prefix or '.'}")
        mode = stat.S_IMODE(before.st_mode)
        if sealed:
            require(mode == 0o555, f"sealed directory mode differs: {prefix or '.'}")
        with os.scandir(descriptor) as iterator:
            children = sorted(
                ((entry.name, entry.stat(follow_symlinks=False)) for entry in iterator),
                key=lambda row: row[0],
            )
        require(_identity(before) == _identity(os.fstat(descriptor)), f"tree directory changed: {prefix or '.'}")
        for name, info in children:
            require(name not in (".", "..") and "/" not in name, "tree entry name is unsafe")
            relative = f"{prefix}/{name}" if prefix else name
            require(not stat.S_ISLNK(info.st_mode), f"tree contains a symlink: {relative}")
            if owner is not None:
                require(info.st_uid == owner, f"tree entry owner differs: {relative}")
            if stat.S_ISDIR(info.st_mode):
                child = os.open(name, directory_flags, dir_fd=descriptor)
                try:
                    require(_identity(info) == _identity(os.fstat(child)), f"directory raced: {relative}")
                    rows.append({"path": relative, "kind": "directory", "mode": stat.S_IMODE(info.st_mode)})
                    visit(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(info.st_mode):
                require(info.st_nlink == 1, f"tree file has unsafe hard-link count: {relative}")
                if sealed:
                    require(stat.S_IMODE(info.st_mode) == 0o444, f"sealed file mode differs: {relative}")
                child = os.open(name, file_flags, dir_fd=descriptor)
                try:
                    opened = os.fstat(child)
                    require(_identity(info) == _identity(opened), f"file raced: {relative}")
                    digest = hashlib.sha256()
                    while True:
                        block = os.read(child, 16 * 1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                    require(_identity(opened) == _identity(os.fstat(child)), f"file changed: {relative}")
                    rows.append(
                        {
                            "path": relative,
                            "kind": "file",
                            "mode": stat.S_IMODE(info.st_mode),
                            "size": info.st_size,
                            "sha256": digest.hexdigest(),
                        }
                    )
                finally:
                    os.close(child)
            else:
                raise RuntimeContractError(f"tree contains a special entry: {relative}")

    try:
        visit(root_fd, "")
    finally:
        os.close(root_fd)
    return rows


def m1_snapshot_inventory(repo_root: Path = REPOSITORY_ROOT) -> dict[str, str]:
    """Return the exact M2A bootstrap/source inventory (legacy public name)."""

    root = repo_root.resolve(strict=True)
    relative_paths: list[Path] = [PACKAGE_RELATIVE / name for name in PACKAGE_SNAPSHOT_FILES]
    frozen_launch8_inventory = launch8_verifier_source_inventory(root)
    relative_paths.extend(Path(path) for path in frozen_launch8_inventory)
    relative_paths.append(
        Path("configs/experiment/treewm_v2_grounded_gauge_formal_v1.yaml")
    )
    relative_paths = sorted(set(relative_paths), key=str)
    inventory: dict[str, str] = {}
    for relative in relative_paths:
        safe_relative(relative, "snapshot source path")
        payload, digest, _info = authenticated_regular_bytes(root / relative, f"snapshot source {relative}")
        require(payload is not None, f"snapshot source is unreadable: {relative}")
        inventory[str(relative)] = digest
    require(
        all(inventory.get(path) == digest for path, digest in frozen_launch8_inventory.items()),
        "M2A snapshot does not bind the complete frozen Launch8 verifier closure",
    )
    return inventory


def verify_inventory_sources(repo_root: Path, inventory: Mapping[str, str]) -> None:
    require(inventory and all(isinstance(key, str) for key in inventory), "snapshot inventory is empty")
    require(list(inventory) == sorted(inventory), "snapshot inventory order differs")
    for raw_relative, expected in inventory.items():
        relative = safe_relative(raw_relative, "snapshot inventory path")
        require(SHA256.fullmatch(str(expected)) is not None, f"snapshot SHA256 is malformed: {relative}")
        _payload, actual, _info = authenticated_regular_bytes(repo_root / relative, f"snapshot inventory {relative}")
        require(actual == expected, f"snapshot source drift: {relative}")


def create_source_snapshot(repo_root: Path, snapshot_root: Path, inventory: Mapping[str, str]) -> None:
    """Copy the verified exact inventory and seal it 0555/0444."""

    require(not snapshot_root.exists() and not snapshot_root.is_symlink(), "snapshot target already exists")
    verify_inventory_sources(repo_root, inventory)
    _mkdir_exact(snapshot_root, 0o700, "snapshot root")
    created_directories: set[Path] = {snapshot_root}
    try:
        for raw_relative, expected in inventory.items():
            relative = safe_relative(raw_relative, "snapshot copy path")
            destination = snapshot_root / relative
            current = snapshot_root
            for part in relative.parts[:-1]:
                current = current / part
                if current not in created_directories:
                    _mkdir_exact(current, 0o700, "snapshot directory")
                    created_directories.add(current)
            payload, digest, before = authenticated_regular_bytes(repo_root / relative, f"live snapshot source {relative}")
            require(digest == expected, f"live source differs during snapshot: {relative}")
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(destination, flags, 0o600)
            try:
                offset = 0
                while offset < len(payload):
                    count = os.write(descriptor, payload[offset:])
                    require(count > 0, f"snapshot write stalled: {relative}")
                    offset += count
                os.fsync(descriptor)
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _fsync_directory(destination.parent)
            _again, live_digest, after = authenticated_regular_bytes(repo_root / relative, f"post-copy source {relative}")
            require(live_digest == expected and _identity(before) == _identity(after), f"source changed around snapshot: {relative}")
        for directory in sorted(created_directories, key=lambda path: len(path.parts), reverse=True):
            os.chmod(directory, 0o555, follow_symlinks=False)
            _fsync_directory(directory)
            if directory != snapshot_root:
                _fsync_directory(directory.parent)
        verify_snapshot_files(snapshot_root, inventory)
    except BaseException:
        # A private/incomplete snapshot is never accepted because only a complete
        # sealed tree can verify.  The caller owns cleanup/recovery of this path.
        raise


def verify_snapshot_files(snapshot_root: Path, inventory: Mapping[str, str]) -> None:
    rows = _tree_rows(snapshot_root, sealed=True, owner=os.getuid())
    files = {row["path"]: row["sha256"] for row in rows if row["kind"] == "file"}
    require(files == dict(inventory), "snapshot tree differs from its exact inventory")
    expected_dirs: set[str] = set()
    for raw_relative in inventory:
        parent = PurePosixPath(raw_relative).parent
        while str(parent) != ".":
            expected_dirs.add(str(parent))
            parent = parent.parent
    actual_dirs = {row["path"] for row in rows if row["kind"] == "directory"}
    require(actual_dirs == expected_dirs, "snapshot directory inventory differs")


def verify_emergency_snapshot_files(
    snapshot_root: Path,
    contract: Mapping[str, Any],
) -> dict[str, str]:
    """Verify only the stable cancel/recover capsule named by contract v1.

    This is intentionally narrower than :func:`verify_snapshot_files`.  A lost
    trainer/config artifact must stop queued science, but must not prevent an
    operator from authenticating the sealed exact job IDs and canceling them.
    """

    envelope = contract.get("emergency_dispatch")
    require(isinstance(envelope, dict) and set(envelope) == {
        "schema_version", "campaign_id", "package_relative", "python",
        "targets", "policy",
    }, "emergency dispatch envelope schema differs")
    require(
        envelope.get("schema_version") == 1
        and envelope.get("campaign_id") == CAMPAIGN_ID
        and envelope.get("package_relative") == str(PACKAGE_RELATIVE)
        and envelope.get("policy") == "stable_v1_minimal_snapshot_cancel_recover_capsule",
        "emergency dispatch envelope identity differs",
    )
    targets = envelope.get("targets")
    expected_names = [str(PACKAGE_RELATIVE / name) for name in EMERGENCY_DISPATCH_FILES]
    require(
        isinstance(targets, dict)
        and list(targets) == sorted(expected_names)
        and all(SHA256.fullmatch(str(value)) is not None for value in targets.values()),
        "emergency dispatch targets differ",
    )
    inventory = contract.get("snapshot_inventory")
    require(isinstance(inventory, dict), "emergency dispatch snapshot inventory differs")
    for relative, expected in targets.items():
        require(inventory.get(relative) == expected, f"emergency target inventory binding differs: {relative}")
        _payload, digest, info = authenticated_regular_bytes(
            snapshot_root / safe_relative(relative, "emergency target path"),
            f"emergency snapshot target {relative}",
        )
        require(
            digest == expected
            and info.st_uid == os.getuid()
            and info.st_gid == os.getgid()
            and info.st_nlink == 1
            and stat.S_IMODE(info.st_mode) == 0o444,
            f"emergency snapshot target differs: {relative}",
        )
    return {str(key): str(value) for key, value in targets.items()}


def verify_submission_snapshot_identity(
    manifest: Mapping[str, Any],
    snapshot_root: Path,
    inventory: Mapping[str, str],
) -> None:
    """Bind in-memory submission decisions and live control bytes to the snapshot."""

    import campaign

    verify_snapshot_files(snapshot_root, inventory)
    snapshot_manifest, _file_sha = authenticated_immutable_json(
        snapshot_root / PACKAGE_RELATIVE / "manifest.json",
        "sealed submission snapshot manifest",
    )
    campaign.validate_manifest(snapshot_manifest)
    require(
        snapshot_manifest == dict(manifest)
        and campaign.manifest_sha256(snapshot_manifest) == campaign.manifest_sha256(manifest),
        "sealed snapshot manifest differs from the authenticated in-memory manifest",
    )
    live_control_files = {
        "campaign.py": Path(str(campaign.__file__)),
        "runtime.py": Path(__file__),
        "submit.py": PACKAGE_DIR / "submit.py",
    }
    for name, source in live_control_files.items():
        relative = str(PACKAGE_RELATIVE / name)
        _payload, digest, _info = authenticated_regular_bytes(
            source,
            f"live executing control source {name}",
        )
        require(
            inventory.get(relative) == digest,
            f"live executing control source differs from sealed snapshot: {name}",
        )


def _restore_private_modes(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for current, directories, files in os.walk(root, topdown=False, followlinks=False):
        for name in files:
            path = Path(current) / name
            if not path.is_symlink():
                os.chmod(path, 0o600, follow_symlinks=False)
        for name in directories:
            path = Path(current) / name
            if not path.is_symlink():
                os.chmod(path, 0o700, follow_symlinks=False)
        os.chmod(current, 0o700, follow_symlinks=False)


def snapshot_test(repo_root: Path = REPOSITORY_ROOT) -> dict[str, Any]:
    """Exercise snapshot copy + isolated ``-I -S -B`` bootstrap in private temp."""

    inventory = m1_snapshot_inventory(repo_root)
    task_root = Path(tempfile.mkdtemp(prefix="exp24-m1-snapshot-"))
    try:
        snapshot_root = task_root / "repo"
        create_source_snapshot(repo_root, snapshot_root, inventory)
        inventory_path = task_root / "inventory.json"
        exclusive_json(inventory_path, {"schema_version": 1, "files": inventory}, mode=0o400)
        command = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(snapshot_root / PACKAGE_RELATIVE / "submit.py"),
            "--internal-snapshot-probe",
            "--snapshot-root",
            str(snapshot_root),
            "--inventory-file",
            str(inventory_path),
        ]
        completed = subprocess.run(
            command,
            cwd=snapshot_root,
            env={"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C"},
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        require(completed.returncode == 0, f"isolated snapshot probe failed: {completed.stderr.strip()}")
        result = parse_json_bytes(completed.stdout.encode("utf-8"), "snapshot probe stdout")
        require(result.get("status") == "verified_m2a_snapshot_bootstrap", "snapshot probe status differs")
        require(result.get("inventory_sha256") == stable_hash(inventory), "snapshot probe inventory differs")
        return {
            "schema_version": 1,
            "status": "verified_ephemeral_m2a_snapshot",
            "file_count": len(inventory),
            "inventory_sha256": stable_hash(inventory),
            "isolated_flags": ["-I", "-S", "-B"],
            "persistent_writes_performed": False,
            "scheduler_calls": [],
            "temporary_snapshot_removed": True,
        }
    finally:
        _restore_private_modes(task_root)
        shutil.rmtree(task_root)


Runner = Callable[
    [Sequence[str], Path, Mapping[str, str], Sequence[int]],
    subprocess.CompletedProcess[str],
]
Observer = Callable[[], Mapping[str, Any]]


def default_runner(
    command: Sequence[str],
    cwd: Path,
    environment: Mapping[str, str],
    inherited_fds: Sequence[int] = (),
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        env=dict(environment),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        pass_fds=tuple(inherited_fds),
        timeout=SCHEDULER_CLIENT_TIMEOUT_SECONDS,
    )


def _resolved_executable_observation(path: Path, label: str) -> dict[str, Any]:
    lexical_before = path.lstat()
    require(stat.S_ISREG(lexical_before.st_mode) or stat.S_ISLNK(lexical_before.st_mode), f"{label} is not a file/link")
    resolved = path.resolve(strict=True)
    payload, digest, info = authenticated_regular_bytes(resolved, f"resolved {label}")
    lexical_after = path.lstat()
    require(_identity(lexical_before) == _identity(lexical_after), f"{label} lexical entry changed")
    require(info.st_uid == 0, f"{label} executable is not root-owned")
    require(not stat.S_IMODE(info.st_mode) & 0o022, f"{label} executable is group/world writable")
    return {
        "lexical": str(path),
        "lexical_identity": list(_identity(lexical_before)),
        "resolved": str(resolved),
        "sha256": digest,
        "size": len(payload),
        "resolved_identity": list(_identity(info)),
    }


def _root_owned_regular_observation(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    """Read an exact root-owned 0644 file through a root-owned directory chain."""

    parts = _absolute_parts(path.parent, f"{label} parent")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptors: list[int] = [os.open(os.sep, directory_flags)]
    try:
        root_info = os.fstat(descriptors[0])
        require(root_info.st_uid == 0 and not stat.S_IMODE(root_info.st_mode) & 0o022, f"{label} root ownership/mode differs")
        for part in parts:
            listed = os.stat(part, dir_fd=descriptors[-1], follow_symlinks=False)
            require(stat.S_ISDIR(listed.st_mode), f"{label} parent contains a non-directory")
            child = os.open(part, directory_flags, dir_fd=descriptors[-1])
            opened = os.fstat(child)
            require(_identity(listed) == _identity(opened), f"{label} parent raced")
            require(opened.st_uid == 0 and not stat.S_IMODE(opened.st_mode) & 0o022, f"{label} parent ownership/mode differs")
            descriptors.append(child)
        listed_file = os.stat(path.name, dir_fd=descriptors[-1], follow_symlinks=False)
        file_fd = os.open(
            path.name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=descriptors[-1],
        )
        try:
            opened_file = os.fstat(file_fd)
            require(_identity(listed_file) == _identity(opened_file), f"{label} raced before read")
            require(
                stat.S_ISREG(opened_file.st_mode)
                and opened_file.st_uid == 0
                and opened_file.st_gid == 0
                and opened_file.st_nlink == 1
                and stat.S_IMODE(opened_file.st_mode) == 0o644,
                f"{label} must be root:root regular 0644 with one link",
            )
            chunks: list[bytes] = []
            digest = hashlib.sha256()
            total = 0
            while True:
                block = os.read(file_fd, 1024 * 1024)
                if not block:
                    break
                total += len(block)
                require(total <= 16 * 1024 * 1024, f"{label} exceeds 16 MiB")
                digest.update(block)
                chunks.append(block)
            require(_identity(opened_file) == _identity(os.fstat(file_fd)), f"{label} changed during read")
            require(_identity(opened_file) == _identity(os.stat(path.name, dir_fd=descriptors[-1], follow_symlinks=False)), f"{label} lexical entry changed")
            return b"".join(chunks), {
                "path": str(path),
                "sha256": digest.hexdigest(),
                "identity": list(_identity(opened_file)),
            }
        finally:
            os.close(file_fd)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def parse_slurm_config(payload: bytes, contract: Mapping[str, Any]) -> dict[str, Any]:
    require(dict(contract) == CONTROL_PLANE_CONTRACT, "scheduler control-plane contract differs")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeContractError(f"Slurm configuration is not UTF-8: {exc}") from exc
    require("\x00" not in text, "Slurm configuration contains NUL")
    directives: dict[str, list[str]] = {}
    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        require(re.match(r"(?i)^include(?:\s|=)", line) is None, "Slurm configuration may not include another file")
        require("=" in line, "Slurm configuration contains a malformed directive")
        raw_key, raw_value = line.split("=", 1)
        key, value = raw_key.strip().lower(), raw_value.strip()
        require(key and value, "Slurm configuration contains an empty directive")
        directives.setdefault(key, []).append(value)
    require(directives.get("clustername") == [contract["cluster_name"]], "Slurm ClusterName differs")
    require(directives.get("slurmctldhost") == contract["slurmctld_hosts"], "Slurm controller hosts differ")
    require(directives.get("slurmctldport", [str(contract["slurmctld_port"])]) == [str(contract["slurmctld_port"])], "Slurm controller port differs")
    require(directives.get("authtype") == [contract["auth_type"]], "Slurm AuthType differs")
    require(directives.get("grestypes") == [",".join(contract["gres_types"])], "Slurm GresTypes differs")
    require(directives.get("clifilterplugins") == [",".join(contract["cli_filter_plugins"])], "Slurm CliFilterPlugins differs")
    require(directives.get("jobsubmitplugins") == [",".join(contract["job_submit_plugins"])], "Slurm JobSubmitPlugins differs")
    require(directives.get("communicationparameters") == ["NoAddrCache"], "Slurm communication parameters differ")
    return {
        "cluster_name": contract["cluster_name"],
        "slurmctld_hosts": list(contract["slurmctld_hosts"]),
        "slurmctld_port": contract["slurmctld_port"],
        "auth_type": contract["auth_type"],
        "gres_types": list(contract["gres_types"]),
        "cli_filter_plugins": list(contract["cli_filter_plugins"]),
        "job_submit_plugins": list(contract["job_submit_plugins"]),
    }


def _scheduler_policy_observation(config_path: Path) -> dict[str, Any]:
    policy_root = config_path.parent
    policy_directory = policy_root / SCHEDULER_POLICY_DIRECTORY
    directory_fd = open_absolute_directory(policy_directory, "Slurm policy directory")
    try:
        info = os.fstat(directory_fd)
        require(info.st_uid == 0 and not stat.S_IMODE(info.st_mode) & 0o022, "Slurm policy directory ownership/mode differs")
        with os.scandir(directory_fd) as iterator:
            entries = sorted((entry.name, entry.stat(follow_symlinks=False)) for entry in iterator)
    finally:
        os.close(directory_fd)
    module_names = [name for name, _info in entries]
    require(SCHEDULER_REQUIRED_POLICY_MODULES.issubset(module_names), "Slurm policy modules are incomplete")
    for name, entry_info in entries:
        require(name not in ("", ".", "..") and "/" not in name, "Slurm policy entry name differs")
        require(stat.S_ISREG(entry_info.st_mode), f"Slurm policy entry is not regular: {name}")
    files: dict[str, Any] = {}
    names = [*SCHEDULER_POLICY_FILES, *(f"{SCHEDULER_POLICY_DIRECTORY}/{name}" for name in module_names)]
    for name in names:
        _payload, observation = _root_owned_regular_observation(policy_root / name, f"Slurm policy {name}")
        files[name] = observation
    return {"files": files, "tree_sha256": stable_hash(files)}


def capture_scheduler_control_plane_bundle(execution: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    contract = execution.get("scheduler_control_plane") or CONTROL_PLANE_CONTRACT
    require(contract == CONTROL_PLANE_CONTRACT, "scheduler control-plane contract differs")
    config_path = Path(str(contract["slurm_conf"]))
    payload_before, config_before = _root_owned_regular_observation(config_path, "Slurm configuration")
    critical = parse_slurm_config(payload_before, contract)
    policy_before = _scheduler_policy_observation(config_path)
    clients = {
        name: _resolved_executable_observation(Path(str(execution[name])), name)
        for name in ("sbatch", "squeue", "scontrol", "scancel")
    }
    payload_after, config_after = _root_owned_regular_observation(config_path, "Slurm configuration revalidation")
    policy_after = _scheduler_policy_observation(config_path)
    require(payload_after == payload_before and config_after == config_before, "Slurm configuration changed during capture")
    require(policy_after == policy_before, "Slurm policy changed during capture")
    return payload_before, {
        "schema_version": 1,
        "contract": contract,
        "trust_model": SCHEDULER_TRUST_MODEL,
        "config": config_before,
        "critical": critical,
        "cli_filter_policy": policy_before,
        "clients": clients,
    }


def capture_scheduler_control_plane(execution: Mapping[str, Any]) -> dict[str, Any]:
    return capture_scheduler_control_plane_bundle(execution)[1]


def scheduler_fallback_binding(payload: bytes, observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "exact_accepted_job_reconciliation_and_cancellation_only_never_submission",
        "encoding": "base64",
        "payload_base64": base64.b64encode(payload).decode("ascii"),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "source_control_plane": dict(observation),
    }


def boundary_from_submission_contract(
    execution: Mapping[str, Any],
    contract: Mapping[str, Any],
    *,
    runner: Runner = default_runner,
) -> SchedulerBoundary:
    """Reconstruct canonical+sealed-fallback scheduler authority after a crash."""

    value = contract.get("scheduler_fallback")
    require(isinstance(value, dict) and set(value) == {
        "schema_version", "purpose", "encoding", "payload_base64", "sha256",
        "size", "source_control_plane",
    }, "submission scheduler fallback schema differs")
    require(
        value.get("schema_version") == 1
        and value.get("purpose")
        == "exact_accepted_job_reconciliation_and_cancellation_only_never_submission"
        and value.get("encoding") == "base64"
        and isinstance(value.get("payload_base64"), str)
        and SHA256.fullmatch(str(value.get("sha256", ""))) is not None
        and isinstance(value.get("size"), int)
        and 0 < value["size"] <= 16 * 1024 * 1024,
        "submission scheduler fallback metadata differs",
    )
    try:
        payload = base64.b64decode(value["payload_base64"], validate=True)
    except (ValueError, UnicodeError) as exc:
        raise RuntimeContractError(f"scheduler fallback encoding differs: {exc}") from exc
    require(
        len(payload) == value["size"]
        and hashlib.sha256(payload).hexdigest() == value["sha256"],
        "scheduler fallback bytes differ",
    )
    expected = value.get("source_control_plane")
    require(isinstance(expected, dict) and expected == contract.get("scheduler_control_plane"), "scheduler fallback source observation differs")
    require(parse_slurm_config(payload, execution["scheduler_control_plane"]) == expected.get("critical"), "scheduler fallback critical config differs")
    return SchedulerBoundary(
        runner=runner,
        observer=lambda: capture_scheduler_control_plane(execution),
        expected=expected,
        fallback_payload=payload,
    )


@contextmanager
def _sealed_fallback_descriptor(payload: bytes) -> Iterator[int]:
    require(hasattr(os, "memfd_create"), "scheduler fallback requires memfd_create")
    descriptor = os.memfd_create(
        "treewm-exp24-slurm-conf",
        getattr(os, "MFD_CLOEXEC", 0) | getattr(os, "MFD_ALLOW_SEALING", 0),
    )
    try:
        offset = 0
        while offset < len(payload):
            written = os.write(descriptor, payload[offset:])
            require(written > 0, "scheduler fallback write stalled")
            offset += written
        os.fchmod(descriptor, 0o400)
        os.lseek(descriptor, 0, os.SEEK_SET)
        seals = (
            getattr(fcntl, "F_SEAL_SEAL", 0)
            | getattr(fcntl, "F_SEAL_SHRINK", 0)
            | getattr(fcntl, "F_SEAL_GROW", 0)
            | getattr(fcntl, "F_SEAL_WRITE", 0)
        )
        require(seals, "scheduler fallback seals are unavailable")
        fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, seals)
        require(fcntl.fcntl(descriptor, fcntl.F_GET_SEALS) & seals == seals, "scheduler fallback descriptor is not sealed")
        yield descriptor
    finally:
        os.close(descriptor)


class SchedulerBoundary:
    """Bind every scheduler call to the exact preclaim control-plane observation."""

    def __init__(
        self,
        *,
        runner: Runner,
        observer: Observer,
        expected: Mapping[str, Any],
        environment: Mapping[str, str] = CONTROL_ENVIRONMENT,
        fallback_payload: bytes | None = None,
    ) -> None:
        self.runner = runner
        self.observer = observer
        self.expected = dict(expected)
        self.environment = dict(environment)
        self.fallback_payload = fallback_payload
        self.calls: list[list[str]] = []
        self.recovery_events: list[dict[str, Any]] = []

    def call(self, command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        before = dict(self.observer())
        require(before == self.expected, "scheduler control plane differs before client call")
        completed: subprocess.CompletedProcess[str] | None = None
        try:
            completed = self.runner(tuple(command), cwd, self.environment, ())
        finally:
            after = dict(self.observer())
            require(after == before, "scheduler control plane changed during client call")
        self.calls.append(list(command))
        assert completed is not None
        return completed

    def recovery_call(self, command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        """Use canonical boundary, then sealed original config only on boundary failure."""

        require(Path(str(command[0])).name in {"squeue", "scancel"}, "fallback is restricted to squeue/scancel")
        try:
            return self.call(command, cwd)
        except BaseException as canonical_exc:
            require(self.fallback_payload is not None, f"canonical scheduler boundary failed without fallback: {canonical_exc}")
            with _sealed_fallback_descriptor(self.fallback_payload) as descriptor:
                environment = {
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "LANG": "C",
                    "LC_ALL": "C",
                    "SLURM_CONF": f"/proc/self/fd/{descriptor}",
                }
                completed = self.runner(tuple(command), cwd, environment, (descriptor,))
            self.recovery_events.append(
                {
                    "command": list(command),
                    "mode": "sealed_original_config_fallback",
                    "canonical_error": repr(canonical_exc),
                    "fallback_sha256": hashlib.sha256(self.fallback_payload).hexdigest(),
                }
            )
            return completed


def _squeue_rows(completed: subprocess.CompletedProcess[str], name: str, comment: str) -> list[str]:
    require(completed.returncode == 0, f"cannot reconcile {name}: {completed.stderr.strip()}")
    expected_user = pwd.getpwuid(os.getuid()).pw_name
    values: set[str] = set()
    for raw in completed.stdout.splitlines():
        if not raw.strip():
            continue
        fields = [value.strip() for value in raw.split("|", 5)]
        require(len(fields) == 6, f"scheduler reconciliation row is malformed for {name}")
        parent_id, scheduler_row_id, actual_name, user, state, actual_comment = fields
        require(
            JOB_ID.fullmatch(parent_id) is not None
            and JOB_ID.fullmatch(scheduler_row_id) is not None,
            f"scheduler returned malformed parent/row ID for {name}",
        )
        require(actual_name == name and actual_comment == comment, f"scheduler transaction identity differs for {name}")
        require(user == expected_user and state, f"scheduler owner/state differs for {name}")
        values.add(parent_id)
    return sorted(values, key=int)


def _squeue_state_rows(
    completed: subprocess.CompletedProcess[str],
    name: str,
    comment: str,
) -> list[dict[str, str]]:
    require(completed.returncode == 0, f"cannot reconcile {name}: {completed.stderr.strip()}")
    expected_user = pwd.getpwuid(os.getuid()).pw_name
    rows: list[dict[str, str]] = []
    for raw in completed.stdout.splitlines():
        if not raw.strip():
            continue
        fields = [value.strip() for value in raw.split("|", 5)]
        require(len(fields) == 6, f"scheduler reconciliation row is malformed for {name}")
        parent_id, scheduler_row_id, actual_name, user, state, actual_comment = fields
        require(
            JOB_ID.fullmatch(parent_id) is not None
            and JOB_ID.fullmatch(scheduler_row_id) is not None,
            f"scheduler returned malformed parent/row ID for {name}",
        )
        require(actual_name == name and actual_comment == comment, f"scheduler transaction identity differs for {name}")
        require(user == expected_user and state, f"scheduler owner/state differs for {name}")
        rows.append({"parent_job_id": parent_id, "scheduler_row_id": scheduler_row_id, "state": state})
    return sorted(rows, key=lambda row: (int(row["parent_job_id"]), int(row["scheduler_row_id"])))


def reconcile_job_ids(
    boundary: SchedulerBoundary,
    execution: Mapping[str, Any],
    name: str,
    comment: str,
    cwd: Path,
) -> list[str]:
    command = [
        str(execution["squeue"]),
        "--noheader",
        f"--name={name}",
        f"--user={pwd.getpwuid(os.getuid()).pw_name}",
        "--format=%F|%A|%j|%u|%T|%k",
    ]
    return _squeue_rows(boundary.recovery_call(command, cwd), name, comment)


def submit_one(
    boundary: SchedulerBoundary,
    execution: Mapping[str, Any],
    command: Sequence[str],
    *,
    name: str,
    comment: str,
    cwd: Path,
) -> tuple[str, dict[str, Any]]:
    completed: subprocess.CompletedProcess[str] | None = None
    try:
        completed = boundary.call(command, cwd)
    except BaseException as exc:
        reconciled: list[str] = []
        reconciliation_error: str | None = None
        try:
            reconciled = reconcile_job_ids(boundary, execution, name, comment, cwd)
        except BaseException as reconcile_exc:
            reconciliation_error = repr(reconcile_exc)
        raise SchedulerTransactionError(
            f"scheduler boundary failed for {name}: {exc}; first reconciliation={reconciliation_error}",
            reconciled,
        ) from exc
    response = completed.stdout.strip()
    match = SBATCH_RESPONSE.fullmatch(response)
    parsed = [match.group("job_id")] if match is not None else []
    if completed.returncode != 0:
        try:
            reconciled = reconcile_job_ids(boundary, execution, name, comment, cwd)
        except BaseException as exc:
            raise SchedulerTransactionError(f"sbatch and reconciliation failed for {name}: {exc}", parsed) from exc
        raise SchedulerTransactionError(
            f"sbatch failed for {name}: {completed.stderr.strip()}",
            [*parsed, *reconciled],
        )
    reconciled = reconcile_job_ids(boundary, execution, name, comment, cwd)
    if match is None:
        if len(reconciled) != 1:
            raise SchedulerTransactionError(
                f"ambiguous sbatch response for {name}; reconciled {len(reconciled)} IDs",
                reconciled,
            )
        job_id = reconciled[0]
    else:
        job_id = match.group("job_id")
        if reconciled != [job_id]:
            raise SchedulerTransactionError(
                f"parseable sbatch ID did not reconcile exactly for {name}",
                [job_id, *reconciled],
            )
    return job_id, {
        "command": list(command),
        "response_mode": (
            "parsed_sbatch_stdout"
            if match is not None
            else "exact_name_comment_reconciliation_after_unparseable_stdout"
        ),
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "reconciled_job_ids": reconciled,
        "control_plane": boundary.expected,
    }


def _oneliner_records(stdout: str) -> list[dict[str, str]]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    require(lines, "scontrol returned no job record")
    records: list[dict[str, str]] = []
    for line in lines:
        fields: dict[str, str] = {}
        for token in line.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            require(key and key not in fields, f"scontrol field inventory differs: {key!r}")
            fields[key] = value
        require(fields, "scontrol returned an empty job record")
        records.append(fields)
    return records


def _oneliner_field(fields: Mapping[str, str], field: str) -> str:
    value = fields.get(field)
    require(isinstance(value, str) and value, f"scontrol field differs: {field}")
    return value


def _optional_oneliner_field(fields: Mapping[str, str], field: str) -> str | None:
    value = fields.get(field)
    require(value is None or (isinstance(value, str) and value), f"scontrol optional field differs: {field}")
    return value


def _array_task_set(expression: str, expected_throttle: int) -> set[int]:
    """Parse only Slurm's canonical integer/range array expression surface."""

    require(expression and expression.count("%") <= 1, "accepted array expression differs")
    if "%" in expression:
        body, throttle = expression.rsplit("%", 1)
        require(throttle == str(expected_throttle), "accepted array expression throttle differs")
    else:
        body = expression
    values: set[int] = set()
    for segment in body.split(","):
        match = re.fullmatch(r"([0-9]+)(?:-([0-9]+))?", segment)
        require(match is not None, "accepted array task expression differs")
        first = int(match.group(1))
        last = first if match.group(2) is None else int(match.group(2))
        require(first <= last, "accepted array range is descending")
        expanded = set(range(first, last + 1))
        require(values.isdisjoint(expanded), "accepted array task expression overlaps")
        values.update(expanded)
    require(values, "accepted array task expression is empty")
    return values


def _normalized_memory_values(raw: str) -> set[str]:
    """Return exact request plus Slurm's integral G-to-M normalization."""

    values = {raw}
    match = re.fullmatch(r"([1-9][0-9]*)G", raw)
    if match is not None:
        values.add(f"{int(match.group(1)) * 1024}M")
    return values


def expected_dependency_string(predecessor_job_id: str, predecessor_elements: int) -> str:
    require(JOB_ID.fullmatch(predecessor_job_id) is not None, "dependency job ID is malformed")
    suffix = "_*" if predecessor_elements > 1 else ""
    return f"afterok:{predecessor_job_id}{suffix}(unfulfilled)"


def validate_accepted_job_stdout(
    stdout: str,
    *,
    job_id: str,
    name: str,
    comment: str,
    predecessor_job_id: str | None,
    predecessor_elements: int | None,
    manifest: Mapping[str, Any],
    node: Mapping[str, Any],
    submit_command: Sequence[str],
    cwd: Path,
    root_lifecycle: str = "held",
) -> dict[str, Any]:
    """Validate the exact accepted Slurm shape from preserved oneliner bytes."""

    records = _oneliner_records(stdout)
    options = {
        value[2:].split("=", 1)[0]: value.split("=", 1)[1]
        for value in submit_command
        if value.startswith("--") and "=" in value
    }
    execution = manifest["execution"]
    gpu = str(node["name"]).startswith("train_") or node["name"] == "heldout_eval"
    requested_gpu_partitions = str(execution["gpu_partitions"])
    allowed_partitions = (
        [
            *requested_gpu_partitions.split(","),
            requested_gpu_partitions,
        ]
        if gpu
        else [str(execution["cpu_partition"])]
    )
    script = next((value for value in submit_command if value.endswith(".slurm")), None)
    require(script is not None, f"accepted script is absent for {name}")
    expected_elements = int(node["elements"])
    expected_task_ids = set(range(expected_elements)) if expected_elements > 1 else set()
    expected_throttle = 40
    observed_task_ids: set[int] = set()
    dependencies: set[str] = set()
    kill_policies: set[str | None] = set()
    states: set[str] = set()
    reasons: set[str | None] = set()
    scheduler_job_ids: set[str] = set()
    parent_id_records = 0
    for fields in records:
        actual_job_id = _oneliner_field(fields, "JobId")
        require(JOB_ID.fullmatch(actual_job_id) is not None, f"accepted scheduler row ID differs for {name}")
        require(actual_job_id not in scheduler_job_ids, f"accepted scheduler row ID is duplicated for {name}")
        scheduler_job_ids.add(actual_job_id)
        parent_id_records += int(actual_job_id == job_id)
        require(_oneliner_field(fields, "JobName") == name, "accepted job name differs")
        require(_oneliner_field(fields, "Comment") == comment, "accepted job comment differs")
        state = _oneliner_field(fields, "JobState")
        states.add(state)
        allowed_states = {"PENDING"}
        if predecessor_job_id is None and expected_elements > 1:
            require(root_lifecycle in {"held", "released"}, "root lifecycle expectation differs")
            if root_lifecycle == "released":
                allowed_states.update({"CONFIGURING", "RUNNING"})
        require(state in allowed_states, f"accepted job state differs for {name}: {state}")
        reason = _optional_oneliner_field(fields, "Reason")
        reasons.add(reason)
        if predecessor_job_id is None and expected_elements > 1:
            if root_lifecycle == "held":
                require(
                    state == "PENDING" and reason == "JobHeldUser",
                    "root job is not exactly user-held",
                )
            else:
                require(
                    reason not in {"JobHeldUser", "JobHeldAdmin"},
                    "released root remains held",
                )
        dependency = _oneliner_field(fields, "Dependency")
        kill_invalid = _optional_oneliner_field(fields, "KillOInInvalidDependent")
        dependencies.add(dependency)
        kill_policies.add(kill_invalid)
        if predecessor_job_id is None:
            require(dependency in {"(null)", "None"}, "root job unexpectedly has a dependency")
            require(kill_invalid in {None, "No", "N/A"}, "root job invalid-dependency policy differs")
        else:
            require(predecessor_elements is not None, "accepted predecessor geometry is absent")
            expected = expected_dependency_string(predecessor_job_id, predecessor_elements)
            require(dependency == expected, f"accepted dependency differs for {name}: {dependency!r} != {expected!r}")
            require(kill_invalid == "Yes", f"accepted invalid-dependency policy differs for {name}")
        require(_oneliner_field(fields, "Partition") in allowed_partitions, f"accepted partition differs for {name}")
        require(
            _oneliner_field(fields, "Account")
            == "edgeai_tao-ptm_image-foundation-model-clip",
            f"accepted account differs for {name}",
        )
        require(_oneliner_field(fields, "QOS") == "normal", f"accepted QoS differs for {name}")
        require(_oneliner_field(fields, "NumCPUs") == str(execution["cpus_per_task"]), f"accepted CPU count differs for {name}")
        require(_oneliner_field(fields, "CPUs/Task") == str(execution["cpus_per_task"]), f"accepted CPUs/task differs for {name}")
        require(_oneliner_field(fields, "NumNodes") in {"1", "1-1"}, f"accepted node count differs for {name}")
        require(_oneliner_field(fields, "NumTasks") == "1", f"accepted task count differs for {name}")
        require(_oneliner_field(fields, "MinMemoryNode") in _normalized_memory_values(str(execution["memory_per_task"])), f"accepted memory differs for {name}")
        require(_oneliner_field(fields, "TimeLimit") == str(execution["walltime"]), f"accepted time differs for {name}")
        require(_oneliner_field(fields, "WorkDir") == str(cwd), f"accepted workdir differs for {name}")
        require(_oneliner_field(fields, "Command") == script, f"accepted script differs for {name}")
        array_task = _optional_oneliner_field(fields, "ArrayTaskId")
        array_throttle = _optional_oneliner_field(fields, "ArrayTaskThrottle")
        array_job = _optional_oneliner_field(fields, "ArrayJobId")
        if expected_elements > 1:
            require(array_job == job_id, f"accepted array root differs for {name}")
            require(array_task is not None, f"accepted array task set is absent for {name}")
            tasks = _array_task_set(array_task, expected_throttle)
            require(observed_task_ids.isdisjoint(tasks), f"accepted array task records overlap for {name}")
            observed_task_ids.update(tasks)
            require(array_throttle == str(expected_throttle), f"accepted array throttle differs for {name}")
            output_values = {
                options["output"],
                options["output"].replace("%A", job_id).replace("%a", "4294967294"),
                *(options["output"].replace("%A", job_id).replace("%a", str(task)) for task in tasks),
            }
        else:
            nulls = {None, "N/A", "(null)"}
            require(len(records) == 1 and actual_job_id == job_id, f"accepted scalar job ID differs for {name}")
            require(array_job in nulls, f"scalar job became an array root: {name}")
            require(array_task in nulls, f"scalar job became an array: {name}")
            require(array_throttle in {*nulls, "0"}, f"scalar array throttle differs: {name}")
            output_values = {options["output"], options["output"].replace("%j", job_id)}
        accepted_output = _oneliner_field(fields, "StdOut")
        require(accepted_output in output_values, f"accepted output differs for {name}")
        require(_oneliner_field(fields, "StdErr") == accepted_output, f"accepted stderr path differs for {name}")
        require(_oneliner_field(fields, "Requeue") == ("1" if gpu else "0"), f"accepted requeue policy differs for {name}")
        tres_per_node = _optional_oneliner_field(fields, "TresPerNode")
        if gpu:
            require(tres_per_node == "gres:gpu:1", f"accepted GPU resources differ for {name}")
        else:
            require(tres_per_node in {None, "N/A", "(null)"}, f"CPU job acquired GPU resources: {name}")
    if expected_elements > 1:
        require(observed_task_ids == expected_task_ids, f"accepted array task coverage differs for {name}")
        require(parent_id_records <= 1, f"accepted array repeats its parent scheduler ID for {name}")
    require(len(dependencies) == 1, f"accepted dependency varies across {name}")
    if predecessor_job_id is None:
        normalized_kill = "disabled_or_absent"
    else:
        require(kill_policies == {"Yes"}, f"accepted dependency policy varies across {name}")
        normalized_kill = "Yes"
    return {
        "record_count": len(records),
        "array_task_ids": sorted(observed_task_ids),
        "states": sorted(states),
        "reasons": sorted("<absent>" if value is None else value for value in reasons),
        "root_lifecycle": root_lifecycle if predecessor_job_id is None else "not_root",
        "dependency": next(iter(dependencies)),
        "kill_on_invalid_dependency": normalized_kill,
    }


def observe_accepted_job(
    boundary: SchedulerBoundary,
    execution: Mapping[str, Any],
    *,
    job_id: str,
    name: str,
    comment: str,
    predecessor_job_id: str | None,
    predecessor_elements: int | None,
    manifest: Mapping[str, Any],
    node: Mapping[str, Any],
    submit_command: Sequence[str],
    cwd: Path,
    root_lifecycle: str = "held",
) -> dict[str, Any]:
    command = [str(execution["scontrol"]), "show", "job", job_id, "--oneliner"]
    completed = boundary.call(command, cwd)
    require(completed.returncode == 0, f"cannot observe accepted job {job_id}: {completed.stderr.strip()}")
    normalized = validate_accepted_job_stdout(
        completed.stdout,
        job_id=job_id,
        name=name,
        comment=comment,
        predecessor_job_id=predecessor_job_id,
        predecessor_elements=predecessor_elements,
        manifest=manifest,
        node=node,
        submit_command=submit_command,
        cwd=cwd,
        root_lifecycle=root_lifecycle,
    )
    return {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "dependency": normalized["dependency"],
        "kill_on_invalid_dependency": normalized["kill_on_invalid_dependency"],
        "normalized_shape": normalized,
        "control_plane": boundary.expected,
    }


def _defined_sbatch_options(stderr: str, label: str) -> dict[str, str]:
    lines = stderr.splitlines()
    try:
        start = lines.index("sbatch: defined options")
        end = lines.index("sbatch: end of defined options", start + 1)
    except ValueError as exc:
        raise RuntimeContractError(f"{label} scheduler test omitted defined options") from exc
    options: dict[str, str] = {}
    for line in lines[start + 1 : end]:
        match = re.fullmatch(r"sbatch: ([a-z0-9-]+)\s+:\s+(.*)", line)
        if match is None:
            continue
        key, value = match.groups()
        require(key not in options, f"{label} scheduler test duplicated {key}")
        options[key] = value
    return options


def scheduler_test_only(
    boundary: SchedulerBoundary,
    command: Sequence[str],
    *,
    label: str,
    cwd: Path,
    dependency: str | None,
    expected_array: str | None,
    expected_options: Mapping[str, str],
    expected_partitions: Sequence[str],
    expected_processors: int,
) -> dict[str, Any]:
    """Run and parse a zero-job ``sbatch --test-only`` decision."""

    test_command = [command[0], "-vvv", "--test-only", *command[1:]]
    completed = boundary.call(test_command, cwd)
    require(completed.returncode == 0, f"{label} sbatch --test-only failed: {completed.stderr.strip()}")
    require(not completed.stdout, f"{label} sbatch --test-only unexpectedly wrote stdout")
    options = _defined_sbatch_options(completed.stderr, label)
    exact_expected = dict(expected_options)
    exact_expected["test-only"] = "set"
    exact_expected["verbose"] = "3"
    require(options == exact_expected, f"{label} scheduler test options differ")
    require(options.get("dependency") == dependency if dependency is not None else "dependency" not in options, f"{label} test dependency differs")
    require(options.get("array") == expected_array if expected_array is not None else "array" not in options, f"{label} test array differs")
    decisions: list[tuple[str, str, str, str, str]] = []
    for line in completed.stderr.splitlines():
        match = re.fullmatch(
            r"sbatch: Job ([0-9]+) to start at (\S+) using ([0-9]+) processors on nodes (\S+) in partition (\S+)",
            line,
        )
        if match is not None:
            decisions.append(match.groups())
    require(len(decisions) == 1, f"{label} scheduler test decision differs")
    _synthetic, _start, processors, _nodes, partition = decisions[0]
    require(int(processors) == expected_processors, f"{label} scheduler processor decision differs")
    require(partition in expected_partitions, f"{label} scheduler partition decision differs")
    value = {
        "command": test_command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "defined_options": options,
        "decision": {
            "processors": int(processors),
            "partition": partition,
        },
        "control_plane": boundary.expected,
        "zero_job": True,
    }
    validate_scheduler_test_evidence(
        value,
        command=command,
        label=label,
        dependency=dependency,
        expected_array=expected_array,
        expected_options=expected_options,
        expected_partitions=expected_partitions,
        expected_processors=expected_processors,
        expected_control_plane=boundary.expected,
    )
    return value


def validate_scheduler_test_evidence(
    value: Mapping[str, Any],
    *,
    command: Sequence[str],
    label: str,
    dependency: str | None,
    expected_array: str | None,
    expected_options: Mapping[str, str],
    expected_partitions: Sequence[str],
    expected_processors: int,
    expected_control_plane: Mapping[str, Any],
) -> None:
    require(set(value) == {
        "command", "returncode", "stdout", "stderr", "defined_options",
        "decision", "control_plane", "zero_job",
    }, f"{label} scheduler test evidence schema differs")
    expected_command = [str(command[0]), "-vvv", "--test-only", *[str(item) for item in command[1:]]]
    require(
        value.get("command") == expected_command
        and value.get("returncode") == 0
        and value.get("stdout") == ""
        and isinstance(value.get("stderr"), str)
        and value.get("control_plane") == dict(expected_control_plane)
        and value.get("zero_job") is True,
        f"{label} scheduler test evidence identity differs",
    )
    options = _defined_sbatch_options(str(value["stderr"]), label)
    exact_expected = dict(expected_options)
    exact_expected.update({"test-only": "set", "verbose": "3"})
    require(options == exact_expected == value.get("defined_options"), f"{label} scheduler test options evidence differs")
    require(options.get("dependency") == dependency if dependency is not None else "dependency" not in options, f"{label} scheduler test dependency evidence differs")
    require(options.get("array") == expected_array if expected_array is not None else "array" not in options, f"{label} scheduler test array evidence differs")
    decisions: list[tuple[str, str, str, str, str]] = []
    for line in str(value["stderr"]).splitlines():
        match = re.fullmatch(
            r"sbatch: Job ([0-9]+) to start at (\S+) using ([0-9]+) processors on nodes (\S+) in partition (\S+)",
            line,
        )
        if match is not None:
            decisions.append(match.groups())
    require(len(decisions) == 1, f"{label} scheduler decision evidence differs")
    _synthetic, _start, processors, _nodes, partition = decisions[0]
    require(
        value.get("decision") == {"processors": int(processors), "partition": partition}
        and int(processors) == expected_processors
        and partition in expected_partitions,
        f"{label} scheduler decision evidence does not match policy",
    )


def expected_test_options(
    manifest: Mapping[str, Any],
    record: Mapping[str, Any],
    *,
    dependency: str | None,
) -> tuple[dict[str, str], list[str]]:
    execution = manifest["execution"]
    node = record["node"]
    command_options = {
        value[2:].split("=", 1)[0]: value.split("=", 1)[1]
        for value in record["command"]
        if value.startswith("--") and "=" in value
    }
    gpu = str(node["name"]).startswith("train_") or node["name"] == "heldout_eval"
    partition = execution["gpu_partitions"] if gpu else execution["cpu_partition"]
    expected = {
        "account": "edgeai_tao-ptm_image-foundation-model-clip",
        "comment": command_options["comment"],
        "cpus-per-task": str(execution["cpus_per_task"]),
        "export": "NONE",
        "job-name": command_options["job-name"],
        "mem": str(execution["memory_per_task"]),
        "nodes": "1",
        "ntasks-per-node": "1",
        "open-mode": "a",
        "output": command_options["output"],
        "parsable": "set",
        "partition": str(partition),
        "qos": "normal",
        "time": str(execution["walltime"]),
    }
    if int(node["elements"]) > 1:
        expected["array"] = (
            str(execution["training_array"])
            if int(node["elements"]) == 40
            else str(execution["heldout_array"])
        )
    if dependency is not None:
        expected["dependency"] = dependency
        expected["kill-on-invalid-dep"] = "yes"
    if node["name"] == "train_2000":
        expected["hold"] = "set"
    if gpu:
        expected.update(
            {
                "gpus-per-node": str(execution["gpus_per_task"]),
                "requeue": "requeue",
                "signal": f"B:USR1@{execution['signal_seconds_before_end']}",
            }
        )
    else:
        expected["no-requeue"] = "no-requeue"
    return expected, str(partition).split(",")


def _parse_controller_configuration(stdout: str, contract: Mapping[str, Any]) -> dict[str, Any]:
    values: dict[str, str] = {}
    hosts: dict[int, str] = {}
    for raw in stdout.splitlines():
        match = re.match(r"^([^=]+?)\s*=\s*(.*?)\s*$", raw)
        if match is None:
            continue
        key, value = match.groups()
        host = re.fullmatch(r"SlurmctldHost\[([0-9]+)\]", key.strip())
        if host is not None:
            index = int(host.group(1))
            require(index not in hosts, "controller duplicates a host")
            hosts[index] = value
        elif key.strip() in {
            "AuthType", "CliFilterPlugins", "ClusterName", "GresTypes",
            "JobSubmitPlugins", "SlurmctldPort",
        }:
            normalized = key.strip()
            require(normalized not in values, f"controller duplicates {normalized}")
            values[normalized] = value
    expected = {
        "AuthType": contract["auth_type"],
        "CliFilterPlugins": ",".join(contract["cli_filter_plugins"]),
        "ClusterName": contract["cluster_name"],
        "GresTypes": ",".join(contract["gres_types"]),
        "JobSubmitPlugins": ",".join(contract["job_submit_plugins"]),
        "SlurmctldPort": str(contract["slurmctld_port"]),
    }
    require(values == expected, "live controller configuration differs")
    require(hosts == dict(enumerate(contract["slurmctld_hosts"])), "live controller hosts differ")
    return {**values, "SlurmctldHosts": [hosts[index] for index in sorted(hosts)]}


def scheduler_preclaim_test(
    manifest: Mapping[str, Any],
    *,
    repo_root: Path,
    boundary: SchedulerBoundary,
) -> dict[str, Any]:
    """Read-only controller + eleven-role site-policy test before the claim."""

    import campaign

    execution = manifest["execution"]
    controller = boundary.call([str(execution["scontrol"]), "show", "config"], repo_root)
    require(controller.returncode == 0, f"scontrol show config failed: {controller.stderr.strip()}")
    controller_contract = _parse_controller_configuration(
        controller.stdout,
        execution["scheduler_control_plane"],
    )
    nodes = campaign.scheduler_dag(manifest)
    fake_ids = {node.name: str(900_000_000 + index) for index, node in enumerate(nodes)}
    dummy_root = repo_root / "exp24-scheduler-test-never-created"
    commands = scheduler_commands(
        manifest,
        repo_root,
        dummy_root,
        "0" * 64,
        fake_ids,
    )
    require(len(commands) == 11, "preclaim command topology differs")
    tests: list[dict[str, Any]] = []
    for record in commands:
        # Preclaim validates script/resources/array without an external job ID.  The
        # exact accepted predecessor is tested again immediately before each real
        # dependent submission and observed again immediately after acceptance.
        stripped = [
            value for value in record["command"]
            if not value.startswith("--dependency=") and value != "--kill-on-invalid-dep=yes"
        ]
        expected_array = None
        elements = int(record["node"]["elements"])
        if elements == 40:
            expected_array = execution["training_array"]
        elif elements == 200:
            expected_array = execution["heldout_array"]
        tests.append(
            scheduler_test_only(
                boundary,
                stripped,
                label=f"preclaim {record['node']['name']}",
                cwd=repo_root,
                dependency=None,
                expected_array=expected_array,
                expected_options=expected_test_options(
                    manifest,
                    record,
                    dependency=None,
                )[0],
                expected_partitions=expected_test_options(
                    manifest,
                    record,
                    dependency=None,
                )[1],
                expected_processors=int(execution["cpus_per_task"]),
            )
        )
    return {
        "schema_version": 1,
        "status": "eleven_node_zero_job_preclaim_verified",
        "controller": controller_contract,
        "tests": tests,
        "test_count": len(tests),
        "scheduler_jobs_created": 0,
        "control_plane": boundary.expected,
    }


def cancel_exact(
    boundary: SchedulerBoundary,
    execution: Mapping[str, Any],
    job_ids: Sequence[str],
    cwd: Path,
) -> dict[str, Any]:
    exact: list[str] = []
    for value in job_ids:
        text = str(value)
        require(JOB_ID.fullmatch(text) is not None, f"refusing non-exact cancellation target: {text!r}")
        if text not in exact:
            exact.append(text)
    require(exact, "exact cancellation target set is empty")
    # Exact receipt/reconciled IDs only.  --quiet makes retries and already-complete
    # ancestors idempotent without broadening the cancellation target.
    command = [str(execution["scancel"]), "--quiet", *exact]
    completed = boundary.recovery_call(command, cwd)
    require(completed.returncode == 0, f"exact cancellation failed: {completed.stderr.strip()}")
    return {
        "command": command,
        "job_ids": exact,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "control_plane": boundary.expected,
    }


def post_cancel_reconciliation(
    boundary: SchedulerBoundary,
    execution: Mapping[str, Any],
    *,
    job_names: Mapping[str, str],
    comment: str,
    cwd: Path,
    submission_root: Path,
    max_attempts: int = 12,
    delay_seconds: float = 0.25,
    sleeper: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Boundedly poll exact names; terminal/not-found is cancellation completion."""

    require(1 <= max_attempts <= 120 and 0.0 <= delay_seconds <= 5.0, "cancel polling bounds differ")
    terminal = {
        "CANCELLED", "COMPLETED", "FAILED", "TIMEOUT", "NODE_FAIL",
        "PREEMPTED", "BOOT_FAIL", "OUT_OF_MEMORY", "DEADLINE",
    }
    attempts: list[dict[str, Any]] = []
    for attempt in range(1, max_attempts + 1):
        observed: dict[str, list[dict[str, str]]] = {}
        active: dict[str, list[dict[str, str]]] = {}
        for role, name in job_names.items():
            command = [
                str(execution["squeue"]), "--noheader", f"--name={name}",
                f"--user={pwd.getpwuid(os.getuid()).pw_name}",
                "--format=%F|%A|%j|%u|%T|%k",
            ]
            rows = _squeue_state_rows(boundary.recovery_call(command, cwd), name, comment)
            observed[role] = rows
            nonterminal = [row for row in rows if row["state"].upper() not in terminal]
            if nonterminal:
                active[role] = nonterminal
        record = {"attempt": attempt, "observed": observed, "active": active}
        attempts.append(record)
        _recovery_evidence(submission_root, "POST_CANCEL_RECONCILIATION", record)
        if not active:
            return {
                "status": "terminal_or_absent",
                "attempt_count": attempt,
                "attempts": attempts,
            }
        if attempt < max_attempts:
            sleeper(delay_seconds)
    return {
        "status": "convergence_pending",
        "attempt_count": max_attempts,
        "attempts": attempts,
        "active": attempts[-1]["active"],
    }


def append_journal(
    submission_root: Path,
    ordinal: int,
    event: str,
    payload: Mapping[str, Any],
) -> Path:
    require(0 <= ordinal <= 9999 and re.fullmatch(r"[A-Z0-9_]+", event) is not None, "journal identity is invalid")
    value = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "ordinal": ordinal,
        "event": event,
        "payload": dict(payload),
    }
    path = submission_root / "journal" / f"{ordinal:04d}_{event}.json"
    exclusive_json(path, value)
    return path


def _node_job_names(nodes: Sequence[Any], token: str) -> dict[str, str]:
    require(re.fullmatch(r"[0-9a-f]{16}", token) is not None, "scheduler token is malformed")
    return {node.name: f"exp24-{token}-{node.name.replace('_', '-')}" for node in nodes}


def _node_script_and_arguments(node: Any, snapshot_root: Path, submission_root: Path, submission_sha256: str) -> tuple[Path, list[str]]:
    package = snapshot_root / PACKAGE_RELATIVE
    common = [str(snapshot_root), str(submission_root), submission_sha256, node.name]
    if node.name.startswith("train_"):
        return package / "train.slurm", [*common, node.name.removeprefix("train_")]
    if node.name.startswith("gate_"):
        return package / "gate.slurm", [*common, node.name.removeprefix("gate_")]
    if node.name == "heldout_eval":
        return package / "final_eval.slurm", common
    if node.name == "aggregate":
        return package / "aggregate.slurm", common
    if node.name == "formal_report":
        return package / "report.slurm", common
    raise RuntimeContractError(f"unknown DAG role: {node.name}")


def scheduler_commands(
    manifest: Mapping[str, Any],
    snapshot_root: Path,
    submission_root: Path,
    submission_sha256: str,
    job_ids: Mapping[str, str] | None = None,
    *,
    through_index: int | None = None,
) -> list[dict[str, Any]]:
    """Derive an exact reachable prefix (or all eleven with complete IDs)."""

    import campaign

    require(SHA256.fullmatch(submission_sha256) is not None, "submission SHA256 is malformed")
    nodes = campaign.scheduler_dag(manifest)
    require(len(nodes) == 11, "Exp24 scheduler DAG must contain exactly eleven nodes")
    if through_index is None:
        limit = len(nodes) - 1
    else:
        require(
            isinstance(through_index, int)
            and not isinstance(through_index, bool)
            and 0 <= through_index < len(nodes),
            "scheduler command prefix index differs",
        )
        limit = through_index
    names = _node_job_names(nodes, submission_sha256[:16])
    execution = manifest["execution"]
    comment = f"treewm-exp24:{submission_sha256}"
    known = {} if job_ids is None else dict(job_ids)
    records: list[dict[str, Any]] = []
    prior_by_name = {node.name: node for node in nodes}
    for node in nodes[: limit + 1]:
        script, arguments = _node_script_and_arguments(node, snapshot_root, submission_root, submission_sha256)
        command = [
            str(execution["sbatch"]),
            "--parsable",
            "--export=NONE",
            f"--job-name={names[node.name]}",
            f"--comment={comment}",
            f"--output={submission_root / 'logs' / (node.name + '_%A_%a.out' if node.elements > 1 else node.name + '_%j.out')}",
        ]
        if node.elements > 1:
            array = execution["training_array"] if node.elements == 40 else execution["heldout_array"]
            command.append(f"--array={array}")
        if node.name == "train_2000":
            command.append("--hold")
        predecessor_id: str | None = None
        predecessor_elements: int | None = None
        if node.dependency is not None:
            predecessor_id = known.get(node.dependency)
            require(predecessor_id is not None, f"dependency ID is unavailable for {node.name}")
            require(JOB_ID.fullmatch(predecessor_id) is not None, f"dependency ID is malformed for {node.name}")
            predecessor_elements = int(prior_by_name[node.dependency].elements)
            command.extend(
                [
                    f"--dependency=afterok:{predecessor_id}",
                    "--kill-on-invalid-dep=yes",
                ]
            )
        command.extend([str(script), *arguments])
        records.append(
            {
                "node": asdict(node),
                "job_name": names[node.name],
                "comment": comment,
                "command": command,
                "predecessor_job_id": predecessor_id,
                "predecessor_elements": predecessor_elements,
            }
        )
    return records


def _preclaim_directories(root: Path) -> tuple[bool, bool]:
    descriptor = open_absolute_directory(root, "private preclaim root")
    try:
        info = os.fstat(descriptor)
        require(
            info.st_uid == os.getuid()
            and info.st_gid == os.getgid()
            and stat.S_IMODE(info.st_mode) == 0o700,
            "private preclaim root identity/mode differs",
        )
        with os.scandir(descriptor) as iterator:
            entries = {entry.name: entry.stat(follow_symlinks=False) for entry in iterator}
    finally:
        os.close(descriptor)
    require(set(entries) <= {"journal", "logs"}, "private preclaim tree contains an unexpected entry")
    for name, child in entries.items():
        require(
            stat.S_ISDIR(child.st_mode)
            and child.st_uid == os.getuid()
            and child.st_gid == os.getgid()
            and stat.S_IMODE(child.st_mode) == 0o700,
            f"private preclaim {name} identity/mode differs",
        )
    return "journal" in entries, "logs" in entries


def _remove_empty_preclaim(root: Path) -> None:
    has_journal, has_logs = _preclaim_directories(root)
    for name, present in (("journal", has_journal), ("logs", has_logs)):
        if not present:
            continue
        child_fd = open_absolute_directory(root / name, f"empty preclaim {name}")
        try:
            with os.scandir(child_fd) as iterator:
                require(not list(iterator), f"private preclaim {name} is not empty")
        finally:
            os.close(child_fd)
        os.rmdir(root / name)
    parent_fd = open_absolute_directory(root.parent, "private preclaim cleanup parent")
    try:
        os.rmdir(root.name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        os.close(parent_fd)


def _complete_private_preclaim(root: Path) -> bool:
    has_journal, has_logs = _preclaim_directories(root)
    if not has_journal:
        return False
    repair_publication_residues(
        root / "journal",
        allowed_final=re.compile(r"0000_CLAIMED\.json"),
    )
    journal_fd = open_absolute_directory(root / "journal", "private preclaim journal")
    try:
        with os.scandir(journal_fd) as iterator:
            names = sorted(entry.name for entry in iterator)
    finally:
        os.close(journal_fd)
    if names != ["0000_CLAIMED.json"] or not has_logs:
        return False
    logs_fd = open_absolute_directory(root / "logs", "private preclaim logs")
    try:
        with os.scandir(logs_fd) as iterator:
            require(not list(iterator), "private preclaim logs are not empty")
    finally:
        os.close(logs_fd)
    claimed, _digest = authenticated_immutable_json(root / "journal" / "0000_CLAIMED.json", "private transaction claim")
    payload = claimed.get("payload")
    require(
        set(claimed) == {"schema_version", "campaign_id", "ordinal", "event", "payload"}
        and claimed.get("schema_version") == 1
        and claimed.get("campaign_id") == CAMPAIGN_ID
        and claimed.get("ordinal") == 0
        and claimed.get("event") == "CLAIMED"
        and isinstance(payload, dict)
        and set(payload) == {"claim_token", "pid", "created_ns"}
        and re.fullmatch(r"[0-9a-f]{64}", str(payload.get("claim_token", ""))) is not None
        and isinstance(payload.get("pid"), int)
        and isinstance(payload.get("created_ns"), int),
        "private transaction claim differs",
    )
    return True


def begin_transaction(submission_root: Path, claim_token: str) -> None:
    """Build the initial namespace privately, then claim its final name atomically."""

    require(re.fullmatch(r"[0-9a-f]{64}", claim_token) is not None, "claim token is malformed")
    require(submission_root.is_absolute(), "submission root must be absolute")
    parent_fd = open_absolute_directory(submission_root.parent, "submission claim parent")
    private_name = f".{submission_root.name}.PRECLAIM"
    private_root = submission_root.parent / private_name
    try:
        try:
            os.stat(submission_root.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise FileExistsError(errno.EEXIST, "submission namespace is already claimed", str(submission_root))
        try:
            os.stat(private_name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            _mkdir_exact(private_root, 0o700, "private submission claim")
        if not _complete_private_preclaim(private_root):
            has_journal, has_logs = _preclaim_directories(private_root)
            if has_journal:
                journal_fd = open_absolute_directory(private_root / "journal", "incomplete preclaim journal")
                try:
                    with os.scandir(journal_fd) as iterator:
                        require(not list(iterator), "incomplete private claim contains a published journal")
                finally:
                    os.close(journal_fd)
            if not has_journal:
                _mkdir_exact(private_root / "journal", 0o700, "private claim journal")
            if not has_logs:
                _mkdir_exact(private_root / "logs", 0o700, "private claim logs")
            append_journal(
                private_root,
                0,
                "CLAIMED",
                {"claim_token": claim_token, "pid": os.getpid(), "created_ns": time.time_ns()},
            )
            require(_complete_private_preclaim(private_root), "private transaction claim did not complete")
        _fsync_directory(private_root / "journal")
        _fsync_directory(private_root / "logs")
        _fsync_directory(private_root)
        _rename_noreplace(parent_fd, private_name, submission_root.name)
    finally:
        os.close(parent_fd)


def submission_contract(
    manifest: Mapping[str, Any],
    *,
    submission_root: Path,
    snapshot_root: Path,
    snapshot_inventory: Mapping[str, str],
    control_plane: Mapping[str, Any],
    scheduler_preclaim: Mapping[str, Any],
    scheduler_fallback: Mapping[str, Any],
    interpreter_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    import campaign

    nodes = campaign.scheduler_dag(manifest)
    interpreter = (
        capture_interpreter_provenance(manifest)
        if interpreter_provenance is None
        else dict(interpreter_provenance)
    )
    validate_interpreter_provenance(manifest, interpreter)
    m2a_schema, m2a_schema_file_sha = campaign.load_m2a_schema()
    m2a_relative = str(PACKAGE_RELATIVE / "m2a_schema.json")
    negative_relative = str(PACKAGE_RELATIVE / "launch7_negative.binding.json")
    negative_value, negative_file_sha = authenticated_json(
        PACKAGE_DIR / "launch7_negative.binding.json",
        "checked-in Launch7 terminal-negative binding",
    )
    positive_relative = str(PACKAGE_RELATIVE / "accepted_engineering_pilot.binding.json")
    positive_value, positive_file_sha = authenticated_json(
        PACKAGE_DIR / "accepted_engineering_pilot.binding.json",
        "checked-in future accepted-pilot binding",
    )
    campaign.validate_launch7_negative_binding(
        negative_value,
        evidence_root=REPOSITORY_ROOT,
    )
    require(
        positive_value == accepted_engineering_pilot_placeholder(),
        "checked-in future accepted-pilot placeholder differs",
    )
    adapter_relative = str(PACKAGE_RELATIVE / "engineering_pilot_binder.py")
    _adapter_bytes, adapter_file_sha, _adapter_info = authenticated_regular_bytes(
        PACKAGE_DIR / "engineering_pilot_binder.py",
        "future accepted-pilot adapter interface",
    )
    adapter_runtime_relative = str(PACKAGE_RELATIVE / "runtime.py")
    _runtime_bytes, adapter_runtime_file_sha, _runtime_info = authenticated_regular_bytes(
        PACKAGE_DIR / "runtime.py",
        "future accepted-pilot adapter runtime dependency",
    )
    negative_evidence_relative = str(campaign.LAUNCH7_NEGATIVE_EVIDENCE_RELATIVE)
    adapter_description = engineering_pilot_adapter_description()
    require(
        snapshot_inventory.get(m2a_relative) == m2a_schema_file_sha
        and snapshot_inventory.get(negative_relative) == negative_file_sha
        and snapshot_inventory.get(negative_evidence_relative)
        == campaign.LAUNCH7_NEGATIVE_EVIDENCE_RAW_SHA256
        and snapshot_inventory.get(positive_relative) == positive_file_sha
        and snapshot_inventory.get(adapter_relative) == adapter_file_sha
        and snapshot_inventory.get(adapter_runtime_relative) == adapter_runtime_file_sha,
        "M2A schema or pilot prerequisite binding differs from snapshot inventory",
    )
    emergency_targets = {
        str(PACKAGE_RELATIVE / name): snapshot_inventory[str(PACKAGE_RELATIVE / name)]
        for name in EMERGENCY_DISPATCH_FILES
    }
    seed = {
        "schema_version": 1,
        "status": "prepared_scheduler_transaction",
        "campaign_id": CAMPAIGN_ID,
        "submission_root": str(submission_root),
        "snapshot_root": str(snapshot_root),
        "manifest_sha256": campaign.manifest_sha256(manifest),
        "m2a_schema": {
            "relative_path": m2a_relative,
            "file_sha256": m2a_schema_file_sha,
            "semantic_sha256": stable_hash(m2a_schema),
        },
        "interpreter_provenance": interpreter,
        "launch7_negative_binding": {
            "relative_path": negative_relative,
            "file_sha256": negative_file_sha,
            "semantic_sha256": stable_hash(negative_value),
            "status": negative_value.get("status"),
            "accepted": negative_value.get("accepted"),
            "reusable": negative_value.get("reusable"),
            "formal_submission_allowed": negative_value.get("formal_submission_allowed"),
            "negative_binding_sha256": negative_value.get("negative_binding_sha256"),
            "evidence_file_sha256": negative_value["evidence"]["raw_file_sha256"],
            "evidence_canonical_sha256": negative_value["evidence"][
                "canonical_json_sha256"
            ],
            "evidence_git_commit": negative_value["evidence"]["evidence_git_commit"],
        },
        "engineering_pilot_adapter_interface": {
            "relative_path": adapter_relative,
            "adapter_file_sha256": adapter_file_sha,
            "adapter_runtime_file_sha256": adapter_runtime_file_sha,
            "adapter_description_sha256": stable_hash(adapter_description),
            "adapter_state": adapter_description["adapter_state"],
            "expected_campaign_id": adapter_description["expected_campaign_id"],
            "forbidden_positive_campaign_id": adapter_description[
                "forbidden_positive_campaign_id"
            ],
            "frozen_source_commit": adapter_description["frozen_source_commit"],
            "frozen_protocol_sha256": adapter_description[
                "frozen_protocol_sha256"
            ],
            "frozen_source_inventory_sha256": adapter_description[
                "frozen_source_inventory_sha256"
            ],
            "frozen_source_file_count": adapter_description[
                "frozen_source_file_count"
            ],
        },
        "accepted_engineering_pilot_binding": {
            "relative_path": positive_relative,
            "file_sha256": positive_file_sha,
            "semantic_sha256": stable_hash(positive_value),
            "status": positive_value.get("status"),
            "formal_submission_allowed": positive_value.get("formal_submission_allowed"),
            "adapter_file_sha256": positive_value.get("adapter_file_sha256"),
            "adapter_runtime_file_sha256": positive_value.get(
                "adapter_runtime_file_sha256"
            ),
            "adapter_description_sha256": positive_value.get(
                "adapter_description_sha256"
            ),
            "report_commit_file_sha256": positive_value.get(
                "report_commit_file_sha256"
            ),
            "binding_sha256": positive_value.get("binding_sha256"),
        },
        "snapshot_inventory": dict(snapshot_inventory),
        "snapshot_inventory_sha256": stable_hash(snapshot_inventory),
        # This deliberately small, versioned envelope is the compatibility
        # boundary used by the stdlib-only live cancel launcher.  Emergency
        # cancellation/recovery authenticates these four snapshot files and
        # the receipt/journals, but does not depend on unrelated training
        # source remaining readable.
        "emergency_dispatch": {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "package_relative": str(PACKAGE_RELATIVE),
            "python": str(manifest["paths"]["python"]),
            "targets": emergency_targets,
            "policy": "stable_v1_minimal_snapshot_cancel_recover_capsule",
        },
        "dag": [asdict(node) for node in nodes],
        "scheduler_control_plane": dict(control_plane),
        "scheduler_preclaim": dict(scheduler_preclaim),
        "scheduler_fallback": dict(scheduler_fallback),
    }
    seed["contract_body_sha256"] = stable_hash(seed)
    return seed


def validate_submission_contract(
    contract: Mapping[str, Any],
    *,
    submission_root: Path,
    digest: str,
    verify_runtime_authority: bool = True,
) -> None:
    """Validate the exact contract; emergency mode needs only its v1 capsule."""

    require(set(contract) == SUBMISSION_CONTRACT_KEYS, "submission contract schema differs")
    body = dict(contract)
    claimed_body_sha = body.pop("contract_body_sha256", None)
    require(
        SHA256.fullmatch(digest) is not None
        and claimed_body_sha == stable_hash(body),
        "submission contract body hash differs",
    )
    snapshot_root = submission_root / "source-snapshot" / "repo"
    require(
        contract.get("schema_version") == 1
        and contract.get("status") == "prepared_scheduler_transaction"
        and contract.get("campaign_id") == CAMPAIGN_ID
        and contract.get("submission_root") == str(submission_root)
        and contract.get("snapshot_root") == str(snapshot_root)
        and SHA256.fullmatch(str(contract.get("manifest_sha256", ""))) is not None,
        "submission contract identity/status differs",
    )
    inventory = contract.get("snapshot_inventory")
    require(
        isinstance(inventory, dict)
        and inventory
        and list(inventory) == sorted(inventory)
        and all(
            isinstance(key, str) and SHA256.fullmatch(str(value)) is not None
            for key, value in inventory.items()
        )
        and contract.get("snapshot_inventory_sha256") == stable_hash(inventory),
        "submission contract snapshot inventory differs",
    )
    verify_emergency_snapshot_files(snapshot_root, contract)
    if not verify_runtime_authority:
        return
    snapshot_schema, snapshot_schema_file_sha = authenticated_immutable_json(
        snapshot_root / PACKAGE_RELATIVE / "m2a_schema.json",
        "snapshot M2A authority schema",
    )
    import campaign

    campaign.validate_m2a_schema(snapshot_schema)
    require(
        set(snapshot_schema["schemas"]["submission_contract"])
        == SUBMISSION_CONTRACT_KEYS
        and set(snapshot_schema["schemas"]["submission_receipt"])
        == SUBMISSION_RECEIPT_KEYS
        and set(snapshot_schema["schemas"]["interpreter_provenance"])
        == INTERPRETER_PROVENANCE_KEYS
        and set(snapshot_schema["schemas"]["root_release_authorization"])
        == ROOT_RELEASE_AUTHORIZATION_KEYS
        and set(snapshot_schema["schemas"]["root_activation_result"])
        == ROOT_ACTIVATION_RESULT_KEYS,
        "M2A schema/runtime contract fields differ",
    )
    schema_binding = contract.get("m2a_schema")
    require(
        isinstance(schema_binding, dict)
        and set(schema_binding) == {"relative_path", "file_sha256", "semantic_sha256"}
        and schema_binding.get("relative_path") == str(PACKAGE_RELATIVE / "m2a_schema.json")
        and schema_binding.get("file_sha256") == snapshot_schema_file_sha
        == inventory.get(str(PACKAGE_RELATIVE / "m2a_schema.json"))
        and schema_binding.get("semantic_sha256") == stable_hash(snapshot_schema),
        "submission M2A authority-schema binding differs",
    )
    snapshot_manifest, _snapshot_manifest_file_sha = authenticated_immutable_json(
        snapshot_root / PACKAGE_RELATIVE / "manifest.json",
        "snapshot submission manifest",
    )
    campaign.validate_manifest(snapshot_manifest)
    interpreter = contract.get("interpreter_provenance")
    require(isinstance(interpreter, dict), "submission interpreter provenance is absent")
    validate_interpreter_provenance(snapshot_manifest, interpreter)
    negative_value, negative_file_sha = authenticated_immutable_json(
        snapshot_root / PACKAGE_RELATIVE / "launch7_negative.binding.json",
        "snapshot Launch7 terminal-negative binding",
    )
    negative_evidence_value, negative_evidence_file_sha = authenticated_immutable_json(
        snapshot_root / campaign.LAUNCH7_NEGATIVE_EVIDENCE_RELATIVE,
        "snapshot Launch7 terminal-negative provenance",
    )
    campaign.validate_launch7_negative_binding(
        negative_value,
        evidence_root=snapshot_root,
        evidence_value=negative_evidence_value,
        evidence_raw_sha256=negative_evidence_file_sha,
    )
    negative_binding = contract.get("launch7_negative_binding")
    require(
        isinstance(negative_binding, dict)
        and set(negative_binding)
        == {
            "relative_path", "file_sha256", "semantic_sha256", "status",
            "accepted", "reusable", "formal_submission_allowed",
            "negative_binding_sha256", "evidence_file_sha256",
            "evidence_canonical_sha256", "evidence_git_commit",
        }
        and negative_binding.get("relative_path")
        == str(PACKAGE_RELATIVE / "launch7_negative.binding.json")
        and negative_binding.get("file_sha256") == negative_file_sha
        == inventory.get(str(PACKAGE_RELATIVE / "launch7_negative.binding.json"))
        and negative_binding.get("semantic_sha256") == stable_hash(negative_value)
        and negative_binding.get("status") == negative_value.get("status")
        and negative_binding.get("accepted") is negative_value.get("accepted") is False
        and negative_binding.get("reusable") is negative_value.get("reusable") is False
        and negative_binding.get("formal_submission_allowed")
        is negative_value.get("formal_submission_allowed") is False
        and negative_binding.get("negative_binding_sha256")
        == negative_value.get("negative_binding_sha256")
        and negative_binding.get("evidence_file_sha256")
        == negative_value["evidence"]["raw_file_sha256"]
        == inventory.get(str(campaign.LAUNCH7_NEGATIVE_EVIDENCE_RELATIVE))
        and negative_binding.get("evidence_canonical_sha256")
        == negative_value["evidence"]["canonical_json_sha256"]
        and negative_binding.get("evidence_git_commit")
        == negative_value["evidence"]["evidence_git_commit"],
        "submission Launch7 terminal-negative binding leaf differs",
    )
    _adapter_bytes, adapter_file_sha, adapter_info = authenticated_regular_bytes(
        snapshot_root / PACKAGE_RELATIVE / "engineering_pilot_binder.py",
        "snapshot future accepted-pilot adapter interface",
    )
    _runtime_bytes, adapter_runtime_file_sha, adapter_runtime_info = authenticated_regular_bytes(
        snapshot_root / PACKAGE_RELATIVE / "runtime.py",
        "snapshot future accepted-pilot adapter runtime dependency",
    )
    adapter_description = engineering_pilot_adapter_description()
    frozen_launch8_inventory = launch8_verifier_source_inventory(
        snapshot_root,
        immutable=True,
    )
    require(
        all(inventory.get(path) == digest for path, digest in frozen_launch8_inventory.items()),
        "snapshot inventory does not bind the frozen Launch8 verifier closure",
    )
    adapter_binding = contract.get("engineering_pilot_adapter_interface")
    require(
        adapter_info.st_uid == os.getuid()
        and adapter_info.st_gid == os.getgid()
        and adapter_info.st_nlink == 1
        and stat.S_IMODE(adapter_info.st_mode) == 0o444
        and adapter_runtime_info.st_uid == os.getuid()
        and adapter_runtime_info.st_gid == os.getgid()
        and adapter_runtime_info.st_nlink == 1
        and stat.S_IMODE(adapter_runtime_info.st_mode) == 0o444
        and isinstance(adapter_binding, dict)
        and set(adapter_binding)
        == {
            "relative_path", "adapter_file_sha256",
            "adapter_runtime_file_sha256", "adapter_description_sha256",
            "adapter_state", "expected_campaign_id",
            "forbidden_positive_campaign_id", "frozen_source_commit",
            "frozen_protocol_sha256", "frozen_source_inventory_sha256",
            "frozen_source_file_count",
        }
        and adapter_binding.get("relative_path")
        == str(PACKAGE_RELATIVE / "engineering_pilot_binder.py")
        and adapter_binding.get("adapter_file_sha256") == adapter_file_sha
        == inventory.get(str(PACKAGE_RELATIVE / "engineering_pilot_binder.py"))
        and adapter_binding.get("adapter_runtime_file_sha256")
        == adapter_runtime_file_sha
        == inventory.get(str(PACKAGE_RELATIVE / "runtime.py"))
        and adapter_binding.get("adapter_description_sha256")
        == stable_hash(adapter_description)
        and adapter_binding.get("adapter_state") == ENGINEERING_PILOT_ADAPTER_STATE
        and adapter_binding.get("expected_campaign_id")
        == EXPECTED_ACCEPTED_PILOT_CAMPAIGN_ID
        and adapter_binding.get("forbidden_positive_campaign_id")
        == FORBIDDEN_POSITIVE_PILOT_CAMPAIGN_ID
        and adapter_binding.get("frozen_source_commit")
        == FROZEN_LAUNCH8_SOURCE_COMMIT
        and adapter_binding.get("frozen_protocol_sha256")
        == FROZEN_LAUNCH8_PROTOCOL_SHA256
        and adapter_binding.get("frozen_source_inventory_sha256")
        == FROZEN_LAUNCH8_VERIFIER_SOURCE_INVENTORY_SHA256
        and adapter_binding.get("frozen_source_file_count")
        == FROZEN_LAUNCH8_VERIFIER_SOURCE_FILE_COUNT,
        "submission future accepted-pilot adapter interface differs",
    )
    positive_value, positive_file_sha = authenticated_immutable_json(
        snapshot_root / PACKAGE_RELATIVE / "accepted_engineering_pilot.binding.json",
        "snapshot future accepted-pilot binding",
    )
    require(
        positive_value == accepted_engineering_pilot_placeholder(),
        "snapshot future accepted-pilot placeholder differs",
    )
    positive_binding = contract.get("accepted_engineering_pilot_binding")
    require(
        isinstance(positive_binding, dict)
        and set(positive_binding)
        == {
            "relative_path", "file_sha256", "semantic_sha256", "status",
            "formal_submission_allowed", "adapter_file_sha256",
            "adapter_runtime_file_sha256", "adapter_description_sha256",
            "report_commit_file_sha256",
            "binding_sha256",
        }
        and positive_binding.get("relative_path")
        == str(PACKAGE_RELATIVE / "accepted_engineering_pilot.binding.json")
        and positive_binding.get("file_sha256") == positive_file_sha
        == inventory.get(str(PACKAGE_RELATIVE / "accepted_engineering_pilot.binding.json"))
        and positive_binding.get("semantic_sha256") == stable_hash(positive_value)
        and positive_binding.get("status") == positive_value.get("status")
        and positive_binding.get("formal_submission_allowed")
        is positive_value.get("formal_submission_allowed") is False
        and positive_binding.get("adapter_file_sha256")
        is positive_value.get("adapter_file_sha256") is None
        and positive_binding.get("adapter_runtime_file_sha256")
        is positive_value.get("adapter_runtime_file_sha256") is None
        and positive_binding.get("adapter_description_sha256")
        is positive_value.get("adapter_description_sha256") is None
        and positive_binding.get("report_commit_file_sha256")
        is positive_value.get("report_commit_file_sha256") is None
        and positive_binding.get("binding_sha256")
        is positive_value.get("binding_sha256") is None,
        "submission future accepted-pilot binding leaf differs",
    )


def _receipt_from_jobs(
    manifest: Mapping[str, Any],
    submission_root: Path,
    snapshot_root: Path,
    submission_sha256: str,
    jobs: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    import campaign

    return {
        "schema_version": 1,
        "status": "submitted",
        "campaign_id": CAMPAIGN_ID,
        "submission_root": str(submission_root),
        "snapshot_root": str(snapshot_root),
        "submission_sha256": submission_sha256,
        "manifest_sha256": campaign.manifest_sha256(manifest),
        "jobs": [dict(row) for row in jobs],
        "dag_names": [node.name for node in campaign.scheduler_dag(manifest)],
        "training_array": "0-39%40",
        "heldout_array": "0-199%40",
    }


def _require_pre_receipt_budget(
    execution: Mapping[str, Any],
    started: float,
    phase: str,
) -> None:
    barrier = execution.get("queued_receipt_barrier_timeout_seconds")
    budget = execution.get("pre_receipt_transaction_budget_seconds")
    margin = execution.get("receipt_barrier_safety_margin_seconds")
    client_timeout = execution.get("scheduler_client_timeout_seconds")
    require(
        barrier == 900
        and budget == PRE_RECEIPT_TRANSACTION_BUDGET_SECONDS
        and margin == 300
        and client_timeout == SCHEDULER_CLIENT_TIMEOUT_SECONDS
        and budget + margin == barrier
        and margin >= 2 * client_timeout,
        "pre-receipt scheduler/barrier timing contract differs",
    )
    elapsed = time.monotonic() - started
    require(
        0.0 <= elapsed <= budget,
        f"pre-receipt transaction exceeded its {budget}s budget at {phase}",
    )


def submit_dag_transaction(
    manifest: Mapping[str, Any],
    *,
    submission_root: Path,
    snapshot_root: Path,
    submission_sha256: str,
    boundary: SchedulerBoundary,
) -> dict[str, Any]:
    """Submit all eleven nodes or reconcile/cancel every accepted exact ID."""

    import campaign

    nodes = campaign.scheduler_dag(manifest)
    require(len(nodes) == 11, "transaction topology is not eleven nodes")
    execution = manifest["execution"]
    jobs_by_role: dict[str, str] = {}
    accepted: list[dict[str, Any]] = []
    active_role: str | None = None
    ready_to_commit = False
    transaction_started = time.monotonic()
    try:
        for index, node in enumerate(nodes):
            active_role = node.name
            _require_pre_receipt_budget(execution, transaction_started, f"before_{node.name}_test")
            record = scheduler_commands(
                manifest,
                snapshot_root,
                submission_root,
                submission_sha256,
                jobs_by_role,
                through_index=index,
            )[index]
            dependency = (
                None
                if node.dependency is None
                else f"afterok:{record['predecessor_job_id']}"
            )
            expected_array = None
            if node.elements == 40:
                expected_array = execution["training_array"]
            elif node.elements == 200:
                expected_array = execution["heldout_array"]
            exact_options, exact_partitions = expected_test_options(
                manifest,
                record,
                dependency=dependency,
            )
            submit_test = scheduler_test_only(
                boundary,
                record["command"],
                label=f"sealed submit {node.name}",
                cwd=snapshot_root,
                dependency=dependency,
                expected_array=expected_array,
                expected_options=exact_options,
                expected_partitions=exact_partitions,
                expected_processors=int(execution["cpus_per_task"]),
            )
            _require_pre_receipt_budget(execution, transaction_started, f"after_{node.name}_test")
            append_journal(
                submission_root,
                10 + 3 * index,
                f"{node.name.upper()}_SUBMIT_TESTED",
                {"role": node.name, "dependency": dependency, "test": submit_test},
            )
            job_id, submit_record = submit_one(
                boundary,
                execution,
                record["command"],
                name=record["job_name"],
                comment=record["comment"],
                cwd=snapshot_root,
            )
            _require_pre_receipt_budget(execution, transaction_started, f"after_{node.name}_submit")
            jobs_by_role[node.name] = job_id
            accepted_row = {
                "role": node.name,
                "job_id": job_id,
                "job_name": record["job_name"],
                "dependency_role": node.dependency,
                "dependency_job_id": record["predecessor_job_id"],
                "elements": node.elements,
                "scheduler_test": submit_test,
                "submit": submit_record,
            }
            accepted.append(accepted_row)
            append_journal(submission_root, 11 + 3 * index, f"{node.name.upper()}_ACCEPTED", accepted_row)
            observation = observe_accepted_job(
                boundary,
                execution,
                job_id=job_id,
                name=record["job_name"],
                comment=record["comment"],
                predecessor_job_id=record["predecessor_job_id"],
                predecessor_elements=record["predecessor_elements"],
                manifest=manifest,
                node=record["node"],
                submit_command=record["command"],
                cwd=snapshot_root,
            )
            _require_pre_receipt_budget(execution, transaction_started, f"after_{node.name}_observe")
            accepted_row["accepted_observation"] = observation
            append_journal(
                submission_root,
                12 + 3 * index,
                f"{node.name.upper()}_OBSERVED",
                {"role": node.name, "job_id": job_id, "observation": observation},
            )
        # A root element may have started while the remaining dependency chain
        # was submitted. Reauthenticate every accepted parent immediately before
        # READY so a failed/mutated root can never be covered by a success receipt.
        complete_commands = scheduler_commands(
            manifest,
            snapshot_root,
            submission_root,
            submission_sha256,
            jobs_by_role,
        )
        for index, (node, record, accepted_row) in enumerate(
            zip(nodes, complete_commands, accepted, strict=True)
        ):
            _require_pre_receipt_budget(
                execution,
                transaction_started,
                f"before_{node.name}_precommit_observe",
            )
            precommit_observation = observe_accepted_job(
                boundary,
                execution,
                job_id=str(accepted_row["job_id"]),
                name=str(accepted_row["job_name"]),
                comment=str(record["comment"]),
                predecessor_job_id=record["predecessor_job_id"],
                predecessor_elements=record["predecessor_elements"],
                manifest=manifest,
                node=record["node"],
                submit_command=record["command"],
                cwd=snapshot_root,
            )
            _require_pre_receipt_budget(
                execution,
                transaction_started,
                f"after_{node.name}_precommit_observe",
            )
            accepted_row["precommit_observation"] = precommit_observation
            append_journal(
                submission_root,
                50 + index,
                f"{node.name.upper()}_PRECOMMIT_REOBSERVED",
                {
                    "role": node.name,
                    "job_id": accepted_row["job_id"],
                    "observation": precommit_observation,
                },
            )
        receipt = _receipt_from_jobs(
            manifest,
            submission_root,
            snapshot_root,
            submission_sha256,
            accepted,
        )
        _require_pre_receipt_budget(execution, transaction_started, "before_ready_to_commit")
        ready_to_commit = True
        try:
            append_journal(submission_root, 90, "READY_TO_COMMIT", receipt)
            exclusive_json(submission_root / "SUBMISSION_RECEIPT.json", receipt)
        except BaseException as exc:
            raise CommitRecoveryRequired(
                f"receipt commit is ambiguous after durable READY: {exc}"
            ) from exc
        return receipt
    except BaseException as exc:
        if ready_to_commit or isinstance(exc, CommitRecoveryRequired):
            if isinstance(exc, CommitRecoveryRequired):
                raise
            raise CommitRecoveryRequired(
                f"transaction failed after durable READY: {exc}"
            ) from exc
        discovered: dict[str, list[str]] = {node.name: [] for node in nodes}
        for role, job_id in jobs_by_role.items():
            discovered[role].append(job_id)
        discovered.setdefault(active_role or "", []).extend(getattr(exc, "job_ids", ()))
        reconciliation_errors: dict[str, str] = {}
        # Query every transaction-unique role, including the active role whose
        # successful sbatch response may have been lost.
        names = _node_job_names(nodes, submission_sha256[:16])
        comment = f"treewm-exp24:{submission_sha256}"
        for index, node in enumerate(nodes):
            try:
                values = reconcile_job_ids(boundary, execution, names[node.name], comment, snapshot_root)
                discovered[node.name].extend(values)
                append_journal(
                    submission_root,
                    800 + index,
                    f"{node.name.upper()}_RECONCILED",
                    {"role": node.name, "job_ids": sorted(set(discovered[node.name]), key=int)},
                )
            except BaseException as reconcile_exc:
                reconciliation_errors[node.name] = repr(reconcile_exc)
        reverse_ids: list[str] = []
        for node in reversed(nodes):
            for job_id in sorted(set(discovered[node.name]), key=int, reverse=True):
                if job_id not in reverse_ids:
                    reverse_ids.append(job_id)
        cancellation: dict[str, Any] | None = None
        cancellation_error: str | None = None
        if reverse_ids:
            try:
                cancellation = cancel_exact(boundary, execution, reverse_ids, snapshot_root)
                post_cancel = post_cancel_reconciliation(
                    boundary,
                    execution,
                    job_names=names,
                    comment=comment,
                    cwd=snapshot_root,
                    submission_root=submission_root,
                )
                cancellation["post_cancel_reconciliation"] = post_cancel
                require(post_cancel["status"] == "terminal_or_absent", "rollback cancellation convergence remains pending")
                append_journal(submission_root, 900, "ROLLBACK_CANCELED", cancellation)
            except BaseException as cancel_exc:
                cancellation_error = repr(cancel_exc)
        abort = {
            "error": repr(exc),
            "active_role": active_role,
            "accepted_job_ids_by_role": {
                role: sorted(set(values), key=int) for role, values in discovered.items()
            },
            "reconciliation_errors": reconciliation_errors,
            "cancellation": cancellation,
            "cancellation_error": cancellation_error,
            "receipt_committed": (submission_root / "SUBMISSION_RECEIPT.json").exists(),
        }
        abort_journal_error: str | None = None
        try:
            append_journal(submission_root, 999, "ABORTED", abort)
        except BaseException as journal_exc:
            abort_journal_error = repr(journal_exc)
        raise SchedulerTransactionError(
            f"Exp24 transaction aborted at {active_role}: {exc}; "
            f"cancellation={cancellation_error}; abort_journal={abort_journal_error}",
            reverse_ids,
        ) from exc


def _authorized_submit_locked(
    manifest: Mapping[str, Any],
    *,
    submission_root: Path,
    fallback_binding: Mapping[str, Any],
    observation: Mapping[str, Any],
    scheduler_preclaim: Mapping[str, Any],
    boundary: SchedulerBoundary,
    interpreter_provenance: Mapping[str, Any],
) -> dict[str, Any]:
    claim_token = secrets.token_hex(32)
    begin_transaction(submission_root, claim_token)
    try:
        inventory = m1_snapshot_inventory(REPOSITORY_ROOT)
        snapshot_root = submission_root / "source-snapshot" / "repo"
        _mkdir_exact(submission_root / "source-snapshot", 0o700, "source snapshot parent")
        create_source_snapshot(REPOSITORY_ROOT, snapshot_root, inventory)
        verify_submission_snapshot_identity(manifest, snapshot_root, inventory)
        os.chmod(submission_root / "source-snapshot", 0o555, follow_symlinks=False)
        _fsync_directory(submission_root / "source-snapshot")
        _fsync_directory(submission_root)
        append_journal(
            submission_root,
            1,
            "SNAPSHOT_SEALED",
            {"inventory_sha256": stable_hash(inventory), "file_count": len(inventory)},
        )
        contract = submission_contract(
            manifest,
            submission_root=submission_root,
            snapshot_root=snapshot_root,
            snapshot_inventory=inventory,
            control_plane=observation,
            scheduler_preclaim=scheduler_preclaim,
            scheduler_fallback=fallback_binding,
            interpreter_provenance=interpreter_provenance,
        )
        submission_sha256 = exclusive_json(submission_root / "SUBMISSION_CONTRACT.json", contract)
        append_journal(submission_root, 2, "CONTRACT_SEALED", {"submission_sha256": submission_sha256})
        receipt = submit_dag_transaction(
            manifest,
            submission_root=submission_root,
            snapshot_root=snapshot_root,
            submission_sha256=submission_sha256,
            boundary=boundary,
        )
        authorization = authorize_root_release_after_receipt(
            submission_root,
            boundary=boundary,
        )
        return {
            "schema_version": 1,
            "status": "submission_receipt_and_held_root_release_authorization_committed",
            "receipt": receipt,
            "authorization": authorization,
        }
    except BaseException as exc:
        if isinstance(exc, CommitRecoveryRequired):
            raise
        if not (submission_root / "SUBMISSION_RECEIPT.json").exists():
            try:
                append_journal(
                    submission_root,
                    998,
                    "OUTER_ABORTED",
                    {
                        "error": repr(exc),
                        "recovery_required": (submission_root / "SUBMISSION_CONTRACT.json").exists(),
                    },
                )
            except BaseException as journal_exc:
                raise RuntimeContractError(
                    f"submission failed and OUTER_ABORTED persistence also failed: "
                    f"original={exc!r}; journal={journal_exc!r}"
                ) from exc
        raise


def authorized_submit(
    manifest: Mapping[str, Any],
    *,
    boundary_factory: Callable[[Mapping[str, Any]], SchedulerBoundary] | None = None,
) -> dict[str, Any]:
    """The sole production submit path; authority is checked before first mutation."""

    import campaign

    campaign.assert_launch_authorized(manifest)
    readiness = execution_readiness(manifest)
    require(readiness["ready"], "formal runtime remains blocked: " + "; ".join(readiness["blockers"]))
    interpreter_provenance = capture_interpreter_provenance(manifest)
    submission_root = Path(str(manifest["paths"]["run_root"]) + "-submission")
    require(submission_root.parent.exists(), "submission parent is absent")
    fallback_payload, observation = capture_scheduler_control_plane_bundle(manifest["execution"])
    fallback_binding = scheduler_fallback_binding(fallback_payload, observation)
    boundary = (
        boundary_factory(observation)
        if boundary_factory is not None
        else SchedulerBoundary(
            runner=default_runner,
            observer=lambda: capture_scheduler_control_plane(manifest["execution"]),
            expected=observation,
            fallback_payload=fallback_payload,
        )
    )
    scheduler_preclaim = scheduler_preclaim_test(
        manifest,
        repo_root=REPOSITORY_ROOT,
        boundary=boundary,
    )
    with transaction_recovery_lock(submission_root):
        prepared = _authorized_submit_locked(
            manifest,
            submission_root=submission_root,
            fallback_binding=fallback_binding,
            observation=observation,
            scheduler_preclaim=scheduler_preclaim,
            boundary=boundary,
            interpreter_provenance=interpreter_provenance,
        )
    # No released worker can now block behind the claim-to-receipt lock.  The
    # durable authorization is sufficient worker authority if this process
    # crashes after release but before the activation-result record commits.
    activation = activate_root_after_receipt(
        submission_root,
        boundary=boundary,
    )
    return {
        "schema_version": 1,
        "status": "submission_receipt_committed_and_root_activated",
        "receipt": prepared["receipt"],
        "authorization": prepared["authorization"],
        "activation": activation,
    }


def execution_readiness(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Read-only M2A readiness inventory.  It intentionally returns ``ready=false``."""

    import campaign

    m2a_schema, m2a_schema_file_sha = campaign.load_m2a_schema()
    objective = manifest["objective"]["objective_version"]
    config = REPOSITORY_ROOT / "configs" / "experiment" / f"{objective}.yaml"
    delta, delta_sha = authenticated_json(PACKAGE_DIR / "formal_objective_delta.json", "formal objective delta")
    blockers = []
    if delta.get("status") != "applied_and_verified":
        blockers.append("formal objective delta is documented but not applied to shared trainer/config")
    blockers.append(
        "training and held-out durable signal/resume adapters are not implemented or killpoint-verified"
    )
    blockers.append(
        "same-stage requeue mutation is hard-disabled until exact scientific identity and an idempotent scheduler transaction are sealed"
    )
    blockers.append(
        "interpreter binary/pyvenv/path identity is captured, but environment-content closure is unbound"
    )
    blockers.append(
        "clean committed execution-source revision and exact snapshot provenance are unverified"
    )
    if not config.is_file() or config.is_symlink():
        blockers.append("formal objective Hydra config is absent")
    if manifest.get("scientific_protocol_sealed") is not True:
        blockers.append("scientific protocol is unsealed")
    if manifest.get("per_setting_contract_state") != "sealed_all_ten":
        blockers.append("all-ten per-setting contracts are unbound")
    if manifest["execution"].get("feasibility_status") != "verified":
        blockers.append("training/evaluation/requeue feasibility is unverified")
    if manifest["launch_dependency"].get("binding_state") != "sealed":
        blockers.append("future Launch8 accepted_engineering_pilot binding is unsealed")
    if manifest["launch_dependency"].get("adapter_state") != "sealed_versioned_adapter":
        blockers.append("future Launch8 versioned semantic adapter is unsealed")
    return {
        "schema_version": 1,
        "ready": not blockers,
        "phase": "m2a_orders_0_3_runtime_authority_scaffold_execution_blocked",
        "m2a_schema_sha256": stable_hash(m2a_schema),
        "m2a_schema_file_sha256": m2a_schema_file_sha,
        "held_root_activation_implemented": True,
        "interpreter_binary_pyvenv_path_provenance_implemented": True,
        "interpreter_environment_content_provenance_implemented": False,
        "same_stage_requeue_mutation_implemented": False,
        "launch7_terminal_negative_binding_state": (
            manifest["launch7_negative_dependency"]["binding_state"]
        ),
        "engineering_pilot_adapter_state": manifest["launch_dependency"]["adapter_state"],
        "accepted_engineering_pilot_binding_state": manifest["launch_dependency"]["binding_state"],
        "formal_objective_delta_sha256": delta_sha,
        "blockers": blockers,
    }


def load_receipt(
    submission_root: Path,
    *,
    verify_full_snapshot: bool = True,
) -> tuple[dict[str, Any], str]:
    import campaign

    value, digest = authenticated_immutable_json(submission_root / "SUBMISSION_RECEIPT.json", "submission receipt")
    require(set(value) == SUBMISSION_RECEIPT_KEYS, "submission receipt schema differs")
    require(
        value.get("schema_version") == 1
        and value.get("status") == "submitted"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_root") == str(submission_root)
        and SHA256.fullmatch(str(value.get("submission_sha256", ""))) is not None
        and SHA256.fullmatch(str(value.get("manifest_sha256", ""))) is not None,
        "submission receipt identity/status differs",
    )
    contract, contract_digest = authenticated_immutable_json(
        submission_root / "SUBMISSION_CONTRACT.json",
        "submission contract",
    )
    require(contract_digest == value["submission_sha256"], "receipt does not bind submission contract bytes")
    validate_submission_contract(
        contract,
        submission_root=submission_root,
        digest=contract_digest,
        verify_runtime_authority=verify_full_snapshot,
    )
    require(
        contract.get("schema_version") == 1
        and contract.get("status") == "prepared_scheduler_transaction"
        and contract.get("campaign_id") == CAMPAIGN_ID
        and contract.get("submission_root") == str(submission_root)
        and contract.get("snapshot_root") == value["snapshot_root"]
        and contract.get("manifest_sha256") == value["manifest_sha256"],
        "receipt and submission contract differ",
    )
    inventory = contract.get("snapshot_inventory")
    require(isinstance(inventory, dict) and inventory, "contract snapshot inventory differs")
    snapshot_root = Path(str(value["snapshot_root"]))
    require(snapshot_root.is_absolute(), "receipt snapshot root differs")
    if verify_full_snapshot:
        verify_snapshot_files(snapshot_root, inventory)
    snapshot_manifest, _snapshot_manifest_file_sha = authenticated_immutable_json(
        snapshot_root / PACKAGE_RELATIVE / "manifest.json",
        "snapshot manifest",
    )
    campaign.validate_manifest(snapshot_manifest)
    require(campaign.manifest_sha256(snapshot_manifest) == value["manifest_sha256"], "receipt snapshot manifest differs")
    canonical_nodes = campaign.scheduler_dag(snapshot_manifest)
    canonical_dag = [asdict(node) for node in canonical_nodes]
    canonical_names = [node.name for node in canonical_nodes]
    require(contract.get("dag") == canonical_dag, "contract DAG differs from canonical eleven nodes")
    require(value.get("dag_names") == canonical_names, "receipt DAG names differ from canonical eleven nodes")
    require(
        value.get("training_array") == snapshot_manifest["execution"]["training_array"] == "0-39%40"
        and value.get("heldout_array") == snapshot_manifest["execution"]["heldout_array"] == "0-199%40",
        "receipt array geometry differs",
    )
    jobs = value.get("jobs")
    require(isinstance(jobs, list) and len(jobs) == 11, "submission receipt job inventory differs")
    roles: list[str] = []
    ids: set[str] = set()
    jobs_by_role: dict[str, str] = {}
    for node, row in zip(canonical_nodes, jobs, strict=True):
        require(isinstance(row, dict) and set(row) == {
            "role", "job_id", "job_name", "dependency_role",
            "dependency_job_id", "elements", "scheduler_test", "submit",
            "accepted_observation", "precommit_observation",
        }, "submission receipt job row schema differs")
        role, job_id = row.get("role"), str(row.get("job_id", ""))
        require(isinstance(role, str) and JOB_ID.fullmatch(job_id) is not None, "submission receipt role/ID differs")
        require(role not in roles and job_id not in ids, "submission receipt duplicates a role/ID")
        require(
            role == node.name
            and row.get("dependency_role") == node.dependency
            and row.get("elements") == node.elements,
            f"receipt canonical role shape differs: {node.name}",
        )
        expected_dependency_id = jobs_by_role.get(node.dependency) if node.dependency else None
        require(row.get("dependency_job_id") == expected_dependency_id, f"receipt dependency ID differs: {node.name}")
        roles.append(role)
        ids.add(job_id)
        jobs_by_role[role] = job_id
    require(roles == canonical_names, "submission receipt DAG order differs")
    expected_commands = scheduler_commands(
        snapshot_manifest,
        snapshot_root,
        submission_root,
        contract_digest,
        jobs_by_role,
    )
    for node, row, expected in zip(canonical_nodes, jobs, expected_commands, strict=True):
        require(row["job_name"] == expected["job_name"], f"receipt job name differs: {node.name}")
        dependency_option = (
            None
            if node.dependency is None
            else f"afterok:{row['dependency_job_id']}"
        )
        expected_array = None
        if node.elements == 40:
            expected_array = snapshot_manifest["execution"]["training_array"]
        elif node.elements == 200:
            expected_array = snapshot_manifest["execution"]["heldout_array"]
        exact_options, exact_partitions = expected_test_options(
            snapshot_manifest,
            expected,
            dependency=dependency_option,
        )
        scheduler_test = row["scheduler_test"]
        require(isinstance(scheduler_test, dict), f"receipt scheduler test is not an object: {node.name}")
        validate_scheduler_test_evidence(
            scheduler_test,
            command=expected["command"],
            label=f"receipt sealed submit {node.name}",
            dependency=dependency_option,
            expected_array=expected_array,
            expected_options=exact_options,
            expected_partitions=exact_partitions,
            expected_processors=int(snapshot_manifest["execution"]["cpus_per_task"]),
            expected_control_plane=contract["scheduler_control_plane"],
        )
        submit = row["submit"]
        require(isinstance(submit, dict) and set(submit) == {
            "command", "response_mode", "returncode", "stdout", "stderr",
            "reconciled_job_ids", "control_plane",
        }, f"receipt submit evidence schema differs: {node.name}")
        require(
            submit["command"] == expected["command"]
            and submit["returncode"] == 0
            and isinstance(submit["stderr"], str)
            and submit["reconciled_job_ids"] == [row["job_id"]]
            and submit["control_plane"] == contract["scheduler_control_plane"],
            f"receipt submit evidence differs: {node.name}",
        )
        if submit["response_mode"] == "parsed_sbatch_stdout":
            require(submit["stdout"].strip() == row["job_id"], f"receipt parsed sbatch response differs: {node.name}")
        else:
            require(
                submit["response_mode"] == "exact_name_comment_reconciliation_after_unparseable_stdout"
                and SBATCH_RESPONSE.fullmatch(str(submit["stdout"]).strip()) is None,
                f"receipt reconciled sbatch response differs: {node.name}",
            )
        predecessor_elements = (
            None
            if node.dependency is None
            else canonical_nodes[canonical_names.index(node.dependency)].elements
        )
        for observation_name in ("accepted_observation", "precommit_observation"):
            observation = row[observation_name]
            require(isinstance(observation, dict) and set(observation) == {
                "command", "stdout", "stderr", "dependency",
                "kill_on_invalid_dependency", "normalized_shape", "control_plane",
            }, f"receipt {observation_name} schema differs: {node.name}")
            require(
                observation["command"]
                == [str(snapshot_manifest["execution"]["scontrol"]), "show", "job", row["job_id"], "--oneliner"]
                and observation["control_plane"] == contract["scheduler_control_plane"],
                f"receipt {observation_name} identity differs: {node.name}",
            )
            normalized = validate_accepted_job_stdout(
                str(observation["stdout"]),
                job_id=str(row["job_id"]),
                name=str(row["job_name"]),
                comment=expected["comment"],
                predecessor_job_id=(None if node.dependency is None else str(row["dependency_job_id"])),
                predecessor_elements=predecessor_elements,
                manifest=snapshot_manifest,
                node=expected["node"],
                submit_command=expected["command"],
                cwd=snapshot_root,
            )
            require(
                observation["normalized_shape"] == normalized
                and observation["dependency"] == normalized["dependency"]
                and observation["kill_on_invalid_dependency"] == normalized["kill_on_invalid_dependency"],
                f"receipt {observation_name} normalization differs: {node.name}",
            )
    journals = load_journals(submission_root)
    expected_journal_identities: list[tuple[int, str]] = [
        (0, "CLAIMED"),
        (1, "SNAPSHOT_SEALED"),
        (2, "CONTRACT_SEALED"),
    ]
    for index, node in enumerate(canonical_nodes):
        expected_journal_identities.extend(
            [
                (10 + 3 * index, f"{node.name.upper()}_SUBMIT_TESTED"),
                (11 + 3 * index, f"{node.name.upper()}_ACCEPTED"),
                (12 + 3 * index, f"{node.name.upper()}_OBSERVED"),
            ]
        )
    expected_journal_identities.extend(
        (50 + index, f"{node.name.upper()}_PRECOMMIT_REOBSERVED")
        for index, node in enumerate(canonical_nodes)
    )
    expected_journal_identities.append((90, "READY_TO_COMMIT"))
    require(
        [(int(row["ordinal"]), str(row["event"])) for row in journals]
        == expected_journal_identities,
        "committed transaction journal inventory differs",
    )
    claim = journals[0]["payload"]
    require(
        set(claim) == {"claim_token", "pid", "created_ns"}
        and re.fullmatch(r"[0-9a-f]{64}", str(claim.get("claim_token", ""))) is not None
        and isinstance(claim.get("pid"), int)
        and not isinstance(claim.get("pid"), bool)
        and claim["pid"] > 0
        and isinstance(claim.get("created_ns"), int)
        and not isinstance(claim.get("created_ns"), bool)
        and claim["created_ns"] > 0,
        "committed transaction claim payload differs",
    )
    require(
        journals[1]["payload"]
        == {
            "inventory_sha256": contract["snapshot_inventory_sha256"],
            "file_count": len(inventory),
        },
        "committed snapshot-seal journal differs",
    )
    require(
        journals[2]["payload"] == {"submission_sha256": contract_digest},
        "committed contract-seal journal differs",
    )
    ready = [row for row in journals if row["event"] == "READY_TO_COMMIT"]
    require(len(ready) == 1 and ready[0]["payload"] == value, "receipt differs from unique durable READY")
    for index, (node, row) in enumerate(zip(canonical_nodes, jobs, strict=True)):
        dependency_option = (
            None
            if node.dependency is None
            else f"afterok:{row['dependency_job_id']}"
        )
        accepted_event = f"{node.name.upper()}_ACCEPTED"
        observed_event = f"{node.name.upper()}_OBSERVED"
        precommit_event = f"{node.name.upper()}_PRECOMMIT_REOBSERVED"
        tested_event = f"{node.name.upper()}_SUBMIT_TESTED"
        tested_rows = [entry for entry in journals if entry["event"] == tested_event]
        accepted_rows = [entry for entry in journals if entry["event"] == accepted_event]
        observed_rows = [entry for entry in journals if entry["event"] == observed_event]
        precommit_rows = [entry for entry in journals if entry["event"] == precommit_event]
        expected_accepted = dict(row)
        expected_accepted.pop("accepted_observation")
        expected_accepted.pop("precommit_observation")
        require(
            len(tested_rows) == 1
            and tested_rows[0]["payload"]
            == {"role": node.name, "dependency": dependency_option, "test": row["scheduler_test"]},
            f"receipt scheduler-test journal differs: {node.name}",
        )
        require(
            len(accepted_rows) == 1 and accepted_rows[0]["payload"] == expected_accepted,
            f"receipt accepted journal differs: {node.name}",
        )
        require(
            len(observed_rows) == 1
            and observed_rows[0]["payload"]
            == {"role": node.name, "job_id": row["job_id"], "observation": row["accepted_observation"]},
            f"receipt observation journal differs: {node.name}",
        )
        require(
            len(precommit_rows) == 1
            and precommit_rows[0]["payload"]
            == {"role": node.name, "job_id": row["job_id"], "observation": row["precommit_observation"]},
            f"receipt precommit observation journal differs: {node.name}",
        )
    return value, digest


ROOT_RELEASE_AUTHORIZATION_KEYS = frozenset(
    {
        "schema_version", "status", "campaign_id", "submission_root",
        "submission_sha256", "submission_receipt_sha256", "root_role",
        "root_job_id", "root_job_name", "root_comment", "held_observation",
        "held_observation_sha256", "release_command", "scheduler_control_plane",
        "authorization_body_sha256",
    }
)
ROOT_ACTIVATION_RESULT_KEYS = frozenset(
    {
        "schema_version", "status", "campaign_id", "submission_root",
        "submission_sha256", "submission_receipt_sha256",
        "root_release_authorization_sha256", "root_role", "root_job_id",
        "release_command", "release_response_mode", "release_returncode",
        "release_stdout", "release_stderr", "released_observation",
        "result_body_sha256",
    }
)


def _root_activation_context(
    submission_root: Path,
) -> dict[str, Any]:
    """Reconstruct exact root authority solely from sealed receipt/contract bytes."""

    receipt, receipt_sha = load_receipt(submission_root, verify_full_snapshot=False)
    contract, contract_sha = authenticated_immutable_json(
        submission_root / "SUBMISSION_CONTRACT.json",
        "activation submission contract",
    )
    require(contract_sha == receipt["submission_sha256"], "activation contract/receipt differs")
    snapshot_root = Path(str(receipt["snapshot_root"]))
    snapshot_manifest, _manifest_file_sha = authenticated_immutable_json(
        snapshot_root / PACKAGE_RELATIVE / "manifest.json",
        "activation snapshot manifest",
    )
    import campaign

    campaign.validate_manifest(snapshot_manifest)
    validate_submission_contract(
        contract,
        submission_root=submission_root,
        digest=contract_sha,
        verify_runtime_authority=True,
    )
    jobs = list(receipt["jobs"])
    roots = [row for row in jobs if row["role"] == "train_2000"]
    require(len(roots) == 1 and jobs[0] == roots[0], "activation root receipt row differs")
    root = roots[0]
    ids = {str(row["role"]): str(row["job_id"]) for row in jobs}
    commands = scheduler_commands(
        snapshot_manifest,
        snapshot_root,
        submission_root,
        contract_sha,
        ids,
    )
    root_command = commands[0]
    require(
        root_command["node"]["name"] == "train_2000"
        and root_command["predecessor_job_id"] is None
        and "--hold" in root_command["command"],
        "activation canonical held-root command differs",
    )
    return {
        "receipt": receipt,
        "receipt_sha256": receipt_sha,
        "contract": contract,
        "contract_sha256": contract_sha,
        "manifest": snapshot_manifest,
        "snapshot_root": snapshot_root,
        "root": root,
        "root_command": root_command,
        "release_command": [
            str(snapshot_manifest["execution"]["scontrol"]),
            "release",
            str(root["job_id"]),
        ],
    }


def _root_observation_from_completed(
    completed: subprocess.CompletedProcess[str],
    *,
    context: Mapping[str, Any],
    boundary: SchedulerBoundary,
    lifecycle: str,
) -> dict[str, Any]:
    root = context["root"]
    record = context["root_command"]
    manifest = context["manifest"]
    snapshot_root = context["snapshot_root"]
    require(completed.returncode == 0, f"cannot observe activation root {root['job_id']}: {completed.stderr.strip()}")
    normalized = validate_accepted_job_stdout(
        completed.stdout,
        job_id=str(root["job_id"]),
        name=str(root["job_name"]),
        comment=str(record["comment"]),
        predecessor_job_id=None,
        predecessor_elements=None,
        manifest=manifest,
        node=record["node"],
        submit_command=record["command"],
        cwd=snapshot_root,
        root_lifecycle=lifecycle,
    )
    return {
        "command": [
            str(manifest["execution"]["scontrol"]),
            "show",
            "job",
            str(root["job_id"]),
            "--oneliner",
        ],
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "dependency": normalized["dependency"],
        "kill_on_invalid_dependency": normalized["kill_on_invalid_dependency"],
        "normalized_shape": normalized,
        "control_plane": boundary.expected,
    }


def _observe_root_lifecycle(
    boundary: SchedulerBoundary,
    context: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    command = [
        str(context["manifest"]["execution"]["scontrol"]),
        "show",
        "job",
        str(context["root"]["job_id"]),
        "--oneliner",
    ]
    completed = boundary.call(command, context["snapshot_root"])
    errors: dict[str, str] = {}
    matches: list[tuple[dict[str, Any], str]] = []
    for lifecycle in ("held", "released"):
        try:
            observation = _root_observation_from_completed(
                completed,
                context=context,
                boundary=boundary,
                lifecycle=lifecycle,
            )
        except RuntimeContractError as exc:
            errors[lifecycle] = str(exc)
        else:
            matches.append((observation, lifecycle))
    require(
        len(matches) == 1,
        f"activation root is neither uniquely held nor released: {errors}",
    )
    return matches[0]


def _authorization_seed(
    context: Mapping[str, Any],
    held_observation: Mapping[str, Any],
) -> dict[str, Any]:
    seed = {
        "schema_version": 1,
        "status": "receipt_committed_root_release_authorized",
        "campaign_id": CAMPAIGN_ID,
        "submission_root": str(context["receipt"]["submission_root"]),
        "submission_sha256": context["contract_sha256"],
        "submission_receipt_sha256": context["receipt_sha256"],
        "root_role": "train_2000",
        "root_job_id": str(context["root"]["job_id"]),
        "root_job_name": str(context["root"]["job_name"]),
        "root_comment": str(context["root_command"]["comment"]),
        "held_observation": dict(held_observation),
        "held_observation_sha256": stable_hash(held_observation),
        "release_command": list(context["release_command"]),
        "scheduler_control_plane": context["contract"]["scheduler_control_plane"],
    }
    seed["authorization_body_sha256"] = stable_hash(seed)
    return seed


def validate_root_release_authorization(
    submission_root: Path,
    value: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = _root_activation_context(submission_root) if context is None else dict(context)
    require(set(value) == ROOT_RELEASE_AUTHORIZATION_KEYS, "root release-authorization schema differs")
    body = dict(value)
    body_hash = body.pop("authorization_body_sha256", None)
    require(body_hash == stable_hash(body), "root release-authorization self-hash differs")
    held = value.get("held_observation")
    require(
        isinstance(held, dict)
        and set(held)
        == {
            "command", "stdout", "stderr", "dependency",
            "kill_on_invalid_dependency", "normalized_shape", "control_plane",
        }
        and isinstance(held.get("stdout"), str)
        and isinstance(held.get("stderr"), str),
        "root held observation schema differs",
    )
    root = context["root"]
    record = context["root_command"]
    normalized = validate_accepted_job_stdout(
        str(held.get("stdout", "")),
        job_id=str(root["job_id"]),
        name=str(root["job_name"]),
        comment=str(record["comment"]),
        predecessor_job_id=None,
        predecessor_elements=None,
        manifest=context["manifest"],
        node=record["node"],
        submit_command=record["command"],
        cwd=context["snapshot_root"],
        root_lifecycle="held",
    )
    expected_held = {
        "command": [
            str(context["manifest"]["execution"]["scontrol"]), "show", "job",
            str(root["job_id"]), "--oneliner",
        ],
        "stdout": held.get("stdout"),
        "stderr": held.get("stderr"),
        "dependency": normalized["dependency"],
        "kill_on_invalid_dependency": normalized["kill_on_invalid_dependency"],
        "normalized_shape": normalized,
        "control_plane": context["contract"]["scheduler_control_plane"],
    }
    expected = _authorization_seed(context, expected_held)
    require(dict(value) == expected, "root release authorization differs from receipt/held evidence")
    return expected


def load_root_release_authorization(
    submission_root: Path,
    *,
    context: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    value, digest = authenticated_immutable_json(
        submission_root / "ROOT_RELEASE_AUTHORIZATION.json",
        "root release authorization",
    )
    validate_root_release_authorization(submission_root, value, context=context)
    return value, digest


def validate_root_activation_result(
    submission_root: Path,
    value: Mapping[str, Any],
    *,
    context: Mapping[str, Any] | None = None,
    authorization_sha256: str | None = None,
) -> dict[str, Any]:
    context = _root_activation_context(submission_root) if context is None else dict(context)
    authorization, loaded_authorization_sha = load_root_release_authorization(
        submission_root,
        context=context,
    )
    if authorization_sha256 is not None:
        require(authorization_sha256 == loaded_authorization_sha, "activation authorization digest differs")
    require(set(value) == ROOT_ACTIVATION_RESULT_KEYS, "root activation-result schema differs")
    body = dict(value)
    body_hash = body.pop("result_body_sha256", None)
    require(body_hash == stable_hash(body), "root activation-result self-hash differs")
    observation = value.get("released_observation")
    require(
        isinstance(observation, dict)
        and isinstance(observation.get("stdout"), str)
        and isinstance(observation.get("stderr"), str),
        "released root observation is absent/malformed",
    )
    normalized = validate_accepted_job_stdout(
        str(observation.get("stdout", "")),
        job_id=str(context["root"]["job_id"]),
        name=str(context["root"]["job_name"]),
        comment=str(context["root_command"]["comment"]),
        predecessor_job_id=None,
        predecessor_elements=None,
        manifest=context["manifest"],
        node=context["root_command"]["node"],
        submit_command=context["root_command"]["command"],
        cwd=context["snapshot_root"],
        root_lifecycle="released",
    )
    require(
        set(observation)
        == {
            "command", "stdout", "stderr", "dependency",
            "kill_on_invalid_dependency", "normalized_shape", "control_plane",
        }
        and observation.get("command")
        == [
            str(context["manifest"]["execution"]["scontrol"]), "show", "job",
            str(context["root"]["job_id"]), "--oneliner",
        ]
        and observation.get("normalized_shape") == normalized
        and observation.get("dependency") == normalized["dependency"]
        and observation.get("kill_on_invalid_dependency")
        == normalized["kill_on_invalid_dependency"]
        and observation.get("control_plane")
        == context["contract"]["scheduler_control_plane"],
        "released root observation evidence differs",
    )
    mode = value.get("release_response_mode")
    if mode == "direct_zero_exit":
        require(
            value.get("release_returncode") == 0
            and isinstance(value.get("release_stdout"), str)
            and isinstance(value.get("release_stderr"), str),
            "direct root release response differs",
        )
    elif mode == "reconciled_nonzero_release_response":
        require(
            isinstance(value.get("release_returncode"), int)
            and not isinstance(value.get("release_returncode"), bool)
            and value["release_returncode"] != 0
            and isinstance(value.get("release_stdout"), str)
            and isinstance(value.get("release_stderr"), str),
            "reconciled nonzero root release response differs",
        )
    else:
        require(
            mode == "reconciled_already_released_after_durable_authorization"
            and value.get("release_returncode") is None
            and value.get("release_stdout") is None
            and value.get("release_stderr") is None,
            "reconciled prior root release response differs",
        )
    require(
        value.get("schema_version") == 1
        and value.get("status") == "train_2000_released_and_observed"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_root") == str(submission_root)
        and value.get("submission_sha256") == context["contract_sha256"]
        and value.get("submission_receipt_sha256") == context["receipt_sha256"]
        and value.get("root_release_authorization_sha256") == loaded_authorization_sha
        and value.get("root_role") == "train_2000"
        and value.get("root_job_id") == str(context["root"]["job_id"])
        and value.get("release_command") == context["release_command"],
        "root activation-result identity differs",
    )
    del authorization
    return dict(value)


def _authorize_root_release_locked(
    submission_root: Path,
    *,
    boundary: SchedulerBoundary,
    fault_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Durably authorize only while the root is held; never release it here.

    This runs under the outer claim-to-receipt transaction lock.  Keeping the
    root held makes every local fsync/controller-observation delay harmless to
    queued workers.  The outer lock is released before the separate activation
    function can issue ``scontrol release``.
    """

    hook = (lambda _ordinal: None) if fault_hook is None else fault_hook
    repair_publication_residues(
        submission_root,
        allowed_final=re.compile(r"ROOT_(?:RELEASE_AUTHORIZATION|ACTIVATION_RESULT)\.json"),
    )
    for name in ("CANCEL_REQUESTED.json", "CANCEL_RESULT.json", "TRANSACTION_RECOVERY_RESULT.json"):
        require(
            not lexical_entry_exists(submission_root, name, f"root authorization marker {name}"),
            f"root authorization conflicts with {name}",
        )
    context = _root_activation_context(submission_root)
    require(
        boundary.expected == context["contract"]["scheduler_control_plane"],
        "root authorization scheduler control-plane binding differs",
    )
    authorization_path = submission_root / "ROOT_RELEASE_AUTHORIZATION.json"
    result_path = submission_root / "ROOT_ACTIVATION_RESULT.json"
    if lexical_entry_exists(submission_root, result_path.name, "root activation result"):
        result, result_sha = authenticated_immutable_json(result_path, "root activation result")
        validate_root_activation_result(submission_root, result, context=context)
        authorization, authorization_sha = load_root_release_authorization(
            submission_root,
            context=context,
        )
        return {
            **authorization,
            "root_release_authorization_sha256": authorization_sha,
            "activation_already_complete": True,
            "root_activation_result_sha256": result_sha,
            "retry": True,
        }
    if lexical_entry_exists(
        submission_root,
        authorization_path.name,
        "root release authorization",
    ):
        authorization, authorization_sha = load_root_release_authorization(
            submission_root,
            context=context,
        )
        return {
            **authorization,
            "root_release_authorization_sha256": authorization_sha,
            "activation_already_complete": False,
            "retry": True,
        }
    observation, lifecycle = _observe_root_lifecycle(boundary, context)
    require(lifecycle == "held", "root was released before durable release authorization")
    authorization = _authorization_seed(context, observation)
    authorization_sha = exclusive_json(authorization_path, authorization)
    hook("authorization_published")
    return {
        **authorization,
        "root_release_authorization_sha256": authorization_sha,
        "activation_already_complete": False,
        "retry": False,
    }


def authorize_root_release_after_receipt(
    submission_root: Path,
    *,
    boundary: SchedulerBoundary,
    fault_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Serialize durable held-root authorization against exact-ID cancel."""

    with report_cancel_lock(submission_root):
        return _authorize_root_release_locked(
            submission_root,
            boundary=boundary,
            fault_hook=fault_hook,
        )


def _activate_root_locked(
    submission_root: Path,
    *,
    boundary: SchedulerBoundary,
    fault_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Release under cancel serialization, then publish under activation lock."""

    hook = (lambda _ordinal: None) if fault_hook is None else fault_hook
    repair_publication_residues(
        submission_root,
        allowed_final=re.compile(r"ROOT_(?:RELEASE_AUTHORIZATION|ACTIVATION_RESULT)\.json"),
    )
    for name in ("CANCEL_REQUESTED.json", "CANCEL_RESULT.json", "TRANSACTION_RECOVERY_RESULT.json"):
        require(
            not lexical_entry_exists(submission_root, name, f"root activation marker {name}"),
            f"root activation conflicts with {name}",
        )
    context = _root_activation_context(submission_root)
    require(
        boundary.expected == context["contract"]["scheduler_control_plane"],
        "activation scheduler control-plane binding differs",
    )
    authorization_path = submission_root / "ROOT_RELEASE_AUTHORIZATION.json"
    result_path = submission_root / "ROOT_ACTIVATION_RESULT.json"
    if lexical_entry_exists(submission_root, result_path.name, "root activation result"):
        result, result_sha = authenticated_immutable_json(result_path, "root activation result")
        validate_root_activation_result(submission_root, result, context=context)
        with report_cancel_lock(submission_root):
            for name in ("CANCEL_REQUESTED.json", "CANCEL_RESULT.json", "TRANSACTION_RECOVERY_RESULT.json"):
                require(
                    not lexical_entry_exists(submission_root, name, f"root activation return marker {name}"),
                    f"root activation return conflicts with {name}",
                )
        return {**result, "root_activation_result_sha256": result_sha, "retry": True}

    # Cancellation is excluded only through the release side effect and its
    # bounded scheduler observation.  Result publication is historical audit
    # evidence and may coexist with a later cancellation, so its unbounded fsync
    # must not hold the emergency-cancel lock.
    with report_cancel_lock(submission_root):
        for name in ("CANCEL_REQUESTED.json", "CANCEL_RESULT.json", "TRANSACTION_RECOVERY_RESULT.json"):
            require(
                not lexical_entry_exists(submission_root, name, f"root release marker {name}"),
                f"root release conflicts with {name}",
            )
        observation, lifecycle = _observe_root_lifecycle(boundary, context)
        if lexical_entry_exists(
            submission_root,
            authorization_path.name,
            "root release authorization",
        ):
            authorization, authorization_sha = load_root_release_authorization(
                submission_root,
                context=context,
            )
        else:
            require(lifecycle == "held", "root was released before durable release authorization")
            authorization = _authorization_seed(context, observation)
            authorization_sha = exclusive_json(authorization_path, authorization)
            hook("authorization_published")
        release_response_mode: str
        release_returncode: int | None
        release_stdout: str | None
        release_stderr: str | None
        if lifecycle == "held":
            hook("before_release_call")
            try:
                released = boundary.call(context["release_command"], context["snapshot_root"])
            except BaseException as exc:
                raise ActivationRecoveryRequired(
                    f"root release response is ambiguous after durable authorization: {exc}"
                ) from exc
            hook("after_release_call")
            release_returncode = released.returncode
            release_stdout = released.stdout
            release_stderr = released.stderr
            release_response_mode = (
                "direct_zero_exit"
                if released.returncode == 0
                else "reconciled_nonzero_release_response"
            )
            try:
                observation, lifecycle = _observe_root_lifecycle(boundary, context)
            except BaseException as exc:
                raise ActivationRecoveryRequired(
                    f"root release requires post-call recovery observation: {exc}"
                ) from exc
            require(lifecycle == "released", "root remained held after exact release call")
        else:
            release_response_mode = "reconciled_already_released_after_durable_authorization"
            release_returncode = None
            release_stdout = None
            release_stderr = None
    hook("released_observed")
    result_seed = {
        "schema_version": 1,
        "status": "train_2000_released_and_observed",
        "campaign_id": CAMPAIGN_ID,
        "submission_root": str(submission_root),
        "submission_sha256": context["contract_sha256"],
        "submission_receipt_sha256": context["receipt_sha256"],
        "root_release_authorization_sha256": authorization_sha,
        "root_role": "train_2000",
        "root_job_id": str(context["root"]["job_id"]),
        "release_command": list(context["release_command"]),
        "release_response_mode": release_response_mode,
        "release_returncode": release_returncode,
        "release_stdout": release_stdout,
        "release_stderr": release_stderr,
        "released_observation": observation,
    }
    result_seed["result_body_sha256"] = stable_hash(result_seed)
    validate_root_activation_result(
        submission_root,
        result_seed,
        context=context,
        authorization_sha256=authorization_sha,
    )
    result_sha = exclusive_json(result_path, result_seed)
    hook("activation_result_published")
    # Linearize the caller-visible success return against cancellation without
    # holding the emergency lock during the potentially unbounded result fsync.
    with report_cancel_lock(submission_root):
        for name in ("CANCEL_REQUESTED.json", "CANCEL_RESULT.json", "TRANSACTION_RECOVERY_RESULT.json"):
            require(
                not lexical_entry_exists(submission_root, name, f"root activation return marker {name}"),
                f"root activation return conflicts with {name}",
            )
    del authorization
    return {**result_seed, "root_activation_result_sha256": result_sha, "retry": False}


def activate_root_after_receipt(
    submission_root: Path,
    *,
    boundary: SchedulerBoundary,
    fault_hook: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Serialize activators without holding emergency cancel during result fsync."""

    with root_activation_lock(submission_root):
        return _activate_root_locked(
            submission_root,
            boundary=boundary,
            fault_hook=fault_hook,
        )


@contextmanager
def queued_transaction_barrier(
    submission_root: Path,
    *,
    timeout_seconds: int = 900,
    poll_seconds: float = 0.25,
    sleeper: Callable[[float], None] = time.sleep,
    max_attempts: int | None = None,
    require_committed_receipt: bool = False,
) -> Iterator[dict[str, Any]]:
    """Bypass the outer lock after durable authorization, else wait/share it.

    A correctly released root necessarily has the immutable receipt and release
    authorization already present.  That state is self-authenticating and no
    longer needs the outer claim-to-receipt lock, so a later recovery process
    cannot make running workers time out behind that lock.
    """

    require(timeout_seconds == 900 and poll_seconds == 0.25, "queued receipt barrier contract differs")
    attempts = (
        int(timeout_seconds / poll_seconds) + 1
        if max_attempts is None
        else max_attempts
    )
    require(isinstance(attempts, int) and 1 <= attempts <= 3601, "queued receipt barrier attempt bound differs")
    if require_committed_receipt and _queued_commit_state(submission_root) == "committed":
        yield {
            "schema_version": 1,
            "status": "durable_release_authorization_bypassed_outer_transaction_lock",
            "attempt": 0,
            "maximum_attempts": attempts,
            "timeout_seconds": timeout_seconds,
            "poll_seconds": poll_seconds,
        }
        return
    parent_fd = open_absolute_directory(submission_root.parent, "queued transaction lock parent")
    lock_name = f".{submission_root.name}.TRANSACTION.lock"
    descriptor = os.open(
        lock_name,
        os.O_RDWR | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        dir_fd=parent_fd,
    )
    acquired = False
    try:
        info = os.fstat(descriptor)
        require(
            stat.S_ISREG(info.st_mode)
            and info.st_uid == os.getuid()
            and info.st_gid == os.getgid()
            and info.st_nlink == 1
            and stat.S_IMODE(info.st_mode) == 0o600,
            "queued transaction lock identity/mode differs",
        )
        acquired_attempt: int | None = None
        for attempt in range(1, attempts + 1):
            try:
                fcntl.flock(descriptor, fcntl.LOCK_SH | fcntl.LOCK_NB)
                acquired = True
            except BlockingIOError:
                pass
            else:
                if not require_committed_receipt or _queued_commit_state(submission_root) == "committed":
                    acquired_attempt = attempt
                    break
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                acquired = False
            if attempt < attempts:
                sleeper(poll_seconds)
        require(acquired_attempt is not None, "queued receipt barrier timed out before submission commit/rollback")
        yield {
            "schema_version": 1,
            "status": "exclusive_submission_lifetime_completed",
            "attempt": acquired_attempt,
            "maximum_attempts": attempts,
            "timeout_seconds": timeout_seconds,
            "poll_seconds": poll_seconds,
        }
    finally:
        if acquired:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        os.close(parent_fd)


def _queued_commit_state(submission_root: Path) -> str:
    """Classify immutable receipt+authorization vs recovery-pending state."""

    root_fd = open_absolute_directory(submission_root, "queued commit-state root")
    try:
        with os.scandir(root_fd) as iterator:
            root_entries = {entry.name: entry.stat(follow_symlinks=False) for entry in iterator}
    finally:
        os.close(root_fd)
    terminal = {"CANCEL_REQUESTED.json", "CANCEL_RESULT.json", "TRANSACTION_RECOVERY_RESULT.json"}
    require(not terminal.intersection(root_entries), "queued transaction has cancellation/recovery terminal state")
    try:
        journals = load_journals(submission_root)
    except RuntimeContractError:
        journal_fd = open_absolute_directory(submission_root / "journal", "queued pending journal")
        try:
            with os.scandir(journal_fd) as iterator:
                journal_names = [entry.name for entry in iterator]
        finally:
            os.close(journal_fd)
        if any(re.fullmatch(r"\.[0-9]{4}_[A-Z0-9_]+\.json\.PUBLISHING", name) for name in journal_names):
            return "recovery_pending"
        raise
    abort_events = {"ROLLBACK_CANCELED", "ABORTED", "OUTER_ABORTED", "PRECOMMIT_ABORT_RECOVERED"}
    observed_abort = sorted(row["event"] for row in journals if row["event"] in abort_events)
    require(not observed_abort, f"queued transaction has abort/rollback evidence: {observed_abort}")
    receipt_info = root_entries.get("SUBMISSION_RECEIPT.json")
    receipt_temp = root_entries.get(".SUBMISSION_RECEIPT.json.PUBLISHING")
    if receipt_info is None:
        require(
            receipt_temp is None
            or (
                stat.S_ISREG(receipt_temp.st_mode)
                and receipt_temp.st_uid == os.getuid()
                and receipt_temp.st_gid == os.getgid()
                and receipt_temp.st_nlink == 1
            ),
            "queued unpublished receipt residue differs",
        )
        return "recovery_pending"
    if receipt_info.st_nlink == 2:
        require(
            receipt_temp is not None
            and receipt_temp.st_dev == receipt_info.st_dev
            and receipt_temp.st_ino == receipt_info.st_ino
            and receipt_temp.st_nlink == 2
            and stat.S_IMODE(receipt_info.st_mode) == 0o444,
            "queued linked receipt residue differs",
        )
        return "recovery_pending"
    require(
        receipt_temp is None
        and stat.S_ISREG(receipt_info.st_mode)
        and receipt_info.st_uid == os.getuid()
        and receipt_info.st_gid == os.getgid()
        and receipt_info.st_nlink == 1
        and stat.S_IMODE(receipt_info.st_mode) == 0o444,
        "queued committed receipt identity/mode differs",
    )
    authorization_info = root_entries.get("ROOT_RELEASE_AUTHORIZATION.json")
    authorization_temp = root_entries.get(".ROOT_RELEASE_AUTHORIZATION.json.PUBLISHING")
    if authorization_info is None:
        require(
            authorization_temp is None
            or (
                stat.S_ISREG(authorization_temp.st_mode)
                and authorization_temp.st_uid == os.getuid()
                and authorization_temp.st_gid == os.getgid()
                and authorization_temp.st_nlink == 1
            ),
            "queued unpublished release-authorization residue differs",
        )
        return "recovery_pending"
    if authorization_info.st_nlink == 2:
        require(
            authorization_temp is not None
            and authorization_temp.st_dev == authorization_info.st_dev
            and authorization_temp.st_ino == authorization_info.st_ino
            and authorization_temp.st_nlink == 2
            and stat.S_IMODE(authorization_info.st_mode) == 0o444,
            "queued linked release-authorization residue differs",
        )
        return "recovery_pending"
    require(
        authorization_temp is None
        and stat.S_ISREG(authorization_info.st_mode)
        and authorization_info.st_uid == os.getuid()
        and authorization_info.st_gid == os.getgid()
        and authorization_info.st_nlink == 1
        and stat.S_IMODE(authorization_info.st_mode) == 0o444,
        "queued release-authorization identity/mode differs",
    )
    return "committed"


def _assert_queued_transaction_not_aborted(submission_root: Path) -> None:
    journals = load_journals(submission_root)
    forbidden = {
        "ROLLBACK_CANCELED", "ABORTED", "OUTER_ABORTED", "PRECOMMIT_ABORT_RECOVERED",
    }
    observed = sorted(row["event"] for row in journals if row["event"] in forbidden)
    require(not observed, f"queued transaction has abort/rollback evidence: {observed}")
    root_fd = open_absolute_directory(submission_root, "queued marker root")
    try:
        for name in ("CANCEL_REQUESTED.json", "CANCEL_RESULT.json", "TRANSACTION_RECOVERY_RESULT.json"):
            try:
                os.stat(name, dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            else:
                raise RuntimeContractError(f"queued transaction has a terminal/cancel marker: {name}")
    finally:
        os.close(root_fd)


def bootstrap_queued_entry(
    *,
    submission_root: Path,
    submission_sha256: str,
    snapshot_root: Path,
    executing_file: Path,
    expected_relative: Path,
) -> dict[str, Any]:
    """Reauthenticate contract, full snapshot, receipt, and executing source."""

    require(sys.executable == PINNED_PYTHON, "queued entry requires exact pinned Python 3.11")
    require(sys.version_info[:3] == (3, 11, 15), "queued entry requires Python 3.11.15")
    require(sys.flags.isolated == 1, "queued entry requires Python -I")
    require(sys.flags.no_site == 1, "queued entry requires Python -S")
    require(sys.flags.dont_write_bytecode == 1, "queued entry requires Python -B")
    require("sitecustomize" not in sys.modules and "usercustomize" not in sys.modules, "customize module loaded before bootstrap")
    require(not any("site-packages" in item for item in sys.path), "site-packages loaded before queued bootstrap")
    require(SHA256.fullmatch(submission_sha256) is not None, "queued submission SHA256 differs")
    expected_relative = safe_relative(expected_relative, "queued executing source")
    require(
        snapshot_root == submission_root / "source-snapshot" / "repo",
        "queued snapshot location differs",
    )
    for path, expected_mode, label in (
        (submission_root, 0o700, "queued submission root"),
        (submission_root / "source-snapshot", 0o555, "queued snapshot parent"),
        (snapshot_root, 0o555, "queued snapshot root"),
    ):
        descriptor = open_absolute_directory(path, label)
        try:
            info = os.fstat(descriptor)
            require(
                info.st_uid == os.getuid()
                and info.st_gid == os.getgid()
                and stat.S_IMODE(info.st_mode) == expected_mode,
                f"{label} ownership/mode differs",
            )
        finally:
            os.close(descriptor)
    contract, contract_sha = authenticated_immutable_json(
        submission_root / "SUBMISSION_CONTRACT.json",
        "queued submission contract",
    )
    _contract_probe_bytes, contract_probe_sha, contract_probe_info = authenticated_regular_bytes(
        submission_root / "SUBMISSION_CONTRACT.json",
        "queued submission contract identity probe",
    )
    require(contract_probe_sha == contract_sha, "queued contract identity probe differs")
    require(contract_sha == submission_sha256, "queued submission contract bytes differ")
    require(contract.get("submission_root") == str(submission_root), "queued contract root differs")
    require(contract.get("snapshot_root") == str(snapshot_root), "queued contract snapshot differs")
    validate_submission_contract(
        contract,
        submission_root=submission_root,
        digest=contract_sha,
    )
    inventory = contract.get("snapshot_inventory")
    require(isinstance(inventory, dict) and inventory, "queued snapshot inventory differs")
    require(contract.get("snapshot_inventory_sha256") == stable_hash(inventory), "queued snapshot inventory hash differs")
    verify_snapshot_files(snapshot_root, inventory)
    expected_path = snapshot_root / expected_relative
    require(executing_file.absolute() == expected_path, "queued program was not executed from exact snapshot path")
    _payload, executing_sha, executing_info = authenticated_regular_bytes(expected_path, "queued executing source")
    require(
        executing_sha == inventory.get(str(expected_relative))
        and stat.S_IMODE(executing_info.st_mode) == 0o444,
        "queued executing source identity differs",
    )
    import campaign

    snapshot_manifest, _manifest_file_sha = authenticated_immutable_json(
        snapshot_root / PACKAGE_RELATIVE / "manifest.json",
        "queued snapshot manifest",
    )
    campaign.validate_manifest(snapshot_manifest)
    execution = snapshot_manifest["execution"]
    with queued_transaction_barrier(
        submission_root,
        timeout_seconds=int(execution["queued_receipt_barrier_timeout_seconds"]),
        poll_seconds=float(execution["queued_receipt_barrier_poll_seconds"]),
        require_committed_receipt=True,
    ) as barrier:
        _assert_queued_transaction_not_aborted(submission_root)
        _receipt_bytes, receipt_probe_sha, receipt_probe_info = authenticated_regular_bytes(
            submission_root / "SUBMISSION_RECEIPT.json",
            "queued receipt identity probe",
        )
        receipt, receipt_sha = load_receipt(submission_root)
        require(
            receipt["submission_sha256"] == submission_sha256
            and receipt_probe_sha == receipt_sha
            and receipt_probe_info.st_uid == os.getuid()
            and receipt_probe_info.st_gid == os.getgid()
            and stat.S_IMODE(receipt_probe_info.st_mode) == 0o444,
            "queued receipt submission/identity differs",
        )
        _authorization_bytes, authorization_probe_sha, authorization_probe_info = authenticated_regular_bytes(
            submission_root / "ROOT_RELEASE_AUTHORIZATION.json",
            "queued root release-authorization identity probe",
        )
        activation_context = _root_activation_context(submission_root)
        authorization, authorization_sha = load_root_release_authorization(
            submission_root,
            context=activation_context,
        )
        require(
            authorization_probe_sha == authorization_sha
            and authorization_probe_info.st_uid == os.getuid()
            and authorization_probe_info.st_gid == os.getgid()
            and stat.S_IMODE(authorization_probe_info.st_mode) == 0o444,
            "queued root release-authorization identity differs",
        )
        contract_again, contract_sha_again = authenticated_immutable_json(
            submission_root / "SUBMISSION_CONTRACT.json",
            "queued submission contract revalidation",
        )
        require(
            contract_sha_again == contract_sha and contract_again == contract,
            "queued submission contract changed across bootstrap",
        )
        _contract_probe_bytes_again, contract_probe_sha_again, contract_probe_info_again = authenticated_regular_bytes(
            submission_root / "SUBMISSION_CONTRACT.json",
            "queued submission contract identity revalidation",
        )
        require(
            contract_probe_sha_again == contract_sha_again
            and _identity(contract_probe_info_again) == _identity(contract_probe_info),
            "queued submission contract entry was replaced across bootstrap",
        )
        verify_snapshot_files(snapshot_root, inventory)
        _payload_again, executing_sha_again, executing_info_again = authenticated_regular_bytes(
            expected_path,
            "queued executing source revalidation",
        )
        require(
            executing_sha_again == executing_sha
            and _identity(executing_info_again) == _identity(executing_info),
            "queued executing source changed across bootstrap",
        )
        _receipt_bytes_again, receipt_probe_sha_again, receipt_probe_info_again = authenticated_regular_bytes(
            submission_root / "SUBMISSION_RECEIPT.json",
            "queued receipt identity revalidation",
        )
        receipt_again, receipt_sha_again = load_receipt(submission_root)
        require(
            receipt_again == receipt and receipt_sha_again == receipt_sha,
            "queued receipt changed across bootstrap",
        )
        require(
            receipt_probe_sha_again == receipt_sha_again
            and _identity(receipt_probe_info_again) == _identity(receipt_probe_info),
            "queued receipt entry was replaced across bootstrap",
        )
        _authorization_bytes_again, authorization_probe_sha_again, authorization_probe_info_again = authenticated_regular_bytes(
            submission_root / "ROOT_RELEASE_AUTHORIZATION.json",
            "queued root release-authorization identity revalidation",
        )
        authorization_again, authorization_sha_again = load_root_release_authorization(
            submission_root,
            context=activation_context,
        )
        require(
            authorization_again == authorization
            and authorization_sha_again == authorization_sha
            and authorization_probe_sha_again == authorization_sha_again
            and _identity(authorization_probe_info_again)
            == _identity(authorization_probe_info),
            "queued root release authorization changed across bootstrap",
        )
        _assert_queued_transaction_not_aborted(submission_root)
    return {
        "schema_version": 1,
        "status": "queued_snapshot_and_submission_reauthenticated",
        "contract": contract,
        "submission_sha256": submission_sha256,
        "snapshot_inventory_sha256": contract["snapshot_inventory_sha256"],
        "receipt": receipt,
        "receipt_sha256": receipt_sha,
        "root_release_authorization": authorization,
        "root_release_authorization_sha256": authorization_sha,
        "executing_source_sha256": executing_sha,
        "transaction_barrier": barrier,
    }


def load_journals(submission_root: Path) -> list[dict[str, Any]]:
    journal_root = submission_root / "journal"
    descriptor = open_absolute_directory(journal_root, "submission journal")
    try:
        with os.scandir(descriptor) as iterator:
            entries = sorted(
                ((entry.name, entry.stat(follow_symlinks=False)) for entry in iterator),
                key=lambda row: row[0],
            )
    finally:
        os.close(descriptor)
    rows: list[dict[str, Any]] = []
    names: set[str] = set()
    for name, info in entries:
        match = re.fullmatch(r"([0-9]{4})_([A-Z0-9_]+)\.json", name)
        require(match is not None, f"journal entry name differs: {name}")
        require(name not in names, f"journal entry is duplicated: {name}")
        names.add(name)
        require(
            stat.S_ISREG(info.st_mode)
            and not stat.S_ISLNK(info.st_mode)
            and info.st_uid == os.getuid()
            and info.st_gid == os.getgid()
            and info.st_nlink == 1
            and stat.S_IMODE(info.st_mode) == 0o444,
            f"journal entry identity/mode differs: {name}",
        )
        payload, digest, opened = authenticated_regular_bytes(journal_root / name, f"journal {name}")
        require(_identity(opened) == _identity(info), f"journal directory/open identity differs: {name}")
        value = parse_json_bytes(payload, f"journal {name}")
        require(
            set(value) == {"schema_version", "campaign_id", "ordinal", "event", "payload"}
            and value.get("schema_version") == 1
            and value.get("campaign_id") == CAMPAIGN_ID
            and value.get("ordinal") == int(match.group(1))
            and value.get("event") == match.group(2)
            and isinstance(value.get("payload"), dict),
            f"journal schema/identity differs: {name}",
        )
        rows.append({**value, "sha256": digest, "path": str(journal_root / name)})
    require(rows and rows[0]["event"] == "CLAIMED", "submission claim journal is absent")
    return rows


class _TransactionRecoveryLockHandle:
    """Process-local proof that the exact external transaction lock is held."""

    def __init__(self, submission_root: Path, descriptor: int) -> None:
        self.submission_root = submission_root
        self.descriptor = descriptor
        self.active = True


def _validate_transaction_recovery_lock_handle(
    submission_root: Path,
    handle: _TransactionRecoveryLockHandle,
) -> None:
    require(
        isinstance(handle, _TransactionRecoveryLockHandle)
        and handle.active
        and handle.submission_root == submission_root,
        "existing transaction lock handle differs",
    )
    info = os.fstat(handle.descriptor)
    parent_fd = open_absolute_directory(submission_root.parent, "existing transaction lock parent")
    try:
        entry = os.stat(
            f".{submission_root.name}.TRANSACTION.lock",
            dir_fd=parent_fd,
            follow_symlinks=False,
        )
    finally:
        os.close(parent_fd)
    require(
        _identity(info) == _identity(entry)
        and stat.S_ISREG(info.st_mode)
        and info.st_uid == os.getuid()
        and info.st_gid == os.getgid()
        and info.st_nlink == 1
        and stat.S_IMODE(info.st_mode) == 0o600,
        "existing transaction lock identity/mode differs",
    )


@contextmanager
def transaction_recovery_lock(
    submission_root: Path,
) -> Iterator[_TransactionRecoveryLockHandle]:
    """Serialize initial claim, submission, and recovery outside the claimed root."""

    require(submission_root.is_absolute(), "transaction lock submission root differs")
    parent_fd = open_absolute_directory(submission_root.parent, "transaction lock parent")
    name = f".{submission_root.name}.TRANSACTION.lock"
    descriptor = os.open(
        name,
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=parent_fd,
    )
    try:
        info = os.fstat(descriptor)
        require(
            stat.S_ISREG(info.st_mode)
            and info.st_uid == os.getuid()
            and info.st_gid == os.getgid()
            and info.st_nlink == 1
            and stat.S_IMODE(info.st_mode) == 0o600,
            "transaction lock identity/mode differs",
        )
        os.fsync(parent_fd)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        handle = _TransactionRecoveryLockHandle(submission_root, descriptor)
        yield handle
    finally:
        if "handle" in locals():
            handle.active = False
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        os.close(parent_fd)


def _recovery_evidence(submission_root: Path, event: str, value: Mapping[str, Any]) -> Path:
    root = submission_root / "recovery"
    try:
        _mkdir_exact(root, 0o700, "recovery evidence")
    except FileExistsError:
        descriptor = open_absolute_directory(root, "existing recovery evidence")
        try:
            info = os.fstat(descriptor)
            require(
                info.st_uid == os.getuid()
                and info.st_gid == os.getgid()
                and stat.S_IMODE(info.st_mode) == 0o700,
                "existing recovery evidence root identity/mode differs",
            )
        finally:
            os.close(descriptor)
    repair_publication_residues(
        root,
        allowed_final=re.compile(r"[0-9]+-[0-9]+-[0-9a-f]{16}_[A-Z0-9_]+\.json"),
    )
    token = f"{time.time_ns()}-{os.getpid()}-{secrets.token_hex(8)}"
    path = root / f"{token}_{event}.json"
    exclusive_json(
        path,
        {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "event": event,
            "payload": dict(value),
        },
    )
    return path


def _journal_ledger(journals: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "ordinal": int(row["ordinal"]),
            "event": str(row["event"]),
            "sha256": str(row["sha256"]),
        }
        for row in journals
    ]


def _validated_role_id_map(
    value: object,
    roles: Sequence[str],
    label: str,
) -> dict[str, list[str]]:
    require(
        isinstance(value, dict) and set(value) == set(roles),
        f"{label} role map differs",
    )
    result: dict[str, list[str]] = {}
    global_ids: set[str] = set()
    for role in roles:
        raw = value.get(role)
        require(isinstance(raw, list), f"{label} IDs are not a list: {role}")
        ids = [str(item) for item in raw]
        require(
            ids == sorted(set(ids), key=int)
            and all(JOB_ID.fullmatch(item) is not None for item in ids)
            and global_ids.isdisjoint(ids),
            f"{label} IDs differ: {role}",
        )
        global_ids.update(ids)
        result[role] = ids
    return result


def _validate_precontract_journals(journals: Sequence[Mapping[str, Any]]) -> None:
    """Prove that no scheduler-mutation evidence exists without a contract."""

    identities = [(int(row["ordinal"]), str(row["event"])) for row in journals]
    allowed = {
        ((0, "CLAIMED"),),
        ((0, "CLAIMED"), (1, "SNAPSHOT_SEALED")),
        ((0, "CLAIMED"), (998, "OUTER_ABORTED")),
        ((0, "CLAIMED"), (1, "SNAPSHOT_SEALED"), (998, "OUTER_ABORTED")),
    }
    require(tuple(identities) in allowed, "pre-contract journal inventory is not admissible")
    claim = journals[0]["payload"]
    require(
        set(claim) == {"claim_token", "pid", "created_ns"}
        and re.fullmatch(r"[0-9a-f]{64}", str(claim.get("claim_token", ""))) is not None
        and isinstance(claim.get("pid"), int)
        and not isinstance(claim.get("pid"), bool)
        and claim["pid"] > 0
        and isinstance(claim.get("created_ns"), int)
        and not isinstance(claim.get("created_ns"), bool)
        and claim["created_ns"] > 0,
        "pre-contract claim payload differs",
    )
    snapshot_rows = [row for row in journals if row["event"] == "SNAPSHOT_SEALED"]
    if snapshot_rows:
        payload = snapshot_rows[0]["payload"]
        require(
            set(payload) == {"inventory_sha256", "file_count"}
            and SHA256.fullmatch(str(payload.get("inventory_sha256", ""))) is not None
            and isinstance(payload.get("file_count"), int)
            and not isinstance(payload.get("file_count"), bool)
            and payload["file_count"] > 0,
            "pre-contract snapshot journal payload differs",
        )
    aborted_rows = [row for row in journals if row["event"] == "OUTER_ABORTED"]
    if aborted_rows:
        payload = aborted_rows[0]["payload"]
        require(
            set(payload) == {"error", "recovery_required"}
            and isinstance(payload.get("error"), str)
            and payload["error"]
            and payload.get("recovery_required") is False,
            "pre-contract outer-abort payload differs",
        )


def _validate_terminal_recovery_result(
    value: Mapping[str, Any],
    *,
    manifest: Mapping[str, Any],
    submission_root: Path,
    journals: Sequence[Mapping[str, Any]],
) -> None:
    """Authenticate a prior terminal result before it can suppress scheduler work."""

    import campaign

    ledger = _journal_ledger(journals)
    common = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "submission_root": str(submission_root),
        "journal_ledger": ledger,
        "journal_ledger_sha256": stable_hash(ledger),
        "new_jobs_created": 0,
    }
    status = value.get("status")
    if status == "aborted_before_scheduler_contract":
        require(set(value) == {
            *common,
            "status",
            "scheduler_calls",
        }, "pre-contract recovery result schema differs")
        _validate_precontract_journals(journals)
        require(
            all(value.get(key) == expected for key, expected in common.items())
            and value.get("scheduler_calls") == 0
            and not (submission_root / "SUBMISSION_CONTRACT.json").exists()
            and not (submission_root / "SUBMISSION_RECEIPT.json").exists()
            and not any(row["event"] == "READY_TO_COMMIT" for row in journals),
            "pre-contract recovery result evidence differs",
        )
        return
    require(status == "precommit_transaction_reconciled_and_aborted", "terminal recovery result status differs")
    expected_fields = {
        *common,
        "status", "submission_sha256", "manifest_sha256", "snapshot_root",
        "dag_names", "historical_job_ids_by_role", "live_job_ids_before_cancel_by_role",
        "prior_rollback_canceled", "cancellation", "scheduler_calls",
    }
    require(set(value) == expected_fields, "precommit recovery result schema differs")
    require(all(value.get(key) == expected for key, expected in common.items()), "precommit recovery common evidence differs")
    contract, contract_sha = authenticated_immutable_json(
        submission_root / "SUBMISSION_CONTRACT.json",
        "terminal recovery submission contract",
    )
    validate_submission_contract(
        contract,
        submission_root=submission_root,
        digest=contract_sha,
        verify_runtime_authority=False,
    )
    require(
        value.get("submission_sha256") == contract_sha
        and value.get("manifest_sha256") == campaign.manifest_sha256(manifest)
        and contract.get("manifest_sha256") == value["manifest_sha256"]
        and contract.get("submission_root") == str(submission_root),
        "terminal recovery contract binding differs",
    )
    snapshot_root = Path(str(value.get("snapshot_root", "")))
    require(
        snapshot_root.is_absolute()
        and str(snapshot_root) == contract.get("snapshot_root"),
        "terminal recovery snapshot binding differs",
    )
    snapshot_manifest, _manifest_sha = authenticated_immutable_json(
        snapshot_root / PACKAGE_RELATIVE / "manifest.json",
        "terminal recovery snapshot manifest",
    )
    campaign.validate_manifest(snapshot_manifest)
    require(campaign.manifest_sha256(snapshot_manifest) == value["manifest_sha256"], "terminal recovery snapshot manifest differs")
    nodes = campaign.scheduler_dag(snapshot_manifest)
    roles = [node.name for node in nodes]
    require(value.get("dag_names") == roles, "terminal recovery DAG names differ")
    historical = _validated_role_id_map(value.get("historical_job_ids_by_role"), roles, "historical recovery")
    live = _validated_role_id_map(value.get("live_job_ids_before_cancel_by_role"), roles, "live recovery")
    derived: dict[str, list[str]] = {role: [] for role in roles}
    for row in journals:
        payload = row["payload"]
        role = payload.get("role")
        if role not in derived:
            continue
        candidates: list[str] = []
        if JOB_ID.fullmatch(str(payload.get("job_id", ""))) is not None:
            candidates.append(str(payload["job_id"]))
        raw_ids = payload.get("job_ids")
        if isinstance(raw_ids, list):
            candidates.extend(str(item) for item in raw_ids if JOB_ID.fullmatch(str(item)) is not None)
        derived[str(role)].extend(candidates)
    derived = {role: sorted(set(ids), key=int) for role, ids in derived.items()}
    require(historical == derived, "terminal recovery historical IDs differ from journals")
    prior_rollback = any(row["event"] == "ROLLBACK_CANCELED" for row in journals)
    require(value.get("prior_rollback_canceled") is prior_rollback, "terminal recovery prior rollback evidence differs")
    reverse_ids: list[str] = []
    for node in reversed(nodes):
        reverse_ids.extend(sorted(live[node.name], key=int, reverse=True))
    cancellation = value.get("cancellation")
    if not reverse_ids:
        require(cancellation is None, "terminal recovery records cancellation without live IDs")
    else:
        require(isinstance(cancellation, dict) and set(cancellation) == {
            "command", "job_ids", "stdout", "stderr", "control_plane",
            "post_cancel_reconciliation",
        }, "terminal recovery cancellation schema differs")
        require(
            cancellation.get("command")
            == [str(snapshot_manifest["execution"]["scancel"]), "--quiet", *reverse_ids]
            and cancellation.get("job_ids") == reverse_ids
            and isinstance(cancellation.get("stdout"), str)
            and isinstance(cancellation.get("stderr"), str)
            and cancellation.get("control_plane") == contract.get("scheduler_control_plane"),
            "terminal recovery cancellation identity differs",
        )
        post = cancellation.get("post_cancel_reconciliation")
        require(
            isinstance(post, dict)
            and set(post) == {"status", "attempt_count", "attempts"}
            and post.get("status") == "terminal_or_absent"
            and isinstance(post.get("attempt_count"), int)
            and isinstance(post.get("attempts"), list)
            and len(post["attempts"]) == post["attempt_count"]
            and post["attempts"]
            and post["attempts"][-1].get("active") == {},
            "terminal recovery post-cancel evidence differs",
        )
    require(isinstance(value.get("scheduler_calls"), int) and value["scheduler_calls"] >= 11, "terminal recovery scheduler-call count differs")


@contextmanager
def _existing_transaction_recovery_lock(
    submission_root: Path,
    handle: _TransactionRecoveryLockHandle,
) -> Iterator[None]:
    """Validate a caller-held lock before and after a locked recovery body."""

    _validate_transaction_recovery_lock_handle(submission_root, handle)
    try:
        yield
    finally:
        _validate_transaction_recovery_lock_handle(submission_root, handle)


def _recover_transaction_locked(
    manifest: Mapping[str, Any],
    *,
    submission_root: Path,
    boundary: SchedulerBoundary,
    lock_handle: _TransactionRecoveryLockHandle,
) -> dict[str, Any]:
    """Recover while the caller continuously holds the exact external lock."""

    import campaign

    campaign.validate_manifest(manifest)
    with _existing_transaction_recovery_lock(submission_root, lock_handle):
        if not submission_root.exists():
            private_preclaim = submission_root.parent / f".{submission_root.name}.PRECLAIM"
            require(private_preclaim.exists(), "no Exp24 transaction namespace or private preclaim exists")
            begin_transaction(submission_root, secrets.token_hex(32))
        repair_publication_residues(
            submission_root,
            allowed_final=re.compile(
                r"(?:SUBMISSION_CONTRACT|SUBMISSION_RECEIPT|TRANSACTION_RECOVERY_RESULT)\.json"
            ),
        )
        repair_publication_residues(
            submission_root / "journal",
            allowed_final=re.compile(r"[0-9]{4}_[A-Z0-9_]+\.json"),
        )
        journals = load_journals(submission_root)
        recovery_result_path = submission_root / "TRANSACTION_RECOVERY_RESULT.json"
        if recovery_result_path.exists():
            prior, prior_sha = authenticated_immutable_json(recovery_result_path, "transaction recovery result")
            _validate_terminal_recovery_result(
                prior,
                manifest=manifest,
                submission_root=submission_root,
                journals=journals,
            )
            return {**prior, "recovery_result_sha256": prior_sha, "retry": True}
        ready = [row for row in journals if row["event"] == "READY_TO_COMMIT"]
        require(len(ready) <= 1, "transaction has multiple READY_TO_COMMIT records")
        contradictory = [
            row["event"]
            for row in journals
            if row["event"] in {"ROLLBACK_CANCELED", "ABORTED", "OUTER_ABORTED"}
        ]
        receipt_path = submission_root / "SUBMISSION_RECEIPT.json"
        if receipt_path.exists():
            receipt, receipt_sha = load_receipt(
                submission_root,
                verify_full_snapshot=False,
            )
            require(len(ready) == 1, "receipt exists without durable READY_TO_COMMIT")
            require(receipt == ready[0]["payload"], "receipt differs from durable READY_TO_COMMIT")
            require(not contradictory, f"receipt conflicts with rollback/abort history: {contradictory}")
            with report_cancel_lock(submission_root):
                if (submission_root / "CANCEL_REQUESTED.json").exists() or (
                    submission_root / "CANCEL_RESULT.json"
                ).exists():
                    cancellation = _explicit_cancel_locked(
                        submission_root,
                        plan=cancellation_plan(submission_root),
                        boundary=boundary,
                        execution=manifest["execution"],
                    )
                    return {
                        "schema_version": 1,
                        "status": "receipt_committed_cancellation_resumed",
                        "receipt": receipt,
                        "receipt_sha256": receipt_sha,
                        "cancellation": cancellation,
                        "scheduler_calls": len(boundary.calls),
                    }
                authorization = _authorize_root_release_locked(
                    submission_root,
                    boundary=boundary,
                )
            return {
                "schema_version": 1,
                "status": "receipt_already_committed_root_release_authorized",
                "receipt": receipt,
                "receipt_sha256": receipt_sha,
                "authorization": authorization,
                "scheduler_calls": len(boundary.calls),
                "_post_transaction_activation_status": (
                    "receipt_already_committed_root_activation_recovered"
                ),
            }
        if ready:
            require(not contradictory, f"READY_TO_COMMIT conflicts with rollback/abort history: {contradictory}")
            require(not (submission_root / "CANCEL_REQUESTED.json").exists(), "READY_TO_COMMIT conflicts with cancellation")
            candidate = ready[0]["payload"]
            require(
                candidate.get("campaign_id") == CAMPAIGN_ID
                and candidate.get("submission_root") == str(submission_root)
                and isinstance(candidate.get("jobs"), list)
                and len(candidate["jobs"]) == 11,
                "READY_TO_COMMIT receipt differs",
            )
            exclusive_json(receipt_path, candidate)
            receipt, receipt_sha = load_receipt(
                submission_root,
                verify_full_snapshot=False,
            )
            evidence = _recovery_evidence(
                submission_root,
                "RECEIPT_COMMITTED_FROM_READY",
                {"ready_journal_sha256": ready[0]["sha256"], "receipt_sha256": receipt_sha},
            )
            with report_cancel_lock(submission_root):
                authorization = _authorize_root_release_locked(
                    submission_root,
                    boundary=boundary,
                )
            return {
                "schema_version": 1,
                "status": "receipt_committed_from_durable_ready_root_release_authorized",
                "receipt": receipt,
                "receipt_sha256": receipt_sha,
                "evidence": str(evidence),
                "authorization": authorization,
                "scheduler_calls": len(boundary.calls),
                "_post_transaction_activation_status": (
                    "receipt_committed_from_durable_ready_and_root_activated"
                ),
            }

        contract_path = submission_root / "SUBMISSION_CONTRACT.json"
        if not contract_path.exists():
            ledger = _journal_ledger(journals)
            result = {
                "schema_version": 1,
                "campaign_id": CAMPAIGN_ID,
                "submission_root": str(submission_root),
                "status": "aborted_before_scheduler_contract",
                "journal_ledger": ledger,
                "journal_ledger_sha256": stable_hash(ledger),
                "new_jobs_created": 0,
                "scheduler_calls": 0,
            }
            _validate_terminal_recovery_result(
                result,
                manifest=manifest,
                submission_root=submission_root,
                journals=journals,
            )
            result_sha = exclusive_json(recovery_result_path, result)
            return {**result, "recovery_result_sha256": result_sha, "retry": False}

        contract, submission_sha = authenticated_immutable_json(contract_path, "submission contract")
        validate_submission_contract(
            contract,
            submission_root=submission_root,
            digest=submission_sha,
            verify_runtime_authority=False,
        )
        require(
            contract.get("campaign_id") == CAMPAIGN_ID
            and contract.get("submission_root") == str(submission_root)
            and contract.get("manifest_sha256") == campaign.manifest_sha256(manifest),
            "recovery submission contract differs",
        )
        snapshot_root = Path(str(contract.get("snapshot_root", "")))
        require(snapshot_root.is_absolute(), "recovery snapshot root differs")
        nodes = campaign.scheduler_dag(manifest)
        names = _node_job_names(nodes, submission_sha[:16])
        comment = f"treewm-exp24:{submission_sha}"
        discovered: dict[str, list[str]] = {node.name: [] for node in nodes}
        # Durable acceptance records accelerate recovery, but exact-name scheduler
        # reconciliation is authoritative and also catches a lost sbatch response.
        for row in journals:
            payload = row["payload"]
            role = payload.get("role")
            if role in discovered:
                values: list[str] = []
                if JOB_ID.fullmatch(str(payload.get("job_id", ""))):
                    values.append(str(payload["job_id"]))
                raw_ids = payload.get("job_ids")
                if isinstance(raw_ids, list):
                    values.extend(str(value) for value in raw_ids if JOB_ID.fullmatch(str(value)))
                discovered[str(role)].extend(values)
        live: dict[str, list[str]] = {node.name: [] for node in nodes}
        for role, name in names.items():
            values = reconcile_job_ids(boundary, manifest["execution"], name, comment, snapshot_root)
            live[role].extend(values)
        reverse_ids: list[str] = []
        for node in reversed(nodes):
            for job_id in sorted(set(live[node.name]), key=int, reverse=True):
                if job_id not in reverse_ids:
                    reverse_ids.append(job_id)
        cancellation: dict[str, Any] | None = None
        if reverse_ids:
            cancellation = cancel_exact(
                boundary,
                manifest["execution"],
                reverse_ids,
                snapshot_root,
            )
            cancellation["post_cancel_reconciliation"] = post_cancel_reconciliation(
                boundary,
                manifest["execution"],
                job_names=names,
                comment=comment,
                cwd=snapshot_root,
                submission_root=submission_root,
            )
            require(
                cancellation["post_cancel_reconciliation"]["status"] == "terminal_or_absent",
                "recovery cancellation convergence remains pending",
            )
        result = {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "submission_root": str(submission_root),
            "status": "precommit_transaction_reconciled_and_aborted",
            "submission_sha256": submission_sha,
            "manifest_sha256": campaign.manifest_sha256(manifest),
            "snapshot_root": str(snapshot_root),
            "dag_names": [node.name for node in nodes],
            "historical_job_ids_by_role": {
                role: sorted(set(values), key=int) for role, values in discovered.items()
            },
            "live_job_ids_before_cancel_by_role": {
                role: sorted(set(values), key=int) for role, values in live.items()
            },
            "prior_rollback_canceled": "ROLLBACK_CANCELED" in contradictory,
            "cancellation": cancellation,
            "journal_ledger": _journal_ledger(journals),
            "journal_ledger_sha256": stable_hash(_journal_ledger(journals)),
            "new_jobs_created": 0,
            "scheduler_calls": len(boundary.calls) + len(boundary.recovery_events),
        }
        _validate_terminal_recovery_result(
            result,
            manifest=manifest,
            submission_root=submission_root,
            journals=journals,
        )
        result_sha = exclusive_json(recovery_result_path, result)
        return {**result, "recovery_result_sha256": result_sha, "retry": False}


def recover_transaction(
    manifest: Mapping[str, Any],
    *,
    submission_root: Path,
    boundary: SchedulerBoundary,
) -> dict[str, Any]:
    """Recover a crash at any claim-to-receipt point under the external lock."""

    with transaction_recovery_lock(submission_root) as lock_handle:
        result = _recover_transaction_locked(
            manifest,
            submission_root=submission_root,
            boundary=boundary,
            lock_handle=lock_handle,
        )
    post_status = result.pop("_post_transaction_activation_status", None)
    if post_status is None:
        return result
    require(isinstance(post_status, str) and post_status, "post-transaction activation status differs")
    activation = activate_root_after_receipt(
        submission_root,
        boundary=boundary,
    )
    return {
        **result,
        "status": post_status,
        "activation": activation,
        "scheduler_calls": len(boundary.calls),
    }


@contextmanager
def root_activation_lock(submission_root: Path) -> Iterator[None]:
    """Serialize activation/recovery without blocking emergency cancellation."""

    parent_fd = open_absolute_directory(submission_root, "root activation lock parent")
    descriptor = os.open(
        "ROOT_ACTIVATION.lock",
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=parent_fd,
    )
    try:
        info = os.fstat(descriptor)
        require(
            stat.S_ISREG(info.st_mode)
            and info.st_uid == os.getuid()
            and info.st_gid == os.getgid()
            and info.st_nlink == 1
            and stat.S_IMODE(info.st_mode) == 0o600,
            "root activation lock identity/mode differs",
        )
        os.fsync(parent_fd)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        os.close(parent_fd)


@contextmanager
def report_cancel_lock(submission_root: Path) -> Iterator[None]:
    parent_fd = open_absolute_directory(submission_root, "report/cancel lock parent")
    descriptor = os.open(
        "REPORT_CANCEL.lock",
        os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
        0o600,
        dir_fd=parent_fd,
    )
    try:
        info = os.fstat(descriptor)
        require(
            stat.S_ISREG(info.st_mode)
            and info.st_uid == os.getuid()
            and info.st_gid == os.getgid()
            and info.st_nlink == 1
            and stat.S_IMODE(info.st_mode) == 0o600,
            "report/cancel lock identity/mode differs",
        )
        os.fsync(parent_fd)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        os.close(parent_fd)


def cancellation_plan(submission_root: Path) -> dict[str, Any]:
    receipt, receipt_sha = load_receipt(
        submission_root,
        verify_full_snapshot=False,
    )
    jobs = list(receipt["jobs"])
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": receipt["submission_sha256"],
        "receipt_sha256": receipt_sha,
        "job_ids_reverse_dag": [str(row["job_id"]) for row in reversed(jobs)],
        "roles_reverse_dag": [str(row["role"]) for row in reversed(jobs)],
    }


def _explicit_cancel_locked(
    submission_root: Path,
    *,
    plan: Mapping[str, Any],
    boundary: SchedulerBoundary,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    repair_publication_residues(
        submission_root,
        allowed_final=re.compile(r"(?:CANCEL_REQUESTED|CANCEL_RESULT)\.json"),
    )
    require(not (submission_root / "immutable-report" / "REPORT_COMMIT.json").exists(), "immutable report already committed")
    latch = {
        **plan,
        "status": "cancellation_latched_before_scheduler_call",
    }
    latch_path = submission_root / "CANCEL_REQUESTED.json"
    if latch_path.exists():
        existing_latch, latch_sha = authenticated_immutable_json(latch_path, "cancellation latch")
        require(existing_latch == latch, "existing cancellation latch differs")
    else:
        latch_sha = exclusive_json(latch_path, latch)
    final_path = submission_root / "CANCEL_RESULT.json"
    expected_final = {
        **plan,
        "status": "cancellation_converged_terminal_or_absent",
        "cancel_latch_sha256": latch_sha,
    }
    if final_path.exists():
        existing_final, final_sha = authenticated_immutable_json(final_path, "cancellation result")
        require(existing_final == expected_final, "existing cancellation result differs")
        return {**existing_final, "cancel_result_sha256": final_sha, "retry": True}
    intent = {
        **plan,
        "status": "exact_cancel_attempt_intent",
        "cancel_latch_sha256": latch_sha,
    }
    intent_path = _recovery_evidence(submission_root, "CANCEL_ATTEMPT_INTENT", intent)
    try:
        result = cancel_exact(boundary, execution, plan["job_ids_reverse_dag"], submission_root)
    except BaseException as exc:
        failed = {**intent, "status": "exact_cancel_attempt_failed", "error": repr(exc)}
        failure_path = _recovery_evidence(submission_root, "CANCEL_ATTEMPT_FAILED", failed)
        raise RuntimeContractError(
            f"exact cancellation failed after durable latch/intent; retry is safe; evidence={failure_path}: {exc}"
        ) from exc
    receipt, _receipt_sha = load_receipt(
        submission_root,
        verify_full_snapshot=False,
    )
    names = {str(row["role"]): str(row["job_name"]) for row in receipt["jobs"]}
    result["post_cancel_reconciliation"] = post_cancel_reconciliation(
        boundary,
        execution,
        job_names=names,
        comment=f"treewm-exp24:{receipt['submission_sha256']}",
        cwd=Path(str(receipt["snapshot_root"])),
        submission_root=submission_root,
    )
    attempt = {
        **intent,
        "status": (
            "exact_cancel_converged"
            if result["post_cancel_reconciliation"]["status"] == "terminal_or_absent"
            else "exact_cancel_convergence_pending"
        ),
        "intent_path": str(intent_path),
        "scheduler": result,
    }
    attempt_path = _recovery_evidence(submission_root, "CANCEL_ATTEMPT_RESULT", attempt)
    if result["post_cancel_reconciliation"]["status"] != "terminal_or_absent":
        return {**attempt, "attempt_path": str(attempt_path), "retry_required": True}
    final_sha = exclusive_json(final_path, expected_final)
    return {
        **expected_final,
        "cancel_result_sha256": final_sha,
        "attempt_path": str(attempt_path),
        "retry": False,
    }


def explicit_cancel(
    submission_root: Path,
    *,
    boundary: SchedulerBoundary,
    execution: Mapping[str, Any],
) -> dict[str, Any]:
    """Latch and idempotently finish exact-ID cancellation under one lock."""

    plan = cancellation_plan(submission_root)
    with report_cancel_lock(submission_root):
        return _explicit_cancel_locked(
            submission_root,
            plan=plan,
            boundary=boundary,
            execution=execution,
        )


def publish_report_quartet(
    submission_root: Path,
    *,
    bundle: Mapping[str, Any],
    decision: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Remain fail-closed until the exact scientific evidence schemas are sealed."""

    del submission_root, bundle, decision, provenance
    raise RuntimeContractError(
        "M2A report publication is disabled: exact all-stage/all-40 lineage, "
        "200-cell paired-rail, seed-level df=3 inference, provenance, and formal-job "
        "evidence schemas are not sealed"
    )


def validate_requeue_ready(
    submission_root: Path,
    generation_root: Path,
    value: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    """Purely authenticate one same-stage readiness record before any mutation."""

    receipt, receipt_sha = load_receipt(submission_root)
    required = {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "submission_receipt_sha256",
        "role",
        "stage_target",
        "cell_index",
        "restart_count",
        "array_job_id",
        "array_task_id",
        "completed_updates",
        "checkpoint_sha256",
        "checkpoint_identity_sha256",
        "run_identity_sha256",
        "wandb_id",
        "promotion_authority",
    }
    require(set(value) == required, "requeue-ready schema differs")
    require(
        value.get("schema_version") == 1
        and value.get("status") == "ready_for_same_stage_same_cell_requeue"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == receipt["submission_sha256"]
        and value.get("submission_receipt_sha256") == receipt_sha,
        "requeue-ready submission identity differs",
    )
    role = value.get("role")
    require(isinstance(role, str), "requeue role differs")
    matching = [row for row in receipt["jobs"] if row["role"] == role]
    require(len(matching) == 1, "requeue role does not identify one receipt job")
    job = matching[0]
    require(str(value.get("array_job_id")) == str(job["job_id"]), "requeue array job differs from receipt")
    task = value.get("array_task_id")
    cell = value.get("cell_index")
    restart = value.get("restart_count")
    require(
        isinstance(task, int)
        and not isinstance(task, bool)
        and isinstance(cell, int)
        and not isinstance(cell, bool)
        and task == cell,
        "requeue array/cell identity differs",
    )
    require(
        isinstance(restart, int)
        and not isinstance(restart, bool)
        and restart >= 0,
        "requeue current-attempt restart count differs",
    )
    if role.startswith("train_"):
        stage = int(role.removeprefix("train_"))
        require(stage in (2_000, 25_000, 100_000, 1_000_000), "requeue training stage differs")
        require(value.get("stage_target") == stage, "requeue stage target differs")
        require(0 <= task < 40, "training requeue cell differs")
        completed = value.get("completed_updates")
        require(
            isinstance(completed, int)
            and not isinstance(completed, bool)
            and 0 <= completed <= stage,
            "training requeue update differs",
        )
    elif role == "heldout_eval":
        require(value.get("stage_target") is None, "held-out requeue has a training stage")
        require(0 <= task < 200, "held-out requeue cell differs")
        require(
            isinstance(value.get("completed_updates"), int)
            and not isinstance(value.get("completed_updates"), bool)
            and value["completed_updates"] >= 0,
            "held-out episode progress differs",
        )
    else:
        raise RuntimeContractError("only array roles may requeue")
    require(value.get("promotion_authority") == "none_within_stage_requeue_only", "requeue attempts cross-stage promotion")
    require(
        all(SHA256.fullmatch(str(value.get(name, ""))) for name in (
            "checkpoint_sha256", "checkpoint_identity_sha256", "run_identity_sha256"
        )),
        "requeue checkpoint/run identity hash differs",
    )
    require(isinstance(value.get("wandb_id"), str) and value["wandb_id"], "requeue W&B identity differs")
    try:
        generation_root.relative_to(submission_root)
    except ValueError as exc:
        raise RuntimeContractError("requeue generation root escapes submission") from exc
    return receipt, receipt_sha


def seal_requeue_ready(
    submission_root: Path,
    generation_root: Path,
    value: Mapping[str, Any],
) -> str:
    """Publish a structural requeue-intent scaffold with no mutation authority."""

    validate_requeue_ready(submission_root, generation_root, value)
    return exclusive_json(generation_root / "REQUEUE_READY.json", value)


def call_same_run_requeue(
    submission_root: Path,
    generation_root: Path,
    ready: Mapping[str, Any],
    *,
    boundary: SchedulerBoundary,
    execution: Mapping[str, Any],
    cwd: Path,
) -> dict[str, Any]:
    """Reject every requeue mutation until its scientific identity transaction seals.

    ``REQUEUE_READY`` is only a structural, non-authoritative scaffold in M2A.  A
    later protocol revision must bind exact run/W&B/checkpoint lineage and add an
    idempotent scheduler observation/result transaction serialized with cancel.
    Keeping this rejection before every read, record publication, and scheduler
    call makes a syntactically plausible or pre-positioned record powerless.
    """

    del submission_root, generation_root, ready, boundary, execution, cwd
    raise RuntimeContractError(
        "same-stage requeue mutation is disabled in M2A: scientific identity, "
        "cancel serialization, and lost-response reconciliation are unsealed"
    )


def runtime_description(manifest: Mapping[str, Any]) -> dict[str, Any]:
    import campaign

    nodes = campaign.scheduler_dag(manifest)
    return {
        "schema_version": 1,
        "phase": "m2a_orders_0_3_runtime_authority_scaffold_execution_blocked",
        "campaign_id": CAMPAIGN_ID,
        "nodes": [asdict(node) for node in nodes],
        "node_count": len(nodes),
        "edge_count": sum(node.dependency is not None for node in nodes),
        "transaction": {
            "exclusive_claim": True,
            "append_only_journals": True,
            "reconcile_all_transaction_names_on_any_failure": True,
            "cancel_exact_ids_reverse_dag": True,
            "receipt_commits_submission_inventory_but_is_not_execution_activation": True,
            "root_is_submitted_user_held": True,
            "durable_authorization_precedes_exact_release": True,
            "outer_transaction_lock_released_before_root_release": True,
            "queued_workers_bypass_outer_lock_after_authorization": True,
            "execution_activation_implemented": True,
            "activation_crash_recovery_implemented": True,
            "release_side_effect_serialized_with_cancellation": True,
            "activation_result_fsync_does_not_hold_cancel_lock": True,
            "accepted_dependencies_observed": True,
            "array_dependency_wildcards_required": True,
            "kill_invalid_dependency_required": True,
            "recovery_uses_contract_names_at_any_precommit_ordinal": True,
        },
        "interpreter_provenance": {
            "capture_in_submission_contract": True,
            "queued_bootstrap_exact_reauthentication": True,
            "python_version": "3.11.15",
            "binary_pyvenv_and_path_identity_implemented": True,
            "environment_content_closure_implemented": False,
        },
        "engineering_pilot_prerequisites": {
            "launch7_terminal_negative_binding_state": (
                manifest["launch7_negative_dependency"]["binding_state"]
            ),
            "launch7_positive_authority_forbidden": True,
            "future_launch8_adapter_state": manifest["launch_dependency"]["adapter_state"],
            "future_launch8_positive_binding_state": manifest["launch_dependency"]["binding_state"],
            "positive_adapter_implemented": True,
            "frozen_source_commit": FROZEN_LAUNCH8_SOURCE_COMMIT,
            "frozen_protocol_sha256": FROZEN_LAUNCH8_PROTOCOL_SHA256,
            "frozen_source_inventory_sha256": (
                FROZEN_LAUNCH8_VERIFIER_SOURCE_INVENTORY_SHA256
            ),
        },
        "same_stage_requeue": {
            "structural_intent_record_only": True,
            "scheduler_mutation_implemented": False,
            "scientific_identity_sealed": False,
            "lost_response_reconciliation_implemented": False,
        },
        "readiness": execution_readiness(manifest),
    }
