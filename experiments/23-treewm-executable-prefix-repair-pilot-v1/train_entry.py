#!/usr/bin/env python3
"""Fail-closed, signal-safe entry point for one sealed Exp23 trainer launch.

The worker deliberately starts this wrapper rather than ``scripts/train.py``.  The
wrapper installs signal latches before importing torch/Hydra, proves that its argv,
environment, package and source snapshot are the sealed launch, registers Exp23's
mandatory post-update cadence state, and then invokes the unmodified trainer exactly
once.  It does not instrument or repeat the first forward pass.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import stat
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


OBJECTIVE = "treewm_v2_grounded_executable_prefix_pilot_v1"
PACKAGE_RELATIVE = Path("experiments/23-treewm-executable-prefix-repair-pilot-v1")
ENTRY_PATH = Path(os.path.abspath(__file__))
PACKAGE_DIR = ENTRY_PATH.parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]
STOP_ENVIRONMENT = "TREEWM_STOP_AFTER_UPDATE"
HEADLESS_RUNTIME_ENVIRONMENT = {
    "MUJOCO_GL": "egl",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}
READY_BYTE = b"R"
PINNED_PYTHON = Path(
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/envs/treewm-formal-py311/bin/python"
)
PINNED_SITE_DIRECTORIES = (
    Path(
        "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
        "users/chrislin/envs/treewm-formal-py311/lib/python3.11/site-packages"
    ),
    Path(
        "/lustre/fsw/portfolios/edgeai/users/chrislin/envs/maniskill-conda/"
        "lib/python3.11/site-packages"
    ),
)
SHA256_CHARS = frozenset("0123456789abcdef")
SNAPSHOT_IMPORT_FILES = {
    "configs/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "scripts/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}
TELEMETRY_CONTRACT_EVIDENCE = {
    "schema_version": 1,
    "status": "telemetry_contract_verified",
    "validation_namespace_sha256": (
        "8f58904f1d6ead6530902886b6ae24dc9529c58ec3562ee25c79c67369368883"
    ),
    "terminal_evaluation_namespace_sha256": (
        "40b28ebb3e286038da6815396452389d1d78a2ef0d42d44bc5cb149145ed2c54"
    ),
    "monitor_evaluation_namespace_sha256": (
        "0d7ec142f9f0715d9e8f3f2f028ab54f5f6d5b9b2066b46da73d3102c8e53e9b"
    ),
    "visualization_namespace_sha256": (
        "cab5e85de7fb86cdd42757529daf307584886ad0f8349b892b12fbbc58a247e2"
    ),
    "float32_identity_bits": "0x3f800000",
    "identical_duplicate_suppressed": True,
    "conflicting_duplicate_rejected": True,
    "batch_preflight_atomic": True,
    "out_of_order_step_rejected": True,
    "invalid_step_rejected": True,
    "backend_writes_performed": 0,
    "persistent_writes_performed": 0,
}
TRAINER_BOOTSTRAP_SMOKE_FIELDS = frozenset(
    {
        "schema_version", "status", "cell_index", "python_flags",
        "entry_relative_path", "config_package_relative_path",
        "config_package_sha256", "snapshot_inventory_sha256", "launch_sha256",
        "resolved_config_sha256", "stdout_sha256", "stdout_bytes",
        "cuda_visible_devices", "full_output_fingerprint_before",
        "full_output_fingerprint_after", "scientific_output_fingerprint_before",
        "scientific_output_fingerprint_after", "persistent_writes_performed",
        "scheduler_calls",
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
SUBMISSION_CONTRACT_FIELDS = frozenset(
    {
        "schema_version", "status", "campaign_id", "formal_validation",
        "submission_root", "snapshot_root", "package_protocol_sha256",
        "manifest_sha256", "trainer_code_fingerprint", "runtime_sha256",
        "orchestration_interpreter", "scheduler_control_plane_contract",
        "scheduler_preclaim", "scheduler_fallback_config",
        "weight_audit_artifact_sha256",
        "prefix_target_artifact_sha256", "resolved_config_artifact_sha256",
        "causal_parity_artifact_sha256", "snapshot_inventory",
        "snapshot_inventory_sha256", "live_audit_replays", "snapshot_audit_replays",
        "direct_hydra_compositions", "trainer_bootstrap_smoke",
        "scientific_output_fingerprint_before",
        "scientific_output_fingerprint_after", "full_output_fingerprint_before",
        "full_output_fingerprint_after", "snapshot_full_output_fingerprint_before",
        "snapshot_full_output_fingerprint_after",
        "snapshot_scientific_output_fingerprint_before",
        "snapshot_scientific_output_fingerprint_after", "git_provenance", "launches",
        "array", "fresh_start", "production_authorization_prerequisite",
    }
)
PRODUCTION_AUTHORIZATION_PREREQUISITE_FIELDS = frozenset(
    {
        "schema_version", "status", "attempt", "path", "raw_sha256",
        "canonical_sha256", "report_raw_sha256", "source_protocol_sha256",
        "source_commit", "state_root", "state_file_map_canonical_sha256",
        "canary_token", "job_ids_by_role", "accepted_attempt_sha256",
        "production_authorization_evidence_sha256",
        "sealed_package_protocol_sha256",
    }
)
INTERPRETER_CONTRACT_FIELDS = frozenset(
    {
        "lexical_executable", "lexical_symlink_target", "resolved_executable",
        "resolved_executable_sha256", "resolved_executable_size", "base_executable",
        "venv_site_packages", "base_site_packages", "python_version",
    }
)
SUBMISSION_LAUNCH_FIELDS = frozenset(
    {
        "index", "path", "launch_sha256", "launch_file_sha256", "setting_id",
        "arm_id", "seed", "weight_audit_artifact_sha256",
        "prefix_target_artifact_sha256", "resolved_config_artifact_sha256",
        "causal_parity_artifact_sha256",
    }
)

_EARLY_SIGNAL: int | None = None


class EntryContractError(RuntimeError):
    """The process is not the exact package-authorized invocation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EntryContractError(message)


def _early_handler(signum: int, _frame: object) -> None:
    """Latch only.  The trainer performs all I/O at its own safe boundary."""
    global _EARLY_SIGNAL
    # Cancellation has precedence while both signals are still in the import bridge.
    if _EARLY_SIGNAL is None or signum == signal.SIGTERM:
        _EARLY_SIGNAL = int(signum)


def install_early_signal_handlers() -> None:
    signal.signal(signal.SIGUSR1, _early_handler)
    signal.signal(signal.SIGTERM, _early_handler)
    # The worker starts us with these signals blocked.  That inherited mask closes
    # the otherwise fatal exec-to-handler window (including a direct Slurm TERM to
    # every process in the batch step).  Unblock only after both handlers exist.
    signal.pthread_sigmask(
        signal.SIG_UNBLOCK,
        {signal.SIGUSR1, signal.SIGTERM},
    )


def notify_worker_ready(fd: int) -> None:
    """Tell the worker that USR1/TERM can no longer take the default disposition."""
    _require(type(fd) is int and fd >= 3, "signal-ready fd is invalid")
    try:
        written = os.write(fd, READY_BYTE)
        _require(written == len(READY_BYTE), "short signal-ready write")
    except OSError as exc:
        raise EntryContractError(f"cannot publish signal readiness: {exc}") from exc
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


_DIRECTORY_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)


def _sha256_string(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA256_CHARS


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _exact_json_equal(left: object, right: object) -> bool:
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        return (
            isinstance(left, Mapping)
            and isinstance(right, Mapping)
            and set(left) == set(right)
            and all(_exact_json_equal(left[key], right[key]) for key in left)
        )
    if isinstance(left, list) or isinstance(right, list):
        return (
            isinstance(left, list)
            and isinstance(right, list)
            and len(left) == len(right)
            and all(_exact_json_equal(a, b) for a, b in zip(left, right))
        )
    return type(left) is type(right) and left == right


def _job_id(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value[0] in "123456789"
        and all(character in "0123456789" for character in value)
    )


def _validated_production_authorization_prerequisite(
    value: object,
    package_protocol_sha256: object,
    *,
    manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    _require(
        isinstance(value, Mapping)
        and set(value) == PRODUCTION_AUTHORIZATION_PREREQUISITE_FIELDS,
        "production authorization prerequisite fields differ",
    )
    result = dict(value)
    hash_fields = (
        "raw_sha256", "canonical_sha256", "report_raw_sha256",
        "source_protocol_sha256", "state_file_map_canonical_sha256",
        "accepted_attempt_sha256", "production_authorization_evidence_sha256",
        "sealed_package_protocol_sha256",
    )
    roles = result.get("job_ids_by_role")
    state_root = result.get("state_root")
    _require(
        type(result.get("schema_version")) is int
        and result.get("schema_version") == 1
        and result.get("status")
        == "canary2_production_authorization_prerequisite_satisfied"
        and result.get("attempt") == "canary2"
        and result.get("path") == "canary2_acceptance_provenance.json"
        and all(_sha256_string(result.get(field)) for field in hash_fields)
        and result.get("sealed_package_protocol_sha256")
        == package_protocol_sha256
        and isinstance(result.get("source_commit"), str)
        and len(result["source_commit"]) == 40
        and set(result["source_commit"]) <= SHA256_CHARS
        and isinstance(state_root, str)
        and Path(state_root).is_absolute()
        and os.path.normpath(state_root) == state_root
        and not state_root.startswith("//")
        and isinstance(result.get("canary_token"), str)
        and len(result["canary_token"]) == 16
        and set(result["canary_token"]) <= SHA256_CHARS
        and isinstance(roles, Mapping)
        and set(roles) == {"wave0", "wave1", "report"}
        and all(
            isinstance(roles[role], list)
            and len(roles[role]) == 1
            and _job_id(roles[role][0])
            for role in ("wave0", "wave1", "report")
        )
        and len({roles[role][0] for role in ("wave0", "wave1", "report")})
        == 3,
        "production authorization prerequisite differs",
    )
    if manifest is not None:
        launch = manifest.get("launch_contract")
        canary = (
            launch.get("real_gpu_two_wave_canary")
            if isinstance(launch, Mapping)
            else None
        )
        accepted = (
            canary.get("accepted_attempts")
            if isinstance(canary, Mapping)
            else None
        )
        evidence = (
            canary.get("production_authorization_evidence")
            if isinstance(canary, Mapping)
            else None
        )
        _require(
            isinstance(accepted, list)
            and len(accepted) == 1
            and isinstance(accepted[0], Mapping)
            and isinstance(evidence, Mapping),
            "snapshot production authorization evidence differs",
        )
        attempt = accepted[0]
        expected = {
            "schema_version": 1,
            "status": "canary2_production_authorization_prerequisite_satisfied",
            "attempt": "canary2",
            "path": "canary2_acceptance_provenance.json",
            "raw_sha256": evidence.get("raw_sha256"),
            "canonical_sha256": evidence.get("canonical_sha256"),
            "report_raw_sha256": evidence.get("report_raw_sha256"),
            "source_protocol_sha256": evidence.get("source_protocol_sha256"),
            "source_commit": attempt.get("source_commit"),
            "state_root": attempt.get("state_root"),
            "state_file_map_canonical_sha256": attempt.get(
                "state_file_map_canonical_sha256"
            ),
            "canary_token": attempt.get("canary_token"),
            "job_ids_by_role": attempt.get("job_ids_by_role"),
            "accepted_attempt_sha256": _stable_hash(attempt),
            "production_authorization_evidence_sha256": _stable_hash(evidence),
            "sealed_package_protocol_sha256": package_protocol_sha256,
        }
        _require(
            _exact_json_equal(result, expected),
            "submission/snapshot production authorization prerequisite differs",
        )
    return result


def _absolute(path: str | Path, label: str) -> Path:
    value = Path(path)
    _require(value.is_absolute(), f"{label} is not absolute")
    _require(all(part not in {"", ".", ".."} for part in value.parts[1:]), f"{label} is not normalized")
    return value


def _safe_relative(path: str | Path, label: str) -> Path:
    value = Path(path)
    _require(
        not value.is_absolute() and bool(value.parts)
        and all(part not in {"", ".", ".."} for part in value.parts),
        f"{label} is not a safe relative path",
    )
    return value


def _open_directory(path: str | Path, label: str) -> int:
    absolute = _absolute(path, label)
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        for part in absolute.parts[1:]:
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except OSError as exc:
                raise EntryContractError(f"{label} has a symlink/non-directory component: {exc}") from exc
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _open_relative_directory(root_fd: int, relative: Path, label: str) -> int:
    descriptor = os.dup(root_fd)
    try:
        for part in relative.parts:
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except OSError as exc:
                raise EntryContractError(f"{label} has a symlink/non-directory component: {exc}") from exc
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _open_regular(path: str | Path, label: str) -> tuple[int, os.stat_result]:
    absolute = _absolute(path, label)
    parent_fd = _open_directory(absolute.parent, f"parent of {label}")
    try:
        try:
            descriptor = os.open(absolute.name, _FILE_FLAGS, dir_fd=parent_fd)
        except OSError as exc:
            raise EntryContractError(f"{label} is unavailable or symlinked: {exc}") from exc
    finally:
        os.close(parent_fd)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise EntryContractError(f"{label} is not a regular nonsymlink file")
    return descriptor, info


def _identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_stable(descriptor: int, label: str) -> tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    _require(stat.S_ISREG(before.st_mode), f"{label} is not regular")
    offset = 0
    chunks: list[bytes] = []
    while block := os.pread(descriptor, 16 * 1024 * 1024, offset):
        chunks.append(block)
        offset += len(block)
    after = os.fstat(descriptor)
    _require(_identity(before) == _identity(after), f"{label} changed while open")
    payload = b"".join(chunks)
    _require(len(payload) == before.st_size, f"{label} short read")
    return payload, before


def _decode_json(payload: bytes, path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise EntryContractError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                EntryContractError(f"non-finite JSON value in {path}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise EntryContractError(f"cannot read JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _read_json_artifact(path: Path, label: str, *, mode: int | None = None) -> tuple[dict[str, Any], str]:
    descriptor, opened = _open_regular(path, label)
    try:
        payload, stable = _read_stable(descriptor, label)
    finally:
        os.close(descriptor)
    _require(_identity(opened) == _identity(stable), f"{label} path/open identity differs")
    if mode is not None:
        _require(stat.S_IMODE(stable.st_mode) == mode, f"{label} mode differs")
    return _decode_json(payload, path), hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return _read_json_artifact(path, f"JSON artifact {path}")[0]


def _regular_nonsymlink(path: Path, label: str) -> None:
    descriptor, _ = _open_regular(path, label)
    os.close(descriptor)


def _directory_nonsymlink(path: Path, label: str) -> None:
    descriptor = _open_directory(path, label)
    os.close(descriptor)


def _contained(path: Path, root: Path, label: str) -> Path:
    candidate = _absolute(path, label)
    expected_root = _absolute(root, f"{label} root")
    try:
        relative = candidate.relative_to(expected_root)
    except ValueError as exc:
        raise EntryContractError(f"{label} escapes its declared root") from exc
    root_fd = _open_directory(expected_root, f"{label} root")
    try:
        parent_fd = _open_relative_directory(root_fd, relative.parent, f"parent of {label}")
        os.close(parent_fd)
    finally:
        os.close(root_fd)
    return candidate


def assert_isolated_runtime() -> None:
    _require(sys.flags.safe_path == 1, "train entry requires Python -P safe-path mode")
    _require(sys.flags.isolated == 0, "train entry forbids -I because it ignores PYTHONHASHSEED")
    _require(sys.flags.no_site == 1, "train entry requires Python -S")
    _require(bool(sys.dont_write_bytecode), "train entry requires Python -B")
    _require("" not in sys.path, "current directory is importable before bootstrap")
    _require(not any("site-packages" in item for item in sys.path), "site-packages active before bootstrap")
    _require("sitecustomize" not in sys.modules, "sitecustomize loaded before bootstrap")
    _require("usercustomize" not in sys.modules, "usercustomize loaded before bootstrap")
    _require("treewm" not in sys.modules and "torch" not in sys.modules, "scientific modules loaded before bootstrap")
    _require("PYTHONPATH" not in os.environ, "PYTHONPATH is forbidden in trainer bootstrap")
    hash_seed = os.environ.get("PYTHONHASHSEED", "")
    _require(hash_seed.isascii() and hash_seed.isdigit(), "PYTHONHASHSEED is absent or malformed")


def _verify_snapshot_tree(snapshot_root: Path, inventory: Mapping[str, Any]) -> None:
    expected: dict[str, str] = {}
    expected_directories: set[str] = set()
    for raw_relative, raw_digest in inventory.items():
        _require(isinstance(raw_relative, str), "snapshot inventory path is not text")
        relative = _safe_relative(raw_relative, "snapshot inventory path")
        _require(_sha256_string(raw_digest), f"snapshot inventory digest is malformed: {relative}")
        rendered = str(relative)
        _require(rendered not in expected, f"duplicate snapshot inventory path: {rendered}")
        expected[rendered] = str(raw_digest)
        for parent in relative.parents:
            if parent != Path("."):
                expected_directories.add(str(parent))
    for required in (
        str(PACKAGE_RELATIVE / "worker.py"),
        str(PACKAGE_RELATIVE / "train_entry.py"),
        str(PACKAGE_RELATIVE / "train.slurm"),
        "scripts/train.py",
    ):
        _require(required in expected, f"snapshot inventory omits lifecycle source: {required}")
    for required, digest in SNAPSHOT_IMPORT_FILES.items():
        _require(
            expected.get(required) == digest,
            f"snapshot inventory omits/replaces exact import marker: {required}",
        )

    root_fd = _open_directory(snapshot_root, "snapshot root")
    actual: dict[str, str] = {}
    actual_directories: set[str] = set()

    def walk(directory_fd: int, prefix: Path) -> None:
        before = os.fstat(directory_fd)
        _require(stat.S_IMODE(before.st_mode) == 0o555, f"snapshot directory mode differs: {prefix}")
        for name in sorted(os.listdir(directory_fd)):
            _require(name not in {"", ".", ".."} and "/" not in name, "invalid snapshot entry")
            relative = prefix / name if prefix != Path(".") else Path(name)
            rendered = str(relative)
            entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISDIR(entry.st_mode):
                _require(stat.S_IMODE(entry.st_mode) == 0o555, f"snapshot directory mode differs: {rendered}")
                _require(rendered in expected_directories, f"snapshot has an extra directory: {rendered}")
                child = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
                try:
                    _require(_identity(entry) == _identity(os.fstat(child)), f"snapshot directory swapped: {rendered}")
                    actual_directories.add(rendered)
                    walk(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(entry.st_mode):
                _require(stat.S_IMODE(entry.st_mode) == 0o444, f"snapshot file mode differs: {rendered}")
                _require(rendered in expected, f"snapshot has an unclaimed file: {rendered}")
                descriptor = os.open(name, _FILE_FLAGS, dir_fd=directory_fd)
                try:
                    payload, opened = _read_stable(descriptor, f"snapshot file {rendered}")
                finally:
                    os.close(descriptor)
                _require(_identity(entry) == _identity(opened), f"snapshot file swapped: {rendered}")
                digest = hashlib.sha256(payload).hexdigest()
                _require(digest == expected[rendered], f"snapshot bytes differ: {rendered}")
                actual[rendered] = digest
            else:
                raise EntryContractError(f"snapshot contains symlink/special file: {rendered}")
        _require(
            _identity(os.fstat(directory_fd)) == _identity(before),
            f"snapshot directory changed while enumerating: {prefix}",
        )

    try:
        walk(root_fd, Path("."))
    finally:
        os.close(root_fd)
    _require(actual == expected, "snapshot file coverage differs from inventory")
    _require(actual_directories == expected_directories, "snapshot directory coverage differs from inventory")


def _verify_snapshot_location(snapshot_root: Path, submission_root: Path) -> None:
    """Require the exact owned launch namespace before executable imports.

    Same-UID malicious processes are trusted here (they could ptrace or alter this
    process directly).  Exact permissions and immediate revalidation catch
    accidental/concurrent drift while the worker has launched no ambient writer.
    """

    _require(
        snapshot_root == submission_root / "source-snapshot" / "repo",
        "snapshot root is outside the exact source-snapshot namespace",
    )
    for path, mode, label in (
        (submission_root, 0o700, "submission root"),
        (submission_root / "source-snapshot", 0o555, "source-snapshot parent"),
        (snapshot_root, 0o555, "snapshot root"),
    ):
        descriptor = _open_directory(path, label)
        try:
            info = os.fstat(descriptor)
            _require(
                info.st_uid == os.getuid()
                and info.st_gid == os.getgid()
                and stat.S_IMODE(info.st_mode) == mode,
                f"{label} ownership/mode differs",
            )
        finally:
            os.close(descriptor)


def _revalidate_snapshot_before_import(
    snapshot_root: Path,
    submission_root: Path,
    contract: Mapping[str, Any],
) -> None:
    inventory = contract.get("snapshot_inventory")
    _require(isinstance(inventory, Mapping) and bool(inventory), "snapshot inventory is absent")
    _verify_snapshot_location(snapshot_root, submission_root)
    _verify_snapshot_tree(snapshot_root, inventory)


def _verify_interpreter_contract(value: object) -> None:
    _require(isinstance(value, Mapping), "submission interpreter contract is absent")
    _require(set(value) == INTERPRETER_CONTRACT_FIELDS, "submission interpreter fields differ")
    _require(
        value.get("lexical_executable") == str(PINNED_PYTHON)
        and os.path.normpath(os.path.abspath(sys.executable)) == str(PINNED_PYTHON),
        "train-entry interpreter binding differs",
    )
    lexical_info = PINNED_PYTHON.lstat()
    _require(stat.S_ISLNK(lexical_info.st_mode), "pinned lexical interpreter is not the sealed venv symlink")
    lexical_target = os.readlink(PINNED_PYTHON) if stat.S_ISLNK(lexical_info.st_mode) else None
    _require(value.get("lexical_symlink_target") == lexical_target, "interpreter symlink binding differs")
    resolved = PINNED_PYTHON.resolve(strict=True)
    _require(value.get("resolved_executable") == str(resolved), "resolved interpreter path differs")
    descriptor, info = _open_regular(resolved, "resolved pinned interpreter")
    try:
        payload, stable = _read_stable(descriptor, "resolved pinned interpreter")
    finally:
        os.close(descriptor)
    _require(_identity(info) == _identity(stable), "resolved interpreter changed")
    _require(
        value.get("resolved_executable_sha256") == hashlib.sha256(payload).hexdigest()
        and value.get("resolved_executable_size") == info.st_size,
        "resolved interpreter identity differs",
    )
    _require(value.get("base_executable") == str(getattr(sys, "_base_executable", "")), "base interpreter differs")
    _require(
        value.get("venv_site_packages") == str(PINNED_SITE_DIRECTORIES[0])
        and value.get("base_site_packages") == str(PINNED_SITE_DIRECTORIES[1]),
        "site-package binding differs",
    )
    _require(
        value.get("python_version") == f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "Python version binding differs",
    )


def configure_verified_import_paths(snapshot_root: Path) -> None:
    suffix = [str(snapshot_root), *(str(path) for path in PINNED_SITE_DIRECTORIES)]
    _require(not any(item in sys.path for item in suffix), "verified import root was pre-injected")
    for directory in PINNED_SITE_DIRECTORIES:
        info = directory.lstat()
        _require(stat.S_ISDIR(info.st_mode), f"pinned site path is not a literal directory: {directory}")
    sys.path.extend(suffix)
    _require(sys.path[-len(suffix):] == suffix, "verified import path suffix differs")


def bootstrap_submission(
    submission_root: Path,
    submission_sha256: str,
) -> tuple[Path, dict[str, Any]]:
    _require(_sha256_string(submission_sha256), "submission SHA256 is malformed")
    submission = _absolute(submission_root, "submission root")
    submission_fd = _open_directory(submission, "submission root")
    os.close(submission_fd)
    contract_path = submission / "SUBMISSION_CONTRACT.json"
    contract, digest = _read_json_artifact(contract_path, "submission contract", mode=0o444)
    _require(digest == submission_sha256, "submission contract bytes differ")
    _require(set(contract) == SUBMISSION_CONTRACT_FIELDS, "submission contract fields differ")
    _require(contract.get("schema_version") == 1, "submission contract schema differs")
    _require(contract.get("status") == "sealed_for_submission", "submission is not sealed")
    _require(contract.get("campaign_id") == "treewm-executable-prefix-repair-pilot-v1-launch8", "campaign differs")
    _require(contract.get("formal_validation") is False, "formal-validation label differs")
    _require(contract.get("array") == "0-19%20" and contract.get("fresh_start") is True, "submission lifecycle differs")
    _validated_production_authorization_prerequisite(
        contract.get("production_authorization_prerequisite"),
        contract.get("package_protocol_sha256"),
    )
    _require(
        contract.get("scheduler_control_plane_contract") == SCHEDULER_CONTROL_PLANE,
        "scheduler control-plane contract differs",
    )
    scheduler_preclaim = contract.get("scheduler_preclaim")
    _require(
        isinstance(scheduler_preclaim, Mapping)
        and scheduler_preclaim.get("schema_version") == 1
        and scheduler_preclaim.get("status") == "scheduler_preclaim_verified"
        and scheduler_preclaim.get("campaign_id")
        == "treewm-executable-prefix-repair-pilot-v1-launch8"
        and scheduler_preclaim.get("scheduler_calls") == 10
        and scheduler_preclaim.get("scheduler_mutation_calls") == 0
        and scheduler_preclaim.get("persistent_writes_performed") == 0,
        "scheduler preclaim differs",
    )
    _require(
        isinstance(scheduler_preclaim.get("scheduler_probe_commands"), list)
        and len(scheduler_preclaim["scheduler_probe_commands"]) == 10
        and scheduler_preclaim.get("dependency_tests")
        == DEPENDENCY_TEST_REQUIREMENT
        and scheduler_preclaim.get("zero_job_proof")
        == {
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
        "scheduler preclaim call ledger differs",
    )
    scheduler_fallback = contract.get("scheduler_fallback_config")
    _require(
        isinstance(scheduler_fallback, Mapping)
        and scheduler_fallback.get("schema_version") == 1
        and scheduler_fallback.get("purpose")
        == (
            "accepted-job exact reconciliation, cancellation, dependency verification, "
            "and wave-zero release only; never submission or compute-side execution"
        )
        and scheduler_fallback.get("encoding") == "base64"
        and isinstance(scheduler_fallback.get("payload_base64"), str)
        and _sha256_string(str(scheduler_fallback.get("sha256", "")))
        and isinstance(scheduler_fallback.get("size"), int)
        and 0 < scheduler_fallback["size"] <= 16 * 1024 * 1024,
        "scheduler fallback metadata differs",
    )
    try:
        scheduler_fallback_bytes = base64.b64decode(
            scheduler_fallback["payload_base64"], validate=True
        )
    except (ValueError, UnicodeError) as exc:
        raise EntryContractError(f"scheduler fallback encoding differs: {exc}") from exc
    _require(
        len(scheduler_fallback_bytes) == scheduler_fallback["size"]
        and hashlib.sha256(scheduler_fallback_bytes).hexdigest()
        == scheduler_fallback["sha256"]
        and scheduler_fallback.get("source_control_plane")
        == scheduler_preclaim.get("scheduler_control_plane"),
        "scheduler fallback bytes differ",
    )
    _require(contract.get("submission_root") == str(submission), "submission root binding differs")
    snapshot = _absolute(str(contract.get("snapshot_root", "")), "snapshot root")
    _require(snapshot == REPOSITORY_ROOT, "entry snapshot root binding differs")
    try:
        snapshot.relative_to(submission)
    except ValueError as exc:
        raise EntryContractError("snapshot root escapes submission root") from exc
    _verify_interpreter_contract(contract.get("orchestration_interpreter"))
    for prefix in ("", "snapshot_"):
        for flavor in ("full_output", "scientific_output"):
            _require(
                contract.get(f"{prefix}{flavor}_fingerprint_before")
                == contract.get(f"{prefix}{flavor}_fingerprint_after"),
                f"submission {prefix}{flavor} fingerprint drifted",
            )
    inventory = contract.get("snapshot_inventory")
    _require(isinstance(inventory, Mapping) and bool(inventory), "snapshot inventory is absent")
    _require(_stable_hash(inventory) == contract.get("snapshot_inventory_sha256"), "snapshot inventory hash differs")
    smoke = contract.get("trainer_bootstrap_smoke")
    compositions = contract.get("direct_hydra_compositions")
    launches = contract.get("launches")
    _require(
        isinstance(smoke, Mapping) and set(smoke) == TRAINER_BOOTSTRAP_SMOKE_FIELDS,
        "trainer bootstrap smoke fields differ",
    )
    _require(
        type(smoke.get("schema_version")) is int
        and smoke.get("schema_version") == 1
        and smoke.get("status") == "sealed_trainer_hydra_composition_verified"
        and type(smoke.get("cell_index")) is int
        and smoke.get("cell_index") == 0
        and smoke.get("python_flags") == ["-P", "-S", "-B"]
        and smoke.get("entry_relative_path") == str(PACKAGE_RELATIVE / "train_entry.py")
        and smoke.get("config_package_relative_path") == "configs/__init__.py"
        and smoke.get("config_package_sha256") == SNAPSHOT_IMPORT_FILES["configs/__init__.py"]
        and smoke.get("snapshot_inventory_sha256") == contract.get("snapshot_inventory_sha256")
        and smoke.get("cuda_visible_devices") == ""
        and type(smoke.get("persistent_writes_performed")) is int
        and smoke.get("persistent_writes_performed") == 0
        and type(smoke.get("scheduler_calls")) is int
        and smoke.get("scheduler_calls") == 0,
        "trainer bootstrap smoke contract differs",
    )
    _require(
        isinstance(compositions, list) and len(compositions) == 20
        and isinstance(launches, list) and len(launches) == 20
        and smoke.get("launch_sha256") == launches[0].get("launch_sha256")
        and smoke.get("resolved_config_sha256")
        == compositions[0].get("resolved_config_sha256"),
        "trainer bootstrap smoke launch/config binding differs",
    )
    _require(
        _sha256_string(smoke.get("stdout_sha256"))
        and type(smoke.get("stdout_bytes")) is int
        and smoke["stdout_bytes"] > 0,
        "trainer bootstrap smoke stdout evidence differs",
    )
    for flavor in ("full_output", "scientific_output"):
        _require(
            _sha256_string(smoke.get(f"{flavor}_fingerprint_before"))
            and _sha256_string(smoke.get(f"{flavor}_fingerprint_after"))
            and smoke.get(f"{flavor}_fingerprint_before")
            == smoke.get(f"{flavor}_fingerprint_after"),
            f"trainer bootstrap smoke {flavor} drifted",
        )
        _require(
            smoke.get(f"{flavor}_fingerprint_before")
            == contract.get(f"snapshot_{flavor}_fingerprint_before")
            == contract.get(f"snapshot_{flavor}_fingerprint_after"),
            f"trainer bootstrap smoke {flavor} evidence is detached",
        )
    _verify_snapshot_location(snapshot, submission)
    _verify_snapshot_tree(snapshot, inventory)
    configure_verified_import_paths(snapshot)
    return submission, contract


def _load_campaign() -> ModuleType:
    path = PACKAGE_DIR / "campaign.py"
    _regular_nonsymlink(path, "campaign verifier")
    name = "_treewm_exp23_campaign_for_train_entry"
    spec = importlib.util.spec_from_file_location(name, path)
    _require(spec is not None and spec.loader is not None, "cannot load campaign verifier")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def reject_forbidden_environment(
    launch_environment: Mapping[str, Any],
    environ: Mapping[str, str] | None = None,
) -> None:
    environment = os.environ if environ is None else environ
    _require(STOP_ENVIRONMENT not in environment, f"{STOP_ENVIRONMENT} is forbidden")
    _require("PYTHONPATH" not in environment, "PYTHONPATH is forbidden after isolated bootstrap")
    allowed_treewm = {str(key) for key in launch_environment if str(key).startswith("TREEWM_")}
    unexpected = sorted(
        key
        for key in environment
        if key.startswith("TREEWM_") and key not in allowed_treewm
    )
    _require(not unexpected, "unexpected TREEWM environment: " + ", ".join(unexpected))
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        _require(name not in environment, f"unexpected distributed environment: {name}")


def validate_headless_runtime_environment(
    launch_environment: Mapping[str, Any],
    environment: Mapping[str, str],
) -> None:
    """Require the sealed launch and actual child to share exact headless GPU knobs."""
    for name, value in HEADLESS_RUNTIME_ENVIRONMENT.items():
        _require(
            launch_environment.get(name) == value,
            f"sealed headless runtime binding differs: {name}",
        )
        _require(
            environment.get(name) == value,
            f"trainer headless runtime binding differs: {name}",
        )


def verify_exact_invocation(
    launch_path: Path,
    trainer_args: Sequence[str],
    *,
    submission_root: Path,
    bootstrap_contract: Mapping[str, Any] | None = None,
    campaign: ModuleType | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[ModuleType, dict[str, Any], dict[str, Any]]:
    """Re-derive and compare the complete launch before importing the trainer."""
    environment = os.environ if environ is None else environ
    snapshot_root = REPOSITORY_ROOT
    package = _contained(PACKAGE_DIR, snapshot_root, "package")
    _require(package == snapshot_root / PACKAGE_RELATIVE, "package root differs")
    _directory_nonsymlink(submission_root, "submission root")
    submission_root = _absolute(submission_root, "submission root")
    launch_path = _contained(launch_path, submission_root, "launch path")
    _require(
        launch_path.parent == submission_root / "launches",
        "launch path is outside the sealed launches directory",
    )
    launch, launch_file_sha256 = _read_json_artifact(
        launch_path,
        "launch path",
        mode=0o444,
    )

    if campaign is None:
        # This is the final filesystem boundary before importing executable bytes.
        # No snapshot writer is launched by the worker/entry bootstrap.
        _revalidate_snapshot_before_import(
            snapshot_root, submission_root, bootstrap_contract or {}
        )
        campaign = _load_campaign()
    manifest, weight_lock = campaign.load_contract(snapshot_root)
    protocol = campaign.verify_protocol_lock(PACKAGE_DIR)
    _validated_production_authorization_prerequisite(
        (bootstrap_contract or {}).get("production_authorization_prerequisite"),
        protocol,
        manifest=manifest,
    )
    cell_value = launch.get("cell")
    _require(isinstance(cell_value, Mapping), "launch cell is absent")
    index = cell_value.get("index")
    _require(type(index) is int, "launch cell index is invalid")
    _require(0 <= index < 20, "launch cell index is out of range")
    _require(launch_path == submission_root / "launches" / f"cell-{index:02d}.json", "launch filename/index differs")
    if bootstrap_contract is not None:
        launches = bootstrap_contract.get("launches")
        _require(isinstance(launches, list) and len(launches) == 20, "submission launch inventory differs")
        row = launches[index]
        _require(isinstance(row, Mapping) and set(row) == SUBMISSION_LAUNCH_FIELDS, "submission launch row fields differ")
        _require(
            row.get("index") == index
            and row.get("path") == f"launches/cell-{index:02d}.json"
            and row.get("launch_sha256") == launch.get("launch_sha256")
            and row.get("launch_file_sha256") == launch_file_sha256,
            "submission launch binding differs",
        )
        _require(
            row.get("setting_id") == cell_value.get("setting")
            and row.get("arm_id") == cell_value.get("arm")
            and row.get("seed") == cell_value.get("seed"),
            "submission launch cell identity differs",
        )
        for audit_name in (
            "weight_audit_artifact_sha256",
            "prefix_target_artifact_sha256",
            "resolved_config_artifact_sha256",
            "causal_parity_artifact_sha256",
        ):
            _require(
                row.get(audit_name) == bootstrap_contract.get(audit_name),
                f"submission launch {audit_name} differs",
            )
    cells = campaign.expand_matrix(manifest)
    _require(0 <= index < len(cells), "launch cell index is out of range")
    expected = campaign.trainer_command(
        manifest,
        weight_lock,
        cells[index],
        repo_root=snapshot_root,
        package_protocol_sha256=protocol,
    )
    _require(launch == expected, "launch JSON differs from snapshot re-derivation")
    launch_body = dict(launch)
    claimed_launch_sha256 = launch_body.pop("launch_sha256", None)
    _require(claimed_launch_sha256 == _stable_hash(launch_body), "launch self hash differs")
    if bootstrap_contract is not None:
        _require(
            bootstrap_contract.get("package_protocol_sha256") == protocol
            and bootstrap_contract.get("manifest_sha256") == campaign.manifest_sha256(manifest),
            "submission protocol/manifest binding differs",
        )
    expected_audits = {
        "weight_audit_artifact_sha256": manifest["weight_audit"]["artifact_sha256"],
        "prefix_target_artifact_sha256": manifest["prefix_target_contract"]["artifact_sha256"],
        "resolved_config_artifact_sha256": manifest["resolved_config_contract"]["artifact_sha256"],
        "causal_parity_artifact_sha256": manifest["causal_parity_contract"]["artifact_sha256"],
    }
    hashes = launch.get("hashes")
    _require(isinstance(hashes, Mapping), "launch hashes are invalid")
    for name, value in expected_audits.items():
        _require(hashes.get(name) == value, f"launch {name} differs")
        if bootstrap_contract is not None:
            _require(bootstrap_contract.get(name) == value, f"submission {name} differs")
    if bootstrap_contract is not None:
        _require(
            bootstrap_contract.get("trainer_code_fingerprint") == hashes.get("source_sha256")
            and bootstrap_contract.get("runtime_sha256") == hashes.get("runtime_sha256"),
            "submission source/runtime binding differs",
        )
    argv = launch.get("argv")
    _require(isinstance(argv, list) and len(argv) >= 3, "trainer argv is invalid")
    _require(argv[0] == manifest["paths"]["python"], "trainer interpreter differs")
    _require(argv[0] == str(PINNED_PYTHON), "trainer does not use pinned interpreter")
    _regular_nonsymlink(Path(str(argv[1])), "direct trainer entrypoint")
    _require(
        Path(str(argv[1])) == snapshot_root / "scripts/train.py",
        "direct trainer path differs",
    )
    _require(list(trainer_args) == argv[2:], "trainer arguments differ from sealed launch")
    _require("resume=auto" in trainer_args, "sealed trainer invocation lacks resume=auto")
    _require("train.steps=25000" in trainer_args, "sealed trainer invocation is not 25k")
    _require(
        not any(str(arg).startswith("TREEWM_STOP_AFTER_UPDATE") for arg in trainer_args),
        "staged stop leaked into trainer arguments",
    )
    launch_environment = launch.get("environment")
    _require(isinstance(launch_environment, Mapping), "launch environment is invalid")
    validate_headless_runtime_environment(launch_environment, environment)
    _require(
        environment.get("PYTHONHASHSEED") == str(cell_value.get("seed")),
        "trainer Python hash seed differs from sealed cell seed",
    )
    _require(
        launch_environment.get("TREEWM_RESOLVED_CONFIG_SHA256")
        == expected_audits["resolved_config_artifact_sha256"],
        "resolved-config audit environment binding differs",
    )
    _require(
        launch_environment.get("TREEWM_CAUSAL_PARITY_SHA256")
        == expected_audits["causal_parity_artifact_sha256"],
        "causal-parity audit environment binding differs",
    )
    reject_forbidden_environment(launch_environment, environment)
    for key, value in launch_environment.items():
        _require(
            environment.get(str(key)) == str(value),
            f"trainer environment differs: {key}",
        )
    return campaign, manifest, launch


def _register_and_import_trainer() -> ModuleType:
    """Register strict cadence and bridge any signal latched during imports."""
    _require("scripts.train" not in sys.modules, "trainer was imported before signal bridge")
    from treewm.utils import checkpoint as checkpoint_utils

    checkpoint_utils.OBJECTIVES_REQUIRING_POST_UPDATE_CADENCE = frozenset(
        {*checkpoint_utils.OBJECTIVES_REQUIRING_POST_UPDATE_CADENCE, OBJECTIVE}
    )
    original = checkpoint_utils.StopController

    class BridgedStopController(original):  # type: ignore[misc, valid-type]
        def install(self) -> None:
            super().install()
            pending = _EARLY_SIGNAL
            if pending is not None:
                self.request(signal.Signals(pending).name)

    checkpoint_utils.StopController = BridgedStopController
    from scripts import train

    _require(
        OBJECTIVE in train.TREEWM_V2_OBJECTIVES
        and train.BOUNDED_PILOT_OBJECTIVES.get(OBJECTIVE) == 25_000
        and OBJECTIVE in train.LATENT_GAUGE_OBJECTIVES,
        "shared trainer does not contain the exact bounded Exp23 objective",
    )
    return train


def _decode_smoke_object(raw: str, label: str) -> dict[str, Any]:
    try:
        payload = raw.encode("ascii")
    except UnicodeEncodeError as exc:
        raise EntryContractError(f"{label} is not canonical ASCII JSON") from exc
    value = _decode_json(payload, Path(f"<{label}>"))
    _require(
        _canonical_json(value) == raw,
        f"{label} is not canonical JSON",
    )
    return value


def _verify_imported_module(module_name: str, expected_path: Path) -> None:
    module = sys.modules.get(module_name)
    _require(module is not None, f"{module_name} was not imported")
    path = Path(os.path.abspath(str(getattr(module, "__file__", ""))))
    _require(path == expected_path, f"{module_name} imported from an unexpected path")
    _regular_nonsymlink(path, f"{module_name} imported module")


def hydra_composition_smoke(argv: Sequence[str] | None = None) -> int:
    """Compose one exact sealed launch through the production module-import bridge."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--_hydra-composition-smoke", action="store_true", required=True)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--snapshot-inventory-sha256", required=True)
    parser.add_argument("--snapshot-inventory-json", required=True)
    parser.add_argument("--launch-json", required=True)
    args = parser.parse_args(argv)

    assert_isolated_runtime()
    _require(_EARLY_SIGNAL is None, "signal state exists before composition smoke")
    _require(
        os.environ.get("CUDA_VISIBLE_DEVICES") == "",
        "composition smoke must hide every CUDA device",
    )
    snapshot = _absolute(args.snapshot_root, "smoke snapshot root")
    _require(snapshot == REPOSITORY_ROOT, "smoke snapshot root differs from entry root")
    _require(
        ENTRY_PATH == snapshot / PACKAGE_RELATIVE / "train_entry.py",
        "composition smoke entry path differs",
    )
    inventory = _decode_smoke_object(
        args.snapshot_inventory_json, "snapshot inventory JSON"
    )
    _require(
        _sha256_string(args.snapshot_inventory_sha256)
        and _stable_hash(inventory) == args.snapshot_inventory_sha256,
        "composition smoke snapshot inventory hash differs",
    )
    launch = _decode_smoke_object(args.launch_json, "launch JSON")
    launch_body = dict(launch)
    launch_sha256 = launch_body.pop("launch_sha256", None)
    _require(
        _sha256_string(launch_sha256)
        and launch_sha256 == _stable_hash(launch_body),
        "composition smoke launch self hash differs",
    )
    _require(
        launch.get("schema_version") == 1
        and launch.get("campaign_id") == "treewm-executable-prefix-repair-pilot-v1-launch8",
        "composition smoke launch identity differs",
    )
    cell = launch.get("cell")
    _require(isinstance(cell, Mapping), "composition smoke cell is absent")
    seed = cell.get("seed")
    _require(type(seed) is int, "composition smoke seed is invalid")
    _require(
        os.environ.get("PYTHONHASHSEED") == str(seed),
        "composition smoke Python hash seed differs",
    )
    launch_environment = launch.get("environment")
    _require(isinstance(launch_environment, Mapping), "composition smoke environment is absent")
    validate_headless_runtime_environment(launch_environment, os.environ)
    reject_forbidden_environment(launch_environment, os.environ)
    for key, value in launch_environment.items():
        _require(
            os.environ.get(str(key)) == str(value),
            f"composition smoke environment differs: {key}",
        )
    launch_argv = launch.get("argv")
    _require(
        isinstance(launch_argv, list) and len(launch_argv) >= 3,
        "composition smoke trainer argv is invalid",
    )
    _require(
        launch_argv[0] == str(PINNED_PYTHON)
        and Path(str(launch_argv[1])) == snapshot / "scripts/train.py",
        "composition smoke trainer executable differs",
    )
    trainer_args = [str(value) for value in launch_argv[2:]]
    _require(
        "resume=auto" in trainer_args and "train.steps=25000" in trainer_args,
        "composition smoke trainer lifecycle differs",
    )
    _require(
        not any(
            value == "--resolve"
            or value.startswith("--resolve=")
            or value == "--cfg"
            or value.startswith("--cfg=")
            for value in trainer_args
        ),
        "composition smoke launch contains unsealed Hydra control flags",
    )

    _verify_snapshot_tree(snapshot, inventory)
    configure_verified_import_paths(snapshot)
    train = _register_and_import_trainer()
    for module_name, expected_path in (
        ("treewm", snapshot / "treewm/__init__.py"),
        ("treewm.utils.checkpoint", snapshot / "treewm/utils/checkpoint.py"),
        ("scripts.train", snapshot / "scripts/train.py"),
    ):
        _verify_imported_module(module_name, expected_path)
    _require(
        train.telemetry_contract_self_test() == TELEMETRY_CONTRACT_EVIDENCE,
        "trainer telemetry contract self-test differs",
    )

    # ``--cfg job --resolve`` returns before the decorated application function is
    # entered, so this exercises Hydra's real module-relative package lookup without
    # constructing a model, opening a run directory, or contacting W&B.
    sys.argv = [str(launch_argv[1]), *trainer_args, "--cfg", "job", "--resolve"]
    train.main()
    _verify_imported_module("configs", snapshot / "configs/__init__.py")
    return 0


def _parse(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--launch", type=Path, required=True)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--submission-sha256", required=True)
    parser.add_argument("--ready-fd", type=int, required=True)
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.trainer_args and args.trainer_args[0] == "--":
        args.trainer_args = args.trainer_args[1:]
    _require(bool(args.trainer_args), "trainer arguments are absent")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    assert_isolated_runtime()
    _require(STOP_ENVIRONMENT not in os.environ, f"{STOP_ENVIRONMENT} is forbidden")
    _require("PYTHONPATH" not in os.environ, "PYTHONPATH is forbidden after isolated bootstrap")
    for distributed_name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        _require(distributed_name not in os.environ, f"unexpected distributed environment: {distributed_name}")
    args = _parse(argv)
    install_early_signal_handlers()
    submission_root, submission_contract = bootstrap_submission(
        args.submission_root,
        args.submission_sha256,
    )
    _campaign, _manifest, launch = verify_exact_invocation(
        args.launch,
        args.trainer_args,
        submission_root=submission_root,
        bootstrap_contract=submission_contract,
    )
    # The byte is sent only after the complete stdlib-only snapshot/submission audit
    # and exact launch re-derivation. Signals were already latched safely above.
    notify_worker_ready(args.ready_fd)
    train = _register_and_import_trainer()
    for module_name, expected_path in (
        ("treewm", REPOSITORY_ROOT / "treewm/__init__.py"),
        ("treewm.utils.checkpoint", REPOSITORY_ROOT / "treewm/utils/checkpoint.py"),
        ("scripts.train", REPOSITORY_ROOT / "scripts/train.py"),
    ):
        _verify_imported_module(module_name, expected_path)

    # Hydra now sees the same argv it would see under direct scripts/train.py.
    sys.argv = [str(launch["argv"][1]), *args.trainer_args]
    train.main()
    return 0


if __name__ == "__main__":
    try:
        raw_argv = sys.argv[1:]
        if raw_argv and raw_argv[0] == "--_hydra-composition-smoke":
            raise SystemExit(hydra_composition_smoke(raw_argv))
        raise SystemExit(main(raw_argv))
    except EntryContractError as exc:
        print(f"Exp23 train-entry contract error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(2)
