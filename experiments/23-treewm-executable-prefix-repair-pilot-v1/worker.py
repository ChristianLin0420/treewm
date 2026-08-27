#!/usr/bin/env python3
"""Run one sealed Exp23 cell from scratch to 25k with exact safe requeue.

There are no lifecycle stages.  Update 5k is only a retrospective report boundary.
Every restart uses the same ``resume=auto`` invocation and accepts the same cell's
verified checkpoint at any durable update, including update 25k while terminal final
evaluation is pending.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pwd
import select
import signal
import stat
import subprocess
import sys
from types import ModuleType
from typing import Any, Mapping, Sequence


OBJECTIVE = "treewm_v2_grounded_executable_prefix_pilot_v1"
CAMPAIGN_ID = "treewm-executable-prefix-repair-pilot-v1"
STOP_ENVIRONMENT = "TREEWM_STOP_AFTER_UPDATE"
HEADLESS_RUNTIME_ENVIRONMENT = {
    "MUJOCO_GL": "egl",
    "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
}
PACKAGE_RELATIVE = Path("experiments/23-treewm-executable-prefix-repair-pilot-v1")
TOTAL_UPDATES = 25_000
LOG_EVERY = 50
BATCH_SIZE = 256
GRAD_ACCUM = 1
GRACEFUL_EXIT_CODE = 75
CANCELLED_EXIT_CODE = 76
CONTRACT_EXIT_CODE = 2
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

SUBMISSION_CONTRACT_NAME = "SUBMISSION_CONTRACT.json"
GLOBAL_CANCEL_NAME = "CANCEL_REQUESTED.json"
TASK_LAUNCH_NAME = "LAUNCH.json"
TASK_COMPLETE_NAME = "WORKER_COMPLETE.json"
TASK_CANCEL_NAME = "CANCEL_REQUESTED.json"
GENERATION_ROOT_NAME = "requeue"
GENERATION_START_NAME = "START.json"
WORKER_SIGNAL_READY_NAME = "WORKER_SIGNAL_READY.json"
USR1_REQUEST_NAME = "USR1_REQUESTED.json"
TERM_REQUEST_NAME = "TERM_REQUESTED.json"
REQUEUE_READY_NAME = "REQUEUE_READY.json"
REQUEUE_CALLING_NAME = "REQUEUE_CALLING.json"
GENERATION_COMPLETE_NAME = "WORKER_COMPLETE.json"
GENERATION_CANCELLED_NAME = "WORKER_CANCELLED.json"
GENERATION_FAILED_NAME = "WORKER_FAILED.json"

SHA256 = frozenset("0123456789abcdef")
SUBMISSION_CONTRACT_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "formal_validation",
        "submission_root",
        "snapshot_root",
        "package_protocol_sha256",
        "manifest_sha256",
        "trainer_code_fingerprint",
        "runtime_sha256",
        "orchestration_interpreter",
        "weight_audit_artifact_sha256",
        "prefix_target_artifact_sha256",
        "resolved_config_artifact_sha256",
        "causal_parity_artifact_sha256",
        "snapshot_inventory",
        "snapshot_inventory_sha256",
        "live_audit_replays",
        "snapshot_audit_replays",
        "direct_hydra_compositions",
        "scientific_output_fingerprint_before",
        "scientific_output_fingerprint_after",
        "full_output_fingerprint_before",
        "full_output_fingerprint_after",
        "snapshot_full_output_fingerprint_before",
        "snapshot_full_output_fingerprint_after",
        "snapshot_scientific_output_fingerprint_before",
        "snapshot_scientific_output_fingerprint_after",
        "git_provenance",
        "launches",
        "array",
        "fresh_start",
    }
)
INTERPRETER_CONTRACT_FIELDS = frozenset(
    {
        "lexical_executable",
        "lexical_symlink_target",
        "resolved_executable",
        "resolved_executable_sha256",
        "resolved_executable_size",
        "base_executable",
        "venv_site_packages",
        "base_site_packages",
        "python_version",
    }
)
ARTIFACT_BASE_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "submission_sha256",
        "launch_sha256",
        "cell_index",
        "restart_count",
        "array_job_id",
        "array_task_id",
    }
)
REQUEUE_READY_FIELDS = ARTIFACT_BASE_FIELDS | frozenset(
    {
        "status",
        "trainer_exit_code",
        "checkpoint_kind",
        "completed_updates",
        "phase",
        "pending_eval_step",
        "checkpoint_sha256",
        "checkpoint_file_identity",
        "final_eval_progress_sha256",
    }
)
REQUEUE_CALLING_FIELDS = ARTIFACT_BASE_FIELDS | frozenset(
    {
        "status",
        "requeue_target",
        "requeue_ready_sha256",
        "checkpoint_sha256",
        "checkpoint_file_identity",
    }
)
SIGNAL_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "cell_index",
        "restart_count",
        "array_job_id",
        "array_task_id",
        "signal",
    }
)
SUBMISSION_LAUNCH_FIELDS = frozenset(
    {
        "index",
        "path",
        "launch_sha256",
        "launch_file_sha256",
        "setting_id",
        "arm_id",
        "seed",
        "weight_audit_artifact_sha256",
        "prefix_target_artifact_sha256",
        "resolved_config_artifact_sha256",
        "causal_parity_artifact_sha256",
    }
)
COMPLETION_FIELDS = frozenset(
    {
        "schema_version",
        "objective_version",
        "status",
        "run_identity",
        "identity_sha256",
        "evaluation_seed_tables",
        "evaluation_seed_tables_sha256",
        "final_seed_table_sha256",
        "protocol_sha256",
        "code_sha256",
        "runtime_sha256",
        "runtime",
        "data_manifest_sha256",
        "calibration_sha256",
        "future_recipe_sha256",
        "recipe_code_sha256",
        "recipe_runtime_sha256",
        "arm",
        "model_class",
        "scorer",
        "setting",
        "env_name",
        "dataset_kind",
        "source_name",
        "dataset_dir",
        "seed",
        "wandb_id",
        "wandb_group",
        "completed_updates",
        "scheduler_total_steps",
        "final_eval_step",
        "task_ids",
        "episodes_per_task",
        "node_budget",
        "branch_factor",
        "gradient_checkpointing",
        "future_set_cache",
        "shared_cache",
        "retrieval_enabled",
        "retrieval_num_keys",
        "final_evaluation",
        "checkpoint",
        "final_eval_progress",
    }
)
EPISODE_FIELDS = frozenset(
    {
        "success",
        "steps",
        "replans",
        "nodes",
        "final_goal_distance",
        "best_goal_distance",
        "chunk_lengths",
        "selected_depths",
        "initial_goal_distance",
        "displacement",
        "path_length",
        "action_magnitude",
        "no_action_plans",
        "guard_plans",
        "guard_rejections",
        "guard_candidate_count",
        "guard_accepted_count",
        "guard_best_predicted_improvements",
        "guard_selected_predicted_improvements",
        "trajectory",
        "progress",
        "task_index",
        "task_id",
        "episode_index",
        "episode_seed",
        "planning_wall_clock_s",
    }
)
PER_UPDATE_TRACKER_TAGS = frozenset(
    {
        "train/grad_norm",
        "train/grad_norm_world_rest",
        "train/grad_norm_branch_transformer",
        "train/grad_norm_world",
        "train/grad_norm_gain",
        "train/grad_clip_coefficient_world_rest",
        "train/grad_clip_coefficient_branch_transformer",
        "train/grad_clip_coefficient_world",
        "train/grad_clip_coefficient_gain",
        "train/learning_rate",
        "train/learning_rate_branch_transformer",
        "train/learning_rate_gain",
        "train/weight_decay",
        "train/weight_decay_gain",
    }
)
PER_EXAMPLE_TRACKER_TAGS = frozenset(
    {
        "train/loss_total",
        "train/loss_executable_prefix_action",
        "train/loss_executable_prefix_latent",
        "train/loss_executable_prefix_endpoint",
        "train/loss_raw/executable_prefix_action",
        "train/loss_raw/executable_prefix_latent",
        "train/loss_raw/executable_prefix_endpoint",
        "train/loss_effective/executable_prefix_action",
        "train/loss_effective/executable_prefix_latent",
        "train/loss_effective/executable_prefix_endpoint",
        "train/loss_weight/executable_prefix_action",
        "train/loss_weight/executable_prefix_latent",
        "train/loss_weight/executable_prefix_endpoint",
        "train/executable_prefix/schema_version",
        "train/executable_prefix/valid_anchor_fraction",
        "train/executable_prefix/prefix_steps_mean",
    }
)


class LifecycleError(RuntimeError):
    """A launch, checkpoint, lineage, or terminal artifact is unsafe."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise LifecycleError(message)


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


def sha256_string(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA256


def lexical_exists(path: str | Path) -> bool:
    """Existence of a directory entry, including a broken symlink."""
    return os.path.lexists(os.fspath(path))


_DIRECTORY_FLAGS = (
    os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
)
_FILE_FLAGS = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)


def _absolute_path(path: str | Path, label: str) -> Path:
    value = Path(path)
    require(value.is_absolute(), f"{label} is not absolute")
    require(all(part not in {"", ".", ".."} for part in value.parts[1:]), f"{label} is not normalized")
    return value


def _safe_relative(path: str | Path, label: str) -> Path:
    value = Path(path)
    require(
        not value.is_absolute()
        and bool(value.parts)
        and all(part not in {"", ".", ".."} for part in value.parts),
        f"{label} is not a safe relative path",
    )
    return value


def _open_absolute_directory(
    path: str | Path,
    label: str,
    *,
    create: bool = False,
    create_mode: int = 0o700,
) -> int:
    """Open every absolute component with O_NOFOLLOW, optionally mkdir+fsync."""
    absolute = _absolute_path(path, label)
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        for part in absolute.parts[1:]:
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                require(create, f"{label} is unavailable: {absolute}")
                try:
                    os.mkdir(part, create_mode, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
                os.fsync(child)
                os.fsync(descriptor)
            except OSError as exc:
                raise LifecycleError(f"{label} has a symlink/non-directory component: {exc}") from exc
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_relative_directory(
    root_fd: int,
    relative: Path,
    label: str,
    *,
    create: bool = False,
    create_mode: int = 0o700,
) -> int:
    descriptor = os.dup(root_fd)
    try:
        for part in relative.parts:
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                require(create, f"{label} is unavailable")
                try:
                    os.mkdir(part, create_mode, dir_fd=descriptor)
                except FileExistsError:
                    pass
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
                os.fsync(child)
                os.fsync(descriptor)
            except OSError as exc:
                raise LifecycleError(f"{label} has a symlink/non-directory component: {exc}") from exc
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _open_absolute_directory_optional(path: str | Path, label: str) -> int | None:
    """As _open_absolute_directory, but return None only for a genuinely absent component."""
    absolute = _absolute_path(path, label)
    descriptor = os.open("/", _DIRECTORY_FLAGS)
    try:
        for part in absolute.parts[1:]:
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                os.close(descriptor)
                return None
            except OSError as exc:
                raise LifecycleError(f"{label} has a symlink/non-directory component: {exc}") from exc
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _open_relative_directory_optional(root_fd: int, relative: Path, label: str) -> int | None:
    descriptor = os.dup(root_fd)
    try:
        for part in relative.parts:
            try:
                child = os.open(part, _DIRECTORY_FLAGS, dir_fd=descriptor)
            except FileNotFoundError:
                os.close(descriptor)
                return None
            except OSError as exc:
                raise LifecycleError(f"{label} has a symlink/non-directory component: {exc}") from exc
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        try:
            os.close(descriptor)
        except OSError:
            pass
        raise


def _open_absolute_regular(path: str | Path, label: str) -> tuple[int, os.stat_result]:
    absolute = _absolute_path(path, label)
    require(len(absolute.parts) > 1, f"{label} cannot be filesystem root")
    parent_fd = _open_absolute_directory(absolute.parent, f"parent of {label}")
    try:
        try:
            descriptor = os.open(absolute.name, _FILE_FLAGS, dir_fd=parent_fd)
        except OSError as exc:
            raise LifecycleError(f"{label} is unavailable or symlinked: {exc}") from exc
    finally:
        os.close(parent_fd)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        os.close(descriptor)
        raise LifecycleError(f"{label} is not a regular nonsymlink file")
    return descriptor, info


def _stat_identity(info: os.stat_result) -> dict[str, int]:
    return {
        "device": int(info.st_dev),
        "inode": int(info.st_ino),
        "size": int(info.st_size),
        "mtime_ns": int(info.st_mtime_ns),
        "ctime_ns": int(info.st_ctime_ns),
    }


def validate_file_identity(value: object, label: str) -> dict[str, int]:
    expected = {"device", "inode", "size", "mtime_ns", "ctime_ns"}
    require(isinstance(value, Mapping) and set(value) == expected, f"{label} fields differ")
    result: dict[str, int] = {}
    for key in sorted(expected):
        item = value[key]
        require(type(item) is int and int(item) >= 0, f"{label} {key} differs")
        result[key] = int(item)
    return result


def _read_fd_stable(descriptor: int, label: str) -> tuple[bytes, os.stat_result]:
    before = os.fstat(descriptor)
    require(stat.S_ISREG(before.st_mode), f"{label} is not regular")
    os.lseek(descriptor, 0, os.SEEK_SET)
    chunks: list[bytes] = []
    while block := os.read(descriptor, 16 * 1024 * 1024):
        chunks.append(block)
    after = os.fstat(descriptor)
    require(_stat_identity(before) == _stat_identity(after), f"{label} changed while open")
    data = b"".join(chunks)
    require(len(data) == before.st_size, f"{label} short read")
    return data, before


def _hash_fd_stable(descriptor: int, label: str) -> tuple[str, os.stat_result]:
    payload, info = _read_fd_stable(descriptor, label)
    return hashlib.sha256(payload).hexdigest(), info


def file_sha256(path: str | Path) -> str:
    descriptor, _ = _open_absolute_regular(path, f"SHA256 source {path}")
    try:
        digest, _ = _hash_fd_stable(descriptor, f"SHA256 source {path}")
        return digest
    finally:
        os.close(descriptor)


def _pairs(path: Path):
    def hook(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise LifecycleError(f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    return hook


def _decode_json_bytes(payload: bytes, source: Path) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8")
        value = json.loads(
            text,
            object_pairs_hook=_pairs(source),
            parse_constant=lambda token: (_ for _ in ()).throw(
                LifecycleError(f"non-finite JSON value in {source}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LifecycleError(f"cannot read {source}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {source}")
    return value


def read_json(path: str | Path) -> dict[str, Any]:
    source = _absolute_path(path, f"JSON artifact {path}")
    descriptor, _ = _open_absolute_regular(source, f"JSON artifact {source}")
    try:
        payload, _ = _read_fd_stable(descriptor, f"JSON artifact {source}")
    finally:
        os.close(descriptor)
    return _decode_json_bytes(payload, source)


def read_json_artifact(
    path: str | Path,
    label: str,
    *,
    required_mode: int | None = None,
) -> tuple[dict[str, Any], str, dict[str, int]]:
    source = _absolute_path(path, label)
    descriptor, path_info = _open_absolute_regular(source, label)
    try:
        payload, opened = _read_fd_stable(descriptor, label)
    finally:
        os.close(descriptor)
    require(_stat_identity(path_info) == _stat_identity(opened), f"{label} path/open identity differs")
    if required_mode is not None:
        require(stat.S_IMODE(opened.st_mode) == required_mode, f"{label} mode differs")
    return (
        _decode_json_bytes(payload, source),
        hashlib.sha256(payload).hexdigest(),
        _stat_identity(opened),
    )


def _fsync_directory(path: Path) -> None:
    descriptor = _open_absolute_directory(path, f"fsync directory {path}")
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def ensure_directory(path: str | Path, label: str, *, create: bool = False) -> Path:
    directory = _absolute_path(path, label)
    descriptor = _open_absolute_directory(directory, label, create=create)
    os.close(descriptor)
    return directory


def ensure_contained_directory(
    path: str | Path,
    root: str | Path,
    label: str,
    *,
    create: bool = False,
) -> Path:
    expected_root = ensure_directory(root, f"{label} declared root")
    directory = _absolute_path(path, label)
    try:
        relative = directory.relative_to(expected_root)
    except ValueError as exc:
        raise LifecycleError(f"{label} escapes its declared root") from exc
    root_fd = _open_absolute_directory(expected_root, f"{label} declared root")
    try:
        descriptor = _open_relative_directory(
            root_fd,
            relative,
            label,
            create=create,
        )
        os.close(descriptor)
    finally:
        os.close(root_fd)
    return directory


def require_regular_nonsymlink(path: str | Path, label: str) -> Path:
    source = _absolute_path(path, label)
    descriptor, _ = _open_absolute_regular(source, label)
    os.close(descriptor)
    return source


def contained(path: str | Path, root: str | Path, label: str, *, strict: bool) -> Path:
    root_path = ensure_directory(root, f"{label} root")
    candidate = _absolute_path(path, label)
    try:
        relative = candidate.relative_to(root_path)
    except ValueError as exc:
        raise LifecycleError(f"{label} escapes its declared root") from exc
    if strict:
        parent_fd = _open_absolute_directory(root_path, f"{label} root")
        try:
            if relative.parts:
                parent = _open_relative_directory(parent_fd, relative.parent, f"parent of {label}")
                os.close(parent)
        finally:
            os.close(parent_fd)
    return candidate


def seal_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Create one immutable JSON object without exposing partial contents."""
    destination = _absolute_path(path, f"artifact {path}")
    parent_fd = _open_absolute_directory(
        destination.parent,
        f"parent of {destination}",
        create=True,
    )
    payload = (json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    temporary = f".{destination.name}.{os.getpid()}.{os.urandom(8).hex()}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=parent_fd,
        )
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            os.fchmod(handle.fileno(), 0o444)
            os.fsync(handle.fileno())
        try:
            os.link(
                temporary,
                destination.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
            os.fsync(parent_fd)
        except FileExistsError:
            existing_fd, existing_info = _open_absolute_regular(destination, f"existing artifact {destination}")
            os.close(existing_fd)
            require(
                stat.S_IMODE(existing_info.st_mode) == 0o444,
                f"immutable artifact mode differs: {destination}",
            )
            require(read_json(destination) == dict(value), f"immutable artifact differs: {destination}")
        return destination
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            os.unlink(temporary, dir_fd=parent_fd)
            os.fsync(parent_fd)
        except FileNotFoundError:
            pass
        os.close(parent_fd)


def assert_isolated_runtime() -> None:
    require(sys.flags.isolated == 1, "worker requires Python -I")
    require(sys.flags.no_site == 1, "worker requires Python -S")
    require(bool(sys.dont_write_bytecode), "worker requires Python -B")
    require("sitecustomize" not in sys.modules, "sitecustomize loaded before bootstrap")
    require("usercustomize" not in sys.modules, "usercustomize loaded before bootstrap")
    require("treewm" not in sys.modules and "torch" not in sys.modules, "scientific modules loaded before bootstrap")
    require("" not in sys.path, "current directory is importable before bootstrap")
    require(
        not any("site-packages" in item for item in sys.path),
        "site-packages was active before verified bootstrap",
    )


def _verify_snapshot_tree(snapshot_root: Path, inventory: Mapping[str, Any]) -> None:
    expected: dict[str, str] = {}
    expected_directories: set[str] = set()
    for raw_relative, digest in inventory.items():
        require(isinstance(raw_relative, str), "snapshot inventory path is not text")
        relative = _safe_relative(raw_relative, "snapshot inventory path")
        require(sha256_string(digest), f"snapshot inventory SHA256 is malformed: {relative}")
        rendered = str(relative)
        require(rendered not in expected, f"duplicate snapshot inventory path: {rendered}")
        expected[rendered] = str(digest)
        for parent in relative.parents:
            if parent != Path("."):
                expected_directories.add(str(parent))
    for required in (
        str(PACKAGE_RELATIVE / "worker.py"),
        str(PACKAGE_RELATIVE / "train_entry.py"),
        str(PACKAGE_RELATIVE / "train.slurm"),
        "scripts/__init__.py",
        "scripts/train.py",
    ):
        require(required in expected, f"snapshot inventory omits lifecycle source: {required}")

    root_fd = _open_absolute_directory(snapshot_root, "snapshot root")
    actual: dict[str, str] = {}
    actual_directories: set[str] = set()

    def walk(directory_fd: int, prefix: Path) -> None:
        info = os.fstat(directory_fd)
        require(stat.S_IMODE(info.st_mode) == 0o555, f"snapshot directory mode differs: {prefix}")
        try:
            names = sorted(os.listdir(directory_fd))
        except OSError as exc:
            raise LifecycleError(f"cannot enumerate snapshot directory {prefix}: {exc}") from exc
        for name in names:
            require(name not in {"", ".", ".."} and "/" not in name, "invalid snapshot entry name")
            relative = prefix / name if prefix != Path(".") else Path(name)
            rendered = str(relative)
            try:
                entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except OSError as exc:
                raise LifecycleError(f"cannot lstat snapshot entry {rendered}: {exc}") from exc
            if stat.S_ISDIR(entry.st_mode):
                require(stat.S_IMODE(entry.st_mode) == 0o555, f"snapshot directory mode differs: {rendered}")
                require(rendered in expected_directories, f"snapshot has an extra directory: {rendered}")
                child = os.open(name, _DIRECTORY_FLAGS, dir_fd=directory_fd)
                try:
                    require(_stat_identity(entry) == _stat_identity(os.fstat(child)), f"snapshot directory swapped: {rendered}")
                    actual_directories.add(rendered)
                    walk(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(entry.st_mode):
                require(stat.S_IMODE(entry.st_mode) == 0o444, f"snapshot file mode differs: {rendered}")
                require(rendered in expected, f"snapshot has an unclaimed file: {rendered}")
                descriptor = os.open(name, _FILE_FLAGS, dir_fd=directory_fd)
                try:
                    digest, opened = _hash_fd_stable(descriptor, f"snapshot file {rendered}")
                    require(_stat_identity(entry) == _stat_identity(opened), f"snapshot file swapped: {rendered}")
                finally:
                    os.close(descriptor)
                require(digest == expected[rendered], f"snapshot bytes differ: {rendered}")
                actual[rendered] = digest
            else:
                raise LifecycleError(f"snapshot contains symlink/special file: {rendered}")

    try:
        walk(root_fd, Path("."))
    finally:
        os.close(root_fd)
    require(actual == expected, "snapshot file coverage differs from exact inventory")
    require(actual_directories == expected_directories, "snapshot directory coverage differs from exact inventory")


def configure_verified_import_paths(snapshot_root: Path) -> None:
    """Append literal import roots without executing site.py or any .pth file."""
    expected_suffix = [str(snapshot_root), *(str(path) for path in PINNED_SITE_DIRECTORIES)]
    if sys.path[-len(expected_suffix) :] == expected_suffix:
        return
    require(not any(path in sys.path for path in expected_suffix), "verified import root was pre-injected")
    for directory in PINNED_SITE_DIRECTORIES:
        try:
            info = directory.lstat()
        except OSError as exc:
            raise LifecycleError(f"pinned site directory is unavailable: {directory}: {exc}") from exc
        require(stat.S_ISDIR(info.st_mode), f"pinned site path is not a literal directory: {directory}")
    sys.path.append(str(snapshot_root))
    for directory in PINNED_SITE_DIRECTORIES:
        sys.path.append(str(directory))
    require(sys.path[-len(expected_suffix) :] == expected_suffix, "verified import path suffix differs")


def bootstrap_submission(
    submission_root: Path,
    submission_sha256: str,
    *,
    expected_snapshot_root: Path | None = None,
    configure_imports: bool = False,
) -> tuple[Path, Path, dict[str, Any]]:
    """Verify the trusted contract and entire read-only snapshot using stdlib only."""
    require(sha256_string(submission_sha256), "submission SHA256 is malformed")
    submission = ensure_directory(submission_root, "submission root")
    contract_path = submission / SUBMISSION_CONTRACT_NAME
    descriptor, info = _open_absolute_regular(contract_path, "submission contract")
    try:
        payload, opened = _read_fd_stable(descriptor, "submission contract")
    finally:
        os.close(descriptor)
    require(_stat_identity(info) == _stat_identity(opened), "submission contract path/open identity differs")
    require(stat.S_IMODE(opened.st_mode) == 0o444, "submission contract mode is not 0444")
    require(hashlib.sha256(payload).hexdigest() == submission_sha256, "submission contract bytes differ")
    contract = _decode_json_bytes(payload, contract_path)
    require(set(contract) == SUBMISSION_CONTRACT_FIELDS, "submission contract fields differ")
    require(contract.get("schema_version") == 1, "submission contract schema differs")
    require(contract.get("status") == "sealed_for_submission", "submission is not sealed")
    require(contract.get("campaign_id") == CAMPAIGN_ID, "submission campaign differs")
    require(contract.get("formal_validation") is False, "submission formal-validation label differs")
    require(contract.get("array") == "0-19%20", "submission array differs")
    require(contract.get("fresh_start") is True, "submission is not fresh-start")
    require(contract.get("submission_root") == str(submission), "submission root binding differs")
    interpreter = contract.get("orchestration_interpreter")
    require(isinstance(interpreter, Mapping), "submission interpreter contract is absent")
    require(set(interpreter) == INTERPRETER_CONTRACT_FIELDS, "submission interpreter fields differ")
    require(
        interpreter.get("lexical_executable") == str(PINNED_PYTHON)
        and os.path.normpath(os.path.abspath(sys.executable)) == str(PINNED_PYTHON),
        "worker interpreter binding differs",
    )
    try:
        lexical_info = PINNED_PYTHON.lstat()
    except OSError as exc:
        raise LifecycleError(f"pinned lexical interpreter is unavailable: {exc}") from exc
    require(stat.S_ISLNK(lexical_info.st_mode), "pinned lexical interpreter is not the sealed venv symlink")
    lexical_target = os.readlink(PINNED_PYTHON) if stat.S_ISLNK(lexical_info.st_mode) else None
    require(
        interpreter.get("lexical_symlink_target") == lexical_target,
        "worker interpreter symlink binding differs",
    )
    resolved_python = PINNED_PYTHON.resolve(strict=True)
    require(
        interpreter.get("resolved_executable") == str(resolved_python),
        "worker resolved interpreter path differs",
    )
    resolved_fd, resolved_info = _open_absolute_regular(
        resolved_python,
        "resolved pinned interpreter",
    )
    try:
        resolved_digest, stable_resolved_info = _hash_fd_stable(
            resolved_fd,
            "resolved pinned interpreter",
        )
    finally:
        os.close(resolved_fd)
    require(
        _stat_identity(resolved_info) == _stat_identity(stable_resolved_info)
        and interpreter.get("resolved_executable_sha256") == resolved_digest
        and interpreter.get("resolved_executable_size") == resolved_info.st_size,
        "worker resolved interpreter identity differs",
    )
    require(
        interpreter.get("base_executable") == str(getattr(sys, "_base_executable", "")),
        "worker base interpreter binding differs",
    )
    require(
        interpreter.get("venv_site_packages") == str(PINNED_SITE_DIRECTORIES[0])
        and interpreter.get("base_site_packages") == str(PINNED_SITE_DIRECTORIES[1]),
        "worker site-package binding differs",
    )
    require(
        interpreter.get("python_version")
        == f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "worker Python version differs",
    )
    for prefix in ("", "snapshot_"):
        for flavor in ("full_output", "scientific_output"):
            before = contract.get(f"{prefix}{flavor}_fingerprint_before")
            after = contract.get(f"{prefix}{flavor}_fingerprint_after")
            require(before == after, f"submission {prefix}{flavor} fingerprint drifted")
    snapshot = _absolute_path(str(contract.get("snapshot_root", "")), "snapshot root binding")
    try:
        snapshot.relative_to(submission)
    except ValueError as exc:
        raise LifecycleError("snapshot root escapes submission root") from exc
    if expected_snapshot_root is not None:
        require(snapshot == _absolute_path(expected_snapshot_root, "requested snapshot root"), "snapshot root binding differs")
    inventory = contract.get("snapshot_inventory")
    require(isinstance(inventory, Mapping) and bool(inventory), "snapshot inventory is absent")
    require(stable_hash(inventory) == contract.get("snapshot_inventory_sha256"), "snapshot inventory hash differs")
    _verify_snapshot_tree(snapshot, inventory)
    if configure_imports:
        configure_verified_import_paths(snapshot)
    return snapshot, submission, contract


def launch_path_for(submission_root: Path, cell_index: int) -> Path:
    return submission_root / "launches" / f"cell-{cell_index:02d}.json"


def task_root_for(submission_root: Path, cell_index: int) -> Path:
    return submission_root / "tasks" / f"cell-{cell_index:02d}"


def generation_root_for(task_root: Path, restart_count: int) -> Path:
    return task_root / GENERATION_ROOT_NAME / str(restart_count)


def _load_campaign(snapshot_root: Path) -> ModuleType:
    package = snapshot_root / PACKAGE_RELATIVE
    path = package / "campaign.py"
    require_regular_nonsymlink(path, "Exp23 campaign verifier")
    name = "_treewm_exp23_campaign_for_worker"
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, "cannot load Exp23 campaign")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def validate_submission_contract(
    submission_root: Path,
    submission_sha256: str,
    *,
    contract: Mapping[str, Any] | None = None,
    snapshot_root: Path | None = None,
    protocol_sha256: str | None = None,
    manifest_sha256: str | None = None,
    cell_index: int | None = None,
    launch: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if contract is None:
        boot_snapshot, boot_submission, loaded = bootstrap_submission(
            submission_root,
            submission_sha256,
            expected_snapshot_root=snapshot_root,
        )
        submission_root = boot_submission
        snapshot_root = boot_snapshot if snapshot_root is None else snapshot_root
        contract = loaded
    contract = dict(contract)
    require(contract.get("submission_root") == str(submission_root), "submission root binding differs")
    if snapshot_root is not None:
        require(contract.get("snapshot_root") == str(snapshot_root), "snapshot root binding differs")
    if protocol_sha256 is not None:
        require(contract.get("package_protocol_sha256") == protocol_sha256, "submission protocol differs")
    if manifest_sha256 is not None:
        require(contract.get("manifest_sha256") == manifest_sha256, "submission manifest differs")
    launches = contract.get("launches")
    require(isinstance(launches, list) and len(launches) == 20, "submission launch inventory differs")
    audit_values = {
        name: contract[name]
        for name in (
            "weight_audit_artifact_sha256",
            "prefix_target_artifact_sha256",
            "resolved_config_artifact_sha256",
            "causal_parity_artifact_sha256",
        )
    }
    for index, row in enumerate(launches):
        require(isinstance(row, Mapping), f"submission launch row {index} is invalid")
        require(set(row) == SUBMISSION_LAUNCH_FIELDS, f"submission launch row {index} fields differ")
        expected_relative = f"launches/cell-{index:02d}.json"
        require(row.get("index") == index, f"submission launch index {index} differs")
        require(row.get("path") == expected_relative, f"submission launch path {index} differs")
        require(sha256_string(row.get("launch_sha256")), f"submission launch hash {index} is malformed")
        require(
            sha256_string(row.get("launch_file_sha256")),
            f"submission launch file hash {index} is malformed",
        )
        for name, expected_audit in audit_values.items():
            require(row.get(name) == expected_audit, f"submission launch audit {index}/{name} differs")
        inventory_path = submission_root / expected_relative
        # Check the original directory entry before resolution; otherwise a symlink
        # to an in-root regular file would pass the subsequent lstat.
        contained(inventory_path, submission_root, f"submission launch file {index}", strict=True)
        inventory_launch, inventory_digest, _ = read_json_artifact(
            inventory_path,
            f"submission launch file {index}",
            required_mode=0o444,
        )
        require(
            inventory_digest == row["launch_file_sha256"],
            f"submission launch file bytes {index} differ",
        )
        require(
            inventory_launch.get("launch_sha256") == row["launch_sha256"],
            f"submission launch content hash {index} differs",
        )
        inventory_body = dict(inventory_launch)
        inventory_claim = inventory_body.pop("launch_sha256", None)
        require(inventory_claim == stable_hash(inventory_body), f"submission launch self hash {index} differs")
    if cell_index is not None and launch is not None:
        row = launches[cell_index]
        expected_relative = f"launches/cell-{cell_index:02d}.json"
        require(row.get("path") == expected_relative, "submission launch path differs")
        require(row.get("launch_sha256") == launch.get("launch_sha256"), "submission launch hash differs")
        require(
            row.get("launch_file_sha256")
            == file_sha256(submission_root / expected_relative),
            "submission launch file hash differs",
        )
    return contract


def load_launch_context(
    *,
    snapshot_root: Path,
    submission_root: Path,
    submission_sha256: str,
    cell_index: int,
    bootstrap_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    snapshot_root = ensure_directory(snapshot_root, "snapshot root")
    submission_root = ensure_directory(submission_root, "submission root")
    require(0 <= cell_index < 20, "cell index is out of range")
    if bootstrap_contract is None:
        boot_snapshot, boot_submission, bootstrap_contract = bootstrap_submission(
            submission_root,
            submission_sha256,
            expected_snapshot_root=snapshot_root,
            configure_imports=True,
        )
        require(boot_snapshot == snapshot_root and boot_submission == submission_root, "bootstrap roots differ")
    else:
        configure_verified_import_paths(snapshot_root)
    validate_submission_contract(
        submission_root,
        submission_sha256,
        contract=bootstrap_contract,
        snapshot_root=snapshot_root,
    )
    package = contained(snapshot_root / PACKAGE_RELATIVE, snapshot_root, "Exp23 package", strict=True)
    campaign = _load_campaign(snapshot_root)
    manifest, weight_lock = campaign.load_contract(snapshot_root)
    protocol = campaign.verify_protocol_lock(package)
    cells = campaign.expand_matrix(manifest)
    launch_path = launch_path_for(submission_root, cell_index)
    require_regular_nonsymlink(launch_path, "cell launch")
    contained(launch_path, submission_root, "launch path", strict=True)
    launch = read_json(launch_path)
    expected = campaign.trainer_command(
        manifest,
        weight_lock,
        cells[cell_index],
        repo_root=snapshot_root,
        package_protocol_sha256=protocol,
    )
    require(launch == expected, "cell launch differs from snapshot re-derivation")
    launch_body = dict(launch)
    claimed_launch_sha = launch_body.pop("launch_sha256", None)
    require(claimed_launch_sha == stable_hash(launch_body), "launch self hash differs")
    require("resume=auto" in launch["argv"], "launch does not use resume=auto")
    require("train.steps=25000" in launch["argv"], "launch is not scratch-to-25k")
    require(
        STOP_ENVIRONMENT not in launch["environment"],
        "launch contains staged-stop environment",
    )
    for name, value in HEADLESS_RUNTIME_ENVIRONMENT.items():
        require(
            launch["environment"].get(name) == value,
            f"launch headless runtime binding differs: {name}",
        )
    validate_submission_contract(
        submission_root,
        submission_sha256,
        contract=bootstrap_contract,
        snapshot_root=snapshot_root,
        protocol_sha256=protocol,
        manifest_sha256=campaign.manifest_sha256(manifest),
        cell_index=cell_index,
        launch=launch,
    )
    config_lock = campaign.read_json(package / "resolved_config.lock.json")
    row = config_lock["matrix"][cell_index]
    require(row.get("index") == cell_index, "resolved-config row differs")
    require(
        stable_hash(row["resolved_config"]) == row["resolved_config_sha256"],
        "resolved-config row hash differs",
    )
    relative_argv = [
        launch["argv"][0],
        str(Path(launch["argv"][1]).relative_to(snapshot_root)),
        *launch["argv"][2:],
    ]
    require(
        relative_argv == row.get("trainer_argv_repo_relative")
        and stable_hash(relative_argv) == row.get("trainer_argv_sha256"),
        "direct Hydra argv differs from the frozen resolved-config audit",
    )
    expected_audits = {
        "weight_audit_artifact_sha256": manifest["weight_audit"]["artifact_sha256"],
        "prefix_target_artifact_sha256": manifest["prefix_target_contract"]["artifact_sha256"],
        "resolved_config_artifact_sha256": manifest["resolved_config_contract"]["artifact_sha256"],
        "causal_parity_artifact_sha256": manifest["causal_parity_contract"]["artifact_sha256"],
    }
    for name, value in expected_audits.items():
        require(sha256_string(value) and launch["hashes"].get(name) == value, f"launch {name} differs")
    require(
        launch["environment"].get("TREEWM_RESOLVED_CONFIG_SHA256")
        == expected_audits["resolved_config_artifact_sha256"],
        "resolved-config environment binding differs",
    )
    require(
        launch["environment"].get("TREEWM_CAUSAL_PARITY_SHA256")
        == expected_audits["causal_parity_artifact_sha256"],
        "causal-parity environment binding differs",
    )
    expected_final_rows = [
        {
            "task_index": task_index,
            "task_id": int(task_id),
            "episode_index": episode_index,
            "episode_seed": int(manifest["scientific_contract"]["evaluation_seed"])
            + 1000 * task_index
            + episode_index,
        }
        for task_index, task_id in enumerate(manifest["scientific_contract"]["task_ids"])
        for episode_index in range(int(manifest["scientific_contract"]["final_episodes_per_task"]))
    ]
    require(
        len(expected_final_rows) == 25
        and launch["hashes"].get("actual_final_evaluation_rows_sha256")
        == stable_hash(expected_final_rows),
        "actual terminal episode identity binding differs",
    )
    declared_run_root = _absolute_path(manifest["paths"]["run_root"], "declared run root")
    run_directory = _absolute_path(launch["cell"]["run_directory"], "cell run directory")
    try:
        run_relative = run_directory.relative_to(declared_run_root)
    except ValueError as exc:
        raise LifecycleError("cell run directory escapes declared run root") from exc
    require(bool(run_relative.parts), "cell run directory equals declared run root")
    source_contract = campaign.source_contract(snapshot_root)
    return {
        "snapshot_root": snapshot_root,
        "submission_root": submission_root,
        "submission_sha256": submission_sha256,
        "package": package,
        "campaign": campaign,
        "manifest": manifest,
        "protocol": protocol,
        "cell": cells[cell_index],
        "launch_path": launch_path,
        "launch": launch,
        "resolved_row": row,
        "submission_contract": dict(bootstrap_contract),
        "run_root": declared_run_root,
        "run_directory": run_directory,
        "run_relative": run_relative,
        "source_contract": source_contract,
    }


def _finite_mapping(value: object, label: str) -> dict[str, float]:
    require(isinstance(value, Mapping), f"{label} is not a mapping")
    result: dict[str, float] = {}
    for key, item in value.items():
        require(isinstance(key, str) and key, f"{label} has an invalid key")
        require(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item)),
            f"{label} contains a non-finite value: {key}",
        )
        result[key] = float(item)
    return result


def _config_string(config: Mapping[str, Any], key: str) -> str:
    return str(config[key]) if key in config else ""


def expected_evaluation_seed_tables(context: Mapping[str, Any]) -> dict[str, Any]:
    """Reconstruct the complete seed artifact from the sealed snapshot and launch."""
    config = context["resolved_row"]["resolved_config"]
    manifest = context["manifest"]
    launch = context["launch"]
    snapshot_root = context["snapshot_root"]
    require(str(snapshot_root) in sys.path, "snapshot import root was not verified")
    from treewm.evaluation import rollout

    rollout_path = require_regular_nonsymlink(
        Path(str(rollout.__file__)),
        "evaluation seed implementation",
    )
    require(
        rollout_path.is_relative_to(snapshot_root),
        "evaluation seed implementation imported outside snapshot",
    )
    task_ids = [int(item) for item in manifest["scientific_contract"]["task_ids"]]
    return rollout.build_evaluation_seed_tables(
        launch["environment"]["TREEWM_EVALUATION_SEED_PROTOCOL_SHA256"],
        int(config["seed"]),
        task_ids,
        int(config["eval"]["episodes_per_task"]),
        int(config["eval"]["final_episodes_per_task"]),
    )


def expected_run_identity(context: Mapping[str, Any]) -> dict[str, Any]:
    config = context["resolved_row"]["resolved_config"]
    launch = context["launch"]
    manifest = context["manifest"]
    cell = context["cell"]
    setting = next(row for row in manifest["settings"] if row["id"] == cell.setting)
    identity_config = copy.deepcopy(config)
    identity_config["resume"] = None

    task_ids = [int(item) for item in manifest["scientific_contract"]["task_ids"]]
    tables = expected_evaluation_seed_tables(context)
    result = {
        "schema_version": 1,
        "objective_version": OBJECTIVE,
        "run_dir": launch["cell"]["run_directory"],
        "run_name": launch["cell"]["run_name"],
        "arm": str(config["arm"]),
        "env_name": str(config["env"]["name"]),
        "setting": str(config["env"]["short_name"]),
        "dataset_kind": str(config["env"].get("dataset_kind", "standard")),
        "source_name": str(config["env"].get("source_name", config["env"]["name"])),
        "seed": int(config["seed"]),
        "total_steps": int(config["train"]["steps"]),
        "scheduler_total_steps": int(config["train"]["scheduler_total_steps"]),
        "world_size": 1,
        "model_class": str(manifest["method"]["model_class"]),
        "scorer": str(config["tree"]["scorer"]),
        "node_budget": int(config["tree"]["node_budget"]),
        "branch_factor": int(config["model"]["branch_factor"]),
        "gradient_checkpointing": bool(config["train"]["gradient_checkpointing"]),
        "future_set_cache": bool(config["future_sets"]["cache"]),
        "shared_cache": bool(config["future_sets"]["shared_cache"]),
        "retrieval_enabled": bool(config["retrieval"]["enabled"]),
        "retrieval_num_keys": int(config["retrieval"]["num_keys"]),
        "task_ids": task_ids,
        "final_episodes_per_task": int(config["eval"]["final_episodes_per_task"]),
        "evaluation_seed_protocol_sha256": launch["environment"]["TREEWM_EVALUATION_SEED_PROTOCOL_SHA256"],
        "evaluation_seed_tables_sha256": tables["sha256"],
        "monitor_seed_table_sha256": tables["monitor"]["sha256"],
        "final_seed_table_sha256": tables["final"]["sha256"],
        "config_sha256": stable_hash(identity_config),
        "protocol_sha256": launch["environment"]["TREEWM_PROTOCOL_SHA256"],
        "code_sha256": launch["hashes"]["source_sha256"],
        "runtime_sha256": launch["hashes"]["runtime_sha256"],
        "data_manifest_sha256": launch["hashes"]["data_manifest_sha256"],
        "calibration_sha256": launch["hashes"]["calibration_sha256"],
        "future_recipe_sha256": launch["hashes"]["future_recipe_sha256"],
        "recipe_anchor_policy": str(config["future_sets"].get("recipe_anchor_policy", "selected_seed")),
        "train_anchor_count": int(setting["published_union_train_anchors"]),
        "validation_anchor_count": int(setting["published_union_validation_anchors"]),
        "recipe_code_sha256": launch["environment"]["TREEWM_RECIPE_CODE_SHA256"],
        "recipe_runtime_sha256": launch["environment"]["TREEWM_RECIPE_RUNTIME_SHA256"],
        "campaign_source_sha256": _config_string(config, "campaign_source_sha256"),
        "campaign_protocol_sha256": _config_string(config, "campaign_protocol_sha256"),
        "campaign_config_sha256": _config_string(config, "campaign_config_sha256"),
        "campaign_input_contract_sha256": _config_string(config, "campaign_input_contract_sha256"),
        "campaign_factorial_arm": _config_string(config, "campaign_factorial_arm"),
        "campaign_prerequisite_binding_sha256": _config_string(config, "campaign_prerequisite_binding_sha256"),
        "campaign_selected_recipe_sha256": _config_string(config, "campaign_selected_recipe_sha256"),
        "wandb_project": launch["environment"]["WANDB_PROJECT"],
        "wandb_entity": "",
        "wandb_group": launch["environment"]["WANDB_RUN_GROUP"],
        "wandb_mode": launch["environment"]["WANDB_MODE"],
        "wandb_id": launch["environment"]["WANDB_RUN_ID"],
    }
    return result


def validate_metric_tracker_state(
    state: object,
    completed_updates: int,
    *,
    log_every: int = LOG_EVERY,
    batch_size: int = BATCH_SIZE,
) -> None:
    require(isinstance(state, Mapping), "checkpoint metric-tracker state is absent")
    require(set(state) == {"schema_version", "sums", "counts", "hists"}, "metric-tracker fields differ")
    require(state.get("schema_version") == 1, "metric-tracker schema differs")
    sums = state.get("sums")
    counts = state.get("counts")
    hists = state.get("hists")
    require(isinstance(sums, Mapping) and isinstance(counts, Mapping), "metric-tracker scalars malformed")
    require(set(sums) == set(counts), "metric-tracker sum/count keys differ")
    require(isinstance(hists, Mapping), "metric-tracker histograms malformed")
    require(not hists, "metric-tracker histogram window is stale")
    require(not any(str(name).endswith("__nonfinite") for name in sums), "metric-tracker records non-finite metrics")
    remainder = completed_updates % log_every
    if remainder == 0:
        require(not sums and not counts, "metric-tracker is nonempty at a logging boundary")
        return
    require(PER_UPDATE_TRACKER_TAGS <= set(sums), "metric-tracker lacks per-update tags")
    require(PER_EXAMPLE_TRACKER_TAGS <= set(sums), "metric-tracker lacks per-example tags")
    for name in sums:
        total = sums[name]
        count = counts[name]
        require(
            isinstance(total, (int, float))
            and not isinstance(total, bool)
            and math.isfinite(float(total)),
            f"metric-tracker sum is invalid: {name}",
        )
        require(
            isinstance(count, (int, float))
            and not isinstance(count, bool)
            and math.isfinite(float(count))
            and float(count) > 0.0,
            f"metric-tracker count is invalid: {name}",
        )
        expected_count = (
            remainder
            if name in PER_UPDATE_TRACKER_TAGS
            or name.startswith("train/grad_")
            or name.startswith("train/learning_rate")
            or name.startswith("train/weight_decay")
            else remainder * batch_size
        )
        require(float(count) == float(expected_count), f"metric-tracker cadence differs: {name}")


def validate_loader_state(
    state: object,
    completed_updates: int,
    *,
    train_anchor_count: int,
    batch_size: int = BATCH_SIZE,
    grad_accum: int = GRAD_ACCUM,
) -> None:
    require(isinstance(state, Mapping), "checkpoint loader state is absent")
    require(
        set(state) == {"epoch", "batches_yielded_in_epoch", "epoch_generator_state"},
        "checkpoint loader fields differ",
    )
    batches_per_epoch = train_anchor_count // batch_size
    require(batches_per_epoch > 0 and grad_accum == 1, "loader contract differs")
    consumed = completed_updates * grad_accum
    if consumed == 0:
        expected_epoch, expected_yielded = 0, 0
        require(state["epoch_generator_state"] is None, "zero-update loader has generator snapshot")
    else:
        expected_epoch = (consumed - 1) // batches_per_epoch
        expected_yielded = ((consumed - 1) % batches_per_epoch) + 1
        generator = state["epoch_generator_state"]
        require(hasattr(generator, "numel") and int(generator.numel()) > 0, "loader generator state is invalid")
    require(type(state.get("epoch")) is int and state["epoch"] == expected_epoch, "loader epoch differs")
    require(
        type(state.get("batches_yielded_in_epoch")) is int
        and state["batches_yielded_in_epoch"] == expected_yielded,
        "loader cursor differs",
    )


def validate_post_update_cadence(
    payload: Mapping[str, Any], completed_updates: int, config: Mapping[str, Any]
) -> dict[str, Any]:
    cadence = payload.get("post_update_cadence")
    require(isinstance(cadence, Mapping), "checkpoint lacks post_update_cadence")
    require(
        set(cadence)
        == {"schema_version", "committed_update", "completed_update", "replay_action"},
        "post-update cadence fields differ",
    )
    require(cadence.get("schema_version") == 1, "post-update cadence schema differs")
    require(cadence.get("committed_update") == completed_updates, "cadence committed update differs")
    phase = payload.get("phase")
    pending = payload.get("pending_eval_step")
    replay = cadence.get("replay_action")
    done = cadence.get("completed_update")
    if done == completed_updates:
        require(replay is None, "complete cadence retains replay intent")
        if phase == "train":
            require(pending is None, "complete training cadence retains evaluation intent")
        else:
            # Entering terminal evaluation deliberately uses a complete update-25k
            # cadence together with pending_eval_step=25000.  This is not periodic
            # evaluation replay and must remain resumable even after all 25 raw rows
            # were durably written but before the terminal checkpoint transaction.
            require(
                phase == "final_eval" and pending in {None, completed_updates},
                "terminal evaluation pending step differs",
            )
    else:
        require(done == max(0, completed_updates - 1), "cadence completed update differs")
        require(replay in {"evaluation", "visualization"}, "cadence replay action differs")
        if replay == "evaluation":
            require(
                completed_updates > 0
                and completed_updates % int(config["train"]["eval_every"]) == 0
                and pending == completed_updates,
                "evaluation replay intent is impossible",
            )
        else:
            early_until = int(config["train"]["viz_early_until"])
            stride = int(
                config["train"]["viz_every_early"]
                if completed_updates <= early_until
                else config["train"]["viz_every"]
            )
            require(
                completed_updates > 0
                and completed_updates % max(stride, 1) == 0
                and pending is None,
                "visualization replay intent is impossible",
            )
    if phase == "final_eval":
        require(done == completed_updates and replay is None, "final evaluation has incomplete cadence")
    return dict(cadence)


def _validate_number_list(value: object, label: str, *, integral: bool = False) -> None:
    require(isinstance(value, list), f"{label} is not a list")
    for item in value:
        if integral:
            require(type(item) is int and item >= 0, f"{label} contains an invalid integer")
        else:
            require(
                isinstance(item, (int, float))
                and not isinstance(item, bool)
                and math.isfinite(float(item)),
                f"{label} contains a non-finite value",
            )


def _validate_finite_numeric_tree(value: object, label: str) -> None:
    """Validate JSON trajectory nesting without silently accepting NaN/strings."""
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_finite_numeric_tree(item, f"{label}[{index}]")
        return
    require(
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value)),
        f"{label} contains a non-finite or nonnumeric leaf",
    )


def validate_episode_row(row: object, expected: tuple[int, int, int, int]) -> dict[str, Any]:
    require(isinstance(row, Mapping), "final evaluation row is not an object")
    require(set(row) == EPISODE_FIELDS, "final evaluation row fields differ")
    task_index, task_id, episode_index, episode_seed = expected
    require(
        (row["task_index"], row["task_id"], row["episode_index"], row["episode_seed"])
        == expected
        and all(type(row[name]) is int for name in ("task_index", "task_id", "episode_index", "episode_seed")),
        "final evaluation row identity/order differs",
    )
    require(type(row["success"]) is bool, "final evaluation success is not boolean")
    for name in (
        "steps",
        "replans",
        "nodes",
        "no_action_plans",
        "guard_plans",
        "guard_rejections",
        "guard_candidate_count",
        "guard_accepted_count",
    ):
        require(type(row[name]) is int and row[name] >= 0, f"final evaluation {name} is invalid")
    for name in (
        "final_goal_distance",
        "best_goal_distance",
        "initial_goal_distance",
        "displacement",
        "path_length",
        "action_magnitude",
        "planning_wall_clock_s",
    ):
        require(
            isinstance(row[name], (int, float))
            and not isinstance(row[name], bool)
            and math.isfinite(float(row[name]))
            and float(row[name]) >= 0.0,
            f"final evaluation {name} is invalid",
        )
    for name in ("chunk_lengths", "selected_depths"):
        _validate_number_list(row[name], f"final evaluation {name}", integral=True)
    for name in (
        "guard_best_predicted_improvements",
        "guard_selected_predicted_improvements",
    ):
        _validate_number_list(row[name], f"final evaluation {name}")
    require(isinstance(row["trajectory"], list), "final evaluation trajectory is invalid")
    _validate_finite_numeric_tree(row["trajectory"], "final evaluation trajectory")
    require(isinstance(row["progress"], Mapping), "final evaluation progress is invalid")
    # Domain progress is defined as a flat dict[str, float].  Rejecting nested
    # structures is stricter than merely walking them and matches evaluate()'s mean.
    _finite_mapping(row["progress"], "final evaluation progress")
    return dict(row)


def _expected_episode_keys(manifest: Mapping[str, Any]) -> list[tuple[int, int, int, int]]:
    task_ids = [int(item) for item in manifest["scientific_contract"]["task_ids"]]
    episodes = int(manifest["scientific_contract"]["final_episodes_per_task"])
    base = int(manifest["scientific_contract"]["evaluation_seed"])
    result = [
        (task_index, task_id, episode_index, base + 1000 * task_index + episode_index)
        for task_index, task_id in enumerate(task_ids)
        for episode_index in range(episodes)
    ]
    require(len(result) == 25, "terminal episode contract is not exactly 25")
    return result


def _run_layout(context: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    root = _absolute_path(context["run_root"], "declared scientific run root")
    relative = _safe_relative(context["run_relative"], "scientific run relative path")
    run_directory = _absolute_path(context["run_directory"], "scientific run directory")
    require(root / relative == run_directory, "scientific run layout binding differs")
    return root, relative, run_directory


def _open_run_directory(context: Mapping[str, Any], *, create: bool = False) -> int:
    root, relative, _ = _run_layout(context)
    root_fd = _open_absolute_directory(
        root,
        "declared scientific run root",
        create=create,
    )
    try:
        return _open_relative_directory(
            root_fd,
            relative,
            "scientific run directory",
            create=create,
        )
    finally:
        os.close(root_fd)


def _scientific_run_exists(context: Mapping[str, Any]) -> bool:
    root, relative, _ = _run_layout(context)
    root_fd = _open_absolute_directory_optional(root, "declared scientific run root")
    if root_fd is None:
        return False
    try:
        run_fd = _open_relative_directory_optional(
            root_fd,
            relative,
            "scientific run directory",
        )
    finally:
        os.close(root_fd)
    if run_fd is None:
        return False
    os.close(run_fd)
    return True


def _entry_stat(directory_fd: int, name: str, label: str) -> os.stat_result | None:
    require(name not in {"", ".", ".."} and "/" not in name, f"{label} leaf is invalid")
    try:
        info = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise LifecycleError(f"cannot lstat {label}: {exc}") from exc
    require(stat.S_ISREG(info.st_mode), f"{label} is not a regular nonsymlink file")
    return info


def _open_run_regular(
    context: Mapping[str, Any],
    relative: str | Path,
    label: str,
) -> tuple[int, os.stat_result]:
    artifact = _safe_relative(relative, label)
    run_fd = _open_run_directory(context)
    parent_fd = run_fd
    try:
        if artifact.parent != Path("."):
            parent_fd = _open_relative_directory(
                run_fd,
                artifact.parent,
                f"parent of {label}",
            )
        try:
            descriptor = os.open(artifact.name, _FILE_FLAGS, dir_fd=parent_fd)
        except OSError as exc:
            raise LifecycleError(f"{label} is unavailable or symlinked: {exc}") from exc
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            os.close(descriptor)
            raise LifecycleError(f"{label} is not a regular nonsymlink file")
        return descriptor, info
    finally:
        if parent_fd != run_fd:
            os.close(parent_fd)
        os.close(run_fd)


def _read_run_json_artifact(
    context: Mapping[str, Any],
    relative: str | Path,
    label: str,
) -> tuple[dict[str, Any], str, dict[str, int]]:
    descriptor, path_info = _open_run_regular(context, relative, label)
    try:
        payload, opened = _read_fd_stable(descriptor, label)
    finally:
        os.close(descriptor)
    require(_stat_identity(path_info) == _stat_identity(opened), f"{label} path/open identity differs")
    return (
        _decode_json_bytes(payload, Path(str(relative))),
        hashlib.sha256(payload).hexdigest(),
        _stat_identity(opened),
    )


def _optional_run_artifact_kind(
    context: Mapping[str, Any], relative: str | Path, label: str
) -> bool:
    artifact = _safe_relative(relative, label)
    run_fd = _open_run_directory(context)
    parent_fd = run_fd
    try:
        if artifact.parent != Path("."):
            try:
                parent_fd = _open_relative_directory(
                    run_fd,
                    artifact.parent,
                    f"parent of {label}",
                )
            except LifecycleError as exc:
                if "unavailable" in str(exc):
                    return False
                raise
        return _entry_stat(parent_fd, artifact.name, label) is not None
    finally:
        if parent_fd != run_fd:
            os.close(parent_fd)
        os.close(run_fd)


def _validate_final_progress_value(
    value: Mapping[str, Any],
    artifact_sha256: str,
    *,
    expected_identity_sha256: str,
    expected_seed_table_sha256: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    value = dict(value)
    expected_keys = _expected_episode_keys(manifest)
    require(value.get("schema_version") == 1, "final-progress schema differs")
    require(value.get("objective_version") == OBJECTIVE, "final-progress objective differs")
    require(value.get("status") in {"in_progress", "complete"}, "final-progress status differs")
    expected_fields = {
        "schema_version",
        "objective_version",
        "status",
        "identity_sha256",
        "seed_table_sha256",
        "task_ids",
        "episodes_per_task",
        "completed_results",
        "generator_state",
    }
    if value.get("status") == "complete":
        expected_fields.add("metrics")
    require(set(value) == expected_fields, "final-progress fields differ")
    require(value.get("identity_sha256") == expected_identity_sha256, "final-progress identity differs")
    require(value.get("seed_table_sha256") == expected_seed_table_sha256, "final-progress seed table differs")
    task_ids = [key[1] for key in expected_keys[::5]]
    require(value.get("task_ids") == task_ids, "final-progress task IDs differ")
    require(value.get("episodes_per_task") == 5, "final-progress episode count differs")
    rows = value.get("completed_results")
    require(isinstance(rows, list) and 1 <= len(rows) <= 25, "final-progress rows differ")
    normalized = [
        validate_episode_row(row, expected_keys[index]) for index, row in enumerate(rows)
    ]
    generator = value.get("generator_state")
    require(
        isinstance(generator, list)
        and generator
        and all(type(item) is int and 0 <= item <= 255 for item in generator),
        "final-progress generator state differs",
    )
    metrics = None
    if value["status"] == "complete":
        require(len(rows) == 25, "complete final-progress does not have 25 rows")
        metrics = _finite_mapping(value.get("metrics"), "final-progress metrics")
    else:
        require("metrics" not in value, "in-progress final evaluation contains terminal metrics")
    return {
        "status": value["status"],
        "row_count": len(rows),
        "rows": normalized,
        "rows_sha256": stable_hash(normalized),
        "metrics": metrics,
        "sha256": artifact_sha256,
        "value": value,
    }


def validate_final_progress(
    path: Path,
    *,
    expected_identity_sha256: str,
    expected_seed_table_sha256: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not lexical_exists(path):
        return None
    value, digest, _ = read_json_artifact(path, "final evaluation progress")
    return _validate_final_progress_value(
        value,
        digest,
        expected_identity_sha256=expected_identity_sha256,
        expected_seed_table_sha256=expected_seed_table_sha256,
        manifest=manifest,
    )


def validate_checkpoint_payload(
    payload: Mapping[str, Any],
    context: Mapping[str, Any],
    *,
    validate_shared: bool = True,
) -> dict[str, Any]:
    config = context["resolved_row"]["resolved_config"]
    identity = expected_run_identity(context)
    if validate_shared:
        from treewm.utils import checkpoint as checkpoint_utils

        checkpoint_utils.OBJECTIVES_REQUIRING_POST_UPDATE_CADENCE = frozenset(
            {*checkpoint_utils.OBJECTIVES_REQUIRING_POST_UPDATE_CADENCE, OBJECTIVE}
        )
        checkpoint_utils.validate_exact_resume_payload(
            payload,
            expected_identity=identity,
            expected_world_size=1,
            require_cuda_rng=True,
        )
    require(payload.get("run_identity") == identity, "checkpoint run identity differs")
    require(payload.get("identity_sha256") == stable_hash(identity), "checkpoint identity hash differs")
    require(payload.get("config") == config, "checkpoint resolved config differs")
    completed = payload.get("completed_updates")
    require(type(completed) is int and 0 <= completed <= TOTAL_UPDATES, "checkpoint update differs")
    require(payload.get("step") == completed and payload.get("next_step") == completed, "checkpoint step differs")
    rank_states = payload.get("rank_states")
    require(isinstance(rank_states, list) and len(rank_states) == 1, "checkpoint rank coverage differs")
    rank = rank_states[0]
    require(isinstance(rank, Mapping) and rank.get("rank") == 0, "checkpoint rank0 state differs")
    validate_metric_tracker_state(rank.get("metric_tracker"), completed)
    validate_loader_state(
        rank.get("loader"),
        completed,
        train_anchor_count=int(identity["train_anchor_count"]),
        batch_size=int(config["train"]["batch_size"]),
        grad_accum=int(config["train"]["grad_accum"]),
    )
    require(payload.get("epoch") == rank["loader"]["epoch"], "checkpoint epoch/loader differ")
    cadence = validate_post_update_cadence(payload, completed, config)
    phase = payload.get("phase")
    require(phase in {"train", "final_eval"}, "checkpoint phase differs")
    final_eval = payload.get("final_eval")
    if final_eval is not None:
        final_eval = _finite_mapping(final_eval, "checkpoint final evaluation")
    progress = None
    if _optional_run_artifact_kind(
        context,
        "final_eval_progress.json",
        "final evaluation progress",
    ):
        progress_value, progress_sha256, _ = _read_run_json_artifact(
            context,
            "final_eval_progress.json",
            "final evaluation progress",
        )
        progress = _validate_final_progress_value(
            progress_value,
            progress_sha256,
            expected_identity_sha256=stable_hash(identity),
            expected_seed_table_sha256=identity["final_seed_table_sha256"],
            manifest=context["manifest"],
        )
    pending = payload.get("pending_eval_step")
    reason = payload.get("reason")
    require(isinstance(reason, str) and reason, "checkpoint reason differs")
    if phase == "train":
        require(progress is None, "training checkpoint has stale final-progress artifact")
        require(final_eval is None, "training checkpoint has stale final-evaluation metrics")
        kind = "train_at_target" if completed == TOTAL_UPDATES else "train"
    else:
        require(completed == TOTAL_UPDATES, "final-eval checkpoint is not at 25k")
        require(
            cadence["completed_update"] == TOTAL_UPDATES
            and cadence["replay_action"] is None,
            "final-eval checkpoint cadence is incomplete",
        )
        if pending == TOTAL_UPDATES:
            require(
                reason == "final-evaluation-pending" or reason.startswith("graceful-stop:"),
                "pending final-evaluation checkpoint reason differs",
            )
            if progress is None or progress["status"] == "in_progress":
                require(final_eval is None, "partial final evaluation has terminal metrics")
            else:
                require(final_eval == progress["metrics"], "pending exact-25 metrics differ")
            kind = "final_pending"
        else:
            require(pending is None, "final-eval pending step differs")
            require(reason == "final-evaluation-complete", "terminal checkpoint reason differs")
            require(progress is not None and progress["status"] == "complete", "terminal checkpoint progress incomplete")
            require(final_eval == progress["metrics"], "terminal checkpoint metrics differ")
            kind = "final_checkpoint_complete"
    return {
        "kind": kind,
        "completed_updates": completed,
        "phase": phase,
        "pending_eval_step": pending,
        "reason": reason,
        "cadence": cadence,
        "progress": progress,
        "final_eval": final_eval,
        "identity": identity,
        "identity_sha256": stable_hash(identity),
        "config": copy.deepcopy(config),
    }


def resolve_checkpoint(context: Mapping[str, Any]) -> dict[str, Any]:
    """Hash and deserialize the exact same rooted, nonsymlink checkpoint inode."""
    descriptor, opened = _open_run_regular(
        context,
        "checkpoints/latest.pt",
        "latest checkpoint",
    )
    try:
        import torch

        before = os.fstat(descriptor)
        require(_stat_identity(opened) == _stat_identity(before), "latest checkpoint open identity differs")
        digest = hashlib.sha256()
        offset = 0
        while block := os.pread(descriptor, 16 * 1024 * 1024, offset):
            digest.update(block)
            offset += len(block)
        require(offset == before.st_size, "latest checkpoint short read")
        with os.fdopen(os.dup(descriptor), "rb", closefd=True) as handle:
            payload = torch.load(handle, map_location="cpu", weights_only=False)
        after = os.fstat(descriptor)
        require(_stat_identity(before) == _stat_identity(after), "latest checkpoint changed while open")
    except Exception as exc:
        raise LifecycleError(f"cannot load latest checkpoint: {exc}") from exc
    finally:
        os.close(descriptor)
    require(isinstance(payload, Mapping), "checkpoint payload is not a mapping")
    state = validate_checkpoint_payload(payload, context)
    state["checkpoint_sha256"] = digest.hexdigest()
    state["checkpoint_file_identity"] = _stat_identity(after)
    state["checkpoint_path"] = str(context["run_directory"] / Path("checkpoints/latest.pt"))
    return state


def load_and_validate_checkpoint(path: Path, context: Mapping[str, Any]) -> dict[str, Any]:
    expected = _run_layout(context)[2] / "checkpoints/latest.pt"
    require(_absolute_path(path, "latest checkpoint path") == expected, "latest checkpoint path binding differs")
    return resolve_checkpoint(context)


def _close(left: object, right: float, tolerance: float = 1e-6) -> bool:
    return (
        isinstance(left, (int, float))
        and not isinstance(left, bool)
        and math.isfinite(float(left))
        and abs(float(left) - right) <= tolerance * max(1.0, abs(right))
    )


def validate_completion_metrics(metrics: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> None:
    successes = float(sum(bool(row["success"]) for row in rows))
    progress = sum(
        (float(row["initial_goal_distance"]) - float(row["final_goal_distance"]))
        / max(float(row["initial_goal_distance"]), 1e-6)
        for row in rows
    ) / 25.0
    require(_close(metrics.get("eval/num_episodes"), 25.0), "terminal metric episode count differs")
    require(_close(metrics.get("eval/successes"), successes), "terminal success count differs")
    require(_close(metrics.get("eval/success_rate"), successes / 25.0), "terminal success rate differs")
    require(_close(metrics.get("eval/distance_reduction_frac"), progress), "terminal progress differs")
    for task_index, task_id in enumerate((1, 2, 3, 4, 5)):
        task_rows = rows[task_index * 5 : (task_index + 1) * 5]
        task_success = float(sum(bool(row["success"]) for row in task_rows))
        require(_close(metrics.get(f"eval/task{task_id}/num_episodes"), 5.0), "terminal task episode count differs")
        require(_close(metrics.get(f"eval/task{task_id}/successes"), task_success), "terminal task successes differ")
        require(_close(metrics.get(f"eval/task{task_id}/success_rate"), task_success / 5.0), "terminal task success rate differs")


def validate_complete_run(
    run_dir: Path, checkpoint_state: Mapping[str, Any], context: Mapping[str, Any]
) -> dict[str, Any]:
    require(checkpoint_state["kind"] == "final_checkpoint_complete", "completion lacks terminal checkpoint")
    require(_absolute_path(run_dir, "completion run directory") == _run_layout(context)[2], "completion run path differs")
    completion, completion_sha256, _ = _read_run_json_artifact(
        context,
        "COMPLETED.json",
        "trainer completion",
    )
    require(set(completion) == COMPLETION_FIELDS, "completion fields differ")
    identity = checkpoint_state["identity"]
    config = checkpoint_state["config"]
    progress = checkpoint_state["progress"]
    expected = {
        "schema_version": 1,
        "objective_version": OBJECTIVE,
        "status": "complete",
        "run_identity": identity,
        "identity_sha256": checkpoint_state["identity_sha256"],
        "evaluation_seed_tables": "evaluation_seed_tables.json",
        "evaluation_seed_tables_sha256": identity["evaluation_seed_tables_sha256"],
        "final_seed_table_sha256": identity["final_seed_table_sha256"],
        "protocol_sha256": identity["protocol_sha256"],
        "code_sha256": identity["code_sha256"],
        "runtime_sha256": identity["runtime_sha256"],
        "runtime": context["source_contract"]["runtime"]["software"],
        "data_manifest_sha256": identity["data_manifest_sha256"],
        "calibration_sha256": identity["calibration_sha256"],
        "future_recipe_sha256": identity["future_recipe_sha256"],
        "recipe_code_sha256": identity["recipe_code_sha256"],
        "recipe_runtime_sha256": identity["recipe_runtime_sha256"],
        "arm": identity["arm"],
        "model_class": identity["model_class"],
        "scorer": identity["scorer"],
        "setting": identity["setting"],
        "env_name": identity["env_name"],
        "dataset_kind": identity["dataset_kind"],
        "source_name": identity["source_name"],
        "dataset_dir": str(config["env"]["dataset_dir"]),
        "seed": identity["seed"],
        "wandb_id": identity["wandb_id"],
        "wandb_group": identity["wandb_group"],
        "completed_updates": TOTAL_UPDATES,
        "scheduler_total_steps": identity["scheduler_total_steps"],
        "final_eval_step": TOTAL_UPDATES,
        "task_ids": identity["task_ids"],
        "episodes_per_task": identity["final_episodes_per_task"],
        "node_budget": identity["node_budget"],
        "branch_factor": identity["branch_factor"],
        "gradient_checkpointing": identity["gradient_checkpointing"],
        "future_set_cache": identity["future_set_cache"],
        "shared_cache": identity["shared_cache"],
        "retrieval_enabled": identity["retrieval_enabled"],
        "retrieval_num_keys": identity["retrieval_num_keys"],
        "checkpoint": "checkpoints/latest.pt",
        "final_eval_progress": "final_eval_progress.json",
    }
    for key, value in expected.items():
        require(completion.get(key) == value, f"completion {key} differs")
    metrics = _finite_mapping(completion.get("final_evaluation"), "completion final evaluation")
    require(progress is not None and progress["status"] == "complete", "completion progress is incomplete")
    require(metrics == progress["metrics"] == checkpoint_state["final_eval"], "terminal metrics disagree")
    validate_completion_metrics(metrics, progress["rows"])

    progress_value, progress_sha256, _ = _read_run_json_artifact(
        context,
        "final_eval_progress.json",
        "terminal final evaluation progress",
    )
    require(
        progress_value == progress["value"] and progress_sha256 == progress["sha256"],
        "terminal final-progress changed after checkpoint validation",
    )
    seed_tables, _, _ = _read_run_json_artifact(
        context,
        "evaluation_seed_tables.json",
        "evaluation seed tables",
    )
    expected_seed_tables = expected_evaluation_seed_tables(context)
    require(seed_tables == expected_seed_tables, "evaluation seed tables differ from reconstruction")
    require(seed_tables.get("sha256") == identity["evaluation_seed_tables_sha256"], "evaluation seed tables differ")
    require(completion.get("evaluation_seed_tables_sha256") == seed_tables["sha256"], "completion seed-table hash differs")
    require(completion.get("final_seed_table_sha256") == identity["final_seed_table_sha256"], "completion final seed table differs")
    return {
        "status": "complete",
        "completed_updates": TOTAL_UPDATES,
        "checkpoint_sha256": checkpoint_state["checkpoint_sha256"],
        "completion_sha256": completion_sha256,
        "final_eval_progress_sha256": progress["sha256"],
        "completed_results_sha256": progress["rows_sha256"],
        "identity_sha256": checkpoint_state["identity_sha256"],
        "final_metrics": metrics,
    }


def inspect_run(context: Mapping[str, Any]) -> dict[str, Any]:
    run_dir = _run_layout(context)[2]
    if not _scientific_run_exists(context):
        return {"kind": "absent", "run_dir": str(run_dir)}
    latest = run_dir / "checkpoints/latest.pt"
    latest_exists = _optional_run_artifact_kind(context, "checkpoints/latest.pt", "latest checkpoint")
    completion_exists = _optional_run_artifact_kind(context, "COMPLETED.json", "trainer completion")
    progress_exists = _optional_run_artifact_kind(
        context,
        "final_eval_progress.json",
        "final evaluation progress",
    )
    if not latest_exists:
        require(not completion_exists, "completion exists without latest checkpoint")
        require(not progress_exists, "final progress exists without checkpoint")
        return {"kind": "claimed_empty", "run_dir": str(run_dir)}
    state = load_and_validate_checkpoint(latest, context)
    if completion_exists:
        complete = validate_complete_run(run_dir, state, context)
        return {**state, "kind": "complete", "complete": complete, "run_dir": str(run_dir)}
    return {**state, "run_dir": str(run_dir)}


def _artifact_base(args: argparse.Namespace, context: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": args.submission_sha256,
        "launch_sha256": context["launch"]["launch_sha256"],
        "cell_index": args.cell_index,
        "restart_count": args.restart_count,
        "array_job_id": str(args.array_job_id),
        "array_task_id": int(args.array_task_id),
    }


def signal_request_path(task_root: Path, generation_root: Path, name: str) -> Path:
    return generation_root / name


def record_signal_request(
    *,
    submission_root: Path,
    submission_sha256: str,
    cell_index: int,
    restart_count: int,
    array_job_id: str,
    array_task_id: int,
    signal_name: str,
    bootstrap_contract: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    submission_root = ensure_directory(submission_root, "submission root")
    validate_submission_contract(
        submission_root,
        submission_sha256,
        contract=bootstrap_contract,
    )
    require(signal_name in {"USR1", "TERM"}, "unsupported lifecycle signal")
    task_root = ensure_contained_directory(
        task_root_for(submission_root, cell_index),
        submission_root,
        "task state root",
        create=True,
    )
    generation = ensure_contained_directory(
        generation_root_for(task_root, restart_count),
        task_root,
        "generation state root",
        create=True,
    )
    value = {
        "schema_version": 1,
        "status": "signal_requested",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "cell_index": cell_index,
        "restart_count": restart_count,
        "array_job_id": str(array_job_id),
        "array_task_id": int(array_task_id),
        "signal": signal_name,
    }
    require(set(value) == SIGNAL_REQUEST_FIELDS, "signal-request marker fields differ")
    name = USR1_REQUEST_NAME if signal_name == "USR1" else TERM_REQUEST_NAME
    seal_json(generation / name, value)
    if signal_name == "TERM":
        seal_json(task_root / TASK_CANCEL_NAME, {**value, "status": "task_cancellation_requested"})
    return value


def cancellation_requested(submission_root: Path, task_root: Path) -> bool:
    return lexical_exists(submission_root / GLOBAL_CANCEL_NAME) or lexical_exists(
        task_root / TASK_CANCEL_NAME
    )


def requeue_requested(generation: Path) -> bool:
    return lexical_exists(generation / USR1_REQUEST_NAME)


def validate_ready_checkpoint_binding(
    ready: Mapping[str, Any],
    checkpoint_state: Mapping[str, Any],
    label: str,
) -> dict[str, int]:
    """Bind every requeue claim to the reopened checkpoint and progress bytes."""
    checkpoint_identity = validate_file_identity(
        ready.get("checkpoint_file_identity"),
        f"{label} checkpoint identity",
    )
    expected = {
        "trainer_exit_code": GRACEFUL_EXIT_CODE,
        "checkpoint_kind": checkpoint_state.get("kind"),
        "completed_updates": checkpoint_state.get("completed_updates"),
        "phase": checkpoint_state.get("phase"),
        "pending_eval_step": checkpoint_state.get("pending_eval_step"),
        "checkpoint_sha256": checkpoint_state.get("checkpoint_sha256"),
        "checkpoint_file_identity": checkpoint_state.get("checkpoint_file_identity"),
        "final_eval_progress_sha256": (
            checkpoint_state.get("progress") or {}
        ).get("sha256"),
    }
    for key, value in expected.items():
        require(ready.get(key) == value, f"{label} {key} differs from checkpoint")
    return checkpoint_identity


def _verify_previous_lineage(
    args: argparse.Namespace,
    task_root: Path,
    context: Mapping[str, Any],
) -> None:
    restart_count = int(args.restart_count)
    require(restart_count > 0, "previous lineage requested for generation zero")
    previous = generation_root_for(task_root, restart_count - 1)
    ensure_contained_directory(previous, task_root, "previous generation root")
    ready_path = previous / REQUEUE_READY_NAME
    calling_path = previous / REQUEUE_CALLING_NAME
    ready, ready_sha256, _ = read_json_artifact(
        ready_path,
        "previous requeue READY",
        required_mode=0o444,
    )
    calling, _, _ = read_json_artifact(
        calling_path,
        "previous requeue CALLING",
        required_mode=0o444,
    )
    require(set(ready) == REQUEUE_READY_FIELDS, "previous READY fields differ")
    require(set(calling) == REQUEUE_CALLING_FIELDS, "previous CALLING fields differ")
    common = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": args.submission_sha256,
        "launch_sha256": context["launch"]["launch_sha256"],
        "cell_index": args.cell_index,
        "restart_count": restart_count - 1,
        "array_job_id": str(args.array_job_id),
        "array_task_id": int(args.array_task_id),
    }
    for key, value in common.items():
        require(ready.get(key) == value, f"previous READY {key} differs")
        require(calling.get(key) == value, f"previous CALLING {key} differs")
    require(ready.get("status") == "requeue_ready", "previous generation is not requeue-ready")
    require(calling.get("status") == "scontrol_requeue_calling", "previous generation did not call requeue")
    require(
        calling.get("requeue_target") == f"{args.array_job_id}_{args.array_task_id}",
        "previous requeue target differs",
    )
    require(calling.get("requeue_ready_sha256") == ready_sha256, "previous CALLING/READY differ")
    require(
        calling.get("checkpoint_sha256") == ready.get("checkpoint_sha256")
        and sha256_string(ready.get("checkpoint_sha256")),
        "previous checkpoint lineage hash differs",
    )
    current = resolve_checkpoint(context)
    ready_identity = validate_ready_checkpoint_binding(ready, current, "previous READY")
    require(
        validate_file_identity(
            calling.get("checkpoint_file_identity"),
            "previous CALLING checkpoint identity",
        )
        == ready_identity,
        "previous CALLING/READY checkpoint identity differs",
    )
    require(
        current["checkpoint_sha256"] == ready.get("checkpoint_sha256")
        and current["checkpoint_file_identity"] == ready_identity,
        "checkpoint advanced/swapped after READY",
    )


class SignalRelay:
    def __init__(self, args: argparse.Namespace, task_root: Path, generation: Path) -> None:
        self.args = args
        self.task_root = task_root
        self.generation = generation
        self.child_pid: int | None = None
        self.child_ready = False
        self.forwarded: int | None = None
        self.pending_signal: int | None = None

    def install(self) -> None:
        signal.signal(signal.SIGUSR1, self._handle)
        signal.signal(signal.SIGTERM, self._handle)

    def _handle(self, signum: int, _frame: object) -> None:
        # Python invokes this on the main thread, possibly in the middle of JSON,
        # import, or checkpoint validation.  Assignment is the only safe operation
        # here; durable I/O occurs at explicit service points below. TERM wins.
        if self.pending_signal is None or signum == signal.SIGTERM:
            self.pending_signal = int(signum)

    def service_pending(self) -> None:
        signum = self.pending_signal
        if signum is not None:
            self.pending_signal = None
            name = "TERM" if signum == signal.SIGTERM else "USR1"
            try:
                record_signal_request(
                    submission_root=Path(self.args.submission_root),
                    submission_sha256=self.args.submission_sha256,
                    cell_index=self.args.cell_index,
                    restart_count=self.args.restart_count,
                    array_job_id=self.args.array_job_id,
                    array_task_id=self.args.array_task_id,
                    signal_name=name,
                    bootstrap_contract=getattr(self.args, "_bootstrap_contract", None),
                )
            except BaseException as exc:
                # Keep the request pending so a later service boundary retries.  The
                # batch shell also records before forwarding, giving two safe paths.
                self.pending_signal = signum
                print(f"Exp23 signal latch failed: {exc}", file=sys.stderr, flush=True)
        self.forward_pending()

    def forward_pending(self) -> None:
        if self.child_pid is None or not self.child_ready:
            return
        desired = None
        if cancellation_requested(Path(self.args.submission_root), self.task_root):
            desired = signal.SIGTERM
        elif requeue_requested(self.generation):
            desired = signal.SIGUSR1
        if desired is None or self.forwarded == desired:
            return
        try:
            os.kill(self.child_pid, desired)
            self.forwarded = desired
        except ProcessLookupError:
            pass


def controlled_child_environment(context: Mapping[str, Any]) -> dict[str, str]:
    require(STOP_ENVIRONMENT not in os.environ, f"inherited {STOP_ENVIRONMENT} is forbidden")
    launch_environment = {str(key): str(value) for key, value in context["launch"]["environment"].items()}
    require(STOP_ENVIRONMENT not in launch_environment, "sealed launch contains staged stop")
    for name, value in HEADLESS_RUNTIME_ENVIRONMENT.items():
        require(
            launch_environment.get(name) == value,
            f"sealed launch headless runtime binding differs: {name}",
        )
    for name in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
        require(name not in launch_environment, f"sealed launch contains distributed variable {name}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    require(visible and "," not in visible, "exactly one CUDA_VISIBLE_DEVICES entry is required")
    python = Path(context["launch"]["argv"][0])
    require(python == PINNED_PYTHON, "trainer child interpreter is not pinned")
    home = pwd.getpwuid(os.getuid()).pw_dir
    environment = {
        **launch_environment,
        "HOME": home,
        "PATH": f"{python.parent}:/usr/bin:/bin",
        "PYTHONHASHSEED": str(context["cell"].seed),
        "CUDA_VISIBLE_DEVICES": visible,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "TMPDIR": "/tmp",
    }
    for name in (
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_RESTART_COUNT",
        "SLURM_JOB_GPUS",
    ):
        if name in os.environ:
            environment[name] = os.environ[name]
    return environment


def _spawn_trainer(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    relay: SignalRelay,
) -> int:
    read_fd, write_fd = os.pipe()
    entry = context["package"] / "train_entry.py"
    require_regular_nonsymlink(entry, "Exp23 train entry")
    argv = [
        context["launch"]["argv"][0],
        "-P",
        "-S",
        "-B",
        str(entry),
        "--launch",
        str(context["launch_path"]),
        "--submission-root",
        str(context["submission_root"]),
        "--submission-sha256",
        str(context["submission_sha256"]),
        "--ready-fd",
        str(write_fd),
        "--",
        *context["launch"]["argv"][2:],
    ]
    blocked = {signal.SIGUSR1, signal.SIGTERM}
    previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, blocked)
    try:
        # The blocked mask survives exec. train_entry installs both handlers before
        # unblocking, so even a cgroup-wide Slurm TERM cannot hit default disposition.
        try:
            process = subprocess.Popen(
                argv,
                cwd=context["snapshot_root"],
                env=controlled_child_environment(context),
                pass_fds=(write_fd,),
            )
        except BaseException:
            os.close(read_fd)
            os.close(write_fd)
            raise
    finally:
        signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)
    os.close(write_fd)
    os.set_blocking(read_fd, False)
    relay.child_pid = process.pid
    try:
        while True:
            if not relay.child_ready:
                readable, _, _ = select.select([read_fd], [], [], 0.1)
                if readable:
                    value = os.read(read_fd, 1)
                    require(value == b"R", "trainer signal-ready handshake differs")
                    relay.child_ready = True
            relay.service_pending()
            status = process.poll()
            if status is not None:
                require(relay.child_ready, "trainer exited before signal-ready handshake")
                return int(status)
    finally:
        os.close(read_fd)


def _claim_fresh_run(context: Mapping[str, Any]) -> None:
    run_root, relative, _ = _run_layout(context)
    root_fd = _open_absolute_directory(
        run_root,
        "fresh declared run root",
        create=True,
    )
    parent_fd = root_fd
    try:
        if relative.parent != Path("."):
            parent_fd = _open_relative_directory(
                root_fd,
                relative.parent,
                "fresh run parent",
                create=True,
            )
        try:
            existing = os.stat(relative.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            existing = None
        require(existing is None, "generation zero run directory already exists")
        try:
            os.mkdir(relative.name, 0o700, dir_fd=parent_fd)
        except FileExistsError as exc:
            raise LifecycleError("fresh run claim raced with an existing path") from exc
        run_fd = os.open(relative.name, _DIRECTORY_FLAGS, dir_fd=parent_fd)
        try:
            os.fsync(run_fd)
            os.fsync(parent_fd)
        finally:
            os.close(run_fd)
    finally:
        if parent_fd != root_fd:
            os.close(parent_fd)
        os.close(root_fd)


def _complete_marker(args: argparse.Namespace, context: Mapping[str, Any], complete: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **_artifact_base(args, context),
        **dict(complete),
        # validate_complete_run's internal status is "complete"; the lifecycle
        # receipt has a distinct status and must win the merge.
        "status": "worker_complete",
    }


def _execute_generation(
    args: argparse.Namespace,
    context: Mapping[str, Any],
    task_root: Path,
    generation: Path,
    relay: SignalRelay,
) -> int:
    submission = context["submission_root"]
    require(args.array_task_id == args.cell_index, "Slurm array task/cell mapping differs")
    base = _artifact_base(args, context)
    seal_json(
        task_root / TASK_LAUNCH_NAME,
        {
            "schema_version": 1,
            "status": "launch_sealed",
            "campaign_id": CAMPAIGN_ID,
            "submission_sha256": args.submission_sha256,
            "launch_sha256": context["launch"]["launch_sha256"],
            "cell_index": args.cell_index,
            "array_job_id": str(args.array_job_id),
            "array_task_id": int(args.array_task_id),
            "launch": context["launch"],
        },
    )

    if cancellation_requested(submission, task_root):
        seal_json(generation / GENERATION_CANCELLED_NAME, {**base, "status": "cancelled_before_launch"})
        return CANCELLED_EXIT_CODE

    if args.restart_count == 0:
        _claim_fresh_run(context)
        input_state = {"kind": "fresh", "checkpoint_sha256": None}
    else:
        _verify_previous_lineage(args, task_root, context)
        input_state = inspect_run(context)
        require(input_state["kind"] not in {"absent", "claimed_empty"}, "requeue has no exact checkpoint")

    seal_json(
        generation / GENERATION_START_NAME,
        {
            **base,
            "status": "generation_started",
            "input_kind": input_state["kind"],
            "input_checkpoint_sha256": input_state.get("checkpoint_sha256"),
        },
    )
    if input_state["kind"] == "complete":
        marker = _complete_marker(args, context, input_state["complete"])
        seal_json(generation / GENERATION_COMPLETE_NAME, marker)
        seal_json(task_root / TASK_COMPLETE_NAME, marker)
        return 0

    status = _spawn_trainer(args, context, relay)
    relay.service_pending()
    output_state = inspect_run(context)
    relay.service_pending()
    if output_state["kind"] == "complete":
        marker = _complete_marker(args, context, output_state["complete"])
        seal_json(generation / GENERATION_COMPLETE_NAME, marker)
        seal_json(task_root / TASK_COMPLETE_NAME, marker)
        return 0

    if cancellation_requested(submission, task_root):
        cancellation = {
            **base,
            "status": "worker_cancelled",
            "trainer_exit_code": status,
            "checkpoint_kind": output_state.get("kind"),
            "completed_updates": output_state.get("completed_updates"),
            "checkpoint_sha256": output_state.get("checkpoint_sha256"),
        }
        seal_json(generation / GENERATION_CANCELLED_NAME, cancellation)
        return CANCELLED_EXIT_CODE

    if requeue_requested(generation):
        require(status == GRACEFUL_EXIT_CODE, "USR1 trainer exit is not graceful code 75")
        require(output_state["kind"] not in {"absent", "claimed_empty"}, "USR1 produced no resumable checkpoint")
        require(
            str(output_state.get("reason", "")).startswith("graceful-stop:"),
            "USR1 checkpoint lacks graceful-stop reason",
        )
        ready = {
            **base,
            "status": "requeue_ready",
            "trainer_exit_code": status,
            "checkpoint_kind": output_state["kind"],
            "completed_updates": output_state["completed_updates"],
            "phase": output_state["phase"],
            "pending_eval_step": output_state["pending_eval_step"],
            "checkpoint_sha256": output_state["checkpoint_sha256"],
            "checkpoint_file_identity": output_state["checkpoint_file_identity"],
            "final_eval_progress_sha256": (
                output_state.get("progress") or {}
            ).get("sha256"),
        }
        require(set(ready) == REQUEUE_READY_FIELDS, "requeue READY fields differ")
        seal_json(generation / REQUEUE_READY_NAME, ready)
        return GRACEFUL_EXIT_CODE

    raise LifecycleError(
        f"trainer exited {status} without a complete run or durable lifecycle request"
    )


def run_worker(args: argparse.Namespace) -> int:
    # Reject the historical staged-stop mechanism before creating any task/run state.
    require(STOP_ENVIRONMENT not in os.environ, f"inherited {STOP_ENVIRONMENT} is forbidden")
    snapshot = ensure_directory(args.snapshot_root, "snapshot root")
    submission = ensure_directory(args.submission_root, "submission root")
    bootstrap_contract = getattr(args, "_bootstrap_contract", None)
    require(isinstance(bootstrap_contract, Mapping), "run command lacks verified bootstrap")

    # Install an in-memory handler before potentially expensive snapshot/recipe
    # validation.  No readiness marker or durable task state is published until the
    # complete context has validated; train.slurm buffers its durable signal request.
    provisional_task = task_root_for(submission, args.cell_index)
    provisional_generation = generation_root_for(provisional_task, args.restart_count)
    relay = SignalRelay(args, provisional_task, provisional_generation)
    relay.install()
    context = load_launch_context(
        snapshot_root=snapshot,
        submission_root=submission,
        submission_sha256=args.submission_sha256,
        cell_index=args.cell_index,
        bootstrap_contract=bootstrap_contract,
    )
    task_root = ensure_contained_directory(
        provisional_task,
        submission,
        "task state root",
        create=True,
    )
    generation = ensure_contained_directory(
        provisional_generation,
        task_root,
        "generation state root",
        create=True,
    )
    relay.task_root = task_root
    relay.generation = generation
    relay.service_pending()
    base = _artifact_base(args, context)
    # The batch wrapper forwards only after this immutable marker exists.  At this
    # point the request, snapshot, submission, all 20 launches, protocol, direct
    # Hydra command, and this cell's complete scientific identity have validated.
    seal_json(
        generation / WORKER_SIGNAL_READY_NAME,
        {**base, "status": "worker_signal_ready"},
    )
    try:
        return _execute_generation(args, context, task_root, generation, relay)
    except Exception as exc:
        try:
            seal_json(
                generation / GENERATION_FAILED_NAME,
                {
                    **base,
                    "status": "worker_failed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        except Exception as marker_exc:
            print(
                f"Exp23 failure-marker publication failed: {marker_exc}",
                file=sys.stderr,
                flush=True,
            )
        raise


def mark_requeue_calling(args: argparse.Namespace) -> int:
    submission = ensure_directory(args.submission_root, "submission root")
    bootstrap_contract = getattr(args, "_bootstrap_contract", None)
    snapshot = getattr(args, "_bootstrap_snapshot_root", None)
    require(isinstance(bootstrap_contract, Mapping) and isinstance(snapshot, Path), "CALLING lacks verified bootstrap")
    context = load_launch_context(
        snapshot_root=snapshot,
        submission_root=submission,
        submission_sha256=args.submission_sha256,
        cell_index=args.cell_index,
        bootstrap_contract=bootstrap_contract,
    )
    task_root = ensure_contained_directory(
        task_root_for(submission, args.cell_index),
        submission,
        "task state root",
    )
    generation = ensure_contained_directory(
        generation_root_for(task_root, args.restart_count),
        task_root,
        "generation state root",
    )
    require(not cancellation_requested(submission, task_root), "cancellation raced with requeue")
    ready_path = generation / REQUEUE_READY_NAME
    ready, ready_sha256, _ = read_json_artifact(
        ready_path,
        "requeue READY",
        required_mode=0o444,
    )
    require(set(ready) == REQUEUE_READY_FIELDS, "requeue READY fields differ")
    require(ready.get("status") == "requeue_ready", "generation is not requeue-ready")
    expected_common = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": args.submission_sha256,
        "launch_sha256": context["launch"]["launch_sha256"],
        "cell_index": args.cell_index,
        "restart_count": args.restart_count,
        "array_job_id": str(args.array_job_id),
        "array_task_id": int(args.array_task_id),
    }
    for key, value in expected_common.items():
        require(ready.get(key) == value, f"requeue READY {key} differs")
    current = resolve_checkpoint(context)
    ready_identity = validate_ready_checkpoint_binding(ready, current, "requeue READY")
    require(
        sha256_string(ready.get("checkpoint_sha256"))
        and current["checkpoint_sha256"] == ready["checkpoint_sha256"]
        and current["checkpoint_file_identity"] == ready_identity,
        "checkpoint advanced/swapped before scontrol requeue",
    )
    require(not cancellation_requested(submission, task_root), "cancellation raced before CALLING")
    target = f"{args.array_job_id}_{args.array_task_id}"
    calling = {
        **expected_common,
        "status": "scontrol_requeue_calling",
        "requeue_target": target,
        "requeue_ready_sha256": ready_sha256,
        "checkpoint_sha256": ready["checkpoint_sha256"],
        "checkpoint_file_identity": ready_identity,
    }
    require(set(calling) == REQUEUE_CALLING_FIELDS, "requeue CALLING fields differ")
    seal_json(
        generation / REQUEUE_CALLING_NAME,
        calling,
    )
    return 0


def _common_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--submission-sha256", required=True)
    parser.add_argument("--cell-index", type=int, required=True)
    parser.add_argument("--restart-count", type=int, required=True)
    parser.add_argument("--array-job-id", required=True)
    parser.add_argument("--array-task-id", type=int, required=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    run = sub.add_parser("run")
    _common_parser(run)
    run.add_argument("--snapshot-root", type=Path, required=True)
    record = sub.add_parser("record-signal")
    _common_parser(record)
    record.add_argument("--signal", choices=("USR1", "TERM"), required=True)
    calling = sub.add_parser("mark-requeue-calling")
    _common_parser(calling)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    assert_isolated_runtime()
    args = _parser().parse_args(argv)
    require(args.command is not None, "worker command is required")
    require(0 <= args.cell_index < 20, "cell index is out of range")
    require(args.restart_count >= 0, "restart count is negative")
    require(args.array_task_id == args.cell_index, "array task/cell mapping differs")
    require(STOP_ENVIRONMENT not in os.environ, f"inherited {STOP_ENVIRONMENT} is forbidden")
    expected_snapshot = args.snapshot_root if args.command == "run" else None
    bootstrap_snapshot, bootstrap_submission_root, bootstrap_contract = bootstrap_submission(
        args.submission_root,
        args.submission_sha256,
        expected_snapshot_root=expected_snapshot,
        configure_imports=False,
    )
    executing_worker = _absolute_path(os.path.abspath(__file__), "executing worker source")
    require(
        executing_worker == bootstrap_snapshot / PACKAGE_RELATIVE / "worker.py",
        "worker was not executed from the sealed snapshot",
    )
    args.submission_root = bootstrap_submission_root
    args._bootstrap_snapshot_root = bootstrap_snapshot
    args._bootstrap_contract = bootstrap_contract
    validate_submission_contract(
        bootstrap_submission_root,
        args.submission_sha256,
        contract=bootstrap_contract,
        snapshot_root=bootstrap_snapshot,
    )
    if args.command == "run":
        args.snapshot_root = bootstrap_snapshot
    if args.command == "record-signal":
        record_signal_request(
            submission_root=args.submission_root,
            submission_sha256=args.submission_sha256,
            cell_index=args.cell_index,
            restart_count=args.restart_count,
            array_job_id=args.array_job_id,
            array_task_id=args.array_task_id,
            signal_name=args.signal,
            bootstrap_contract=bootstrap_contract,
        )
        return 0
    if args.command == "mark-requeue-calling":
        return mark_requeue_calling(args)
    return run_worker(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except LifecycleError as exc:
        # Best-effort failure record; never overwrite a previously sealed result.
        print(f"Exp23 lifecycle contract error: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(CONTRACT_EXIT_CODE)
    except Exception as exc:
        # Campaign/source validators use their own fail-closed exception classes.
        # Keep the public worker failure code stable without masking the diagnostic.
        print(
            f"Exp23 lifecycle validation failed ({type(exc).__name__}): {exc}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(CONTRACT_EXIT_CODE)
