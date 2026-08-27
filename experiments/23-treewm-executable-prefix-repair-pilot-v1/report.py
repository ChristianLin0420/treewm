#!/usr/bin/env python3
"""Assemble, gate, and atomically publish the terminal Exp23 report.

The default/``--test-only`` action is read-only.  ``report.slurm`` uses the explicit
``--publish`` action after the complete twenty-cell array succeeds.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
import time
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
PACKAGE_RELATIVE = Path("experiments/23-treewm-executable-prefix-repair-pilot-v1")
PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]
CAMPAIGN_ID = "treewm-executable-prefix-repair-pilot-v1-launch2"
BOUNDARIES = (5_000, 25_000)
SHA256 = frozenset("0123456789abcdef")
WORKER_COMPLETE_KEYS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "submission_sha256",
        "launch_sha256",
        "cell_index",
        "restart_count",
        "array_job_id",
        "array_task_id",
        "status",
        "completed_updates",
        "checkpoint_sha256",
        "completion_sha256",
        "final_eval_progress_sha256",
        "completed_results_sha256",
        "identity_sha256",
        "final_metrics",
    }
)
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


class ReportError(RuntimeError):
    """An engineering artifact is absent, ambiguous, unsafe, or inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def sha256_string(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA256


def _pairs(path: Path):
    def hook(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    return hook


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    regular_nonsymlink(source, f"JSON artifact {source}")
    try:
        with source.open("r", encoding="utf-8") as handle:
            value = json.load(
                handle,
                object_pairs_hook=_pairs(source),
                parse_constant=lambda token: (_ for _ in ()).throw(
                    ReportError(f"non-finite JSON value in {source}: {token}")
                ),
            )
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read {source}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {source}")
    return value


def regular_nonsymlink(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ReportError(f"{label} is unavailable: {exc}") from exc
    require(stat.S_ISREG(info.st_mode), f"{label} is not a regular nonsymlink file")
    return path.resolve(strict=True)


def nonsymlink_directory(path: Path, label: str) -> Path:
    lexical = path.absolute()
    current = Path(lexical.anchor)
    for part in lexical.parts[1:]:
        current /= part
        try:
            component = current.lstat()
        except OSError as exc:
            raise ReportError(f"{label} path component is unavailable: {current}: {exc}") from exc
        require(not stat.S_ISLNK(component.st_mode), f"{label} has a symlink path component: {current}")
    try:
        info = path.lstat()
    except OSError as exc:
        raise ReportError(f"{label} is unavailable: {exc}") from exc
    require(stat.S_ISDIR(info.st_mode), f"{label} is not a nonsymlink directory")
    return path.resolve(strict=True)


class _ReportCancelLock:
    """Linearize the final report commit against the durable cancel latch."""

    def __init__(self, submission_root: Path) -> None:
        self.root = submission_root
        self.path = submission_root / ".REPORT_CANCEL.lock"
        self.descriptor: int | None = None

    def __enter__(self) -> "_ReportCancelLock":
        root = nonsymlink_directory(self.root, "report/cancel lock root")
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            raise ReportError(f"cannot open report/cancel lock: {exc}") from exc
        try:
            opened = os.fstat(descriptor)
            named = self.path.lstat()
            require(stat.S_ISREG(opened.st_mode), "report/cancel lock is not regular")
            require((opened.st_dev, opened.st_ino) == (named.st_dev, named.st_ino), "report/cancel lock path raced")
            require(opened.st_uid == os.getuid() and opened.st_nlink == 1, "report/cancel lock ownership differs")
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            _fsync_directory(root)
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


def activate_isolated_runtime(manifest: Mapping[str, Any]) -> dict[str, Any]:
    require(bool(sys.flags.isolated) and bool(sys.flags.no_site), "report requires Python -I -S")
    expected = Path(str(manifest["paths"]["python"]))
    require(expected.is_absolute(), "pinned Python path is not absolute")
    try:
        info = expected.lstat()
    except OSError as exc:
        raise ReportError(f"pinned Python is unavailable: {exc}") from exc
    require(stat.S_ISLNK(info.st_mode) or stat.S_ISREG(info.st_mode), "pinned Python has invalid type")
    require(
        os.path.normpath(os.path.abspath(sys.executable)) == str(expected),
        f"report must use exact lexical pinned Python {expected}",
    )
    target = expected.resolve(strict=True)
    regular_nonsymlink(target, "resolved pinned Python")
    venv_root = expected.parent.parent
    pyvenv = venv_root / "pyvenv.cfg"
    regular_nonsymlink(pyvenv, "pinned pyvenv.cfg")
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
    for site_path in sites:
        try:
            site_info = site_path.lstat()
        except OSError as exc:
            raise ReportError(f"bound site-packages is unavailable: {exc}") from exc
        require(stat.S_ISDIR(site_info.st_mode), "bound site-packages is not a nonsymlink directory")
    existing = [value for value in sys.path if "site-packages" in value]
    require(not existing or existing == [str(value) for value in sites], "unexpected site-package bootstrap path")
    for site_path in sites:
        if str(site_path) not in sys.path:
            sys.path.append(str(site_path))
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


def verify_snapshot_inventory(
    snapshot_root: Path,
    submission_root: Path,
    submission_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the contract/receipt and exact read-only snapshot using stdlib only."""

    submission = nonsymlink_directory(submission_root, "submission root")
    snapshot = nonsymlink_directory(snapshot_root, "snapshot root")
    require(snapshot.is_relative_to(submission), "snapshot root escapes submission root")
    contract_path = submission / "SUBMISSION_CONTRACT.json"
    contained_regular(contract_path, submission, "submission contract")
    require(stat.S_IMODE(contract_path.lstat().st_mode) == 0o444, "submission contract mode differs")
    require(file_sha256(contract_path) == submission_sha256, "submission contract hash differs")
    seal_path = submission / "journal" / "0002_CONTRACT_SEALED.json"
    contained_regular(seal_path, submission, "contract-seal journal")
    require(stat.S_IMODE(seal_path.lstat().st_mode) == 0o444, "contract-seal journal mode differs")
    require(
        read_json(seal_path)
        == {
            "schema_version": 1,
            "record": "contract_sealed",
            "submission_sha256": submission_sha256,
            "launch_count": 20,
        },
        "contract-seal journal differs",
    )
    contract = read_json(contract_path)
    require(contract.get("schema_version") == 1 and contract.get("status") == "sealed_for_submission", "submission contract is not sealed")
    require(contract.get("campaign_id") == CAMPAIGN_ID, "submission contract campaign differs")
    require(contract.get("submission_root") == str(submission), "submission contract root differs")
    require(contract.get("snapshot_root") == str(snapshot), "submission snapshot root differs")
    inventory = contract.get("snapshot_inventory")
    require(isinstance(inventory, Mapping) and inventory, "snapshot inventory is absent")
    normalized: dict[str, str] = {}
    for raw_relative, digest in inventory.items():
        relative = str(_safe_relative(raw_relative, "snapshot inventory path"))
        require(sha256_string(digest), f"snapshot inventory digest is malformed: {relative}")
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
    require(snapshot.lstat().st_mode & 0o222 == 0, "snapshot root is writable")
    for path in snapshot.rglob("*"):
        entry = path.lstat()
        relative = str(path.relative_to(snapshot))
        require(not stat.S_ISLNK(entry.st_mode), f"snapshot contains symlink: {relative}")
        if stat.S_ISREG(entry.st_mode):
            actual_files.add(relative)
            require(relative in normalized, f"snapshot contains unclaimed file: {relative}")
            require(entry.st_mode & 0o222 == 0, f"snapshot file is writable: {relative}")
            require(file_sha256(path) == normalized[relative], f"snapshot file hash differs: {relative}")
        elif stat.S_ISDIR(entry.st_mode):
            actual_dirs.add(relative)
            require(entry.st_mode & 0o222 == 0, f"snapshot directory is writable: {relative}")
        else:
            raise ReportError(f"snapshot contains special entry: {relative}")
    require(actual_files == set(normalized), "snapshot file coverage differs")
    require(actual_dirs == expected_dirs, "snapshot directory coverage differs")

    receipt_path = submission / "SUBMISSION_RECEIPT.json"
    contained_regular(receipt_path, submission, "submission receipt")
    require(stat.S_IMODE(receipt_path.lstat().st_mode) == 0o444, "submission receipt mode differs")
    receipt = read_json(receipt_path)
    require(set(receipt) == RECEIPT_KEYS, "submission receipt schema differs")
    require(receipt["schema_version"] == 1 and receipt["status"] == "submitted", "submission receipt is not committed")
    require(receipt["campaign_id"] == CAMPAIGN_ID, "submission receipt campaign differs")
    require(receipt["submission_root"] == str(submission), "submission receipt root differs")
    require(receipt["snapshot_root"] == str(snapshot), "submission receipt snapshot differs")
    require(receipt["submission_sha256"] == submission_sha256, "submission receipt hash differs")
    train_id = str(receipt["train_array_job_id"])
    report_id = str(receipt["report_job_id"])
    require(train_id.isdigit() and int(train_id) > 0, "training receipt job ID is malformed")
    require(report_id.isdigit() and int(report_id) > 0 and report_id != train_id, "report receipt job ID is malformed")
    require(receipt["array"] == "0-19%20", "submission receipt array differs")
    require(receipt["dependency"] == f"afterok:{train_id}", "submission receipt dependency differs")
    for role, ordinal, expected_id in (("train", 3, train_id), ("report", 4, report_id)):
        journal_path = submission / "journal" / f"{ordinal:04d}_{role.upper()}_SUBMITTED.json"
        contained_regular(journal_path, submission, f"{role} submission journal")
        require(stat.S_IMODE(journal_path.lstat().st_mode) == 0o444, f"{role} submission journal mode differs")
        journal = read_json(journal_path)
        require(
            journal.get("record") == f"{role}_submitted"
            and str(journal.get("job_id")) == expected_id,
            f"submission receipt {role} job differs from durable journal",
        )
    scheduler_job = os.environ.get("SLURM_JOB_ID")
    if scheduler_job is not None:
        require(scheduler_job == report_id, "active report Slurm job differs from committed receipt")
    return contract, receipt


def contained_regular(path: Path, root: Path, label: str) -> Path:
    regular_nonsymlink(path, label)
    resolved = path.resolve(strict=True)
    expected_root = nonsymlink_directory(root, f"{label} root")
    require(resolved.is_relative_to(expected_root), f"{label} escapes its declared root")
    return resolved


def _load_module(name: str, path: Path, root: Path) -> ModuleType:
    resolved = contained_regular(path, root, name)
    unique = f"_treewm_exp23_report_{name}_{os.getpid()}_{time.time_ns()}"
    spec = importlib.util.spec_from_file_location(unique, resolved)
    require(spec is not None and spec.loader is not None, f"cannot load {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[unique] = module
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(unique, None)
        raise
    require(Path(str(module.__file__)).resolve(strict=True) == resolved, f"{name} imported outside snapshot")
    return module


def reject_environment(environ: Mapping[str, str] | None = None) -> None:
    environment = os.environ if environ is None else environ
    failures = sorted(
        key
        for key in environment
        if key.startswith("TREEWM_") or key in {"RANK", "WORLD_SIZE", "LOCAL_RANK"}
    )
    require(not failures, "forbidden inherited environment: " + ", ".join(failures))


def _decode_text_tensor(event: Any, path: Path) -> str:
    try:
        from tensorboard.util import tensor_util

        values = tensor_util.make_ndarray(event.tensor_proto).reshape(-1)
    except Exception as exc:
        raise ReportError(f"cannot decode TensorBoard text in {path}: {exc}") from exc
    require(len(values) == 1, f"TensorBoard text event is not scalar: {path}")
    value = values[0]
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReportError(f"TensorBoard text event is not UTF-8: {path}") from exc
    return str(value)


def parse_event_files(
    run_dir: Path,
    expected_sampler: Mapping[str, Any],
) -> dict[str, Any]:
    """Parse every event file; accept only value-identical scalar duplicates."""

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    run_root = nonsymlink_directory(run_dir, "scientific run directory")
    # Only root writers are training generations.  The terminal hparams writer lives
    # below hparams/ and intentionally has no fixed-validation text.
    paths = sorted(run_root.glob("events.out.tfevents.*"))
    hparams_root = run_root / "hparams"
    if os.path.lexists(hparams_root):
        nonsymlink_directory(hparams_root, "TensorBoard hparams directory")
        hparam_paths = sorted(hparams_root.rglob("events.out.tfevents.*"))
    else:
        hparam_paths = []
    all_event_paths = sorted(run_root.rglob("events.out.tfevents.*"))
    require(
        set(all_event_paths) == set(paths) | set(hparam_paths),
        "unexpected TensorBoard event file outside root/hparams writers",
    )
    require(paths, f"no TensorBoard event files in {run_root}")
    merged: dict[str, dict[int, tuple[bytes, float]]] = {}
    duplicate_counts: dict[str, int] = {}
    expected_text = "<pre>" + json.dumps(
        dict(expected_sampler), sort_keys=True, indent=2, allow_nan=False
    ) + "</pre>"
    fixed_text_events = 0
    excluded_eval_tags: set[str] = set()
    event_hashes: dict[str, str] = {}
    for path in paths:
        contained_regular(path, run_root, "TensorBoard event file")
        before_info = path.lstat()
        before_hash = file_sha256(path)
        accumulator = EventAccumulator(
            str(path), size_guidance={"scalars": 0, "tensors": 0}
        )
        try:
            accumulator.Reload()
        except Exception as exc:
            raise ReportError(f"unreadable TensorBoard event file {path}: {exc}") from exc
        tags = accumulator.Tags()
        for tag in tags.get("scalars", []):
            require(isinstance(tag, str) and tag, f"invalid scalar tag in {path}")
            if tag.startswith("eval/"):
                # Periodic 1/task and terminal 5/task evaluations intentionally share
                # update 25k.  They are not training/calibration inputs and terminal
                # outcomes come exclusively from the 25 raw progress rows.
                excluded_eval_tags.add(tag)
                continue
            for event in accumulator.Scalars(tag):
                step = int(event.step)
                value = float(event.value)
                wall_time = float(event.wall_time)
                require(step >= 0 and math.isfinite(value) and math.isfinite(wall_time), f"invalid scalar event {tag}@{step}")
                bits = struct.pack(">d", value)
                previous = merged.setdefault(tag, {}).get(step)
                if previous is None:
                    merged[tag][step] = (bits, value)
                else:
                    require(previous[0] == bits, f"conflicting duplicate scalar {tag}@{step}")
                    duplicate_counts[tag] = duplicate_counts.get(tag, 0) + 1
        fixed_in_file = 0
        for tag in tags.get("tensors", []):
            if tag != "meta/fixed_validation_sample/text_summary":
                continue
            for event in accumulator.Tensors(tag):
                require(int(event.step) == 0 and math.isfinite(float(event.wall_time)), "fixed-validation text has invalid metadata")
                text = _decode_text_tensor(event, path)
                require(text == expected_text, "fixed-validation sampler text differs from frozen summary")
                fixed_text_events += 1
                fixed_in_file += 1
        require(fixed_in_file == 1, f"training generation {path.name} lacks one exact fixed-validation text")
        after_info = path.lstat()
        require(
            (before_info.st_dev, before_info.st_ino, before_info.st_size, before_info.st_mtime_ns, before_info.st_ctime_ns)
            == (after_info.st_dev, after_info.st_ino, after_info.st_size, after_info.st_mtime_ns, after_info.st_ctime_ns)
            and file_sha256(path) == before_hash,
            f"TensorBoard event file changed while reporting: {path}",
        )
        event_hashes[str(path.relative_to(run_root))] = before_hash
    hparams_hashes: dict[str, str] = {}
    for path in hparam_paths:
        contained_regular(path, run_root, "TensorBoard hparams event file")
        hparams_hashes[str(path.relative_to(run_root))] = file_sha256(path)
    require(fixed_text_events == len(paths), "fixed-validation text generation coverage differs")
    scalars = {
        tag: {step: item[1] for step, item in sorted(points.items())}
        for tag, points in sorted(merged.items())
    }
    return {
        "scalars": scalars,
        "event_files": [str(path.relative_to(run_root)) for path in paths],
        "event_file_sha256": event_hashes,
        "hparams_event_files": [str(path.relative_to(run_root)) for path in hparam_paths],
        "hparams_event_file_sha256": hparams_hashes,
        "excluded_eval_tags": sorted(excluded_eval_tags),
        "fixed_validation_text_events": fixed_text_events,
        "identical_scalar_duplicates": duplicate_counts,
    }


def _axis(target: int, cadence: int, window: int | None = None) -> tuple[int, ...]:
    lower = cadence if window is None else max(cadence, target - min(window, target))
    first = ((lower + cadence - 1) // cadence) * cadence
    return tuple(range(first, target + 1, cadence))


def _require_axis(
    scalars: Mapping[str, Mapping[int, float]],
    tags: Sequence[str],
    axis: Sequence[int],
    label: str,
) -> None:
    expected = tuple(axis)
    require(expected, f"{label}: empty expected axis")
    for tag in tags:
        actual = tuple(sorted(scalars.get(tag, {})))
        require(actual == expected, f"{label}: scalar axis differs for {tag}")


def validate_boundary_axes(
    scalars: Mapping[str, Mapping[int, float]],
    gate: ModuleType,
    manifest: Mapping[str, Any],
) -> None:
    scientific = manifest["scientific_contract"]
    train_cadence = int(scientific["training_telemetry_every_updates"])
    validation_cadence = int(scientific["validation_every_updates"])
    training_tags = (
        *gate.GAUGE_EXACT_TAGS,
        *gate.GRADIENT_NORM_TAGS,
        *gate.GRADIENT_CLIP_TAGS,
        *(gate.TRAIN_PREFIX + suffix for suffix in gate.PREFIX_COMMON_SUFFIXES),
    )
    validation_tags = (
        *(tag for tag in gate.METHOD_EXACT_TAGS if tag != "data/validation_fixed_sample_count"),
        *(gate.PREFIX + suffix for suffix in gate.PREFIX_COMMON_SUFFIXES),
    )
    training_axis = tuple(range(train_cadence, 25_000 + 1, train_cadence))
    validation_axis = tuple(range(validation_cadence, 25_000 + 1, validation_cadence))
    _require_axis(scalars, training_tags, training_axis, "full training telemetry")
    _require_axis(scalars, validation_tags, validation_axis, "full validation telemetry")
    _require_axis(
        scalars,
        ("data/validation_fixed_sample_count",),
        (0, *validation_axis),
        "fixed-validation sample-count telemetry",
    )


def _serial_scalars(
    scalars: Mapping[str, Mapping[int, float]], target: int
) -> dict[str, list[list[float | int]]]:
    return {
        tag: [[int(step), float(value)] for step, value in sorted(points.items()) if step <= target]
        for tag, points in sorted(scalars.items())
        if any(step <= target for step in points)
    }


def _prefix_contract(
    setting_id: str,
    prefix_lock: Mapping[str, Any],
    actual_sampler: Mapping[str, Any],
) -> dict[str, Any]:
    expected = prefix_lock["settings"][setting_id]
    require(dict(actual_sampler) == expected["fixed_validation_sampler"], f"{setting_id}: sampler summary differs")
    return {
        "setting_id": setting_id,
        "target_contract_sha256": expected["target_contract_sha256"],
        "prefix_target_artifact_sha256": prefix_lock["artifact_sha256"],
        "validation_manifest_sha256": expected["validation_manifest_sha256"],
        "fixed_validation_sampler": dict(actual_sampler),
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


def validate_worker_receipt(
    marker: Mapping[str, Any],
    *,
    index: int,
    launch: Mapping[str, Any],
    submission_sha256: str,
    train_array_job_id: str,
    terminal: Mapping[str, Any],
) -> None:
    require(set(marker) == WORKER_COMPLETE_KEYS, f"cell{index}: WORKER_COMPLETE fields differ")
    require(marker["schema_version"] == 1, f"cell{index}: worker receipt schema differs")
    require(marker["status"] == "worker_complete", f"cell{index}: worker receipt status differs")
    require(marker["campaign_id"] == CAMPAIGN_ID, f"cell{index}: worker receipt campaign differs")
    require(marker["submission_sha256"] == submission_sha256, f"cell{index}: worker receipt submission differs")
    require(marker["launch_sha256"] == launch["launch_sha256"], f"cell{index}: worker receipt launch differs")
    require(type(marker["cell_index"]) is int and marker["cell_index"] == index, f"cell{index}: worker receipt cell differs")
    require(type(marker["restart_count"]) is int and marker["restart_count"] >= 0, f"cell{index}: restart count differs")
    require(isinstance(marker["array_job_id"], str) and marker["array_job_id"].isdigit(), f"cell{index}: array job ID differs")
    require(marker["array_job_id"] == train_array_job_id, f"cell{index}: worker array job differs from submission receipt")
    require(type(marker["array_task_id"]) is int and marker["array_task_id"] == index, f"cell{index}: array task differs")
    for key in (
        "completed_updates",
        "checkpoint_sha256",
        "completion_sha256",
        "final_eval_progress_sha256",
        "completed_results_sha256",
        "identity_sha256",
    ):
        require(marker[key] == terminal[key], f"cell{index}: worker receipt {key} differs from independent verification")
    require(_finite_mapping(marker["final_metrics"], f"cell{index} worker metrics") == terminal["final_metrics"], f"cell{index}: worker receipt metrics differ")


def _outcome_from_terminal(terminal: Mapping[str, Any]) -> dict[str, Any]:
    progress = terminal["progress"]
    rows = progress["rows"]
    require(len(rows) == 25 and progress["status"] == "complete", "terminal progress is incomplete")
    successes = sum(bool(row["success"]) for row in rows)
    progress_values = [
        (float(row["initial_goal_distance"]) - float(row["final_goal_distance"]))
        / max(float(row["initial_goal_distance"]), 1e-6)
        for row in rows
    ]
    return {
        "source": "terminal_final_evaluation",
        "status": "complete",
        "task_ids": [1, 2, 3, 4, 5],
        "episodes_per_task": 5,
        "num_episodes": 25,
        "successes": successes,
        "success_rate": successes / 25.0,
        "distance_reduction_frac": sum(progress_values) / 25.0,
        "completed_results": rows,
        "completed_results_sha256": stable_hash(rows),
        "completion_sha256": terminal["completion_sha256"],
        "final_eval_progress_sha256": terminal["final_eval_progress_sha256"],
        "checkpoint_sha256": terminal["checkpoint_sha256"],
    }


def _context_modules(
    snapshot_root: Path, interpreter: Mapping[str, Any]
) -> tuple[ModuleType, ModuleType, ModuleType]:
    root = nonsymlink_directory(snapshot_root, "snapshot root")
    verified_tail = [
        str(root),
        str(interpreter["venv_site_packages"]),
        str(interpreter["base_site_packages"]),
    ]
    for value in verified_tail:
        while value in sys.path:
            sys.path.remove(value)
    sys.path.extend(verified_tail)
    require(sys.path[-3:] == verified_tail, "verified report import-path order differs")
    package = root / PACKAGE_RELATIVE
    nonsymlink_directory(package, "snapshot Exp23 package")
    campaign = _load_module("campaign", package / "campaign.py", root)
    worker = _load_module("worker", package / "worker.py", root)
    gate = _load_module("gate", package / "gate.py", root)
    return campaign, worker, gate


def assemble_report(
    snapshot_root: Path,
    submission_root: Path,
    submission_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reject_environment()
    snapshot_root = nonsymlink_directory(snapshot_root, "snapshot root")
    submission_root = nonsymlink_directory(submission_root, "submission root")
    require(sha256_string(submission_sha256), "submission SHA256 is malformed")
    require(
        not os.path.lexists(submission_root / "CANCEL_REQUESTED.json"),
        "cancelled/ambiguous submission cannot report",
    )
    contract, receipt = verify_snapshot_inventory(
        snapshot_root, submission_root, submission_sha256
    )
    bootstrap_manifest = read_json(snapshot_root / PACKAGE_RELATIVE / "manifest.json")
    runtime_interpreter = activate_isolated_runtime(bootstrap_manifest)
    require(
        contract.get("orchestration_interpreter") == runtime_interpreter,
        "report interpreter differs from submission contract",
    )
    require(
        not any(name == "treewm" or name.startswith("treewm.") for name in sys.modules),
        "treewm was imported before snapshot verification",
    )
    campaign, worker, gate = _context_modules(snapshot_root, runtime_interpreter)
    package = snapshot_root / PACKAGE_RELATIVE
    manifest, _weight_lock = campaign.load_contract(snapshot_root)
    require(manifest == bootstrap_manifest, "manifest changed during report bootstrap")
    protocol = campaign.verify_protocol_lock(package)
    worker_contract = worker.validate_submission_contract(
        submission_root,
        submission_sha256,
        contract=contract,
        snapshot_root=snapshot_root,
        protocol_sha256=protocol,
        manifest_sha256=campaign.manifest_sha256(manifest),
    )
    require(worker_contract == contract, "worker and reporter contract parsing differ")
    require(contract.get("trainer_code_fingerprint") == manifest["core_binding"]["trainer_code_fingerprint"], "submission trainer binding differs")
    required_contract_bindings = {
        "weight_audit_artifact_sha256": manifest["weight_audit"]["artifact_sha256"],
        "prefix_target_artifact_sha256": manifest["prefix_target_contract"]["artifact_sha256"],
        "resolved_config_artifact_sha256": manifest["resolved_config_contract"]["artifact_sha256"],
        "causal_parity_artifact_sha256": manifest["causal_parity_contract"]["artifact_sha256"],
    }
    for key, expected in required_contract_bindings.items():
        require(contract.get(key) == expected, f"submission {key} differs")
    prefix_lock = campaign.read_json(package / "prefix_target.lock.json")

    # Phase 1 is outcome blind.  Require all durable marker entries to exist, but do
    # not open their outcome-bearing JSON until every telemetry/calibration boundary
    # has been parsed and evaluated.
    contexts: list[dict[str, Any]] = []
    marker_paths: list[Path] = []
    cells: list[dict[str, Any]] = []
    boundary_evaluations: list[dict[str, Any]] = []
    event_provenance: list[dict[str, Any]] = []
    for index in range(20):
        marker_path = submission_root / "tasks" / f"cell-{index:02d}" / "WORKER_COMPLETE.json"
        contained_regular(marker_path, submission_root, f"cell{index} WORKER_COMPLETE")
        marker_paths.append(marker_path)
        context = worker.load_launch_context(
            snapshot_root=snapshot_root,
            submission_root=submission_root,
            submission_sha256=submission_sha256,
            cell_index=index,
            bootstrap_contract=contract,
        )
        contexts.append(context)
        launch = context["launch"]
        cell = launch["cell"]
        run_dir = Path(cell["run_directory"])
        expected_sampler = prefix_lock["settings"][cell["setting"]]["fixed_validation_sampler"]
        parsed = parse_event_files(run_dir, expected_sampler)
        validate_boundary_axes(parsed["scalars"], gate, manifest)
        prefix = _prefix_contract(cell["setting"], prefix_lock, expected_sampler)
        boundaries = {
            str(target): {
                "update": target,
                "scalars": _serial_scalars(parsed["scalars"], target),
                "prefix_contract": prefix,
            }
            for target in BOUNDARIES
        }
        evaluated = {
            str(target): gate.evaluate_boundary(
                boundaries[str(target)],
                cell_label=f"{cell['setting']}/{cell['arm']}/seed{cell['seed']}",
                target=target,
                arm_id=cell["arm"],
                setting_id=cell["setting"],
                manifest=manifest,
            )
            for target in BOUNDARIES
        }
        boundary_evaluations.append(
            {
                "index": index,
                "setting_id": cell["setting"],
                "arm_id": cell["arm"],
                "seed": cell["seed"],
                "boundaries": evaluated,
            }
        )
        cells.append(
            {
                "index": index,
                "setting_id": cell["setting"],
                "arm_id": cell["arm"],
                "seed": cell["seed"],
                "fresh_start": True,
                "boundaries": boundaries,
            }
        )
        event_provenance.append(
            {
                "index": index,
                "event_files": parsed["event_files"],
                "event_file_sha256": parsed["event_file_sha256"],
                "hparams_event_files": parsed["hparams_event_files"],
                "hparams_event_file_sha256": parsed["hparams_event_file_sha256"],
                "excluded_eval_tags": parsed["excluded_eval_tags"],
                "fixed_validation_text_events": parsed["fixed_validation_text_events"],
                "identical_scalar_duplicates": parsed["identical_scalar_duplicates"],
            }
        )
    # Freeze paired calibration computation before any terminal result is opened.
    paired_calibration = gate._paired_calibration(boundary_evaluations, manifest)
    outcome_blind_phase = {
        "status": "all_boundaries_parsed_and_calibration_computed_before_outcomes",
        "boundary_evaluations_sha256": stable_hash(boundary_evaluations),
        "paired_calibration_sha256": stable_hash(paired_calibration),
    }

    # Phase 2 opens and independently validates the terminal checkpoint/progress/
    # completion triplet.  WORKER_COMPLETE is only corroboration, never authority.
    terminal_provenance: list[dict[str, Any]] = []
    for index, (context, marker_path) in enumerate(zip(contexts, marker_paths, strict=True)):
        marker = read_json(marker_path)
        inspected = worker.inspect_run(context)
        require(inspected.get("kind") == "complete", f"cell{index}: trainer triplet is not terminal")
        complete = inspected["complete"]
        terminal = {
            **complete,
            "progress": inspected["progress"],
        }
        validate_worker_receipt(
            marker,
            index=index,
            launch=context["launch"],
            submission_sha256=submission_sha256,
            terminal=terminal,
            train_array_job_id=str(receipt["train_array_job_id"]),
        )
        cells[index]["boundaries"]["25000"]["outcome"] = _outcome_from_terminal(terminal)
        terminal_provenance.append(
            {
                "index": index,
                "worker_complete_sha256": file_sha256(marker_path),
                "restart_count": marker["restart_count"],
                "array_job_id": marker["array_job_id"],
                "checkpoint_sha256": complete["checkpoint_sha256"],
                "completion_sha256": complete["completion_sha256"],
                "final_eval_progress_sha256": complete["final_eval_progress_sha256"],
                "completed_results_sha256": complete["completed_results_sha256"],
                "identity_sha256": complete["identity_sha256"],
            }
        )

    bundle = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        **gate.package_binding(manifest),
        "cells": cells,
    }
    decision = gate.evaluate_bundle(bundle, manifest)
    require(decision.get("status") in {"accepted_engineering_pilot", "rejected"}, "gate returned an invalid scientific status")
    provenance = {
        "schema_version": 1,
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "outcome_blind_phase": outcome_blind_phase,
        "event_artifacts": event_provenance,
        "terminal_artifacts": terminal_provenance,
        "report_bundle_sha256": stable_hash(bundle),
        "gate_sha256": decision["gate_sha256"],
    }
    return bundle, decision, provenance


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def seal_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = (json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    except FileExistsError:
        contained_regular(path, path.parent, f"immutable report artifact {path.name}")
        require(path.read_bytes() == payload, f"immutable report artifact differs: {path}")
        return digest
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, f"short report artifact write: {path}")
            view = view[written:]
        os.fsync(descriptor)
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return digest


def _publish_report_locked(
    submission_root: Path,
    submission_sha256: str,
    bundle: Mapping[str, Any],
    decision: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    submission_root = nonsymlink_directory(submission_root, "submission root")
    report_root = submission_root / "report"
    bundle_hash = stable_hash(bundle)
    gate_hash = str(decision["gate_sha256"])
    bundle_name = f"REPORT_BUNDLE.{bundle_hash}.json"
    decision_name = f"GATE_DECISION.{gate_hash}.json"
    provenance_hash = stable_hash(provenance)
    provenance_name = f"REPORT_PROVENANCE.{provenance_hash}.json"
    bundle_payload_sha = hashlib.sha256(
        (json.dumps(dict(bundle), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    ).hexdigest()
    decision_payload_sha = hashlib.sha256(
        (json.dumps(dict(decision), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    ).hexdigest()
    provenance_payload_sha = hashlib.sha256(
        (json.dumps(dict(provenance), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    ).hexdigest()
    commit = {
        "schema_version": 1,
        "status": decision["status"],
        "scientific_rejection": decision["status"] == "rejected",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "report_bundle": bundle_name,
        "report_bundle_sha256": bundle_hash,
        "report_bundle_file_sha256": bundle_payload_sha,
        "gate_decision": decision_name,
        "gate_sha256": gate_hash,
        "gate_decision_file_sha256": decision_payload_sha,
        "provenance": provenance_name,
        "provenance_sha256": provenance_hash,
        "provenance_file_sha256": provenance_payload_sha,
    }

    expected_names = {bundle_name, decision_name, provenance_name, "REPORT_COMMIT.json"}

    def validate_existing() -> None:
        root = nonsymlink_directory(report_root, "published report root")
        require(root.lstat().st_mode & 0o222 == 0, "published report root is writable")
        actual = {path.name for path in root.iterdir()}
        require(actual == expected_names, "published report file coverage differs")
        commit_path = contained_regular(root / "REPORT_COMMIT.json", root, "published report commit")
        require(commit_path.lstat().st_mode & 0o222 == 0, "published report commit is writable")
        require(read_json(commit_path) == commit, "published report commit differs")
        for name, digest in (
            (bundle_name, bundle_payload_sha),
            (decision_name, decision_payload_sha),
            (provenance_name, provenance_payload_sha),
        ):
            path = contained_regular(root / name, root, f"published report {name}")
            require(path.lstat().st_mode & 0o222 == 0, f"published report file is writable: {name}")
            require(file_sha256(path) == digest, f"published report file differs: {name}")

    if os.path.lexists(report_root):
        validate_existing()
        return commit

    staging = submission_root / f".report.tmp.{os.getpid()}.{time.time_ns()}"
    staging.mkdir(mode=0o700)
    _fsync_directory(submission_root)
    try:
        require(seal_json(staging / bundle_name, bundle) == bundle_payload_sha, "staged bundle hash differs")
        require(seal_json(staging / decision_name, decision) == decision_payload_sha, "staged decision hash differs")
        require(seal_json(staging / provenance_name, provenance) == provenance_payload_sha, "staged provenance hash differs")
        seal_json(staging / "REPORT_COMMIT.json", commit)
        _fsync_directory(staging)
        staging.chmod(0o555)
        _fsync_directory(submission_root)
        require(
            not os.path.lexists(submission_root / "CANCEL_REQUESTED.json"),
            "cancellation latch appeared before report commit",
        )
        require(not os.path.lexists(report_root), "report publication target appeared concurrently")
        os.rename(staging, report_root)
        _fsync_directory(submission_root)
    except BaseException:
        if os.path.lexists(staging):
            try:
                staging.chmod(0o700)
                for path in staging.iterdir():
                    path.chmod(0o600)
                    path.unlink()
                staging.rmdir()
                _fsync_directory(submission_root)
            except OSError:
                pass
        raise
    validate_existing()
    return commit


def publish_report(
    submission_root: Path,
    submission_sha256: str,
    bundle: Mapping[str, Any],
    decision: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish only on the report side of the report/cancellation linearization."""

    submission_root = nonsymlink_directory(submission_root, "submission root")
    with _ReportCancelLock(submission_root):
        require(
            not os.path.lexists(submission_root / "CANCEL_REQUESTED.json"),
            "cancelled/ambiguous submission cannot publish a report",
        )
        return _publish_report_locked(
            submission_root,
            submission_sha256,
            bundle,
            decision,
            provenance,
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--test-only", action="store_true", help="read-only assembly (default)")
    actions.add_argument("--publish", action="store_true", help="atomically publish the gate decision")
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--submission-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        bundle, decision, provenance = assemble_report(
            args.snapshot_root, args.submission_root, args.submission_sha256
        )
        if args.publish:
            result = publish_report(
                args.submission_root,
                args.submission_sha256,
                bundle,
                decision,
                provenance,
            )
        else:
            result = {
                "schema_version": 1,
                "status": "read_only_report_verified",
                "scientific_status": decision["status"],
                "report_bundle_sha256": stable_hash(bundle),
                "gate_sha256": decision["gate_sha256"],
                "writes_performed": 0,
                "scheduler_calls": 0,
            }
        print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        # There is intentionally no error marker here: absence of REPORT_COMMIT.json
        # is the engineering-failure state, and it cannot be confused with a frozen
        # scientific rejection.
        print(f"Exp23 report engineering error: {exc}", file=sys.stderr, flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
