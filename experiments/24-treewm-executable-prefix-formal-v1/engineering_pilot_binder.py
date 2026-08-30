#!/usr/bin/env python3
"""Versioned, read-only semantic verifier for an accepted Exp23 Launch8 report.

The adapter is sealed to the committed Launch8 source at ``33122e15``. It
authenticates the immutable report quartet, replays that exact gate in an isolated
subprocess, and independently checks reporter-only provenance and terminal joins.
The absolute trust chain and every source/report authority object remain open by
descriptor through the final lexical identity recheck and return boundary.
It never writes or publishes ``accepted_engineering_pilot.binding.json``; successful
verification returns an unpublished candidate record and leaves Exp24 unbound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any, Mapping, Sequence

sys.dont_write_bytecode = True
PACKAGE_DIR = Path(__file__).resolve().parent
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import runtime


EXPECTED_ACCEPTED_CAMPAIGN_ID = runtime.EXPECTED_ACCEPTED_PILOT_CAMPAIGN_ID
FORBIDDEN_POSITIVE_CAMPAIGN_ID = runtime.FORBIDDEN_POSITIVE_PILOT_CAMPAIGN_ID
POSITIVE_ADAPTER_STATE = runtime.ENGINEERING_PILOT_ADAPTER_STATE
POSITIVE_BINDING_STATE = "unbound"
ADAPTER_REQUIREMENTS = runtime.ENGINEERING_PILOT_ADAPTER_REQUIREMENTS
FROZEN_SOURCE_COMMIT = runtime.FROZEN_LAUNCH8_SOURCE_COMMIT
FROZEN_PACKAGE_RELATIVE = runtime.FROZEN_LAUNCH8_PACKAGE_RELATIVE
FROZEN_PACKAGE_BINDING = runtime.FROZEN_LAUNCH8_PACKAGE_BINDING

PACKAGE_BINDING_KEYS = frozenset(FROZEN_PACKAGE_BINDING)
REPORT_COMMIT_KEYS = frozenset(
    {
        "schema_version", "status", "scientific_rejection", "campaign_id",
        "submission_sha256", "report_bundle", "report_bundle_sha256",
        "report_bundle_file_sha256", "gate_decision", "gate_sha256",
        "gate_decision_file_sha256", "provenance", "provenance_sha256",
        "provenance_file_sha256",
    }
)
BUNDLE_KEYS = frozenset({"schema_version", "campaign_id", "cells", *PACKAGE_BINDING_KEYS})
RAW_CELL_KEYS = frozenset(
    {"index", "setting_id", "arm_id", "seed", "fresh_start", "boundaries"}
)
RAW_BOUNDARY_KEYS = frozenset({"update", "scalars", "prefix_contract"})
RAW_OUTCOME_KEYS = frozenset(
    {
        "source", "status", "task_ids", "episodes_per_task", "num_episodes",
        "successes", "success_rate", "distance_reduction_frac",
        "completed_results", "completed_results_sha256", "completion_sha256",
        "final_eval_progress_sha256", "checkpoint_sha256",
    }
)
EPISODE_RESULT_KEYS = frozenset(
    {
        "success", "steps", "replans", "nodes", "final_goal_distance",
        "best_goal_distance", "chunk_lengths", "selected_depths",
        "initial_goal_distance", "displacement", "path_length",
        "action_magnitude", "no_action_plans", "guard_plans",
        "guard_rejections", "guard_candidate_count", "guard_accepted_count",
        "guard_best_predicted_improvements",
        "guard_selected_predicted_improvements", "trajectory", "progress",
        "task_index", "task_id", "episode_index", "episode_seed",
        "planning_wall_clock_s",
    }
)
PREFIX_CONTRACT_KEYS = frozenset(
    {
        "setting_id", "target_contract_sha256", "prefix_target_artifact_sha256",
        "validation_manifest_sha256", "fixed_validation_sampler",
    }
)
PROVENANCE_KEYS = frozenset(
    {
        "schema_version", "campaign_id", "submission_sha256",
        "production_authorization_prerequisite",
        "production_authorization_prerequisite_sha256", "outcome_blind_phase",
        "event_artifacts", "terminal_artifacts", "report_bundle_sha256",
        "gate_sha256",
    }
)
PRODUCTION_AUTHORIZATION_KEYS = frozenset(
    {
        "schema_version", "status", "attempt", "path", "raw_sha256",
        "canonical_sha256", "report_raw_sha256", "source_protocol_sha256",
        "source_commit", "state_root", "state_file_map_canonical_sha256",
        "canary_token", "job_ids_by_role", "accepted_attempt_sha256",
        "production_authorization_evidence_sha256",
        "sealed_package_protocol_sha256",
    }
)
OUTCOME_BLIND_PHASE_KEYS = frozenset(
    {"status", "boundary_evaluations_sha256", "paired_calibration_sha256"}
)
EVENT_PROVENANCE_KEYS = frozenset(
    {
        "index", "event_files", "event_file_sha256", "hparams_event_files",
        "hparams_event_file_sha256", "excluded_eval_tags",
        "fixed_validation_text_events", "identical_scalar_duplicates",
    }
)
TERMINAL_PROVENANCE_KEYS = frozenset(
    {
        "index", "worker_complete_sha256", "wave_index", "array_job_id",
        "checkpoint_sha256", "completion_sha256", "final_eval_progress_sha256",
        "completed_results_sha256", "identity_sha256", "wave_lineage",
    }
)
WAVE_LINEAGE_KEYS = frozenset(
    {
        "branch", "wave0_start_sha256", "wave0_predecessor_evidence_name",
        "wave0_predecessor_evidence_sha256", "wave0_checkpoint_sha256",
        "wave1_start_sha256", "wave1_input_checkpoint_sha256",
        "wave1_predecessor_evidence_sha256", "wave1_noop_sha256",
    }
)
DECISION_KEYS = frozenset(
    {
        "schema_version", "campaign_id", "status", "formal_validation",
        "supports_1m_claim", "report_bundle_sha256", "package_binding",
        "manifest_sha256", "matrix", "gates", "causal_parity_audit",
        "paired_25k_calibration", "outcomes_25000", "cells", "gate_sha256",
    }
)
NORMALIZED_CELL_KEYS = frozenset(
    {"index", "setting_id", "arm_id", "seed", "boundaries", "outcome"}
)
NORMALIZED_BOUNDARY_KEYS = frozenset(
    {
        "structural_passed", "candidate_passed", "structural_gates",
        "method_gates", "candidate_gates", "prefix_contract",
        "train_prefix_telemetry", "validation_prefix_telemetry",
        "prefix_term_telemetry", "calibration", "recent_gauge_min_ratio",
        "clip_fraction_below_0_05_by_tag", "gain_recent_means",
        "horizon_empirical_prior_entropy",
    }
)
STRUCTURAL_GATE_KEYS = frozenset(
    {
        "required_exact_boundary_telemetry_finite", "fixed_validation_axis_and_count",
        "complete_recent_gradient_axis", "nonzero_world_gain_and_split_gradients",
        "gradient_clip_coefficients_in_0_to_1", "complete_recent_gauge_axis",
        "gauge_reference_sealed_at_update_zero", "gauge_ratio_consistent",
        "prefix_contract_exact", "train_prefix_telemetry_structural",
        "fixed_validation_prefix_telemetry_structural",
        "train_validation_prefix_domain_parity", "generic_prefix_term_telemetry_exact",
        "calibration_telemetry_structural",
    }
)
METHOD_GATE_KEYS = frozenset(
    {
        "validation_nonregression", "self_fed_multistep_validation_nonregression",
        "horizon_ce_below_uniform_and_empirical_prior", "q_advantage",
        "gain_rank_pair_eligibility_and_coverage", "support_recall_and_precision",
    }
)
CANDIDATE_GATE_KEYS = frozenset(
    {
        "structural", "bounded_branch_rest_and_gain_clipping",
        "absolute_gauge_retention", "unchanged_method_gates",
        "absolute_action_h4_calibration",
    }
)
TOP_GATE_KEYS = frozenset(
    {
        "frozen_outcome_blind_causal_parity_audit_bound",
        "fixed_common_validation_samples",
        "all_gs_cells_structurally_valid_at_both_boundaries",
        "all_gsep_cells_pass_unchanged_and_absolute_gates_at_both_boundaries",
        "paired_25k_calibration", "absolute_and_paired_25k_outcomes",
    }
)
PAIRED_CALIBRATION_GATE_KEYS = frozenset(
    {
        "all_paired_calibration_values_finite",
        "gsep_macro_endpoint_error_strictly_lower",
        "at_least_three_settings_lower_endpoint_error",
        "gsep_macro_action_distortion_not_higher",
        "gsep_macro_action_clip_fraction_not_higher",
    }
)
ABSOLUTE_OUTCOME_GATE_KEYS = frozenset(
    {
        "all_twenty_outcomes_structurally_valid",
        "each_candidate_seed_has_success_and_positive_mean_progress",
        "at_least_one_setting_has_success_both_seeds",
        "at_least_three_settings_have_positive_progress_both_seeds",
    }
)
PAIRED_OUTCOME_GATE_KEYS = frozenset(
    {
        "gsep_macro_success_not_worse", "gsep_macro_distance_reduction_not_worse",
        "at_least_one_macro_outcome_strictly_better",
        "at_least_three_settings_distance_reduction_not_worse",
    }
)


class EngineeringPilotBindingError(runtime.RuntimeContractError):
    """The immutable positive evidence or frozen adapter dependency differs."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EngineeringPilotBindingError(message)


def _exact_keys(
    value: object, expected: frozenset[str], label: str
) -> Mapping[str, Any]:
    _require(isinstance(value, Mapping), f"{label} must be an object")
    _require(set(value) == expected, f"{label} fields differ")
    return value


def _sha256_string(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and set(value) <= set("0123456789abcdef")
    )


def _positive_int_string(value: object) -> bool:
    return (
        isinstance(value, str)
        and bool(value)
        and value[0] in "123456789"
        and all(character in "0123456789" for character in value)
    )


def _normalized_absolute(path: Path, label: str) -> Path:
    _require(path.is_absolute(), f"{label} must be absolute")
    _require(os.path.normpath(str(path)) == str(path), f"{label} must be normalized")
    _require(not str(path).startswith("//"), f"{label} must have one absolute root")
    return path


def _pretty_json_bytes(value: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise EngineeringPilotBindingError(
            "report JSON is noncanonical/nonfinite"
        ) from exc


def _report_tree_entries(descriptor: int) -> dict[str, Any]:
    def scan() -> dict[str, tuple[int, ...]]:
        with os.scandir(descriptor) as iterator:
            rows = sorted(
                (entry.name, entry.stat(follow_symlinks=False)) for entry in iterator
            )
        result: dict[str, tuple[int, ...]] = {}
        for name, child in rows:
            _require(
                name not in {"", ".", ".."}
                and "/" not in name
                and stat.S_ISREG(child.st_mode)
                and not stat.S_ISLNK(child.st_mode)
                and child.st_uid == os.getuid()
                and child.st_gid == os.getgid()
                and child.st_nlink == 1
                and stat.S_IMODE(child.st_mode) == 0o444,
                f"Launch8 report entry differs: {name}",
            )
            result[name] = runtime._identity(child)
        return result

    info = os.fstat(descriptor)
    _require(
        info.st_uid == os.getuid()
        and info.st_gid == os.getgid()
        and info.st_nlink == 2
        and stat.S_IMODE(info.st_mode) == 0o555,
        "Launch8 report root ownership/mode differs",
    )
    result = scan()
    _require(
        len(result) == 4,
        "Launch8 report tree must contain exactly four files",
    )
    _require(
        scan() == result
        and runtime._identity(os.fstat(descriptor)) == runtime._identity(info),
        "Launch8 report root changed while scanning",
    )
    return {
        "root_identity": runtime._identity(info),
        "entries": result,
    }


_DIRECTORY_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_DIRECTORY", 0)
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)
_FILE_OPEN_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_CLOEXEC", 0)
)


def _safe_entry_name(name: str, label: str) -> str:
    _require(
        isinstance(name, str)
        and name not in {"", ".", ".."}
        and "/" not in name,
        f"{label} entry name differs",
    )
    return name


def _open_directory_at(parent_fd: int, name: str, label: str) -> int:
    _safe_entry_name(name, label)
    try:
        descriptor = os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise EngineeringPilotBindingError(f"cannot open {label}: {exc}") from exc
    info = os.fstat(descriptor)
    if not stat.S_ISDIR(info.st_mode):
        os.close(descriptor)
        raise EngineeringPilotBindingError(f"{label} is not a directory")
    return descriptor


def _open_regular_at(parent_fd: int, name: str, label: str) -> int:
    _safe_entry_name(name, label)
    try:
        descriptor = os.open(name, _FILE_OPEN_FLAGS, dir_fd=parent_fd)
    except OSError as exc:
        raise EngineeringPilotBindingError(f"cannot open {label}: {exc}") from exc
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(descriptor)
        raise EngineeringPilotBindingError(
            f"{label} is not a single-link regular file"
        )
    return descriptor


def _entry_absent_at(parent_fd: int, name: str, label: str) -> None:
    _safe_entry_name(name, label)
    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        return
    except OSError as exc:
        raise EngineeringPilotBindingError(f"cannot inspect {label}: {exc}") from exc
    raise EngineeringPilotBindingError(f"{label} conflicts with accepted report")


def _read_regular_fd(
    descriptor: int,
    label: str,
    *,
    immutable: bool = True,
) -> tuple[bytes, str, os.stat_result]:
    before = os.fstat(descriptor)
    _require(stat.S_ISREG(before.st_mode), f"{label} is not a regular file")
    if immutable:
        _require(
            before.st_uid == os.getuid()
            and before.st_gid == os.getgid()
            and before.st_nlink == 1
            and stat.S_IMODE(before.st_mode) == 0o444,
            f"{label} ownership/mode differs",
        )
    try:
        os.lseek(descriptor, 0, os.SEEK_SET)
    except OSError as exc:
        raise EngineeringPilotBindingError(f"cannot rewind {label}: {exc}") from exc
    digest = hashlib.sha256()
    chunks: list[bytes] = []
    while True:
        block = os.read(descriptor, 16 * 1024 * 1024)
        if not block:
            break
        digest.update(block)
        chunks.append(block)
    after = os.fstat(descriptor)
    _require(
        runtime._identity(after) == runtime._identity(before),
        f"{label} changed while reading",
    )
    return b"".join(chunks), digest.hexdigest(), before


def _read_immutable_pretty_json_fd(
    descriptor: int,
    label: str,
    *,
    require_pretty: bool = True,
) -> tuple[dict[str, Any], str, os.stat_result, bytes]:
    payload, digest, info = _read_regular_fd(descriptor, label)
    try:
        value = runtime.parse_json_bytes(payload, label)
    except runtime.RuntimeContractError as exc:
        raise EngineeringPilotBindingError(str(exc)) from exc
    if require_pretty:
        _require(
            payload == _pretty_json_bytes(value),
            f"{label} byte serialization differs",
        )
    return value, digest, info, payload


class _RetainedLaunch8Trust:
    """Retain the complete lexical trust chain and all verifier authority fds."""

    def __init__(self, submission_root: Path) -> None:
        self.submission_root = submission_root
        self._descriptors: list[int] = []
        self._descriptor_by_label: dict[str, int] = {}
        self._identity_by_label: dict[str, tuple[int, ...]] = {}
        self.absolute_chain_labels: list[str] = []
        self.report_files: dict[str, int] = {}
        self.journal_fd: int | None = None
        try:
            self._open_absolute_chain()
            self.report_fd = self._open_directory(
                self.submission_fd, "report", "submission/report"
            )
            self.source_snapshot_fd = self._open_directory(
                self.submission_fd,
                "source-snapshot",
                "submission/source-snapshot",
            )
            self.repo_fd = self._open_directory(
                self.source_snapshot_fd,
                "repo",
                "submission/source-snapshot/repo",
            )
            parent = self.repo_fd
            prefix = ""
            self.package_directory_fds: dict[str, int] = {}
            for part in FROZEN_PACKAGE_RELATIVE.parts:
                prefix = f"{prefix}/{part}" if prefix else part
                parent = self._open_directory(
                    parent, part, f"source/{prefix}"
                )
                self.package_directory_fds[prefix] = parent
            self.package_fd = parent
            self.gate_fd = self._open_regular(
                self.package_fd, "gate.py", "source/package/gate.py"
            )
            self.manifest_fd = self._open_regular(
                self.package_fd, "manifest.json", "source/package/manifest.json"
            )
            try:
                self.journal_fd = self._open_directory(
                    self.submission_fd, "journal", "submission/journal"
                )
            except EngineeringPilotBindingError as exc:
                try:
                    os.stat(
                        "journal",
                        dir_fd=self.submission_fd,
                        follow_symlinks=False,
                    )
                except FileNotFoundError:
                    self.journal_fd = None
                except OSError:
                    raise exc
                else:
                    raise exc
            self.core_identity_by_label = dict(self._identity_by_label)
        except BaseException:
            self.close()
            raise

    def _retain(self, descriptor: int, label: str) -> int:
        _require(label not in self._descriptor_by_label, f"duplicate retained fd label: {label}")
        self._descriptors.append(descriptor)
        self._descriptor_by_label[label] = descriptor
        self._identity_by_label[label] = runtime._identity(os.fstat(descriptor))
        return descriptor

    def _open_absolute_chain(self) -> None:
        try:
            descriptor = os.open(os.sep, _DIRECTORY_OPEN_FLAGS)
        except OSError as exc:
            raise EngineeringPilotBindingError(
                f"cannot open absolute trust root: {exc}"
            ) from exc
        descriptor = self._retain(descriptor, "absolute:/")
        self.absolute_chain_labels.append("absolute:/")
        prefix = ""
        for part in tuple(
            component
            for component in str(self.submission_root).split(os.sep)
            if component
        ):
            prefix = f"{prefix}/{part}"
            child = _open_directory_at(
                descriptor, part, f"absolute submission chain {prefix}"
            )
            descriptor = self._retain(child, f"absolute:{prefix}")
            self.absolute_chain_labels.append(f"absolute:{prefix}")
        self.submission_fd = descriptor

    def _open_directory(self, parent_fd: int, name: str, label: str) -> int:
        return self._retain(_open_directory_at(parent_fd, name, label), label)

    def _open_regular(self, parent_fd: int, name: str, label: str) -> int:
        return self._retain(_open_regular_at(parent_fd, name, label), label)

    def open_report_file(self, name: str) -> int:
        if name in self.report_files:
            return self.report_files[name]
        descriptor = self._open_regular(
            self.report_fd, name, f"report/{name}"
        )
        self.report_files[name] = descriptor
        return descriptor

    def assert_retained_identities(self) -> None:
        for label, expected in self._identity_by_label.items():
            descriptor = self._descriptor_by_label[label]
            _require(
                runtime._identity(os.fstat(descriptor)) == expected,
                f"retained Launch8 trust object changed: {label}",
            )

    def close(self) -> None:
        while self._descriptors:
            descriptor = self._descriptors.pop()
            try:
                os.close(descriptor)
            except OSError:
                pass
        self._descriptor_by_label.clear()

    def __enter__(self) -> "_RetainedLaunch8Trust":
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: object,
    ) -> None:
        try:
            if exception_type is None:
                _check_cancel_latches(self)
                self.assert_retained_identities()
        finally:
            self.close()


def _expected_source_tree_contract() -> tuple[frozenset[str], frozenset[str], dict[str, int]]:
    """Return the frozen file, parent-directory, and link-count contract.

    The selected paths are themselves authenticated by the frozen aggregate hash
    below.  Deriving the parent set from those exact paths keeps empty or otherwise
    ignored directories outside the accepted namespace.
    """

    expected_files = frozenset(
        str(relative)
        for relative in runtime.launch8_verifier_dependency_relatives(
            runtime.REPOSITORY_ROOT
        )
    )
    _require(
        len(expected_files) == runtime.FROZEN_LAUNCH8_VERIFIER_SOURCE_FILE_COUNT,
        "frozen Launch8 expected file coverage differs",
    )
    directories: set[str] = set()
    for raw_relative in expected_files:
        relative = Path(raw_relative)
        for parent in relative.parents:
            if parent != Path("."):
                directories.add(str(parent))
    expected_directories = frozenset(directories)
    all_directories = {Path("."), *(Path(value) for value in expected_directories)}
    expected_nlinks = {
        str(directory): 2
        + sum(1 for child in all_directories if child != Path(".") and child.parent == directory)
        for directory in all_directories
    }
    return expected_files, expected_directories, expected_nlinks


def _exact_source_tree_inventory(source_root_fd: int) -> dict[str, Any]:
    """No-follow inventory of every entry in the sealed Launch8 source tree.

    Both semantic bytes and full filesystem identities are retained.  The latter
    intentionally includes inode, ctime, and directory metadata so an add/remove or
    replace/restore cycle during replay cannot disappear behind equal final bytes.
    """

    expected_files, expected_directories, expected_nlinks = (
        _expected_source_tree_contract()
    )
    expected_uid = os.getuid()
    expected_gid = os.getgid()
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    file_flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    directory_identities: dict[str, tuple[int, ...]] = {}
    file_rows: dict[str, dict[str, Any]] = {}

    def children(descriptor: int) -> dict[str, tuple[int, ...]]:
        with os.scandir(descriptor) as iterator:
            entries = sorted(
                (
                    entry.name,
                    entry.stat(follow_symlinks=False),
                )
                for entry in iterator
            )
        result: dict[str, tuple[int, ...]] = {}
        for name, info in entries:
            _require(
                name not in {"", ".", ".."} and "/" not in name,
                "frozen Launch8 source contains an unsafe entry name",
            )
            result[name] = runtime._identity(info)
        return result

    def validate_directory(info: os.stat_result, relative: str) -> None:
        key = relative or "."
        _require(
            stat.S_ISDIR(info.st_mode)
            and info.st_uid == expected_uid
            and info.st_gid == expected_gid
            and stat.S_IMODE(info.st_mode) == 0o555,
            f"frozen Launch8 source directory authority differs: {key}",
        )

    def visit(descriptor: int, prefix: str) -> None:
        before = os.fstat(descriptor)
        validate_directory(before, prefix)
        directory_identities[prefix or "."] = runtime._identity(before)
        first_children = children(descriptor)
        for name in sorted(first_children):
            relative = f"{prefix}/{name}" if prefix else name
            listed_identity = first_children[name]
            listed_mode = listed_identity[2]
            if stat.S_ISDIR(listed_mode):
                _require(
                    relative in expected_directories,
                    f"frozen Launch8 source contains an unexpected directory: {relative}",
                )
                child = os.open(name, directory_flags, dir_fd=descriptor)
                try:
                    opened = os.fstat(child)
                    _require(
                        runtime._identity(opened) == listed_identity,
                        f"frozen Launch8 source directory raced: {relative}",
                    )
                    visit(child, relative)
                finally:
                    os.close(child)
            elif stat.S_ISREG(listed_mode):
                _require(
                    relative in expected_files,
                    f"frozen Launch8 source contains an unexpected file: {relative}",
                )
                child = os.open(name, file_flags, dir_fd=descriptor)
                try:
                    opened = os.fstat(child)
                    _require(
                        runtime._identity(opened) == listed_identity
                        and opened.st_uid == expected_uid
                        and opened.st_gid == expected_gid
                        and opened.st_nlink == 1
                        and stat.S_IMODE(opened.st_mode) == 0o444,
                        f"frozen Launch8 source file authority differs: {relative}",
                    )
                    digest = hashlib.sha256()
                    while True:
                        block = os.read(child, 16 * 1024 * 1024)
                        if not block:
                            break
                        digest.update(block)
                    _require(
                        runtime._identity(os.fstat(child)) == listed_identity,
                        f"frozen Launch8 source file changed while hashing: {relative}",
                    )
                    file_rows[relative] = {
                        "identity": listed_identity,
                        "sha256": digest.hexdigest(),
                    }
                finally:
                    os.close(child)
            else:
                raise EngineeringPilotBindingError(
                    f"frozen Launch8 source contains a symlink or special entry: {relative}"
                )
        _require(
            before.st_nlink == expected_nlinks[prefix or "."],
            f"frozen Launch8 source directory link count differs: {prefix or '.'}",
        )
        _require(
            children(descriptor) == first_children
            and runtime._identity(os.fstat(descriptor)) == runtime._identity(before),
            f"frozen Launch8 source directory changed while scanning: {prefix or '.'}",
        )

    visit(source_root_fd, "")

    _require(
        set(file_rows) == expected_files,
        "frozen Launch8 source exact file set differs",
    )
    _require(
        set(directory_identities) == {".", *expected_directories},
        "frozen Launch8 source exact parent-directory set differs",
    )
    file_sha256 = {
        relative: file_rows[relative]["sha256"] for relative in sorted(file_rows)
    }
    _require(
        runtime.stable_hash(file_sha256)
        == runtime.FROZEN_LAUNCH8_VERIFIER_SOURCE_INVENTORY_SHA256,
        "frozen Launch8 source file set/hash differs from 33122e15",
    )
    for name, expected in runtime.FROZEN_LAUNCH8_ENTRYPOINT_SHA256.items():
        relative = str(runtime.FROZEN_LAUNCH8_PACKAGE_RELATIVE / name)
        _require(
            file_sha256.get(relative) == expected,
            f"frozen Launch8 source entry point differs: {name}",
        )
    return {
        "directory_identities": {
            relative: directory_identities[relative]
            for relative in sorted(directory_identities)
        },
        "files": {relative: file_rows[relative] for relative in sorted(file_rows)},
        "file_sha256": file_sha256,
    }


def _validate_raw_bundle_schema(
    bundle: Mapping[str, Any]
) -> list[Mapping[str, Any]]:
    _exact_keys(bundle, BUNDLE_KEYS, "report bundle")
    _require(bundle.get("schema_version") == 1, "report bundle schema differs")
    _require(
        bundle.get("campaign_id") == EXPECTED_ACCEPTED_CAMPAIGN_ID,
        "report bundle campaign differs",
    )
    cells = bundle.get("cells")
    _require(
        isinstance(cells, list) and len(cells) == 20,
        "report bundle cell coverage differs",
    )
    _require(
        [cell.get("index") for cell in cells if isinstance(cell, Mapping)]
        == list(range(20)),
        "raw report cell order/index coverage differs",
    )
    for index, raw_cell in enumerate(cells):
        cell = _exact_keys(raw_cell, RAW_CELL_KEYS, f"raw cell{index}")
        _require(
            type(cell.get("index")) is int and cell["index"] == index,
            f"raw cell{index} index differs",
        )
        _require(cell.get("fresh_start") is True, f"raw cell{index} is not fresh")
        boundaries = cell.get("boundaries")
        _require(
            isinstance(boundaries, Mapping)
            and set(boundaries) == {"5000", "25000"},
            f"raw cell{index} boundary coverage differs",
        )
        for target in (5_000, 25_000):
            boundary = boundaries[str(target)]
            expected = RAW_BOUNDARY_KEYS | (
                {"outcome"} if target == 25_000 else set()
            )
            _exact_keys(boundary, frozenset(expected), f"raw cell{index}@{target}")
            _require(
                type(boundary.get("update")) is int
                and boundary["update"] == target,
                f"raw cell{index}@{target} update differs",
            )
            _exact_keys(
                boundary.get("prefix_contract"),
                PREFIX_CONTRACT_KEYS,
                f"raw cell{index}@{target} prefix contract",
            )
            scalars = boundary.get("scalars")
            _require(
                isinstance(scalars, Mapping) and bool(scalars),
                f"raw cell{index}@{target} scalars missing",
            )
            for tag, points in scalars.items():
                _require(
                    isinstance(tag, str) and bool(tag),
                    f"raw cell{index}@{target} scalar tag differs",
                )
                _require(
                    isinstance(points, list),
                    f"raw cell{index}@{target}:{tag} axis differs",
                )
                prior = -1
                for point in points:
                    _require(
                        isinstance(point, list)
                        and len(point) == 2
                        and type(point[0]) is int
                        and point[0] > prior,
                        f"raw cell{index}@{target}:{tag} axis order differs",
                    )
                    prior = point[0]
            if target == 25_000:
                outcome = _exact_keys(
                    boundary.get("outcome"),
                    RAW_OUTCOME_KEYS,
                    f"raw cell{index} outcome",
                )
                rows = outcome.get("completed_results")
                _require(
                    isinstance(rows, list) and len(rows) == 25,
                    f"raw cell{index} completed-result coverage differs",
                )
                for episode_index, row in enumerate(rows):
                    _exact_keys(
                        row,
                        EPISODE_RESULT_KEYS,
                        f"raw cell{index} episode{episode_index}",
                    )
    return cells


def _validate_gate_schema(decision: Mapping[str, Any]) -> None:
    _exact_keys(decision, DECISION_KEYS, "gate decision")
    _require(
        decision.get("schema_version") == 1
        and decision.get("campaign_id") == EXPECTED_ACCEPTED_CAMPAIGN_ID
        and decision.get("status") == "accepted_engineering_pilot"
        and decision.get("formal_validation") is False
        and decision.get("supports_1m_claim") is False,
        "gate decision identity/status differs",
    )
    _exact_keys(decision.get("gates"), TOP_GATE_KEYS, "top gate set")
    _require(
        all(value is True for value in decision["gates"].values()),
        "not every recomputed top gate passed",
    )
    paired = decision.get("paired_25k_calibration")
    _require(isinstance(paired, Mapping), "paired calibration is absent")
    _exact_keys(
        paired.get("gates"),
        PAIRED_CALIBRATION_GATE_KEYS,
        "paired calibration gate set",
    )
    _require(
        paired.get("passed") is True
        and all(value is True for value in paired["gates"].values()),
        "recomputed paired calibration did not pass",
    )
    outcomes = decision.get("outcomes_25000")
    _require(isinstance(outcomes, Mapping), "normalized outcomes are absent")
    _exact_keys(
        outcomes.get("absolute_gates"),
        ABSOLUTE_OUTCOME_GATE_KEYS,
        "absolute outcome gate set",
    )
    _exact_keys(
        outcomes.get("paired_gates"),
        PAIRED_OUTCOME_GATE_KEYS,
        "paired outcome gate set",
    )
    _require(
        outcomes.get("passed") is True
        and all(value is True for value in outcomes["absolute_gates"].values())
        and all(value is True for value in outcomes["paired_gates"].values()),
        "recomputed absolute/paired outcomes did not pass",
    )
    cells = decision.get("cells")
    _require(
        isinstance(cells, list) and len(cells) == 20,
        "normalized gate cell coverage differs",
    )
    _require(
        [cell.get("index") for cell in cells if isinstance(cell, Mapping)]
        == list(range(20)),
        "normalized gate index coverage differs",
    )
    for raw_cell in cells:
        cell = _exact_keys(raw_cell, NORMALIZED_CELL_KEYS, "normalized gate cell")
        boundaries = cell.get("boundaries")
        _require(
            isinstance(boundaries, Mapping)
            and set(boundaries) == {"5000", "25000"},
            "normalized gate boundary coverage differs",
        )
        for target in ("5000", "25000"):
            boundary = _exact_keys(
                boundaries[target],
                NORMALIZED_BOUNDARY_KEYS,
                f"normalized boundary {target}",
            )
            _exact_keys(
                boundary.get("structural_gates"),
                STRUCTURAL_GATE_KEYS,
                f"normalized structural gates {target}",
            )
            _exact_keys(
                boundary.get("method_gates"),
                METHOD_GATE_KEYS,
                f"normalized method gates {target}",
            )
            _exact_keys(
                boundary.get("candidate_gates"),
                CANDIDATE_GATE_KEYS,
                f"normalized candidate gates {target}",
            )
            _require(
                boundary.get("structural_passed") is True,
                f"normalized structural boundary {target} failed",
            )
            if cell.get("arm_id") == "GSEP":
                _require(
                    boundary.get("candidate_passed") is True
                    and all(
                        value is True
                        for value in boundary["candidate_gates"].values()
                    ),
                    f"normalized candidate boundary {target} failed",
                )
            else:
                _require(
                    cell.get("arm_id") == "GS"
                    and boundary.get("candidate_passed") is False,
                    "normalized arm/candidate semantics differ",
                )


def _expected_production_authorization(
    manifest: Mapping[str, Any]
) -> dict[str, Any]:
    canary = manifest["launch_contract"]["real_gpu_two_wave_canary"]
    evidence = canary["production_authorization_evidence"]
    attempts = canary["accepted_attempts"]
    _require(
        isinstance(attempts, list)
        and len(attempts) == 1
        and isinstance(attempts[0], Mapping),
        "frozen successful-canary attempt differs",
    )
    attempt = attempts[0]
    roles = attempt.get("job_ids_by_role")
    _require(
        isinstance(roles, Mapping)
        and set(roles) == {"wave0", "wave1", "report"},
        "frozen successful-canary roles differ",
    )
    normalized_roles: dict[str, list[str]] = {}
    for role in ("wave0", "wave1", "report"):
        values = roles[role]
        _require(
            isinstance(values, list)
            and len(values) == 1
            and _positive_int_string(values[0]),
            f"frozen successful-canary {role} ID differs",
        )
        normalized_roles[role] = list(values)
    return {
        "schema_version": 1,
        "status": "canary2_production_authorization_prerequisite_satisfied",
        "attempt": "canary2",
        "path": "canary2_acceptance_provenance.json",
        "raw_sha256": evidence["raw_sha256"],
        "canonical_sha256": evidence["canonical_sha256"],
        "report_raw_sha256": evidence["report_raw_sha256"],
        "source_protocol_sha256": evidence["source_protocol_sha256"],
        "source_commit": attempt["source_commit"],
        "state_root": attempt["state_root"],
        "state_file_map_canonical_sha256": attempt[
            "state_file_map_canonical_sha256"
        ],
        "canary_token": attempt["canary_token"],
        "job_ids_by_role": normalized_roles,
        "accepted_attempt_sha256": runtime.stable_hash(attempt),
        "production_authorization_evidence_sha256": runtime.stable_hash(evidence),
        "sealed_package_protocol_sha256": runtime.FROZEN_LAUNCH8_PROTOCOL_SHA256,
    }


def _validate_event_provenance(rows: object) -> None:
    _require(
        isinstance(rows, list) and len(rows) == 20,
        "event provenance coverage differs",
    )
    _require(
        [row.get("index") for row in rows if isinstance(row, Mapping)]
        == list(range(20)),
        "event provenance index join differs",
    )
    for index, raw_row in enumerate(rows):
        row = _exact_keys(
            raw_row, EVENT_PROVENANCE_KEYS, f"event provenance cell{index}"
        )
        for names_key, hashes_key, hparams in (
            ("event_files", "event_file_sha256", False),
            ("hparams_event_files", "hparams_event_file_sha256", True),
        ):
            names = row.get(names_key)
            hashes = row.get(hashes_key)
            _require(
                isinstance(names, list) and names == sorted(set(names)),
                f"event provenance cell{index} {names_key} differs",
            )
            _require(
                isinstance(hashes, Mapping) and set(hashes) == set(names),
                f"event provenance cell{index} {hashes_key} join differs",
            )
            for name in names:
                relative = runtime.safe_relative(
                    name, f"event provenance cell{index} path"
                )
                _require(
                    (relative.parts[0] == "hparams") is hparams,
                    f"event provenance cell{index} path class differs",
                )
                _require(
                    _sha256_string(hashes[name]),
                    f"event provenance cell{index} hash differs",
                )
        _require(
            bool(row["event_files"]),
            f"event provenance cell{index} has no training event file",
        )
        _require(
            type(row.get("fixed_validation_text_events")) is int
            and row["fixed_validation_text_events"] == len(row["event_files"]),
            f"event provenance cell{index} fixed-validation text coverage differs",
        )
        excluded = row.get("excluded_eval_tags")
        _require(
            isinstance(excluded, list)
            and excluded == sorted(set(excluded))
            and all(
                isinstance(tag, str) and tag.startswith("eval/") for tag in excluded
            ),
            f"event provenance cell{index} excluded-eval schema differs",
        )
        duplicates = row.get("identical_scalar_duplicates")
        _require(
            isinstance(duplicates, Mapping)
            and all(
                isinstance(tag, str)
                and bool(tag)
                and type(count) is int
                and count > 0
                for tag, count in duplicates.items()
            ),
            f"event provenance cell{index} duplicate inventory differs",
        )


def _validate_terminal_provenance(
    rows: object, raw_cells: list[Mapping[str, Any]]
) -> None:
    _require(
        isinstance(rows, list) and len(rows) == 20,
        "terminal provenance coverage differs",
    )
    _require(
        [row.get("index") for row in rows if isinstance(row, Mapping)]
        == list(range(20)),
        "terminal provenance index join differs",
    )
    job_ids_by_wave: dict[int, str] = {}
    for index, raw_row in enumerate(rows):
        row = _exact_keys(
            raw_row,
            TERMINAL_PROVENANCE_KEYS,
            f"terminal provenance cell{index}",
        )
        outcome = raw_cells[index]["boundaries"]["25000"]["outcome"]
        for key in (
            "completed_results_sha256", "completion_sha256",
            "final_eval_progress_sha256", "checkpoint_sha256",
        ):
            _require(
                _sha256_string(row.get(key)) and row[key] == outcome[key],
                f"terminal provenance cell{index} {key} join differs",
            )
        _require(
            _sha256_string(row.get("worker_complete_sha256"))
            and _sha256_string(row.get("identity_sha256")),
            f"terminal provenance cell{index} receipt identity differs",
        )
        wave = row.get("wave_index")
        job_id = row.get("array_job_id")
        _require(
            type(wave) is int and wave in {0, 1} and _positive_int_string(job_id),
            f"terminal provenance cell{index} wave identity differs",
        )
        prior = job_ids_by_wave.setdefault(wave, job_id)
        _require(
            prior == job_id,
            f"terminal provenance wave{wave} array job ID is inconsistent",
        )
        lineage = _exact_keys(
            row.get("wave_lineage"),
            WAVE_LINEAGE_KEYS,
            f"terminal provenance cell{index} lineage",
        )
        for key in (
            "wave0_start_sha256", "wave0_predecessor_evidence_sha256",
            "wave0_checkpoint_sha256", "wave1_start_sha256",
            "wave1_input_checkpoint_sha256",
            "wave1_predecessor_evidence_sha256",
        ):
            _require(
                _sha256_string(lineage.get(key)),
                f"terminal provenance cell{index} lineage {key} differs",
            )
        _require(
            lineage["wave1_input_checkpoint_sha256"]
            == lineage["wave0_checkpoint_sha256"]
            and lineage["wave1_predecessor_evidence_sha256"]
            == lineage["wave0_predecessor_evidence_sha256"],
            f"terminal provenance cell{index} predecessor join differs",
        )
        if lineage.get("branch") == "wave0_complete_wave1_noop":
            _require(
                wave == 0
                and lineage.get("wave0_predecessor_evidence_name")
                == "WORKER_COMPLETE.json"
                and lineage.get("wave0_predecessor_evidence_sha256")
                == row["worker_complete_sha256"]
                and lineage.get("wave0_checkpoint_sha256")
                == row["checkpoint_sha256"]
                and _sha256_string(lineage.get("wave1_noop_sha256")),
                f"terminal provenance cell{index} wave-zero/no-op lineage differs",
            )
        else:
            _require(
                lineage.get("branch") == "wave0_ready_wave1_resume"
                and wave == 1
                and lineage.get("wave0_predecessor_evidence_name")
                == "CONTINUATION_READY.json"
                and lineage.get("wave1_noop_sha256") is None,
                f"terminal provenance cell{index} wave-one/resume lineage differs",
            )


def _run_frozen_gate(
    *,
    repo_fd: int,
    package_fd: int,
    gate_fd: int,
    bundle_fd: int,
    manifest_fd: int,
) -> dict[str, Any]:
    procfd = Path("/proc/self/fd")
    command = [
        sys.executable, "-I", "-S", "-B",
        str(procfd / str(package_fd) / "gate.py"),
        "--report", str(procfd / str(bundle_fd)),
        "--manifest", str(procfd / str(manifest_fd)),
    ]
    environment = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    try:
        result = subprocess.run(
            command,
            cwd=str(procfd / str(repo_fd)),
            env=environment,
            check=False,
            shell=False,
            close_fds=True,
            pass_fds=tuple(
                sorted({repo_fd, package_fd, gate_fd, bundle_fd, manifest_fd})
            ),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EngineeringPilotBindingError(
            f"frozen Launch8 gate could not run: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise EngineeringPilotBindingError(
            "frozen Launch8 gate did not accept the raw bundle"
            + (f": {detail}" if detail else "")
        )
    try:
        return runtime.parse_json_bytes(result.stdout, "frozen Launch8 gate stdout")
    except runtime.RuntimeContractError as exc:
        raise EngineeringPilotBindingError(str(exc)) from exc


def adapter_description() -> dict[str, Any]:
    return runtime.engineering_pilot_adapter_description()


def _check_cancel_latches(trust: _RetainedLaunch8Trust) -> None:
    _entry_absent_at(
        trust.submission_fd,
        "CANCEL_REQUESTED.json",
        "durable cancellation latch",
    )
    if trust.journal_fd is None:
        _entry_absent_at(
            trust.submission_fd,
            "journal",
            "previously absent submission journal parent",
        )
        return
    for name, label in (
        ("PREREQUISITE_MISSING.json", "prerequisite-missing journal"),
        ("9000_RECOVERY_CANCELLED.json", "recovery-cancelled journal"),
        (
            "9001_PRODUCTION_PREREQUISITE_MISSING.json",
            "production-prerequisite-missing journal",
        ),
    ):
        _entry_absent_at(trust.journal_fd, name, label)


def _join_source_handles(
    trust: _RetainedLaunch8Trust,
    source_tree: Mapping[str, Any],
) -> None:
    directories = source_tree["directory_identities"]
    files = source_tree["files"]
    _require(
        directories.get(".") == runtime._identity(os.fstat(trust.repo_fd)),
        "retained Launch8 repository root is not the scanned root",
    )
    for relative, descriptor in trust.package_directory_fds.items():
        _require(
            directories.get(relative) == runtime._identity(os.fstat(descriptor)),
            f"retained Launch8 source directory join differs: {relative}",
        )
    for name, descriptor in (
        ("gate.py", trust.gate_fd),
        ("manifest.json", trust.manifest_fd),
    ):
        relative = str(FROZEN_PACKAGE_RELATIVE / name)
        _require(
            files.get(relative, {}).get("identity")
            == runtime._identity(os.fstat(descriptor)),
            f"retained Launch8 source file join differs: {relative}",
        )


def _verify_fresh_lexical_bindings(
    trust: _RetainedLaunch8Trust,
    quartet_names: set[str],
) -> None:
    """Reopen every lexical authority root and join it to the retained objects."""

    with _RetainedLaunch8Trust(trust.submission_root) as fresh:
        _check_cancel_latches(fresh)
        _require(
            fresh.core_identity_by_label == trust.core_identity_by_label,
            "lexical Launch8 trust chain no longer resolves to retained objects",
        )
        fresh_tree = _report_tree_entries(fresh.report_fd)
        retained_tree = _report_tree_entries(trust.report_fd)
        _require(
            fresh_tree == retained_tree
            and set(fresh_tree["entries"]) == quartet_names,
            "lexical Launch8 report root no longer resolves to retained quartet",
        )
        for name in sorted(quartet_names):
            fresh_descriptor = fresh.open_report_file(name)
            retained_descriptor = trust.report_files[name]
            _require(
                runtime._identity(os.fstat(fresh_descriptor))
                == runtime._identity(os.fstat(retained_descriptor)),
                f"lexical Launch8 quartet entry is not retained object: {name}",
            )
        fresh.assert_retained_identities()


def _verify_engineering_pilot_report_quartet_retained(
    trust: _RetainedLaunch8Trust,
    *,
    report_root: Path,
    submission_root: Path,
    expected_submission_sha256: str,
    expected_package_binding: Mapping[str, str],
) -> dict[str, Any]:
    _check_cancel_latches(trust)

    source_tree_before = _exact_source_tree_inventory(trust.repo_fd)
    _join_source_handles(trust, source_tree_before)
    source_inventory_before = source_tree_before["file_sha256"]
    manifest, manifest_file_sha, manifest_info, manifest_payload = (
        _read_immutable_pretty_json_fd(
            trust.manifest_fd,
            "frozen Launch8 manifest",
            require_pretty=False,
        )
    )
    gate_payload, gate_file_sha, gate_info = _read_regular_fd(
        trust.gate_fd, "frozen Launch8 gate source"
    )
    _require(
        manifest.get("schema_version") == 1
        and manifest.get("campaign_id") == EXPECTED_ACCEPTED_CAMPAIGN_ID
        and manifest.get("status") == "sealed_launch_ready_unsubmitted",
        "frozen Launch8 campaign schema/status differs",
    )

    tree_before = _report_tree_entries(trust.report_fd)
    commit_fd = trust.open_report_file("REPORT_COMMIT.json")
    _require(
        runtime._identity(os.fstat(commit_fd))
        == tree_before["entries"].get("REPORT_COMMIT.json"),
        "retained Launch8 report commit is not the inventoried entry",
    )
    commit, commit_file_sha, commit_info, commit_payload = (
        _read_immutable_pretty_json_fd(
            commit_fd, "Launch8 report commit"
        )
    )
    _exact_keys(commit, REPORT_COMMIT_KEYS, "Launch8 report commit")
    _require(
        type(commit.get("schema_version")) is int
        and commit.get("schema_version") == 1
        and commit.get("status") == "accepted_engineering_pilot"
        and commit.get("scientific_rejection") is False
        and commit.get("campaign_id") == EXPECTED_ACCEPTED_CAMPAIGN_ID
        and commit.get("submission_sha256") == expected_submission_sha256,
        "Launch8 report commit identity/status differs",
    )
    for key in (
        "report_bundle_sha256", "report_bundle_file_sha256", "gate_sha256",
        "gate_decision_file_sha256", "provenance_sha256",
        "provenance_file_sha256",
    ):
        _require(
            _sha256_string(commit.get(key)),
            f"Launch8 report commit {key} differs",
        )
    expected_names = {
        "REPORT_COMMIT.json",
        f"REPORT_BUNDLE.{commit['report_bundle_sha256']}.json",
        f"GATE_DECISION.{commit['gate_sha256']}.json",
        f"REPORT_PROVENANCE.{commit['provenance_sha256']}.json",
    }
    _require(
        commit.get("report_bundle")
        == f"REPORT_BUNDLE.{commit['report_bundle_sha256']}.json"
        and commit.get("gate_decision")
        == f"GATE_DECISION.{commit['gate_sha256']}.json"
        and commit.get("provenance")
        == f"REPORT_PROVENANCE.{commit['provenance_sha256']}.json"
        and set(tree_before["entries"]) == expected_names,
        "Launch8 report quartet names/coverage differ",
    )

    bundle_fd = trust.open_report_file(commit["report_bundle"])
    decision_fd = trust.open_report_file(commit["gate_decision"])
    provenance_fd = trust.open_report_file(commit["provenance"])
    for name, descriptor in trust.report_files.items():
        _require(
            runtime._identity(os.fstat(descriptor))
            == tree_before["entries"].get(name),
            f"retained Launch8 quartet file is not inventoried entry: {name}",
        )
    bundle, bundle_file_sha, bundle_info, bundle_payload = (
        _read_immutable_pretty_json_fd(
            bundle_fd, "Launch8 raw report bundle"
        )
    )
    decision, decision_file_sha, decision_info, decision_payload = (
        _read_immutable_pretty_json_fd(
            decision_fd, "Launch8 gate decision"
        )
    )
    provenance, provenance_file_sha, provenance_info, provenance_payload = (
        _read_immutable_pretty_json_fd(
            provenance_fd, "Launch8 report provenance"
        )
    )
    quartet_snapshots = {
        "REPORT_COMMIT.json": (commit_payload, commit_file_sha, runtime._identity(commit_info)),
        commit["report_bundle"]: (
            bundle_payload, bundle_file_sha, runtime._identity(bundle_info)
        ),
        commit["gate_decision"]: (
            decision_payload, decision_file_sha, runtime._identity(decision_info)
        ),
        commit["provenance"]: (
            provenance_payload, provenance_file_sha, runtime._identity(provenance_info)
        ),
    }
    _require(
        bundle_file_sha == commit["report_bundle_file_sha256"]
        and decision_file_sha == commit["gate_decision_file_sha256"]
        and provenance_file_sha == commit["provenance_file_sha256"],
        "Launch8 quartet raw file hash join differs",
    )
    _require(
        runtime.stable_hash(bundle) == commit["report_bundle_sha256"]
        and runtime.stable_hash(provenance) == commit["provenance_sha256"],
        "Launch8 quartet semantic hash join differs",
    )
    decision_body = dict(decision)
    embedded_gate_sha = decision_body.pop("gate_sha256", None)
    _require(
        embedded_gate_sha == commit["gate_sha256"]
        and runtime.stable_hash(decision_body) == commit["gate_sha256"],
        "Launch8 decision self-hash differs",
    )

    raw_cells = _validate_raw_bundle_schema(bundle)
    for key in PACKAGE_BINDING_KEYS:
        _require(
            bundle.get(key) == expected_package_binding[key],
            f"raw bundle package binding differs: {key}",
        )
    normalized = _run_frozen_gate(
        repo_fd=trust.repo_fd,
        package_fd=trust.package_fd,
        gate_fd=trust.gate_fd,
        bundle_fd=bundle_fd,
        manifest_fd=trust.manifest_fd,
    )
    _require(
        normalized == decision,
        "published gate decision differs from frozen raw-bundle recomputation",
    )
    _validate_gate_schema(normalized)
    _require(
        normalized.get("report_bundle_sha256")
        == commit["report_bundle_sha256"]
        and normalized.get("gate_sha256") == commit["gate_sha256"]
        and normalized.get("package_binding") == expected_package_binding
        and normalized.get("manifest_sha256")
        == expected_package_binding["manifest_sha256"],
        "recomputed gate/package/report join differs",
    )

    _exact_keys(provenance, PROVENANCE_KEYS, "Launch8 report provenance")
    _require(
        provenance.get("schema_version") == 1
        and provenance.get("campaign_id") == EXPECTED_ACCEPTED_CAMPAIGN_ID
        and provenance.get("submission_sha256") == expected_submission_sha256
        and provenance.get("report_bundle_sha256")
        == commit["report_bundle_sha256"]
        and provenance.get("gate_sha256") == commit["gate_sha256"],
        "Launch8 provenance identity/quartet join differs",
    )
    production = _exact_keys(
        provenance.get("production_authorization_prerequisite"),
        PRODUCTION_AUTHORIZATION_KEYS,
        "Launch8 production authorization prerequisite",
    )
    expected_production = _expected_production_authorization(manifest)
    _require(
        dict(production) == expected_production,
        "Launch8 production authorization source binding differs",
    )
    _require(
        provenance.get("production_authorization_prerequisite_sha256")
        == runtime.stable_hash(production),
        "Launch8 production authorization hash differs",
    )
    phase = _exact_keys(
        provenance.get("outcome_blind_phase"),
        OUTCOME_BLIND_PHASE_KEYS,
        "Launch8 outcome-blind phase",
    )
    boundary_evaluations = [
        {
            "index": cell["index"],
            "setting_id": cell["setting_id"],
            "arm_id": cell["arm_id"],
            "seed": cell["seed"],
            "boundaries": cell["boundaries"],
        }
        for cell in normalized["cells"]
    ]
    _require(
        phase.get("status")
        == "all_boundaries_parsed_and_calibration_computed_before_outcomes"
        and phase.get("boundary_evaluations_sha256")
        == runtime.stable_hash(boundary_evaluations)
        and phase.get("paired_calibration_sha256")
        == runtime.stable_hash(normalized["paired_25k_calibration"]),
        "Launch8 outcome-blind telemetry-to-decision phase hashes differ",
    )
    _validate_event_provenance(provenance.get("event_artifacts"))
    _validate_terminal_provenance(
        provenance.get("terminal_artifacts"), raw_cells
    )

    tree_after = _report_tree_entries(trust.report_fd)
    source_tree_after = _exact_source_tree_inventory(trust.repo_fd)
    _join_source_handles(trust, source_tree_after)
    _require(
        tree_after == tree_before,
        "Launch8 report quartet changed during verification",
    )
    _require(
        source_tree_after == source_tree_before,
        "frozen Launch8 verifier source tree changed during gate replay",
    )
    for name, (expected_payload, expected_digest, expected_identity) in (
        quartet_snapshots.items()
    ):
        payload, digest, info = _read_regular_fd(
            trust.report_files[name], f"post-replay report/{name}"
        )
        _require(
            payload == expected_payload
            and digest == expected_digest
            and runtime._identity(info) == expected_identity,
            f"retained Launch8 quartet contents/identity changed: {name}",
        )
    manifest_after, manifest_sha_after, manifest_info_after = _read_regular_fd(
        trust.manifest_fd, "post-replay frozen Launch8 manifest"
    )
    gate_after, gate_sha_after, gate_info_after = _read_regular_fd(
        trust.gate_fd, "post-replay frozen Launch8 gate source"
    )
    _require(
        manifest_after == manifest_payload
        and manifest_sha_after == manifest_file_sha
        and runtime._identity(manifest_info_after) == runtime._identity(manifest_info)
        and gate_after == gate_payload
        and gate_sha_after == gate_file_sha
        and runtime._identity(gate_info_after) == runtime._identity(gate_info),
        "retained Launch8 manifest/gate changed during replay",
    )

    adapter_payload, adapter_file_sha, _adapter_info = (
        runtime.authenticated_regular_bytes(
            PACKAGE_DIR / "engineering_pilot_binder.py",
            "Exp24 engineering-pilot adapter",
        )
    )
    runtime_payload, adapter_runtime_file_sha, _runtime_info = (
        runtime.authenticated_regular_bytes(
            PACKAGE_DIR / "runtime.py", "Exp24 adapter runtime"
        )
    )
    _require(
        bool(adapter_payload) and bool(runtime_payload),
        "Exp24 adapter dependency is empty",
    )
    result = {
        "schema_version": 1,
        "status": "accepted_engineering_pilot_semantics_verified_unpublished",
        "campaign_id": EXPECTED_ACCEPTED_CAMPAIGN_ID,
        "binding_state": POSITIVE_BINDING_STATE,
        "formal_submission_allowed": False,
        "source_commit": FROZEN_SOURCE_COMMIT,
        "source_protocol_sha256": runtime.FROZEN_LAUNCH8_PROTOCOL_SHA256,
        "source_inventory_sha256": runtime.stable_hash(source_inventory_before),
        "source_file_count": len(source_inventory_before),
        "submission_root": str(submission_root),
        "submission_sha256": expected_submission_sha256,
        "report_root": str(report_root),
        "report_commit_file_sha256": commit_file_sha,
        "report_bundle_sha256": commit["report_bundle_sha256"],
        "gate_sha256": commit["gate_sha256"],
        "provenance_sha256": commit["provenance_sha256"],
        "adapter_file_sha256": adapter_file_sha,
        "adapter_runtime_file_sha256": adapter_runtime_file_sha,
        "adapter_description_sha256": runtime.stable_hash(adapter_description()),
        "recomputed_top_gates": dict(normalized["gates"]),
        "recomputed_paired_calibration_gates": dict(
            normalized["paired_25k_calibration"]["gates"]
        ),
        "recomputed_absolute_outcome_gates": dict(
            normalized["outcomes_25000"]["absolute_gates"]
        ),
        "recomputed_paired_outcome_gates": dict(
            normalized["outcomes_25000"]["paired_gates"]
        ),
        "persistent_writes_performed": False,
        "binding_published": False,
    }
    result["verification_sha256"] = runtime.stable_hash(result)
    _check_cancel_latches(trust)
    trust.assert_retained_identities()
    _verify_fresh_lexical_bindings(trust, expected_names)
    _check_cancel_latches(trust)
    trust.assert_retained_identities()
    return result


def verify_engineering_pilot_report_quartet(
    report_root: Path,
    *,
    expected_report_root: Path,
    expected_submission_root: Path,
    expected_submission_sha256: str,
    expected_package_binding: Mapping[str, str],
) -> dict[str, Any]:
    """Authenticate and replay one caller-selected immutable accepted quartet.

    This function is read-only. Its return value is deliberately not the checked-in
    positive-binding schema and cannot authorize Exp24 submission.
    """

    report_root = _normalized_absolute(report_root, "report root")
    expected_report_root = _normalized_absolute(
        expected_report_root, "expected report root"
    )
    submission_root = _normalized_absolute(
        expected_submission_root, "expected submission root"
    )
    _require(
        report_root == expected_report_root == submission_root / "report",
        "report/submission root identity differs",
    )
    _require(
        _sha256_string(expected_submission_sha256),
        "expected submission SHA256 is malformed",
    )
    _exact_keys(
        expected_package_binding,
        PACKAGE_BINDING_KEYS,
        "expected frozen package binding",
    )
    _require(
        dict(expected_package_binding) == FROZEN_PACKAGE_BINDING,
        "expected package binding differs from frozen Launch8",
    )

    with _RetainedLaunch8Trust(submission_root) as trust:
        return _verify_engineering_pilot_report_quartet_retained(
            trust,
            report_root=report_root,
            submission_root=submission_root,
            expected_submission_sha256=expected_submission_sha256,
            expected_package_binding=expected_package_binding,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args(argv)
    if not args.describe:
        raise EngineeringPilotBindingError(
            "the CLI is description-only; call the read-only verifier with explicit expected roots"
        )
    print(runtime.canonical_json(adapter_description()))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (EngineeringPilotBindingError, runtime.RuntimeContractError) as exc:
        print(f"EXP24_ENGINEERING_PILOT_BINDER_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
