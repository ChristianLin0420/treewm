#!/usr/bin/env python3
"""Assemble, gate, and atomically publish the terminal Exp23 report.

The default/``--test-only`` action is read-only.  ``report.slurm`` uses the explicit
``--publish`` action after the complete twenty-cell array succeeds.  A separately
authorized, append-only engineering repair may use ``--publish-repair``; it never
changes the scientific assembly or gate inputs.
"""

from __future__ import annotations

import argparse
import base64
import ctypes
import errno
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import tempfile
import time
from types import ModuleType
from typing import Any, Mapping, Sequence


sys.dont_write_bytecode = True
PACKAGE_RELATIVE = Path("experiments/23-treewm-executable-prefix-repair-pilot-v1")
PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]
CAMPAIGN_ID = "treewm-executable-prefix-repair-pilot-v1-launch8"
BOUNDARIES = (5_000, 25_000)
SHA256 = frozenset("0123456789abcdef")
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
WORKER_COMPLETE_KEYS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "submission_sha256",
        "launch_sha256",
        "cell_index",
        "wave_index",
        "array_job_id",
        "array_task_id",
        "predecessor_array_job_id",
        "submission_authorization_sha256",
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
ARTIFACT_BASE_KEYS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "submission_sha256",
        "launch_sha256",
        "cell_index",
        "wave_index",
        "array_job_id",
        "array_task_id",
        "predecessor_array_job_id",
        "submission_authorization_sha256",
    }
)
GENERATION_START_KEYS = ARTIFACT_BASE_KEYS | frozenset(
    {
        "status",
        "input_kind",
        "input_checkpoint_sha256",
        "predecessor_evidence_sha256",
    }
)
CONTINUATION_READY_KEYS = ARTIFACT_BASE_KEYS | frozenset(
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
REPORT_REPAIR_AUTHORIZATION_KEYS = frozenset(
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
REPORT_REPAIR_SOURCE_AUTHORITY_NAME = "SOURCE_AUTHORITY.json"
REPORT_REPAIR_SUBMIT_CALLING_KEYS = frozenset(
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
REPORT_REPAIR_RELEASE_KEYS = frozenset(
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
REPORT_REPAIR_RELEASE_CALLING_KEYS = frozenset(
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
REPORT_REPAIR_RELEASE_RESULT_KEYS = frozenset(
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
REPORT_REPAIR_FAILURE_KEYS = frozenset(
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
EXPECTED_REPAIR_REASSEMBLY = {
    "schema_version": 1,
    "status": "rejected",
    "report_bundle_sha256": "b9102090021c103fa2362663d1a51310d239d50223108dba0106758b199d9b83",
    "gate_sha256": "d41b37f6806c77f15557ecd0329596da8385c02db5b06cecfb29247bb5f4682a",
    "report_bundle_file_sha256": "1a72e7968c5bc1639845eb18a64584db2204310c70c6301cdcccf804f576f139",
    "gate_decision_file_sha256": "53a7af1c91e4b09b8a04fdab7c1c0192d2076a88eb495855d9eafe39601f64b6",
    "original_provenance_v1_file_sha256": "3e99d102d6f5faa92699fb9bed4e1607e00a08349f03107048153c8d0764e858",
    "original_provenance_v1_sha256": "3fca5a3893cfd2e948f922438ee57bcc03e7763cfdb615500429700153820f77",
    "report_bundle_file_size": 424_013_704,
    "gate_decision_file_size": 704_147,
    "original_provenance_v1_file_size": 236_577,
    "worker_marker_aggregate_sha256": "ab1ced2e9b736edede8e1353297682feb800865f03da0c25b681208ce7d8cfc8",
    "deterministic_reassembly_allowed": True,
    "scientific_input_change_allowed": False,
    "gate_change_allowed": False,
}


class ReportError(RuntimeError):
    """An engineering artifact is absent, ambiguous, unsafe, or inconsistent."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReportError(message)


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
        raise ReportError(
            f"cannot determine whether {label or source} exists: {exc}"
        ) from exc


def _lexical_exists(path: str | Path, label: str | None = None) -> bool:
    return _lstat_if_present(path, label) is not None


def _durable_original_cleanup_prefix_exists(submission_root: Path) -> bool:
    journal = submission_root / "journal"
    return any(
        _lexical_exists(path)
        for path in (
            submission_root / "CANCEL_REQUESTED.json",
            journal / "PREREQUISITE_MISSING.json",
            journal / "9000_RECOVERY_CANCELLED.json",
            journal / "9001_PRODUCTION_PREREQUISITE_MISSING.json",
        )
    )


def _durable_repair_stop_prefix_exists(submission_root: Path) -> bool:
    journal = submission_root / "journal"
    try:
        names = {
            entry.name
            for entry in os.scandir(journal)
            if entry.name.startswith("REPORT_REPAIR_")
            or entry.name.startswith("CALLING_REPORT_REPAIR_")
        }
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ReportError(f"cannot inventory durable cleanup prefix: {exc}") from exc
    allowed_positive = {
        "REPORT_REPAIR_0001_ORIGINAL_FAILURE.json",
        "CALLING_REPORT_REPAIR_0001_SUBMIT.json",
        "REPORT_REPAIR_0001_SUBMITTED.json",
        "REPORT_REPAIR_0001_AUTHORIZED.json",
        "REPORT_REPAIR_0001_RELEASED.json",
    }
    allowed_positive_patterns = (
        re.compile(r"CALLING_REPORT_REPAIR_0001_RELEASE_[0-9]{4}\.json\Z"),
        re.compile(r"REPORT_REPAIR_0001_RELEASE_RESULT_[0-9]{4}\.json\Z"),
    )
    if any(
        name not in allowed_positive
        and not any(pattern.fullmatch(name) for pattern in allowed_positive_patterns)
        for name in names
    ):
        return True
    release_result_pattern = allowed_positive_patterns[1]
    for name in sorted(names):
        if not release_result_pattern.fullmatch(name):
            continue
        try:
            value = read_json(journal / name)
        except (OSError, ReportError, ValueError, TypeError):
            return True
        if value.get("mode") == "lost_response_reconciled_ambiguous_identity":
            return True
    return False


def _durable_cleanup_prefix_exists(submission_root: Path) -> bool:
    return _durable_original_cleanup_prefix_exists(
        submission_root
    ) or _durable_repair_stop_prefix_exists(submission_root)


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


def file_sha256(path: str | Path) -> str:
    _payload, digest, _info = _authenticated_regular_bytes(
        Path(path), f"SHA256 source {path}", capture=False
    )
    return digest


def sha256_string(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= SHA256


def git_commit_string(value: object) -> bool:
    return isinstance(value, str) and len(value) == 40 and set(value) <= SHA256


def _validated_production_authorization_prerequisite(
    manifest: Mapping[str, Any], package_protocol_sha256: str
) -> dict[str, Any]:
    """Project the exact positive-canary evidence frozen by submission.

    The snapshot campaign validator authenticates the complete provenance artifact.
    This compact, duplicated projection is intentional: an older snapshot reporter
    that does not know about the accepted-canary gate cannot manufacture the field
    now required in every Launch8 submission contract and report provenance record.
    """

    launch = manifest.get("launch_contract")
    canary = (
        launch.get("real_gpu_two_wave_canary")
        if isinstance(launch, Mapping)
        else None
    )
    evidence = (
        canary.get("production_authorization_evidence")
        if isinstance(canary, Mapping)
        else None
    )
    accepted = canary.get("accepted_attempts") if isinstance(canary, Mapping) else None
    require(
        isinstance(evidence, Mapping)
        and isinstance(accepted, list)
        and len(accepted) == 1
        and isinstance(accepted[0], Mapping),
        "successful canary production-authorization prerequisite differs",
    )
    attempt = accepted[0]
    sha_fields = (
        "raw_sha256",
        "canonical_sha256",
        "report_raw_sha256",
        "source_protocol_sha256",
    )
    require(
        evidence.get("attempt") == "canary2"
        and evidence.get("path") == "canary2_acceptance_provenance.json"
        and evidence.get("required") is True
        and evidence.get("satisfied") is True
        and evidence.get("artifact_evidence_consumption_allowed") is True
        and evidence.get("scientific_runtime_input_consumption_allowed") is False
        and attempt.get("attempt") == "canary2"
        and attempt.get("status") == "terminal_positive_canary_provenance_frozen"
        and attempt.get("production_authorization_prerequisite_satisfied") is True
        and attempt.get("topology_canary_passed") is True
        and type(attempt.get("active_scheduler_jobs_after_terminal")) is int
        and attempt.get("active_scheduler_jobs_after_terminal") == 0
        and all(
            sha256_string(evidence.get(field))
            and evidence.get(field) == attempt.get(field)
            for field in sha_fields
        )
        and evidence.get("path") == attempt.get("path"),
        "successful canary production-authorization evidence differs",
    )
    raw_roles = attempt.get("job_ids_by_role")
    require(
        isinstance(raw_roles, Mapping)
        and set(raw_roles) == {"wave0", "wave1", "report"},
        "successful canary production-authorization job IDs differ",
    )
    roles: dict[str, list[str]] = {}
    for role in ("wave0", "wave1", "report"):
        values = raw_roles[role]
        require(
            isinstance(values, list)
            and len(values) == 1
            and isinstance(values[0], str)
            and bool(values[0])
            and values[0][0] in "123456789"
            and all(character in "0123456789" for character in values[0]),
            f"successful canary production-authorization {role} ID differs",
        )
        roles[role] = list(values)
    require(
        len({item for values in roles.values() for item in values}) == 3,
        "successful canary production-authorization job IDs are not injective",
    )
    token = attempt.get("canary_token")
    state_root = attempt.get("state_root")
    source_commit = attempt.get("source_commit")
    state_map = attempt.get("state_file_map_canonical_sha256")
    require(
        isinstance(token, str)
        and len(token) == 16
        and set(token) <= SHA256
        and isinstance(state_root, str)
        and Path(state_root).is_absolute()
        and os.path.normpath(state_root) == state_root
        and not state_root.startswith("//")
        and isinstance(source_commit, str)
        and len(source_commit) == 40
        and set(source_commit) <= SHA256
        and sha256_string(state_map)
        and sha256_string(package_protocol_sha256),
        "successful canary production-authorization identity differs",
    )
    return {
        "schema_version": 1,
        "status": "canary2_production_authorization_prerequisite_satisfied",
        "attempt": "canary2",
        "path": "canary2_acceptance_provenance.json",
        "raw_sha256": evidence["raw_sha256"],
        "canonical_sha256": evidence["canonical_sha256"],
        "report_raw_sha256": evidence["report_raw_sha256"],
        "source_protocol_sha256": evidence["source_protocol_sha256"],
        "source_commit": source_commit,
        "state_root": state_root,
        "state_file_map_canonical_sha256": state_map,
        "canary_token": token,
        "job_ids_by_role": roles,
        "accepted_attempt_sha256": stable_hash(attempt),
        "production_authorization_evidence_sha256": stable_hash(evidence),
        "sealed_package_protocol_sha256": package_protocol_sha256,
    }


def _validated_snapshot_production_authorization_prerequisite(
    submission_root: Path,
    submission_sha256: str,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Authenticate the hermetic snapshot evidence that authorizes reporting."""

    submission = nonsymlink_directory(submission_root, "submission root")
    contract_path = contained_regular(
        submission / "SUBMISSION_CONTRACT.json",
        submission,
        "report successful-canary submission contract",
    )
    contract_payload, contract_sha256, contract_info = _authenticated_regular_bytes(
        contract_path,
        "report successful-canary submission contract",
        capture=True,
    )
    assert contract_payload is not None
    require(
        stat.S_IMODE(contract_info.st_mode) == 0o444
        and contract_sha256 == submission_sha256,
        "report successful-canary submission contract bytes differ",
    )
    contract = _decode_json_object(contract_path, contract_payload)
    require(
        type(contract.get("schema_version")) is int
        and contract.get("schema_version") == 1
        and contract.get("status") == "sealed_for_submission"
        and contract.get("campaign_id") == CAMPAIGN_ID
        and contract.get("submission_root") == str(submission),
        "report successful-canary submission contract identity differs",
    )
    snapshot = nonsymlink_directory(
        Path(str(contract.get("snapshot_root", ""))),
        "report successful-canary snapshot root",
    )
    require(
        snapshot == submission / "source-snapshot" / "repo"
        and contract.get("snapshot_root") == str(snapshot),
        "report successful-canary snapshot root differs",
    )
    inventory = contract.get("snapshot_inventory")
    require(
        isinstance(inventory, Mapping) and bool(inventory),
        "report successful-canary snapshot inventory is absent",
    )
    normalized: dict[str, str] = {}
    for raw_relative, raw_digest in inventory.items():
        require(
            isinstance(raw_relative, str),
            "report successful-canary snapshot inventory path differs",
        )
        relative = str(
            _safe_relative(
                raw_relative, "report successful-canary snapshot inventory path"
            )
        )
        require(
            sha256_string(raw_digest) and relative not in normalized,
            f"report successful-canary snapshot inventory row differs: {relative}",
        )
        normalized[relative] = raw_digest
    require(
        stable_hash(normalized) == contract.get("snapshot_inventory_sha256"),
        "report successful-canary snapshot inventory hash differs",
    )

    required_relatives = {
        "manifest": PACKAGE_RELATIVE / "manifest.json",
        "protocol": PACKAGE_RELATIVE / "protocol.sha256",
        "artifact": PACKAGE_RELATIVE / "canary2_acceptance_provenance.json",
    }
    payloads: dict[str, bytes] = {}
    raw_hashes: dict[str, str] = {}
    for label, relative in required_relatives.items():
        relative_text = relative.as_posix()
        require(
            relative_text in normalized,
            f"report successful-canary {label} is absent from snapshot inventory",
        )
        path = contained_regular(
            snapshot / relative,
            snapshot,
            f"report successful-canary {label}",
        )
        payload, digest, info = _authenticated_regular_bytes(
            path, f"report successful-canary {label}", capture=True
        )
        assert payload is not None
        require(
            stat.S_IMODE(info.st_mode) == 0o444
            and digest == normalized[relative_text],
            f"report successful-canary {label} bytes differ",
        )
        payloads[label] = payload
        raw_hashes[label] = digest

    manifest_path = snapshot / required_relatives["manifest"]
    manifest = _decode_json_object(manifest_path, payloads["manifest"])
    require(
        stable_hash(manifest) == contract.get("manifest_sha256"),
        "report successful-canary manifest binding differs",
    )
    try:
        protocol = payloads["protocol"].decode("ascii")
    except UnicodeDecodeError as exc:
        raise ReportError(
            f"report successful-canary protocol lock is not ASCII: {exc}"
        ) from exc
    protocol_value = protocol.removesuffix("\n")
    require(
        protocol == f"{protocol_value}\n"
        and sha256_string(protocol_value)
        and protocol_value == contract.get("package_protocol_sha256"),
        "report successful-canary protocol binding differs",
    )
    prerequisite = _validated_production_authorization_prerequisite(
        manifest, protocol_value
    )
    require(
        exact_json_equal(
            contract.get("production_authorization_prerequisite"), prerequisite
        ),
        "report successful-canary contract projection differs",
    )
    artifact_path = snapshot / required_relatives["artifact"]
    artifact = _decode_json_object(artifact_path, payloads["artifact"])
    require(
        raw_hashes["artifact"] == prerequisite["raw_sha256"]
        and stable_hash(artifact) == prerequisite["canonical_sha256"],
        "report successful-canary provenance artifact binding differs",
    )
    if provenance is not None:
        require(
            provenance.get("campaign_id") == CAMPAIGN_ID
            and provenance.get("submission_sha256") == submission_sha256
            and exact_json_equal(
                provenance.get("production_authorization_prerequisite"),
                prerequisite,
            )
            and provenance.get("production_authorization_prerequisite_sha256")
            == stable_hash(prerequisite),
            "report provenance successful-canary prerequisite differs",
        )
    return prerequisite


def _pairs(path: Path):
    def hook(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key in {path}: {key}")
            result[key] = value
        return result

    return hook


def _decode_json_object(path: Path, payload: bytes) -> dict[str, Any]:
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs(path),
            parse_constant=lambda token: (_ for _ in ()).throw(
                ReportError(f"non-finite JSON value in {path}: {token}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    payload, _digest, _info = _authenticated_regular_bytes(
        source, f"JSON artifact {source}", capture=True
    )
    assert payload is not None
    return _decode_json_object(source, payload)


def regular_nonsymlink(path: Path, label: str) -> Path:
    try:
        info = path.lstat()
    except OSError as exc:
        raise ReportError(f"{label} is unavailable: {exc}") from exc
    require(stat.S_ISREG(info.st_mode), f"{label} is not a regular nonsymlink file")
    return path.resolve(strict=True)


def nonsymlink_directory(path: Path, label: str) -> Path:
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
            raise ReportError(f"{label} path component is unavailable: {current}: {exc}") from exc
        require(not stat.S_ISLNK(component.st_mode), f"{label} has a symlink path component: {current}")
    try:
        info = path.lstat()
    except OSError as exc:
        raise ReportError(f"{label} is unavailable: {exc}") from exc
    require(stat.S_ISDIR(info.st_mode), f"{label} is not a nonsymlink directory")
    # Preserve the lexical path proved above.  Resolving after the final lstat would
    # introduce a new pathname-follow window before the fd-based consumer reopens
    # every component with O_NOFOLLOW.
    return lexical


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
        raise ReportError(f"{label} root cannot be opened: {exc}") from exc
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


def _open_relative_regular(
    root: Path, relative: Path, label: str
) -> tuple[int, os.stat_result]:
    relative = _safe_relative(relative, label)
    directory_fd = _open_directory_components(root, f"{label} root")
    descriptors = [directory_fd]
    descriptor: int | None = None
    try:
        for part in relative.parts[:-1]:
            child = os.open(
                part,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                dir_fd=directory_fd,
            )
            descriptors.append(child)
            directory_fd = child
        descriptor = os.open(
            relative.name,
            os.O_RDONLY
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=directory_fd,
        )
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or not stat.S_IMODE(info.st_mode) & 0o444:
            os.close(descriptor)
            descriptor = None
            raise ReportError(f"{label} is not a readable regular file")
        result = descriptor
        descriptor = None
        return result, info
    except OSError as exc:
        if descriptor is not None:
            os.close(descriptor)
        raise ReportError(f"{label} cannot be opened without symlinks: {exc}") from exc
    finally:
        for opened in reversed(descriptors):
            os.close(opened)


def _authenticated_relative_regular(
    root: Path,
    relative: Path,
    label: str,
    *,
    capture: bool,
    copy_fd: int | None = None,
) -> tuple[bytes | None, str, os.stat_result]:
    descriptor, before = _open_relative_regular(root, relative, label)
    digest = hashlib.sha256()
    chunks: list[bytes] | None = [] if capture else None
    try:
        while block := os.read(descriptor, 16 * 1024 * 1024):
            digest.update(block)
            if chunks is not None:
                chunks.append(block)
            if copy_fd is not None:
                view = memoryview(block)
                while view:
                    written = os.write(copy_fd, view)
                    require(written > 0, f"short private copy for {label}")
                    view = view[written:]
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ReportError(f"{label} cannot be read: {exc}") from exc
    finally:
        os.close(descriptor)
    require(_file_identity(after) == _file_identity(before), f"{label} changed while reading")
    return (None if chunks is None else b"".join(chunks), digest.hexdigest(), before)


def _authenticated_regular_bytes(
    path: Path, label: str, *, capture: bool
) -> tuple[bytes | None, str, os.stat_result]:
    parent = nonsymlink_directory(path.parent, f"{label} parent")
    return _authenticated_relative_regular(
        parent, Path(path.name), label, capture=capture
    )


def _stable_open_fd_sha256(descriptor: int, label: str) -> tuple[str, os.stat_result]:
    before = os.fstat(descriptor)
    require(stat.S_ISREG(before.st_mode), f"{label} is not regular")
    digest = hashlib.sha256()
    offset = 0
    try:
        while block := os.pread(descriptor, 16 * 1024 * 1024, offset):
            digest.update(block)
            offset += len(block)
        after = os.fstat(descriptor)
    except OSError as exc:
        raise ReportError(f"{label} cannot be read: {exc}") from exc
    require(
        _file_identity(after) == _file_identity(before) and offset == before.st_size,
        f"{label} changed while reading",
    )
    return digest.hexdigest(), before


def _verify_tfrecord_fd(descriptor: int, label: str, masked_crc32c: Any) -> None:
    """Require exact TFRecord framing, CRCs, and EOF on one pinned event inode."""

    before = os.fstat(descriptor)
    require(stat.S_ISREG(before.st_mode), f"{label} is not regular")

    def exact(offset: int, size: int) -> bytes:
        chunks: list[bytes] = []
        remaining = size
        cursor = offset
        try:
            while remaining:
                block = os.pread(descriptor, min(16 * 1024 * 1024, remaining), cursor)
                require(block, f"{label} has a truncated TFRecord")
                chunks.append(block)
                cursor += len(block)
                remaining -= len(block)
        except OSError as exc:
            raise ReportError(f"{label} cannot be read: {exc}") from exc
        return b"".join(chunks)

    offset = 0
    while offset < before.st_size:
        require(before.st_size - offset >= 12, f"{label} has trailing/truncated TFRecord bytes")
        length_bytes = exact(offset, 8)
        length_crc = struct.unpack("<I", exact(offset + 8, 4))[0]
        require(
            int(masked_crc32c(length_bytes)) == length_crc,
            f"{label} TFRecord length CRC differs",
        )
        length = struct.unpack("<Q", length_bytes)[0]
        require(
            length <= before.st_size - offset - 16,
            f"{label} TFRecord length exceeds remaining bytes",
        )
        payload = exact(offset + 12, int(length))
        payload_crc = struct.unpack("<I", exact(offset + 12 + int(length), 4))[0]
        require(
            int(masked_crc32c(payload)) == payload_crc,
            f"{label} TFRecord payload CRC differs",
        )
        offset += 16 + int(length)
    require(offset == before.st_size, f"{label} TFRecord EOF differs")
    require(
        _file_identity(os.fstat(descriptor)) == _file_identity(before),
        f"{label} changed during TFRecord validation",
    )


def _secure_tree_rows(
    root: Path,
    label: str,
    *,
    hash_files: bool,
    allow_wandb_symlink_leaves: bool = False,
) -> list[dict[str, Any]]:
    root = nonsymlink_directory(root, f"{label} root")
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
                        raise ReportError(f"cannot stat {label} entry {parent / entry.name}: {exc}") from exc
        except OSError as exc:
            raise ReportError(f"cannot enumerate {label} directory {parent}: {exc}") from exc
        for name, listed in sorted(children, key=lambda value: value[0]):
            relative = parent / name
            if stat.S_ISLNK(listed.st_mode):
                # W&B creates four observational convenience links per run.  They are
                # not scientific inputs: authenticate each link inode and exact target
                # text as leaf metadata, and never follow it.  The wandb directory
                # itself and every non-W&B symlink remain forbidden.
                require(
                    allow_wandb_symlink_leaves
                    and len(relative.parts) >= 2
                    and relative.parts[0] == "wandb",
                    f"{label} contains symlink: {relative}",
                )
                path_flag = getattr(os, "O_PATH", 0)
                require(path_flag != 0, f"{label} symlink authentication requires O_PATH")
                link_fd: int | None = None
                try:
                    link_fd = os.open(
                        name,
                        path_flag
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_CLOEXEC", 0),
                        dir_fd=directory_fd,
                    )
                    opened = os.fstat(link_fd)
                    require(
                        stat.S_ISLNK(opened.st_mode)
                        and opened.st_uid == os.getuid()
                        and opened.st_nlink == 1
                        and _file_identity(opened) == _file_identity(listed),
                        f"{label} symlink raced: {relative}",
                    )
                    target = os.readlink(b"", dir_fd=link_fd)
                    middle = os.fstat(link_fd)
                    target_after = os.readlink(b"", dir_fd=link_fd)
                    after = os.fstat(link_fd)
                    named_after = os.stat(
                        name, dir_fd=directory_fd, follow_symlinks=False
                    )
                    require(
                        len(target) == opened.st_size
                        and target_after == target
                        and _file_identity(middle) == _file_identity(opened)
                        and _file_identity(after) == _file_identity(opened)
                        and _file_identity(named_after) == _file_identity(opened),
                        f"{label} symlink changed: {relative}",
                    )
                except OSError as exc:
                    raise ReportError(
                        f"cannot authenticate {label} symlink {relative}: {exc}"
                    ) from exc
                finally:
                    if link_fd is not None:
                        os.close(link_fd)
                rows.append(
                    {
                        "path": str(relative),
                        "kind": "symlink",
                        "mode": stat.S_IMODE(opened.st_mode),
                        "device": opened.st_dev,
                        "inode": opened.st_ino,
                        "uid": opened.st_uid,
                        "gid": opened.st_gid,
                        "nlink": opened.st_nlink,
                        "size": opened.st_size,
                        "mtime_ns": opened.st_mtime_ns,
                        "ctime_ns": opened.st_ctime_ns,
                        "readlink_bytes_hex": target.hex(),
                    }
                )
            elif stat.S_ISDIR(listed.st_mode):
                permissions = stat.S_IMODE(listed.st_mode)
                require(
                    permissions & 0o444 != 0 and permissions & 0o111 != 0,
                    f"{label} directory is not traversable: {relative}",
                )
                try:
                    child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise ReportError(f"cannot open {label} directory {relative}: {exc}") from exc
                try:
                    opened = os.fstat(child_fd)
                    require(_file_identity(opened) == _file_identity(listed), f"{label} directory raced: {relative}")
                    require(opened.st_uid == os.getuid(), f"{label} directory owner differs: {relative}")
                    rows.append(
                        {
                            "path": str(relative),
                            "kind": "directory",
                            "mode": stat.S_IMODE(opened.st_mode),
                            "device": opened.st_dev,
                            "inode": opened.st_ino,
                            "uid": opened.st_uid,
                            "gid": opened.st_gid,
                            "nlink": opened.st_nlink,
                            "size": opened.st_size,
                            "mtime_ns": opened.st_mtime_ns,
                            "ctime_ns": opened.st_ctime_ns,
                        }
                    )
                    walk(child_fd, relative, opened)
                    require(_file_identity(os.fstat(child_fd)) == _file_identity(opened), f"{label} directory changed: {relative}")
                finally:
                    os.close(child_fd)
            elif stat.S_ISREG(listed.st_mode):
                require(stat.S_IMODE(listed.st_mode) & 0o444 != 0, f"{label} file is unreadable: {relative}")
                try:
                    child_fd = os.open(name, file_flags, dir_fd=directory_fd)
                except OSError as exc:
                    raise ReportError(f"cannot open {label} file {relative}: {exc}") from exc
                digest = hashlib.sha256() if hash_files else None
                try:
                    opened = os.fstat(child_fd)
                    require(_file_identity(opened) == _file_identity(listed), f"{label} file raced: {relative}")
                    require(
                        opened.st_uid == os.getuid() and opened.st_nlink == 1,
                        f"{label} file ownership/link count differs: {relative}",
                    )
                    if digest is not None:
                        while block := os.read(child_fd, 16 * 1024 * 1024):
                            digest.update(block)
                    require(_file_identity(os.fstat(child_fd)) == _file_identity(opened), f"{label} file changed: {relative}")
                except OSError as exc:
                    raise ReportError(f"cannot read {label} file {relative}: {exc}") from exc
                finally:
                    os.close(child_fd)
                row = {
                    "path": str(relative),
                    "kind": "file",
                    "mode": stat.S_IMODE(opened.st_mode),
                    "device": opened.st_dev,
                    "inode": opened.st_ino,
                    "uid": opened.st_uid,
                    "gid": opened.st_gid,
                    "nlink": opened.st_nlink,
                    "size": opened.st_size,
                    "mtime_ns": opened.st_mtime_ns,
                    "ctime_ns": opened.st_ctime_ns,
                }
                if digest is not None:
                    row["sha256"] = digest.hexdigest()
                rows.append(row)
            else:
                raise ReportError(f"{label} contains special file: {relative}")
        require(_file_identity(os.fstat(directory_fd)) == _file_identity(before), f"{label} directory changed: {parent}")

    try:
        opened_root = os.fstat(root_fd)
        require(opened_root.st_uid == os.getuid(), f"{label} root owner differs")
        rows.append(
            {
                "path": "",
                "kind": "root",
                "mode": stat.S_IMODE(opened_root.st_mode),
                "device": opened_root.st_dev,
                "inode": opened_root.st_ino,
                "uid": opened_root.st_uid,
                "gid": opened_root.st_gid,
                "nlink": opened_root.st_nlink,
                "size": opened_root.st_size,
                "mtime_ns": opened_root.st_mtime_ns,
                "ctime_ns": opened_root.st_ctime_ns,
            }
        )
        walk(root_fd, Path(), opened_root)
        require(_file_identity(os.fstat(root_fd)) == _file_identity(opened_root), f"{label} root changed")
    finally:
        os.close(root_fd)
    rows.sort(key=lambda row: str(row["path"]))
    return rows


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
            opened_after_lock = os.fstat(descriptor)
            named_after_lock = self.path.lstat()
            require(
                stat.S_ISREG(opened_after_lock.st_mode)
                and _file_identity(opened_after_lock)
                == _file_identity(named_after_lock)
                and opened_after_lock.st_uid == os.getuid()
                and opened_after_lock.st_nlink == 1
                and stat.S_IMODE(opened_after_lock.st_mode) == 0o600,
                "report/cancel lock binding changed while waiting for flock",
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
    pyvenv_payload, _pyvenv_digest, _pyvenv_info = _authenticated_regular_bytes(
        pyvenv, "pinned pyvenv.cfg", capture=True
    )
    assert pyvenv_payload is not None
    values: dict[str, str] = {}
    try:
        pyvenv_text = pyvenv_payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReportError(f"pinned pyvenv.cfg is not UTF-8: {exc}") from exc
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
        "lexical_symlink_target": os.readlink(expected) if stat.S_ISLNK(info.st_mode) else None,
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
    *,
    require_publish_job: bool = False,
    repair_attempt: int | None = None,
    repair_authorization_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate the contract/receipt and exact read-only snapshot using stdlib only."""

    submission = nonsymlink_directory(submission_root, "submission root")
    snapshot = nonsymlink_directory(snapshot_root, "snapshot root")
    require(snapshot.is_relative_to(submission), "snapshot root escapes submission root")
    require(
        snapshot == submission / "source-snapshot" / "repo",
        "snapshot root differs from exact source-snapshot namespace",
    )
    for path, mode, label in (
        (submission, 0o700, "submission root"),
        (submission / "source-snapshot", 0o555, "source-snapshot parent"),
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
    contract_path = submission / "SUBMISSION_CONTRACT.json"
    contained_regular(contract_path, submission, "submission contract")
    require(stat.S_IMODE(contract_path.lstat().st_mode) == 0o444, "submission contract mode differs")
    require(file_sha256(contract_path) == submission_sha256, "submission contract hash differs")
    seal_path = submission / "journal" / "0002_CONTRACT_SEALED.json"
    contained_regular(seal_path, submission, "contract-seal journal")
    require(stat.S_IMODE(seal_path.lstat().st_mode) == 0o444, "contract-seal journal mode differs")
    require(
        exact_json_equal(
            read_json(seal_path),
            {
            "schema_version": 1,
            "record": "contract_sealed",
            "submission_sha256": submission_sha256,
            "launch_count": 20,
            },
        ),
        "contract-seal journal differs",
    )
    contract = read_json(contract_path)
    require(
        type(contract.get("schema_version")) is int
        and contract.get("schema_version") == 1
        and contract.get("status") == "sealed_for_submission",
        "submission contract is not sealed",
    )
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
    for row in _secure_tree_rows(snapshot, "sealed snapshot", hash_files=True):
        relative = str(row["path"])
        if row["kind"] == "root":
            require(int(row["mode"]) == 0o555, "snapshot root mode differs")
            continue
        if row["kind"] == "file":
            actual_files.add(relative)
            require(relative in normalized, f"snapshot contains unclaimed file: {relative}")
            require(int(row["mode"]) & 0o222 == 0, f"snapshot file is writable: {relative}")
            require(row["sha256"] == normalized[relative], f"snapshot file hash differs: {relative}")
        else:
            actual_dirs.add(relative)
            require(int(row["mode"]) & 0o222 == 0, f"snapshot directory is writable: {relative}")
    require(actual_files == set(normalized), "snapshot file coverage differs")
    require(actual_dirs == expected_dirs, "snapshot directory coverage differs")

    _validated_snapshot_production_authorization_prerequisite(
        submission, submission_sha256
    )

    receipt_path = submission / "SUBMISSION_RECEIPT.json"
    contained_regular(receipt_path, submission, "submission receipt")
    require(stat.S_IMODE(receipt_path.lstat().st_mode) == 0o444, "submission receipt mode differs")
    receipt = read_json(receipt_path)
    require(set(receipt) == RECEIPT_KEYS, "submission receipt schema differs")
    require(
        type(receipt["schema_version"]) is int
        and receipt["schema_version"] == 1
        and receipt["status"] == "committed_two_wave_dag",
        "submission receipt is not committed",
    )
    require(receipt["campaign_id"] == CAMPAIGN_ID, "submission receipt campaign differs")
    require(receipt["submission_sha256"] == submission_sha256, "submission receipt hash differs")
    wave0_id = receipt["wave0_array_job_id"]
    wave1_id = receipt["wave1_array_job_id"]
    report_id = receipt["report_job_id"]
    require(
        all(
            isinstance(value, str)
            and value
            and value[0] in "123456789"
            and all(character in "0123456789" for character in value)
            for value in (wave0_id, wave1_id, report_id)
        )
        and len({wave0_id, wave1_id, report_id}) == 3,
        "DAG receipt job IDs are malformed",
    )
    require(receipt["array"] == "0-19%20", "submission receipt array differs")
    require(
        receipt["wave1_dependency"] == f"afterok:{wave0_id}"
        and receipt["report_dependency"] == f"afterok:{wave1_id}"
        and exact_json_equal(
            receipt["kill_on_invalid_dependency"],
            {"wave1": "yes", "report": "yes"},
        )
        and receipt["within_wave_requeue"] is False
        and receipt["wave0_submitted_held"] is True,
        "submission receipt DAG differs",
    )
    authorization_path = submission / "SUBMISSION_AUTHORIZATION.json"
    contained_regular(authorization_path, submission, "submission authorization")
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
        and authorization.get("campaign_id") == CAMPAIGN_ID
        and authorization.get("submission_sha256") == submission_sha256
        and authorization.get("array") == "0-19%20"
        and exact_json_equal(authorization.get("job_ids"), job_ids)
        and exact_json_equal(authorization.get("dependencies"), dependencies)
        and exact_json_equal(
            authorization.get("kill_on_invalid_dependency"),
            {"wave1": "yes", "report": "yes"},
        )
        and authorization.get("within_wave_requeue") is False
        and authorization.get("wave0_submitted_held") is True
        and sha256_string(authorization.get("accepted_job_evidence_sha256"))
        and isinstance(authorization.get("authorized_at_utc"), str)
        and bool(authorization["authorized_at_utc"]),
        "submission authorization differs",
    )
    records: dict[str, Any] = {}
    for role, ordinal, expected_id in (
        ("wave0", 3, wave0_id),
        ("wave1", 4, wave1_id),
        ("report", 5, report_id),
    ):
        journal_path = submission / "journal" / f"{ordinal:04d}_{role.upper()}_SUBMITTED.json"
        contained_regular(journal_path, submission, f"{role} submission journal")
        require(stat.S_IMODE(journal_path.lstat().st_mode) == 0o444, f"{role} submission journal mode differs")
        journal = read_json(journal_path)
        require(
            type(journal.get("schema_version")) is int
            and journal.get("schema_version") == 1
            and journal.get("record") == f"{role}_submitted"
            and isinstance(journal.get("job_id"), str)
            and journal.get("job_id") == expected_id,
            f"submission receipt {role} job differs from durable journal",
        )
        records[role] = {
            key: value
            for key, value in journal.items()
            if key not in {"schema_version", "record", "job_id"}
        }
    claim_path = submission / "journal" / "0000_CLAIMED.json"
    contained_regular(claim_path, submission, "DAG transaction claim")
    require(stat.S_IMODE(claim_path.lstat().st_mode) == 0o444, "DAG transaction claim mode differs")
    claim = read_json(claim_path)
    calling_records = {}
    calling_sha256_by_role = {}
    for role in ("wave0", "wave1", "report"):
        calling_path = submission / "journal" / f"CALLING_{role.upper()}.json"
        contained_regular(calling_path, submission, f"{role} scheduler calling record")
        require(stat.S_IMODE(calling_path.lstat().st_mode) == 0o444, f"{role} scheduler calling record mode differs")
        calling_records[role] = read_json(calling_path)
        calling_sha256_by_role[role] = file_sha256(calling_path)
    manifest = read_json(snapshot / PACKAGE_RELATIVE / "manifest.json")
    dag_evidence = _load_module(
        "DAG evidence verifier",
        snapshot / PACKAGE_RELATIVE / "dag_evidence.py",
        snapshot,
    )
    try:
        semantic_hash = dag_evidence.validate_dag_records(
            records,
            calling_records,
            calling_sha256_by_role,
            manifest=manifest,
            snapshot_root=snapshot,
            submission_root=submission,
            submission_sha256=submission_sha256,
            claim_token=str(claim.get("claim_token", "")),
            job_ids=job_ids,
            expected_control_plane=contract["scheduler_preclaim"][
                "scheduler_control_plane"
            ],
        )
    except BaseException as exc:
        raise ReportError(f"accepted DAG scheduler evidence differs: {exc}") from exc
    require(
        semantic_hash == authorization["accepted_job_evidence_sha256"],
        "accepted-job journal evidence differs",
    )
    authorized_path = submission / "journal" / "0006_DAG_AUTHORIZED.json"
    contained_regular(authorized_path, submission, "durable DAG authorization journal")
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
    ready_path = submission / "journal" / "0007_READY_TO_COMMIT.json"
    contained_regular(ready_path, submission, "durable ready-to-commit journal")
    require(stat.S_IMODE(ready_path.lstat().st_mode) == 0o444, "ready-to-commit journal mode differs")
    require(
        exact_json_equal(
            read_json(ready_path),
            {"schema_version": 1, "record": "ready_to_commit", **receipt},
        ),
        "durable ready-to-commit journal differs",
    )
    released_path = submission / "journal" / "0008_WAVE0_RELEASED.json"
    contained_regular(released_path, submission, "durable wave-zero release journal")
    require(stat.S_IMODE(released_path.lstat().st_mode) == 0o444, "wave-zero release journal mode differs")
    released = read_json(released_path)
    require(
        set(released)
        == {
            "schema_version",
            "record",
            "wave0_array_job_id",
            "submission_authorization_sha256",
            "calling_sha256",
            "release_evidence",
        }
        and type(released.get("schema_version")) is int
        and released.get("schema_version") == 1
        and released.get("record") == "wave0_released"
        and released.get("wave0_array_job_id") == wave0_id
        and released.get("submission_authorization_sha256")
        == receipt["submission_authorization_sha256"]
        and sha256_string(released.get("calling_sha256"))
        and isinstance(released.get("release_evidence"), Mapping),
        "durable wave-zero release journal differs",
    )
    release_calling_path = submission / "journal" / "CALLING_WAVE0_RELEASE.json"
    contained_regular(
        release_calling_path, submission, "wave-zero release calling record"
    )
    release_calling = read_json(release_calling_path)
    require(
        file_sha256(release_calling_path) == released["calling_sha256"]
        and exact_json_equal(
            release_calling,
            {
            "schema_version": 1,
            "status": "scheduler_calling",
            "campaign_id": CAMPAIGN_ID,
            "submission_sha256": submission_sha256,
            "claim_token": claim["claim_token"],
            "role": "wave0_release",
            "job_name": f"exp23-launch8-{submission_sha256[:16]}-wave0",
            "scheduler_comment": f"treewm-exp23:{submission_sha256}",
            "command": [
                str(manifest["execution"]["scontrol"]),
                "release",
                wave0_id,
            ],
            "transaction_lock": calling_records["wave0"]["transaction_lock"],
            },
        ),
        "wave-zero release calling record differs",
    )
    evidence = dict(released["release_evidence"])
    require(
        set(evidence)
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
        "wave-zero release evidence fields differ",
    )
    release_command = evidence["release_command"]
    require(
        (
            release_command == ["/usr/local/bin/scontrol", "release", wave0_id]
            and type(evidence["release_returncode"]) is int
            and evidence["release_returncode"] == 0
            and isinstance(evidence["release_stdout"], str)
            and isinstance(evidence["release_stderr"], str)
            and exact_json_equal(
                evidence["release_scheduler_control_plane"],
                (contract.get("scheduler_preclaim") or {}).get(
                    "scheduler_control_plane"
                ),
            )
        )
        or (
            release_command is None
            and evidence["release_returncode"] is None
            and evidence["release_stdout"] == ""
            and evidence["release_stderr"] == ""
            and evidence["release_scheduler_control_plane"] is None
        ),
        "wave-zero release command differs",
    )
    require(
        evidence["show_command"]
        == ["/usr/local/bin/scontrol", "show", "job", wave0_id, "--oneliner"]
        and type(evidence["show_returncode"]) is int
        and evidence["show_returncode"] == 0
        and isinstance(evidence["show_stdout"], str)
        and isinstance(evidence["show_stderr"], str)
        and len(evidence["show_stdout"]) <= 1024 * 1024
        and len(evidence["show_stderr"]) <= 1024 * 1024
        and exact_json_equal(
            evidence["show_scheduler_control_plane"],
            (contract.get("scheduler_preclaim") or {}).get(
                "scheduler_control_plane"
            ),
        ),
        "wave-zero release show/control evidence differs",
    )
    fields = {
        token.split("=", 1)[0]: token.split("=", 1)[1]
        for token in evidence["show_stdout"].split()
        if "=" in token
    }
    require(
        fields.get("JobId") == wave0_id
        and fields.get("JobName") == f"exp23-launch8-{submission_sha256[:16]}-wave0"
        and fields.get("Comment") == f"treewm-exp23:{submission_sha256}"
        and fields.get("JobState") == evidence["state"]
        and evidence["state"] in {"PENDING", "RUNNING", "COMPLETING", "COMPLETED"}
        and fields.get("Reason") == evidence["reason"]
        and evidence["reason"] != "JobHeldUser",
        "wave-zero release identity/state differs",
    )
    scheduler_job = os.environ.get("SLURM_JOB_ID")
    if repair_attempt is not None or repair_authorization_sha256 is not None:
        require(
            type(repair_attempt) is int
            and repair_attempt == 1
            and sha256_string(repair_authorization_sha256),
            "report repair publication arguments differ",
        )
        authority = _validated_report_repair_authorization(
            submission,
            submission_sha256,
            receipt,
            attempt=repair_attempt,
            expected_raw_sha256=str(repair_authorization_sha256),
        )
        require(
            scheduler_job is not None
            and scheduler_job == authority["repair_report_job_id"],
            "active repair-report Slurm job differs from repair authorization",
        )
    elif require_publish_job:
        require(
            scheduler_job is not None and scheduler_job == report_id,
            "active report Slurm job is absent or differs from committed receipt",
        )
    elif scheduler_job is not None:
        require(
            scheduler_job == report_id,
            "active report Slurm job differs from committed receipt",
        )
    return contract, receipt


def contained_regular(path: Path, root: Path, label: str) -> Path:
    regular_nonsymlink(path, label)
    resolved = path.absolute()
    expected_root = nonsymlink_directory(root, f"{label} root")
    require(resolved.is_relative_to(expected_root), f"{label} escapes its declared root")
    return resolved


def _receipt_file_map(submission_root: Path) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for index in range(20):
        relative = Path("tasks") / f"cell-{index:02d}" / "WORKER_COMPLETE.json"
        path = contained_regular(
            submission_root / relative,
            submission_root,
            f"cell{index} repair worker receipt",
        )
        info = path.lstat()
        require(
            stat.S_IMODE(info.st_mode) == 0o444
            and info.st_uid == os.getuid()
            and info.st_nlink == 1,
            f"cell{index} repair worker receipt identity differs",
        )
        files[relative.as_posix()] = {
            "mode": 0o444,
            "size": info.st_size,
            "sha256": file_sha256(path),
        }
    return {"schema_version": 1, "files": files}


def _repair_authorization_path(submission_root: Path, attempt: int) -> Path:
    return (
        submission_root
        / "journal"
        / f"REPORT_REPAIR_{attempt:04d}_AUTHORIZED.json"
    )


def _repair_release_path(submission_root: Path, attempt: int) -> Path:
    return (
        submission_root
        / "journal"
        / f"REPORT_REPAIR_{attempt:04d}_RELEASED.json"
    )


def _wait_for_repair_release_evidence(
    submission_root: Path,
    submission_sha256: str,
    *,
    attempt: int,
    authorization_sha256: str,
    monotonic: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> None:
    require(type(attempt) is int and attempt == 1, "report repair wait attempt differs")
    require(sha256_string(authorization_sha256), "report repair wait authorization hash differs")
    require(
        REPAIR_RELEASE_EVIDENCE_WAIT_SECONDS + REPAIR_ASSEMBLY_BUDGET_SECONDS
        == REPAIR_WALLTIME_SECONDS,
        "report repair wait/assembly walltime budget differs",
    )
    authorization_path = _repair_authorization_path(submission_root, attempt)
    payload, digest, info = _authenticated_regular_bytes(
        authorization_path, "report repair wait authorization", capture=True
    )
    assert payload is not None
    require(
        stat.S_IMODE(info.st_mode) == 0o444
        and info.st_uid == os.getuid()
        and info.st_nlink == 1
        and digest == authorization_sha256,
        "report repair wait authorization identity/hash differs",
    )
    authorization = _decode_json_object(authorization_path, payload)
    job_id = authorization.get("repair_report_job_id")
    require(
        set(authorization) == REPORT_REPAIR_AUTHORIZATION_KEYS
        and type(authorization.get("schema_version")) is int
        and authorization.get("schema_version") == 1
        and authorization.get("status") == "authorized_terminal_report_repair"
        and authorization.get("campaign_id") == CAMPAIGN_ID
        and authorization.get("submission_sha256") == submission_sha256
        and type(authorization.get("attempt")) is int
        and authorization.get("attempt") == attempt
        and isinstance(job_id, str)
        and job_id.isdigit()
        and job_id[0] in "123456789"
        and exact_json_equal(
            authorization.get("worker_handoff"), REPAIR_WORKER_HANDOFF
        )
        and authorization.get("publication_allowed") is True
        and authorization.get("scheduler_submission_allowed") is False,
        "report repair wait authorization fields differ",
    )
    require(
        os.environ.get("SLURM_JOB_ID") == job_id
        and os.environ.get("SLURM_RESTART_COUNT") == "0"
        and "SLURM_ARRAY_JOB_ID" not in os.environ
        and "SLURM_ARRAY_TASK_ID" not in os.environ,
        "active report repair scheduler identity/restart differs",
    )
    started = monotonic()
    require(
        isinstance(started, (int, float))
        and not isinstance(started, bool)
        and math.isfinite(float(started)),
        "report repair monotonic clock differs",
    )
    deadline = float(started) + REPAIR_RELEASE_EVIDENCE_WAIT_SECONDS
    release_path = _repair_release_path(submission_root, attempt)
    while True:
        require(
            not _durable_original_cleanup_prefix_exists(submission_root)
            and not _durable_repair_stop_prefix_exists(submission_root),
            "repair cleanup/terminal authority won before release evidence",
        )
        release_info = _lstat_if_present(release_path, "report repair release evidence")
        if release_info is not None:
            require(
                stat.S_ISREG(release_info.st_mode)
                and stat.S_IMODE(release_info.st_mode) == 0o444
                and release_info.st_uid == os.getuid()
                and release_info.st_nlink == 1,
                "report repair release evidence identity differs",
            )
            return
        now = monotonic()
        require(
            isinstance(now, (int, float))
            and not isinstance(now, bool)
            and math.isfinite(float(now))
            and float(now) >= float(started),
            "report repair monotonic clock regressed",
        )
        remaining = deadline - float(now)
        require(remaining > 0, "report repair release-evidence wait exhausted")
        sleep(min(REPAIR_RELEASE_POLL_SECONDS, remaining))


def _repair_stream_bytes(value: object, label: str) -> bytes:
    require(
        isinstance(value, Mapping)
        and set(value) == {"encoding", "size", "sha256", "data"}
        and value.get("encoding") == "base64"
        and type(value.get("size")) is int
        and value["size"] >= 0
        and sha256_string(value.get("sha256"))
        and isinstance(value.get("data"), str),
        f"{label} stream evidence differs",
    )
    try:
        payload = base64.b64decode(value["data"], validate=True)
    except (ValueError, TypeError) as exc:
        raise ReportError(f"{label} stream base64 differs: {exc}") from exc
    require(
        len(payload) == value["size"]
        and hashlib.sha256(payload).hexdigest() == value["sha256"],
        f"{label} stream payload differs",
    )
    return payload


def _validated_repair_command_evidence(
    value: object,
    *,
    expected_argv: Sequence[str] | None,
    label: str,
) -> dict[str, Any]:
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
        require(value["argv"] == list(expected_argv), f"{label} command differs")
    _repair_stream_bytes(value["stdout"], f"{label} stdout")
    _repair_stream_bytes(value["stderr"], f"{label} stderr")
    return dict(value)


def _validated_repair_census(
    value: object,
    *,
    submission_sha256: str,
    label: str,
) -> dict[str, Any]:
    require(
        isinstance(value, Mapping)
        and set(value) == {"schema_version", "rounds", "settled_rows", "captured_at_utc"}
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and isinstance(value.get("rounds"), list)
        and len(value["rounds"]) == 3
        and isinstance(value.get("settled_rows"), list)
        and isinstance(value.get("captured_at_utc"), str)
        and bool(value["captured_at_utc"]),
        f"{label} census shape differs",
    )
    reconstructed: list[list[dict[str, str]]] = []
    for index, round_value in enumerate(value["rounds"]):
        require(
            isinstance(round_value, Mapping)
            and set(round_value) == {"round", "raw", "relevant_rows"}
            and type(round_value.get("round")) is int
            and round_value.get("round") == index
            and isinstance(round_value.get("relevant_rows"), list),
            f"{label} census round differs",
        )
        raw = _validated_repair_command_evidence(
            round_value["raw"], expected_argv=None, label=f"{label} round {index}"
        )
        require(
            raw["argv"][0:2] == ["/usr/local/bin/squeue", "--noheader"]
            and any(item.startswith("--user=") for item in raw["argv"])
            and "--format=%A|%j|%u|%T|%k|%r" in raw["argv"]
            and raw["returncode"] == 0
            and _repair_stream_bytes(raw["stderr"], f"{label} round {index} stderr")
            == b"",
            f"{label} census command differs",
        )
        stdout_payload = _repair_stream_bytes(
            raw["stdout"], f"{label} round {index} stdout"
        )
        try:
            stdout_text = stdout_payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReportError(f"{label} census stdout is not UTF-8: {exc}") from exc
        owner_options = [item for item in raw["argv"] if item.startswith("--user=")]
        require(len(owner_options) == 1, f"{label} census owner option differs")
        expected_owner = owner_options[0].split("=", 1)[1]
        parsed_relevant: list[dict[str, str]] = []
        for line in stdout_text.splitlines():
            if not line:
                continue
            fields = line.split("|", 5)
            require(len(fields) == 6, f"{label} census raw row differs")
            job_id, job_name, owner, state, comment, reason = fields
            require(
                job_id.isdigit() and job_id[0] in "123456789",
                f"{label} census raw job ID differs",
            )
            if job_name.startswith("exp23-launch8-") or comment.startswith(
                "treewm-exp23"
            ):
                require(owner == expected_owner, f"{label} census raw owner differs")
                parsed_relevant.append(
                    {
                        "job_id": job_id,
                        "job_name": job_name,
                        "owner": owner,
                        "state": state,
                        "comment": comment,
                        "reason": reason,
                    }
                )
        rows: list[dict[str, str]] = []
        for row in round_value["relevant_rows"]:
            require(
                isinstance(row, Mapping)
                and set(row)
                == {"job_id", "job_name", "owner", "state", "comment", "reason"}
                and isinstance(row.get("job_id"), str)
                and row["job_id"].isdigit()
                and row["job_id"][0] in "123456789"
                and all(isinstance(row.get(key), str) for key in row),
                f"{label} census row differs",
            )
            rows.append(dict(row))
        require(
            exact_json_equal(rows, parsed_relevant),
            f"{label} census parsed/raw rows differ",
        )
        reconstructed.append(rows)
    require(
        exact_json_equal(reconstructed[-2], reconstructed[-1])
        and exact_json_equal(value["settled_rows"], reconstructed[-1]),
        f"{label} census did not settle",
    )
    return dict(value)


def _validated_repair_worker_liveness(
    value: Mapping[str, Any],
    *,
    post_release: Mapping[str, Any],
    repair_authorization: Mapping[str, Any],
    authority_environment: Mapping[str, Any],
    contract: Mapping[str, Any],
    submission_sha256: str,
) -> dict[str, Any]:
    job_id = str(repair_authorization["repair_report_job_id"])
    mode = value.get("mode")
    if mode == "active_squeue_identity":
        require(
            set(value)
            == {
                "schema_version",
                "mode",
                "repair_report_job_id",
                "state",
                "reason",
                "scheduler_census_sha256",
            }
            and type(value.get("schema_version")) is int
            and value.get("schema_version") == 1
            and value.get("repair_report_job_id") == job_id
            and value.get("scheduler_census_sha256") == stable_hash(post_release),
            "report repair squeue worker liveness differs",
        )
        rows = [
            row
            for row in post_release["settled_rows"]
            if row["job_name"] == repair_authorization["repair_job_name"]
            and row["comment"] == repair_authorization["scheduler_comment"]
        ]
        require(
            len(post_release["settled_rows"]) == 1
            and len(rows) == 1
            and rows[0]["job_id"] == job_id
            and rows[0]["state"]
            in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
            and not (
                rows[0]["state"] == "PENDING"
                and rows[0]["reason"] in {"JobHeldUser", "JobHeldAdmin"}
            )
            and value.get("state") == rows[0]["state"]
            and value.get("reason") == rows[0]["reason"],
            "report repair squeue worker liveness identity differs",
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
        observation = value["accounting_observation"]
        require(
            set(observation)
            == {
                "schema_version",
                "captured_at_utc",
                "scheduler_control_plane",
                "raw",
                "canonical",
                "canonical_sha256",
                "parsed_row",
            }
            and type(observation.get("schema_version")) is int
            and observation.get("schema_version") == 1
            and isinstance(observation.get("captured_at_utc"), str)
            and bool(observation["captured_at_utc"])
            and exact_json_equal(
                observation.get("scheduler_control_plane"),
                contract.get("scheduler_control_plane_contract"),
            )
            and isinstance(observation.get("canonical"), Mapping)
            and observation.get("canonical_sha256")
            == stable_hash(observation["canonical"])
            and isinstance(observation.get("parsed_row"), Mapping),
            "report repair accounting liveness observation differs",
        )
        raw = _validated_repair_command_evidence(
            observation.get("raw"),
            expected_argv=[
                "/usr/local/bin/sacct",
                "-X",
                "-n",
                "-j",
                job_id,
                "-o",
                ",".join(REPAIR_SACCT_FIELDS),
                "-P",
            ],
            label="report repair worker accounting",
        )
        require(
            exact_json_equal(raw["environment"], authority_environment)
            and raw["returncode"] == 0
            and _repair_stream_bytes(
                raw["stderr"], "report repair worker accounting stderr"
            )
            == b"",
            "report repair worker accounting command differs",
        )
        stdout = _repair_stream_bytes(
            raw["stdout"], "report repair worker accounting stdout"
        )
        try:
            text = stdout.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ReportError(
                f"report repair worker accounting stdout is not UTF-8: {exc}"
            ) from exc
        lines = [line for line in text.splitlines() if line]
        require(len(lines) == 1, "report repair worker accounting row count differs")
        row = lines[0].split("|")
        require(
            len(row) == len(REPAIR_SACCT_FIELDS),
            "report repair worker accounting field count differs",
        )
        parsed = dict(zip(REPAIR_SACCT_FIELDS, row, strict=True))
        state = parsed["State"].split(maxsplit=1)[0].removesuffix("+")
        require(
            parsed["JobIDRaw"] == job_id
            and parsed["JobName"] == repair_authorization["repair_job_name"]
            and parsed["User"] == authority_environment["USER"]
            and parsed["Comment"] == repair_authorization["scheduler_comment"]
            and state
            in {
                "PENDING",
                "RUNNING",
                "CONFIGURING",
                "COMPLETING",
                "SUSPENDED",
                "RESIZING",
                "STAGE_OUT",
            }
            and not (
                state == "PENDING"
                and parsed["Reason"] in {"JobHeldUser", "JobHeldAdmin"}
            )
            and exact_json_equal(observation["parsed_row"], parsed)
            and exact_json_equal(
                observation["canonical"],
                {
                    "schema_version": 1,
                    "fields": list(REPAIR_SACCT_FIELDS),
                    "rows": [row],
                },
            )
            and not post_release["settled_rows"]
            and not [
                item
                for item in post_release["settled_rows"]
                if item["job_name"] == repair_authorization["repair_job_name"]
                and item["comment"] == repair_authorization["scheduler_comment"]
            ],
            "report repair accounting worker liveness identity differs",
        )
    else:
        raise ReportError("report repair worker liveness mode differs")
    return dict(value)


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


def _validated_report_repair_authorization(
    submission_root: Path,
    submission_sha256: str,
    receipt: Mapping[str, Any],
    *,
    attempt: int,
    expected_raw_sha256: str,
) -> dict[str, Any]:
    """Authenticate the one append-only publication-only Launch8 repair."""

    require(type(attempt) is int and attempt == 1, "report repair attempt differs")
    require(sha256_string(expected_raw_sha256), "report repair authorization SHA256 differs")
    path = contained_regular(
        _repair_authorization_path(submission_root, attempt),
        submission_root,
        "report repair authorization",
    )
    info = path.lstat()
    require(
        stat.S_IMODE(info.st_mode) == 0o444
        and info.st_uid == os.getuid()
        and info.st_nlink == 1
        and file_sha256(path) == expected_raw_sha256,
        "report repair authorization identity/hash differs",
    )
    value = read_json(path)
    historical_ids = {
        receipt.get("wave0_array_job_id"),
        receipt.get("wave1_array_job_id"),
        receipt.get("report_job_id"),
        "33285485",
        "33285486",
        "33295657",
        "33295659",
        "33295661",
    }
    require(
        set(value) == REPORT_REPAIR_AUTHORIZATION_KEYS
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("status") == "authorized_terminal_report_repair"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and type(value.get("attempt")) is int
        and value.get("attempt") == attempt
        and value.get("original_report_job_id") == receipt.get("report_job_id")
        and isinstance(value.get("repair_report_job_id"), str)
        and value["repair_report_job_id"].isdigit()
        and value["repair_report_job_id"][0] in "123456789"
        and value["repair_report_job_id"] not in historical_ids
        and value.get("repair_job_name")
        == f"exp23-launch8-{submission_sha256[:16]}-report-repair-0001"
        and value.get("scheduler_comment")
        == f"treewm-exp23-report-repair:{submission_sha256}:0001"
        and exact_json_equal(value.get("worker_handoff"), REPAIR_WORKER_HANDOFF)
        and exact_json_equal(value.get("expected_reassembly"), EXPECTED_REPAIR_REASSEMBLY)
        and value.get("publication_allowed") is True
        and value.get("deterministic_reassembly_allowed") is True
        and value.get("scientific_input_change_allowed") is False
        and value.get("gate_change_allowed") is False
        and value.get("scheduler_submission_allowed") is False
        and isinstance(value.get("authorized_at_utc"), str)
        and bool(value["authorized_at_utc"]),
        "report repair authorization fields differ",
    )
    contract = read_json(submission_root / "SUBMISSION_CONTRACT.json")
    require(
        value.get("snapshot_root") == contract.get("snapshot_root")
        and value.get("snapshot_inventory_sha256")
        == contract.get("snapshot_inventory_sha256")
        and value.get("original_package_protocol_sha256")
        == contract.get("package_protocol_sha256"),
        "report repair original snapshot binding differs",
    )

    failure_name = value.get("original_failure_evidence")
    require(
        failure_name == f"journal/REPORT_REPAIR_{attempt:04d}_ORIGINAL_FAILURE.json",
        "report repair failure-evidence name differs",
    )
    failure_path = contained_regular(
        submission_root / str(failure_name),
        submission_root,
        "report repair original failure evidence",
    )
    failure_info = failure_path.lstat()
    require(
        stat.S_IMODE(failure_info.st_mode) == 0o444
        and failure_info.st_uid == os.getuid()
        and failure_info.st_nlink == 1
        and file_sha256(failure_path) == value.get("original_failure_evidence_sha256"),
        "report repair original failure evidence differs",
    )
    failure = read_json(failure_path)
    receipt_map = _receipt_file_map(submission_root)
    require(
        set(failure) == REPORT_REPAIR_FAILURE_KEYS
        and failure.get("schema_version") == 1
        and failure.get("status") == "original_report_terminal_failure_authenticated"
        and failure.get("submission_sha256") == submission_sha256
        and failure.get("original_report_job_id") == receipt.get("report_job_id")
        and failure.get("original_report_job_name")
        == f"exp23-launch8-{submission_sha256[:16]}-report"
        and failure.get("scheduler_comment") == f"treewm-exp23:{submission_sha256}"
        and failure.get("original_report_calling_sha256")
        == "e0fd250dcd21fc7a0a62b5da0fe2c3d95401a24e6ad92c4e806082867e623047"
        and failure.get("original_report_submitted_sha256")
        == "923f49755df3fcab99a547e0347b158ed42daef20cd640e24c605848b0769e57"
        and failure.get("submission_authorization_sha256")
        == "371ae8df4add6338b98469eca6a287902cb69325dfda9d5be6ce5b1600e6fd55"
        and failure.get("submission_receipt_sha256")
        == "58d1fd0f004efae049afd51e9592a79e963ba3fc8c2d3aae8a4af0bb7791a6a7"
        and failure.get("snapshot_root") == contract.get("snapshot_root")
        and failure.get("snapshot_inventory_sha256")
        == contract.get("snapshot_inventory_sha256")
        and failure.get("original_source_commit")
        == "33122e15d0aaf3661893a4c853fd5ac49173c685"
        and failure.get("original_package_protocol_sha256")
        == contract.get("package_protocol_sha256")
        and exact_json_equal(failure.get("worker_receipt_map"), receipt_map)
        and failure.get("worker_receipt_map_sha256") == stable_hash(receipt_map)
        and exact_json_equal(failure.get("expected_reassembly"), EXPECTED_REPAIR_REASSEMBLY)
        and exact_json_equal(value.get("worker_receipt_map"), receipt_map)
        and value.get("worker_receipt_map_sha256") == stable_hash(receipt_map),
        "report repair original terminal failure fields differ",
    )
    log = failure.get("report_log")
    require(
        isinstance(log, Mapping)
        and set(log)
        == {"path", "mode", "size", "uid", "nlink", "sha256", "encoding", "data"}
        and log.get("path") == "logs/report_33311218.out"
        and log.get("mode") == 0o600
        and log.get("size") == 384
        and log.get("uid") == os.getuid()
        and log.get("nlink") == 1
        and log.get("sha256")
        == "2c5a23103e00fc07196886c62e7c9d069ed1b011fb9f44095a4242cc926e43a6"
        and log.get("encoding") == "base64",
        "report repair failure-log evidence differs",
    )
    log_payload = _repair_stream_bytes(
        {
            "encoding": log["encoding"],
            "size": log["size"],
            "sha256": log["sha256"],
            "data": log["data"],
        },
        "report repair failure log",
    )
    require(
        b"staged report artifact identity differs" in log_payload
        and EXPECTED_REPAIR_REASSEMBLY["report_bundle_sha256"].encode("ascii")
        in log_payload,
        "report repair failure-log content differs",
    )
    terminal = failure.get("terminal_scheduler_observation")
    require(
        isinstance(terminal, Mapping)
        and terminal.get("schema_version") == 1
        and exact_json_equal(
            terminal.get("scheduler_control_plane"),
            contract.get("scheduler_control_plane_contract"),
        )
        and isinstance(terminal.get("canonical"), Mapping)
        and terminal.get("canonical_sha256") == stable_hash(terminal["canonical"])
        and isinstance(terminal.get("parsed_row"), Mapping),
        "report repair terminal scheduler observation differs",
    )
    parsed = terminal["parsed_row"]
    sacct_fields = [
        "JobIDRaw", "JobName", "State", "ExitCode", "ElapsedRaw", "AllocNodes",
        "NodeList", "Submit", "Eligible", "Start", "End", "Comment",
    ]
    require(
        set(parsed) == set(sacct_fields)
        and parsed.get("JobIDRaw") == "33311218"
        and parsed.get("JobName") == f"exp23-launch8-{submission_sha256[:16]}-report"
        and parsed.get("State") == "FAILED"
        and parsed.get("ExitCode") == "2:0"
        and parsed.get("ElapsedRaw") == "355"
        and parsed.get("AllocNodes") == "1"
        and parsed.get("NodeList") == "cpu-00090"
        and parsed.get("Start") == "2026-08-29T08:28:49"
        and parsed.get("End") == "2026-08-29T08:34:44"
        and _original_report_timeline_is_ordered(parsed)
        and parsed.get("Comment") == f"treewm-exp23:{submission_sha256}"
        and terminal["canonical"].get("fields") == sacct_fields
        and terminal["canonical"].get("rows")
        == [[parsed[field] for field in sacct_fields]],
        "report repair terminal scheduler row differs",
    )
    raw_sacct = _validated_repair_command_evidence(
        terminal.get("raw"), expected_argv=None, label="report repair terminal sacct"
    )
    require(
        raw_sacct["argv"]
        == [
            "/usr/local/bin/sacct", "-X", "-n", "-j", "33311218", "-o",
            ",".join(sacct_fields), "-P",
        ]
        and raw_sacct["returncode"] == 0
        and _repair_stream_bytes(raw_sacct["stderr"], "report repair terminal sacct stderr") == b"",
        "report repair terminal sacct command differs",
    )
    sacct_stdout = _repair_stream_bytes(
        raw_sacct["stdout"], "report repair terminal sacct stdout"
    )
    require(
        sacct_stdout
        == ("|".join(parsed[field] for field in sacct_fields) + "\n").encode(
            "utf-8"
        ),
        "report repair terminal sacct raw/canonical row differs",
    )
    pre = _validated_repair_census(
        failure.get("pre_submit_active_census"),
        submission_sha256=submission_sha256,
        label="report repair pre-submit",
    )
    require(pre["settled_rows"] == [], "report repair pre-submit jobs were active")
    require(
        exact_json_equal(
            failure.get("publication_state"),
            {
                "report_absent": True,
                "staging_entries": [],
                "cleanup_prefixes": [],
                "journal_directory": str(submission_root / "journal"),
            },
        ),
        "report repair original publication state differs",
    )

    source_root = nonsymlink_directory(
        Path(str(value.get("repair_source_root"))), "report repair source root"
    )
    require(
        source_root == submission_root / "report-repair" / "attempt-0001" / "source"
        and stat.S_IMODE(source_root.lstat().st_mode) == 0o555,
        "report repair source root differs",
    )
    source_files = value.get("repair_source_files")
    require(
        isinstance(source_files, Mapping)
        and set(source_files)
        == {"report.py", "report_repair.py", "report_repair.slurm", "protocol.sha256"}
        and value.get("repair_source_files_sha256") == stable_hash(source_files)
        and git_commit_string(value.get("repair_source_commit"))
        and sha256_string(value.get("repair_package_protocol_sha256")),
        "report repair source inventory/identity differs",
    )
    for name, expected in source_files.items():
        require(
            isinstance(expected, Mapping)
            and set(expected) == {"mode", "size", "sha256"}
            and expected.get("mode") == 0o444
            and type(expected.get("size")) is int
            and expected["size"] > 0
            and sha256_string(expected.get("sha256")),
            f"report repair source inventory row differs: {name}",
        )
        source_path = contained_regular(
            source_root / name, source_root, f"report repair source {name}"
        )
        source_info = source_path.lstat()
        require(
            stat.S_IMODE(source_info.st_mode) == 0o444
            and source_info.st_size == expected["size"]
            and source_info.st_uid == os.getuid()
            and source_info.st_nlink == 1
            and file_sha256(source_path) == expected["sha256"],
            f"report repair source bytes differ: {name}",
        )
    require(
        {entry.name for entry in os.scandir(source_root)}
        == {*source_files, REPORT_REPAIR_SOURCE_AUTHORITY_NAME},
        "report repair sealed source coverage differs",
    )
    source_authority_path = contained_regular(
        source_root / REPORT_REPAIR_SOURCE_AUTHORITY_NAME,
        source_root,
        "report repair sealed source authority",
    )
    source_authority_info = source_authority_path.lstat()
    source_authority = read_json(source_authority_path)
    require(
        stat.S_IMODE(source_authority_info.st_mode) == 0o444
        and source_authority_info.st_uid == os.getuid()
        and source_authority_info.st_nlink == 1
        and exact_json_equal(
            source_authority,
            {
                "schema_version": 1,
                "repair_source_commit": value["repair_source_commit"],
                "repair_package_protocol_sha256": value[
                    "repair_package_protocol_sha256"
                ],
                "repair_source_files": value["repair_source_files"],
                "repair_source_files_sha256": value[
                    "repair_source_files_sha256"
                ],
            },
        ),
        "report repair sealed source authority differs",
    )
    require(
        Path(__file__).absolute() == source_root / "report.py",
        "active report repair publisher is outside its sealed source",
    )
    protocol_payload, _protocol_sha, _protocol_info = _authenticated_regular_bytes(
        source_root / "protocol.sha256", "report repair protocol", capture=True
    )
    assert protocol_payload is not None
    require(
        protocol_payload == f"{value['repair_package_protocol_sha256']}\n".encode("ascii"),
        "report repair source protocol differs",
    )

    calling_path = contained_regular(
        submission_root / "journal/CALLING_REPORT_REPAIR_0001_SUBMIT.json",
        submission_root,
        "report repair submit calling",
    )
    submitted_path = contained_regular(
        submission_root / "journal/REPORT_REPAIR_0001_SUBMITTED.json",
        submission_root,
        "report repair submitted evidence",
    )
    require(
        file_sha256(calling_path) == value.get("submit_calling_sha256")
        and value.get("submitted_evidence")
        == "journal/REPORT_REPAIR_0001_SUBMITTED.json"
        and file_sha256(submitted_path) == value.get("submitted_evidence_sha256"),
        "report repair submit evidence hashes differ",
    )
    calling = read_json(calling_path)
    submitted = read_json(submitted_path)
    expected_submit_command = [
        "/usr/local/bin/sbatch",
        "--parsable",
        "--hold",
        "--no-requeue",
        "--export=NONE",
        f"--job-name={value['repair_job_name']}",
        f"--comment={value['scheduler_comment']}",
        f"--output={submission_root / 'logs/report-repair-0001-%j.out'}",
        str(source_root / "report_repair.slurm"),
        str(value["snapshot_root"]),
        str(submission_root),
        submission_sha256,
        "1",
        str(source_root),
        str(source_root / "report.py"),
        str(source_files["report.py"]["sha256"]),
        str(source_files["report.py"]["size"]),
    ]
    require(
        set(calling) == REPORT_REPAIR_SUBMIT_CALLING_KEYS
        and type(calling.get("schema_version")) is int
        and calling.get("schema_version") == 1
        and calling.get("status") == "calling_held_report_repair_submission"
        and calling.get("campaign_id") == CAMPAIGN_ID
        and calling.get("submission_sha256") == submission_sha256
        and type(calling.get("attempt")) is int
        and calling.get("attempt") == attempt
        and calling.get("original_failure_evidence")
        == "journal/REPORT_REPAIR_0001_ORIGINAL_FAILURE.json"
        and calling.get("original_failure_evidence_sha256")
        == value["original_failure_evidence_sha256"]
        and calling.get("repair_source_root") == str(source_root)
        and calling.get("repair_source_commit") == value["repair_source_commit"]
        and calling.get("repair_package_protocol_sha256")
        == value["repair_package_protocol_sha256"]
        and exact_json_equal(
            calling.get("repair_source_files"), value["repair_source_files"]
        )
        and calling.get("repair_source_files_sha256")
        == value["repair_source_files_sha256"]
        and calling.get("command") == expected_submit_command
        and isinstance(calling.get("scheduler_environment"), Mapping)
        and isinstance(calling.get("transaction_lock"), Mapping)
        and isinstance(calling.get("report_cancel_lock"), Mapping)
        and isinstance(calling.get("called_at_utc"), str)
        and bool(calling["called_at_utc"])
        and submitted.get("status") == "held_report_repair_submitted"
        and submitted.get("submit_calling_sha256") == value["submit_calling_sha256"]
        and submitted.get("repair_report_job_id") == value["repair_report_job_id"],
        "report repair submit evidence semantics differ",
    )
    pre_submit_census = _validated_repair_census(
        calling.get("scheduler_pre_submit_census"),
        submission_sha256=submission_sha256,
        label="report repair fresh pre-submit",
    )
    pre_submit_environment = pre_submit_census["rounds"][0]["raw"]["environment"]
    require(
        calling.get("scheduler_pre_submit_census_sha256")
        == stable_hash(pre_submit_census)
        and pre_submit_census["settled_rows"] == []
        and pre_submit_census["captured_at_utc"] <= calling["called_at_utc"]
        and all(
            exact_json_equal(
                round_value["raw"]["environment"], pre_submit_environment
            )
            for round_value in pre_submit_census["rounds"]
        )
        and exact_json_equal(
            calling["scheduler_environment"], pre_submit_environment
        ),
        "report repair fresh pre-submit scheduler authority differs",
    )
    authority_census = _validated_repair_census(
        value.get("scheduler_authority_census"),
        submission_sha256=submission_sha256,
        label="report repair authorization",
    )
    require(
        value.get("scheduler_authority_census_sha256") == stable_hash(authority_census)
        and len(authority_census["settled_rows"]) == 1,
        "report repair authorization census binding differs",
    )
    held = authority_census["settled_rows"][0]
    require(
        held.get("job_id") == value["repair_report_job_id"]
        and held.get("job_name") == value["repair_job_name"]
        and held.get("comment") == value["scheduler_comment"]
        and held.get("state") == "PENDING"
        and held.get("reason") in {"JobHeldUser", "JobHeldAdmin"},
        "report repair authorization held-job evidence differs",
    )
    authority_environment = authority_census["rounds"][0]["raw"]["environment"]
    require(
        all(
            exact_json_equal(
                round_value["raw"]["environment"], authority_environment
            )
            for round_value in authority_census["rounds"]
        )
        and exact_json_equal(authority_environment, pre_submit_environment),
        "report repair authorization scheduler environment changed",
    )

    release_path = contained_regular(
        _repair_release_path(submission_root, attempt),
        submission_root,
        "report repair release result",
    )
    release_info = release_path.lstat()
    release_sha256 = file_sha256(release_path)
    release = read_json(release_path)
    require(
        stat.S_IMODE(release_info.st_mode) == 0o444
        and release_info.st_uid == os.getuid()
        and release_info.st_nlink == 1
        and set(release) == REPORT_REPAIR_RELEASE_KEYS
        and type(release.get("schema_version")) is int
        and release.get("schema_version") == 1
        and release.get("status") == "report_repair_released"
        and release.get("campaign_id") == CAMPAIGN_ID
        and release.get("submission_sha256") == submission_sha256
        and type(release.get("attempt")) is int
        and release.get("attempt") == attempt
        and release.get("repair_report_job_id") == value["repair_report_job_id"]
        and release.get("authorization_sha256") == expected_raw_sha256
        and isinstance(release.get("release_attempts"), list)
        and bool(release["release_attempts"])
        and release.get("release_attempts_sha256") == stable_hash(release["release_attempts"])
        and isinstance(release.get("post_release_census"), Mapping)
        and release.get("post_release_census_sha256")
        == stable_hash(release["post_release_census"])
        and isinstance(release.get("worker_liveness_observation"), Mapping)
        and release.get("worker_liveness_observation_sha256")
        == stable_hash(release["worker_liveness_observation"])
        and isinstance(release.get("released_at_utc"), str)
        and bool(release["released_at_utc"]),
        "report repair release result differs",
    )
    release_modes: list[str] = []
    for index, entry in enumerate(release["release_attempts"]):
        require(
            isinstance(entry, Mapping)
            and set(entry)
            == {"release_attempt", "calling", "calling_sha256", "result", "result_sha256"}
            and type(entry.get("release_attempt")) is int
            and entry.get("release_attempt") == index
            and entry.get("calling")
            == f"CALLING_REPORT_REPAIR_0001_RELEASE_{index:04d}.json"
            and entry.get("result")
            == f"REPORT_REPAIR_0001_RELEASE_RESULT_{index:04d}.json"
            and sha256_string(entry.get("calling_sha256"))
            and sha256_string(entry.get("result_sha256")),
            "report repair release attempt chain differs",
        )
        release_calling_path = contained_regular(
            submission_root / "journal" / entry["calling"],
            submission_root,
            f"report repair release calling {index}",
        )
        release_result_path = contained_regular(
            submission_root / "journal" / entry["result"],
            submission_root,
            f"report repair release result {index}",
        )
        require(
            stat.S_IMODE(release_calling_path.lstat().st_mode) == 0o444
            and release_calling_path.lstat().st_uid == os.getuid()
            and release_calling_path.lstat().st_nlink == 1
            and stat.S_IMODE(release_result_path.lstat().st_mode) == 0o444
            and release_result_path.lstat().st_uid == os.getuid()
            and release_result_path.lstat().st_nlink == 1
            and file_sha256(release_calling_path) == entry["calling_sha256"]
            and file_sha256(release_result_path) == entry["result_sha256"],
            "report repair release attempt hashes differ",
        )
        release_calling = read_json(release_calling_path)
        release_result = read_json(release_result_path)
        require(
            set(release_calling) == REPORT_REPAIR_RELEASE_CALLING_KEYS
            and type(release_calling.get("schema_version")) is int
            and release_calling.get("schema_version") == 1
            and release_calling.get("status") == "calling_report_repair_release"
            and release_calling.get("campaign_id") == CAMPAIGN_ID
            and release_calling.get("submission_sha256") == submission_sha256
            and type(release_calling.get("attempt")) is int
            and release_calling.get("attempt") == attempt
            and type(release_calling.get("release_attempt")) is int
            and release_calling.get("release_attempt") == index
            and release_calling.get("repair_report_job_id")
            == value["repair_report_job_id"]
            and release_calling.get("authorization_sha256") == expected_raw_sha256
            and release_calling.get("command")
            == ["/usr/local/bin/scontrol", "release", value["repair_report_job_id"]]
            and exact_json_equal(
                release_calling.get("scheduler_environment"),
                authority_environment,
            )
            and isinstance(release_calling.get("transaction_lock"), Mapping)
            and isinstance(release_calling.get("report_cancel_lock"), Mapping)
            and isinstance(release_calling.get("called_at_utc"), str)
            and bool(release_calling["called_at_utc"])
            and set(release_result) == REPORT_REPAIR_RELEASE_RESULT_KEYS
            and type(release_result.get("schema_version")) is int
            and release_result.get("schema_version") == 1
            and release_result.get("status")
            == "report_repair_release_attempt_observed"
            and release_result.get("campaign_id") == CAMPAIGN_ID
            and release_result.get("submission_sha256") == submission_sha256
            and type(release_result.get("attempt")) is int
            and release_result.get("attempt") == attempt
            and type(release_result.get("release_attempt")) is int
            and release_result.get("release_attempt") == index
            and release_result.get("repair_report_job_id")
            == value["repair_report_job_id"]
            and release_result.get("release_calling_sha256")
            == entry["calling_sha256"]
            and release_result.get("authorization_sha256") == expected_raw_sha256,
            "report repair release attempt semantics differ",
        )
        mode = release_result.get("mode")
        release_modes.append(str(mode))
        evidence = release_result.get("scheduler_evidence")
        require(
            mode
            in {
                "direct_release_response",
                "lost_response_reconciled_still_held",
                "lost_response_reconciled_release_effect",
            }
            and isinstance(evidence, Mapping)
            and isinstance(release_result.get("observed_at_utc"), str)
            and bool(release_result["observed_at_utc"]),
            "report repair release result mode differs",
        )
        if mode == "direct_release_response":
            direct = _validated_repair_command_evidence(
                evidence,
                expected_argv=[
                    "/usr/local/bin/scontrol",
                    "release",
                    value["repair_report_job_id"],
                ],
                label=f"report repair direct release result {index}",
            )
            require(
                exact_json_equal(direct["environment"], authority_environment)
                and direct["returncode"] == 0
                and _repair_stream_bytes(
                    direct["stderr"], f"report repair direct release stderr {index}"
                )
                == b"",
                "report repair direct release result differs",
            )
        else:
            require(
                set(evidence) == {"census", "census_sha256"}
                and evidence.get("census_sha256") == stable_hash(evidence.get("census")),
                "report repair reconciled release evidence differs",
            )
            reconciled = _validated_repair_census(
                evidence.get("census"),
                submission_sha256=submission_sha256,
                label=f"report repair reconciled release {index}",
            )
            require(
                all(
                    exact_json_equal(
                        round_value["raw"]["environment"], authority_environment
                    )
                    for round_value in reconciled["rounds"]
                ),
                "report repair reconciled release environment differs",
            )
            reconciled_rows = [
                row
                for row in reconciled["settled_rows"]
                if row["job_name"] == value["repair_job_name"]
                and row["comment"] == value["scheduler_comment"]
            ]
            released_or_absent = not reconciled_rows or not (
                len(reconciled_rows) == 1
                and reconciled_rows[0]["job_id"] == value["repair_report_job_id"]
                and reconciled_rows[0]["state"] == "PENDING"
                and reconciled_rows[0]["reason"]
                in {"JobHeldUser", "JobHeldAdmin"}
            )
            require(
                len(reconciled["settled_rows"]) == len(reconciled_rows)
                and len(reconciled_rows) <= 1
                and (
                    not reconciled_rows
                    or reconciled_rows[0]["job_id"]
                    == value["repair_report_job_id"]
                )
                and (
                    mode == "lost_response_reconciled_release_effect"
                )
                == released_or_absent,
                "report repair reconciled release effect differs",
            )
    expected_release_callings = {
        str(entry["calling"]) for entry in release["release_attempts"]
    }
    expected_release_results = {
        str(entry["result"]) for entry in release["release_attempts"]
    }
    journal_names = {entry.name for entry in os.scandir(submission_root / "journal")}
    actual_release_callings = {
        name
        for name in journal_names
        if name.startswith("CALLING_REPORT_REPAIR_0001_RELEASE_")
    }
    actual_release_results = {
        name
        for name in journal_names
        if name.startswith("REPORT_REPAIR_0001_RELEASE_RESULT_")
    }
    require(
        len(release["release_attempts"]) <= 3
        and actual_release_callings == expected_release_callings
        and actual_release_results == expected_release_results
        and release_modes.count("lost_response_reconciled_release_effect") <= 1
        and (
            "lost_response_reconciled_release_effect" not in release_modes
            or release_modes[-1] == "lost_response_reconciled_release_effect"
        ),
        "report repair release namespace/prefix differs",
    )
    post_release = _validated_repair_census(
        release["post_release_census"],
        submission_sha256=submission_sha256,
        label="report repair post-release",
    )
    require(
        all(
            exact_json_equal(
                round_value["raw"]["environment"], authority_environment
            )
            for round_value in post_release["rounds"]
        ),
        "report repair post-release scheduler environment differs",
    )
    repair_rows = [
        row
        for row in post_release["settled_rows"]
        if row["job_name"] == value["repair_job_name"]
        and row["comment"] == value["scheduler_comment"]
    ]
    require(
        len(post_release["settled_rows"]) == len(repair_rows)
        and len(repair_rows) <= 1
        and (
            not repair_rows
            or (
                repair_rows[0]["job_id"] == value["repair_report_job_id"]
                and not (
                    repair_rows[0]["state"] == "PENDING"
                    and repair_rows[0]["reason"] in {"JobHeldUser", "JobHeldAdmin"}
                )
            )
        ),
        "report repair post-release scheduler state differs",
    )
    _validated_repair_worker_liveness(
        release["worker_liveness_observation"],
        post_release=post_release,
        repair_authorization=value,
        authority_environment=authority_environment,
        contract=contract,
        submission_sha256=submission_sha256,
    )
    restart_count = os.environ.get("SLURM_RESTART_COUNT")
    require(
        os.environ.get("SLURM_JOB_ID") == value["repair_report_job_id"]
        and restart_count == "0"
        and "SLURM_ARRAY_JOB_ID" not in os.environ
        and "SLURM_ARRAY_TASK_ID" not in os.environ,
        "active report repair scheduler identity/restart differs",
    )
    result = dict(value)
    result["_validated_release_sha256"] = release_sha256
    return result


def _repair_publication_authority(
    repair: Mapping[str, Any], authorization_sha256: str, attempt: int
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "authorized_terminal_report_repair",
        "attempt": attempt,
        "authorization": f"journal/REPORT_REPAIR_{attempt:04d}_AUTHORIZED.json",
        "authorization_sha256": authorization_sha256,
        "release": f"journal/REPORT_REPAIR_{attempt:04d}_RELEASED.json",
        "release_sha256": repair["_validated_release_sha256"],
        "original_report_job_id": repair["original_report_job_id"],
        "repair_report_job_id": repair["repair_report_job_id"],
        "original_failure_evidence": repair["original_failure_evidence"],
        "original_failure_evidence_sha256": repair[
            "original_failure_evidence_sha256"
        ],
        "worker_receipt_map_sha256": repair["worker_receipt_map_sha256"],
        "original_snapshot_root": repair["snapshot_root"],
        "original_snapshot_inventory_sha256": repair[
            "snapshot_inventory_sha256"
        ],
        "original_package_protocol_sha256": repair[
            "original_package_protocol_sha256"
        ],
        "repair_source_root": repair["repair_source_root"],
        "repair_source_commit": repair["repair_source_commit"],
        "repair_package_protocol_sha256": repair[
            "repair_package_protocol_sha256"
        ],
        "repair_source_files_sha256": repair["repair_source_files_sha256"],
        "expected_report_bundle_sha256": EXPECTED_REPAIR_REASSEMBLY[
            "report_bundle_sha256"
        ],
        "expected_report_bundle_file_sha256": EXPECTED_REPAIR_REASSEMBLY[
            "report_bundle_file_sha256"
        ],
        "expected_gate_sha256": EXPECTED_REPAIR_REASSEMBLY["gate_sha256"],
        "expected_gate_decision_file_sha256": EXPECTED_REPAIR_REASSEMBLY[
            "gate_decision_file_sha256"
        ],
        "deterministic_reassembly_allowed": True,
        "scientific_input_change_allowed": False,
        "gate_change_allowed": False,
    }


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
    require(Path(str(module.__file__)).absolute() == resolved, f"{name} imported outside snapshot")
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

    from tensorboard.backend.event_processing import event_file_loader
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c

    run_root = nonsymlink_directory(run_dir, "scientific run directory")
    tree_rows = _secure_tree_rows(
        run_root,
        "scientific run tree",
        hash_files=True,
        allow_wandb_symlink_leaves=True,
    )
    initial_rows = {str(row["path"]): row for row in tree_rows}
    event_relatives = sorted(
        Path(str(row["path"]))
        for row in tree_rows
        if row["kind"] == "file"
        and Path(str(row["path"])).name.startswith("events.out.tfevents.")
    )
    # Only root writers are training generations.  The terminal hparams writer lives
    # below hparams/ and intentionally has no fixed-validation text.
    path_relatives = [relative for relative in event_relatives if len(relative.parts) == 1]
    hparam_relatives = [
        relative
        for relative in event_relatives
        if len(relative.parts) >= 2 and relative.parts[0] == "hparams"
    ]
    require(
        set(event_relatives) == set(path_relatives) | set(hparam_relatives),
        "unexpected TensorBoard event file outside root/hparams writers",
    )
    require(path_relatives, f"no TensorBoard event files in {run_root}")
    merged: dict[str, dict[int, tuple[bytes, float]]] = {}
    duplicate_counts: dict[str, int] = {}
    expected_text = "<pre>" + json.dumps(
        dict(expected_sampler), sort_keys=True, indent=2, allow_nan=False
    ) + "</pre>"
    fixed_text_events = 0
    excluded_eval_tags: set[str] = set()
    event_hashes: dict[str, str] = {}
    # Parse only a private exact-byte copy.  The shared source is reauthenticated
    # afterward, so EventAccumulator can neither pathname-reopen an ABA replacement
    # nor certify bytes which differ from the live sealed event inode.
    temporary_parent = nonsymlink_directory(Path("/tmp"), "event-copy temporary parent")
    with tempfile.TemporaryDirectory(
        prefix=f"treewm-exp23-report-events-{os.getuid()}-", dir=temporary_parent
    ) as raw_event_root:
        private_root = Path(raw_event_root)
        for index, relative in enumerate(path_relatives):
            path = run_root / relative
            private_path = private_root / f"event-{index:03d}.tfevents"
            private_fd = os.open(
                private_path,
                os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
            try:
                _payload, before_hash, before_info = _authenticated_relative_regular(
                    run_root,
                    relative,
                    "TensorBoard event file",
                    capture=False,
                    copy_fd=private_fd,
                )
                os.fsync(private_fd)
                initial = initial_rows.get(str(relative))
                require(initial is not None and initial.get("kind") == "file", f"event vanished from initial inventory: {relative}")
                require(
                    initial.get("sha256") == before_hash
                    and (
                        initial.get("device"),
                        initial.get("inode"),
                        initial.get("mode"),
                        initial.get("uid"),
                        initial.get("gid"),
                        initial.get("nlink"),
                        initial.get("size"),
                        initial.get("mtime_ns"),
                        initial.get("ctime_ns"),
                    )
                    == (
                        before_info.st_dev,
                        before_info.st_ino,
                        stat.S_IMODE(before_info.st_mode),
                        before_info.st_uid,
                        before_info.st_gid,
                        before_info.st_nlink,
                        before_info.st_size,
                        before_info.st_mtime_ns,
                        before_info.st_ctime_ns,
                    ),
                    f"TensorBoard event differs from initial inventory: {relative}",
                )
                private_info = os.fstat(private_fd)
                require(
                    stat.S_ISREG(private_info.st_mode)
                    and private_info.st_uid == os.getuid()
                    and private_info.st_nlink == 1
                    and stat.S_IMODE(private_info.st_mode) == 0o600,
                    f"private TensorBoard copy is unsafe: {relative}",
                )
                os.unlink(private_path)
                private_hash, private_before = _stable_open_fd_sha256(
                    private_fd, f"private TensorBoard copy {relative}"
                )
                require(private_hash == before_hash, f"private TensorBoard copy differs: {relative}")
                _verify_tfrecord_fd(
                    private_fd,
                    f"private TensorBoard copy {relative}",
                    masked_crc32c,
                )

                # EventAccumulator only accepts a pathname.  Pin its loader to the
                # anonymous exact-copy inode through /proc/self/fd while retaining
                # the descriptor for the entire parse; no mutable directory entry
                # is reopened.
                accumulator = EventAccumulator(
                    str(private_root),
                    size_guidance={"scalars": 0, "tensors": 0},
                    purge_orphaned_data=False,
                )
                accumulator._generator = event_file_loader.LegacyEventFileLoader(  # type: ignore[attr-defined]
                    f"/proc/self/fd/{private_fd}"
                )
                try:
                    accumulator.Reload()
                except Exception as exc:
                    raise ReportError(f"unreadable TensorBoard event file {path}: {exc}") from exc
                tags = accumulator.Tags()
                for tag in tags.get("scalars", []):
                    require(isinstance(tag, str) and tag, f"invalid scalar tag in {path}")
                    if tag.startswith("eval/"):
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
                del accumulator
                private_after_hash, private_after = _stable_open_fd_sha256(
                    private_fd, f"private TensorBoard copy {relative}"
                )
                require(
                    private_after_hash == private_hash
                    and _file_identity(private_after) == _file_identity(private_before),
                    f"private TensorBoard copy changed while reporting: {relative}",
                )
            finally:
                os.close(private_fd)
            _payload, after_hash, after_info = _authenticated_relative_regular(
                run_root, relative, "TensorBoard event file", capture=False
            )
            require(
                _file_identity(after_info) == _file_identity(before_info)
                and after_hash == before_hash,
                f"TensorBoard event file changed while reporting: {path}",
            )
            event_hashes[str(relative)] = before_hash
    hparams_hashes: dict[str, str] = {}
    for relative in hparam_relatives:
        descriptor, info = _open_relative_regular(
            run_root, relative, "TensorBoard hparams event file"
        )
        try:
            digest, stable_info = _stable_open_fd_sha256(
                descriptor, f"TensorBoard hparams event file {relative}"
            )
            require(
                _file_identity(stable_info) == _file_identity(info),
                f"hparams event open identity differs: {relative}",
            )
            _verify_tfrecord_fd(
                descriptor,
                f"TensorBoard hparams event file {relative}",
                masked_crc32c,
            )
        finally:
            os.close(descriptor)
        initial = initial_rows.get(str(relative))
        require(
            initial is not None
            and initial.get("sha256") == digest
            and initial.get("device") == info.st_dev
            and initial.get("inode") == info.st_ino
            and initial.get("ctime_ns") == info.st_ctime_ns,
            f"hparams event differs from initial inventory: {relative}",
        )
        hparams_hashes[str(relative)] = digest
    require(
        _secure_tree_rows(
            run_root,
            "scientific run tree",
            hash_files=True,
            allow_wandb_symlink_leaves=True,
        )
        == tree_rows,
        "scientific run tree changed while parsing TensorBoard events",
    )
    require(fixed_text_events == len(path_relatives), "fixed-validation text generation coverage differs")
    scalars = {
        tag: {step: item[1] for step, item in sorted(points.items())}
        for tag, points in sorted(merged.items())
    }
    return {
        "scalars": scalars,
        "event_files": [str(path) for path in path_relatives],
        "event_file_sha256": event_hashes,
        "hparams_event_files": [str(path) for path in hparam_relatives],
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
        *gate.DENSE_TRAIN_METHOD_TAGS,
        *(gate.TRAIN_PREFIX + suffix for suffix in gate.PREFIX_COMMON_SUFFIXES),
    )
    dense_method_tags = frozenset(gate.DENSE_TRAIN_METHOD_TAGS)
    require(
        dense_method_tags <= frozenset(gate.METHOD_EXACT_TAGS),
        "dense training method tags are not a subset of exact method tags",
    )
    validation_tags = (
        *(
            tag
            for tag in gate.METHOD_EXACT_TAGS
            if tag != "data/validation_fixed_sample_count"
            and tag not in dense_method_tags
        ),
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
    wave0_array_job_id: str,
    wave1_array_job_id: str,
    submission_authorization_sha256: str,
    task_root: Path,
    terminal: Mapping[str, Any],
) -> dict[str, Any]:
    require(set(marker) == WORKER_COMPLETE_KEYS, f"cell{index}: WORKER_COMPLETE fields differ")
    require(
        type(marker["schema_version"]) is int and marker["schema_version"] == 1,
        f"cell{index}: worker receipt schema differs",
    )
    require(marker["status"] == "worker_complete", f"cell{index}: worker receipt status differs")
    require(marker["campaign_id"] == CAMPAIGN_ID, f"cell{index}: worker receipt campaign differs")
    require(marker["submission_sha256"] == submission_sha256, f"cell{index}: worker receipt submission differs")
    require(marker["launch_sha256"] == launch["launch_sha256"], f"cell{index}: worker receipt launch differs")
    require(type(marker["cell_index"]) is int and marker["cell_index"] == index, f"cell{index}: worker receipt cell differs")
    require(
        type(marker["wave_index"]) is int and marker["wave_index"] in {0, 1},
        f"cell{index}: completion wave differs",
    )
    require(
        isinstance(marker["array_job_id"], str)
        and marker["array_job_id"]
        and marker["array_job_id"][0] in "123456789"
        and all(character in "0123456789" for character in marker["array_job_id"]),
        f"cell{index}: array job ID differs",
    )
    expected_array_id = wave0_array_job_id if marker["wave_index"] == 0 else wave1_array_job_id
    expected_predecessor = "none" if marker["wave_index"] == 0 else wave0_array_job_id
    require(
        marker["array_job_id"] == expected_array_id
        and marker["predecessor_array_job_id"] == expected_predecessor
        and marker["submission_authorization_sha256"]
        == submission_authorization_sha256,
        f"cell{index}: worker DAG authorization differs",
    )
    require(type(marker["array_task_id"]) is int and marker["array_task_id"] == index, f"cell{index}: array task differs")

    def exact_base(value: Mapping[str, Any], wave_index: int) -> None:
        expected_base = {
            "schema_version": 1,
            "campaign_id": CAMPAIGN_ID,
            "submission_sha256": submission_sha256,
            "launch_sha256": launch["launch_sha256"],
            "cell_index": index,
            "wave_index": wave_index,
            "array_job_id": (
                wave0_array_job_id if wave_index == 0 else wave1_array_job_id
            ),
            "array_task_id": index,
            "predecessor_array_job_id": (
                "none" if wave_index == 0 else wave0_array_job_id
            ),
            "submission_authorization_sha256": submission_authorization_sha256,
        }
        require(
            all(
                exact_json_equal(value.get(key), expected)
                for key, expected in expected_base.items()
            ),
            f"cell{index}: wave{wave_index} artifact base differs",
        )

    wave0_start_path = task_root / "waves" / "0" / "START.json"
    contained_regular(wave0_start_path, task_root, f"cell{index} wave-zero START")
    require(
        stat.S_IMODE(wave0_start_path.lstat().st_mode) == 0o444,
        f"cell{index}: wave-zero START mode differs",
    )
    wave0_start = read_json(wave0_start_path)
    require(
        set(wave0_start) == GENERATION_START_KEYS
        and wave0_start.get("status") == "wave_started"
        and wave0_start.get("input_kind") == "fresh"
        and wave0_start.get("input_checkpoint_sha256") is None
        and wave0_start.get("predecessor_evidence_sha256") is None,
        f"cell{index}: wave-zero scratch START differs",
    )
    exact_base(wave0_start, 0)
    wave0_start_sha256 = file_sha256(wave0_start_path)
    for key in (
        "completed_updates",
        "checkpoint_sha256",
        "completion_sha256",
        "final_eval_progress_sha256",
        "completed_results_sha256",
        "identity_sha256",
    ):
        require(
            exact_json_equal(marker[key], terminal[key]),
            f"cell{index}: worker receipt {key} differs from independent verification",
        )
    require(_finite_mapping(marker["final_metrics"], f"cell{index} worker metrics") == terminal["final_metrics"], f"cell{index}: worker receipt metrics differ")
    wave_marker_path = task_root / "waves" / str(marker["wave_index"]) / "WORKER_COMPLETE.json"
    contained_regular(wave_marker_path, task_root, f"cell{index} wave completion")
    require(
        stat.S_IMODE(wave_marker_path.lstat().st_mode) == 0o444
        and stat.S_IMODE((task_root / "WORKER_COMPLETE.json").lstat().st_mode) == 0o444,
        f"cell{index}: worker completion marker mode differs",
    )
    require(
        exact_json_equal(read_json(wave_marker_path), marker)
        and file_sha256(wave_marker_path) == file_sha256(task_root / "WORKER_COMPLETE.json"),
        f"cell{index}: task/wave completion markers differ",
    )
    if marker["wave_index"] == 0:
        require(
            not _lexical_exists(task_root / "waves" / "0" / "CONTINUATION_READY.json"),
            f"cell{index}: wave-zero completion conflicts with continuation READY",
        )
        noop_path = task_root / "waves" / "1" / "WORKER_COMPLETE.json"
        contained_regular(noop_path, task_root, f"cell{index} wave-one no-op completion")
        require(stat.S_IMODE(noop_path.lstat().st_mode) == 0o444, f"cell{index}: wave-one no-op mode differs")
        noop = read_json(noop_path)
        require(
            set(noop) == WORKER_COMPLETE_KEYS
            and noop["status"] == "wave_one_noop_after_wave_zero_complete"
            and type(noop["wave_index"]) is int
            and noop["wave_index"] == 1
            and noop["array_job_id"] == wave1_array_job_id
            and noop["predecessor_array_job_id"] == wave0_array_job_id
            and noop["submission_authorization_sha256"]
            == submission_authorization_sha256,
            f"cell{index}: wave-one completion no-op differs",
        )
        for key in WORKER_COMPLETE_KEYS - {
            "status", "wave_index", "array_job_id", "predecessor_array_job_id"
        }:
            require(
                exact_json_equal(noop[key], marker[key]),
                f"cell{index}: wave-one no-op payload differs: {key}",
            )
        wave1_start_path = task_root / "waves" / "1" / "START.json"
        contained_regular(wave1_start_path, task_root, f"cell{index} wave-one no-op START")
        require(stat.S_IMODE(wave1_start_path.lstat().st_mode) == 0o444, f"cell{index}: wave-one no-op START mode differs")
        wave1_start = read_json(wave1_start_path)
        wave0_complete_sha256 = file_sha256(wave_marker_path)
        require(
            set(wave1_start) == GENERATION_START_KEYS
            and wave1_start.get("status") == "wave_started"
            and wave1_start.get("input_kind") == "complete"
            and wave1_start.get("input_checkpoint_sha256")
            == marker["checkpoint_sha256"]
            and wave1_start.get("predecessor_evidence_sha256")
            == wave0_complete_sha256,
            f"cell{index}: wave-one no-op predecessor lineage differs",
        )
        exact_base(wave1_start, 1)
        return {
            "branch": "wave0_complete_wave1_noop",
            "wave0_start_sha256": wave0_start_sha256,
            "wave0_predecessor_evidence_name": "WORKER_COMPLETE.json",
            "wave0_predecessor_evidence_sha256": wave0_complete_sha256,
            "wave0_checkpoint_sha256": marker["checkpoint_sha256"],
            "wave1_start_sha256": file_sha256(wave1_start_path),
            "wave1_input_checkpoint_sha256": wave1_start["input_checkpoint_sha256"],
            "wave1_predecessor_evidence_sha256": wave1_start[
                "predecessor_evidence_sha256"
            ],
            "wave1_noop_sha256": file_sha256(noop_path),
        }

    ready_path = task_root / "waves" / "0" / "CONTINUATION_READY.json"
    contained_regular(ready_path, task_root, f"cell{index} wave-zero continuation READY")
    require(stat.S_IMODE(ready_path.lstat().st_mode) == 0o444, f"cell{index}: wave-zero READY mode differs")
    require(
        not _lexical_exists(task_root / "waves" / "0" / "WORKER_COMPLETE.json"),
        f"cell{index}: wave-zero READY conflicts with completion",
    )
    ready = read_json(ready_path)
    exact_base(ready, 0)
    ready_identity = ready.get("checkpoint_file_identity")
    require(
        set(ready) == CONTINUATION_READY_KEYS
        and ready.get("status") == "continuation_ready"
        and type(ready.get("trainer_exit_code")) is int
        and ready.get("trainer_exit_code") == 75
        and isinstance(ready.get("checkpoint_kind"), str)
        and bool(ready["checkpoint_kind"])
        and type(ready.get("completed_updates")) is int
        and 0 <= ready["completed_updates"] <= 25_000
        and isinstance(ready.get("phase"), str)
        and bool(ready["phase"])
        and (
            ready.get("pending_eval_step") is None
            or type(ready.get("pending_eval_step")) is int
        )
        and isinstance(ready.get("checkpoint_sha256"), str)
        and len(ready["checkpoint_sha256"]) == 64
        and set(ready["checkpoint_sha256"]) <= SHA256,
        f"cell{index}: wave-zero continuation READY differs",
    )
    require(
        isinstance(ready_identity, Mapping)
        and set(ready_identity) == {"device", "inode", "size", "mtime_ns", "ctime_ns"}
        and all(
            type(ready_identity.get(key)) is int and ready_identity[key] >= 0
            for key in ready_identity
        )
        and (
            ready.get("final_eval_progress_sha256") is None
            or (
                isinstance(ready["final_eval_progress_sha256"], str)
                and len(ready["final_eval_progress_sha256"]) == 64
                and set(ready["final_eval_progress_sha256"]) <= SHA256
            )
        ),
        f"cell{index}: wave-zero continuation READY checkpoint evidence differs",
    )
    ready_sha256 = file_sha256(ready_path)
    wave1_start_path = task_root / "waves" / "1" / "START.json"
    contained_regular(wave1_start_path, task_root, f"cell{index} wave-one resume START")
    require(stat.S_IMODE(wave1_start_path.lstat().st_mode) == 0o444, f"cell{index}: wave-one resume START mode differs")
    wave1_start = read_json(wave1_start_path)
    exact_base(wave1_start, 1)
    require(
        set(wave1_start) == GENERATION_START_KEYS
        and wave1_start.get("status") == "wave_started"
        and wave1_start.get("input_kind") == ready["checkpoint_kind"]
        and wave1_start.get("input_checkpoint_sha256")
        == ready["checkpoint_sha256"]
        and wave1_start.get("predecessor_evidence_sha256") == ready_sha256,
        f"cell{index}: wave-one checkpoint predecessor lineage differs",
    )
    return {
        "branch": "wave0_ready_wave1_resume",
        "wave0_start_sha256": wave0_start_sha256,
        "wave0_predecessor_evidence_name": "CONTINUATION_READY.json",
        "wave0_predecessor_evidence_sha256": ready_sha256,
        "wave0_checkpoint_sha256": ready["checkpoint_sha256"],
        "wave1_start_sha256": file_sha256(wave1_start_path),
        "wave1_input_checkpoint_sha256": wave1_start["input_checkpoint_sha256"],
        "wave1_predecessor_evidence_sha256": wave1_start[
            "predecessor_evidence_sha256"
        ],
        "wave1_noop_sha256": None,
    }


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
    *,
    require_publish_job: bool = False,
    repair_attempt: int | None = None,
    repair_authorization_sha256: str | None = None,
    allow_repair_cleanup_for_audit: bool = False,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    reject_environment()
    snapshot_root = nonsymlink_directory(snapshot_root, "snapshot root")
    submission_root = nonsymlink_directory(submission_root, "submission root")
    require(sha256_string(submission_sha256), "submission SHA256 is malformed")
    require(
        type(allow_repair_cleanup_for_audit) is bool
        and (
            not allow_repair_cleanup_for_audit
            or (
                not require_publish_job
                and repair_attempt is None
                and repair_authorization_sha256 is None
            )
        ),
        "repair cleanup audit mode cannot authorize publication",
    )
    require(
        not _durable_original_cleanup_prefix_exists(submission_root),
        "cancelled/ambiguous submission cannot report",
    )
    if not allow_repair_cleanup_for_audit:
        require(
            not _durable_repair_stop_prefix_exists(submission_root),
            "terminal report repair state cannot publish",
        )
    contract, receipt = verify_snapshot_inventory(
        snapshot_root,
        submission_root,
        submission_sha256,
        require_publish_job=require_publish_job,
        repair_attempt=repair_attempt,
        repair_authorization_sha256=repair_authorization_sha256,
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
    # Revalidate at the import boundary.  Same-UID malicious processes are trusted
    # (they could ptrace this process); no ambient snapshot writer has been launched,
    # so this catches accidental/concurrent path drift before executable bytes load.
    second_contract, second_receipt = verify_snapshot_inventory(
        snapshot_root,
        submission_root,
        submission_sha256,
        require_publish_job=require_publish_job,
        repair_attempt=repair_attempt,
        repair_authorization_sha256=repair_authorization_sha256,
    )
    require(
        second_contract == contract and second_receipt == receipt,
        "snapshot/submission changed before report imports",
    )
    campaign, worker, gate = _context_modules(snapshot_root, runtime_interpreter)
    package = snapshot_root / PACKAGE_RELATIVE
    manifest, _weight_lock = campaign.load_contract(snapshot_root)
    require(manifest == bootstrap_manifest, "manifest changed during report bootstrap")
    protocol = campaign.verify_protocol_lock(package)
    production_authorization_prerequisite = (
        _validated_production_authorization_prerequisite(manifest, protocol)
    )
    require(
        exact_json_equal(
            contract.get("production_authorization_prerequisite"),
            production_authorization_prerequisite,
        ),
        "report successful-canary prerequisite differs from submission contract",
    )
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
        wave_lineage = validate_worker_receipt(
            marker,
            index=index,
            launch=context["launch"],
            submission_sha256=submission_sha256,
            terminal=terminal,
            wave0_array_job_id=receipt["wave0_array_job_id"],
            wave1_array_job_id=receipt["wave1_array_job_id"],
            submission_authorization_sha256=str(
                receipt["submission_authorization_sha256"]
            ),
            task_root=marker_path.parent,
        )
        cells[index]["boundaries"]["25000"]["outcome"] = _outcome_from_terminal(terminal)
        terminal_provenance.append(
            {
                "index": index,
                "worker_complete_sha256": file_sha256(marker_path),
                "wave_index": marker["wave_index"],
                "array_job_id": marker["array_job_id"],
                "checkpoint_sha256": complete["checkpoint_sha256"],
                "completion_sha256": complete["completion_sha256"],
                "final_eval_progress_sha256": complete["final_eval_progress_sha256"],
                "completed_results_sha256": complete["completed_results_sha256"],
                "identity_sha256": complete["identity_sha256"],
                "wave_lineage": wave_lineage,
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
        "production_authorization_prerequisite": (
            production_authorization_prerequisite
        ),
        "production_authorization_prerequisite_sha256": stable_hash(
            production_authorization_prerequisite
        ),
        "outcome_blind_phase": outcome_blind_phase,
        "event_artifacts": event_provenance,
        "terminal_artifacts": terminal_provenance,
        "report_bundle_sha256": stable_hash(bundle),
        "gate_sha256": decision["gate_sha256"],
    }
    if repair_attempt is not None:
        assert repair_authorization_sha256 is not None
        bundle_payload = (
            json.dumps(dict(bundle), sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        decision_payload = (
            json.dumps(dict(decision), sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        provenance_v1_payload = (
            json.dumps(dict(provenance), sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        require(
            stable_hash(bundle) == EXPECTED_REPAIR_REASSEMBLY["report_bundle_sha256"]
            and decision.get("status") == EXPECTED_REPAIR_REASSEMBLY["status"]
            and decision.get("gate_sha256")
            == EXPECTED_REPAIR_REASSEMBLY["gate_sha256"]
            and len(bundle_payload)
            == EXPECTED_REPAIR_REASSEMBLY["report_bundle_file_size"]
            and hashlib.sha256(bundle_payload).hexdigest()
            == EXPECTED_REPAIR_REASSEMBLY["report_bundle_file_sha256"]
            and len(decision_payload)
            == EXPECTED_REPAIR_REASSEMBLY["gate_decision_file_size"]
            and hashlib.sha256(decision_payload).hexdigest()
            == EXPECTED_REPAIR_REASSEMBLY["gate_decision_file_sha256"]
            and stable_hash(provenance)
            == EXPECTED_REPAIR_REASSEMBLY["original_provenance_v1_sha256"]
            and len(provenance_v1_payload)
            == EXPECTED_REPAIR_REASSEMBLY["original_provenance_v1_file_size"]
            and hashlib.sha256(provenance_v1_payload).hexdigest()
            == EXPECTED_REPAIR_REASSEMBLY[
                "original_provenance_v1_file_sha256"
            ],
            "report repair deterministic reassembly differs from the failed reporter",
        )
        repair = _validated_report_repair_authorization(
            submission_root,
            submission_sha256,
            receipt,
            attempt=repair_attempt,
            expected_raw_sha256=repair_authorization_sha256,
        )
        provenance["schema_version"] = 2
        provenance["publication_authority"] = _repair_publication_authority(
            repair, repair_authorization_sha256, repair_attempt
        )
    return bundle, decision, provenance


def _fsync_directory(path: Path) -> None:
    descriptor = _open_directory_components(path, f"fsync directory {path}")
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def seal_json(path: Path, value: Mapping[str, Any]) -> str:
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
        contained_regular(path, path.parent, f"immutable report artifact {path.name}")
        existing, _existing_digest, _existing_info = _authenticated_regular_bytes(
            path, f"immutable report artifact {path.name}", capture=True
        )
        require(existing == payload, f"immutable report artifact differs: {path}")
        return digest
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, f"short report artifact write: {path}")
            view = view[written:]
        os.fsync(descriptor)
        # Creation modes are filtered by the inherited process umask.  Seal via
        # the retained descriptor so a strict controller umask (notably 0077)
        # cannot turn the intended 0444 evidence into 0400, and never relax the
        # subsequent identity/size/owner/link predicates.
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        require(
            stat.S_ISREG(opened.st_mode)
            and opened.st_uid == os.getuid()
            and opened.st_nlink == 1
            and stat.S_IMODE(opened.st_mode) == 0o444
            and opened.st_size == len(payload),
            f"staged report artifact identity differs: {path}",
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        readback = bytearray()
        while len(readback) < len(payload):
            chunk = os.read(
                descriptor, min(1024 * 1024, len(payload) - len(readback))
            )
            require(bool(chunk), f"short staged report artifact readback: {path}")
            readback.extend(chunk)
        require(
            os.read(descriptor, 1) == b""
            and bytes(readback) == payload
            and hashlib.sha256(readback).hexdigest() == digest,
            f"staged report artifact readback differs: {path}",
        )
        named = path.lstat()
        reopened = os.fstat(descriptor)
        require(
            _file_identity(named) == _file_identity(reopened)
            and stat.S_IMODE(named.st_mode) == 0o444
            and named.st_size == len(payload),
            f"staged report artifact path identity differs: {path}",
        )
    except BaseException:
        os.close(descriptor)
        path.unlink(missing_ok=True)
        raise
    else:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return digest


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically publish one directory without replacing any target entry."""

    require(source.parent == target.parent, "report publication rename parents differ")
    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    require(renameat2 is not None, "renameat2 is unavailable for no-replace publication")
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        -100,  # AT_FDCWD
        os.fsencode(source),
        -100,
        os.fsencode(target),
        1,  # RENAME_NOREPLACE
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), str(target))
    raise OSError(error, os.strerror(error), f"{source} -> {target}")


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
        rows = _secure_tree_rows(root, "published report tree", hash_files=True)
        root_rows = [row for row in rows if row["kind"] == "root"]
        require(
            len(root_rows) == 1 and int(root_rows[0]["mode"]) == 0o555,
            "published report root mode differs",
        )
        file_rows = {
            str(row["path"]): row for row in rows if row["kind"] == "file"
        }
        require(
            len(file_rows) == len(rows) - 1,
            "published report contains a non-file entry",
        )
        actual = set(file_rows)
        require(actual == expected_names, "published report file coverage differs")
        commit_path = contained_regular(root / "REPORT_COMMIT.json", root, "published report commit")
        require(int(file_rows["REPORT_COMMIT.json"]["mode"]) == 0o444, "published report commit mode differs")
        require(read_json(commit_path) == commit, "published report commit differs")
        for name, digest in (
            (bundle_name, bundle_payload_sha),
            (decision_name, decision_payload_sha),
            (provenance_name, provenance_payload_sha),
        ):
            path = contained_regular(root / name, root, f"published report {name}")
            require(int(file_rows[name]["mode"]) == 0o444, f"published report file mode differs: {name}")
            require(file_rows[name]["sha256"] == digest, f"published report file differs: {name}")

    if _lexical_exists(report_root):
        validate_existing()
        return commit

    staging = submission_root / f".report.tmp.{os.getpid()}.{time.time_ns()}"
    staging.mkdir(mode=0o700)
    _fsync_directory(submission_root)
    try:
        require(seal_json(staging / bundle_name, bundle) == bundle_payload_sha, "staged bundle hash differs")
        require(seal_json(staging / decision_name, decision) == decision_payload_sha, "staged decision hash differs")
        require(seal_json(staging / provenance_name, provenance) == provenance_payload_sha, "staged provenance hash differs")
        commit_payload_sha = seal_json(staging / "REPORT_COMMIT.json", commit)
        _fsync_directory(staging)
        staged_rows = _secure_tree_rows(
            staging, "completed report staging tree", hash_files=True
        )
        staged_root_rows = [row for row in staged_rows if row["kind"] == "root"]
        staged_files = {
            str(row["path"]): row
            for row in staged_rows
            if row["kind"] == "file"
        }
        require(
            len(staged_root_rows) == 1
            and int(staged_root_rows[0]["mode"]) == 0o700
            and len(staged_files) == len(staged_rows) - 1
            and set(staged_files) == expected_names
            and all(int(row["mode"]) == 0o444 for row in staged_files.values())
            and staged_files[bundle_name]["sha256"] == bundle_payload_sha
            and staged_files[decision_name]["sha256"] == decision_payload_sha
            and staged_files[provenance_name]["sha256"] == provenance_payload_sha
            and staged_files["REPORT_COMMIT.json"]["sha256"]
            == commit_payload_sha,
            "completed report staging inventory differs",
        )
        staging_fd = _open_directory_components(staging, "report staging root")
        try:
            staging_info = os.fstat(staging_fd)
            require(
                staging_info.st_uid == os.getuid()
                and stat.S_IMODE(staging_info.st_mode) == 0o700,
                "report staging root identity differs",
            )
            os.fchmod(staging_fd, 0o555)
            os.fsync(staging_fd)
            require(
                stat.S_IMODE(os.fstat(staging_fd).st_mode) == 0o555,
                "report staging root seal differs",
            )
        finally:
            os.close(staging_fd)
        _fsync_directory(submission_root)
        require(
            not _durable_cleanup_prefix_exists(submission_root),
            "cancellation/cleanup prefix appeared before report commit",
        )
        require(not _lexical_exists(report_root), "report publication target appeared concurrently")
        _rename_directory_noreplace(staging, report_root)
        _fsync_directory(submission_root)
    except BaseException:
        if _lexical_exists(staging):
            try:
                staging_fd = _open_directory_components(staging, "failed report staging root")
                try:
                    staging_info = os.fstat(staging_fd)
                    require(
                        staging_info.st_uid == os.getuid(),
                        "failed report staging root owner differs",
                    )
                    os.fchmod(staging_fd, 0o700)
                    with os.scandir(staging_fd) as iterator:
                        entries = [
                            (entry.name, entry.stat(follow_symlinks=False))
                            for entry in iterator
                        ]
                    for name, listed in entries:
                        require(
                            stat.S_ISREG(listed.st_mode)
                            and listed.st_uid == os.getuid()
                            and listed.st_nlink == 1,
                            f"failed report staging contains unsafe entry: {name}",
                        )
                        artifact_fd = os.open(
                            name,
                            os.O_RDONLY
                            | getattr(os, "O_NONBLOCK", 0)
                            | getattr(os, "O_NOFOLLOW", 0),
                            dir_fd=staging_fd,
                        )
                        try:
                            opened = os.fstat(artifact_fd)
                            require(
                                _file_identity(opened) == _file_identity(listed),
                                f"failed report staging entry raced: {name}",
                            )
                            os.fchmod(artifact_fd, 0o600)
                            os.unlink(name, dir_fd=staging_fd)
                        finally:
                            os.close(artifact_fd)
                    os.fsync(staging_fd)
                finally:
                    os.close(staging_fd)
                os.rmdir(staging)
                _fsync_directory(submission_root)
            except OSError as cleanup_exc:
                raise ReportError(
                    f"failed report staging cleanup could not complete: {cleanup_exc}"
                ) from cleanup_exc
            require(not _lexical_exists(staging), "failed report staging survived cleanup")
        raise
    validate_existing()
    return commit


def _validated_report_publication_prerequisite(
    submission_root: Path,
    submission_sha256: str,
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind publication to the positive-canary projection in the sealed contract."""

    return _validated_snapshot_production_authorization_prerequisite(
        submission_root,
        submission_sha256,
        provenance=provenance,
    )


def _validated_report_publication_receipt(
    submission_root: Path, submission_sha256: str
) -> dict[str, Any]:
    receipt_path = contained_regular(
        submission_root / "SUBMISSION_RECEIPT.json",
        submission_root,
        "report publication receipt",
    )
    receipt = read_json(receipt_path)
    require(
        set(receipt) == RECEIPT_KEYS
        and receipt.get("submission_sha256") == submission_sha256,
        "report publication receipt differs",
    )
    return receipt


def publish_report(
    submission_root: Path,
    submission_sha256: str,
    bundle: Mapping[str, Any],
    decision: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    repair_attempt: int | None = None,
    repair_authorization_sha256: str | None = None,
) -> dict[str, Any]:
    """Publish only on the report side of the report/cancellation linearization."""

    submission_root = nonsymlink_directory(submission_root, "submission root")
    with _ReportCancelLock(submission_root):
        _validated_report_publication_prerequisite(
            submission_root, submission_sha256, provenance
        )
        receipt = _validated_report_publication_receipt(
            submission_root, submission_sha256
        )
        if repair_attempt is None and repair_authorization_sha256 is None:
            require(
                os.environ.get("SLURM_JOB_ID") == receipt.get("report_job_id"),
                "report publication requires the exact committed report Slurm job",
            )
        else:
            require(
                type(repair_attempt) is int
                and repair_attempt == 1
                and sha256_string(repair_authorization_sha256),
                "report repair publication arguments differ",
            )
            authority = _validated_report_repair_authorization(
                submission_root,
                submission_sha256,
                receipt,
                attempt=repair_attempt,
                expected_raw_sha256=str(repair_authorization_sha256),
            )
            expected_publication_authority = _repair_publication_authority(
                authority, str(repair_authorization_sha256), repair_attempt
            )
            require(
                provenance.get("schema_version") == 2
                and exact_json_equal(
                    provenance.get("publication_authority"),
                    expected_publication_authority,
                ),
                "report repair provenance authority differs",
            )
        require(
            not _durable_cleanup_prefix_exists(submission_root),
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
    actions.add_argument(
        "--publish-repair",
        action="store_true",
        help="publish under one exact append-only terminal-report repair authority",
    )
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--submission-root", type=Path, required=True)
    parser.add_argument("--submission-sha256", required=True)
    parser.add_argument("--repair-attempt", type=int)
    parser.add_argument("--repair-authorization-sha256")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        require(
            (args.publish_repair and args.repair_attempt == 1 and sha256_string(args.repair_authorization_sha256))
            or (
                not args.publish_repair
                and args.repair_attempt is None
                and args.repair_authorization_sha256 is None
            ),
            "repair arguments require --publish-repair attempt 1 and an authorization SHA256",
        )
        if args.publish_repair:
            reject_environment()
            _wait_for_repair_release_evidence(
                args.submission_root,
                args.submission_sha256,
                attempt=args.repair_attempt,
                authorization_sha256=args.repair_authorization_sha256,
            )
        bundle, decision, provenance = assemble_report(
            args.snapshot_root,
            args.submission_root,
            args.submission_sha256,
            require_publish_job=args.publish,
            repair_attempt=(args.repair_attempt if args.publish_repair else None),
            repair_authorization_sha256=(
                args.repair_authorization_sha256 if args.publish_repair else None
            ),
        )
        if args.publish or args.publish_repair:
            result = publish_report(
                args.submission_root,
                args.submission_sha256,
                bundle,
                decision,
                provenance,
                repair_attempt=(args.repair_attempt if args.publish_repair else None),
                repair_authorization_sha256=(
                    args.repair_authorization_sha256
                    if args.publish_repair
                    else None
                ),
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
