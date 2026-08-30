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
import contextlib
import contextvars
import ctypes
import errno
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import pwd
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
    "schema_version": 2,
    "slurm_walltime_seconds": REPAIR_WALLTIME_SECONDS,
    "release_evidence_wait_seconds": REPAIR_RELEASE_EVIDENCE_WAIT_SECONDS,
    "minimum_assembly_budget_seconds": REPAIR_ASSEMBLY_BUDGET_SECONDS,
    "clock": "time.monotonic",
    "poll_interval_seconds": REPAIR_RELEASE_POLL_SECONDS,
    "restart_environment_policy": {
        "allowed_first_start_representations": ["absent", "0"],
        "nonzero_or_malformed_forbidden": True,
        "scheduler_requeue_required": 0,
        "scheduler_restarts_required": 0,
    },
}
EXPECTED_ATTEMPT1_JOB_ID = "33349323"
EXPECTED_ATTEMPT1_JOB_NAME = (
    "exp23-launch8-bbeaa71f8f37f22c-report-repair-0001"
)
EXPECTED_ATTEMPT1_COMMENT = (
    "treewm-exp23-report-repair:"
    "bbeaa71f8f37f22cbe74c16c68b733742e8a4366838812832180257d145f5418:0001"
)
EXPECTED_ATTEMPT1_CHAIN_SHA256 = {
    "REPORT_REPAIR_0001_ORIGINAL_FAILURE.json": "fb7119e73abb9ef745b03f85a83e3492364a08f9354e6986056e24537ef02a9d",
    "CALLING_REPORT_REPAIR_0001_SUBMIT.json": "17d7478dfbee2ef76e7c881121f436044153d871dec60a6ca29a13b0a38e058d",
    "REPORT_REPAIR_0001_SUBMITTED.json": "4b9f6998cd5e4c92fb4ee67afa8d9da8539e6108d9f397c740f959d3c2626a06",
    "REPORT_REPAIR_0001_AUTHORIZED.json": "f921ab2114908315cc5fe5f628cd6eda594fa2b826c6e4495a00c1152321f46a",
    "CALLING_REPORT_REPAIR_0001_RELEASE_0000.json": "55607fc914e817de939ba4353fe222a9a56235e63536446f59a885c98af9e86b",
    "REPORT_REPAIR_0001_RELEASE_RESULT_0000.json": "eb114ed55e70f9b0f36c8b17a22967af7bc90b859782905b0d3d917df68bc0ab",
    "REPORT_REPAIR_0001_RELEASED.json": "a59ff5dece424bd253576920287115c0cecfca6a54a19dc4b34b54a54d923b81",
}
EXPECTED_ATTEMPT1_SOURCE_AUTHORITY_SHA256 = "f23a9e83478d602724d86faa86ea6783bd48e4edcd8a6ddcefd94d0bdc9d5930"
EXPECTED_ATTEMPT1_SOURCE_COMMIT = "8fd45297ce09bb4a7e6d048ffc3f085d3841f254"
EXPECTED_ATTEMPT1_SOURCE_PROTOCOL = "d2b2f87bf15ea1a488208098155a2b3c230f564d5b6faf5bbba55b4ace548d95"
EXPECTED_ATTEMPT1_SOURCE_FILES_SHA256 = "c5d310c2abf34dc6a1598bb92ad9b1bc29f8765492157316e1728aea670480ce"
EXPECTED_ATTEMPT1_LOG_SHA256 = "5f6501b43be9014aa74d8c4e427466687735b82f0e3098db35d638f7e6b5ef01"
EXPECTED_ATTEMPT1_TERMINAL_SACCT_STDOUT_SHA256 = "8a02a2bb3de891045f67c6fdb12781693aabf1f783bca28454dbef2fa38a60d8"
EXPECTED_ATTEMPT1_ENV_STDOUT_SHA256 = "1a64556b0bd9c2ff3f0f2521825b453151eae9eebb9978c1f318303982c1e73a"
EXPECTED_ATTEMPT1_ENV_STDOUT_SIZE = 748
EXPECTED_ATTEMPT1_BATCH_STDOUT_SHA256 = "1138a359fa9162dd1bd3ab57f978401ca33964c215f5dd3baa5374decad0f4a5"
RENAME_NOREPLACE_FALLBACK_ERRNOS = frozenset(
    {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}
)
INSTALL_METHOD_PRIMARY = "renameat2_noreplace"
INSTALL_METHOD_FALLBACK_PREFIX = "locked_same_parent_rename_after_errno_"
DIRECT_FINAL_INSTALL_METHOD = "direct_final_name_o_excl"
SOURCE_ARCHIVE_INSTALL_METHOD = "direct_final_source_archive_o_excl"
PUBLICATION_ARCHIVE_INSTALL_METHOD = "direct_final_publication_archive_o_excl"
SOURCE_ARCHIVE_NAME = "REPORT_REPAIR_0002_SOURCE_ARCHIVE.bin"
SOURCE_ARCHIVE_KIND = "treewm_exp23_report_repair_source"
SOURCE_ARCHIVE_MARKER = b"__TREEWM_EXP23_SOURCE_ARCHIVE_V2_PAYLOAD__\n"
SOURCE_ARCHIVE_END = b"\n__TREEWM_EXP23_SOURCE_ARCHIVE_V2_END__\n"
PUBLICATION_ARCHIVE_KIND = "treewm_exp23_report_repair_publication"
PUBLICATION_ARCHIVE_MAGIC = b"TREEWM_EXP23_REPORT_REPAIR_PUBLICATION_V2\n"
PUBLICATION_ARCHIVE_PREFIX = "REPORT_REPAIR_0002_PUBLICATION."
PUBLICATION_ARCHIVE_SUFFIX = ".archive"
PUBLICATION_ARCHIVE_ENTRY_ORDER = (
    "report_bundle",
    "gate_decision",
    "provenance",
    "report_commit",
)
RENAME_NOREPLACE = 1
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
ATTEMPT1_PUBLIC_ENVIRONMENT_EVIDENCE = {
    "schema_version": 1,
    "raw_stdout_sha256": EXPECTED_ATTEMPT1_ENV_STDOUT_SHA256,
    "raw_stdout_size": EXPECTED_ATTEMPT1_ENV_STDOUT_SIZE,
    "allowlisted_projection": {
        "slurm_export_env": "NONE",
        "slurm_restart_count_present": False,
    },
}


def _valid_installation_method(value: object) -> bool:
    if value in {
        INSTALL_METHOD_PRIMARY,
        DIRECT_FINAL_INSTALL_METHOD,
        SOURCE_ARCHIVE_INSTALL_METHOD,
        PUBLICATION_ARCHIVE_INSTALL_METHOD,
    }:
        return True
    if not isinstance(value, str) or not value.startswith(
        INSTALL_METHOD_FALLBACK_PREFIX
    ):
        return False
    suffix = value.removeprefix(INSTALL_METHOD_FALLBACK_PREFIX)
    return suffix.isascii() and suffix.isdigit() and int(suffix) in {
        int(item) for item in RENAME_NOREPLACE_FALLBACK_ERRNOS
    }
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
REPORT_COMMIT_KEYS = frozenset(
    {
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
)
REPORT_PROVENANCE_V1_KEYS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "submission_sha256",
        "production_authorization_prerequisite",
        "production_authorization_prerequisite_sha256",
        "outcome_blind_phase",
        "event_artifacts",
        "terminal_artifacts",
        "report_bundle_sha256",
        "gate_sha256",
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
        "predecessor_failure_evidence",
        "predecessor_failure_evidence_sha256",
        "worker_receipt_map",
        "worker_receipt_map_sha256",
        "repair_source_root",
        "repair_source_commit",
        "repair_package_protocol_sha256",
        "repair_source_files",
        "repair_source_files_sha256",
        "repair_source_installation_method",
        "repair_source_archive",
        "repair_source_archive_sha256",
        "repair_source_archive_size",
        "repair_source_archive_format",
        "report_publication_installation_method",
        "submit_calling_sha256",
        "submitted_evidence",
        "submitted_evidence_sha256",
        "scheduler_authority_census",
        "scheduler_authority_census_sha256",
        "scheduler_job_control_observation",
        "scheduler_job_control_observation_sha256",
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
REPORT_REPAIR_PUBLICATION_AUTHORITY_KEYS = frozenset(
    {
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
        "predecessor_failure_evidence",
        "predecessor_failure_evidence_sha256",
        "attempt1_environment_evidence",
        "worker_receipt_map_sha256",
        "original_snapshot_root",
        "original_snapshot_inventory_sha256",
        "original_package_protocol_sha256",
        "repair_source_root",
        "repair_source_commit",
        "repair_package_protocol_sha256",
        "repair_source_files_sha256",
        "repair_source_installation_method",
        "repair_source_archive",
        "repair_source_archive_sha256",
        "repair_source_archive_size",
        "repair_source_archive_format",
        "report_publication_installation_method",
        "scheduler_job_control_observation_sha256",
        "worker_handoff_sha256",
        "expected_report_bundle_sha256",
        "expected_report_bundle_file_sha256",
        "expected_gate_sha256",
        "expected_gate_decision_file_sha256",
        "deterministic_reassembly_allowed",
        "scientific_input_change_allowed",
        "gate_change_allowed",
    }
)
REPORT_REPAIR_SOURCE_AUTHORITY_NAME = "SOURCE_AUTHORITY.json"
REPORT_REPAIR_SOURCE_AUTHORITY_V1_KEYS = frozenset(
    {
        "schema_version",
        "repair_source_commit",
        "repair_package_protocol_sha256",
        "repair_source_files",
        "repair_source_files_sha256",
    }
)
REPORT_REPAIR_SOURCE_AUTHORITY_V2_KEYS = frozenset(
    {*REPORT_REPAIR_SOURCE_AUTHORITY_V1_KEYS, "repair_source_installation_method"}
)
REPORT_REPAIR_SOURCE_ARCHIVE_EVIDENCE_KEYS = frozenset(
    {
        *REPORT_REPAIR_SOURCE_AUTHORITY_V2_KEYS,
        "repair_source_archive",
        "repair_source_archive_sha256",
        "repair_source_archive_size",
        "repair_source_archive_format",
    }
)
REPORT_REPAIR_ATTEMPT1_SOURCE_EVIDENCE_KEYS = frozenset(
    {
        "root",
        "authority",
        "authority_sha256",
        *REPORT_REPAIR_SOURCE_AUTHORITY_V1_KEYS,
    }
)
REPORT_REPAIR_JOB_ID_RE = re.compile(r"[1-9][0-9]*\Z")
REPORT_REPAIR_SBATCH_CLUSTER_RE = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z"
)
REPORT_REPAIR_SUBMIT_CALLING_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "attempt",
        "original_failure_evidence",
        "original_failure_evidence_sha256",
        "predecessor_failure_evidence",
        "predecessor_failure_evidence_sha256",
        "repair_source_root",
        "repair_source_commit",
        "repair_package_protocol_sha256",
        "repair_source_files",
        "repair_source_files_sha256",
        "repair_source_installation_method",
        "repair_source_archive",
        "repair_source_archive_sha256",
        "repair_source_archive_size",
        "repair_source_archive_format",
        "repair_source_archive_file_identity",
        "scheduler_source_archive_input",
        "scheduler_pre_submit_census",
        "scheduler_pre_submit_census_sha256",
        "command",
        "scheduler_environment",
        "transaction_lock",
        "report_cancel_lock",
        "called_at_utc",
    }
)
REPORT_REPAIR_SUBMITTED_KEYS = frozenset(
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
        "scheduler_pre_release_census",
        "scheduler_pre_release_census_sha256",
        "scheduler_pre_release_job_control_observation",
        "scheduler_pre_release_job_control_observation_sha256",
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
REPORT_REPAIR_ATTEMPT1_TERMINAL_KEYS = frozenset(
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


def _repair_journal_artifact_name_is_allowed(name: str) -> bool:
    """Exact finite JSON evidence grammar; archives are separate classes."""

    static = {
        *EXPECTED_ATTEMPT1_CHAIN_SHA256,
        "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json",
        "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json",
        "CALLING_REPORT_REPAIR_0002_SUBMIT.json",
        "REPORT_REPAIR_0002_SUBMITTED.json",
        "REPORT_REPAIR_0002_TERMINAL_SUBMIT_FAILURE.json",
        "REPORT_REPAIR_0002_AUTHORIZED.json",
        "REPORT_REPAIR_0002_RELEASED.json",
        "REPORT_REPAIR_0002_TERMINAL_RELEASE_DENIED.json",
        "REPORT_REPAIR_0002_TERMINAL_WORKER_FAILURE.json",
        "REPORT_REPAIR_0002_COMPLETED.json",
    }
    patterns = (
        re.compile(r"CALLING_REPORT_REPAIR_0002_RELEASE_[0-9]{4}\.json\Z"),
        re.compile(r"REPORT_REPAIR_0002_RELEASE_RESULT_[0-9]{4}\.json\Z"),
        re.compile(r"REPORT_REPAIR_0002_CANCEL_AUTHORIZED_[0-9]{4}\.json\Z"),
        re.compile(
            r"CALLING_REPORT_REPAIR_0002_SCANCEL_[0-9]{4}_[0-9]{4}\.json\Z"
        ),
        re.compile(
            r"REPORT_REPAIR_0002_SCANCEL_RESULT_[0-9]{4}_[0-9]{4}\.json\Z"
        ),
        re.compile(r"REPORT_REPAIR_0002_CANCEL_TERMINAL_[0-9]{4}\.json\Z"),
    )
    return name in static or any(pattern.fullmatch(name) for pattern in patterns)


def _repair_root_artifact_name_is_allowed(name: str) -> bool:
    if name == SOURCE_ARCHIVE_NAME or _repair_journal_artifact_name_is_allowed(name):
        return True
    return (
        re.fullmatch(
            re.escape(PUBLICATION_ARCHIVE_PREFIX)
            + r"[0-9a-f]{64}"
            + re.escape(PUBLICATION_ARCHIVE_SUFFIX),
            name,
        )
        is not None
    )


def _repair_root_name_is_reserved(name: str) -> bool:
    """Return every root spelling that could claim repair authority.

    This deliberately overmatches generations and malformed suffixes.  The
    caller must subsequently require ``_repair_root_artifact_name_is_allowed``;
    an unknown ``0003`` (or otherwise malformed) spelling is evidence, never
    an unrelated file that may be ignored.
    """

    return (
        name.startswith("REPORT_REPAIR_")
        or name.startswith("CALLING_REPORT_REPAIR_")
        or name.startswith(".REPORT_REPAIR_")
        or name.startswith(".CALLING_REPORT_REPAIR_")
    )
REPORT_REPAIR_PREDECESSOR_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "attempt",
        "predecessor_attempt",
        "predecessor_report_job_id",
        "predecessor_job_name",
        "predecessor_scheduler_comment",
        "predecessor_chain",
        "predecessor_chain_sha256",
        "predecessor_source",
        "predecessor_source_sha256",
        "terminal_worker_failure_evidence",
        "terminal_worker_failure_evidence_sha256",
        "retained_environment_evidence",
        "retained_environment_evidence_sha256",
        "retained_batch_script_evidence",
        "retained_batch_script_evidence_sha256",
        "failure_log",
        "failure_log_sha256",
        "worker_receipt_map",
        "worker_receipt_map_sha256",
        "publication_state",
        "restart_failure_classification",
        "transaction_lock",
        "report_cancel_lock",
        "publication_allowed",
        "retry_predecessor_allowed",
        "successor_attempt",
        "successor_scheduler_submission_allowed",
        "sealed_at_utc",
    }
)
REPORT_REPAIR_COMPLETED_KEYS = frozenset(
    {
        "schema_version",
        "status",
        "campaign_id",
        "submission_sha256",
        "attempt",
        "repair_report_job_id",
        "predecessor_failure_evidence",
        "predecessor_failure_evidence_sha256",
        "authorization",
        "authorization_sha256",
        "release",
        "release_sha256",
        "report_commit",
        "report_commit_sha256",
        "report_commit_value",
        "report_commit_value_sha256",
        "publication_archive",
        "publication_archive_sha256",
        "publication_archive_size",
        "publication_archive_header_sha256",
        "publication_archive_file_identity",
        "publication_authority_sha256",
        "repair_source_installation_method",
        "report_publication_installation_method",
        "expected_reassembly",
        "publication_complete",
        "retry_allowed",
        "successor_attempt_allowed",
        "completed_at_utc",
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


def _durable_repair_stop_prefix_exists(
    submission_root: Path, *, repair_attempt: int | None = None
) -> bool:
    journal = submission_root / "journal"
    try:
        journal_names = {
            entry.name
            for entry in os.scandir(journal)
            if entry.name.startswith("REPORT_REPAIR_")
            or entry.name.startswith("CALLING_REPORT_REPAIR_")
        }
        reserved_root_names = {
            entry.name
            for entry in os.scandir(submission_root)
            if _repair_root_name_is_reserved(entry.name)
        }
        if not all(
            _repair_root_artifact_name_is_allowed(name)
            for name in reserved_root_names
        ):
            return True
        root_names = {
            name
            for name in reserved_root_names
            if _repair_journal_artifact_name_is_allowed(name)
        }
    except FileNotFoundError:
        return False
    except OSError as exc:
        raise ReportError(f"cannot inventory durable cleanup prefix: {exc}") from exc
    if any("_0002_" in name for name in journal_names):
        return True
    names = journal_names | root_names
    if repair_attempt is None:
        return bool(names)
    require(repair_attempt == 2, "repair stop-prefix attempt differs")
    allowed_positive = {
        *EXPECTED_ATTEMPT1_CHAIN_SHA256,
        "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json",
        "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json",
        "CALLING_REPORT_REPAIR_0002_SUBMIT.json",
        "REPORT_REPAIR_0002_SUBMITTED.json",
        "REPORT_REPAIR_0002_AUTHORIZED.json",
        "REPORT_REPAIR_0002_RELEASED.json",
        "REPORT_REPAIR_0002_COMPLETED.json",
    }
    allowed_positive_patterns = (
        re.compile(r"CALLING_REPORT_REPAIR_0002_RELEASE_[0-9]{4}\.json\Z"),
        re.compile(r"REPORT_REPAIR_0002_RELEASE_RESULT_[0-9]{4}\.json\Z"),
    )
    if any(
        name not in allowed_positive
        and not any(pattern.fullmatch(name) for pattern in allowed_positive_patterns)
        for name in names
    ):
        return True
    if (
        "REPORT_REPAIR_0002_COMPLETED.json" in names
        and len(
            {
                name
                for name in os.listdir(submission_root)
                if name.startswith(PUBLICATION_ARCHIVE_PREFIX)
                and name.endswith(PUBLICATION_ARCHIVE_SUFFIX)
            }
        )
        != 1
    ):
        return True
    release_result_pattern = allowed_positive_patterns[1]
    for name in sorted(names):
        if not release_result_pattern.fullmatch(name):
            continue
        try:
            value = read_json(
                submission_root / name if "_0002_" in name else journal / name
            )
        except (OSError, ReportError, ValueError, TypeError):
            return True
        if value.get("mode") == "lost_response_reconciled_ambiguous_identity":
            return True
    return False


def _durable_cleanup_prefix_exists(
    submission_root: Path, *, repair_attempt: int | None = None
) -> bool:
    return _durable_original_cleanup_prefix_exists(
        submission_root
    ) or _durable_repair_stop_prefix_exists(
        submission_root, repair_attempt=repair_attempt
    )


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


def _repair_first_start_restart_count_is_valid(value: object) -> bool:
    """Accept Slurm's two first-start encodings and no restarted value."""

    return value is None or (type(value) is str and value == "0")


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


DIRECT_FINAL_FILE_IDENTITY_KEYS = frozenset(
    {"device", "inode", "uid", "mode", "nlink", "size"}
)


def _direct_final_file_identity(info: os.stat_result) -> dict[str, int]:
    require(stat.S_ISREG(info.st_mode), "direct-final identity is not regular")
    return {
        "device": info.st_dev,
        "inode": info.st_ino,
        "uid": info.st_uid,
        "mode": stat.S_IMODE(info.st_mode),
        "nlink": info.st_nlink,
        "size": info.st_size,
    }


def _validated_direct_final_file_identity(
    value: object, *, size: int, label: str
) -> dict[str, int]:
    require(
        isinstance(value, Mapping)
        and set(value) == DIRECT_FINAL_FILE_IDENTITY_KEYS
        and all(type(value[key]) is int for key in DIRECT_FINAL_FILE_IDENTITY_KEYS)
        and value["device"] >= 0
        and value["inode"] > 0
        and value["uid"] == os.getuid()
        and value["mode"] == 0o444
        and value["nlink"] == 1
        and value["size"] == size,
        f"{label} differs",
    )
    return dict(value)


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
    active_binding_var = globals().get("_ACTIVE_REPAIR_PUBLICATION_BINDING")
    if active_binding_var is not None:
        binding = active_binding_var.get()
        if binding is not None:
            retained = binding.open_retained_regular(root / relative)
            if retained is None:
                retained = binding.open_retained_scientific_regular(root, relative)
            if retained is not None:
                return retained
            require(
                not binding.manages_regular_path(root / relative),
                f"{label} is absent from the retained authority graph",
            )
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
    active_binding_var = globals().get("_ACTIVE_REPAIR_PUBLICATION_BINDING")
    if active_binding_var is not None:
        binding = active_binding_var.get()
        if binding is not None:
            retained = binding.retained_regular(path)
            if retained is not None:
                payload, info = retained
                return (
                    payload if capture else None,
                    hashlib.sha256(payload).hexdigest(),
                    info,
                )
            require(
                not binding.manages_regular_path(path),
                f"{label} is absent from the retained authority graph",
            )
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
    active_binding_var = globals().get("_ACTIVE_REPAIR_PUBLICATION_BINDING")
    if active_binding_var is not None:
        binding = active_binding_var.get()
        if binding is not None:
            retained = binding.retained_scientific_tree_rows(root)
            if retained is not None:
                require(
                    hash_files and allow_wandb_symlink_leaves,
                    f"{label} retained scientific census arguments differ",
                )
                return retained
            retained_authority = binding.retained_authority_tree_rows(root)
            if retained_authority is not None:
                require(
                    hash_files and not allow_wandb_symlink_leaves,
                    f"{label} retained authority census arguments differ",
                )
                return retained_authority
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

    def binding(self) -> dict[str, Any]:
        require(self.descriptor is not None, "report/cancel lock is not held")
        opened = os.fstat(self.descriptor)
        named = self.path.lstat()
        require(
            stat.S_ISREG(opened.st_mode)
            and _file_identity(opened) == _file_identity(named)
            and opened.st_uid == os.getuid()
            and opened.st_nlink == 1
            and stat.S_IMODE(opened.st_mode) == 0o600,
            "report/cancel lock binding changed",
        )
        return {
            "path": str(self.path),
            "device": opened.st_dev,
            "inode": opened.st_ino,
            "uid": opened.st_uid,
            "mode": stat.S_IMODE(opened.st_mode),
        }


def _repair_transaction_lock_path(submission_root: Path) -> Path:
    absolute = submission_root.absolute()
    require(
        absolute.name == "submission" and absolute.parent.name == "state",
        "report repair publication requires the sealed run layout",
    )
    token = hashlib.sha256(str(absolute).encode("utf-8")).hexdigest()[:16]
    return absolute.parents[2] / f".exp23-{token}.transaction.lock"


class _ExistingPublicationLock:
    """Open and rebind one already-created production lock inode."""

    def __init__(self, path: Path, label: str) -> None:
        self.path = path
        self.label = label
        self.descriptor: int | None = None

    def __enter__(self) -> "_ExistingPublicationLock":
        nonsymlink_directory(self.path.parent, f"{self.label} parent")
        try:
            descriptor = os.open(
                self.path,
                os.O_RDWR
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
            )
        except OSError as exc:
            raise ReportError(f"cannot open existing {self.label}: {exc}") from exc
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
            opened_after = os.fstat(descriptor)
            named_after = self.path.lstat()
            require(
                _file_identity(opened_after) == _file_identity(named_after)
                and opened_after.st_uid == os.getuid()
                and opened_after.st_nlink == 1
                and stat.S_IMODE(opened_after.st_mode) == 0o600,
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
            and opened.st_uid == os.getuid()
            and opened.st_nlink == 1
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


class _RepairPublicationLocks:
    """Hold transaction then report/cancel locks for repaired publication."""

    def __init__(self, submission_root: Path) -> None:
        self.transaction = _ExistingPublicationLock(
            _repair_transaction_lock_path(submission_root),
            "production transaction lock",
        )
        self.report_cancel = _ExistingPublicationLock(
            submission_root / ".REPORT_CANCEL.lock", "report/cancel lock"
        )

    def __enter__(self) -> "_RepairPublicationLocks":
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
            and repair_attempt == 2
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
    name = f"REPORT_REPAIR_{attempt:04d}_AUTHORIZED.json"
    return submission_root / name if attempt == 2 else submission_root / "journal" / name


def _repair_release_path(submission_root: Path, attempt: int) -> Path:
    name = f"REPORT_REPAIR_{attempt:04d}_RELEASED.json"
    return submission_root / name if attempt == 2 else submission_root / "journal" / name


def _wait_for_repair_release_evidence(
    submission_root: Path,
    submission_sha256: str,
    *,
    attempt: int,
    authorization_sha256: str,
    phase_binding: "_RepairPublicationPhaseBinding | None" = None,
    monotonic: Any = time.monotonic,
    sleep: Any = time.sleep,
) -> None:
    require(type(attempt) is int and attempt == 2, "report repair wait attempt differs")
    require(sha256_string(authorization_sha256), "report repair wait authorization hash differs")
    require(
        REPAIR_RELEASE_EVIDENCE_WAIT_SECONDS + REPAIR_ASSEMBLY_BUDGET_SECONDS
        == REPAIR_WALLTIME_SECONDS,
        "report repair wait/assembly walltime budget differs",
    )
    active = _ACTIVE_REPAIR_PUBLICATION_BINDING.get()
    binding = phase_binding if phase_binding is not None else active
    require(
        binding is not None
        and binding.submission_root == submission_root,
        "report repair wait retained binding is absent",
    )
    authorization_path = _repair_authorization_path(submission_root, attempt)
    retained_authorization = binding.retained_regular(authorization_path)
    require(
        retained_authorization is not None,
        "report repair wait authorization is unbound",
    )
    payload, info = retained_authorization
    digest = hashlib.sha256(payload).hexdigest()
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
        and REPORT_REPAIR_JOB_ID_RE.fullmatch(job_id) is not None
        and exact_json_equal(
            authorization.get("worker_handoff"), REPAIR_WORKER_HANDOFF
        )
        and authorization.get("publication_allowed") is True
        and authorization.get("scheduler_submission_allowed") is False,
        "report repair wait authorization fields differ",
    )
    require(
        os.environ.get("SLURM_JOB_ID") == job_id
        and _repair_first_start_restart_count_is_valid(
            os.environ.get("SLURM_RESTART_COUNT")
        )
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
            binding.release_wait_is_open(),
            "report repair cleanup/terminal authority precedes release",
        )
        if binding.admit_release_evidence(release_path.name):
            retained_release = binding.retained_regular(release_path)
            require(
                retained_release is not None,
                "report repair release evidence is unbound",
            )
            _release_payload, release_info = retained_release
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


def _validated_repair_sbatch_stdout(stdout: bytes) -> dict[str, Any]:
    """Parse the exact one-line contract emitted by ``sbatch --parsable``."""

    try:
        text = stdout.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ReportError(
            f"report repair sbatch stdout is not ASCII: {exc}"
        ) from exc
    require(
        text.endswith("\n")
        and text.count("\n") == 1
        and "\r" not in text
        and text[:-1]
        and text[:-1] == text[:-1].strip(),
        "report repair sbatch stdout differs",
    )
    fields = text[:-1].split(";")
    require(
        len(fields) in {1, 2},
        "report repair sbatch stdout field count differs",
    )
    job_id = fields[0]
    cluster = fields[1] if len(fields) == 2 else None
    require(
        REPORT_REPAIR_JOB_ID_RE.fullmatch(job_id) is not None,
        "report repair sbatch job ID differs",
    )
    require(
        cluster is None
        or REPORT_REPAIR_SBATCH_CLUSTER_RE.fullmatch(cluster) is not None,
        "report repair sbatch cluster differs",
    )
    return {"schema_version": 1, "job_id": job_id, "cluster": cluster}


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
    expected_environment: Mapping[str, str],
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
    require(
        isinstance(expected_environment, Mapping)
        and set(expected_environment)
        == {"PATH", "LANG", "LC_ALL", "TZ", "SLURM_CONF", "USER", "LOGNAME"}
        and all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in expected_environment.items()
        )
        and bool(expected_environment["USER"]),
        f"{label} expected scheduler environment differs",
    )
    expected_argv = [
        "/usr/local/bin/squeue",
        "--noheader",
        f"--user={expected_environment['USER']}",
        "--format=%A|%j|%u|%T|%k|%r",
    ]
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
            round_value["raw"],
            expected_argv=expected_argv,
            label=f"{label} round {index}",
        )
        require(
            exact_json_equal(raw["environment"], expected_environment)
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
        expected_owner = expected_environment["USER"]
        parsed_relevant: list[dict[str, str]] = []
        for line in stdout_text.splitlines():
            if not line:
                continue
            fields = line.split("|", 5)
            require(len(fields) == 6, f"{label} census raw row differs")
            job_id, job_name, owner, state, comment, reason = fields
            require(
                REPORT_REPAIR_JOB_ID_RE.fullmatch(job_id) is not None,
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
                and REPORT_REPAIR_JOB_ID_RE.fullmatch(row["job_id"])
                is not None
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


def _repair_expected_scheduler_environment(
    contract: Mapping[str, Any], username: object
) -> dict[str, str]:
    control_plane = contract.get("scheduler_control_plane_contract")
    require(
        isinstance(control_plane, Mapping)
        and isinstance(control_plane.get("slurm_conf"), str)
        and bool(control_plane["slurm_conf"])
        and isinstance(username, str)
        and bool(username),
        "attempt1 scheduler environment authority differs",
    )
    return {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "TZ": "UTC",
        "USER": username,
        "LOGNAME": username,
        "SLURM_CONF": str(control_plane["slurm_conf"]),
    }


def _repair_current_scheduler_environment(
    contract: Mapping[str, Any],
) -> dict[str, str]:
    return _repair_expected_scheduler_environment(
        contract, pwd.getpwuid(os.getuid()).pw_name
    )


def _validated_attempt1_terminal_scheduler_observation(
    value: object, contract: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, str]]:
    require(
        isinstance(value, Mapping)
        and set(value)
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
        and value.get("canonical_sha256") == stable_hash(value["canonical"])
        and isinstance(value.get("parsed_row"), Mapping),
        "attempt1 terminal scheduler observation differs",
    )
    parsed = value["parsed_row"]
    require(
        set(parsed) == set(REPAIR_SACCT_FIELDS)
        and parsed.get("JobIDRaw") == EXPECTED_ATTEMPT1_JOB_ID
        and parsed.get("JobName") == EXPECTED_ATTEMPT1_JOB_NAME
        and parsed.get("User")
        == _repair_current_scheduler_environment(contract)["USER"]
        and parsed.get("State") == "FAILED"
        and parsed.get("ExitCode") == "2:0"
        and parsed.get("ElapsedRaw") == "5"
        and parsed.get("AllocNodes") == "1"
        and parsed.get("NodeList") == "cpu-00049"
        and parsed.get("Submit") == "2026-08-29T13:22:38"
        and parsed.get("Eligible") == "2026-08-29T13:22:41"
        and parsed.get("Start") == "2026-08-29T13:22:43"
        and parsed.get("End") == "2026-08-29T13:22:48"
        and parsed["Submit"] <= parsed["Eligible"] <= parsed["Start"] < parsed["End"]
        and parsed.get("Comment") == EXPECTED_ATTEMPT1_COMMENT
        and parsed.get("Reason") == "None",
        "attempt1 terminal scheduler row differs",
    )
    canonical = {
        "schema_version": 1,
        "fields": list(REPAIR_SACCT_FIELDS),
        "rows": [[parsed[field] for field in REPAIR_SACCT_FIELDS]],
    }
    expected_environment = _repair_current_scheduler_environment(contract)
    raw = _validated_repair_command_evidence(
        value["raw"],
        expected_argv=[
            "/usr/local/bin/sacct",
            "-X",
            "-n",
            "-j",
            EXPECTED_ATTEMPT1_JOB_ID,
            "-o",
            ",".join(REPAIR_SACCT_FIELDS),
            "-P",
        ],
        label="attempt1 terminal accounting",
    )
    stdout = _repair_stream_bytes(
        raw["stdout"], "attempt1 terminal accounting stdout"
    )
    require(
        exact_json_equal(value["canonical"], canonical)
        and exact_json_equal(raw["environment"], expected_environment)
        and raw["returncode"] == 0
        and _repair_stream_bytes(
            raw["stderr"], "attempt1 terminal accounting stderr"
        )
        == b""
        and hashlib.sha256(stdout).hexdigest()
        == EXPECTED_ATTEMPT1_TERMINAL_SACCT_STDOUT_SHA256
        and stdout
        == ("|".join(parsed[field] for field in REPAIR_SACCT_FIELDS) + "\n").encode(
            "utf-8"
        ),
        "attempt1 terminal scheduler raw/canonical evidence differs",
    )
    return dict(value), expected_environment


def _validated_attempt1_worker_failure_terminal(
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    *,
    relative_name: str,
    expected_raw_sha256: str,
    phase_binding: "_RepairPublicationPhaseBinding",
) -> tuple[dict[str, Any], dict[str, str]]:
    terminal_path = submission_root / relative_name
    terminal, terminal_sha256, terminal_info = _retained_repair_json(
        phase_binding,
        terminal_path,
        "attempt1 worker-failure terminal",
    )
    expected_release_attempts = [
        {
            "release_attempt": 0,
            "calling": "CALLING_REPORT_REPAIR_0001_RELEASE_0000.json",
            "calling_sha256": EXPECTED_ATTEMPT1_CHAIN_SHA256[
                "CALLING_REPORT_REPAIR_0001_RELEASE_0000.json"
            ],
            "result": "REPORT_REPAIR_0001_RELEASE_RESULT_0000.json",
            "result_sha256": EXPECTED_ATTEMPT1_CHAIN_SHA256[
                "REPORT_REPAIR_0001_RELEASE_RESULT_0000.json"
            ],
        }
    ]
    require(
        stat.S_IMODE(terminal_info.st_mode) == 0o444
        and terminal_info.st_uid == os.getuid()
        and terminal_info.st_nlink == 1
        and terminal_sha256 == expected_raw_sha256
        and set(terminal) == REPORT_REPAIR_ATTEMPT1_TERMINAL_KEYS
        and type(terminal.get("schema_version")) is int
        and terminal.get("schema_version") == 1
        and terminal.get("status") == "report_repair_terminal_worker_failure"
        and terminal.get("campaign_id") == CAMPAIGN_ID
        and terminal.get("submission_sha256") == submission_sha256
        and type(terminal.get("attempt")) is int
        and terminal.get("attempt") == 1
        and terminal.get("repair_report_job_id") == EXPECTED_ATTEMPT1_JOB_ID
        and terminal.get("authorization_sha256")
        == EXPECTED_ATTEMPT1_CHAIN_SHA256["REPORT_REPAIR_0001_AUTHORIZED.json"]
        and terminal.get("reason")
        == "repair_worker_terminal_after_release_evidence"
        and exact_json_equal(
            terminal.get("release_attempts"), expected_release_attempts
        )
        and terminal.get("release_attempts_sha256")
        == stable_hash(expected_release_attempts)
        and terminal.get("released_evidence")
        == "journal/REPORT_REPAIR_0001_RELEASED.json"
        and terminal.get("released_evidence_sha256")
        == EXPECTED_ATTEMPT1_CHAIN_SHA256["REPORT_REPAIR_0001_RELEASED.json"]
        and isinstance(terminal.get("post_release_census"), Mapping)
        and terminal.get("post_release_census_sha256")
        == stable_hash(terminal["post_release_census"])
        and isinstance(terminal.get("terminal_scheduler_observation"), Mapping)
        and terminal.get("terminal_scheduler_observation_sha256")
        == stable_hash(terminal["terminal_scheduler_observation"])
        and terminal.get("publication_allowed") is False
        and terminal.get("retry_allowed") is False
        and isinstance(terminal.get("sealed_at_utc"), str)
        and bool(terminal["sealed_at_utc"]),
        "attempt1 worker-failure terminal fields differ",
    )
    _terminal_observation, attempt1_environment = (
        _validated_attempt1_terminal_scheduler_observation(
            terminal["terminal_scheduler_observation"], contract
        )
    )
    terminal_census = _validated_repair_census(
        terminal["post_release_census"],
        submission_sha256=submission_sha256,
        expected_environment=attempt1_environment,
        label="attempt1 worker-failure terminal",
    )
    require(
        terminal_census["settled_rows"] == [],
        "attempt1 worker-failure terminal census is not empty",
    )
    require(
        all(
            exact_json_equal(
                round_value["raw"]["environment"], attempt1_environment
            )
            for round_value in terminal_census["rounds"]
        ),
        "attempt1 worker-failure terminal census environment differs",
    )
    return terminal, attempt1_environment


def _validated_attempt2_predecessor(
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    receipt_map: Mapping[str, Any],
    *,
    expected_raw_sha256: str,
    phase_binding: "_RepairPublicationPhaseBinding",
) -> dict[str, Any]:
    path = submission_root / "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
    value, predecessor_sha256, info = _retained_repair_json(
        phase_binding, path, "attempt2 predecessor failure"
    )
    require(
        stat.S_IMODE(info.st_mode) == 0o444
        and info.st_uid == os.getuid()
        and info.st_nlink == 1
        and predecessor_sha256 == expected_raw_sha256,
        "attempt2 predecessor identity/hash differs",
    )
    require(
        set(value) == REPORT_REPAIR_PREDECESSOR_KEYS
        and value.get("schema_version") == 1
        and value.get("status")
        == "attempt1_terminal_failure_authorized_for_attempt2"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and value.get("attempt") == 2
        and value.get("predecessor_attempt") == 1
        and value.get("predecessor_report_job_id") == EXPECTED_ATTEMPT1_JOB_ID
        and value.get("predecessor_job_name") == EXPECTED_ATTEMPT1_JOB_NAME
        and value.get("predecessor_scheduler_comment") == EXPECTED_ATTEMPT1_COMMENT
        and exact_json_equal(
            value.get("predecessor_chain"), EXPECTED_ATTEMPT1_CHAIN_SHA256
        )
        and value.get("predecessor_chain_sha256")
        == stable_hash(EXPECTED_ATTEMPT1_CHAIN_SHA256)
        and isinstance(value.get("predecessor_source"), Mapping)
        and value.get("predecessor_source_sha256")
        == stable_hash(value["predecessor_source"])
        and value.get("terminal_worker_failure_evidence")
        == "journal/REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json"
        and sha256_string(value.get("terminal_worker_failure_evidence_sha256"))
        and isinstance(value.get("retained_environment_evidence"), Mapping)
        and value.get("retained_environment_evidence_sha256")
        == stable_hash(value["retained_environment_evidence"])
        and isinstance(value.get("retained_batch_script_evidence"), Mapping)
        and value.get("retained_batch_script_evidence_sha256")
        == stable_hash(value["retained_batch_script_evidence"])
        and isinstance(value.get("failure_log"), Mapping)
        and value.get("failure_log_sha256") == stable_hash(value["failure_log"])
        and exact_json_equal(value.get("worker_receipt_map"), receipt_map)
        and value.get("worker_receipt_map_sha256") == stable_hash(receipt_map)
        and exact_json_equal(
            value.get("publication_state"),
            {
                "report_absent": True,
                "staging_entries": [],
                "cleanup_prefixes": [],
                "journal_directory": str(submission_root / "journal"),
            },
        )
        and exact_json_equal(
            value.get("restart_failure_classification"),
            {
                "schema_version": 1,
                "failed_guard": "restart_count_presence_and_zero_required",
                "retained_environment_variable_present": False,
                "runtime_absence_directly_recorded": False,
                "runtime_absence_inferred_from_exact_guard_and_log": True,
                "durable_attempt1_scontrol_observation_available": False,
                "successor_first_start_allowed_representations": ["absent", "0"],
                "successor_requires_fresh_scheduler_requeue": 0,
                "successor_requires_fresh_scheduler_restarts": 0,
            },
        )
        and isinstance(value.get("transaction_lock"), Mapping)
        and isinstance(value.get("report_cancel_lock"), Mapping)
        and value.get("publication_allowed") is False
        and value.get("retry_predecessor_allowed") is False
        and value.get("successor_attempt") == 2
        and value.get("successor_scheduler_submission_allowed") is True
        and isinstance(value.get("sealed_at_utc"), str)
        and bool(value["sealed_at_utc"]),
        "attempt2 predecessor fields differ",
    )
    for name, expected_sha in EXPECTED_ATTEMPT1_CHAIN_SHA256.items():
        predecessor_path = submission_root / "journal" / name
        _predecessor_payload, predecessor_digest, predecessor_info = (
            _retained_repair_regular(
                phase_binding,
                predecessor_path,
                f"attempt1 predecessor {name}",
            )
        )
        require(
            stat.S_IMODE(predecessor_info.st_mode) == 0o444
            and predecessor_info.st_uid == os.getuid()
            and predecessor_info.st_nlink == 1
            and predecessor_digest == expected_sha,
            f"attempt1 predecessor differs: {name}",
        )
    source = value["predecessor_source"]
    source_root = submission_root / "report-repair/attempt-0001/source"
    source_root_info, source_names = phase_binding.retained_directory(
        source_root
    )
    require(
        set(source) == REPORT_REPAIR_ATTEMPT1_SOURCE_EVIDENCE_KEYS
        and source_root_info.st_uid == os.getuid()
        and source_root_info.st_nlink == 2
        and stat.S_IMODE(source_root_info.st_mode) == 0o555
        and set(source_names)
        == {
            "report.py",
            "report_repair.py",
            "report_repair.slurm",
            "protocol.sha256",
            REPORT_REPAIR_SOURCE_AUTHORITY_NAME,
        }
        and source.get("root") == str(source_root)
        and source.get("authority")
        == "report-repair/attempt-0001/source/SOURCE_AUTHORITY.json"
        and source.get("authority_sha256")
        == EXPECTED_ATTEMPT1_SOURCE_AUTHORITY_SHA256
        and source.get("schema_version") == 1
        and source.get("repair_source_commit") == EXPECTED_ATTEMPT1_SOURCE_COMMIT
        and source.get("repair_package_protocol_sha256")
        == EXPECTED_ATTEMPT1_SOURCE_PROTOCOL
        and source.get("repair_source_files_sha256")
        == EXPECTED_ATTEMPT1_SOURCE_FILES_SHA256
        and isinstance(source.get("repair_source_files"), Mapping)
        and set(source["repair_source_files"])
        == {"report.py", "report_repair.py", "report_repair.slurm", "protocol.sha256"},
        "attempt1 predecessor source fields differ",
    )
    authority_path = source_root / REPORT_REPAIR_SOURCE_AUTHORITY_NAME
    authority, authority_sha256, authority_info = _retained_repair_json(
        phase_binding,
        authority_path,
        "attempt1 predecessor source authority",
    )
    require(
        stat.S_IMODE(authority_info.st_mode) == 0o444
        and authority_info.st_uid == os.getuid()
        and authority_info.st_nlink == 1
        and authority_sha256 == EXPECTED_ATTEMPT1_SOURCE_AUTHORITY_SHA256
        and set(authority) == REPORT_REPAIR_SOURCE_AUTHORITY_V1_KEYS
        and exact_json_equal(
            authority,
            {
                "schema_version": 1,
                "repair_source_commit": EXPECTED_ATTEMPT1_SOURCE_COMMIT,
                "repair_package_protocol_sha256": EXPECTED_ATTEMPT1_SOURCE_PROTOCOL,
                "repair_source_files": source["repair_source_files"],
                "repair_source_files_sha256": EXPECTED_ATTEMPT1_SOURCE_FILES_SHA256,
            },
        ),
        "attempt1 predecessor source authority differs",
    )
    for name, expected in source["repair_source_files"].items():
        source_path = source_root / name
        _source_payload, source_sha256, source_info = _retained_repair_regular(
            phase_binding,
            source_path,
            f"attempt1 predecessor source {name}",
        )
        require(
            isinstance(expected, Mapping)
            and set(expected) == {"mode", "size", "sha256"}
            and expected.get("mode") == 0o444
            and stat.S_IMODE(source_info.st_mode) == 0o444
            and source_info.st_uid == os.getuid()
            and source_info.st_nlink == 1
            and source_info.st_size == expected.get("size")
            and source_sha256 == expected.get("sha256"),
            f"attempt1 predecessor source bytes differ: {name}",
        )
    terminal, attempt1_environment = _validated_attempt1_worker_failure_terminal(
        submission_root,
        submission_sha256,
        contract,
        relative_name=str(value["terminal_worker_failure_evidence"]),
        expected_raw_sha256=str(value["terminal_worker_failure_evidence_sha256"]),
        phase_binding=phase_binding,
    )
    for key, option, expected_sha in (
        ("retained_environment_evidence", "--env-vars", EXPECTED_ATTEMPT1_ENV_STDOUT_SHA256),
        ("retained_batch_script_evidence", "--batch-script", EXPECTED_ATTEMPT1_BATCH_STDOUT_SHA256),
    ):
        evidence = _validated_repair_command_evidence(
            value[key],
            expected_argv=["/usr/local/bin/sacct", "-j", EXPECTED_ATTEMPT1_JOB_ID, option],
            label=f"attempt1 retained {option}",
        )
        stdout = _repair_stream_bytes(evidence["stdout"], f"attempt1 retained {option} stdout")
        require(
            evidence["returncode"] == 0
            and exact_json_equal(evidence["environment"], attempt1_environment)
            and _repair_stream_bytes(
                evidence["stderr"], f"attempt1 retained {option} stderr"
            )
            == b""
            and hashlib.sha256(stdout).hexdigest() == expected_sha,
            f"attempt1 retained {option} bytes differ",
        )
        if option == "--env-vars":
            require(
                b"SLURM_EXPORT_ENV=NONE" in stdout
                and b"SLURM_RESTART_COUNT" not in stdout,
                "attempt1 retained restart environment differs",
            )
        else:
            require(
                b'[[ -n "${SLURM_RESTART_COUNT+x}" && "$SLURM_RESTART_COUNT" == "0" ]]'
                in stdout,
                "attempt1 retained batch restart guard differs",
            )
    log = value["failure_log"]
    require(
        set(log) == {"path", "mode", "size", "uid", "nlink", "sha256", "encoding", "data"}
        and log.get("path") == f"logs/report-repair-0001-{EXPECTED_ATTEMPT1_JOB_ID}.out"
        and log.get("mode") == 0o600
        and log.get("size") == 38
        and log.get("uid") == os.getuid()
        and log.get("nlink") == 1
        and log.get("sha256") == EXPECTED_ATTEMPT1_LOG_SHA256
        and _repair_stream_bytes(
            {key: log[key] for key in ("encoding", "size", "sha256", "data")},
            "attempt1 failure log",
        )
        == b"repair publication cannot be requeued\n",
        "attempt1 failure log evidence differs",
    )
    log_path = submission_root / log["path"]
    _log_payload, log_sha256, log_info = _retained_repair_regular(
        phase_binding,
        log_path,
        "attempt1 failure log",
        expected_mode=0o600,
    )
    require(
        stat.S_IMODE(log_info.st_mode) == 0o600
        and log_info.st_uid == os.getuid()
        and log_info.st_nlink == 1
        and log_info.st_size == 38
        and log_sha256 == EXPECTED_ATTEMPT1_LOG_SHA256,
        "attempt1 failure log live identity differs",
    )
    return dict(value)


def _validated_repair_job_control(
    value: object,
    *,
    submission_root: Path,
    submission_sha256: str,
    repair_authorization: Mapping[str, Any],
    authority_environment: Mapping[str, Any],
    contract: Mapping[str, Any],
    scheduler_source_argument: str,
) -> dict[str, Any]:
    require(
        isinstance(value, Mapping)
        and set(value)
        == {
            "schema_version",
            "captured_at_utc",
            "scheduler_control_plane",
            "raw",
            "projection",
            "projection_sha256",
        }
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and isinstance(value.get("captured_at_utc"), str)
        and bool(value["captured_at_utc"])
        and exact_json_equal(
            value.get("scheduler_control_plane"),
            contract.get("scheduler_control_plane_contract"),
        )
        and isinstance(value.get("projection"), Mapping)
        and value.get("projection_sha256") == stable_hash(value["projection"]),
        "report repair job-control observation differs",
    )
    job_id = repair_authorization["repair_report_job_id"]
    raw = _validated_repair_command_evidence(
        value["raw"],
        expected_argv=["/usr/local/bin/scontrol", "show", "job", "-dd", job_id],
        label="report repair held-job scontrol",
    )
    require(
        raw["returncode"] == 0
        and _repair_stream_bytes(raw["stderr"], "report repair scontrol stderr")
        == b""
        and exact_json_equal(raw["environment"], authority_environment),
        "report repair held-job scontrol command differs",
    )
    stdout = _repair_stream_bytes(raw["stdout"], "report repair scontrol stdout")
    try:
        text = stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReportError(f"report repair scontrol output is not UTF-8: {exc}") from exc
    fields: dict[str, str] = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, item = token.split("=", 1)
        require(key not in fields, f"report repair scontrol field duplicated: {key}")
        fields[key] = item
    username = str(authority_environment["USER"])
    expected_user_id = f"{username}({os.getuid()})"
    source_root = Path(str(repair_authorization["repair_source_root"]))
    require(
        source_root == submission_root / SOURCE_ARCHIVE_NAME,
        "report repair held-job source archive path differs",
    )
    require(
        re.fullmatch(r"/proc/self/fd/[0-9]+", scheduler_source_argument)
        is not None,
        "report repair held-job scheduler source argument differs",
    )
    expected_output = str(
        submission_root / "logs" / f"report-repair-0002-{job_id}.out"
    )
    array_fields = {key for key in fields if key.startswith("Array")}
    heterogeneous_fields = {key for key in fields if key.startswith("Het")}
    selected = (
        "JobId", "JobName", "UserId", "JobState", "Reason", "Requeue",
        "Restarts", "BatchFlag", "TimeLimit", "Comment", "Partition", "Account", "QOS", "NumNodes",
        "NumCPUs", "NumTasks", "CPUs/Task", "MinMemoryNode", "Command",
        "WorkDir", "StdOut", "StdErr", "StdIn",
    )
    projection = {
        "schema_version": 1,
        "fields": {key: fields.get(key) for key in selected},
        "no_requeue": True,
        "restart_count": 0,
        "held": True,
        "array_identity_absent": True,
        "heterogeneous_identity_absent": True,
    }
    require(
        fields.get("JobId") == job_id
        and fields.get("JobName") == repair_authorization["repair_job_name"]
        and fields.get("UserId") == expected_user_id
        and fields.get("JobState") == "PENDING"
        and fields.get("Reason") in {"JobHeldUser", "JobHeldAdmin"}
        and fields.get("Requeue") == "0"
        and fields.get("Restarts") == "0"
        and fields.get("BatchFlag") == "1"
        and fields.get("TimeLimit") == "04:00:00"
        and fields.get("Comment") == repair_authorization["scheduler_comment"]
        and fields.get("Partition") == "cpu"
        and fields.get("Account") == "edgeai_tao-ptm_image-foundation-model-clip"
        and fields.get("QOS") == "normal"
        and fields.get("NumNodes") == "1"
        and fields.get("NumCPUs") == "12"
        and fields.get("NumTasks") == "1"
        and fields.get("CPUs/Task") == "12"
        and fields.get("MinMemoryNode") == "64G"
        and fields.get("Command") == scheduler_source_argument
        and fields.get("WorkDir")
        == str(submission_root / "source-snapshot" / "repo")
        and fields.get("StdOut") == expected_output
        and fields.get("StdErr") == expected_output
        and fields.get("StdIn") == "/dev/null"
        and not array_fields
        and not heterogeneous_fields
        and exact_json_equal(value["projection"], projection),
        "report repair held-job scheduler control differs",
    )
    return dict(value)


def _validated_active_source_archive(
    submission_root: Path,
    phase_binding: "_RepairPublicationPhaseBinding",
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Authenticate the exact retained archive FD used to execute this worker."""

    descriptor = globals().get("__EXP23_REPAIR_SOURCE_ARCHIVE_FD__")
    raw_path = globals().get("__EXP23_REPAIR_SOURCE_ARCHIVE_PATH__")
    expected_sha = globals().get("__EXP23_REPAIR_SOURCE_ARCHIVE_SHA256__")
    expected_size = globals().get("__EXP23_REPAIR_SOURCE_ARCHIVE_SIZE__")
    injected_value = globals().get("__EXP23_REPAIR_SOURCE_ARCHIVE_VALUE__")
    execution_path = globals().get(
        "__EXP23_REPAIR_SOURCE_ARCHIVE_EXECUTION_PATH__"
    )
    path = Path(str(raw_path))
    require(
        type(descriptor) is int
        and descriptor >= 0
        and path == submission_root / SOURCE_ARCHIVE_NAME
        and sha256_string(expected_sha)
        and type(expected_size) is int
        and 0 < expected_size <= 8 * 1024 * 1024
        and isinstance(injected_value, Mapping),
        "active repair source archive bootstrap differs",
    )
    before = os.fstat(descriptor)
    retained = phase_binding.retained_regular(path)
    require(retained is not None, "active repair source archive is not retained")
    retained_payload, retained_info = retained
    execution = phase_binding.execution_source_archive
    require(
        stat.S_ISREG(before.st_mode)
        and execution is not None
        and descriptor == execution[0]
        and _file_identity(before) == execution[1]
        and isinstance(execution_path, str)
        and bool(execution_path)
        and stat.S_ISREG(retained_info.st_mode)
        and retained_info.st_uid == os.getuid()
        and retained_info.st_nlink == 1
        and stat.S_IMODE(retained_info.st_mode) == 0o444
        and before.st_size == expected_size,
        "active repair source archive execution/canonical identity differs",
    )
    raw = bytearray()
    offset = 0
    while offset < expected_size:
        block = os.pread(descriptor, min(1024 * 1024, expected_size - offset), offset)
        require(block, "active repair source archive is truncated")
        raw.extend(block)
        offset += len(block)
    after = os.fstat(descriptor)
    payload = bytes(raw)
    require(
        _file_identity(after) == _file_identity(before)
        and payload == execution[2]
        and payload == retained_payload
        and hashlib.sha256(payload).hexdigest() == expected_sha
        and payload.count(SOURCE_ARCHIVE_MARKER) == 1
        and payload.endswith(SOURCE_ARCHIVE_END),
        "active repair source archive bytes differ",
    )
    prefix, tail = payload.split(SOURCE_ARCHIVE_MARKER, 1)
    prefix += SOURCE_ARCHIVE_MARKER
    body = tail[: -len(SOURCE_ARCHIVE_END)]
    try:
        value = json.loads(body.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"active repair source archive JSON differs: {exc}") from exc
    require(
        body
        == json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        and exact_json_equal(value, injected_value)
        and isinstance(value, Mapping)
        and set(value) == {"archive_kind", "schema_version", "authority", "files"}
        and value.get("archive_kind") == SOURCE_ARCHIVE_KIND
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 2
        and isinstance(value.get("authority"), Mapping)
        and isinstance(value.get("files"), Mapping)
        and set(value["files"])
        == {"protocol.sha256", "report.py", "report_repair.py", "report_repair.slurm"},
        "active repair source archive envelope differs",
    )
    decoded: dict[str, bytes] = {}
    projection: dict[str, Any] = {}
    for name in sorted(value["files"]):
        row = value["files"][name]
        require(
            isinstance(row, Mapping)
            and set(row) == {"data_base64", "mode", "sha256", "size"}
            and row.get("mode") == 0o444
            and type(row.get("size")) is int
            and row["size"] >= 0
            and sha256_string(row.get("sha256"))
            and isinstance(row.get("data_base64"), str),
            f"active repair source archive row differs: {name}",
        )
        try:
            item = base64.b64decode(row["data_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise ReportError(f"active repair source archive base64 differs: {name}") from exc
        require(
            len(item) == row["size"]
            and hashlib.sha256(item).hexdigest() == row["sha256"],
            f"active repair source archive payload differs: {name}",
        )
        decoded[name] = item
        projection[name] = {"mode": 0o444, "size": len(item), "sha256": row["sha256"]}
    authority = value["authority"]
    require(
        set(authority) == REPORT_REPAIR_SOURCE_AUTHORITY_V2_KEYS
        and authority.get("schema_version") == 2
        and authority.get("repair_source_installation_method")
        == SOURCE_ARCHIVE_INSTALL_METHOD
        and exact_json_equal(authority.get("repair_source_files"), projection)
        and authority.get("repair_source_files_sha256") == stable_hash(projection)
        and decoded["report_repair.slurm"] == prefix
        and decoded["protocol.sha256"]
        == f"{authority.get('repair_package_protocol_sha256')}\n".encode("ascii")
        and str(__file__) == f"{path}::report.py",
        "active repair source archive authority differs",
    )
    evidence = {
        **dict(authority),
        "repair_source_archive": str(path),
        "repair_source_archive_sha256": expected_sha,
        "repair_source_archive_size": expected_size,
        "repair_source_archive_format": SOURCE_ARCHIVE_KIND,
    }
    require(
        set(evidence) == REPORT_REPAIR_SOURCE_ARCHIVE_EVIDENCE_KEYS,
        "active repair source archive evidence differs",
    )
    return evidence, decoded


def _validated_report_repair_filesystem_namespace(
    submission_root: Path,
) -> None:
    """Require immutable attempt-1 history plus the attempt-2 source archive."""

    repair_parent = nonsymlink_directory(
        submission_root / "report-repair", "report repair state root"
    )
    parent_info = repair_parent.lstat()
    require(
        parent_info.st_uid == os.getuid()
        and stat.S_IMODE(parent_info.st_mode) == 0o700
        and {entry.name for entry in os.scandir(repair_parent)}
        == {"attempt-0001"},
        "report repair attempt namespace differs",
    )
    attempt_root = nonsymlink_directory(
        repair_parent / "attempt-0001", "report repair attempt1 root"
    )
    attempt_info = attempt_root.lstat()
    require(
        attempt_info.st_uid == os.getuid()
        and stat.S_IMODE(attempt_info.st_mode) == 0o700
        and {entry.name for entry in os.scandir(attempt_root)} == {"source"},
        "report repair attempt1 source namespace differs",
    )
    source_root = nonsymlink_directory(
        attempt_root / "source", "report repair attempt1 source root"
    )
    source_info = source_root.lstat()
    require(
        source_info.st_uid == os.getuid()
        and source_info.st_nlink == 2
        and stat.S_IMODE(source_info.st_mode) == 0o555,
        "report repair attempt1 source root identity differs",
    )
    archive = contained_regular(
        submission_root / SOURCE_ARCHIVE_NAME,
        submission_root,
        "report repair source archive",
    )
    archive_info = archive.lstat()
    require(
        archive_info.st_uid == os.getuid()
        and archive_info.st_nlink == 1
        and stat.S_IMODE(archive_info.st_mode) == 0o444,
        "report repair source archive identity differs",
    )


def _validated_repair_publication_namespace(submission_root: Path) -> None:
    names = set(os.listdir(submission_root))
    reserved = {
        name
        for name in names
        if name == "report"
        or name.startswith(".report")
        or (
            name.startswith(PUBLICATION_ARCHIVE_PREFIX)
            and name.endswith(PUBLICATION_ARCHIVE_SUFFIX)
        )
    }
    require(
        len(reserved) <= 1
        and all(
            name.startswith(PUBLICATION_ARCHIVE_PREFIX)
            and name.endswith(PUBLICATION_ARCHIVE_SUFFIX)
            for name in reserved
        ),
        "report repair publication namespace differs",
    )


class _RepairPublicationPhaseBinding:
    """Pin the exact immutable repair journal consumed by the publisher."""

    def __init__(
        self,
        submission_root: Path,
        *,
        allow_completed_stage: bool = False,
        locks: "_RepairPublicationLocks | None" = None,
    ):
        self.submission_root = submission_root
        self.locks = locks
        self.lock_bindings = locks.bindings() if locks is not None else None
        self.root_descriptor = _open_directory_components(
            submission_root, "report repair publication submission root"
        )
        self.root_identity = os.fstat(self.root_descriptor)
        self.root_reserved = self._reserved_root_names()
        self.source_descriptors: list[int] = []
        self.source_file_rows: list[
            tuple[int, str, int, tuple[int, ...], bytes]
        ] = []
        self.stream_file_rows: list[
            tuple[str, int, tuple[int, ...], str, int]
        ] = []
        self.source_rows: list[
            tuple[Path, int, tuple[int, ...], frozenset[str]]
        ] = []
        self.execution_source_archive: tuple[
            int, tuple[int, ...], bytes
        ] | None = None
        self.journal_file_rows: dict[
            str, tuple[int, tuple[int, ...], bytes]
        ] = {}
        self.scientific_runs: dict[Path, dict[str, Any]] = {}
        self.root = submission_root / "journal"
        self.allow_completed_stage = allow_completed_stage
        self.descriptor = _open_directory_components(
            self.root, "report repair publication journal"
        )
        try:
            self._capture_sources()
            self.identity = os.fstat(self.descriptor)
            self.journal_namespace = frozenset(os.listdir(self.descriptor))
            self._capture_journal_authority()
            self.names, self.payloads, self.stage_state = self._snapshot()
        except BaseException:
            self.close()
            raise

    def _reserved_root_names(self) -> frozenset[str]:
        return frozenset(
            name
            for name in os.listdir(self.root_descriptor)
            if name == "report"
            or name == "report-repair"
            or name == "journal"
            or name == "CANCEL_REQUESTED.json"
            or name.startswith(".report")
            or name == SOURCE_ARCHIVE_NAME
            or (
                name.startswith(PUBLICATION_ARCHIVE_PREFIX)
                and name.endswith(PUBLICATION_ARCHIVE_SUFFIX)
            )
            or _repair_root_name_is_reserved(name)
        )

    @staticmethod
    def _read_fd(descriptor: int) -> bytes:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)

    def _capture_sources(self) -> None:
        repair_parent = self.submission_root / "report-repair"
        repair_parent_fd = _open_directory_components(
            repair_parent, "repair publication attempt namespace"
        )
        self.source_descriptors.append(repair_parent_fd)
        repair_parent_info = os.fstat(repair_parent_fd)
        repair_parent_names = frozenset(os.listdir(repair_parent_fd))
        require(
            repair_parent_names == frozenset({"attempt-0001"})
            and repair_parent_info.st_uid == os.getuid()
            and repair_parent_info.st_nlink == 3
            and stat.S_IMODE(repair_parent_info.st_mode) == 0o700,
            "repair publication attempt namespace differs",
        )
        self.source_rows.append(
            (
                repair_parent,
                repair_parent_fd,
                _file_identity(repair_parent_info),
                repair_parent_names,
            )
        )
        attempt_root = repair_parent / "attempt-0001"
        attempt_fd = os.open(
            "attempt-0001",
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=repair_parent_fd,
        )
        self.source_descriptors.append(attempt_fd)
        attempt_info = os.fstat(attempt_fd)
        attempt_listed = os.stat(
            "attempt-0001", dir_fd=repair_parent_fd, follow_symlinks=False
        )
        attempt_names = frozenset(os.listdir(attempt_fd))
        require(
            _file_identity(attempt_info) == _file_identity(attempt_listed)
            and attempt_info.st_uid == os.getuid()
            and attempt_info.st_nlink == 3
            and stat.S_IMODE(attempt_info.st_mode) == 0o700
            and attempt_names == frozenset({"source"}),
            "attempt1 repair publication namespace differs",
        )
        self.source_rows.append(
            (
                attempt_root,
                attempt_fd,
                _file_identity(attempt_info),
                attempt_names,
            )
        )
        root = attempt_root / "source"
        descriptor = os.open(
            "source",
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=attempt_fd,
        )
        self.source_descriptors.append(descriptor)
        opened = os.fstat(descriptor)
        listed = os.stat("source", dir_fd=attempt_fd, follow_symlinks=False)
        names = frozenset(os.listdir(descriptor))
        require(
            stat.S_ISDIR(opened.st_mode)
            and _file_identity(opened) == _file_identity(listed)
            and opened.st_uid == listed.st_uid == os.getuid()
            and opened.st_nlink == listed.st_nlink == 2
            and stat.S_IMODE(opened.st_mode)
            == stat.S_IMODE(listed.st_mode)
            == 0o555,
            "attempt1 retained repair source root differs",
        )
        self.source_rows.append((root, descriptor, _file_identity(opened), names))
        for name in sorted(names):
            self._retain_authority_file(
                descriptor,
                name,
                "attempt1 retained repair source file",
            )

        root_names = set(os.listdir(self.root_descriptor))
        root_repair_names = {
            name for name in root_names if _repair_root_name_is_reserved(name)
        }
        require(
            all(_repair_root_artifact_name_is_allowed(name) for name in root_repair_names),
            "repair publication root contains an unknown attempt2 artifact",
        )
        require(
            SOURCE_ARCHIVE_NAME in root_names,
            "repair publication source archive is absent",
        )
        inherited_archive_fd = globals().get("__EXP23_REPAIR_SOURCE_ARCHIVE_FD__")
        require(
            type(inherited_archive_fd) is int and inherited_archive_fd >= 0,
            "repair publication inherited source archive descriptor is absent",
        )
        self._retain_authority_file(
            self.root_descriptor,
            SOURCE_ARCHIVE_NAME,
            "repair publication canonical source archive",
        )
        canonical_rows = [
            row
            for row in self.source_file_rows
            if row[0] == self.root_descriptor and row[1] == SOURCE_ARCHIVE_NAME
        ]
        require(
            len(canonical_rows) == 1,
            "repair publication canonical source archive is ambiguous",
        )
        canonical_payload = canonical_rows[0][4]
        opened_archive = os.fstat(inherited_archive_fd)
        archive_payload = self._read_fd(inherited_archive_fd)
        require(
            stat.S_ISREG(opened_archive.st_mode)
            and archive_payload == canonical_payload,
            "repair publication spooled source archive bytes differ",
        )
        self.execution_source_archive = (
            inherited_archive_fd,
            _file_identity(opened_archive),
            archive_payload,
        )
        for name in sorted(root_names):
            if (
                name != SOURCE_ARCHIVE_NAME
                and not (
                    name.startswith(PUBLICATION_ARCHIVE_PREFIX)
                    and name.endswith(PUBLICATION_ARCHIVE_SUFFIX)
                )
                and _repair_root_name_is_reserved(name)
                and _repair_journal_artifact_name_is_allowed(name)
            ):
                self._retain_authority_file(
                    self.root_descriptor,
                    name,
                    "repair publication root successor",
                )
        require(
            len(
                {
                    (parent_fd, name)
                    for parent_fd, name, _descriptor, _identity, _payload in self.source_file_rows
                }
            )
            == len(self.source_file_rows),
            "repair publication retained file graph is not bijective",
        )
        publication_names = sorted(
            name
            for name in root_names
            if name.startswith(PUBLICATION_ARCHIVE_PREFIX)
            and name.endswith(PUBLICATION_ARCHIVE_SUFFIX)
        )
        require(
            len(publication_names) <= 1,
            "repair publication archive generation differs",
        )
        for name in publication_names:
            self._retain_stream_root_file(name)
        for name in (
            "SUBMISSION_CONTRACT.json",
            "SUBMISSION_RECEIPT.json",
            "SUBMISSION_AUTHORIZATION.json",
        ):
            require(name in root_names, f"repair publication authority is absent: {name}")
            self._retain_authority_file(
                self.root_descriptor, name, "repair publication root authority"
            )

        snapshot = self.submission_root / "source-snapshot"
        require(
            _lexical_exists(snapshot, "repair publication source snapshot"),
            "repair publication source snapshot is absent",
        )
        self._capture_authority_tree(snapshot, "repair publication source snapshot")
        self._capture_compatible_contract_authority(snapshot)

        tasks = self.submission_root / "tasks"
        task_names = frozenset(f"cell-{index:02d}" for index in range(20))
        tasks_fd = self._retain_authority_directory(
            tasks,
            "repair publication task root",
            required_names=task_names,
            exact_names=True,
            expected_mode=0o700,
        )
        del tasks_fd
        for cell_name in sorted(task_names):
            cell_fd = self._retain_authority_directory(
                tasks / cell_name,
                f"repair publication task {cell_name}",
                required_names=frozenset(
                    {"LAUNCH.json", "WORKER_COMPLETE.json", "waves"}
                ),
                exact_names=True,
                expected_mode=0o700,
            )
            for name in ("LAUNCH.json", "WORKER_COMPLETE.json"):
                self._retain_authority_file(
                    cell_fd,
                    name,
                    f"repair publication task {cell_name} artifact",
                    allowed_modes=frozenset({0o444}),
                )
            waves = tasks / cell_name / "waves"
            waves_fd = self._retain_authority_directory(
                waves,
                f"repair publication task {cell_name} waves",
                required_names=frozenset({"0", "1"}),
                exact_names=True,
                expected_mode=0o700,
            )
            del waves_fd
            for wave_index in (0, 1):
                wave = waves / str(wave_index)
                wave_fd = self._retain_authority_directory(
                    wave,
                    f"repair publication task {cell_name} wave{wave_index}",
                    expected_mode=0o700,
                )
                names = frozenset(os.listdir(wave_fd))
                mandatory = {"START.json", "WORKER_SIGNAL_READY.json"}
                optional_signals = {"USR1_REQUESTED.json", "TERM_REQUESTED.json"}
                terminal = (
                    {"WORKER_COMPLETE.json", "CONTINUATION_READY.json"}
                    if wave_index == 0
                    else {"WORKER_COMPLETE.json"}
                )
                require(
                    mandatory <= names
                    and len(names & terminal) == 1
                    and names <= mandatory | terminal | optional_signals,
                    f"repair publication task {cell_name} wave{wave_index} namespace differs",
                )
                for name in sorted(names):
                    self._retain_authority_file(
                        wave_fd,
                        name,
                        f"repair publication task {cell_name} wave{wave_index} artifact",
                        allowed_modes=frozenset({0o444}),
                    )

        launches = self.submission_root / "launches"
        launch_names = frozenset(
            f"cell-{index:02d}.json" for index in range(20)
        )
        launches_fd = self._retain_authority_directory(
            launches,
            "repair publication launch root",
            required_names=launch_names,
            exact_names=True,
            expected_mode=0o700,
        )
        for name in sorted(launch_names):
            self._retain_authority_file(
                launches_fd,
                name,
                "repair publication launch artifact",
                allowed_modes=frozenset({0o444}),
            )

        logs = self.submission_root / "logs"
        attempt1_log = f"report-repair-0001-{EXPECTED_ATTEMPT1_JOB_ID}.out"
        logs_fd = self._retain_authority_directory(
            logs,
            "repair publication logs",
            required_names=frozenset({attempt1_log}),
            expected_mode=0o700,
        )
        self._retain_authority_file(
            logs_fd, attempt1_log, "repair publication attempt1 failure log"
        )

    def _capture_compatible_contract_authority(self, snapshot: Path) -> None:
        """Retain the external JSON inputs used by ``trainer_command``.

        The immutable source snapshot carries the canonical absolute contract
        root and setting order.  Report reconstruction calls
        ``campaign.trainer_command(..., verify_recipe_files=False)`` for every
        launch; that path consumes exactly the setting input contract and the
        corresponding future-recipe manifest.  Retain those ten files before
        any snapshot code is executed so the imported campaign module cannot
        fall back to mutable pathname reads.
        """

        manifest_path = (
            snapshot / "repo" / PACKAGE_RELATIVE / "manifest.json"
        )
        retained_manifest = self.retained_regular(manifest_path)
        if retained_manifest is None:
            # Small unit fixtures may intentionally omit the Exp23 package.
            # A real repair is later rejected by exact snapshot validation if
            # its manifest is absent, so this cannot create a production
            # bypass.
            return
        manifest_payload, _manifest_info = retained_manifest
        manifest = _decode_json_object(manifest_path, manifest_payload)
        paths = manifest.get("paths")
        settings = manifest.get("settings")
        require(
            isinstance(paths, Mapping)
            and isinstance(paths.get("compatible_contract_root"), str)
            and bool(paths["compatible_contract_root"])
            and isinstance(settings, list)
            and len(settings) == 5,
            "repair compatible-contract authority is absent",
        )
        compatible_root = Path(paths["compatible_contract_root"])
        require(
            compatible_root.is_absolute()
            and all(
                part not in {"", ".", ".."}
                for part in compatible_root.parts[1:]
            ),
            "repair compatible-contract root differs",
        )
        setting_ids: list[str] = []
        for row in settings:
            setting_id = row.get("id") if isinstance(row, Mapping) else None
            require(
                isinstance(setting_id, str)
                and setting_id
                and setting_id.isascii()
                and setting_id not in setting_ids
                and "/" not in setting_id
                and "\\" not in setting_id
                and setting_id not in {".", ".."},
                "repair compatible-contract setting differs",
            )
            setting_ids.append(setting_id)

        data_root = compatible_root / "data"
        data_names = frozenset(f"{setting_id}.json" for setting_id in setting_ids)
        data_fd = self._retain_authority_directory(
            data_root,
            "repair compatible-contract data root",
            required_names=data_names,
        )
        for name in sorted(data_names):
            self._retain_authority_file(
                data_fd,
                name,
                "repair compatible-contract data artifact",
                allowed_modes=frozenset({0o444}),
            )

        recipe_root = compatible_root / "future-recipes"
        recipe_names = frozenset(setting_ids)
        recipe_root_fd = self._retain_authority_directory(
            recipe_root,
            "repair compatible-contract recipe root",
            required_names=recipe_names,
        )
        del recipe_root_fd
        for setting_id in setting_ids:
            setting_root = recipe_root / setting_id
            setting_fd = self._retain_authority_directory(
                setting_root,
                f"repair compatible-contract recipe {setting_id}",
                required_names=frozenset({"manifest.json"}),
            )
            self._retain_authority_file(
                setting_fd,
                "manifest.json",
                f"repair compatible-contract recipe {setting_id} manifest",
                allowed_modes=frozenset({0o444}),
            )

    def _retain_authority_directory(
        self,
        path: Path,
        label: str,
        *,
        required_names: frozenset[str] = frozenset(),
        exact_names: bool = False,
        expected_mode: int | None = None,
    ) -> int:
        descriptor = _open_directory_components(path, label)
        self.source_descriptors.append(descriptor)
        opened = os.fstat(descriptor)
        named = path.lstat()
        names = frozenset(os.listdir(descriptor))
        require(
            stat.S_ISDIR(opened.st_mode)
            and _file_identity(opened) == _file_identity(named)
            and opened.st_uid == named.st_uid == os.getuid()
            and opened.st_nlink == named.st_nlink
            and opened.st_nlink >= 2
            and (
                expected_mode is None
                or stat.S_IMODE(opened.st_mode)
                == stat.S_IMODE(named.st_mode)
                == expected_mode
            )
            and (names == required_names if exact_names else required_names <= names),
            f"{label} identity differs",
        )
        self.source_rows.append((path, descriptor, _file_identity(opened), names))
        return descriptor

    def _retain_authority_file(
        self,
        parent_fd: int,
        name: str,
        label: str,
        *,
        allowed_modes: frozenset[int] = frozenset({0o444, 0o600}),
    ) -> None:
        listed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        file_descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_fd,
        )
        try:
            opened_file = os.fstat(file_descriptor)
            require(
                stat.S_ISREG(opened_file.st_mode)
                and _file_identity(opened_file) == _file_identity(listed)
                and opened_file.st_uid == listed.st_uid == os.getuid()
                and opened_file.st_nlink == listed.st_nlink == 1
                and stat.S_IMODE(opened_file.st_mode)
                == stat.S_IMODE(listed.st_mode)
                in allowed_modes,
                f"{label} differs: {name}",
            )
            payload = self._read_fd(file_descriptor)
            self.source_file_rows.append(
                (
                    parent_fd,
                    name,
                    file_descriptor,
                    _file_identity(opened_file),
                    payload,
                )
            )
        except BaseException:
            os.close(file_descriptor)
            raise

    def _retain_stream_root_file(self, name: str) -> None:
        listed = os.stat(
            name, dir_fd=self.root_descriptor, follow_symlinks=False
        )
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=self.root_descriptor,
        )
        try:
            opened = os.fstat(descriptor)
            digest, after = _stable_open_fd_sha256(
                descriptor, f"retained publication archive {name}"
            )
            require(
                stat.S_ISREG(opened.st_mode)
                and _file_identity(opened) == _file_identity(listed)
                == _file_identity(after)
                and opened.st_uid == os.getuid()
                and opened.st_nlink == 1
                and stat.S_IMODE(opened.st_mode) == 0o444
                and name
                == f"{PUBLICATION_ARCHIVE_PREFIX}{digest}{PUBLICATION_ARCHIVE_SUFFIX}",
                "retained publication archive identity differs",
            )
            self.stream_file_rows.append(
                (name, descriptor, _file_identity(opened), digest, opened.st_size)
            )
        except BaseException:
            os.close(descriptor)
            raise

    def _capture_journal_authority(self) -> None:
        """Retain every pre-existing journal file, not only repair successors."""

        for name in sorted(self.journal_namespace):
            listed = os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
            require(
                stat.S_ISREG(listed.st_mode),
                f"repair publication journal contains a special entry: {name}",
            )
            self._read(name)

    def _capture_authority_tree(self, path: Path, label: str) -> None:
        descriptor = self._retain_authority_directory(path, label)
        for name in sorted(os.listdir(descriptor)):
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                self._capture_authority_tree(path / name, f"{label}/{name}")
            elif stat.S_ISREG(info.st_mode):
                self._retain_authority_file(descriptor, name, label)
            else:
                raise ReportError(f"{label} has a special entry: {name}")

    @staticmethod
    def _scientific_row(
        relative: Path,
        kind: str,
        info: os.stat_result,
        *,
        digest: str | None = None,
        target: bytes | None = None,
    ) -> dict[str, Any]:
        row: dict[str, Any] = {
            "path": "" if not relative.parts else str(relative),
            "kind": kind,
            "mode": stat.S_IMODE(info.st_mode),
            "device": info.st_dev,
            "inode": info.st_ino,
            "uid": info.st_uid,
            "gid": info.st_gid,
            "nlink": info.st_nlink,
            "size": info.st_size,
            "mtime_ns": info.st_mtime_ns,
            "ctime_ns": info.st_ctime_ns,
        }
        if digest is not None:
            row["sha256"] = digest
        if target is not None:
            row["readlink_bytes_hex"] = target.hex()
        return row

    def retain_scientific_run_tree(self, run_root: Path) -> None:
        """Pin one exact scientific run tree through publication/COMPLETED."""

        root = run_root.absolute()
        require(
            root.is_absolute()
            and all(part not in {"", ".", ".."} for part in root.parts[1:]),
            "scientific run retained root path differs",
        )
        if root in self.scientific_runs:
            self._revalidate_scientific_run(root, full_hash=True)
            return
        self.revalidate()
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
        directories: dict[
            Path, tuple[int, tuple[int, ...], frozenset[str], int | None, str | None]
        ] = {}
        files: dict[
            Path, tuple[int, tuple[int, ...], str, int, str]
        ] = {}
        symlinks: dict[
            Path, tuple[int, tuple[int, ...], bytes, int, str]
        ] = {}
        rows: list[dict[str, Any]] = []

        def close_partial() -> None:
            for descriptor, *_rest in files.values():
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            for descriptor, *_rest in symlinks.values():
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            for descriptor, *_rest in reversed(list(directories.values())):
                try:
                    os.close(descriptor)
                except OSError:
                    pass

        def walk(
            directory_fd: int,
            relative: Path,
            opened: os.stat_result,
        ) -> None:
            children = []
            try:
                with os.scandir(directory_fd) as iterator:
                    for entry in iterator:
                        children.append(
                            (entry.name, entry.stat(follow_symlinks=False))
                        )
            except OSError as exc:
                raise ReportError(
                    f"cannot enumerate retained scientific run {root / relative}: {exc}"
                ) from exc
            names = frozenset(name for name, _info in children)
            previous = directories[relative]
            directories[relative] = (
                previous[0],
                previous[1],
                names,
                previous[3],
                previous[4],
            )
            for name, listed in sorted(children, key=lambda item: item[0]):
                child_relative = relative / name
                if stat.S_ISDIR(listed.st_mode):
                    require(
                        listed.st_uid == os.getuid()
                        and stat.S_IMODE(listed.st_mode) & 0o444 != 0
                        and stat.S_IMODE(listed.st_mode) & 0o111 != 0,
                        f"retained scientific directory differs: {child_relative}",
                    )
                    child_fd = os.open(
                        name, directory_flags, dir_fd=directory_fd
                    )
                    child_opened = os.fstat(child_fd)
                    require(
                        _file_identity(child_opened) == _file_identity(listed),
                        f"retained scientific directory raced: {child_relative}",
                    )
                    directories[child_relative] = (
                        child_fd,
                        _file_identity(child_opened),
                        frozenset(),
                        directory_fd,
                        name,
                    )
                    rows.append(
                        self._scientific_row(
                            child_relative, "directory", child_opened
                        )
                    )
                    walk(child_fd, child_relative, child_opened)
                elif stat.S_ISREG(listed.st_mode):
                    require(
                        listed.st_uid == os.getuid()
                        and listed.st_nlink == 1
                        and stat.S_IMODE(listed.st_mode) & 0o444 != 0,
                        f"retained scientific file differs: {child_relative}",
                    )
                    child_fd = os.open(name, file_flags, dir_fd=directory_fd)
                    child_opened = os.fstat(child_fd)
                    digest, after = _stable_open_fd_sha256(
                        child_fd, f"retained scientific file {child_relative}"
                    )
                    require(
                        _file_identity(child_opened)
                        == _file_identity(listed)
                        == _file_identity(after),
                        f"retained scientific file raced: {child_relative}",
                    )
                    files[child_relative] = (
                        child_fd,
                        _file_identity(child_opened),
                        digest,
                        directory_fd,
                        name,
                    )
                    rows.append(
                        self._scientific_row(
                            child_relative,
                            "file",
                            child_opened,
                            digest=digest,
                        )
                    )
                elif stat.S_ISLNK(listed.st_mode):
                    require(
                        len(child_relative.parts) >= 2
                        and child_relative.parts[0] == "wandb"
                        and listed.st_uid == os.getuid()
                        and listed.st_nlink == 1
                        and getattr(os, "O_PATH", 0) != 0,
                        f"retained scientific symlink differs: {child_relative}",
                    )
                    child_fd = os.open(
                        name,
                        getattr(os, "O_PATH", 0)
                        | getattr(os, "O_NOFOLLOW", 0)
                        | getattr(os, "O_CLOEXEC", 0),
                        dir_fd=directory_fd,
                    )
                    child_opened = os.fstat(child_fd)
                    target = os.readlink(b"", dir_fd=child_fd)
                    require(
                        _file_identity(child_opened) == _file_identity(listed)
                        and len(target) == child_opened.st_size,
                        f"retained scientific symlink raced: {child_relative}",
                    )
                    symlinks[child_relative] = (
                        child_fd,
                        _file_identity(child_opened),
                        target,
                        directory_fd,
                        name,
                    )
                    rows.append(
                        self._scientific_row(
                            child_relative,
                            "symlink",
                            child_opened,
                            target=target,
                        )
                    )
                else:
                    raise ReportError(
                        f"retained scientific run contains a special file: {child_relative}"
                    )
            require(
                _file_identity(os.fstat(directory_fd)) == _file_identity(opened),
                f"retained scientific directory changed: {relative}",
            )

        try:
            root_fd = _open_directory_components(root, "retained scientific run root")
            root_info = os.fstat(root_fd)
            named_root = root.lstat()
            require(
                stat.S_ISDIR(root_info.st_mode)
                and _file_identity(root_info) == _file_identity(named_root)
                and root_info.st_uid == os.getuid()
                and stat.S_IMODE(root_info.st_mode) & 0o444 != 0
                and stat.S_IMODE(root_info.st_mode) & 0o111 != 0,
                "retained scientific run root identity differs",
            )
            directories[Path()] = (
                root_fd,
                _file_identity(root_info),
                frozenset(),
                None,
                None,
            )
            rows.append(self._scientific_row(Path(), "root", root_info))
            walk(root_fd, Path(), root_info)
            rows.sort(key=lambda row: str(row["path"]))
            self.scientific_runs[root] = {
                "directories": directories,
                "files": files,
                "symlinks": symlinks,
                "rows": tuple(rows),
            }
            self._revalidate_scientific_run(root, full_hash=True)
            self.revalidate()
        except BaseException:
            self.scientific_runs.pop(root, None)
            close_partial()
            raise

    def _revalidate_scientific_run(
        self, run_root: Path, *, full_hash: bool
    ) -> None:
        retained = self.scientific_runs.get(run_root.absolute())
        require(retained is not None, "retained scientific run is absent")
        directories = retained["directories"]
        for relative, (
            descriptor,
            identity,
            names,
            parent_fd,
            leaf,
        ) in directories.items():
            opened = os.fstat(descriptor)
            named = (
                run_root.lstat()
                if parent_fd is None
                else os.stat(str(leaf), dir_fd=parent_fd, follow_symlinks=False)
            )
            require(
                _file_identity(opened) == _file_identity(named) == identity
                and frozenset(os.listdir(descriptor)) == names,
                f"retained scientific directory changed: {relative}",
            )
        for relative, (descriptor, identity, digest, parent_fd, leaf) in retained[
            "files"
        ].items():
            opened = os.fstat(descriptor)
            named = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            require(
                _file_identity(opened) == _file_identity(named) == identity
                and (
                    not full_hash
                    or _stable_open_fd_sha256(
                        descriptor, f"retained scientific file {relative}"
                    )[0]
                    == digest
                ),
                f"retained scientific file changed: {relative}",
            )
        for relative, (descriptor, identity, target, parent_fd, leaf) in retained[
            "symlinks"
        ].items():
            opened = os.fstat(descriptor)
            named = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            require(
                _file_identity(opened) == _file_identity(named) == identity
                and os.readlink(b"", dir_fd=descriptor) == target,
                f"retained scientific symlink changed: {relative}",
            )

    def retained_scientific_tree_rows(
        self, run_root: Path
    ) -> list[dict[str, Any]] | None:
        root = run_root.absolute()
        retained = self.scientific_runs.get(root)
        if retained is None:
            return None
        self._revalidate_scientific_run(root, full_hash=False)
        return [dict(row) for row in retained["rows"]]

    def open_retained_scientific_regular(
        self, run_root: Path, relative: Path
    ) -> tuple[int, os.stat_result] | None:
        root = run_root.absolute()
        retained = self.scientific_runs.get(root)
        if retained is None:
            return None
        relative = _safe_relative(relative, "retained scientific artifact")
        row = retained["files"].get(relative)
        if row is None:
            return None
        descriptor, identity, _digest, parent_fd, leaf = row
        opened = os.fstat(descriptor)
        named = os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
        require(
            _file_identity(opened) == _file_identity(named) == identity,
            f"retained scientific artifact changed: {relative}",
        )
        duplicate = os.dup(descriptor)
        os.set_inheritable(duplicate, False)
        os.lseek(duplicate, 0, os.SEEK_SET)
        return duplicate, os.fstat(duplicate)

    def open_any_retained_scientific_regular(
        self, path: Path
    ) -> tuple[int, os.stat_result] | None:
        lexical = path.absolute()
        for root in self.scientific_runs:
            try:
                relative = lexical.relative_to(root)
            except ValueError:
                continue
            if not relative.parts:
                return None
            return self.open_retained_scientific_regular(root, relative)
        return None

    def manages_regular_path(self, path: Path) -> bool:
        lexical = path.absolute()
        if lexical == self.submission_root or self.submission_root in lexical.parents:
            return True
        if any(
            lexical == root or root in lexical.parents
            for root, _descriptor, _identity, _names in self.source_rows
        ):
            return True
        return any(
            lexical == root or root in lexical.parents
            for root in self.scientific_runs
        )

    def retained_scientific_artifact_exists(
        self, run_root: Path, relative: Path
    ) -> bool:
        retained = self.scientific_runs.get(run_root.absolute())
        require(retained is not None, "retained scientific run is absent")
        relative = _safe_relative(relative, "retained scientific artifact")
        return (
            relative in retained["files"]
            or relative in retained["directories"]
            or relative in retained["symlinks"]
        )

    def has_retained_scientific_run(self, run_root: Path) -> bool:
        return run_root.absolute() in self.scientific_runs

    def open_retained_scientific_directory(self, run_root: Path) -> int:
        retained = self.scientific_runs.get(run_root.absolute())
        require(retained is not None, "retained scientific run is absent")
        descriptor = retained["directories"][Path()][0]
        duplicate = os.dup(descriptor)
        os.set_inheritable(duplicate, False)
        return duplicate

    def _require_submission_and_sources(
        self, *, allowed_reserved_additions: frozenset[str] = frozenset()
    ) -> None:
        opened_root = os.fstat(self.root_descriptor)
        named_root = self.submission_root.lstat()
        require(
            (opened_root.st_dev, opened_root.st_ino)
            == (named_root.st_dev, named_root.st_ino)
            == (self.root_identity.st_dev, self.root_identity.st_ino)
            and opened_root.st_uid
            == named_root.st_uid
            == self.root_identity.st_uid
            == os.getuid()
            and stat.S_IMODE(opened_root.st_mode)
            == stat.S_IMODE(named_root.st_mode)
            == stat.S_IMODE(self.root_identity.st_mode)
            and self._reserved_root_names()
            == self.root_reserved | allowed_reserved_additions
            and (
                self.locks is None
                or exact_json_equal(self.locks.bindings(), self.lock_bindings)
            ),
            "report repair publication root/lock binding differs",
        )
        root_names = set(os.listdir(self.root_descriptor))
        require(
            not {name for name in root_names if name.startswith(".report")},
            "report repair publication transition namespace differs",
        )
        for root, descriptor, identity, names in self.source_rows:
            opened = os.fstat(descriptor)
            named = root.lstat()
            require(
                _file_identity(opened) == _file_identity(named) == identity
                and frozenset(os.listdir(descriptor)) == names,
                f"retained repair publication source changed: {root}",
            )
        for parent_fd, name, descriptor, identity, payload in self.source_file_rows:
            opened = os.fstat(descriptor)
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            require(
                _file_identity(opened) == _file_identity(named) == identity
                and self._read_fd(descriptor) == payload
                and _file_identity(os.fstat(descriptor)) == identity,
                f"retained repair publication source file changed: {name}",
            )
        require(
            self.execution_source_archive is not None,
            "retained repair execution archive is absent",
        )
        execution_fd, execution_identity, execution_payload = (
            self.execution_source_archive
        )
        execution_info = os.fstat(execution_fd)
        require(
            stat.S_ISREG(execution_info.st_mode)
            and _file_identity(execution_info) == execution_identity
            and self._read_fd(execution_fd) == execution_payload
            and _file_identity(os.fstat(execution_fd)) == execution_identity,
            "retained spooled repair source archive changed",
        )
        for name, descriptor, identity, digest, size in self.stream_file_rows:
            opened = os.fstat(descriptor)
            named = os.stat(
                name, dir_fd=self.root_descriptor, follow_symlinks=False
            )
            require(
                _file_identity(opened) == _file_identity(named) == identity
                and opened.st_uid == os.getuid()
                and opened.st_nlink == 1
                and stat.S_IMODE(opened.st_mode) == 0o444
                and opened.st_size == size
                and _stable_open_fd_sha256(
                    descriptor, f"retained publication archive {name}"
                )[0]
                == digest
                and _file_identity(os.fstat(descriptor)) == identity,
                f"retained streaming publication file changed: {name}",
            )
        for run_root in self.scientific_runs:
            self._revalidate_scientific_run(run_root, full_hash=False)

    def retained_regular(
        self, path: Path
    ) -> tuple[bytes, os.stat_result] | None:
        """Return bytes/stat from the already-retained descriptor, if present."""

        lexical = path.absolute()
        parent_fd: int | None = None
        if lexical.parent == self.submission_root:
            parent_fd = self.root_descriptor
        elif lexical.parent == self.root:
            parent_fd = self.descriptor
        else:
            for parent_path, descriptor, _identity, _names in self.source_rows:
                if lexical.parent == parent_path:
                    parent_fd = descriptor
                    break
        if parent_fd is None:
            return None
        rows = [
            row
            for row in self.source_file_rows
            if row[0] == parent_fd and row[1] == lexical.name
        ]
        if not rows and parent_fd == self.descriptor:
            retained = self.journal_file_rows.get(lexical.name)
            if retained is not None:
                descriptor, identity, payload = retained
                require(
                    _file_identity(os.fstat(descriptor)) == identity
                    and self._read_fd(descriptor) == payload,
                    f"retained report repair file changed: {lexical}",
                )
                return payload, os.fstat(descriptor)
        if not rows:
            return None
        require(len(rows) == 1, f"retained report repair file is ambiguous: {lexical}")
        _parent_fd, _name, descriptor, identity, payload = rows[0]
        require(
            _file_identity(os.fstat(descriptor)) == identity
            and self._read_fd(descriptor) == payload,
            f"retained report repair file changed: {lexical}",
        )
        return payload, os.fstat(descriptor)

    def open_retained_regular(
        self, path: Path
    ) -> tuple[int, os.stat_result] | None:
        """Duplicate one already-retained authority FD without reopening its path."""

        lexical = path.absolute()
        parent_fd: int | None = None
        if lexical.parent == self.submission_root:
            parent_fd = self.root_descriptor
        elif lexical.parent == self.root:
            parent_fd = self.descriptor
        else:
            for parent_path, descriptor, _identity, _names in self.source_rows:
                if lexical.parent == parent_path:
                    parent_fd = descriptor
                    break
        descriptor: int | None = None
        identity: tuple[int, ...] | None = None
        if parent_fd == self.descriptor:
            retained = self.journal_file_rows.get(lexical.name)
            if retained is not None:
                descriptor, identity, _payload = retained
        if descriptor is None and parent_fd is not None:
            rows = [
                row
                for row in self.source_file_rows
                if row[0] == parent_fd and row[1] == lexical.name
            ]
            if rows:
                require(
                    len(rows) == 1,
                    f"retained report repair file is ambiguous: {lexical}",
                )
                _parent_fd, _name, descriptor, identity, _payload = rows[0]
        if descriptor is None or identity is None:
            return None
        opened = os.fstat(descriptor)
        require(
            _file_identity(opened) == identity,
            f"retained report repair file changed: {lexical}",
        )
        duplicate = os.dup(descriptor)
        os.set_inheritable(duplicate, False)
        os.lseek(duplicate, 0, os.SEEK_SET)
        return duplicate, os.fstat(duplicate)

    def retained_directory(
        self, path: Path
    ) -> tuple[os.stat_result, frozenset[str]]:
        """Return a directory solely from the already-retained tree graph."""

        lexical = path.absolute()
        if lexical == self.submission_root:
            self._require_submission_and_sources()
            return os.fstat(self.root_descriptor), self.root_reserved
        if lexical == self.root:
            self._require_root()
            return os.fstat(self.descriptor), self.journal_namespace
        rows = [row for row in self.source_rows if row[0] == lexical]
        require(
            len(rows) == 1,
            f"retained report repair directory binding differs: {lexical}",
        )
        _root, descriptor, identity, names = rows[0]
        opened = os.fstat(descriptor)
        require(
            _file_identity(opened) == identity
            and frozenset(os.listdir(descriptor)) == names,
            f"retained report repair directory changed: {lexical}",
        )
        return opened, names

    def retained_authority_tree_rows(
        self, root: Path
    ) -> list[dict[str, Any]] | None:
        """Project one recursively retained authority tree without walking paths."""

        lexical = root.absolute()
        directory_rows = [
            row
            for row in self.source_rows
            if row[0] == lexical or lexical in row[0].parents
        ]
        if not any(path == lexical for path, *_rest in directory_rows):
            return None
        require(
            len({path for path, *_rest in directory_rows}) == len(directory_rows),
            f"retained authority directory graph is ambiguous: {lexical}",
        )
        parent_by_fd = {
            descriptor: path
            for path, descriptor, _identity, _names in directory_rows
        }
        rows: list[dict[str, Any]] = []
        for path, descriptor, identity, _names in directory_rows:
            info = os.fstat(descriptor)
            require(
                _file_identity(info) == identity,
                f"retained authority directory changed: {path}",
            )
            relative = path.relative_to(lexical)
            rows.append(
                {
                    "path": "" if not relative.parts else str(relative),
                    "kind": "root" if not relative.parts else "directory",
                    "mode": stat.S_IMODE(info.st_mode),
                    "device": info.st_dev,
                    "inode": info.st_ino,
                    "uid": info.st_uid,
                    "gid": info.st_gid,
                    "nlink": info.st_nlink,
                    "size": info.st_size,
                    "mtime_ns": info.st_mtime_ns,
                    "ctime_ns": info.st_ctime_ns,
                }
            )
        seen_files: set[Path] = set()
        for parent_fd, name, descriptor, identity, payload in self.source_file_rows:
            parent = parent_by_fd.get(parent_fd)
            if parent is None:
                continue
            path = parent / name
            require(path not in seen_files, f"retained authority file is ambiguous: {path}")
            seen_files.add(path)
            info = os.fstat(descriptor)
            require(
                _file_identity(info) == identity
                and self._read_fd(descriptor) == payload,
                f"retained authority file changed: {path}",
            )
            rows.append(
                {
                    "path": str(path.relative_to(lexical)),
                    "kind": "file",
                    "mode": stat.S_IMODE(info.st_mode),
                    "device": info.st_dev,
                    "inode": info.st_ino,
                    "uid": info.st_uid,
                    "gid": info.st_gid,
                    "nlink": info.st_nlink,
                    "size": info.st_size,
                    "mtime_ns": info.st_mtime_ns,
                    "ctime_ns": info.st_ctime_ns,
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
        rows.sort(key=lambda row: str(row["path"]))
        return rows

    def validate_exact_snapshot_authority(
        self, contract: Mapping[str, Any]
    ) -> None:
        """Validate the sealed snapshot solely from the retained FD graph."""

        snapshot_parent = self.submission_root / "source-snapshot"
        snapshot_root = snapshot_parent / "repo"
        require(
            contract.get("snapshot_root") == str(snapshot_root),
            "retained repair snapshot root differs",
        )
        raw_inventory = contract.get("snapshot_inventory")
        require(
            isinstance(raw_inventory, Mapping) and bool(raw_inventory),
            "retained repair snapshot inventory is absent",
        )
        inventory: dict[str, str] = {}
        for raw_relative, raw_digest in raw_inventory.items():
            relative = Path(raw_relative) if isinstance(raw_relative, str) else Path()
            require(
                isinstance(raw_relative, str)
                and sha256_string(raw_digest)
                and not relative.is_absolute()
                and bool(relative.parts)
                and all(part not in {"", ".", ".."} for part in relative.parts)
                and relative.as_posix() == raw_relative,
                "retained repair snapshot inventory row differs",
            )
            inventory[raw_relative] = raw_digest
        require(
            stable_hash(inventory) == contract.get("snapshot_inventory_sha256"),
            "retained repair snapshot inventory hash differs",
        )

        retained_directories = {
            path: (descriptor, identity, names)
            for path, descriptor, identity, names in self.source_rows
            if path == snapshot_parent
            or path == snapshot_root
            or snapshot_root in path.parents
        }
        require(
            len(retained_directories)
            == sum(
                1
                for path, *_rest in self.source_rows
                if path == snapshot_parent
                or path == snapshot_root
                or snapshot_root in path.parents
            ),
            "retained repair snapshot directory graph is ambiguous",
        )
        expected_files = {snapshot_root / relative for relative in inventory}
        expected_directories = {snapshot_parent, snapshot_root}
        for file_path in expected_files:
            parent = file_path.parent
            while parent != snapshot_root:
                expected_directories.add(parent)
                parent = parent.parent
        require(
            set(retained_directories) == expected_directories,
            "retained repair snapshot directory coverage differs",
        )
        path_by_fd = {
            descriptor: path
            for path, (descriptor, _identity, _names) in retained_directories.items()
        }
        retained_files: dict[
            Path, tuple[int, tuple[int, ...], bytes]
        ] = {}
        for parent_fd, name, descriptor, identity, payload in self.source_file_rows:
            parent_path = path_by_fd.get(parent_fd)
            if parent_path is None:
                continue
            path = parent_path / name
            require(
                path not in retained_files,
                f"retained repair snapshot file is ambiguous: {path}",
            )
            retained_files[path] = (descriptor, identity, payload)
        require(
            set(retained_files) == expected_files,
            "retained repair snapshot file coverage differs",
        )
        expected_children: dict[Path, set[str]] = {
            path: set() for path in expected_directories
        }
        for directory in expected_directories - {snapshot_parent}:
            expected_children[directory.parent].add(directory.name)
        for file_path in expected_files:
            expected_children[file_path.parent].add(file_path.name)
        for path, (descriptor, identity, names) in retained_directories.items():
            opened = os.fstat(descriptor)
            require(
                _file_identity(opened) == identity
                and opened.st_uid == os.getuid()
                and opened.st_nlink >= 2
                and stat.S_IMODE(opened.st_mode) == 0o555
                and names == frozenset(expected_children[path]),
                f"retained repair snapshot directory differs: {path}",
            )
        observed_inventory: dict[str, str] = {}
        for relative, expected_digest in inventory.items():
            descriptor, identity, payload = retained_files[
                snapshot_root / relative
            ]
            opened = os.fstat(descriptor)
            digest = hashlib.sha256(payload).hexdigest()
            require(
                _file_identity(opened) == identity
                and opened.st_uid == os.getuid()
                and opened.st_nlink == 1
                and stat.S_IMODE(opened.st_mode) == 0o444
                and opened.st_size == len(payload)
                and digest == expected_digest,
                f"retained repair snapshot file differs: {relative}",
            )
            observed_inventory[relative] = digest
        require(
            exact_json_equal(observed_inventory, inventory),
            "retained repair snapshot inventory differs",
        )

    def admit_release_evidence(self, name: str) -> bool:
        """Admit exactly RELEASED through the retained root descriptor."""

        require(
            name == "REPORT_REPAIR_0002_RELEASED.json",
            "repair release admission target differs",
        )
        current_reserved = self._reserved_root_names()
        if current_reserved == self.root_reserved:
            self.revalidate()
            return name in self.names
        require(
            current_reserved == self.root_reserved | {name}
            and name not in self.root_reserved,
            "repair release wait observed a foreign successor",
        )
        self._require_submission_and_sources(
            allowed_reserved_additions=frozenset({name})
        )
        self._retain_authority_file(
            self.root_descriptor, name, "repair release wait successor"
        )
        self.root_reserved = current_reserved
        names, payloads, stage_state = self._snapshot()
        require(
            names == self.names | {name}
            and payloads[name]
            == self.retained_regular(self.submission_root / name)[0]
            and stage_state is None,
            "repair release wait successor graph differs",
        )
        self.names = names
        self.payloads = payloads
        self.stage_state = None
        self.revalidate()
        return True

    def release_wait_is_open(self) -> bool:
        """Reject every retained attempt2 stop/successor before RELEASED."""

        self.revalidate()
        allowed_fixed = {
            *EXPECTED_ATTEMPT1_CHAIN_SHA256,
            "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json",
            "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json",
            "CALLING_REPORT_REPAIR_0002_SUBMIT.json",
            "REPORT_REPAIR_0002_SUBMITTED.json",
            "REPORT_REPAIR_0002_AUTHORIZED.json",
            "REPORT_REPAIR_0002_RELEASED.json",
        }
        allowed_patterns = (
            re.compile(r"CALLING_REPORT_REPAIR_0002_RELEASE_[0-9]{4}\.json\Z"),
            re.compile(r"REPORT_REPAIR_0002_RELEASE_RESULT_[0-9]{4}\.json\Z"),
        )
        return all(
            name in allowed_fixed
            or any(pattern.fullmatch(name) for pattern in allowed_patterns)
            for name in self.names
        )

    def _require_root(self) -> None:
        opened = os.fstat(self.descriptor)
        named = self.root.lstat()
        require(
            stat.S_ISDIR(opened.st_mode)
            and _file_identity(opened)
            == _file_identity(named)
            == _file_identity(self.identity)
            and opened.st_uid == named.st_uid == os.getuid()
            and stat.S_IMODE(opened.st_mode)
            == stat.S_IMODE(named.st_mode)
            == 0o700,
            "report repair publication journal binding differs",
        )
        require(
            frozenset(os.listdir(self.descriptor)) == self.journal_namespace,
            "report repair publication journal namespace changed",
        )

    def _read(self, name: str, *, expected_nlink: int = 1) -> bytes:
        listed = os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
        retained = self.journal_file_rows.get(name)
        if retained is not None:
            descriptor, identity, payload = retained
            opened = os.fstat(descriptor)
            require(
                stat.S_ISREG(opened.st_mode)
                and _file_identity(opened) == _file_identity(listed) == identity
                and opened.st_uid == os.getuid()
                and opened.st_nlink == expected_nlink
                and stat.S_IMODE(opened.st_mode) == 0o444
                and self._read_fd(descriptor) == payload
                and _file_identity(os.fstat(descriptor)) == identity,
                f"retained report repair publication journal changed: {name}",
            )
            return payload
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=self.descriptor,
        )
        keep_descriptor = False
        try:
            opened = os.fstat(descriptor)
            require(
                stat.S_ISREG(opened.st_mode)
                and _file_identity(opened) == _file_identity(listed)
                and opened.st_uid == os.getuid()
                and opened.st_nlink == expected_nlink
                and stat.S_IMODE(opened.st_mode) == 0o444,
                f"report repair publication journal identity differs: {name}",
            )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            require(
                _file_identity(after) == _file_identity(opened),
                f"report repair publication journal changed: {name}",
            )
            payload = b"".join(chunks)
            self.journal_file_rows[name] = (
                descriptor,
                _file_identity(opened),
                payload,
            )
            keep_descriptor = True
            return payload
        finally:
            if not keep_descriptor:
                os.close(descriptor)

    def _legacy_snapshot(
        self,
    ) -> tuple[frozenset[str], dict[str, bytes], tuple[object, ...] | None]:
        self._require_root()
        all_names = set(os.listdir(self.descriptor))
        names = frozenset(
            name
            for name in all_names
            if name.startswith("REPORT_REPAIR_")
            or name.startswith("CALLING_REPORT_REPAIR_")
        )
        stages = {
            name
            for name in all_names
            if name.startswith(".") and name.endswith(".seal.tmp")
        }
        completed_stage = ".REPORT_REPAIR_0002_COMPLETED.json.seal.tmp"
        require(
            not stages,
            "attempt2 publication journal staging is permanent fail-stop evidence",
        )
        stage_state: tuple[object, ...] | None = None
        linked_completed = False
        if "REPORT_REPAIR_0002_COMPLETED.json" in names:
            require(
                _lexical_exists(self.root.parent / "report"),
                "completed report repair evidence lacks a published report",
            )
        if stages:
            require(
                _lexical_exists(self.root.parent / "report"),
                "completed report repair stage lacks a published report",
            )
            stage_info = os.stat(
                completed_stage,
                dir_fd=self.descriptor,
                follow_symlinks=False,
            )
            target_name = "REPORT_REPAIR_0002_COMPLETED.json"
            target_info = (
                os.stat(target_name, dir_fd=self.descriptor, follow_symlinks=False)
                if target_name in names
                else None
            )
            linked_completed = (
                target_info is not None
                and (stage_info.st_dev, stage_info.st_ino)
                == (target_info.st_dev, target_info.st_ino)
            )
            require(
                stat.S_ISREG(stage_info.st_mode)
                and stage_info.st_uid == os.getuid()
                and stat.S_IMODE(stage_info.st_mode) in {0o600, 0o444}
                and (
                    (
                        target_info is None
                        and stage_info.st_nlink == 1
                    )
                    or (
                        linked_completed
                        and stat.S_IMODE(stage_info.st_mode) == 0o444
                        and stage_info.st_nlink == target_info.st_nlink == 2
                    )
                ),
                "completed report repair target/staging identity differs",
            )
            if target_info is not None:
                require(
                    linked_completed,
                    "completed report repair target and stage are distinct",
                )
            stage_payload = (
                self._read(
                    completed_stage,
                    expected_nlink=2 if linked_completed else 1,
                )
                if stat.S_IMODE(stage_info.st_mode) == 0o444
                else None
            )
            stage_state = (
                stat.S_IMODE(stage_info.st_mode),
                linked_completed,
                stage_payload,
            )
        payloads = {
            name: self._read(
                name,
                expected_nlink=(
                    2
                    if linked_completed
                    and name == "REPORT_REPAIR_0002_COMPLETED.json"
                    else 1
                ),
            )
            for name in sorted(names)
        }
        self._require_root()
        require(
            set(os.listdir(self.descriptor)) == all_names,
            "report repair publication journal namespace changed",
        )
        return names, payloads, stage_state

    def _read_root_successor(self, name: str) -> bytes:
        listed = os.stat(
            name, dir_fd=self.root_descriptor, follow_symlinks=False
        )
        retained = next(
            (
                row
                for row in self.source_file_rows
                if row[0] == self.root_descriptor and row[1] == name
            ),
            None,
        )
        if retained is not None:
            _parent_fd, _name, descriptor, identity, payload = retained
            opened = os.fstat(descriptor)
            require(
                _file_identity(opened) == _file_identity(listed) == identity
                and stat.S_IMODE(opened.st_mode) == 0o444
                and opened.st_uid == os.getuid()
                and opened.st_nlink == 1
                and self._read_fd(descriptor) == payload,
                f"retained root-level report repair artifact changed: {name}",
            )
            return payload
        self._retain_authority_file(
            self.root_descriptor, name, "root-level report repair artifact"
        )
        return self._read_root_successor(name)

    def _snapshot(
        self,
    ) -> tuple[frozenset[str], dict[str, bytes], tuple[object, ...] | None]:
        self._require_root()
        journal_names = {
            name
            for name in os.listdir(self.descriptor)
            if name.startswith("REPORT_REPAIR_")
            or name.startswith("CALLING_REPORT_REPAIR_")
        }
        require(
            not {name for name in journal_names if "_0002_" in name},
            "attempt2 publication artifacts must be root-level",
        )
        all_root_names = set(os.listdir(self.root_descriptor))
        root_repair_names = {
            name
            for name in all_root_names
            if _repair_root_name_is_reserved(name)
        }
        require(
            all(_repair_root_artifact_name_is_allowed(name) for name in root_repair_names),
            "repair publication root contains an unknown attempt2 artifact",
        )
        root_names = {
            name
            for name in root_repair_names
            if _repair_journal_artifact_name_is_allowed(name)
        }
        stages = {
            name
            for name in set(os.listdir(self.descriptor))
            | set(os.listdir(self.root_descriptor))
            if name.startswith(".") and name.endswith(".seal.tmp")
        }
        require(not stages, "attempt2 local staging is permanent fail-stop evidence")
        names = frozenset(journal_names | root_names)
        publication_names = {
            name
            for name in os.listdir(self.root_descriptor)
            if name.startswith(PUBLICATION_ARCHIVE_PREFIX)
            and name.endswith(PUBLICATION_ARCHIVE_SUFFIX)
        }
        require(
            len(publication_names) <= 1
            and (
                "REPORT_REPAIR_0002_COMPLETED.json" not in names
                or len(publication_names) == 1
            ),
            "completed report repair evidence lacks its publication archive",
        )
        payloads = {
            name: (
                self._read_root_successor(name)
                if "_0002_" in name
                else self._read(name)
            )
            for name in sorted(names)
        }
        self._require_submission_and_sources()
        self._require_root()
        return names, payloads, None

    def revalidate(self) -> None:
        self._require_submission_and_sources()
        names, payloads, stage_state = self._snapshot()
        require(
            names == self.names
            and payloads == self.payloads
            and stage_state == self.stage_state,
            "report repair publication journal changed",
        )

    def create_root_file_from_fd(
        self,
        path: Path,
        source_fd: int,
        *,
        size: int,
        digest: str,
        label: str,
    ) -> int:
        """Directly claim a root file and retain its creator descriptor."""

        require(
            path.parent == self.submission_root
            and path.name not in {"", ".", ".."}
            and type(size) is int
            and size > 0
            and sha256_string(digest)
            and stat.S_ISREG(os.fstat(source_fd).st_mode),
            f"{label} source/target differs",
        )
        self.revalidate()
        require(
            path.name not in set(os.listdir(self.root_descriptor)),
            f"{label} target already exists",
        )
        descriptor = -1
        try:
            descriptor = os.open(
                path.name,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=self.root_descriptor,
            )
            created = os.fstat(descriptor)
            named = os.stat(
                path.name,
                dir_fd=self.root_descriptor,
                follow_symlinks=False,
            )
            require(
                stat.S_ISREG(created.st_mode)
                and _file_identity(created) == _file_identity(named)
                and created.st_uid == named.st_uid == os.getuid()
                and created.st_nlink == named.st_nlink == 1
                and stat.S_IMODE(created.st_mode)
                == stat.S_IMODE(named.st_mode)
                == 0o600,
                f"{label} creation differs",
            )
            offset = 0
            while offset < size:
                block = os.pread(source_fd, min(16 * 1024 * 1024, size - offset), offset)
                require(block, f"{label} source is truncated")
                view = memoryview(block)
                while view:
                    written = os.write(descriptor, view)
                    require(written > 0, f"short {label} write")
                    view = view[written:]
                offset += len(block)
            require(
                not os.pread(source_fd, 1, size),
                f"{label} source has trailing bytes",
            )
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            observed_digest, sealed = _stable_open_fd_sha256(descriptor, label)
            rebound = os.stat(
                path.name,
                dir_fd=self.root_descriptor,
                follow_symlinks=False,
            )
            require(
                observed_digest == digest
                and sealed.st_size == size
                and _file_identity(sealed) == _file_identity(rebound)
                and sealed.st_uid == os.getuid()
                and sealed.st_nlink == 1
                and stat.S_IMODE(sealed.st_mode) == 0o444,
                f"{label} seal differs",
            )
            os.fsync(self.root_descriptor)
            expected_reserved = self.root_reserved | {path.name}
            require(
                self._reserved_root_names() == expected_reserved
                and _file_identity(os.fstat(descriptor))
                == _file_identity(
                    os.stat(
                        path.name,
                        dir_fd=self.root_descriptor,
                        follow_symlinks=False,
                    )
                ),
                f"{label} changed after parent fsync",
            )
            self.root_reserved = expected_reserved
            self.stream_file_rows.append(
                (path.name, descriptor, _file_identity(sealed), digest, size)
            )
            descriptor = -1
            self._require_submission_and_sources()
            return self.stream_file_rows[-1][1]
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def create_root_file_bytes(
        self, path: Path, payload: bytes, *, label: str
    ) -> tuple[str, int]:
        require(isinstance(payload, bytes) and bool(payload), f"{label} bytes differ")
        memfd = os.memfd_create(
            "exp23-repair-direct-final",
            getattr(os, "MFD_CLOEXEC", 0),
        )
        try:
            view = memoryview(payload)
            while view:
                written = os.write(memfd, view)
                require(written > 0, f"short private {label} write")
                view = view[written:]
            digest = hashlib.sha256(payload).hexdigest()
            descriptor = self.create_root_file_from_fd(
                path,
                memfd,
                size=len(payload),
                digest=digest,
                label=label,
            )
            name, retained, identity, retained_digest, retained_size = (
                self.stream_file_rows.pop()
            )
            require(
                descriptor == retained
                and name == path.name
                and retained_digest == digest
                and retained_size == len(payload),
                f"{label} retained binding differs",
            )
            self.source_file_rows.append(
                (self.root_descriptor, name, retained, identity, payload)
            )
            return digest, len(payload)
        finally:
            os.close(memfd)

    def advance_direct_report_directory(self) -> None:
        """Admit only creation of the final ``report`` directory."""

        opened_root = os.fstat(self.root_descriptor)
        named_root = self.submission_root.lstat()
        current_reserved = self._reserved_root_names()
        report_root = self.submission_root / "report"
        report_info = report_root.lstat()
        require(
            "report" not in self.root_reserved
            and current_reserved == self.root_reserved | {"report"}
            and (opened_root.st_dev, opened_root.st_ino)
            == (named_root.st_dev, named_root.st_ino)
            == (self.root_identity.st_dev, self.root_identity.st_ino)
            and exact_json_equal(self.locks.bindings(), self.lock_bindings)
            and stat.S_ISDIR(report_info.st_mode)
            and report_info.st_uid == os.getuid()
            and report_info.st_nlink == 2
            and stat.S_IMODE(report_info.st_mode) == 0o700,
            "direct-final report directory append differs",
        )
        self.root_reserved = current_reserved
        self.revalidate()

    def retain_direct_report_tree(self, report_root: Path) -> None:
        """Retain the completed quartet before the publisher releases its FDs."""

        require(
            report_root == self.submission_root / "report"
            and "report" in self.root_reserved
            and not any(path == report_root for path, *_rest in self.source_rows),
            "direct-final report retention authority differs",
        )
        self.revalidate()
        self._capture_authority_tree(report_root, "direct-final repaired report")
        self.revalidate()

    def _legacy_seal_completed(
        self, path: Path, value: Mapping[str, Any]
    ) -> str:
        """Append COMPLETED while retaining the exact pre-append FD graph."""

        completed_name = "REPORT_REPAIR_0002_COMPLETED.json"
        stage_name = f".{completed_name}.seal.tmp"
        require(
            path == self.root / completed_name
            and self.allow_completed_stage,
            "completed report repair append target differs",
        )
        self.revalidate()
        before_all_names = set(os.listdir(self.descriptor))
        before_root = os.fstat(self.descriptor)
        payload = (
            json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        digest = seal_json(path, value)
        require(
            hashlib.sha256(payload).hexdigest() == digest,
            "completed report repair append digest differs",
        )
        self._require_submission_and_sources()
        expected_names = (before_all_names - {stage_name}) | {completed_name}
        opened_root = os.fstat(self.descriptor)
        named_root = self.root.lstat()
        require(
            (opened_root.st_dev, opened_root.st_ino)
            == (named_root.st_dev, named_root.st_ino)
            == (before_root.st_dev, before_root.st_ino)
            and opened_root.st_uid
            == named_root.st_uid
            == before_root.st_uid
            == os.getuid()
            and opened_root.st_gid == named_root.st_gid == before_root.st_gid
            and opened_root.st_nlink
            == named_root.st_nlink
            == before_root.st_nlink
            and stat.S_IMODE(opened_root.st_mode)
            == stat.S_IMODE(named_root.st_mode)
            == stat.S_IMODE(before_root.st_mode)
            == 0o700
            and set(os.listdir(self.descriptor)) == expected_names,
            "completed report repair append namespace differs",
        )
        # Directory timestamps/size may change only because seal_json linked or
        # unlinked this exact successor stage.  Rebase those mutable fields only
        # after proving the retained descriptor is still the named directory and
        # the namespace changed by precisely that append.
        self.identity = opened_root
        target = os.stat(
            completed_name, dir_fd=self.descriptor, follow_symlinks=False
        )
        require(
            stat.S_ISREG(target.st_mode)
            and target.st_uid == os.getuid()
            and target.st_nlink == 1
            and stat.S_IMODE(target.st_mode) == 0o444
            and target.st_size == len(payload),
            "completed report repair append identity differs",
        )
        retained_target_fd: int | None = None
        retained_rows: dict[str, tuple[int, tuple[int, ...], bytes]] = {}
        for name, (descriptor, identity, retained_payload) in (
            self.journal_file_rows.items()
        ):
            if name in {completed_name, stage_name}:
                opened = os.fstat(descriptor)
                require(
                    (opened.st_dev, opened.st_ino)
                    == (target.st_dev, target.st_ino)
                    and opened.st_uid == target.st_uid == os.getuid()
                    and opened.st_nlink == target.st_nlink == 1
                    and stat.S_IMODE(opened.st_mode)
                    == stat.S_IMODE(target.st_mode)
                    == 0o444
                    and self._read_fd(descriptor) == payload,
                    "completed report repair retained stage changed",
                )
                if retained_target_fd is None:
                    retained_target_fd = descriptor
                else:
                    os.close(descriptor)
                continue
            named = os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
            opened = os.fstat(descriptor)
            require(
                _file_identity(opened) == _file_identity(named) == identity
                and self._read_fd(descriptor) == retained_payload
                and _file_identity(os.fstat(descriptor)) == identity,
                f"report repair predecessor changed after completion: {name}",
            )
            retained_rows[name] = (descriptor, identity, retained_payload)
        if retained_target_fd is None:
            retained_target_fd = os.open(
                completed_name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=self.descriptor,
            )
        opened_target = os.fstat(retained_target_fd)
        require(
            _file_identity(opened_target) == _file_identity(target)
            and self._read_fd(retained_target_fd) == payload,
            "completed report repair appended bytes differ",
        )
        retained_rows[completed_name] = (
            retained_target_fd,
            _file_identity(opened_target),
            payload,
        )
        self.journal_file_rows = retained_rows
        self.names = frozenset(set(self.names) | {completed_name})
        self.payloads = {**self.payloads, completed_name: payload}
        self.stage_state = None
        self.revalidate()
        return digest

    def seal_completed(
        self, path: Path, value: Mapping[str, Any]
    ) -> str:
        """Directly append root-level COMPLETED and retain the creator FD."""

        completed_name = "REPORT_REPAIR_0002_COMPLETED.json"
        require(
            len(self.scientific_runs) == 20,
            "completed report repair scientific run generation differs",
        )
        for run_root in self.scientific_runs:
            self._revalidate_scientific_run(run_root, full_hash=True)
        require(
            path == self.submission_root / completed_name
            and self.allow_completed_stage
            and len(
                {
                    name
                    for name in os.listdir(self.root_descriptor)
                    if name.startswith(PUBLICATION_ARCHIVE_PREFIX)
                    and name.endswith(PUBLICATION_ARCHIVE_SUFFIX)
                }
            )
            == 1,
            "completed report repair append target/archive differs",
        )
        payload = (
            json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        digest, size = self.create_root_file_bytes(
            path, payload, label="completed report repair evidence"
        )
        require(size == len(payload), "completed report repair size differs")
        names, payloads, stage_state = self._snapshot()
        require(
            names == self.names | {completed_name}
            and payloads[completed_name] == payload
            and stage_state is None,
            "completed report repair successor graph differs",
        )
        self.names = names
        self.payloads = payloads
        self.stage_state = None
        self.revalidate()
        for run_root in self.scientific_runs:
            self._revalidate_scientific_run(run_root, full_hash=True)
        return digest

    def close(self) -> None:
        for retained in reversed(list(self.scientific_runs.values())):
            for descriptor, *_rest in retained["files"].values():
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            for descriptor, *_rest in retained["symlinks"].values():
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            for descriptor, *_rest in reversed(
                list(retained["directories"].values())
            ):
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        self.scientific_runs.clear()
        for _name, descriptor, _identity, _digest, _size in reversed(
            self.stream_file_rows
        ):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.stream_file_rows.clear()
        for descriptor, _identity, _payload in reversed(
            list(self.journal_file_rows.values())
        ):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.journal_file_rows.clear()
        for _parent_fd, _name, descriptor, _identity, _payload in reversed(
            self.source_file_rows
        ):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.source_file_rows.clear()
        if self.execution_source_archive is not None:
            try:
                os.close(self.execution_source_archive[0])
            except OSError:
                pass
            self.execution_source_archive = None
        for descriptor in reversed(self.source_descriptors):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.source_descriptors.clear()
        for descriptor in (getattr(self, "descriptor", -1), self.root_descriptor):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


_ACTIVE_REPAIR_PUBLICATION_BINDING: contextvars.ContextVar[
    _RepairPublicationPhaseBinding | None
] = contextvars.ContextVar("active_exp23_repair_publication_binding", default=None)


@contextlib.contextmanager
def _active_repair_publication_binding(
    binding: _RepairPublicationPhaseBinding,
) -> Any:
    inherited = _ACTIVE_REPAIR_PUBLICATION_BINDING.get()
    if inherited is not None:
        require(inherited is binding, "nested repair publication binding differs")
        binding.revalidate()
        yield binding
        binding.revalidate()
        return
    token = _ACTIVE_REPAIR_PUBLICATION_BINDING.set(binding)
    try:
        binding.revalidate()
        yield binding
        binding.revalidate()
    finally:
        _ACTIVE_REPAIR_PUBLICATION_BINDING.reset(token)


def _retained_repair_regular(
    phase_binding: _RepairPublicationPhaseBinding,
    path: Path,
    label: str,
    *,
    expected_mode: int = 0o444,
) -> tuple[bytes, str, os.stat_result]:
    """Read one authority artifact only through its retained descriptor."""

    retained = phase_binding.retained_regular(path)
    require(retained is not None, f"{label} is absent from retained authority")
    payload, info = retained
    digest = hashlib.sha256(payload).hexdigest()
    require(
        stat.S_ISREG(info.st_mode)
        and info.st_uid == os.getuid()
        and info.st_nlink == 1
        and stat.S_IMODE(info.st_mode) == expected_mode
        and info.st_size == len(payload),
        f"{label} retained identity differs",
    )
    return payload, digest, info


def _retained_repair_json(
    phase_binding: _RepairPublicationPhaseBinding,
    path: Path,
    label: str,
    *,
    expected_mode: int = 0o444,
) -> tuple[dict[str, Any], str, os.stat_result]:
    payload, digest, info = _retained_repair_regular(
        phase_binding, path, label, expected_mode=expected_mode
    )
    return _decode_json_object(path, payload), digest, info


def _retained_receipt_file_map(
    submission_root: Path,
    phase_binding: _RepairPublicationPhaseBinding,
) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for index in range(20):
        relative = Path("tasks") / f"cell-{index:02d}" / "WORKER_COMPLETE.json"
        payload, digest, info = _retained_repair_regular(
            phase_binding,
            submission_root / relative,
            f"cell{index} retained repair worker receipt",
        )
        _decode_json_object(submission_root / relative, payload)
        files[relative.as_posix()] = {
            "mode": 0o444,
            "size": info.st_size,
            "sha256": digest,
        }
    return {"schema_version": 1, "files": files}


def _validated_publication_archive_bound(
    submission_root: Path,
    submission_sha256: str,
    phase_binding: _RepairPublicationPhaseBinding,
    expected_publication_authority: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Mapping[str, Any]]:
    """Authenticate the logical quartet from the retained archive descriptor."""

    require(
        len(phase_binding.stream_file_rows) == 1,
        "completed publication archive generation differs",
    )
    archive_name, descriptor, identity, archive_sha256, archive_size = (
        phase_binding.stream_file_rows[0]
    )
    archive_path = submission_root / archive_name
    require(
        archive_name
        == f"{PUBLICATION_ARCHIVE_PREFIX}{archive_sha256}{PUBLICATION_ARCHIVE_SUFFIX}",
        "completed publication archive name differs",
    )

    def exact(offset: int, size: int) -> bytes:
        pieces: list[bytes] = []
        cursor = offset
        remaining = size
        while remaining:
            block = os.pread(descriptor, min(16 * 1024 * 1024, remaining), cursor)
            require(block, "completed publication archive is truncated")
            pieces.append(block)
            cursor += len(block)
            remaining -= len(block)
        return b"".join(pieces)

    opened = os.fstat(descriptor)
    named = archive_path.lstat()
    require(
        _file_identity(opened) == _file_identity(named) == identity
        and opened.st_uid == named.st_uid == os.getuid()
        and opened.st_nlink == named.st_nlink == 1
        and stat.S_IMODE(opened.st_mode) == stat.S_IMODE(named.st_mode) == 0o444
        and opened.st_size == named.st_size == archive_size
        and _stable_open_fd_sha256(
            descriptor, "completed retained publication archive"
        )[0]
        == archive_sha256,
        "completed publication archive retained identity differs",
    )
    offset = 0
    require(
        exact(offset, len(PUBLICATION_ARCHIVE_MAGIC)) == PUBLICATION_ARCHIVE_MAGIC,
        "completed publication archive magic differs",
    )
    offset += len(PUBLICATION_ARCHIVE_MAGIC)
    header_size = int.from_bytes(exact(offset, 8), "big")
    offset += 8
    require(
        0 < header_size <= 1 << 20,
        "completed publication archive header size differs",
    )
    header_payload = exact(offset, header_size)
    offset += header_size
    try:
        header = json.loads(
            header_payload.decode("ascii"), object_pairs_hook=_pairs(archive_path)
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError(
            f"completed publication archive header differs: {exc}"
        ) from exc
    require(
        header_payload
        == json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        and set(header)
        == {
            "archive_kind",
            "schema_version",
            "campaign_id",
            "submission_sha256",
            "entry_order",
            "entries",
            "report_commit_sha256",
            "report_commit_value_sha256",
        }
        and header.get("archive_kind") == PUBLICATION_ARCHIVE_KIND
        and type(header.get("schema_version")) is int
        and header.get("schema_version") == 2
        and header.get("campaign_id") == CAMPAIGN_ID
        and header.get("submission_sha256") == submission_sha256
        and header.get("entry_order") == list(PUBLICATION_ARCHIVE_ENTRY_ORDER)
        and isinstance(header.get("entries"), list)
        and len(header["entries"]) == len(PUBLICATION_ARCHIVE_ENTRY_ORDER),
        "completed publication archive header differs",
    )
    captured: dict[str, bytes] = {}
    rows: dict[str, Mapping[str, Any]] = {}
    for kind, row in zip(
        PUBLICATION_ARCHIVE_ENTRY_ORDER, header["entries"], strict=True
    ):
        require(
            isinstance(row, Mapping)
            and set(row) == {"kind", "name", "size", "sha256", "logical_sha256"}
            and row.get("kind") == kind
            and isinstance(row.get("name"), str)
            and Path(row["name"]).name == row["name"]
            and type(row.get("size")) is int
            and row["size"] > 0
            and sha256_string(row.get("sha256"))
            and sha256_string(row.get("logical_sha256")),
            f"completed publication archive row differs: {kind}",
        )
        name_size = int.from_bytes(exact(offset, 8), "big")
        offset += 8
        require(0 < name_size <= 256, "publication archive entry name size differs")
        try:
            frame_name = exact(offset, name_size).decode("ascii")
        except UnicodeDecodeError as exc:
            raise ReportError("publication archive entry name is not ASCII") from exc
        offset += name_size
        frame_size = int.from_bytes(exact(offset, 8), "big")
        offset += 8
        require(
            frame_name == row["name"] and frame_size == row["size"],
            f"completed publication archive frame differs: {kind}",
        )
        digest = hashlib.sha256()
        pieces: list[bytes] | None = [] if kind != "report_bundle" else None
        remaining = frame_size
        while remaining:
            block_size = min(16 * 1024 * 1024, remaining)
            block = exact(offset, block_size)
            digest.update(block)
            if pieces is not None:
                pieces.append(block)
            offset += block_size
            remaining -= block_size
        require(
            digest.hexdigest() == row["sha256"],
            f"completed publication archive payload hash differs: {kind}",
        )
        rows[kind] = row
        if pieces is not None:
            captured[kind] = b"".join(pieces)
    require(offset == archive_size, "completed publication archive trailing bytes differ")
    bundle_row = rows["report_bundle"]
    gate_row = rows["gate_decision"]
    provenance_row = rows["provenance"]
    commit_row = rows["report_commit"]
    require(
        bundle_row["name"]
        == f"REPORT_BUNDLE.{EXPECTED_REPAIR_REASSEMBLY['report_bundle_sha256']}.json"
        and bundle_row["size"]
        == EXPECTED_REPAIR_REASSEMBLY["report_bundle_file_size"]
        and bundle_row["sha256"]
        == EXPECTED_REPAIR_REASSEMBLY["report_bundle_file_sha256"]
        and bundle_row["logical_sha256"]
        == EXPECTED_REPAIR_REASSEMBLY["report_bundle_sha256"]
        and gate_row["name"]
        == f"GATE_DECISION.{EXPECTED_REPAIR_REASSEMBLY['gate_sha256']}.json"
        and gate_row["size"]
        == EXPECTED_REPAIR_REASSEMBLY["gate_decision_file_size"]
        and gate_row["sha256"]
        == EXPECTED_REPAIR_REASSEMBLY["gate_decision_file_sha256"]
        and gate_row["logical_sha256"] == EXPECTED_REPAIR_REASSEMBLY["gate_sha256"]
        and commit_row["name"] == "REPORT_COMMIT.json",
        "completed publication archive scientific rows differ",
    )
    provenance = _decode_json_object(archive_path, captured["provenance"])
    commit = _decode_json_object(archive_path, captured["report_commit"])
    gate = _decode_json_object(archive_path, captured["gate_decision"])
    provenance_v1 = dict(provenance)
    provenance_v1.pop("publication_authority", None)
    provenance_v1["schema_version"] = 1
    require(
        captured["gate_decision"]
        == (
            json.dumps(gate, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        and stable_hash(gate) == EXPECTED_REPAIR_REASSEMBLY["gate_sha256"]
        and captured["provenance"]
        == (
            json.dumps(provenance, sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        and set(provenance) == REPORT_PROVENANCE_V1_KEYS | {"publication_authority"}
        and provenance.get("schema_version") == 2
        and provenance.get("submission_sha256") == submission_sha256
        and exact_json_equal(
            provenance.get("publication_authority"), expected_publication_authority
        )
        and set(provenance_v1) == REPORT_PROVENANCE_V1_KEYS
        and stable_hash(provenance_v1)
        == EXPECTED_REPAIR_REASSEMBLY["original_provenance_v1_sha256"]
        and len(
            (
                json.dumps(
                    provenance_v1, sort_keys=True, indent=2, allow_nan=False
                )
                + "\n"
            ).encode("utf-8")
        )
        == EXPECTED_REPAIR_REASSEMBLY["original_provenance_v1_file_size"]
        and hashlib.sha256(
            (
                json.dumps(
                    provenance_v1, sort_keys=True, indent=2, allow_nan=False
                )
                + "\n"
            ).encode("utf-8")
        ).hexdigest()
        == EXPECTED_REPAIR_REASSEMBLY["original_provenance_v1_file_sha256"],
        "completed publication archive provenance differs",
    )
    require(
        captured["report_commit"]
        == (
            json.dumps(commit, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        and set(commit) == REPORT_COMMIT_KEYS
        and commit.get("schema_version") == 1
        and commit.get("status") == EXPECTED_REPAIR_REASSEMBLY["status"]
        and commit.get("scientific_rejection") is True
        and commit.get("campaign_id") == CAMPAIGN_ID
        and commit.get("submission_sha256") == submission_sha256
        and commit.get("report_bundle") == bundle_row["name"]
        and commit.get("report_bundle_sha256") == bundle_row["logical_sha256"]
        and commit.get("report_bundle_file_sha256") == bundle_row["sha256"]
        and commit.get("gate_decision") == gate_row["name"]
        and commit.get("gate_sha256") == gate_row["logical_sha256"]
        and commit.get("gate_decision_file_sha256") == gate_row["sha256"]
        and commit.get("provenance") == provenance_row["name"]
        and commit.get("provenance_sha256") == provenance_row["logical_sha256"]
        and commit.get("provenance_file_sha256") == provenance_row["sha256"]
        and provenance_row["logical_sha256"] == stable_hash(provenance)
        and commit_row["sha256"] == hashlib.sha256(captured["report_commit"]).hexdigest()
        and commit_row["logical_sha256"] == stable_hash(commit)
        and header.get("report_commit_sha256") == commit_row["sha256"]
        and header.get("report_commit_value_sha256") == stable_hash(commit),
        "completed publication archive commit differs",
    )
    evidence = {
        "schema_version": 2,
        "archive_kind": PUBLICATION_ARCHIVE_KIND,
        "archive": archive_name,
        "archive_sha256": archive_sha256,
        "archive_size": archive_size,
        "header": header,
        "header_sha256": stable_hash(header),
        "descriptor": descriptor,
        "file_identity": _direct_final_file_identity(opened),
    }
    phase_binding.publication_archive_evidence = evidence
    phase_binding.revalidate()
    return commit, header, evidence


def _validate_completed_recovery_phase(
    submission_root: Path,
    submission_sha256: str,
    phase_binding: _RepairPublicationPhaseBinding,
    authorization: Mapping[str, Any],
    authorization_sha256: str,
    attempt: int,
) -> None:
    completed_name = "REPORT_REPAIR_0002_COMPLETED.json"
    completed_is_durable = completed_name in phase_binding.names
    if not completed_is_durable:
        return
    expected_publication_authority = _repair_publication_authority(
        authorization, authorization_sha256, attempt
    )
    commit, header, evidence = _validated_publication_archive_bound(
        submission_root,
        submission_sha256,
        phase_binding,
        expected_publication_authority,
    )
    completed_payload = phase_binding.payloads[completed_name]
    completed_value = _decode_json_object(
        submission_root / completed_name, completed_payload
    )
    require(
        completed_payload
        == (
            json.dumps(
                completed_value,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
        "completed report repair serialization differs",
    )
    commit_row = next(
        row for row in header["entries"] if row.get("kind") == "report_commit"
    )
    _validated_completed_value(
        completed_value,
        submission_sha256=submission_sha256,
        authorization=authorization,
        authorization_sha256=authorization_sha256,
        commit=commit,
        commit_sha256=str(commit_row["sha256"]),
        publication_authority=expected_publication_authority,
        publication_archive=evidence,
    )


def _validated_report_repair_authorization(
    submission_root: Path,
    submission_sha256: str,
    receipt: Mapping[str, Any],
    *,
    attempt: int,
    expected_raw_sha256: str,
    lock_bindings: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Authenticate the one append-only publication-only Launch8 repair."""

    active = _ACTIVE_REPAIR_PUBLICATION_BINDING.get()
    if active is not None:
        require(
            active.submission_root == submission_root,
            "active report repair authorization binding differs",
        )
        return _validated_report_repair_authorization_bound(
            submission_root,
            submission_sha256,
            receipt,
            attempt=attempt,
            expected_raw_sha256=expected_raw_sha256,
            lock_bindings=lock_bindings,
            phase_binding=active,
        )
    binding = _RepairPublicationPhaseBinding(
        submission_root, allow_completed_stage=True
    )
    try:
        return _validated_report_repair_authorization_bound(
            submission_root,
            submission_sha256,
            receipt,
            attempt=attempt,
            expected_raw_sha256=expected_raw_sha256,
            lock_bindings=lock_bindings,
            phase_binding=binding,
        )
    finally:
        binding.close()


def _validated_report_repair_authorization_bound(
    submission_root: Path,
    submission_sha256: str,
    receipt: Mapping[str, Any],
    *,
    attempt: int,
    expected_raw_sha256: str,
    lock_bindings: tuple[Mapping[str, Any], Mapping[str, Any]] | None,
    phase_binding: _RepairPublicationPhaseBinding,
) -> dict[str, Any]:
    """Validate repair authority while retaining the journal directory FD."""

    require(type(attempt) is int and attempt == 2, "report repair attempt differs")
    require(sha256_string(expected_raw_sha256), "report repair authorization SHA256 differs")
    phase_binding.revalidate()
    path = _repair_authorization_path(submission_root, attempt)
    value, authorization_sha256, info = _retained_repair_json(
        phase_binding, path, "report repair authorization"
    )
    require(
        stat.S_IMODE(info.st_mode) == 0o444
        and info.st_uid == os.getuid()
        and info.st_nlink == 1
        and authorization_sha256 == expected_raw_sha256,
        "report repair authorization identity/hash differs",
    )
    historical_ids = {
        receipt.get("wave0_array_job_id"),
        receipt.get("wave1_array_job_id"),
        receipt.get("report_job_id"),
        "33285485",
        "33285486",
        "33295657",
        "33295659",
        "33295661",
        EXPECTED_ATTEMPT1_JOB_ID,
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
        and REPORT_REPAIR_JOB_ID_RE.fullmatch(value["repair_report_job_id"])
        is not None
        and value["repair_report_job_id"] not in historical_ids
        and value.get("repair_job_name")
        == f"exp23-launch8-{submission_sha256[:16]}-report-repair-0002"
        and value.get("scheduler_comment")
        == f"treewm-exp23-report-repair:{submission_sha256}:0002"
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
    contract, _contract_sha256, _contract_info = _retained_repair_json(
        phase_binding,
        submission_root / "SUBMISSION_CONTRACT.json",
        "retained submission contract",
    )
    phase_binding.validate_exact_snapshot_authority(contract)
    expected_scheduler_environment = _repair_current_scheduler_environment(
        contract
    )
    require(
        value.get("snapshot_root") == contract.get("snapshot_root")
        and value.get("snapshot_inventory_sha256")
        == contract.get("snapshot_inventory_sha256")
        and value.get("original_package_protocol_sha256")
        == contract.get("package_protocol_sha256")
        and value.get("predecessor_failure_evidence")
        == "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
        and sha256_string(value.get("predecessor_failure_evidence_sha256")),
        "report repair original snapshot binding differs",
    )

    failure_name = value.get("original_failure_evidence")
    require(
        failure_name == "journal/REPORT_REPAIR_0001_ORIGINAL_FAILURE.json",
        "report repair failure-evidence name differs",
    )
    failure_path = submission_root / str(failure_name)
    failure, failure_sha256, failure_info = _retained_repair_json(
        phase_binding,
        failure_path,
        "report repair original failure evidence",
    )
    require(
        stat.S_IMODE(failure_info.st_mode) == 0o444
        and failure_info.st_uid == os.getuid()
        and failure_info.st_nlink == 1
        and failure_sha256 == value.get("original_failure_evidence_sha256"),
        "report repair original failure evidence differs",
    )
    receipt_map = _retained_receipt_file_map(submission_root, phase_binding)
    predecessor = _validated_attempt2_predecessor(
        submission_root,
        submission_sha256,
        contract,
        receipt_map,
        expected_raw_sha256=value["predecessor_failure_evidence_sha256"],
        phase_binding=phase_binding,
    )
    if lock_bindings is not None:
        require(
            exact_json_equal(predecessor.get("transaction_lock"), lock_bindings[0])
            and exact_json_equal(
                predecessor.get("report_cancel_lock"), lock_bindings[1]
            ),
            "attempt2 predecessor lock authority differs",
        )
    require(
        set(failure) == REPORT_REPAIR_FAILURE_KEYS
        and failure.get("schema_version") == 1
        and failure.get("status") == "original_report_terminal_failure_authenticated"
        and failure.get("submission_sha256") == submission_sha256
        and failure.get("attempt") == 1
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
        expected_environment=expected_scheduler_environment,
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

    active_source, active_source_files = _validated_active_source_archive(
        submission_root, phase_binding
    )
    source_root = Path(str(value.get("repair_source_root")))
    _source_archive_payload, _source_archive_sha, source_root_info = (
        _retained_repair_regular(
            phase_binding, source_root, "report repair source archive"
        )
    )
    require(
        source_root == submission_root / SOURCE_ARCHIVE_NAME
        and stat.S_ISREG(source_root_info.st_mode)
        and source_root_info.st_uid == os.getuid()
        and source_root_info.st_nlink == 1
        and stat.S_IMODE(source_root_info.st_mode) == 0o444
        and exact_json_equal(active_source, {
            key: value[key] for key in REPORT_REPAIR_SOURCE_ARCHIVE_EVIDENCE_KEYS
        }),
        "report repair source archive differs",
    )
    source_files = value.get("repair_source_files")
    require(
        isinstance(source_files, Mapping)
        and set(source_files)
        == {"report.py", "report_repair.py", "report_repair.slurm", "protocol.sha256"}
        and value.get("repair_source_files_sha256") == stable_hash(source_files)
        and git_commit_string(value.get("repair_source_commit"))
        and sha256_string(value.get("repair_package_protocol_sha256"))
        and value.get("repair_source_installation_method")
        == SOURCE_ARCHIVE_INSTALL_METHOD
        and value.get("report_publication_installation_method")
        == PUBLICATION_ARCHIVE_INSTALL_METHOD,
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
        require(
            len(active_source_files[name]) == expected["size"]
            and hashlib.sha256(active_source_files[name]).hexdigest()
            == expected["sha256"],
            f"report repair source bytes differ: {name}",
        )
    require(
        set(active_source_files) == set(source_files),
        "report repair sealed source coverage differs",
    )
    source_authority = {
        key: active_source[key] for key in REPORT_REPAIR_SOURCE_AUTHORITY_V2_KEYS
    }
    expected_source_authority = {
        "schema_version": 2,
        "repair_source_commit": value["repair_source_commit"],
        "repair_package_protocol_sha256": value[
            "repair_package_protocol_sha256"
        ],
        "repair_source_files": value["repair_source_files"],
        "repair_source_files_sha256": value[
            "repair_source_files_sha256"
        ],
        "repair_source_installation_method": value[
            "repair_source_installation_method"
        ],
    }
    expected_source_authority_payload = (
        json.dumps(
            expected_source_authority,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    require(
        set(source_authority) == REPORT_REPAIR_SOURCE_AUTHORITY_V2_KEYS
        and exact_json_equal(source_authority, expected_source_authority),
        "report repair sealed source authority differs",
    )
    require(
        str(__file__) == f"{source_root}::report.py",
        "active report repair publisher is outside its sealed source",
    )
    protocol_payload = active_source_files["protocol.sha256"]
    require(
        protocol_payload == f"{value['repair_package_protocol_sha256']}\n".encode("ascii"),
        "report repair source protocol differs",
    )

    calling_path = submission_root / "CALLING_REPORT_REPAIR_0002_SUBMIT.json"
    submitted_path = submission_root / "REPORT_REPAIR_0002_SUBMITTED.json"
    calling, calling_sha256, calling_info = _retained_repair_json(
        phase_binding, calling_path, "report repair submit calling"
    )
    submitted, submitted_sha256, submitted_info = _retained_repair_json(
        phase_binding, submitted_path, "report repair submitted evidence"
    )
    require(
        stat.S_IMODE(calling_info.st_mode) == 0o444
        and calling_info.st_uid == os.getuid()
        and calling_info.st_nlink == 1
        and stat.S_IMODE(submitted_info.st_mode) == 0o444
        and submitted_info.st_uid == os.getuid()
        and submitted_info.st_nlink == 1
        and calling_sha256 == value.get("submit_calling_sha256")
        and value.get("submitted_evidence")
        == "REPORT_REPAIR_0002_SUBMITTED.json"
        and submitted_sha256 == value.get("submitted_evidence_sha256"),
        "report repair submit evidence hashes differ",
    )
    scheduler_source_input = calling.get("scheduler_source_archive_input")
    require(
        isinstance(scheduler_source_input, Mapping)
        and set(scheduler_source_input)
        == {
            "schema_version",
            "transport",
            "descriptor",
            "argument",
            "source_archive",
            "sha256",
            "size",
            "file_identity",
        }
        and scheduler_source_input.get("schema_version") == 1
        and scheduler_source_input.get("transport") == "inherited_proc_fd"
        and type(scheduler_source_input.get("descriptor")) is int
        and scheduler_source_input["descriptor"] >= 0
        and scheduler_source_input.get("argument")
        == f"/proc/self/fd/{scheduler_source_input['descriptor']}"
        and scheduler_source_input.get("source_archive") == str(source_root)
        and scheduler_source_input.get("sha256")
        == value["repair_source_archive_sha256"]
        and scheduler_source_input.get("size")
        == value["repair_source_archive_size"]
        and exact_json_equal(
            _validated_direct_final_file_identity(
                scheduler_source_input.get("file_identity"),
                size=int(value["repair_source_archive_size"]),
                label="report repair scheduler source identity",
            ),
            _direct_final_file_identity(source_root_info),
        ),
        "report repair scheduler source archive input differs",
    )
    expected_submit_command = [
        "/usr/local/bin/sbatch",
        "--parsable",
        "--hold",
        "--no-requeue",
        "--export=NONE",
        f"--job-name={value['repair_job_name']}",
        f"--comment={value['scheduler_comment']}",
        f"--output={submission_root / 'logs/report-repair-0002-%j.out'}",
        str(scheduler_source_input["argument"]),
        str(value["snapshot_root"]),
        str(submission_root),
        submission_sha256,
        "2",
        str(source_root),
        str(value["repair_source_archive_sha256"]),
        str(value["repair_source_archive_size"]),
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
        and calling.get("predecessor_failure_evidence")
        == "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
        and calling.get("predecessor_failure_evidence_sha256")
        == value["predecessor_failure_evidence_sha256"]
        and calling.get("repair_source_root") == str(source_root)
        and calling.get("repair_source_commit") == value["repair_source_commit"]
        and calling.get("repair_package_protocol_sha256")
        == value["repair_package_protocol_sha256"]
        and exact_json_equal(
            calling.get("repair_source_files"), value["repair_source_files"]
        )
        and calling.get("repair_source_files_sha256")
        == value["repair_source_files_sha256"]
        and calling.get("repair_source_installation_method")
        == value["repair_source_installation_method"]
        and calling.get("repair_source_archive")
        == value["repair_source_archive"]
        == str(source_root)
        and calling.get("repair_source_archive_sha256")
        == value["repair_source_archive_sha256"]
        and calling.get("repair_source_archive_size")
        == value["repair_source_archive_size"]
        and calling.get("repair_source_archive_format")
        == value["repair_source_archive_format"]
        == SOURCE_ARCHIVE_KIND
        and exact_json_equal(
            _validated_direct_final_file_identity(
                calling.get("repair_source_archive_file_identity"),
                size=int(value["repair_source_archive_size"]),
                label="report repair source creation identity",
            ),
            _direct_final_file_identity(source_root_info),
        )
        and calling.get("command") == expected_submit_command
        and isinstance(calling.get("scheduler_environment"), Mapping)
        and isinstance(calling.get("transaction_lock"), Mapping)
        and isinstance(calling.get("report_cancel_lock"), Mapping)
        and isinstance(calling.get("called_at_utc"), str)
        and bool(calling["called_at_utc"])
        and set(submitted) == REPORT_REPAIR_SUBMITTED_KEYS
        and type(submitted.get("schema_version")) is int
        and submitted.get("schema_version") == 1
        and submitted.get("status") == "held_report_repair_submitted"
        and submitted.get("campaign_id") == CAMPAIGN_ID
        and submitted.get("submission_sha256") == submission_sha256
        and type(submitted.get("attempt")) is int
        and submitted.get("attempt") == attempt
        and submitted.get("submit_calling_sha256") == value["submit_calling_sha256"]
        and submitted.get("repair_report_job_id") == value["repair_report_job_id"]
        and isinstance(submitted.get("submission_evidence"), Mapping)
        and isinstance(submitted.get("accepted_at_utc"), str)
        and bool(submitted["accepted_at_utc"]),
        "report repair submit evidence semantics differ",
    )
    if lock_bindings is not None:
        require(
            exact_json_equal(calling["transaction_lock"], lock_bindings[0])
            and exact_json_equal(calling["report_cancel_lock"], lock_bindings[1]),
            "report repair submit lock authority differs",
        )
    pre_submit_census = _validated_repair_census(
        calling.get("scheduler_pre_submit_census"),
        submission_sha256=submission_sha256,
        expected_environment=expected_scheduler_environment,
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
    submission_evidence = submitted["submission_evidence"]
    if submission_evidence.get("mode") == "direct_sbatch_response":
        require(
            set(submission_evidence) == {"mode", "raw"},
            "report repair direct submission evidence shape differs",
        )
        direct_submission = _validated_repair_command_evidence(
            submission_evidence["raw"],
            expected_argv=expected_submit_command,
            label="report repair direct submission",
        )
        direct_stdout = _repair_stream_bytes(
            direct_submission["stdout"], "report repair direct submission stdout"
        )
        require(
            exact_json_equal(
                direct_submission["environment"], calling["scheduler_environment"]
            )
            and direct_submission["returncode"] == 0
            and _repair_stream_bytes(
                direct_submission["stderr"], "report repair direct submission stderr"
            )
            == b""
            and _validated_repair_sbatch_stdout(direct_stdout)["job_id"]
            == value["repair_report_job_id"],
            "report repair direct submission evidence differs",
        )
    elif submission_evidence.get("mode") == "lost_response_census_adoption":
        require(
            set(submission_evidence) == {"mode", "census", "census_sha256"}
            and isinstance(submission_evidence.get("census"), Mapping)
            and submission_evidence.get("census_sha256")
            == stable_hash(submission_evidence["census"]),
            "report repair lost-response submission evidence shape differs",
        )
        adoption = _validated_repair_census(
            submission_evidence["census"],
            submission_sha256=submission_sha256,
            expected_environment=expected_scheduler_environment,
            label="report repair lost-response adoption",
        )
        adopted_rows = [
            row
            for row in adoption["settled_rows"]
            if row["job_name"] == value["repair_job_name"]
            and row["comment"] == value["scheduler_comment"]
        ]
        require(
            len(adoption["settled_rows"]) == 1
            and len(adopted_rows) == 1
            and adopted_rows[0]["job_id"] == value["repair_report_job_id"]
            and adopted_rows[0]["job_id"] not in historical_ids
            and adopted_rows[0]["state"] == "PENDING"
            and adopted_rows[0]["reason"] in {"JobHeldUser", "JobHeldAdmin"},
            "report repair lost-response adopted job was not exact and held",
        )
    else:
        raise ReportError("report repair submission evidence mode differs")
    authority_census = _validated_repair_census(
        value.get("scheduler_authority_census"),
        submission_sha256=submission_sha256,
        expected_environment=expected_scheduler_environment,
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
    job_control = value.get("scheduler_job_control_observation")
    require(
        isinstance(job_control, Mapping)
        and value.get("scheduler_job_control_observation_sha256")
        == stable_hash(job_control),
        "report repair scheduler job-control hash differs",
    )
    validated_job_control = _validated_repair_job_control(
        job_control,
        submission_root=submission_root,
        submission_sha256=submission_sha256,
        repair_authorization=value,
        authority_environment=authority_environment,
        contract=contract,
        scheduler_source_argument=str(scheduler_source_input["argument"]),
    )
    require(
        authority_census["captured_at_utc"]
        <= validated_job_control["captured_at_utc"]
        <= value["authorized_at_utc"],
        "report repair authorization observation order differs",
    )

    release_path = _repair_release_path(submission_root, attempt)
    release, release_sha256, release_info = _retained_repair_json(
        phase_binding, release_path, "report repair release result"
    )
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
            == f"CALLING_REPORT_REPAIR_0002_RELEASE_{index:04d}.json"
            and entry.get("result")
            == f"REPORT_REPAIR_0002_RELEASE_RESULT_{index:04d}.json"
            and sha256_string(entry.get("calling_sha256"))
            and sha256_string(entry.get("result_sha256")),
            "report repair release attempt chain differs",
        )
        release_calling_path = submission_root / str(entry["calling"])
        release_result_path = submission_root / str(entry["result"])
        release_calling, release_calling_sha256, release_calling_info = (
            _retained_repair_json(
                phase_binding,
                release_calling_path,
                f"report repair release calling {index}",
            )
        )
        release_result, release_result_sha256, release_result_info = (
            _retained_repair_json(
                phase_binding,
                release_result_path,
                f"report repair release result {index}",
            )
        )
        require(
            stat.S_IMODE(release_calling_info.st_mode) == 0o444
            and release_calling_info.st_uid == os.getuid()
            and release_calling_info.st_nlink == 1
            and stat.S_IMODE(release_result_info.st_mode) == 0o444
            and release_result_info.st_uid == os.getuid()
            and release_result_info.st_nlink == 1
            and release_calling_sha256 == entry["calling_sha256"]
            and release_result_sha256 == entry["result_sha256"],
            "report repair release attempt hashes differ",
        )
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
            and isinstance(
                release_calling.get("scheduler_pre_release_census"), Mapping
            )
            and release_calling.get("scheduler_pre_release_census_sha256")
            == stable_hash(release_calling["scheduler_pre_release_census"])
            and isinstance(
                release_calling.get(
                    "scheduler_pre_release_job_control_observation"
                ),
                Mapping,
            )
            and release_calling.get(
                "scheduler_pre_release_job_control_observation_sha256"
            )
            == stable_hash(
                release_calling[
                    "scheduler_pre_release_job_control_observation"
                ]
            )
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
        pre_release_census = _validated_repair_census(
            release_calling["scheduler_pre_release_census"],
            submission_sha256=submission_sha256,
            expected_environment=expected_scheduler_environment,
            label=f"report repair pre-release {index}",
        )
        pre_release_rows = [
            row
            for row in pre_release_census["settled_rows"]
            if row["job_name"] == value["repair_job_name"]
            and row["comment"] == value["scheduler_comment"]
        ]
        require(
            len(pre_release_census["settled_rows"]) == 1
            and len(pre_release_rows) == 1
            and pre_release_rows[0]["job_id"]
            == value["repair_report_job_id"]
            and pre_release_rows[0]["state"] == "PENDING"
            and pre_release_rows[0]["reason"]
            in {"JobHeldUser", "JobHeldAdmin"},
            "report repair pre-release census is not one exact held job",
        )
        pre_release_control = _validated_repair_job_control(
            release_calling[
                "scheduler_pre_release_job_control_observation"
            ],
            submission_root=submission_root,
            submission_sha256=submission_sha256,
            repair_authorization=value,
            authority_environment=expected_scheduler_environment,
            contract=contract,
            scheduler_source_argument=str(
                scheduler_source_input["argument"]
            ),
        )
        require(
            pre_release_census["captured_at_utc"]
            <= pre_release_control["captured_at_utc"]
            <= release_calling["called_at_utc"],
            "report repair pre-release authority order differs",
        )
        if lock_bindings is not None:
            require(
                exact_json_equal(
                    release_calling["transaction_lock"], lock_bindings[0]
                )
                and exact_json_equal(
                    release_calling["report_cancel_lock"], lock_bindings[1]
                ),
                "report repair release lock authority differs",
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
                expected_environment=expected_scheduler_environment,
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
    journal_names = set(phase_binding.names)
    actual_release_callings = {
        name
        for name in journal_names
        if name.startswith("CALLING_REPORT_REPAIR_0002_RELEASE_")
    }
    actual_release_results = {
        name
        for name in journal_names
        if name.startswith("REPORT_REPAIR_0002_RELEASE_RESULT_")
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
    expected_repair_journal = {
        *EXPECTED_ATTEMPT1_CHAIN_SHA256,
        "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json",
        "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json",
        "CALLING_REPORT_REPAIR_0002_SUBMIT.json",
        "REPORT_REPAIR_0002_SUBMITTED.json",
        "REPORT_REPAIR_0002_AUTHORIZED.json",
        "REPORT_REPAIR_0002_RELEASED.json",
        *expected_release_callings,
        *expected_release_results,
    }
    completed_name = "REPORT_REPAIR_0002_COMPLETED.json"
    completed_is_durable = completed_name in journal_names
    if completed_is_durable:
        expected_repair_journal.add(completed_name)
    actual_repair_journal = {
        name
        for name in journal_names
        if name.startswith("REPORT_REPAIR_")
        or name.startswith("CALLING_REPORT_REPAIR_")
    }
    journal_seal_staging = {
        name
        for name in journal_names
        if name.startswith(".") and name.endswith(".seal.tmp")
    }
    require(
        actual_repair_journal == expected_repair_journal
        and not journal_seal_staging,
        "report repair publication journal namespace differs",
    )
    post_release = _validated_repair_census(
        release["post_release_census"],
        submission_sha256=submission_sha256,
        expected_environment=expected_scheduler_environment,
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
        and _repair_first_start_restart_count_is_valid(restart_count)
        and "SLURM_ARRAY_JOB_ID" not in os.environ
        and "SLURM_ARRAY_TASK_ID" not in os.environ,
        "active report repair scheduler identity/restart differs",
    )
    result = dict(value)
    result["_validated_release_sha256"] = release_sha256
    _validate_completed_recovery_phase(
        submission_root,
        submission_sha256,
        phase_binding,
        result,
        expected_raw_sha256,
        attempt,
    )
    phase_binding.revalidate()
    return result


def _repair_publication_authority(
    repair: Mapping[str, Any], authorization_sha256: str, attempt: int
) -> dict[str, Any]:
    value = {
        "schema_version": 2,
        "status": "authorized_terminal_report_repair",
        "attempt": attempt,
        "authorization": f"REPORT_REPAIR_{attempt:04d}_AUTHORIZED.json",
        "authorization_sha256": authorization_sha256,
        "release": f"REPORT_REPAIR_{attempt:04d}_RELEASED.json",
        "release_sha256": repair["_validated_release_sha256"],
        "original_report_job_id": repair["original_report_job_id"],
        "repair_report_job_id": repair["repair_report_job_id"],
        "original_failure_evidence": repair["original_failure_evidence"],
        "original_failure_evidence_sha256": repair[
            "original_failure_evidence_sha256"
        ],
        "predecessor_failure_evidence": repair[
            "predecessor_failure_evidence"
        ],
        "predecessor_failure_evidence_sha256": repair[
            "predecessor_failure_evidence_sha256"
        ],
        "attempt1_environment_evidence": dict(
            ATTEMPT1_PUBLIC_ENVIRONMENT_EVIDENCE
        ),
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
        "repair_source_installation_method": repair[
            "repair_source_installation_method"
        ],
        "repair_source_archive": repair["repair_source_archive"],
        "repair_source_archive_sha256": repair[
            "repair_source_archive_sha256"
        ],
        "repair_source_archive_size": repair["repair_source_archive_size"],
        "repair_source_archive_format": repair["repair_source_archive_format"],
        "report_publication_installation_method": repair[
            "report_publication_installation_method"
        ],
        "scheduler_job_control_observation_sha256": repair[
            "scheduler_job_control_observation_sha256"
        ],
        "worker_handoff_sha256": stable_hash(repair["worker_handoff"]),
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
    require(
        set(value) == REPORT_REPAIR_PUBLICATION_AUTHORITY_KEYS,
        "report repair publication authority schema differs",
    )
    return value


def _validated_completed_value(
    value: Mapping[str, Any],
    *,
    submission_sha256: str,
    authorization: Mapping[str, Any],
    authorization_sha256: str,
    commit: Mapping[str, Any],
    commit_sha256: str,
    publication_authority: Mapping[str, Any],
    publication_archive: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        set(value) == REPORT_REPAIR_COMPLETED_KEYS
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("status")
        == "report_repair_terminal_publication_complete"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and type(value.get("attempt")) is int
        and value.get("attempt") == 2
        and value.get("repair_report_job_id")
        == authorization["repair_report_job_id"]
        and value.get("predecessor_failure_evidence")
        == authorization["predecessor_failure_evidence"]
        and value.get("predecessor_failure_evidence_sha256")
        == authorization["predecessor_failure_evidence_sha256"]
        and value.get("authorization")
        == "REPORT_REPAIR_0002_AUTHORIZED.json"
        and value.get("authorization_sha256") == authorization_sha256
        and value.get("release") == "REPORT_REPAIR_0002_RELEASED.json"
        and value.get("release_sha256")
        == authorization["_validated_release_sha256"]
        and value.get("report_commit") == "REPORT_COMMIT.json"
        and value.get("report_commit_sha256") == commit_sha256
        and exact_json_equal(value.get("report_commit_value"), commit)
        and value.get("report_commit_value_sha256") == stable_hash(commit)
        and value.get("publication_archive") == publication_archive["archive"]
        and value.get("publication_archive_sha256")
        == publication_archive["archive_sha256"]
        and value.get("publication_archive_size")
        == publication_archive["archive_size"]
        and value.get("publication_archive_header_sha256")
        == publication_archive["header_sha256"]
        and exact_json_equal(
            _validated_direct_final_file_identity(
                value.get("publication_archive_file_identity"),
                size=int(publication_archive["archive_size"]),
                label="completed publication creation identity",
            ),
            publication_archive["file_identity"],
        )
        and value.get("publication_authority_sha256")
        == stable_hash(publication_authority)
        and value.get("repair_source_installation_method")
        == authorization["repair_source_installation_method"]
        and value.get("report_publication_installation_method")
        == authorization["report_publication_installation_method"]
        and exact_json_equal(
            value.get("expected_reassembly"), authorization["expected_reassembly"]
        )
        and value.get("publication_complete") is True
        and value.get("retry_allowed") is False
        and value.get("successor_attempt_allowed") is False
        and isinstance(value.get("completed_at_utc"), str)
        and bool(value["completed_at_utc"]),
        "completed report repair evidence differs",
    )
    return dict(value)


class _RetainedSnapshotLoader:
    """Execute one snapshot module exclusively from an already-retained FD."""

    def __init__(
        self,
        finder: "_RetainedSnapshotFinder",
        fullname: str,
        path: Path,
        *,
        is_package: bool,
    ) -> None:
        self.finder = finder
        self.fullname = fullname
        self.path = path
        self._is_package = is_package

    def create_module(self, _spec: Any) -> None:
        return None

    def exec_module(self, module: ModuleType) -> None:
        binding = self.finder.binding
        binding.revalidate()
        retained = binding.retained_regular(self.path)
        require(retained is not None, f"retained snapshot module is absent: {self.path}")
        payload, info = retained
        require(
            stat.S_ISREG(info.st_mode)
            and info.st_uid == os.getuid()
            and info.st_nlink == 1
            and stat.S_IMODE(info.st_mode) == 0o444
            and bool(payload),
            f"retained snapshot module identity differs: {self.path}",
        )
        module.__file__ = str(self.path)
        module.__cached__ = None
        if self._is_package:
            module.__path__ = [str(self.path.parent)]
        self.finder.loaded_names.add(self.fullname)
        try:
            code = compile(payload, str(self.path), "exec", dont_inherit=True)
            exec(code, module.__dict__)
            self.finder.patch_explicit_imports(module)
            binding.revalidate()
        except BaseException:
            self.finder.loaded_names.discard(self.fullname)
            sys.modules.pop(self.fullname, None)
            raise


class _RetainedImportUtilProxy:
    """Route snapshot modules' explicit spec loaders back to retained bytes."""

    def __init__(self, finder: "_RetainedSnapshotFinder") -> None:
        self.finder = finder

    def spec_from_file_location(
        self,
        name: str,
        location: str | os.PathLike[str],
        loader: object | None = None,
        *,
        submodule_search_locations: object | None = None,
    ) -> Any:
        require(
            loader is None and submodule_search_locations is None,
            "retained snapshot explicit loader arguments differ",
        )
        return self.finder.spec_for_path(str(name), Path(location))

    @staticmethod
    def module_from_spec(spec: Any) -> ModuleType:
        return importlib.util.module_from_spec(spec)


class _RetainedImportlibProxy:
    def __init__(self, finder: "_RetainedSnapshotFinder") -> None:
        self.util = _RetainedImportUtilProxy(finder)


class _RetainedSnapshotFinder:
    """Meta-path finder whose complete module image is the retained snapshot."""

    def __init__(
        self,
        binding: _RepairPublicationPhaseBinding,
        snapshot_root: Path,
    ) -> None:
        self.binding = binding
        self.snapshot_root = snapshot_root.absolute()
        self.path_by_module: dict[str, tuple[Path, bool]] = {}
        self.retained_python_paths: set[Path] = set()
        self.loaded_names: set[str] = set()
        parent_by_fd = {
            descriptor: path
            for path, descriptor, _identity, _names in binding.source_rows
            if path == self.snapshot_root or self.snapshot_root in path.parents
        }
        for parent_fd, name, _descriptor, _identity, _payload in (
            binding.source_file_rows
        ):
            parent = parent_by_fd.get(parent_fd)
            if parent is None or not name.endswith(".py"):
                continue
            path = parent / name
            self.retained_python_paths.add(path)
        relative_python = {
            path.relative_to(self.snapshot_root): path
            for path in self.retained_python_paths
        }
        package_dirs = {
            relative.parent
            for relative in relative_python
            if relative.name == "__init__.py"
        }
        for relative, path in sorted(
            relative_python.items(), key=lambda item: item[0].as_posix()
        ):
            if relative.name == "__init__.py":
                parts = relative.parent.parts
                is_package = True
            else:
                parts = (*relative.parent.parts, relative.stem)
                is_package = False
            # Only normal import packages are intercepted.  Experiment files
            # with a hyphenated directory are loaded explicitly through
            # ``spec_for_path`` below.
            if not parts or not all(part.isidentifier() for part in parts):
                continue
            required_packages = [
                Path(*parts[:index])
                for index in range(1, len(parts) if not is_package else len(parts) + 1)
            ]
            if not all(package in package_dirs for package in required_packages):
                continue
            fullname = ".".join(parts)
            require(
                fullname not in self.path_by_module,
                f"retained snapshot module name is ambiguous: {fullname}",
            )
            self.path_by_module[fullname] = (path, is_package)
        require(
            "treewm" in self.path_by_module,
            "retained snapshot import closure lacks treewm",
        )
        self.managed_import_roots = frozenset(
            name.split(".", 1)[0] for name in self.path_by_module
        )

    def spec_for_path(self, fullname: str, path: Path) -> Any:
        lexical = path.absolute()
        require(
            lexical in self.retained_python_paths
            and lexical == self.snapshot_root / lexical.relative_to(self.snapshot_root),
            f"explicit snapshot module escaped retained files: {path}",
        )
        is_package = lexical.name == "__init__.py"
        loader = _RetainedSnapshotLoader(
            self, fullname, lexical, is_package=is_package
        )
        spec = importlib.util.spec_from_loader(
            fullname,
            loader,
            origin=str(lexical),
            is_package=is_package,
        )
        require(spec is not None, f"cannot define retained snapshot module: {fullname}")
        return spec

    def find_spec(
        self, fullname: str, _path: object = None, _target: object = None
    ) -> Any:
        row = self.path_by_module.get(fullname)
        if row is None:
            root = fullname.split(".", 1)[0]
            require(
                root not in self.managed_import_roots,
                f"snapshot import is absent from retained closure: {fullname}",
            )
            return None
        path, _is_package = row
        return self.spec_for_path(fullname, path)

    def patch_explicit_imports(self, module: ModuleType) -> None:
        if "importlib" in module.__dict__:
            module.__dict__["importlib"] = _RetainedImportlibProxy(self)
        if callable(module.__dict__.get("file_sha256")):
            module.__dict__["file_sha256"] = self.retained_file_sha256
        if callable(module.__dict__.get("read_json")):
            module.__dict__["read_json"] = self.retained_json
        if callable(module.__dict__.get("_regular_file_bytes")):
            module.__dict__["_regular_file_bytes"] = self.retained_regular_bytes
        if callable(module.__dict__.get("verify_protocol_lock")):
            module.__dict__["verify_protocol_lock"] = (
                self.retained_verify_protocol_lock(module)
            )
        if module.__name__ == "treewm.utils.provenance":
            module.__dict__["trainer_code_fingerprint"] = (
                self.retained_trainer_code_fingerprint
            )
        if module.__dict__.get("CAMPAIGN_MODULE_PATH") is not None:
            for name in (
                "load_manifest",
                "load_prefix_target_lock",
                "load_resolved_config_lock",
                "load_causal_parity_lock",
            ):
                original = module.__dict__.get(name)
                if callable(original):
                    module.__dict__[name] = self.retained_gate_reader(
                        module, original
                    )

    def retained_bytes(self, path: str | os.PathLike[str]) -> bytes:
        lexical = Path(path).absolute()
        retained = self.binding.retained_regular(lexical)
        require(retained is not None, f"snapshot read escaped retained files: {lexical}")
        payload, info = retained
        require(
            stat.S_ISREG(info.st_mode)
            and info.st_uid == os.getuid()
            and info.st_nlink == 1
            and stat.S_IMODE(info.st_mode) == 0o444,
            f"retained snapshot file identity differs: {lexical}",
        )
        return payload

    def retained_file_sha256(self, path: str | os.PathLike[str]) -> str:
        return hashlib.sha256(self.retained_bytes(path)).hexdigest()

    def retained_json(self, path: str | os.PathLike[str]) -> dict[str, Any]:
        lexical = Path(path).absolute()
        return _decode_json_object(lexical, self.retained_bytes(lexical))

    def retained_regular_bytes(
        self,
        path: Path,
        label: str,
        *,
        max_bytes: int,
    ) -> bytes:
        require(
            type(max_bytes) is int and max_bytes >= 0,
            f"{label} byte boundary differs",
        )
        payload = self.retained_bytes(path)
        require(len(payload) <= max_bytes, f"{label} exceeds its byte boundary")
        return payload

    def retained_trainer_code_fingerprint(
        self, repo_root: str | Path
    ) -> dict[str, Any]:
        root = Path(repo_root).absolute()
        require(
            root == self.snapshot_root,
            "trainer fingerprint escaped retained snapshot",
        )
        paths = [
            root / "scripts" / "__init__.py",
            root / "scripts" / "train.py",
            root / "configs" / "__init__.py",
            *sorted(
                path
                for path in self.retained_python_paths
                if path == root / "treewm" or root / "treewm" in path.parents
            ),
        ]
        parent_by_fd = {
            descriptor: path
            for path, descriptor, _identity, _names in self.binding.source_rows
            if path == root or root in path.parents
        }
        yaml_paths = sorted(
            parent / name
            for parent_fd, name, _descriptor, _identity, _payload in (
                self.binding.source_file_rows
            )
            if (parent := parent_by_fd.get(parent_fd)) is not None
            and name.endswith(".yaml")
            and (parent == root / "configs" or root / "configs" in parent.parents)
        )
        paths.extend(yaml_paths)
        require(
            len(paths) == len(set(paths)),
            "retained trainer fingerprint file set is ambiguous",
        )
        files: dict[str, str] = {}
        manifest = hashlib.sha256()
        for path in paths:
            relative = path.relative_to(root).as_posix()
            digest = self.retained_file_sha256(path)
            files[relative] = digest
            manifest.update(
                relative.encode("utf-8")
                + b"\0"
                + digest.encode("ascii")
                + b"\n"
            )
        return {"manifest_sha256": manifest.hexdigest(), "files": files}

    def retained_verify_protocol_lock(self, module: ModuleType) -> Any:
        """Replace campaign's final direct ``Path.read_text`` authority read."""

        finder = self

        def wrapper(root: str | Path | None = None) -> str:
            package = Path(
                module.__dict__["PACKAGE_DIR"] if root is None else root
            ).absolute()
            require(
                package
                == finder.snapshot_root
                / "experiments"
                / "23-treewm-executable-prefix-repair-pilot-v1",
                "retained protocol package path differs",
            )
            lock_path = package / "protocol.sha256"
            try:
                locked = finder.retained_bytes(lock_path).decode("ascii").strip()
            except UnicodeDecodeError as exc:
                raise ReportError("retained protocol lock encoding differs") from exc
            live = module.__dict__["protocol_sha256"](package)
            module.__dict__["require"](
                sha256_string(locked) and locked == live,
                "protocol lock stale",
            )
            finder.binding.revalidate()
            return live

        return wrapper

    def retained_gate_reader(self, module: ModuleType, original: Any) -> Any:
        finder = self

        class RetainedTextPath:
            def __init__(self, value: object) -> None:
                self.path = Path(value)

            def read_text(self, *, encoding: str = "utf-8") -> str:
                require(encoding == "utf-8", "retained gate encoding differs")
                return finder.retained_bytes(self.path).decode("utf-8")

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            current = module.__dict__.get("Path")
            require(current is Path, "retained gate Path binding differs")
            module.__dict__["Path"] = RetainedTextPath
            try:
                return original(*args, **kwargs)
            finally:
                require(
                    module.__dict__.get("Path") is RetainedTextPath,
                    "retained gate Path binding changed",
                )
                module.__dict__["Path"] = current

        return wrapper


_ACTIVE_RETAINED_SNAPSHOT_FINDER: contextvars.ContextVar[
    _RetainedSnapshotFinder | None
] = contextvars.ContextVar("active_exp23_retained_snapshot_finder", default=None)


@contextlib.contextmanager
def _retained_snapshot_imports(
    binding: _RepairPublicationPhaseBinding, snapshot_root: Path
):
    require(
        _ACTIVE_RETAINED_SNAPSHOT_FINDER.get() is None,
        "retained snapshot importer is already active",
    )
    finder = _RetainedSnapshotFinder(binding, snapshot_root)
    require(
        not set(finder.path_by_module) & set(sys.modules),
        "snapshot module was imported before retained importer activation",
    )
    token = _ACTIVE_RETAINED_SNAPSHOT_FINDER.set(finder)
    sys.meta_path.insert(0, finder)
    try:
        binding.revalidate()
        yield finder
        binding.revalidate()
    finally:
        require(
            finder in sys.meta_path,
            "retained snapshot importer disappeared",
        )
        sys.meta_path.remove(finder)
        for name in sorted(finder.loaded_names, reverse=True):
            sys.modules.pop(name, None)
        _ACTIVE_RETAINED_SNAPSHOT_FINDER.reset(token)


def _load_module(name: str, path: Path, root: Path) -> ModuleType:
    active = _ACTIVE_REPAIR_PUBLICATION_BINDING.get()
    finder = _ACTIVE_RETAINED_SNAPSHOT_FINDER.get()
    if active is not None:
        require(
            finder is not None and finder.binding is active,
            "repair snapshot module load lacks its retained importer",
        )
        resolved = path.absolute()
        try:
            relative = resolved.relative_to(root.absolute())
        except ValueError as exc:
            raise ReportError(f"{name} escaped retained snapshot") from exc
        require(
            bool(relative.parts)
            and all(part not in {"", ".", ".."} for part in relative.parts)
            and resolved in finder.retained_python_paths,
            f"{name} is not an exact retained snapshot module",
        )
        retained = active.retained_regular(resolved)
        require(retained is not None, f"{name} retained module is absent")
        payload, info = retained
        require(
            stat.S_ISREG(info.st_mode)
            and info.st_uid == os.getuid()
            and info.st_nlink == 1
            and stat.S_IMODE(info.st_mode) == 0o444
            and bool(payload),
            f"{name} retained module identity differs",
        )
        unique = f"_treewm_exp23_report_{name}_{os.getpid()}_{time.time_ns()}"
        module = ModuleType(unique)
        module.__file__ = str(resolved)
        module.__package__ = ""
        module.__cached__ = None
        sys.modules[unique] = module
        try:
            code = compile(payload, str(resolved), "exec", dont_inherit=True)
            exec(code, module.__dict__)
            finder.patch_explicit_imports(module)
            finder.loaded_names.add(unique)
            active.revalidate()
        except BaseException:
            sys.modules.pop(unique, None)
            raise
        require(
            Path(str(module.__file__)).absolute() == resolved,
            f"{name} imported outside retained snapshot",
        )
        return module

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
    active_binding = _ACTIVE_REPAIR_PUBLICATION_BINDING.get()
    retained_finder = _ACTIVE_RETAINED_SNAPSHOT_FINDER.get()
    require(
        (active_binding is None and retained_finder is None)
        or (
            active_binding is not None
            and retained_finder is not None
            and retained_finder.binding is active_binding
            and retained_finder.snapshot_root == root
        ),
        "report snapshot import authority differs",
    )
    # In repair mode the snapshot root is deliberately absent from normal
    # PathFinder search.  Every snapshot module, including top-level modules,
    # must resolve through the retained-FD finder; a transient shadow file can
    # therefore never fall through to pathname import.
    verified_tail = [
        str(interpreter["venv_site_packages"]),
        str(interpreter["base_site_packages"]),
    ]
    if active_binding is None:
        verified_tail.insert(0, str(root))
    while str(root) in sys.path:
        sys.path.remove(str(root))
    for value in verified_tail:
        while value in sys.path:
            sys.path.remove(value)
    sys.path.extend(verified_tail)
    require(
        sys.path[-len(verified_tail) :] == verified_tail
        and (active_binding is None or str(root) not in sys.path),
        "verified report import-path order differs",
    )
    package = root / PACKAGE_RELATIVE
    nonsymlink_directory(package, "snapshot Exp23 package")
    campaign = _load_module("campaign", package / "campaign.py", root)
    worker = _load_module("worker", package / "worker.py", root)
    gate = _load_module("gate", package / "gate.py", root)
    return campaign, worker, gate


def _bind_worker_retained_scientific_io(
    worker: ModuleType,
    binding: _RepairPublicationPhaseBinding,
) -> None:
    """Force snapshot worker inspection to use the long-lived run-tree FDs."""

    def run_root(context: Mapping[str, Any]) -> Path:
        value = context.get("run_directory")
        require(
            isinstance(value, (str, Path)),
            "retained worker scientific run path differs",
        )
        root = Path(value).absolute()
        require(
            binding.has_retained_scientific_run(root),
            "worker scientific run was not retained before access",
        )
        return root

    def scientific_run_exists(context: Mapping[str, Any]) -> bool:
        return binding.has_retained_scientific_run(
            Path(context["run_directory"]).absolute()
        )

    def open_absolute_regular(
        path: str | Path, label: str
    ) -> tuple[int, os.stat_result]:
        lexical = Path(path).absolute()
        retained = binding.open_retained_regular(lexical)
        if retained is None:
            retained = binding.open_any_retained_scientific_regular(lexical)
        require(
            retained is not None,
            f"worker authority read escaped retained descriptors: {label}",
        )
        return retained

    def open_run_directory(
        context: Mapping[str, Any], *, create: bool = False
    ) -> int:
        require(not create, "report repair cannot create a scientific run")
        return binding.open_retained_scientific_directory(run_root(context))

    def open_run_regular(
        context: Mapping[str, Any], relative: str | Path, label: str
    ) -> tuple[int, os.stat_result]:
        retained = binding.open_retained_scientific_regular(
            run_root(context), _safe_relative(relative, label)
        )
        require(retained is not None, f"retained worker artifact is absent: {label}")
        return retained

    def optional_run_artifact_kind(
        context: Mapping[str, Any], relative: str | Path, label: str
    ) -> bool:
        retained = binding.open_retained_scientific_regular(
            run_root(context), _safe_relative(relative, label)
        )
        if retained is None:
            return False
        descriptor, _info = retained
        os.close(descriptor)
        return True

    def verify_snapshot_tree(
        snapshot_root: Path, inventory: Mapping[str, Any]
    ) -> None:
        binding.validate_exact_snapshot_authority(
            {
                "snapshot_root": str(Path(snapshot_root).absolute()),
                "snapshot_inventory": dict(inventory),
                "snapshot_inventory_sha256": stable_hash(dict(inventory)),
            }
        )

    def verify_snapshot_location(
        snapshot_root: Path, submission_root: Path
    ) -> None:
        require(
            Path(snapshot_root).absolute()
            == binding.submission_root / "source-snapshot" / "repo"
            and Path(submission_root).absolute() == binding.submission_root,
            "retained worker snapshot location differs",
        )
        binding.revalidate()

    worker.__dict__["_scientific_run_exists"] = scientific_run_exists
    worker.__dict__["_open_absolute_regular"] = open_absolute_regular
    worker.__dict__["_open_run_directory"] = open_run_directory
    worker.__dict__["_open_run_regular"] = open_run_regular
    worker.__dict__["_optional_run_artifact_kind"] = optional_run_artifact_kind
    worker.__dict__["_verify_snapshot_tree"] = verify_snapshot_tree
    worker.__dict__["_verify_snapshot_location"] = verify_snapshot_location


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
            not _durable_repair_stop_prefix_exists(
                submission_root, repair_attempt=repair_attempt
            ),
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
    active_binding = _ACTIVE_REPAIR_PUBLICATION_BINDING.get()
    if repair_attempt is not None or active_binding is not None:
        require(
            active_binding is not None,
            "report repair assembly lacks its retained publication binding",
        )
        _bind_worker_retained_scientific_io(worker, active_binding)
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
        if active_binding is not None:
            assert active_binding is not None
            active_binding.retain_scientific_run_tree(run_dir)
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


def _legacy_staged_seal_json(path: Path, value: Mapping[str, Any]) -> str:
    """Atomically append JSON through a retained-FD hardlink stage."""

    require(
        path.is_absolute() and path.name not in {"", ".", ".."},
        "immutable report artifact path differs",
    )
    payload = (
        json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    parent = nonsymlink_directory(path.parent, "immutable report artifact parent")
    parent_fd = _open_directory_components(parent, "immutable report artifact parent")
    target_name = path.name
    stage_name = f".{target_name}.seal.tmp"
    target_fd: int | None = None
    stage_fd: int | None = None

    def same_inode(left: os.stat_result, right: os.stat_result) -> bool:
        return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)

    try:
        parent_info = os.fstat(parent_fd)
        named_parent = parent.lstat()
        require(
            same_inode(parent_info, named_parent)
            and parent_info.st_uid == os.getuid()
            and stat.S_IMODE(parent_info.st_mode) == 0o700,
            "immutable report artifact parent identity differs",
        )

        initial_names = set(os.listdir(parent_fd))
        baseline_names = initial_names - {target_name, stage_name}
        require(
            not {
                name
                for name in baseline_names
                if name.startswith(".") and name.endswith(".seal.tmp")
            },
            "immutable report artifact namespace contains a foreign stage",
        )

        def require_namespace(expected: set[str]) -> None:
            opened = os.fstat(parent_fd)
            named = parent.lstat()
            require(
                same_inode(opened, parent_info)
                and same_inode(named, parent_info)
                and opened.st_uid == named.st_uid == os.getuid()
                and stat.S_IMODE(opened.st_mode)
                == stat.S_IMODE(named.st_mode)
                == 0o700
                and set(os.listdir(parent_fd)) == baseline_names | expected,
                "immutable report artifact parent/namespace binding changed",
            )

        def listed(name: str) -> os.stat_result | None:
            try:
                return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return None

        def open_bound(name: str) -> tuple[int, os.stat_result]:
            named = listed(name)
            require(named is not None, f"immutable report artifact vanished: {name}")
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_fd,
            )
            opened = os.fstat(descriptor)
            require(
                stat.S_ISREG(opened.st_mode)
                and same_inode(opened, named)
                and opened.st_uid == os.getuid(),
                f"immutable report artifact identity differs: {name}",
            )
            return descriptor, opened

        def read_exact(
            descriptor: int,
            name: str,
            *,
            expected_mode: int,
            expected_nlink: int,
        ) -> os.stat_result:
            opened = os.fstat(descriptor)
            named = listed(name)
            chunks: list[bytes] = []
            offset = 0
            while offset <= len(payload):
                chunk = os.pread(descriptor, len(payload) + 1 - offset, offset)
                if not chunk:
                    break
                chunks.append(chunk)
                offset += len(chunk)
            after = os.fstat(descriptor)
            require(
                named is not None
                and same_inode(opened, named)
                and _file_identity(after) == _file_identity(opened)
                and opened.st_uid == os.getuid()
                and opened.st_nlink == expected_nlink
                and stat.S_IMODE(opened.st_mode) == expected_mode
                and b"".join(chunks) == payload,
                f"immutable report artifact differs: {name}",
            )
            return opened

        require_namespace(initial_names & {target_name, stage_name})
        target_info = listed(target_name)
        stage_info = listed(stage_name)
        if target_info is not None and stage_info is not None:
            require(
                same_inode(target_info, stage_info)
                and target_info.st_uid == stage_info.st_uid == os.getuid()
                and target_info.st_nlink == stage_info.st_nlink == 2
                and stat.S_IMODE(target_info.st_mode)
                == stat.S_IMODE(stage_info.st_mode)
                == 0o444,
                "immutable report artifact target/staging state differs",
            )
            target_fd, _ = open_bound(target_name)
            stage_fd, _ = open_bound(stage_name)
            read_exact(target_fd, target_name, expected_mode=0o444, expected_nlink=2)
            read_exact(stage_fd, stage_name, expected_mode=0o444, expected_nlink=2)
            os.fsync(stage_fd)
            require_namespace({target_name, stage_name})
            read_exact(stage_fd, stage_name, expected_mode=0o444, expected_nlink=2)
            os.unlink(stage_name, dir_fd=parent_fd)
            require(
                os.fstat(stage_fd).st_nlink == 1
                and same_inode(os.fstat(stage_fd), os.fstat(target_fd)),
                "immutable report artifact recovered staging identity changed",
            )
            os.fsync(parent_fd)
            require_namespace({target_name})
            read_exact(target_fd, target_name, expected_mode=0o444, expected_nlink=1)
            return digest
        if target_info is not None:
            require(stage_info is None, "immutable report artifact staging differs")
            target_fd, _ = open_bound(target_name)
            read_exact(target_fd, target_name, expected_mode=0o444, expected_nlink=1)
            os.fsync(target_fd)
            os.fsync(parent_fd)
            require_namespace({target_name})
            read_exact(target_fd, target_name, expected_mode=0o444, expected_nlink=1)
            return digest

        resume_sealed_stage = False
        if stage_info is not None:
            require(
                stat.S_ISREG(stage_info.st_mode)
                and stage_info.st_uid == os.getuid()
                and stage_info.st_nlink == 1
                and stat.S_IMODE(stage_info.st_mode) in {0o600, 0o444},
                "immutable report artifact staging identity differs",
            )
            stage_fd, _ = open_bound(stage_name)
            if stat.S_IMODE(stage_info.st_mode) == 0o444:
                read_exact(stage_fd, stage_name, expected_mode=0o444, expected_nlink=1)
                resume_sealed_stage = True
            else:
                require_namespace({stage_name})
                os.fchmod(stage_fd, 0o600)
                os.fsync(stage_fd)
                reopened = os.fstat(stage_fd)
                renamed = listed(stage_name)
                require(
                    renamed is not None
                    and same_inode(reopened, renamed)
                    and reopened.st_uid == os.getuid()
                    and reopened.st_nlink == 1
                    and stat.S_IMODE(reopened.st_mode) == 0o600,
                    "immutable report artifact partial staging changed",
                )
                require_namespace({stage_name})
                os.unlink(stage_name, dir_fd=parent_fd)
                require(
                    os.fstat(stage_fd).st_nlink == 0,
                    "immutable report artifact partial staging remains linked",
                )
                os.fsync(parent_fd)
                require_namespace(set())
                os.close(stage_fd)
                stage_fd = None

        if not resume_sealed_stage:
            require_namespace(set())
            stage_fd = os.open(
                stage_name,
                os.O_RDWR
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0),
                0o600,
                dir_fd=parent_fd,
            )
            created = os.fstat(stage_fd)
            named_created = listed(stage_name)
            require(
                named_created is not None
                and same_inode(created, named_created)
                and created.st_uid == os.getuid()
                and created.st_nlink == 1
                and stat.S_IMODE(created.st_mode) == 0o600,
                "immutable report artifact staging creation differs",
            )
            require_namespace({stage_name})
            view = memoryview(payload)
            while view:
                written = os.write(stage_fd, view)
                require(written > 0, f"short report artifact write: {path}")
                view = view[written:]
            os.fsync(stage_fd)
            require_namespace({stage_name})
            os.fchmod(stage_fd, 0o444)
            os.fsync(stage_fd)
            read_exact(stage_fd, stage_name, expected_mode=0o444, expected_nlink=1)
            require_namespace({stage_name})

        assert stage_fd is not None
        read_exact(stage_fd, stage_name, expected_mode=0o444, expected_nlink=1)
        require_namespace({stage_name})
        os.link(
            stage_name,
            target_name,
            src_dir_fd=parent_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.fsync(parent_fd)
        require_namespace({stage_name, target_name})
        target_fd, _ = open_bound(target_name)
        read_exact(stage_fd, stage_name, expected_mode=0o444, expected_nlink=2)
        read_exact(target_fd, target_name, expected_mode=0o444, expected_nlink=2)
        require_namespace({stage_name, target_name})
        os.unlink(stage_name, dir_fd=parent_fd)
        require(
            os.fstat(stage_fd).st_nlink == 1
            and same_inode(os.fstat(stage_fd), os.fstat(target_fd)),
            "immutable report artifact linked staging identity changed",
        )
        os.fsync(parent_fd)
        require_namespace({target_name})
        read_exact(target_fd, target_name, expected_mode=0o444, expected_nlink=1)
        require(listed(stage_name) is None, "immutable report artifact staging remains")
        return digest
    finally:
        if target_fd is not None:
            os.close(target_fd)
        if stage_fd is not None:
            os.close(stage_fd)
        os.close(parent_fd)


def seal_json(path: Path, value: Mapping[str, Any]) -> str:
    """Append immutable JSON directly at its final fail-stop pathname."""

    require(
        path.is_absolute() and path.name not in {"", ".", ".."},
        "immutable report artifact path differs",
    )
    payload = (
        json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    parent = nonsymlink_directory(path.parent, "immutable report artifact parent")
    parent_fd = _open_directory_components(parent, "immutable report artifact parent")
    descriptor = -1
    try:
        parent_info = os.fstat(parent_fd)
        named_parent = parent.lstat()
        names = set(os.listdir(parent_fd))
        require(
            _file_identity(parent_info) == _file_identity(named_parent)
            and parent_info.st_uid == named_parent.st_uid == os.getuid()
            and stat.S_IMODE(parent_info.st_mode)
            == stat.S_IMODE(named_parent.st_mode)
            == 0o700
            and not {
                name
                for name in names
                if name.startswith(".") and name.endswith(".seal.tmp")
            },
            "immutable report artifact parent/fail-stop namespace differs",
        )

        def require_parent(expected_names: set[str]) -> None:
            opened = os.fstat(parent_fd)
            named = parent.lstat()
            require(
                (opened.st_dev, opened.st_ino)
                == (named.st_dev, named.st_ino)
                == (parent_info.st_dev, parent_info.st_ino)
                and opened.st_uid == named.st_uid == os.getuid()
                and stat.S_IMODE(opened.st_mode)
                == stat.S_IMODE(named.st_mode)
                == 0o700
                and set(os.listdir(parent_fd)) == expected_names,
                "immutable report artifact parent/namespace binding changed",
            )

        def read_bound(expected_mode: int) -> bytes:
            opened = os.fstat(descriptor)
            named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            chunks: list[bytes] = []
            offset = 0
            while True:
                chunk = os.pread(descriptor, 1024 * 1024, offset)
                if not chunk:
                    break
                chunks.append(chunk)
                offset += len(chunk)
            after = os.fstat(descriptor)
            require(
                stat.S_ISREG(opened.st_mode)
                and _file_identity(opened)
                == _file_identity(named)
                == _file_identity(after)
                and opened.st_uid == os.getuid()
                and opened.st_nlink == 1
                and stat.S_IMODE(opened.st_mode) == expected_mode,
                "immutable report final artifact identity differs",
            )
            return b"".join(chunks)

        if path.name in names:
            descriptor = os.open(
                path.name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=parent_fd,
            )
            require_parent(names)
            require(
                read_bound(0o444) == payload,
                    "immutable report final artifact bytes differs",
            )
            os.fsync(descriptor)
            os.fsync(parent_fd)
            require_parent(names)
            require(
                read_bound(0o444) == payload,
                "immutable report final artifact changed",
            )
            return digest

        baseline = set(names)
        require_parent(baseline)
        descriptor = os.open(
            path.name,
            os.O_RDWR
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            0o600,
            dir_fd=parent_fd,
        )
        require_parent(baseline | {path.name})
        require(
            read_bound(0o600) == b"",
            "immutable report final artifact creation differs",
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, f"short immutable report artifact write: {path}")
            view = view[written:]
        os.fsync(descriptor)
        require_parent(baseline | {path.name})
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        require(
            read_bound(0o444) == payload,
            "immutable report final artifact seal differs",
        )
        os.fsync(parent_fd)
        require_parent(baseline | {path.name})
        require(
            read_bound(0o444) == payload,
            "immutable report final artifact changed after parent fsync",
        )
        return digest
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _renameat2_noreplace(
    parent_descriptor: int, source_name: str, target_name: str
) -> None:
    """Invoke Linux ``renameat2(RENAME_NOREPLACE)`` in one directory."""

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise OSError(
            errno.ENOSYS,
            os.strerror(errno.ENOSYS),
            f"{source_name} -> {target_name}",
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        parent_descriptor,
        os.fsencode(source_name),
        parent_descriptor,
        os.fsencode(target_name),
        RENAME_NOREPLACE,
    )
    if result == 0:
        return
    error = ctypes.get_errno()
    if error == errno.EEXIST:
        raise FileExistsError(error, os.strerror(error), target_name)
    raise OSError(error, os.strerror(error), f"{source_name} -> {target_name}")


def _rename_directory_noreplace(source: Path, target: Path) -> None:
    """Primary no-replace publication used by the original reporter."""

    require(source.parent == target.parent, "report publication rename parents differ")
    parent_fd = _open_directory_components(
        source.parent, "report publication rename parent"
    )
    try:
        _renameat2_noreplace(parent_fd, source.name, target.name)
    finally:
        os.close(parent_fd)


def _directory_install_identity(value: os.stat_result) -> tuple[int, ...]:
    """Return the directory fields that must survive an atomic rename.

    Rename legitimately updates directory ctime (and can update mtime on some
    filesystems), so those timestamps are not identity fields.  Content is
    authenticated separately before and after installation.
    """

    return (
        value.st_dev,
        value.st_ino,
        value.st_uid,
        stat.S_IMODE(value.st_mode),
        value.st_nlink,
        value.st_size,
    )


def _install_repaired_report_directory(
    source: Path,
    target: Path,
    *,
    installation_method: str,
    locks: _RepairPublicationLocks,
    baseline_names: set[str],
    validate_tree: Any,
    phase_check: Any | None = None,
) -> None:
    """Install the sealed repaired report with a lock-serialized fallback.

    The plain-rename fallback is intentionally limited to the capability
    result already sealed in AUTHORIZED.  It protects every cooperating
    writer through the transaction and report/cancel locks; it makes no
    no-replace claim about an actor that ignores those locks.
    """

    require(
        source.is_absolute()
        and target.is_absolute()
        and source.parent == target.parent
        and source.name.startswith(".report.tmp.")
        and target.name == "report"
        and _valid_installation_method(installation_method),
        "repaired report installation path/method differs",
    )
    parent = nonsymlink_directory(source.parent, "repaired report parent")
    lock_bindings = locks.bindings()
    parent_fd = _open_directory_components(parent, "repaired report parent")
    source_fd: int | None = None
    try:
        parent_info = os.fstat(parent_fd)
        parent_named = parent.lstat()
        require(
            _file_identity(parent_info) == _file_identity(parent_named)
            and parent_info.st_uid == os.getuid(),
            "repaired report parent identity differs",
        )
        source_listed = os.stat(
            source.name, dir_fd=parent_fd, follow_symlinks=False
        )
        source_fd = os.open(
            source.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        source_opened = os.fstat(source_fd)
        require(
            stat.S_ISDIR(source_opened.st_mode)
            and _directory_install_identity(source_opened)
            == _directory_install_identity(source_listed)
            and source_opened.st_uid == os.getuid()
            and source_opened.st_nlink == 2
            and stat.S_IMODE(source_opened.st_mode) == 0o555,
            "repaired report staging identity differs",
        )
        source_identity = _directory_install_identity(source_opened)

        def require_bindings() -> None:
            opened = os.fstat(parent_fd)
            named = parent.lstat()
            require(
                stat.S_ISDIR(opened.st_mode)
                and stat.S_ISDIR(named.st_mode)
                and opened.st_dev == named.st_dev == parent_info.st_dev
                and opened.st_ino == named.st_ino == parent_info.st_ino
                and opened.st_uid == named.st_uid == os.getuid()
                and opened.st_nlink == named.st_nlink == parent_info.st_nlink
                and stat.S_IMODE(opened.st_mode)
                == stat.S_IMODE(named.st_mode)
                == stat.S_IMODE(parent_info.st_mode)
                and locks.bindings() == lock_bindings,
                "repaired report installation authority changed",
            )
            if phase_check is not None:
                phase_check()

        def require_target_absent() -> None:
            try:
                os.stat(target.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return
            except OSError as exc:
                raise ReportError(
                    f"repaired report target absence cannot be proven: {exc}"
                ) from exc
            raise FileExistsError(
                errno.EEXIST, os.strerror(errno.EEXIST), str(target)
            )

        def require_preinstall() -> None:
            require_bindings()
            require_target_absent()
            listed = os.stat(
                source.name, dir_fd=parent_fd, follow_symlinks=False
            )
            require(
                _directory_install_identity(listed) == source_identity
                and _directory_install_identity(os.fstat(source_fd))
                == source_identity
                and set(os.listdir(parent_fd))
                == baseline_names | {source.name},
                "repaired report preinstall namespace/identity differs",
            )
            validate_tree(source)
            require_target_absent()
            listed_after = os.stat(
                source.name, dir_fd=parent_fd, follow_symlinks=False
            )
            require(
                _directory_install_identity(listed_after) == source_identity
                and _directory_install_identity(os.fstat(source_fd))
                == source_identity
                and set(os.listdir(parent_fd))
                == baseline_names | {source.name},
                "repaired report staging changed during validation",
            )
            require_bindings()

        def require_postinstall() -> None:
            require_bindings()
            installed = os.stat(
                target.name, dir_fd=parent_fd, follow_symlinks=False
            )
            require(
                _directory_install_identity(installed) == source_identity
                and _directory_install_identity(os.fstat(source_fd))
                == source_identity
                and set(os.listdir(parent_fd))
                == baseline_names | {target.name},
                "repaired report postinstall namespace/identity differs",
            )
            try:
                os.stat(source.name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise ReportError(
                    f"repaired report staging absence cannot be proven: {exc}"
                ) from exc
            else:
                raise ReportError("repaired report staging remains after install")

        require_preinstall()
        if installation_method == INSTALL_METHOD_PRIMARY:
            _renameat2_noreplace(parent_fd, source.name, target.name)
        else:
            require_preinstall()
            os.rename(
                source.name,
                target.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        require_postinstall()
        validate_tree(target)
        require_postinstall()
        os.fsync(parent_fd)
        require_postinstall()
    finally:
        if source_fd is not None:
            os.close(source_fd)
        os.close(parent_fd)


def _legacy_remove_report_staging(
    staging: Path,
    *,
    expected_payloads: Mapping[str, bytes],
    validate_complete: Any,
    phase_check: Any | None = None,
) -> None:
    """Invalidate then remove one owned partial/completed report staging tree."""

    require(staging.is_absolute(), "report staging cleanup path differs")
    parent = nonsymlink_directory(staging.parent, "report staging cleanup parent")
    require(
        staging.name.startswith(".report.tmp.")
        and staging.name not in {".", ".."},
        "report staging cleanup name differs",
    )
    parent_fd = _open_directory_components(parent, "report staging cleanup parent")
    staging_fd: int | None = None
    transient_fds: set[int] = set()

    def stable_file_identity(value: os.stat_result) -> tuple[int, ...]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_uid,
            value.st_nlink,
            value.st_size,
        )

    try:
        parent_opened = os.fstat(parent_fd)
        parent_named = parent.lstat()
        require(
            _file_identity(parent_opened) == _file_identity(parent_named),
            "report staging cleanup parent identity differs",
        )
        staging_listed = os.stat(
            staging.name, dir_fd=parent_fd, follow_symlinks=False
        )
        staging_fd = os.open(
            staging.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        staging_opened = os.fstat(staging_fd)
        mode = stat.S_IMODE(staging_opened.st_mode)
        staging_identity = (
            staging_opened.st_dev,
            staging_opened.st_ino,
            staging_opened.st_uid,
            staging_opened.st_nlink,
        )
        require(
            stat.S_ISDIR(staging_opened.st_mode)
            and _file_identity(staging_opened) == _file_identity(staging_listed)
            and staging_opened.st_uid == os.getuid()
            and staging_opened.st_nlink == 2
            and mode in {0o700, 0o555},
            "report staging cleanup identity differs",
        )

        def require_bindings() -> None:
            opened_parent = os.fstat(parent_fd)
            named_parent = parent.lstat()
            opened_staging = os.fstat(staging_fd)
            named_staging = os.stat(
                staging.name, dir_fd=parent_fd, follow_symlinks=False
            )
            require(
                opened_parent.st_dev == named_parent.st_dev == parent_opened.st_dev
                and opened_parent.st_ino
                == named_parent.st_ino
                == parent_opened.st_ino
                and opened_staging.st_dev
                == named_staging.st_dev
                == staging_identity[0]
                and opened_staging.st_ino
                == named_staging.st_ino
                == staging_identity[1]
                and opened_staging.st_uid
                == named_staging.st_uid
                == staging_identity[2]
                == os.getuid()
                and opened_staging.st_nlink
                == named_staging.st_nlink
                == staging_identity[3]
                == 2,
                "report staging cleanup binding changed",
            )
            if phase_check is not None:
                phase_check()

        require_bindings()
        initial_names = os.listdir(staging_fd)
        require(
            len(initial_names) == len(set(initial_names)),
            "report staging cleanup namespace is not unique",
        )
        remaining_names = set(initial_names)

        def require_names(expected: set[str]) -> None:
            require_bindings()
            observed = os.listdir(staging_fd)
            require(
                len(observed) == len(set(observed))
                and set(observed) == expected,
                "report staging cleanup namespace changed",
            )

        seal_stages = {
            name
            for name in initial_names
            if name.startswith(".") and name.endswith(".seal.tmp")
        }
        require(
            len(seal_stages) <= 1
            and all(name[1:-9] in expected_payloads for name in seal_stages)
            and (mode == 0o700 or not seal_stages),
            "report staging artifact-seal namespace differs",
        )
        staged_rows: list[
            tuple[
                str,
                str,
                str,
                tuple[int, ...],
                tuple[int, ...] | None,
                int,
            ]
        ] = []
        # Classify every inner atomic-seal prefix before changing any inode.
        # A completed detached stage is resumable; it is never treated as an
        # unauthenticated partial file.  A distinct target is an ambiguous
        # competing publication and must remain untouched.
        for stage_name in sorted(seal_stages):
            target_name = stage_name[1:-9]
            staged = os.stat(
                stage_name, dir_fd=staging_fd, follow_symlinks=False
            )
            try:
                target = os.stat(
                    target_name, dir_fd=staging_fd, follow_symlinks=False
                )
            except FileNotFoundError:
                target = None
            require(
                stat.S_ISREG(staged.st_mode)
                and staged.st_uid == os.getuid()
                and stat.S_IMODE(staged.st_mode) in {0o600, 0o444},
                "report staging artifact-seal identity differs",
            )
            stage_mode = stat.S_IMODE(staged.st_mode)
            stage_identity = stable_file_identity(staged)
            target_identity = (
                stable_file_identity(target) if target is not None else None
            )
            same_inode = target is not None and (
                staged.st_dev,
                staged.st_ino,
            ) == (target.st_dev, target.st_ino)
            if target is not None:
                require(
                    same_inode
                    and stat.S_ISREG(target.st_mode)
                    and target.st_uid == os.getuid()
                    and stage_mode == 0o444
                    and stat.S_IMODE(target.st_mode) == 0o444
                    and staged.st_nlink == target.st_nlink == 2,
                    "report staging target and artifact-seal differ",
                )
                disposition = "linked_complete"
            else:
                require(
                    staged.st_nlink == 1,
                    "report staging detached artifact-seal link count differs",
                )
                disposition = (
                    "detached_complete" if stage_mode == 0o444 else "partial"
                )
            stage_fd = os.open(
                stage_name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=staging_fd,
            )
            transient_fds.add(stage_fd)
            try:
                opened_stage = os.fstat(stage_fd)
                require(
                    stable_file_identity(opened_stage) == stage_identity
                    and stat.S_IMODE(opened_stage.st_mode) == stage_mode,
                    "report staging artifact-seal raced during classification",
                )
                if stage_mode == 0o444:
                    expected = expected_payloads[target_name]
                    chunks: list[bytes] = []
                    remaining = len(expected) + 1
                    while remaining:
                        chunk = os.read(stage_fd, remaining)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        remaining -= len(chunk)
                    require(
                        b"".join(chunks) == expected
                        and stable_file_identity(os.fstat(stage_fd))
                        == stage_identity,
                        "report staging completed artifact-seal bytes differ",
                    )
            except BaseException:
                os.close(stage_fd)
                transient_fds.discard(stage_fd)
                raise
            staged_rows.append(
                (
                    stage_name,
                    target_name,
                    disposition,
                    stage_identity,
                    target_identity,
                    stage_fd,
                )
            )

        require_bindings()
        for (
            stage_name,
            target_name,
            disposition,
            stage_identity,
            target_identity,
            stage_fd,
        ) in staged_rows:
            require_names(remaining_names)
            staged = os.stat(
                stage_name, dir_fd=staging_fd, follow_symlinks=False
            )
            require(
                stable_file_identity(staged) == stage_identity
                and stat.S_IMODE(staged.st_mode)
                == (0o600 if disposition == "partial" else 0o444),
                "report staging artifact-seal changed before resolution",
            )
            if disposition == "partial":
                try:
                    os.stat(target_name, dir_fd=staging_fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise ReportError(
                        "report staging partial artifact acquired a target"
                    )
                try:
                    require(
                        stable_file_identity(os.fstat(stage_fd)) == stage_identity,
                        "report staging partial artifact-seal raced",
                    )
                    os.fchmod(stage_fd, 0o600)
                    os.fsync(stage_fd)
                    rebound_stage = os.stat(
                        stage_name,
                        dir_fd=staging_fd,
                        follow_symlinks=False,
                    )
                    rebound_opened = os.fstat(stage_fd)
                    require(
                        (rebound_stage.st_dev, rebound_stage.st_ino)
                        == (rebound_opened.st_dev, rebound_opened.st_ino)
                        == (stage_identity[0], stage_identity[1])
                        and rebound_stage.st_uid
                        == rebound_opened.st_uid
                        == os.getuid()
                        and rebound_stage.st_nlink
                        == rebound_opened.st_nlink
                        == 1
                        and stat.S_IMODE(rebound_stage.st_mode)
                        == stat.S_IMODE(rebound_opened.st_mode)
                        == 0o600,
                        "report staging partial artifact-seal changed before removal",
                    )
                    require_names(remaining_names)
                    os.unlink(stage_name, dir_fd=staging_fd)
                    remaining_names.remove(stage_name)
                    require(
                        os.fstat(stage_fd).st_nlink == 0,
                        "report staging partial artifact remains linked",
                    )
                    os.fsync(staging_fd)
                    require_names(remaining_names)
                finally:
                    os.close(stage_fd)
                    transient_fds.discard(stage_fd)
            elif disposition == "detached_complete":
                try:
                    os.stat(target_name, dir_fd=staging_fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise ReportError(
                        "report staging completed artifact acquired a target"
                    )
                require(
                    stable_file_identity(os.fstat(stage_fd)) == stage_identity,
                    "report staging completed artifact-seal raced before link",
                )
                require_names(remaining_names)
                os.link(
                    stage_name,
                    target_name,
                    src_dir_fd=staging_fd,
                    dst_dir_fd=staging_fd,
                    follow_symlinks=False,
                )
                remaining_names.add(target_name)
                require_names(remaining_names)
                require(
                    os.fstat(stage_fd).st_nlink == 2,
                    "report staging resumed artifact link count differs",
                )
                linked_stage = os.stat(
                    stage_name, dir_fd=staging_fd, follow_symlinks=False
                )
                linked_target = os.stat(
                    target_name, dir_fd=staging_fd, follow_symlinks=False
                )
                require(
                    (linked_stage.st_dev, linked_stage.st_ino)
                    == (linked_target.st_dev, linked_target.st_ino)
                    == (stage_identity[0], stage_identity[1])
                    and linked_stage.st_uid == linked_target.st_uid == os.getuid()
                    and linked_stage.st_nlink == linked_target.st_nlink == 2
                    and stat.S_IMODE(linked_stage.st_mode)
                    == stat.S_IMODE(linked_target.st_mode)
                    == 0o444,
                    "report staging resumed artifact-seal differs",
                )
                os.fsync(staging_fd)
                rebound_stage = os.stat(
                    stage_name, dir_fd=staging_fd, follow_symlinks=False
                )
                rebound_target = os.stat(
                    target_name, dir_fd=staging_fd, follow_symlinks=False
                )
                require(
                    (rebound_stage.st_dev, rebound_stage.st_ino)
                    == (rebound_target.st_dev, rebound_target.st_ino)
                    == (stage_identity[0], stage_identity[1])
                    and rebound_stage.st_uid
                    == rebound_target.st_uid
                    == os.getuid()
                    and rebound_stage.st_nlink
                    == rebound_target.st_nlink
                    == 2
                    and stat.S_IMODE(rebound_stage.st_mode)
                    == stat.S_IMODE(rebound_target.st_mode)
                    == 0o444,
                    "report staging resumed artifact-seal changed before unlink",
                )
                require_names(remaining_names)
                try:
                    os.unlink(stage_name, dir_fd=staging_fd)
                    remaining_names.remove(stage_name)
                    require(
                        os.fstat(stage_fd).st_nlink == 1,
                        "report staging resumed target link count differs",
                    )
                    os.fsync(staging_fd)
                    require_names(remaining_names)
                finally:
                    os.close(stage_fd)
                    transient_fds.discard(stage_fd)
                final_target = os.stat(
                    target_name, dir_fd=staging_fd, follow_symlinks=False
                )
                require(
                    (final_target.st_dev, final_target.st_ino)
                    == (stage_identity[0], stage_identity[1])
                    and final_target.st_uid == os.getuid()
                    and final_target.st_nlink == 1
                    and stat.S_IMODE(final_target.st_mode) == 0o444,
                    "report staging resumed artifact target differs",
                )
            else:
                require(
                    disposition == "linked_complete" and target_identity is not None,
                    "report staging artifact-seal disposition differs",
                )
                target = os.stat(
                    target_name, dir_fd=staging_fd, follow_symlinks=False
                )
                require(
                    stable_file_identity(target) == target_identity
                    and (staged.st_dev, staged.st_ino)
                    == (target.st_dev, target.st_ino)
                    and stat.S_IMODE(target.st_mode) == 0o444,
                    "report staging linked artifact-seal changed",
                )
                try:
                    require(
                        stable_file_identity(os.fstat(stage_fd)) == stage_identity,
                        "report staging linked artifact raced before unlink",
                    )
                    require_names(remaining_names)
                    os.unlink(stage_name, dir_fd=staging_fd)
                    remaining_names.remove(stage_name)
                    require(
                        os.fstat(stage_fd).st_nlink == 1,
                        "report staging linked target link count differs",
                    )
                    os.fsync(staging_fd)
                    require_names(remaining_names)
                finally:
                    os.close(stage_fd)
                    transient_fds.discard(stage_fd)
                final_target = os.stat(
                    target_name, dir_fd=staging_fd, follow_symlinks=False
                )
                require(
                    (final_target.st_dev, final_target.st_ino)
                    == (target_identity[0], target_identity[1])
                    and final_target.st_uid == os.getuid()
                    and final_target.st_nlink == 1
                    and stat.S_IMODE(final_target.st_mode) == 0o444,
                    "report staging linked artifact target differs",
                )
        if seal_stages:
            os.fsync(staging_fd)
            require_names(remaining_names)

        names_list = os.listdir(staging_fd)
        names = set(names_list)
        require(
            len(names) == len(names_list) and names <= set(expected_payloads),
            "report staging cleanup coverage differs",
        )
        require(names == remaining_names, "report staging cleanup state differs")
        opened_rows: dict[str, tuple[tuple[int, ...], int, int]] = {}
        for name in sorted(names):
            listed = os.stat(name, dir_fd=staging_fd, follow_symlinks=False)
            try:
                descriptor = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NONBLOCK", 0),
                    dir_fd=staging_fd,
                )
            except OSError as exc:
                raise ReportError(
                    f"report staging cleanup entry cannot be opened: {name}: {exc}"
                ) from exc
            transient_fds.add(descriptor)
            try:
                opened = os.fstat(descriptor)
                entry_mode = stat.S_IMODE(opened.st_mode)
                identity = stable_file_identity(opened)
                require(
                    stat.S_ISREG(opened.st_mode)
                    and _file_identity(opened) == _file_identity(listed)
                    and opened.st_uid == os.getuid()
                    and opened.st_nlink == 1
                    and entry_mode in {0o600, 0o444},
                    f"report staging cleanup entry differs: {name}",
                )
                opened_rows[name] = (identity, entry_mode, descriptor)
                if entry_mode == 0o444:
                    expected = expected_payloads[name]
                    chunks: list[bytes] = []
                    remaining = len(expected) + 1
                    while remaining:
                        chunk = os.read(descriptor, remaining)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        remaining -= len(chunk)
                    require(
                        b"".join(chunks) == expected
                        and stable_file_identity(os.fstat(descriptor)) == identity,
                        f"report staging completed artifact differs: {name}",
                    )
            except BaseException:
                os.close(descriptor)
                transient_fds.discard(descriptor)
                raise
        commit_name = "REPORT_COMMIT.json"
        completed_commit = (
            commit_name in opened_rows and opened_rows[commit_name][1] == 0o444
        )
        if mode == 0o555 or completed_commit:
            require(
                names == set(expected_payloads)
                and all(row[1] == 0o444 for row in opened_rows.values()),
                "completed report staging coverage differs",
            )
            # A 0700 tree with a still-0444 commit is the legitimate crash
            # prefix immediately after the directory was reopened for cleanup.
            # Exact same-fd payload validation above is sufficient there;
            # the normal complete-tree validator additionally authenticates
            # the sealed 0555 state.
            if mode == 0o555:
                validate_complete(staging)
                require_bindings()
                names_after_validation = os.listdir(staging_fd)
                require(
                    len(names_after_validation) == len(names_list)
                    and set(names_after_validation) == names,
                    "completed report staging changed during validation",
                )

        os.fchmod(staging_fd, 0o700)
        os.fsync(staging_fd)
        require_bindings()
        ordered_names = [
            *([commit_name] if commit_name in names else []),
            *sorted(names - {commit_name}),
        ]
        # REPORT_COMMIT is the completed-tree authority.  Invalidate it first
        # so every later crash leaves only removal-only partial state.
        for name in ordered_names:
            require_names(names)
            listed = os.stat(name, dir_fd=staging_fd, follow_symlinks=False)
            descriptor = opened_rows[name][2]
            try:
                opened = os.fstat(descriptor)
                require(
                    stable_file_identity(opened) == opened_rows[name][0]
                    and _file_identity(opened) == _file_identity(listed)
                    and stat.S_ISREG(opened.st_mode)
                    and stat.S_IMODE(opened.st_mode) in {0o600, 0o444},
                    f"report staging invalidation raced: {name}",
                )
                os.fchmod(descriptor, 0o600)
                os.fsync(descriptor)
                rebound_named = os.stat(
                    name, dir_fd=staging_fd, follow_symlinks=False
                )
                require(
                    stable_file_identity(os.fstat(descriptor))
                    == opened_rows[name][0]
                    and stable_file_identity(rebound_named)
                    == opened_rows[name][0]
                    and stat.S_IMODE(os.fstat(descriptor).st_mode) == 0o600,
                    f"report staging invalidation differs: {name}",
                )
                require_names(names)
            except BaseException:
                raise
        os.fsync(staging_fd)
        require_names(names)
        remaining_names = set(names)
        for name in [
            *sorted(names - {commit_name}),
            *([commit_name] if commit_name in names else []),
        ]:
            require_names(remaining_names)
            listed = os.stat(name, dir_fd=staging_fd, follow_symlinks=False)
            descriptor = opened_rows[name][2]
            try:
                opened = os.fstat(descriptor)
                require(
                    stable_file_identity(listed)
                    == stable_file_identity(opened)
                    == opened_rows[name][0]
                    and stat.S_ISREG(opened.st_mode)
                    and stat.S_IMODE(opened.st_mode) == 0o600,
                    f"report staging removal state differs: {name}",
                )
                os.fsync(descriptor)
                rebound_named = os.stat(
                    name, dir_fd=staging_fd, follow_symlinks=False
                )
                require(
                    stable_file_identity(rebound_named)
                    == stable_file_identity(os.fstat(descriptor))
                    == opened_rows[name][0],
                    f"report staging removal binding changed: {name}",
                )
                require_names(remaining_names)
                os.unlink(name, dir_fd=staging_fd)
                remaining_names.remove(name)
                require(
                    os.fstat(descriptor).st_nlink == 0,
                    f"report staging removed inode remains linked: {name}",
                )
                os.fsync(staging_fd)
                require_names(remaining_names)
            finally:
                os.close(descriptor)
                transient_fds.discard(descriptor)
        os.fsync(staging_fd)
        require_names(set())
        os.rmdir(staging.name, dir_fd=parent_fd)
        require(
            os.fstat(staging_fd).st_nlink == 0,
            "report staging cleanup directory remains linked",
        )
        os.fsync(parent_fd)
        final_parent_opened = os.fstat(parent_fd)
        final_parent_named = parent.lstat()
        require(
            final_parent_opened.st_dev
            == final_parent_named.st_dev
            == parent_opened.st_dev
            and final_parent_opened.st_ino
            == final_parent_named.st_ino
            == parent_opened.st_ino
            and final_parent_opened.st_uid
            == final_parent_named.st_uid
            == parent_opened.st_uid
            == os.getuid()
            and stat.S_IMODE(final_parent_opened.st_mode)
            == stat.S_IMODE(final_parent_named.st_mode)
            == stat.S_IMODE(parent_opened.st_mode),
            "report staging cleanup parent binding changed after removal",
        )
        if phase_check is not None:
            phase_check()
        try:
            os.stat(staging.name, dir_fd=parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            pass
        else:
            raise ReportError("report staging cleanup directory remains")
    finally:
        for descriptor in transient_fds:
            try:
                os.close(descriptor)
            except OSError:
                pass
        if staging_fd is not None:
            os.close(staging_fd)
        os.close(parent_fd)


def _remove_report_staging(
    staging: Path,
    *,
    expected_payloads: Mapping[str, bytes],
    validate_complete: Any,
    phase_check: Any | None = None,
) -> None:
    del expected_payloads, validate_complete, phase_check
    require(staging.is_absolute(), "report fail-stop residue path differs")
    raise ReportError(
        "attempt2 report staging/residue is permanent fail-stop evidence"
    )


def _legacy_staged_publish_report_locked(
    submission_root: Path,
    submission_sha256: str,
    bundle: Mapping[str, Any],
    decision: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    repair_installation_method: str | None = None,
    repair_locks: _RepairPublicationLocks | None = None,
    repair_phase_binding: _RepairPublicationPhaseBinding | None = None,
) -> dict[str, Any]:
    submission_root = nonsymlink_directory(submission_root, "submission root")
    report_root = submission_root / "report"
    bundle_hash = stable_hash(bundle)
    gate_hash = str(decision["gate_sha256"])
    bundle_name = f"REPORT_BUNDLE.{bundle_hash}.json"
    decision_name = f"GATE_DECISION.{gate_hash}.json"
    provenance_hash = stable_hash(provenance)
    provenance_name = f"REPORT_PROVENANCE.{provenance_hash}.json"
    bundle_payload = (
        json.dumps(dict(bundle), sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    decision_payload = (
        json.dumps(dict(decision), sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    provenance_payload = (
        json.dumps(dict(provenance), sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf-8")
    bundle_payload_sha = hashlib.sha256(bundle_payload).hexdigest()
    decision_payload_sha = hashlib.sha256(decision_payload).hexdigest()
    provenance_payload_sha = hashlib.sha256(provenance_payload).hexdigest()
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
    commit_payload = (
        json.dumps(commit, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    commit_payload_sha = hashlib.sha256(commit_payload).hexdigest()

    expected_names = {bundle_name, decision_name, provenance_name, "REPORT_COMMIT.json"}
    expected_payloads = {
        bundle_name: bundle_payload,
        decision_name: decision_payload,
        provenance_name: provenance_payload,
        "REPORT_COMMIT.json": commit_payload,
    }

    def require_repair_phase() -> None:
        if repair_phase_binding is None:
            return
        repair_phase_binding.revalidate()
        _validated_report_repair_filesystem_namespace(submission_root)
        _validated_repair_publication_namespace(submission_root)

    def validate_tree(path: Path) -> None:
        root = nonsymlink_directory(path, "published report root")
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

    require_repair_phase()
    if _lexical_exists(report_root):
        validate_tree(report_root)
        require_repair_phase()
        return commit

    leftovers = sorted(submission_root.glob(".report.tmp.*"))
    require(len(leftovers) <= 1, "multiple report staging trees exist")
    all_baseline_names = set(os.listdir(submission_root))
    leftover_names = {path.name for path in leftovers}
    baseline_names = all_baseline_names - leftover_names
    require(
        "report" not in baseline_names
        and not {
            name for name in baseline_names if name.startswith(".report.tmp.")
        }
        and not {
            name
            for name in baseline_names
            if name.startswith(".report.install-probe-")
        },
        "report publication baseline namespace differs",
    )
    staging: Path
    reuse_completed = False
    if leftovers:
        staging = leftovers[0]
        staging_info = staging.lstat()
        if stat.S_IMODE(staging_info.st_mode) == 0o555:
            validate_tree(staging)
            reuse_completed = True
        else:
            require_repair_phase()
            _remove_report_staging(
                staging,
                expected_payloads=expected_payloads,
                validate_complete=validate_tree,
                phase_check=require_repair_phase,
            )
            baseline_names = set(os.listdir(submission_root))
            staging = submission_root / f".report.tmp.{os.getpid()}.{time.time_ns()}"
    else:
        staging = submission_root / f".report.tmp.{os.getpid()}.{time.time_ns()}"
    if not reuse_completed:
        require_repair_phase()
        staging.mkdir(mode=0o700)
        _fsync_directory(submission_root)
    try:
        if not reuse_completed:
            require_repair_phase()
            require(seal_json(staging / bundle_name, bundle) == bundle_payload_sha, "staged bundle hash differs")
            require_repair_phase()
            require(seal_json(staging / decision_name, decision) == decision_payload_sha, "staged decision hash differs")
            require_repair_phase()
            require(seal_json(staging / provenance_name, provenance) == provenance_payload_sha, "staged provenance hash differs")
            require_repair_phase()
            require(
                seal_json(staging / "REPORT_COMMIT.json", commit)
                == commit_payload_sha,
                "staged report commit hash differs",
            )
            _fsync_directory(staging)
            require_repair_phase()
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
        require_repair_phase()
        validate_tree(staging)
        _fsync_directory(submission_root)
        require(
            not _durable_cleanup_prefix_exists(
                submission_root,
                repair_attempt=(2 if provenance.get("schema_version") == 2 else None),
            ),
            "cancellation/cleanup prefix appeared before report commit",
        )
        require(not _lexical_exists(report_root), "report publication target appeared concurrently")
        require_repair_phase()
        if repair_installation_method is None and repair_locks is None:
            _rename_directory_noreplace(staging, report_root)
        else:
            require(
                repair_locks is not None
                and _valid_installation_method(repair_installation_method),
                "repaired report installation authority differs",
            )
            _install_repaired_report_directory(
                staging,
                report_root,
                installation_method=str(repair_installation_method),
                locks=repair_locks,
                baseline_names=baseline_names,
                validate_tree=validate_tree,
                phase_check=require_repair_phase,
            )
        require_repair_phase()
        _fsync_directory(submission_root)
    except BaseException:
        if _lexical_exists(staging):
            try:
                require_repair_phase()
                _remove_report_staging(
                    staging,
                    expected_payloads=expected_payloads,
                    validate_complete=validate_tree,
                    phase_check=require_repair_phase,
                )
            except (OSError, ReportError) as cleanup_exc:
                raise ReportError(
                    f"failed report staging cleanup could not complete: {cleanup_exc}"
                ) from cleanup_exc
            require(not _lexical_exists(staging), "failed report staging survived cleanup")
        raise
    require_repair_phase()
    validate_tree(report_root)
    return commit


def _legacy_direct_publish_report_locked(
    submission_root: Path,
    submission_sha256: str,
    bundle: Mapping[str, Any],
    decision: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    repair_installation_method: str | None = None,
    repair_locks: _RepairPublicationLocks | None = None,
    repair_phase_binding: _RepairPublicationPhaseBinding | None = None,
) -> dict[str, Any]:
    """Publish directly into the final report directory with fail-stop crashes."""

    submission_root = nonsymlink_directory(submission_root, "submission root")
    report_root = submission_root / "report"
    bundle_hash = stable_hash(bundle)
    gate_hash = str(decision["gate_sha256"])
    provenance_hash = stable_hash(provenance)
    bundle_name = f"REPORT_BUNDLE.{bundle_hash}.json"
    decision_name = f"GATE_DECISION.{gate_hash}.json"
    provenance_name = f"REPORT_PROVENANCE.{provenance_hash}.json"
    values: list[tuple[str, Mapping[str, Any]]] = [
        (bundle_name, bundle),
        (decision_name, decision),
        (provenance_name, provenance),
    ]
    payloads = {
        name: (
            json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        for name, value in values
    }
    commit = {
        "schema_version": 1,
        "status": decision["status"],
        "scientific_rejection": decision["status"] == "rejected",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "report_bundle": bundle_name,
        "report_bundle_sha256": bundle_hash,
        "report_bundle_file_sha256": hashlib.sha256(
            payloads[bundle_name]
        ).hexdigest(),
        "gate_decision": decision_name,
        "gate_sha256": gate_hash,
        "gate_decision_file_sha256": hashlib.sha256(
            payloads[decision_name]
        ).hexdigest(),
        "provenance": provenance_name,
        "provenance_sha256": provenance_hash,
        "provenance_file_sha256": hashlib.sha256(
            payloads[provenance_name]
        ).hexdigest(),
    }
    commit_payload = (
        json.dumps(commit, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    payloads["REPORT_COMMIT.json"] = commit_payload
    values.append(("REPORT_COMMIT.json", commit))
    expected_names = set(payloads)

    def validate_complete(path: Path) -> None:
        root = nonsymlink_directory(path, "published report root")
        root_info = root.lstat()
        require(
            root_info.st_uid == os.getuid()
            and root_info.st_nlink == 2
            and stat.S_IMODE(root_info.st_mode) == 0o555
            and set(os.listdir(root)) == expected_names,
            "published direct-final report root differs",
        )
        for name, payload in payloads.items():
            opened = contained_regular(root / name, root, f"published report {name}")
            info = opened.lstat()
            require(
                info.st_uid == os.getuid()
                and info.st_nlink == 1
                and stat.S_IMODE(info.st_mode) == 0o444
                and opened.read_bytes() == payload,
                f"published direct-final report file differs: {name}",
            )

    if repair_phase_binding is not None:
        repair_phase_binding.revalidate()
    _validated_repair_publication_namespace(submission_root)
    if os.path.lexists(report_root):
        validate_complete(report_root)
        if repair_phase_binding is not None:
            repair_phase_binding.revalidate()
        return commit
    require(
        (repair_installation_method is None and repair_locks is None)
        or (
            repair_locks is not None
            and repair_installation_method == DIRECT_FINAL_INSTALL_METHOD
        ),
        "direct-final repaired report installation authority differs",
    )
    submission_fd = _open_directory_components(
        submission_root, "direct-final report parent"
    )
    report_fd = -1
    retained: dict[str, tuple[int, tuple[int, ...], bytes]] = {}
    try:
        submission_info = os.fstat(submission_fd)
        submission_names = set(os.listdir(submission_fd))
        require("report" not in submission_names, "report target already exists")
        os.mkdir("report", mode=0o700, dir_fd=submission_fd)
        os.fsync(submission_fd)
        report_fd = os.open(
            "report",
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=submission_fd,
        )
        report_info = os.fstat(report_fd)
        named_report = os.stat(
            "report", dir_fd=submission_fd, follow_symlinks=False
        )
        require(
            _file_identity(report_info) == _file_identity(named_report)
            and report_info.st_uid == os.getuid()
            and report_info.st_nlink == 2
            and stat.S_IMODE(report_info.st_mode) == 0o700
            and os.listdir(report_fd) == [],
            "direct-final report directory creation differs",
        )
        if repair_phase_binding is not None:
            repair_phase_binding.advance_direct_report_directory()

        def rebind(expected: set[str], *, final_mode: int = 0o700) -> None:
            opened_submission = os.fstat(submission_fd)
            named_submission = submission_root.lstat()
            opened_report = os.fstat(report_fd)
            current_report = os.stat(
                "report", dir_fd=submission_fd, follow_symlinks=False
            )
            require(
                (opened_submission.st_dev, opened_submission.st_ino)
                == (named_submission.st_dev, named_submission.st_ino)
                == (submission_info.st_dev, submission_info.st_ino)
                and (opened_report.st_dev, opened_report.st_ino)
                == (current_report.st_dev, current_report.st_ino)
                == (report_info.st_dev, report_info.st_ino)
                and opened_submission.st_uid
                == opened_report.st_uid
                == os.getuid()
                and opened_report.st_nlink == 2
                and stat.S_IMODE(opened_report.st_mode) == final_mode
                and set(os.listdir(submission_fd))
                == submission_names | {"report"}
                and set(os.listdir(report_fd)) == expected,
                "direct-final report retained directory binding changed",
            )
            for name, (descriptor, identity, payload) in retained.items():
                opened = os.fstat(descriptor)
                named = os.stat(name, dir_fd=report_fd, follow_symlinks=False)
                os.lseek(descriptor, 0, os.SEEK_SET)
                observed = b""
                while len(observed) < len(payload) + 1:
                    chunk = os.read(descriptor, len(payload) + 1 - len(observed))
                    if not chunk:
                        break
                    observed += chunk
                require(
                    _file_identity(opened)
                    == _file_identity(named)
                    == identity
                    and observed == payload,
                    f"direct-final report retained file changed: {name}",
                )
            if repair_phase_binding is not None:
                repair_phase_binding.revalidate()

        written: set[str] = set()
        for name, value in values:
            rebind(written)
            require(
                seal_json(report_root / name, value)
                == hashlib.sha256(payloads[name]).hexdigest(),
                f"direct-final report artifact hash differs: {name}",
            )
            descriptor = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=report_fd,
            )
            retained[name] = (
                descriptor,
                _file_identity(os.fstat(descriptor)),
                payloads[name],
            )
            written.add(name)
            rebind(written)
        os.fsync(report_fd)
        os.fchmod(report_fd, 0o555)
        os.fsync(report_fd)
        os.fsync(submission_fd)
        rebind(expected_names, final_mode=0o555)
        validate_complete(report_root)
        rebind(expected_names, final_mode=0o555)
        if repair_phase_binding is not None:
            repair_phase_binding.retain_direct_report_tree(report_root)
            repair_phase_binding.revalidate()
        return commit
    finally:
        for descriptor, _identity, _payload in retained.values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        if report_fd >= 0:
            os.close(report_fd)
        os.close(submission_fd)


def _pretty_json_memfd(value: Mapping[str, Any], label: str) -> tuple[int, int, str]:
    """Serialize canonical pretty JSON into a private anonymous descriptor."""

    descriptor = os.memfd_create(label, getattr(os, "MFD_CLOEXEC", 0))
    digest = hashlib.sha256()
    size = 0
    try:
        encoder = json.JSONEncoder(
            sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
        )
        for text in encoder.iterencode(dict(value)):
            block = text.encode("utf-8")
            digest.update(block)
            size += len(block)
            view = memoryview(block)
            while view:
                written = os.write(descriptor, view)
                require(written > 0, f"short private {label} write")
                view = view[written:]
        digest.update(b"\n")
        size += 1
        require(os.write(descriptor, b"\n") == 1, f"short private {label} newline")
        os.fsync(descriptor)
        require(
            os.fstat(descriptor).st_size == size
            and _stable_open_fd_sha256(descriptor, label)[0] == digest.hexdigest(),
            f"private {label} serialization differs",
        )
        return descriptor, size, digest.hexdigest()
    except BaseException:
        os.close(descriptor)
        raise


def _copy_fd_range(source_fd: int, target_fd: int, size: int, label: str) -> None:
    offset = 0
    while offset < size:
        block = os.pread(source_fd, min(16 * 1024 * 1024, size - offset), offset)
        require(block, f"{label} source is truncated")
        view = memoryview(block)
        while view:
            written = os.write(target_fd, view)
            require(written > 0, f"short {label} write")
            view = view[written:]
        offset += len(block)
    require(not os.pread(source_fd, 1, size), f"{label} source has trailing bytes")


def _build_publication_archive(
    submission_sha256: str,
    bundle: Mapping[str, Any],
    decision: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> tuple[int, int, str, dict[str, Any], dict[str, Any]]:
    """Build the exact logical quartet as one private framed stream."""

    bundle_hash = stable_hash(bundle)
    gate_hash = str(decision["gate_sha256"])
    provenance_hash = stable_hash(provenance)
    names = {
        "report_bundle": f"REPORT_BUNDLE.{bundle_hash}.json",
        "gate_decision": f"GATE_DECISION.{gate_hash}.json",
        "provenance": f"REPORT_PROVENANCE.{provenance_hash}.json",
        "report_commit": "REPORT_COMMIT.json",
    }
    entry_fds: dict[str, tuple[int, int, str, str]] = {}
    archive_fd = -1
    try:
        for kind, value, logical_hash in (
            ("report_bundle", bundle, bundle_hash),
            ("gate_decision", decision, gate_hash),
            ("provenance", provenance, provenance_hash),
        ):
            descriptor, size, digest = _pretty_json_memfd(value, f"exp23-{kind}")
            entry_fds[kind] = (descriptor, size, digest, logical_hash)
        commit = {
            "schema_version": 1,
            "status": decision["status"],
            "scientific_rejection": decision["status"] == "rejected",
            "campaign_id": CAMPAIGN_ID,
            "submission_sha256": submission_sha256,
            "report_bundle": names["report_bundle"],
            "report_bundle_sha256": bundle_hash,
            "report_bundle_file_sha256": entry_fds["report_bundle"][2],
            "gate_decision": names["gate_decision"],
            "gate_sha256": gate_hash,
            "gate_decision_file_sha256": entry_fds["gate_decision"][2],
            "provenance": names["provenance"],
            "provenance_sha256": provenance_hash,
            "provenance_file_sha256": entry_fds["provenance"][2],
        }
        require(set(commit) == REPORT_COMMIT_KEYS, "publication archive commit differs")
        descriptor, size, digest = _pretty_json_memfd(commit, "exp23-report-commit")
        entry_fds["report_commit"] = (
            descriptor,
            size,
            digest,
            stable_hash(commit),
        )
        entries = [
            {
                "kind": kind,
                "name": names[kind],
                "size": entry_fds[kind][1],
                "sha256": entry_fds[kind][2],
                "logical_sha256": entry_fds[kind][3],
            }
            for kind in PUBLICATION_ARCHIVE_ENTRY_ORDER
        ]
        header = {
            "archive_kind": PUBLICATION_ARCHIVE_KIND,
            "schema_version": 2,
            "campaign_id": CAMPAIGN_ID,
            "submission_sha256": submission_sha256,
            "entry_order": list(PUBLICATION_ARCHIVE_ENTRY_ORDER),
            "entries": entries,
            "report_commit_sha256": entry_fds["report_commit"][2],
            "report_commit_value_sha256": stable_hash(commit),
        }
        header_bytes = json.dumps(
            header,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        archive_fd = os.memfd_create(
            "exp23-publication-archive", getattr(os, "MFD_CLOEXEC", 0)
        )
        prefix = PUBLICATION_ARCHIVE_MAGIC + len(header_bytes).to_bytes(8, "big")
        for block in (prefix, header_bytes):
            view = memoryview(block)
            while view:
                written = os.write(archive_fd, view)
                require(written > 0, "short private publication archive write")
                view = view[written:]
        for kind in PUBLICATION_ARCHIVE_ENTRY_ORDER:
            name_bytes = names[kind].encode("ascii")
            size = entry_fds[kind][1]
            frame = (
                len(name_bytes).to_bytes(8, "big")
                + name_bytes
                + size.to_bytes(8, "big")
            )
            view = memoryview(frame)
            while view:
                written = os.write(archive_fd, view)
                require(written > 0, "short publication archive frame write")
                view = view[written:]
            _copy_fd_range(
                entry_fds[kind][0], archive_fd, size, f"publication {kind}"
            )
        os.fsync(archive_fd)
        archive_digest, archive_info = _stable_open_fd_sha256(
            archive_fd, "private publication archive"
        )
        return archive_fd, archive_info.st_size, archive_digest, commit, header
    except BaseException:
        if archive_fd >= 0:
            os.close(archive_fd)
        raise
    finally:
        for descriptor, _size, _digest, _logical in entry_fds.values():
            try:
                os.close(descriptor)
            except OSError:
                pass


def _publish_report_locked(
    submission_root: Path,
    submission_sha256: str,
    bundle: Mapping[str, Any],
    decision: Mapping[str, Any],
    provenance: Mapping[str, Any],
    *,
    repair_installation_method: str | None = None,
    repair_locks: _RepairPublicationLocks | None = None,
    repair_phase_binding: _RepairPublicationPhaseBinding | None = None,
) -> dict[str, Any]:
    """Publish one content-addressed archive directly at its final root name."""

    require(
        repair_phase_binding is not None
        and repair_locks is not None
        and repair_phase_binding.locks is repair_locks
        and repair_installation_method == PUBLICATION_ARCHIVE_INSTALL_METHOD,
        "publication archive authority differs",
    )
    require(
        len(repair_phase_binding.scientific_runs) == 20,
        "publication archive scientific run generation differs",
    )
    for run_root in repair_phase_binding.scientific_runs:
        repair_phase_binding._revalidate_scientific_run(
            run_root, full_hash=True
        )
    archive_fd, archive_size, archive_sha, commit, header = (
        _build_publication_archive(
            submission_sha256, bundle, decision, provenance
        )
    )
    archive_name = (
        f"{PUBLICATION_ARCHIVE_PREFIX}{archive_sha}{PUBLICATION_ARCHIVE_SUFFIX}"
    )
    archive_path = submission_root / archive_name
    try:
        for run_root in repair_phase_binding.scientific_runs:
            repair_phase_binding._revalidate_scientific_run(
                run_root, full_hash=True
            )
        existing = {
            name
            for name in os.listdir(repair_phase_binding.root_descriptor)
            if name.startswith(PUBLICATION_ARCHIVE_PREFIX)
            and name.endswith(PUBLICATION_ARCHIVE_SUFFIX)
        }
        require(
            not existing or existing == {archive_name},
            "publication archive namespace differs",
        )
        require(
            not existing,
            "preexisting publication archive is permanent fail-stop evidence",
        )
        retained_fd = repair_phase_binding.create_root_file_from_fd(
            archive_path,
            archive_fd,
            size=archive_size,
            digest=archive_sha,
            label="report publication archive",
        )
        repair_phase_binding.publication_archive_evidence = {
            "schema_version": 2,
            "archive_kind": PUBLICATION_ARCHIVE_KIND,
            "archive": archive_name,
            "archive_sha256": archive_sha,
            "archive_size": archive_size,
            "header": header,
            "header_sha256": stable_hash(header),
            "descriptor": retained_fd,
            "file_identity": _direct_final_file_identity(
                os.fstat(retained_fd)
            ),
        }
        repair_phase_binding._require_submission_and_sources()
        return commit
    finally:
        os.close(archive_fd)


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


class _CompletedReportTreeBinding:
    def __init__(self, submission_root: Path):
        self.root = submission_root / "report"
        self.file_rows: dict[
            str, tuple[int, tuple[int, ...], bytes]
        ] = {}
        self.descriptor = _open_directory_components(
            self.root, "completed repaired report root"
        )
        try:
            self.identity = os.fstat(self.descriptor)
            self.payloads = self._read_tree()
        except BaseException:
            for descriptor, _identity, _payload in self.file_rows.values():
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            self.file_rows.clear()
            os.close(self.descriptor)
            raise

    def _require_root(self) -> None:
        opened = os.fstat(self.descriptor)
        named = self.root.lstat()
        require(
            stat.S_ISDIR(opened.st_mode)
            and _file_identity(opened)
            == _file_identity(named)
            == _file_identity(self.identity)
            and opened.st_uid == named.st_uid == os.getuid()
            and opened.st_nlink == named.st_nlink == 2
            and stat.S_IMODE(opened.st_mode)
            == stat.S_IMODE(named.st_mode)
            == 0o555,
            "completed repaired report root identity differs",
        )

    @staticmethod
    def _read_retained(descriptor: int) -> bytes:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)

    def _read_file(self, name: str) -> bytes:
        require(
            Path(name).name == name and name not in {"", ".", ".."},
            "completed repaired report filename differs",
        )
        listed = os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
        retained = self.file_rows.get(name)
        if retained is not None:
            descriptor, identity, payload = retained
            opened = os.fstat(descriptor)
            require(
                stat.S_ISREG(opened.st_mode)
                and _file_identity(opened) == _file_identity(listed) == identity
                and opened.st_uid == os.getuid()
                and opened.st_nlink == 1
                and stat.S_IMODE(opened.st_mode) == 0o444
                and self._read_retained(descriptor) == payload
                and _file_identity(os.fstat(descriptor)) == identity,
                f"retained completed repaired report file changed: {name}",
            )
            return payload
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=self.descriptor,
        )
        keep_descriptor = False
        try:
            opened = os.fstat(descriptor)
            require(
                stat.S_ISREG(opened.st_mode)
                and _file_identity(opened) == _file_identity(listed)
                and opened.st_uid == os.getuid()
                and opened.st_nlink == 1
                and stat.S_IMODE(opened.st_mode) == 0o444,
                f"completed repaired report file identity differs: {name}",
            )
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            after = os.fstat(descriptor)
            require(
                _file_identity(after) == _file_identity(opened)
                and after.st_size == opened.st_size,
                f"completed repaired report file changed: {name}",
            )
            payload = b"".join(chunks)
            self.file_rows[name] = (
                descriptor,
                _file_identity(opened),
                payload,
            )
            keep_descriptor = True
            return payload
        finally:
            if not keep_descriptor:
                os.close(descriptor)

    def _read_tree(self) -> dict[str, bytes]:
        self._require_root()
        commit_payload = self._read_file("REPORT_COMMIT.json")
        commit = _decode_json_object(self.root / "REPORT_COMMIT.json", commit_payload)
        require(
            set(commit) == REPORT_COMMIT_KEYS
            and isinstance(commit.get("report_bundle"), str)
            and isinstance(commit.get("gate_decision"), str)
            and isinstance(commit.get("provenance"), str),
            "completed repaired report commit differs",
        )
        names = {
            "REPORT_COMMIT.json",
            str(commit["report_bundle"]),
            str(commit["gate_decision"]),
            str(commit["provenance"]),
        }
        require(
            len(names) == 4
            and all(Path(name).name == name and name not in {"", ".", ".."} for name in names)
            and set(os.listdir(self.descriptor)) == names,
            "completed repaired report file coverage differs",
        )
        payloads = {"REPORT_COMMIT.json": commit_payload}
        for name in sorted(names - {"REPORT_COMMIT.json"}):
            payloads[name] = self._read_file(name)
        self._require_root()
        require(
            set(os.listdir(self.descriptor)) == names,
            "completed repaired report namespace changed",
        )
        return payloads

    def revalidate(self) -> None:
        require(
            self._read_tree() == self.payloads,
            "completed repaired report tree changed",
        )

    def authenticate(
        self,
        *,
        submission_sha256: str,
        commit: Mapping[str, Any],
        publication_authority: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Authenticate the retained quartet, not merely its structure."""

        self.revalidate()

        def decoded_exact(name: str, label: str) -> dict[str, Any]:
            payload = self.payloads[name]
            value = _decode_json_object(self.root / name, payload)
            require(
                payload
                == (
                    json.dumps(value, sort_keys=True, indent=2, allow_nan=False)
                    + "\n"
                ).encode("utf-8"),
                f"completed repaired {label} serialization differs",
            )
            return value

        sealed_commit = decoded_exact("REPORT_COMMIT.json", "report commit")
        require(
            set(sealed_commit) == REPORT_COMMIT_KEYS
            and exact_json_equal(sealed_commit, commit)
            and type(sealed_commit.get("schema_version")) is int
            and sealed_commit.get("schema_version") == 1
            and sealed_commit.get("status") == EXPECTED_REPAIR_REASSEMBLY["status"]
            and sealed_commit.get("scientific_rejection") is True
            and sealed_commit.get("campaign_id") == CAMPAIGN_ID
            and sealed_commit.get("submission_sha256") == submission_sha256,
            "completed repaired report commit authority differs",
        )
        bundle_name = str(sealed_commit["report_bundle"])
        gate_name = str(sealed_commit["gate_decision"])
        provenance_name = str(sealed_commit["provenance"])
        require(
            bundle_name
            == f"REPORT_BUNDLE.{EXPECTED_REPAIR_REASSEMBLY['report_bundle_sha256']}.json"
            and gate_name
            == f"GATE_DECISION.{EXPECTED_REPAIR_REASSEMBLY['gate_sha256']}.json"
            and provenance_name
            == f"REPORT_PROVENANCE.{sealed_commit['provenance_sha256']}.json",
            "completed repaired report filenames differ",
        )
        bundle_payload = self.payloads[bundle_name]
        gate_payload = self.payloads[gate_name]
        provenance_payload = self.payloads[provenance_name]
        bundle = decoded_exact(bundle_name, "bundle")
        gate = decoded_exact(gate_name, "gate decision")
        provenance = decoded_exact(provenance_name, "provenance")
        provenance_v1 = dict(provenance)
        provenance_v1.pop("publication_authority", None)
        provenance_v1["schema_version"] = 1
        provenance_v1_payload = (
            json.dumps(
                provenance_v1, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
        require(
            stable_hash(bundle)
            == sealed_commit.get("report_bundle_sha256")
            == EXPECTED_REPAIR_REASSEMBLY["report_bundle_sha256"]
            and len(bundle_payload)
            == EXPECTED_REPAIR_REASSEMBLY["report_bundle_file_size"]
            and hashlib.sha256(bundle_payload).hexdigest()
            == sealed_commit.get("report_bundle_file_sha256")
            == EXPECTED_REPAIR_REASSEMBLY["report_bundle_file_sha256"]
            and gate.get("status") == EXPECTED_REPAIR_REASSEMBLY["status"]
            and gate.get("gate_sha256")
            == sealed_commit.get("gate_sha256")
            == EXPECTED_REPAIR_REASSEMBLY["gate_sha256"]
            and len(gate_payload)
            == EXPECTED_REPAIR_REASSEMBLY["gate_decision_file_size"]
            and hashlib.sha256(gate_payload).hexdigest()
            == sealed_commit.get("gate_decision_file_sha256")
            == EXPECTED_REPAIR_REASSEMBLY["gate_decision_file_sha256"]
            and set(provenance)
            == REPORT_PROVENANCE_V1_KEYS | {"publication_authority"}
            and type(provenance.get("schema_version")) is int
            and provenance.get("schema_version") == 2
            and provenance.get("campaign_id") == CAMPAIGN_ID
            and provenance.get("submission_sha256") == submission_sha256
            and exact_json_equal(
                provenance.get("publication_authority"), publication_authority
            )
            and stable_hash(provenance)
            == sealed_commit.get("provenance_sha256")
            and hashlib.sha256(provenance_payload).hexdigest()
            == sealed_commit.get("provenance_file_sha256")
            and set(provenance_v1) == REPORT_PROVENANCE_V1_KEYS
            and stable_hash(provenance_v1)
            == EXPECTED_REPAIR_REASSEMBLY["original_provenance_v1_sha256"]
            and len(provenance_v1_payload)
            == EXPECTED_REPAIR_REASSEMBLY["original_provenance_v1_file_size"]
            and hashlib.sha256(provenance_v1_payload).hexdigest()
            == EXPECTED_REPAIR_REASSEMBLY[
                "original_provenance_v1_file_sha256"
            ],
            "completed repaired report quartet authority differs",
        )
        self.revalidate()
        return sealed_commit

    def close(self) -> None:
        for descriptor, _identity, _payload in reversed(
            list(self.file_rows.values())
        ):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.file_rows.clear()
        os.close(self.descriptor)


def _legacy_seal_report_repair_completed(
    submission_root: Path,
    submission_sha256: str,
    authorization: Mapping[str, Any],
    authorization_sha256: str,
    commit: Mapping[str, Any],
    publication_authority: Mapping[str, Any],
    locks: "_RepairPublicationLocks",
    journal_binding: _RepairPublicationPhaseBinding,
) -> dict[str, Any]:
    """Seal the terminal attempt-2 publication boundary after the quartet."""

    require(journal_binding.locks is locks, "completed publication lock binding differs")
    journal_binding.revalidate()
    binding = _CompletedReportTreeBinding(submission_root)
    try:
        return _legacy_seal_report_repair_completed_bound(
            submission_root,
            submission_sha256,
            authorization,
            authorization_sha256,
            commit,
            publication_authority,
            binding,
            journal_binding,
        )
    finally:
        binding.close()


def _legacy_seal_report_repair_completed_bound(
    submission_root: Path,
    submission_sha256: str,
    authorization: Mapping[str, Any],
    authorization_sha256: str,
    commit: Mapping[str, Any],
    publication_authority: Mapping[str, Any],
    binding: _CompletedReportTreeBinding,
    journal_binding: _RepairPublicationPhaseBinding,
) -> dict[str, Any]:

    path = submission_root / "REPORT_REPAIR_0002_COMPLETED.json"
    authenticated_commit = binding.authenticate(
        submission_sha256=submission_sha256,
        commit=commit,
        publication_authority=publication_authority,
    )
    require(
        exact_json_equal(authenticated_commit, commit),
        "completed repaired report retained commit differs",
    )
    commit_path = contained_regular(
        submission_root / "report/REPORT_COMMIT.json",
        submission_root,
        "completed repair report commit",
    )
    commit_info = commit_path.lstat()
    commit_sha256 = file_sha256(commit_path)
    require(
        stat.S_IMODE(commit_info.st_mode) == 0o444
        and commit_info.st_uid == os.getuid()
        and commit_info.st_nlink == 1
        and exact_json_equal(read_json(commit_path), commit)
        and set(commit) == REPORT_COMMIT_KEYS,
        "completed repair report commit differs",
    )
    if _lexical_exists(path, "completed report repair"):
        completed_path = contained_regular(
            path, submission_root, "completed report repair"
        )
        completed_info = completed_path.lstat()
        sealed = read_json(completed_path)
        stage_path = path.parent / f".{path.name}.seal.tmp"
        linked_completed_stage = False
        if completed_info.st_nlink == 2 and _lexical_exists(
            stage_path, "completed report repair stage"
        ):
            stage_info = stage_path.lstat()
            linked_completed_stage = (
                stat.S_ISREG(stage_info.st_mode)
                and stage_info.st_uid == os.getuid()
                and stage_info.st_nlink == 2
                and stat.S_IMODE(stage_info.st_mode) == 0o444
                and _file_identity(stage_info) == _file_identity(completed_info)
            )
        require(
            stat.S_IMODE(completed_info.st_mode) == 0o444
            and completed_info.st_uid == os.getuid()
            and (completed_info.st_nlink == 1 or linked_completed_stage)
            and set(sealed) == REPORT_REPAIR_COMPLETED_KEYS
            and type(sealed.get("schema_version")) is int
            and sealed.get("schema_version") == 1
            and sealed.get("status")
            == "report_repair_terminal_publication_complete"
            and sealed.get("campaign_id") == CAMPAIGN_ID
            and sealed.get("submission_sha256") == submission_sha256
            and type(sealed.get("attempt")) is int
            and sealed.get("attempt") == 2
            and sealed.get("repair_report_job_id")
            == authorization["repair_report_job_id"]
            and sealed.get("predecessor_failure_evidence")
            == authorization["predecessor_failure_evidence"]
            and sealed.get("predecessor_failure_evidence_sha256")
            == authorization["predecessor_failure_evidence_sha256"]
            and sealed.get("authorization")
            == "REPORT_REPAIR_0002_AUTHORIZED.json"
            and sealed.get("authorization_sha256") == authorization_sha256
            and sealed.get("release") == "REPORT_REPAIR_0002_RELEASED.json"
            and sealed.get("release_sha256")
            == authorization["_validated_release_sha256"]
            and sealed.get("report_commit") == "report/REPORT_COMMIT.json"
            and sealed.get("report_commit_sha256") == commit_sha256
            and exact_json_equal(sealed.get("report_commit_value"), commit)
            and sealed.get("report_commit_value_sha256") == stable_hash(commit)
            and sealed.get("publication_authority_sha256")
            == stable_hash(publication_authority)
            and sealed.get("repair_source_installation_method")
            == authorization["repair_source_installation_method"]
            and sealed.get("report_publication_installation_method")
            == authorization["report_publication_installation_method"]
            and exact_json_equal(
                sealed.get("expected_reassembly"), authorization["expected_reassembly"]
            )
            and sealed.get("publication_complete") is True
            and sealed.get("retry_allowed") is False
            and sealed.get("successor_attempt_allowed") is False
            and isinstance(sealed.get("completed_at_utc"), str)
            and bool(sealed["completed_at_utc"]),
            "completed report repair evidence differs",
        )
        binding.revalidate()
        journal_binding.revalidate()
        binding.authenticate(
            submission_sha256=submission_sha256,
            commit=commit,
            publication_authority=publication_authority,
        )
        journal_binding.seal_completed(path, sealed)
        binding.revalidate()
        require(
            "REPORT_REPAIR_0002_COMPLETED.json" in journal_binding.names,
            "completed report repair is absent after recovery",
        )
        journal_binding.revalidate()
        rebound_path = contained_regular(
            path, submission_root, "completed report repair"
        )
        rebound_info = rebound_path.lstat()
        rebound = read_json(rebound_path)
        require(
            rebound_info.st_nlink == 1 and exact_json_equal(rebound, sealed),
            "completed report repair recovery differs",
        )
        return dict(rebound)
    staged_completed: Mapping[str, Any] | None = None
    if journal_binding.stage_state is not None:
        staged_mode, staged_linked, staged_payload = journal_binding.stage_state
        if staged_mode == 0o444 and staged_linked is False:
            require(
                isinstance(staged_payload, bytes),
                "completed report repair staged payload differs",
            )
            staged_completed = _decode_json_object(
                path.parent / f".{path.name}.seal.tmp", staged_payload
            )
            require(
                set(staged_completed) == REPORT_REPAIR_COMPLETED_KEYS
                and isinstance(staged_completed.get("completed_at_utc"), str)
                and bool(staged_completed["completed_at_utc"]),
                "completed report repair staged value differs",
            )
    value = {
        "schema_version": 1,
        "status": "report_repair_terminal_publication_complete",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "attempt": 2,
        "repair_report_job_id": authorization["repair_report_job_id"],
        "predecessor_failure_evidence": authorization[
            "predecessor_failure_evidence"
        ],
        "predecessor_failure_evidence_sha256": authorization[
            "predecessor_failure_evidence_sha256"
        ],
        "authorization": "REPORT_REPAIR_0002_AUTHORIZED.json",
        "authorization_sha256": authorization_sha256,
        "release": "REPORT_REPAIR_0002_RELEASED.json",
        "release_sha256": authorization["_validated_release_sha256"],
        "report_commit": "report/REPORT_COMMIT.json",
        "report_commit_sha256": commit_sha256,
        "report_commit_value": dict(commit),
        "report_commit_value_sha256": stable_hash(commit),
        "publication_authority_sha256": stable_hash(publication_authority),
        "repair_source_installation_method": authorization[
            "repair_source_installation_method"
        ],
        "report_publication_installation_method": authorization[
            "report_publication_installation_method"
        ],
        "expected_reassembly": dict(authorization["expected_reassembly"]),
        "publication_complete": True,
        "retry_allowed": False,
        "successor_attempt_allowed": False,
        "completed_at_utc": (
            staged_completed["completed_at_utc"]
            if staged_completed is not None
            else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ),
    }
    if staged_completed is not None:
        require(
            exact_json_equal(staged_completed, value),
            "completed report repair staged authority differs",
        )
    binding.revalidate()
    journal_binding.revalidate()
    binding.authenticate(
        submission_sha256=submission_sha256,
        commit=commit,
        publication_authority=publication_authority,
    )
    journal_binding.seal_completed(path, value)
    binding.revalidate()
    require(
        "REPORT_REPAIR_0002_COMPLETED.json" in journal_binding.names,
        "completed report repair is absent after seal",
    )
    journal_binding.revalidate()
    sealed = read_json(
        contained_regular(path, submission_root, "completed report repair")
    )
    require(
        set(sealed) == REPORT_REPAIR_COMPLETED_KEYS
        and exact_json_equal(sealed, value),
        "completed report repair evidence differs",
    )
    return dict(sealed)


def _seal_report_repair_completed(
    submission_root: Path,
    submission_sha256: str,
    authorization: Mapping[str, Any],
    authorization_sha256: str,
    commit: Mapping[str, Any],
    publication_authority: Mapping[str, Any],
    locks: "_RepairPublicationLocks",
    journal_binding: _RepairPublicationPhaseBinding,
) -> dict[str, Any]:
    """Bind COMPLETED to the same retained publication-archive descriptor."""

    require(
        journal_binding.locks is locks
        and hasattr(journal_binding, "publication_archive_evidence"),
        "completed publication archive binding differs",
    )
    evidence = journal_binding.publication_archive_evidence
    require(
        isinstance(evidence, Mapping)
        and set(evidence)
        == {
            "schema_version",
            "archive_kind",
            "archive",
            "archive_sha256",
            "archive_size",
            "header",
            "header_sha256",
            "descriptor",
            "file_identity",
        }
        and evidence.get("schema_version") == 2
        and evidence.get("archive_kind") == PUBLICATION_ARCHIVE_KIND
        and sha256_string(evidence.get("archive_sha256"))
        and type(evidence.get("archive_size")) is int
        and evidence["archive_size"] > 0
        and isinstance(evidence.get("header"), Mapping)
        and evidence.get("header_sha256") == stable_hash(evidence["header"])
        and type(evidence.get("descriptor")) is int,
        "completed publication archive evidence differs",
    )
    archive_digest, archive_info = _stable_open_fd_sha256(
        int(evidence["descriptor"]), "completed publication archive"
    )
    commit_rows = [
        row
        for row in evidence["header"]["entries"]
        if row.get("kind") == "report_commit"
    ]
    require(
        archive_digest == evidence["archive_sha256"]
        and archive_info.st_size == evidence["archive_size"]
        and exact_json_equal(
            _validated_direct_final_file_identity(
                evidence.get("file_identity"),
                size=archive_info.st_size,
                label="publication archive creator identity",
            ),
            _direct_final_file_identity(archive_info),
        )
        and evidence["archive"]
        == f"{PUBLICATION_ARCHIVE_PREFIX}{archive_digest}{PUBLICATION_ARCHIVE_SUFFIX}"
        and len(commit_rows) == 1
        and commit_rows[0].get("name") == "REPORT_COMMIT.json"
        and commit_rows[0].get("logical_sha256") == stable_hash(commit)
        and set(commit) == REPORT_COMMIT_KEYS,
        "completed publication archive/commit differs",
    )
    path = submission_root / "REPORT_REPAIR_0002_COMPLETED.json"
    base_value = {
        "schema_version": 1,
        "status": "report_repair_terminal_publication_complete",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "attempt": 2,
        "repair_report_job_id": authorization["repair_report_job_id"],
        "predecessor_failure_evidence": authorization[
            "predecessor_failure_evidence"
        ],
        "predecessor_failure_evidence_sha256": authorization[
            "predecessor_failure_evidence_sha256"
        ],
        "authorization": "REPORT_REPAIR_0002_AUTHORIZED.json",
        "authorization_sha256": authorization_sha256,
        "release": "REPORT_REPAIR_0002_RELEASED.json",
        "release_sha256": authorization["_validated_release_sha256"],
        "report_commit": "REPORT_COMMIT.json",
        "report_commit_sha256": commit_rows[0]["sha256"],
        "report_commit_value": dict(commit),
        "report_commit_value_sha256": stable_hash(commit),
        "publication_archive": evidence["archive"],
        "publication_archive_sha256": evidence["archive_sha256"],
        "publication_archive_size": evidence["archive_size"],
        "publication_archive_header_sha256": evidence["header_sha256"],
        "publication_archive_file_identity": evidence["file_identity"],
        "publication_authority_sha256": stable_hash(publication_authority),
        "repair_source_installation_method": authorization[
            "repair_source_installation_method"
        ],
        "report_publication_installation_method": authorization[
            "report_publication_installation_method"
        ],
        "expected_reassembly": dict(authorization["expected_reassembly"]),
        "publication_complete": True,
        "retry_allowed": False,
        "successor_attempt_allowed": False,
    }
    if "REPORT_REPAIR_0002_COMPLETED.json" in journal_binding.names:
        sealed = _decode_json_object(
            path,
            journal_binding.payloads["REPORT_REPAIR_0002_COMPLETED.json"],
        )
        require(
            set(sealed) == REPORT_REPAIR_COMPLETED_KEYS
            and isinstance(sealed.get("completed_at_utc"), str)
            and bool(sealed["completed_at_utc"])
            and exact_json_equal(
                {key: value for key, value in sealed.items() if key != "completed_at_utc"},
                base_value,
            ),
            "completed publication archive recovery differs",
        )
        journal_binding.revalidate()
        return dict(sealed)
    value = {
        **base_value,
        "completed_at_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
        ),
    }
    require(set(value) == REPORT_REPAIR_COMPLETED_KEYS, "completed schema differs")
    journal_binding.revalidate()
    require(
        _stable_open_fd_sha256(
            int(evidence["descriptor"]), "publication archive before completed"
        )[0]
        == evidence["archive_sha256"],
        "publication archive changed before completed",
    )
    journal_binding.seal_completed(path, value)
    require(
        _stable_open_fd_sha256(
            int(evidence["descriptor"]), "publication archive after completed"
        )[0]
        == evidence["archive_sha256"],
        "publication archive changed after completed",
    )
    journal_binding.revalidate()
    return value


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

    def publish_under_lock(
        *,
        repair_locks: _RepairPublicationLocks | None = None,
        phase_binding: _RepairPublicationPhaseBinding | None = None,
    ) -> dict[str, Any]:
        owns_phase_binding = False
        try:
            if repair_attempt is not None or repair_authorization_sha256 is not None:
                require(
                    repair_locks is not None,
                    "report repair retained publication locks are absent",
                )
                if phase_binding is None:
                    phase_binding = _RepairPublicationPhaseBinding(
                        submission_root,
                        allow_completed_stage=True,
                        locks=repair_locks,
                    )
                    owns_phase_binding = True
                require(
                    phase_binding.locks is repair_locks,
                    "report repair retained publication lock graph differs",
                )
                phase_binding.revalidate()
            _validated_report_publication_prerequisite(
                submission_root, submission_sha256, provenance
            )
            if phase_binding is not None:
                phase_binding.revalidate()
            receipt = _validated_report_publication_receipt(
                submission_root, submission_sha256
            )
            if phase_binding is not None:
                phase_binding.revalidate()
            if repair_attempt is None and repair_authorization_sha256 is None:
                require(
                    os.environ.get("SLURM_JOB_ID") == receipt.get("report_job_id"),
                    "report publication requires the exact committed report Slurm job",
                )
            else:
                require(
                    type(repair_attempt) is int
                    and repair_attempt == 2
                    and sha256_string(repair_authorization_sha256),
                    "report repair publication arguments differ",
                )
                assert phase_binding is not None and repair_locks is not None
                authority = _validated_report_repair_authorization_bound(
                    submission_root,
                    submission_sha256,
                    receipt,
                    attempt=repair_attempt,
                    expected_raw_sha256=str(repair_authorization_sha256),
                    lock_bindings=repair_locks.bindings(),
                    phase_binding=phase_binding,
                )
                phase_binding.revalidate()
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
                not _durable_cleanup_prefix_exists(
                    submission_root, repair_attempt=repair_attempt
                ),
                "cancelled/ambiguous submission cannot publish a report",
            )
            publisher = (
                _publish_report_locked
                if repair_attempt is not None
                else _legacy_staged_publish_report_locked
            )
            commit = publisher(
                submission_root,
                submission_sha256,
                bundle,
                decision,
                provenance,
                repair_installation_method=(
                    str(authority["report_publication_installation_method"])
                    if repair_attempt is not None
                    else None
                ),
                repair_locks=repair_locks,
                repair_phase_binding=phase_binding,
            )
            if repair_attempt is not None:
                assert phase_binding is not None and repair_locks is not None
                _seal_report_repair_completed(
                    submission_root,
                    submission_sha256,
                    authority,
                    str(repair_authorization_sha256),
                    commit,
                    expected_publication_authority,
                    repair_locks,
                    phase_binding,
                )
            return commit
        finally:
            if phase_binding is not None and owns_phase_binding:
                phase_binding.close()

    if repair_attempt is None and repair_authorization_sha256 is None:
        with _ReportCancelLock(submission_root):
            return publish_under_lock()
    active = _ACTIVE_REPAIR_PUBLICATION_BINDING.get()
    if active is not None:
        require(
            active.submission_root == submission_root
            and active.locks is not None,
            "active report repair publication binding differs",
        )
        return publish_under_lock(
            repair_locks=active.locks,
            phase_binding=active,
        )
    with _RepairPublicationLocks(submission_root) as repair_locks:
        return publish_under_lock(repair_locks=repair_locks)


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
            (args.publish_repair and args.repair_attempt == 2 and sha256_string(args.repair_authorization_sha256))
            or (
                not args.publish_repair
                and args.repair_attempt is None
                and args.repair_authorization_sha256 is None
            ),
            "repair arguments require --publish-repair attempt 2 and an authorization SHA256",
        )
        def run_pipeline() -> dict[str, Any]:
            bundle, decision, provenance = assemble_report(
                args.snapshot_root,
                args.submission_root,
                args.submission_sha256,
                require_publish_job=args.publish,
                repair_attempt=(args.repair_attempt if args.publish_repair else None),
                repair_authorization_sha256=(
                    args.repair_authorization_sha256
                    if args.publish_repair
                    else None
                ),
            )
            if args.publish or args.publish_repair:
                return publish_report(
                    args.submission_root,
                    args.submission_sha256,
                    bundle,
                    decision,
                    provenance,
                    repair_attempt=(
                        args.repair_attempt if args.publish_repair else None
                    ),
                    repair_authorization_sha256=(
                        args.repair_authorization_sha256
                        if args.publish_repair
                        else None
                    ),
                )
            return {
                "schema_version": 1,
                "status": "read_only_report_verified",
                "scientific_status": decision["status"],
                "report_bundle_sha256": stable_hash(bundle),
                "gate_sha256": decision["gate_sha256"],
                "writes_performed": 0,
                "scheduler_calls": 0,
            }

        if args.publish_repair:
            reject_environment()
            with _RepairPublicationLocks(args.submission_root) as repair_locks:
                phase_binding = _RepairPublicationPhaseBinding(
                    args.submission_root,
                    allow_completed_stage=True,
                    locks=repair_locks,
                )
                try:
                    with _active_repair_publication_binding(phase_binding):
                        with _retained_snapshot_imports(
                            phase_binding, args.snapshot_root
                        ):
                            _wait_for_repair_release_evidence(
                                args.submission_root,
                                args.submission_sha256,
                                attempt=args.repair_attempt,
                                authorization_sha256=args.repair_authorization_sha256,
                                phase_binding=phase_binding,
                            )
                            result = run_pipeline()
                finally:
                    phase_binding.close()
        else:
            result = run_pipeline()
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
