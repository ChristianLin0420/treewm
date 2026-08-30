#!/usr/bin/env python3
"""Second-generation, append-only repair for the failed Launch8 report publisher.

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
import contextlib
import contextlib
import contextvars
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
ATTEMPT = 2
PREDECESSOR_ATTEMPT = 1
CONFIRMATION = "SUBMIT_EXP23_LAUNCH8_REPORT_REPAIR_0002"
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
EXPECTED_ORIGINAL_GIT_PROVENANCE = {
    "branch": "main",
    "head": EXPECTED_ORIGINAL_SOURCE_COMMIT,
    "object_format": "sha1",
    "origin_main": EXPECTED_ORIGINAL_SOURCE_COMMIT,
    "remote_origin": "git@github.com:ChristianLin0420/treewm.git",
    "worktree_status": "clean",
    "worktree_status_sha256": (
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    ),
}
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
EXPECTED_ATTEMPT1_JOB_ID = "33349323"
EXPECTED_ATTEMPT1_JOB_NAME = (
    f"exp23-launch8-{EXPECTED_SUBMISSION_SHA256[:16]}-report-repair-0001"
)
EXPECTED_ATTEMPT1_COMMENT = (
    f"treewm-exp23-report-repair:{EXPECTED_SUBMISSION_SHA256}:0001"
)
EXPECTED_ATTEMPT1_LOG_SHA256 = (
    "5f6501b43be9014aa74d8c4e427466687735b82f0e3098db35d638f7e6b5ef01"
)
EXPECTED_ATTEMPT1_LOG_SIZE = 38
EXPECTED_ATTEMPT1_LOG_BYTES = b"repair publication cannot be requeued\n"
EXPECTED_ATTEMPT1_TERMINAL_SACCT_STDOUT_SHA256 = (
    "8a02a2bb3de891045f67c6fdb12781693aabf1f783bca28454dbef2fa38a60d8"
)
EXPECTED_ATTEMPT1_ENV_STDOUT_SHA256 = (
    "1a64556b0bd9c2ff3f0f2521825b453151eae9eebb9978c1f318303982c1e73a"
)
EXPECTED_ATTEMPT1_ENV_STDOUT_SIZE = 748
EXPECTED_ATTEMPT1_BATCH_STDOUT_SHA256 = (
    "1138a359fa9162dd1bd3ab57f978401ca33964c215f5dd3baa5374decad0f4a5"
)
EXPECTED_ATTEMPT1_BATCH_STDOUT_SIZE = 5_930
ATTEMPT1_PUBLIC_ENVIRONMENT_EVIDENCE = {
    "schema_version": 1,
    "raw_stdout_sha256": EXPECTED_ATTEMPT1_ENV_STDOUT_SHA256,
    "raw_stdout_size": EXPECTED_ATTEMPT1_ENV_STDOUT_SIZE,
    "allowlisted_projection": {
        "slurm_export_env": "NONE",
        "slurm_restart_count_present": False,
    },
}
EXPECTED_ATTEMPT1_CHAIN_SHA256 = {
    "REPORT_REPAIR_0001_ORIGINAL_FAILURE.json": (
        "fb7119e73abb9ef745b03f85a83e3492364a08f9354e6986056e24537ef02a9d"
    ),
    "CALLING_REPORT_REPAIR_0001_SUBMIT.json": (
        "17d7478dfbee2ef76e7c881121f436044153d871dec60a6ca29a13b0a38e058d"
    ),
    "REPORT_REPAIR_0001_SUBMITTED.json": (
        "4b9f6998cd5e4c92fb4ee67afa8d9da8539e6108d9f397c740f959d3c2626a06"
    ),
    "REPORT_REPAIR_0001_AUTHORIZED.json": (
        "f921ab2114908315cc5fe5f628cd6eda594fa2b826c6e4495a00c1152321f46a"
    ),
    "CALLING_REPORT_REPAIR_0001_RELEASE_0000.json": (
        "55607fc914e817de939ba4353fe222a9a56235e63536446f59a885c98af9e86b"
    ),
    "REPORT_REPAIR_0001_RELEASE_RESULT_0000.json": (
        "eb114ed55e70f9b0f36c8b17a22967af7bc90b859782905b0d3d917df68bc0ab"
    ),
    "REPORT_REPAIR_0001_RELEASED.json": (
        "a59ff5dece424bd253576920287115c0cecfca6a54a19dc4b34b54a54d923b81"
    ),
}
EXPECTED_ATTEMPT1_SOURCE_AUTHORITY_SHA256 = (
    "f23a9e83478d602724d86faa86ea6783bd48e4edcd8a6ddcefd94d0bdc9d5930"
)
EXPECTED_ATTEMPT1_SOURCE_COMMIT = "8fd45297ce09bb4a7e6d048ffc3f085d3841f254"
EXPECTED_ATTEMPT1_SOURCE_PROTOCOL = (
    "d2b2f87bf15ea1a488208098155a2b3c230f564d5b6faf5bbba55b4ace548d95"
)
EXPECTED_ATTEMPT1_SOURCE_FILES_SHA256 = (
    "c5d310c2abf34dc6a1598bb92ad9b1bc29f8765492157316e1728aea670480ce"
)
EXPECTED_ATTEMPT1_SOURCE_FILES = {
    "report.py": {
        "mode": 0o444,
        "size": 184_358,
        "sha256": "31df11e598f4d0da9ed7958c387d7777de617b532e435ff14d601eb0888f3a07",
    },
    "report_repair.py": {
        "mode": 0o444,
        "size": 231_673,
        "sha256": "047d63ad280e36958987b048aac3c616b2355c40d4ace8324f9fce06c2ff06d8",
    },
    "report_repair.slurm": {
        "mode": 0o444,
        "size": 5_822,
        "sha256": "15ce6712f16c0655b4ad3d544987aec25574531cfc02b470c1ed9395bd363962",
    },
    "protocol.sha256": {
        "mode": 0o444,
        "size": 65,
        "sha256": "d0c90b5039073f6f358e170c19fdcbba17b0282de58b6d7f0fb68b13577376a1",
    },
}
SOURCE_INSTALL_METHOD_PRIMARY = "renameat2_noreplace"
SOURCE_INSTALL_METHOD_FALLBACK_PREFIX = "locked_same_parent_rename_after_errno_"
DIRECT_FINAL_INSTALL_METHOD = "direct_final_name_o_excl"
SOURCE_ARCHIVE_INSTALL_METHOD = "direct_final_source_archive_o_excl"
PUBLICATION_ARCHIVE_INSTALL_METHOD = "direct_final_publication_archive_o_excl"
SOURCE_ARCHIVE_NAME = "REPORT_REPAIR_0002_SOURCE_ARCHIVE.bin"
SOURCE_ARCHIVE_KIND = "treewm_exp23_report_repair_source"
SOURCE_ARCHIVE_MARKER = b"__TREEWM_EXP23_SOURCE_ARCHIVE_V2_PAYLOAD__\n"
SOURCE_ARCHIVE_END = b"\n__TREEWM_EXP23_SOURCE_ARCHIVE_V2_END__\n"
PUBLICATION_ARCHIVE_PREFIX = "REPORT_REPAIR_0002_PUBLICATION."
PUBLICATION_ARCHIVE_SUFFIX = ".archive"
PUBLICATION_ARCHIVE_KIND = "treewm_exp23_report_repair_publication"
PUBLICATION_ARCHIVE_MAGIC = b"TREEWM_EXP23_REPORT_REPAIR_PUBLICATION_V2\n"
PUBLICATION_ARCHIVE_ENTRY_ORDER = (
    "report_bundle",
    "gate_decision",
    "provenance",
    "report_commit",
)
SOURCE_NAMES = (
    "report.py",
    "report_repair.py",
    "report_repair.slurm",
    "protocol.sha256",
)
SOURCE_AUTHORITY_NAME = "SOURCE_AUTHORITY.json"
SOURCE_AUTHORITY_V1_KEYS = frozenset(
    {
        "schema_version",
        "repair_source_commit",
        "repair_package_protocol_sha256",
        "repair_source_files",
        "repair_source_files_sha256",
    }
)
SOURCE_AUTHORITY_V2_KEYS = frozenset(
    {*SOURCE_AUTHORITY_V1_KEYS, "repair_source_installation_method"}
)
SOURCE_ARCHIVE_EVIDENCE_KEYS = frozenset(
    {
        *SOURCE_AUTHORITY_V2_KEYS,
        "repair_source_archive",
        "repair_source_archive_sha256",
        "repair_source_archive_size",
        "repair_source_archive_format",
    }
)
ATTEMPT1_SOURCE_EVIDENCE_KEYS = frozenset(
    {"root", "authority", "authority_sha256", *SOURCE_AUTHORITY_V1_KEYS}
)
RENAME_NOREPLACE = 1
RENAME_NOREPLACE_FALLBACK_ERRNOS = frozenset(
    {errno.EINVAL, errno.ENOSYS, errno.EOPNOTSUPP}
)
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
GIT_RE = re.compile(r"[0-9a-f]{40}\Z")
JOB_ID_RE = re.compile(r"[1-9][0-9]*\Z")
SBATCH_CLUSTER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
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

ATTEMPT1_TERMINAL_KEYS = frozenset(
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
ATTEMPT2_PREDECESSOR_KEYS = frozenset(
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
COMPLETED_KEYS = frozenset(
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
PROVENANCE_V1_KEYS = frozenset(
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
PUBLICATION_AUTHORITY_KEYS = frozenset(
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
        EXPECTED_ATTEMPT1_JOB_ID,
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


class RepairError(RuntimeError):
    """The one-generation report-repair state is unsafe or ambiguous."""


class CommandResult(NamedTuple):
    returncode: int
    stdout: bytes
    stderr: bytes


Runner = Callable[[Sequence[str], Path, Mapping[str, str]], CommandResult]

_ACTIVE_RUN_PASS_FDS: contextvars.ContextVar[tuple[int, ...]] = (
    contextvars.ContextVar("active_exp23_scheduler_pass_fds", default=())
)


def _valid_installation_method(value: object) -> bool:
    if value in {
        SOURCE_INSTALL_METHOD_PRIMARY,
        DIRECT_FINAL_INSTALL_METHOD,
        SOURCE_ARCHIVE_INSTALL_METHOD,
    }:
        return True
    if not isinstance(value, str) or not value.startswith(
        SOURCE_INSTALL_METHOD_FALLBACK_PREFIX
    ):
        return False
    suffix = value.removeprefix(SOURCE_INSTALL_METHOD_FALLBACK_PREFIX)
    return suffix.isascii() and suffix.isdigit() and int(suffix) in {
        int(item) for item in RENAME_NOREPLACE_FALLBACK_ERRNOS
    }


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
    active_transition_var = globals().get("_ACTIVE_REPAIR_TRANSITION")
    if active_transition_var is not None:
        transition = active_transition_var.get()
        if transition is not None:
            retained = transition._maybe_retained_regular(path)
            if retained is not None:
                payload, info = retained
                require(len(payload) <= max_size, f"{label} size differs")
                return payload, hashlib.sha256(payload).hexdigest(), info
            require(
                not transition.manages_regular_path(path),
                f"{label} escaped the retained authority graph",
            )
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


def _revalidated_sealed_json(
    path: Path,
    expected_value: Mapping[str, Any],
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    """Rebind one immutable predecessor by name, inode, bytes, and value."""

    value, digest, info = read_json(path, label)
    require(
        stat.S_ISREG(info.st_mode)
        and stat.S_IMODE(info.st_mode) == 0o444
        and info.st_uid == os.getuid()
        and info.st_nlink == 1
        and digest == expected_sha256
        and exact_json_equal(value, expected_value),
        f"{label} changed",
    )
    return value


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


def _renameat2_noreplace(
    parent_descriptor: int, source_name: str, target_name: str
) -> None:
    """Invoke Linux ``renameat2(RENAME_NOREPLACE)`` within one directory."""

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


def _rename_directory_noreplace(
    source: Path,
    target: Path,
    source_authority: Mapping[str, Any],
    locks: _RepairLocks,
    installation_method: str,
) -> str:
    """Install one sealed source directory without replacing an authorized peer.

    ``renameat2(RENAME_NOREPLACE)`` is the primary operation.  Some Lustre
    deployments reject that flag for directories.  For only that narrow
    capability-error set, the already-held transaction and report/cancel locks,
    one exact staging-only namespace, and one revalidated parent dirfd serialize
    every authorized writer before a same-dirfd ``rename`` fallback.  The
    fallback deliberately does not claim kernel no-replace protection against
    an actor that ignores those locks.
    """

    require(
        source.is_absolute()
        and target.is_absolute()
        and source.parent == target.parent,
        "repair snapshot rename parents differ",
    )
    require(
        source.name.startswith(".source.tmp.")
        and source.name not in {".", ".."}
        and target.name == "source",
        "repair snapshot rename names differ",
    )
    parent = _canonical_existing_directory(
        source.parent, "repair snapshot rename parent"
    )
    lock_bindings = locks.bindings()
    parent_descriptor = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    source_descriptor: int | None = None
    try:
        parent_opened = os.fstat(parent_descriptor)
        parent_named = parent.lstat()
        require(
            stat.S_ISDIR(parent_opened.st_mode)
            and _file_identity(parent_opened) == _file_identity(parent_named)
            and parent_opened.st_uid == os.getuid()
            and stat.S_IMODE(parent_opened.st_mode) == 0o700,
            "repair snapshot rename parent identity differs",
        )
        source_listed = os.stat(
            source.name, dir_fd=parent_descriptor, follow_symlinks=False
        )
        source_descriptor = os.open(
            source.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_descriptor,
        )
        source_opened = os.fstat(source_descriptor)
        require(
            stat.S_ISDIR(source_opened.st_mode)
            and _file_identity(source_opened) == _file_identity(source_listed)
            and source_opened.st_uid == os.getuid()
            and source_opened.st_nlink == 2
            and stat.S_IMODE(source_opened.st_mode) == 0o555,
            "repair snapshot staging identity differs before install",
        )
        source_identity = _file_identity(source_opened)

        def require_parent_and_lock_bindings() -> None:
            opened = os.fstat(parent_descriptor)
            named = parent.lstat()
            require(
                stat.S_ISDIR(opened.st_mode)
                and opened.st_dev == parent_opened.st_dev
                and opened.st_ino == parent_opened.st_ino
                and opened.st_uid == parent_opened.st_uid == os.getuid()
                and stat.S_IMODE(opened.st_mode)
                == stat.S_IMODE(parent_opened.st_mode)
                == 0o700
                and named.st_dev == opened.st_dev
                and named.st_ino == opened.st_ino
                and named.st_uid == opened.st_uid
                and stat.S_IMODE(named.st_mode) == 0o700
                and locks.bindings() == lock_bindings,
                "repair snapshot install authority binding changed",
            )

        def require_target_absent() -> None:
            try:
                os.stat(
                    target.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                return
            except OSError as exc:
                raise RepairError(
                    f"repair snapshot target absence cannot be proven: {exc}"
                ) from exc
            raise FileExistsError(
                errno.EEXIST, os.strerror(errno.EEXIST), str(target)
            )

        def require_exact_preinstall_state() -> None:
            require_parent_and_lock_bindings()
            listed = os.stat(
                source.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
            opened = os.fstat(source_descriptor)
            require_target_absent()
            require(
                _file_identity(listed) == source_identity
                and _file_identity(opened) == source_identity
                and set(os.listdir(parent_descriptor)) == {source.name},
                "repair snapshot preinstall namespace/identity differs",
            )
            _validate_sealed_repair_source(source, source_authority)
            listed_after = os.stat(
                source.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
            require_target_absent()
            require(
                _file_identity(listed_after) == source_identity
                and _file_identity(os.fstat(source_descriptor)) == source_identity
                and set(os.listdir(parent_descriptor)) == {source.name},
                "repair snapshot staging changed during validation",
            )
            require_parent_and_lock_bindings()

        def require_exact_postinstall_state() -> None:
            require_parent_and_lock_bindings()
            target_listed = os.stat(
                target.name, dir_fd=parent_descriptor, follow_symlinks=False
            )
            require(
                _file_identity(target_listed) == source_identity
                and _file_identity(os.fstat(source_descriptor))
                == source_identity
                and set(os.listdir(parent_descriptor)) == {target.name},
                "installed repair snapshot identity/namespace differs",
            )
            try:
                os.stat(
                    source.name,
                    dir_fd=parent_descriptor,
                    follow_symlinks=False,
                )
            except FileNotFoundError:
                pass
            except OSError as exc:
                raise RepairError(
                    f"repair snapshot staging absence cannot be proven: {exc}"
                ) from exc
            else:
                raise RepairError("repair snapshot staging remains after install")

        require_exact_preinstall_state()
        if installation_method == SOURCE_INSTALL_METHOD_PRIMARY:
            _renameat2_noreplace(
                parent_descriptor, source.name, target.name
            )
        else:
            match = re.fullmatch(
                rf"{re.escape(SOURCE_INSTALL_METHOD_FALLBACK_PREFIX)}([0-9]+)",
                installation_method,
            )
            require(match is not None, "repair snapshot installation method differs")
            fallback_errno = int(match.group(1))
            require(
                fallback_errno in RENAME_NOREPLACE_FALLBACK_ERRNOS,
                "repair snapshot fallback errno differs",
            )
            require_exact_preinstall_state()
            os.rename(
                source.name,
                target.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )

        require_exact_postinstall_state()
        _validate_sealed_repair_source(target, source_authority)
        require_exact_postinstall_state()
        os.fsync(parent_descriptor)
        require_exact_postinstall_state()
        return installation_method
    finally:
        if source_descriptor is not None:
            os.close(source_descriptor)
        os.close(parent_descriptor)


def _directory_install_method_probe(
    parent: Path,
    probe_label: str,
    locks: _RepairLocks,
    *,
    expected_baseline: set[str] | None = None,
) -> str:
    """Select one exact directory-install method without leaving probe state."""

    root = _canonical_existing_directory(parent, f"{probe_label} probe parent")
    require(
        re.fullmatch(r"[a-z0-9-]+", probe_label) is not None,
        "directory-install probe label differs",
    )
    source_name = f".{probe_label}.install-probe-source"
    target_name = f".{probe_label}.install-probe-target"
    probe_names = {source_name, target_name}
    lock_bindings = locks.bindings()
    descriptor = os.open(
        root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        parent_info = os.fstat(descriptor)
        named_parent = root.lstat()
        require(
            stat.S_ISDIR(parent_info.st_mode)
            and _file_identity(parent_info) == _file_identity(named_parent)
            and parent_info.st_uid == os.getuid()
            and stat.S_IMODE(parent_info.st_mode) == 0o700,
            "directory-install probe parent identity differs",
        )

        def require_bindings() -> None:
            opened = os.fstat(descriptor)
            named = root.lstat()
            require(
                opened.st_dev == named.st_dev == parent_info.st_dev
                and opened.st_ino == named.st_ino == parent_info.st_ino
                and opened.st_uid == named.st_uid == os.getuid()
                and stat.S_IMODE(opened.st_mode)
                == stat.S_IMODE(named.st_mode)
                == 0o700
                and locks.bindings() == lock_bindings,
                "directory-install probe authority changed",
            )

        listed_names = set(os.listdir(descriptor))
        foreign_probe_names = {
            name
            for name in listed_names
            if name.startswith(f".{probe_label}.install-probe-")
            and name not in probe_names
        }
        require(
            not foreign_probe_names,
            "directory-install probe namespace contains a foreign entry",
        )
        baseline = listed_names - probe_names
        if expected_baseline is not None:
            require(
                baseline == expected_baseline,
                "directory-install probe baseline namespace differs",
            )
        # Entries which this invocation creates or renames must retain that
        # exact inode through cleanup.  An exact pre-existing single residue
        # is separately handled as the protocol's crash prefix, but it may
        # not replace a probe already observed in this invocation.
        created_probe_identities: dict[str, tuple[int, int, int, int, int, int]] = {}

        def remove_probe_residue() -> None:
            present = set(os.listdir(descriptor)) & probe_names
            require(
                len(present) <= 1,
                "directory-install probe has ambiguous crash residue",
            )
            for name in present:
                require_bindings()
                expected_before = baseline | {name}
                require(
                    set(os.listdir(descriptor)) == expected_before,
                    "directory-install probe residue namespace differs",
                )
                listed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                require(
                    stat.S_ISDIR(listed.st_mode)
                    and listed.st_uid == os.getuid()
                    and listed.st_nlink == 2
                    and stat.S_IMODE(listed.st_mode) == 0o700,
                    "directory-install probe residue identity differs",
                )
                expected_identity = created_probe_identities.get(name)
                require(
                    expected_identity is None
                    or _file_identity(listed) == expected_identity,
                    "directory-install probe created residue was replaced",
                )
                child = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
                try:
                    opened_child = os.fstat(child)
                    require(
                        (opened_child.st_dev, opened_child.st_ino)
                        == (listed.st_dev, listed.st_ino)
                        and opened_child.st_uid == listed.st_uid == os.getuid()
                        and opened_child.st_nlink == listed.st_nlink == 2
                        and stat.S_IMODE(opened_child.st_mode)
                        == stat.S_IMODE(listed.st_mode)
                        == 0o700
                        and os.listdir(child) == [],
                        "directory-install probe residue is not empty",
                    )
                    os.fsync(child)
                    rebound_child = os.fstat(child)
                    rebound_named = os.stat(
                        name, dir_fd=descriptor, follow_symlinks=False
                    )
                    require(
                        (rebound_child.st_dev, rebound_child.st_ino)
                        == (rebound_named.st_dev, rebound_named.st_ino)
                        == (opened_child.st_dev, opened_child.st_ino)
                        and rebound_child.st_uid
                        == rebound_named.st_uid
                        == os.getuid()
                        and rebound_child.st_nlink
                        == rebound_named.st_nlink
                        == 2
                        and stat.S_IMODE(rebound_child.st_mode)
                        == stat.S_IMODE(rebound_named.st_mode)
                        == 0o700
                        and os.listdir(child) == []
                        and set(os.listdir(descriptor)) == expected_before,
                        "directory-install probe residue changed before removal",
                    )
                    require(
                        expected_identity is None
                        or _file_identity(rebound_child) == expected_identity,
                        "directory-install probe created residue binding changed",
                    )
                    require_bindings()
                    os.rmdir(name, dir_fd=descriptor)
                    require(
                        os.fstat(child).st_nlink == 0,
                        "directory-install probe residue remains linked",
                    )
                    os.fsync(descriptor)
                    require(
                        set(os.listdir(descriptor)) == baseline
                        and name not in set(os.listdir(descriptor)),
                        "directory-install probe residue survived removal",
                    )
                    require_bindings()
                finally:
                    os.close(child)

        require_bindings()
        remove_probe_residue()
        require(
            set(os.listdir(descriptor)) == baseline,
            "directory-install probe baseline namespace changed",
        )
        os.mkdir(source_name, mode=0o700, dir_fd=descriptor)
        source_info = os.stat(
            source_name, dir_fd=descriptor, follow_symlinks=False
        )
        created_probe_identities[source_name] = _file_identity(source_info)
        os.fsync(descriptor)
        require_bindings()
        try:
            _renameat2_noreplace(descriptor, source_name, target_name)
        except FileExistsError:
            raise
        except OSError as exc:
            if exc.errno not in RENAME_NOREPLACE_FALLBACK_ERRNOS:
                raise
            current = os.stat(
                source_name, dir_fd=descriptor, follow_symlinks=False
            )
            require(
                _file_identity(current) == _file_identity(source_info)
                and target_name not in set(os.listdir(descriptor)),
                "directory-install unsupported probe result differs",
            )
            method = f"{SOURCE_INSTALL_METHOD_FALLBACK_PREFIX}{exc.errno}"
        else:
            installed = os.stat(
                target_name, dir_fd=descriptor, follow_symlinks=False
            )
            require(
                _file_identity(installed) == _file_identity(source_info)
                and source_name not in set(os.listdir(descriptor)),
                "directory-install primary probe result differs",
            )
            created_probe_identities.pop(source_name, None)
            created_probe_identities[target_name] = _file_identity(installed)
            method = SOURCE_INSTALL_METHOD_PRIMARY
        require_bindings()
        remove_probe_residue()
        require(
            set(os.listdir(descriptor)) == baseline,
            "directory-install probe did not restore its namespace",
        )
        require_bindings()
        return method
    finally:
        os.close(descriptor)


def _legacy_staged_seal_json(path: Path, value: Mapping[str, Any]) -> str:
    """Atomically append one immutable JSON object through a pinned FD graph."""

    require(
        path.is_absolute()
        and path.name not in {"", ".", ".."}
        and path.parent != path,
        "immutable repair artifact path differs",
    )
    payload = (
        json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    parent = _directory(path.parent, "immutable repair artifact parent")
    parent_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
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
            stat.S_ISDIR(parent_info.st_mode)
            and same_inode(parent_info, named_parent)
            and parent_info.st_uid == os.getuid()
            and stat.S_IMODE(parent_info.st_mode) == 0o700,
            "immutable repair artifact parent identity differs",
        )
        initial_names = set(os.listdir(parent_fd))
        baseline_names = initial_names - {target_name, stage_name}
        require(
            not {
                name
                for name in baseline_names
                if name.startswith(".") and name.endswith(".seal.tmp")
            },
            "immutable repair artifact namespace contains a foreign stage",
        )

        def listed(name: str) -> os.stat_result | None:
            try:
                return os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                return None

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
                "immutable repair artifact parent/namespace binding changed",
            )

        def open_bound(name: str) -> tuple[int, os.stat_result]:
            named = listed(name)
            require(named is not None, f"immutable repair artifact vanished: {name}")
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
                f"immutable repair artifact identity differs: {name}",
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
                f"immutable repair artifact differs: {name}",
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
                "immutable repair artifact target/staging state differs",
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
                "immutable repair artifact recovered staging identity changed",
            )
            os.fsync(parent_fd)
            require_namespace({target_name})
            read_exact(target_fd, target_name, expected_mode=0o444, expected_nlink=1)
            return digest
        if target_info is not None:
            require(stage_info is None, "immutable repair artifact staging differs")
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
                "immutable repair artifact staging identity differs",
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
                    "immutable repair artifact partial staging changed",
                )
                require_namespace({stage_name})
                os.unlink(stage_name, dir_fd=parent_fd)
                require(
                    os.fstat(stage_fd).st_nlink == 0,
                    "immutable repair artifact partial staging remains linked",
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
                "immutable repair artifact staging creation differs",
            )
            require_namespace({stage_name})
            view = memoryview(payload)
            while view:
                written = os.write(stage_fd, view)
                require(written > 0, f"short repair artifact write: {path}")
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
            "immutable repair artifact linked staging identity changed",
        )
        os.fsync(parent_fd)
        require_namespace({target_name})
        read_exact(target_fd, target_name, expected_mode=0o444, expected_nlink=1)
        require(listed(stage_name) is None, "immutable repair artifact staging remains")
        return digest
    finally:
        if target_fd is not None:
            os.close(target_fd)
        if stage_fd is not None:
            os.close(stage_fd)
        os.close(parent_fd)


def seal_json(path: Path, value: Mapping[str, Any]) -> str:
    """Append one immutable JSON object directly at its final name.

    Attempt 2 is fail-stop for local filesystem crashes.  No visible staging
    pathname is ever created or removed.  A crash while the final inode is
    mode 0600 leaves permanent stop evidence; recovery must not resume, chmod,
    unlink, or replace it.
    """

    require(
        path.is_absolute()
        and path.name not in {"", ".", ".."}
        and path.parent != path,
        "immutable repair artifact path differs",
    )
    payload = (
        json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    parent = _directory(path.parent, "immutable repair artifact parent")
    parent_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    descriptor = -1
    try:
        parent_info = os.fstat(parent_fd)
        named_parent = parent.lstat()
        names = set(os.listdir(parent_fd))
        require(
            stat.S_ISDIR(parent_info.st_mode)
            and _file_identity(parent_info) == _file_identity(named_parent)
            and parent_info.st_uid == named_parent.st_uid == os.getuid()
            and stat.S_IMODE(parent_info.st_mode)
            == stat.S_IMODE(named_parent.st_mode)
            == 0o700
            and not {
                name
                for name in names
                if name.startswith(".") and name.endswith(".seal.tmp")
            },
            "immutable repair artifact parent/fail-stop namespace differs",
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
                "immutable repair artifact parent/namespace binding changed",
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
                "immutable repair final artifact identity differs",
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
                "immutable repair final artifact bytes differ",
            )
            os.fsync(descriptor)
            os.fsync(parent_fd)
            require_parent(names)
            require(
                read_bound(0o444) == payload,
                "immutable repair final artifact changed",
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
            "immutable repair final artifact creation differs",
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, f"short immutable repair artifact write: {path}")
            view = view[written:]
        os.fsync(descriptor)
        require_parent(baseline | {path.name})
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        require(
            read_bound(0o444) == payload,
            "immutable repair final artifact seal differs",
        )
        os.fsync(parent_fd)
        require_parent(baseline | {path.name})
        require(
            read_bound(0o444) == payload,
            "immutable repair final artifact changed after parent fsync",
        )
        return digest
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


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
        pass_fds=_ACTIVE_RUN_PASS_FDS.get(),
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


def _contract_scheduler_environment(
    contract: Mapping[str, Any],
) -> dict[str, str]:
    control_plane = contract.get("scheduler_control_plane_contract")
    require(
        isinstance(control_plane, Mapping)
        and isinstance(control_plane.get("slurm_conf"), str)
        and bool(control_plane["slurm_conf"]),
        "repair scheduler control-plane contract differs",
    )
    return _scheduler_environment(control_plane["slurm_conf"])


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
    *,
    pass_fds: Sequence[int] = (),
) -> tuple[CommandResult, dict[str, Any]]:
    active_transition = _ACTIVE_REPAIR_TRANSITION.get()
    if active_transition is not None:
        active_transition.revalidate()
    before = locks.bindings()
    normalized_pass_fds = tuple(pass_fds)
    require(
        all(type(item) is int and item >= 0 for item in normalized_pass_fds)
        and len(set(normalized_pass_fds)) == len(normalized_pass_fds),
        "repair scheduler inherited descriptor set differs",
    )
    pass_token = _ACTIVE_RUN_PASS_FDS.set(normalized_pass_fds)
    try:
        result = runner(tuple(argv), cwd, environment)
    finally:
        _ACTIVE_RUN_PASS_FDS.reset(pass_token)
    require(locks.bindings() == before, "repair scheduler-call lock lease changed")
    if active_transition is not None:
        active_transition.revalidate()
    require(
        type(result.returncode) is int
        and isinstance(result.stdout, bytes)
        and isinstance(result.stderr, bytes),
        "repair scheduler result types differ",
    )
    return result, _command_evidence(argv, environment, result)


def _load_module_bytes(name: str, path: Path, payload: bytes) -> ModuleType:
    require(
        isinstance(payload, bytes) and bool(payload),
        f"module {name} bytes differ",
    )
    digest = hashlib.sha256(payload).hexdigest()
    require(SHA256_RE.fullmatch(digest) is not None, f"module {name} hash differs")
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


def _load_module(name: str, path: Path) -> ModuleType:
    payload, digest, info = _regular_bytes(path, f"module {name}")
    require(
        stat.S_IMODE(info.st_mode) in {0o444, 0o644}
        and bool(payload)
        and SHA256_RE.fullmatch(digest) is not None,
        f"module {name} identity differs",
    )
    return _load_module_bytes(name, path, payload)


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
    if "_0002_" in name:
        return submission_root / name
    return submission_root / "journal" / name


def _repair_root(submission_root: Path) -> Path:
    return submission_root / "report-repair" / "attempt-0002"


def _repair_source_root(submission_root: Path) -> Path:
    # Historical name retained in the journal API.  Attempt 2 is one sealed
    # regular archive directly under the already-bound submission root.
    return submission_root / SOURCE_ARCHIVE_NAME


def _repair_name(submission_sha256: str) -> str:
    return f"exp23-launch8-{submission_sha256[:16]}-report-repair-0002"


def _repair_comment(submission_sha256: str) -> str:
    return f"treewm-exp23-report-repair:{submission_sha256}:0002"


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


def _validated_original_git_provenance(value: object) -> dict[str, str]:
    expected = EXPECTED_ORIGINAL_GIT_PROVENANCE
    require(
        isinstance(value, Mapping)
        and set(value) == set(expected)
        and all(type(value[key]) is str for key in expected)
        and exact_json_equal(value, expected),
        "original submission git provenance differs",
    )
    return {key: value[key] for key in expected}


def _validate_original_submission(
    submission_root: Path,
    submission_sha256: str,
    *,
    report_program: Path | tuple[Path, bytes],
    locks: _RepairLocks | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Authenticate original snapshot/receipt/all cells and exact deterministic assembly."""

    require(
        submission_root == CANONICAL_PRODUCTION_SUBMISSION_ROOT,
        "report repair submission root differs",
    )
    require(submission_sha256 == EXPECTED_SUBMISSION_SHA256, "report repair submission SHA differs")
    transition = _ACTIVE_REPAIR_TRANSITION.get()
    if transition is not None:
        require(
            transition.submission_root == submission_root,
            "original submission retained authority root differs",
        )

        def retained_json(
            name: str, label: str
        ) -> tuple[dict[str, Any], str, os.stat_result]:
            path = submission_root / name
            payload = transition._retained_payload(path)
            row = transition._retained_file_row(path)
            info = os.fstat(row[2])
            require(
                _file_identity(info) == row[3]
                and info.st_size == len(payload),
                f"{label} retained identity differs",
            )
            return (
                _decode_json(path, payload),
                hashlib.sha256(payload).hexdigest(),
                info,
            )

        contract, contract_digest, contract_info = retained_json(
            "SUBMISSION_CONTRACT.json", "original submission contract"
        )
    else:
        _directory(submission_root, "original submission root")
        contract, contract_digest, contract_info = read_json(
            submission_root / "SUBMISSION_CONTRACT.json",
            "original submission contract",
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
    _validated_original_git_provenance(contract.get("git_provenance"))
    snapshot_root = Path(str(contract.get("snapshot_root", "")))
    require(
        snapshot_root == submission_root / "source-snapshot" / "repo",
        "original snapshot root differs",
    )
    if transition is None:
        _directory(snapshot_root, "original source snapshot")
        receipt, receipt_digest, receipt_info = read_json(
            submission_root / "SUBMISSION_RECEIPT.json",
            "original submission receipt",
        )
    else:
        receipt, receipt_digest, receipt_info = retained_json(
            "SUBMISSION_RECEIPT.json", "original submission receipt"
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
    if transition is None:
        authorization_row = _json_file_row(
            submission_root / "SUBMISSION_AUTHORIZATION.json",
            "original submission authorization",
        )
    else:
        authorization_path = submission_root / "SUBMISSION_AUTHORIZATION.json"
        authorization_payload = transition._retained_payload(authorization_path)
        _decode_json(authorization_path, authorization_payload)
        authorization_record = transition._retained_file_row(authorization_path)
        authorization_info = os.fstat(authorization_record[2])
        require(
            _file_identity(authorization_info) == authorization_record[3]
            and authorization_info.st_uid == os.getuid()
            and authorization_info.st_nlink == 1
            and stat.S_IMODE(authorization_info.st_mode) == 0o444,
            "original submission authorization retained identity differs",
        )
        authorization_row = {
            "mode": 0o444,
            "size": len(authorization_payload),
            "sha256": hashlib.sha256(authorization_payload).hexdigest(),
        }
    require(
        authorization_row["sha256"] == EXPECTED_AUTHORIZATION_RAW_SHA256,
        "original submission authorization hash differs",
    )
    receipt_map = _worker_receipt_map(submission_root)
    if transition is not None:
        # Install the immutable production expectation on the *already-live*
        # binding before report code, scientific inputs, or scheduler evidence
        # is consumed.  No later transition may recapture a clean pathname and
        # bless a different inode graph.
        _bind_transition_authority_expectation(
            transition.locks,
            submission_root=submission_root,
            submission_sha256=submission_sha256,
            contract=contract,
            receipt_map=receipt_map,
        )
        installed = getattr(
            transition.locks, "_transition_authority_expectation", None
        )
        require(
            isinstance(installed, Mapping),
            "retained production authority expectation is absent",
        )
        transition.install_authority_expectation(installed)
    if isinstance(report_program, tuple):
        report_program_path, report_program_bytes = report_program
        require(
            isinstance(report_program_path, Path)
            and isinstance(report_program_bytes, bytes),
            "sealed repaired report program differs",
        )
        repaired_report = _load_module_bytes(
            "sealed_repaired_report", report_program_path, report_program_bytes
        )
    else:
        repaired_report = _load_module("sealed_repaired_report", report_program)
    require(
        transition is not None
        and transition.submission_root == submission_root,
        "controller report assembly lacks its retained authority graph",
    )
    publication_binding = repaired_report._ACTIVE_REPAIR_PUBLICATION_BINDING
    binding_token = publication_binding.set(transition)
    try:
        with repaired_report._retained_snapshot_imports(
            transition, snapshot_root
        ):
            bundle, decision, provenance = repaired_report.assemble_report(
                snapshot_root,
                submission_root,
                submission_sha256,
                allow_repair_cleanup_for_audit=True,
            )
            transition.revalidate()
    finally:
        publication_binding.reset(binding_token)
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
    if locks is not None and transition is None:
        _bind_transition_authority_expectation(
            locks,
            submission_root=submission_root,
            submission_sha256=submission_sha256,
            contract=contract,
            receipt_map=receipt_map,
        )
    return contract, receipt, receipt_map, _expected_reassembly()


def _normalized_snapshot_inventory(contract: Mapping[str, Any]) -> dict[str, str]:
    inventory = contract.get("snapshot_inventory")
    require(
        isinstance(inventory, Mapping) and bool(inventory),
        "report repair snapshot inventory is absent",
    )
    normalized: dict[str, str] = {}
    for raw_name, raw_digest in inventory.items():
        require(
            isinstance(raw_name, str)
            and isinstance(raw_digest, str)
            and SHA256_RE.fullmatch(raw_digest) is not None,
            "report repair snapshot inventory row differs",
        )
        relative = Path(raw_name)
        require(
            not relative.is_absolute()
            and relative.parts
            and all(part not in {"", ".", ".."} for part in relative.parts)
            and relative.as_posix() == raw_name
            and raw_name not in normalized,
            f"report repair snapshot inventory path differs: {raw_name}",
        )
        normalized[raw_name] = raw_digest
    require(
        stable_hash(normalized) == contract.get("snapshot_inventory_sha256")
        == EXPECTED_SNAPSHOT_INVENTORY_SHA256,
        "report repair snapshot inventory hash differs",
    )
    return normalized


def _bind_transition_authority_expectation(
    locks: _RepairLocks,
    *,
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    receipt_map: Mapping[str, Any],
) -> None:
    """Carry the validated immutable production authority into every transition."""

    normalized = _normalized_snapshot_inventory(contract)
    expectation = {
        "submission_root": str(submission_root),
        "submission_contract_sha256": submission_sha256,
        "submission_receipt_sha256": EXPECTED_RECEIPT_RAW_SHA256,
        "submission_authorization_sha256": EXPECTED_AUTHORIZATION_RAW_SHA256,
        "snapshot_root": str(submission_root / "source-snapshot/repo"),
        "snapshot_inventory": normalized,
        "snapshot_inventory_sha256": stable_hash(normalized),
        "worker_receipt_map": dict(receipt_map),
        "worker_receipt_map_sha256": stable_hash(receipt_map),
        "attempt1_log_sha256": EXPECTED_ATTEMPT1_LOG_SHA256,
        "attempt1_log_size": len(EXPECTED_ATTEMPT1_LOG_BYTES),
    }
    prior = getattr(locks, "_transition_authority_expectation", None)
    require(
        prior is None or exact_json_equal(prior, expectation),
        "report repair retained authority expectation changed",
    )
    setattr(locks, "_transition_authority_expectation", expectation)


def _secure_snapshot_inventory_for_transition(snapshot_root: Path) -> dict[str, str]:
    """Read one sealed snapshot through nofollow directory descriptors.

    The surrounding transition retains the same tree descriptors.  This pass
    authenticates exact content and read-only topology; the retained binding
    then proves that the authenticated objects remain the named objects for the
    lifetime of scheduler evidence and its successor append.
    """

    root_fd = os.open(
        snapshot_root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    rows: dict[str, str] = {}
    directories: set[str] = set()

    def walk(descriptor: int, prefix: Path) -> None:
        opened = os.fstat(descriptor)
        names = tuple(sorted(os.listdir(descriptor)))
        require(
            stat.S_ISDIR(opened.st_mode)
            and opened.st_uid == os.getuid()
            and opened.st_nlink >= 2
            and opened.st_mode & 0o222 == 0,
            f"retained snapshot directory identity differs: {prefix}",
        )
        for name in names:
            require(
                name not in {"", ".", ".."} and "/" not in name,
                "retained snapshot entry name differs",
            )
            listed = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            relative = prefix / name if prefix.parts else Path(name)
            if stat.S_ISDIR(listed.st_mode):
                child_fd = os.open(
                    name,
                    os.O_RDONLY
                    | getattr(os, "O_DIRECTORY", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    dir_fd=descriptor,
                )
                try:
                    child_opened = os.fstat(child_fd)
                    require(
                        _file_identity(child_opened) == _file_identity(listed),
                        f"retained snapshot directory raced: {relative}",
                    )
                    directories.add(relative.as_posix())
                    walk(child_fd, relative)
                    require(
                        _file_identity(os.fstat(child_fd))
                        == _file_identity(child_opened)
                        and _file_identity(
                            os.stat(
                                name,
                                dir_fd=descriptor,
                                follow_symlinks=False,
                            )
                        )
                        == _file_identity(child_opened),
                        f"retained snapshot directory changed: {relative}",
                    )
                finally:
                    os.close(child_fd)
                continue
            require(
                stat.S_ISREG(listed.st_mode)
                and listed.st_uid == os.getuid()
                and listed.st_nlink == 1
                and listed.st_mode & 0o222 == 0,
                f"retained snapshot file identity differs: {relative}",
            )
            file_fd = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=descriptor,
            )
            try:
                file_opened = os.fstat(file_fd)
                require(
                    _file_identity(file_opened) == _file_identity(listed),
                    f"retained snapshot file raced: {relative}",
                )
                digest = hashlib.sha256()
                total = 0
                while chunk := os.read(file_fd, 16 * 1024 * 1024):
                    digest.update(chunk)
                    total += len(chunk)
                require(
                    total == file_opened.st_size
                    and _file_identity(os.fstat(file_fd))
                    == _file_identity(file_opened)
                    and _file_identity(
                        os.stat(
                            name,
                            dir_fd=descriptor,
                            follow_symlinks=False,
                        )
                    )
                    == _file_identity(file_opened),
                    f"retained snapshot file changed: {relative}",
                )
                rows[relative.as_posix()] = digest.hexdigest()
            finally:
                os.close(file_fd)
        require(
            _file_identity(os.fstat(descriptor)) == _file_identity(opened)
            and tuple(sorted(os.listdir(descriptor))) == names,
            f"retained snapshot namespace changed: {prefix}",
        )

    try:
        named_root = snapshot_root.lstat()
        opened_root = os.fstat(root_fd)
        require(
            _file_identity(opened_root) == _file_identity(named_root),
            "retained snapshot root raced",
        )
        walk(root_fd, Path())
        require(
            _file_identity(os.fstat(root_fd)) == _file_identity(opened_root)
            and _file_identity(snapshot_root.lstat())
            == _file_identity(opened_root),
            "retained snapshot root changed",
        )
    finally:
        os.close(root_fd)

    expected_directories = {
        parent.as_posix()
        for raw_relative in rows
        for parent in list(Path(raw_relative).parents)[:-1]
    }
    require(
        directories == expected_directories,
        "retained snapshot directory coverage differs",
    )
    return rows


def _publication_state(submission_root: Path) -> dict[str, Any]:
    journal = submission_root / "journal"
    cleanup = [
        "CANCEL_REQUESTED.json",
        "journal/PREREQUISITE_MISSING.json",
        "journal/9000_RECOVERY_CANCELLED.json",
        "journal/9001_PRODUCTION_PREREQUISITE_MISSING.json",
    ]
    active = _ACTIVE_REPAIR_TRANSITION.get()
    if active is not None and active.submission_root == submission_root:
        root_row = next(
            row for row in active.directory_rows if row[0] == submission_root
        )
        journal_row = next(
            row for row in active.directory_rows if row[0] == journal
        )
        root_names = set(root_row[3])
        journal_names = set(journal_row[3])
        present_cleanup = [
            name
            for name in cleanup
            if (
                name in root_names
                if "/" not in name
                else Path(name).name in journal_names
            )
        ]
        staging = sorted(
            name
            for name in root_names
            if name == "report"
            or name.startswith(".report")
            or name.startswith(PUBLICATION_ARCHIVE_PREFIX)
        )
    else:
        present_cleanup = [
            name
            for name in cleanup
            if os.path.lexists(submission_root / name)
        ]
        staging = sorted(
            name
            for name in os.listdir(submission_root)
            if name == "report"
            or name.startswith(".report")
            or name.startswith(PUBLICATION_ARCHIVE_PREFIX)
        )
    require(
        not present_cleanup
        and not staging,
        "report publication/cancellation/staging state is not fresh for repair",
    )
    return {
        "report_absent": True,
        "staging_entries": [],
        "cleanup_prefixes": [],
        "journal_directory": str(journal),
    }


def _require_report_install_probe_namespace(
    submission_root: Path,
    *,
    allow_exact_crash_residue: bool,
) -> None:
    active = _ACTIVE_REPAIR_TRANSITION.get()
    if active is not None and active.submission_root == submission_root:
        names = set(
            next(
                row[3]
                for row in active.directory_rows
                if row[0] == submission_root
            )
        )
    else:
        names = set(os.listdir(submission_root))
    present = {
        name
        for name in names
        if name == "report" or name.startswith(".report")
    }
    require(
        not present and type(allow_exact_crash_residue) is bool,
        "report install-probe residue is permanent fail-stop evidence",
    )


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


def _validated_scheduler_census(
    census: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
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
    environment = _contract_scheduler_environment(contract)
    owner = environment["USER"]
    expected_argv = [
        "/usr/local/bin/squeue",
        "--noheader",
        f"--user={owner}",
        f"--format={SQUEUE_FORMAT}",
    ]
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
        result = _validated_command_evidence(
            raw,
            label=f"report repair scheduler census round {index}",
            expected_argv=expected_argv,
            expected_environment=environment,
        )
        rows = _parse_squeue_rows(result)
        relevant = [
            row
            for row in rows
            if row["job_name"].startswith("exp23-launch8-")
            or row["comment"].startswith("treewm-exp23")
        ]
        require(
            all(row["owner"] == owner for row in relevant)
            and exact_json_equal(observation["relevant_rows"], relevant),
            f"report repair scheduler census parsed/raw rows differ at round {index}",
        )
        reconstructed.append(relevant)
    require(
        exact_json_equal(reconstructed[-2], reconstructed[-1])
        and exact_json_equal(census["settled_rows"], reconstructed[-1]),
        "report repair scheduler census did not settle exactly",
    )
    return dict(census)


def _repair_rows(
    census: Mapping[str, Any],
    submission_sha256: str,
    contract: Mapping[str, Any],
) -> list[dict[str, str]]:
    _validated_scheduler_census(census, contract)
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
        not _repair_rows(census, submission_sha256, contract)
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
        "attempt": PREDECESSOR_ATTEMPT,
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
        and value.get("attempt") == PREDECESSOR_ATTEMPT
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
    _validated_scheduler_census(pre, contract)
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


def _source_with_installation_method(
    source: Mapping[str, Any], installation_method: str
) -> dict[str, Any]:
    require(
        set(source) == SOURCE_AUTHORITY_V1_KEYS
        and source.get("schema_version") == 1,
        "uninstalled repair source identity differs",
    )
    return {
        **dict(source),
        "schema_version": 2,
        "repair_source_installation_method": installation_method,
    }


def _source_base_matches(
    sealed: Mapping[str, Any], source: Mapping[str, Any]
) -> bool:
    return exact_json_equal(
        {
            "schema_version": 1,
            "repair_source_commit": sealed.get("repair_source_commit"),
            "repair_package_protocol_sha256": sealed.get(
                "repair_package_protocol_sha256"
            ),
            "repair_source_files": sealed.get("repair_source_files"),
            "repair_source_files_sha256": sealed.get(
                "repair_source_files_sha256"
            ),
        },
        source,
    )


def _source_authority_from_bound_value(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "repair_source_commit": value.get("repair_source_commit"),
        "repair_package_protocol_sha256": value.get(
            "repair_package_protocol_sha256"
        ),
        "repair_source_files": value.get("repair_source_files"),
        "repair_source_files_sha256": value.get("repair_source_files_sha256"),
        "repair_source_installation_method": value.get(
            "repair_source_installation_method"
        ),
        "repair_source_archive": value.get("repair_source_archive"),
        "repair_source_archive_sha256": value.get(
            "repair_source_archive_sha256"
        ),
        "repair_source_archive_size": value.get("repair_source_archive_size"),
        "repair_source_archive_format": value.get(
            "repair_source_archive_format"
        ),
    }


def _legacy_cleanup_write_sealed_file(path: Path, payload: bytes) -> None:
    require(
        path.is_absolute() and path.name not in {"", ".", ".."},
        "repair source seal path differs",
    )
    parent = _directory(path.parent, "repair source seal parent")
    parent_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    descriptor = -1
    try:
        parent_info = os.fstat(parent_fd)
        named_parent = parent.lstat()
        baseline_names = set(os.listdir(parent_fd))
        require(
            stat.S_ISDIR(parent_info.st_mode)
            and (parent_info.st_dev, parent_info.st_ino)
            == (named_parent.st_dev, named_parent.st_ino)
            and parent_info.st_uid == named_parent.st_uid == os.getuid()
            and stat.S_IMODE(parent_info.st_mode)
            == stat.S_IMODE(named_parent.st_mode)
            == 0o700
            and path.name not in baseline_names,
            "repair source seal parent/namespace differs",
        )

        def require_parent(expected_names: set[str]) -> None:
            opened_parent = os.fstat(parent_fd)
            current_parent = parent.lstat()
            require(
                (opened_parent.st_dev, opened_parent.st_ino)
                == (current_parent.st_dev, current_parent.st_ino)
                == (parent_info.st_dev, parent_info.st_ino)
                and opened_parent.st_uid
                == current_parent.st_uid
                == os.getuid()
                and stat.S_IMODE(opened_parent.st_mode)
                == stat.S_IMODE(current_parent.st_mode)
                == 0o700
                and set(os.listdir(parent_fd)) == expected_names,
                "repair source seal parent/namespace binding changed",
            )

        require_parent(baseline_names)
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
        created = os.fstat(descriptor)
        named_created = os.stat(
            path.name, dir_fd=parent_fd, follow_symlinks=False
        )
        require(
            stat.S_ISREG(created.st_mode)
            and (created.st_dev, created.st_ino)
            == (named_created.st_dev, named_created.st_ino)
            and created.st_uid == named_created.st_uid == os.getuid()
            and created.st_nlink == named_created.st_nlink == 1
            and stat.S_IMODE(created.st_mode)
            == stat.S_IMODE(named_created.st_mode)
            == 0o600,
            "repair source seal creation differs",
        )
        require_parent(baseline_names | {path.name})
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
        named = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        require(
            bytes(readback) == payload
            and os.read(descriptor, 1) == b""
            and _file_identity(info) == _file_identity(named)
            and info.st_uid == os.getuid()
            and info.st_nlink == 1
            and stat.S_IMODE(info.st_mode) == 0o444,
            f"repair source seal differs: {path}",
        )
        require_parent(baseline_names | {path.name})
        os.fsync(parent_fd)
        require_parent(baseline_names | {path.name})
    except BaseException as exc:
        if descriptor >= 0:
            try:
                opened = os.fstat(descriptor)
                named = os.stat(
                    path.name, dir_fd=parent_fd, follow_symlinks=False
                )
                require(
                    stat.S_ISREG(opened.st_mode)
                    and (opened.st_dev, opened.st_ino)
                    == (named.st_dev, named.st_ino)
                    and opened.st_uid == named.st_uid == os.getuid()
                    and opened.st_nlink == named.st_nlink == 1
                    and stat.S_IMODE(opened.st_mode) in {0o600, 0o444}
                    and set(os.listdir(parent_fd))
                    == baseline_names | {path.name},
                    "repair source failed-seal cleanup is ambiguous",
                )
                os.fchmod(descriptor, 0o600)
                os.fsync(descriptor)
                rebound = os.fstat(descriptor)
                rebound_named = os.stat(
                    path.name, dir_fd=parent_fd, follow_symlinks=False
                )
                require(
                    (rebound.st_dev, rebound.st_ino)
                    == (rebound_named.st_dev, rebound_named.st_ino)
                    == (opened.st_dev, opened.st_ino)
                    and rebound.st_nlink == rebound_named.st_nlink == 1
                    and stat.S_IMODE(rebound.st_mode)
                    == stat.S_IMODE(rebound_named.st_mode)
                    == 0o600
                    and set(os.listdir(parent_fd))
                    == baseline_names | {path.name},
                    "repair source failed-seal cleanup changed",
                )
                require_parent(baseline_names | {path.name})
                os.unlink(path.name, dir_fd=parent_fd)
                require(
                    os.fstat(descriptor).st_nlink == 0,
                    "repair source failed-seal inode remains linked",
                )
                os.fsync(parent_fd)
                require_parent(baseline_names)
            except BaseException as cleanup_exc:
                raise RepairError(
                    f"repair source failed-seal cleanup refused ambiguity: {cleanup_exc}"
                ) from exc
        raise
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _write_sealed_file(path: Path, payload: bytes) -> None:
    """Create one final source file; failures deliberately leave stop evidence."""

    require(
        path.is_absolute() and path.name not in {"", ".", ".."},
        "repair source seal path differs",
    )
    parent = _directory(path.parent, "repair source seal parent")
    parent_fd = os.open(
        parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    descriptor = -1
    try:
        parent_info = os.fstat(parent_fd)
        named_parent = parent.lstat()
        baseline = set(os.listdir(parent_fd))
        require(
            stat.S_ISDIR(parent_info.st_mode)
            and _file_identity(parent_info) == _file_identity(named_parent)
            and parent_info.st_uid == named_parent.st_uid == os.getuid()
            and stat.S_IMODE(parent_info.st_mode)
            == stat.S_IMODE(named_parent.st_mode)
            == 0o700
            and path.name not in baseline,
            "repair source final-file parent/namespace differs",
        )

        def require_parent(expected: set[str]) -> None:
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
                and set(os.listdir(parent_fd)) == expected,
                "repair source final-file parent binding changed",
            )

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
        expected_names = baseline | {path.name}
        require_parent(expected_names)
        created = os.fstat(descriptor)
        named_created = os.stat(
            path.name, dir_fd=parent_fd, follow_symlinks=False
        )
        require(
            stat.S_ISREG(created.st_mode)
            and _file_identity(created) == _file_identity(named_created)
            and created.st_uid == named_created.st_uid == os.getuid()
            and created.st_nlink == named_created.st_nlink == 1
            and stat.S_IMODE(created.st_mode)
            == stat.S_IMODE(named_created.st_mode)
            == 0o600,
            "repair source final-file creation differs",
        )
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            require(written > 0, f"short repair source write: {path}")
            view = view[written:]
        os.fsync(descriptor)
        require_parent(expected_names)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
        observed = bytearray()
        offset = 0
        while True:
            chunk = os.pread(descriptor, 1024 * 1024, offset)
            if not chunk:
                break
            observed.extend(chunk)
            offset += len(chunk)
        sealed = os.fstat(descriptor)
        named_sealed = os.stat(
            path.name, dir_fd=parent_fd, follow_symlinks=False
        )
        require(
            bytes(observed) == payload
            and _file_identity(sealed) == _file_identity(named_sealed)
            and sealed.st_uid == os.getuid()
            and sealed.st_nlink == 1
            and stat.S_IMODE(sealed.st_mode) == 0o444,
            "repair source final-file seal differs",
        )
        os.fsync(parent_fd)
        require_parent(expected_names)
        require(
            _file_identity(os.fstat(descriptor))
            == _file_identity(
                os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
            ),
            "repair source final-file changed after parent fsync",
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(parent_fd)


def _legacy_validate_sealed_repair_source_directory(
    source_root: Path, source: Mapping[str, Any]
) -> None:
    root = _directory(source_root, "sealed repair source root")
    root_info = root.lstat()
    require(
        set(source) == SOURCE_AUTHORITY_V2_KEYS
        and type(source.get("schema_version")) is int
        and source.get("schema_version") == 2
        and isinstance(source.get("repair_source_commit"), str)
        and GIT_RE.fullmatch(source["repair_source_commit"]) is not None
        and isinstance(source.get("repair_package_protocol_sha256"), str)
        and SHA256_RE.fullmatch(source["repair_package_protocol_sha256"])
        is not None
        and _valid_installation_method(
            source.get("repair_source_installation_method")
        ),
        "sealed repair source identity differs",
    )
    require(
        root_info.st_uid == os.getuid()
        and root_info.st_nlink == 2
        and stat.S_IMODE(root_info.st_mode) == 0o555,
        "sealed repair source root identity/mode differs",
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
    authority, authority_sha256, authority_info = read_json(
        root / SOURCE_AUTHORITY_NAME, "sealed repair source authority"
    )
    require(
        stat.S_IMODE(authority_info.st_mode) == 0o444
        and authority_info.st_uid == os.getuid()
        and authority_info.st_nlink == 1
        and authority_info.st_size == _pretty_json_size(source)
        and authority_sha256 == _pretty_json_sha(source)
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


def _legacy_load_sealed_repair_source_directory(source_root: Path) -> dict[str, Any]:
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


def _legacy_remove_repair_source_staging(
    staging: Path,
    source: Mapping[str, Any],
    attempt_root: Path,
    locks: _RepairLocks,
) -> None:
    require(
        staging.parent == attempt_root
        and re.fullmatch(r"\.source\.tmp\.[1-9][0-9]*\.[1-9][0-9]*", staging.name)
        is not None,
        "repair source staging name/path differs",
    )
    attempt_fd = os.open(
        attempt_root,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    staging_fd = -1
    try:
        attempt_identity = os.fstat(attempt_fd)
        named_attempt = attempt_root.lstat()
        require(
            stat.S_ISDIR(attempt_identity.st_mode)
            and _file_identity(attempt_identity) == _file_identity(named_attempt)
            and attempt_identity.st_uid == os.getuid()
            and stat.S_IMODE(attempt_identity.st_mode) == 0o700,
            "repair source staging parent identity differs",
        )
        staging_fd = os.open(
            staging.name,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=attempt_fd,
        )
        staging_identity = os.fstat(staging_fd)
        named_staging = os.stat(
            staging.name, dir_fd=attempt_fd, follow_symlinks=False
        )
        mode = stat.S_IMODE(staging_identity.st_mode)
        initial_lock_bindings = locks.bindings()
        require(
            stat.S_ISDIR(staging_identity.st_mode)
            and _file_identity(staging_identity) == _file_identity(named_staging)
            and staging_identity.st_uid == os.getuid()
            and staging_identity.st_nlink == 2
            and mode in {0o700, 0o555},
            "repair source staging identity differs",
        )

        def rebind(*, staging_present: bool = True) -> None:
            opened_attempt = os.fstat(attempt_fd)
            current_attempt = attempt_root.lstat()
            require(
                (opened_attempt.st_dev, opened_attempt.st_ino)
                == (attempt_identity.st_dev, attempt_identity.st_ino)
                == (current_attempt.st_dev, current_attempt.st_ino)
                and opened_attempt.st_uid == current_attempt.st_uid == os.getuid()
                and stat.S_IMODE(opened_attempt.st_mode)
                == stat.S_IMODE(current_attempt.st_mode)
                == 0o700
                and exact_json_equal(locks.bindings(), initial_lock_bindings),
                "repair source staging parent/lock binding changed",
            )
            if staging_present:
                opened_staging = os.fstat(staging_fd)
                current_staging = os.stat(
                    staging.name, dir_fd=attempt_fd, follow_symlinks=False
                )
                require(
                    (opened_staging.st_dev, opened_staging.st_ino)
                    == (staging_identity.st_dev, staging_identity.st_ino)
                    == (current_staging.st_dev, current_staging.st_ino)
                    and opened_staging.st_uid
                    == current_staging.st_uid
                    == os.getuid()
                    and opened_staging.st_nlink == current_staging.st_nlink == 2
                    and stat.S_IMODE(opened_staging.st_mode)
                    == stat.S_IMODE(current_staging.st_mode)
                    in {0o700, 0o555},
                    "repair source staging binding changed",
                )
            else:
                try:
                    os.stat(staging.name, dir_fd=attempt_fd, follow_symlinks=False)
                except FileNotFoundError:
                    pass
                else:
                    raise RepairError("repair source staging survived removal")

        entry_descriptors: dict[str, int] = {}
        entry_payloads: dict[str, bytes] = {}
        entry_initial_infos: dict[str, os.stat_result] = {}

        def regular_bytes(name: str) -> tuple[bytes, os.stat_result]:
            rebind()
            listed = os.stat(name, dir_fd=staging_fd, follow_symlinks=False)
            if name in entry_descriptors:
                descriptor = entry_descriptors[name]
                opened = os.fstat(descriptor)
                initial = entry_initial_infos[name]
                payload = entry_payloads[name]
                observed = bytearray()
                offset = 0
                while offset < len(payload) + 1:
                    chunk = os.pread(
                        descriptor, len(payload) + 1 - offset, offset
                    )
                    if not chunk:
                        break
                    observed.extend(chunk)
                    offset += len(chunk)
                require(
                    _file_identity(opened)
                    == _file_identity(listed)
                    == _file_identity(initial)
                    and bytes(observed) == payload,
                    f"repair source retained staging entry changed: {name}",
                )
                return payload, opened
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
                raise RepairError(
                    f"repair source staging entry cannot be opened safely: {name}: {exc}"
                ) from exc
            keep_descriptor = False
            try:
                opened = os.fstat(descriptor)
                require(
                    stat.S_ISREG(opened.st_mode)
                    and _file_identity(opened) == _file_identity(listed)
                    and opened.st_uid == os.getuid()
                    and opened.st_nlink == 1
                    and stat.S_IMODE(opened.st_mode) in {0o600, 0o444},
                    f"repair source staging entry identity differs: {name}",
                )
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
                    _file_identity(after) == _file_identity(opened),
                    f"repair source staging entry changed: {name}",
                )
                payload = b"".join(chunks)
                entry_descriptors[name] = descriptor
                entry_payloads[name] = payload
                entry_initial_infos[name] = opened
                keep_descriptor = True
                return payload, opened
            finally:
                if not keep_descriptor:
                    os.close(descriptor)

        rebind()
        entry_names = set(os.listdir(staging_fd))
        allowed_names = {*SOURCE_NAMES, SOURCE_AUTHORITY_NAME}
        require(
            entry_names <= allowed_names,
            "repair source staging coverage differs",
        )
        effective_source: Mapping[str, Any] = source
        completed_authority = False
        if SOURCE_AUTHORITY_NAME in entry_names:
            authority_payload, authority_info = regular_bytes(SOURCE_AUTHORITY_NAME)
            if stat.S_IMODE(authority_info.st_mode) == 0o444:
                completed_authority = True
                require(
                    entry_names == allowed_names,
                    "repair source staging authority coverage differs",
                )
                effective_source = _decode_json(staging / SOURCE_AUTHORITY_NAME, authority_payload)
                require(
                    set(effective_source) == SOURCE_AUTHORITY_V2_KEYS
                    and type(effective_source.get("schema_version")) is int
                    and effective_source.get("schema_version") == 2
                    and _valid_installation_method(
                        effective_source.get("repair_source_installation_method")
                    )
                    and _source_base_matches(effective_source, source)
                    and authority_payload
                    == (
                        json.dumps(
                            dict(effective_source),
                            sort_keys=True,
                            indent=2,
                            allow_nan=False,
                        )
                        + "\n"
                    ).encode("utf-8"),
                    "repair source staging authority differs",
                )
        if mode == 0o555:
            require(
                entry_names == allowed_names,
                "repair source sealed staging coverage differs",
            )
        entry_modes: dict[str, int] = {}
        entry_identities: dict[str, tuple[int, int, int, int, int, int]] = {}
        for name in sorted(entry_names):
            payload, entry_info = regular_bytes(name)
            entry_mode = stat.S_IMODE(entry_info.st_mode)
            entry_modes[name] = entry_mode
            entry_identities[name] = _file_identity(entry_info)
            require(
                mode != 0o555 or entry_mode == 0o444,
                "repair source sealed staging entry mode differs",
            )
            if entry_mode == 0o600:
                continue
            if name == SOURCE_AUTHORITY_NAME:
                continue
            expected = effective_source["repair_source_files"][name]
            require(
                expected["mode"] == 0o444
                and expected["size"] == len(payload)
                and expected["sha256"] == hashlib.sha256(payload).hexdigest(),
                f"repair source staging differs: {name}",
            )
        require(
            not completed_authority
            or all(entry_mode == 0o444 for entry_mode in entry_modes.values()),
            "repair source completed authority has partial source entries",
        )

        def require_entry_names(expected: set[str]) -> None:
            rebind()
            observed = os.listdir(staging_fd)
            require(
                len(observed) == len(set(observed))
                and set(observed) == expected,
                "repair source staging namespace changed",
            )

        require_entry_names(entry_names)
        os.fchmod(staging_fd, 0o700)
        os.fsync(staging_fd)
        require_entry_names(entry_names)
        invalidation_order = [
            *([SOURCE_AUTHORITY_NAME] if SOURCE_AUTHORITY_NAME in entry_names else []),
            *(name for name in sorted(entry_names) if name != SOURCE_AUTHORITY_NAME),
        ]
        for name in invalidation_order:
            require_entry_names(entry_names)
            listed = os.stat(name, dir_fd=staging_fd, follow_symlinks=False)
            descriptor = entry_descriptors[name]
            opened = os.fstat(descriptor)
            require(
                stat.S_ISREG(opened.st_mode)
                and _file_identity(opened) == _file_identity(listed)
                and _file_identity(opened) == entry_identities[name]
                and opened.st_uid == os.getuid()
                and opened.st_nlink == 1
                and stat.S_IMODE(opened.st_mode) in {0o600, 0o444},
                "repair source staging invalidation identity differs",
            )
            os.fchmod(descriptor, 0o600)
            os.fsync(descriptor)
            rebound_opened = os.fstat(descriptor)
            rebound_named = os.stat(
                name, dir_fd=staging_fd, follow_symlinks=False
            )
            require(
                (rebound_opened.st_dev, rebound_opened.st_ino)
                == (rebound_named.st_dev, rebound_named.st_ino)
                == (opened.st_dev, opened.st_ino)
                and rebound_opened.st_uid
                == rebound_named.st_uid
                == os.getuid()
                and rebound_opened.st_nlink
                == rebound_named.st_nlink
                == 1
                and stat.S_IMODE(rebound_opened.st_mode)
                == stat.S_IMODE(rebound_named.st_mode)
                == 0o600,
                "repair source staging invalidation changed",
            )
            expected_identity = entry_identities[name]
            require(
                (
                    rebound_opened.st_dev,
                    rebound_opened.st_ino,
                    rebound_opened.st_uid,
                    rebound_opened.st_nlink,
                    rebound_opened.st_size,
                )
                == (
                    expected_identity[0],
                    expected_identity[1],
                    expected_identity[2],
                    expected_identity[3],
                    expected_identity[5],
                ),
                "repair source staging invalidation inode differs",
            )
            require_entry_names(entry_names)
        os.fsync(staging_fd)
        require_entry_names(entry_names)
        removal_order = [
            *(name for name in sorted(entry_names) if name != SOURCE_AUTHORITY_NAME),
            *([SOURCE_AUTHORITY_NAME] if SOURCE_AUTHORITY_NAME in entry_names else []),
        ]
        remaining_names = set(entry_names)
        for name in removal_order:
            require_entry_names(remaining_names)
            listed = os.stat(name, dir_fd=staging_fd, follow_symlinks=False)
            descriptor = entry_descriptors[name]
            opened = os.fstat(descriptor)
            require(
                stat.S_ISREG(opened.st_mode)
                and _file_identity(opened) == _file_identity(listed)
                and opened.st_uid == listed.st_uid == os.getuid()
                and opened.st_nlink == listed.st_nlink == 1
                and stat.S_IMODE(opened.st_mode)
                == stat.S_IMODE(listed.st_mode)
                == 0o600,
                "repair source staging invalidation differs",
            )
            expected_identity = entry_identities[name]
            require(
                (
                    opened.st_dev,
                    opened.st_ino,
                    opened.st_uid,
                    opened.st_nlink,
                    opened.st_size,
                )
                == (
                    expected_identity[0],
                    expected_identity[1],
                    expected_identity[2],
                    expected_identity[3],
                    expected_identity[5],
                ),
                "repair source staging removal inode differs",
            )
            os.fsync(descriptor)
            rebound_named = os.stat(
                name, dir_fd=staging_fd, follow_symlinks=False
            )
            rebound_opened = os.fstat(descriptor)
            require(
                _file_identity(rebound_named)
                == _file_identity(rebound_opened)
                == _file_identity(opened)
                and rebound_named.st_nlink
                == rebound_opened.st_nlink
                == 1
                and stat.S_IMODE(rebound_named.st_mode)
                == stat.S_IMODE(rebound_opened.st_mode)
                == 0o600,
                "repair source staging removal binding changed",
            )
            require_entry_names(remaining_names)
            os.unlink(name, dir_fd=staging_fd)
            remaining_names.remove(name)
            require(
                os.fstat(descriptor).st_nlink == 0,
                "repair source staging removed inode remains linked",
            )
            os.fsync(staging_fd)
            require_entry_names(remaining_names)
        os.fsync(staging_fd)
        require_entry_names(set())
        require(os.listdir(staging_fd) == [], "repair source staging is not empty")
        os.rmdir(staging.name, dir_fd=attempt_fd)
        require(
            os.fstat(staging_fd).st_nlink == 0,
            "repair source staging directory remains linked",
        )
        os.fsync(attempt_fd)
        rebind(staging_present=False)
    finally:
        for descriptor in locals().get("entry_descriptors", {}).values():
            try:
                os.close(descriptor)
            except OSError:
                pass
        if staging_fd >= 0:
            os.close(staging_fd)
        os.close(attempt_fd)


def _remove_repair_source_staging(
    staging: Path,
    source: Mapping[str, Any],
    attempt_root: Path,
    locks: _RepairLocks,
) -> None:
    del source, locks
    require(
        staging.parent == attempt_root,
        "repair source fail-stop residue path differs",
    )
    raise RepairError(
        "attempt2 source staging/residue is permanent fail-stop evidence"
    )


def _legacy_seal_repair_source_directory(
    submission_root: Path,
    source: Mapping[str, Any],
    locks: _RepairLocks,
    *,
    enforce_phase: bool = False,
    transition: "_RepairTransitionBinding | None" = None,
) -> Path:
    require(type(enforce_phase) is bool, "repair source phase policy differs")

    def phase_boundary(*, installed: bool = False) -> None:
        if enforce_phase:
            _classify_repair_phase(
                submission_root, source_must_be_installed=installed
            )

    phase_boundary()
    repair_parent = submission_root / "report-repair"
    attempt_root = _repair_root(submission_root)
    source_root = _repair_source_root(submission_root)
    lock_bindings = locks.bindings()

    if os.path.lexists(source_root):
        phase_boundary(installed=True)
        sealed_source = _load_sealed_repair_source(source_root)
        require(
            _source_base_matches(sealed_source, source)
            and sealed_source.get("repair_source_installation_method")
            == DIRECT_FINAL_INSTALL_METHOD,
            "sealed direct-final repair source differs from requested identity",
        )
        return source_root

    # An existing attempt-0002 directory without a complete source authority is
    # permanent fail-stop evidence.  It is never cleaned, populated, or reused.
    require(
        not os.path.lexists(attempt_root),
        "incomplete direct-final attempt2 source namespace is terminal",
    )
    if transition is not None:
        transition.revalidate()
    if not os.path.lexists(repair_parent):
        repair_parent.mkdir(mode=0o700)
        _fsync_directory(repair_parent.parent)
    repair_parent = _directory(repair_parent, "repair state parent")
    repair_parent_fd = os.open(
        repair_parent,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    attempt_fd = -1
    source_fd = -1
    try:
        repair_parent_info = os.fstat(repair_parent_fd)
        repair_parent_named = repair_parent.lstat()
        baseline_attempts = set(os.listdir(repair_parent_fd))
        require(
            _file_identity(repair_parent_info)
            == _file_identity(repair_parent_named)
            and repair_parent_info.st_uid == os.getuid()
            and stat.S_IMODE(repair_parent_info.st_mode) == 0o700
            and "attempt-0002" not in baseline_attempts
            and exact_json_equal(locks.bindings(), lock_bindings),
            "repair source direct-final parent authority differs",
        )
        os.mkdir("attempt-0002", mode=0o700, dir_fd=repair_parent_fd)
        os.fsync(repair_parent_fd)
        attempt_fd = os.open(
            "attempt-0002",
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=repair_parent_fd,
        )
        attempt_info = os.fstat(attempt_fd)
        attempt_named = os.stat(
            "attempt-0002", dir_fd=repair_parent_fd, follow_symlinks=False
        )
        require(
            _file_identity(attempt_info) == _file_identity(attempt_named)
            and attempt_info.st_uid == os.getuid()
            and attempt_info.st_nlink == 2
            and stat.S_IMODE(attempt_info.st_mode) == 0o700
            and os.listdir(attempt_fd) == []
            and set(os.listdir(repair_parent_fd))
            == baseline_attempts | {"attempt-0002"}
            and exact_json_equal(locks.bindings(), lock_bindings),
            "repair source direct-final attempt root differs",
        )
        os.mkdir("source", mode=0o700, dir_fd=attempt_fd)
        os.fsync(attempt_fd)
        source_fd = os.open(
            "source",
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=attempt_fd,
        )
        source_info = os.fstat(source_fd)
        source_named = os.stat("source", dir_fd=attempt_fd, follow_symlinks=False)
        require(
            _file_identity(source_info) == _file_identity(source_named)
            and source_info.st_uid == os.getuid()
            and source_info.st_nlink == 2
            and stat.S_IMODE(source_info.st_mode) == 0o700
            and os.listdir(source_fd) == []
            and set(os.listdir(attempt_fd)) == {"source"}
            and exact_json_equal(locks.bindings(), lock_bindings),
            "repair source direct-final source root differs",
        )
        if transition is not None:
            transition.admit_direct_source_tree(attempt_root, source_root)

        sealed_source = _source_with_installation_method(
            source, DIRECT_FINAL_INSTALL_METHOD
        )
        expected_files = sealed_source["repair_source_files"]

        def rebind(expected_names: set[str]) -> None:
            if transition is not None:
                transition._revalidate_retained(validate_phase=False)
            opened_parent = os.fstat(repair_parent_fd)
            named_parent = repair_parent.lstat()
            opened_attempt = os.fstat(attempt_fd)
            named_attempt = os.stat(
                "attempt-0002", dir_fd=repair_parent_fd, follow_symlinks=False
            )
            opened_source = os.fstat(source_fd)
            named_source = os.stat(
                "source", dir_fd=attempt_fd, follow_symlinks=False
            )
            require(
                (opened_parent.st_dev, opened_parent.st_ino)
                == (named_parent.st_dev, named_parent.st_ino)
                == (repair_parent_info.st_dev, repair_parent_info.st_ino)
                and (opened_attempt.st_dev, opened_attempt.st_ino)
                == (named_attempt.st_dev, named_attempt.st_ino)
                == (attempt_info.st_dev, attempt_info.st_ino)
                and (opened_source.st_dev, opened_source.st_ino)
                == (named_source.st_dev, named_source.st_ino)
                == (source_info.st_dev, source_info.st_ino)
                and opened_parent.st_uid
                == opened_attempt.st_uid
                == opened_source.st_uid
                == os.getuid()
                and stat.S_IMODE(opened_parent.st_mode) == 0o700
                and stat.S_IMODE(opened_attempt.st_mode) == 0o700
                and stat.S_IMODE(opened_source.st_mode) in {0o700, 0o555}
                and set(os.listdir(repair_parent_fd))
                == baseline_attempts | {"attempt-0002"}
                and set(os.listdir(attempt_fd)) == {"source"}
                and set(os.listdir(source_fd)) == expected_names
                and exact_json_equal(locks.bindings(), lock_bindings),
                "repair source direct-final retained authority changed",
            )

        written_names: set[str] = set()
        for name in SOURCE_NAMES:
            rebind(written_names)
            payload, digest, _info = _regular_bytes(
                PACKAGE_DIR / name, f"live repair source snapshot {name}"
            )
            require(
                expected_files[name]["size"] == len(payload)
                and expected_files[name]["sha256"] == digest,
                f"live repair source changed before snapshot: {name}",
            )
            _write_sealed_file(source_root / name, payload)
            written_names.add(name)
            if transition is not None:
                transition.advance_direct_source_file(
                    source_root / name, payload
                )
            rebind(written_names)
        authority_payload = (
            json.dumps(
                dict(sealed_source), sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
        rebind(written_names)
        _write_sealed_file(source_root / SOURCE_AUTHORITY_NAME, authority_payload)
        written_names.add(SOURCE_AUTHORITY_NAME)
        if transition is not None:
            transition.advance_direct_source_file(
                source_root / SOURCE_AUTHORITY_NAME, authority_payload
            )
        rebind(written_names)
        os.fsync(source_fd)
        os.fchmod(source_fd, 0o555)
        os.fsync(source_fd)
        os.fsync(attempt_fd)
        os.fsync(repair_parent_fd)
        if transition is not None:
            transition.seal_direct_source_root(source_root)
        rebind(written_names)
    finally:
        if source_fd >= 0:
            os.close(source_fd)
        if attempt_fd >= 0:
            os.close(attempt_fd)
        os.close(repair_parent_fd)
    phase_boundary(installed=True)
    _validate_sealed_repair_source(source_root, sealed_source)
    if transition is not None:
        transition.revalidate()
    return source_root


def _source_archive_payload(
    source: Mapping[str, Any], submission_root: Path
) -> tuple[bytes, dict[str, Any]]:
    """Build the one canonical, shell-spoolable attempt-2 source archive."""

    require(
        set(source) == SOURCE_AUTHORITY_V1_KEYS
        and source.get("schema_version") == 1,
        "uninstalled source archive identity differs",
    )
    expected_files = source.get("repair_source_files")
    require(
        isinstance(expected_files, Mapping)
        and set(expected_files) == set(SOURCE_NAMES),
        "uninstalled source archive inventory differs",
    )
    encoded_files: dict[str, Any] = {}
    raw_files: dict[str, bytes] = {}
    projection: dict[str, Any] = {}
    for name in SOURCE_NAMES:
        payload, digest, info = _regular_bytes(
            PACKAGE_DIR / name, f"live source archive {name}", max_size=8 << 20
        )
        expected = expected_files[name]
        require(
            isinstance(expected, Mapping)
            and set(expected) == {"mode", "size", "sha256"}
            and expected.get("mode") == 0o444
            and expected.get("size") == len(payload)
            and expected.get("sha256") == digest
            and info.st_uid == os.getuid()
            and info.st_nlink == 1,
            f"live source archive input differs: {name}",
        )
        raw_files[name] = payload
        projection[name] = {
            "mode": 0o444,
            "size": len(payload),
            "sha256": digest,
        }
        encoded_files[name] = {
            **projection[name],
            "data_base64": base64.b64encode(payload).decode("ascii"),
        }
    authority = _source_with_installation_method(
        source, SOURCE_ARCHIVE_INSTALL_METHOD
    )
    require(
        exact_json_equal(authority["repair_source_files"], projection)
        and authority["repair_source_files_sha256"] == stable_hash(projection),
        "source archive authority projection differs",
    )
    slurm_prefix = raw_files["report_repair.slurm"]
    require(
        slurm_prefix.endswith(SOURCE_ARCHIVE_MARKER)
        and slurm_prefix.count(SOURCE_ARCHIVE_MARKER) == 1
        and SOURCE_ARCHIVE_END not in slurm_prefix,
        "source archive batch prefix differs",
    )
    envelope = {
        "archive_kind": SOURCE_ARCHIVE_KIND,
        "schema_version": 2,
        "authority": authority,
        "files": encoded_files,
    }
    body = json.dumps(
        envelope,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    archive = slurm_prefix + body + SOURCE_ARCHIVE_END
    evidence = {
        **authority,
        "repair_source_archive": str(submission_root / SOURCE_ARCHIVE_NAME),
        "repair_source_archive_sha256": hashlib.sha256(archive).hexdigest(),
        "repair_source_archive_size": len(archive),
        "repair_source_archive_format": SOURCE_ARCHIVE_KIND,
    }
    require(set(evidence) == SOURCE_ARCHIVE_EVIDENCE_KEYS, "source archive evidence differs")
    return archive, evidence


def _parse_source_archive_payload(
    source_archive: Path,
    payload: bytes,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    require(
        payload.count(SOURCE_ARCHIVE_MARKER) == 1
        and payload.endswith(SOURCE_ARCHIVE_END),
        "sealed source archive framing differs",
    )
    prefix, tail = payload.split(SOURCE_ARCHIVE_MARKER, 1)
    prefix += SOURCE_ARCHIVE_MARKER
    body = tail[: -len(SOURCE_ARCHIVE_END)]
    try:
        envelope = json.loads(body.decode("ascii"), object_pairs_hook=_pairs(source_archive))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairError(f"sealed source archive JSON differs: {exc}") from exc
    require(
        isinstance(envelope, Mapping)
        and set(envelope) == {"archive_kind", "schema_version", "authority", "files"}
        and envelope.get("archive_kind") == SOURCE_ARCHIVE_KIND
        and type(envelope.get("schema_version")) is int
        and envelope.get("schema_version") == 2
        and isinstance(envelope.get("authority"), Mapping)
        and isinstance(envelope.get("files"), Mapping)
        and set(envelope["files"]) == set(SOURCE_NAMES)
        and body
        == json.dumps(
            envelope,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii"),
        "sealed source archive envelope differs",
    )
    decoded: dict[str, bytes] = {}
    projection: dict[str, Any] = {}
    for name in SOURCE_NAMES:
        row = envelope["files"][name]
        require(
            isinstance(row, Mapping)
            and set(row) == {"data_base64", "mode", "sha256", "size"}
            and row.get("mode") == 0o444
            and type(row.get("size")) is int
            and row["size"] >= 0
            and isinstance(row.get("sha256"), str)
            and SHA256_RE.fullmatch(row["sha256"]) is not None
            and isinstance(row.get("data_base64"), str),
            f"sealed source archive row differs: {name}",
        )
        try:
            item = base64.b64decode(row["data_base64"], validate=True)
        except (ValueError, TypeError) as exc:
            raise RepairError(f"sealed source archive base64 differs: {name}") from exc
        require(
            len(item) == row["size"]
            and hashlib.sha256(item).hexdigest() == row["sha256"],
            f"sealed source archive bytes differ: {name}",
        )
        decoded[name] = item
        projection[name] = {
            "mode": 0o444,
            "size": len(item),
            "sha256": row["sha256"],
        }
    authority = envelope["authority"]
    require(
        set(authority) == SOURCE_AUTHORITY_V2_KEYS
        and authority.get("schema_version") == 2
        and authority.get("repair_source_installation_method")
        == SOURCE_ARCHIVE_INSTALL_METHOD
        and exact_json_equal(authority.get("repair_source_files"), projection)
        and authority.get("repair_source_files_sha256") == stable_hash(projection)
        and decoded["report_repair.slurm"] == prefix
        and decoded["protocol.sha256"]
        == f"{authority.get('repair_package_protocol_sha256')}\n".encode("ascii"),
        "sealed source archive authority differs",
    )
    evidence = {
        **dict(authority),
        "repair_source_archive": str(source_archive),
        "repair_source_archive_sha256": hashlib.sha256(payload).hexdigest(),
        "repair_source_archive_size": len(payload),
        "repair_source_archive_format": SOURCE_ARCHIVE_KIND,
    }
    require(set(evidence) == SOURCE_ARCHIVE_EVIDENCE_KEYS, "sealed source archive evidence differs")
    return evidence, decoded


def _validate_sealed_repair_source(
    source_archive: Path, source: Mapping[str, Any]
) -> None:
    require(
        source_archive == source_archive.parent / SOURCE_ARCHIVE_NAME
        and set(source) == SOURCE_ARCHIVE_EVIDENCE_KEYS,
        "sealed source archive path/evidence differs",
    )
    payload, digest, info = _regular_bytes(
        source_archive, "sealed repair source archive", max_size=8 << 20
    )
    require(
        stat.S_IMODE(info.st_mode) == 0o444
        and info.st_uid == os.getuid()
        and info.st_nlink == 1
        and source.get("repair_source_archive") == str(source_archive)
        and source.get("repair_source_archive_sha256") == digest
        and source.get("repair_source_archive_size") == len(payload)
        and source.get("repair_source_archive_format") == SOURCE_ARCHIVE_KIND,
        "sealed repair source archive identity differs",
    )
    observed, _decoded = _parse_source_archive_payload(source_archive, payload)
    require(exact_json_equal(observed, source), "sealed repair source archive differs")


def _load_sealed_repair_source(source_archive: Path) -> dict[str, Any]:
    payload, _digest, info = _regular_bytes(
        source_archive, "sealed repair source archive", max_size=8 << 20
    )
    require(
        stat.S_IMODE(info.st_mode) == 0o444
        and info.st_uid == os.getuid()
        and info.st_nlink == 1,
        "sealed repair source archive identity differs",
    )
    source, _decoded = _parse_source_archive_payload(source_archive, payload)
    return source


def _source_archive_report_program(
    source_archive: Path,
) -> tuple[Path, bytes]:
    """Return the authenticated report module bytes without a source directory."""

    payload, _digest, info = _regular_bytes(
        source_archive, "sealed repair source archive", max_size=8 << 20
    )
    require(
        stat.S_IMODE(info.st_mode) == 0o444
        and info.st_uid == os.getuid()
        and info.st_nlink == 1,
        "sealed repair source archive identity differs",
    )
    _source, decoded = _parse_source_archive_payload(source_archive, payload)
    return Path(f"{source_archive}::report.py"), decoded["report.py"]


def _seal_repair_source_snapshot(
    submission_root: Path,
    source: Mapping[str, Any],
    locks: _RepairLocks,
    *,
    enforce_phase: bool = False,
    transition: "_RepairTransitionBinding | None" = None,
    prepared_archive: bytes | None = None,
    prepared_evidence: Mapping[str, Any] | None = None,
) -> Path:
    """Seal one final source archive; a partial target is permanent stop evidence."""

    require(type(enforce_phase) is bool, "source archive phase policy differs")
    source_archive = _repair_source_root(submission_root)
    if os.path.lexists(source_archive):
        sealed = _load_sealed_repair_source(source_archive)
        require(
            _source_base_matches(sealed, source)
            and sealed.get("repair_source_installation_method")
            == SOURCE_ARCHIVE_INSTALL_METHOD,
            "existing source archive differs from requested source",
        )
        return source_archive
    require(
        transition is not None
        and _ACTIVE_REPAIR_TRANSITION.get() is transition,
        "source archive creation lacks its retained transition",
    )
    if enforce_phase:
        _classify_repair_phase(
            submission_root, source_must_be_installed=False
        )
    if prepared_archive is None:
        require(
            prepared_evidence is None,
            "prepared source archive evidence lacks its bytes",
        )
        archive, evidence = _source_archive_payload(source, submission_root)
    else:
        require(
            isinstance(prepared_evidence, Mapping),
            "prepared source archive evidence is absent",
        )
        archive = prepared_archive
        evidence, _decoded = _parse_source_archive_payload(
            source_archive, archive
        )
        require(
            exact_json_equal(evidence, prepared_evidence)
            and _source_base_matches(evidence, source),
            "prepared source archive authority differs",
        )
    transition.revalidate()
    digest, size = transition.create_direct_final_file(
        source_archive, archive, label="repair source archive"
    )
    require(
        digest == evidence["repair_source_archive_sha256"]
        and size == evidence["repair_source_archive_size"],
        "created source archive digest differs",
    )
    # The creator descriptor is already part of ``transition.file_rows``.
    # Promote the retained phase in-place so every later scheduler observation
    # and successor append remains under this same binding; never close and
    # recapture the archive through its pathname.
    transition.source_must_be_installed = True
    transition.phase = _classify_repair_phase(
        submission_root, source_must_be_installed=True
    )
    _validate_sealed_repair_source(source_archive, evidence)
    if enforce_phase:
        _classify_repair_phase(
            submission_root, source_must_be_installed=True
        )
    transition.revalidate()
    return source_archive


def _failure_path(submission_root: Path) -> Path:
    return _journal_path(
        submission_root, "REPORT_REPAIR_0001_ORIGINAL_FAILURE.json"
    )


def _submit_calling_path(submission_root: Path) -> Path:
    return _journal_path(
        submission_root, "CALLING_REPORT_REPAIR_0002_SUBMIT.json"
    )


def _submitted_path(submission_root: Path) -> Path:
    return _journal_path(submission_root, "REPORT_REPAIR_0002_SUBMITTED.json")


def _submit_failure_terminal_path(submission_root: Path) -> Path:
    return _journal_path(
        submission_root, "REPORT_REPAIR_0002_TERMINAL_SUBMIT_FAILURE.json"
    )


def _authorization_path(submission_root: Path) -> Path:
    return _journal_path(submission_root, "REPORT_REPAIR_0002_AUTHORIZED.json")


def _released_path(submission_root: Path) -> Path:
    return _journal_path(submission_root, "REPORT_REPAIR_0002_RELEASED.json")


def _release_denied_path(submission_root: Path) -> Path:
    return _journal_path(
        submission_root, "REPORT_REPAIR_0002_TERMINAL_RELEASE_DENIED.json"
    )


def _worker_failure_terminal_path(submission_root: Path) -> Path:
    return _journal_path(
        submission_root, "REPORT_REPAIR_0002_TERMINAL_WORKER_FAILURE.json"
    )


def _attempt1_worker_failure_terminal_path(submission_root: Path) -> Path:
    return _journal_path(
        submission_root, "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json"
    )


def _attempt2_predecessor_path(submission_root: Path) -> Path:
    return _journal_path(
        submission_root, "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
    )


def _completed_path(submission_root: Path) -> Path:
    return _journal_path(submission_root, "REPORT_REPAIR_0002_COMPLETED.json")


def _repair_log_path(submission_root: Path) -> Path:
    return submission_root / "logs" / "report-repair-0002-%j.out"


def _repair_journal_namespace_names(submission_root: Path) -> list[str]:
    active_var = globals().get("_ACTIVE_REPAIR_TRANSITION")
    if active_var is not None:
        active = active_var.get()
        if active is not None and active.submission_root == submission_root:
            return list(active._classify_retained_phase().durable_names)
    historical = {
        entry.name
        for entry in os.scandir(submission_root / "journal")
        if entry.name.startswith("REPORT_REPAIR_")
        or entry.name.startswith("CALLING_REPORT_REPAIR_")
    }
    require(
        not {name for name in historical if "_0002_" in name},
        "attempt2 journal artifacts must be direct submission-root files",
    )
    root_repair_names = {
        entry.name
        for entry in os.scandir(submission_root)
        if _repair_root_name_is_reserved(entry.name)
    }
    publication_pattern = re.compile(
        re.escape(PUBLICATION_ARCHIVE_PREFIX)
        + r"[0-9a-f]{64}"
        + re.escape(PUBLICATION_ARCHIVE_SUFFIX)
        + r"\Z"
    )
    require(
        all(
            name == SOURCE_ARCHIVE_NAME
            or publication_pattern.fullmatch(name) is not None
            or _repair_journal_artifact_name_is_allowed(name)
            for name in root_repair_names
        ),
        "attempt2 root namespace contains an unknown repair artifact",
    )
    successors = {
        name
        for name in root_repair_names
        if _repair_journal_artifact_name_is_allowed(name)
    }
    return sorted(historical | successors)


def _repair_root_name_is_reserved(name: str) -> bool:
    """Exhaustively reserve repair-looking root names across generations."""

    return (
        name.startswith("REPORT_REPAIR_")
        or name.startswith("CALLING_REPORT_REPAIR_")
        or name.startswith(".REPORT_REPAIR_")
        or name.startswith(".CALLING_REPORT_REPAIR_")
    )


def _require_repair_filesystem_namespace(
    submission_root: Path,
    *,
    source_must_be_installed: bool,
    durable_journal_names: Sequence[str] | None = None,
) -> None:
    """Authenticate the reserved attempt/source/report namespace by phase."""

    active_var = globals().get("_ACTIVE_REPAIR_TRANSITION")
    if active_var is not None:
        active = active_var.get()
        if active is not None and active.submission_root == submission_root:
            active.source_must_be_installed = (
                active.source_must_be_installed or source_must_be_installed
            )
            phase = active._classify_retained_phase()
            if durable_journal_names is not None:
                require(
                    set(durable_journal_names) == set(phase.durable_names),
                    "retained report repair durable namespace differs",
                )
            active.revalidate()
            return

    durable = set(
        _repair_journal_namespace_names(submission_root)
        if durable_journal_names is None
        else durable_journal_names
    )
    root_names = set(os.listdir(submission_root))
    report_archives = {
        name
        for name in root_names
        if name.startswith(PUBLICATION_ARCHIVE_PREFIX)
        and name.endswith(PUBLICATION_ARCHIVE_SUFFIX)
    }
    report_reserved = {
        name
        for name in root_names
        if name == "report"
        or name.startswith(".report")
        or (
            name.startswith("REPORT_REPAIR_0002_PUBLICATION")
            and name not in report_archives
        )
    }
    require(
        not report_reserved and len(report_archives) <= 1,
        "report repair install-probe/local staging residue is permanent fail-stop evidence",
    )
    for archive_name in report_archives:
        report_info = (submission_root / archive_name).lstat()
        require(
            stat.S_ISREG(report_info.st_mode)
            and
            report_info.st_uid == os.getuid()
            and report_info.st_nlink == 1
            and stat.S_IMODE(report_info.st_mode) in {0o600, 0o444},
            "published repaired report archive identity differs",
        )

    repair_parent = _directory(
        submission_root / "report-repair", "report repair state root"
    )
    repair_parent_info = repair_parent.lstat()
    attempt_names = {entry.name for entry in os.scandir(repair_parent)}
    require(
        repair_parent_info.st_uid == os.getuid()
        and stat.S_IMODE(repair_parent_info.st_mode) == 0o700
        and attempt_names == {"attempt-0001"},
        "report repair attempt namespace differs",
    )
    attempt1 = _directory(repair_parent / "attempt-0001", "attempt1 state root")
    attempt1_info = attempt1.lstat()
    require(
        attempt1_info.st_uid == os.getuid()
        and stat.S_IMODE(attempt1_info.st_mode) == 0o700
        and {entry.name for entry in os.scandir(attempt1)} == {"source"},
        "attempt1 source namespace differs",
    )
    attempt1_source = _directory(
        attempt1 / "source", "attempt1 sealed source root"
    )
    attempt1_source_info = attempt1_source.lstat()
    require(
        attempt1_source_info.st_uid == os.getuid()
        and attempt1_source_info.st_nlink == 2
        and stat.S_IMODE(attempt1_source_info.st_mode) == 0o555,
        "attempt1 sealed source root identity differs",
    )
    source_archive = submission_root / SOURCE_ARCHIVE_NAME
    if not os.path.lexists(source_archive):
        require(
            not source_must_be_installed,
            "attempt2 source archive is absent after authorization",
        )
        return
    source_info = source_archive.lstat()
    require(
        stat.S_ISREG(source_info.st_mode)
        and source_info.st_uid == os.getuid()
        and source_info.st_nlink == 1
        and stat.S_IMODE(source_info.st_mode) in {0o600, 0o444}
        and (
            stat.S_IMODE(source_info.st_mode) == 0o444
            or not source_must_be_installed
        )
        and "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json" in durable,
        "attempt2 direct-final source archive differs",
    )


def _repair_journal_artifact_name_is_allowed(name: str) -> bool:
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


def _repair_journal_seal_staging_names(submission_root: Path) -> list[str]:
    active_var = globals().get("_ACTIVE_REPAIR_TRANSITION")
    active = None if active_var is None else active_var.get()
    if active is not None and active.submission_root == submission_root:
        journal_names: set[str] = set()
        for path, _descriptor, _identity, names in active.directory_rows:
            if path in {submission_root, submission_root / "journal"}:
                journal_names.update(names)
    else:
        journal_names = set(os.listdir(submission_root / "journal")) | set(
            os.listdir(submission_root)
        )
    candidates = sorted(
        name
        for name in journal_names
        if name.startswith(".") and name.endswith(".seal.tmp")
    )
    require(
        not candidates,
        "attempt2 journal staging is permanent fail-stop evidence",
    )
    return candidates


class _RepairPhaseSnapshot(NamedTuple):
    durable_names: tuple[str, ...]
    virtual_names: tuple[str, ...]
    staged_target: str | None
    staged_mode: int | None
    staged_linked: bool
    report_present: bool
    journal_identity: tuple[int, int, int, int, int]
    staged_identity: tuple[int, int, int, int, int, int] | None


def _journal_artifact_keyset(name: str) -> frozenset[str]:
    static = {
        "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json": ATTEMPT1_TERMINAL_KEYS,
        "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json": ATTEMPT2_PREDECESSOR_KEYS,
        "CALLING_REPORT_REPAIR_0002_SUBMIT.json": SUBMIT_CALLING_KEYS,
        "REPORT_REPAIR_0002_SUBMITTED.json": SUBMITTED_KEYS,
        "REPORT_REPAIR_0002_TERMINAL_SUBMIT_FAILURE.json": SUBMIT_FAILURE_TERMINAL_KEYS,
        "REPORT_REPAIR_0002_AUTHORIZED.json": AUTHORIZATION_KEYS,
        "REPORT_REPAIR_0002_RELEASED.json": RELEASED_KEYS,
        "REPORT_REPAIR_0002_TERMINAL_RELEASE_DENIED.json": RELEASE_DENIED_KEYS,
        "REPORT_REPAIR_0002_TERMINAL_WORKER_FAILURE.json": WORKER_FAILURE_TERMINAL_KEYS,
        "REPORT_REPAIR_0002_COMPLETED.json": COMPLETED_KEYS,
    }
    if name in static:
        return static[name]
    if re.fullmatch(r"CALLING_REPORT_REPAIR_0002_RELEASE_[0-9]{4}\.json", name):
        return RELEASE_CALLING_KEYS
    if re.fullmatch(r"REPORT_REPAIR_0002_RELEASE_RESULT_[0-9]{4}\.json", name):
        return RELEASE_RESULT_KEYS
    if re.fullmatch(r"REPORT_REPAIR_0002_CANCEL_AUTHORIZED_[0-9]{4}\.json", name):
        return CANCEL_AUTHORIZATION_KEYS
    if re.fullmatch(
        r"CALLING_REPORT_REPAIR_0002_SCANCEL_[0-9]{4}_[0-9]{4}\.json", name
    ):
        return CANCEL_CALLING_KEYS
    if re.fullmatch(
        r"REPORT_REPAIR_0002_SCANCEL_RESULT_[0-9]{4}_[0-9]{4}\.json", name
    ):
        return CANCEL_RESULT_KEYS
    if re.fullmatch(r"REPORT_REPAIR_0002_CANCEL_TERMINAL_[0-9]{4}\.json", name):
        return CANCEL_TERMINAL_KEYS
    raise RepairError(f"report repair journal artifact class differs: {name}")


def _journal_artifact_status(name: str) -> str:
    static = {
        "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json": "report_repair_terminal_worker_failure",
        "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json": "attempt1_terminal_failure_authorized_for_attempt2",
        "CALLING_REPORT_REPAIR_0002_SUBMIT.json": "calling_held_report_repair_submission",
        "REPORT_REPAIR_0002_SUBMITTED.json": "held_report_repair_submitted",
        "REPORT_REPAIR_0002_TERMINAL_SUBMIT_FAILURE.json": "report_repair_terminal_submit_failure",
        "REPORT_REPAIR_0002_AUTHORIZED.json": "authorized_terminal_report_repair",
        "REPORT_REPAIR_0002_RELEASED.json": "report_repair_released",
        "REPORT_REPAIR_0002_TERMINAL_RELEASE_DENIED.json": "report_repair_terminal_release_denied",
        "REPORT_REPAIR_0002_TERMINAL_WORKER_FAILURE.json": "report_repair_terminal_worker_failure",
        "REPORT_REPAIR_0002_COMPLETED.json": "report_repair_terminal_publication_complete",
    }
    if name in static:
        return static[name]
    if name.startswith("CALLING_REPORT_REPAIR_0002_RELEASE_"):
        return "calling_report_repair_release"
    if name.startswith("REPORT_REPAIR_0002_RELEASE_RESULT_"):
        return "report_repair_release_attempt_observed"
    if name.startswith("REPORT_REPAIR_0002_CANCEL_AUTHORIZED_"):
        return "report_repair_cleanup_authorized"
    if name.startswith("CALLING_REPORT_REPAIR_0002_SCANCEL_"):
        return "calling_report_repair_cleanup"
    if name.startswith("REPORT_REPAIR_0002_SCANCEL_RESULT_"):
        return "report_repair_cleanup_attempt_observed"
    if name.startswith("REPORT_REPAIR_0002_CANCEL_TERMINAL_"):
        return "report_repair_terminal_cleanup_complete"
    raise RepairError(f"report repair journal artifact status differs: {name}")


def _journal_artifact_dependencies(name: str) -> set[str]:
    attempt1_terminal = "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json"
    predecessor = "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
    calling = "CALLING_REPORT_REPAIR_0002_SUBMIT.json"
    submitted = "REPORT_REPAIR_0002_SUBMITTED.json"
    authorized = "REPORT_REPAIR_0002_AUTHORIZED.json"
    released = "REPORT_REPAIR_0002_RELEASED.json"
    static = {
        attempt1_terminal: set(EXPECTED_ATTEMPT1_CHAIN_SHA256),
        predecessor: {attempt1_terminal},
        calling: {attempt1_terminal, predecessor},
        submitted: {calling},
        "REPORT_REPAIR_0002_TERMINAL_SUBMIT_FAILURE.json": {calling},
        authorized: {submitted},
        "REPORT_REPAIR_0002_TERMINAL_RELEASE_DENIED.json": {authorized},
        # A worker can terminate while still held, before RELEASED is sealed.
        "REPORT_REPAIR_0002_TERMINAL_WORKER_FAILURE.json": {authorized},
        released: {authorized},
        "REPORT_REPAIR_0002_COMPLETED.json": {released},
    }
    if name in static:
        return static[name]
    match = re.fullmatch(r"CALLING_REPORT_REPAIR_0002_RELEASE_([0-9]{4})\.json", name)
    if match is not None:
        index = int(match.group(1))
        dependencies = {authorized}
        if index:
            dependencies |= {
                f"CALLING_REPORT_REPAIR_0002_RELEASE_{index - 1:04d}.json",
                f"REPORT_REPAIR_0002_RELEASE_RESULT_{index - 1:04d}.json",
            }
        return dependencies
    match = re.fullmatch(r"REPORT_REPAIR_0002_RELEASE_RESULT_([0-9]{4})\.json", name)
    if match is not None:
        index = int(match.group(1))
        return {f"CALLING_REPORT_REPAIR_0002_RELEASE_{index:04d}.json"}
    match = re.fullmatch(r"REPORT_REPAIR_0002_CANCEL_AUTHORIZED_([0-9]{4})\.json", name)
    if match is not None:
        generation = int(match.group(1))
        dependencies = {calling}
        if generation:
            dependencies.add(
                f"REPORT_REPAIR_0002_CANCEL_TERMINAL_{generation - 1:04d}.json"
            )
        return dependencies
    match = re.fullmatch(
        r"CALLING_REPORT_REPAIR_0002_SCANCEL_([0-9]{4})_([0-9]{4})\.json",
        name,
    )
    if match is not None:
        generation, index = (int(item) for item in match.groups())
        dependencies = {
            f"REPORT_REPAIR_0002_CANCEL_AUTHORIZED_{generation:04d}.json"
        }
        if index:
            dependencies |= {
                f"CALLING_REPORT_REPAIR_0002_SCANCEL_{generation:04d}_{index - 1:04d}.json",
                f"REPORT_REPAIR_0002_SCANCEL_RESULT_{generation:04d}_{index - 1:04d}.json",
            }
        return dependencies
    match = re.fullmatch(
        r"REPORT_REPAIR_0002_SCANCEL_RESULT_([0-9]{4})_([0-9]{4})\.json",
        name,
    )
    if match is not None:
        generation, index = (int(item) for item in match.groups())
        return {
            f"CALLING_REPORT_REPAIR_0002_SCANCEL_{generation:04d}_{index:04d}.json"
        }
    match = re.fullmatch(r"REPORT_REPAIR_0002_CANCEL_TERMINAL_([0-9]{4})\.json", name)
    if match is not None:
        generation = int(match.group(1))
        return {f"REPORT_REPAIR_0002_CANCEL_AUTHORIZED_{generation:04d}.json"}
    raise RepairError(f"report repair journal artifact dependency class differs: {name}")


def _read_relative_regular(
    parent_fd: int,
    name: str,
    info: os.stat_result,
    *,
    expected_nlink: int,
) -> bytes:
    descriptor = os.open(
        name,
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NONBLOCK", 0),
        dir_fd=parent_fd,
    )
    try:
        opened = os.fstat(descriptor)
        require(
            stat.S_ISREG(opened.st_mode)
            and _file_identity(opened) == _file_identity(info)
            and opened.st_uid == os.getuid()
            and opened.st_nlink == expected_nlink,
            f"report repair journal artifact identity differs: {name}",
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
            f"report repair journal artifact changed: {name}",
        )
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _validated_virtual_journal_payload(name: str, payload: bytes) -> dict[str, Any]:
    value = _decode_json(Path(name), payload)
    expected_status = _journal_artifact_status(name)
    status_is_valid = value.get("status") == expected_status
    if name.startswith("REPORT_REPAIR_0002_CANCEL_TERMINAL_"):
        status_is_valid = value.get("status") in {
            expected_status,
            "report_repair_cleanup_residual_jobs",
        }
    require(
        payload
        == (
            json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
        and set(value) == _journal_artifact_keyset(name)
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and status_is_valid
        and type(value.get("attempt")) is int
        and value.get("attempt")
        == (
            PREDECESSOR_ATTEMPT
            if name == "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json"
            else ATTEMPT
        )
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == EXPECTED_SUBMISSION_SHA256,
        f"report repair staged artifact payload differs: {name}",
    )
    return value


def _legacy_classify_repair_journal_phase(
    submission_root: Path,
    *,
    report_present: bool,
) -> _RepairPhaseSnapshot:
    """Pin and classify the durable/virtual append-only repair graph read-only."""

    journal = _directory(submission_root / "journal", "report repair journal")
    journal_fd = os.open(
        journal,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        parent = os.fstat(journal_fd)
        named_parent = journal.lstat()
        require(
            stat.S_ISDIR(parent.st_mode)
            and _file_identity(parent) == _file_identity(named_parent)
            and parent.st_uid == os.getuid()
            and parent.st_nlink >= 2
            and stat.S_IMODE(parent.st_mode) == 0o700,
            "report repair journal identity differs",
        )
        all_names = set(os.listdir(journal_fd))
        durable = sorted(
            name
            for name in all_names
            if name.startswith("REPORT_REPAIR_")
            or name.startswith("CALLING_REPORT_REPAIR_")
        )
        stages = sorted(
            name
            for name in all_names
            if name.startswith(".") and name.endswith(".seal.tmp")
        )
        require(
            all(_repair_journal_artifact_name_is_allowed(name) for name in durable)
            and not stages,
            "attempt2 journal staging/generation namespace is fail-stop",
        )
        stage_name = stages[0] if stages else None
        staged_target = stage_name[1:-9] if stage_name is not None else None
        staged_mode: int | None = None
        staged_linked = False
        stage_info: os.stat_result | None = None
        if stage_name is not None:
            stage_info = os.stat(stage_name, dir_fd=journal_fd, follow_symlinks=False)
        durable_info: dict[str, os.stat_result] = {}
        for name in durable:
            info = os.stat(name, dir_fd=journal_fd, follow_symlinks=False)
            durable_info[name] = info
            linked = (
                stage_info is not None
                and staged_target == name
                and _file_identity(stage_info) == _file_identity(info)
            )
            require(
                stat.S_ISREG(info.st_mode)
                and info.st_uid == os.getuid()
                and stat.S_IMODE(info.st_mode) == 0o444
                and info.st_nlink == (2 if linked else 1),
                f"report repair durable journal identity differs: {name}",
            )
        virtual = set(durable)
        if stage_name is not None:
            assert stage_info is not None and staged_target is not None
            staged_mode = stat.S_IMODE(stage_info.st_mode)
            target_info = (
                os.stat(staged_target, dir_fd=journal_fd, follow_symlinks=False)
                if staged_target in durable
                else None
            )
            staged_linked = (
                target_info is not None
                and _file_identity(stage_info) == _file_identity(target_info)
            )
            require(
                stat.S_ISREG(stage_info.st_mode)
                and stage_info.st_uid == os.getuid()
                and staged_mode in {0o600, 0o444}
                and (
                    (
                        target_info is None
                        and stage_info.st_nlink == 1
                    )
                    or (
                        staged_linked
                        and staged_mode == 0o444
                        and stage_info.st_nlink == target_info.st_nlink == 2
                    )
                ),
                "report repair journal target/staging identity differs",
            )
            if target_info is not None:
                require(
                    staged_linked,
                    "report repair journal target and staging are distinct",
                )
            dependencies = _journal_artifact_dependencies(staged_target)
            require(
                dependencies <= set(durable),
                "report repair journal stage is not phase-next",
            )
            virtual.add(staged_target)
            if staged_mode == 0o444:
                payload = _read_relative_regular(
                    journal_fd,
                    stage_name,
                    stage_info,
                    expected_nlink=2 if staged_linked else 1,
                )
                _validated_virtual_journal_payload(staged_target, payload)
        require(
            not report_present
            or staged_target in {None, "REPORT_REPAIR_0002_COMPLETED.json"},
            "published repaired report conflicts with a staged journal successor",
        )
        _require_repair_prefix_graph(
            submission_root,
            durable,
            report_present=report_present,
            validate_disk_cleanup=False,
        )
        _require_repair_prefix_graph(
            submission_root,
            sorted(virtual),
            report_present=report_present,
            validate_disk_cleanup=False,
        )
        worker_failure = "REPORT_REPAIR_0002_TERMINAL_WORKER_FAILURE.json"
        released = "REPORT_REPAIR_0002_RELEASED.json"
        completed = "REPORT_REPAIR_0002_COMPLETED.json"
        release_result_values: list[tuple[int, dict[str, Any]]] = []
        for name in sorted(virtual):
            match = re.fullmatch(
                r"REPORT_REPAIR_0002_RELEASE_RESULT_([0-9]{4})\.json",
                name,
            )
            if match is None:
                continue
            if name in durable_info:
                result_payload = _read_relative_regular(
                    journal_fd,
                    name,
                    durable_info[name],
                    expected_nlink=(
                        2
                        if staged_linked and staged_target == name
                        else 1
                    ),
                )
            else:
                # A detached 0600 stage has no authority-bearing bytes and is
                # removable only after the virtual graph proves it phase-next.
                if staged_mode == 0o600:
                    continue
                assert stage_name is not None and stage_info is not None
                result_payload = _read_relative_regular(
                    journal_fd,
                    stage_name,
                    stage_info,
                    expected_nlink=1,
                )
            release_result_values.append(
                (
                    int(match.group(1)),
                    _validated_virtual_journal_payload(name, result_payload),
                )
            )
        allowed_release_modes = {
            "direct_release_response",
            "lost_response_reconciled_still_held",
            "lost_response_reconciled_release_effect",
            "lost_response_reconciled_ambiguous_identity",
        }
        require(
            all(
                result.get("mode") in allowed_release_modes
                for _index, result in release_result_values
            ),
            "attempt2 release result mode differs",
        )
        ambiguous_indices = [
            index
            for index, result in release_result_values
            if result.get("mode")
            == "lost_response_reconciled_ambiguous_identity"
        ]
        effect_indices = [
            index
            for index, result in release_result_values
            if result.get("mode")
            == "lost_response_reconciled_release_effect"
        ]
        release_calling_indices = [
            int(match.group(1))
            for name in virtual
            if (
                match := re.fullmatch(
                    r"CALLING_REPORT_REPAIR_0002_RELEASE_([0-9]{4})\.json",
                    name,
                )
            )
            is not None
        ]
        require(
            len(ambiguous_indices) <= 1
            and len(effect_indices) <= 1
            and (
                not ambiguous_indices
                or (
                    ambiguous_indices[0] == max(release_calling_indices)
                    and not report_present
                    and released not in virtual
                    and worker_failure not in virtual
                    and completed not in virtual
                )
            )
            and (
                not effect_indices
                or effect_indices[0] == max(release_calling_indices)
            ),
            "attempt2 release result terminality differs",
        )
        if worker_failure in virtual and released not in virtual:
            result_names = sorted(
                name
                for name in virtual
                if re.fullmatch(
                    r"REPORT_REPAIR_0002_RELEASE_RESULT_[0-9]{4}\.json",
                    name,
                )
            )
            require(
                bool(result_names),
                "attempt2 before-release worker terminal lacks release results",
            )
            for name in result_names:
                info = durable_info.get(name)
                require(
                    info is not None,
                    "attempt2 worker terminal cannot depend on a staged release result",
                )
                payload = _read_relative_regular(
                    journal_fd, name, info, expected_nlink=1
                )
                result = _validated_virtual_journal_payload(name, payload)
                require(
                    result.get("mode")
                    != "lost_response_reconciled_ambiguous_identity",
                    "attempt2 before-release worker terminal has ambiguous release evidence",
                )
        require(
            _file_identity(os.fstat(journal_fd)) == _file_identity(parent)
            and _file_identity(journal.lstat()) == _file_identity(parent),
            "report repair journal binding changed",
        )
        return _RepairPhaseSnapshot(
            tuple(durable),
            tuple(sorted(virtual)),
            staged_target,
            staged_mode,
            staged_linked,
            report_present,
            (
                parent.st_dev,
                parent.st_ino,
                parent.st_uid,
                parent.st_nlink,
                stat.S_IMODE(parent.st_mode),
            ),
            (
                (
                    stage_info.st_dev,
                    stage_info.st_ino,
                    stage_info.st_uid,
                    stage_info.st_nlink,
                    stat.S_IMODE(stage_info.st_mode),
                    stage_info.st_size,
                )
                if stage_info is not None
                else None
            ),
        )
    finally:
        os.close(journal_fd)


def _classify_repair_journal_phase(
    submission_root: Path,
    *,
    report_present: bool,
) -> _RepairPhaseSnapshot:
    """Classify the fail-stop graph with attempt-2 artifacts at root level."""

    journal = _directory(submission_root / "journal", "report repair journal")
    journal_info = journal.lstat()
    require(
        journal_info.st_uid == os.getuid()
        and journal_info.st_nlink >= 2
        and stat.S_IMODE(journal_info.st_mode) == 0o700,
        "report repair journal identity differs",
    )
    durable = _repair_journal_namespace_names(submission_root)
    _repair_journal_seal_staging_names(submission_root)
    require(
        all(_repair_journal_artifact_name_is_allowed(name) for name in durable),
        "attempt2 journal generation namespace is fail-stop",
    )
    for name in durable:
        path = _journal_path(submission_root, name)
        payload, _digest, info = _regular_bytes(
            path, f"report repair durable artifact {name}"
        )
        require(
            stat.S_IMODE(info.st_mode) == 0o444
            and info.st_uid == os.getuid()
            and info.st_nlink == 1,
            f"report repair durable artifact identity differs: {name}",
        )
        # Attempt-1 bytes are authenticated by the dedicated frozen-chain
        # validators.  This structural classifier owns only the attempt-2
        # append graph and must not reinterpret historical schemas.
        if "_0002_" in name:
            _validated_virtual_journal_payload(name, payload)

    _require_repair_prefix_graph(
        submission_root,
        durable,
        report_present=report_present,
        validate_disk_cleanup=False,
    )
    worker_failure = "REPORT_REPAIR_0002_TERMINAL_WORKER_FAILURE.json"
    released = "REPORT_REPAIR_0002_RELEASED.json"
    completed = "REPORT_REPAIR_0002_COMPLETED.json"
    release_results: list[tuple[int, Mapping[str, Any]]] = []
    release_calling_indices: list[int] = []
    for name in durable:
        calling_match = re.fullmatch(
            r"CALLING_REPORT_REPAIR_0002_RELEASE_([0-9]{4})\.json", name
        )
        if calling_match is not None:
            release_calling_indices.append(int(calling_match.group(1)))
        result_match = re.fullmatch(
            r"REPORT_REPAIR_0002_RELEASE_RESULT_([0-9]{4})\.json", name
        )
        if result_match is None:
            continue
        value, _digest, _info = read_json(
            _journal_path(submission_root, name),
            f"report repair release result {name}",
        )
        release_results.append((int(result_match.group(1)), value))
    ambiguous = [
        index
        for index, value in release_results
        if value.get("mode") == "lost_response_reconciled_ambiguous_identity"
    ]
    effects = [
        index
        for index, value in release_results
        if value.get("mode") == "lost_response_reconciled_release_effect"
    ]
    require(
        len(ambiguous) <= 1
        and len(effects) <= 1
        and (
            not ambiguous
            or (
                release_calling_indices
                and ambiguous[0] == max(release_calling_indices)
                and not report_present
                and released not in durable
                and worker_failure not in durable
                and completed not in durable
            )
        )
        and (
            not effects
            or (
                release_calling_indices
                and effects[0] == max(release_calling_indices)
            )
        ),
        "attempt2 release result terminality differs",
    )
    if worker_failure in durable and released not in durable:
        require(
            bool(release_results)
            and all(
                value.get("mode")
                != "lost_response_reconciled_ambiguous_identity"
                for _index, value in release_results
            ),
            "attempt2 before-release worker terminal evidence differs",
        )
    return _RepairPhaseSnapshot(
        tuple(durable),
        tuple(durable),
        None,
        None,
        False,
        report_present,
        (
            journal_info.st_dev,
            journal_info.st_ino,
            journal_info.st_uid,
            journal_info.st_nlink,
            stat.S_IMODE(journal_info.st_mode),
        ),
        None,
    )


def _require_known_single_generation_namespace(submission_root: Path) -> list[str]:
    names = _repair_journal_namespace_names(submission_root)
    _repair_journal_seal_staging_names(submission_root)
    require(
        all(_repair_journal_artifact_name_is_allowed(name) for name in names),
        "report repair namespace contains a forbidden generation or artifact",
    )
    return names


def _publication_archive_names(submission_root: Path) -> list[str]:
    active_var = globals().get("_ACTIVE_REPAIR_TRANSITION")
    if active_var is not None:
        active = active_var.get()
        if active is not None and active.submission_root == submission_root:
            names = sorted(
                name
                for _parent_fd, name, _descriptor, _identity, _digest, _size in active.stream_file_rows
                if name.startswith(PUBLICATION_ARCHIVE_PREFIX)
                and name.endswith(PUBLICATION_ARCHIVE_SUFFIX)
            )
            require(
                len(names) <= 1,
                "multiple retained report publication archives exist",
            )
            return names
    names = sorted(
        name
        for name in os.listdir(submission_root)
        if name.startswith(PUBLICATION_ARCHIVE_PREFIX)
        and name.endswith(PUBLICATION_ARCHIVE_SUFFIX)
    )
    require(len(names) <= 1, "multiple report publication archives exist")
    return names


def _classify_repair_phase(
    submission_root: Path,
    *,
    source_must_be_installed: bool,
) -> _RepairPhaseSnapshot:
    active_var = globals().get("_ACTIVE_REPAIR_TRANSITION")
    if active_var is not None:
        active = active_var.get()
        if active is not None and active.submission_root == submission_root:
            require(
                active.source_must_be_installed == source_must_be_installed
                or (
                    source_must_be_installed
                    and SOURCE_ARCHIVE_NAME
                    in active._current_submission_reserved(
                        active.directory_rows[0][1]
                    )
                ),
                "retained repair phase source requirement differs",
            )
            if source_must_be_installed:
                active.source_must_be_installed = True
            return active._classify_retained_phase()
    report_present = bool(_publication_archive_names(submission_root))
    phase = _classify_repair_journal_phase(
        submission_root, report_present=report_present
    )
    _require_repair_filesystem_namespace(
        submission_root,
        source_must_be_installed=source_must_be_installed,
        durable_journal_names=phase.durable_names,
    )
    return phase


class _RepairTransitionBinding:
    """Retain the exact local authority spanning one append-only transition.

    Scheduler evidence is meaningful only while the immutable journal prefix,
    every virtual seal stage, both sealed source trees, reserved namespaces,
    submission root, and both production lock inodes remain the same.  This
    object keeps nofollow descriptors for those objects and compares their
    same-fd bytes and named identities at every mutation boundary.
    """

    def __init__(
        self,
        submission_root: Path,
        locks: _RepairLocks,
        *,
        source_must_be_installed: bool,
    ) -> None:
        self.submission_root = submission_root
        self.locks = locks
        self.source_must_be_installed = source_must_be_installed
        self.lock_bindings = locks.bindings()
        raw_expectation = getattr(
            locks, "_transition_authority_expectation", None
        )
        require(
            raw_expectation is None or isinstance(raw_expectation, Mapping),
            "report repair retained authority expectation differs",
        )
        self.authority_expectation = (
            None
            if raw_expectation is None
            else json.loads(
                json.dumps(
                    dict(raw_expectation),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
            )
        )
        self.directory_rows: list[
            tuple[Path, int, tuple[int, int, int, int, int], frozenset[str]]
        ] = []
        self.auxiliary_directory_rows: list[
            tuple[
                Path,
                int,
                tuple[int, int, int, int, int],
                frozenset[str],
            ]
        ] = []
        self.exact_auxiliary_directory_fds: set[int] = set()
        self.file_rows: list[
            tuple[int, str, int, tuple[int, int, int, int, int, int], bytes]
        ] = []
        self.stream_file_rows: list[
            tuple[int, str, int, tuple[int, int, int, int, int, int], str, int]
        ] = []
        self.scientific_runs: dict[Path, dict[str, Any]] = {}
        self.phase: _RepairPhaseSnapshot
        try:
            self._capture()
            # Admission is intentionally after the finite namespace and every
            # authority inode have been retained.  Semantic phase decoding
            # below consumes those bytes; a clean pathname presented only to
            # a pre-capture classifier can never bless a different retained
            # graph.
            self.phase = self._classify_retained_phase()
            self._validate_expected_authority()
            self.revalidate()
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _directory_identity(info: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            info.st_dev,
            info.st_ino,
            info.st_uid,
            info.st_nlink,
            stat.S_IMODE(info.st_mode),
        )

    @staticmethod
    def _read_descriptor(descriptor: int) -> bytes:
        os.lseek(descriptor, 0, os.SEEK_SET)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)

    def _open_directory(self, path: Path) -> int:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        opened = os.fstat(descriptor)
        named = path.lstat()
        require(
            stat.S_ISDIR(opened.st_mode)
            and _file_identity(opened) == _file_identity(named)
            and opened.st_uid == named.st_uid == os.getuid(),
            f"report repair transition directory differs: {path}",
        )
        names = frozenset(os.listdir(descriptor))
        self.directory_rows.append(
            (path, descriptor, self._directory_identity(opened), names)
        )
        return descriptor

    def _retain_file(self, parent_fd: int, name: str) -> None:
        listed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NONBLOCK", 0),
            dir_fd=parent_fd,
        )
        try:
            opened = os.fstat(descriptor)
            require(
                stat.S_ISREG(opened.st_mode)
                and _file_identity(opened) == _file_identity(listed)
                and opened.st_uid == listed.st_uid == os.getuid()
                and stat.S_IMODE(opened.st_mode) in {0o444, 0o600},
                f"report repair transition file differs: {name}",
            )
            payload = self._read_descriptor(descriptor)
            require(
                _file_identity(os.fstat(descriptor)) == _file_identity(opened),
                f"report repair transition file changed: {name}",
            )
            self.file_rows.append(
                (parent_fd, name, descriptor, _file_identity(opened), payload)
            )
        except BaseException:
            os.close(descriptor)
            raise

    def _retain_stream_file(self, parent_fd: int, name: str) -> None:
        listed = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
        descriptor = os.open(
            name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=parent_fd,
        )
        try:
            opened = os.fstat(descriptor)
            require(
                stat.S_ISREG(opened.st_mode)
                and _file_identity(opened) == _file_identity(listed)
                and opened.st_uid == os.getuid()
                and opened.st_nlink == 1
                and stat.S_IMODE(opened.st_mode) == 0o444,
                f"report repair retained streaming file differs: {name}",
            )
            digest = hashlib.sha256()
            offset = 0
            while True:
                block = os.pread(descriptor, 16 * 1024 * 1024, offset)
                if not block:
                    break
                digest.update(block)
                offset += len(block)
            after = os.fstat(descriptor)
            require(
                _file_identity(after) == _file_identity(opened)
                and offset == opened.st_size,
                f"report repair retained streaming file changed: {name}",
            )
            self.stream_file_rows.append(
                (
                    parent_fd,
                    name,
                    descriptor,
                    _file_identity(opened),
                    digest.hexdigest(),
                    opened.st_size,
                )
            )
        except BaseException:
            os.close(descriptor)
            raise

    def _open_auxiliary_directory(
        self,
        path: Path,
        retained_names: frozenset[str],
        *,
        exact_names: bool = False,
        expected_mode: int = 0o700,
    ) -> int:
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        opened = os.fstat(descriptor)
        named = path.lstat()
        observed_names = frozenset(os.listdir(descriptor))
        require(
            stat.S_ISDIR(opened.st_mode)
            and _file_identity(opened) == _file_identity(named)
            and opened.st_uid == named.st_uid == os.getuid()
            and opened.st_nlink == named.st_nlink
            and opened.st_nlink >= 2
            and stat.S_IMODE(opened.st_mode)
            == stat.S_IMODE(named.st_mode)
            == expected_mode
            and (
                observed_names == retained_names
                if exact_names
                else retained_names <= observed_names
            ),
            f"report repair auxiliary authority directory differs: {path}",
        )
        self.auxiliary_directory_rows.append(
            (path, descriptor, self._directory_identity(opened), retained_names)
        )
        if exact_names:
            self.exact_auxiliary_directory_fds.add(descriptor)
        return descriptor

    def _capture_full_tree(self, path: Path) -> None:
        descriptor = self._open_directory(path)
        for name in sorted(os.listdir(descriptor)):
            info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
            child = path / name
            if stat.S_ISDIR(info.st_mode):
                self._capture_full_tree(child)
            elif stat.S_ISREG(info.st_mode):
                self._retain_file(descriptor, name)
            else:
                raise RepairError(
                    f"report repair retained authority tree has a special entry: {child}"
                )

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

    @staticmethod
    def _descriptor_sha256(descriptor: int) -> tuple[str, os.stat_result]:
        before = os.fstat(descriptor)
        digest = hashlib.sha256()
        offset = 0
        while True:
            block = os.pread(descriptor, 16 * 1024 * 1024, offset)
            if not block:
                break
            digest.update(block)
            offset += len(block)
        after = os.fstat(descriptor)
        require(
            _file_identity(before) == _file_identity(after)
            and offset == before.st_size,
            "retained scientific file changed while hashing",
        )
        return digest.hexdigest(), after

    def _capture_scientific_run_tree(self, run_root: Path) -> None:
        """Retain one complete scientific run before report code is loaded."""

        root = run_root.absolute()
        require(root not in self.scientific_runs, "scientific run root is duplicated")
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
        files: dict[Path, tuple[int, tuple[int, ...], str, int, str]] = {}
        symlinks: dict[Path, tuple[int, tuple[int, ...], bytes, int, str]] = {}
        rows: list[dict[str, Any]] = []

        def walk(directory_fd: int, relative: Path, opened: os.stat_result) -> None:
            children = [
                (entry.name, entry.stat(follow_symlinks=False))
                for entry in os.scandir(directory_fd)
            ]
            current = directories[relative]
            directories[relative] = (
                current[0],
                current[1],
                frozenset(name for name, _info in children),
                current[3],
                current[4],
            )
            for name, listed in sorted(children, key=lambda item: item[0]):
                child_relative = relative / name
                if stat.S_ISDIR(listed.st_mode):
                    require(
                        listed.st_uid == os.getuid()
                        and stat.S_IMODE(listed.st_mode) & 0o444
                        and stat.S_IMODE(listed.st_mode) & 0o111,
                        f"retained scientific directory differs: {child_relative}",
                    )
                    child_fd = os.open(name, directory_flags, dir_fd=directory_fd)
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
                        and stat.S_IMODE(listed.st_mode) & 0o444,
                        f"retained scientific file differs: {child_relative}",
                    )
                    child_fd = os.open(name, file_flags, dir_fd=directory_fd)
                    child_opened = os.fstat(child_fd)
                    digest, after = self._descriptor_sha256(child_fd)
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
                            child_relative, "file", child_opened, digest=digest
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
                            child_relative, "symlink", child_opened, target=target
                        )
                    )
                else:
                    raise RepairError(
                        f"retained scientific run has a special entry: {child_relative}"
                    )
            require(
                _file_identity(os.fstat(directory_fd)) == _file_identity(opened),
                f"retained scientific directory changed: {relative}",
            )

        root_fd = self._open_directory_fd_untracked(root)
        root_info = os.fstat(root_fd)
        named_root = root.lstat()
        require(
            stat.S_ISDIR(root_info.st_mode)
            and _file_identity(root_info) == _file_identity(named_root)
            and root_info.st_uid == os.getuid()
            and stat.S_IMODE(root_info.st_mode) & 0o444
            and stat.S_IMODE(root_info.st_mode) & 0o111,
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
        try:
            walk(root_fd, Path(), root_info)
            rows.sort(key=lambda row: str(row["path"]))
            self.scientific_runs[root] = {
                "directories": directories,
                "files": files,
                "symlinks": symlinks,
                "rows": tuple(rows),
            }
            self._revalidate_scientific_run(root, full_hash=True)
        except BaseException:
            for descriptor, *_rest in files.values():
                os.close(descriptor)
            for descriptor, *_rest in symlinks.values():
                os.close(descriptor)
            for descriptor, *_rest in reversed(list(directories.values())):
                os.close(descriptor)
            raise

    @staticmethod
    def _open_directory_fd_untracked(path: Path) -> int:
        return os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )

    def _revalidate_scientific_run(self, root: Path, *, full_hash: bool) -> None:
        retained = self.scientific_runs.get(root.absolute())
        require(retained is not None, "retained scientific run is absent")
        for relative, (descriptor, identity, names, parent_fd, leaf) in retained[
            "directories"
        ].items():
            opened = os.fstat(descriptor)
            named = (
                root.lstat()
                if parent_fd is None
                else os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
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
                    or self._descriptor_sha256(descriptor)[0] == digest
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

    def _retained_file_row(
        self, path: Path
    ) -> tuple[int, str, int, tuple[int, int, int, int, int, int], bytes]:
        parent_descriptor: int | None = None
        for parent_path, descriptor, _identity, _names in self.directory_rows:
            if parent_path == path.parent:
                parent_descriptor = descriptor
                break
        if parent_descriptor is None:
            for parent_path, descriptor, _identity, _names in (
                self.auxiliary_directory_rows
            ):
                if parent_path == path.parent:
                    parent_descriptor = descriptor
                    break
        require(parent_descriptor is not None, f"retained file parent is absent: {path}")
        rows = [
            row
            for row in self.file_rows
            if row[0] == parent_descriptor and row[1] == path.name
        ]
        require(len(rows) == 1, f"retained file binding differs: {path}")
        return rows[0]

    def _maybe_retained_regular(
        self, path: Path
    ) -> tuple[bytes, os.stat_result] | None:
        lexical = path.absolute()
        parent_descriptors = [
            descriptor
            for parent_path, descriptor, _identity, _names in (
                self.directory_rows + self.auxiliary_directory_rows
            )
            if parent_path == lexical.parent
        ]
        if not parent_descriptors:
            return None
        require(
            len(parent_descriptors) == 1,
            f"retained file parent is ambiguous: {lexical}",
        )
        rows = [
            row
            for row in self.file_rows
            if row[0] == parent_descriptors[0] and row[1] == lexical.name
        ]
        require(len(rows) == 1, f"retained file binding differs: {lexical}")
        _parent_fd, _name, descriptor, identity, payload = rows[0]
        require(
            _file_identity(os.fstat(descriptor)) == identity
            and self._read_descriptor(descriptor) == payload,
            f"retained file changed: {lexical}",
        )
        return payload, os.fstat(descriptor)

    @property
    def source_rows(
        self,
    ) -> list[tuple[Path, int, tuple[int, ...], frozenset[str]]]:
        """Compatibility view consumed by the retained snapshot importer."""

        return [*self.directory_rows, *self.auxiliary_directory_rows]

    @property
    def source_file_rows(
        self,
    ) -> list[tuple[int, str, int, tuple[int, ...], bytes]]:
        return self.file_rows

    def retained_regular(
        self, path: Path
    ) -> tuple[bytes, os.stat_result] | None:
        return self._maybe_retained_regular(path.absolute())

    def open_retained_regular(
        self, path: Path
    ) -> tuple[int, os.stat_result] | None:
        retained = self._maybe_retained_regular(path.absolute())
        if retained is None:
            return None
        row = self._retained_file_row(path.absolute())
        descriptor = os.dup(row[2])
        os.set_inheritable(descriptor, False)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor, os.fstat(descriptor)

    def retain_scientific_run_tree(self, run_root: Path) -> None:
        require(
            run_root.absolute() in self.scientific_runs,
            "scientific run was not retained before report assembly",
        )
        self._revalidate_scientific_run(run_root.absolute(), full_hash=False)

    def retained_scientific_tree_rows(
        self, run_root: Path
    ) -> list[dict[str, Any]] | None:
        retained = self.scientific_runs.get(run_root.absolute())
        if retained is None:
            return None
        self._revalidate_scientific_run(run_root.absolute(), full_hash=False)
        return [dict(row) for row in retained["rows"]]

    def has_retained_scientific_run(self, run_root: Path) -> bool:
        return run_root.absolute() in self.scientific_runs

    def open_retained_scientific_regular(
        self, run_root: Path, relative: Path
    ) -> tuple[int, os.stat_result] | None:
        retained = self.scientific_runs.get(run_root.absolute())
        if retained is None:
            return None
        relative = Path(relative)
        require(
            not relative.is_absolute()
            and bool(relative.parts)
            and all(part not in {"", ".", ".."} for part in relative.parts),
            "retained scientific relative path differs",
        )
        row = retained["files"].get(relative)
        if row is None:
            return None
        source_fd, identity, _digest, parent_fd, leaf = row
        require(
            _file_identity(os.fstat(source_fd))
            == _file_identity(
                os.stat(leaf, dir_fd=parent_fd, follow_symlinks=False)
            )
            == identity,
            f"retained scientific artifact changed: {relative}",
        )
        descriptor = os.dup(source_fd)
        os.set_inheritable(descriptor, False)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor, os.fstat(descriptor)

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

    def open_retained_scientific_directory(self, run_root: Path) -> int:
        retained = self.scientific_runs.get(run_root.absolute())
        require(retained is not None, "retained scientific run is absent")
        descriptor = os.dup(retained["directories"][Path()][0])
        os.set_inheritable(descriptor, False)
        return descriptor

    def retained_scientific_artifact_exists(
        self, run_root: Path, relative: Path
    ) -> bool:
        retained = self.scientific_runs.get(run_root.absolute())
        require(retained is not None, "retained scientific run is absent")
        relative = Path(relative)
        return (
            relative in retained["files"]
            or relative in retained["directories"]
            or relative in retained["symlinks"]
        )

    def manages_regular_path(self, path: Path) -> bool:
        lexical = path.absolute()
        if lexical == self.submission_root or self.submission_root in lexical.parents:
            return True
        if any(
            lexical == root or root in lexical.parents
            for root in self.scientific_runs
        ):
            return True
        return any(
            lexical == root or root in lexical.parents
            for root, _descriptor, _identity, _names in self.source_rows
        )

    def validate_exact_snapshot_authority(
        self, expectation: Mapping[str, Any]
    ) -> None:
        require(
            isinstance(expectation.get("snapshot_root"), str)
            and isinstance(expectation.get("snapshot_inventory"), Mapping),
            "retained snapshot validation expectation differs",
        )
        root = Path(expectation["snapshot_root"])
        inventory = dict(expectation["snapshot_inventory"])
        require(
            root == self.submission_root / "source-snapshot" / "repo"
            and stable_hash(inventory)
            == expectation.get("snapshot_inventory_sha256"),
            "retained snapshot validation identity differs",
        )
        for relative, digest in inventory.items():
            retained = self._maybe_retained_regular(root / relative)
            require(
                retained is not None
                and hashlib.sha256(retained[0]).hexdigest() == digest,
                f"retained snapshot validation differs: {relative}",
            )
        self.revalidate()

    def _retained_payload(self, path: Path) -> bytes:
        row = self._retained_file_row(path)
        _parent_fd, _name, descriptor, identity, payload = row
        require(
            _file_identity(os.fstat(descriptor)) == identity
            and self._read_descriptor(descriptor) == payload,
            f"retained file changed: {path}",
        )
        return payload

    def _capture(self) -> None:
        submission_fd = self._open_directory(self.submission_root)
        # The submission tree has live logs and task metadata; pin only its
        # reserved repair/publication names while retaining its root inode.
        submission_names = set(os.listdir(submission_fd))
        reserved = frozenset(
            name
            for name in submission_names
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
        path, descriptor, identity, _names = self.directory_rows[-1]
        self.directory_rows[-1] = (path, descriptor, identity, reserved)
        for name in (
            "SUBMISSION_CONTRACT.json",
            "SUBMISSION_RECEIPT.json",
            "SUBMISSION_AUTHORIZATION.json",
        ):
            require(
                name in submission_names,
                f"report repair transition root authority is absent: {name}",
            )
            self._retain_file(submission_fd, name)
        for name in sorted(reserved):
            if name.startswith(PUBLICATION_ARCHIVE_PREFIX) and name.endswith(
                PUBLICATION_ARCHIVE_SUFFIX
            ):
                self._retain_stream_file(submission_fd, name)
            elif name == SOURCE_ARCHIVE_NAME or _repair_root_name_is_reserved(name):
                self._retain_file(submission_fd, name)

        snapshot = self.submission_root / "source-snapshot"
        require(
            os.path.lexists(snapshot),
            "report repair transition source snapshot is absent",
        )
        self._capture_full_tree(snapshot)
        self._capture_compatible_contract_authority(snapshot)

        tasks = self.submission_root / "tasks"
        cell_names = frozenset(f"cell-{index:02d}" for index in range(20))
        tasks_fd = self._open_auxiliary_directory(
            tasks, cell_names, exact_names=True
        )
        del tasks_fd
        for cell_name in sorted(cell_names):
            cell = tasks / cell_name
            cell_fd = self._open_auxiliary_directory(
                cell,
                frozenset({"LAUNCH.json", "WORKER_COMPLETE.json", "waves"}),
                exact_names=True,
            )
            for name in ("LAUNCH.json", "WORKER_COMPLETE.json"):
                self._retain_file(cell_fd, name)
            waves = cell / "waves"
            self._open_auxiliary_directory(
                waves,
                frozenset({"0", "1"}),
                exact_names=True,
            )
            for wave_index in (0, 1):
                wave = waves / str(wave_index)
                wave_fd = self._open_auxiliary_directory(
                    wave, frozenset(), exact_names=False
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
                    f"report repair task {cell_name} wave{wave_index} namespace differs",
                )
                # Upgrade this directory to an exact namespace after the
                # branch-aware production grammar has been established.
                for index, row in enumerate(self.auxiliary_directory_rows):
                    if row[1] == wave_fd:
                        self.auxiliary_directory_rows[index] = (
                            row[0], row[1], row[2], names
                        )
                        self.exact_auxiliary_directory_fds.add(wave_fd)
                        break
                for name in sorted(names):
                    self._retain_file(wave_fd, name)

        launches = self.submission_root / "launches"
        launch_names = frozenset(
            f"cell-{index:02d}.json" for index in range(20)
        )
        launches_fd = self._open_auxiliary_directory(
            launches, launch_names, exact_names=True
        )
        for name in sorted(launch_names):
            self._retain_file(launches_fd, name)
        run_roots: list[Path] = []
        for index, name in enumerate(sorted(launch_names)):
            launch_path = launches / name
            launch = _decode_json(launch_path, self._retained_payload(launch_path))
            cell = launch.get("cell")
            run_value = cell.get("run_directory") if isinstance(cell, Mapping) else None
            require(
                isinstance(run_value, str) and bool(run_value),
                f"report repair retained launch run directory differs: cell{index}",
            )
            run_root = Path(run_value)
            require(
                run_root.is_absolute()
                and all(part not in {"", ".", ".."} for part in run_root.parts[1:])
                and run_root not in run_roots,
                f"report repair retained scientific run identity differs: cell{index}",
            )
            run_roots.append(run_root)
        for run_root in run_roots:
            self._capture_scientific_run_tree(run_root)

        logs = self.submission_root / "logs"
        attempt1_log = f"report-repair-0001-{EXPECTED_ATTEMPT1_JOB_ID}.out"
        logs_fd = self._open_auxiliary_directory(
            logs, frozenset({attempt1_log})
        )
        self._retain_file(logs_fd, attempt1_log)

        journal_fd = self._open_directory(self.submission_root / "journal")
        journal_names = set(os.listdir(journal_fd))
        for name in sorted(journal_names):
            info = os.stat(name, dir_fd=journal_fd, follow_symlinks=False)
            require(
                stat.S_ISREG(info.st_mode),
                f"report repair transition journal has a special entry: {name}",
            )
            self._retain_file(journal_fd, name)

        repair_parent = self.submission_root / "report-repair"
        if os.path.lexists(repair_parent):
            repair_fd = self._open_directory(repair_parent)
            for attempt_name in sorted(os.listdir(repair_fd)):
                attempt_path = repair_parent / attempt_name
                attempt_fd = self._open_directory(attempt_path)
                for name in sorted(os.listdir(attempt_fd)):
                    child = attempt_path / name
                    info = child.lstat()
                    if stat.S_ISDIR(info.st_mode):
                        child_fd = self._open_directory(child)
                        for entry_name in sorted(os.listdir(child_fd)):
                            entry_info = os.stat(
                                entry_name,
                                dir_fd=child_fd,
                                follow_symlinks=False,
                            )
                            if stat.S_ISREG(entry_info.st_mode):
                                self._retain_file(child_fd, entry_name)
                            else:
                                raise RepairError(
                                    "report repair retained source has a special "
                                    f"entry: {child / entry_name}"
                                )
                    else:
                        raise RepairError(
                            "report repair attempt namespace has a non-directory "
                            f"entry: {child}"
                        )

    def _capture_compatible_contract_authority(self, snapshot: Path) -> None:
        manifest_path = snapshot / "repo" / PACKAGE_RELATIVE / "manifest.json"
        try:
            retained = self._maybe_retained_regular(manifest_path)
        except RepairError:
            retained = None
        if retained is None:
            # Minimal state-machine fixtures omit the scientific package; the
            # real contract validator rejects that shape before mutation.
            return
        manifest = _decode_json(manifest_path, retained[0])
        paths = manifest.get("paths")
        settings = manifest.get("settings")
        require(
            isinstance(paths, Mapping)
            and isinstance(paths.get("compatible_contract_root"), str)
            and isinstance(settings, list)
            and len(settings) == 5,
            "retained compatible-contract authority differs",
        )
        root = Path(paths["compatible_contract_root"])
        require(
            root.is_absolute()
            and all(part not in {"", ".", ".."} for part in root.parts[1:]),
            "retained compatible-contract root differs",
        )
        setting_ids: list[str] = []
        for row in settings:
            setting_id = row.get("id") if isinstance(row, Mapping) else None
            require(
                isinstance(setting_id, str)
                and setting_id.isascii()
                and bool(setting_id)
                and setting_id not in setting_ids
                and "/" not in setting_id
                and "\\" not in setting_id,
                "retained compatible-contract setting differs",
            )
            setting_ids.append(setting_id)
        data_names = frozenset(f"{item}.json" for item in setting_ids)
        data_fd = self._open_auxiliary_directory(
            root / "data", data_names, exact_names=False
        )
        for name in sorted(data_names):
            self._retain_file(data_fd, name)
        recipe_names = frozenset(setting_ids)
        recipe_fd = self._open_auxiliary_directory(
            root / "future-recipes", recipe_names, exact_names=False
        )
        del recipe_fd
        for setting_id in setting_ids:
            setting_fd = self._open_auxiliary_directory(
                root / "future-recipes" / setting_id,
                frozenset({"manifest.json"}),
                exact_names=False,
            )
            self._retain_file(setting_fd, "manifest.json")

    def _classify_retained_phase(self) -> _RepairPhaseSnapshot:
        """Decode the repair phase solely from the already-retained graph."""

        directory_map = {
            path: (descriptor, identity, names)
            for path, descriptor, identity, names in self.directory_rows
        }
        require(
            len(directory_map) == len(self.directory_rows),
            "report repair retained directory graph is ambiguous",
        )
        root_fd, _root_identity, root_reserved = directory_map[
            self.submission_root
        ]
        journal_path = self.submission_root / "journal"
        journal_fd, journal_identity, journal_names = directory_map[journal_path]

        root_repair_names = {
            name for name in root_reserved if _repair_root_name_is_reserved(name)
        }
        require(
            all(
                name == SOURCE_ARCHIVE_NAME
                or _repair_journal_artifact_name_is_allowed(name)
                or re.fullmatch(
                    re.escape(PUBLICATION_ARCHIVE_PREFIX)
                    + r"[0-9a-f]{64}"
                    + re.escape(PUBLICATION_ARCHIVE_SUFFIX),
                    name,
                )
                is not None
                for name in root_repair_names
            ),
            "attempt2 root namespace contains an unknown repair artifact",
        )
        require(
            not {
                name
                for name in root_reserved
                if name == "report" or name.startswith(".report")
            },
            "report repair local staging/probe residue is permanent fail-stop evidence",
        )

        historical = {
            name
            for name in journal_names
            if name.startswith("REPORT_REPAIR_")
            or name.startswith("CALLING_REPORT_REPAIR_")
        }
        require(
            not {name for name in historical if "_0002_" in name}
            and not {
                name
                for name in journal_names
                if name.startswith(".") and name.endswith(".seal.tmp")
            },
            "attempt2 journal artifacts/stages must not exist under journal",
        )
        successors = {
            name
            for name in root_repair_names
            if _repair_journal_artifact_name_is_allowed(name)
        }
        durable = tuple(sorted(historical | successors))
        require(
            all(_repair_journal_artifact_name_is_allowed(name) for name in durable),
            "attempt2 journal generation namespace is fail-stop",
        )

        retained_by_parent_name = {
            (parent_fd, name): (descriptor, identity, payload)
            for parent_fd, name, descriptor, identity, payload in self.file_rows
        }
        require(
            len(retained_by_parent_name) == len(self.file_rows),
            "report repair retained file graph is not bijective",
        )
        values: dict[str, dict[str, Any]] = {}
        for name in durable:
            parent_fd = root_fd if "_0002_" in name else journal_fd
            row = retained_by_parent_name.get((parent_fd, name))
            require(row is not None, f"retained repair artifact is absent: {name}")
            descriptor, identity, payload = row
            info = os.fstat(descriptor)
            require(
                _file_identity(info) == identity
                and info.st_uid == os.getuid()
                and info.st_nlink == 1
                and stat.S_IMODE(info.st_mode) == 0o444
                and self._read_descriptor(descriptor) == payload,
                f"retained repair artifact identity differs: {name}",
            )
            if "_0002_" in name:
                values[name] = _validated_virtual_journal_payload(name, payload)

        publication_rows = [
            row
            for row in self.stream_file_rows
            if row[0] == root_fd
            and row[1].startswith(PUBLICATION_ARCHIVE_PREFIX)
            and row[1].endswith(PUBLICATION_ARCHIVE_SUFFIX)
        ]
        require(
            len(publication_rows) <= 1,
            "multiple report publication archives exist",
        )
        report_present = bool(publication_rows)
        _require_repair_prefix_graph(
            self.submission_root,
            durable,
            report_present=report_present,
            validate_disk_cleanup=False,
        )

        release_callings = [
            int(match.group(1))
            for name in durable
            if (
                match := re.fullmatch(
                    r"CALLING_REPORT_REPAIR_0002_RELEASE_([0-9]{4})\.json",
                    name,
                )
            )
            is not None
        ]
        release_results = [
            (int(match.group(1)), values[name])
            for name in durable
            if (
                match := re.fullmatch(
                    r"REPORT_REPAIR_0002_RELEASE_RESULT_([0-9]{4})\.json",
                    name,
                )
            )
            is not None
        ]
        ambiguous = [
            index
            for index, value in release_results
            if value.get("mode")
            == "lost_response_reconciled_ambiguous_identity"
        ]
        effects = [
            index
            for index, value in release_results
            if value.get("mode") == "lost_response_reconciled_release_effect"
        ]
        require(
            len(ambiguous) <= 1
            and len(effects) <= 1
            and (
                not ambiguous
                or (
                    bool(release_callings)
                    and ambiguous[0] == max(release_callings)
                    and not report_present
                    and "REPORT_REPAIR_0002_RELEASED.json" not in durable
                    and "REPORT_REPAIR_0002_TERMINAL_WORKER_FAILURE.json"
                    not in durable
                    and "REPORT_REPAIR_0002_COMPLETED.json" not in durable
                )
            )
            and (
                not effects
                or (
                    bool(release_callings)
                    and effects[0] == max(release_callings)
                )
            ),
            "attempt2 release result terminality differs",
        )

        repair_parent = self.submission_root / "report-repair"
        require(
            repair_parent in directory_map
            and directory_map[repair_parent][2] == frozenset({"attempt-0001"}),
            "report repair attempt namespace differs",
        )
        attempt1 = repair_parent / "attempt-0001"
        source1 = attempt1 / "source"
        require(
            attempt1 in directory_map
            and directory_map[attempt1][2] == frozenset({"source"})
            and source1 in directory_map
            and directory_map[source1][1][3] == 2
            and directory_map[source1][1][4] == 0o555,
            "attempt1 retained source namespace differs",
        )
        source_name_present = SOURCE_ARCHIVE_NAME in root_reserved
        if self.source_must_be_installed:
            require(
                source_name_present,
                "attempt2 source archive is absent after authorization",
            )
        if source_name_present:
            source_row = retained_by_parent_name.get((root_fd, SOURCE_ARCHIVE_NAME))
            require(
                source_row is not None
                and source_row[1][3] == 1
                and source_row[1][4] in ({0o444} if self.source_must_be_installed else {0o600, 0o444})
                and "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json" in durable,
                "attempt2 direct-final source archive differs",
            )
        return _RepairPhaseSnapshot(
            durable,
            durable,
            None,
            None,
            False,
            report_present,
            journal_identity,
            None,
        )

    def _validate_expected_authority(self) -> None:
        """Authenticate retained production roots against the sealed contract."""

        expected = self.authority_expectation
        if expected is None:
            return
        require(
            set(expected)
            == {
                "submission_root",
                "submission_contract_sha256",
                "submission_receipt_sha256",
                "submission_authorization_sha256",
                "snapshot_root",
                "snapshot_inventory",
                "snapshot_inventory_sha256",
                "worker_receipt_map",
                "worker_receipt_map_sha256",
                "attempt1_log_sha256",
                "attempt1_log_size",
            }
            and expected.get("submission_root") == str(self.submission_root),
            "report repair retained authority expectation schema differs",
        )
        for name, digest_key, label in (
            (
                "SUBMISSION_CONTRACT.json",
                "submission_contract_sha256",
                "submission contract",
            ),
            (
                "SUBMISSION_RECEIPT.json",
                "submission_receipt_sha256",
                "submission receipt",
            ),
            (
                "SUBMISSION_AUTHORIZATION.json",
                "submission_authorization_sha256",
                "submission authorization",
            ),
        ):
            path = self.submission_root / name
            payload = self._retained_payload(path)
            digest = hashlib.sha256(payload).hexdigest()
            _decode_json(path, payload)
            _parent_fd, _name, descriptor, _identity, _payload = (
                self._retained_file_row(path)
            )
            info = os.fstat(descriptor)
            require(
                digest == expected.get(digest_key)
                and info.st_uid == os.getuid()
                and info.st_nlink == 1
                and stat.S_IMODE(info.st_mode) == 0o444,
                f"retained {label} differs",
            )

        snapshot_parent = self.submission_root / "source-snapshot"
        snapshot_root = Path(str(expected.get("snapshot_root", "")))
        expected_inventory = expected.get("snapshot_inventory")
        require(
            isinstance(expected_inventory, Mapping),
            "retained snapshot expected inventory differs",
        )
        normalized_inventory: dict[str, str] = {}
        for raw_relative, raw_digest in expected_inventory.items():
            relative = Path(raw_relative) if isinstance(raw_relative, str) else Path()
            require(
                isinstance(raw_relative, str)
                and isinstance(raw_digest, str)
                and SHA256_RE.fullmatch(raw_digest) is not None
                and not relative.is_absolute()
                and bool(relative.parts)
                and all(part not in {"", ".", ".."} for part in relative.parts)
                and relative.as_posix() == raw_relative,
                "retained snapshot expected inventory row differs",
            )
            normalized_inventory[raw_relative] = raw_digest
        require(
            snapshot_root == snapshot_parent / "repo"
            and bool(normalized_inventory),
            "retained source snapshot root differs",
        )

        retained_directories = {
            path: (descriptor, identity, names)
            for path, descriptor, identity, names in self.directory_rows
            if path == snapshot_parent
            or path == snapshot_root
            or snapshot_root in path.parents
        }
        require(
            len(retained_directories)
            == sum(
                1
                for path, *_rest in self.directory_rows
                if path == snapshot_parent
                or path == snapshot_root
                or snapshot_root in path.parents
            ),
            "retained source snapshot directory graph is ambiguous",
        )
        expected_files = {
            snapshot_root / relative for relative in normalized_inventory
        }
        expected_directories = {snapshot_parent, snapshot_root}
        for file_path in expected_files:
            parent = file_path.parent
            while parent != snapshot_root:
                expected_directories.add(parent)
                parent = parent.parent
        require(
            set(retained_directories) == expected_directories,
            "retained source snapshot directory coverage differs",
        )
        directory_by_fd = {
            descriptor: path
            for path, (descriptor, _identity, _names) in retained_directories.items()
        }
        retained_files: dict[
            Path, tuple[int, tuple[int, int, int, int, int, int], bytes]
        ] = {}
        for parent_fd, name, descriptor, identity, payload in self.file_rows:
            parent_path = directory_by_fd.get(parent_fd)
            if parent_path is None:
                continue
            path = parent_path / name
            require(
                path not in retained_files,
                f"retained source snapshot file is ambiguous: {path}",
            )
            retained_files[path] = (descriptor, identity, payload)
        require(
            set(retained_files) == expected_files,
            "retained source snapshot file coverage differs",
        )
        expected_children: dict[Path, set[str]] = {
            path: set() for path in expected_directories
        }
        for directory in expected_directories - {snapshot_parent}:
            expected_children[directory.parent].add(directory.name)
        for file_path in expected_files:
            expected_children[file_path.parent].add(file_path.name)
        for path, (descriptor, identity, names) in retained_directories.items():
            info = os.fstat(descriptor)
            require(
                self._directory_identity(info) == identity
                and info.st_uid == os.getuid()
                and info.st_nlink >= 2
                and stat.S_IMODE(info.st_mode) == 0o555
                and names == frozenset(expected_children[path]),
                f"retained source snapshot directory differs: {path}",
            )
        snapshot_inventory: dict[str, str] = {}
        for relative, expected_digest in normalized_inventory.items():
            descriptor, identity, payload = retained_files[
                snapshot_root / relative
            ]
            info = os.fstat(descriptor)
            digest = hashlib.sha256(payload).hexdigest()
            require(
                _file_identity(info) == identity
                and info.st_uid == os.getuid()
                and info.st_nlink == 1
                and stat.S_IMODE(info.st_mode) == 0o444
                and info.st_size == len(payload)
                and digest == expected_digest,
                f"retained source snapshot file differs: {relative}",
            )
            snapshot_inventory[relative] = digest
        require(
            exact_json_equal(
                snapshot_inventory, normalized_inventory
            )
            and stable_hash(snapshot_inventory)
            == expected.get("snapshot_inventory_sha256"),
            "retained source snapshot inventory differs",
        )

        expected_receipt_map = expected.get("worker_receipt_map")
        expected_receipt_names = {
            (Path("tasks") / f"cell-{index:02d}" / "WORKER_COMPLETE.json").as_posix()
            for index in range(20)
        }
        require(
            isinstance(expected_receipt_map, Mapping)
            and set(expected_receipt_map) == {"schema_version", "files"}
            and expected_receipt_map.get("schema_version") == 1
            and isinstance(expected_receipt_map.get("files"), Mapping)
            and set(expected_receipt_map["files"]) == expected_receipt_names,
            "retained worker receipt expectation differs",
        )
        receipt_rows: dict[str, Any] = {}
        for relative, expected_row in expected_receipt_map["files"].items():
            path = self.submission_root / relative
            payload = self._retained_payload(path)
            _parent_fd, _name, descriptor, _identity, _payload = (
                self._retained_file_row(path)
            )
            info = os.fstat(descriptor)
            receipt_rows[relative] = {
                "mode": stat.S_IMODE(info.st_mode),
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            _decode_json(path, payload)
            require(
                info.st_uid == os.getuid()
                and info.st_nlink == 1
                and stat.S_IMODE(info.st_mode) == 0o444
                and info.st_size == len(payload)
                and exact_json_equal(receipt_rows[relative], expected_row),
                f"retained worker receipt differs: {relative}",
            )
        receipt_map = {"schema_version": 1, "files": receipt_rows}
        require(
            exact_json_equal(receipt_map, expected.get("worker_receipt_map"))
            and stable_hash(receipt_map)
            == expected.get("worker_receipt_map_sha256"),
            "retained worker receipt authority differs",
        )

        attempt1_log = (
            self.submission_root
            / "logs"
            / f"report-repair-0001-{EXPECTED_ATTEMPT1_JOB_ID}.out"
        )
        payload = self._retained_payload(attempt1_log)
        digest = hashlib.sha256(payload).hexdigest()
        _parent_fd, _name, descriptor, _identity, _payload = (
            self._retained_file_row(attempt1_log)
        )
        info = os.fstat(descriptor)
        require(
            digest == expected.get("attempt1_log_sha256")
            and len(payload) == expected.get("attempt1_log_size")
            and payload == EXPECTED_ATTEMPT1_LOG_BYTES
            and info.st_uid == os.getuid()
            and info.st_nlink == 1
            and stat.S_IMODE(info.st_mode) == 0o600,
            "retained attempt1 failure log differs",
        )

    def install_authority_expectation(
        self, expectation: Mapping[str, Any]
    ) -> None:
        """Attach production semantics to this already-retained inode graph.

        Construction intentionally captures before it decodes.  The original
        contract validator subsequently derives this expectation from those
        retained bytes and installs it on the same object; replacing the
        binding with a fresh capture would reintroduce the show-clean/restore
        attack this boundary is designed to prevent.
        """

        normalized = json.loads(
            json.dumps(
                dict(expectation),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
        require(
            self.authority_expectation is None
            or exact_json_equal(self.authority_expectation, normalized),
            "report repair retained authority expectation changed",
        )
        self.authority_expectation = normalized
        self._validate_expected_authority()
        self.revalidate()

    def _current_submission_reserved(self, descriptor: int) -> frozenset[str]:
        return frozenset(
            name
            for name in os.listdir(descriptor)
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

    def create_direct_final_file(
        self,
        path: Path,
        payload: bytes,
        *,
        label: str,
    ) -> tuple[str, int]:
        """Claim, seal, and retain one final regular file without reopening."""

        require(
            path.parent
            in {self.submission_root, self.submission_root / "journal"}
            and path.name not in {"", ".", ".."}
            and isinstance(payload, bytes)
            and bool(payload),
            f"{label} target/payload differs",
        )
        self.revalidate()
        submission_index = next(
            (
                index
                for index, row in enumerate(self.directory_rows)
                if row[0] == path.parent
            ),
            None,
        )
        require(submission_index is not None, f"{label} parent is unbound")
        row_path, parent_fd, parent_identity, names = self.directory_rows[
            submission_index
        ]
        require(path.name not in names, f"{label} target already exists")
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
                dir_fd=parent_fd,
            )
            created = os.fstat(descriptor)
            named = os.stat(
                path.name, dir_fd=parent_fd, follow_symlinks=False
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
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                require(written > 0, f"short {label} write")
                view = view[written:]
            os.fsync(descriptor)
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
            digest = hashlib.sha256(payload).hexdigest()
            observed = bytearray()
            offset = 0
            while offset < len(payload) + 1:
                chunk = os.pread(
                    descriptor, len(payload) + 1 - offset, offset
                )
                if not chunk:
                    break
                observed.extend(chunk)
                offset += len(chunk)
            sealed = os.fstat(descriptor)
            rebound = os.stat(
                path.name, dir_fd=parent_fd, follow_symlinks=False
            )
            expected_names = frozenset(set(names) | {path.name})
            observed_names = (
                self._current_submission_reserved(parent_fd)
                if path.parent == self.submission_root
                else frozenset(os.listdir(parent_fd))
            )
            require(
                bytes(observed) == payload
                and _file_identity(sealed) == _file_identity(rebound)
                and sealed.st_uid == os.getuid()
                and sealed.st_nlink == 1
                and stat.S_IMODE(sealed.st_mode) == 0o444
                and observed_names == expected_names,
                f"{label} seal differs",
            )
            os.fsync(parent_fd)
            require(
                self._directory_identity(os.fstat(parent_fd))
                == self._directory_identity(row_path.lstat())
                == parent_identity
                and _file_identity(os.fstat(descriptor))
                == _file_identity(
                    os.stat(
                        path.name,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                )
                and (
                    self._current_submission_reserved(parent_fd)
                    if path.parent == self.submission_root
                    else frozenset(os.listdir(parent_fd))
                )
                == expected_names,
                f"{label} changed after parent fsync",
            )
            self.directory_rows[submission_index] = (
                row_path,
                parent_fd,
                parent_identity,
                expected_names,
            )
            self.file_rows.append(
                (
                    parent_fd,
                    path.name,
                    descriptor,
                    _file_identity(sealed),
                    payload,
                )
            )
            descriptor = -1
            self._revalidate_retained(validate_phase=False)
            return digest, len(payload)
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def _revalidate_retained(self, *, validate_phase: bool) -> None:
        require(
            exact_json_equal(self.locks.bindings(), self.lock_bindings),
            "report repair transition lock binding changed",
        )
        current_expectation = getattr(
            self.locks, "_transition_authority_expectation", None
        )
        if self.authority_expectation is None and isinstance(
            current_expectation, Mapping
        ):
            # The outer transaction is intentionally captured before the
            # original contract is interpreted.  Admit that expectation only
            # once, from the contract-derived value, then authenticate it
            # against the already-retained descriptors.  This is an upgrade of
            # the existing binding, not a fresh pathname capture.
            self.authority_expectation = json.loads(
                json.dumps(
                    dict(current_expectation),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=True,
                    allow_nan=False,
                )
            )
            self._validate_expected_authority()
        require(
            (
                self.authority_expectation is None
                and current_expectation is None
            )
            or (
                self.authority_expectation is not None
                and isinstance(current_expectation, Mapping)
                and exact_json_equal(
                    self.authority_expectation, current_expectation
                )
            ),
            "report repair retained authority expectation changed",
        )
        if validate_phase:
            phase = self._classify_retained_phase()
            require(
                phase == self.phase,
                "report repair transition phase changed",
            )
        for index, (path, descriptor, identity, names) in enumerate(
            self.directory_rows
        ):
            opened = os.fstat(descriptor)
            named = path.lstat()
            observed_names = (
                self._current_submission_reserved(descriptor)
                if index == 0
                else frozenset(os.listdir(descriptor))
            )
            require(
                stat.S_ISDIR(opened.st_mode)
                and self._directory_identity(opened)
                == self._directory_identity(named)
                == identity
                and observed_names == names,
                f"report repair transition directory changed: {path}",
            )
        for path, descriptor, identity, retained_names in (
            self.auxiliary_directory_rows
        ):
            opened = os.fstat(descriptor)
            named = path.lstat()
            require(
                stat.S_ISDIR(opened.st_mode)
                and self._directory_identity(opened)
                == self._directory_identity(named)
                == identity
                and (
                    frozenset(os.listdir(descriptor)) == retained_names
                    if descriptor in self.exact_auxiliary_directory_fds
                    else retained_names <= frozenset(os.listdir(descriptor))
                ),
                f"report repair auxiliary authority changed: {path}",
            )
        for parent_fd, name, descriptor, identity, payload in self.file_rows:
            opened = os.fstat(descriptor)
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            require(
                _file_identity(opened) == _file_identity(named) == identity
                and self._read_descriptor(descriptor) == payload
                and _file_identity(os.fstat(descriptor)) == identity,
                f"report repair transition predecessor changed: {name}",
            )
        for parent_fd, name, descriptor, identity, digest, size in (
            self.stream_file_rows
        ):
            opened = os.fstat(descriptor)
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            observed = hashlib.sha256()
            offset = 0
            while True:
                block = os.pread(descriptor, 16 * 1024 * 1024, offset)
                if not block:
                    break
                observed.update(block)
                offset += len(block)
            require(
                _file_identity(opened) == _file_identity(named) == identity
                and opened.st_size == size
                and offset == size
                and observed.hexdigest() == digest
                and _file_identity(os.fstat(descriptor)) == identity,
                f"report repair retained stream changed: {name}",
            )
        for root in self.scientific_runs:
            self._revalidate_scientific_run(root, full_hash=False)

    def revalidate(self) -> None:
        self._revalidate_retained(validate_phase=True)

    def retained_publication_archive(
        self,
    ) -> tuple[Path, int, str, int] | None:
        rows = [
            row
            for row in self.stream_file_rows
            if row[1].startswith(PUBLICATION_ARCHIVE_PREFIX)
            and row[1].endswith(PUBLICATION_ARCHIVE_SUFFIX)
        ]
        require(len(rows) <= 1, "retained publication archive generation differs")
        if not rows:
            return None
        _parent_fd, name, descriptor, _identity, digest, size = rows[0]
        return self.submission_root / name, descriptor, digest, size

    def retained_direct_file_identity(self, path: Path) -> dict[str, int]:
        """Project creator/recovery identity only from this retained graph."""

        lexical = path.absolute()
        if (
            lexical.parent == self.submission_root
            and lexical.name.startswith(PUBLICATION_ARCHIVE_PREFIX)
            and lexical.name.endswith(PUBLICATION_ARCHIVE_SUFFIX)
        ):
            rows = [row for row in self.stream_file_rows if row[1] == lexical.name]
            require(len(rows) == 1, "retained publication identity is absent")
            _parent_fd, _name, descriptor, identity, _digest, size = rows[0]
            info = os.fstat(descriptor)
            require(
                _file_identity(info) == identity and info.st_size == size,
                "retained publication identity changed",
            )
            return _direct_final_file_identity(info)
        _parent_fd, _name, descriptor, identity, _payload = self._retained_file_row(
            lexical
        )
        info = os.fstat(descriptor)
        require(
            _file_identity(info) == identity,
            "retained direct-final identity changed",
        )
        return _direct_final_file_identity(info)

    def retained_direct_file_descriptor(self, path: Path) -> int:
        lexical = path.absolute()
        _parent_fd, _name, descriptor, identity, _payload = (
            self._retained_file_row(lexical)
        )
        require(
            _file_identity(os.fstat(descriptor)) == identity,
            "retained direct-final descriptor changed",
        )
        return descriptor

    def admit_direct_source_tree(
        self, attempt_root: Path, source_root: Path
    ) -> None:
        """Admit only the final attempt-0002/source directory creation."""

        repair_parent = attempt_root.parent
        parent_found = False
        for index, (path, descriptor, identity, names) in enumerate(
            list(self.directory_rows)
        ):
            expected = names
            opened = os.fstat(descriptor)
            named = path.lstat()
            if path == repair_parent:
                expected = frozenset(set(names) | {attempt_root.name})
                parent_found = True
                require(
                    (opened.st_dev, opened.st_ino, opened.st_uid)
                    == (named.st_dev, named.st_ino, named.st_uid)
                    == identity[:3]
                    and opened.st_nlink == named.st_nlink == identity[3] + 1
                    and stat.S_IMODE(opened.st_mode)
                    == stat.S_IMODE(named.st_mode)
                    == identity[4]
                    and frozenset(os.listdir(descriptor)) == expected,
                    f"direct-final source parent transition differs: {path}",
                )
                self.directory_rows[index] = (
                    path,
                    descriptor,
                    self._directory_identity(opened),
                    expected,
                )
            else:
                observed = (
                    self._current_submission_reserved(descriptor)
                    if path == self.submission_root
                    else frozenset(os.listdir(descriptor))
                )
                require(
                    self._directory_identity(opened)
                    == self._directory_identity(named)
                    == identity
                    and observed == expected,
                    f"direct-final source parent transition differs: {path}",
                )
        require(parent_found, "direct-final source repair parent is unbound")
        attempt_fd = self._open_directory(attempt_root)
        source_fd = self._open_directory(source_root)
        require(
            os.listdir(attempt_fd) == [source_root.name]
            and os.listdir(source_fd) == [],
            "direct-final source directory append differs",
        )
        self._revalidate_retained(validate_phase=False)

    def advance_direct_source_file(self, path: Path, payload: bytes) -> None:
        """Admit one exact final source file without permitting any residue."""

        source_fd: int | None = None
        for index, (row_path, descriptor, identity, names) in enumerate(
            list(self.directory_rows)
        ):
            expected = names
            if row_path == path.parent:
                require(path.name not in names, "direct-final source file already bound")
                expected = frozenset(set(names) | {path.name})
                source_fd = descriptor
                self.directory_rows[index] = (
                    row_path,
                    descriptor,
                    identity,
                    expected,
                )
            opened = os.fstat(descriptor)
            named = row_path.lstat()
            observed = (
                self._current_submission_reserved(descriptor)
                if row_path == self.submission_root
                else frozenset(os.listdir(descriptor))
            )
            require(
                self._directory_identity(opened)
                == self._directory_identity(named)
                == identity
                and observed == expected,
                f"direct-final source file parent changed: {row_path}",
            )
        require(source_fd is not None, "direct-final source root is unbound")
        self._retain_file(source_fd, path.name)
        require(
            self.file_rows[-1][4] == payload,
            f"direct-final source file bytes differ: {path.name}",
        )
        self._revalidate_retained(validate_phase=False)

    def seal_direct_source_root(self, source_root: Path) -> None:
        """Admit only the final 0700 -> 0555 source-directory seal."""

        found = False
        for index, (path, descriptor, identity, names) in enumerate(
            list(self.directory_rows)
        ):
            if path != source_root:
                continue
            opened = os.fstat(descriptor)
            named = path.lstat()
            require(
                (opened.st_dev, opened.st_ino, opened.st_uid, opened.st_nlink)
                == (named.st_dev, named.st_ino, named.st_uid, named.st_nlink)
                == identity[:4]
                and identity[4] == 0o700
                and stat.S_IMODE(opened.st_mode)
                == stat.S_IMODE(named.st_mode)
                == 0o555
                and frozenset(os.listdir(descriptor)) == names,
                "direct-final source root seal differs",
            )
            self.directory_rows[index] = (
                path,
                descriptor,
                self._directory_identity(opened),
                names,
            )
            found = True
        require(found, "direct-final source root seal is unbound")
        self._revalidate_retained(validate_phase=False)

    def advance_after_append(
        self,
        path: Path,
        value: Mapping[str, Any],
        digest: str,
    ) -> None:
        """Admit exactly one append while retaining every prior authority FD."""

        journal = self.submission_root / "journal"
        require(
            path.parent == journal
            and path.name not in {"", ".", ".."}
            and (
                self.phase.staged_target is None
                or self.phase.staged_target == path.name
            ),
            "report repair transition append target differs",
        )
        payload = (
            json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        require(
            hashlib.sha256(payload).hexdigest() == digest,
            "report repair transition append digest differs",
        )
        require(
            exact_json_equal(self.locks.bindings(), self.lock_bindings),
            "report repair transition lock binding changed after append",
        )
        rebound_phase = _classify_repair_phase(
            self.submission_root,
            source_must_be_installed=self.source_must_be_installed,
        )
        expected_durable = tuple(
            sorted(set(self.phase.durable_names) | {path.name})
        )
        require(
            rebound_phase.durable_names == expected_durable
            and rebound_phase.virtual_names == expected_durable
            and rebound_phase.staged_target is None
            and rebound_phase.staged_mode is None
            and rebound_phase.staged_linked is False
            and rebound_phase.report_present == self.phase.report_present
            and rebound_phase.journal_identity == self.phase.journal_identity,
            "report repair transition append phase differs",
        )

        journal_fd: int | None = None
        stage_name = f".{path.name}.seal.tmp"
        for index, (row_path, descriptor, identity, names) in enumerate(
            list(self.directory_rows)
        ):
            opened = os.fstat(descriptor)
            named = row_path.lstat()
            expected_names = names
            if row_path == journal:
                journal_fd = descriptor
                expected_names = frozenset(
                    (set(names) - {stage_name}) | {path.name}
                )
                self.directory_rows[index] = (
                    row_path,
                    descriptor,
                    identity,
                    expected_names,
                )
            observed_names = (
                self._current_submission_reserved(descriptor)
                if row_path == self.submission_root
                else frozenset(os.listdir(descriptor))
            )
            require(
                stat.S_ISDIR(opened.st_mode)
                and self._directory_identity(opened)
                == self._directory_identity(named)
                == identity
                and observed_names == expected_names,
                f"report repair transition directory changed after append: {row_path}",
            )
        require(journal_fd is not None, "report repair transition journal is absent")

        named_target = os.stat(
            path.name, dir_fd=journal_fd, follow_symlinks=False
        )
        require(
            stat.S_ISREG(named_target.st_mode)
            and named_target.st_uid == os.getuid()
            and named_target.st_nlink == 1
            and stat.S_IMODE(named_target.st_mode) == 0o444
            and named_target.st_size == len(payload),
            "report repair transition appended target identity differs",
        )
        retained_target_fd: int | None = None
        retained_rows: list[
            tuple[int, str, int, tuple[int, int, int, int, int, int], bytes]
        ] = []
        for row in self.file_rows:
            parent_fd, name, descriptor, identity, retained_payload = row
            if parent_fd == journal_fd and name in {path.name, stage_name}:
                opened = os.fstat(descriptor)
                require(
                    (opened.st_dev, opened.st_ino)
                    == (named_target.st_dev, named_target.st_ino)
                    and opened.st_uid == named_target.st_uid == os.getuid()
                    and opened.st_nlink == named_target.st_nlink == 1
                    and stat.S_IMODE(opened.st_mode)
                    == stat.S_IMODE(named_target.st_mode)
                    == 0o444
                    and self._read_descriptor(descriptor) == payload,
                    "report repair transition retained stage changed after append",
                )
                if retained_target_fd is None:
                    retained_target_fd = descriptor
                else:
                    os.close(descriptor)
                continue
            opened = os.fstat(descriptor)
            named = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            require(
                _file_identity(opened) == _file_identity(named) == identity
                and self._read_descriptor(descriptor) == retained_payload
                and _file_identity(os.fstat(descriptor)) == identity,
                f"report repair transition predecessor changed after append: {name}",
            )
            retained_rows.append(row)
        self.file_rows = retained_rows
        if retained_target_fd is None:
            retained_target_fd = os.open(
                path.name,
                os.O_RDONLY
                | getattr(os, "O_NOFOLLOW", 0)
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NONBLOCK", 0),
                dir_fd=journal_fd,
            )
        retained_target = os.fstat(retained_target_fd)
        require(
            _file_identity(retained_target) == _file_identity(named_target)
            and self._read_descriptor(retained_target_fd) == payload,
            "report repair transition appended target bytes differ",
        )
        self.file_rows.append(
            (
                journal_fd,
                path.name,
                retained_target_fd,
                _file_identity(retained_target),
                payload,
            )
        )
        self.phase = rebound_phase
        self.revalidate()

    def seal_successor(self, path: Path, value: Mapping[str, Any]) -> str:
        """Directly create a successor and retain the creator FD continuously."""

        payload = (
            json.dumps(dict(value), sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode("utf-8")
        self.revalidate()
        digest, size = self.create_direct_final_file(
            path, payload, label=f"repair successor {path.name}"
        )
        require(size == len(payload), "repair successor size differs")
        rebound_phase = _classify_repair_phase(
            self.submission_root,
            source_must_be_installed=self.source_must_be_installed,
        )
        expected_durable = tuple(
            sorted(set(self.phase.durable_names) | {path.name})
        )
        require(
            rebound_phase.durable_names == expected_durable
            and rebound_phase.virtual_names == expected_durable
            and rebound_phase.staged_target is None
            and rebound_phase.report_present == self.phase.report_present,
            "report repair direct successor phase differs",
        )
        self.phase = rebound_phase
        self.revalidate()
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
        for _parent_fd, _name, descriptor, _identity, _digest, _size in reversed(
            self.stream_file_rows
        ):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.stream_file_rows.clear()
        for _parent_fd, _name, descriptor, _identity, _payload in reversed(
            self.file_rows
        ):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.file_rows.clear()
        for _path, descriptor, _identity, _names in reversed(self.directory_rows):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.directory_rows.clear()
        for _path, descriptor, _identity, _names in reversed(
            self.auxiliary_directory_rows
        ):
            try:
                os.close(descriptor)
            except OSError:
                pass
        self.auxiliary_directory_rows.clear()
        self.exact_auxiliary_directory_fds.clear()

    def __enter__(self) -> "_RepairTransitionBinding":
        return self

    def __exit__(self, _kind: object, _value: object, _traceback: object) -> None:
        self.close()


_ACTIVE_REPAIR_TRANSITION: contextvars.ContextVar[
    _RepairTransitionBinding | None
] = contextvars.ContextVar("active_exp23_repair_transition", default=None)


@contextlib.contextmanager
def _retained_transition_scope(
    submission_root: Path,
    locks: _RepairLocks,
    *,
    source_must_be_installed: bool,
) -> Any:
    """Retain one full local authority boundary across evidence and mutation."""

    inherited = _ACTIVE_REPAIR_TRANSITION.get()
    if inherited is not None:
        require(
            inherited.submission_root == submission_root
            and inherited.locks is locks
            and (
                inherited.source_must_be_installed
                or not source_must_be_installed
            ),
            "nested report repair transition authority differs",
        )
        inherited.revalidate()
        yield inherited
        inherited.revalidate()
        return
    transition = _RepairTransitionBinding(
        submission_root,
        locks,
        source_must_be_installed=source_must_be_installed,
    )
    token = _ACTIVE_REPAIR_TRANSITION.set(transition)
    try:
        transition.revalidate()
        yield transition
        transition.revalidate()
    finally:
        _ACTIVE_REPAIR_TRANSITION.reset(token)
        transition.close()


def _active_transition_required(
    submission_root: Path, locks: _RepairLocks
) -> _RepairTransitionBinding:
    transition = _ACTIVE_REPAIR_TRANSITION.get()
    require(
        transition is not None
        and transition.submission_root == submission_root
        and transition.locks is locks,
        "retained report repair transition is absent",
    )
    return transition


def _seal_transition_json(
    submission_root: Path,
    path: Path,
    value: Mapping[str, Any],
    locks: _RepairLocks,
    *,
    source_must_be_installed: bool,
) -> str:
    """Seal one successor only after a retained full-boundary recheck."""

    inherited = _ACTIVE_REPAIR_TRANSITION.get()
    if inherited is not None:
        require(
            inherited.submission_root == submission_root
            and inherited.locks is locks
            and (
                inherited.source_must_be_installed
                or not source_must_be_installed
            ),
            "active report repair transition differs",
        )
        return inherited.seal_successor(path, value)
    with _RepairTransitionBinding(
        submission_root,
        locks,
        source_must_be_installed=source_must_be_installed,
    ) as transition:
        return transition.seal_successor(path, value)


def _phase_staged_value(
    submission_root: Path, phase: _RepairPhaseSnapshot
) -> dict[str, Any] | None:
    if phase.staged_target is None or phase.staged_mode != 0o444:
        return None
    stage = (
        submission_root
        / "journal"
        / f".{phase.staged_target}.seal.tmp"
    )
    payload, _digest, info = _regular_bytes(
        stage, f"report repair staged {phase.staged_target}"
    )
    require(
        stat.S_IMODE(info.st_mode) == 0o444
        and info.st_uid == os.getuid()
        and info.st_nlink == (2 if phase.staged_linked else 1),
        "report repair phase stage identity changed",
    )
    return _validated_virtual_journal_payload(phase.staged_target, payload)


def _legacy_discard_phase_next_partial_stage(
    submission_root: Path,
    phase: _RepairPhaseSnapshot,
) -> None:
    require(
        phase.staged_target is not None
        and phase.staged_mode == 0o600
        and not phase.staged_linked
        and phase.staged_identity is not None,
        "report repair partial stage is not discardable",
    )
    journal = submission_root / "journal"
    journal_fd = os.open(
        journal,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0),
    )
    stage_name = f".{phase.staged_target}.seal.tmp"
    descriptor = -1
    try:
        parent = os.fstat(journal_fd)
        named_parent = journal.lstat()
        expected_names = set(phase.durable_names) | {stage_name}
        staged = os.stat(stage_name, dir_fd=journal_fd, follow_symlinks=False)
        require(
            (
                parent.st_dev,
                parent.st_ino,
                parent.st_uid,
                parent.st_nlink,
                stat.S_IMODE(parent.st_mode),
            )
            == phase.journal_identity
            and (
                named_parent.st_dev,
                named_parent.st_ino,
                named_parent.st_uid,
                named_parent.st_nlink,
                stat.S_IMODE(named_parent.st_mode),
            )
            == phase.journal_identity
            and stat.S_ISREG(staged.st_mode)
            and (
                staged.st_dev,
                staged.st_ino,
                staged.st_uid,
                staged.st_nlink,
                stat.S_IMODE(staged.st_mode),
                staged.st_size,
            )
            == phase.staged_identity
            and phase.staged_target not in set(os.listdir(journal_fd))
            and set(os.listdir(journal_fd)) == expected_names,
            "report repair partial stage binding changed",
        )
        descriptor = os.open(
            stage_name,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            dir_fd=journal_fd,
        )
        opened = os.fstat(descriptor)
        require(
            (
                opened.st_dev,
                opened.st_ino,
                opened.st_uid,
                opened.st_nlink,
                stat.S_IMODE(opened.st_mode),
                opened.st_size,
            )
            == phase.staged_identity,
            "report repair partial stage raced",
        )
        os.fsync(descriptor)
        rebound_parent = os.fstat(journal_fd)
        rebound_named_parent = journal.lstat()
        rebound_stage = os.stat(
            stage_name, dir_fd=journal_fd, follow_symlinks=False
        )
        require(
            (
                rebound_parent.st_dev,
                rebound_parent.st_ino,
                rebound_parent.st_uid,
                rebound_parent.st_nlink,
                stat.S_IMODE(rebound_parent.st_mode),
            )
            == phase.journal_identity
            and (
                rebound_named_parent.st_dev,
                rebound_named_parent.st_ino,
                rebound_named_parent.st_uid,
                rebound_named_parent.st_nlink,
                stat.S_IMODE(rebound_named_parent.st_mode),
            )
            == phase.journal_identity
            and (
                rebound_stage.st_dev,
                rebound_stage.st_ino,
                rebound_stage.st_uid,
                rebound_stage.st_nlink,
                stat.S_IMODE(rebound_stage.st_mode),
                rebound_stage.st_size,
            )
            == phase.staged_identity,
            "report repair partial stage binding changed before discard",
        )
        require(
            set(os.listdir(journal_fd)) == expected_names,
            "report repair partial stage namespace changed before discard",
        )
        os.unlink(stage_name, dir_fd=journal_fd)
        require(
            os.fstat(descriptor).st_nlink == 0
            and set(os.listdir(journal_fd)) == set(phase.durable_names),
            "report repair partial stage unlink differs",
        )
        os.fsync(journal_fd)
        final_parent = os.fstat(journal_fd)
        final_named_parent = journal.lstat()
        require(
            (
                final_parent.st_dev,
                final_parent.st_ino,
                final_parent.st_uid,
                final_parent.st_nlink,
                stat.S_IMODE(final_parent.st_mode),
            )
            == phase.journal_identity
            and (
                final_named_parent.st_dev,
                final_named_parent.st_ino,
                final_named_parent.st_uid,
                final_named_parent.st_nlink,
                stat.S_IMODE(final_named_parent.st_mode),
            )
            == phase.journal_identity
            and stage_name not in set(os.listdir(journal_fd))
            and set(os.listdir(journal_fd)) == set(phase.durable_names),
            "report repair partial stage discard binding differs",
        )
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        os.close(journal_fd)


def _discard_phase_next_partial_stage(
    submission_root: Path,
    phase: _RepairPhaseSnapshot,
) -> None:
    """Refuse every named attempt-2 staging artifact without touching it."""

    del submission_root, phase
    raise RepairError(
        "attempt2 local staging is permanent fail-stop evidence"
    )


def _rebind_scheduler_calling_before_mutation(
    submission_root: Path,
    *,
    source: Mapping[str, Any],
    calling_path: Path,
    calling: Mapping[str, Any],
    calling_sha256: str,
    locks: _RepairLocks,
    label: str,
) -> dict[str, Any]:
    """Rebind every filesystem authority after CALLING and before mutation."""

    phase = _classify_repair_phase(
        submission_root, source_must_be_installed=True
    )
    require(
        phase.staged_target is None,
        f"{label} conflicts with a staged repair successor",
    )
    _require_report_install_probe_namespace(
        submission_root, allow_exact_crash_residue=False
    )
    _validate_sealed_repair_source(_repair_source_root(submission_root), source)
    locks.bindings()
    rebound, rebound_sha256, rebound_info = read_json(calling_path, label)
    require(
        stat.S_IMODE(rebound_info.st_mode) == 0o444
        and rebound_info.st_uid == os.getuid()
        and rebound_info.st_nlink == 1
        and rebound_sha256 == calling_sha256
        and exact_json_equal(rebound, calling),
        f"{label} binding changed before scheduler mutation",
    )
    return rebound


def _require_repair_prefix_graph(
    submission_root: Path,
    names: Sequence[str],
    *,
    report_present: bool,
    validate_disk_cleanup: bool = True,
) -> None:
    """Reject impossible append-only prefixes before any new durable action."""

    present = set(names)
    fixed_attempt1 = set(EXPECTED_ATTEMPT1_CHAIN_SHA256)
    require(
        fixed_attempt1 <= present,
        "report repair fixed attempt1 chain is incomplete",
    )
    attempt1_terminal = "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json"
    predecessor = "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
    calling = "CALLING_REPORT_REPAIR_0002_SUBMIT.json"
    submitted = "REPORT_REPAIR_0002_SUBMITTED.json"
    submit_failure = "REPORT_REPAIR_0002_TERMINAL_SUBMIT_FAILURE.json"
    authorized = "REPORT_REPAIR_0002_AUTHORIZED.json"
    released = "REPORT_REPAIR_0002_RELEASED.json"
    release_denied = "REPORT_REPAIR_0002_TERMINAL_RELEASE_DENIED.json"
    worker_failure = "REPORT_REPAIR_0002_TERMINAL_WORKER_FAILURE.json"
    completed = "REPORT_REPAIR_0002_COMPLETED.json"

    require(
        predecessor not in present or attempt1_terminal in present,
        "attempt2 predecessor lacks attempt1 terminal evidence",
    )
    require(
        calling not in present
        or {attempt1_terminal, predecessor} <= present,
        "attempt2 submit calling lacks predecessor evidence",
    )
    require(
        submitted not in present or calling in present,
        "attempt2 submitted evidence lacks submit calling",
    )
    require(
        submit_failure not in present or calling in present,
        "attempt2 submit-failure terminal lacks submit calling",
    )
    require(
        authorized not in present or submitted in present,
        "attempt2 authorization lacks submitted evidence",
    )

    release_callings: dict[int, str] = {}
    release_results: dict[int, str] = {}
    for name in present:
        match = re.fullmatch(
            r"CALLING_REPORT_REPAIR_0002_RELEASE_([0-9]{4})\.json", name
        )
        if match is not None:
            release_callings[int(match.group(1))] = name
        match = re.fullmatch(
            r"REPORT_REPAIR_0002_RELEASE_RESULT_([0-9]{4})\.json", name
        )
        if match is not None:
            release_results[int(match.group(1))] = name
    release_successors = {
        *release_callings.values(),
        *release_results.values(),
        released,
        release_denied,
        worker_failure,
        completed,
    } & present
    require(
        not release_successors or authorized in present,
        "attempt2 release successor lacks authorization evidence",
    )
    require(
        (not release_callings
         or set(release_callings) == set(range(max(release_callings) + 1)))
        and set(release_results) <= set(release_callings)
        and (
            not release_results
            or set(release_results) == set(range(max(release_results) + 1))
        )
        and len(release_callings) - len(release_results) in {0, 1}
        and (
            len(release_callings) == len(release_results)
            or max(release_callings) not in release_results
        )
        and len(release_callings) <= 3,
        "attempt2 release prefix is not gap-free",
    )
    require(
        released not in present
        or (
            bool(release_results)
            and set(release_callings) == set(release_results)
        ),
        "attempt2 release terminal lacks a paired release prefix",
    )
    require(
        worker_failure not in present
        or (
            authorized in present
            and (
                released in present
                or (
                    bool(release_results)
                    and set(release_callings) == set(release_results)
                )
            )
        ),
        "attempt2 worker-failure terminal lacks complete release evidence",
    )
    require(
        release_denied not in present
        or not (
            set(release_callings.values())
            | set(release_results.values())
            | {released, worker_failure, completed}
        )
        & present,
        "attempt2 release-denied terminal has release successor state",
    )
    require(
        completed not in present or (report_present and released in present),
        "completed report repair exists without published report/release",
    )
    require(
        not report_present or (authorized in present and released in present),
        "published repaired report lacks authorization/release evidence",
    )

    cancel_authorities: dict[int, str] = {}
    cancel_callings: dict[int, dict[int, str]] = {}
    cancel_results: dict[int, dict[int, str]] = {}
    cancel_terminals: dict[int, str] = {}
    for name in present:
        match = re.fullmatch(
            r"REPORT_REPAIR_0002_CANCEL_AUTHORIZED_([0-9]{4})\.json", name
        )
        if match is not None:
            cancel_authorities[int(match.group(1))] = name
            continue
        match = re.fullmatch(
            r"CALLING_REPORT_REPAIR_0002_SCANCEL_([0-9]{4})_([0-9]{4})\.json",
            name,
        )
        if match is not None:
            generation, index = (int(item) for item in match.groups())
            cancel_callings.setdefault(generation, {})[index] = name
            continue
        match = re.fullmatch(
            r"REPORT_REPAIR_0002_SCANCEL_RESULT_([0-9]{4})_([0-9]{4})\.json",
            name,
        )
        if match is not None:
            generation, index = (int(item) for item in match.groups())
            cancel_results.setdefault(generation, {})[index] = name
            continue
        match = re.fullmatch(
            r"REPORT_REPAIR_0002_CANCEL_TERMINAL_([0-9]{4})\.json", name
        )
        if match is not None:
            cancel_terminals[int(match.group(1))] = name
    cancel_names = {
        *cancel_authorities.values(),
        *(name for rows in cancel_callings.values() for name in rows.values()),
        *(name for rows in cancel_results.values() for name in rows.values()),
        *cancel_terminals.values(),
    }
    require(
        not cancel_names or calling in present,
        "attempt2 cleanup prefix lacks submit calling",
    )
    if cancel_names:
        authority_indices = set(cancel_authorities)
        require(
            bool(authority_indices)
            and authority_indices == set(range(max(authority_indices) + 1))
            and set(cancel_callings) <= authority_indices
            and set(cancel_results) <= authority_indices
            and set(cancel_terminals) <= authority_indices,
            "attempt2 cleanup generation prefix is not gap-free",
        )
        for generation in sorted(authority_indices):
            callings = cancel_callings.get(generation, {})
            results = cancel_results.get(generation, {})
            require(
                (not callings or set(callings) == set(range(max(callings) + 1)))
                and set(results) <= set(callings)
                and (not results or set(results) == set(range(max(results) + 1)))
                and len(callings) - len(results) in {0, 1}
                and (
                    len(callings) == len(results)
                    or max(callings) not in results
                )
                and len(callings) <= 3
                and (
                    generation not in cancel_terminals
                    or set(callings) == set(results)
                )
                and (
                    generation == max(authority_indices)
                    or generation in cancel_terminals
                ),
                "attempt2 cleanup attempt prefix is not gap-free",
            )
    require(
        not (
            submit_failure in present
            and ({submitted, authorized} & present or release_successors)
        )
        and not (release_denied in present and released in present)
        and not (worker_failure in present and release_denied in present)
        and not (
            completed in present
            and (
                submit_failure in present
                or release_denied in present
                or worker_failure in present
                or bool(cancel_names)
            )
        ),
        "report repair terminal/successor prefix is contradictory",
    )
    require(
        not report_present
        or not (
            {submit_failure, release_denied, worker_failure} & present
            or cancel_names
        ),
        "published repaired report conflicts with terminal cleanup state",
    )
    # This read-only validator also proves cleanup generations/attempts are
    # contiguous and have no dependent evidence without an authorization.
    if validate_disk_cleanup:
        _cancel_generation_count(submission_root)


def _validated_attempt1_source(submission_root: Path) -> dict[str, Any]:
    root = _directory(
        submission_root / "report-repair" / "attempt-0001" / "source",
        "attempt1 sealed repair source",
    )
    root_info = root.lstat()
    require(
        root_info.st_uid == os.getuid()
        and root_info.st_nlink == 2
        and stat.S_IMODE(root_info.st_mode) == 0o555
        and {entry.name for entry in os.scandir(root)}
        == {*EXPECTED_ATTEMPT1_SOURCE_FILES, SOURCE_AUTHORITY_NAME},
        "attempt1 sealed repair source identity/coverage differs",
    )
    authority, authority_sha256, authority_info = read_json(
        root / SOURCE_AUTHORITY_NAME, "attempt1 sealed repair source authority"
    )
    expected_authority = {
        "schema_version": 1,
        "repair_source_commit": EXPECTED_ATTEMPT1_SOURCE_COMMIT,
        "repair_package_protocol_sha256": EXPECTED_ATTEMPT1_SOURCE_PROTOCOL,
        "repair_source_files": EXPECTED_ATTEMPT1_SOURCE_FILES,
        "repair_source_files_sha256": EXPECTED_ATTEMPT1_SOURCE_FILES_SHA256,
    }
    require(
        stat.S_IMODE(authority_info.st_mode) == 0o444
        and authority_info.st_uid == os.getuid()
        and authority_info.st_nlink == 1
        and authority_sha256 == EXPECTED_ATTEMPT1_SOURCE_AUTHORITY_SHA256
        and exact_json_equal(authority, expected_authority),
        "attempt1 sealed repair source authority differs",
    )
    for name, expected in EXPECTED_ATTEMPT1_SOURCE_FILES.items():
        payload, digest, info = _regular_bytes(
            root / name, f"attempt1 sealed repair source {name}"
        )
        require(
            len(payload) == expected["size"]
            and digest == expected["sha256"]
            and stat.S_IMODE(info.st_mode) == expected["mode"] == 0o444
            and info.st_uid == os.getuid()
            and info.st_nlink == 1,
            f"attempt1 sealed repair source differs: {name}",
        )
    return {
        "root": str(root),
        "authority": f"report-repair/attempt-0001/source/{SOURCE_AUTHORITY_NAME}",
        "authority_sha256": authority_sha256,
        **expected_authority,
    }


def _validated_attempt1_chain(submission_root: Path) -> dict[str, Any]:
    journal = submission_root / "journal"
    values: dict[str, dict[str, Any]] = {}
    for name, expected_sha256 in EXPECTED_ATTEMPT1_CHAIN_SHA256.items():
        value, digest, info = read_json(journal / name, f"attempt1 chain {name}")
        require(
            stat.S_IMODE(info.st_mode) == 0o444
            and info.st_uid == os.getuid()
            and info.st_nlink == 1
            and digest == expected_sha256,
            f"attempt1 chain identity/hash differs: {name}",
        )
        values[name] = value
    failure = values["REPORT_REPAIR_0001_ORIGINAL_FAILURE.json"]
    calling = values["CALLING_REPORT_REPAIR_0001_SUBMIT.json"]
    submitted = values["REPORT_REPAIR_0001_SUBMITTED.json"]
    authorization = values["REPORT_REPAIR_0001_AUTHORIZED.json"]
    release_calling = values["CALLING_REPORT_REPAIR_0001_RELEASE_0000.json"]
    release_result = values["REPORT_REPAIR_0001_RELEASE_RESULT_0000.json"]
    released = values["REPORT_REPAIR_0001_RELEASED.json"]
    require(
        failure.get("attempt") == PREDECESSOR_ATTEMPT
        and failure.get("submission_sha256") == EXPECTED_SUBMISSION_SHA256
        and calling.get("attempt") == PREDECESSOR_ATTEMPT
        and calling.get("original_failure_evidence")
        == "journal/REPORT_REPAIR_0001_ORIGINAL_FAILURE.json"
        and calling.get("original_failure_evidence_sha256")
        == EXPECTED_ATTEMPT1_CHAIN_SHA256[
            "REPORT_REPAIR_0001_ORIGINAL_FAILURE.json"
        ]
        and submitted.get("attempt") == PREDECESSOR_ATTEMPT
        and submitted.get("submit_calling_sha256")
        == EXPECTED_ATTEMPT1_CHAIN_SHA256[
            "CALLING_REPORT_REPAIR_0001_SUBMIT.json"
        ]
        and submitted.get("repair_report_job_id") == EXPECTED_ATTEMPT1_JOB_ID
        and authorization.get("attempt") == PREDECESSOR_ATTEMPT
        and authorization.get("repair_report_job_id") == EXPECTED_ATTEMPT1_JOB_ID
        and authorization.get("repair_job_name") == EXPECTED_ATTEMPT1_JOB_NAME
        and authorization.get("scheduler_comment") == EXPECTED_ATTEMPT1_COMMENT
        and authorization.get("original_failure_evidence_sha256")
        == EXPECTED_ATTEMPT1_CHAIN_SHA256[
            "REPORT_REPAIR_0001_ORIGINAL_FAILURE.json"
        ]
        and authorization.get("submit_calling_sha256")
        == EXPECTED_ATTEMPT1_CHAIN_SHA256[
            "CALLING_REPORT_REPAIR_0001_SUBMIT.json"
        ]
        and authorization.get("submitted_evidence_sha256")
        == EXPECTED_ATTEMPT1_CHAIN_SHA256[
            "REPORT_REPAIR_0001_SUBMITTED.json"
        ]
        and release_calling.get("attempt") == PREDECESSOR_ATTEMPT
        and release_calling.get("release_attempt") == 0
        and release_calling.get("repair_report_job_id") == EXPECTED_ATTEMPT1_JOB_ID
        and release_calling.get("authorization_sha256")
        == EXPECTED_ATTEMPT1_CHAIN_SHA256[
            "REPORT_REPAIR_0001_AUTHORIZED.json"
        ]
        and release_result.get("attempt") == PREDECESSOR_ATTEMPT
        and release_result.get("release_attempt") == 0
        and release_result.get("repair_report_job_id") == EXPECTED_ATTEMPT1_JOB_ID
        and release_result.get("authorization_sha256")
        == EXPECTED_ATTEMPT1_CHAIN_SHA256[
            "REPORT_REPAIR_0001_AUTHORIZED.json"
        ]
        and release_result.get("release_calling_sha256")
        == EXPECTED_ATTEMPT1_CHAIN_SHA256[
            "CALLING_REPORT_REPAIR_0001_RELEASE_0000.json"
        ]
        and released.get("attempt") == PREDECESSOR_ATTEMPT
        and released.get("repair_report_job_id") == EXPECTED_ATTEMPT1_JOB_ID
        and released.get("authorization_sha256")
        == EXPECTED_ATTEMPT1_CHAIN_SHA256[
            "REPORT_REPAIR_0001_AUTHORIZED.json"
        ]
        and exact_json_equal(
            released.get("release_attempts"),
            [
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
            ],
        ),
        "attempt1 journal chain semantics differ",
    )
    source = _validated_attempt1_source(submission_root)
    require(
        calling.get("repair_source_root") == source["root"]
        and calling.get("repair_source_commit") == source["repair_source_commit"]
        and calling.get("repair_package_protocol_sha256")
        == source["repair_package_protocol_sha256"]
        and exact_json_equal(
            calling.get("repair_source_files"), source["repair_source_files"]
        )
        and calling.get("repair_source_files_sha256")
        == source["repair_source_files_sha256"]
        and authorization.get("repair_source_root") == source["root"]
        and authorization.get("repair_source_commit")
        == source["repair_source_commit"]
        and authorization.get("repair_package_protocol_sha256")
        == source["repair_package_protocol_sha256"]
        and exact_json_equal(
            authorization.get("repair_source_files"), source["repair_source_files"]
        )
        and authorization.get("repair_source_files_sha256")
        == source["repair_source_files_sha256"],
        "attempt1 journal/source binding differs",
    )
    return {
        "schema_version": 1,
        "files": dict(EXPECTED_ATTEMPT1_CHAIN_SHA256),
        "source": source,
    }


def _attempt1_accounting_argv() -> list[str]:
    return [
        "/usr/local/bin/sacct",
        "-X",
        "-n",
        "-j",
        EXPECTED_ATTEMPT1_JOB_ID,
        "-o",
        ",".join(REPAIR_SACCT_FIELDS),
        "-P",
    ]


def _attempt1_terminal_scheduler_observation(
    submission_root: Path,
    contract: Mapping[str, Any],
    runner: Runner,
    locks: _RepairLocks,
) -> dict[str, Any]:
    control_plane = contract.get("scheduler_control_plane_contract")
    require(isinstance(control_plane, Mapping), "attempt1 control plane differs")
    environment = _scheduler_environment(str(control_plane.get("slurm_conf", "")))
    result, raw = _run(
        runner,
        _attempt1_accounting_argv(),
        submission_root,
        environment,
        locks,
    )
    require(
        result.returncode == 0
        and result.stderr == b""
        and hashlib.sha256(result.stdout).hexdigest()
        == EXPECTED_ATTEMPT1_TERMINAL_SACCT_STDOUT_SHA256,
        "attempt1 terminal accounting query differs",
    )
    try:
        lines = [line for line in result.stdout.decode("utf-8").splitlines() if line]
    except UnicodeDecodeError as exc:
        raise RepairError(f"attempt1 terminal accounting is not UTF-8: {exc}") from exc
    require(len(lines) == 1, "attempt1 terminal accounting row count differs")
    row = lines[0].split("|")
    require(
        len(row) == len(REPAIR_SACCT_FIELDS),
        "attempt1 terminal accounting field count differs",
    )
    parsed = dict(zip(REPAIR_SACCT_FIELDS, row, strict=True))
    require(
        parsed["JobIDRaw"] == EXPECTED_ATTEMPT1_JOB_ID
        and parsed["JobName"] == EXPECTED_ATTEMPT1_JOB_NAME
        and parsed["User"] == environment["USER"]
        and parsed["State"] == "FAILED"
        and parsed["ExitCode"] == "2:0"
        and parsed["ElapsedRaw"] == "5"
        and parsed["AllocNodes"] == "1"
        and parsed["NodeList"] == "cpu-00049"
        and parsed["Submit"] == "2026-08-29T13:22:38"
        and parsed["Eligible"] == "2026-08-29T13:22:41"
        and parsed["Start"] == "2026-08-29T13:22:43"
        and parsed["End"] == "2026-08-29T13:22:48"
        and parsed["Submit"] <= parsed["Eligible"] <= parsed["Start"] < parsed["End"]
        and parsed["Comment"] == EXPECTED_ATTEMPT1_COMMENT
        and parsed["Reason"] == "None",
        "attempt1 terminal accounting semantics differ",
    )
    canonical = {
        "schema_version": 1,
        "fields": list(REPAIR_SACCT_FIELDS),
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


def _validated_attempt1_terminal_scheduler_observation(
    value: Mapping[str, Any], contract: Mapping[str, Any]
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
        and set(value["canonical"]) == {"schema_version", "fields", "rows"}
        and value.get("canonical_sha256") == stable_hash(value["canonical"])
        and isinstance(value.get("parsed_row"), Mapping),
        "attempt1 terminal scheduler observation differs",
    )
    parsed = value["parsed_row"]
    expected_environment = _contract_scheduler_environment(contract)
    require(
        set(parsed) == set(REPAIR_SACCT_FIELDS)
        and parsed["JobIDRaw"] == EXPECTED_ATTEMPT1_JOB_ID
        and parsed["JobName"] == EXPECTED_ATTEMPT1_JOB_NAME
        and parsed["User"] == expected_environment["USER"]
        and parsed["State"] == "FAILED"
        and parsed["ExitCode"] == "2:0"
        and parsed["ElapsedRaw"] == "5"
        and parsed["AllocNodes"] == "1"
        and parsed["NodeList"] == "cpu-00049"
        and parsed["Submit"] == "2026-08-29T13:22:38"
        and parsed["Eligible"] == "2026-08-29T13:22:41"
        and parsed["Start"] == "2026-08-29T13:22:43"
        and parsed["End"] == "2026-08-29T13:22:48"
        and parsed["Submit"] <= parsed["Eligible"] <= parsed["Start"] < parsed["End"]
        and parsed["Comment"] == EXPECTED_ATTEMPT1_COMMENT
        and parsed["Reason"] == "None",
        "attempt1 terminal scheduler row differs",
    )
    canonical = value["canonical"]
    require(
        exact_json_equal(
            canonical,
            {
                "schema_version": 1,
                "fields": list(REPAIR_SACCT_FIELDS),
                "rows": [[parsed[field] for field in REPAIR_SACCT_FIELDS]],
            },
        ),
        "attempt1 terminal scheduler canonical row differs",
    )
    result = _validated_command_evidence(
        value["raw"],
        label="attempt1 terminal accounting",
        expected_argv=_attempt1_accounting_argv(),
        expected_environment=expected_environment,
    )
    require(
        result.returncode == 0
        and result.stderr == b""
        and hashlib.sha256(result.stdout).hexdigest()
        == EXPECTED_ATTEMPT1_TERMINAL_SACCT_STDOUT_SHA256
        and result.stdout
        == ("|".join(parsed[field] for field in REPAIR_SACCT_FIELDS) + "\n").encode(
            "utf-8"
        ),
        "attempt1 terminal scheduler raw row differs",
    )
    return dict(value)


def _validated_attempt1_worker_failure_terminal(
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str] | None:
    path = _attempt1_worker_failure_terminal_path(submission_root)
    durable_identity = candidate is None
    if durable_identity:
        if not os.path.lexists(path):
            return None
        value, digest, info = read_json(path, "attempt1 worker-failure terminal")
        identity_valid = (
            stat.S_IMODE(info.st_mode) == 0o444
            and info.st_uid == os.getuid()
            and info.st_nlink == 1
        )
    else:
        value = dict(candidate)
        digest = _pretty_json_sha(value)
        identity_valid = True
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
        identity_valid
        and set(value) == ATTEMPT1_TERMINAL_KEYS
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("status") == "report_repair_terminal_worker_failure"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and type(value.get("attempt")) is int
        and value.get("attempt") == PREDECESSOR_ATTEMPT
        and value.get("repair_report_job_id") == EXPECTED_ATTEMPT1_JOB_ID
        and value.get("authorization_sha256")
        == EXPECTED_ATTEMPT1_CHAIN_SHA256[
            "REPORT_REPAIR_0001_AUTHORIZED.json"
        ]
        and value.get("reason")
        == "repair_worker_terminal_after_release_evidence"
        and exact_json_equal(value.get("release_attempts"), expected_release_attempts)
        and value.get("release_attempts_sha256")
        == stable_hash(expected_release_attempts)
        and value.get("released_evidence")
        == "journal/REPORT_REPAIR_0001_RELEASED.json"
        and value.get("released_evidence_sha256")
        == EXPECTED_ATTEMPT1_CHAIN_SHA256["REPORT_REPAIR_0001_RELEASED.json"]
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
        "attempt1 worker-failure terminal differs",
    )
    census = _validated_scheduler_census(value["post_release_census"], contract)
    expected_environment = _scheduler_environment(
        str(contract["scheduler_control_plane_contract"]["slurm_conf"])
    )
    require(
        census["settled_rows"] == []
        and all(
            exact_json_equal(
                round_value["raw"]["environment"], expected_environment
            )
            for round_value in census["rounds"]
        ),
        "attempt1 worker-failure terminal census is not empty",
    )
    _validated_attempt1_terminal_scheduler_observation(
        value["terminal_scheduler_observation"], contract
    )
    return dict(value), digest


def _seal_attempt1_worker_failure_terminal(
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    census: Mapping[str, Any],
    accounting: Mapping[str, Any],
    locks: _RepairLocks,
) -> tuple[dict[str, Any], str]:
    require(
        _validated_scheduler_census(census, contract)["settled_rows"] == [],
        "attempt1 terminal cannot be sealed with active relevant jobs",
    )
    _validated_attempt1_terminal_scheduler_observation(accounting, contract)
    release_attempts = [
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
    value = {
        "schema_version": 1,
        "status": "report_repair_terminal_worker_failure",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "attempt": PREDECESSOR_ATTEMPT,
        "repair_report_job_id": EXPECTED_ATTEMPT1_JOB_ID,
        "authorization_sha256": EXPECTED_ATTEMPT1_CHAIN_SHA256[
            "REPORT_REPAIR_0001_AUTHORIZED.json"
        ],
        "reason": "repair_worker_terminal_after_release_evidence",
        "release_attempts": release_attempts,
        "release_attempts_sha256": stable_hash(release_attempts),
        "released_evidence": "journal/REPORT_REPAIR_0001_RELEASED.json",
        "released_evidence_sha256": EXPECTED_ATTEMPT1_CHAIN_SHA256[
            "REPORT_REPAIR_0001_RELEASED.json"
        ],
        "post_release_census": dict(census),
        "post_release_census_sha256": stable_hash(census),
        "terminal_scheduler_observation": dict(accounting),
        "terminal_scheduler_observation_sha256": stable_hash(accounting),
        "publication_allowed": False,
        "retry_allowed": False,
        "sealed_at_utc": _utc_now(),
    }
    digest = _seal_transition_json(
        submission_root,
        _attempt1_worker_failure_terminal_path(submission_root),
        value,
        locks,
        source_must_be_installed=False,
    )
    validated = _validated_attempt1_worker_failure_terminal(
        submission_root, submission_sha256, contract
    )
    require(validated is not None, "attempt1 worker-failure terminal was not sealed")
    return validated


def _capture_attempt1_log(submission_root: Path) -> dict[str, Any]:
    path = (
        submission_root
        / "logs"
        / f"report-repair-0001-{EXPECTED_ATTEMPT1_JOB_ID}.out"
    )
    payload, digest, info = _regular_bytes(path, "attempt1 repair failure log")
    relative = path.relative_to(submission_root)
    require(
        payload == EXPECTED_ATTEMPT1_LOG_BYTES
        and digest == EXPECTED_ATTEMPT1_LOG_SHA256
        and len(payload) == EXPECTED_ATTEMPT1_LOG_SIZE
        and info.st_uid == os.getuid()
        and info.st_nlink == 1
        and stat.S_IMODE(info.st_mode) == 0o600,
        "attempt1 repair failure log differs",
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


def _attempt1_retained_command(
    submission_root: Path,
    contract: Mapping[str, Any],
    runner: Runner,
    locks: _RepairLocks,
    option: str,
) -> dict[str, Any]:
    require(option in {"--env-vars", "--batch-script"}, "attempt1 retained option differs")
    environment = _scheduler_environment(
        str(contract["scheduler_control_plane_contract"]["slurm_conf"])
    )
    argv = ["/usr/local/bin/sacct", "-j", EXPECTED_ATTEMPT1_JOB_ID, option]
    result, evidence = _run(runner, argv, submission_root, environment, locks)
    expected_sha = (
        EXPECTED_ATTEMPT1_ENV_STDOUT_SHA256
        if option == "--env-vars"
        else EXPECTED_ATTEMPT1_BATCH_STDOUT_SHA256
    )
    expected_size = (
        EXPECTED_ATTEMPT1_ENV_STDOUT_SIZE
        if option == "--env-vars"
        else EXPECTED_ATTEMPT1_BATCH_STDOUT_SIZE
    )
    require(
        result.returncode == 0
        and result.stderr == b""
        and len(result.stdout) == expected_size
        and hashlib.sha256(result.stdout).hexdigest() == expected_sha,
        f"attempt1 retained {option} evidence differs",
    )
    if option == "--env-vars":
        require(
            b"SLURM_EXPORT_ENV=NONE" in result.stdout
            and b"SLURM_RESTART_COUNT" not in result.stdout,
            "attempt1 retained restart environment differs",
        )
    else:
        require(
            b'[[ -n "${SLURM_RESTART_COUNT+x}" && "$SLURM_RESTART_COUNT" == "0" ]]'
            in result.stdout
            and EXPECTED_ATTEMPT1_SOURCE_FILES["report_repair.slurm"]["sha256"]
            == "15ce6712f16c0655b4ad3d544987aec25574531cfc02b470c1ed9395bd363962",
            "attempt1 retained batch restart guard differs",
        )
    return evidence


def _validated_attempt1_retained_command(
    value: Mapping[str, Any],
    contract: Mapping[str, Any],
    option: str,
) -> dict[str, Any]:
    environment = _scheduler_environment(
        str(contract["scheduler_control_plane_contract"]["slurm_conf"])
    )
    result = _validated_command_evidence(
        value,
        label=f"attempt1 retained {option}",
        expected_argv=["/usr/local/bin/sacct", "-j", EXPECTED_ATTEMPT1_JOB_ID, option],
        expected_environment=environment,
    )
    expected_sha = (
        EXPECTED_ATTEMPT1_ENV_STDOUT_SHA256
        if option == "--env-vars"
        else EXPECTED_ATTEMPT1_BATCH_STDOUT_SHA256
    )
    expected_size = (
        EXPECTED_ATTEMPT1_ENV_STDOUT_SIZE
        if option == "--env-vars"
        else EXPECTED_ATTEMPT1_BATCH_STDOUT_SIZE
    )
    require(
        result.returncode == 0
        and result.stderr == b""
        and len(result.stdout) == expected_size
        and hashlib.sha256(result.stdout).hexdigest() == expected_sha,
        f"attempt1 retained {option} raw evidence differs",
    )
    if option == "--env-vars":
        require(
            b"SLURM_EXPORT_ENV=NONE" in result.stdout
            and b"SLURM_RESTART_COUNT" not in result.stdout,
            "attempt1 retained restart environment semantics differ",
        )
    else:
        require(
            b'[[ -n "${SLURM_RESTART_COUNT+x}" && "$SLURM_RESTART_COUNT" == "0" ]]'
            in result.stdout,
            "attempt1 retained batch guard semantics differ",
        )
    return dict(value)


def _attempt2_predecessor_publication_state(
    submission_root: Path,
) -> dict[str, Any]:
    report_absent = not os.path.lexists(submission_root / "report") and not any(
        name.startswith(PUBLICATION_ARCHIVE_PREFIX)
        and name.endswith(PUBLICATION_ARCHIVE_SUFFIX)
        for name in os.listdir(submission_root)
    )
    staging_entries = sorted(
        path.name for path in submission_root.glob(".report.tmp.*")
    )
    original_cleanup = [
        relative
        for relative in (
            "CANCEL_REQUESTED.json",
            "journal/PREREQUISITE_MISSING.json",
            "journal/9000_RECOVERY_CANCELLED.json",
            "journal/9001_PRODUCTION_PREREQUISITE_MISSING.json",
        )
        if os.path.lexists(submission_root / relative)
    ]
    require(
        report_absent and not staging_entries and not original_cleanup,
        "attempt2 predecessor publication state differs",
    )
    return {
        "report_absent": True,
        "staging_entries": [],
        "cleanup_prefixes": [],
        "journal_directory": str(submission_root / "journal"),
    }


def _validated_attempt2_predecessor(
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    receipt_map: Mapping[str, Any],
    locks: _RepairLocks,
    *,
    candidate: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], str] | None:
    path = _attempt2_predecessor_path(submission_root)
    durable_identity = candidate is None
    if durable_identity:
        if not os.path.lexists(path):
            return None
        value, digest, info = read_json(path, "attempt2 predecessor failure")
        identity_valid = (
            stat.S_IMODE(info.st_mode) == 0o444
            and info.st_uid == os.getuid()
            and info.st_nlink == 1
        )
    else:
        value = dict(candidate)
        digest = _pretty_json_sha(value)
        identity_valid = True
    terminal = _validated_attempt1_worker_failure_terminal(
        submission_root, submission_sha256, contract
    )
    require(terminal is not None, "attempt2 predecessor lacks attempt1 terminal")
    terminal_value, terminal_sha256 = terminal
    chain = _validated_attempt1_chain(submission_root)
    transaction, report_cancel = locks.bindings()
    require(
        identity_valid
        and set(value) == ATTEMPT2_PREDECESSOR_KEYS
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("status")
        == "attempt1_terminal_failure_authorized_for_attempt2"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and type(value.get("attempt")) is int
        and value.get("attempt") == ATTEMPT
        and type(value.get("predecessor_attempt")) is int
        and value.get("predecessor_attempt") == PREDECESSOR_ATTEMPT
        and value.get("predecessor_report_job_id") == EXPECTED_ATTEMPT1_JOB_ID
        and value.get("predecessor_job_name") == EXPECTED_ATTEMPT1_JOB_NAME
        and value.get("predecessor_scheduler_comment") == EXPECTED_ATTEMPT1_COMMENT
        and exact_json_equal(value.get("predecessor_chain"), chain["files"])
        and value.get("predecessor_chain_sha256") == stable_hash(chain["files"])
        and exact_json_equal(value.get("predecessor_source"), chain["source"])
        and value.get("predecessor_source_sha256") == stable_hash(chain["source"])
        and value.get("terminal_worker_failure_evidence")
        == "journal/REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json"
        and value.get("terminal_worker_failure_evidence_sha256")
        == terminal_sha256
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
        and exact_json_equal(value.get("transaction_lock"), transaction)
        and exact_json_equal(value.get("report_cancel_lock"), report_cancel)
        and value.get("publication_allowed") is False
        and value.get("retry_predecessor_allowed") is False
        and value.get("successor_attempt") == ATTEMPT
        and value.get("successor_scheduler_submission_allowed") is True
        and isinstance(value.get("sealed_at_utc"), str)
        and bool(value["sealed_at_utc"]),
        "attempt2 predecessor evidence differs",
    )
    _validated_attempt1_retained_command(
        value["retained_environment_evidence"], contract, "--env-vars"
    )
    _validated_attempt1_retained_command(
        value["retained_batch_script_evidence"], contract, "--batch-script"
    )
    require(
        exact_json_equal(value["failure_log"], _capture_attempt1_log(submission_root))
        and exact_json_equal(
            terminal_value["terminal_scheduler_observation"],
            _validated_attempt1_terminal_scheduler_observation(
                terminal_value["terminal_scheduler_observation"], contract
            ),
        ),
        "attempt2 predecessor durable evidence changed",
    )
    return dict(value), digest


def _seal_attempt2_predecessor(
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    receipt_map: Mapping[str, Any],
    terminal: Mapping[str, Any],
    terminal_sha256: str,
    runner: Runner,
    locks: _RepairLocks,
) -> tuple[dict[str, Any], str]:
    chain = _validated_attempt1_chain(submission_root)
    retained_environment = _attempt1_retained_command(
        submission_root, contract, runner, locks, "--env-vars"
    )
    retained_batch = _attempt1_retained_command(
        submission_root, contract, runner, locks, "--batch-script"
    )
    failure_log = _capture_attempt1_log(submission_root)
    publication_state = _attempt2_predecessor_publication_state(submission_root)
    transaction, report_cancel = locks.bindings()
    value = {
        "schema_version": 1,
        "status": "attempt1_terminal_failure_authorized_for_attempt2",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "attempt": ATTEMPT,
        "predecessor_attempt": PREDECESSOR_ATTEMPT,
        "predecessor_report_job_id": EXPECTED_ATTEMPT1_JOB_ID,
        "predecessor_job_name": EXPECTED_ATTEMPT1_JOB_NAME,
        "predecessor_scheduler_comment": EXPECTED_ATTEMPT1_COMMENT,
        "predecessor_chain": dict(chain["files"]),
        "predecessor_chain_sha256": stable_hash(chain["files"]),
        "predecessor_source": dict(chain["source"]),
        "predecessor_source_sha256": stable_hash(chain["source"]),
        "terminal_worker_failure_evidence": (
            "journal/REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json"
        ),
        "terminal_worker_failure_evidence_sha256": terminal_sha256,
        "retained_environment_evidence": retained_environment,
        "retained_environment_evidence_sha256": stable_hash(retained_environment),
        "retained_batch_script_evidence": retained_batch,
        "retained_batch_script_evidence_sha256": stable_hash(retained_batch),
        "failure_log": failure_log,
        "failure_log_sha256": stable_hash(failure_log),
        "worker_receipt_map": dict(receipt_map),
        "worker_receipt_map_sha256": stable_hash(receipt_map),
        "publication_state": publication_state,
        "restart_failure_classification": {
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
        "transaction_lock": transaction,
        "report_cancel_lock": report_cancel,
        "publication_allowed": False,
        "retry_predecessor_allowed": False,
        "successor_attempt": ATTEMPT,
        "successor_scheduler_submission_allowed": True,
        "sealed_at_utc": _utc_now(),
    }
    require(
        exact_json_equal(
            terminal["terminal_scheduler_observation"],
            _validated_attempt1_terminal_scheduler_observation(
                terminal["terminal_scheduler_observation"], contract
            ),
        ),
        "attempt2 predecessor terminal input differs",
    )
    _classify_repair_phase(
        submission_root, source_must_be_installed=False
    )
    _seal_transition_json(
        submission_root,
        _attempt2_predecessor_path(submission_root),
        value,
        locks,
        source_must_be_installed=False,
    )
    validated = _validated_attempt2_predecessor(
        submission_root, submission_sha256, contract, receipt_map, locks
    )
    require(validated is not None, "attempt2 predecessor was not sealed")
    return validated


def _sbatch_command(
    source_root: Path,
    submission_root: Path,
    submission_sha256: str,
    source: Mapping[str, Any],
    scheduler_input: Mapping[str, Any],
) -> list[str]:
    snapshot_root = submission_root / "source-snapshot" / "repo"
    require(
        source_root == submission_root / SOURCE_ARCHIVE_NAME
        and source.get("repair_source_archive") == str(source_root),
        "repair source archive command path differs",
    )
    require(
        isinstance(scheduler_input, Mapping)
        and set(scheduler_input)
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
        and scheduler_input.get("schema_version") == 1
        and scheduler_input.get("transport") == "inherited_proc_fd"
        and type(scheduler_input.get("descriptor")) is int
        and scheduler_input["descriptor"] >= 0
        and scheduler_input.get("argument")
        == f"/proc/self/fd/{scheduler_input['descriptor']}"
        and scheduler_input.get("source_archive") == str(source_root)
        and scheduler_input.get("sha256")
        == source.get("repair_source_archive_sha256")
        and scheduler_input.get("size")
        == source.get("repair_source_archive_size"),
        "repair source archive scheduler input differs",
    )
    return [
        "/usr/local/bin/sbatch",
        "--parsable",
        "--hold",
        "--no-requeue",
        "--export=NONE",
        f"--job-name={_repair_name(submission_sha256)}",
        f"--comment={_repair_comment(submission_sha256)}",
        f"--output={_repair_log_path(submission_root)}",
        str(scheduler_input["argument"]),
        str(snapshot_root),
        str(submission_root),
        submission_sha256,
        str(ATTEMPT),
        str(source_root),
        str(source["repair_source_archive_sha256"]),
        str(source["repair_source_archive_size"]),
    ]


def _parse_sbatch_job_id(result: CommandResult) -> str:
    require(result.returncode == 0 and result.stderr == b"", "report repair sbatch failed")
    return str(_parsed_sbatch_stdout(result.stdout)["job_id"])


def _parsed_sbatch_stdout(stdout: bytes) -> dict[str, Any]:
    """Parse the exact one-line contract emitted by ``sbatch --parsable``."""

    try:
        text = stdout.decode("ascii")
    except UnicodeDecodeError as exc:
        raise RepairError(f"report repair sbatch stdout is not ASCII: {exc}") from exc
    require(
        text.endswith("\n")
        and text.count("\n") == 1
        and "\r" not in text
        and text[:-1]
        and text[:-1] == text[:-1].strip(),
        "report repair sbatch stdout differs",
    )
    fields = text[:-1].split(";")
    require(len(fields) in {1, 2}, "report repair sbatch stdout field count differs")
    job_id = fields[0]
    cluster = fields[1] if len(fields) == 2 else None
    require(
        JOB_ID_RE.fullmatch(job_id) is not None,
        "report repair sbatch job ID differs",
    )
    require(
        cluster is None or SBATCH_CLUSTER_RE.fullmatch(cluster) is not None,
        "report repair sbatch cluster differs",
    )
    return {"schema_version": 1, "job_id": job_id, "cluster": cluster}


def _validated_submit_failure_terminal(
    submission_root: Path,
    submission_sha256: str,
    calling: Mapping[str, Any],
    calling_sha256: str,
    contract: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    path = _submit_failure_terminal_path(submission_root)
    if candidate is None:
        if not os.path.lexists(path):
            return None
        value, _digest, info = read_json(
            path, "report repair terminal submit failure"
        )
        identity_valid = (
            stat.S_IMODE(info.st_mode) == 0o444
            and info.st_uid == os.getuid()
            and info.st_nlink == 1
        )
    else:
        value = dict(candidate)
        identity_valid = True
    require(
        identity_valid
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
    _validated_scheduler_census(value["post_failure_census"], contract)
    require(
        result.returncode != 0
        and not _repair_rows(
            value["post_failure_census"], submission_sha256, contract
        ),
        "report repair terminal submit failure outcome differs",
    )
    return dict(value)


def _scheduler_source_archive_input(
    transition: _RepairTransitionBinding,
    source_root: Path,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    descriptor = transition.retained_direct_file_descriptor(source_root)
    return {
        "schema_version": 1,
        "transport": "inherited_proc_fd",
        "descriptor": descriptor,
        "argument": f"/proc/self/fd/{descriptor}",
        "source_archive": str(source_root),
        "sha256": source.get("repair_source_archive_sha256"),
        "size": source.get("repair_source_archive_size"),
        "file_identity": transition.retained_direct_file_identity(source_root),
    }


def _scheduler_source_argument_from_calling(
    calling: Mapping[str, Any],
) -> str:
    scheduler_input = calling.get("scheduler_source_archive_input")
    require(
        isinstance(scheduler_input, Mapping)
        and type(scheduler_input.get("descriptor")) is int
        and scheduler_input["descriptor"] >= 0
        and scheduler_input.get("argument")
        == f"/proc/self/fd/{scheduler_input['descriptor']}",
        "report repair scheduler source argument differs",
    )
    return str(scheduler_input["argument"])


def _retained_scheduler_source_argument(
    submission_root: Path, locks: _RepairLocks
) -> str:
    transition = _active_transition_required(submission_root, locks)
    path = _submit_calling_path(submission_root)
    payload = transition._retained_payload(path)
    calling = _decode_json(path, payload)
    require(
        set(calling) == SUBMIT_CALLING_KEYS,
        "retained report repair submit calling differs",
    )
    return _scheduler_source_argument_from_calling(calling)


def _validate_submit_calling(
    value: Mapping[str, Any],
    *,
    submission_root: Path,
    submission_sha256: str,
    failure_sha256: str,
    predecessor_sha256: str,
    source: Mapping[str, Any],
    contract: Mapping[str, Any],
    locks: _RepairLocks,
) -> dict[str, Any]:
    source_root = _repair_source_root(submission_root)
    transition = _active_transition_required(submission_root, locks)
    scheduler_input = value.get("scheduler_source_archive_input")
    require(
        isinstance(scheduler_input, Mapping)
        and set(scheduler_input)
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
        and scheduler_input.get("schema_version") == 1
        and scheduler_input.get("transport") == "inherited_proc_fd"
        and type(scheduler_input.get("descriptor")) is int
        and scheduler_input["descriptor"] >= 0
        and scheduler_input.get("argument")
        == f"/proc/self/fd/{scheduler_input['descriptor']}"
        and scheduler_input.get("source_archive") == str(source_root)
        and scheduler_input.get("sha256")
        == source.get("repair_source_archive_sha256")
        and scheduler_input.get("size")
        == source.get("repair_source_archive_size"),
        "report repair scheduler source archive input differs",
    )
    expected_command = _sbatch_command(
        source_root,
        submission_root,
        submission_sha256,
        source,
        scheduler_input,
    )
    transaction, report_cancel = locks.bindings()
    source_size = source.get("repair_source_archive_size")
    require(type(source_size) is int and source_size > 0, "repair source size differs")
    source_identity = transition.retained_direct_file_identity(source_root)
    claimed_source_identity = _validated_direct_final_file_identity(
        value.get("repair_source_archive_file_identity"),
        size=source_size,
        label="report repair source creation identity",
    )
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
        and value.get("predecessor_failure_evidence")
        == "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
        and value.get("predecessor_failure_evidence_sha256")
        == predecessor_sha256
        and value.get("repair_source_root") == str(source_root)
        and value.get("repair_source_commit") == source.get("repair_source_commit")
        and value.get("repair_package_protocol_sha256")
        == source.get("repair_package_protocol_sha256")
        and exact_json_equal(
            value.get("repair_source_files"), source.get("repair_source_files")
        )
        and value.get("repair_source_files_sha256")
        == source.get("repair_source_files_sha256")
        and value.get("repair_source_installation_method")
        == source.get("repair_source_installation_method")
        == SOURCE_ARCHIVE_INSTALL_METHOD
        and value.get("repair_source_archive")
        == source.get("repair_source_archive")
        == str(source_root)
        and value.get("repair_source_archive_sha256")
        == source.get("repair_source_archive_sha256")
        and value.get("repair_source_archive_size")
        == source.get("repair_source_archive_size")
        and value.get("repair_source_archive_format")
        == source.get("repair_source_archive_format")
        == SOURCE_ARCHIVE_KIND
        and exact_json_equal(claimed_source_identity, source_identity)
        and exact_json_equal(
            scheduler_input.get("file_identity"),
            source_identity,
        )
        and isinstance(value.get("scheduler_pre_submit_census"), Mapping)
        and value.get("scheduler_pre_submit_census_sha256")
        == stable_hash(value["scheduler_pre_submit_census"])
        and value.get("command") == expected_command
        and exact_json_equal(
            value.get("scheduler_environment"),
            _contract_scheduler_environment(contract),
        )
        and exact_json_equal(value.get("transaction_lock"), transaction)
        and exact_json_equal(value.get("report_cancel_lock"), report_cancel)
        and isinstance(value.get("called_at_utc"), str)
        and bool(value["called_at_utc"]),
        "report repair submit-calling evidence differs",
    )
    pre_submit = _validated_scheduler_census(
        value["scheduler_pre_submit_census"], contract
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
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    job_id = value.get("repair_report_job_id")
    evidence = value.get("submission_evidence")
    require(
        set(value) == SUBMITTED_KEYS
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("status") == "held_report_repair_submitted"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and type(value.get("attempt")) is int
        and value.get("attempt") == ATTEMPT
        and value.get("submit_calling_sha256") == calling_sha256
        and isinstance(job_id, str)
        and JOB_ID_RE.fullmatch(job_id) is not None
        and isinstance(evidence, Mapping)
        and evidence.get("mode") in {"direct_sbatch_response", "lost_response_census_adoption"}
        and isinstance(value.get("accepted_at_utc"), str)
        and bool(value["accepted_at_utc"]),
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
        _validated_scheduler_census(evidence["census"], contract)
        rows = _repair_rows(evidence["census"], submission_sha256, contract)
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


def _job_control_argv(job_id: str) -> list[str]:
    return ["/usr/local/bin/scontrol", "show", "job", "-dd", job_id]


def _job_control_projection(
    stdout: bytes,
    *,
    submission_root: Path,
    submission_sha256: str,
    job_id: str,
    source_root: Path,
    scheduler_command: str | None = None,
) -> dict[str, Any]:
    try:
        text = stdout.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RepairError(f"repair job-control output is not UTF-8: {exc}") from exc
    fields: dict[str, str] = {}
    for token in text.split():
        if "=" not in token:
            continue
        key, item = token.split("=", 1)
        require(key not in fields, f"repair job-control field is duplicated: {key}")
        fields[key] = item
    username = pwd.getpwuid(os.getuid()).pw_name
    expected_user_id = f"{username}({os.getuid()})"
    expected_stdout = str(
        submission_root / "logs" / f"report-repair-0002-{job_id}.out"
    )
    require(
        source_root == submission_root / SOURCE_ARCHIVE_NAME,
        "repair held-job source archive path differs",
    )
    expected_command = (
        str(source_root) if scheduler_command is None else scheduler_command
    )
    require(
        isinstance(expected_command, str)
        and (
            expected_command == str(source_root)
            or re.fullmatch(r"/proc/self/fd/[0-9]+", expected_command)
            is not None
        ),
        "repair held-job scheduler command differs",
    )
    expected_workdir = str(submission_root / "source-snapshot" / "repo")
    expected_stdin = "/dev/null"
    array_fields = {key for key in fields if key.startswith("Array")}
    heterogeneous_fields = {key for key in fields if key.startswith("Het")}
    require(
        fields.get("JobId") == job_id
        and fields.get("JobName") == _repair_name(submission_sha256)
        and fields.get("UserId") == expected_user_id
        and fields.get("JobState") == "PENDING"
        and fields.get("Reason") in {"JobHeldUser", "JobHeldAdmin"}
        and fields.get("Requeue") == "0"
        and fields.get("Restarts") == "0"
        and fields.get("BatchFlag") == "1"
        and fields.get("TimeLimit") == "04:00:00"
        and fields.get("Comment") == _repair_comment(submission_sha256)
        and fields.get("Partition") == "cpu"
        and fields.get("Account") == "edgeai_tao-ptm_image-foundation-model-clip"
        and fields.get("QOS") == "normal"
        and fields.get("NumNodes") == "1"
        and fields.get("NumCPUs") == "12"
        and fields.get("NumTasks") == "1"
        and fields.get("CPUs/Task") == "12"
        and fields.get("MinMemoryNode") == "64G"
        and fields.get("Command") == expected_command
        and fields.get("WorkDir") == expected_workdir
        and fields.get("StdOut") == expected_stdout
        and fields.get("StdErr") == expected_stdout
        and fields.get("StdIn") == expected_stdin
        and not array_fields
        and not heterogeneous_fields,
        "repair held-job control-plane identity differs",
    )
    selected = (
        "JobId",
        "JobName",
        "UserId",
        "JobState",
        "Reason",
        "Requeue",
        "Restarts",
        "BatchFlag",
        "TimeLimit",
        "Comment",
        "Partition",
        "Account",
        "QOS",
        "NumNodes",
        "NumCPUs",
        "NumTasks",
        "CPUs/Task",
        "MinMemoryNode",
        "Command",
        "WorkDir",
        "StdOut",
        "StdErr",
        "StdIn",
    )
    return {
        "schema_version": 1,
        "fields": {key: fields[key] for key in selected},
        "no_requeue": True,
        "restart_count": 0,
        "held": True,
        "array_identity_absent": True,
        "heterogeneous_identity_absent": True,
    }


def _job_control_observation(
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    job_id: str,
    source_root: Path,
    runner: Runner,
    locks: _RepairLocks,
    *,
    scheduler_command: str | None = None,
) -> dict[str, Any]:
    environment = _scheduler_environment(
        str(contract["scheduler_control_plane_contract"]["slurm_conf"])
    )
    result, raw = _run(
        runner,
        _job_control_argv(job_id),
        submission_root,
        environment,
        locks,
    )
    require(
        result.returncode == 0 and result.stderr == b"",
        "repair held-job scontrol query failed",
    )
    projection = _job_control_projection(
        result.stdout,
        submission_root=submission_root,
        submission_sha256=submission_sha256,
        job_id=job_id,
        source_root=source_root,
        scheduler_command=scheduler_command,
    )
    return {
        "schema_version": 1,
        "captured_at_utc": _utc_now(),
        "scheduler_control_plane": dict(
            contract["scheduler_control_plane_contract"]
        ),
        "raw": raw,
        "projection": projection,
        "projection_sha256": stable_hash(projection),
    }


def _validated_job_control_observation(
    value: Mapping[str, Any],
    *,
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    job_id: str,
    source_root: Path,
    scheduler_command: str | None = None,
) -> dict[str, Any]:
    require(
        set(value)
        == {
            "schema_version",
            "captured_at_utc",
            "scheduler_control_plane",
            "raw",
            "projection",
            "projection_sha256",
        }
        and value.get("schema_version") == 1
        and isinstance(value.get("captured_at_utc"), str)
        and bool(value["captured_at_utc"])
        and exact_json_equal(
            value.get("scheduler_control_plane"),
            contract.get("scheduler_control_plane_contract"),
        )
        and isinstance(value.get("projection"), Mapping)
        and value.get("projection_sha256") == stable_hash(value["projection"]),
        "repair job-control observation differs",
    )
    result = _validated_command_evidence(
        value["raw"],
        label="repair held-job scontrol",
        expected_argv=_job_control_argv(job_id),
        expected_environment=_scheduler_environment(
            str(contract["scheduler_control_plane_contract"]["slurm_conf"])
        ),
    )
    require(
        result.returncode == 0
        and result.stderr == b""
        and exact_json_equal(
            value["projection"],
            _job_control_projection(
                result.stdout,
                submission_root=submission_root,
                submission_sha256=submission_sha256,
                job_id=job_id,
                source_root=source_root,
                scheduler_command=scheduler_command,
            ),
        ),
        "repair held-job scontrol raw/projection differs",
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
    predecessor_sha256: str,
    calling_sha256: str,
    submitted_sha256: str,
    job_id: str,
    census: Mapping[str, Any],
    job_control: Mapping[str, Any],
    report_installation_method: str,
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
        "predecessor_failure_evidence": (
            "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
        ),
        "predecessor_failure_evidence_sha256": predecessor_sha256,
        "worker_receipt_map": dict(receipt_map),
        "worker_receipt_map_sha256": stable_hash(receipt_map),
        "repair_source_root": str(_repair_source_root(submission_root)),
        "repair_source_commit": source["repair_source_commit"],
        "repair_package_protocol_sha256": source["repair_package_protocol_sha256"],
        "repair_source_files": dict(source["repair_source_files"]),
        "repair_source_files_sha256": source["repair_source_files_sha256"],
        "repair_source_installation_method": source[
            "repair_source_installation_method"
        ],
        "repair_source_archive": source["repair_source_archive"],
        "repair_source_archive_sha256": source[
            "repair_source_archive_sha256"
        ],
        "repair_source_archive_size": source["repair_source_archive_size"],
        "repair_source_archive_format": source["repair_source_archive_format"],
        "report_publication_installation_method": report_installation_method,
        "submit_calling_sha256": calling_sha256,
        "submitted_evidence": "REPORT_REPAIR_0002_SUBMITTED.json",
        "submitted_evidence_sha256": submitted_sha256,
        "scheduler_authority_census": dict(census),
        "scheduler_authority_census_sha256": stable_hash(census),
        "scheduler_job_control_observation": dict(job_control),
        "scheduler_job_control_observation_sha256": stable_hash(job_control),
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
    predecessor_sha256: str,
    calling_sha256: str,
    submitted_sha256: str,
    calling: Mapping[str, Any] | None = None,
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
        and value.get("predecessor_failure_evidence")
        == "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
        and value.get("predecessor_failure_evidence_sha256")
        == predecessor_sha256
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
        and value.get("repair_source_installation_method")
        == source.get("repair_source_installation_method")
        == SOURCE_ARCHIVE_INSTALL_METHOD
        and value.get("repair_source_archive")
        == source.get("repair_source_archive")
        == str(_repair_source_root(submission_root))
        and value.get("repair_source_archive_sha256")
        == source.get("repair_source_archive_sha256")
        and value.get("repair_source_archive_size")
        == source.get("repair_source_archive_size")
        and value.get("repair_source_archive_format")
        == source.get("repair_source_archive_format")
        == SOURCE_ARCHIVE_KIND
        and value.get("report_publication_installation_method")
        == PUBLICATION_ARCHIVE_INSTALL_METHOD
        and value.get("submit_calling_sha256") == calling_sha256
        and value.get("submitted_evidence")
        == "REPORT_REPAIR_0002_SUBMITTED.json"
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
    rows = _repair_rows(census, submission_sha256, contract)
    require(
        len(rows) == 1
        and len(census["settled_rows"]) == 1
        and rows[0]["job_id"] == value["repair_report_job_id"]
        and value["repair_report_job_id"] not in HISTORICAL_JOB_IDS
        and rows[0]["state"] == "PENDING"
        and rows[0]["reason"] in {"JobHeldUser", "JobHeldAdmin"},
        "report repair authorization is not bound to one exact held job",
    )
    job_control = value.get("scheduler_job_control_observation")
    require(
        isinstance(job_control, Mapping)
        and value.get("scheduler_job_control_observation_sha256")
        == stable_hash(job_control),
        "report repair authorization job-control binding differs",
    )
    scheduler_source_argument: str | None = None
    if calling is not None:
        scheduler_source_argument = _scheduler_source_argument_from_calling(
            calling
        )
    validated_job_control = _validated_job_control_observation(
        job_control,
        submission_root=submission_root,
        submission_sha256=submission_sha256,
        contract=contract,
        job_id=value["repair_report_job_id"],
        source_root=_repair_source_root(submission_root),
        scheduler_command=scheduler_source_argument,
    )
    require(
        census["captured_at_utc"]
        <= validated_job_control["captured_at_utc"]
        <= value["authorized_at_utc"],
        "report repair authorization observation order differs",
    )
    return dict(value)


def _release_calling_path(submission_root: Path, index: int) -> Path:
    return _journal_path(
        submission_root,
        f"CALLING_REPORT_REPAIR_0002_RELEASE_{index:04d}.json",
    )


def _release_result_path(submission_root: Path, index: int) -> Path:
    return _journal_path(
        submission_root,
        f"REPORT_REPAIR_0002_RELEASE_RESULT_{index:04d}.json",
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
        and isinstance(value.get("scheduler_pre_release_census"), Mapping)
        and value.get("scheduler_pre_release_census_sha256")
        == stable_hash(value["scheduler_pre_release_census"])
        and isinstance(
            value.get("scheduler_pre_release_job_control_observation"), Mapping
        )
        and value.get("scheduler_pre_release_job_control_observation_sha256")
        == stable_hash(value["scheduler_pre_release_job_control_observation"])
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
    census = _validated_scheduler_census(
        value["scheduler_pre_release_census"], contract
    )
    rows = _repair_rows(census, submission_sha256, contract)
    require(
        len(census["settled_rows"]) == 1
        and len(rows) == 1
        and rows[0]["job_id"] == job_id
        and job_id not in HISTORICAL_JOB_IDS
        and rows[0]["state"] == "PENDING"
        and rows[0]["reason"] in {"JobHeldUser", "JobHeldAdmin"},
        "report repair release-calling census is not one exact held job",
    )
    observation = _validated_job_control_observation(
        value["scheduler_pre_release_job_control_observation"],
        submission_root=submission_root,
        submission_sha256=submission_sha256,
        contract=contract,
        job_id=job_id,
        source_root=_repair_source_root(submission_root),
        scheduler_command=_retained_scheduler_source_argument(
            submission_root, locks
        ),
    )
    require(
        census["captured_at_utc"]
        <= observation["captured_at_utc"]
        <= value["called_at_utc"],
        "report repair release-calling authority order differs",
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
    contract: Mapping[str, Any],
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
        _validated_scheduler_census(evidence["census"], contract)
        rows = _repair_rows(evidence["census"], submission_sha256, contract)
        ambiguous = _release_census_is_ambiguous(
            evidence["census"], submission_sha256, job_id, contract
        )
        if value["mode"] == "lost_response_reconciled_ambiguous_identity":
            require(ambiguous, "report repair reconciled ambiguity differs")
        else:
            require(not ambiguous, "report repair reconciled release is ambiguous")
            released = _job_is_released_or_absent(
                evidence["census"], submission_sha256, job_id, contract
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
    active = _ACTIVE_REPAIR_TRANSITION.get()
    if active is not None:
        require(
            active.submission_root == submission_root,
            "release prefix retained transition differs",
        )
        active.revalidate()
        namespace_names = set(active.phase.durable_names)
    else:
        namespace_names = set(_repair_journal_namespace_names(submission_root))
    calling_pattern = re.compile(
        r"CALLING_REPORT_REPAIR_0002_RELEASE_([0-9]{4})\.json\Z"
    )
    result_pattern = re.compile(
        r"REPORT_REPAIR_0002_RELEASE_RESULT_([0-9]{4})\.json\Z"
    )
    calling_indices: list[int] = []
    result_indices: list[int] = []
    for name in sorted(namespace_names):
        match = calling_pattern.fullmatch(name)
        if match is None:
            continue
        require(match is not None, "report repair release-calling name differs")
        calling_indices.append(int(match.group(1)))
    for name in sorted(namespace_names):
        match = result_pattern.fullmatch(name)
        if match is None:
            continue
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
    while _release_calling_path(submission_root, index).name in namespace_names:
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
        if result_path.name not in namespace_names:
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
            contract=contract,
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
        _release_result_path(submission_root, index).name not in namespace_names,
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
    census: Mapping[str, Any],
    submission_sha256: str,
    job_id: str,
    contract: Mapping[str, Any],
) -> bool:
    rows = _repair_rows(census, submission_sha256, contract)
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
    census: Mapping[str, Any],
    submission_sha256: str,
    job_id: str,
    contract: Mapping[str, Any],
) -> bool:
    rows = _repair_rows(census, submission_sha256, contract)
    settled = census.get("settled_rows")
    require(isinstance(settled, list), "report repair release census rows differ")
    return (
        len(settled) != len(rows)
        or len(rows) > 1
        or (len(rows) == 1 and rows[0]["job_id"] != job_id)
    )


def _active_squeue_worker_liveness(
    census: Mapping[str, Any],
    submission_sha256: str,
    job_id: str,
    contract: Mapping[str, Any],
) -> dict[str, Any] | None:
    rows = _repair_rows(census, submission_sha256, contract)
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
        expected = _active_squeue_worker_liveness(
            census, submission_sha256, job_id, contract
        )
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
            and not _repair_rows(census, submission_sha256, contract)
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
    contract: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    path = _release_denied_path(submission_root)
    if candidate is None:
        if not os.path.lexists(path):
            return None
        value, _digest, info = read_json(
            path, "report repair release-denied terminal"
        )
        identity_valid = (
            stat.S_IMODE(info.st_mode) == 0o444
            and info.st_uid == os.getuid()
            and info.st_nlink == 1
        )
    else:
        value = dict(candidate)
        identity_valid = True
    require(
        identity_valid
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
    _validated_scheduler_census(value["pre_release_census"], contract)
    require(
        _repair_rows(
            value["pre_release_census"], submission_sha256, contract
        )
        == [],
        "report repair release-denied terminal has a live repair identity",
    )
    require(
        not os.path.lexists(_released_path(submission_root)),
        "report repair release-denied terminal has release successor state",
    )
    active = _ACTIVE_REPAIR_TRANSITION.get()
    if active is not None:
        active.revalidate()
        namespace_names = set(active.phase.durable_names)
    else:
        namespace_names = set(_repair_journal_namespace_names(submission_root))
    require(
        not any(
            name.startswith("CALLING_REPORT_REPAIR_0002_RELEASE_")
            or name.startswith("REPORT_REPAIR_0002_RELEASE_RESULT_")
            for name in namespace_names
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
    contract: Mapping[str, Any],
    locks: _RepairLocks,
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
    _seal_transition_json(
        submission_root,
        _release_denied_path(submission_root),
        value,
        locks,
        source_must_be_installed=True,
    )
    validated = _validated_release_denied_terminal(
        submission_root,
        submission_sha256,
        job_id,
        authorization_sha256,
        contract,
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
    *,
    candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    path = _worker_failure_terminal_path(submission_root)
    if candidate is None:
        if not os.path.lexists(path):
            return None
        value, _digest, info = read_json(
            path, "report repair worker-failure terminal"
        )
        identity_valid = (
            stat.S_IMODE(info.st_mode) == 0o444
            and info.st_uid == os.getuid()
            and info.st_nlink == 1
        )
    else:
        value = dict(candidate)
        identity_valid = True
    require(
        identity_valid
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
    _validated_scheduler_census(value["post_release_census"], contract)
    require(
        not _repair_rows(
            value["post_release_census"], submission_sha256, contract
        ),
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
            == "REPORT_REPAIR_0002_RELEASED.json"
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
        not os.path.lexists(submission_root / "report")
        and not _publication_archive_names(submission_root),
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
        released_evidence = "REPORT_REPAIR_0002_RELEASED.json"
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
    _seal_transition_json(
        submission_root,
        _worker_failure_terminal_path(submission_root),
        value,
        locks,
        source_must_be_installed=True,
    )
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
    locks: _RepairLocks,
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
    return value, _seal_transition_json(
        submission_root,
        _released_path(submission_root),
        value,
        locks,
        source_must_be_installed=True,
    )


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
        and not _repair_rows(census, submission_sha256, contract),
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
        and not _repair_rows(census, submission_sha256, contract),
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


def _staged_cleanup_target(name: str | None) -> bool:
    if name is None:
        return False
    return any(
        pattern.fullmatch(name) is not None
        for pattern in (
            re.compile(r"REPORT_REPAIR_0002_CANCEL_AUTHORIZED_[0-9]{4}\.json"),
            re.compile(
                r"CALLING_REPORT_REPAIR_0002_SCANCEL_[0-9]{4}_[0-9]{4}\.json"
            ),
            re.compile(
                r"REPORT_REPAIR_0002_SCANCEL_RESULT_[0-9]{4}_[0-9]{4}\.json"
            ),
            re.compile(r"REPORT_REPAIR_0002_CANCEL_TERMINAL_[0-9]{4}\.json"),
        )
    )


def _legacy_resolve_staged_cleanup_before_positive_transition(
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    phase: _RepairPhaseSnapshot,
    runner: Runner,
    locks: _RepairLocks,
    *,
    reason: str,
    sleep: Callable[[float], None],
) -> dict[str, Any] | None:
    """Give any valid virtual cleanup edge total precedence over success."""

    if not _staged_cleanup_target(phase.staged_target):
        return None
    census = _scheduler_census(
        submission_root, contract, runner, locks, sleep=sleep
    )
    assert phase.staged_target is not None
    staged_authority = re.fullmatch(
        r"REPORT_REPAIR_0002_CANCEL_AUTHORIZED_[0-9]{4}\.json",
        phase.staged_target,
    )
    if staged_authority is not None and phase.staged_mode == 0o600:
        _discard_phase_next_partial_stage(submission_root, phase)
        rebound = _classify_repair_phase(
            submission_root, source_must_be_installed=True
        )
        require(
            rebound.staged_target is None,
            "discarded cleanup authorization stage was replaced",
        )
        return {
            "schema_version": 1,
            "status": "report_repair_cleanup_partial_authority_discarded",
            "campaign_id": CAMPAIGN_ID,
            "submission_sha256": submission_sha256,
            "attempt": ATTEMPT,
            "staged_cleanup_authority": phase.staged_target,
            "pre_discard_census": census,
            "pre_discard_census_sha256": stable_hash(census),
            "publication_allowed": False,
            "scheduler_calls": 3,
        }
    return _cleanup_repair_rows(
        submission_root,
        submission_sha256,
        contract,
        census,
        reason,
        runner,
        locks,
        sleep=sleep,
    )


def _resolve_staged_cleanup_before_positive_transition(
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    phase: _RepairPhaseSnapshot,
    runner: Runner,
    locks: _RepairLocks,
    *,
    reason: str,
    sleep: Callable[[float], None],
) -> dict[str, Any] | None:
    """Attempt 2 has no recoverable local staging state."""

    del submission_root, submission_sha256, contract, runner, locks, reason, sleep
    require(
        phase.staged_target is None
        and phase.staged_mode is None
        and phase.staged_linked is False
        and phase.staged_identity is None,
        "attempt2 local staging is permanent fail-stop evidence",
    )
    return None


def _release_authorized_job_bound(
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    authorization: Mapping[str, Any],
    authorization_sha256: str,
    runner: Runner,
    locks: _RepairLocks,
    transition: _RepairTransitionBinding,
    *,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    job_id = str(authorization["repair_report_job_id"])

    def rebind_authorization(label: str) -> None:
        _revalidated_sealed_json(
            _authorization_path(submission_root),
            authorization,
            authorization_sha256,
            label,
        )

    rebind_authorization("report repair authorization at release entry")
    phase = _classify_repair_phase(
        submission_root, source_must_be_installed=True
    )
    staged_cleanup = _resolve_staged_cleanup_before_positive_transition(
        submission_root,
        submission_sha256,
        contract,
        phase,
        runner,
        locks,
        reason="staged_cleanup_prefix_before_release",
        sleep=sleep,
    )
    if staged_cleanup is not None:
        return staged_cleanup
    staged_target = phase.staged_target
    recovered_released: dict[str, Any] | None = None
    if staged_target is not None:
        calling_match = re.fullmatch(
            r"CALLING_REPORT_REPAIR_0002_RELEASE_([0-9]{4})\.json",
            staged_target,
        )
        result_match = re.fullmatch(
            r"REPORT_REPAIR_0002_RELEASE_RESULT_([0-9]{4})\.json",
            staged_target,
        )
        if calling_match is not None or result_match is not None:
            if phase.staged_mode == 0o600:
                _discard_phase_next_partial_stage(submission_root, phase)
            else:
                staged_value = _phase_staged_value(submission_root, phase)
                assert staged_value is not None
                if calling_match is not None:
                    index = int(calling_match.group(1))
                    _validate_release_calling(
                        staged_value,
                        submission_root=submission_root,
                        submission_sha256=submission_sha256,
                        index=index,
                        job_id=job_id,
                        authorization_sha256=authorization_sha256,
                        contract=contract,
                        locks=locks,
                    )
                else:
                    assert result_match is not None
                    index = int(result_match.group(1))
                    calling_value, calling_sha, calling_info = read_json(
                        _release_calling_path(submission_root, index),
                        f"report repair staged result calling {index}",
                    )
                    require(
                        stat.S_IMODE(calling_info.st_mode) == 0o444
                        and calling_info.st_uid == os.getuid()
                        and calling_info.st_nlink == 1,
                        "report repair staged result calling identity differs",
                    )
                    _validate_release_calling(
                        calling_value,
                        submission_root=submission_root,
                        submission_sha256=submission_sha256,
                        index=index,
                        job_id=job_id,
                        authorization_sha256=authorization_sha256,
                        contract=contract,
                        locks=locks,
                    )
                    _validate_release_result(
                        staged_value,
                        submission_sha256=submission_sha256,
                        index=index,
                        job_id=job_id,
                        authorization_sha256=authorization_sha256,
                        calling_sha256=calling_sha,
                        scheduler_environment=calling_value["scheduler_environment"],
                        contract=contract,
                    )
                _seal_transition_json(
                    submission_root,
                    _journal_path(submission_root, staged_target),
                    staged_value,
                    locks,
                    source_must_be_installed=True,
                )
        elif staged_target in {
            _release_denied_path(submission_root).name,
            _worker_failure_terminal_path(submission_root).name,
            _released_path(submission_root).name,
        }:
            if phase.staged_mode == 0o600:
                _discard_phase_next_partial_stage(submission_root, phase)
            else:
                staged_value = _phase_staged_value(submission_root, phase)
                assert staged_value is not None
                if staged_target == _release_denied_path(submission_root).name:
                    validated_stage = _validated_release_denied_terminal(
                        submission_root,
                        submission_sha256,
                        job_id,
                        authorization_sha256,
                        contract,
                        candidate=staged_value,
                    )
                elif staged_target == _worker_failure_terminal_path(
                    submission_root
                ).name:
                    validated_stage = _validated_worker_failure_terminal(
                        submission_root,
                        submission_sha256,
                        job_id,
                        authorization_sha256,
                        contract,
                        locks,
                        candidate=staged_value,
                    )
                else:
                    validated_stage = _validate_existing_release(
                        submission_root,
                        submission_sha256,
                        authorization_sha256,
                        job_id,
                        contract,
                        locks,
                        candidate=staged_value,
                    )
                require(
                    validated_stage is not None,
                    "staged report repair terminal/release differs",
                )
                _seal_transition_json(
                    submission_root,
                    _journal_path(submission_root, staged_target),
                    staged_value,
                    locks,
                    source_must_be_installed=True,
                )
                if staged_target == _released_path(submission_root).name:
                    recovered_released = validated_stage
    if recovered_released is not None:
        rebound_released = _validate_existing_release(
            submission_root,
            submission_sha256,
            authorization_sha256,
            job_id,
            contract,
            locks,
        )
        require(
            rebound_released is not None
            and exact_json_equal(rebound_released, recovered_released),
            "recovered report repair release differs",
        )
        return rebound_released
    _require_report_install_probe_namespace(
        submission_root, allow_exact_crash_residue=False
    )
    cleanup_phase = _classify_repair_phase(
        submission_root, source_must_be_installed=True
    )
    staged_cleanup = _resolve_staged_cleanup_before_positive_transition(
        submission_root,
        submission_sha256,
        contract,
        cleanup_phase,
        runner,
        locks,
        reason="staged_cleanup_prefix_before_release",
        sleep=sleep,
    )
    if staged_cleanup is not None:
        return staged_cleanup
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
        if _repair_rows(delayed_census, submission_sha256, contract):
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
            cleanup_census, submission_sha256, contract
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
        submission_root,
        submission_sha256,
        job_id,
        authorization_sha256,
        contract,
    )
    if denied is not None:
        delayed_census = _scheduler_census(
            submission_root, contract, runner, locks, sleep=sleep
        )
        if _repair_rows(delayed_census, submission_sha256, contract):
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
        if _repair_rows(cleanup_census, submission_sha256, contract):
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
        effect_rows = _repair_rows(effect_recheck, submission_sha256, contract)
        effect_ambiguous = _release_census_is_ambiguous(
            effect_recheck, submission_sha256, job_id, contract
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
                effect_recheck, submission_sha256, job_id, contract
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
            locks,
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
        observed_rows = _repair_rows(census, submission_sha256, contract)
        ambiguous = _release_census_is_ambiguous(
            census, submission_sha256, job_id, contract
        )
        released = (
            False
            if ambiguous
            else _job_is_released_or_absent(
                census, submission_sha256, job_id, contract
            )
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
        _classify_repair_phase(
            submission_root, source_must_be_installed=True
        )
        result_sha = _seal_transition_json(
            submission_root,
            _release_result_path(submission_root, unpaired["index"]),
            result,
            locks,
            source_must_be_installed=True,
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
                    census, submission_sha256, job_id, contract
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
                locks,
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
    pre_release_rows = _repair_rows(
        pre_release_census, submission_sha256, contract
    )
    if records:
        completed_attempt_ambiguous = _release_census_is_ambiguous(
            pre_release_census, submission_sha256, job_id, contract
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
            pre_release_census, submission_sha256, job_id, contract
        ):
            if pre_release_rows:
                worker_liveness = _active_squeue_worker_liveness(
                    pre_release_census, submission_sha256, job_id, contract
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
                locks,
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
            contract,
            locks,
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
    pre_release_job_control = _job_control_observation(
        submission_root,
        submission_sha256,
        contract,
        job_id,
        _repair_source_root(submission_root),
        runner,
        locks,
        scheduler_command=_retained_scheduler_source_argument(
            submission_root, locks
        ),
    )
    _validated_job_control_observation(
        pre_release_job_control,
        submission_root=submission_root,
        submission_sha256=submission_sha256,
        contract=contract,
        job_id=job_id,
        source_root=_repair_source_root(submission_root),
        scheduler_command=_retained_scheduler_source_argument(
            submission_root, locks
        ),
    )
    rebind_authorization("report repair authorization before release calling")
    _classify_repair_phase(
        submission_root, source_must_be_installed=True
    )
    _require_report_install_probe_namespace(
        submission_root, allow_exact_crash_residue=False
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
        "scheduler_pre_release_census": dict(pre_release_census),
        "scheduler_pre_release_census_sha256": stable_hash(
            pre_release_census
        ),
        "scheduler_pre_release_job_control_observation": dict(
            pre_release_job_control
        ),
        "scheduler_pre_release_job_control_observation_sha256": stable_hash(
            pre_release_job_control
        ),
        "command": command,
        "scheduler_environment": environment,
        "transaction_lock": transaction,
        "report_cancel_lock": report_cancel,
        "called_at_utc": _utc_now(),
    }
    transition.revalidate()
    try:
        calling_sha = transition.seal_successor(
            _release_calling_path(submission_root, next_index), calling
        )
        rebound_calling = _rebind_scheduler_calling_before_mutation(
            submission_root,
            source=_source_authority_from_bound_value(authorization),
            calling_path=_release_calling_path(submission_root, next_index),
            calling=calling,
            calling_sha256=calling_sha,
            locks=locks,
            label=f"report repair release calling {next_index}",
        )
        _validate_release_calling(
            rebound_calling,
            submission_root=submission_root,
            submission_sha256=submission_sha256,
            index=next_index,
            job_id=job_id,
            authorization_sha256=authorization_sha256,
            contract=contract,
            locks=locks,
        )
        rebind_authorization("report repair authorization before scontrol")
        transition.revalidate()
        _rebind_scheduler_calling_before_mutation(
            submission_root,
            source=_source_authority_from_bound_value(authorization),
            calling_path=_release_calling_path(submission_root, next_index),
            calling=calling,
            calling_sha256=calling_sha,
            locks=locks,
            label=f"report repair release calling {next_index}",
        )
        transition.revalidate()
        result, evidence = _run(
            runner, command, submission_root, environment, locks
        )
        transition.revalidate()
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
        transition.revalidate()
        result_sha = transition.seal_successor(
            _release_result_path(submission_root, next_index), result_value
        )
    finally:
        transition.revalidate()
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
    direct_rows = _repair_rows(census, submission_sha256, contract)
    direct_ambiguous = _release_census_is_ambiguous(
        census, submission_sha256, job_id, contract
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
        _job_is_released_or_absent(
            census, submission_sha256, job_id, contract
        ),
        "report repair job remains held after successful release",
    )
    if direct_rows:
        worker_liveness = _active_squeue_worker_liveness(
            census, submission_sha256, job_id, contract
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
        locks,
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
    with _retained_transition_scope(
        submission_root, locks, source_must_be_installed=True
    ) as transition:
        return _release_authorized_job_bound(
            submission_root,
            submission_sha256,
            contract,
            authorization,
            authorization_sha256,
            runner,
            locks,
            transition,
            sleep=sleep,
        )


def _cancel_generation_count(submission_root: Path) -> int:
    active = _ACTIVE_REPAIR_TRANSITION.get()
    if active is not None:
        require(
            active.submission_root == submission_root,
            "cancel generation retained transition differs",
        )
        active.revalidate()
        namespace_names = set(active.phase.durable_names)
    else:
        namespace_names = set(_repair_journal_namespace_names(submission_root))
    existing: list[int] = []
    pattern = re.compile(r"REPORT_REPAIR_0002_CANCEL_AUTHORIZED_([0-9]{4})\.json\Z")
    for name in sorted(namespace_names):
        match = pattern.fullmatch(name)
        if match is None:
            continue
        require(match is not None, "report repair cancel-authorization name differs")
        existing.append(int(match.group(1)))
    require(
        sorted(existing) == list(range(len(existing))),
        "report repair cancel generations are not contiguous",
    )
    count = len(existing)
    dependent_patterns = (
        (
            "CALLING_REPORT_REPAIR_0002_SCANCEL_*.json",
            re.compile(
                r"CALLING_REPORT_REPAIR_0002_SCANCEL_([0-9]{4})_([0-9]{4})\.json\Z"
            ),
        ),
        (
            "REPORT_REPAIR_0002_SCANCEL_RESULT_*.json",
            re.compile(
                r"REPORT_REPAIR_0002_SCANCEL_RESULT_([0-9]{4})_([0-9]{4})\.json\Z"
            ),
        ),
        (
            "REPORT_REPAIR_0002_CANCEL_TERMINAL_*.json",
            re.compile(r"REPORT_REPAIR_0002_CANCEL_TERMINAL_([0-9]{4})\.json\Z"),
        ),
    )
    for _glob_pattern, dependent_pattern in dependent_patterns:
        for name in sorted(namespace_names):
            match = dependent_pattern.fullmatch(name)
            if match is None:
                continue
            require(
                match is not None and int(match.group(1)) < count,
                "report repair cleanup evidence exists without authorization",
            )
    return count


def _cancel_authority_path(submission_root: Path, generation: int) -> Path:
    return _journal_path(
        submission_root,
        f"REPORT_REPAIR_0002_CANCEL_AUTHORIZED_{generation:04d}.json",
    )


def _cancel_calling_path(
    submission_root: Path, generation: int, cancel_attempt: int
) -> Path:
    return _journal_path(
        submission_root,
        f"CALLING_REPORT_REPAIR_0002_SCANCEL_{generation:04d}_{cancel_attempt:04d}.json",
    )


def _cancel_result_path(
    submission_root: Path, generation: int, cancel_attempt: int
) -> Path:
    return _journal_path(
        submission_root,
        f"REPORT_REPAIR_0002_SCANCEL_RESULT_{generation:04d}_{cancel_attempt:04d}.json",
    )


def _cancel_terminal_path(submission_root: Path, generation: int) -> Path:
    return _journal_path(
        submission_root,
        f"REPORT_REPAIR_0002_CANCEL_TERMINAL_{generation:04d}.json",
    )


def _validated_cancel_authority(
    value: Mapping[str, Any],
    *,
    submission_sha256: str,
    generation: int,
    contract: Mapping[str, Any],
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
        and isinstance(value.get("authorized_at_utc"), str)
        and bool(value["authorized_at_utc"]),
        "report repair cleanup authorization differs",
    )
    validated_census = _validated_scheduler_census(
        value["pre_cancel_census"], contract
    )
    require(
        validated_census["captured_at_utc"] <= value["authorized_at_utc"],
        "report repair cleanup authorization observation order differs",
    )
    bound_ids = sorted(
        {
            row["job_id"]
            for row in _repair_rows(
                value["pre_cancel_census"], submission_sha256, contract
            )
        },
        key=int,
    )
    require(bound_ids == value["job_ids"], "report repair cleanup authority targets differ")
    return dict(value)


def _validated_cancel_calling_value(
    value: Mapping[str, Any],
    *,
    submission_sha256: str,
    generation: int,
    index: int,
    authority_sha256: str,
    job_ids: Sequence[str],
    contract: Mapping[str, Any],
    locks: _RepairLocks,
) -> dict[str, Any]:
    transaction, report_cancel = locks.bindings()
    require(
        set(value) == CANCEL_CALLING_KEYS
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("status") == "calling_report_repair_cleanup"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and type(value.get("attempt")) is int
        and value.get("attempt") == ATTEMPT
        and type(value.get("cancel_generation")) is int
        and value.get("cancel_generation") == generation
        and type(value.get("cancel_attempt")) is int
        and value.get("cancel_attempt") == index
        and value.get("authorization_sha256") == authority_sha256
        and value.get("job_ids") == list(job_ids)
        and value.get("command") == ["/usr/local/bin/scancel", *job_ids]
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
        "report repair cleanup calling differs",
    )
    return dict(value)


def _validated_cancel_result_value(
    value: Mapping[str, Any],
    *,
    submission_sha256: str,
    generation: int,
    index: int,
    authority_sha256: str,
    calling_sha256: str,
    calling: Mapping[str, Any],
    job_ids: Sequence[str],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    require(
        set(value) == CANCEL_RESULT_KEYS
        and type(value.get("schema_version")) is int
        and value.get("schema_version") == 1
        and value.get("status") == "report_repair_cleanup_attempt_observed"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and type(value.get("attempt")) is int
        and value.get("attempt") == ATTEMPT
        and type(value.get("cancel_generation")) is int
        and value.get("cancel_generation") == generation
        and type(value.get("cancel_attempt")) is int
        and value.get("cancel_attempt") == index
        and value.get("authorization_sha256") == authority_sha256
        and value.get("calling_sha256") == calling_sha256
        and value.get("job_ids") == list(job_ids)
        and value.get("mode")
        in {
            "direct_scancel_response",
            "lost_response_reconciled_still_live",
            "lost_response_reconciled_cancel_effect",
        }
        and isinstance(value.get("scheduler_evidence"), Mapping)
        and isinstance(value.get("observed_at_utc"), str)
        and bool(value["observed_at_utc"]),
        "report repair cleanup result differs",
    )
    evidence = value["scheduler_evidence"]
    if value["mode"] == "direct_scancel_response":
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
        _validated_scheduler_census(evidence["census"], contract)
        live_ids = {
            row["job_id"]
            for row in _repair_rows(
                evidence["census"], submission_sha256, contract
            )
        }
        require(
            (value["mode"] == "lost_response_reconciled_still_live")
            == bool(set(job_ids) & live_ids),
            "report repair reconciled cleanup effect differs",
        )
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
    active = _ACTIVE_REPAIR_TRANSITION.get()
    if active is not None:
        require(
            active.submission_root == submission_root,
            "cleanup prefix retained transition differs",
        )
        active.revalidate()
        namespace_names = set(active.phase.durable_names)
    else:
        namespace_names = set(_repair_journal_namespace_names(submission_root))
    records: list[dict[str, Any]] = []
    index = 0
    unpaired: dict[str, Any] | None = None
    while (
        _cancel_calling_path(submission_root, generation, index).name
        in namespace_names
    ):
        calling, calling_sha, calling_info = read_json(
            _cancel_calling_path(submission_root, generation, index),
            f"report repair cleanup calling {generation}/{index}",
        )
        require(
            stat.S_IMODE(calling_info.st_mode) == 0o444
            and calling_info.st_uid == os.getuid()
            and calling_info.st_nlink == 1,
            "report repair cleanup calling identity differs",
        )
        _validated_cancel_calling_value(
            calling,
            submission_sha256=submission_sha256,
            generation=generation,
            index=index,
            authority_sha256=authority_sha256,
            job_ids=job_ids,
            contract=contract,
            locks=locks,
        )
        result_path = _cancel_result_path(submission_root, generation, index)
        if result_path.name not in namespace_names:
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
            and result_info.st_nlink == 1,
            "report repair cleanup result identity differs",
        )
        _validated_cancel_result_value(
            result,
            submission_sha256=submission_sha256,
            generation=generation,
            index=index,
            authority_sha256=authority_sha256,
            calling_sha256=calling_sha,
            calling=calling,
            job_ids=job_ids,
            contract=contract,
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
        _cancel_result_path(submission_root, generation, index).name
        not in namespace_names,
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
    remaining = _repair_rows(post, submission_sha256, contract)
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
    _seal_transition_json(
        submission_root,
        _cancel_terminal_path(submission_root, generation),
        terminal,
        locks,
        source_must_be_installed=True,
    )
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
    terminal, _terminal_sha, terminal_info = read_json(
        _cancel_terminal_path(submission_root, generation),
        f"report repair cleanup terminal {generation}",
    )
    require(
        stat.S_IMODE(terminal_info.st_mode) == 0o444
        and terminal_info.st_uid == os.getuid()
        and terminal_info.st_nlink == 1,
        "report repair cleanup terminal identity differs",
    )
    return _validated_cleanup_terminal_value(
        terminal,
        submission_root=submission_root,
        submission_sha256=submission_sha256,
        generation=generation,
        contract=contract,
        locks=locks,
    )


def _validated_cleanup_terminal_value(
    terminal: Mapping[str, Any],
    *,
    submission_root: Path,
    submission_sha256: str,
    generation: int,
    contract: Mapping[str, Any],
    locks: _RepairLocks,
) -> dict[str, Any]:
    """Validate a durable or phase-next cleanup terminal against its prefix."""

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
        contract=contract,
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
    require(
        set(terminal) == CANCEL_TERMINAL_KEYS
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
    _validated_scheduler_census(terminal["post_cancel_census"], contract)
    remaining = sorted(
        {
            row["job_id"]
            for row in _repair_rows(
                terminal["post_cancel_census"], submission_sha256, contract
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

    _validated_scheduler_census(census, contract)
    phase = _classify_repair_phase(
        submission_root, source_must_be_installed=True
    )
    staged_match = (
        re.fullmatch(
            r"REPORT_REPAIR_0002_CANCEL_AUTHORIZED_([0-9]{4})\.json",
            phase.staged_target,
        )
        if phase.staged_target is not None
        else None
    )
    if staged_match is not None:
        generation = int(staged_match.group(1))
        if phase.staged_mode == 0o600:
            _discard_phase_next_partial_stage(submission_root, phase)
        else:
            staged_authority = _phase_staged_value(submission_root, phase)
            assert staged_authority is not None
            _validated_cancel_authority(
                staged_authority,
                submission_sha256=submission_sha256,
                generation=generation,
                contract=contract,
                locks=locks,
            )
            _seal_transition_json(
                submission_root,
                _cancel_authority_path(submission_root, generation),
                staged_authority,
                locks,
                source_must_be_installed=True,
            )
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
                contract=contract,
                locks=locks,
            ),
            authority_sha,
        )

    job_ids = sorted(
        {
            row["job_id"]
            for row in _repair_rows(census, submission_sha256, contract)
        },
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
    authority_sha = _seal_transition_json(
        submission_root,
        _cancel_authority_path(submission_root, generation),
        authority,
        locks,
        source_must_be_installed=True,
    )
    return generation, authority, authority_sha


def _reconcile_cleanup_stage(
    submission_root: Path,
    submission_sha256: str,
    generation: int,
    authority_sha256: str,
    job_ids: Sequence[str],
    contract: Mapping[str, Any],
    locks: _RepairLocks,
) -> dict[str, Any] | None:
    """Promote only an exact phase-next cleanup calling/result stage."""

    phase = _classify_repair_phase(
        submission_root, source_must_be_installed=True
    )
    target = phase.staged_target
    if target is None:
        return None
    calling_match = re.fullmatch(
        r"CALLING_REPORT_REPAIR_0002_SCANCEL_([0-9]{4})_([0-9]{4})\.json",
        target,
    )
    result_match = re.fullmatch(
        r"REPORT_REPAIR_0002_SCANCEL_RESULT_([0-9]{4})_([0-9]{4})\.json",
        target,
    )
    terminal_match = re.fullmatch(
        r"REPORT_REPAIR_0002_CANCEL_TERMINAL_([0-9]{4})\.json", target
    )
    if terminal_match is not None:
        require(
            int(terminal_match.group(1)) == generation,
            "report repair staged cleanup terminal generation differs",
        )
        if phase.staged_mode == 0o600:
            _discard_phase_next_partial_stage(submission_root, phase)
            return None
        staged_terminal = _phase_staged_value(submission_root, phase)
        assert staged_terminal is not None
        validated_terminal = _validated_cleanup_terminal_value(
            staged_terminal,
            submission_root=submission_root,
            submission_sha256=submission_sha256,
            generation=generation,
            contract=contract,
            locks=locks,
        )
        _seal_transition_json(
            submission_root,
            _cancel_terminal_path(submission_root, generation),
            staged_terminal,
            locks,
            source_must_be_installed=True,
        )
        return validated_terminal
    if calling_match is None and result_match is None:
        return None
    match = calling_match if calling_match is not None else result_match
    assert match is not None
    staged_generation, index = (int(item) for item in match.groups())
    require(
        staged_generation == generation,
        "report repair staged cleanup generation differs",
    )
    if phase.staged_mode == 0o600:
        _discard_phase_next_partial_stage(submission_root, phase)
        return None
    staged = _phase_staged_value(submission_root, phase)
    assert staged is not None
    if calling_match is not None:
        _validated_cancel_calling_value(
            staged,
            submission_sha256=submission_sha256,
            generation=generation,
            index=index,
            authority_sha256=authority_sha256,
            job_ids=job_ids,
            contract=contract,
            locks=locks,
        )
    else:
        calling, calling_sha, calling_info = read_json(
            _cancel_calling_path(submission_root, generation, index),
            f"staged cleanup result calling {generation}/{index}",
        )
        require(
            stat.S_IMODE(calling_info.st_mode) == 0o444
            and calling_info.st_uid == os.getuid()
            and calling_info.st_nlink == 1,
            "staged cleanup result calling identity differs",
        )
        _validated_cancel_calling_value(
            calling,
            submission_sha256=submission_sha256,
            generation=generation,
            index=index,
            authority_sha256=authority_sha256,
            job_ids=job_ids,
            contract=contract,
            locks=locks,
        )
        _validated_cancel_result_value(
            staged,
            submission_sha256=submission_sha256,
            generation=generation,
            index=index,
            authority_sha256=authority_sha256,
            calling_sha256=calling_sha,
            calling=calling,
            job_ids=job_ids,
            contract=contract,
        )
    _seal_transition_json(
        submission_root,
        _journal_path(submission_root, target),
        staged,
        locks,
        source_must_be_installed=True,
    )
    return None


def _cleanup_repair_rows_bound(
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    census: Mapping[str, Any],
    reason: str,
    runner: Runner,
    locks: _RepairLocks,
    transition: _RepairTransitionBinding,
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

    def rebind_cleanup_authority(label: str) -> None:
        _revalidated_sealed_json(
            _cancel_authority_path(submission_root, generation),
            authority,
            authority_sha,
            label,
        )

    rebind_cleanup_authority("report repair cleanup authorization at entry")

    job_ids = list(authority["job_ids"])
    staged_terminal = _reconcile_cleanup_stage(
        submission_root,
        submission_sha256,
        generation,
        authority_sha,
        job_ids,
        contract,
        locks,
    )
    if staged_terminal is not None:
        return staged_terminal
    records, next_attempt, unpaired = _cancel_attempt_prefix(
        submission_root,
        submission_sha256,
        generation,
        authority_sha,
        job_ids,
        contract,
        locks,
    )
    current_rows = _repair_rows(census, submission_sha256, contract)
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
        _classify_repair_phase(
            submission_root, source_must_be_installed=True
        )
        result_sha = _seal_transition_json(
            submission_root,
            _cancel_result_path(
                submission_root, generation, unpaired["cancel_attempt"]
            ),
            result_value,
            locks,
            source_must_be_installed=True,
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
        _classify_repair_phase(
            submission_root, source_must_be_installed=True
        )
        transition.revalidate()
        try:
            calling_sha = transition.seal_successor(
                _cancel_calling_path(
                    submission_root, generation, next_attempt
                ),
                calling,
            )
            rebound_calling = _rebind_scheduler_calling_before_mutation(
                submission_root,
                source=_load_sealed_repair_source(
                    _repair_source_root(submission_root)
                ),
                calling_path=_cancel_calling_path(
                    submission_root, generation, next_attempt
                ),
                calling=calling,
                calling_sha256=calling_sha,
                locks=locks,
                label=(
                    "report repair cleanup calling "
                    f"{generation}/{next_attempt}"
                ),
            )
            _validated_cancel_calling_value(
                rebound_calling,
                submission_sha256=submission_sha256,
                generation=generation,
                index=next_attempt,
                authority_sha256=authority_sha,
                job_ids=job_ids,
                contract=contract,
                locks=locks,
            )
            rebind_cleanup_authority(
                "report repair cleanup authorization before scancel"
            )
            transition.revalidate()
            _rebind_scheduler_calling_before_mutation(
                submission_root,
                source=_load_sealed_repair_source(
                    _repair_source_root(submission_root)
                ),
                calling_path=_cancel_calling_path(
                    submission_root, generation, next_attempt
                ),
                calling=calling,
                calling_sha256=calling_sha,
                locks=locks,
                label=(
                    "report repair cleanup calling "
                    f"{generation}/{next_attempt}"
                ),
            )
            transition.revalidate()
            result, scheduler_evidence = _run(
                runner, command, submission_root, environment, locks
            )
            transition.revalidate()
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
            transition.revalidate()
            result_sha = transition.seal_successor(
                _cancel_result_path(submission_root, generation, next_attempt),
                result_value,
            )
        finally:
            transition.revalidate()
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
    inherited = _ACTIVE_REPAIR_TRANSITION.get()
    with _retained_transition_scope(
        submission_root, locks, source_must_be_installed=True
    ) as transition:
        if inherited is None:
            # A census supplied by an outer recovery branch cannot authorize a
            # cleanup mutation after a fresh pathname capture.  Re-observe it
            # under this retained boundary instead of blessing drift.
            census = _scheduler_census(
                submission_root, contract, runner, locks, sleep=sleep
            )
            transition.revalidate()
        return _cleanup_repair_rows_bound(
            submission_root,
            submission_sha256,
            contract,
            census,
            reason,
            runner,
            locks,
            transition,
            sleep=sleep,
        )


def _validate_existing_release(
    submission_root: Path,
    submission_sha256: str,
    authorization_sha256: str,
    job_id: str,
    contract: Mapping[str, Any],
    locks: _RepairLocks,
    *,
    candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    path = _released_path(submission_root)
    if candidate is None and not os.path.lexists(path):
        return None
    require(
        not os.path.lexists(_release_denied_path(submission_root)),
        "report repair released and release-denied states conflict",
    )
    if candidate is None:
        value, digest, info = read_json(path, "report repair released result")
        identity_valid = (
            stat.S_IMODE(info.st_mode) == 0o444
            and info.st_uid == os.getuid()
            and info.st_nlink == 1
        )
    else:
        value = dict(candidate)
        digest = _pretty_json_sha(value)
        identity_valid = True
    require(
        identity_valid
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
    _validated_scheduler_census(value["post_release_census"], contract)
    require(
        _job_is_released_or_absent(
            value["post_release_census"], submission_sha256, job_id, contract
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


def _reconcile_released_worker_bound(
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
    _revalidated_sealed_json(
        _released_path(submission_root),
        released,
        _pretty_json_sha(released),
        "report repair released evidence at reconciliation entry",
    )
    job_id = str(released["repair_report_job_id"])
    phase = _classify_repair_phase(
        submission_root, source_must_be_installed=True
    )
    staged_cleanup = _resolve_staged_cleanup_before_positive_transition(
        submission_root,
        submission_sha256,
        contract,
        phase,
        runner,
        locks,
        reason="staged_cleanup_prefix_after_release",
        sleep=sleep,
    )
    if staged_cleanup is not None:
        return staged_cleanup
    if phase.staged_target == _worker_failure_terminal_path(
        submission_root
    ).name:
        if phase.staged_mode == 0o600:
            _discard_phase_next_partial_stage(submission_root, phase)
        else:
            staged_terminal = _phase_staged_value(submission_root, phase)
            assert staged_terminal is not None
            validated_staged_terminal = _validated_worker_failure_terminal(
                submission_root,
                submission_sha256,
                job_id,
                authorization_sha256,
                contract,
                locks,
                candidate=staged_terminal,
            )
            require(
                validated_staged_terminal is not None,
                "staged released-worker terminal differs",
            )
            _seal_transition_json(
                submission_root,
                _worker_failure_terminal_path(submission_root),
                staged_terminal,
                locks,
                source_must_be_installed=True,
            )
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
    rows = _repair_rows(census, submission_sha256, contract)
    cleanup_phase = _classify_repair_phase(
        submission_root, source_must_be_installed=True
    )
    staged_cleanup = _resolve_staged_cleanup_before_positive_transition(
        submission_root,
        submission_sha256,
        contract,
        cleanup_phase,
        runner,
        locks,
        reason="staged_cleanup_prefix_after_release",
        sleep=sleep,
    )
    if staged_cleanup is not None:
        return staged_cleanup
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
        census, submission_sha256, job_id, contract
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
        _active_squeue_worker_liveness(
            census, submission_sha256, job_id, contract
        )
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
    with _retained_transition_scope(
        submission_root, locks, source_must_be_installed=True
    ):
        return _reconcile_released_worker_bound(
            submission_root,
            submission_sha256,
            contract,
            authorization_sha256,
            released,
            runner,
            locks,
            sleep=sleep,
        )


def _expected_publication_authority(
    authorization: Mapping[str, Any],
    authorization_sha256: str,
    release_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": "authorized_terminal_report_repair",
        "attempt": ATTEMPT,
        "authorization": "REPORT_REPAIR_0002_AUTHORIZED.json",
        "authorization_sha256": authorization_sha256,
        "release": "REPORT_REPAIR_0002_RELEASED.json",
        "release_sha256": release_sha256,
        "original_report_job_id": authorization.get("original_report_job_id"),
        "repair_report_job_id": authorization.get("repair_report_job_id"),
        "original_failure_evidence": authorization.get(
            "original_failure_evidence"
        ),
        "original_failure_evidence_sha256": authorization.get(
            "original_failure_evidence_sha256"
        ),
        "predecessor_failure_evidence": authorization.get(
            "predecessor_failure_evidence"
        ),
        "predecessor_failure_evidence_sha256": authorization.get(
            "predecessor_failure_evidence_sha256"
        ),
        "attempt1_environment_evidence": dict(
            ATTEMPT1_PUBLIC_ENVIRONMENT_EVIDENCE
        ),
        "worker_receipt_map_sha256": authorization.get(
            "worker_receipt_map_sha256"
        ),
        "original_snapshot_root": authorization.get("snapshot_root"),
        "original_snapshot_inventory_sha256": authorization.get(
            "snapshot_inventory_sha256"
        ),
        "original_package_protocol_sha256": authorization.get(
            "original_package_protocol_sha256"
        ),
        "repair_source_root": authorization.get("repair_source_root"),
        "repair_source_commit": authorization.get("repair_source_commit"),
        "repair_package_protocol_sha256": authorization.get(
            "repair_package_protocol_sha256"
        ),
        "repair_source_files_sha256": authorization.get(
            "repair_source_files_sha256"
        ),
        "repair_source_installation_method": authorization.get(
            "repair_source_installation_method"
        ),
        "repair_source_archive": authorization.get("repair_source_archive"),
        "repair_source_archive_sha256": authorization.get(
            "repair_source_archive_sha256"
        ),
        "repair_source_archive_size": authorization.get(
            "repair_source_archive_size"
        ),
        "repair_source_archive_format": authorization.get(
            "repair_source_archive_format"
        ),
        "report_publication_installation_method": authorization.get(
            "report_publication_installation_method"
        ),
        "scheduler_job_control_observation_sha256": authorization.get(
            "scheduler_job_control_observation_sha256"
        ),
        "worker_handoff_sha256": stable_hash(
            authorization.get("worker_handoff")
        ),
        "expected_report_bundle_sha256": EXPECTED_BUNDLE_SHA256,
        "expected_report_bundle_file_sha256": EXPECTED_BUNDLE_FILE_SHA256,
        "expected_gate_sha256": EXPECTED_GATE_SHA256,
        "expected_gate_decision_file_sha256": EXPECTED_DECISION_FILE_SHA256,
        "deterministic_reassembly_allowed": True,
        "scientific_input_change_allowed": False,
        "gate_change_allowed": False,
    }


class _ReportTreeBinding:
    def __init__(self, report_root: Path):
        self.root = report_root
        self.file_rows: dict[
            str, tuple[int, tuple[int, ...], bytes]
        ] = {}
        self.descriptor = os.open(
            report_root,
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            self.identity = os.fstat(self.descriptor)
            self.payloads = self._read_exact_tree()
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
            "published repaired report root identity differs",
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
            "published repaired report filename differs",
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
                f"retained published repaired report file changed: {name}",
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
                f"published repaired report file identity differs: {name}",
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
                f"published repaired report file changed: {name}",
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

    def _read_exact_tree(self) -> dict[str, bytes]:
        self._require_root()
        commit_payload = self._read_file("REPORT_COMMIT.json")
        commit = _decode_json(self.root / "REPORT_COMMIT.json", commit_payload)
        require(
            isinstance(commit.get("report_bundle"), str)
            and isinstance(commit.get("gate_decision"), str)
            and isinstance(commit.get("provenance"), str),
            "published repaired report commit paths differ",
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
            "published repaired report file coverage differs",
        )
        payloads = {"REPORT_COMMIT.json": commit_payload}
        for name in sorted(names - {"REPORT_COMMIT.json"}):
            payloads[name] = self._read_file(name)
        self._require_root()
        require(
            set(os.listdir(self.descriptor)) == names,
            "published repaired report namespace changed",
        )
        return payloads

    def revalidate(self) -> None:
        current = self._read_exact_tree()
        require(
            current == self.payloads,
            "published repaired report tree changed",
        )

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


def _legacy_validated_repaired_report_tree(
    submission_root: Path,
    submission_sha256: str,
) -> dict[str, Any] | None:
    report_root = submission_root / "report"
    if not os.path.lexists(report_root):
        return None
    binding = _ReportTreeBinding(report_root)
    try:
        return _legacy_validated_repaired_report_tree_bound(
            submission_root, submission_sha256, binding
        )
    finally:
        binding.close()


def _legacy_validated_repaired_report_tree_bound(
    submission_root: Path,
    submission_sha256: str,
    binding: _ReportTreeBinding,
) -> dict[str, Any]:
    root = binding.root
    commit_payload = binding.payloads["REPORT_COMMIT.json"]
    commit = _decode_json(
        root / "REPORT_COMMIT.json", commit_payload
    )
    canonical_commit_payload = (
        json.dumps(commit, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    bundle_name = f"REPORT_BUNDLE.{EXPECTED_BUNDLE_SHA256}.json"
    gate_name = f"GATE_DECISION.{EXPECTED_GATE_SHA256}.json"
    require(
        commit_payload == canonical_commit_payload
        and set(commit) == REPORT_COMMIT_KEYS
        and commit.get("schema_version") == 1
        and commit.get("status") == EXPECTED_REPORT_STATUS
        and commit.get("scientific_rejection") is True
        and commit.get("campaign_id") == CAMPAIGN_ID
        and commit.get("submission_sha256") == submission_sha256
        and commit.get("report_bundle_sha256") == EXPECTED_BUNDLE_SHA256
        and commit.get("report_bundle_file_sha256") == EXPECTED_BUNDLE_FILE_SHA256
        and commit.get("gate_sha256") == EXPECTED_GATE_SHA256
        and commit.get("gate_decision_file_sha256") == EXPECTED_DECISION_FILE_SHA256
        and commit.get("report_bundle") == bundle_name
        and commit.get("gate_decision") == gate_name
        and isinstance(commit.get("provenance_sha256"), str)
        and SHA256_RE.fullmatch(commit["provenance_sha256"]) is not None
        and commit.get("provenance")
        == f"REPORT_PROVENANCE.{commit['provenance_sha256']}.json",
        "published repaired report commit differs",
    )
    names = {
        "REPORT_COMMIT.json",
        bundle_name,
        gate_name,
        str(commit["provenance"]),
    }
    require(
        set(binding.payloads) == names,
        "published repaired report file coverage differs",
    )
    require(
        hashlib.sha256(binding.payloads[bundle_name]).hexdigest()
        == EXPECTED_BUNDLE_FILE_SHA256
        and hashlib.sha256(binding.payloads[gate_name]).hexdigest()
        == EXPECTED_DECISION_FILE_SHA256,
        "published repaired scientific bundle/gate bytes differ",
    )
    provenance_payload = binding.payloads[str(commit["provenance"])]
    provenance = _decode_json(
        root / str(commit["provenance"]), provenance_payload
    )
    canonical_provenance_payload = (
        json.dumps(provenance, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    provenance_v1 = dict(provenance)
    provenance_v1.pop("publication_authority", None)
    provenance_v1["schema_version"] = 1
    require(
        provenance_payload == canonical_provenance_payload
        and hashlib.sha256(provenance_payload).hexdigest()
        == commit.get("provenance_file_sha256")
        and stable_hash(provenance) == commit.get("provenance_sha256")
        and set(provenance) == PROVENANCE_V1_KEYS | {"publication_authority"}
        and provenance.get("schema_version") == 2
        and provenance.get("submission_sha256") == submission_sha256
        and isinstance(provenance.get("publication_authority"), Mapping),
        "published repaired report provenance differs",
    )
    require(
        set(provenance_v1) == PROVENANCE_V1_KEYS
        and provenance_v1.get("campaign_id") == CAMPAIGN_ID
        and provenance_v1.get("submission_sha256") == submission_sha256
        and stable_hash(provenance_v1) == EXPECTED_PROVENANCE_V1_SHA256
        and _pretty_json_sha(provenance_v1) == EXPECTED_PROVENANCE_V1_FILE_SHA256
        and _pretty_json_size(provenance_v1) == EXPECTED_PROVENANCE_V1_FILE_SIZE
        and provenance_v1.get("report_bundle_sha256") == EXPECTED_BUNDLE_SHA256
        and provenance_v1.get("gate_sha256") == EXPECTED_GATE_SHA256,
        "published repaired provenance-v1 reconstruction differs",
    )
    authority = provenance["publication_authority"]
    authorization, authorization_digest, authorization_info = read_json(
        submission_root / "REPORT_REPAIR_0002_AUTHORIZED.json",
        "published repaired authorization",
    )
    release, release_digest, release_info = read_json(
        submission_root / "REPORT_REPAIR_0002_RELEASED.json",
        "published repaired release",
    )
    expected_authority = _expected_publication_authority(
        authorization, authorization_digest, release_digest
    )
    require(
        stat.S_IMODE(authorization_info.st_mode) == 0o444
        and authorization_info.st_uid == os.getuid()
        and authorization_info.st_nlink == 1
        and stat.S_IMODE(release_info.st_mode) == 0o444
        and release_info.st_uid == os.getuid()
        and release_info.st_nlink == 1
        and release.get("authorization_sha256") == authorization_digest
        and set(authority) == PUBLICATION_AUTHORITY_KEYS
        and exact_json_equal(authority, expected_authority)
        and authority.get("schema_version") == 2
        and authority.get("status") == "authorized_terminal_report_repair"
        and authority.get("attempt") == ATTEMPT
        and authority.get("original_report_job_id") == EXPECTED_ORIGINAL_REPORT_JOB_ID
        and isinstance(authority.get("repair_report_job_id"), str)
        and JOB_ID_RE.fullmatch(authority["repair_report_job_id"]) is not None
        and authority.get("authorization")
        == "REPORT_REPAIR_0002_AUTHORIZED.json"
        and SHA256_RE.fullmatch(str(authority.get("authorization_sha256", "")))
        is not None
        and authority.get("release") == "REPORT_REPAIR_0002_RELEASED.json"
        and SHA256_RE.fullmatch(str(authority.get("release_sha256", ""))) is not None
        and authority.get("original_failure_evidence")
        == "journal/REPORT_REPAIR_0001_ORIGINAL_FAILURE.json"
        and SHA256_RE.fullmatch(
            str(authority.get("original_failure_evidence_sha256", ""))
        )
        is not None
        and authority.get("predecessor_failure_evidence")
        == "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
        and SHA256_RE.fullmatch(
            str(authority.get("predecessor_failure_evidence_sha256", ""))
        )
        is not None
        and exact_json_equal(
            authority.get("attempt1_environment_evidence"),
            ATTEMPT1_PUBLIC_ENVIRONMENT_EVIDENCE,
        )
        and authority.get("original_snapshot_root")
        == str(submission_root / "source-snapshot/repo")
        and authority.get("original_snapshot_inventory_sha256")
        == EXPECTED_SNAPSHOT_INVENTORY_SHA256
        and authority.get("original_package_protocol_sha256")
        == EXPECTED_ORIGINAL_PROTOCOL
        and authority.get("repair_source_installation_method")
        == DIRECT_FINAL_INSTALL_METHOD
        and authority.get("report_publication_installation_method")
        == DIRECT_FINAL_INSTALL_METHOD
        and SHA256_RE.fullmatch(
            str(authority.get("scheduler_job_control_observation_sha256", ""))
        )
        is not None
        and authority.get("worker_handoff_sha256")
        == stable_hash(REPAIR_WORKER_HANDOFF)
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
        (
            "predecessor_failure_evidence",
            "predecessor_failure_evidence_sha256",
            "attempt2 predecessor failure",
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
    binding.revalidate()
    return dict(commit)


def _read_publication_archive_fd(
    submission_root: Path,
    submission_sha256: str,
    path: Path,
    descriptor: int,
    expected_digest: str,
    expected_size: int,
    *,
    transition: _RepairTransitionBinding | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate a publication archive without materializing its 424MB bundle."""

    def exact(offset: int, size: int) -> bytes:
        chunks: list[bytes] = []
        cursor = offset
        remaining = size
        while remaining:
            block = os.pread(descriptor, min(16 * 1024 * 1024, remaining), cursor)
            require(block, "publication archive is truncated")
            chunks.append(block)
            cursor += len(block)
            remaining -= len(block)
        return b"".join(chunks)

    opened = os.fstat(descriptor)
    named = path.lstat()
    require(
        stat.S_ISREG(opened.st_mode)
        and _file_identity(opened) == _file_identity(named)
        and opened.st_uid == named.st_uid == os.getuid()
        and opened.st_nlink == named.st_nlink == 1
        and stat.S_IMODE(opened.st_mode) == stat.S_IMODE(named.st_mode) == 0o444
        and opened.st_size == expected_size
        and path.name
        == f"{PUBLICATION_ARCHIVE_PREFIX}{expected_digest}{PUBLICATION_ARCHIVE_SUFFIX}",
        "publication archive identity differs",
    )
    archive_hash = hashlib.sha256()
    archive_offset = 0
    while archive_offset < expected_size:
        block = os.pread(
            descriptor,
            min(16 * 1024 * 1024, expected_size - archive_offset),
            archive_offset,
        )
        require(block, "publication archive hash stream is truncated")
        archive_hash.update(block)
        archive_offset += len(block)
    require(
        archive_hash.hexdigest() == expected_digest,
        "publication archive hash differs",
    )
    offset = 0
    require(
        exact(offset, len(PUBLICATION_ARCHIVE_MAGIC)) == PUBLICATION_ARCHIVE_MAGIC,
        "publication archive magic differs",
    )
    offset += len(PUBLICATION_ARCHIVE_MAGIC)
    header_size = int.from_bytes(exact(offset, 8), "big")
    offset += 8
    require(0 < header_size <= 1 << 20, "publication archive header size differs")
    header_payload = exact(offset, header_size)
    offset += header_size
    try:
        header = json.loads(header_payload.decode("ascii"), object_pairs_hook=_pairs(path))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepairError(f"publication archive header differs: {exc}") from exc
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
        and header.get("schema_version") == 2
        and header.get("campaign_id") == CAMPAIGN_ID
        and header.get("submission_sha256") == submission_sha256
        and header.get("entry_order") == list(PUBLICATION_ARCHIVE_ENTRY_ORDER)
        and isinstance(header.get("entries"), list)
        and len(header["entries"]) == 4,
        "publication archive header differs",
    )
    captured: dict[str, bytes] = {}
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
            and SHA256_RE.fullmatch(str(row.get("sha256", ""))) is not None
            and SHA256_RE.fullmatch(str(row.get("logical_sha256", ""))) is not None,
            f"publication archive header row differs: {kind}",
        )
        name_size = int.from_bytes(exact(offset, 8), "big")
        offset += 8
        require(0 < name_size <= 256, "publication archive entry name size differs")
        try:
            frame_name = exact(offset, name_size).decode("ascii")
        except UnicodeDecodeError as exc:
            raise RepairError("publication archive entry name is not ASCII") from exc
        offset += name_size
        frame_size = int.from_bytes(exact(offset, 8), "big")
        offset += 8
        require(
            frame_name == row["name"] and frame_size == row["size"],
            f"publication archive frame differs: {kind}",
        )
        digest = hashlib.sha256()
        pieces: list[bytes] | None = [] if kind != "report_bundle" else None
        remaining = frame_size
        while remaining:
            block = exact(offset, min(16 * 1024 * 1024, remaining))
            digest.update(block)
            if pieces is not None:
                pieces.append(block)
            offset += len(block)
            remaining -= len(block)
        require(digest.hexdigest() == row["sha256"], f"publication entry hash differs: {kind}")
        if pieces is not None:
            captured[kind] = b"".join(pieces)
    require(offset == expected_size, "publication archive has trailing bytes")
    rows = {row["kind"]: row for row in header["entries"]}
    require(
        rows["report_bundle"]["name"]
        == f"REPORT_BUNDLE.{EXPECTED_BUNDLE_SHA256}.json"
        and rows["report_bundle"]["size"] == EXPECTED_BUNDLE_FILE_SIZE
        and rows["report_bundle"]["sha256"] == EXPECTED_BUNDLE_FILE_SHA256
        and rows["report_bundle"]["logical_sha256"] == EXPECTED_BUNDLE_SHA256
        and rows["gate_decision"]["name"]
        == f"GATE_DECISION.{EXPECTED_GATE_SHA256}.json"
        and rows["gate_decision"]["size"] == EXPECTED_DECISION_FILE_SIZE
        and rows["gate_decision"]["sha256"] == EXPECTED_DECISION_FILE_SHA256
        and rows["gate_decision"]["logical_sha256"] == EXPECTED_GATE_SHA256
        and rows["report_commit"]["name"] == "REPORT_COMMIT.json",
        "publication archive pinned scientific entries differ",
    )
    provenance = _decode_json(path, captured["provenance"])
    commit = _decode_json(path, captured["report_commit"])
    provenance_v1 = dict(provenance)
    provenance_v1.pop("publication_authority", None)
    provenance_v1["schema_version"] = 1
    require(
        captured["provenance"]
        == (json.dumps(provenance, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        and rows["provenance"]["name"]
        == f"REPORT_PROVENANCE.{stable_hash(provenance)}.json"
        and rows["provenance"]["sha256"]
        == hashlib.sha256(captured["provenance"]).hexdigest()
        and rows["provenance"]["logical_sha256"] == stable_hash(provenance)
        and set(provenance) == PROVENANCE_V1_KEYS | {"publication_authority"}
        and provenance.get("schema_version") == 2
        and stable_hash(provenance_v1) == EXPECTED_PROVENANCE_V1_SHA256
        and _pretty_json_sha(provenance_v1) == EXPECTED_PROVENANCE_V1_FILE_SHA256
        and _pretty_json_size(provenance_v1) == EXPECTED_PROVENANCE_V1_FILE_SIZE,
        "publication archive provenance differs",
    )
    require(
        captured["report_commit"]
        == (json.dumps(commit, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
        and set(commit) == REPORT_COMMIT_KEYS
        and commit.get("schema_version") == 1
        and commit.get("status") == EXPECTED_REPORT_STATUS
        and commit.get("scientific_rejection") is True
        and commit.get("campaign_id") == CAMPAIGN_ID
        and commit.get("submission_sha256") == submission_sha256
        and commit.get("report_bundle") == rows["report_bundle"]["name"]
        and commit.get("report_bundle_sha256") == EXPECTED_BUNDLE_SHA256
        and commit.get("report_bundle_file_sha256") == EXPECTED_BUNDLE_FILE_SHA256
        and commit.get("gate_decision") == rows["gate_decision"]["name"]
        and commit.get("gate_sha256") == EXPECTED_GATE_SHA256
        and commit.get("gate_decision_file_sha256") == EXPECTED_DECISION_FILE_SHA256
        and commit.get("provenance") == rows["provenance"]["name"]
        and commit.get("provenance_sha256") == rows["provenance"]["logical_sha256"]
        and commit.get("provenance_file_sha256") == rows["provenance"]["sha256"]
        and rows["report_commit"]["sha256"]
        == hashlib.sha256(captured["report_commit"]).hexdigest()
        and rows["report_commit"]["logical_sha256"] == stable_hash(commit)
        and header.get("report_commit_sha256") == rows["report_commit"]["sha256"]
        and header.get("report_commit_value_sha256") == stable_hash(commit),
        "publication archive commit differs",
    )
    if transition is not None:
        authorization_payload = transition._retained_payload(
            _authorization_path(submission_root)
        )
        release_payload = transition._retained_payload(_released_path(submission_root))
        authorization = _decode_json(
            _authorization_path(submission_root), authorization_payload
        )
        release = _decode_json(_released_path(submission_root), release_payload)
        authorization_digest = hashlib.sha256(authorization_payload).hexdigest()
        release_digest = hashlib.sha256(release_payload).hexdigest()
    else:
        authorization, authorization_digest, _info = read_json(
            _authorization_path(submission_root), "publication authorization"
        )
        release, release_digest, _info = read_json(
            _released_path(submission_root), "publication release"
        )
    expected_authority = _expected_publication_authority(
        authorization, authorization_digest, release_digest
    )
    require(
        release.get("authorization_sha256") == authorization_digest
        and exact_json_equal(provenance.get("publication_authority"), expected_authority),
        "publication archive public authority differs",
    )
    if transition is not None:
        transition.revalidate()
    return commit, header


def _validated_repaired_report_tree(
    submission_root: Path,
    submission_sha256: str,
    *,
    transition: _RepairTransitionBinding | None = None,
) -> dict[str, Any] | None:
    names = _publication_archive_names(submission_root)
    if not names:
        return None
    path = submission_root / names[0]
    expected_digest = names[0][
        len(PUBLICATION_ARCHIVE_PREFIX) : -len(PUBLICATION_ARCHIVE_SUFFIX)
    ]
    require(
        SHA256_RE.fullmatch(expected_digest) is not None,
        "publication archive name hash differs",
    )
    owned_descriptor = -1
    if transition is not None:
        retained = transition.retained_publication_archive()
        require(
            retained is not None
            and retained[0] == path
            and retained[2] == expected_digest,
            "publication archive retained binding differs",
        )
        descriptor, expected_size = retained[1], retained[3]
    else:
        owned_descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
        descriptor = owned_descriptor
        expected_size = os.fstat(descriptor).st_size
    try:
        commit, _header = _read_publication_archive_fd(
            submission_root,
            submission_sha256,
            path,
            descriptor,
            expected_digest,
            expected_size,
            transition=transition,
        )
        return commit
    finally:
        if owned_descriptor >= 0:
            os.close(owned_descriptor)


def _legacy_validated_repair_completed(
    submission_root: Path,
    submission_sha256: str,
    authorization: Mapping[str, Any],
    authorization_sha256: str,
    commit: Mapping[str, Any],
    *,
    candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    path = _completed_path(submission_root)
    if candidate is None:
        if not os.path.lexists(path):
            return None
        value, _digest, info = read_json(path, "completed report repair")
        stage_path = path.parent / f".{path.name}.seal.tmp"
        linked_completed_stage = False
        if info.st_nlink == 2 and os.path.lexists(stage_path):
            stage_info = stage_path.lstat()
            linked_completed_stage = (
                stat.S_ISREG(stage_info.st_mode)
                and stage_info.st_uid == os.getuid()
                and stage_info.st_nlink == 2
                and stat.S_IMODE(stage_info.st_mode) == 0o444
                and _file_identity(stage_info) == _file_identity(info)
            )
        identity_valid = (
            stat.S_IMODE(info.st_mode) == 0o444
            and info.st_uid == os.getuid()
            and (info.st_nlink == 1 or linked_completed_stage)
        )
    else:
        value = dict(candidate)
        identity_valid = True
    commit_value, commit_sha256, commit_info = read_json(
        submission_root / "report/REPORT_COMMIT.json",
        "completed repaired report commit",
    )
    release_value, release_sha256, release_info = read_json(
        _released_path(submission_root), "completed report repair release"
    )
    provenance, _provenance_sha256, _provenance_info = read_json(
        submission_root / "report" / str(commit["provenance"]),
        "completed repaired report provenance",
    )
    publication_authority = provenance.get("publication_authority")
    require(
        identity_valid
        and stat.S_IMODE(commit_info.st_mode) == 0o444
        and commit_info.st_uid == os.getuid()
        and commit_info.st_nlink == 1
        and stat.S_IMODE(release_info.st_mode) == 0o444
        and release_info.st_uid == os.getuid()
        and release_info.st_nlink == 1
        and set(value) == COMPLETED_KEYS
        and value.get("schema_version") == 1
        and value.get("status")
        == "report_repair_terminal_publication_complete"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and value.get("attempt") == ATTEMPT
        and value.get("repair_report_job_id")
        == authorization.get("repair_report_job_id")
        and value.get("predecessor_failure_evidence")
        == "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
        and value.get("predecessor_failure_evidence_sha256")
        == authorization.get("predecessor_failure_evidence_sha256")
        and value.get("authorization")
        == "REPORT_REPAIR_0002_AUTHORIZED.json"
        and value.get("authorization_sha256") == authorization_sha256
        and value.get("release") == "REPORT_REPAIR_0002_RELEASED.json"
        and value.get("release_sha256") == release_sha256
        and release_value.get("authorization_sha256") == authorization_sha256
        and value.get("report_commit") == "report/REPORT_COMMIT.json"
        and value.get("report_commit_sha256") == commit_sha256
        and exact_json_equal(commit_value, commit)
        and exact_json_equal(value.get("report_commit_value"), commit)
        and value.get("report_commit_value_sha256") == stable_hash(commit)
        and isinstance(publication_authority, Mapping)
        and value.get("publication_authority_sha256")
        == stable_hash(publication_authority)
        and value.get("repair_source_installation_method")
        == authorization.get("repair_source_installation_method")
        and value.get("report_publication_installation_method")
        == authorization.get("report_publication_installation_method")
        and exact_json_equal(
            value.get("expected_reassembly"), _expected_reassembly()
        )
        and value.get("publication_complete") is True
        and value.get("retry_allowed") is False
        and value.get("successor_attempt_allowed") is False
        and isinstance(value.get("completed_at_utc"), str)
        and bool(value["completed_at_utc"]),
        "completed report repair evidence differs",
    )
    return dict(value)


def _legacy_seal_repair_completed(
    submission_root: Path,
    submission_sha256: str,
    authorization: Mapping[str, Any],
    authorization_sha256: str,
    commit: Mapping[str, Any],
    locks: _RepairLocks,
) -> dict[str, Any]:
    with _retained_transition_scope(
        submission_root, locks, source_must_be_installed=True
    ) as transition:
        binding = _ReportTreeBinding(submission_root / "report")
        try:
            transition.revalidate()
            return _seal_repair_completed_bound(
                submission_root,
                submission_sha256,
                authorization,
                authorization_sha256,
                commit,
                binding,
                locks,
                transition,
            )
        finally:
            binding.close()


def _legacy_seal_repair_completed_bound(
    submission_root: Path,
    submission_sha256: str,
    authorization: Mapping[str, Any],
    authorization_sha256: str,
    commit: Mapping[str, Any],
    binding: _ReportTreeBinding,
    locks: _RepairLocks,
    transition: _RepairTransitionBinding,
) -> dict[str, Any]:
    authenticated_commit = _validated_repaired_report_tree_bound(
        submission_root, submission_sha256, binding
    )
    require(
        exact_json_equal(authenticated_commit, commit),
        "completed report repair tree authority differs",
    )
    phase = _classify_repair_phase(
        submission_root, source_must_be_installed=True
    )
    require(
        phase.staged_target in {None, _completed_path(submission_root).name},
        "completed report repair has a foreign staged successor",
    )
    if phase.staged_target == _completed_path(submission_root).name:
        if phase.staged_mode == 0o600:
            _discard_phase_next_partial_stage(submission_root, phase)
    if (
        phase.staged_target == _completed_path(submission_root).name
        and phase.staged_mode == 0o444
        and not os.path.lexists(_completed_path(submission_root))
    ):
        staged_completed = _phase_staged_value(submission_root, phase)
        assert staged_completed is not None
        existing = _validated_repair_completed(
            submission_root,
            submission_sha256,
            authorization,
            authorization_sha256,
            commit,
            candidate=staged_completed,
        )
    else:
        existing = _validated_repair_completed(
            submission_root,
            submission_sha256,
            authorization,
            authorization_sha256,
            commit,
        )
    if existing is not None:
        preexisting_phase = _classify_repair_phase(
            submission_root, source_must_be_installed=True
        )
        require(
            preexisting_phase.staged_target
            in {None, _completed_path(submission_root).name},
            "completed report repair changed before recovery seal",
        )
        binding.revalidate()
        transition.seal_successor(
            _completed_path(submission_root), existing
        )
        binding.revalidate()
        binding.revalidate()
        rebound_phase = _classify_repair_phase(
            submission_root, source_must_be_installed=True
        )
        require(
            rebound_phase.staged_target is None,
            "completed report repair stage survived recovery",
        )
        rebound = _validated_repair_completed(
            submission_root,
            submission_sha256,
            authorization,
            authorization_sha256,
            commit,
        )
        require(rebound is not None, "completed report repair recovery differs")
        return rebound
    commit_value, commit_sha256, _commit_info = read_json(
        submission_root / "report/REPORT_COMMIT.json",
        "completed repaired report commit",
    )
    _release_value, release_sha256, _release_info = read_json(
        _released_path(submission_root), "completed report repair release"
    )
    provenance, _provenance_sha256, _provenance_info = read_json(
        submission_root / "report" / str(commit["provenance"]),
        "completed repaired report provenance",
    )
    value = {
        "schema_version": 1,
        "status": "report_repair_terminal_publication_complete",
        "campaign_id": CAMPAIGN_ID,
        "submission_sha256": submission_sha256,
        "attempt": ATTEMPT,
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
        "release_sha256": release_sha256,
        "report_commit": "report/REPORT_COMMIT.json",
        "report_commit_sha256": commit_sha256,
        "report_commit_value": dict(commit_value),
        "report_commit_value_sha256": stable_hash(commit_value),
        "publication_authority_sha256": stable_hash(
            provenance["publication_authority"]
        ),
        "repair_source_installation_method": authorization[
            "repair_source_installation_method"
        ],
        "report_publication_installation_method": authorization[
            "report_publication_installation_method"
        ],
        "expected_reassembly": _expected_reassembly(),
        "publication_complete": True,
        "retry_allowed": False,
        "successor_attempt_allowed": False,
        "completed_at_utc": _utc_now(),
    }
    rebound_commit = _validated_repaired_report_tree_bound(
        submission_root, submission_sha256, binding
    )
    require(
        exact_json_equal(rebound_commit, commit),
        "completed report repair tree changed before terminal seal",
    )
    preseal_phase = _classify_repair_phase(
        submission_root, source_must_be_installed=True
    )
    require(
        preseal_phase.staged_target is None,
        "completed report repair phase changed before terminal seal",
    )
    binding.revalidate()
    transition.seal_successor(_completed_path(submission_root), value)
    binding.revalidate()
    binding.revalidate()
    completed_phase = _classify_repair_phase(
        submission_root, source_must_be_installed=True
    )
    require(
        completed_phase.staged_target is None,
        "completed report repair stage survived seal",
    )
    validated = _validated_repair_completed(
        submission_root,
        submission_sha256,
        authorization,
        authorization_sha256,
        commit,
    )
    require(validated is not None, "completed report repair was not sealed")
    return validated


def _validated_repair_completed(
    submission_root: Path,
    submission_sha256: str,
    authorization: Mapping[str, Any],
    authorization_sha256: str,
    commit: Mapping[str, Any],
    *,
    transition: _RepairTransitionBinding,
    candidate: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    path = _completed_path(submission_root)
    if candidate is None:
        if path.name not in transition.phase.durable_names:
            return None
        payload = transition._retained_payload(path)
        value = _decode_json(path, payload)
        identity_valid = True
    else:
        value = dict(candidate)
        identity_valid = True
    retained = transition.retained_publication_archive()
    require(retained is not None, "completed publication archive is absent")
    archive_path, archive_fd, archive_sha, archive_size = retained
    authenticated_commit, header = _read_publication_archive_fd(
        submission_root,
        submission_sha256,
        archive_path,
        archive_fd,
        archive_sha,
        archive_size,
        transition=transition,
    )
    release_payload = transition._retained_payload(_released_path(submission_root))
    release = _decode_json(_released_path(submission_root), release_payload)
    release_sha = hashlib.sha256(release_payload).hexdigest()
    expected_authority = _expected_publication_authority(
        authorization, authorization_sha256, release_sha
    )
    commit_rows = [
        row for row in header["entries"] if row.get("kind") == "report_commit"
    ]
    require(
        len(commit_rows) == 1
        and exact_json_equal(authenticated_commit, commit)
        and identity_valid
        and set(value) == COMPLETED_KEYS
        and value.get("schema_version") == 1
        and value.get("status") == "report_repair_terminal_publication_complete"
        and value.get("campaign_id") == CAMPAIGN_ID
        and value.get("submission_sha256") == submission_sha256
        and value.get("attempt") == ATTEMPT
        and value.get("repair_report_job_id")
        == authorization.get("repair_report_job_id")
        and value.get("predecessor_failure_evidence")
        == "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
        and value.get("predecessor_failure_evidence_sha256")
        == authorization.get("predecessor_failure_evidence_sha256")
        and value.get("authorization") == "REPORT_REPAIR_0002_AUTHORIZED.json"
        and value.get("authorization_sha256") == authorization_sha256
        and value.get("release") == "REPORT_REPAIR_0002_RELEASED.json"
        and value.get("release_sha256") == release_sha
        and release.get("authorization_sha256") == authorization_sha256
        and value.get("report_commit") == "REPORT_COMMIT.json"
        and value.get("report_commit_sha256") == commit_rows[0]["sha256"]
        and exact_json_equal(value.get("report_commit_value"), commit)
        and value.get("report_commit_value_sha256") == stable_hash(commit)
        and value.get("publication_archive") == archive_path.name
        and value.get("publication_archive_sha256") == archive_sha
        and value.get("publication_archive_size") == archive_size
        and value.get("publication_archive_header_sha256") == stable_hash(header)
        and exact_json_equal(
            _validated_direct_final_file_identity(
                value.get("publication_archive_file_identity"),
                size=archive_size,
                label="completed publication creation identity",
            ),
            transition.retained_direct_file_identity(archive_path),
        )
        and value.get("publication_authority_sha256")
        == stable_hash(expected_authority)
        and value.get("repair_source_installation_method")
        == SOURCE_ARCHIVE_INSTALL_METHOD
        and value.get("report_publication_installation_method")
        == PUBLICATION_ARCHIVE_INSTALL_METHOD
        and exact_json_equal(value.get("expected_reassembly"), _expected_reassembly())
        and value.get("publication_complete") is True
        and value.get("retry_allowed") is False
        and value.get("successor_attempt_allowed") is False
        and isinstance(value.get("completed_at_utc"), str)
        and bool(value["completed_at_utc"]),
        "completed report repair archive evidence differs",
    )
    transition.revalidate()
    return dict(value)


def _seal_repair_completed(
    submission_root: Path,
    submission_sha256: str,
    authorization: Mapping[str, Any],
    authorization_sha256: str,
    commit: Mapping[str, Any],
    locks: _RepairLocks,
    *,
    transition: _RepairTransitionBinding | None = None,
) -> dict[str, Any]:
    scope = (
        _retained_transition_scope(
            submission_root, locks, source_must_be_installed=True
        )
        if transition is None
        else contextlib.nullcontext(transition)
    )
    with scope as transition:
        retained = transition.retained_publication_archive()
        require(retained is not None, "completed publication archive is absent")
        archive_path, archive_fd, archive_sha, archive_size = retained
        authenticated_commit, header = _read_publication_archive_fd(
            submission_root,
            submission_sha256,
            archive_path,
            archive_fd,
            archive_sha,
            archive_size,
            transition=transition,
        )
        require(
            exact_json_equal(authenticated_commit, commit),
            "completed publication archive commit differs",
        )
        existing = _validated_repair_completed(
            submission_root,
            submission_sha256,
            authorization,
            authorization_sha256,
            commit,
            transition=transition,
        )
        if existing is not None:
            return existing
        release_payload = transition._retained_payload(
            _released_path(submission_root)
        )
        release_sha = hashlib.sha256(release_payload).hexdigest()
        expected_authority = _expected_publication_authority(
            authorization, authorization_sha256, release_sha
        )
        commit_row = next(
            row for row in header["entries"] if row.get("kind") == "report_commit"
        )
        value = {
            "schema_version": 1,
            "status": "report_repair_terminal_publication_complete",
            "campaign_id": CAMPAIGN_ID,
            "submission_sha256": submission_sha256,
            "attempt": ATTEMPT,
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
            "release_sha256": release_sha,
            "report_commit": "REPORT_COMMIT.json",
            "report_commit_sha256": commit_row["sha256"],
            "report_commit_value": dict(commit),
            "report_commit_value_sha256": stable_hash(commit),
            "publication_archive": archive_path.name,
            "publication_archive_sha256": archive_sha,
            "publication_archive_size": archive_size,
            "publication_archive_header_sha256": stable_hash(header),
            "publication_archive_file_identity": (
                transition.retained_direct_file_identity(archive_path)
            ),
            "publication_authority_sha256": stable_hash(expected_authority),
            "repair_source_installation_method": SOURCE_ARCHIVE_INSTALL_METHOD,
            "report_publication_installation_method": PUBLICATION_ARCHIVE_INSTALL_METHOD,
            "expected_reassembly": _expected_reassembly(),
            "publication_complete": True,
            "retry_allowed": False,
            "successor_attempt_allowed": False,
            "completed_at_utc": _utc_now(),
        }
        require(set(value) == COMPLETED_KEYS, "completed report repair schema differs")
        transition.revalidate()
        transition.seal_successor(path := _completed_path(submission_root), value)
        transition.revalidate()
        validated = _validated_repair_completed(
            submission_root,
            submission_sha256,
            authorization,
            authorization_sha256,
            commit,
            transition=transition,
        )
        require(validated is not None and path.name in transition.phase.durable_names,
                "completed report repair was not sealed")
        return validated


def _complete_existing_publication(
    submission_root: Path,
    submission_sha256: str,
    contract: Mapping[str, Any],
    receipt: Mapping[str, Any],
    receipt_map: Mapping[str, Any],
    expected_reassembly: Mapping[str, Any],
    predecessor_sha256: str,
    locks: _RepairLocks,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate archive-to-COMPLETED under one retained descriptor graph."""

    with _retained_transition_scope(
        submission_root, locks, source_must_be_installed=True
    ) as transition:
        published = _validated_repaired_report_tree(
            submission_root,
            submission_sha256,
            transition=transition,
        )
        require(published is not None, "published repaired report is absent")
        calling_payload = transition._retained_payload(
            _submit_calling_path(submission_root)
        )
        calling = _decode_json(_submit_calling_path(submission_root), calling_payload)
        calling_sha = hashlib.sha256(calling_payload).hexdigest()
        source = _source_from_calling(calling)
        _validate_sealed_repair_source(_repair_source_root(submission_root), source)
        transition.revalidate()
        failure_payload = transition._retained_payload(_failure_path(submission_root))
        failure = _decode_json(_failure_path(submission_root), failure_payload)
        failure_sha = hashlib.sha256(failure_payload).hexdigest()
        _validate_failure_evidence(
            failure,
            submission_root=submission_root,
            submission_sha256=submission_sha256,
            contract=contract,
            receipt_map=receipt_map,
            expected_reassembly=expected_reassembly,
        )
        _validate_submit_calling(
            calling,
            submission_root=submission_root,
            submission_sha256=submission_sha256,
            failure_sha256=failure_sha,
            predecessor_sha256=predecessor_sha256,
            source=source,
            contract=contract,
            locks=locks,
        )
        submitted_payload = transition._retained_payload(
            _submitted_path(submission_root)
        )
        submitted = _decode_json(_submitted_path(submission_root), submitted_payload)
        submitted_sha = hashlib.sha256(submitted_payload).hexdigest()
        _validate_submitted(
            submitted,
            submission_sha256=submission_sha256,
            calling_sha256=calling_sha,
            calling=calling,
            contract=contract,
        )
        authorization_payload = transition._retained_payload(
            _authorization_path(submission_root)
        )
        authorization = _decode_json(
            _authorization_path(submission_root), authorization_payload
        )
        authorization_sha = hashlib.sha256(authorization_payload).hexdigest()
        _validate_authorization(
            authorization,
            submission_root=submission_root,
            submission_sha256=submission_sha256,
            contract=contract,
            receipt=receipt,
            receipt_map=receipt_map,
            expected_reassembly=expected_reassembly,
            source=source,
            failure_sha256=failure_sha,
            predecessor_sha256=predecessor_sha256,
            calling_sha256=calling_sha,
            submitted_sha256=submitted_sha,
            calling=calling,
        )
        transition.revalidate()
        released = _validate_existing_release(
            submission_root,
            submission_sha256,
            authorization_sha,
            str(authorization["repair_report_job_id"]),
            contract,
            locks,
        )
        require(released is not None, "published repaired report lacks release evidence")
        transition.revalidate()
        completed = _seal_repair_completed(
            submission_root,
            submission_sha256,
            authorization,
            authorization_sha,
            published,
            locks,
            transition=transition,
        )
        transition.revalidate()
        return published, completed


def _source_from_calling(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "repair_source_commit": value.get("repair_source_commit"),
        "repair_package_protocol_sha256": value.get(
            "repair_package_protocol_sha256"
        ),
        "repair_source_files": value.get("repair_source_files"),
        "repair_source_files_sha256": value.get("repair_source_files_sha256"),
        "repair_source_installation_method": value.get(
            "repair_source_installation_method"
        ),
        "repair_source_archive": value.get("repair_source_archive"),
        "repair_source_archive_sha256": value.get(
            "repair_source_archive_sha256"
        ),
        "repair_source_archive_size": value.get("repair_source_archive_size"),
        "repair_source_archive_format": value.get(
            "repair_source_archive_format"
        ),
    }


def _begin_pre_submit_boundary(
    submission_root: Path,
    contract: Mapping[str, Any],
    runner: Runner,
    locks: _RepairLocks,
    *,
    sleep: Callable[[float], None],
) -> tuple[_RepairTransitionBinding, dict[str, Any]]:
    """Pin all local authority before collecting the final submit census."""

    inherited = _ACTIVE_REPAIR_TRANSITION.get()
    if inherited is not None:
        require(
            inherited.submission_root == submission_root
            and inherited.locks is locks
            and inherited.source_must_be_installed,
            "pre-submit retained transition differs",
        )
        inherited.revalidate()
        census = _scheduler_census(
            submission_root, contract, runner, locks, sleep=sleep
        )
        inherited.revalidate()
        return inherited, census

    transition = _RepairTransitionBinding(
        submission_root,
        locks,
        source_must_be_installed=True,
    )
    token = _ACTIVE_REPAIR_TRANSITION.set(transition)
    try:
        transition.revalidate()
        census = _scheduler_census(
            submission_root, contract, runner, locks, sleep=sleep
        )
        transition.revalidate()
        return transition, census
    except BaseException:
        transition.close()
        raise
    finally:
        _ACTIVE_REPAIR_TRANSITION.reset(token)


def _execute_report_repair_impl(
    repo_root: Path,
    submission_root: Path,
    submission_sha256: str,
    *,
    allow_initial_submission: bool,
    runner: Runner = _default_runner,
    sleep: Callable[[float], None] = time.sleep,
    _bound_locks: _RepairLocks | None = None,
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
    lock_scope: Any = (
        _RepairLocks(submission)
        if _bound_locks is None
        else contextlib.nullcontext(_bound_locks)
    )
    with lock_scope as locks:
        prepared_live_source: dict[str, Any] | None = None
        prepared_source_archive: bytes | None = None
        prepared_source_evidence: dict[str, Any] | None = None
        report_present = bool(_publication_archive_names(submission))
        calling_path = _submit_calling_path(submission)
        submitted_path = _submitted_path(submission)
        authorization_path = _authorization_path(submission)
        source_present_at_entry = os.path.lexists(_repair_source_root(submission))
        require(
            not (source_present_at_entry and not os.path.lexists(calling_path)),
            "unpaired source archive is permanent fail-stop evidence",
        )
        require(
            not (
                report_present
                and not os.path.lexists(_completed_path(submission))
            ),
            "unpaired publication archive is permanent fail-stop evidence",
        )
        phase = _classify_repair_phase(
            submission,
            source_must_be_installed=(
                report_present
                or os.path.lexists(calling_path)
                or os.path.lexists(authorization_path)
                or os.path.lexists(_repair_source_root(submission))
            ),
        )
        repair_namespace_names = list(phase.durable_names)
        _require_repair_prefix_graph(
            submission,
            phase.virtual_names,
            report_present=report_present,
            validate_disk_cleanup=False,
        )
        _require_report_install_probe_namespace(
            submission,
            allow_exact_crash_residue=not os.path.lexists(authorization_path),
        )
        release_successor_present = any(
            name
            in {
                "REPORT_REPAIR_0002_RELEASED.json",
                "REPORT_REPAIR_0002_TERMINAL_RELEASE_DENIED.json",
                "REPORT_REPAIR_0002_TERMINAL_WORKER_FAILURE.json",
            }
            or name.startswith("CALLING_REPORT_REPAIR_0002_RELEASE_")
            or name.startswith("REPORT_REPAIR_0002_RELEASE_RESULT_")
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
            report_program = _source_archive_report_program(
                _repair_source_root(submission)
            )
        elif os.path.lexists(_repair_source_root(submission)):
            _load_sealed_repair_source(_repair_source_root(submission))
            report_program = _source_archive_report_program(
                _repair_source_root(submission)
            )
        else:
            # Authenticate, frame, and retain the exact future worker image
            # before deterministic reassembly.  The same immutable bytes are
            # later written through the creator FD; validation can never run
            # one transient report.py and launch a separately reread file.
            prepared_live_source = _verified_live_repair_source(root)
            (
                prepared_source_archive,
                prepared_source_evidence,
            ) = _source_archive_payload(prepared_live_source, submission)
            _prepared_source, prepared_files = _parse_source_archive_payload(
                _repair_source_root(submission), prepared_source_archive
            )
            require(
                exact_json_equal(_prepared_source, prepared_source_evidence),
                "prepared source archive changed before report validation",
            )
            report_program = (
                Path(f"{_repair_source_root(submission)}::report.py"),
                prepared_files["report.py"],
            )

        contract, receipt, receipt_map, expected_reassembly = (
            _validate_original_submission(
                submission,
                submission_sha256,
                report_program=report_program,
                locks=locks,
            )
        )
        _active_transition_required(submission, locks).revalidate()
        _validated_attempt1_chain(submission)
        phase = _classify_repair_phase(
            submission,
            source_must_be_installed=os.path.lexists(
                _repair_source_root(submission)
            ),
        )
        if phase.staged_target == _attempt1_worker_failure_terminal_path(
            submission
        ).name:
            if phase.staged_mode == 0o600:
                _discard_phase_next_partial_stage(submission, phase)
            else:
                staged_terminal = _phase_staged_value(submission, phase)
                assert staged_terminal is not None
                validated_staged_terminal = (
                    _validated_attempt1_worker_failure_terminal(
                        submission,
                        submission_sha256,
                        contract,
                        candidate=staged_terminal,
                    )
                )
                require(
                    validated_staged_terminal is not None,
                    "staged attempt1 terminal evidence differs",
                )
                _seal_transition_json(
                    submission,
                    _attempt1_worker_failure_terminal_path(submission),
                    staged_terminal,
                    locks,
                    source_must_be_installed=os.path.lexists(
                        _repair_source_root(submission)
                    ),
                )
        attempt1_terminal = _validated_attempt1_worker_failure_terminal(
            submission, submission_sha256, contract
        )
        phase = _classify_repair_phase(
            submission,
            source_must_be_installed=os.path.lexists(
                _repair_source_root(submission)
            ),
        )
        if phase.staged_target == _attempt2_predecessor_path(submission).name:
            if phase.staged_mode == 0o600:
                _discard_phase_next_partial_stage(submission, phase)
            else:
                staged_predecessor = _phase_staged_value(submission, phase)
                assert staged_predecessor is not None
                validated_staged_predecessor = _validated_attempt2_predecessor(
                    submission,
                    submission_sha256,
                    contract,
                    receipt_map,
                    locks,
                    candidate=staged_predecessor,
                )
                require(
                    validated_staged_predecessor is not None,
                    "staged attempt2 predecessor evidence differs",
                )
                _seal_transition_json(
                    submission,
                    _attempt2_predecessor_path(submission),
                    staged_predecessor,
                    locks,
                    source_must_be_installed=os.path.lexists(
                        _repair_source_root(submission)
                    ),
                )
        predecessor = _validated_attempt2_predecessor(
            submission, submission_sha256, contract, receipt_map, locks
        )
        if report_present:
            require(
                attempt1_terminal is not None and predecessor is not None,
                "published attempt2 report lacks its terminal predecessor chain",
            )
        else:
            if attempt1_terminal is None:
                require(
                    allow_initial_submission,
                    "attempt2 recovery cannot create the attempt1 terminal boundary",
                )
                with _retained_transition_scope(
                    submission,
                    locks,
                    source_must_be_installed=os.path.lexists(
                        _repair_source_root(submission)
                    ),
                ):
                    terminal_census = _scheduler_census(
                        submission, contract, runner, locks, sleep=sleep
                    )
                    require(
                        terminal_census["settled_rows"] == [],
                        "attempt1 terminal boundary has active relevant jobs",
                    )
                    attempt1_accounting = _attempt1_terminal_scheduler_observation(
                        submission, contract, runner, locks
                    )
                    attempt1_terminal = _seal_attempt1_worker_failure_terminal(
                        submission,
                        submission_sha256,
                        contract,
                        terminal_census,
                        attempt1_accounting,
                        locks,
                    )
            if predecessor is None:
                terminal_value, terminal_sha256 = attempt1_terminal
                with _retained_transition_scope(
                    submission,
                    locks,
                    source_must_be_installed=os.path.lexists(
                        _repair_source_root(submission)
                    ),
                ):
                    predecessor = _seal_attempt2_predecessor(
                        submission,
                        submission_sha256,
                        contract,
                        receipt_map,
                        terminal_value,
                        terminal_sha256,
                        runner,
                        locks,
                    )
        assert attempt1_terminal is not None and predecessor is not None
        predecessor_value, predecessor_sha256 = predecessor
        phase = _classify_repair_phase(
            submission,
            source_must_be_installed=os.path.lexists(
                _repair_source_root(submission)
            ),
        )
        if phase.staged_target == calling_path.name:
            if phase.staged_mode == 0o600:
                _discard_phase_next_partial_stage(submission, phase)
            else:
                staged_calling = _phase_staged_value(submission, phase)
                assert staged_calling is not None
                staged_source = _source_from_calling(staged_calling)
                _validate_sealed_repair_source(
                    _repair_source_root(submission), staged_source
                )
                staged_failure, staged_failure_sha, staged_failure_info = read_json(
                    _failure_path(submission),
                    "staged submit-calling original failure evidence",
                )
                require(
                    stat.S_IMODE(staged_failure_info.st_mode) == 0o444
                    and staged_failure_info.st_uid == os.getuid()
                    and staged_failure_info.st_nlink == 1,
                    "staged submit-calling failure identity differs",
                )
                _validate_failure_evidence(
                    staged_failure,
                    submission_root=submission,
                    submission_sha256=submission_sha256,
                    contract=contract,
                    receipt_map=receipt_map,
                    expected_reassembly=expected_reassembly,
                )
                _validate_submit_calling(
                    staged_calling,
                    submission_root=submission,
                    submission_sha256=submission_sha256,
                    failure_sha256=staged_failure_sha,
                    predecessor_sha256=predecessor_sha256,
                    source=staged_source,
                    contract=contract,
                    locks=locks,
                )
                _seal_transition_json(
                    submission,
                    calling_path,
                    staged_calling,
                    locks,
                    source_must_be_installed=True,
                )
        repair_namespace_names = _require_known_single_generation_namespace(
            submission
        )
        if report_present:
            require(
                not os.path.lexists(_submit_failure_terminal_path(submission))
                and not os.path.lexists(_release_denied_path(submission))
                and not os.path.lexists(_worker_failure_terminal_path(submission))
                and _cancel_generation_count(submission) == 0,
                "published repaired report conflicts with terminal cleanup state",
            )
            published, completed = _complete_existing_publication(
                submission,
                submission_sha256,
                contract,
                receipt,
                receipt_map,
                expected_reassembly,
                predecessor_sha256,
                locks,
            )
            return {
                "schema_version": 1,
                "status": "report_repair_already_published",
                "commit": published,
                "completed": completed,
                "scheduler_calls": 0,
            }
        publication_state = _publication_state(submission)
        require(publication_state["report_absent"] is True, "repair publication state differs")

        failure_path = _failure_path(submission)
        if not os.path.lexists(calling_path):
            allowed_pre_calling_names = {
                *EXPECTED_ATTEMPT1_CHAIN_SHA256,
                "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json",
                "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json",
            }
            no_calling_repair_names = sorted(
                name
                for name in repair_namespace_names
                if name not in allowed_pre_calling_names
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
                or os.path.lexists(_attempt2_predecessor_path(submission)),
                "report repair recovery cannot create the initial scheduler call",
            )
            require(failure_was_sealed, "attempt2 original failure evidence is absent")
            if source_was_sealed:
                source_root = _repair_source_root(submission)
                source = _load_sealed_repair_source(source_root)
            else:
                with _retained_transition_scope(
                    submission, locks, source_must_be_installed=False
                ) as source_transition:
                    require(
                        prepared_live_source is not None
                        and prepared_source_archive is not None
                        and prepared_source_evidence is not None,
                        "prepared source archive is unavailable",
                    )
                    live_source = prepared_live_source
                    source_transition.revalidate()
                    source_root = _seal_repair_source_snapshot(
                        submission,
                        live_source,
                        locks,
                        enforce_phase=True,
                        transition=source_transition,
                        prepared_archive=prepared_source_archive,
                        prepared_evidence=prepared_source_evidence,
                    )
                source = _load_sealed_repair_source(source_root)
            _require_repair_filesystem_namespace(
                submission, source_must_be_installed=True
            )
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

            # The failure artifact's census is historical evidence about the
            # original report failure, not authority for this new scheduler
            # mutation.  Every path that has not yet sealed SUBMIT CALLING
            # therefore takes a new owner-wide settled census immediately at
            # the mutation boundary.  In particular, this covers recovery
            # from a kill after ORIGINAL_FAILURE was sealed.
            transition, scheduler_pre_submit_census = (
                _begin_pre_submit_boundary(
                    submission,
                    contract,
                    runner,
                    locks,
                    sleep=sleep,
                )
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
                == sorted(
                    {
                        *EXPECTED_ATTEMPT1_CHAIN_SHA256,
                        "REPORT_REPAIR_0001_TERMINAL_WORKER_FAILURE.json",
                        "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json",
                    }
                ),
                "report repair pre-submit append-only state differs",
            )
            rebound_predecessor = _validated_attempt2_predecessor(
                submission, submission_sha256, contract, receipt_map, locks
            )
            require(
                rebound_predecessor is not None
                and rebound_predecessor[1] == predecessor_sha256
                and exact_json_equal(
                    rebound_predecessor[0], predecessor_value
                ),
                "attempt2 predecessor changed before submission",
            )
            environment = _scheduler_environment(
                str(contract["scheduler_control_plane_contract"]["slurm_conf"])
            )
            transaction, report_cancel = locks.bindings()
            scheduler_source_input = _scheduler_source_archive_input(
                transition, source_root, source
            )
            command = _sbatch_command(
                source_root,
                submission,
                submission_sha256,
                source,
                scheduler_source_input,
            )
            calling = {
                "schema_version": 1,
                "status": "calling_held_report_repair_submission",
                "campaign_id": CAMPAIGN_ID,
                "submission_sha256": submission_sha256,
                "attempt": ATTEMPT,
                "original_failure_evidence": "journal/REPORT_REPAIR_0001_ORIGINAL_FAILURE.json",
                "original_failure_evidence_sha256": failure_sha,
                "predecessor_failure_evidence": (
                    "REPORT_REPAIR_0002_PREDECESSOR_FAILURE.json"
                ),
                "predecessor_failure_evidence_sha256": predecessor_sha256,
                "repair_source_root": str(source_root),
                "repair_source_commit": source["repair_source_commit"],
                "repair_package_protocol_sha256": source[
                    "repair_package_protocol_sha256"
                ],
                "repair_source_files": dict(source["repair_source_files"]),
                "repair_source_files_sha256": source[
                    "repair_source_files_sha256"
                ],
                "repair_source_installation_method": source[
                    "repair_source_installation_method"
                ],
                "repair_source_archive": source["repair_source_archive"],
                "repair_source_archive_sha256": source[
                    "repair_source_archive_sha256"
                ],
                "repair_source_archive_size": source[
                    "repair_source_archive_size"
                ],
                "repair_source_archive_format": source[
                    "repair_source_archive_format"
                ],
                "repair_source_archive_file_identity": (
                    transition.retained_direct_file_identity(source_root)
                ),
                "scheduler_source_archive_input": scheduler_source_input,
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
                predecessor_sha256=predecessor_sha256,
                source=source,
                contract=contract,
                locks=locks,
            )
            _classify_repair_phase(
                submission, source_must_be_installed=True
            )
            failed_census: dict[str, Any] | None = None
            failed_rows: list[dict[str, str]] | None = None
            inherited_transition = _ACTIVE_REPAIR_TRANSITION.get()
            transition_token = (
                None
                if inherited_transition is transition
                else _ACTIVE_REPAIR_TRANSITION.set(transition)
            )
            try:
                calling_sha = transition.seal_successor(
                    calling_path, calling
                )
                rebound_calling = _revalidated_sealed_json(
                    calling_path,
                    calling,
                    calling_sha,
                    "report repair submit calling before sbatch",
                )
                _validate_submit_calling(
                    rebound_calling,
                    submission_root=submission,
                    submission_sha256=submission_sha256,
                    failure_sha256=failure_sha,
                    predecessor_sha256=predecessor_sha256,
                    source=source,
                    contract=contract,
                    locks=locks,
                )
                transition.revalidate()
                result, evidence = _run(
                    runner,
                    command,
                    submission / "source-snapshot" / "repo",
                    environment,
                    locks,
                    pass_fds=(scheduler_source_input["descriptor"],),
                )
                transition.revalidate()
                rebound_calling = _revalidated_sealed_json(
                    calling_path,
                    calling,
                    calling_sha,
                    "report repair submit calling after sbatch",
                )
                _validate_submit_calling(
                    rebound_calling,
                    submission_root=submission,
                    submission_sha256=submission_sha256,
                    failure_sha256=failure_sha,
                    predecessor_sha256=predecessor_sha256,
                    source=source,
                    contract=contract,
                    locks=locks,
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
                    transition.revalidate()
                    transition.seal_successor(
                        _submitted_path(submission), submitted
                    )
                else:
                    failed_census = _scheduler_census(
                        submission, contract, runner, locks, sleep=sleep
                    )
                    transition.revalidate()
                    failed_rows = _repair_rows(
                        failed_census, submission_sha256, contract
                    )
                    if not failed_rows:
                        terminal = {
                            "schema_version": 1,
                            "status": "report_repair_terminal_submit_failure",
                            "campaign_id": CAMPAIGN_ID,
                            "submission_sha256": submission_sha256,
                            "attempt": ATTEMPT,
                            "submit_calling_sha256": calling_sha,
                            "scheduler_evidence": evidence,
                            "post_failure_census": failed_census,
                            "publication_allowed": False,
                            "retry_allowed": False,
                            "sealed_at_utc": _utc_now(),
                        }
                        transition.seal_successor(
                            _submit_failure_terminal_path(submission),
                            terminal,
                        )
                        validated_terminal = _validated_submit_failure_terminal(
                            submission,
                            submission_sha256,
                            calling,
                            calling_sha,
                            contract,
                        )
                        assert validated_terminal is not None
                        return validated_terminal
                    return _cleanup_repair_rows(
                        submission,
                        submission_sha256,
                        contract,
                        failed_census,
                        "nonzero_sbatch_with_live_exact_repair_identity",
                        runner,
                        locks,
                        sleep=sleep,
                    )
            finally:
                if transition_token is not None:
                    _ACTIVE_REPAIR_TRANSITION.reset(transition_token)
                    transition.close()
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
                predecessor_sha256=predecessor_sha256,
                source=source,
                contract=contract,
                locks=locks,
            )

        phase = _classify_repair_phase(
            submission, source_must_be_installed=True
        )
        staged_cleanup = _resolve_staged_cleanup_before_positive_transition(
            submission,
            submission_sha256,
            contract,
            phase,
            runner,
            locks,
            reason="staged_cleanup_prefix_before_positive_recovery",
            sleep=sleep,
        )
        if staged_cleanup is not None:
            return staged_cleanup
        if phase.staged_target == _submit_failure_terminal_path(submission).name:
            if phase.staged_mode == 0o600:
                _discard_phase_next_partial_stage(submission, phase)
            else:
                staged_submit_failure = _phase_staged_value(submission, phase)
                assert staged_submit_failure is not None
                validated_staged_submit_failure = (
                    _validated_submit_failure_terminal(
                        submission,
                        submission_sha256,
                        calling,
                        calling_sha,
                        contract,
                        candidate=staged_submit_failure,
                    )
                )
                require(
                    validated_staged_submit_failure is not None,
                    "staged report repair submit-failure differs",
                )
                _seal_transition_json(
                    submission,
                    _submit_failure_terminal_path(submission),
                    staged_submit_failure,
                    locks,
                    source_must_be_installed=True,
                )
        terminal_submit_failure = _validated_submit_failure_terminal(
            submission,
            submission_sha256,
            calling,
            calling_sha,
            contract,
        )
        if terminal_submit_failure is not None:
            require(
                not os.path.lexists(_submitted_path(submission))
                and not os.path.lexists(_authorization_path(submission))
                and not os.path.lexists(_released_path(submission)),
                "terminal report repair submit failure has successor state",
            )
            if _cancel_generation_count(submission) == 0:
                with _retained_transition_scope(
                    submission, locks, source_must_be_installed=True
                ):
                    delayed_census = _scheduler_census(
                        submission, contract, runner, locks, sleep=sleep
                    )
                    if _repair_rows(
                        delayed_census, submission_sha256, contract
                    ):
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
            with _retained_transition_scope(
                submission, locks, source_must_be_installed=True
            ):
                census = _scheduler_census(
                    submission, contract, runner, locks, sleep=sleep
                )
                latest_terminal = _cancel_terminal_path(
                    submission, existing_cancel_generation - 1
                )
                if os.path.lexists(latest_terminal) and not _repair_rows(
                    census, submission_sha256, contract
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
        phase = _classify_repair_phase(
            submission, source_must_be_installed=True
        )
        if phase.staged_target == _submitted_path(submission).name:
            if phase.staged_mode == 0o600:
                _discard_phase_next_partial_stage(submission, phase)
            else:
                staged_submitted = _phase_staged_value(submission, phase)
                assert staged_submitted is not None
                with _RepairTransitionBinding(
                    submission,
                    locks,
                    source_must_be_installed=True,
                ) as staged_transition:
                    _validate_submitted(
                        staged_submitted,
                        submission_sha256=submission_sha256,
                        calling_sha256=calling_sha,
                        calling=calling,
                        contract=contract,
                    )
                    staged_transition.seal_successor(
                        submitted_path, staged_submitted
                    )
        if not os.path.lexists(submitted_path):
            with _retained_transition_scope(
                submission, locks, source_must_be_installed=True
            ) as submit_recovery_transition:
                census = _scheduler_census(
                    submission, contract, runner, locks, sleep=sleep
                )
                submit_recovery_transition.revalidate()
                rows = _repair_rows(census, submission_sha256, contract)
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
                                row["job_id"] in HISTORICAL_JOB_IDS
                                for row in rows
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
                _classify_repair_phase(
                    submission, source_must_be_installed=True
                )
                submit_recovery_transition.seal_successor(
                    submitted_path, submitted
                )

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
            contract=contract,
        )
        job_id = str(submitted["repair_report_job_id"])

        phase = _classify_repair_phase(
            submission, source_must_be_installed=True
        )
        if phase.staged_target == authorization_path.name:
            if phase.staged_mode == 0o600:
                _discard_phase_next_partial_stage(submission, phase)
            else:
                staged_authorization = _phase_staged_value(submission, phase)
                assert staged_authorization is not None
                with _RepairTransitionBinding(
                    submission,
                    locks,
                    source_must_be_installed=True,
                ) as staged_transition:
                    _validate_authorization(
                        staged_authorization,
                        submission_root=submission,
                        submission_sha256=submission_sha256,
                        contract=contract,
                        receipt=receipt,
                        receipt_map=receipt_map,
                        expected_reassembly=expected_reassembly,
                        source=source,
                        failure_sha256=failure_sha,
                        predecessor_sha256=predecessor_sha256,
                        calling_sha256=calling_sha,
                        submitted_sha256=submitted_sha,
                        calling=calling,
                    )
                    staged_transition.seal_successor(
                        authorization_path, staged_authorization
                    )

        if not os.path.lexists(authorization_path):
            # Attempt 2 uses one universal direct-final O_EXCL publication
            # method.  No capability probe or removable residue is created.
            _classify_repair_phase(
                submission, source_must_be_installed=True
            )
            _require_report_install_probe_namespace(
                submission, allow_exact_crash_residue=False
            )
            report_installation_method = PUBLICATION_ARCHIVE_INSTALL_METHOD
            _require_repair_filesystem_namespace(
                submission, source_must_be_installed=True
            )

            def rebind_authorization_predecessors() -> None:
                rebound_calling = _revalidated_sealed_json(
                    calling_path,
                    calling,
                    calling_sha,
                    "report repair authorization submit calling",
                )
                _validate_submit_calling(
                    rebound_calling,
                    submission_root=submission,
                    submission_sha256=submission_sha256,
                    failure_sha256=failure_sha,
                    predecessor_sha256=predecessor_sha256,
                    source=source,
                    contract=contract,
                    locks=locks,
                )
                rebound_submitted = _revalidated_sealed_json(
                    submitted_path,
                    submitted,
                    submitted_sha,
                    "report repair authorization submitted evidence",
                )
                _validate_submitted(
                    rebound_submitted,
                    submission_sha256=submission_sha256,
                    calling_sha256=calling_sha,
                    calling=calling,
                    contract=contract,
                )
                rebound_failure = _revalidated_sealed_json(
                    failure_path,
                    failure,
                    failure_sha,
                    "report repair authorization original failure",
                )
                _validate_failure_evidence(
                    rebound_failure,
                    submission_root=submission,
                    submission_sha256=submission_sha256,
                    contract=contract,
                    receipt_map=receipt_map,
                    expected_reassembly=expected_reassembly,
                )

            with _retained_transition_scope(
                submission, locks, source_must_be_installed=True
            ) as authorization_transition:
                rebind_authorization_predecessors()
                census = _scheduler_census(
                    submission, contract, runner, locks, sleep=sleep
                )
                authorization_transition.revalidate()
                rows = _repair_rows(census, submission_sha256, contract)
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
                job_control = _job_control_observation(
                    submission,
                    submission_sha256,
                    contract,
                    job_id,
                    _repair_source_root(submission),
                    runner,
                    locks,
                    scheduler_command=_scheduler_source_argument_from_calling(
                        calling
                    ),
                )
                authorization_transition.revalidate()
                # The owner-wide census, scheduler job-control row, immutable
                # predecessor, sealed source, and both lock inodes are one
                # authority boundary.  Rebind them immediately before sealing
                # the authorization consumed by release and by the worker.
                _validate_sealed_repair_source(
                    _repair_source_root(submission), source
                )
                rebound_predecessor = _validated_attempt2_predecessor(
                    submission, submission_sha256, contract, receipt_map, locks
                )
                require(
                    rebound_predecessor is not None
                    and rebound_predecessor[1] == predecessor_sha256
                    and exact_json_equal(
                        rebound_predecessor[0], predecessor_value
                    ),
                    "attempt2 predecessor changed before authorization",
                )
                rebind_authorization_predecessors()
                authorization_transition.revalidate()
                authorization = _authorization_value(
                    submission_root=submission,
                    submission_sha256=submission_sha256,
                    contract=contract,
                    receipt=receipt,
                    receipt_map=receipt_map,
                    expected_reassembly=expected_reassembly,
                    source=source,
                    failure_sha256=failure_sha,
                    predecessor_sha256=predecessor_sha256,
                    calling_sha256=calling_sha,
                    submitted_sha256=submitted_sha,
                    job_id=job_id,
                    census=census,
                    job_control=job_control,
                    report_installation_method=report_installation_method,
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
                    predecessor_sha256=predecessor_sha256,
                    calling_sha256=calling_sha,
                    submitted_sha256=submitted_sha,
                    calling=calling,
                )
                authorization_sha = authorization_transition.seal_successor(
                    authorization_path, authorization
                )
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
                predecessor_sha256=predecessor_sha256,
                calling_sha256=calling_sha,
                submitted_sha256=submitted_sha,
                calling=calling,
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
            _require_repair_filesystem_namespace(
                submission, source_must_be_installed=True
            )
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


def execute_report_repair(
    repo_root: Path,
    submission_root: Path,
    submission_sha256: str,
    *,
    allow_initial_submission: bool,
    runner: Runner = _default_runner,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Advance attempt 2 under one retained authority and both locks.

    The transition is captured before original-contract/scientific validation
    and remains active through every census, accounting query, scheduler call,
    publication recovery, and terminal/successor append.  Nested helpers must
    reuse it; a later pathname capture can never bless drift.
    """

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
    require(
        submission_sha256 == EXPECTED_SUBMISSION_SHA256,
        "report repair submission SHA differs",
    )
    with _RepairLocks(submission) as locks:
        transition = _RepairTransitionBinding(
            submission,
            locks,
            source_must_be_installed=os.path.lexists(
                _repair_source_root(submission)
            ),
        )
        token = _ACTIVE_REPAIR_TRANSITION.set(transition)
        try:
            transition.revalidate()
            result = _execute_report_repair_impl(
                root,
                submission,
                submission_sha256,
                allow_initial_submission=allow_initial_submission,
                runner=runner,
                sleep=sleep,
                _bound_locks=locks,
            )
            transition.revalidate()
            return result
        finally:
            _ACTIVE_REPAIR_TRANSITION.reset(token)
            transition.close()


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
