#!/usr/bin/env python3
"""Fail-closed scientific and launch identities for the sealed Exp23 pilot."""

from __future__ import annotations

import argparse
import base64
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping
import zlib


PACKAGE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = PACKAGE_DIR.parents[1]
PACKAGE_RELATIVE = Path("experiments/23-treewm-executable-prefix-repair-pilot-v1")
MANIFEST_PATH = PACKAGE_DIR / "manifest.json"
WEIGHT_LOCK_PATH = PACKAGE_DIR / "weight_audit.lock.json"
PROTOCOL_LOCK_PATH = PACKAGE_DIR / "protocol.sha256"
PREFIX_TARGET_LOCK_PATH = PACKAGE_DIR / "prefix_target.lock.json"
RESOLVED_CONFIG_LOCK_PATH = PACKAGE_DIR / "resolved_config.lock.json"
CAUSAL_PARITY_LOCK_PATH = PACKAGE_DIR / "causal_parity.lock.json"
SETTINGS = (
    "antmaze-large",
    "scene",
    "puzzle-3x3",
    "puzzle-4x4-100m",
    "cube-quadruple-100m",
)
ARMS = ("GS", "GSEP")
SEEDS = (110, 111)
PREFIX_TERMS = (
    "executable_prefix_action",
    "executable_prefix_latent",
    "executable_prefix_endpoint",
)
WEIGHT_KEYS = tuple(f"losses.weights.{name}" for name in PREFIX_TERMS)
CAUSAL_AUDIT_MANIFEST_INPUT_KEYS = (
    "schema_version",
    "campaign_id",
    "method",
    "design",
    "arms",
    "causal_contrast",
    "weight_audit",
    "prefix_target_contract",
    "resolved_config_contract",
    "core_binding",
    "scientific_contract",
    "settings",
    "compatible_v2_recipe_input",
)
PROTOCOL_FILES = (
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
    "report_repair.py",
    "report_repair.slurm",
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
    "tests/test_report_repair.py",
    "tests/test_two_wave_canary.py",
)
SNAPSHOT_IMPORT_FILES = {
    "configs/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "scripts/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
CAMPAIGN_ID = "treewm-executable-prefix-repair-pilot-v1-launch8"
FAILED_CANARY_ATTEMPTS = [{
    "attempt": "canary1",
    "path": "canary1_negative_provenance.json",
    "raw_sha256": (
        "00f122373b4d7e37fba91c9f6bcc13f6a3c8374114ea30b8524aa80ae2acae20"
    ),
    "canonical_sha256": (
        "44fad8a37283ca276ba49ecade0dec7add0d10cf7c1f0e511d2e59fcdaba644c"
    ),
    "status": "terminal_negative_canary_provenance_frozen",
    "source_commit": "af348afdef0fa84f5e8ad4917d469d9729509f09",
    "source_protocol_sha256": (
        "b4403218d841667b6e68c715fa91cb53090670ca360d284e34d92b0a8763130f"
    ),
    "state_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/exp23-launch8-two-wave-canaries/"
        "exp23-launch8-two-wave-canary-af348af-b4403218"
    ),
    "canary_token": "e09ce7d5a0cef1b0",
    "job_ids_by_role": {
        "wave0": ["33285485"],
        "wave1": ["33285486"],
        "report": [],
    },
    "state_file_map_canonical_sha256": (
        "61662d488cf571f26362e3392e49f63239153ef443b515d7c79bcbf094d05648"
    ),
    "scheduler_terminal_rows": 2,
    "gpu_runtime_seconds": 0,
    "allocated_node_count": 0,
    "wave0_released": False,
    "authorization_published": False,
    "receipt_published": False,
    "report_job_submitted": False,
    "report_published": False,
    "active_scheduler_jobs_after_recovery": 0,
    "reuse_allowed": False,
    "resume_allowed": False,
    "retry_allowed": False,
    "recovery_allowed": False,
    "result_consumption_allowed": False,
}]
ACCEPTED_CANARY_ATTEMPTS = [{
    "attempt": "canary2",
    "path": "canary2_acceptance_provenance.json",
    "raw_sha256": (
        "7d1351f6d8fe900cd17698d79a5eff9d24166c1eaf50f08ab9a8ae1ea3a99fa7"
    ),
    "canonical_sha256": (
        "6ee4d943b92b69f8fd3a8a304d2226a7cee1e83140093d65b705250b073ff6e0"
    ),
    "status": "terminal_positive_canary_provenance_frozen",
    "source_commit": "b688ea652e99479ed5d8c6eccd6c137d77f9e03f",
    "source_protocol_sha256": (
        "79aaa2b6a9713034dc074c49ecbcf3daa17944c4f89287924fe13ddd3abc93d9"
    ),
    "state_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/exp23-launch8-two-wave-canaries/"
        "exp23-launch8-two-wave-canary-b688ea6-79aaa2b6"
    ),
    "canary_token": "b95869841048e511",
    "job_ids_by_role": {
        "wave0": ["33295657"],
        "wave1": ["33295659"],
        "report": ["33295661"],
    },
    "state_file_map_canonical_sha256": (
        "60dded3399d9b5406418209ffeb5020a09f22778e135c1ad79b3f413f9a7a2ce"
    ),
    "report_raw_sha256": (
        "1d4e00bf032f97434fdcabfe9c1f7601ca06786937b7c7d17c1d199b4a072dfd"
    ),
    "scheduler_terminal_rows": 3,
    "active_scheduler_jobs_after_terminal": 0,
    "topology_canary_passed": True,
    "production_authorization_prerequisite_satisfied": True,
    "reuse_allowed": False,
    "resume_allowed": False,
    "retry_allowed": False,
    "recovery_allowed": False,
    "checkpoint_consumption_as_scientific_state_allowed": False,
    "result_consumption_as_scientific_measurement_allowed": False,
}]
ACCEPTED_CANARY_CURRENT_SOURCE_SHA256 = {
    "submit.py": "51671551e331537dc31e87e11423bd6c66e1ce981b1dbb5b85ee48c028160dfa",
    "worker.py": "321cef217725178eda2a3c8750edd77f187631f1637614fe51fd511776ac6cd9",
    "report.py": "31df11e598f4d0da9ed7958c387d7777de617b532e435ff14d601eb0888f3a07",
    "dag_evidence.py": "b653ba6b2c25c017144cacb7b4b17ec5f5e153ad504eab47fba101d2c71dd1bb",
    "train_entry.py": "ac525211824eca20993f696a3e249d3c700b24eefdea48f365c8a997d64a9f33",
    "train.slurm": "fa66ce7d7dccf626ad434b6f87dfe284a904464701549a99a0dd6e94a695c44a",
    "report.slurm": "9930e95a9427d1fa55125387effcd1cdd65367b075b586855cd2d1ba34c1a6c9",
    "canary_worker.py": "4bcbaab866538c527f7a894d015d8b8bfbc505b2eb8e3596e56d419cfa4b3a2d",
    "canary_gpu.slurm": "cdfa25456ea26544450ff00681590adf1e81ef03eeb1434597c0777f6552ec14",
    "canary_report.slurm": "6d4561c7b8462fde2cd7b8666fa9f2b8d377aa30bfc44195bef6924548df1c0b",
    "two_wave_canary.py": "2f803ccaac8b4e42e821b884cfe817c7954a013648ebacd0c6fbf4411824bfe9",
}
ACCEPTED_CANARY_CONTROLLER_SHA256 = (
    "b373d64f410f08d1bd692dd9bb9e732a18b06bf80b6b483c313d82d4a6499436"
)
ACCEPTED_CANARY_CURRENT_CONTROLLER_SHA256 = (
    "2f803ccaac8b4e42e821b884cfe817c7954a013648ebacd0c6fbf4411824bfe9"
)
ACCEPTED_CANARY_POST_ACCEPTANCE_CHANGE_SCOPE = (
    "Post-acceptance package changes are limited to canary historical-identity, "
    "canonical-path, lock/rollback, and cleanup-only recovery guards in "
    "two_wave_canary.py, plus mandatory accepted-canary prerequisite and "
    "report-versus-cancel gating with cleanup-only legacy recovery in submit.py "
    "and report.py and matching prerequisite validation in worker.py and "
    "train_entry.py, plus a one-generation append-only terminal-report repair "
    "controller and a repair-only publication mode in report.py after the "
    "original Launch8 reporter failed during immutable publication; fresh canary "
    "DAG submission commands, scalar dependency "
    "forms, wave0 release payload, canary compute worker and Slurm scripts, and "
    "fresh scientific DAG submission commands and dependency forms remain "
    "unchanged"
)
TERMINAL_REPORT_REPAIR_POLICY = {
    "schema_version": 1,
    "status": "authorized_source_available_unexecuted",
    "scientific": False,
    "attempt": 1,
    "generation_count": 1,
    "default_action": "read-only --describe",
    "test_action": "--test-only",
    "explicit_submit_flag": "--submit-real-report-repair",
    "recovery_action": "--recover-or-cancel-report-repair",
    "confirmation_phrase": "SUBMIT_EXP23_LAUNCH8_REPORT_REPAIR_0001",
    "controller": "report_repair.py",
    "batch": "report_repair.slurm",
    "publisher": "report.py --publish-repair",
    "source_checkout_requirement": "active_controller_clean_origin_main",
    "production_submission_root_independent_of_source_checkout": True,
    "submission_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch8/"
        "state/submission"
    ),
    "submission_sha256": (
        "bbeaa71f8f37f22cbe74c16c68b733742e8a4366838812832180257d145f5418"
    ),
    "original_terminal_report": {
        "source_commit": "33122e15d0aaf3661893a4c853fd5ac49173c685",
        "package_protocol_sha256": (
            "2c0231b61197fe67790432c78a896272a55c3497a777d490598b53a6be67342f"
        ),
        "snapshot_inventory_sha256": (
            "9bff89010f792d1aed8b3b691567655daab8f83135d6421798b5efea29a2f2c5"
        ),
        "submission_authorization_sha256": (
            "371ae8df4add6338b98469eca6a287902cb69325dfda9d5be6ce5b1600e6fd55"
        ),
        "submission_receipt_sha256": (
            "58d1fd0f004efae049afd51e9592a79e963ba3fc8c2d3aae8a4af0bb7791a6a7"
        ),
        "report_calling_sha256": (
            "e0fd250dcd21fc7a0a62b5da0fe2c3d95401a24e6ad92c4e806082867e623047"
        ),
        "report_submitted_sha256": (
            "923f49755df3fcab99a547e0347b158ed42daef20cd640e24c605848b0769e57"
        ),
        "job_id": "33311218",
        "job_name": "exp23-launch8-bbeaa71f8f37f22c-report",
        "scheduler_comment": (
            "treewm-exp23:"
            "bbeaa71f8f37f22cbe74c16c68b733742e8a4366838812832180257d145f5418"
        ),
        "state": "FAILED",
        "exit_code": "2:0",
        "elapsed_raw": "355",
        "allocated_nodes": "1",
        "node_list": "cpu-00090",
        "start": "2026-08-29T08:28:49",
        "end": "2026-08-29T08:34:44",
        "log_path": "logs/report_33311218.out",
        "log_mode": 0o600,
        "log_size": 384,
        "log_sha256": (
            "2c5a23103e00fc07196886c62e7c9d069ed1b011fb9f44095a4242cc926e43a6"
        ),
        "terminal_scheduler_observation_required_before_submit": True,
        "active_original_or_repair_jobs_required": 0,
        "published_report_required_absent": True,
        "staging_and_cleanup_prefixes_required_absent": True,
    },
    "deterministic_reassembly": {
        "status": "rejected",
        "report_bundle_sha256": (
            "b9102090021c103fa2362663d1a51310d239d50223108dba0106758b199d9b83"
        ),
        "report_bundle_file_sha256": (
            "1a72e7968c5bc1639845eb18a64584db2204310c70c6301cdcccf804f576f139"
        ),
        "report_bundle_file_size": 424_013_704,
        "gate_sha256": (
            "d41b37f6806c77f15557ecd0329596da8385c02db5b06cecfb29247bb5f4682a"
        ),
        "gate_decision_file_sha256": (
            "53a7af1c91e4b09b8a04fdab7c1c0192d2076a88eb495855d9eafe39601f64b6"
        ),
        "gate_decision_file_size": 704_147,
        "original_provenance_v1_sha256": (
            "3fca5a3893cfd2e948f922438ee57bcc03e7763cfdb615500429700153820f77"
        ),
        "original_provenance_v1_file_sha256": (
            "3e99d102d6f5faa92699fb9bed4e1607e00a08349f03107048153c8d0764e858"
        ),
        "original_provenance_v1_file_size": 236_577,
        "worker_receipt_map_sha256": (
            "ab1ced2e9b736edede8e1353297682feb800865f03da0c25b681208ce7d8cfc8"
        ),
        "original_report_commit_payload_sha256": (
            "a52fd230482818d2d9bc52e2b0433d95b31fccd945c0a6ec95b8e9aa1834611c"
        ),
    },
    "repair_source_sha256": {
        "report.py": (
            "31df11e598f4d0da9ed7958c387d7777de617b532e435ff14d601eb0888f3a07"
        ),
        "report_repair.py": (
            "756d69eb8a9f32d8e9a9e9eb5e8fdc7706a954953a52900a7c120321c77fcd26"
        ),
        "report_repair.slurm": (
            "15ce6712f16c0655b4ad3d544987aec25574531cfc02b470c1ed9395bd363962"
        ),
    },
    "publication_contract": {
        "provenance_schema_version": 2,
        "publication_authority_required": True,
        "report_commit_schema_version": 1,
        "report_commit_exact_key_count": 14,
        "report_commit_last": True,
        "no_replace_publication": True,
        "sealed_json_mode": "0444",
        "report_directory_mode": "0555",
        "deterministic_reassembly_allowed": True,
        "scientific_input_change_allowed": False,
        "gate_change_allowed": False,
        "scientific_status_change_allowed": False,
        "scientific_bundle_schema_changed": False,
        "report_commit_schema_changed": False,
        "exp24_adapter_change_required": False,
    },
    "scheduler_protocol": {
        "submit_held": True,
        "settled_census_rounds": 3,
        "atomic_sealed_source_authority": True,
        "source_staging_authority_first_cleanup": True,
        "fresh_owner_wide_empty_census_before_submit_calling": True,
        "transaction_then_report_cancel_lock": True,
        "authorization_before_release": True,
        "historical_numeric_id_cleanup_only": True,
        "ambiguous_identity_cleanup_only": True,
        "broad_namespace_ambiguity_cleanup_only": True,
        "retry_after_terminal_repair_failure_allowed": False,
        "slurm_walltime_seconds": 14_400,
        "release_evidence_wait_seconds": 10_800,
        "minimum_assembly_budget_seconds": 3_600,
        "release_wait_clock": "time.monotonic",
        "release_poll_seconds": 0.25,
        "terminal_worker_accounting_required": True,
        "terminal_worker_failure_blocks_publication": True,
        "sealed_source_root_bound_in_sbatch_argv": True,
        "publisher_nofollow_same_fd_hash_before_exec": True,
    },
    "actual_repair_submit_performed": False,
}
SUPERSEDED_LAUNCHES = [{
    "campaign_id": "treewm-executable-prefix-repair-pilot-v1",
    "run_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1"
    ),
    "submission_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1/state/submission"
    ),
    "snapshot_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1/state/submission/"
        "source-snapshot/repo"
    ),
    "wandb_project": "treewm-executable-prefix-repair-pilot-v1",
    "status": "aborted_before_submission_contract",
    "source_commit": "85cd77de2d5956944008b4b2b16267858828fa84",
    "source_commit_claimed_by_journal": False,
    "source_commit_evidence": "independent_137_of_137_snapshot_file_byte_match",
    "proof_scope": (
        "The preserved journal does not record git provenance. The source commit is "
        "linked only by an independent comparison proving that all 137 sealed "
        "snapshot files match commit 85cd77de2d5956944008b4b2b16267858828fa84; "
        "the journal proves only its own claim, snapshot, and pre-contract abort records."
    ),
    "package_protocol_sha256": "3e39fb1e6501e3a31e360f569502eb92d1bbb0ad8093c7e747563e50665c2b6e",
    "manifest_canonical_sha256": "25790db3fe7a9a25c6de4f6b8224ccab33751817dc00f1bcf64d25c7fb497e4e",
    "manifest_raw_sha256": "bb841c5a9290465f864407a5d6a8ed927c907e9f4c2b07eeb58d23062a18d0db",
    "snapshot": {
        "inventory_sha256": "6767520819d42ef8866712023211b2f1bc8d236db3ffc836c8dae429b4e5b326",
        "file_count": 137,
        "independently_matched_files": 137,
        "all_files_match": True,
    },
    "claim_token": "0e5e1be5eace176f6a51ec3a3beb7e2579f6914699ac3080f9b2b9d10e4127e9",
    "scientific_output_fingerprint": "786beb527e80f37a8382059309858437df25ec867c5eb3c1e1b1fe1064b62cd4",
    "journal_sha256": {
        "0000_CLAIMED.json": "e9607ea26d07af65b670f2b70abceee9b3f45460159f28a76cd1ec6807a195d4",
        "0001_SNAPSHOT_SEALED.json": "94945e37ad3b363c04ab89c14230c06a62b30067d28f5c49df35854690de1439",
        "9998_OUTER_ABORTED.json": "27316555fd705a63bbd24521cadadf7d6d6b51b177d9c28622d654c99da16f02",
    },
    "submission_sha256": None,
    "known_job_ids": [],
    "submission_receipt_committed": False,
    "scientific_run_started": False,
    "checkpoint_created": False,
    "wandb_run_created": False,
    "optimizer_updates": 0,
    "results_consumed": False,
    "checkpoints_consumed": False,
    "reuse_allowed": False,
    "resume_allowed": False,
    "retry_allowed": False,
    "recovery_allowed": False,
}, {
    "campaign_id": "treewm-executable-prefix-repair-pilot-v1-launch2",
    "run_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch2"
    ),
    "submission_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch2/state/submission"
    ),
    "snapshot_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch2/state/"
        "submission/source-snapshot/repo"
    ),
    "wandb_project": "treewm-executable-prefix-repair-pilot-v1-launch2",
    "status": "aborted_before_submission_contract",
    "source_commit": "0fd89949a092bd9bbf12b16e3efb058850d50c86",
    "source_commit_claimed_by_journal": False,
    "source_commit_evidence": "independent_137_of_137_snapshot_file_byte_match",
    "proof_scope": (
        "The preserved journal does not record git provenance. Independent evidence "
        "proves that all 137 sealed snapshot files match commit "
        "0fd89949a092bd9bbf12b16e3efb058850d50c86 and that HEAD and origin/main "
        "equaled that commit with a clean worktree at capture; the journal proves "
        "only its own claim, snapshot, and pre-contract abort records."
    ),
    "package_protocol_sha256": "6472ca50fcbc1eaa35c4388876bf627f0f2c03d8310fd05e336f0204c0f49516",
    "manifest_canonical_sha256": "d124566d5834a62028f7416756c7cca36e6e63ae256e72d8c7c788412d558b00",
    "manifest_raw_sha256": "44911238bb06b10b46abbf58a8fe33019e0c107a6d760f8954a9a416382be776",
    "snapshot": {
        "inventory_sha256": "4aef86836e7fb683ace18cdd7588fd6b3904bdb9877e5c9a08146e76e49e2a76",
        "file_count": 137,
        "independently_matched_files": 137,
        "all_files_match": True,
    },
    "claim_token": "a2ea8575200dc47b4e3de67863a0f429d2397ae35fbbd2e7948e9322ffb64802",
    "scientific_output_fingerprint": "786beb527e80f37a8382059309858437df25ec867c5eb3c1e1b1fe1064b62cd4",
    "journal_sha256": {
        "0000_CLAIMED.json": "e353f69fb6a397d1095f3f5b81ca717a5887b6d4d55fa0961ca38faa3460b6dc",
        "0001_SNAPSHOT_SEALED.json": "be9a29c112f56dc0ace53847b99e3892ab964ee410ca2ecfc2a0fd9d39179bdc",
        "9998_OUTER_ABORTED.json": "48eed85ce12306a35927e3ac8b539be28dc52e107465fac9f5546c378c99cb99",
    },
    "submission_sha256": None,
    "known_job_ids": [],
    "submission_contract_committed": False,
    "submission_receipt_committed": False,
    "scientific_run_started": False,
    "checkpoint_created": False,
    "wandb_run_created": False,
    "optimizer_updates": 0,
    "results_consumed": False,
    "checkpoints_consumed": False,
    "reuse_allowed": False,
    "resume_allowed": False,
    "retry_allowed": False,
    "recovery_allowed": False,
}, {
    "campaign_id": "treewm-executable-prefix-repair-pilot-v1-launch3",
    "run_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch3"
    ),
    "submission_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch3/state/submission"
    ),
    "snapshot_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch3/state/"
        "submission/source-snapshot/repo"
    ),
    "wandb_project": "treewm-executable-prefix-repair-pilot-v1-launch3",
    "status": "aborted_after_contract_before_scheduler_submission",
    "source_commit": "ca979a2b0329d6775793cd8ce51d57a9200e6b8a",
    "source_commit_claimed_by_contract": True,
    "source_commit_claimed_by_journal": False,
    "source_commit_evidence": "contract_clean_head_origin_and_independent_137_of_137_snapshot_file_byte_match",
    "proof_scope": (
        "The sealed contract records clean HEAD and origin/main at commit "
        "ca979a2b0329d6775793cd8ce51d57a9200e6b8a, and all 137 snapshot files "
        "were independently byte-matched to that commit. The source order at that "
        "commit and the preserved abort journals prove failure in the first sanitized "
        "train-name squeue reconciliation before either sbatch call. Current exact-name "
        "squeue and sacct emptiness corroborates absence but is not used alone as a "
        "historical proof."
    ),
    "package_protocol_sha256": "6178ed54273d13c88fce750414131c98d002b394231618f11f6d8d6a1a3fb49a",
    "manifest_canonical_sha256": "7259a76924ac6f4566541a00f74923c67151619b36411048dd82ee20747a8d09",
    "manifest_raw_sha256": "92a9e17b78075805503907e4d9b71b732b54f5127da981ca17524feba650f74a",
    "snapshot": {
        "inventory_sha256": "7e237d31f9d49e3b55d0e0598c299b6064ab16b9caa77b62329c9bb8a2839eae",
        "file_count": 137,
        "independently_matched_files": 137,
        "all_files_match": True,
    },
    "preserved_tree": {
        "regular_file_count": 163,
        "symlink_count": 0,
        "snapshot_file_count": 137,
        "launch_file_count": 20,
        "contract_file_count": 1,
        "journal_file_count": 5,
    },
    "claim_token": "2741418c7e528a0b64b8115cafa46cfac391abd53312cf29b0ffc8a4e1afca4d",
    "scientific_output_fingerprint": "786beb527e80f37a8382059309858437df25ec867c5eb3c1e1b1fe1064b62cd4",
    "contract_sha256": "0cd594c8a49499b5e3d10a09ddbf3b89f981264be67bb603dc64836568a1b4c2",
    "submission_sha256": "0cd594c8a49499b5e3d10a09ddbf3b89f981264be67bb603dc64836568a1b4c2",
    "journal_sha256": {
        "0000_CLAIMED.json": "25d607cef5aaf49e932e86a56c2272d37be6936010f089f8e7230bb44166be28",
        "0001_SNAPSHOT_SEALED.json": "d4acf9fb3fa6af98adedf45abd0269d1e491abd48b3cb992509b7641ad05b4c6",
        "0002_CONTRACT_SEALED.json": "3936a8717d096a93cf7d0eb3dea5293e32c14cdd0b986dd7a738f9501a92a044",
        "9999_ABORTED.json": "915dd5a3869a6784f3eb9d8e0d564a16b96fb17689c3d31f5a2e21431365199a",
        "9998_OUTER_ABORTED.json": "e65e9b39d268c2497e98e1d66b0cadd7f80b0f53c98a243057cc03ca399b47b0",
    },
    "failure_phase": "first_sanitized_squeue_before_any_sbatch",
    "no_job_proof": {
        "scheduler_job_names": {
            "train": "exp23-launch3-0cd594c8a49499b5-train",
            "report": "exp23-launch3-0cd594c8a49499b5-report",
        },
        "source_order_at_commit": {
            "contract_sealed_first": True,
            "train_absence_check_line": 3112,
            "report_absence_check_line": 3113,
            "first_sbatch_call_line": 3114,
            "recorded_failure": "first_train_absence_check",
        },
        "preserved_journal_known_job_ids": [],
        "preserved_journal_job_ids_by_role": {"train": [], "report": []},
        "current_scheduler_observation": {
            "environment": {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "LANG": "C",
                "LC_ALL": "C",
                "SLURM_CONF": "/cm/shared/apps/slurm/var/etc/cs-oci-ord/slurm.conf",
            },
            "history_start_utc": "2026-08-01",
            "squeue_command": [
                "/usr/local/bin/squeue", "-h", "-n",
                "exp23-launch3-0cd594c8a49499b5-train,exp23-launch3-0cd594c8a49499b5-report",
                "-o", "%i|%j|%T|%k",
            ],
            "squeue_matching_rows": 0,
            "sacct_command": [
                "/usr/local/bin/sacct", "-X", "-n", "-S", "2026-08-01",
                "--name",
                "exp23-launch3-0cd594c8a49499b5-train,exp23-launch3-0cd594c8a49499b5-report",
                "-o", "JobIDRaw,JobName,State,Comment",
            ],
            "sacct_matching_rows": 0,
        },
    },
    "actual_sbatch_calls": 0,
    "known_job_ids": [],
    "job_ids_by_role": {"train": [], "report": []},
    "submission_contract_committed": True,
    "submission_receipt_committed": False,
    "scientific_run_started": False,
    "checkpoint_created": False,
    "wandb_run_created": False,
    "optimizer_updates": 0,
    "results_consumed": False,
    "checkpoints_consumed": False,
    "reuse_allowed": False,
    "resume_allowed": False,
    "retry_allowed": False,
    "recovery_allowed": False,
}, {
    "campaign_id": "treewm-executable-prefix-repair-pilot-v1-launch4",
    "run_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch4"
    ),
    "submission_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch4/state/submission"
    ),
    "snapshot_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch4/state/"
        "submission/source-snapshot/repo"
    ),
    "transaction_lock": "outputs/.exp23-d3765ecc9f5b5f7a.transaction.lock",
    "wandb_project": "treewm-executable-prefix-repair-pilot-v1-launch4",
    "status": "aborted_after_scheduler_submission_before_any_job_runtime",
    "failure_phase": "canonical_array_dependency_validation_after_report_submission",
    "canonical_array_dependency_validator_error": (
        "SubmissionError('accepted report dependency differs')"
    ),
    "source_commit": "62fbf4631e950187506293138f13be691df1fa37",
    "source_commit_claimed_by_contract": True,
    "source_commit_claimed_by_journal": False,
    "source_commit_evidence": (
        "contract_clean_head_origin_and_independent_137_of_137_snapshot_file_byte_match"
    ),
    "proof_scope": (
        "The sealed contract records clean HEAD and origin/main at commit "
        "62fbf4631e950187506293138f13be691df1fa37, and all 137 snapshot files were "
        "independently byte-matched to that commit. Durable contract and journal bytes "
        "prove both scheduler IDs were accepted, the exact submit arguments, the "
        "canonical dependency validator exception, cancellation, and the absence of a "
        "receipt. The canonical report dependency afterok:33211846_*(unfulfilled) and "
        "KillOInInvalidDependent=Yes were observed live immediately after the abort and "
        "before slurmctld purged the records; that unsealed observation is corroborative "
        "and is not claimed as preserved journal stdout. Current exact-ID squeue absence "
        "from the active queue establishes present absence; current sacct rows corroborate "
        "zero-runtime cancellation. Neither is used alone as historical proof."
    ),
    "package_protocol_sha256": "a838d23a396439dac585a1d4fe72f89b385df1e432f5795d85c5d7a2d818c02b",
    "manifest_canonical_sha256": "7e1130dcc0f781c21e74a323880699e23ae6778d1752c64fe38cdb31a64aa7f8",
    "manifest_raw_sha256": "72342c585b3988df3d410131d33705e06b1eaf99494f027dea3830adb7326534",
    "snapshot": {
        "inventory_sha256": "1a4e42ee751964ab704d2fae6f736862d46174d3eafd1ffe42b4f4f018cf1cbb",
        "file_count": 137,
        "independently_matched_files": 137,
        "all_files_match": True,
    },
    "preserved_tree": {
        "regular_file_count": 164,
        "symlink_count": 0,
        "snapshot_file_count": 137,
        "launch_file_count": 20,
        "contract_file_count": 1,
        "journal_file_count": 6,
        "log_file_count": 0,
        "aggregate_schema_version": 1,
        "aggregate_algorithm": (
            "sha256(json.dumps({schema_version:1,files:{relative_posix_path:"
            "raw_file_sha256}},sort_keys=True,separators=(',',':')).encode('utf-8'))"
        ),
        "aggregate_sha256": "d67768c00795e209a0b1998058cd475360b98cd3aab331b416d1b6934142adb5",
    },
    "claim_token": "4c61d6fffc30ed2861ba6d3aaefb94ab6535616c9a6ce63bf9a097d4aa6162a9",
    "transaction_lock_state": {
        "regular_file": True,
        "symlink": False,
        "mode": "0600",
        "size": 0,
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    },
    "scientific_output_fingerprint": "786beb527e80f37a8382059309858437df25ec867c5eb3c1e1b1fe1064b62cd4",
    "contract_sha256": "0aa63e5787fbdb06331265f03dd5e1aa32c32c32bb9b74728cd3060be7200336",
    "submission_sha256": "0aa63e5787fbdb06331265f03dd5e1aa32c32c32bb9b74728cd3060be7200336",
    "journal_sha256": {
        "0000_CLAIMED.json": "ce5361cf48b00b61e3ae7d12d1e06fb50a2429ecafad3f9453f7c38e1b1c594c",
        "0001_SNAPSHOT_SEALED.json": "a150558bbe7ff1c39caff10f8a48329f42142230bb6f1e1862721111b3505efa",
        "0002_CONTRACT_SEALED.json": "6b6b375e539bf64ceb3bccf2e3f4533b1913ed455a20b0db2ddfd36ee9727c3b",
        "0003_TRAIN_SUBMITTED.json": "196ad9be4302d6d1262914a8d42aaa0201d0b3c8f3a1d1496c1db5614fe6c271",
        "9999_ABORTED.json": "d73201fb9c7a6f89afacdf057613016a85c303ffd5ec2320972a6813ca524701",
        "9998_OUTER_ABORTED.json": "458285c797d33534deb5250cc4766a5a4f383afc3f195165cca46d8648d265d9",
    },
    "durable_failure_evidence": {
        "journal_error": "SubmissionError('accepted report dependency differs')",
        "outer_journal_error": (
            "SchedulerSubmissionError(\"submission transaction aborted: "
            "SubmissionError('accepted report dependency differs')\")"
        ),
        "report_submit_dependency_argument": "--dependency=afterok:33211846",
        "report_submit_kill_argument": "--kill-on-invalid-dep=yes",
        "scontrol_command_preserved": True,
        "scontrol_stdout_preserved": False,
    },
    "unsealed_time_bounded_live_observation": {
        "provenance": "independent_operator_observation_immediately_after_abort_before_slurmctld_purge",
        "preserved_in_launch4_bytes": False,
        "canonical_report_dependency": "afterok:33211846_*(unfulfilled)",
        "kill_on_invalid_dependent": "Yes",
    },
    "scheduler_history": {
        "environment": {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "SLURM_CONF": "/cm/shared/apps/slurm/var/etc/cs-oci-ord/slurm.conf",
        },
        "history_start_utc": "2026-08-01",
        "observation_date_utc": "2026-08-28",
        "squeue_command": [
            "/usr/local/bin/squeue", "-h", "-j", "33211846,33211848",
            "-o", "%A|%a|%j|%u|%T|%k|%R",
        ],
        "squeue_matching_rows": 0,
        "sacct_command": [
            "/usr/local/bin/sacct", "-X", "-n", "-P", "-S", "2026-08-01",
            "-j", "33211846,33211848",
            "-o", (
                "JobIDRaw,JobName%100,State,ElapsedRaw,AllocNodes,NodeList%100,"
                "Submit,Start,End,ExitCode,DerivedExitCode,Reason%100"
            ),
        ],
        "sacct_rows": [{
            "job_id": "33211848",
            "job_name": "exp23-launch4-0aa63e5787fbdb06-report",
            "state": "CANCELLED",
            "raw_state": "CANCELLED by 147230",
            "elapsed_raw": 0,
            "allocated_nodes": 0,
            "node_list": "None assigned",
            "submitted_at": "2026-08-28T02:14:39",
            "started_at": "2026-08-28T02:14:40",
            "ended_at": "2026-08-28T02:14:40",
            "exit_code": "0:0",
            "derived_exit_code": "0:0",
            "reason": "None",
        }, {
            "job_id": "33211846",
            "job_name": "exp23-launch4-0aa63e5787fbdb06-train",
            "state": "CANCELLED",
            "raw_state": "CANCELLED by 147230",
            "elapsed_raw": 0,
            "allocated_nodes": 0,
            "node_list": "None assigned",
            "submitted_at": "2026-08-28T02:14:39",
            "started_at": "2026-08-28T02:14:40",
            "ended_at": "2026-08-28T02:14:40",
            "exit_code": "0:0",
            "derived_exit_code": "0:0",
            "reason": "None",
        }],
        "top_level_row_count": 2,
        "array_task_row_count": 0,
    },
    "actual_sbatch_calls": 2,
    "known_job_ids": ["33211846", "33211848"],
    "job_ids_by_role": {"train": ["33211846"], "report": ["33211848"]},
    "submission_contract_committed": True,
    "submission_receipt_committed": False,
    "train_submission_journal_committed": True,
    "report_submission_journal_committed": False,
    "ready_marker_committed": False,
    "report_journal_committed": False,
    "jobs_cancelled_before_runtime": True,
    "scientific_run_started": False,
    "checkpoint_created": False,
    "wandb_run_created": False,
    "optimizer_updates": 0,
    "results_consumed": False,
    "checkpoints_consumed": False,
    "reuse_allowed": False,
    "resume_allowed": False,
    "retry_allowed": False,
    "recovery_allowed": False,
}, {
    "campaign_id": "treewm-executable-prefix-repair-pilot-v1-launch5",
    "run_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch5"
    ),
    "submission_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch5/state/submission"
    ),
    "snapshot_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch5/state/"
        "submission/source-snapshot/repo"
    ),
    "transaction_lock": "outputs/.exp23-9066d1c600046ae2.transaction.lock",
    "wandb_project": "treewm-executable-prefix-repair-pilot-v1-launch5",
    "status": "cancelled_after_trainer_bootstrap_failure_before_hydra_composition",
    "failure_phase": "sealed_train_entry_import_of_hydra_relative_config_package",
    "failure_error": (
        "Primary config module 'configs' not found. Check that it's correct and "
        "contains an __init__.py file"
    ),
    "source_commit": "332a26f2f88e627f842eebbfc8310978ad606898",
    "source_commit_claimed_by_contract": True,
    "source_commit_claimed_by_journal": False,
    "source_commit_evidence": (
        "contract_clean_head_origin_and_independent_137_of_137_snapshot_file_byte_match"
    ),
    "proof_scope": (
        "The sealed contract records clean HEAD and origin/main at commit "
        "332a26f2f88e627f842eebbfc8310978ad606898, and all 137 snapshot files "
        "were independently byte-matched to that commit. Twelve array tasks entered "
        "the real sealed train_entry bridge and emitted byte-identical Hydra config-"
        "package failures before config composition, model construction, or an optimizer "
        "update. Cancellation was then sealed and exact-ID scancel returned zero. A "
        "later unsealed exact-ID sacct/squeue observation showed that eight remaining tasks "
        "and the reporter were cancelled, accounted for all twenty tasks plus the "
        "reporter, and corroborated terminal absence; its raw stdout was not preserved in launch5 "
        "bytes and is not durable launch-journal proof. Empty run directories are "
        "scheduler/bootstrap residue, not scientific artifacts. No result or checkpoint "
        "from this attempt may be consumed."
    ),
    "package_protocol_sha256": "e9ac9f39e9261ca6ab0dcd5aadeba3dd3eb4ec25c999846c55fc09b2168c62a7",
    "manifest_canonical_sha256": "80665a573a6f7d19b63adb0064d5dd6fd98c7af37a64e6f974cfbe20d92e158e",
    "manifest_raw_sha256": "f4b72bfd77599be2bd00656e518eec8fc21ec0574e743232bca113f4d55f8321",
    "snapshot": {
        "inventory_sha256": "1a74e2356af32a68aba6f6cde78a262c96a2334976f9e1fcab00cd2115ee188e",
        "file_count": 137,
        "independently_matched_files": 137,
        "all_files_match": True,
    },
    "preserved_tree": {
        "regular_file_count": 235,
        "symlink_count": 0,
        "snapshot_file_count": 137,
        "launch_file_count": 20,
        "contract_file_count": 1,
        "receipt_file_count": 1,
        "journal_file_count": 6,
        "task_file_count": 53,
        "log_file_count": 13,
        "cancellation_file_count": 2,
        "aggregate_schema_version": 1,
        "aggregate_algorithm": (
            "sha256(json.dumps({schema_version:1,files:{relative_posix_path:"
            "raw_file_sha256}},sort_keys=True,separators=(',',':')).encode('utf-8'))"
        ),
        "preserved_root_path_base": "run_root",
        "preserved_root_aggregate_canonical_json_bytes": 32248,
        "preserved_root_aggregate_sha256": (
            "c07dce9aa58352f790af94bff8c719a3e9c8639bdd268d5b7d33824db8b7a874"
        ),
        "submission_root_path_base": "submission_root",
        "submission_root_aggregate_canonical_json_bytes": 28253,
        "submission_root_aggregate_sha256": (
            "1c1082dd6a9f43220a1de15eaafcb5d947dfcf88f49f9415633ad2865c3d36d4"
        ),
        "task_root_path_base": "submission_root/tasks",
        "task_root_aggregate_canonical_json_bytes": 5406,
        "task_root_aggregate_sha256": (
            "31d41fd81ee8092c0ebdfc68e9cd5199e81698fb62a7a2a88e5f2c2a6f5666f5"
        ),
    },
    "claim_token": "aded842d0b386521cfe8d99791f4f42c5d4f4396c7f6537d6367780caaf1481e",
    "transaction_lock_state": {
        "regular_file": True,
        "symlink": False,
        "mode": "0600",
        "size": 0,
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    },
    "scientific_output_fingerprint": "786beb527e80f37a8382059309858437df25ec867c5eb3c1e1b1fe1064b62cd4",
    "contract_sha256": "8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
    "submission_sha256": "8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
    "receipt_sha256": "463397088705144887fa8c75d6b40f3e770dca3f891d818e4178c9351672fbd5",
    "journal_sha256": {
        "0000_CLAIMED.json": "1dc871202751f498654c3eafa79e78d31d707603a631c612196943d7de268929",
        "0001_SNAPSHOT_SEALED.json": "d5ec982cc057878eb542433e568693d2d7e73c51bacab20f75545e98f6269368",
        "0002_CONTRACT_SEALED.json": "ac15380a94b4e61de8e5b50c0d0ba834f229d3cef03f7aca86493e350eb85da7",
        "0003_TRAIN_SUBMITTED.json": "b872b9eb574de46320d5dce114ec00a076e15d6f4b02aaf802eaacc3c595819d",
        "0004_REPORT_SUBMITTED.json": "b3add73cb036b8b1a91d834e89d48d0ead4503acd25a3cfdcb401deddbb846b2",
        "0005_READY_TO_COMMIT.json": "d05b39b204bbdefb8846e7074725eba7f85b674dbb81a57c4c72b4933ddcfa84",
    },
    "scheduler_submission": {
        "train_array_job_id": "33217168",
        "report_job_id": "33217171",
        "requested_dependency": "afterok:33217168",
        "accepted_dependency": "afterok:33217168_*(unfulfilled)",
        "kill_on_invalid_dependency": "Yes",
        "actual_sbatch_calls": 2,
    },
    "cancellation": {
        "latch_sha256": "b28ca9e23ffdbb523f3c11f4cea686388563a3e3aefec5c491a97a5e86ff972a",
        "call_token": "1787888274165397643-1491782",
        "call_sha256": "6628089da0e20c4176b3eea9b106d732913326e3df9138bdbda1f9c8085bb13e",
        "result_sha256": "68c09e9031c668a8dff337a171c9ca2fe3f962efd76297a84ed8ae47b9a44108",
        "command": ["/usr/local/bin/scancel", "33217168", "33217171"],
        "returncode": 0,
        "scheduler_calls": 1,
    },
    "failure_logs": {
        "deterministic_failed_cell_indices": list(range(12)),
        "deterministic_log_count": 12,
        "byte_identical": True,
        "raw_log_sha256": "4a624c03f806664ce70d0b98af2b8ea3e6f61a24ef4160357348441aa93b405b",
        "cancellation_only_cell_indices": [17],
        "cancellation_only_log_sha256": "01455d9b4b80863a529f3f27e08b7b6553c2be1bce041721c6ab520295313b90",
        "no_log_cell_indices": [12, 13, 14, 15, 16, 18, 19],
    },
    "unsealed_later_terminal_scheduler_observation": {
        "provenance": (
            "independent_operator_observation_after_exact_cancellation_reached_terminal_state"
        ),
        "observation_utc": "2026-08-28T04:16:52Z",
        "preserved_in_launch5_bytes": False,
        "evidence_policy": (
            "corroborative_current_state_only_not_durable_launch5_journal_proof"
        ),
        "environment": {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "PYTHONHASHSEED": "0",
            "SLURM_CONF": "/cm/shared/apps/slurm/var/etc/cs-oci-ord/slurm.conf",
        },
        "sacct_command": [
            "/usr/local/bin/sacct", "-X", "-n", "-P", "-S", "2026-08-28",
            "-j", "33217168,33217171", "-o",
            "JobID,JobIDRaw,JobName%100,State,ElapsedRaw,AllocNodes,NodeList%100,"
            "Submit,Start,End,ExitCode,DerivedExitCode,Reason%100,Comment%150",
        ],
        "columns": [
            "JobID", "JobIDRaw", "JobName%100", "State", "ElapsedRaw",
            "AllocNodes", "NodeList%100", "Submit", "Start", "End", "ExitCode",
            "DerivedExitCode", "Reason%100", "Comment%150",
        ],
        "sacct_raw_stdout_bytes": 5116,
        "sacct_raw_stdout_sha256": "4a741ee0ece1e84644bf1628ba2666e2e35d3e2c906fdd643a8beaccedff7429",
        "canonical_ledger_algorithm": (
            "sha256(json.dumps({schema_version:1,columns:columns,rows:[row.split('|') "
            "for row in serialized_rows]},sort_keys=True,separators=(',',':'),"
            "ensure_ascii=True,allow_nan=False).encode('ascii'))"
        ),
        "canonical_ledger_sha256": "14e77c0964e2c42e0ef6d1be17926751da6fc0f1d8a2615e240ceba51d5a5061",
        "serialized_rows": [
            "33217168_0|33217200|exp23-launch5-8848790ca118a2fb-train|FAILED|28|1|batch-block5-00642|2026-08-28T03:36:17|2026-08-28T03:36:45|2026-08-28T03:37:13|2:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
            "33217168_1|33217201|exp23-launch5-8848790ca118a2fb-train|FAILED|26|1|batch-block5-00642|2026-08-28T03:36:17|2026-08-28T03:36:45|2026-08-28T03:37:11|2:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
            "33217168_2|33217202|exp23-launch5-8848790ca118a2fb-train|FAILED|26|1|batch-block5-00642|2026-08-28T03:36:17|2026-08-28T03:36:45|2026-08-28T03:37:11|2:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
            "33217168_3|33217203|exp23-launch5-8848790ca118a2fb-train|FAILED|26|1|batch-block5-00642|2026-08-28T03:36:17|2026-08-28T03:36:45|2026-08-28T03:37:11|2:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
            "33217168_4|33217204|exp23-launch5-8848790ca118a2fb-train|FAILED|26|1|batch-block5-00642|2026-08-28T03:36:17|2026-08-28T03:36:45|2026-08-28T03:37:11|2:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
            "33217168_5|33217205|exp23-launch5-8848790ca118a2fb-train|FAILED|30|1|batch-block5-00642|2026-08-28T03:36:17|2026-08-28T03:36:45|2026-08-28T03:37:15|2:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
            "33217168_6|33217206|exp23-launch5-8848790ca118a2fb-train|FAILED|26|1|batch-block5-00642|2026-08-28T03:36:17|2026-08-28T03:36:45|2026-08-28T03:37:11|2:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
            "33217168_7|33217207|exp23-launch5-8848790ca118a2fb-train|FAILED|26|1|batch-block5-01951|2026-08-28T03:36:17|2026-08-28T03:36:45|2026-08-28T03:37:11|2:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
            "33217168_8|33217208|exp23-launch5-8848790ca118a2fb-train|FAILED|26|1|batch-block5-01951|2026-08-28T03:36:17|2026-08-28T03:36:45|2026-08-28T03:37:11|2:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
            "33217168_9|33217209|exp23-launch5-8848790ca118a2fb-train|FAILED|25|1|batch-block5-03415|2026-08-28T03:36:17|2026-08-28T03:36:45|2026-08-28T03:37:10|2:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
            "33217168_10|33217210|exp23-launch5-8848790ca118a2fb-train|FAILED|25|1|batch-block5-03415|2026-08-28T03:36:17|2026-08-28T03:36:45|2026-08-28T03:37:10|2:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
            "33217168_11|33217211|exp23-launch5-8848790ca118a2fb-train|FAILED|25|1|batch-block5-04017|2026-08-28T03:36:17|2026-08-28T03:36:45|2026-08-28T03:37:10|2:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
            "33217168_12|33217258|exp23-launch5-8848790ca118a2fb-train|CANCELLED by 147230|9|1|batch-block5-03951|2026-08-28T03:36:17|2026-08-28T03:37:47|2026-08-28T03:37:56|0:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
            "33217168_13|33217259|exp23-launch5-8848790ca118a2fb-train|CANCELLED by 147230|9|1|batch-block5-00642|2026-08-28T03:36:17|2026-08-28T03:37:47|2026-08-28T03:37:56|0:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
            "33217168_14|33217260|exp23-launch5-8848790ca118a2fb-train|CANCELLED by 147230|9|1|batch-block5-00642|2026-08-28T03:36:17|2026-08-28T03:37:47|2026-08-28T03:37:56|0:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
            "33217168_15|33217261|exp23-launch5-8848790ca118a2fb-train|CANCELLED by 147230|9|1|batch-block5-00642|2026-08-28T03:36:17|2026-08-28T03:37:47|2026-08-28T03:37:56|0:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
            "33217168_16|33217262|exp23-launch5-8848790ca118a2fb-train|CANCELLED by 147230|9|1|batch-block5-00642|2026-08-28T03:36:17|2026-08-28T03:37:47|2026-08-28T03:37:56|0:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
            "33217168_17|33217263|exp23-launch5-8848790ca118a2fb-train|CANCELLED by 147230|9|1|batch-block5-00642|2026-08-28T03:36:17|2026-08-28T03:37:47|2026-08-28T03:37:56|0:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
            "33217168_18|33217264|exp23-launch5-8848790ca118a2fb-train|CANCELLED by 147230|9|1|batch-block5-00642|2026-08-28T03:36:17|2026-08-28T03:37:47|2026-08-28T03:37:56|0:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
            "33217168_19|33217168|exp23-launch5-8848790ca118a2fb-train|CANCELLED by 147230|9|1|batch-block5-00642|2026-08-28T03:36:17|2026-08-28T03:37:47|2026-08-28T03:37:56|0:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
            "33217171|33217171|exp23-launch5-8848790ca118a2fb-report|CANCELLED by 147230|0|0|None assigned|2026-08-28T03:36:18|None|2026-08-28T03:37:56|0:0|0:0|None|treewm-exp23:8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe",
        ],
        "sacct_row_count": 21,
        "failed_task_indices": list(range(12)),
        "cancelled_task_indices": list(range(12, 20)),
        "report_state": "CANCELLED",
        "squeue_command": [
            "/usr/local/bin/squeue", "--array", "--noheader",
            "--jobs=33217168,33217171", "--format=%i|%T|%R|%N",
        ],
        "squeue_raw_stdout_bytes": 0,
        "squeue_raw_stdout_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        "squeue_row_count": 0,
    },
    "scientific_state": {
        "worker_started_cell_count": 13,
        "trainer_bridge_started_cell_count": 12,
        "hydra_composition_completed": False,
        "model_constructed": False,
        "optimizer_updates": 0,
        "empty_run_directory_count": 12,
        "checkpoint_files": 0,
        "result_files": 0,
        "wandb_files": 0,
        "report_files": 0,
        "submission_ready_to_commit_journal_0005_committed": True,
        "scheduler_report_submitted_journal_0004_committed": True,
        "scientific_ready_marker_committed": False,
        "scientific_report_bundle_committed": False,
        "results_consumed": False,
        "checkpoints_consumed": False,
    },
    "known_job_ids": ["33217168", "33217171"],
    "job_ids_by_role": {"train": ["33217168"], "report": ["33217171"]},
    "submission_contract_committed": True,
    "submission_receipt_committed": True,
    "cancel_latch_committed": True,
    "scientific_run_completed": False,
    "reuse_allowed": False,
    "resume_allowed": False,
    "retry_allowed": False,
    "recovery_allowed": False,
}, {
    "campaign_id": "treewm-executable-prefix-repair-pilot-v1-launch6",
    "run_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch6"
    ),
    "submission_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch6/"
        "state/submission"
    ),
    "snapshot_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch6/"
        "state/submission/source-snapshot/repo"
    ),
    "transaction_lock": "outputs/.exp23-34d79ab13d65ef27.transaction.lock",
    "wandb_project": "treewm-executable-prefix-repair-pilot-v1-launch6",
    "status": "cancelled_after_retrospective_tensorboard_scalar_identity_failure",
    "failure_phase": "retrospective_5000_boundary_append_only_scalar_audit",
    "failure_error": (
        "The immutable Launch6 event streams contain 718 repeated (cell,tag,step) "
        "groups, including 560 conflict-classified groups, so no single-valued "
        "append-only report can be assembled."
    ),
    "source_commit": "d09e842acaf7909edf4ea6ba29138ea5c646fc1a",
    "source_commit_claimed_by_contract": True,
    "source_commit_claimed_by_journal": False,
    "source_commit_evidence": (
        "contract_clean_head_origin_and_independent_138_of_138_snapshot_file_byte_match"
    ),
    "proof_scope": (
        "The sealed contract records clean HEAD and origin/main at commit "
        "d09e842acaf7909edf4ea6ba29138ea5c646fc1a, and all 138 snapshot files were "
        "independently byte-matched to that commit. All twenty cells entered the "
        "sealed trainer from a generation-zero scratch start and performed partial "
        "training before an operator's retrospective 5k audit found irreparable "
        "same-tag/same-step scalar conflicts. Durable cancellation bytes prove one "
        "exact-ID scancel call returned zero. The later sacct/squeue ledger and scalar "
        "census are independent observations serialized only into Launch7 negative "
        "provenance, not records originally sealed inside Launch6. No Launch6 result, "
        "checkpoint, W&B identity, optimizer state, or namespace may be consumed."
    ),
    "package_protocol_sha256": (
        "33288668441622bb30b205c98a0373e96f2c11f5ec5ba0e76bd5255098a8b7bd"
    ),
    "manifest_canonical_sha256": (
        "ea35d1e2f36179b99ce950036ed3a8673eb7cc5ed91fa7c022cb1f6c9b03d442"
    ),
    "manifest_raw_sha256": (
        "a621177815c69bb2685c84eee13804cedd5b896c281e25c022310818047078b2"
    ),
    "snapshot": {
        "inventory_sha256": (
            "7982b2b5d9470f4bab4b16c59bc7acd42cb8eebc7905a3a0af404e471e197819"
        ),
        "file_count": 138,
        "independently_matched_files": 138,
        "all_files_match": True,
    },
    "canonical_evidence_serialization": (
        "json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,"
        "allow_nan=False).encode('ascii'); SHA-256 is over those exact bytes"
    ),
    "preserved_tree": {
        "aggregate_schema_version": 1,
        "aggregate_algorithm": (
            "sha256(json.dumps({schema_version:1,files:{relative_posix_path:"
            "raw_file_sha256}},sort_keys=True,separators=(',',':'),"
            "ensure_ascii=True,allow_nan=False).encode('utf-8'))"
        ),
        "run_root": {
            "directory_count_including_root": 297,
            "regular_file_count": 629,
            "regular_file_bytes": 2009101434,
            "symlink_count": 80,
            "special_file_count": 0,
            "canonical_json_bytes": 101693,
            "aggregate_sha256": (
                "0ba872f7a03f42a58dfd9dcbc55afef0ec94dba6bef66744845c12f55cd340a8"
            ),
        },
        "submission_root": {
            "directory_count_including_root": 85,
            "regular_file_count": 309,
            "regular_file_bytes": 13694950,
            "symlink_count": 0,
            "special_file_count": 0,
            "canonical_json_bytes": 36226,
            "aggregate_sha256": (
                "98d131fbd80b46d34e978b9b23f7c0137bc07711cc32b35bd89877a49f8c0242"
            ),
        },
        "task_root": {
            "directory_count_including_root": 61,
            "regular_file_count": 119,
            "regular_file_bytes": 369713,
            "symlink_count": 0,
            "special_file_count": 0,
            "canonical_json_bytes": 12201,
            "aggregate_sha256": (
                "07217ed94c148f7751c46c776b114c593f6b92a17d79c45aceeaa55fa0895ec4"
            ),
        },
        "symlink_target_envelope": {
            "schema_version": 1,
            "policy": "raw readlink target text; links are never followed",
            "canonical_payload_schema": (
                "{schema_version:1,symlinks:{relative_posix_path:"
                "raw_readlink_target}}"
            ),
            "symlink_count": 80,
            "wandb_leaf_links_per_run": 4,
            "canonical_json_bytes": 13357,
            "aggregate_sha256": (
                "fb1ef56fa6e93ae69f01e54438a377c36e9978ab64af8232f9f109582fecbfb4"
            ),
        },
    },
    "claim_token": "45b8bb00c3ea0fd7fcb763cad34ce7cdf7300ca64638e7db3ce0387c3cc3f681",
    "transaction_lock_state": {
        "regular_file": True,
        "symlink": False,
        "mode": "0600",
        "size": 0,
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    },
    "scientific_output_fingerprint_at_claim": (
        "786beb527e80f37a8382059309858437df25ec867c5eb3c1e1b1fe1064b62cd4"
    ),
    "contract_sha256": "e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
    "submission_sha256": "e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
    "receipt_sha256": "ab0574d776dec229420e207752abc257cbe3f17115def2061a1248d22d12a110",
    "journal_sha256": {
        "0000_CLAIMED.json": "6f96d1fa8e6562e459cce1339a09bdbabbba8ca94e33720d5f1b007abfa44e62",
        "0001_SNAPSHOT_SEALED.json": "eea22f3b79588783d13ac0ca65891d84ba49a8830d658c537795cee9a9ffe837",
        "0002_CONTRACT_SEALED.json": "c13c34f7d65dd606741675631087b9a0bd75ee849cea5c53d42c047eaa2be5f3",
        "0003_TRAIN_SUBMITTED.json": "cc9b931f3053513723e816ebfce8e727dac63692283486aa3c063b045906c2d8",
        "0004_REPORT_SUBMITTED.json": "10e704d8ea8c34b50d9e4ae9bfaaae54e81c49258357eadc8d4919b11c9fd1b5",
        "0005_READY_TO_COMMIT.json": "d669d13504b140f68e4a536defd4bc3fa99dd3ec2158e8d9176a84b10c522b09",
    },
    "scheduler_submission": {
        "train_array_job_id": "33223076",
        "report_job_id": "33223079",
        "requested_dependency": "afterok:33223076",
        "accepted_dependency": "afterok:33223076_*(unfulfilled)",
        "kill_on_invalid_dependency": "Yes",
        "actual_sbatch_calls": 2,
    },
    "cancellation": {
        "latch_sha256": "a95defbc689623a08d48cead6c6959a585b461a6a70894b56d57d021c52584aa",
        "call_token": "1787897778572633216-1585470",
        "call_sha256": "f706ebbbd16f2ecaa0a2c10af38a6fb3251b7323e6399f970090878d7f3e82fe",
        "result_sha256": "cb3be62a3e4a0e7ca9b9b61ee30771f32099aa61a671b5efb044cb95c7c5909e",
        "command": ["/usr/local/bin/scancel", "33223076", "33223079"],
        "returncode": 0,
        "stdout": "",
        "stderr": "",
        "scheduler_calls": 1,
        "task_cancel_latch_count": 20,
        "task_cancel_latch_canonical_payload_schema": (
            "{schema_version:1,files:{tasks/cell-NN/CANCEL_REQUESTED.json:"
            "raw_file_sha256}}"
        ),
        "task_cancel_latch_canonical_json_bytes": 2130,
        "task_cancel_latch_aggregate_sha256": (
            "c606ca738b0ee29a6cec57d670dae6bb17abeb25c35d4cbccec6885f1cdc947c"
        ),
    },
    "partial_checkpoints": {
        "canonical_payload_schema": (
            "{schema_version:1,validation,rows:[{cell,completed_updates,"
            "checkpoint_sha256,kind,phase,reason}]}"
        ),
        "validation": (
            "sealed_launch6_worker.inspect_run; exact-resume shared validator enabled"
        ),
        "row_constant_fields": {
            "kind": "train",
            "phase": "train",
            "reason": "graceful-stop:SIGTERM",
        },
        "row_count": 20,
        "minimum_completed_updates": 4337,
        "maximum_completed_updates": 6000,
        "total_completed_updates": 102017,
        "canonical_json_bytes": 3867,
        "aggregate_sha256": (
            "b4498f2787600681b5c90862c48f23571be6fce23343c9f5957aae5e993628b5"
        ),
        "rows": [
            {"cell": 0, "completed_updates": 4579, "checkpoint_sha256": "b953b1cc54cb6193a66a626f008af13fcc4298f22a7e98b4428d99e20688cd73"},
            {"cell": 1, "completed_updates": 4590, "checkpoint_sha256": "d4f6ca833c03ef7fbc0ef7b29c57297a30879ee864de38449e68cc7c3f5a9649"},
            {"cell": 2, "completed_updates": 4711, "checkpoint_sha256": "434cacbfc172a8b188f7cd592874876f50cafd565ed73f901089bd276132105d"},
            {"cell": 3, "completed_updates": 4525, "checkpoint_sha256": "a307b2d4c15b5872c745c0e575485f9237db4d8fa545ee410402b2217d3917d8"},
            {"cell": 4, "completed_updates": 4413, "checkpoint_sha256": "f9ace36a0dc829397d2bc2cc2eb0b76ed13b21416abc37adaa557511012430de"},
            {"cell": 5, "completed_updates": 4721, "checkpoint_sha256": "9fa84a01bb026ced6ae66d5e98406e5dc0db300dda272cd8b0b3eb4006b486c0"},
            {"cell": 6, "completed_updates": 4337, "checkpoint_sha256": "065a68e531bd4400df4cadc8b8be8a9bd20fdaa022437e2226b01a93d08c5b3f"},
            {"cell": 7, "completed_updates": 4574, "checkpoint_sha256": "70c97661f8ebfdf4681d7b400e5c56a2c8c93b7d1b4a16f6501dfb4562e8d689"},
            {"cell": 8, "completed_updates": 5950, "checkpoint_sha256": "c717edd639af6b8152ece728c8b6de69bdb4558b63221836026b0379b77dfd92"},
            {"cell": 9, "completed_updates": 5736, "checkpoint_sha256": "c2ffe9b9669afb837694d34c4e70dbeae99d306a821ed4a656928e92e937d4ba"},
            {"cell": 10, "completed_updates": 5699, "checkpoint_sha256": "7bcb376844d7642edceeda6dc5c3b1299ff9b0fc817c2f9dedf7be5db4b56958"},
            {"cell": 11, "completed_updates": 5913, "checkpoint_sha256": "f2237ff1fdf93bb894b601a03df10d0c65eca9af1b115fd2aa82a9bbac745a87"},
            {"cell": 12, "completed_updates": 4568, "checkpoint_sha256": "30649da40677784021aed0e4772dffdf14ff93289d7e22400195cb69d8747ba8"},
            {"cell": 13, "completed_updates": 4644, "checkpoint_sha256": "bc02094735d90db15a1a9b7d6c71bbd2242c021849a423246e3f8429aa24965b"},
            {"cell": 14, "completed_updates": 4687, "checkpoint_sha256": "3b48d93508100d8c85e11b6d54ce78fa82a7e8b3b3bb6c58ac39a2e1235b9ebb"},
            {"cell": 15, "completed_updates": 4787, "checkpoint_sha256": "61c34a70781c83822c07d2064651d1ae3a644f528d8288de08d9dde8c10f7231"},
            {"cell": 16, "completed_updates": 5938, "checkpoint_sha256": "068a0caa081b5d328486d08a8194dc7fbd7ce1400d41174d66772c3386667cfa"},
            {"cell": 17, "completed_updates": 6000, "checkpoint_sha256": "573111ae23b8d6d95846be27e8239ef931fd097b7d690fe99e0fa853e58229eb"},
            {"cell": 18, "completed_updates": 5853, "checkpoint_sha256": "f4e2359d9b2a0f627d530015925485385a3895272a29d0c5dd874908d78dd187"},
            {"cell": 19, "completed_updates": 5792, "checkpoint_sha256": "4152c7a9a9ac9eeaa83c892c09eb05beb7fd723c3a1aaa331100ab3e666a8832"},
        ],
    },
    "tensorboard_scalar_evidence": {
        "comparison": (
            "EventAccumulator scalar float converted to IEEE-754 binary64 big-endian "
            "bits; group key=(cell,tag,step)"
        ),
        "event_file_count": 20,
        "event_file_map_canonical_payload_schema": (
            "{schema_version:1,files:{run_root_relative_event_path:"
            "raw_file_sha256}}"
        ),
        "event_file_selection": (
            "For cells 0..19, read the sealed launch record's run_directory and take "
            "sorted run_directory.glob('events.out.tfevents.*') at the run root only."
        ),
        "event_file_map_canonical_json_bytes": 3864,
        "event_file_map_sha256": (
            "91f77de5c3313a519312cde4244ae08fea0788180d5f2ceefeb8e08b5ccd8da2"
        ),
        "scalar_event_records": 803745,
        "unique_cell_tag_steps": 802659,
        "duplicate_groups": 718,
        "duplicate_extra_occurrences": 1086,
        "conflict_groups": 560,
        "conflict_extra_occurrences": 722,
        "identical_groups": 158,
        "identical_extra_occurrences": 364,
        "census_construction": (
            "EventAccumulator(size_guidance={'scalars':0,'tensors':0}, "
            "purge_orphaned_data=False); cells ascend 0..19, files and tags are "
            "lexically sorted, scalar float values become IEEE-754 binary64 "
            "big-endian hex, duplicate rows sort by (tag,step), and bit-count rows "
            "sort by hex."
        ),
        "duplicate_row_keys": [
            "cell", "tag", "step", "occurrences", "bit_counts", "classification",
        ],
        "bit_count_row_keys": ["float64_bits_hex", "count"],
        "full_census_payload_schema": (
            "{schema_version:1,comparison,rows:all_duplicate_rows}"
        ),
        "conflict_census_payload_schema": (
            "{schema_version:1,comparison,rows:conflict_classified_rows}"
        ),
        "identical_census_payload_schema": (
            "{schema_version:1,comparison,rows:identical_classified_rows}"
        ),
        "per_cell_census_payload_schema": "{schema_version:1,rows:per_cell_rows}",
        "per_cell_row_keys": [
            "cell", "setting", "arm", "seed", "event_files", "scalar_events",
            "unique_tag_steps", "duplicate_groups", "identical_groups",
            "conflict_groups", "identical_extra_events", "conflict_extra_events",
            "conflict_steps", "conflict_tags",
        ],
        "later_occurrences_compared_with_first": {
            "value_conflicting": 715,
            "bit_identical": 371,
            "total": 1086,
            "counting_rule": (
                "Each occurrence after the first in a group is compared only with "
                "that group's first occurrence; these are not group classifications."
            ),
        },
        "full_census_canonical_json_bytes": 152562,
        "full_census_sha256": (
            "26c7127632470128fdfc92441ce2fce5b74f95a17195c4fa07175ef236926f98"
        ),
        "conflict_census_canonical_json_bytes": 126485,
        "conflict_census_sha256": (
            "f4a6f0f22549617004c5390601dada4e6acbebe6a3393902a0cae90f9679b053"
        ),
        "identical_census_canonical_json_bytes": 26225,
        "identical_census_sha256": (
            "04ce578d994c36af58a3db3ad7fad527290831059c96e62964557b2a84c673ea"
        ),
        "per_cell_census_canonical_json_bytes": 9994,
        "per_cell_census_sha256": (
            "47bd1c52254571a74cf32388484c67f318dbdb41b0f8356741c23d978bb23a5e"
        ),
        "validation_aliases": {
            "tags": [
                "bind/negative_margin_loss",
                "control/loss_metric",
                "control/loss_rank",
                "latent_gauge/future/loss",
                "latent_gauge/loss",
                "latent_gauge/root/loss",
            ],
            "groups": 534,
            "conflict_groups": 479,
            "identical_groups": 55,
            "occurrences_per_group": 2,
            "extra_occurrences": 534,
            "conflict_group_extra_occurrences": 479,
            "identical_group_extra_occurrences": 55,
        },
        "visualization_structural_aliases": {
            "tag_count": 23,
            "maze_cell_count": 4,
            "boundaries": [1000, 2000],
            "groups": 184,
            "conflict_groups": 81,
            "identical_groups": 103,
            "occurrences_per_group": 4,
            "extra_occurrences": 552,
            "conflict_group_extra_occurrences": 243,
            "identical_group_extra_occurrences": 309,
            "later_occurrences_compared_with_first": {
                "value_conflicting": 236,
                "bit_identical": 316,
            },
        },
        "conflict_interpretation": (
            "There are 560 groups containing at least two distinct float64 bit values "
            "and 722 occurrences beyond the first within those groups; the 722 are not "
            "claimed to be pairwise distinct."
        ),
    },
    "report_blockers": {
        "immutable_launch6_axis_contract_defect": {
            "tags": [
                "expansion/gain_rank_correlation",
                "expansion/gain_pairwise_accuracy",
                "expansion/gain_eligible_decision_fraction",
                "expansion/gain_ordered_pair_count",
                "expansion/gain_pair_coverage_fraction",
                "tree/support_recall",
                "tree/support_precision",
            ],
            "producer_axis": "dense training cadence: every 50 optimizer updates",
            "sealed_reporter_axis": "sparse validation cadence: every 1000 optimizer updates",
            "evidence": (
                "The independent twenty-event-file census places every listed tag on "
                "its exact dense 50-update training axis."
            ),
        },
        "immutable_launch6_reporter_order": [
            "durable CANCEL_REQUESTED latch rejects report assembly",
            "if the latch were bypassed, the pre-fix reporter rejects the 80 W&B symlinks",
            "if both were bypassed, conflicting scalar identities reject assembly",
            "if those blockers were repaired, its wrong sparse-axis classification rejects seven dense tags",
            "all cells are below 25k, so terminal 25k axes and triplets are incomplete",
        ],
        "current_launch7_candidate_reporter_probe": (
            "The repaired reporter authenticates four W&B leaf links without following "
            "them, classifies the seven method diagnostics on their dense 50-update "
            "training axis, and then deterministically rejects conflicting duplicate "
            "scalar bind/negative_margin_loss@1000."
        ),
        "worker_complete_count": 0,
        "terminal_triplet_count": 0,
        "report_bundle_created": False,
        "report_job_started": False,
    },
    "unsealed_later_terminal_scheduler_observation": {
        "provenance": (
            "independent_operator_observation_after_exact_cancellation_reached_terminal_state"
        ),
        "observation_date_utc": "2026-08-28",
        "preserved_in_launch6_bytes": False,
        "evidence_policy": (
            "corroborative point-in-time state serialized only into Launch7 negative provenance"
        ),
        "environment": {
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "C",
            "LC_ALL": "C",
            "SLURM_CONF": "/cm/shared/apps/slurm/var/etc/cs-oci-ord/slurm.conf",
        },
        "sacct_command": [
            "/usr/local/bin/sacct", "-X", "-n", "-P", "-S", "2026-08-28",
            "-j", "33223076,33223079", "-o",
            "JobID,JobIDRaw,JobName%100,State,ElapsedRaw,AllocNodes,NodeList%100,"
            "Submit,Start,End,ExitCode,DerivedExitCode,Reason%100,Comment%150",
        ],
        "columns": [
            "JobID", "JobIDRaw", "JobName%100", "State", "ElapsedRaw",
            "AllocNodes", "NodeList%100", "Submit", "Start", "End", "ExitCode",
            "DerivedExitCode", "Reason%100", "Comment%150",
        ],
        "sacct_raw_stdout_bytes": 5320,
        "sacct_raw_stdout_sha256": (
            "ed61d986df986bca184c5355acadabb388446e5ee39ab9282ffb75ea780e470e"
        ),
        "canonical_ledger_json_bytes": 6152,
        "canonical_ledger_sha256": (
            "447a0b38dbbd0ead850aa1eb16010e67334452a698f103824393cd0341cbce4d"
        ),
        "canonical_ledger_payload_schema": (
            "{schema_version:1,columns:columns,rows:[serialized_row.split('|') "
            "for serialized_row in serialized_rows]}"
        ),
        "serialized_row_policy": (
            "Exactly 21 rows in sacct stdout order; each stored string contains "
            "exactly 14 pipe-delimited fields and no trailing delimiter."
        ),
        "serialized_rows": [
            "33223076_0|33223113|exp23-launch6-e2758413a5bb28af-train|CANCELLED by 147230|3461|1|batch-block5-04037|2026-08-28T05:18:02|2026-08-28T05:18:38|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
            "33223076_1|33223114|exp23-launch6-e2758413a5bb28af-train|CANCELLED by 147230|3461|1|batch-block5-03148|2026-08-28T05:18:02|2026-08-28T05:18:38|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
            "33223076_2|33223115|exp23-launch6-e2758413a5bb28af-train|CANCELLED by 147230|3461|1|batch-block5-03148|2026-08-28T05:18:02|2026-08-28T05:18:38|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
            "33223076_3|33223116|exp23-launch6-e2758413a5bb28af-train|CANCELLED by 147230|3461|1|batch-block5-03148|2026-08-28T05:18:02|2026-08-28T05:18:38|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
            "33223076_4|33223117|exp23-launch6-e2758413a5bb28af-train|CANCELLED by 147230|3461|1|batch-block5-03915|2026-08-28T05:18:02|2026-08-28T05:18:38|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
            "33223076_5|33223118|exp23-launch6-e2758413a5bb28af-train|CANCELLED by 147230|3461|1|batch-block5-03073|2026-08-28T05:18:02|2026-08-28T05:18:38|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
            "33223076_6|33223119|exp23-launch6-e2758413a5bb28af-train|CANCELLED by 147230|3461|1|batch-block5-03846|2026-08-28T05:18:02|2026-08-28T05:18:38|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
            "33223076_7|33223120|exp23-launch6-e2758413a5bb28af-train|CANCELLED by 147230|3461|1|batch-block5-03732|2026-08-28T05:18:02|2026-08-28T05:18:38|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
            "33223076_8|33223121|exp23-launch6-e2758413a5bb28af-train|CANCELLED by 147230|3461|1|batch-block5-01993|2026-08-28T05:18:02|2026-08-28T05:18:38|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
            "33223076_9|33223122|exp23-launch6-e2758413a5bb28af-train|CANCELLED by 147230|3461|1|batch-block5-00449|2026-08-28T05:18:02|2026-08-28T05:18:38|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
            "33223076_10|33223123|exp23-launch6-e2758413a5bb28af-train|CANCELLED by 147230|3461|1|batch-block5-00419|2026-08-28T05:18:02|2026-08-28T05:18:38|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
            "33223076_11|33223124|exp23-launch6-e2758413a5bb28af-train|CANCELLED by 147230|3461|1|batch-block5-01592|2026-08-28T05:18:02|2026-08-28T05:18:38|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
            "33223076_12|33223176|exp23-launch6-e2758413a5bb28af-train|CANCELLED by 147230|3402|1|batch-block5-04044|2026-08-28T05:18:02|2026-08-28T05:19:37|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
            "33223076_13|33223330|exp23-launch6-e2758413a5bb28af-train|CANCELLED by 147230|3342|1|batch-block5-04037|2026-08-28T05:18:02|2026-08-28T05:20:37|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
            "33223076_14|33223331|exp23-launch6-e2758413a5bb28af-train|CANCELLED by 147230|3342|1|batch-block5-00297|2026-08-28T05:18:02|2026-08-28T05:20:37|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
            "33223076_15|33223332|exp23-launch6-e2758413a5bb28af-train|CANCELLED by 147230|3342|1|batch-block5-00297|2026-08-28T05:18:02|2026-08-28T05:20:37|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
            "33223076_16|33223333|exp23-launch6-e2758413a5bb28af-train|CANCELLED by 147230|3342|1|batch-block5-00297|2026-08-28T05:18:02|2026-08-28T05:20:37|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
            "33223076_17|33223334|exp23-launch6-e2758413a5bb28af-train|CANCELLED by 147230|3342|1|batch-block5-00297|2026-08-28T05:18:02|2026-08-28T05:20:37|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
            "33223076_18|33223372|exp23-launch6-e2758413a5bb28af-train|CANCELLED by 147230|3282|1|batch-block5-00642|2026-08-28T05:18:02|2026-08-28T05:21:37|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
            "33223076_19|33223076|exp23-launch6-e2758413a5bb28af-train|CANCELLED by 147230|3282|1|batch-block5-03613|2026-08-28T05:18:02|2026-08-28T05:21:37|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
            "33223079|33223079|exp23-launch6-e2758413a5bb28af-report|CANCELLED by 147230|0|0|None assigned|2026-08-28T05:18:07|None|2026-08-28T06:16:19|0:0|0:0|None|treewm-exp23:e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e",
        ],
        "sacct_row_count": 21,
        "cancelled_task_indices": list(range(20)),
        "task_state": "CANCELLED by 147230",
        "report_state": "CANCELLED by 147230",
        "task_exit_codes": ["0:0"],
        "task_derived_exit_codes": ["0:0"],
        "squeue_command": [
            "/usr/local/bin/squeue", "--array", "--noheader",
            "--jobs=33223076,33223079", "--format=%i|%T|%R|%N",
        ],
        "squeue_raw_stdout_bytes": 0,
        "squeue_raw_stdout_sha256": (
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        ),
        "squeue_row_count": 0,
    },
    "scientific_state": {
        "worker_started_cell_count": 20,
        "trainer_bridge_started_cell_count": 20,
        "hydra_composition_completed_cell_count": 20,
        "model_constructed_cell_count": 20,
        "optimizer_update_total_across_cells": 102017,
        "minimum_optimizer_updates_per_cell": 4337,
        "maximum_optimizer_updates_per_cell": 6000,
        "cells_reaching_analysis_boundary_5000": 8,
        "latest_checkpoint_count": 20,
        "best_validation_checkpoint_count": 20,
        "online_wandb_identity_count": 20,
        "scientific_ready_marker_committed": False,
        "scientific_report_bundle_committed": False,
        "results_consumed": False,
        "checkpoints_consumed": False,
    },
    "known_job_ids": ["33223076", "33223079"],
    "job_ids_by_role": {"train": ["33223076"], "report": ["33223079"]},
    "submission_contract_committed": True,
    "submission_receipt_committed": True,
    "cancel_latch_committed": True,
    "scientific_run_started": True,
    "checkpoint_created": True,
    "wandb_run_created": True,
    "scientific_run_completed": False,
    "results_consumed": False,
    "checkpoints_consumed": False,
    "reuse_allowed": False,
    "resume_allowed": False,
    "retry_allowed": False,
    "recovery_allowed": False,
}, {
    "campaign_id": "treewm-executable-prefix-repair-pilot-v1-launch7",
    "run_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch7"
    ),
    "submission_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch7/"
        "state/submission"
    ),
    "snapshot_root": (
        "/lustre/fs11/portfolios/edgeai/projects/"
        "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
        "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch7/"
        "state/submission/source-snapshot/repo"
    ),
    "transaction_lock": "outputs/.exp23-8fc3c9e0775ae4d7.transaction.lock",
    "wandb_project": "treewm-executable-prefix-repair-pilot-v1-launch7",
    "status": "terminal_failed_compute_scheduler_client_topology",
    "source_commit": "bdd5d819d291104d5aa7cbe8934ba935cb76518c",
    "submission_sha256": (
        "fc4ce1695e6e4ed5a23c8cd0a240299e0823ed1f2a17da87461261d74e74d112"
    ),
    "receipt_sha256": (
        "9be7e28cb2ffdfa7ab59f903eeb50e34e8e8f58a4dd4273d7fd789dfbc6ade10"
    ),
    "package_protocol_sha256": (
        "300d4aacde0502477450dbfc6bc2aaaa80eade93bc572776dd59a0cdd2d7d582"
    ),
    "manifest_canonical_sha256": (
        "f43c885bce1734e3b2eab842d1c56dd18e3a1b7bfd302c51ccbdf5a1ebe2ab5d"
    ),
    "negative_provenance": {
        "path": "launch7_negative_provenance.json",
        "raw_sha256": (
            "29051e9839b9ceff4160b8ea0e99e82ce449cd7c2306f1e3604b30f24bb0272e"
        ),
        "canonical_sha256": (
            "48839a4f58214d7a1b616f2f43089e24e16f44717865bac8c3c76845c4457e62"
        ),
        "scheduler_terminal_rows": 21,
        "ready_checkpoint_cells": 16,
        "completed_cells": 4,
        "report_started": False,
        "active_scheduler_jobs_after_terminal": 0,
    },
    "job_ids_by_role": {
        "wave0_train": ["33236584"],
        "report": ["33236586"],
    },
    "proof_scope": (
        "Launch7 started all twenty fresh generation-zero cells. Four cells completed "
        "25k and terminal evaluation; sixteen sealed exact USR1 READY checkpoints, then "
        "failed deterministically because /usr/local/bin/scontrol was absent from every "
        "observed GPU compute-node image. No REQUEUE_CALLING or replacement generation "
        "exists and the reporter was cancelled with DependencyNeverSatisfied. The full "
        "read-only terminal census is protocol-bound by launch7_negative_provenance.json. "
        "Every Launch7 output, checkpoint, W&B identity, task state, receipt, job ID, and "
        "namespace is negative evidence only and cannot enter Launch8."
    ),
    "submission_contract_committed": True,
    "submission_receipt_committed": True,
    "scientific_run_started": True,
    "checkpoint_created": True,
    "wandb_run_created": True,
    "scientific_run_completed": False,
    "report_valid": False,
    "results_consumed": False,
    "checkpoints_consumed": False,
    "reuse_allowed": False,
    "resume_allowed": False,
    "retry_allowed": False,
    "recovery_allowed": False,
}]


class ContractError(RuntimeError):
    pass


@dataclass(frozen=True)
class Cell:
    index: int
    setting_index: int
    arm_index: int
    seed_index: int
    setting: str
    env_config: str
    arm: str
    seed: int
    run_name: str


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def canonical_json(value: object) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )


def exact_json_equal(left: object, right: object) -> bool:
    """Compare JSON values without Python's bool/int equality coercion."""

    return canonical_json(left) == canonical_json(right)


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _regular_file_bytes(path: Path, label: str, *, max_bytes: int) -> bytes:
    require(
        type(max_bytes) is int and max_bytes >= 0,
        f"{label} byte boundary differs",
    )
    try:
        named_before = path.lstat()
        require(
            stat.S_ISREG(named_before.st_mode) and not path.is_symlink(),
            f"{label} is not a regular nonsymlink file",
        )
        descriptor = os.open(
            path,
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
        )
    except ContractError:
        raise
    except OSError as exc:
        raise ContractError(f"{label} is unavailable: {exc}") from exc
    try:
        opened_before = os.fstat(descriptor)
        require(
            stat.S_ISREG(opened_before.st_mode)
            and (opened_before.st_dev, opened_before.st_ino)
            == (named_before.st_dev, named_before.st_ino)
            and opened_before.st_size <= max_bytes,
            f"{label} raced before read",
        )
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        payload = b"".join(chunks)
        opened_after = os.fstat(descriptor)
        named_after = path.lstat()
        require(
            (opened_after.st_dev, opened_after.st_ino, opened_after.st_size)
            == (opened_before.st_dev, opened_before.st_ino, opened_before.st_size)
            and opened_after.st_mtime_ns == opened_before.st_mtime_ns
            and len(payload) == opened_after.st_size
            and (named_after.st_dev, named_after.st_ino, named_after.st_size)
            == (opened_after.st_dev, opened_after.st_ino, opened_after.st_size)
            and named_after.st_mtime_ns == opened_after.st_mtime_ns,
            f"{label} raced during read",
        )
        return payload
    except OSError as exc:
        raise ContractError(f"{label} read failed: {exc}") from exc
    finally:
        os.close(descriptor)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant: {value}")


def read_json(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle, object_pairs_hook=_pairs)
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {path}: {exc}") from exc
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def atomic_json(path: str | Path, value: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(value), handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return stable_hash(manifest)


def _override(name: str, value: object) -> str:
    if isinstance(value, bool):
        rendered = str(value).lower()
    elif value is None:
        rendered = "null"
    elif isinstance(value, (list, tuple)):
        rendered = "[" + ",".join(
            str(item).lower() if isinstance(item, bool) else str(item)
            for item in value
        ) + "]"
    else:
        rendered = str(value)
    return f"{name}={rendered}"


def run_directory(manifest: Mapping[str, Any], cell: Cell) -> Path:
    return Path(manifest["paths"]["run_root"]) / cell.setting / "treewm" / cell.run_name


def recipe_root(manifest: Mapping[str, Any], setting_id: str) -> Path:
    return Path(manifest["paths"]["compatible_contract_root"]) / "future-recipes" / setting_id


def load_compatible_input(
    manifest: Mapping[str, Any], setting_id: str, *, verify_files: bool = False
) -> dict[str, Any]:
    setting = next(row for row in manifest["settings"] if row["id"] == setting_id)
    path = Path(manifest["paths"]["compatible_contract_root"]) / "data" / f"{setting_id}.json"
    contract = read_json(path)
    claimed = contract.get("contract_sha256")
    body = dict(contract)
    body.pop("contract_sha256", None)
    require(claimed == stable_hash(body) == setting["input_contract_sha256"], f"{setting_id}: input contract differs")
    legacy = manifest["compatible_v2_recipe_input"]
    expected = {
        "campaign_id": legacy["campaign_id"],
        "objective_version": legacy["objective_version"],
        "campaign_protocol_sha256": legacy["campaign_protocol_sha256"],
        "setting_id": setting_id,
        "dataset_kind": setting["dataset_kind"],
        "calibration_sha256": setting["calibration_sha256"],
        "future_recipe_sha256": setting["future_recipe_sha256"],
        "raw_cache_read_only": True,
    }
    for key, value in expected.items():
        require(contract.get(key) == value, f"{setting_id}: compatible {key} differs")
    composite = read_json(recipe_root(manifest, setting_id) / "manifest.json")
    require(composite.get("recipe_sha256") == setting["future_recipe_sha256"], f"{setting_id}: recipe differs")
    if verify_files:
        from treewm.data.future_recipe import validate_recipe_manifest

        validate_recipe_manifest(
            recipe_root(manifest, setting_id),
            composite,
            expected_source_manifest_sha256=contract["data_manifest_sha256"],
            expected_normalizer_sha256=contract["normalizer_sha256"],
            expected_calibration_sha256=contract["calibration_sha256"],
            expected_thresholds=contract["chosen_thresholds"],
            expected_train_manifest_sha256=contract["train_manifest_sha256"],
            expected_validation_manifest_sha256=contract["validation_manifest_sha256"],
            expected_code_sha256=legacy["recipe_code_sha256"],
            expected_runtime_sha256=legacy["recipe_runtime_sha256"],
            verify_file_hash=True,
        )
    return contract


def source_contract(repo_root: str | Path = REPOSITORY_ROOT) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from treewm.utils.provenance import runtime_fingerprint, trainer_code_fingerprint

    source = trainer_code_fingerprint(root)
    runtime = runtime_fingerprint()
    return {
        "source_sha256": source["manifest_sha256"],
        "source_files": source["files"],
        "runtime_sha256": runtime["sha256"],
        "runtime": runtime,
    }


def protocol_sha256(root: str | Path = PACKAGE_DIR) -> str:
    package = Path(root).resolve()
    require(len(PROTOCOL_FILES) == len(set(PROTOCOL_FILES)), "duplicate protocol file")
    files: dict[str, str] = {}
    for relative in PROTOCOL_FILES:
        path = package / relative
        require(path.is_file() and not path.is_symlink(), f"missing/symlink protocol file: {relative}")
        files[relative] = file_sha256(path)
    return stable_hash({"schema_version": 1, "files": files})


def validate_snapshot_import_files(repo_root: str | Path = REPOSITORY_ROOT) -> None:
    expected = {
        "configs/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "scripts/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }
    require(SNAPSHOT_IMPORT_FILES == expected, "snapshot import inventory differs")
    root = Path(repo_root).resolve()
    for relative, expected_sha256 in SNAPSHOT_IMPORT_FILES.items():
        candidate = root / relative
        require(
            candidate.is_file()
            and not candidate.is_symlink()
            and candidate.resolve().is_relative_to(root),
            f"snapshot import is unavailable/symlinked: {relative}",
        )
        require(
            file_sha256(candidate) == expected_sha256,
            f"snapshot import bytes differ: {relative}",
        )


def verify_protocol_lock(root: str | Path = PACKAGE_DIR) -> str:
    package = Path(root).resolve()
    try:
        locked = (package / "protocol.sha256").read_text(encoding="ascii").strip()
    except OSError as exc:
        raise ContractError(f"protocol lock unavailable: {exc}") from exc
    live = protocol_sha256(package)
    require(SHA256.fullmatch(locked) is not None and locked == live, "protocol lock stale")
    return live


def expand_matrix(manifest: Mapping[str, Any]) -> list[Cell]:
    settings = manifest["settings"]
    result: list[Cell] = []
    for setting_index, setting in enumerate(settings):
        for arm_index, arm in enumerate(ARMS):
            for seed_index, seed in enumerate(SEEDS):
                index = ((setting_index * len(ARMS)) + arm_index) * len(SEEDS) + seed_index
                require(index == len(result), "matrix mapping is not contiguous")
                result.append(
                    Cell(
                        index=index,
                        setting_index=setting_index,
                        arm_index=arm_index,
                        seed_index=seed_index,
                        setting=str(setting["id"]),
                        env_config=str(setting["env_config"]),
                        arm=arm,
                        seed=seed,
                        run_name=f"exp23-launch8-{setting['id']}-arm{arm.lower()}-seed{seed}",
                    )
                )
    return result


def cell_overrides(cell: Cell, manifest: Mapping[str, Any], lock: Mapping[str, Any]) -> dict[str, Any]:
    """Return declarative Hydra leaves; never execute them."""
    arm = next(row for row in manifest["arms"] if row["id"] == cell.arm)
    weights = arm["executable_prefix_weights"]
    bounds = lock["action_bounds"][cell.setting]
    return {
        "experiment": manifest["method"]["experiment_config"],
        "env": cell.env_config,
        "arm": "treewm",
        "seed": cell.seed,
        "objective_version": manifest["method"]["objective_version"],
        "train.steps": manifest["scientific_contract"]["optimizer_updates"],
        "future_sets.executable_prefix_steps": 4,
        "losses.enabled.executable_prefix_action": True,
        "losses.enabled.executable_prefix_latent": True,
        "losses.enabled.executable_prefix_endpoint": True,
        "losses.weights.executable_prefix_action": weights["action"],
        "losses.weights.executable_prefix_latent": weights["latent"],
        "losses.weights.executable_prefix_endpoint": weights["endpoint"],
        "losses.executable_action_lower_bound": bounds["lower"],
        "losses.executable_action_upper_bound": bounds["upper"],
        "planner.action_lower_bound": bounds["lower"],
        "planner.action_upper_bound": bounds["upper"],
        "+campaign_id": manifest["campaign_id"],
        "+weight_audit_artifact_sha256": manifest["weight_audit"]["artifact_sha256"],
    }


def scientific_overrides(
    cell: Cell,
    manifest: Mapping[str, Any],
    lock: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> list[str]:
    """Return the complete ordered Hydra scientific config for one cell.

    Unique cell/run labels are deliberately absent.  Within a setting/seed pair the
    only arm-dependent leaves are the three audited executable-prefix weights.
    """
    setting = next(row for row in manifest["settings"] if row["id"] == cell.setting)
    scientific = manifest["scientific_contract"]
    future = scientific["future_sets"]
    chosen = contract["chosen_thresholds"]
    bounds = lock["action_bounds"][cell.setting]
    weights = next(row for row in manifest["arms"] if row["id"] == cell.arm)[
        "executable_prefix_weights"
    ]
    values: list[tuple[str, object]] = [
        ("env", cell.env_config),
        ("experiment", manifest["method"]["experiment_config"]),
        ("arm", "treewm"),
        ("objective_version", manifest["method"]["objective_version"]),
        ("seed", cell.seed),
        ("train.steps", scientific["optimizer_updates"]),
        ("train.scheduler_total_steps", scientific["scheduler_total_steps"]),
        ("train.ckpt_every", scientific["checkpoint_every_updates"]),
        ("train.val_every", scientific["validation_every_updates"]),
        ("train.diag_every", scientific["diagnostics_every_updates"]),
        ("train.eval_every", scientific["periodic_evaluation_every_updates"]),
        ("train.log_every", scientific["training_telemetry_every_updates"]),
        ("train.validation_sample_seed", scientific["validation_sample_seed"]),
        ("train.max_train_anchors", setting["published_union_train_anchors"]),
        ("train.max_val_anchors", setting["published_union_validation_anchors"]),
        ("train.num_workers", scientific["data_loader_workers"]),
        ("train.lr", scientific["world_lr"]),
        ("train.weight_decay", scientific["world_weight_decay"]),
        ("train.gradient_checkpointing", scientific["gradient_checkpointing"]),
        ("train.separate_gain_grad_clip", scientific["separate_gain_grad_clip"]),
        ("train.separate_branch_transformer_grad_clip", scientific["separate_branch_transformer_grad_clip"]),
        ("train.world_grad_clip", scientific["world_grad_clip"]),
        ("train.branch_transformer_grad_clip", scientific["branch_transformer_grad_clip"]),
        ("train.gain_grad_clip", scientific["gain_grad_clip"]),
        ("train.gain_loss_every", scientific["gain_loss_every"]),
        ("train.gain_lr", scientific["gain_lr"]),
        ("train.gain_weight_decay", scientific["gain_weight_decay"]),
        ("train.gain_training_scorers", scientific["gain_training_scorers"]),
        ("train.viz_every", scientific["visualization_every_updates"]),
        ("train.viz_every_early", scientific["visualization_every_early_updates"]),
        ("train.viz_early_until", scientific["visualization_early_until_updates"]),
        ("model.dropout", scientific["model_dropout"]),
        ("model.max_depth", scientific["model_max_depth"]),
        ("tree.max_depth", scientific["tree_max_depth"]),
        ("tree.node_budget", manifest["method"]["node_budget"]),
        ("tree.keep_threshold", scientific["keep_threshold"]),
        ("tree.scorer", scientific["tree_scorer"]),
        ("model.branch_factor", manifest["method"]["branch_factor"]),
        ("planner.decoded_metric", "domain_raw"),
        ("planner.execute_mode", "clipped"),
        ("planner.execute_steps", 4),
        ("planner.max_env_steps", setting["max_episode_steps"]),
        ("planner.require_first_edge_improvement", True),
        ("planner.min_first_edge_improvement", 0.0),
    ]
    values.extend((f"future_sets.{key}", value) for key, value in future.items() if key not in {"recipe_anchor_policy"})
    values.extend(
        [
            ("future_sets.recipe_anchor_policy", "published_union"),
            ("future_sets.relative_endpoints", setting["relative_endpoints"]),
            ("future_sets.retrieval_radius", chosen["retrieval_radius"]),
            ("future_sets.displacement_threshold", chosen["displacement_threshold"]),
            ("future_sets.cluster_threshold", chosen["cluster_threshold"]),
            ("+env.task_metric_dims", setting["task_metric_dims"]),
            ("losses.keep_balance", True),
            ("losses.enabled.multistep", True),
            ("losses.weights.multistep", scientific["multistep_weight"]),
            ("losses.scheduled_sampling_p", scientific["scheduled_sampling_p"]),
            ("losses.scheduled_sampling_warmup", scientific["scheduled_sampling_warmup"]),
            ("losses.scheduled_sampling_granularity", scientific["scheduled_sampling_granularity"]),
            ("losses.multistep_transition_mode", "grounded_execution_v2"),
            ("losses.grounded_select_action_weight", scientific["grounded_select_weights"]["action"]),
            ("losses.grounded_select_endpoint_weight", scientific["grounded_select_weights"]["endpoint"]),
            ("losses.grounded_select_horizon_weight", scientific["grounded_select_weights"]["horizon"]),
            ("losses.grounded_loss_latent_weight", scientific["grounded_loss_weights"]["latent"]),
            ("losses.grounded_loss_action_weight", scientific["grounded_loss_weights"]["action"]),
            ("losses.grounded_loss_horizon_weight", scientific["grounded_loss_weights"]["horizon"]),
            ("losses.grounded_loss_endpoint_weight", scientific["grounded_loss_weights"]["endpoint"]),
            ("losses.grounded_detach_self_fed_parent", scientific["grounded_detach_self_fed_parent"]),
            ("losses.multistep_depth_weights", scientific["multistep_depth_weights"]),
            ("losses.enabled.latent_gauge", scientific["latent_gauge_enabled"]),
            ("losses.weights.latent_gauge", scientific["latent_gauge_weight"]),
            ("losses.latent_gauge_epsilon", scientific["latent_gauge_epsilon"]),
            ("losses.latent_gauge_min_reference_scale", scientific["latent_gauge_min_reference_scale"]),
            ("losses.enabled.executable_prefix_action", True),
            ("losses.enabled.executable_prefix_latent", True),
            ("losses.enabled.executable_prefix_endpoint", True),
            ("losses.weights.executable_prefix_action", weights["action"]),
            ("losses.weights.executable_prefix_latent", weights["latent"]),
            ("losses.weights.executable_prefix_endpoint", weights["endpoint"]),
            ("losses.executable_action_lower_bound", bounds["lower"]),
            ("losses.executable_action_upper_bound", bounds["upper"]),
            ("planner.action_lower_bound", bounds["lower"]),
            ("planner.action_upper_bound", bounds["upper"]),
            ("eval.task_split", scientific["task_split"]),
            ("eval.episodes_per_task", scientific["periodic_episodes_per_task"]),
            ("eval.final_episodes_per_task", scientific["final_episodes_per_task"]),
            ("eval.seed", scientific["evaluation_seed"]),
            ("+campaign_id", manifest["campaign_id"]),
            ("+campaign_input_contract_sha256", contract["contract_sha256"]),
            ("+campaign_calibration_sha256", contract["calibration_sha256"]),
            ("+campaign_future_recipe_sha256", contract["future_recipe_sha256"]),
            ("+campaign_compatible_recipe_code_sha256", manifest["compatible_v2_recipe_input"]["recipe_code_sha256"]),
            ("+weight_audit_artifact_sha256", manifest["weight_audit"]["artifact_sha256"]),
            ("+prefix_target_artifact_sha256", manifest["prefix_target_contract"]["artifact_sha256"]),
            ("run_root", manifest["paths"]["run_root"]),
            ("run_name", None),
            ("resume", "auto"),
        ]
    )
    return [_override(name, value) for name, value in values]


def actual_final_evaluation_rows(manifest: Mapping[str, Any]) -> list[dict[str, int]]:
    scientific = manifest["scientific_contract"]
    rows = [
        {
            "task_index": task_index,
            "task_id": int(task_id),
            "episode_index": episode_index,
            "episode_seed": int(scientific["evaluation_seed"]) + 1000 * task_index + episode_index,
        }
        for task_index, task_id in enumerate(scientific["task_ids"])
        for episode_index in range(int(scientific["final_episodes_per_task"]))
    ]
    require(len(rows) == 25, "terminal evaluation row count differs")
    return rows


def trainer_command(
    manifest: Mapping[str, Any],
    lock: Mapping[str, Any],
    cell: Cell,
    *,
    repo_root: str | Path = REPOSITORY_ROOT,
    package_protocol_sha256: str | None = None,
    verify_recipe_files: bool = False,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    protocol = package_protocol_sha256 or verify_protocol_lock(root / "experiments/23-treewm-executable-prefix-repair-pilot-v1")
    contract = load_compatible_input(manifest, cell.setting, verify_files=verify_recipe_files)
    source = source_contract(root)
    overrides = scientific_overrides(cell, manifest, lock, contract)
    config_sha = stable_hash({"schema_version": 1, "overrides": overrides})
    output = run_directory(manifest, cell)
    run_protocol = stable_hash(
        {
            "schema_version": 1,
            "campaign_id": manifest["campaign_id"],
            "package_protocol_sha256": protocol,
            "source_sha256": source["source_sha256"],
            "runtime_sha256": source["runtime_sha256"],
            "config_sha256": config_sha,
            "input_contract_sha256": contract["contract_sha256"],
            "data_manifest_sha256": contract["data_manifest_sha256"],
            "validation_manifest_sha256": contract["validation_manifest_sha256"],
            "future_recipe_sha256": contract["future_recipe_sha256"],
            "weight_audit_artifact_sha256": manifest["weight_audit"]["artifact_sha256"],
            "prefix_target_artifact_sha256": manifest["prefix_target_contract"]["artifact_sha256"],
            "resolved_config_artifact_sha256": manifest["resolved_config_contract"]["artifact_sha256"],
            "causal_parity_artifact_sha256": manifest["causal_parity_contract"]["artifact_sha256"],
        }
    )
    argv = [
        manifest["paths"]["python"],
        str(root / "scripts/train.py"),
        *overrides,
        _override("hydra.run.dir", output / "hydra"),
        _override("hydra.job.chdir", False),
    ]
    wandb_id = stable_hash(
        {"campaign_id": manifest["campaign_id"], "cell": asdict(cell)}
    )[:32]
    environment = {
        "TREEWM_PROTOCOL_SHA256": run_protocol,
        "TREEWM_CODE_SHA256": source["source_sha256"],
        "TREEWM_ACTIVE_SOURCE_SHA256": source["source_sha256"],
        "TREEWM_RUNTIME_SHA256": source["runtime_sha256"],
        "TREEWM_RECIPE_CODE_SHA256": manifest["compatible_v2_recipe_input"]["recipe_code_sha256"],
        "TREEWM_RECIPE_RUNTIME_SHA256": manifest["compatible_v2_recipe_input"]["recipe_runtime_sha256"],
        "TREEWM_CONFIG_SHA256": config_sha,
        "TREEWM_DATA_SHA256": contract["data_manifest_sha256"],
        "TREEWM_CALIBRATION_SHA256": contract["calibration_sha256"],
        "TREEWM_FUTURE_RECIPE_SHA256": contract["future_recipe_sha256"],
        "TREEWM_CAUSAL_PARITY_SHA256": manifest["causal_parity_contract"]["artifact_sha256"],
        "TREEWM_RESOLVED_CONFIG_SHA256": manifest["resolved_config_contract"]["artifact_sha256"],
        "TREEWM_DATA_CONTRACT_SHA256": contract["contract_sha256"],
        "TREEWM_DATA_ROOT": manifest["paths"]["data_root"],
        "TREEWM_CACHE": manifest["paths"]["raw_cache_root"],
        "TREEWM_FUTURE_RECIPE_ROOT": str(recipe_root(manifest, cell.setting)),
        "TREEWM_EVALUATION_SEED_PROTOCOL_SHA256": manifest["scientific_contract"]["evaluation_seed_protocol_sha256"],
        "TREEWM_RUN_NAME": cell.run_name,
        "WANDB_PROJECT": manifest["logging"]["wandb_project"],
        "WANDB_RUN_GROUP": manifest["logging"]["wandb_group"],
        "WANDB_RUN_ID": wandb_id,
        "WANDB_MODE": manifest["logging"]["wandb_mode"],
        "OMP_NUM_THREADS": str(manifest["scientific_contract"]["loader_thread_limit"]),
        "MKL_NUM_THREADS": str(manifest["scientific_contract"]["loader_thread_limit"]),
        "OPENBLAS_NUM_THREADS": str(manifest["scientific_contract"]["loader_thread_limit"]),
        "MUJOCO_GL": manifest["execution"]["sealed_trainer_environment"]["MUJOCO_GL"],
        "XLA_PYTHON_CLIENT_PREALLOCATE": manifest["execution"]["sealed_trainer_environment"]["XLA_PYTHON_CLIENT_PREALLOCATE"],
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    launch: dict[str, Any] = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "cell": {**asdict(cell), "run_directory": str(output), "wandb_id": wandb_id},
        "argv": argv,
        "environment": environment,
        "hashes": {
            "manifest_sha256": manifest_sha256(manifest),
            "source_sha256": source["source_sha256"],
            "runtime_sha256": source["runtime_sha256"],
            "package_protocol_sha256": protocol,
            "config_override_sha256": config_sha,
            "run_protocol_sha256": run_protocol,
            "input_contract_sha256": contract["contract_sha256"],
            "data_manifest_sha256": contract["data_manifest_sha256"],
            "normalizer_sha256": contract["normalizer_sha256"],
            "train_manifest_sha256": contract["train_manifest_sha256"],
            "validation_manifest_sha256": contract["validation_manifest_sha256"],
            "calibration_sha256": contract["calibration_sha256"],
            "future_recipe_sha256": contract["future_recipe_sha256"],
            "weight_audit_artifact_sha256": manifest["weight_audit"]["artifact_sha256"],
            "prefix_target_artifact_sha256": manifest["prefix_target_contract"]["artifact_sha256"],
            "resolved_config_artifact_sha256": manifest["resolved_config_contract"]["artifact_sha256"],
            "causal_parity_artifact_sha256": manifest["causal_parity_contract"]["artifact_sha256"],
            "actual_final_evaluation_rows_sha256": stable_hash(actual_final_evaluation_rows(manifest)),
        },
    }
    launch["launch_sha256"] = stable_hash(launch)
    return launch


def _validate_lock(lock: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    require(lock.get("schema_version") == 1 and lock.get("status") == "frozen", "weight lock not frozen")
    identity = lock["result_identity"]
    for key in ("artifact_sha256", "rows_sha256", "summary_sha256"):
        require(SHA256.fullmatch(str(identity.get(key, ""))) is not None, f"invalid audit {key}")
        require(identity[key] == manifest["weight_audit"][key], f"manifest/audit {key} differs")
    require(identity.get("row_count") == 40, "audit row count differs")
    contract = lock["contract"]
    require(contract["settings"] == list(SETTINGS), "audit settings differ")
    require(contract["regimes"] == ["exp20_gs_exact_5000", "scratch_initialization"], "audit regimes differ")
    require(contract["checkpoint_seeds"] == [108, 109] and contract["scratch_seeds"] == [230, 231], "audit seeds differ")
    require(contract["fixed_batches_per_setting_regime"] == 2 and contract["batch_size"] == 16, "audit batch design differs")
    require(contract["groups"] == ["branch_transformer", "world_rest"], "audit groups differ")
    require(contract["per_component_median_base_gradient_fraction_max"] == 0.03, "component gradient budget differs")
    require(contract["aggregate_every_row_group_base_gradient_fraction_max"] == 0.10, "aggregate gradient budget differs")
    require(lock["derived"]["post_scale_max_aggregate_ratio"] <= 0.10, "audited tuple exceeds aggregate budget")
    expected = manifest["arms"][1]["executable_prefix_weights"]
    actual = lock["derived"]["weights"]
    runtime = lock["derived"]["audit_runtime_float_weights"]
    require(
        actual == {
            "executable_prefix_action": expected["action"],
            "executable_prefix_latent": expected["latent"],
            "executable_prefix_endpoint": expected["endpoint"],
        },
        "treatment weights differ from audit",
    )
    require(
        all(0.0 <= runtime[name] - actual[name] <= math.ulp(runtime[name]) for name in actual),
        "canonical audit decimals are not conservative one-ULP renderings",
    )
    require(len(lock["checkpoint_sha256"]) == 10, "checkpoint hash inventory incomplete")
    require(len(lock["batch_sha256"]) == 20, "batch hash inventory incomplete")
    external_keys = {
        "exp20/manifest.json",
        *(
            f"{setting}/seed{seed}/GAUGE_PILOT_V2_LAUNCH.json"
            for setting in SETTINGS
            for seed in (108, 109)
        ),
    }
    require(
        set(lock.get("external_input_sha256") or {}) == external_keys,
        "external input hash inventory differs",
    )
    for inventory in (
        lock["checkpoint_sha256"],
        lock["external_input_sha256"],
        lock["batch_sha256"],
        lock["source_sha256"],
    ):
        require(all(SHA256.fullmatch(str(value)) for value in inventory.values()), "invalid audit hash")
    require(file_sha256(PACKAGE_DIR / "weight_audit.py") == lock["source_sha256"]["audit"], "auditor source differs")
    api = lock["fail_closed_api_binding"]
    require(api["effective_tree_config_required"] is True, "tree config is not fail closed")
    require("tree_config_for" in api["audit_call"], "audit did not bind effective tree config")
    for setting in SETTINGS:
        bounds = lock["action_bounds"][setting]
        require(bounds["action_dim"] > 0 and bounds["lower"] < bounds["upper"], f"{setting}: bounds invalid")
        require(SHA256.fullmatch(bounds["lower_sha256"]) is not None, f"{setting}: lower hash invalid")
        require(SHA256.fullmatch(bounds["upper_sha256"]) is not None, f"{setting}: upper hash invalid")


def _validate_prefix_target_lock(manifest: Mapping[str, Any]) -> dict[str, Any]:
    lock = read_json(PREFIX_TARGET_LOCK_PATH)
    claimed = lock.get("artifact_sha256")
    body = dict(lock)
    body.pop("artifact_sha256", None)
    require(claimed == stable_hash(body), "prefix-target artifact hash differs")
    binding = manifest["prefix_target_contract"]
    require(claimed == binding["artifact_sha256"], "manifest prefix-target artifact differs")
    require(file_sha256(PACKAGE_DIR / binding["audit_source"]) == binding["source_sha256"] == lock["source_sha256"], "prefix-target auditor source differs")
    require(lock["weight_audit_artifact_sha256"] == manifest["weight_audit"]["artifact_sha256"], "prefix-target/weight audit binding differs")
    weight_lock = read_json(WEIGHT_LOCK_PATH)
    require(
        lock.get("external_input_sha256") == weight_lock.get("external_input_sha256"),
        "prefix-target external-input binding differs",
    )
    require(set(lock["settings"]) == set(SETTINGS), "prefix-target setting coverage differs")
    for setting, row in lock["settings"].items():
        require(row["setting_id"] == setting, f"{setting}: prefix-target setting label differs")
        require(row["anchor_count"] == binding["validation_anchor_count_per_setting"] == 5120, f"{setting}: prefix-target anchor count differs")
        require(row["batch_size"] == binding["batch_size"] == 256 and row["num_batches"] == binding["validation_batches"] == 20, f"{setting}: fixed validation shape differs")
        require(row["all_anchors_have_match"] is True and row["matched_branch_count"] >= row["anchor_count"], f"{setting}: incomplete prefix targets")
        require(row["prefix_length_histogram"] == {"1": 0, "2": 0, "3": 0, "4": row["matched_branch_count"]}, f"{setting}: impossible sealed prefix lengths")
        horizon_histogram = row.get("logged_selected_horizon_histogram")
        require(
            isinstance(horizon_histogram, dict)
            and set(horizon_histogram) == {"4", "8", "16", "32", "64"}
            and all(type(value) is int and value >= 0 for value in horizon_histogram.values())
            and sum(horizon_histogram.values()) == row["matched_branch_count"]
            and SHA256.fullmatch(str(row.get("sorted_logged_selected_horizons_sha256", ""))) is not None,
            f"{setting}: logged continuation horizon evidence differs",
        )
        require(row["prefix_action_step_count"] == 4 * row["matched_branch_count"], f"{setting}: action-step denominator differs")
        require(row["prefix_action_scalar_count"] == row["prefix_action_step_count"] * row["action_dim"], f"{setting}: action-scalar denominator differs")
        require(all(SHA256.fullmatch(str(value)) for key, value in row.items() if key.endswith("sha256")), f"{setting}: malformed prefix-target hash")
    return lock


def _validate_resolved_config_lock(manifest: Mapping[str, Any]) -> dict[str, Any]:
    lock = read_json(RESOLVED_CONFIG_LOCK_PATH)
    claimed = lock.get("artifact_sha256")
    body = dict(lock)
    body.pop("artifact_sha256", None)
    require(claimed == stable_hash(body), "resolved-config artifact hash differs")
    binding = manifest["resolved_config_contract"]
    require(claimed == binding["artifact_sha256"], "manifest resolved-config artifact differs")
    require(file_sha256(PACKAGE_DIR / binding["audit_source"]) == binding["source_sha256"] == lock["source_sha256"], "resolved-config auditor source differs")
    require(lock["direct_entrypoint"] == binding["direct_entrypoint"] == "scripts/train.py", "direct trainer entrypoint differs")
    require(lock["trainer_code_fingerprint"] == manifest["core_binding"]["trainer_code_fingerprint"], "resolved config/core differs")
    rows = lock.get("matrix") or []
    require(len(rows) == binding["cell_count"] == 20 and [row.get("index") for row in rows] == list(range(20)), "resolved-config matrix differs")
    cells = expand_matrix(manifest)
    for cell, row in zip(cells, rows, strict=True):
        require((row["setting_id"], row["arm_id"], row["seed"]) == (cell.setting, cell.arm, cell.seed), f"cell{cell.index}: resolved-config identity differs")
        require(stable_hash(row["resolved_config"]) == row["resolved_config_sha256"], f"cell{cell.index}: resolved config hash differs")
        require(
            row["resolved_config"].get("campaign_id") == manifest["campaign_id"]
            and row["resolved_config"].get("run_root") == manifest["paths"]["run_root"],
            f"cell{cell.index}: resolved campaign/run-root identity differs",
        )
        argv = row.get("trainer_argv_repo_relative")
        expected_hydra_directory = str(run_directory(manifest, cell) / "hydra")
        require(
            isinstance(argv, list)
            and len(argv) >= 3
            and argv[0] == manifest["paths"]["python"]
            and argv[1] == "scripts/train.py"
            and stable_hash(argv) == row["trainer_argv_sha256"],
            f"cell{cell.index}: repo-relative trainer argv differs",
        )
        require(
            f"+campaign_id={manifest['campaign_id']}" in argv
            and f"run_root={manifest['paths']['run_root']}" in argv
            and f"hydra.run.dir={expected_hydra_directory}" in argv,
            f"cell{cell.index}: resolved launch namespace differs",
        )
        require(row["resolved_config"].get("run_name") is None and row["resolved_config"].get("resume") == "auto", f"cell{cell.index}: run-name/resume parity differs")
    for setting in SETTINGS:
        for seed in SEEDS:
            pair = [row for row in rows if row["setting_id"] == setting and row["seed"] == seed]
            require(pair[0]["resolved_config_without_prefix_weights_sha256"] == pair[1]["resolved_config_without_prefix_weights_sha256"], f"{setting}/seed{seed}: resolved configs differ beyond weights")
    return lock


def _validate_causal_parity_lock(manifest: Mapping[str, Any]) -> dict[str, Any]:
    lock = read_json(CAUSAL_PARITY_LOCK_PATH)
    claimed = lock.get("artifact_sha256")
    body = dict(lock)
    body.pop("artifact_sha256", None)
    require(claimed == stable_hash(body), "causal-parity artifact hash differs")
    binding = manifest["causal_parity_contract"]
    require(claimed == binding["artifact_sha256"], "manifest causal-parity artifact differs")
    require(
        file_sha256(PACKAGE_DIR / binding["audit_source"])
        == binding["source_sha256"]
        == lock["source_sha256"],
        "causal-parity auditor source differs",
    )
    audit_manifest_input = {
        key: manifest[key] for key in CAUSAL_AUDIT_MANIFEST_INPUT_KEYS
    }
    require(
        stable_hash(audit_manifest_input)
        == binding["audit_manifest_input_sha256"]
        == lock["audit_manifest_input_sha256"],
        "causal-parity manifest input differs",
    )
    require(lock.get("package_protocol_claimed") is False, "causal audit claims a circular protocol")
    require(
        lock["runtime_sha256"] == binding["runtime_sha256"]
        and lock["runtime_sha256"] == manifest["compatible_v2_recipe_input"]["recipe_runtime_sha256"],
        "causal-parity runtime differs",
    )
    require(
        lock["trainer_code_fingerprint"] == manifest["core_binding"]["trainer_code_fingerprint"]
        and lock["weight_audit_artifact_sha256"] == manifest["weight_audit"]["artifact_sha256"]
        and lock["prefix_target_artifact_sha256"] == manifest["prefix_target_contract"]["artifact_sha256"]
        and lock["resolved_config_artifact_sha256"] == manifest["resolved_config_contract"]["artifact_sha256"],
        "causal-parity upstream binding differs",
    )
    require(
        lock["live_output_fingerprint_before"] == lock["live_output_fingerprint_after"],
        "causal audit changed live output metadata",
    )
    expected_fields = [
        "launch_without_allowed_deltas_sha256",
        "resolved_config_without_prefix_weights_sha256",
        "controlled_cpu_scratch_parameters_sha256",
        "data_identity_sha256",
        "sampler_identity_sha256",
        "controlled_cpu_pre_forward_rng_sha256",
        "fixed_validation_batch_sha256",
        "raw_prefix_targets_sha256",
        "raw_prefix_artifacts_sha256",
        "raw_prefix_telemetry_sha256",
        "raw_prefix_values",
    ]
    expected_pairs = [
        (setting, seed) for setting in SETTINGS for seed in SEEDS
    ]
    rows = lock.get("pairs") or []
    require(
        len(rows) == 10
        and [(row.get("setting_id"), row.get("seed")) for row in rows]
        == expected_pairs,
        "causal-parity pair matrix differs",
    )
    expected_weights = {
        "executable_prefix_action": manifest["arms"][1]["executable_prefix_weights"]["action"],
        "executable_prefix_latent": manifest["arms"][1]["executable_prefix_weights"]["latent"],
        "executable_prefix_endpoint": manifest["arms"][1]["executable_prefix_weights"]["endpoint"],
    }
    config_rows = read_json(RESOLVED_CONFIG_LOCK_PATH)["matrix"]
    for row in rows:
        require(row.get("parity_fields") == expected_fields, "causal parity-field set differs")
        require(len(row["parity_fields"]) == len(set(row["parity_fields"])), "causal parity fields duplicate")
        require(
            row.get("allowed_environment_differences")
            == ["TREEWM_CONFIG_SHA256", "TREEWM_PROTOCOL_SHA256", "TREEWM_RUN_NAME", "WANDB_RUN_ID"],
            "causal launch environment deltas differ",
        )
        arms = row.get("arms") or {}
        require(list(arms) == list(ARMS), "causal arm ordering differs")
        require(
            all(arms["GS"][field] == arms["GSEP"][field] for field in expected_fields),
            "causal parity value differs",
        )
        setting, seed = row["setting_id"], row["seed"]
        for arm in ARMS:
            cell = next(
                value for value in expand_matrix(manifest)
                if value.setting == setting and value.seed == seed and value.arm == arm
            )
            require(
                arms[arm]["resolved_config_sha256"]
                == config_rows[cell.index]["resolved_config_sha256"],
                "causal/resolved config identity differs",
            )
            require(arms[arm]["controlled_cpu_parameters_unchanged"] is True, "causal audit mutated parameters")
        require(
            set(arms["GS"]["effective_prefix_weights"].values()) == {0.0}
            and set(arms["GS"]["effective_prefix_values"].values()) == {0.0},
            "causal control is not monitor-only",
        )
        require(arms["GSEP"]["effective_prefix_weights"] == expected_weights, "causal treatment weights differ")
    return lock


def _validate_core(manifest: Mapping[str, Any], lock: Mapping[str, Any], repo: Path) -> None:
    binding = manifest["core_binding"]
    paths = {
        "trainer_sha256": repo / "scripts/train.py",
        "executable_loss_sha256": repo / "treewm/losses/executable_prefix.py",
        "action_projection_sha256": repo / "treewm/planning/action_execution.py",
        "objective_config_sha256": repo / "configs/experiment/treewm_v2_grounded_executable_prefix_pilot_v1.yaml",
    }
    for key, path in paths.items():
        require(file_sha256(path) == binding[key], f"core source drift: {path.relative_to(repo)}")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from treewm.utils.provenance import trainer_code_fingerprint

    live = trainer_code_fingerprint(repo)["manifest_sha256"]
    require(live == binding["trainer_code_fingerprint"], "trainer code fingerprint drift")
    require(live == lock["source_sha256"]["trainer_code_fingerprint"], "audit/core fingerprint differs")


def _validate_launch7_negative_provenance(
    manifest: Mapping[str, Any], repo: Path
) -> None:
    prior = manifest["superseded_launches"][-1]
    require(
        prior["campaign_id"]
        == "treewm-executable-prefix-repair-pilot-v1-launch7",
        "Launch7 superseded position differs",
    )
    binding = prior["negative_provenance"]
    path = repo / PACKAGE_RELATIVE / str(binding["path"])
    info = path.lstat()
    require(
        stat.S_ISREG(info.st_mode) and not path.is_symlink(),
        "Launch7 negative provenance is not a regular nonsymlink file",
    )
    require(
        file_sha256(path) == binding["raw_sha256"],
        "Launch7 negative provenance raw hash differs",
    )
    value = read_json(path)
    require(
        stable_hash(value) == binding["canonical_sha256"],
        "Launch7 negative provenance canonical hash differs",
    )
    require(
        value.get("schema_version") == 1
        and value.get("status") == "terminal_negative_provenance_frozen"
        and value.get("campaign_id") == prior["campaign_id"]
        and value["immutable_identity"]["source_commit"] == prior["source_commit"]
        and value["immutable_identity"]["submission_sha256"]
        == prior["submission_sha256"]
        and value["immutable_identity"]["receipt_sha256"]
        == prior["receipt_sha256"]
        and value["immutable_identity"]["manifest_canonical_sha256"]
        == prior["manifest_canonical_sha256"],
        "Launch7 negative provenance identity differs",
    )
    scheduler_payload = {
        "schema_version": 1,
        "fields": value["scheduler_terminal_rows_schema"].split("|"),
        "rows": value["scheduler_terminal_rows"],
    }
    require(
        stable_hash(scheduler_payload)
        == value["scheduler_terminal_observation"][
            "canonical_reduced_rows_sha256"
        ]
        == "ba0af0501dda20c6697333311c3c7c9eea2d3e586f348057e2d4aae811afaeac",
        "Launch7 scheduler terminal wrapper differs",
    )
    require(
        len(value["scheduler_terminal_rows"]) == binding["scheduler_terminal_rows"]
        and len(value["ready_checkpoints"]) == binding["ready_checkpoint_cells"]
        and len(value["completed_cells"]) == binding["completed_cells"]
        and value["reporter"]["started"] is binding["report_started"]
        and value["terminal_zero_active_evidence"]["active_job_count"]
        == binding["active_scheduler_jobs_after_terminal"]
        and value["scientific_conclusion"]["reuse_allowed"] is False
        and value["scientific_conclusion"]["resume_allowed"] is False
        and value["scientific_conclusion"]["retry_allowed"] is False
        and value["scientific_conclusion"]["recovery_allowed"] is False,
        "Launch7 terminal/no-reuse census differs",
    )


def _validate_canary1_negative_provenance(
    manifest: Mapping[str, Any], repo: Path
) -> None:
    attempts = manifest["launch_contract"]["real_gpu_two_wave_canary"].get(
        "failed_attempts"
    )
    require(
        exact_json_equal(attempts, FAILED_CANARY_ATTEMPTS),
        "failed real-GPU canary binding differs",
    )
    binding = FAILED_CANARY_ATTEMPTS[0]
    path = repo / PACKAGE_RELATIVE / binding["path"]
    info = path.lstat()
    require(
        stat.S_ISREG(info.st_mode) and not path.is_symlink(),
        "canary1 negative provenance is not a regular nonsymlink file",
    )
    require(
        file_sha256(path) == binding["raw_sha256"],
        "canary1 negative provenance raw hash differs",
    )
    value = read_json(path)
    require(
        stable_hash(value) == binding["canonical_sha256"],
        "canary1 negative provenance canonical hash differs",
    )
    require(
        set(value)
        == {
            "schema_version",
            "status",
            "campaign_id",
            "canary_attempt",
            "scientific",
            "observed_at_utc",
            "scope",
            "canonical_reconstruction",
            "immutable_identity",
            "terminal_state_file_census",
            "durable_controller_outcome",
            "absent_durable_artifacts",
            "retained_wave1_scheduler_observation",
            "scheduler_terminal_observation",
            "scheduler_terminal_rows_schema",
            "scheduler_terminal_rows",
            "root_cause",
            "negative_conclusion",
        }
        and type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and value["status"] == binding["status"]
        and value["campaign_id"] == manifest["campaign_id"]
        and value["canary_attempt"] == binding["attempt"]
        and value["scientific"] is False
        and value["observed_at_utc"] == "2026-08-28T20:13:16Z"
        and value["scope"]
        == (
            "Terminal negative evidence for one non-scientific real-GPU topology "
            "canary. It does not supersede the still-unsubmitted Launch8 scientific "
            "campaign and grants no authority to reuse the failed canary namespace."
        ),
        "canary1 negative provenance envelope differs",
    )
    require(
        exact_json_equal(
            value["canonical_reconstruction"],
            {
                "canonical_json": (
                    "UTF-8 JSON with ensure_ascii=true, allow_nan=false, object keys "
                    "sorted lexicographically, separators comma and colon, and no "
                    "trailing newline."
                ),
                "state_file_map_payload": (
                    "Hash canonical JSON of {schema_version:1,files:"
                    "{state_root_relative_path:{mode,sha256,size}}}; mode is a "
                    "four-digit octal string, size is the exact raw byte count, and "
                    "sha256 is the raw-file SHA-256."
                ),
                "scheduler_rows_payload": (
                    "Hash canonical JSON of {schema_version:1,fields:"
                    "scheduler_terminal_rows_schema.split('|'),rows:"
                    "scheduler_terminal_rows}."
                ),
                "scope": (
                    "The state-file map is a point-in-time read-only census of the "
                    "preserved failed canary root; the repository artifact, not the "
                    "writable owner lock, is the frozen evidence anchor. The embedded "
                    "sacct rows are independently reconstructable from their preserved "
                    "raw-row representation. The retained scontrol digest and parsed "
                    "fields are contemporaneous operator-recorded, non-reconstructable "
                    "corroborative metadata only. Package validation reads only this "
                    "embedded artifact and never reads or imports the failed state root."
                ),
            },
        ),
        "canary1 canonical reconstruction differs",
    )
    identity = value["immutable_identity"]
    require(
        exact_json_equal(
            identity,
            {
                "source_commit_protocol_equivalent": binding["source_commit"],
                "package_protocol_sha256": binding["source_protocol_sha256"],
                "protocol_lock_raw_sha256": (
                    "968e338ece347698ce10d8df9d52b1afe6ba86dcdbab383e2fa6e6de23e0e967"
                ),
                "manifest_raw_sha256": (
                    "1976216ad9630509d2a91f023d3ffa88b9e52558a119b805c0bbc07997ac3704"
                ),
                "manifest_canonical_sha256": (
                    "782baa33c8129e90be8599074b80f93fdaed8e2a4ffe73aed8fc9f8ab0bc41cc"
                ),
                "state_root": binding["state_root"],
                "canary_token": binding["canary_token"],
                "state_root_relative": (
                    "outputs/exp23-launch8-two-wave-canaries/"
                    "exp23-launch8-two-wave-canary-af348af-b4403218"
                ),
                "scheduler_comment": (
                    "treewm-exp23-canary:e09ce7d5a0cef1b0"
                ),
                "controller_identity_sha256": (
                    "4a79c2dfa750ca682cfdbab686db3b0e30e870af53851dcacab9e6aa68afb938"
                ),
                "scheduler_control_plane_sha256": (
                    "b631db129a9330a869436a1487fe67d99a8692ba73eb3b1a0156de4adc349731"
                ),
                "source_sha256": {
                    "canary_gpu.slurm": (
                        "cdfa25456ea26544450ff00681590adf1e81ef03eeb1434597c0777f6552ec14"
                    ),
                    "canary_report.slurm": (
                        "6d4561c7b8462fde2cd7b8666fa9f2b8d377aa30bfc44195bef6924548df1c0b"
                    ),
                    "canary_worker.py": (
                        "f0c20908a481382a0734a003a5facf8709c02e4b4763076d40d2cd7b0a50d87f"
                    ),
                    "two_wave_canary.py": (
                        "2fba2cd0c477317e49545b0ed3f6af7f2c590493fd43e888c183ce13fb7a2823"
                    ),
                },
            },
        ),
        "canary1 immutable identity differs",
    )
    census = value["terminal_state_file_census"]
    files = census.get("files")
    require(
        set(census)
        == {
            "state_file_count",
            "state_file_bytes",
            "directory_paths",
            "directory_modes",
            "logs_file_count",
            "symlink_count",
            "special_file_count",
            "state_file_map_canonical_sha256",
            "files",
        }
        and isinstance(files, dict)
        and type(census.get("state_file_count")) is int
        and census["state_file_count"] == len(files) == 13
        and type(census.get("state_file_bytes")) is int
        and census["state_file_bytes"]
        == sum(row["size"] for row in files.values())
        == 290_380
        and exact_json_equal(census.get("directory_paths"), [".", "logs", "source"])
        and exact_json_equal(
            census.get("directory_modes"),
            {".": "0700", "logs": "0700", "source": "0555"},
        )
        and type(census.get("logs_file_count")) is int
        and census["logs_file_count"] == 0
        and type(census.get("symlink_count")) is int
        and census["symlink_count"] == 0
        and type(census.get("special_file_count")) is int
        and census["special_file_count"] == 0,
        "canary1 state census summary differs",
    )
    expected_file_names = {
        ".CANARY_CONTROLLER.lock",
        "CANARY_ABORTED.json",
        "CANARY_CONTROLLER_IDENTITY.json",
        "CANARY_RECOVERY_CANCELLED.json",
        "CANARY_RECOVERY_CANCEL_CALLING_0000.json",
        "CANARY_RECOVERY_CANCEL_RESULT_0000.json",
        "CANARY_WAVE0_CALLING.json",
        "CANARY_WAVE0_SUBMITTED.json",
        "CANARY_WAVE1_CALLING.json",
        "source/canary_gpu.slurm",
        "source/canary_report.slurm",
        "source/canary_worker.py",
        "source/two_wave_canary.py",
    }
    require(set(files) == expected_file_names, "canary1 state file names differ")
    require(
        all(
            set(row) == {"mode", "sha256", "size"}
            and isinstance(row["mode"], str)
            and re.fullmatch(r"0[0-7]{3}", row["mode"]) is not None
            and isinstance(row["sha256"], str)
            and SHA256.fullmatch(row["sha256"]) is not None
            and type(row["size"]) is int
            and row["size"] >= 0
            for row in files.values()
        ),
        "canary1 state file row differs",
    )
    state_map_payload = {"schema_version": 1, "files": files}
    require(
        stable_hash(state_map_payload)
        == census["state_file_map_canonical_sha256"]
        == binding["state_file_map_canonical_sha256"],
        "canary1 state file map differs",
    )
    require(
        files["CANARY_CONTROLLER_IDENTITY.json"]["sha256"]
        == identity["controller_identity_sha256"]
        and exact_json_equal(
            {
                name: files[f"source/{name}"]["sha256"]
                for name in (
                    "two_wave_canary.py",
                    "canary_worker.py",
                    "canary_gpu.slurm",
                    "canary_report.slurm",
                )
            },
            identity["source_sha256"],
        ),
        "canary1 state census/identity hash links differ",
    )
    outcome = value["durable_controller_outcome"]
    require(
        set(outcome)
        == {
            "abort_status",
            "abort_error",
            "known_job_ids",
            "scheduler_assigned_job_ids_by_role",
            "durably_submitted_marker_ids_by_role",
            "cancellation_authority_job_ids_by_role",
            "calling_intent_sha256_by_role",
            "submitted_record_sha256_by_role",
            "wave0_accepted_hold",
            "authorization_committed",
            "submission_receipt_committed",
            "ready_to_release_committed",
            "release_calling_committed",
            "wave0_released",
            "report_calling_committed",
            "report_submitted",
            "canary_report_committed",
            "recovery_status",
            "recovery_sha256",
            "cancel_calling_sha256",
            "cancel_result_sha256",
            "scheduler_calls_in_recovery",
            "new_jobs_created_in_recovery",
            "post_cancel_active_job_ids_by_role",
            "cancellation_error",
            "reconciliation_errors",
        }
        and outcome["abort_status"] == "two_wave_gpu_canary_aborted"
        and outcome["abort_error"]
        == "SubmissionError('accepted wave1 dependency differs')"
        and exact_json_equal(outcome["known_job_ids"], ["33285485", "33285486"])
        and exact_json_equal(
            outcome["scheduler_assigned_job_ids_by_role"],
            binding["job_ids_by_role"],
        )
        and exact_json_equal(
            outcome["durably_submitted_marker_ids_by_role"],
            {"wave0": ["33285485"], "wave1": [], "report": []},
        )
        and exact_json_equal(
            outcome["cancellation_authority_job_ids_by_role"],
            binding["job_ids_by_role"],
        )
        and outcome["recovery_status"]
        == "canary_recovered_terminal_after_cancel_attempts"
        and type(outcome["scheduler_calls_in_recovery"]) is int
        and outcome["scheduler_calls_in_recovery"] == 9
        and type(outcome["new_jobs_created_in_recovery"]) is int
        and outcome["new_jobs_created_in_recovery"] == 0
        and exact_json_equal(
            outcome["post_cancel_active_job_ids_by_role"],
            {"wave0": [], "wave1": [], "report": []},
        )
        and outcome["cancellation_error"] is None
        and exact_json_equal(outcome["reconciliation_errors"], []),
        "canary1 durable controller outcome differs",
    )
    require(
        exact_json_equal(
            {
                key: outcome.get(key)
                for key in (
                    "authorization_committed",
                    "submission_receipt_committed",
                    "ready_to_release_committed",
                    "release_calling_committed",
                    "wave0_released",
                    "report_calling_committed",
                    "report_submitted",
                    "canary_report_committed",
                )
            },
            {
                "authorization_committed": False,
                "submission_receipt_committed": False,
                "ready_to_release_committed": False,
                "release_calling_committed": False,
                "wave0_released": False,
                "report_calling_committed": False,
                "report_submitted": False,
                "canary_report_committed": False,
            },
        ),
        "canary1 durable lifecycle outcome differs",
    )
    require(
        exact_json_equal(
            outcome["wave0_accepted_hold"],
            {
                "state": "PENDING",
                "reason": "JobHeldUser",
                "stdout_bytes": 2000,
                "stdout_sha256": (
                    "a11edd23eabf58f34049b670b49d6bbe9478e7b80e3b8e1fb26680a256416d3f"
                ),
            },
        ),
        "canary1 durable wave0 hold evidence differs",
    )
    require(
        outcome["calling_intent_sha256_by_role"]["wave0"]
        == files["CANARY_WAVE0_CALLING.json"]["sha256"]
        and outcome["calling_intent_sha256_by_role"]["wave1"]
        == files["CANARY_WAVE1_CALLING.json"]["sha256"]
        and outcome["submitted_record_sha256_by_role"]["wave0"]
        == files["CANARY_WAVE0_SUBMITTED.json"]["sha256"]
        and outcome["submitted_record_sha256_by_role"]["wave1"] is None
        and outcome["submitted_record_sha256_by_role"]["report"] is None
        and outcome["recovery_sha256"]
        == files["CANARY_RECOVERY_CANCELLED.json"]["sha256"]
        and outcome["cancel_calling_sha256"]
        == files["CANARY_RECOVERY_CANCEL_CALLING_0000.json"]["sha256"]
        and outcome["cancel_result_sha256"]
        == files["CANARY_RECOVERY_CANCEL_RESULT_0000.json"]["sha256"],
        "canary1 durable hash links differ",
    )
    require(
        exact_json_equal(
            value["absent_durable_artifacts"],
            [
                "CANARY_WAVE1_SUBMITTED.json",
                "CANARY_REPORT_CALLING.json",
                "CANARY_REPORT_SUBMITTED.json",
                "CANARY_AUTHORIZATION.json",
                "CANARY_SUBMISSION_RECEIPT.json",
                "CANARY_READY_TO_RELEASE.json",
                "CANARY_WAVE0_RELEASE_CALLING.json",
                "CANARY_WAVE0_RELEASED.json",
                "WAVE0_READY.json",
                "WAVE1_COMPLETE.json",
                "CANARY_REPORT.json",
            ],
        ),
        "canary1 absent durable artifact census differs",
    )
    retained = value["retained_wave1_scheduler_observation"]
    require(
        exact_json_equal(
            retained,
            {
                "evidence_scope": (
                    "Contemporaneous operator-recorded scontrol observation made after "
                    "controller failure and before scheduler purge. The raw stdout was "
                    "not preserved in the canary root or package; its byte count and "
                    "SHA-256 are opaque metadata only. The parsed fields cannot be "
                    "reconstructed from preserved bytes and are non-authoritative "
                    "corroboration, not cryptographically authenticated scheduler evidence."
                ),
                "observed_at_utc": "2026-08-28T20:04:37.342Z",
                "environment": {
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "LANG": "C",
                    "LC_ALL": "C",
                    "SLURM_CONF": (
                        "/cm/shared/apps/slurm/var/etc/cs-oci-ord/slurm.conf"
                    ),
                },
                "argv": [
                    "/usr/local/bin/scontrol",
                    "show",
                    "job",
                    "33285486",
                    "--oneliner",
                ],
                "returncode": 0,
                "stdout_bytes": 2136,
                "stdout_sha256": (
                    "75d00d04459a341ae75f9661e236099d62076eee459de2e9796b43d3f398737f"
                ),
                "stderr_bytes": 0,
                "stderr_sha256": hashlib.sha256(b"").hexdigest(),
                "raw_stdout_preserved_in_state_root": False,
                "parsed_fields": {
                    "JobId": "33285486",
                    "JobName": "exp23-launch8-canary-e09ce7d5a0cef1b0-wave1",
                    "JobState": "CANCELLED",
                    "Reason": "Dependency",
                    "Dependency": "afterok:33285485(unfulfilled)",
                    "RunTime": "00:00:00",
                    "StartTime": "2026-08-28T20:03:08",
                    "EndTime": "2026-08-28T20:03:08",
                    "NodeList": "",
                    "Comment": "treewm-exp23-canary:e09ce7d5a0cef1b0",
                    "KillOInInvalidDependent": "Yes",
                },
            },
        ),
        "canary1 retained scalar dependency observation differs",
    )
    expected_rows = [
        (
            "33285485|exp23-launch8-canary-e09ce7d5a0cef1b0-wave0|"
            "CANCELLED by 147230|0:0|00:00:00|2026-08-28T20:03:08|"
            "2026-08-28T20:03:08|None assigned|"
            "treewm-exp23-canary:e09ce7d5a0cef1b0"
        ),
        (
            "33285486|exp23-launch8-canary-e09ce7d5a0cef1b0-wave1|"
            "CANCELLED by 147230|0:0|00:00:00|2026-08-28T20:03:08|"
            "2026-08-28T20:03:08|None assigned|"
            "treewm-exp23-canary:e09ce7d5a0cef1b0"
        ),
    ]
    require(
        exact_json_equal(value["scheduler_terminal_rows"], expected_rows),
        "canary1 scheduler terminal rows differ",
    )
    scheduler_payload = {
        "schema_version": 1,
        "fields": value["scheduler_terminal_rows_schema"].split("|"),
        "rows": value["scheduler_terminal_rows"],
    }
    scheduler_raw = ("\n".join(value["scheduler_terminal_rows"]) + "\n").encode(
        "ascii"
    )
    scheduler = value["scheduler_terminal_observation"]
    require(
        set(scheduler)
        == {
            "observed_at_utc",
            "environment",
            "argv",
            "returncode",
            "stdout_bytes",
            "stdout_sha256",
            "stderr_bytes",
            "stderr_sha256",
            "row_count",
            "canonical_reduced_rows_sha256",
        }
        and scheduler["observed_at_utc"] == "2026-08-28T20:13:16Z"
        and exact_json_equal(
            scheduler["environment"],
            {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "LANG": "C",
                "LC_ALL": "C",
                "SLURM_CONF": (
                    "/cm/shared/apps/slurm/var/etc/cs-oci-ord/slurm.conf"
                ),
            },
        )
        and value["scheduler_terminal_rows_schema"]
        == "JobIDRaw|JobName|State|ExitCode|Elapsed|Start|End|NodeList|Comment"
        and scheduler["argv"]
        == [
            "/usr/local/bin/sacct",
            "-X",
            "-n",
            "-j",
            "33285485,33285486",
            "-o",
            "JobIDRaw,JobName,State,ExitCode,Elapsed,Start,End,NodeList,Comment",
            "-P",
        ]
        and type(scheduler["returncode"]) is int
        and scheduler["returncode"] == 0
        and type(scheduler["row_count"]) is int
        and scheduler["row_count"] == len(expected_rows) == binding["scheduler_terminal_rows"]
        and type(scheduler["stdout_bytes"]) is int
        and scheduler["stdout_bytes"] == len(scheduler_raw) == 354
        and scheduler["stdout_sha256"] == hashlib.sha256(scheduler_raw).hexdigest()
        and type(scheduler["stderr_bytes"]) is int
        and scheduler["stderr_bytes"] == 0
        and scheduler["stderr_sha256"] == hashlib.sha256(b"").hexdigest()
        and scheduler["canonical_reduced_rows_sha256"]
        == stable_hash(scheduler_payload)
        == "a74fcb1e76eb07d5372f0d8b7f0c80729ca4f64c541d66805e9cecea251ffe56",
        "canary1 scheduler terminal evidence differs",
    )
    root_cause = value["root_cause"]
    require(
        exact_json_equal(
            root_cause,
            {
                "classification": "canary_adapter_dependency_validation_bug",
                "observed_scheduler_dependency": "afterok:33285485(unfulfilled)",
                "scheduler_dependency_evidence_scope": (
                    "contemporaneous_operator_recorded_non_reconstructable_"
                    "corroboration"
                ),
                "rejected_validator_dependency": (
                    "afterok:33285485_*(unfulfilled)"
                ),
                "explanation": (
                    "The canary submits scalar wave0 and wave1 jobs. A contemporaneous "
                    "non-reconstructable operator observation recorded the "
                    "scalar-predecessor afterok spelling, while the durable controller "
                    "abort records that its shared validator rejected the accepted wave1 "
                    "dependency. Source inspection identifies the validator's incorrect "
                    "requirement for the production-array _* spelling. The parsed "
                    "scontrol fields are corroborative only; the preserved root "
                    "independently proves that failure occurred before sealing wave1 "
                    "submission or authorizing release."
                ),
                "scheduler_or_topology_drift": False,
                "scientific_runtime_defect": False,
            },
        ),
        "canary1 root-cause classification differs",
    )
    conclusion = value["negative_conclusion"]
    require(
        conclusion.get("required_successor_policy")
        == (
            "Any successor canary must use a fresh state root and token, submit fresh "
            "jobs, and run only from a newly sealed package protocol after the "
            "scalar-predecessor validator repair. No file, job identity, receipt, "
            "source snapshot, or result from canary1 may be read as successor input."
        ),
        "canary1 required successor policy differs",
    )
    require(
        exact_json_equal(
            conclusion,
            {
                "wave0_released": False,
                "authorization_published": False,
                "receipt_published": False,
                "report_job_submitted": False,
                "report_published": False,
                "gpu_runtime_seconds": 0,
                "allocated_node_count": 0,
                "scientific_outputs_created": False,
                "active_scheduler_jobs_after_recovery": 0,
                "reuse_allowed": False,
                "resume_allowed": False,
                "retry_allowed": False,
                "recovery_allowed": False,
                "result_consumption_allowed": False,
                "required_successor_policy": (
                    "Any successor canary must use a fresh state root and token, submit "
                    "fresh jobs, and run only from a newly sealed package protocol after "
                    "the scalar-predecessor validator repair. No file, job identity, "
                    "receipt, source snapshot, or result from canary1 may be read as "
                    "successor input."
                ),
            },
        )
        and conclusion["gpu_runtime_seconds"] == binding["gpu_runtime_seconds"]
        and conclusion["allocated_node_count"] == binding["allocated_node_count"]
        and conclusion["active_scheduler_jobs_after_recovery"]
        == binding["active_scheduler_jobs_after_recovery"],
        "canary1 terminal/no-reuse conclusion differs",
    )
    require(
        outcome["authorization_committed"]
        is conclusion["authorization_published"]
        and outcome["submission_receipt_committed"]
        is conclusion["receipt_published"]
        and outcome["wave0_released"] is conclusion["wave0_released"]
        and outcome["report_submitted"] is conclusion["report_job_submitted"]
        and outcome["canary_report_committed"] is conclusion["report_published"],
        "canary1 durable lifecycle/conclusion binding differs",
    )
    require(
        all(
            conclusion[key] is binding[key]
            for key in (
                "wave0_released",
                "authorization_published",
                "receipt_published",
                "report_job_submitted",
                "report_published",
                "reuse_allowed",
                "resume_allowed",
                "retry_allowed",
                "recovery_allowed",
                "result_consumption_allowed",
            )
        ),
        "canary1 manifest conclusion binding differs",
    )


def _validate_canary2_acceptance_provenance_impl(
    manifest: Mapping[str, Any], repo: Path
) -> None:
    attempts = manifest["launch_contract"]["real_gpu_two_wave_canary"].get(
        "accepted_attempts"
    )
    require(
        exact_json_equal(attempts, ACCEPTED_CANARY_ATTEMPTS),
        "successful real-GPU canary binding differs",
    )
    binding = ACCEPTED_CANARY_ATTEMPTS[0]
    require(
        exact_json_equal(
            manifest["launch_contract"]["real_gpu_two_wave_canary"].get(
                "production_authorization_evidence"
            ),
            {
                "required": True,
                "satisfied": True,
                "attempt": binding["attempt"],
                "path": binding["path"],
                "raw_sha256": binding["raw_sha256"],
                "canonical_sha256": binding["canonical_sha256"],
                "source_protocol_sha256": binding["source_protocol_sha256"],
                "report_raw_sha256": binding["report_raw_sha256"],
                "artifact_evidence_consumption_allowed": True,
                "scientific_runtime_input_consumption_allowed": False,
                "accepted_compute_runtime_source_sha256": {
                    "canary_gpu.slurm": (
                        "cdfa25456ea26544450ff00681590adf1e81ef03eeb1434597c0777f6552ec14"
                    ),
                    "canary_report.slurm": (
                        "6d4561c7b8462fde2cd7b8666fa9f2b8d377aa30bfc44195bef6924548df1c0b"
                    ),
                    "canary_worker.py": (
                        "4bcbaab866538c527f7a894d015d8b8bfbc505b2eb8e3596e56d419cfa4b3a2d"
                    ),
                },
                "accepted_controller_sha256": ACCEPTED_CANARY_CONTROLLER_SHA256,
                "post_acceptance_current_controller_sha256": (
                    ACCEPTED_CANARY_CURRENT_CONTROLLER_SHA256
                ),
                "post_acceptance_current_source_sha256": (
                    ACCEPTED_CANARY_CURRENT_SOURCE_SHA256
                ),
                "post_acceptance_change_scope": (
                    ACCEPTED_CANARY_POST_ACCEPTANCE_CHANGE_SCOPE
                ),
                "canary_rerun_required": False,
            },
        ),
        "production canary-acceptance authorization differs",
    )
    package = repo / PACKAGE_RELATIVE
    runtime_source_bytes = {
        name: _regular_file_bytes(
            package / name,
            f"post-acceptance runtime source {name}",
            max_bytes=4 * 1024 * 1024,
        )
        for name in ACCEPTED_CANARY_CURRENT_SOURCE_SHA256
    }
    require(
        all(
            hashlib.sha256(runtime_source_bytes[name]).hexdigest() == expected
            for name, expected in (
                ACCEPTED_CANARY_CURRENT_SOURCE_SHA256.items()
            )
        ),
        "post-acceptance runtime source bytes differ",
    )
    failed_roots = {row["state_root"] for row in FAILED_CANARY_ATTEMPTS}
    failed_tokens = {row["canary_token"] for row in FAILED_CANARY_ATTEMPTS}
    failed_job_ids = {
        job_id
        for row in FAILED_CANARY_ATTEMPTS
        for values in row["job_ids_by_role"].values()
        for job_id in values
    }
    accepted_job_ids = {
        job_id
        for values in binding["job_ids_by_role"].values()
        for job_id in values
    }
    require(
        binding["state_root"] not in failed_roots
        and binding["canary_token"] not in failed_tokens
        and accepted_job_ids.isdisjoint(failed_job_ids),
        "historical canary identities are not globally injective",
    )
    path = package / binding["path"]
    artifact_payload = _regular_file_bytes(
        path,
        "canary2 acceptance provenance",
        max_bytes=256 * 1024,
    )
    require(
        hashlib.sha256(artifact_payload).hexdigest() == binding["raw_sha256"],
        "canary2 acceptance provenance raw hash differs",
    )
    try:
        value = json.loads(
            artifact_payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ContractError(
            f"canary2 acceptance provenance JSON differs: {exc}"
        ) from exc
    require(
        isinstance(value, dict),
        "canary2 acceptance provenance JSON root differs",
    )
    require(
        stable_hash(value) == binding["canonical_sha256"],
        "canary2 acceptance provenance canonical hash differs",
    )
    require(
        set(value)
        == {
            "schema_version",
            "status",
            "campaign_id",
            "canary_attempt",
            "scientific",
            "observed_at_utc",
            "scope",
            "canonical_reconstruction",
            "immutable_identity",
            "terminal_state_file_census",
            "durable_topology",
            "runtime_result",
            "scheduler_terminal_rows_schema",
            "scheduler_terminal_rows",
            "scheduler_terminal_observation",
            "terminal_zero_active_evidence",
            "absent_durable_artifacts",
            "acceptance_conclusion",
        }
        and type(value["schema_version"]) is int
        and value["schema_version"] == 1
        and value["status"] == binding["status"]
        and value["campaign_id"] == manifest["campaign_id"]
        and value["canary_attempt"] == binding["attempt"]
        and value["scientific"] is False
        and value["observed_at_utc"] == "2026-08-28T23:28:04Z"
        and value["scope"]
        == (
            "Terminal positive evidence for the non-scientific Launch8 two-wave "
            "real-GPU topology canary. This evidence is a required production-"
            "authorization prerequisite, but the canary namespace, checkpoint, "
            "token, and scheduler job identities are permanently evidence-only "
            "and may never be consumed as scientific state."
        ),
        "canary2 acceptance provenance envelope differs",
    )
    require(
        exact_json_equal(
            value["canonical_reconstruction"],
            {
                "canonical_json": (
                    "UTF-8 JSON with ensure_ascii=true, allow_nan=false, object keys "
                    "sorted lexicographically, separators comma and colon, and no "
                    "trailing newline."
                ),
                "state_file_map_payload": (
                    "Hash canonical JSON of {schema_version:1,files:"
                    "{state_root_relative_path:{mode,sha256,size}}}; mode is a "
                    "four-digit octal string, size is the exact raw byte count, and "
                    "sha256 is the raw-file SHA-256."
                ),
                "scheduler_rows_payload": (
                    "Split every nonempty raw stdout line on '|' into a 14-string "
                    "array without dropping the final Comment field, then hash "
                    "canonical JSON of {schema_version:1,fields:"
                    "scheduler_terminal_rows_schema.split('|'),rows:"
                    "scheduler_terminal_rows}."
                ),
                "scheduler_raw_stdout_payload": (
                    "scheduler_terminal_observation.raw_stdout is the exact UTF-8 "
                    "stdout byte sequence, including its final newline; its size and "
                    "SHA-256 are independently bound."
                ),
                "owner_scheduler_census_payload": (
                    "Base64-decode and zlib-decompress terminal_zero_active_evidence."
                    "owner_active_scheduler_census.raw_stdout_zlib_base64 to recover "
                    "the exact owner-wide squeue stdout; verify its byte size and "
                    "SHA-256, parse every nonempty line as five pipe-separated "
                    "strings, and select rows whose JobName starts with exp23-launch8-"
                    "canary-b95869841048e511- or whose Comment equals treewm-exp23-"
                    "canary:b95869841048e511."
                ),
                "scope": (
                    "The state-file map is a point-in-time read-only census of the "
                    "preserved successful canary root. WAVE0_READY, WAVE1_COMPLETE, "
                    "CANARY_REPORT, sacct stdout, reduced sacct rows, and all zero-active "
                    "scheduler observations are embedded and independently "
                    "reconstructable. Large controller lifecycle records remain exact "
                    "immutable raw digests in the literal state map; held-before-"
                    "authorization and release-after-receipt semantics are independently "
                    "pre-freeze-audited projections, not reconstructions of embedded "
                    "record bytes. Package validation reads only this embedded artifact "
                    "and never reads, imports, resumes, or recovers the successful canary "
                    "root."
                ),
            },
        ),
        "canary2 canonical reconstruction differs",
    )
    identity = value["immutable_identity"]
    require(
        exact_json_equal(
            identity,
            {
                "source_commit_protocol_equivalent": binding["source_commit"],
                "source_commit_claimed_by_controller": False,
                "source_commit_evidence": (
                    "Independent clean HEAD, origin/main, and live-remote equality "
                    "plus byte comparison established "
                    "b688ea652e99479ed5d8c6eccd6c137d77f9e03f as the protocol-"
                    "equivalent source commit; the controller itself records only "
                    "the protocol and four source hashes."
                ),
                "package_protocol_sha256": binding["source_protocol_sha256"],
                "protocol_lock_raw_sha256": (
                    "785b52b5858876c52e03628da98f07d987e628b565d74c5ea5230cc0adb3dea1"
                ),
                "manifest_raw_sha256": (
                    "225f1d27fe65a8405679f4937ed2f434d4c8d4a84dd280ec0983fc2743633baa"
                ),
                "manifest_canonical_sha256": (
                    "e7b8f304d46e67a8e428e2aea851191f344578eef7cfe349a7d3a60cfd874088"
                ),
                "state_root": binding["state_root"],
                "state_root_relative": (
                    "outputs/exp23-launch8-two-wave-canaries/"
                    "exp23-launch8-two-wave-canary-b688ea6-79aaa2b6"
                ),
                "canary_token": binding["canary_token"],
                "scheduler_comment": (
                    "treewm-exp23-canary:b95869841048e511"
                ),
                "controller_identity_sha256": (
                    "8ab251408c6a326d3ff83c66869e01687bc60b9e4d9593a769d4c8532fcdd3fe"
                ),
                "scheduler_control_plane_sha256": (
                    "b631db129a9330a869436a1487fe67d99a8692ba73eb3b1a0156de4adc349731"
                ),
                "source_sha256": {
                    "canary_gpu.slurm": (
                        "cdfa25456ea26544450ff00681590adf1e81ef03eeb1434597c0777f6552ec14"
                    ),
                    "canary_report.slurm": (
                        "6d4561c7b8462fde2cd7b8666fa9f2b8d377aa30bfc44195bef6924548df1c0b"
                    ),
                    "canary_worker.py": (
                        "4bcbaab866538c527f7a894d015d8b8bfbc505b2eb8e3596e56d419cfa4b3a2d"
                    ),
                    "two_wave_canary.py": (
                        "b373d64f410f08d1bd692dd9bb9e732a18b06bf80b6b483c313d82d4a6499436"
                    ),
                },
            },
        ),
        "canary2 immutable identity differs",
    )
    production_evidence = manifest["launch_contract"][
        "real_gpu_two_wave_canary"
    ]["production_authorization_evidence"]
    require(
        exact_json_equal(
            production_evidence["accepted_compute_runtime_source_sha256"],
            {
                name: identity["source_sha256"][name]
                for name in (
                    "canary_gpu.slurm",
                    "canary_report.slurm",
                    "canary_worker.py",
                )
            },
        )
        and all(
            production_evidence["post_acceptance_current_source_sha256"][name]
            == identity["source_sha256"][name]
            for name in (
                "canary_gpu.slurm",
                "canary_report.slurm",
                "canary_worker.py",
            )
        )
        and production_evidence["accepted_controller_sha256"]
        == identity["source_sha256"]["two_wave_canary.py"]
        and production_evidence["post_acceptance_current_controller_sha256"]
        == production_evidence["post_acceptance_current_source_sha256"][
            "two_wave_canary.py"
        ],
        "canary2 accepted/final runtime source binding differs",
    )
    census = value["terminal_state_file_census"]
    files = census.get("files")
    require(
        set(census)
        == {
            "state_file_count",
            "state_file_bytes",
            "directory_paths",
            "directory_modes",
            "logs_file_count",
            "logs_total_bytes",
            "symlink_count",
            "special_file_count",
            "cache_file_count",
            "temporary_file_count",
            "state_file_map_canonical_sha256",
            "files",
        }
        and isinstance(files, dict)
        and type(census["state_file_count"]) is int
        and census["state_file_count"] == len(files) == 24
        and type(census["state_file_bytes"]) is int
        and census["state_file_bytes"]
        == sum(row["size"] for row in files.values())
        == 285_696
        and exact_json_equal(census["directory_paths"], [".", "logs", "source"])
        and exact_json_equal(
            census["directory_modes"],
            {".": "0700", "logs": "0700", "source": "0555"},
        )
        and type(census["logs_file_count"]) is int
        and census["logs_file_count"] == 3
        and type(census["logs_total_bytes"]) is int
        and census["logs_total_bytes"] == 0
        and type(census["symlink_count"]) is int
        and census["symlink_count"] == 0
        and type(census["special_file_count"]) is int
        and census["special_file_count"] == 0
        and type(census["cache_file_count"]) is int
        and census["cache_file_count"] == 0
        and type(census["temporary_file_count"]) is int
        and census["temporary_file_count"] == 0,
        "canary2 state census summary differs",
    )
    expected_names = {
        ".CANARY_CONTROLLER.lock",
        "CANARY_AUTHORIZATION.json",
        "CANARY_CONTROLLER_IDENTITY.json",
        "CANARY_READY_TO_RELEASE.json",
        "CANARY_REPORT.json",
        "CANARY_REPORT_CALLING.json",
        "CANARY_REPORT_SUBMITTED.json",
        "CANARY_SUBMISSION_RECEIPT.json",
        "CANARY_WAVE0_CALLING.json",
        "CANARY_WAVE0_RELEASED.json",
        "CANARY_WAVE0_RELEASE_CALLING.json",
        "CANARY_WAVE0_SUBMITTED.json",
        "CANARY_WAVE1_CALLING.json",
        "CANARY_WAVE1_SUBMITTED.json",
        "WAVE0_READY.json",
        "WAVE1_COMPLETE.json",
        "logs/report_33295661.out",
        "logs/wave0_33295657.out",
        "logs/wave1_33295659.out",
        "source/canary_gpu.slurm",
        "source/canary_report.slurm",
        "source/canary_worker.py",
        "source/two_wave_canary.py",
        "wave0_checkpoint.pt",
    }
    require(set(files) == expected_names, "canary2 state file names differ")
    require(
        all(
            set(row) == {"mode", "sha256", "size"}
            and type(row["mode"]) is str
            and re.fullmatch(r"0[0-7]{3}", row["mode"]) is not None
            and type(row["sha256"]) is str
            and SHA256.fullmatch(row["sha256"]) is not None
            and type(row["size"]) is int
            and row["size"] >= 0
            for row in files.values()
        ),
        "canary2 state file row differs",
    )
    require(
        stable_hash({"schema_version": 1, "files": files})
        == census["state_file_map_canonical_sha256"]
        == binding["state_file_map_canonical_sha256"]
        == "60dded3399d9b5406418209ffeb5020a09f22778e135c1ad79b3f413f9a7a2ce",
        "canary2 state file map differs",
    )
    empty_sha = hashlib.sha256(b"").hexdigest()
    require(
        files[".CANARY_CONTROLLER.lock"]
        == {"mode": "0600", "sha256": empty_sha, "size": 0}
        and all(
            files[name] == {"mode": "0644", "sha256": empty_sha, "size": 0}
            for name in (
                "logs/report_33295661.out",
                "logs/wave0_33295657.out",
                "logs/wave1_33295659.out",
            )
        )
        and files["CANARY_CONTROLLER_IDENTITY.json"]["sha256"]
        == identity["controller_identity_sha256"]
        and exact_json_equal(
            {
                name: files[f"source/{name}"]["sha256"]
                for name in (
                    "canary_gpu.slurm",
                    "canary_report.slurm",
                    "canary_worker.py",
                    "two_wave_canary.py",
                )
            },
            identity["source_sha256"],
        ),
        "canary2 census identity/source/log binding differs",
    )
    topology = value["durable_topology"]
    expected_ids = {role: values[0] for role, values in binding["job_ids_by_role"].items()}
    expected_names_by_role = {
        role: f"exp23-launch8-canary-{binding['canary_token']}-{role}"
        for role in ("wave0", "wave1", "report")
    }
    require(
        set(topology)
        == {
            "job_ids",
            "job_names",
            "dependencies",
            "scheduler_observed_dependencies",
            "kill_on_invalid_dependency",
            "accepted_states",
            "accepted_reasons",
            "within_wave_requeue",
            "resources",
            "evidence_sha256",
        }
        and exact_json_equal(topology["job_ids"], expected_ids)
        and exact_json_equal(topology["job_names"], expected_names_by_role)
        and exact_json_equal(
            topology["dependencies"],
            {
                "wave0": "none",
                "wave1": "afterok:33295657",
                "report": "afterok:33295659",
            },
        )
        and exact_json_equal(
            topology["scheduler_observed_dependencies"],
            {
                "wave0": "(null)",
                "wave1": "afterok:33295657(unfulfilled)",
                "report": "afterok:33295659(unfulfilled)",
            },
        )
        and exact_json_equal(
            topology["kill_on_invalid_dependency"],
            {"wave0": None, "wave1": "Yes", "report": "Yes"},
        )
        and exact_json_equal(
            topology["accepted_states"],
            {"wave0": "PENDING", "wave1": "PENDING", "report": "PENDING"},
        )
        and exact_json_equal(
            topology["accepted_reasons"],
            {"wave0": "JobHeldUser", "wave1": "Dependency", "report": "Dependency"},
        )
        and topology["within_wave_requeue"] is False,
        "canary2 scalar topology differs",
    )
    gpu_resources = {
        "partition": "polar4,polar3,polar,grizzly",
        "account": "edgeai_tao-ptm_image-foundation-model-clip",
        "qos": "normal",
        "time_limit": "00:10:00",
        "nodes": "1-1",
        "tasks": 1,
        "cpus": 2,
        "cpus_per_task": 2,
        "memory": "8G",
        "gpus": 1,
        "tres_per_node": "gres:gpu:1",
        "req_tres": "cpu=2,mem=8G,node=1,billing=1,gres/gpu=1",
    }
    require(
        exact_json_equal(
            topology["resources"],
            {
                "wave0": gpu_resources,
                "wave1": gpu_resources,
                "report": {
                    "partition": "cpu",
                    "account": "edgeai_tao-ptm_image-foundation-model-clip",
                    "qos": "normal",
                    "time_limit": "00:05:00",
                    "nodes": "1-1",
                    "tasks": 1,
                    "cpus": 1,
                    "cpus_per_task": 1,
                    "memory": "1G",
                    "gpus": 0,
                    "tres_per_node": None,
                    "req_tres": "cpu=1,mem=1G,node=1",
                },
            },
        ),
        "canary2 accepted scheduler resources differ",
    )
    evidence = topology["evidence_sha256"]
    evidence_files = {
        "controller_identity_raw": "CANARY_CONTROLLER_IDENTITY.json",
        "wave0_calling_raw": "CANARY_WAVE0_CALLING.json",
        "wave0_submitted_raw": "CANARY_WAVE0_SUBMITTED.json",
        "wave1_calling_raw": "CANARY_WAVE1_CALLING.json",
        "wave1_submitted_raw": "CANARY_WAVE1_SUBMITTED.json",
        "report_calling_raw": "CANARY_REPORT_CALLING.json",
        "report_submitted_raw": "CANARY_REPORT_SUBMITTED.json",
        "authorization_raw": "CANARY_AUTHORIZATION.json",
        "submission_receipt_raw": "CANARY_SUBMISSION_RECEIPT.json",
        "ready_to_release_raw": "CANARY_READY_TO_RELEASE.json",
        "wave0_release_calling_raw": "CANARY_WAVE0_RELEASE_CALLING.json",
        "wave0_released_raw": "CANARY_WAVE0_RELEASED.json",
    }
    require(
        set(evidence)
        == set(evidence_files)
        | {"accepted_submission_records", "accepted_scheduler_evidence"}
        and all(evidence[key] == files[name]["sha256"] for key, name in evidence_files.items())
        and evidence["accepted_submission_records"]
        == "46e5c8ddaf1ec5f181d107a90811c9c59d7d8cd679333ef4cbc4818ba1771599"
        and evidence["accepted_scheduler_evidence"]
        == "2973d098d440ddf6cfad8db3575386ec6c1fe8484a50edce032b3d16e1fffbf6",
        "canary2 durable evidence hashes differ",
    )
    runtime = value["runtime_result"]
    require(
        exact_json_equal(
            runtime,
            {
                "wave0_ready": {
                    "raw_sha256": files["WAVE0_READY.json"]["sha256"],
                    "record": {
                        "schema_version": 1,
                        "status": "wave0_ready",
                        "campaign_id": manifest["campaign_id"],
                        "canary_token": binding["canary_token"],
                        "wave0_job_id": expected_ids["wave0"],
                        "checkpoint_sha256": files["wave0_checkpoint.pt"]["sha256"],
                        "cuda_device_name": "NVIDIA A100-SXM4-80GB",
                        "expected_resumed_result": 66_672_896.0,
                        "within_wave_requeue": False,
                    },
                },
                "checkpoint": {
                    "raw_sha256": files["wave0_checkpoint.pt"]["sha256"],
                    "size": 2501,
                    "mode": "0444",
                    "tensor_dtype": "torch.float32",
                    "tensor_shape": [16, 16],
                    "tensor_checksum": 66_672_640.0,
                    "scientific_state": False,
                },
                "wave1_complete": {
                    "raw_sha256": files["WAVE1_COMPLETE.json"]["sha256"],
                    "record": {
                        "schema_version": 1,
                        "status": "wave1_complete",
                        "campaign_id": manifest["campaign_id"],
                        "canary_token": binding["canary_token"],
                        "wave0_job_id": expected_ids["wave0"],
                        "wave1_job_id": expected_ids["wave1"],
                        "ready_sha256": files["WAVE0_READY.json"]["sha256"],
                        "checkpoint_sha256": files["wave0_checkpoint.pt"]["sha256"],
                        "resumed_result": 66_672_896.0,
                        "within_wave_requeue": False,
                    },
                },
                "report": {
                    "raw_sha256": files["CANARY_REPORT.json"]["sha256"],
                    "record": {
                        "schema_version": 1,
                        "status": "two_wave_gpu_canary_passed",
                        "campaign_id": manifest["campaign_id"],
                        "canary_token": binding["canary_token"],
                        "job_ids": expected_ids,
                        "dependencies": topology["dependencies"],
                        "authorization_sha256": evidence["authorization_raw"],
                        "receipt_sha256": evidence["submission_receipt_raw"],
                        "ready_to_release_sha256": evidence["ready_to_release_raw"],
                        "wave0_release_sha256": evidence["wave0_released_raw"],
                        "wave0_ready_sha256": files["WAVE0_READY.json"]["sha256"],
                        "wave1_complete_sha256": files["WAVE1_COMPLETE.json"]["sha256"],
                        "checkpoint_sha256": files["wave0_checkpoint.pt"]["sha256"],
                        "accepted_submission_records_sha256": evidence[
                            "accepted_submission_records"
                        ],
                        "accepted_scheduler_evidence_sha256": evidence[
                            "accepted_scheduler_evidence"
                        ],
                    },
                },
            },
        )
        and runtime["report"]["raw_sha256"] == binding["report_raw_sha256"],
        "canary2 runtime/report lineage differs",
    )
    require(
        all(
            hashlib.sha256(
                (canonical_json(runtime[key]["record"]) + "\n").encode("ascii")
            ).hexdigest()
            == runtime[key]["raw_sha256"]
            for key in ("wave0_ready", "wave1_complete", "report")
        )
        and runtime["checkpoint"]["raw_sha256"]
        == files["wave0_checkpoint.pt"]["sha256"]
        and runtime["checkpoint"]["size"]
        == files["wave0_checkpoint.pt"]["size"]
        and runtime["checkpoint"]["mode"]
        == files["wave0_checkpoint.pt"]["mode"],
        "canary2 reconstructed runtime artifact bytes differ",
    )
    expected_rows = [
        [
            "33295657",
            "exp23-launch8-canary-b95869841048e511-wave0",
            "COMPLETED",
            "0:0",
            "25",
            "1",
            "batch-block5-04028",
            "billing=1,cpu=2,gres/gpu=1,mem=8G,node=1",
            "billing=1,cpu=2,gres/gpu=1,mem=8G,node=1",
            "2026-08-28T22:52:10",
            "2026-08-28T22:52:18",
            "2026-08-28T22:52:45",
            "2026-08-28T22:53:10",
            "treewm-exp23-canary:b95869841048e511",
        ],
        [
            "33295659",
            "exp23-launch8-canary-b95869841048e511-wave1",
            "COMPLETED",
            "0:0",
            "12",
            "1",
            "batch-block5-04028",
            "billing=1,cpu=2,gres/gpu=1,mem=8G,node=1",
            "billing=1,cpu=2,gres/gpu=1,mem=8G,node=1",
            "2026-08-28T22:52:13",
            "2026-08-28T22:53:16",
            "2026-08-28T22:53:45",
            "2026-08-28T22:53:57",
            "treewm-exp23-canary:b95869841048e511",
        ],
        [
            "33295661",
            "exp23-launch8-canary-b95869841048e511-report",
            "COMPLETED",
            "0:0",
            "6",
            "1",
            "cpu-00073",
            "cpu=1,mem=1G,node=1",
            "cpu=2,mem=1G,node=1",
            "2026-08-28T22:52:15",
            "2026-08-28T22:54:10",
            "2026-08-28T22:54:10",
            "2026-08-28T22:54:16",
            "treewm-exp23-canary:b95869841048e511",
        ],
    ]
    require(
        value["scheduler_terminal_rows_schema"]
        == (
            "JobIDRaw|JobName|State|ExitCode|ElapsedRaw|AllocNodes|NodeList|"
            "ReqTRES|AllocTRES|Submit|Eligible|Start|End|Comment"
        )
        and exact_json_equal(value["scheduler_terminal_rows"], expected_rows)
        and len(expected_rows) == binding["scheduler_terminal_rows"],
        "canary2 scheduler terminal rows differ",
    )
    scheduler_payload = {
        "schema_version": 1,
        "fields": value["scheduler_terminal_rows_schema"].split("|"),
        "rows": expected_rows,
    }
    scheduler_raw = (
        "\n".join("|".join(row) for row in expected_rows) + "\n"
    ).encode("ascii")
    scheduler = value["scheduler_terminal_observation"]
    require(
        set(scheduler)
        == {
            "captured_at_utc",
            "cwd",
            "environment",
            "scheduler_control_plane_sha256",
            "scheduler_control_plane_match_observed_separately",
            "scheduler_control_plane_attestation_scope",
            "argv",
            "returncode",
            "raw_stdout",
            "raw_stdout_size",
            "raw_stdout_sha256",
            "raw_stderr",
            "raw_stderr_size",
            "raw_stderr_sha256",
            "canonical_rows_bytes",
            "canonical_rows_sha256",
        }
        and scheduler["captured_at_utc"] == "2026-08-28T23:05:37Z"
        and scheduler["cwd"] == "/"
        and exact_json_equal(
            scheduler["environment"],
            {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "LANG": "C",
                "LC_ALL": "C",
                "SLURM_CONF": "/cm/shared/apps/slurm/var/etc/cs-oci-ord/slurm.conf",
            },
        )
        and scheduler["scheduler_control_plane_sha256"]
        == identity["scheduler_control_plane_sha256"]
        and scheduler["scheduler_control_plane_match_observed_separately"] is True
        and scheduler["scheduler_control_plane_attestation_scope"]
        == (
            "A separate read-only observation matched the root-bound control-plane "
            "digest; this is an operator observation, not a signed scheduler "
            "attestation."
        )
        and exact_json_equal(
            scheduler["argv"],
            [
                "/usr/local/bin/sacct",
                "-X",
                "-n",
                "-j",
                "33295657,33295659,33295661",
                "-o",
                "JobIDRaw,JobName,State,ExitCode,ElapsedRaw,AllocNodes,NodeList,ReqTRES,AllocTRES,Submit,Eligible,Start,End,Comment",
                "-P",
            ],
        )
        and type(scheduler["returncode"]) is int
        and scheduler["returncode"] == 0
        and scheduler["raw_stdout"].encode("ascii") == scheduler_raw
        and type(scheduler["raw_stdout_size"]) is int
        and scheduler["raw_stdout_size"] == len(scheduler_raw) == 819
        and scheduler["raw_stdout_sha256"] == hashlib.sha256(scheduler_raw).hexdigest()
        and scheduler["raw_stderr"] == ""
        and type(scheduler["raw_stderr_size"]) is int
        and scheduler["raw_stderr_size"] == 0
        and scheduler["raw_stderr_sha256"] == empty_sha
        and type(scheduler["canonical_rows_bytes"]) is int
        and scheduler["canonical_rows_bytes"]
        == len(canonical_json(scheduler_payload).encode("ascii"))
        == 1092
        and scheduler["canonical_rows_sha256"]
        == stable_hash(scheduler_payload)
        == "b7b204be3ef714222f5893cc727ace5ca20bb482485301ee20ac33e8320031ef",
        "canary2 scheduler terminal evidence differs",
    )
    zero = value["terminal_zero_active_evidence"]
    common_empty = {
        "returncode": 0,
        "stdout": "",
        "stdout_size": 0,
        "stdout_sha256": empty_sha,
        "stderr": "",
        "stderr_size": 0,
        "stderr_sha256": empty_sha,
    }
    owner = zero.get("owner_active_scheduler_census")
    require(
        isinstance(owner, Mapping)
        and set(owner)
        == {
            "captured_at_utc",
            "argv",
            "environment",
            "returncode",
            "raw_stdout_size",
            "raw_stdout_sha256",
            "compressed_stdout_size",
            "compressed_stdout_sha256",
            "raw_stdout_zlib_base64",
            "raw_stderr",
            "raw_stderr_size",
            "raw_stderr_sha256",
            "row_schema",
            "matching_predicate",
            "matched_rows",
        }
        and owner["captured_at_utc"] == "2026-08-28T23:28:04Z"
        and exact_json_equal(
            owner["argv"],
            [
                "/usr/local/bin/squeue",
                "--noheader",
                "--user=chrislin",
                "--format=%A|%F|%j|%T|%k",
            ],
        )
        and exact_json_equal(
            owner["environment"],
            {
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "LANG": "C",
                "LC_ALL": "C",
                "TZ": "UTC",
                "USER": "chrislin",
                "LOGNAME": "chrislin",
                "SLURM_CONF": "/cm/shared/apps/slurm/var/etc/cs-oci-ord/slurm.conf",
            },
        )
        and type(owner["returncode"]) is int
        and owner["returncode"] == 0
        and type(owner["raw_stdout_size"]) is int
        and owner["raw_stdout_size"] == 9992
        and owner["raw_stdout_sha256"]
        == "1aff7458b2b7ab62fb7e8189af662d7d92c6eadcb21303366917249d4ead5a00"
        and type(owner["compressed_stdout_size"]) is int
        and owner["compressed_stdout_size"] == 1017
        and owner["compressed_stdout_sha256"]
        == "8dccdf089806fe525f74aaeb903eced6c99e6a123642b3c378713912f7c5b988"
        and type(owner["raw_stdout_zlib_base64"]) is str
        and len(owner["raw_stdout_zlib_base64"]) == 1356
        and owner["raw_stderr"] == ""
        and type(owner["raw_stderr_size"]) is int
        and owner["raw_stderr_size"] == 0
        and owner["raw_stderr_sha256"] == empty_sha
        and owner["row_schema"] == "JobID|ArrayJobID|JobName|State|Comment"
        and owner["matching_predicate"]
        == (
            "JobName startswith exp23-launch8-canary-b95869841048e511- OR "
            "Comment equals treewm-exp23-canary:b95869841048e511"
        )
        and exact_json_equal(owner["matched_rows"], []),
        "canary2 owner-wide zero-stray census envelope differs",
    )
    try:
        compressed_owner_stdout = base64.b64decode(
            owner["raw_stdout_zlib_base64"], validate=True
        )
        require(
            len(compressed_owner_stdout) == owner["compressed_stdout_size"]
            and hashlib.sha256(compressed_owner_stdout).hexdigest()
            == owner["compressed_stdout_sha256"],
            "canary2 owner-wide zero-stray compressed bytes differ",
        )
        decompressor = zlib.decompressobj()
        owner_stdout = decompressor.decompress(compressed_owner_stdout, 9993)
        require(
            decompressor.eof
            and not decompressor.unconsumed_tail
            and not decompressor.unused_data
            and len(owner_stdout) <= 9992,
            "canary2 owner-wide zero-stray decompression boundary differs",
        )
        owner_text = owner_stdout.decode("ascii")
    except Exception as exc:
        raise ContractError(
            f"canary2 owner-wide zero-stray census bytes differ: {exc}"
        ) from exc
    owner_rows = [line.split("|") for line in owner_text.splitlines()]
    require(
        len(owner_stdout) == owner["raw_stdout_size"]
        and hashlib.sha256(owner_stdout).hexdigest() == owner["raw_stdout_sha256"]
        and all(len(row) == 5 and all(type(item) is str for item in row) for row in owner_rows),
        "canary2 owner-wide zero-stray census bytes differ",
    )
    matched_owner_rows = [
        row
        for row in owner_rows
        if row[2].startswith("exp23-launch8-canary-b95869841048e511-")
        or row[4] == "treewm-exp23-canary:b95869841048e511"
    ]
    require(
        exact_json_equal(matched_owner_rows, owner["matched_rows"]),
        "canary2 owner-wide zero-stray census selection differs",
    )
    require(
        set(zero)
        == {
            "exact_job_ids",
            "exact_job_names",
            "owner_active_scheduler_census",
            "active_exact_job_count",
            "active_exact_name_count",
            "stray_topology_job_count",
        }
        and exact_json_equal(
            {
                key: zero[key]
                for key in (
                    "exact_job_ids",
                    "exact_job_names",
                    "active_exact_job_count",
                    "active_exact_name_count",
                    "stray_topology_job_count",
                )
            },
            {
                "exact_job_ids": {
                    "captured_at_utc": "2026-08-28T23:10:48Z",
                    "argv": [
                        "/usr/local/bin/squeue",
                        "--noheader",
                        "--jobs=33295657,33295659,33295661",
                        "--format=%A|%F|%j|%T|%k",
                    ],
                    **common_empty,
                },
                "exact_job_names": {
                    "captured_at_utc": "2026-08-28T23:10:49Z",
                    "argv": [
                        "/usr/local/bin/squeue",
                        "--noheader",
                        "--name=exp23-launch8-canary-b95869841048e511-wave0,exp23-launch8-canary-b95869841048e511-wave1,exp23-launch8-canary-b95869841048e511-report",
                        "--format=%A|%F|%j|%T|%k",
                    ],
                    **common_empty,
                },
                "active_exact_job_count": 0,
                "active_exact_name_count": 0,
                "stray_topology_job_count": 0,
            },
        )
        and zero["active_exact_job_count"]
        == binding["active_scheduler_jobs_after_terminal"]
        and value["observed_at_utc"]
        >= max(
            scheduler["captured_at_utc"],
            zero["exact_job_ids"]["captured_at_utc"],
            zero["exact_job_names"]["captured_at_utc"],
            owner["captured_at_utc"],
        ),
        "canary2 terminal zero-active evidence differs",
    )
    require(
        exact_json_equal(
            value["absent_durable_artifacts"],
            {
                "exact_names": [
                    "CANARY_ABORTED.json",
                    "CANARY_RECOVERY_CANCELLED.json",
                ],
                "forbidden_name_fragments": [
                    "ABORT",
                    "CANCEL",
                    "ERROR",
                    "FAIL",
                    "RECOVER",
                    "TEMP",
                    "TMP",
                ],
                "matched_file_count": 0,
            },
        )
        and not any(
            fragment in name.upper()
            for name in files
            for fragment in ("ABORT", "CANCEL", "ERROR", "FAIL", "RECOVER", "TEMP", "TMP")
        ),
        "canary2 absent failure/recovery artifacts differ",
    )
    conclusion = value["acceptance_conclusion"]
    require(
        exact_json_equal(
            conclusion,
            {
                "topology_canary_passed": True,
                "production_authorization_prerequisite_satisfied": True,
                "scalar_dependency_scope_proved": True,
                "held_before_authorization_proved": True,
                "release_after_receipt_proved": True,
                "real_gpu_wave0_proved": True,
                "checkpoint_transfer_proved": True,
                "same_canary_wave0_to_wave1_checkpoint_transfer_completed": True,
                "real_gpu_wave1_proved": True,
                "report_after_wave1_proved": True,
                "all_jobs_terminal_success": True,
                "zero_active_jobs": True,
                "scientific_state": False,
                "reuse_allowed": False,
                "resume_allowed": False,
                "retry_allowed": False,
                "recovery_allowed": False,
                "checkpoint_consumption_as_scientific_state_allowed": False,
                "result_consumption_as_scientific_measurement_allowed": False,
                "allowed_consumption": [
                    "package provenance",
                    "production launch authorization prerequisite",
                    "topology engineering evidence",
                ],
                "required_successor_policy": (
                    "Every future canary must use a fresh root, token, logical job "
                    "namespace, checkpoint, and result namespace. Historical numeric "
                    "Slurm IDs are never authorization, release, resume, or result "
                    "identities; if Slurm recycles one into a fresh exact canary "
                    "namespace, it may only be signalled for cleanup after a settled "
                    "exact owner, name, and token-bound-comment census. Launch8 "
                    "scientific work must use its fresh scientific root, names, "
                    "checkpoints, and results and must never consume canary files or "
                    "identities as scientific state."
                ),
            },
        )
        and all(
            conclusion[key] is binding[key]
            for key in (
                "topology_canary_passed",
                "production_authorization_prerequisite_satisfied",
                "reuse_allowed",
                "resume_allowed",
                "retry_allowed",
                "recovery_allowed",
                "checkpoint_consumption_as_scientific_state_allowed",
                "result_consumption_as_scientific_measurement_allowed",
            )
        ),
        "canary2 terminal acceptance/no-reuse conclusion differs",
    )


def _validate_canary2_acceptance_provenance(
    manifest: Mapping[str, Any], repo: Path
) -> None:
    try:
        _validate_canary2_acceptance_provenance_impl(manifest, repo)
    except ContractError:
        raise
    except Exception as exc:
        raise ContractError(
            f"canary2 acceptance provenance is malformed: {exc}"
        ) from exc


def validate_manifest(
    manifest: Mapping[str, Any],
    lock: Mapping[str, Any],
    repo: str | Path = REPOSITORY_ROOT,
    *,
    verify_resolved_config_lock: bool = True,
    verify_causal_parity_lock: bool = True,
) -> None:
    require(
        type(manifest.get("schema_version")) is int
        and manifest.get("schema_version") == 1,
        "manifest schema differs",
    )
    require(manifest.get("campaign_id") == CAMPAIGN_ID, "campaign ID differs")
    require(
        manifest.get("status") == "sealed_launch_ready_unsubmitted",
        "package launch state differs",
    )
    require(manifest.get("formal_validation") is False, "pilot is marked formal")
    _validate_canary1_negative_provenance(manifest, Path(repo).resolve())
    _validate_canary2_acceptance_provenance(manifest, Path(repo).resolve())
    _validate_launch7_negative_provenance(manifest, Path(repo).resolve())
    require(
        exact_json_equal(
            manifest.get("package_policy"),
            {
                "launch_surface": True,
                "allowed_actions": [
                    "verify_package",
                    "render_matrix",
                    "evaluate_supplied_report_bundle",
                    "rerun_outcome_blind_weight_audit",
                    "scheduler_control_plane_test",
                    "static_launch_test",
                    "dry_run",
                    "explicit_submit",
                    "terminal_report_repair_describe",
                    "terminal_report_repair_test",
                    "explicit_terminal_report_repair_submit",
                    "explicit_terminal_report_repair_recovery",
                ],
                "forbidden_actions": [
                    "implicit_submit",
                    "import_checkpoint",
                    "import_optimizer",
                    "select_at_midpoint",
                    "drop_cell",
                    "write_exp20_exp21_exp22",
                    "consume_exp20_outcomes",
                    "implicit_report_retry",
                    "manual_report_publication",
                    "second_report_repair_generation",
                    "scientific_recomputation",
                    "scientific_input_change",
                    "gate_change",
                ],
                "submission_policy": (
                    "No command submits unless submit.py is invoked with --submit "
                    "under the pinned interpreter, except the exact one-generation "
                    "terminal-report repair bound by launch_contract."
                    "terminal_report_repair, which requires report_repair.py "
                    "--submit-real-report-repair plus its exact confirmation phrase. "
                    "All default, --describe, --scheduler-test, and --test-only "
                    "actions are read-only."
                ),
            },
        ),
        "package action policy differs",
    )
    require(
        manifest.get("superseded_launches") == SUPERSEDED_LAUNCHES,
        "superseded launch identities differ",
    )
    require(
        all(
            prior["reuse_allowed"] is False
            and prior["resume_allowed"] is False
            and prior["retry_allowed"] is False
            and prior["recovery_allowed"] is False
            for prior in SUPERSEDED_LAUNCHES
        ),
        "superseded launch reuse/retry/recovery policy differs",
    )
    require(
        all(
            manifest["paths"]["run_root"] != prior["run_root"]
            and manifest["paths"]["wandb_project"] != prior["wandb_project"]
            for prior in SUPERSEDED_LAUNCHES
        ),
        "superseded namespace was reused",
    )
    require(
        manifest["paths"]["prospective_run_root"]
        == "outputs/treewm-executable-prefix-repair-pilot-v1-launch8"
        and manifest["paths"]["transaction_lock"]
        == "outputs/.exp23-c85fcaba919d617f.transaction.lock"
        and manifest["paths"]["run_root"]
        == (
            "/lustre/fs11/portfolios/edgeai/projects/"
            "edgeai_tao-ptm_image-foundation-model-clip/users/chrislin/projects/"
            "treewm/outputs/treewm-executable-prefix-repair-pilot-v1-launch8"
        ),
        "launch8 run/transaction namespace differs",
    )
    require(
        manifest["paths"]["wandb_project"] == CAMPAIGN_ID
        and manifest["logging"]["wandb_project"] == CAMPAIGN_ID
        and manifest["logging"]["wandb_group"] == CAMPAIGN_ID,
        "launch8 W&B namespace differs",
    )
    require(
        manifest["design"]["fresh_start_policy"]
        == (
            "Every Launch8 wave-zero cell starts from scratch in the fresh Launch8 "
            "namespace; no checkpoint, result, optimizer, model "
            "initialization, W&B identity, or state from any superseded launch may "
            "be imported, reused, resumed, retried, or recovered. Resume=auto and "
            "W&B resume are permitted only for the exact same fresh Launch8 cell in "
            "the predeclared, authorization-bound wave-zero to wave-one DAG."
        ),
        "superseded-launch exclusion policy differs",
    )
    require(manifest["design"]["settings"] == list(SETTINGS), "setting order differs")
    require(manifest["design"]["arms"] == list(ARMS), "arm order differs")
    require(manifest["design"]["seeds"] == list(SEEDS), "seed order differs")
    require(manifest["design"]["expected_cells"] == 20, "cell count differs")
    require(manifest["design"]["analysis_boundaries"] == [5000, 25000], "analysis boundaries differ")
    require(manifest["design"]["periodic_evaluation_boundaries"] == [12500, 25000], "eval boundaries differ")
    require([row["id"] for row in manifest["settings"]] == list(SETTINGS), "settings rows differ")
    require([row["id"] for row in manifest["arms"]] == list(ARMS), "arm rows differ")
    require(manifest["arms"][0]["executable_prefix_enabled"] is True, "control graph disabled")
    require(set(manifest["arms"][0]["executable_prefix_weights"].values()) == {0.0}, "control weights nonzero")
    require(all(math.isfinite(value) and value > 0 for value in manifest["arms"][1]["executable_prefix_weights"].values()), "treatment weights invalid")
    require(manifest["causal_contrast"]["sole_resolved_config_difference"] == list(WEIGHT_KEYS), "causal leaves differ")
    require(manifest["scientific_contract"]["optimizer_updates"] == 25000, "train cap differs")
    future = manifest["scientific_contract"]["future_sets"]
    require(future["horizons"] == [4, 8, 16, 32, 64] and future["h_max"] == 64, "horizon contract differs")
    require(future["executable_prefix_steps"] == 4, "prefix cap differs")
    require("min(4" in manifest["scientific_contract"]["prefix_length_rule"], "branchwise prefix rule missing")
    require("never compared to 4" in manifest["acceptance"]["prefix_structural_gates"]["target_rule"], "mean==4 gate reintroduced")
    cells = expand_matrix(manifest)
    require(len(cells) == 20, "matrix expansion differs")
    require(
        all(cell.run_name.startswith("exp23-launch8-") for cell in cells),
        "launch8 run-name namespace differs",
    )
    launch = manifest["launch_contract"]
    require(launch["array"] == "0-19%20" and launch["array_cells"] == 20, "launch array differs")
    require(
        launch["scientific_cells"] == 20
        and launch["gpu_array_jobs"] == 2
        and launch["scheduled_gpu_task_slots"] == 40
        and launch["gpu_per_cell"] == 1,
        "two-wave GPU topology counts differ",
    )
    require(launch["scratch_to_updates"] == 25_000 and launch["analysis_only_boundary_updates"] == 5_000, "launch boundaries differ")
    require(launch["terminal_final_evaluation"] is True and launch["terminal_final_evaluation_total_episodes"] == 25, "terminal evaluation contract differs")
    require(launch["midpoint_selection"] is False, "midpoint selection enabled")
    require("no TREEWM_STOP_AFTER_UPDATE" in launch["trainer_invocation_policy"], "staged stop reintroduced")
    require(launch["actual_submit_performed"] is False, "manifest claims a submission")
    graph = launch["scheduler_graph"]
    require(
        graph["scientific_cells"] == 20
        and [node["role"] for node in graph["nodes"]]
        == ["wave0", "wave1", "report"]
        and graph["nodes"][0]["initial_state"] == "held"
        and graph["nodes"][0]["within_wave_requeue"] is False
        and graph["nodes"][1]["within_wave_requeue"] is False
        and graph["nodes"][1]["usr1_incomplete_outcome"]
        == "nonzero terminal failure; no successor wave"
        and graph["edges"]
        == [
            {
                "from": "wave0",
                "to": "wave1",
                "dependency": "afterok:<wave0_array_job_id>",
                "kill_on_invalid_dependency": True,
            },
            {
                "from": "wave1",
                "to": "report",
                "dependency": "afterok:<wave1_array_job_id>",
                "kill_on_invalid_dependency": True,
            },
        ]
        and graph["authorization_order"]
        == [
            "submit wave0 held",
            "submit and authenticate wave1 dependency",
            "submit and authenticate report dependency",
            "publish SUBMISSION_AUTHORIZATION.json",
            "publish SUBMISSION_RECEIPT.json",
            "release exact wave0 array",
        ],
        "structured scheduler graph differs",
    )
    feasibility = launch["runtime_feasibility"]
    require(
        feasibility["wave_walltime_seconds"] == 14_400
        and feasibility["signal_seconds_before_end"] == 420
        and feasibility["pre_signal_usable_seconds"] == 13_980
        and feasibility["launch7_minimum_ready_checkpoint_updates"] == 17_518
        and feasibility["worst_case_remaining_updates"] == 7_482
        and feasibility["completed_cell_post_25000_progress_to_scheduler_exit_seconds"]
        == [99, 94, 134, 104]
        and feasibility["conservative_terminal_completion_bound_seconds"] == 180
        and math.isclose(
            feasibility["required_wave1_updates_per_second_after_terminal_reserve"],
            7482 / (13980 - 180),
            rel_tol=0.0,
            abs_tol=1e-15,
        )
        and feasibility["observed_to_required_rate_margin"] > 2.31,
        "two-wave runtime feasibility proof differs",
    )
    require(
        launch["real_gpu_two_wave_canary"]
        == {
            "scientific": False,
            "default_action": "read-only --describe",
            "preflight_invocation": False,
            "explicit_submit_flag": "--submit-real-gpu-two-wave-canary",
            "hard_crash_action": "--recover-or-cancel-real-gpu-canary",
            "confirmation_phrase": (
                "SUBMIT_EXP23_LAUNCH8_REAL_GPU_TWO_WAVE_CANARY"
            ),
            "controller": "two_wave_canary.py",
            "compute_worker": "canary_worker.py",
            "gpu_batch": "canary_gpu.slurm",
            "report_batch": "canary_report.slurm",
            "graph": (
                "one held wave0 GPU job -> one afterok/kill-invalid wave1 GPU job -> "
                "one afterok/kill-invalid CPU report; authorization, receipt, and READY "
                "are durable before wave0 release"
            ),
            "dedicated_state_parent": "outputs/exp23-launch8-two-wave-canaries",
            "controller_bootstrap": (
                "pinned Python -I -S -B with exact live sealed package/protocol "
                "validation; compute worker appends only the pinned venv/base "
                "package roots before location-validating the Torch import"
            ),
            "runtime_proof_scope": (
                "The canary proves isolated lexical pinned-Python execution, Torch "
                "import from one of the two validated nonsymlink package-root "
                "directories, exactly one visible selected CUDA device, a real CUDA "
                "tensor operation, and checkpoint transfer across the two waves. It "
                "does not hash or version-bind the resolved interpreter binary, "
                "pyvenv.cfg, Torch distribution/native libraries, CUDA driver/runtime, "
                "or establish byte-equivalence to the scientific trainer environment."
            ),
            "within_wave_requeue": False,
            "run_during_read_only_preflight": False,
            "failed_attempts": FAILED_CANARY_ATTEMPTS,
            "accepted_attempts": ACCEPTED_CANARY_ATTEMPTS,
            "production_authorization_evidence": {
                "required": True,
                "satisfied": True,
                "attempt": "canary2",
                "path": "canary2_acceptance_provenance.json",
                "raw_sha256": ACCEPTED_CANARY_ATTEMPTS[0]["raw_sha256"],
                "canonical_sha256": ACCEPTED_CANARY_ATTEMPTS[0][
                    "canonical_sha256"
                ],
                "source_protocol_sha256": ACCEPTED_CANARY_ATTEMPTS[0][
                    "source_protocol_sha256"
                ],
                "report_raw_sha256": ACCEPTED_CANARY_ATTEMPTS[0][
                    "report_raw_sha256"
                ],
                "artifact_evidence_consumption_allowed": True,
                "scientific_runtime_input_consumption_allowed": False,
                "accepted_compute_runtime_source_sha256": {
                    "canary_gpu.slurm": (
                        "cdfa25456ea26544450ff00681590adf1e81ef03eeb1434597c0777f6552ec14"
                    ),
                    "canary_report.slurm": (
                        "6d4561c7b8462fde2cd7b8666fa9f2b8d377aa30bfc44195bef6924548df1c0b"
                    ),
                    "canary_worker.py": (
                        "4bcbaab866538c527f7a894d015d8b8bfbc505b2eb8e3596e56d419cfa4b3a2d"
                    ),
                },
                "accepted_controller_sha256": ACCEPTED_CANARY_CONTROLLER_SHA256,
                "post_acceptance_current_controller_sha256": (
                    ACCEPTED_CANARY_CURRENT_CONTROLLER_SHA256
                ),
                "post_acceptance_current_source_sha256": (
                    ACCEPTED_CANARY_CURRENT_SOURCE_SHA256
                ),
                "post_acceptance_change_scope": (
                    ACCEPTED_CANARY_POST_ACCEPTANCE_CHANGE_SCOPE
                ),
                "canary_rerun_required": False,
            },
        },
        "real-GPU two-wave canary contract differs",
    )
    require(
        exact_json_equal(
            launch.get("terminal_report_repair"), TERMINAL_REPORT_REPAIR_POLICY
        ),
        "terminal report repair contract differs",
    )
    repair_package = Path(repo).resolve() / PACKAGE_RELATIVE
    for repair_name, repair_sha256 in TERMINAL_REPORT_REPAIR_POLICY[
        "repair_source_sha256"
    ].items():
        repair_payload = _regular_file_bytes(
            repair_package / repair_name,
            f"terminal report repair source {repair_name}",
            max_bytes=8 * 1024 * 1024,
        )
        require(
            hashlib.sha256(repair_payload).hexdigest() == repair_sha256,
            f"terminal report repair source hash differs: {repair_name}",
        )
    execution = manifest["execution"]
    require("srun" not in execution, "srun execution path reintroduced")
    require(execution["scontrol"] == "/usr/local/bin/scontrol", "scontrol path differs")
    require(
        execution["scheduler_control_plane"]
        == {
            "slurm_conf": "/cm/shared/apps/slurm/var/etc/cs-oci-ord/slurm.conf",
            "cluster_name": "cs-oci-ord",
            "slurmctld_hosts": ["cs-oci-ord-a", "cs-oci-ord-b"],
            "slurmctld_port": 6817,
            "auth_type": "auth/munge",
            "gres_types": ["gpu"],
            "cli_filter_plugins": ["lua"],
            "job_submit_plugins": ["lua"],
            "trust_model": (
                "root-admin mutable scheduler control plane; config and Lua policy bytes "
                "are observation-bound from preclaim through submission; root-owned Slurm "
                "clients, plugin binaries, and shared libraries are trusted mutable "
                "external runtime"
            ),
        },
        "scheduler control-plane contract differs",
    )
    require(execution["control_python_flags"] == ["-I", "-S", "-B"], "control Python flags differ")
    require(execution["trainer_python_flags"] == ["-P", "-S", "-B"], "trainer Python flags differ")
    require(
        execution["sealed_trainer_environment"]
        == {"MUJOCO_GL": "egl", "XLA_PYTHON_CLIENT_PREALLOCATE": "false"},
        "sealed trainer environment differs",
    )
    require(
        execution["within_wave_requeue"] is False
        and "predeclared afterok wave one" in execution["training_lifecycle"]
        and "no srun, compute-side scheduler client, or within-wave requeue"
        in execution["process_topology"]
        and execution["scheduler_client_placement"]
        == (
            "Only submit.py, cancel.py, report_repair.py, and submission-host "
            "recovery may execute scontrol/squeue/sbatch/scancel. Compute "
            "worker.py, train.slurm, report.py, and report_repair.slurm never do."
        ),
        "two-wave process topology differs",
    )
    validate_snapshot_import_files(repo)
    _validate_lock(lock, manifest)
    _validate_prefix_target_lock(manifest)
    if verify_resolved_config_lock:
        _validate_resolved_config_lock(manifest)
    if verify_causal_parity_lock:
        _validate_causal_parity_lock(manifest)
    _validate_core(manifest, lock, Path(repo).resolve())
    for setting in manifest["settings"]:
        audit_data = lock["data_identity"][setting["id"]]
        require(setting["source_manifest_sha256"] == audit_data["source_manifest_sha256"], f"{setting['id']}: source differs")
        require(setting["future_recipe_sha256"] == audit_data["future_recipe_sha256"], f"{setting['id']}: recipe differs")
        require(setting["published_union_train_anchors"] == audit_data["train_population"], f"{setting['id']}: train population differs")
        require(setting["published_union_validation_anchors"] == audit_data["validation_population"], f"{setting['id']}: val population differs")
    for setting in SETTINGS:
        for seed in SEEDS:
            pair = [cell for cell in cells if cell.setting == setting and cell.seed == seed]
            require([cell.arm for cell in pair] == list(ARMS), "matched pair missing")
            resolved = [cell_overrides(cell, manifest, lock) for cell in pair]
            differing = {key for key in resolved[0] if resolved[0][key] != resolved[1][key]}
            require(differing == set(WEIGHT_KEYS), f"{setting}/seed{seed}: arm contrast is not weights-only: {sorted(differing)}")


def load_contract(repo: str | Path = REPOSITORY_ROOT) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_json(MANIFEST_PATH)
    lock = read_json(WEIGHT_LOCK_PATH)
    validate_manifest(manifest, lock, repo)
    return manifest, lock


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--verify", action="store_true")
    actions.add_argument("--matrix-json", action="store_true")
    actions.add_argument("--cell-json", type=int)
    parser.add_argument("--skip-protocol-lock", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        manifest, lock = load_contract()
        protocol = None if args.skip_protocol_lock else verify_protocol_lock()
        cells = expand_matrix(manifest)
        if args.verify:
            payload = {"status": "verified", "cells": len(cells), "protocol_sha256": protocol}
        elif args.matrix_json:
            payload = {"cells": [asdict(cell) for cell in cells]}
        else:
            require(args.cell_json is not None and 0 <= args.cell_json < len(cells), "cell index out of range")
            cell = cells[args.cell_json]
            payload = {"cell": asdict(cell), "declarative_overrides": cell_overrides(cell, manifest, lock)}
        print(canonical_json(payload))
        return 0
    except Exception as exc:
        print(f"Exp23 contract failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
