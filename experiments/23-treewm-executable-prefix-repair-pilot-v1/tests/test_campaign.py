from __future__ import annotations

import copy
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import zlib

import numpy as np
import pytest


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]
SPEC = importlib.util.spec_from_file_location("exp23_campaign", PACKAGE / "campaign.py")
assert SPEC is not None and SPEC.loader is not None
campaign = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = campaign
SPEC.loader.exec_module(campaign)


def contracts():
    manifest = campaign.read_json(PACKAGE / "manifest.json")
    lock = campaign.read_json(PACKAGE / "weight_audit.lock.json")
    return manifest, lock


def test_full_static_contract_and_matrix():
    manifest, lock = contracts()
    campaign.validate_manifest(manifest, lock, REPO)
    cells = campaign.expand_matrix(manifest)
    assert len(cells) == 20
    assert cells[0].index == 0
    assert (cells[0].setting, cells[0].arm, cells[0].seed) == (
        "antmaze-large", "GS", 110
    )
    assert (cells[-1].setting, cells[-1].arm, cells[-1].seed) == (
        "cube-quadruple-100m", "GSEP", 111
    )
    assert [cell.index for cell in cells] == list(range(20))
    assert manifest["campaign_id"] == campaign.CAMPAIGN_ID
    assert all(cell.run_name.startswith("exp23-launch8-") for cell in cells)


def test_launch8_namespace_and_ordered_superseded_launches_are_exact():
    manifest, lock = contracts()
    superseded = manifest["superseded_launches"]
    assert superseded == campaign.SUPERSEDED_LAUNCHES
    assert [row["campaign_id"] for row in superseded] == [
        "treewm-executable-prefix-repair-pilot-v1",
        "treewm-executable-prefix-repair-pilot-v1-launch2",
        "treewm-executable-prefix-repair-pilot-v1-launch3",
        "treewm-executable-prefix-repair-pilot-v1-launch4",
        "treewm-executable-prefix-repair-pilot-v1-launch5",
        "treewm-executable-prefix-repair-pilot-v1-launch6",
        "treewm-executable-prefix-repair-pilot-v1-launch7",
    ]
    launch1, launch2, launch3, launch4, launch5, launch6, launch7 = superseded
    assert launch1["source_commit_claimed_by_journal"] is False
    assert launch1["snapshot"] == {
        "inventory_sha256": "6767520819d42ef8866712023211b2f1bc8d236db3ffc836c8dae429b4e5b326",
        "file_count": 137,
        "independently_matched_files": 137,
        "all_files_match": True,
    }
    assert launch2["source_commit"] == "0fd89949a092bd9bbf12b16e3efb058850d50c86"
    assert launch2["package_protocol_sha256"] == (
        "6472ca50fcbc1eaa35c4388876bf627f0f2c03d8310fd05e336f0204c0f49516"
    )
    assert launch2["manifest_raw_sha256"] == (
        "44911238bb06b10b46abbf58a8fe33019e0c107a6d760f8954a9a416382be776"
    )
    assert launch2["manifest_canonical_sha256"] == (
        "d124566d5834a62028f7416756c7cca36e6e63ae256e72d8c7c788412d558b00"
    )
    assert launch2["snapshot"] == {
        "inventory_sha256": "4aef86836e7fb683ace18cdd7588fd6b3904bdb9877e5c9a08146e76e49e2a76",
        "file_count": 137,
        "independently_matched_files": 137,
        "all_files_match": True,
    }
    assert launch2["claim_token"] == (
        "a2ea8575200dc47b4e3de67863a0f429d2397ae35fbbd2e7948e9322ffb64802"
    )
    assert launch2["journal_sha256"] == {
        "0000_CLAIMED.json": "e353f69fb6a397d1095f3f5b81ca717a5887b6d4d55fa0961ca38faa3460b6dc",
        "0001_SNAPSHOT_SEALED.json": "be9a29c112f56dc0ace53847b99e3892ab964ee410ca2ecfc2a0fd9d39179bdc",
        "9998_OUTER_ABORTED.json": "48eed85ce12306a35927e3ac8b539be28dc52e107465fac9f5546c378c99cb99",
    }
    assert launch3["status"] == "aborted_after_contract_before_scheduler_submission"
    assert launch3["source_commit"] == "ca979a2b0329d6775793cd8ce51d57a9200e6b8a"
    assert launch3["source_commit_claimed_by_contract"] is True
    assert launch3["source_commit_claimed_by_journal"] is False
    assert launch3["package_protocol_sha256"] == (
        "6178ed54273d13c88fce750414131c98d002b394231618f11f6d8d6a1a3fb49a"
    )
    assert launch3["manifest_raw_sha256"] == (
        "92a9e17b78075805503907e4d9b71b732b54f5127da981ca17524feba650f74a"
    )
    assert launch3["manifest_canonical_sha256"] == (
        "7259a76924ac6f4566541a00f74923c67151619b36411048dd82ee20747a8d09"
    )
    assert launch3["snapshot"] == {
        "inventory_sha256": "7e237d31f9d49e3b55d0e0598c299b6064ab16b9caa77b62329c9bb8a2839eae",
        "file_count": 137,
        "independently_matched_files": 137,
        "all_files_match": True,
    }
    assert launch3["preserved_tree"] == {
        "regular_file_count": 163,
        "symlink_count": 0,
        "snapshot_file_count": 137,
        "launch_file_count": 20,
        "contract_file_count": 1,
        "journal_file_count": 5,
    }
    assert launch3["claim_token"] == (
        "2741418c7e528a0b64b8115cafa46cfac391abd53312cf29b0ffc8a4e1afca4d"
    )
    assert launch3["contract_sha256"] == launch3["submission_sha256"] == (
        "0cd594c8a49499b5e3d10a09ddbf3b89f981264be67bb603dc64836568a1b4c2"
    )
    assert launch3["journal_sha256"] == {
        "0000_CLAIMED.json": "25d607cef5aaf49e932e86a56c2272d37be6936010f089f8e7230bb44166be28",
        "0001_SNAPSHOT_SEALED.json": "d4acf9fb3fa6af98adedf45abd0269d1e491abd48b3cb992509b7641ad05b4c6",
        "0002_CONTRACT_SEALED.json": "3936a8717d096a93cf7d0eb3dea5293e32c14cdd0b986dd7a738f9501a92a044",
        "9999_ABORTED.json": "915dd5a3869a6784f3eb9d8e0d564a16b96fb17689c3d31f5a2e21431365199a",
        "9998_OUTER_ABORTED.json": "e65e9b39d268c2497e98e1d66b0cadd7f80b0f53c98a243057cc03ca399b47b0",
    }
    assert launch3["failure_phase"] == "first_sanitized_squeue_before_any_sbatch"
    assert launch3["actual_sbatch_calls"] == 0
    assert launch3["known_job_ids"] == []
    assert launch3["job_ids_by_role"] == {"train": [], "report": []}
    assert launch3["submission_contract_committed"] is True
    assert launch3["no_job_proof"]["source_order_at_commit"] == {
        "contract_sealed_first": True,
        "train_absence_check_line": 3112,
        "report_absence_check_line": 3113,
        "first_sbatch_call_line": 3114,
        "recorded_failure": "first_train_absence_check",
    }
    observation = launch3["no_job_proof"]["current_scheduler_observation"]
    assert observation["environment"] == {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "C",
        "LC_ALL": "C",
        "SLURM_CONF": "/cm/shared/apps/slurm/var/etc/cs-oci-ord/slurm.conf",
    }
    assert observation["history_start_utc"] == "2026-08-01"
    assert observation["squeue_matching_rows"] == 0
    assert observation["sacct_matching_rows"] == 0
    assert launch4["status"] == "aborted_after_scheduler_submission_before_any_job_runtime"
    assert launch4["failure_phase"] == (
        "canonical_array_dependency_validation_after_report_submission"
    )
    assert launch4["canonical_array_dependency_validator_error"] == (
        "SubmissionError('accepted report dependency differs')"
    )
    assert launch4["source_commit"] == "62fbf4631e950187506293138f13be691df1fa37"
    assert launch4["source_commit_claimed_by_contract"] is True
    assert launch4["source_commit_claimed_by_journal"] is False
    assert launch4["package_protocol_sha256"] == (
        "a838d23a396439dac585a1d4fe72f89b385df1e432f5795d85c5d7a2d818c02b"
    )
    assert launch4["manifest_raw_sha256"] == (
        "72342c585b3988df3d410131d33705e06b1eaf99494f027dea3830adb7326534"
    )
    assert launch4["manifest_canonical_sha256"] == (
        "7e1130dcc0f781c21e74a323880699e23ae6778d1752c64fe38cdb31a64aa7f8"
    )
    assert launch4["snapshot"] == {
        "inventory_sha256": "1a4e42ee751964ab704d2fae6f736862d46174d3eafd1ffe42b4f4f018cf1cbb",
        "file_count": 137,
        "independently_matched_files": 137,
        "all_files_match": True,
    }
    assert launch4["preserved_tree"] == {
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
    }
    assert launch4["claim_token"] == (
        "4c61d6fffc30ed2861ba6d3aaefb94ab6535616c9a6ce63bf9a097d4aa6162a9"
    )
    assert launch4["transaction_lock_state"] == {
        "regular_file": True,
        "symlink": False,
        "mode": "0600",
        "size": 0,
        "sha256": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }
    assert launch4["contract_sha256"] == launch4["submission_sha256"] == (
        "0aa63e5787fbdb06331265f03dd5e1aa32c32c32bb9b74728cd3060be7200336"
    )
    assert launch4["journal_sha256"] == {
        "0000_CLAIMED.json": "ce5361cf48b00b61e3ae7d12d1e06fb50a2429ecafad3f9453f7c38e1b1c594c",
        "0001_SNAPSHOT_SEALED.json": "a150558bbe7ff1c39caff10f8a48329f42142230bb6f1e1862721111b3505efa",
        "0002_CONTRACT_SEALED.json": "6b6b375e539bf64ceb3bccf2e3f4533b1913ed455a20b0db2ddfd36ee9727c3b",
        "0003_TRAIN_SUBMITTED.json": "196ad9be4302d6d1262914a8d42aaa0201d0b3c8f3a1d1496c1db5614fe6c271",
        "9999_ABORTED.json": "d73201fb9c7a6f89afacdf057613016a85c303ffd5ec2320972a6813ca524701",
        "9998_OUTER_ABORTED.json": "458285c797d33534deb5250cc4766a5a4f383afc3f195165cca46d8648d265d9",
    }
    assert launch4["actual_sbatch_calls"] == 2
    assert launch4["known_job_ids"] == ["33211846", "33211848"]
    assert launch4["job_ids_by_role"] == {
        "train": ["33211846"],
        "report": ["33211848"],
    }
    assert launch4["submission_contract_committed"] is True
    assert launch4["submission_receipt_committed"] is False
    assert launch4["jobs_cancelled_before_runtime"] is True
    durable = launch4["durable_failure_evidence"]
    assert durable["scontrol_command_preserved"] is True
    assert durable["scontrol_stdout_preserved"] is False
    live = launch4["unsealed_time_bounded_live_observation"]
    assert live == {
        "provenance": (
            "independent_operator_observation_immediately_after_abort_before_slurmctld_purge"
        ),
        "preserved_in_launch4_bytes": False,
        "canonical_report_dependency": "afterok:33211846_*(unfulfilled)",
        "kill_on_invalid_dependent": "Yes",
    }
    history = launch4["scheduler_history"]
    assert history["squeue_matching_rows"] == 0
    assert history["sacct_command"][4:6] == ["-S", "2026-08-01"]
    assert history["top_level_row_count"] == 2
    assert history["array_task_row_count"] == 0
    assert [row["job_id"] for row in history["sacct_rows"]] == ["33211848", "33211846"]
    assert all(
        row["state"] == "CANCELLED"
        and row["elapsed_raw"] == 0
        and row["allocated_nodes"] == 0
        and row["node_list"] == "None assigned"
        and row["exit_code"] == "0:0"
        and row["derived_exit_code"] == "0:0"
        and row["reason"] == "None"
        for row in history["sacct_rows"]
    )
    assert launch5["status"] == (
        "cancelled_after_trainer_bootstrap_failure_before_hydra_composition"
    )
    assert launch5["source_commit"] == "332a26f2f88e627f842eebbfc8310978ad606898"
    assert launch5["package_protocol_sha256"] == (
        "e9ac9f39e9261ca6ab0dcd5aadeba3dd3eb4ec25c999846c55fc09b2168c62a7"
    )
    assert launch5["contract_sha256"] == launch5["submission_sha256"] == (
        "8848790ca118a2fbf07b3a9f2edcceaec032c5005fc5eb7d918d55e066713abe"
    )
    assert launch5["receipt_sha256"] == (
        "463397088705144887fa8c75d6b40f3e770dca3f891d818e4178c9351672fbd5"
    )
    assert launch5["preserved_tree"]["preserved_root_path_base"] == "run_root"
    assert launch5["preserved_tree"]["preserved_root_aggregate_sha256"] == (
        "c07dce9aa58352f790af94bff8c719a3e9c8639bdd268d5b7d33824db8b7a874"
    )
    assert launch5["preserved_tree"]["task_root_aggregate_sha256"] == (
        "31d41fd81ee8092c0ebdfc68e9cd5199e81698fb62a7a2a88e5f2c2a6f5666f5"
    )
    assert launch5["failure_logs"]["deterministic_failed_cell_indices"] == list(
        range(12)
    )
    assert launch5["failure_logs"]["raw_log_sha256"] == (
        "4a624c03f806664ce70d0b98af2b8ea3e6f61a24ef4160357348441aa93b405b"
    )
    later = launch5["unsealed_later_terminal_scheduler_observation"]
    assert later["preserved_in_launch5_bytes"] is False
    assert later["observation_utc"] == "2026-08-28T04:16:52Z"
    assert later["sacct_raw_stdout_bytes"] == 5116
    assert later["sacct_raw_stdout_sha256"] == (
        "4a741ee0ece1e84644bf1628ba2666e2e35d3e2c906fdd643a8beaccedff7429"
    )
    ledger = {
        "schema_version": 1,
        "columns": later["columns"],
        "rows": [row.split("|") for row in later["serialized_rows"]],
    }
    assert len(ledger["rows"]) == later["sacct_row_count"] == 21
    assert {len(row) for row in ledger["rows"]} == {14}
    assert hashlib.sha256(
        json.dumps(
            ledger,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    ).hexdigest() == later["canonical_ledger_sha256"]
    assert later["squeue_raw_stdout_bytes"] == later["squeue_row_count"] == 0
    assert later["squeue_raw_stdout_sha256"] == hashlib.sha256(b"").hexdigest()
    scientific = launch5["scientific_state"]
    assert scientific["submission_ready_to_commit_journal_0005_committed"] is True
    assert scientific["scheduler_report_submitted_journal_0004_committed"] is True
    assert scientific["scientific_ready_marker_committed"] is False
    assert scientific["scientific_report_bundle_committed"] is False
    assert scientific["hydra_composition_completed"] is False
    assert scientific["model_constructed"] is False
    assert scientific["optimizer_updates"] == 0
    assert launch5["submission_contract_committed"] is True
    assert launch5["submission_receipt_committed"] is True
    assert launch5["cancel_latch_committed"] is True
    assert launch6["status"] == (
        "cancelled_after_retrospective_tensorboard_scalar_identity_failure"
    )
    assert launch6["source_commit"] == "d09e842acaf7909edf4ea6ba29138ea5c646fc1a"
    assert launch6["package_protocol_sha256"] == (
        "33288668441622bb30b205c98a0373e96f2c11f5ec5ba0e76bd5255098a8b7bd"
    )
    assert launch6["contract_sha256"] == launch6["submission_sha256"] == (
        "e2758413a5bb28af05b99441f0f6e27e279ba2940840be31403fe7cc6870649e"
    )
    assert launch6["receipt_sha256"] == (
        "ab0574d776dec229420e207752abc257cbe3f17115def2061a1248d22d12a110"
    )
    assert launch6["transaction_lock"] == (
        "outputs/.exp23-34d79ab13d65ef27.transaction.lock"
    )
    preserved = launch6["preserved_tree"]
    assert preserved["run_root"] == {
        "directory_count_including_root": 297,
        "regular_file_count": 629,
        "regular_file_bytes": 2009101434,
        "symlink_count": 80,
        "special_file_count": 0,
        "canonical_json_bytes": 101693,
        "aggregate_sha256": (
            "0ba872f7a03f42a58dfd9dcbc55afef0ec94dba6bef66744845c12f55cd340a8"
        ),
    }
    assert preserved["submission_root"]["aggregate_sha256"] == (
        "98d131fbd80b46d34e978b9b23f7c0137bc07711cc32b35bd89877a49f8c0242"
    )
    assert preserved["task_root"]["aggregate_sha256"] == (
        "07217ed94c148f7751c46c776b114c593f6b92a17d79c45aceeaa55fa0895ec4"
    )
    assert preserved["symlink_target_envelope"]["aggregate_sha256"] == (
        "fb1ef56fa6e93ae69f01e54438a377c36e9978ab64af8232f9f109582fecbfb4"
    )
    cancellation = launch6["cancellation"]
    assert cancellation["latch_sha256"] == (
        "a95defbc689623a08d48cead6c6959a585b461a6a70894b56d57d021c52584aa"
    )
    assert cancellation["call_token"] == "1787897778572633216-1585470"
    assert cancellation["call_sha256"] == (
        "f706ebbbd16f2ecaa0a2c10af38a6fb3251b7323e6399f970090878d7f3e82fe"
    )
    assert cancellation["result_sha256"] == (
        "cb3be62a3e4a0e7ca9b9b61ee30771f32099aa61a671b5efb044cb95c7c5909e"
    )
    assert cancellation["task_cancel_latch_aggregate_sha256"] == (
        "c606ca738b0ee29a6cec57d670dae6bb17abeb25c35d4cbccec6885f1cdc947c"
    )
    checkpoints = launch6["partial_checkpoints"]
    assert checkpoints["row_count"] == len(checkpoints["rows"]) == 20
    assert sum(row["completed_updates"] for row in checkpoints["rows"]) == 102017
    checkpoint_payload = {
        "schema_version": 1,
        "validation": checkpoints["validation"],
        "rows": [
            {**row, **checkpoints["row_constant_fields"]}
            for row in checkpoints["rows"]
        ],
    }
    checkpoint_bytes = json.dumps(
        checkpoint_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    assert len(checkpoint_bytes) == checkpoints["canonical_json_bytes"] == 3867
    assert hashlib.sha256(checkpoint_bytes).hexdigest() == checkpoints[
        "aggregate_sha256"
    ] == "b4498f2787600681b5c90862c48f23571be6fce23343c9f5957aae5e993628b5"
    scalars = launch6["tensorboard_scalar_evidence"]
    assert (
        scalars["scalar_event_records"],
        scalars["unique_cell_tag_steps"],
        scalars["duplicate_groups"],
        scalars["conflict_groups"],
        scalars["identical_groups"],
    ) == (803745, 802659, 718, 560, 158)
    assert (
        scalars["duplicate_extra_occurrences"],
        scalars["conflict_extra_occurrences"],
        scalars["identical_extra_occurrences"],
    ) == (1086, 722, 364)
    assert scalars["later_occurrences_compared_with_first"] == {
        "value_conflicting": 715,
        "bit_identical": 371,
        "total": 1086,
        "counting_rule": (
            "Each occurrence after the first in a group is compared only with "
            "that group's first occurrence; these are not group classifications."
        ),
    }
    assert scalars["event_file_map_sha256"] == (
        "91f77de5c3313a519312cde4244ae08fea0788180d5f2ceefeb8e08b5ccd8da2"
    )
    assert scalars["full_census_sha256"] == (
        "26c7127632470128fdfc92441ce2fce5b74f95a17195c4fa07175ef236926f98"
    )
    assert scalars["conflict_census_sha256"] == (
        "f4a6f0f22549617004c5390601dada4e6acbebe6a3393902a0cae90f9679b053"
    )
    assert scalars["identical_census_sha256"] == (
        "04ce578d994c36af58a3db3ad7fad527290831059c96e62964557b2a84c673ea"
    )
    assert scalars["per_cell_census_sha256"] == (
        "47bd1c52254571a74cf32388484c67f318dbdb41b0f8356741c23d978bb23a5e"
    )
    assert scalars["validation_aliases"]["groups"] == 534
    assert scalars["visualization_structural_aliases"]["groups"] == 184
    assert scalars["visualization_structural_aliases"]["extra_occurrences"] == 552
    blockers = launch6["report_blockers"]
    assert blockers["immutable_launch6_axis_contract_defect"]["tags"] == [
        "expansion/gain_rank_correlation",
        "expansion/gain_pairwise_accuracy",
        "expansion/gain_eligible_decision_fraction",
        "expansion/gain_ordered_pair_count",
        "expansion/gain_pair_coverage_fraction",
        "tree/support_recall",
        "tree/support_precision",
    ]
    assert "pre-fix reporter rejects the 80 W&B symlinks" in blockers[
        "immutable_launch6_reporter_order"
    ][1]
    assert "dense 50-update training axis" in blockers[
        "current_launch7_candidate_reporter_probe"
    ]
    terminal = launch6["unsealed_later_terminal_scheduler_observation"]
    assert terminal["preserved_in_launch6_bytes"] is False
    assert terminal["sacct_raw_stdout_bytes"] == 5320
    assert terminal["sacct_raw_stdout_sha256"] == (
        "ed61d986df986bca184c5355acadabb388446e5ee39ab9282ffb75ea780e470e"
    )
    terminal_payload = {
        "schema_version": 1,
        "columns": terminal["columns"],
        "rows": [row.split("|") for row in terminal["serialized_rows"]],
    }
    assert len(terminal_payload["rows"]) == 21
    assert {len(row) for row in terminal_payload["rows"]} == {14}
    terminal_bytes = json.dumps(
        terminal_payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    assert len(terminal_bytes) == terminal["canonical_ledger_json_bytes"] == 6152
    assert hashlib.sha256(terminal_bytes).hexdigest() == terminal[
        "canonical_ledger_sha256"
    ] == "447a0b38dbbd0ead850aa1eb16010e67334452a698f103824393cd0341cbce4d"
    assert launch6["scientific_state"]["optimizer_update_total_across_cells"] == 102017
    assert launch7["status"] == "terminal_failed_compute_scheduler_client_topology"
    assert [
        *launch7["job_ids_by_role"]["wave0_train"],
        *launch7["job_ids_by_role"]["report"],
    ] == ["33236584", "33236586"]
    assert launch7["negative_provenance"] == {
        "path": "launch7_negative_provenance.json",
        "raw_sha256": "29051e9839b9ceff4160b8ea0e99e82ce449cd7c2306f1e3604b30f24bb0272e",
        "canonical_sha256": "48839a4f58214d7a1b616f2f43089e24e16f44717865bac8c3c76845c4457e62",
        "scheduler_terminal_rows": 21,
        "ready_checkpoint_cells": 16,
        "completed_cells": 4,
        "report_started": False,
        "active_scheduler_jobs_after_terminal": 0,
    }
    for row in superseded[:2]:
        assert row["submission_sha256"] is None
        assert row["known_job_ids"] == []
    for row in superseded[:4]:
        for key in (
            "submission_receipt_committed",
            "scientific_run_started",
            "checkpoint_created",
            "wandb_run_created",
            "results_consumed",
            "checkpoints_consumed",
            "reuse_allowed",
            "resume_allowed",
            "retry_allowed",
            "recovery_allowed",
        ):
            assert row[key] is False
        assert row["optimizer_updates"] == 0
    for row in superseded:
        for key in ("reuse_allowed", "resume_allowed", "retry_allowed", "recovery_allowed"):
            assert row[key] is False
    assert launch2["submission_contract_committed"] is False
    assert manifest["paths"]["run_root"].endswith(
        "/outputs/treewm-executable-prefix-repair-pilot-v1-launch8"
    )
    assert manifest["paths"]["transaction_lock"] == (
        "outputs/.exp23-c85fcaba919d617f.transaction.lock"
    )
    assert all(manifest["paths"]["run_root"] != row["run_root"] for row in superseded)
    assert manifest["logging"]["wandb_project"].endswith("-launch8")
    assert manifest["logging"]["wandb_group"].endswith("-launch8")

    tampered = copy.deepcopy(manifest)
    tampered["paths"]["run_root"] = launch2["run_root"]
    with pytest.raises(campaign.ContractError):
        campaign.validate_manifest(tampered, lock, REPO)
    tampered = copy.deepcopy(manifest)
    tampered["superseded_launches"][1]["resume_allowed"] = True
    with pytest.raises(campaign.ContractError):
        campaign.validate_manifest(tampered, lock, REPO)
    tampered = copy.deepcopy(manifest)
    tampered["design"]["fresh_start_policy"] = (
        "Every Launch8 cell starts from scratch; no superseded state may be "
        "imported, reused, or resumed."
    )
    with pytest.raises(
        campaign.ContractError, match="superseded-launch exclusion policy differs"
    ):
        campaign.validate_manifest(tampered, lock, REPO)
    tampered = copy.deepcopy(manifest)
    tampered["design"]["fresh_start_policy"] = manifest["design"][
        "fresh_start_policy"
    ].replace("authorization-bound", "unbound")
    with pytest.raises(
        campaign.ContractError, match="superseded-launch exclusion policy differs"
    ):
        campaign.validate_manifest(tampered, lock, REPO)


def test_canary1_negative_provenance_is_exact_and_never_reused():
    manifest, _lock = contracts()
    binding = manifest["launch_contract"]["real_gpu_two_wave_canary"][
        "failed_attempts"
    ]
    assert campaign.exact_json_equal(binding, campaign.FAILED_CANARY_ATTEMPTS)
    assert binding[0]["job_ids_by_role"] == {
        "wave0": ["33285485"],
        "wave1": ["33285486"],
        "report": [],
    }
    assert binding[0]["state_file_map_canonical_sha256"] == (
        "61662d488cf571f26362e3392e49f63239153ef443b515d7c79bcbf094d05648"
    )
    campaign._validate_canary1_negative_provenance(manifest, REPO)
    artifact = campaign.read_json(PACKAGE / binding[0]["path"])
    census = artifact["terminal_state_file_census"]
    assert len(census["files"]) == census["state_file_count"] == 13
    assert sum(row["size"] for row in census["files"].values()) == 290380
    assert campaign.stable_hash(
        {"schema_version": 1, "files": census["files"]}
    ) == census["state_file_map_canonical_sha256"]
    terminal = artifact["scheduler_terminal_observation"]
    rows = artifact["scheduler_terminal_rows"]
    assert campaign.stable_hash(
        {
            "schema_version": 1,
            "fields": artifact["scheduler_terminal_rows_schema"].split("|"),
            "rows": rows,
        }
    ) == terminal["canonical_reduced_rows_sha256"]
    conclusion = artifact["negative_conclusion"]
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
    ):
        assert conclusion[key] is False


@pytest.mark.parametrize(
    "mutation,error",
    (
        ("job-id", "durable controller outcome differs"),
        ("state-file-size", "state census summary differs"),
        ("scheduler-row", "scheduler terminal rows differ"),
        ("authorization-flag", "durable lifecycle outcome differs"),
        ("release-flag", "durable lifecycle outcome differs"),
        ("scope", "negative provenance envelope differs"),
        ("reconstruction", "canonical reconstruction differs"),
        ("hold", "durable wave0 hold evidence differs"),
        ("retained-state", "retained scalar dependency observation differs"),
        ("successor-policy", "required successor policy differs"),
        ("reuse", "terminal/no-reuse conclusion differs"),
    ),
)
def test_canary1_negative_provenance_rejects_coherently_rehashed_mutations(
    tmp_path, monkeypatch, mutation, error
):
    manifest, _lock = contracts()
    value = campaign.read_json(PACKAGE / "canary1_negative_provenance.json")
    if mutation == "job-id":
        value["durable_controller_outcome"][
            "scheduler_assigned_job_ids_by_role"
        ]["wave1"] = ["33285487"]
    elif mutation == "state-file-size":
        value["terminal_state_file_census"]["files"][
            "CANARY_ABORTED.json"
        ]["size"] += 1
    elif mutation == "scheduler-row":
        value["scheduler_terminal_rows"][0] = value["scheduler_terminal_rows"][
            0
        ].replace("CANCELLED by 147230", "COMPLETED")
    elif mutation == "authorization-flag":
        value["durable_controller_outcome"]["authorization_committed"] = True
    elif mutation == "release-flag":
        value["durable_controller_outcome"]["wave0_released"] = True
    elif mutation == "scope":
        value["scope"] = "safe to reuse"
    elif mutation == "reconstruction":
        value["canonical_reconstruction"]["scope"] = "external bytes are authoritative"
    elif mutation == "hold":
        value["durable_controller_outcome"]["wave0_accepted_hold"]["state"] = (
            "RUNNING"
        )
    elif mutation == "retained-state":
        value["retained_wave1_scheduler_observation"]["parsed_fields"][
            "JobState"
        ] = "COMPLETED"
    elif mutation == "successor-policy":
        value["negative_conclusion"]["required_successor_policy"] = "reuse allowed"
    else:
        value["negative_conclusion"]["reuse_allowed"] = True
    payload = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("ascii")
    package = tmp_path / mutation / "package"
    package.mkdir(parents=True)
    artifact = package / "canary1_negative_provenance.json"
    artifact.write_bytes(payload)
    binding = copy.deepcopy(campaign.FAILED_CANARY_ATTEMPTS)
    binding[0]["raw_sha256"] = hashlib.sha256(payload).hexdigest()
    binding[0]["canonical_sha256"] = campaign.stable_hash(value)
    tampered = copy.deepcopy(manifest)
    tampered["launch_contract"]["real_gpu_two_wave_canary"][
        "failed_attempts"
    ] = binding
    with monkeypatch.context() as patch:
        patch.setattr(campaign, "PACKAGE_RELATIVE", Path(mutation) / "package")
        patch.setattr(campaign, "FAILED_CANARY_ATTEMPTS", binding)
        with pytest.raises(campaign.ContractError, match=error):
            campaign._validate_canary1_negative_provenance(tampered, tmp_path)


def test_canary1_negative_provenance_rejects_symlink(tmp_path, monkeypatch):
    manifest, _lock = contracts()
    package = tmp_path / "package"
    package.mkdir()
    (package / "canary1_negative_provenance.json").symlink_to(
        PACKAGE / "canary1_negative_provenance.json"
    )
    monkeypatch.setattr(campaign, "PACKAGE_RELATIVE", Path("package"))
    with pytest.raises(campaign.ContractError, match="regular nonsymlink"):
        campaign._validate_canary1_negative_provenance(manifest, tmp_path)


def _canary2_fixture_package(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[dict, Path]:
    package = tmp_path / "package"
    package.mkdir()
    for name in campaign.ACCEPTED_CANARY_CURRENT_SOURCE_SHA256:
        shutil.copyfile(PACKAGE / name, package / name)
    shutil.copyfile(
        PACKAGE / "canary2_acceptance_provenance.json",
        package / "canary2_acceptance_provenance.json",
    )
    monkeypatch.setattr(campaign, "PACKAGE_RELATIVE", Path("package"))
    manifest, _lock = contracts()
    return manifest, package


def _rebind_canary2_fixture(
    manifest: dict,
    package: Path,
    value: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = (campaign.canonical_json(value) + "\n").encode("ascii")
    path = package / "canary2_acceptance_provenance.json"
    path.write_bytes(payload)
    binding = copy.deepcopy(campaign.ACCEPTED_CANARY_ATTEMPTS[0])
    binding["raw_sha256"] = hashlib.sha256(payload).hexdigest()
    binding["canonical_sha256"] = campaign.stable_hash(value)
    manifest["launch_contract"]["real_gpu_two_wave_canary"][
        "accepted_attempts"
    ] = [copy.deepcopy(binding)]
    evidence = manifest["launch_contract"]["real_gpu_two_wave_canary"][
        "production_authorization_evidence"
    ]
    evidence["raw_sha256"] = binding["raw_sha256"]
    evidence["canonical_sha256"] = binding["canonical_sha256"]
    monkeypatch.setattr(campaign, "ACCEPTED_CANARY_ATTEMPTS", [binding])


def test_canary2_acceptance_provenance_is_exact_hermetic_and_authorizing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, package = _canary2_fixture_package(tmp_path, monkeypatch)
    opened: list[Path] = []
    original = campaign._regular_file_bytes

    def guarded(path: Path, label: str, *, max_bytes: int) -> bytes:
        candidate = Path(path)
        assert candidate.is_relative_to(package)
        assert "outputs" not in candidate.parts
        opened.append(candidate)
        return original(candidate, label, max_bytes=max_bytes)

    monkeypatch.setattr(campaign, "_regular_file_bytes", guarded)
    campaign._validate_canary2_acceptance_provenance(manifest, tmp_path)
    assert package / "canary2_acceptance_provenance.json" in opened
    assert len(opened) == (
        len(campaign.ACCEPTED_CANARY_CURRENT_SOURCE_SHA256) + 1
    )
    evidence = manifest["launch_contract"]["real_gpu_two_wave_canary"][
        "production_authorization_evidence"
    ]
    assert evidence["required"] is evidence["satisfied"] is True
    assert evidence["scientific_runtime_input_consumption_allowed"] is False


@pytest.mark.parametrize(
    "mutation,pattern",
    (
        ("state-mode", "state file map differs"),
        ("wave0-token", "runtime/report lineage differs"),
        ("terminal-state", "scheduler terminal rows differ"),
        ("owner-stray", "owner-wide zero-stray census envelope differs"),
        ("reuse", "terminal acceptance/no-reuse conclusion differs"),
    ),
)
def test_canary2_acceptance_rejects_coherently_rebound_semantic_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    pattern: str,
) -> None:
    manifest, package = _canary2_fixture_package(tmp_path, monkeypatch)
    value = json.loads(
        (package / "canary2_acceptance_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    if mutation == "state-mode":
        value["terminal_state_file_census"]["files"][
            "CANARY_AUTHORIZATION.json"
        ]["mode"] = "0777"
        state_payload = {
            "schema_version": 1,
            "files": value["terminal_state_file_census"]["files"],
        }
        changed = campaign.stable_hash(state_payload)
        value["terminal_state_file_census"][
            "state_file_map_canonical_sha256"
        ] = changed
        binding = manifest["launch_contract"]["real_gpu_two_wave_canary"][
            "accepted_attempts"
        ][0]
        binding["state_file_map_canonical_sha256"] = changed
    elif mutation == "wave0-token":
        value["runtime_result"]["wave0_ready"]["record"][
            "canary_token"
        ] = "0123456789abcdef"
    elif mutation == "terminal-state":
        value["scheduler_terminal_rows"][0][2] = "FAILED"
        rows = value["scheduler_terminal_rows"]
        raw = ("\n".join("|".join(row) for row in rows) + "\n").encode("ascii")
        scheduler = value["scheduler_terminal_observation"]
        scheduler["raw_stdout"] = raw.decode("ascii")
        scheduler["raw_stdout_size"] = len(raw)
        scheduler["raw_stdout_sha256"] = hashlib.sha256(raw).hexdigest()
        reduced = {
            "schema_version": 1,
            "fields": value["scheduler_terminal_rows_schema"].split("|"),
            "rows": rows,
        }
        scheduler["canonical_rows_bytes"] = len(
            campaign.canonical_json(reduced).encode("ascii")
        )
        scheduler["canonical_rows_sha256"] = campaign.stable_hash(reduced)
    elif mutation == "owner-stray":
        owner = value["terminal_zero_active_evidence"][
            "owner_active_scheduler_census"
        ]
        compressed = base64.b64decode(owner["raw_stdout_zlib_base64"])
        raw = zlib.decompress(compressed)
        row = (
            "999999|999999|exp23-launch8-canary-b95869841048e511-stray|"
            "PENDING|treewm-exp23-canary:b95869841048e511"
        )
        raw += (row + "\n").encode("ascii")
        compressed = zlib.compress(raw)
        owner["raw_stdout_size"] = len(raw)
        owner["raw_stdout_sha256"] = hashlib.sha256(raw).hexdigest()
        owner["compressed_stdout_size"] = len(compressed)
        owner["compressed_stdout_sha256"] = hashlib.sha256(compressed).hexdigest()
        owner["raw_stdout_zlib_base64"] = base64.b64encode(compressed).decode(
            "ascii"
        )
        owner["matched_rows"] = [row.split("|")]
        value["terminal_zero_active_evidence"]["stray_topology_job_count"] = 1
    else:
        value["acceptance_conclusion"]["reuse_allowed"] = True
    _rebind_canary2_fixture(manifest, package, value, monkeypatch)
    if mutation == "state-mode":
        binding = campaign.ACCEPTED_CANARY_ATTEMPTS[0]
        binding["state_file_map_canonical_sha256"] = value[
            "terminal_state_file_census"
        ]["state_file_map_canonical_sha256"]
        manifest["launch_contract"]["real_gpu_two_wave_canary"][
            "accepted_attempts"
        ] = [copy.deepcopy(binding)]
    with pytest.raises(campaign.ContractError, match=pattern):
        campaign._validate_canary2_acceptance_provenance(manifest, tmp_path)


@pytest.mark.parametrize("payload", (b"NaN\n", b"[]\n", b"{\"x\":null}\n"))
def test_canary2_acceptance_malformed_json_is_always_contract_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
) -> None:
    manifest, package = _canary2_fixture_package(tmp_path, monkeypatch)
    path = package / "canary2_acceptance_provenance.json"
    path.write_bytes(payload)
    binding = copy.deepcopy(campaign.ACCEPTED_CANARY_ATTEMPTS[0])
    binding["raw_sha256"] = hashlib.sha256(payload).hexdigest()
    manifest["launch_contract"]["real_gpu_two_wave_canary"][
        "accepted_attempts"
    ] = [copy.deepcopy(binding)]
    manifest["launch_contract"]["real_gpu_two_wave_canary"][
        "production_authorization_evidence"
    ]["raw_sha256"] = binding["raw_sha256"]
    monkeypatch.setattr(campaign, "ACCEPTED_CANARY_ATTEMPTS", [binding])
    with pytest.raises(campaign.ContractError):
        campaign._validate_canary2_acceptance_provenance(manifest, tmp_path)


def test_canary2_acceptance_missing_or_symlink_artifact_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, package = _canary2_fixture_package(tmp_path, monkeypatch)
    artifact = package / "canary2_acceptance_provenance.json"
    artifact.unlink()
    with pytest.raises(campaign.ContractError, match="unavailable"):
        campaign._validate_canary2_acceptance_provenance(manifest, tmp_path)
    artifact.symlink_to(PACKAGE / "canary2_acceptance_provenance.json")
    with pytest.raises(campaign.ContractError, match="regular nonsymlink"):
        campaign._validate_canary2_acceptance_provenance(manifest, tmp_path)


def test_canary2_acceptance_pins_current_controller_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, package = _canary2_fixture_package(tmp_path, monkeypatch)
    controller = package / "two_wave_canary.py"
    controller.write_bytes(controller.read_bytes() + b"\n")
    with pytest.raises(
        campaign.ContractError, match="post-acceptance runtime source bytes differ"
    ):
        campaign._validate_canary2_acceptance_provenance(manifest, tmp_path)


def test_manifest_schema_version_is_an_exact_integer() -> None:
    manifest, lock = contracts()
    manifest["schema_version"] = True
    with pytest.raises(campaign.ContractError, match="manifest schema differs"):
        campaign.validate_manifest(manifest, lock, REPO)


def test_terminal_report_repair_policy_and_sources_are_exact() -> None:
    manifest, lock = contracts()
    repair = manifest["launch_contract"]["terminal_report_repair"]
    assert repair == campaign.TERMINAL_REPORT_REPAIR_POLICY
    assert repair["attempt"] == repair["generation_count"] == 1
    assert repair["original_terminal_report"]["job_id"] == "33311218"
    assert repair["original_terminal_report"]["state"] == "FAILED"
    assert repair["original_terminal_report"]["exit_code"] == "2:0"
    assert repair["deterministic_reassembly"]["status"] == "rejected"
    assert repair["publication_contract"]["report_commit_exact_key_count"] == 14
    assert repair["publication_contract"]["scientific_input_change_allowed"] is False
    assert repair["publication_contract"]["gate_change_allowed"] is False
    assert repair["publication_contract"]["scientific_bundle_schema_changed"] is False
    assert repair["publication_contract"]["report_commit_schema_changed"] is False
    assert repair["publication_contract"]["exp24_adapter_change_required"] is False
    assert repair["scheduler_protocol"]["submit_held"] is True
    assert (
        repair["scheduler_protocol"][
            "fresh_owner_wide_empty_census_before_submit_calling"
        ]
        is True
    )
    assert repair["scheduler_protocol"]["slurm_walltime_seconds"] == 14_400
    assert repair["scheduler_protocol"]["release_evidence_wait_seconds"] == 10_800
    assert repair["scheduler_protocol"]["minimum_assembly_budget_seconds"] == 3_600
    assert repair["scheduler_protocol"]["release_wait_clock"] == "time.monotonic"
    assert repair["scheduler_protocol"]["terminal_worker_failure_blocks_publication"] is True
    assert repair["scheduler_protocol"]["sealed_source_root_bound_in_sbatch_argv"] is True
    assert repair["scheduler_protocol"]["publisher_nofollow_same_fd_hash_before_exec"] is True
    assert repair["actual_repair_submit_performed"] is False
    for name, expected in repair["repair_source_sha256"].items():
        assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == expected
    campaign.validate_manifest(manifest, lock, REPO)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("attempt",), 2),
        (("generation_count",), 2),
        (("actual_repair_submit_performed",), True),
        (("publication_contract", "report_commit_exact_key_count"), 15),
        (("publication_contract", "scientific_input_change_allowed"), True),
        (("publication_contract", "gate_change_allowed"), True),
        (("scheduler_protocol", "submit_held"), False),
        (("scheduler_protocol", "settled_census_rounds"), 2),
        (("scheduler_protocol", "release_evidence_wait_seconds"), 60),
        (("scheduler_protocol", "terminal_worker_failure_blocks_publication"), False),
        (("scheduler_protocol", "publisher_nofollow_same_fd_hash_before_exec"), False),
    ],
)
def test_terminal_report_repair_policy_drift_fails_closed(path, value) -> None:
    manifest, lock = contracts()
    tampered = copy.deepcopy(manifest)
    leaf = tampered["launch_contract"]["terminal_report_repair"]
    for key in path[:-1]:
        leaf = leaf[key]
    leaf[path[-1]] = value
    with pytest.raises(campaign.ContractError, match="terminal report repair contract"):
        campaign.validate_manifest(tampered, lock, REPO)


def test_each_matched_pair_differs_only_in_three_audited_weights():
    manifest, lock = contracts()
    cells = campaign.expand_matrix(manifest)
    for setting in campaign.SETTINGS:
        for seed in campaign.SEEDS:
            pair = [cell for cell in cells if cell.setting == setting and cell.seed == seed]
            left, right = [campaign.cell_overrides(cell, manifest, lock) for cell in pair]
            differing = {key for key in left if left[key] != right[key]}
            assert differing == set(campaign.WEIGHT_KEYS)
            for key in campaign.WEIGHT_KEYS:
                assert left[key] == 0.0
                assert right[key] > 0.0
            for name in campaign.PREFIX_TERMS:
                assert left[f"losses.enabled.{name}"] is True
                assert right[f"losses.enabled.{name}"] is True


def test_audited_tuple_and_fail_closed_tree_binding_are_exact():
    manifest, lock = contracts()
    assert lock["derived"]["weights"] == {
        "executable_prefix_action": 0.033368419,
        "executable_prefix_latent": 0.027645085,
        "executable_prefix_endpoint": 0.011350645,
    }
    assert lock["derived"]["post_scale_max_aggregate_ratio"] < 0.10
    assert len(lock["checkpoint_sha256"]) == 10
    assert len(lock["batch_sha256"]) == 20
    api = lock["fail_closed_api_binding"]
    assert api["effective_tree_config_required"] is True
    assert api["audit_call"] == "tree_config_for(cfg.arm, cfg_utils.tree_config(cfg), model)"
    source = (PACKAGE / "weight_audit.py").read_text(encoding="utf-8")
    assert "tree_cfg=tree_config_for(" in source
    assert hashlib.sha256((PACKAGE / "weight_audit.py").read_bytes()).hexdigest() == lock["source_sha256"]["audit"]


def test_bounds_are_per_setting_env_arrays_not_an_unbound_scalar_default():
    _, lock = contracts()
    for setting, contract in lock["action_bounds"].items():
        low = np.full((contract["action_dim"],), contract["lower"], dtype=np.float32)
        high = np.full((contract["action_dim"],), contract["upper"], dtype=np.float32)
        assert hashlib.sha256(low.tobytes()).hexdigest() == contract["lower_sha256"], setting
        assert hashlib.sha256(high.tobytes()).hexdigest() == contract["upper_sha256"], setting
        assert np.all(low < high)


def test_prefix_rule_accepts_short_logged_continuations_and_forbids_mean_four_gate():
    manifest, lock = contracts()
    campaign.validate_manifest(manifest, lock, REPO)
    rule = manifest["scientific_contract"]["prefix_length_rule"]
    target_rule = manifest["acceptance"]["prefix_structural_gates"]["target_rule"]
    assert "min(4" in rule
    assert "never compared to 4" in target_rule
    tampered = copy.deepcopy(manifest)
    tampered["acceptance"]["prefix_structural_gates"]["target_rule"] = "require prefix_steps_mean == 4"
    with pytest.raises(campaign.ContractError):
        campaign.validate_manifest(tampered, lock, REPO)


def test_one_continuous_launch_has_no_stage_stop_and_binds_all_four_audits():
    manifest, lock = contracts()
    cell = campaign.expand_matrix(manifest)[0]
    launch = campaign.trainer_command(
        manifest,
        lock,
        cell,
        repo_root=REPO,
        package_protocol_sha256="0" * 64,
    )
    assert "TREEWM_STOP_AFTER_UPDATE" not in launch["environment"]
    assert all("TREEWM_STOP_AFTER_UPDATE" not in item for item in launch["argv"])
    assert launch["environment"]["TREEWM_RESOLVED_CONFIG_SHA256"] == manifest[
        "resolved_config_contract"
    ]["artifact_sha256"]
    assert launch["environment"]["TREEWM_CAUSAL_PARITY_SHA256"] == manifest[
        "causal_parity_contract"
    ]["artifact_sha256"]
    assert launch["environment"]["MUJOCO_GL"] == "egl"
    assert launch["environment"]["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    for key in (
        "weight_audit_artifact_sha256",
        "prefix_target_artifact_sha256",
        "resolved_config_artifact_sha256",
        "causal_parity_artifact_sha256",
    ):
        assert launch["hashes"][key] == {
            "weight_audit_artifact_sha256": manifest["weight_audit"]["artifact_sha256"],
            "prefix_target_artifact_sha256": manifest["prefix_target_contract"]["artifact_sha256"],
            "resolved_config_artifact_sha256": manifest["resolved_config_contract"]["artifact_sha256"],
            "causal_parity_artifact_sha256": manifest["causal_parity_contract"]["artifact_sha256"],
        }[key]


def test_prefix_target_lock_proves_exact_logged_horizon_set():
    target_lock = campaign.read_json(PACKAGE / "prefix_target.lock.json")
    for row in target_lock["settings"].values():
        histogram = row["logged_selected_horizon_histogram"]
        assert set(histogram) == {"4", "8", "16", "32", "64"}
        assert sum(histogram.values()) == row["matched_branch_count"]


def test_generated_config_and_causal_locks_are_exact_prefix_stripped_payloads():
    for name in ("resolved_config.lock.json", "causal_parity.lock.json"):
        path = PACKAGE / name
        raw = path.read_bytes()
        value = campaign.read_json(path)
        # Both auditors print PREFIX + canonical_json(result) + newline.  The lock
        # must be the exact suffix, not a jq/pretty-print round trip that can coerce
        # JSON floats such as 1.0 into integers while leaving Python equality true.
        assert raw == (campaign.canonical_json(value) + "\n").encode("ascii")
        body = dict(value)
        claimed = body.pop("artifact_sha256")
        assert claimed == campaign.stable_hash(body)


def test_resolved_lock_cannot_rebind_a_coherently_rehashed_launch7_namespace(
    tmp_path, monkeypatch
):
    manifest, _ = contracts()
    stale = campaign.read_json(PACKAGE / "resolved_config.lock.json")
    row = stale["matrix"][0]
    row["resolved_config"]["campaign_id"] = (
        "treewm-executable-prefix-repair-pilot-v1-launch7"
    )
    row["resolved_config"]["run_root"] = manifest["paths"]["run_root"].replace(
        "launch8", "launch7"
    )
    row["resolved_config_sha256"] = campaign.stable_hash(row["resolved_config"])
    body = dict(stale)
    body.pop("artifact_sha256")
    stale["artifact_sha256"] = campaign.stable_hash(body)
    tampered_manifest = copy.deepcopy(manifest)
    tampered_manifest["resolved_config_contract"]["artifact_sha256"] = stale[
        "artifact_sha256"
    ]
    path = tmp_path / "resolved_config.lock.json"
    path.write_text(campaign.canonical_json(stale) + "\n", encoding="ascii")
    monkeypatch.setattr(campaign, "RESOLVED_CONFIG_LOCK_PATH", path)
    with pytest.raises(
        campaign.ContractError, match="resolved campaign/run-root identity differs"
    ):
        campaign._validate_resolved_config_lock(tampered_manifest)


def test_protocol_inventory_is_closed_and_deterministic():
    assert campaign.PROTOCOL_FILES == (
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
    assert len(campaign.PROTOCOL_FILES) == len(set(campaign.PROTOCOL_FILES))
    assert "protocol.sha256" not in campaign.PROTOCOL_FILES
    assert campaign.SNAPSHOT_IMPORT_FILES == {
        "configs/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "scripts/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    }
    campaign.validate_snapshot_import_files(REPO)
    assert campaign.protocol_sha256(PACKAGE) == campaign.protocol_sha256(PACKAGE)


def test_snapshot_import_markers_fail_closed_when_absent_or_replaced(tmp_path):
    (tmp_path / "scripts").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "scripts/__init__.py").write_bytes(b"")
    config_marker = tmp_path / "configs/__init__.py"

    with pytest.raises(campaign.ContractError, match="unavailable"):
        campaign.validate_snapshot_import_files(tmp_path)

    config_marker.write_text("# not the sealed empty marker\n", encoding="utf-8")
    with pytest.raises(campaign.ContractError, match="bytes differ"):
        campaign.validate_snapshot_import_files(tmp_path)

    config_marker.write_bytes(b"")
    campaign.validate_snapshot_import_files(tmp_path)
