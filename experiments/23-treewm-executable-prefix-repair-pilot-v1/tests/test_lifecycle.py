from __future__ import annotations

import argparse
import ast
import base64
import copy
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest


PACKAGE = Path(__file__).resolve().parents[1]
REPO = PACKAGE.parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, PACKAGE / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


worker = _load("exp23_lifecycle_worker", "worker.py")
train_entry = _load("exp23_lifecycle_train_entry", "train_entry.py")


def _scheduler_config_bytes() -> bytes:
    return (
        "ClusterName=cs-oci-ord\n"
        "SlurmctldHost=cs-oci-ord-a\n"
        "SlurmctldHost=cs-oci-ord-b\n"
        "SlurmctldPort=6817\n"
        "AuthType=auth/munge\n"
        "GresTypes=gpu\n"
        "CliFilterPlugins=lua\n"
        "JobSubmitPlugins=lua\n"
        "CommunicationParameters=NoAddrCache\n"
    ).encode("utf-8")


def _scheduler_critical() -> dict:
    return {
        key: worker.SCHEDULER_CONTROL_PLANE[key]
        for key in (
            "cluster_name",
            "slurmctld_hosts",
            "slurmctld_port",
            "auth_type",
            "gres_types",
            "cli_filter_plugins",
            "job_submit_plugins",
        )
    }


def _scheduler_observation() -> dict:
    payload = _scheduler_config_bytes()
    return {
        "schema_version": 1,
        "trust_model": worker.SCHEDULER_CONTROL_PLANE["trust_model"],
        "config": {
            "path": worker.SCHEDULER_CONTROL_PLANE["slurm_conf"],
            "sha256": hashlib.sha256(payload).hexdigest(),
            "identity": {"size": len(payload)},
        },
        "critical": _scheduler_critical(),
        "cli_filter_policy": {"files": {}, "tree_sha256": "a" * 64},
    }


def _scheduler_fallback() -> dict:
    payload = _scheduler_config_bytes()
    return {
        "schema_version": 1,
        "purpose": (
            "accepted-job exact reconciliation, cancellation, and requeue only; "
            "never submission"
        ),
        "encoding": "base64",
        "payload_base64": base64.b64encode(payload).decode("ascii"),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size": len(payload),
        "source_control_plane": _scheduler_observation(),
    }


def _scheduler_preclaim() -> dict:
    comment = "treewm-exp23:" + "0" * 64
    train_output = "/repo/scheduler-test-never-executed/logs/train_%A_%a.out"
    report_output = "/repo/scheduler-test-never-executed/logs/report_%j.out"
    return {
        "schema_version": 1,
        "status": "scheduler_preclaim_verified",
        "campaign_id": worker.CAMPAIGN_ID,
        "scheduler_control_plane": _scheduler_observation(),
        "controller_configuration": _scheduler_critical(),
        "sbatch_test_only": {
            "train": {
                "role": "train",
                "defined_options": {
                    "array": "0-19%20", "comment": comment, "export": "NONE",
                    "output": train_output, "parsable": "set", "test-only": "set",
                    "verbose": "3",
                },
                "decision": {},
                "warnings": [],
            },
            "report": {
                "role": "report",
                "defined_options": {
                    "comment": comment, "export": "NONE", "output": report_output,
                    "parsable": "set", "test-only": "set", "verbose": "3",
                },
                "decision": {},
                "warnings": [],
            },
        },
        "report_dependency_test": worker.REPORT_DEPENDENCY_TEST_REQUIREMENT,
        "scheduler_probe_commands": [
            ["scontrol"],
            ["squeue"],
            ["squeue"],
            [
                "sbatch", "-vvv", "--test-only", "--parsable", "--export=NONE",
                "--array=0-19%20",
                f"--comment={comment}", f"--output={train_output}", "train.slurm",
            ],
            [
                "sbatch", "-vvv", "--test-only", "--parsable", "--export=NONE",
                f"--comment={comment}", f"--output={report_output}", "report.slurm",
            ],
            ["squeue"],
            ["squeue"],
        ],
        "zero_job_proof": {
            "job_names": {
                "train": "exp23-launch7-scheduler-test-train",
                "report": "exp23-launch7-scheduler-test-report",
            },
            "pre_queries": 2,
            "post_queries": 2,
            "matching_jobs_before": 0,
            "matching_jobs_after": 0,
        },
        "scheduler_calls": 7,
        "scheduler_mutation_calls": 0,
        "persistent_writes_performed": 0,
    }


class FakeGeneratorState:
    def numel(self) -> int:
        return 16


def _tracker(completed: int) -> dict:
    remainder = completed % worker.LOG_EVERY
    if remainder == 0:
        return {"schema_version": 1, "sums": {}, "counts": {}, "hists": {}}
    counts = {
        name: (
            remainder
            if name in worker.PER_UPDATE_TRACKER_TAGS
            else remainder * worker.BATCH_SIZE
        )
        for name in worker.PER_UPDATE_TRACKER_TAGS | worker.PER_EXAMPLE_TRACKER_TAGS
    }
    return {
        "schema_version": 1,
        "sums": {name: float(count) / 10.0 for name, count in counts.items()},
        "counts": counts,
        "hists": {},
    }


def _loader(completed: int, anchors: int = 758_084) -> dict:
    batches = anchors // worker.BATCH_SIZE
    if completed == 0:
        return {
            "epoch": 0,
            "batches_yielded_in_epoch": 0,
            "epoch_generator_state": None,
        }
    return {
        "epoch": (completed - 1) // batches,
        "batches_yielded_in_epoch": ((completed - 1) % batches) + 1,
        "epoch_generator_state": FakeGeneratorState(),
    }


def _config() -> dict:
    return {
        "train": {
            "batch_size": 256,
            "grad_accum": 1,
            "eval_every": 12_500,
            "viz_early_until": 2_000,
            "viz_every_early": 1_000,
            "viz_every": 25_000,
        }
    }


def _manifest() -> dict:
    return {
        "scientific_contract": {
            "task_ids": [1, 2, 3, 4, 5],
            "final_episodes_per_task": 5,
            "evaluation_seed": 2718,
        }
    }


def _run_layout(run: Path) -> dict:
    return {
        "run_root": run.parent,
        "run_relative": Path(run.name),
        "run_directory": run,
    }


def _episode(task_index: int, episode_index: int) -> dict:
    task_id = task_index + 1
    return {
        "success": episode_index == 0,
        "steps": 10,
        "replans": 2,
        "nodes": 8,
        "final_goal_distance": 0.8,
        "best_goal_distance": 0.7,
        "chunk_lengths": [4, 4, 2],
        "selected_depths": [1, 2, 1],
        "initial_goal_distance": 1.0,
        "displacement": 0.5,
        "path_length": 1.2,
        "action_magnitude": 0.4,
        "no_action_plans": 0,
        "guard_plans": 2,
        "guard_rejections": 1,
        "guard_candidate_count": 8,
        "guard_accepted_count": 7,
        "guard_best_predicted_improvements": [0.3, -0.1],
        "guard_selected_predicted_improvements": [0.2, -0.1],
        "trajectory": [[0.0, 1.0], [0.1, 0.9]],
        "progress": {"subgoal_gain": 0.1},
        "task_index": task_index,
        "task_id": task_id,
        "episode_index": episode_index,
        "episode_seed": 2718 + 1000 * task_index + episode_index,
        "planning_wall_clock_s": 0.25,
    }


def _episodes() -> list[dict]:
    return [_episode(task, episode) for task in range(5) for episode in range(5)]


def _progress(rows: list[dict], status: str = "in_progress") -> dict:
    value = {
        "schema_version": 1,
        "objective_version": worker.OBJECTIVE,
        "status": status,
        "identity_sha256": "a" * 64,
        "seed_table_sha256": "b" * 64,
        "task_ids": [1, 2, 3, 4, 5],
        "episodes_per_task": 5,
        "completed_results": rows,
        "generator_state": [1, 2, 3],
    }
    if status == "complete":
        value["metrics"] = {
            "eval/num_episodes": 25.0,
            "eval/successes": 5.0,
            "eval/success_rate": 0.2,
            "eval/distance_reduction_frac": 0.2,
            **{
                f"eval/task{task_id}/{name}": metric
                for task_id in range(1, 6)
                for name, metric in (
                    ("num_episodes", 5.0),
                    ("successes", 1.0),
                    ("success_rate", 0.2),
                )
            },
        }
    return value


def _write_json(path: Path, value: dict) -> None:
    worker.seal_json(path, value)


def test_metric_tracker_requires_exact_unflushed_window() -> None:
    worker.validate_metric_tracker_state(_tracker(37), 37)
    worker.validate_metric_tracker_state(_tracker(50), 50)

    wrong = _tracker(37)
    name = next(iter(worker.PER_EXAMPLE_TRACKER_TAGS))
    wrong["counts"][name] -= 1
    with pytest.raises(worker.LifecycleError, match="cadence differs"):
        worker.validate_metric_tracker_state(wrong, 37)

    stale_hist = _tracker(37)
    stale_hist["hists"] = {"tree/keep_scores": [[0.5]]}
    with pytest.raises(worker.LifecycleError, match="histogram window is stale"):
        worker.validate_metric_tracker_state(stale_hist, 37)

    with pytest.raises(worker.LifecycleError, match="nonempty at a logging boundary"):
        worker.validate_metric_tracker_state(_tracker(49), 50)


def test_loader_cursor_is_exact_across_epoch_boundary() -> None:
    anchors = 758_084
    batches = anchors // 256
    for completed in (0, 1, batches, batches + 1, 25_000):
        worker.validate_loader_state(
            _loader(completed, anchors),
            completed,
            train_anchor_count=anchors,
        )
    wrong = _loader(batches + 1, anchors)
    wrong["epoch"] = 0
    with pytest.raises(worker.LifecycleError, match="loader epoch differs"):
        worker.validate_loader_state(wrong, batches + 1, train_anchor_count=anchors)


def test_cadence_accepts_periodic_replay_and_terminal_pending_25k() -> None:
    config = _config()
    worker.validate_post_update_cadence(
        {
            "phase": "train",
            "pending_eval_step": 12_500,
            "post_update_cadence": {
                "schema_version": 1,
                "committed_update": 12_500,
                "completed_update": 12_499,
                "replay_action": "evaluation",
            },
        },
        12_500,
        config,
    )
    worker.validate_post_update_cadence(
        {
            "phase": "final_eval",
            "pending_eval_step": 25_000,
            "post_update_cadence": {
                "schema_version": 1,
                "committed_update": 25_000,
                "completed_update": 25_000,
                "replay_action": None,
            },
        },
        25_000,
        config,
    )
    with pytest.raises(worker.LifecycleError, match="training cadence retains"):
        worker.validate_post_update_cadence(
            {
                "phase": "train",
                "pending_eval_step": 25_000,
                "post_update_cadence": {
                    "schema_version": 1,
                    "committed_update": 25_000,
                    "completed_update": 25_000,
                    "replay_action": None,
                },
            },
            25_000,
            config,
        )


def test_checkpoint_accepts_terminal_pending_and_train_at_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    identity = {"train_anchor_count": 758_084, "final_seed_table_sha256": "b" * 64}
    monkeypatch.setattr(worker, "expected_run_identity", lambda _context: identity)
    (tmp_path / "run").mkdir()
    context = {
        "resolved_row": {"resolved_config": _config()},
        "launch": {"cell": {"run_directory": str(tmp_path / "run")}},
        "manifest": _manifest(),
        **_run_layout(tmp_path / "run"),
    }

    def payload(phase: str, pending: int | None, reason: str) -> dict:
        return {
            "run_identity": identity,
            "identity_sha256": worker.stable_hash(identity),
            "config": _config(),
            "completed_updates": 25_000,
            "step": 25_000,
            "next_step": 25_000,
            "epoch": _loader(25_000)["epoch"],
            "rank_states": [
                {
                    "rank": 0,
                    "metric_tracker": _tracker(25_000),
                    "loader": _loader(25_000),
                }
            ],
            "post_update_cadence": {
                "schema_version": 1,
                "committed_update": 25_000,
                "completed_update": 25_000,
                "replay_action": None,
            },
            "phase": phase,
            "pending_eval_step": pending,
            "final_eval": None,
            "reason": reason,
        }

    terminal = worker.validate_checkpoint_payload(
        payload("final_eval", 25_000, "final-evaluation-pending"),
        context,
        validate_shared=False,
    )
    assert terminal["kind"] == "final_pending"
    at_target = worker.validate_checkpoint_payload(
        payload("train", None, "graceful-stop:SIGUSR1"),
        context,
        validate_shared=False,
    )
    assert at_target["kind"] == "train_at_target"


def test_exact_25_in_progress_rows_are_resumable(tmp_path: Path) -> None:
    path = tmp_path / "final_eval_progress.json"
    _write_json(path, _progress(_episodes(), "in_progress"))
    state = worker.validate_final_progress(
        path,
        expected_identity_sha256="a" * 64,
        expected_seed_table_sha256="b" * 64,
        manifest=_manifest(),
    )
    assert state is not None
    assert state["status"] == "in_progress"
    assert state["row_count"] == 25


def test_final_rows_reject_nested_nonfinite_and_nonfinite_progress() -> None:
    expected = (0, 1, 0, 2718)
    nested_nan = _episode(0, 0)
    nested_nan["trajectory"] = [[0.0, [float("nan")]]]
    with pytest.raises(worker.LifecycleError, match="nonnumeric leaf"):
        worker.validate_episode_row(nested_nan, expected)

    bad_progress = _episode(0, 0)
    bad_progress["progress"] = {"gain": float("inf")}
    with pytest.raises(worker.LifecycleError, match="non-finite"):
        worker.validate_episode_row(bad_progress, expected)


def test_final_progress_rejects_extra_fields_and_all_symlinks(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    _write_json(target, _progress([_episode(0, 0)]))
    linked = tmp_path / "linked.json"
    linked.symlink_to(target)
    with pytest.raises(worker.LifecycleError, match="symlink"):
        worker.validate_final_progress(
            linked,
            expected_identity_sha256="a" * 64,
            expected_seed_table_sha256="b" * 64,
            manifest=_manifest(),
        )

    broken = tmp_path / "broken.json"
    broken.symlink_to(tmp_path / "absent.json")
    with pytest.raises(worker.LifecycleError, match="symlink"):
        worker.validate_final_progress(
            broken,
            expected_identity_sha256="a" * 64,
            expected_seed_table_sha256="b" * 64,
            manifest=_manifest(),
        )

    extra = _progress([_episode(0, 0)])
    extra["unexpected"] = 1
    extra_path = tmp_path / "extra.json"
    _write_json(extra_path, extra)
    with pytest.raises(worker.LifecycleError, match="fields differ"):
        worker.validate_final_progress(
            extra_path,
            expected_identity_sha256="a" * 64,
            expected_seed_table_sha256="b" * 64,
            manifest=_manifest(),
        )


def _terminal_case(
    run: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str | None = None,
) -> tuple[dict, dict]:
    run.mkdir(parents=True)
    identity = {
        "evaluation_seed_tables_sha256": "e" * 64,
        "final_seed_table_sha256": "f" * 64,
        "protocol_sha256": "1" * 64,
        "code_sha256": "2" * 64,
        "runtime_sha256": "3" * 64,
        "data_manifest_sha256": "4" * 64,
        "calibration_sha256": "5" * 64,
        "future_recipe_sha256": "6" * 64,
        "recipe_code_sha256": "7" * 64,
        "recipe_runtime_sha256": "8" * 64,
        "arm": "treatment",
        "model_class": "TreeWorldModelV2",
        "scorer": "goal_distance",
        "setting": "setting-a",
        "env_name": "PushCube-v1",
        "dataset_kind": "future_recipe",
        "source_name": "PushCube-v1",
        "seed": 110,
        "wandb_id": "cell-id",
        "wandb_group": "exp23",
        "scheduler_total_steps": 1_000_000,
        "task_ids": [1, 2, 3, 4, 5],
        "final_episodes_per_task": 5,
        "node_budget": 64,
        "branch_factor": 4,
        "gradient_checkpointing": True,
        "future_set_cache": True,
        "shared_cache": True,
        "retrieval_enabled": False,
        "retrieval_num_keys": 0,
    }
    identity_sha256 = worker.stable_hash(identity)
    progress_value = _progress(_episodes(), "complete")
    progress_value["identity_sha256"] = identity_sha256
    progress_value["seed_table_sha256"] = identity["final_seed_table_sha256"]
    progress_path = run / "final_eval_progress.json"
    _write_json(progress_path, progress_value)
    progress = worker.validate_final_progress(
        progress_path,
        expected_identity_sha256=identity_sha256,
        expected_seed_table_sha256=identity["final_seed_table_sha256"],
        manifest=_manifest(),
    )
    assert progress is not None
    seed_tables = {
        "sha256": identity["evaluation_seed_tables_sha256"],
        "final": {"sha256": identity["final_seed_table_sha256"]},
    }
    _write_json(run / "evaluation_seed_tables.json", seed_tables)
    monkeypatch.setattr(worker, "expected_evaluation_seed_tables", lambda _context: seed_tables)
    config = {"env": {"dataset_dir": "/sealed/dataset"}}
    context = {
        **_run_layout(run),
        "manifest": _manifest(),
        "source_contract": {"runtime": {"software": {"python": "3.11.15"}}},
    }
    state = {
        "kind": "final_checkpoint_complete",
        "identity": identity,
        "identity_sha256": identity_sha256,
        "config": config,
        "progress": progress,
        "final_eval": progress["metrics"],
        "checkpoint_sha256": "9" * 64,
    }
    completion = {
        "schema_version": 1,
        "objective_version": worker.OBJECTIVE,
        "status": "complete",
        "run_identity": identity,
        "identity_sha256": identity_sha256,
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
        "dataset_dir": config["env"]["dataset_dir"],
        "seed": identity["seed"],
        "wandb_id": identity["wandb_id"],
        "wandb_group": identity["wandb_group"],
        "completed_updates": 25_000,
        "scheduler_total_steps": identity["scheduler_total_steps"],
        "final_eval_step": 25_000,
        "task_ids": identity["task_ids"],
        "episodes_per_task": identity["final_episodes_per_task"],
        "node_budget": identity["node_budget"],
        "branch_factor": identity["branch_factor"],
        "gradient_checkpointing": identity["gradient_checkpointing"],
        "future_set_cache": identity["future_set_cache"],
        "shared_cache": identity["shared_cache"],
        "retrieval_enabled": identity["retrieval_enabled"],
        "retrieval_num_keys": identity["retrieval_num_keys"],
        "final_evaluation": progress["metrics"],
        "checkpoint": "checkpoints/latest.pt",
        "final_eval_progress": "final_eval_progress.json",
    }
    if mutation == "extra":
        completion["unexpected"] = 1
    elif mutation == "stripped":
        completion.pop("recipe_runtime_sha256")
    elif mutation == "drift":
        completion["dataset_dir"] = "/drifted/dataset"
    _write_json(run / "COMPLETED.json", completion)
    return context, state


def test_completion_requires_exact_full_trainer_schema_and_checkpoint_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, state = _terminal_case(tmp_path / "valid", monkeypatch)
    result = worker.validate_complete_run(tmp_path / "valid", state, context)
    assert result["status"] == "complete" and result["completed_updates"] == 25_000

    for mutation, pattern in (
        ("extra", "fields differ"),
        ("stripped", "fields differ"),
        ("drift", "dataset_dir differs"),
    ):
        case_context, case_state = _terminal_case(tmp_path / mutation, monkeypatch, mutation)
        with pytest.raises(worker.LifecycleError, match=pattern):
            worker.validate_complete_run(tmp_path / mutation, case_state, case_context)


@pytest.mark.parametrize("name", ["checkpoints/latest.pt", "COMPLETED.json", "final_eval_progress.json"])
def test_inspect_run_never_treats_broken_artifact_symlink_as_absent(
    tmp_path: Path, name: str
) -> None:
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    path = run_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(tmp_path / "does-not-exist")
    context = {
        "launch": {"cell": {"run_directory": str(run_dir)}},
        **_run_layout(run_dir),
    }
    with pytest.raises(worker.LifecycleError):
        worker.inspect_run(context)


def _lineage_args(task_index: int = 3) -> argparse.Namespace:
    return argparse.Namespace(
        submission_sha256="c" * 64,
        cell_index=task_index,
        restart_count=1,
        array_job_id="12345",
        array_task_id=task_index,
    )


def test_arbitrary_checkpoint_lineage_accepts_1234_and_rejects_advanced_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _lineage_args()
    task = tmp_path / "task"
    previous = worker.generation_root_for(task, 0)
    previous.mkdir(parents=True)
    run = tmp_path / "run"
    latest = run / "checkpoints/latest.pt"
    latest.parent.mkdir(parents=True)
    latest.write_bytes(b"checkpoint-at-1234")
    checkpoint_sha = worker.file_sha256(latest)
    checkpoint_identity = worker._stat_identity(latest.stat())
    context = {
        "launch": {
            "launch_sha256": "d" * 64,
            "cell": {"run_directory": str(run)},
        },
        "manifest": {"execution": {"scontrol": "/usr/local/bin/scontrol"}},
        "submission_contract": {
            "scheduler_preclaim": _scheduler_preclaim(),
            "scheduler_fallback_config": _scheduler_fallback(),
        },
        **_run_layout(run),
    }
    current_state = {
        "kind": "train",
        "completed_updates": 1234,
        "phase": "train",
        "pending_eval_step": None,
        "progress": None,
    }

    def reopened(_context):
        return {
            **current_state,
            "checkpoint_sha256": worker.file_sha256(latest),
            "checkpoint_file_identity": worker._stat_identity(latest.stat()),
        }

    monkeypatch.setattr(
        worker,
        "resolve_checkpoint",
        reopened,
    )
    common = {
        "schema_version": 1,
        "campaign_id": worker.CAMPAIGN_ID,
        "submission_sha256": args.submission_sha256,
        "launch_sha256": "d" * 64,
        "cell_index": args.cell_index,
        "restart_count": 0,
        "array_job_id": args.array_job_id,
        "array_task_id": args.array_task_id,
    }
    ready = {
        **common,
        "status": "requeue_ready",
        "trainer_exit_code": 75,
        "checkpoint_kind": "train",
        "completed_updates": 1234,
        "phase": "train",
        "pending_eval_step": None,
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_file_identity": checkpoint_identity,
        "final_eval_progress_sha256": None,
    }
    ready_path = previous / worker.REQUEUE_READY_NAME
    _write_json(ready_path, ready)
    _write_json(
        previous / worker.REQUEUE_CALLING_NAME,
        {
            **common,
                "status": "scontrol_requeue_calling",
                "requeue_target": "12345_3",
                "requeue_ready_sha256": worker.file_sha256(ready_path),
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_file_identity": checkpoint_identity,
                "scheduler_control_plane": _scheduler_observation(),
                "scheduler_config_sha256": _scheduler_fallback()["sha256"],
                "scheduler_config_size": _scheduler_fallback()["size"],
                "scontrol_executable_sha256": "e" * 64,
                "scontrol_show_command": [
                    "/usr/local/bin/scontrol", "show", "job", "12345_3", "--oneliner",
                ],
                "scontrol_show_stdout_sha256": "f" * 64,
                "scontrol_requeue_command": [
                    "/usr/local/bin/scontrol", "requeue", "12345_3",
                ],
            },
        )
    worker._verify_previous_lineage(args, task, context)
    current_state["completed_updates"] = 1235
    with pytest.raises(worker.LifecycleError, match="completed_updates differs"):
        worker._verify_previous_lineage(args, task, context)
    current_state["completed_updates"] = 1234
    latest.write_bytes(b"advanced-after-ready")
    with pytest.raises(worker.LifecycleError, match="checkpoint_sha256|advanced/swapped"):
        worker._verify_previous_lineage(args, task, context)


def test_ready_binding_rejects_mutated_final_progress(tmp_path: Path) -> None:
    progress_path = tmp_path / "final_eval_progress.json"
    _write_json(progress_path, {"status": "in_progress", "rows": [1]})
    original_progress_sha = worker.file_sha256(progress_path)
    checkpoint_identity = {
        "device": 1, "inode": 2, "size": 3, "mtime_ns": 4, "ctime_ns": 5,
    }
    ready = {
        "trainer_exit_code": 75,
        "checkpoint_kind": "final_pending",
        "completed_updates": 25_000,
        "phase": "final_eval",
        "pending_eval_step": 25_000,
        "checkpoint_sha256": "a" * 64,
        "checkpoint_file_identity": checkpoint_identity,
        "final_eval_progress_sha256": original_progress_sha,
    }
    progress_path.chmod(0o600)
    progress_path.write_text('{"rows":[1,2],"status":"in_progress"}\n', encoding="utf-8")
    mutated_progress_sha = worker.file_sha256(progress_path)
    reopened = {
        "kind": "final_pending",
        "completed_updates": 25_000,
        "phase": "final_eval",
        "pending_eval_step": 25_000,
        "checkpoint_sha256": "a" * 64,
        "checkpoint_file_identity": checkpoint_identity,
        "progress": {"sha256": mutated_progress_sha},
    }
    with pytest.raises(worker.LifecycleError, match="final_eval_progress_sha256 differs"):
        worker.validate_ready_checkpoint_binding(ready, reopened, "adversarial READY")


def test_checkpoint_resolver_hashes_and_loads_one_open_inode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "run"
    latest = run / "checkpoints/latest.pt"
    latest.parent.mkdir(parents=True)
    original = b"original-checkpoint-bytes"
    latest.write_bytes(original)
    original_identity = worker._stat_identity(latest.stat())

    def fake_load(handle, **_kwargs):
        assert handle.read() == original
        return {"loaded": "original"}

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=fake_load))
    monkeypatch.setattr(
        worker,
        "validate_checkpoint_payload",
        lambda payload, _context: {"kind": "train", "payload": payload},
    )
    context = _run_layout(run)
    state = worker.resolve_checkpoint(context)
    assert state["payload"] == {"loaded": "original"}
    assert state["checkpoint_sha256"] == hashlib.sha256(original).hexdigest()
    assert state["checkpoint_file_identity"] == original_identity

    def swapping_load(handle, **_kwargs):
        assert handle.read() == original
        with latest.open("ab") as writer:
            writer.write(b"-mutated")
            writer.flush()
            os.fsync(writer.fileno())
        return {"loaded": "original"}

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(load=swapping_load))
    with pytest.raises(worker.LifecycleError, match="changed while open"):
        worker.resolve_checkpoint(context)


def test_component_walk_and_sealed_json_reject_mutation_and_symlink_parent(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "state" / "marker.json"
    value = {"schema_version": 1, "status": "sealed"}
    worker.seal_json(artifact, value)
    before = artifact.stat()
    assert stat.S_IMODE(before.st_mode) == 0o444
    worker.seal_json(artifact, value)
    after = artifact.stat()
    assert (before.st_dev, before.st_ino, before.st_mtime_ns) == (
        after.st_dev,
        after.st_ino,
        after.st_mtime_ns,
    )
    with pytest.raises(worker.LifecycleError, match="differs"):
        worker.seal_json(artifact, {**value, "status": "mutated"})
    assert json.loads(artifact.read_text(encoding="utf-8")) == value

    root = tmp_path / "root"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "linked").symlink_to(outside, target_is_directory=True)
    with pytest.raises(worker.LifecycleError, match="symlink/non-directory"):
        worker.ensure_contained_directory(
            root / "linked" / "child",
            root,
            "hostile state path",
            create=True,
        )
    assert not (outside / "child").exists()


def test_fresh_claim_is_root_bound_and_rejects_symlinked_component(tmp_path: Path) -> None:
    run_root = tmp_path / "declared"
    run = run_root / "nested" / "cell"
    context = {
        "run_root": run_root,
        "run_relative": Path("nested/cell"),
        "run_directory": run,
    }
    worker._claim_fresh_run(context)
    assert run.is_dir()
    with pytest.raises(worker.LifecycleError, match="already exists"):
        worker._claim_fresh_run(context)

    hostile_root = tmp_path / "hostile-root"
    outside = tmp_path / "hostile-outside"
    hostile_root.mkdir()
    outside.mkdir()
    (hostile_root / "nested").symlink_to(outside, target_is_directory=True)
    hostile_context = {
        "run_root": hostile_root,
        "run_relative": Path("nested/cell"),
        "run_directory": hostile_root / "nested" / "cell",
    }
    with pytest.raises(worker.LifecycleError, match="symlink/non-directory"):
        worker._claim_fresh_run(hostile_context)
    assert not (outside / "cell").exists()


def _interpreter_contract() -> dict:
    lexical = worker.PINNED_PYTHON
    resolved = lexical.resolve(strict=True)
    info = resolved.stat()
    return {
        "lexical_executable": str(lexical),
        "lexical_symlink_target": os.readlink(lexical),
        "resolved_executable": str(resolved),
        "resolved_executable_sha256": worker.file_sha256(resolved),
        "resolved_executable_size": info.st_size,
        "base_executable": str(getattr(sys, "_base_executable", "")),
        "venv_site_packages": str(worker.PINNED_SITE_DIRECTORIES[0]),
        "base_site_packages": str(worker.PINNED_SITE_DIRECTORIES[1]),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }


def _seal_tree(root: Path) -> dict[str, str]:
    inventory = {
        str(path.relative_to(root)): worker.file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        path.chmod(0o444 if path.is_file() else 0o555)
    root.chmod(0o555)
    return inventory


def _smoke_evidence(
    *,
    inventory_sha256: str,
    launch_sha256: str,
    resolved_config_sha256: str,
    full_output_fingerprint: str,
    scientific_output_fingerprint: str,
) -> dict:
    return {
        "schema_version": 1,
        "status": "sealed_trainer_hydra_composition_verified",
        "cell_index": 0,
        "python_flags": ["-P", "-S", "-B"],
        "entry_relative_path": str(worker.PACKAGE_RELATIVE / "train_entry.py"),
        "config_package_relative_path": "configs/__init__.py",
        "config_package_sha256": hashlib.sha256(b"").hexdigest(),
        "snapshot_inventory_sha256": inventory_sha256,
        "launch_sha256": launch_sha256,
        "resolved_config_sha256": resolved_config_sha256,
        "stdout_sha256": "7" * 64,
        "stdout_bytes": 1,
        "cuda_visible_devices": "",
        "full_output_fingerprint_before": full_output_fingerprint,
        "full_output_fingerprint_after": full_output_fingerprint,
        "scientific_output_fingerprint_before": scientific_output_fingerprint,
        "scientific_output_fingerprint_after": scientific_output_fingerprint,
        "persistent_writes_performed": 0,
        "scheduler_calls": 0,
    }


def _submission(root: Path) -> tuple[Path, str]:
    root.chmod(0o700)
    snapshot = root / "source-snapshot" / "repo"
    package = snapshot / worker.PACKAGE_RELATIVE
    scripts = snapshot / "scripts"
    configs = snapshot / "configs"
    package.mkdir(parents=True)
    scripts.mkdir(parents=True)
    configs.mkdir(parents=True)
    for relative, payload in {
        package / "worker.py": (PACKAGE / "worker.py").read_bytes(),
        package / "train_entry.py": (PACKAGE / "train_entry.py").read_bytes(),
        package / "train.slurm": (PACKAGE / "train.slurm").read_bytes(),
        configs / "__init__.py": (REPO / "configs/__init__.py").read_bytes(),
        scripts / "__init__.py": (REPO / "scripts/__init__.py").read_bytes(),
        scripts / "train.py": (REPO / "scripts/train.py").read_bytes(),
    }.items():
        relative.write_bytes(payload)
    inventory = _seal_tree(snapshot)
    inventory_sha256 = worker.stable_hash(inventory)
    snapshot.parent.chmod(0o555)

    launches = []
    audits = {
        "weight_audit_artifact_sha256": "1" * 64,
        "prefix_target_artifact_sha256": "2" * 64,
        "resolved_config_artifact_sha256": "3" * 64,
        "causal_parity_artifact_sha256": "4" * 64,
    }
    for index in range(20):
        body = {"schema_version": 1, "cell": {"index": index}}
        launch = {**body, "launch_sha256": worker.stable_hash(body)}
        path = root / "launches" / f"cell-{index:02d}.json"
        _write_json(path, launch)
        launches.append(
            {
                "index": index,
                "path": f"launches/cell-{index:02d}.json",
                "launch_sha256": launch["launch_sha256"],
                "launch_file_sha256": worker.file_sha256(path),
                "setting_id": f"setting-{index // 4}",
                "arm_id": "baseline" if index % 2 == 0 else "treatment",
                "seed": index,
                **audits,
            }
        )
    compositions = [
        {
            "index": index,
            "resolved_config_sha256": hashlib.sha256(
                f"config-{index}".encode()
            ).hexdigest(),
            "launch_sha256": launches[index]["launch_sha256"],
            "stdout_sha256": hashlib.sha256(f"stdout-{index}".encode()).hexdigest(),
        }
        for index in range(20)
    ]
    snapshot_full = "5" * 64
    snapshot_scientific = "6" * 64
    contract = {
        "schema_version": 1,
        "status": "sealed_for_submission",
        "campaign_id": worker.CAMPAIGN_ID,
        "formal_validation": False,
        "submission_root": str(root),
        "snapshot_root": str(snapshot),
        "package_protocol_sha256": "e" * 64,
        "manifest_sha256": "f" * 64,
        "trainer_code_fingerprint": "a" * 64,
        "runtime_sha256": "b" * 64,
        "orchestration_interpreter": _interpreter_contract(),
        "scheduler_control_plane_contract": worker.SCHEDULER_CONTROL_PLANE,
        "scheduler_preclaim": _scheduler_preclaim(),
        "scheduler_fallback_config": _scheduler_fallback(),
        **audits,
        "snapshot_inventory": inventory,
        "snapshot_inventory_sha256": inventory_sha256,
        "live_audit_replays": {},
        "snapshot_audit_replays": {},
        "direct_hydra_compositions": compositions,
        "trainer_bootstrap_smoke": _smoke_evidence(
            inventory_sha256=inventory_sha256,
            launch_sha256=launches[0]["launch_sha256"],
            resolved_config_sha256=compositions[0]["resolved_config_sha256"],
            full_output_fingerprint=snapshot_full,
            scientific_output_fingerprint=snapshot_scientific,
        ),
        "scientific_output_fingerprint_before": {},
        "scientific_output_fingerprint_after": {},
        "full_output_fingerprint_before": {},
        "full_output_fingerprint_after": {},
        "snapshot_full_output_fingerprint_before": snapshot_full,
        "snapshot_full_output_fingerprint_after": snapshot_full,
        "snapshot_scientific_output_fingerprint_before": snapshot_scientific,
        "snapshot_scientific_output_fingerprint_after": snapshot_scientific,
        "git_provenance": {},
        "launches": launches,
        "array": "0-19%20",
        "fresh_start": True,
    }
    path = root / worker.SUBMISSION_CONTRACT_NAME
    _write_json(path, contract)
    return path, worker.file_sha256(path)


def test_submission_inventory_rejects_launch_symlink_even_with_same_bytes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "submission"
    root.mkdir()
    _, digest = _submission(root)
    original = root / "launches/cell-07.json"
    copy_path = root / "launches/copied-cell-07.json"
    copy_path.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(copy_path)
    with pytest.raises(worker.LifecycleError, match="symlink"):
        worker.validate_submission_contract(root, digest)


def test_bootstrap_rejects_snapshot_extra_and_swapped_site_binding(tmp_path: Path) -> None:
    root = tmp_path / "submission"
    root.mkdir()
    contract_path, digest = _submission(root)
    snapshot, submission, _ = worker.bootstrap_submission(root, digest)
    assert snapshot == root / "source-snapshot" / "repo" and submission == root

    snapshot.chmod(0o755)
    extra = snapshot / "unclaimed.py"
    extra.write_text("raise RuntimeError('must never import')\n", encoding="utf-8")
    extra.chmod(0o444)
    snapshot.chmod(0o555)
    with pytest.raises(worker.LifecycleError, match="unclaimed|extra"):
        worker.bootstrap_submission(root, digest)

    snapshot.chmod(0o755)
    extra.unlink()
    snapshot.chmod(0o555)
    contract_path.chmod(0o600)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["orchestration_interpreter"]["venv_site_packages"] = "/hostile/site-packages"
    contract_path.write_text(json.dumps(contract, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    contract_path.chmod(0o444)
    swapped_digest = worker.file_sha256(contract_path)
    with pytest.raises(worker.LifecycleError, match="site-package binding"):
        worker.bootstrap_submission(root, swapped_digest)


def test_both_bootstraps_reject_detached_trainer_smoke_evidence(tmp_path: Path) -> None:
    root = tmp_path / "submission"
    root.mkdir()
    contract_path, _digest = _submission(root)
    contract_path.chmod(0o600)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    contract["trainer_bootstrap_smoke"]["launch_sha256"] = "9" * 64
    contract_path.write_text(
        json.dumps(contract, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    contract_path.chmod(0o444)
    detached_digest = worker.file_sha256(contract_path)

    with pytest.raises(worker.LifecycleError, match="smoke launch/config binding"):
        worker.bootstrap_submission(root, detached_digest)

    entry = root / "source-snapshot/repo" / worker.PACKAGE_RELATIVE / "train_entry.py"
    code = (
        "import pathlib,runpy;"
        f"entry=runpy.run_path({str(entry)!r});"
        f"entry['bootstrap_submission'](pathlib.Path({str(root)!r}),{detached_digest!r})"
    )
    environment = {
        "PATH": "/usr/bin:/bin",
        "PYTHONHASHSEED": "110",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    completed = subprocess.run(
        [str(worker.PINNED_PYTHON), "-P", "-S", "-B", "-c", code],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode != 0
    assert "smoke launch/config binding" in completed.stderr


def test_preimport_revalidation_rejects_mutation_after_initial_snapshot_check(
    tmp_path: Path,
) -> None:
    root = tmp_path / "submission"
    root.mkdir()
    _, digest = _submission(root)
    snapshot, submission, contract = worker.bootstrap_submission(root, digest)
    source = snapshot / worker.PACKAGE_RELATIVE / "worker.py"
    original = source.read_bytes()
    source.chmod(0o644)
    source.write_bytes(original + b"\n# deterministic between-check drift\n")
    source.chmod(0o444)
    with pytest.raises(worker.LifecycleError, match="bytes differ"):
        worker._revalidate_snapshot_before_import(snapshot, submission, contract)


def test_isolated_bootstrap_ignores_hostile_pythonpath_customizers(tmp_path: Path) -> None:
    root = tmp_path / "submission"
    root.mkdir()
    _, digest = _submission(root)
    hostile = tmp_path / "hostile"
    hostile.mkdir()
    sentinel = tmp_path / "customizer-ran"
    payload = f"from pathlib import Path\nPath({str(sentinel)!r}).write_text('ran')\n"
    (hostile / "sitecustomize.py").write_text(payload, encoding="utf-8")
    (hostile / "usercustomize.py").write_text(payload, encoding="utf-8")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(hostile)
    completed = subprocess.run(
        [
            str(worker.PINNED_PYTHON), "-I", "-S", "-B",
            str(root / "source-snapshot" / "repo" / worker.PACKAGE_RELATIVE / "worker.py"),
            "record-signal",
            "--submission-root", str(root),
            "--submission-sha256", digest,
            "--cell-index", "0",
            "--restart-count", "0",
            "--array-job-id", "12345",
            "--array-task-id", "0",
            "--signal", "USR1",
        ],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    marker = root / "tasks/cell-00/requeue/0/USR1_REQUESTED.json"
    assert marker.is_file() and stat.S_IMODE(marker.stat().st_mode) == 0o444

    code = (
        "import pathlib,runpy;"
        f"entry=runpy.run_path({str(root / 'source-snapshot' / 'repo' / worker.PACKAGE_RELATIVE / 'train_entry.py')!r});"
        "entry['assert_isolated_runtime']();"
        f"entry['bootstrap_submission'](pathlib.Path({str(root)!r}),{digest!r})"
    )
    entry_environment = dict(environment)
    entry_environment.pop("PYTHONPATH", None)
    entry_environment["PYTHONHASHSEED"] = "110"
    entry_completed = subprocess.run(
        [str(worker.PINNED_PYTHON), "-P", "-S", "-B", "-c", code],
        env=entry_environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert entry_completed.returncode == 0, entry_completed.stderr
    assert not sentinel.exists()


def test_bootstrap_rejects_direct_base_python(tmp_path: Path) -> None:
    root = tmp_path / "submission"
    root.mkdir()
    _, digest = _submission(root)
    code = (
        "import pathlib,runpy,sys;"
        f"scope=runpy.run_path({str(PACKAGE / 'worker.py')!r});"
        "\ntry:\n"
        f" scope['bootstrap_submission'](pathlib.Path({str(root)!r}),{digest!r})\n"
        "except Exception as exc:\n print(exc); sys.exit(0)\n"
        "sys.exit(9)\n"
    )
    base_python = worker.PINNED_PYTHON.resolve(strict=True)
    completed = subprocess.run(
        [str(base_python), "-I", "-S", "-B", "-c", code],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "interpreter binding differs" in completed.stdout


def test_train_entry_rejects_original_launch_symlink(tmp_path: Path) -> None:
    submission = tmp_path / "submission"
    launches = submission / "launches"
    launches.mkdir(parents=True)
    args = ["resume=auto", "train.steps=25000"]
    manifest = {
        "paths": {"python": sys.executable},
        "weight_audit": {"artifact_sha256": "1" * 64},
        "prefix_target_contract": {"artifact_sha256": "2" * 64},
        "resolved_config_contract": {"artifact_sha256": "3" * 64},
        "causal_parity_contract": {"artifact_sha256": "4" * 64},
    }
    launch = {
        "cell": {"index": 0},
        "argv": [sys.executable, str(REPO / "scripts/train.py"), *args],
        "environment": {
            "TREEWM_RESOLVED_CONFIG_SHA256": "3" * 64,
            "TREEWM_CAUSAL_PARITY_SHA256": "4" * 64,
            "MUJOCO_GL": "egl",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        },
        "hashes": {
            "weight_audit_artifact_sha256": "1" * 64,
            "prefix_target_artifact_sha256": "2" * 64,
            "resolved_config_artifact_sha256": "3" * 64,
            "causal_parity_artifact_sha256": "4" * 64,
        },
    }

    class FakeCampaign:
        @staticmethod
        def load_contract(_root):
            return manifest, {}

        @staticmethod
        def verify_protocol_lock(_package):
            return "5" * 64

        @staticmethod
        def expand_matrix(_manifest):
            return [SimpleNamespace(index=0)]

        @staticmethod
        def trainer_command(*_args, **_kwargs):
            return launch

    real = launches / "cell-00-real.json"
    _write_json(real, launch)
    link = launches / "cell-00.json"
    link.symlink_to(real)
    with pytest.raises(train_entry.EntryContractError, match="symlink"):
        train_entry.verify_exact_invocation(
            link,
            args,
            submission_root=submission,
            campaign=FakeCampaign(),
            environ=launch["environment"],
        )


def test_controlled_environment_rejects_stop_and_drops_ambient_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(worker.STOP_ENVIRONMENT, raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setenv("AMBIENT_SECRET_SENTINEL", "must-not-propagate")
    context = {
        "launch": {
            "argv": [sys.executable],
            "environment": {
                "TREEWM_PROTOCOL_SHA256": "a" * 64,
                "MUJOCO_GL": "egl",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            },
        },
        "snapshot_root": tmp_path,
        "cell": SimpleNamespace(seed=110),
    }
    environment = worker.controlled_child_environment(context)
    assert "AMBIENT_SECRET_SENTINEL" not in environment
    assert "PYTHONPATH" not in environment
    assert environment["PYTHONHASHSEED"] == "110"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["MUJOCO_GL"] == "egl"
    assert environment["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    context["launch"]["environment"]["MUJOCO_GL"] = "osmesa"
    with pytest.raises(worker.LifecycleError, match="MUJOCO_GL"):
        worker.controlled_child_environment(context)
    context["launch"]["environment"]["MUJOCO_GL"] = "egl"
    monkeypatch.setenv(worker.STOP_ENVIRONMENT, "5000")
    with pytest.raises(worker.LifecycleError, match="forbidden"):
        worker.controlled_child_environment(context)


def test_trainer_safe_path_preserves_deterministic_python_hash_seed() -> None:
    def observed(flags: list[str], seed: str) -> str:
        environment = dict(os.environ)
        environment.pop("PYTHONPATH", None)
        environment["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [str(worker.PINNED_PYTHON), *flags, "-c", "print(hash('exp23-hash-seed-adversary'))"],
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    seeded = [observed(["-P", "-S", "-B"], "110") for _ in range(3)]
    assert len(set(seeded)) == 1
    assert observed(["-P", "-S", "-B"], "111") != seeded[0]
    ignored = [observed(["-I", "-S", "-B"], "110") for _ in range(4)]
    assert len(set(ignored)) > 1


def test_invalid_context_creates_no_task_or_run_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submission = tmp_path / "submission"
    submission.mkdir()
    run = tmp_path / "run"
    args = argparse.Namespace(
        snapshot_root=REPO,
        submission_root=submission,
        submission_sha256="a" * 64,
        cell_index=0,
        restart_count=0,
        array_job_id="123",
        array_task_id=0,
        _bootstrap_contract={},
    )
    monkeypatch.delenv(worker.STOP_ENVIRONMENT, raising=False)
    monkeypatch.setattr(
        worker,
        "load_launch_context",
        lambda **_kwargs: (_ for _ in ()).throw(worker.LifecycleError("invalid contract")),
    )
    old_usr1 = signal.getsignal(signal.SIGUSR1)
    old_term = signal.getsignal(signal.SIGTERM)
    try:
        with pytest.raises(worker.LifecycleError, match="invalid contract"):
            worker.run_worker(args)
    finally:
        signal.signal(signal.SIGUSR1, old_usr1)
        signal.signal(signal.SIGTERM, old_term)
    assert not (submission / "tasks").exists()
    assert not run.exists()


def test_complete_receipt_status_cannot_be_overwritten() -> None:
    args = argparse.Namespace(
        submission_sha256="a" * 64,
        cell_index=0,
        restart_count=2,
        array_job_id="123",
        array_task_id=0,
    )
    context = {"launch": {"launch_sha256": "b" * 64}}
    marker = worker._complete_marker(
        args,
        context,
        {"status": "complete", "completed_updates": 25_000},
    )
    assert marker["status"] == "worker_complete"


def test_live_submit_contract_schema_matches_both_lifecycle_bootstraps(tmp_path: Path) -> None:
    submit = _load("exp23_lifecycle_live_submit", "submit.py")
    audits = {
        "weight_audit": {"artifact_sha256": "1" * 64},
        "prefix_target_contract": {"artifact_sha256": "2" * 64},
        "resolved_config_contract": {"artifact_sha256": "3" * 64},
        "causal_parity_contract": {"artifact_sha256": "4" * 64},
    }
    manifest = {
        "campaign_id": worker.CAMPAIGN_ID,
        "execution": {"scheduler_control_plane": worker.SCHEDULER_CONTROL_PLANE},
        **audits,
    }
    launches = [
        {
            "cell": {
                "index": index,
                "setting": f"setting-{index // 4}",
                "arm": "baseline" if index % 2 == 0 else "treatment",
                "seed": index,
            },
            "launch_sha256": hashlib.sha256(f"launch-{index}".encode()).hexdigest(),
        }
        for index in range(20)
    ]
    preflight = {
        "launches": launches,
        "orchestration_interpreter": _interpreter_contract(),
        "audit_replays": {},
        "scientific_output_fingerprint_before": {},
        "scientific_output_fingerprint_after": {},
        "full_output_fingerprint_before": {},
        "full_output_fingerprint_after": {},
    }
    snapshot_full = "5" * 64
    snapshot_scientific = "6" * 64
    compositions = [
        {
            "index": index,
            "resolved_config_sha256": hashlib.sha256(
                f"config-{index}".encode()
            ).hexdigest(),
            "launch_sha256": launches[index]["launch_sha256"],
            "stdout_sha256": hashlib.sha256(f"stdout-{index}".encode()).hexdigest(),
        }
        for index in range(20)
    ]
    inventory = {"scripts/train.py": "c" * 64}
    inventory_sha256 = submit.stable_hash(inventory)
    snapshot_preflight = {
        "audit_replays": {},
        "scientific_output_fingerprint_before": snapshot_scientific,
        "scientific_output_fingerprint_after": snapshot_scientific,
        "full_output_fingerprint_before": snapshot_full,
        "full_output_fingerprint_after": snapshot_full,
        "trainer_bootstrap_smoke": _smoke_evidence(
            inventory_sha256=inventory_sha256,
            launch_sha256=launches[0]["launch_sha256"],
            resolved_config_sha256=compositions[0]["resolved_config_sha256"],
            full_output_fingerprint=snapshot_full,
            scientific_output_fingerprint=snapshot_scientific,
        ),
    }
    scheduler_preclaim = _scheduler_preclaim()
    contract = submit._submission_contract(
        manifest=manifest,
        protocol="e" * 64,
        source={"source_sha256": "a" * 64, "runtime_sha256": "b" * 64},
        snapshot_root=tmp_path / "snapshot",
        submission_root=tmp_path / "submission",
        inventory=inventory,
        launch_rows=[{"launch_file_sha256": "d" * 64} for _ in range(20)],
        preflight=preflight,
        compositions=compositions,
        snapshot_preflight=snapshot_preflight,
        git={},
        scheduler_preclaim=scheduler_preclaim,
        scheduler_fallback=_scheduler_fallback(),
    )
    assert set(contract) == worker.SUBMISSION_CONTRACT_FIELDS
    assert worker.SUBMISSION_CONTRACT_FIELDS == train_entry.SUBMISSION_CONTRACT_FIELDS
    assert all(set(row) == worker.SUBMISSION_LAUNCH_FIELDS for row in contract["launches"])


def test_worker_terminal_schemas_match_live_trainer_literals() -> None:
    syntax = ast.parse((REPO / "scripts/train.py").read_text(encoding="utf-8"))
    literal_keysets: list[set[str]] = []
    for node in ast.walk(syntax):
        if not isinstance(node, ast.Dict):
            continue
        keys = {
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        literal_keysets.append(keys)
    completion = [
        keys
        for keys in literal_keysets
        if {"run_identity", "final_evaluation", "checkpoint", "final_eval_progress"} <= keys
    ]
    assert completion == [set(worker.COMPLETION_FIELDS)]
    progress_complete = [
        keys
        for keys in literal_keysets
        if {"completed_results", "generator_state", "seed_table_sha256", "metrics"} <= keys
        and "run_identity" not in keys
    ]
    expected_complete_progress = {
        "schema_version", "objective_version", "status", "identity_sha256",
        "seed_table_sha256", "task_ids", "episodes_per_task", "completed_results",
        "generator_state", "metrics",
    }
    assert expected_complete_progress in progress_complete


def test_batch_script_is_direct_no_stage_and_orders_calling_before_requeue() -> None:
    source = (PACKAGE / "train.slurm").read_text(encoding="utf-8")
    assert "#SBATCH --export=NONE" in source
    assert "#SBATCH --array=0-19%20" in source
    assert "TREEWM_STOP_AFTER_UPDATE=" not in source
    assert "resume=" not in source  # the exact sealed launch owns resume=auto
    assert "s" + "run" not in source
    assert 'exec "$PYTHON_EXECUTABLE" -I -S -B "$WORKER" run' in source
    assert source.count('"$PYTHON_EXECUTABLE" -I -S -B "$WORKER"') == 3
    assert '"$PYTHON_EXECUTABLE" -I -S -B "$WORKER" requeue' in source
    assert "scontrol" not in source.lower()
    assert "SLURM_CONF" not in source
    assert "SCHEDULER_ENV" not in source
    assert "if [[ -f \"$REQUEUE_CALLING\" && ! -L \"$REQUEUE_CALLING\" ]]" in source
    worker_source = (PACKAGE / "worker.py").read_text(encoding="utf-8")
    authenticated = worker_source[
        worker_source.index("def authenticated_requeue"):worker_source.index("def _common_parser")
    ]
    show = authenticated.index("shown = scheduler_runner")
    calling = authenticated.index("seal_json(generation / REQUEUE_CALLING_NAME")
    requeue = authenticated.index("requeued = scheduler_runner")
    assert show < calling < requeue
    trainer_spawn = worker_source[
        worker_source.index("def _spawn_trainer"):worker_source.index("def _claim_fresh_run")
    ]
    assert '"-P",\n        "-S",\n        "-B"' in trainer_spawn
    assert '"-I"' not in trainer_spawn


def test_authenticated_requeue_pins_preclaim_config_across_show_and_requeue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submission = tmp_path / "submission"
    task = worker.task_root_for(submission, 2)
    generation = worker.generation_root_for(task, 1)
    generation.mkdir(parents=True)
    args = argparse.Namespace(
        submission_root=submission,
        submission_sha256="c" * 64,
        cell_index=2,
        restart_count=1,
        array_job_id="7000",
        array_task_id=2,
        _bootstrap_contract={},
        _bootstrap_snapshot_root=tmp_path / "snapshot",
    )
    args._bootstrap_snapshot_root.mkdir()
    context = {
        "launch": {"launch_sha256": "d" * 64},
        "manifest": {
            "execution": {
                "scontrol": "/usr/local/bin/scontrol",
                "scheduler_control_plane": worker.SCHEDULER_CONTROL_PLANE,
            }
        },
        "submission_contract": {
            "scheduler_preclaim": _scheduler_preclaim(),
            "scheduler_fallback_config": _scheduler_fallback(),
        },
    }
    common = {
        "schema_version": 1,
        "campaign_id": worker.CAMPAIGN_ID,
        "submission_sha256": args.submission_sha256,
        "launch_sha256": "d" * 64,
        "cell_index": 2,
        "restart_count": 1,
        "array_job_id": "7000",
        "array_task_id": 2,
    }
    checkpoint_identity = {
        "device": 1,
        "inode": 2,
        "size": 3,
        "mtime_ns": 4,
        "ctime_ns": 5,
    }
    checkpoint = {
        "kind": "train",
        "completed_updates": 1000,
        "phase": "train",
        "pending_eval_step": None,
        "checkpoint_sha256": "f" * 64,
        "checkpoint_file_identity": checkpoint_identity,
        "progress": None,
    }
    _write_json(
        generation / worker.REQUEUE_READY_NAME,
        {
            **common,
            "status": "requeue_ready",
            "trainer_exit_code": 75,
            "checkpoint_kind": "train",
            "completed_updates": 1000,
            "phase": "train",
            "pending_eval_step": None,
            "checkpoint_sha256": "f" * 64,
            "checkpoint_file_identity": checkpoint_identity,
            "final_eval_progress_sha256": None,
        },
    )
    monkeypatch.setattr(worker, "load_launch_context", lambda **_kwargs: context)
    monkeypatch.setattr(worker, "resolve_checkpoint", lambda _context: checkpoint)
    monkeypatch.setattr(
        worker,
        "scheduler_control_plane_observation",
        lambda _context: pytest.fail("requeue reopened mutable canonical scheduler state"),
    )
    calls: list[tuple[list[str], dict[str, str], tuple[int, ...]]] = []
    config_descriptors: list[int] = []
    executable_descriptors: list[int] = []
    canonical_drifted = False

    def runner(command, _cwd, environment, inherited_fds):
        nonlocal canonical_drifted
        values = list(command)
        descriptors = tuple(inherited_fds)
        calls.append((values, dict(environment), descriptors))
        assert set(environment) == {"PATH", "LANG", "LC_ALL", "SLURM_CONF"}
        assert environment["SLURM_CONF"].startswith("/proc/self/fd/")
        assert len(descriptors) == 2
        config_fd, executable_fd = descriptors
        config_descriptors.append(config_fd)
        executable_descriptors.append(executable_fd)
        assert environment["SLURM_CONF"] == f"/proc/self/fd/{config_fd}"
        assert values[0] == f"/proc/self/fd/{executable_fd}"
        assert os.pread(config_fd, len(_scheduler_config_bytes()), 0) == _scheduler_config_bytes()
        if values[1:3] == ["show", "job"]:
            canonical_drifted = True
            return subprocess.CompletedProcess(
                values,
                0,
                stdout=(
                    "JobId=7000_2 ArrayJobId=7000 ArrayTaskId=2 "
                    "JobState=RUNNING\n"
                ),
                stderr="",
            )
        assert canonical_drifted
        assert values[1:] == ["requeue", "7000_2"]
        return subprocess.CompletedProcess(values, 0, stdout="", stderr="")

    assert worker.authenticated_requeue(args, runner) == 0
    assert len(calls) == 2
    assert config_descriptors[0] == config_descriptors[1]
    assert executable_descriptors[0] == executable_descriptors[1]
    calling = json.loads(
        (generation / worker.REQUEUE_CALLING_NAME).read_text(encoding="utf-8")
    )
    assert set(calling) == worker.REQUEUE_CALLING_FIELDS
    assert calling["scheduler_config_sha256"] == _scheduler_fallback()["sha256"]
    assert calling["scheduler_config_size"] == _scheduler_fallback()["size"]
    assert calling["scontrol_show_command"] == [
        "/usr/local/bin/scontrol", "show", "job", "7000_2", "--oneliner",
    ]
    assert calling["scontrol_requeue_command"] == [
        "/usr/local/bin/scontrol", "requeue", "7000_2",
    ]
    for descriptor in {config_descriptors[0], executable_descriptors[0]}:
        with pytest.raises(OSError):
            os.fstat(descriptor)


def test_authenticated_requeue_rejects_fallback_not_bound_to_preclaim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    submission = tmp_path / "submission"
    task = worker.task_root_for(submission, 2)
    generation = worker.generation_root_for(task, 1)
    generation.mkdir(parents=True)
    args = argparse.Namespace(
        submission_root=submission,
        submission_sha256="c" * 64,
        cell_index=2,
        restart_count=1,
        array_job_id="7000",
        array_task_id=2,
        _bootstrap_contract={},
        _bootstrap_snapshot_root=tmp_path / "snapshot",
    )
    args._bootstrap_snapshot_root.mkdir()
    preclaim = _scheduler_preclaim()
    preclaim["scheduler_control_plane"]["cli_filter_policy"]["tree_sha256"] = "9" * 64
    context = {
        "launch": {"launch_sha256": "d" * 64},
        "manifest": {"execution": {"scontrol": "/usr/local/bin/scontrol"}},
        "submission_contract": {
            "scheduler_preclaim": preclaim,
            "scheduler_fallback_config": _scheduler_fallback(),
        },
    }
    checkpoint_identity = {
        "device": 1, "inode": 2, "size": 3, "mtime_ns": 4, "ctime_ns": 5,
    }
    checkpoint = {
        "kind": "train",
        "completed_updates": 1000,
        "phase": "train",
        "pending_eval_step": None,
        "checkpoint_sha256": "f" * 64,
        "checkpoint_file_identity": checkpoint_identity,
        "progress": None,
    }
    _write_json(
        generation / worker.REQUEUE_READY_NAME,
        {
            "schema_version": 1,
            "campaign_id": worker.CAMPAIGN_ID,
            "submission_sha256": args.submission_sha256,
            "launch_sha256": "d" * 64,
            "cell_index": 2,
            "restart_count": 1,
            "array_job_id": "7000",
            "array_task_id": 2,
            "status": "requeue_ready",
            "trainer_exit_code": 75,
            "checkpoint_kind": "train",
            "completed_updates": 1000,
            "phase": "train",
            "pending_eval_step": None,
            "checkpoint_sha256": "f" * 64,
            "checkpoint_file_identity": checkpoint_identity,
            "final_eval_progress_sha256": None,
        },
    )
    monkeypatch.setattr(worker, "load_launch_context", lambda **_kwargs: context)
    monkeypatch.setattr(worker, "resolve_checkpoint", lambda _context: checkpoint)
    with pytest.raises(worker.LifecycleError, match="differs from the exact preclaim"):
        worker.authenticated_requeue(
            args, lambda *_args: pytest.fail("unbound scheduler config reached client")
        )
    assert not (generation / worker.REQUEUE_CALLING_NAME).exists()


@pytest.mark.parametrize("field", ["JobId", "ArrayJobId", "ArrayTaskId", "JobState"])
def test_scontrol_show_rejects_duplicate_identity_or_state_field(field: str) -> None:
    line = (
        "JobId=7000_2 ArrayJobId=7000 ArrayTaskId=2 JobState=RUNNING "
        f"{field}=forged\n"
    )
    with pytest.raises(worker.LifecycleError, match=field):
        worker._scontrol_field(line, field)


def test_train_entry_forbidden_environment_is_fail_closed() -> None:
    with pytest.raises(train_entry.EntryContractError, match="STOP_AFTER_UPDATE"):
        train_entry.reject_forbidden_environment({}, {train_entry.STOP_ENVIRONMENT: "5000"})
    with pytest.raises(train_entry.EntryContractError, match="unexpected TREEWM"):
        train_entry.reject_forbidden_environment({}, {"TREEWM_UNSEALED": "1"})
    with pytest.raises(train_entry.EntryContractError, match="distributed"):
        train_entry.reject_forbidden_environment({}, {"WORLD_SIZE": "1"})


def test_train_entry_headless_runtime_is_exactly_sealed() -> None:
    exact = {
        "MUJOCO_GL": "egl",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    train_entry.validate_headless_runtime_environment(exact, exact)
    with pytest.raises(train_entry.EntryContractError, match="sealed.*MUJOCO_GL"):
        train_entry.validate_headless_runtime_environment({}, exact)
    with pytest.raises(train_entry.EntryContractError, match="trainer.*PREALLOCATE"):
        train_entry.validate_headless_runtime_environment(
            exact,
            {**exact, "XLA_PYTHON_CLIENT_PREALLOCATE": "true"},
        )
