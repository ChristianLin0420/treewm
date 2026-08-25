from __future__ import annotations

import io
import http.client
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import aggregate  # noqa: E402
import campaign  # noqa: E402
import dispatcher  # noqa: E402
import gpu_preflight  # noqa: E402
import prepare_data  # noqa: E402
import submit  # noqa: E402


@pytest.fixture(scope="module")
def manifest():
    return campaign.load_manifest(CAMPAIGN_DIR / "manifest.json")


def completion_artifacts(manifest, run, run_dir, success=0.5):
    contract = campaign.current_trainer_contract()
    if run.dataset_kind == "standard":
        standard_dir = str((run_dir / "identity-data" / "standard").resolve())
        dataset_paths = []
    else:
        standard_dir = None
        setting = next(item for item in manifest["settings"] if item["id"] == run.setting_id)
        stem = setting["dataset"]["file_stem"]
        shard_dir = (run_dir / "identity-data" / run.dataset_directory).resolve()
        dataset_paths = [str(shard_dir / f"{stem}-{index:03d}.npz") for index in range(100)]
    identity = {
        "upstream_commit": manifest["upstream"]["commit"],
        "run_name": run.run_id,
        "run_dir": str(run_dir.resolve()),
        "wandb_id": run.wandb_id,
        "wandb_project": manifest["logging"]["wandb_project"],
        "protocol_sha256": campaign.manifest_sha256(manifest),
        "code_manifest_sha256": contract["code_manifest_sha256"],
        "code_files": contract["code_files"],
        "runtime_software": contract["runtime_software"],
        "run_group": manifest["logging"]["wandb_group"],
        "env_name": run.env_name,
        "seed": run.seed,
        "offline_steps": 1_000_000,
        "online_steps": 0,
        "sparse": run.sparse,
        "p_aug": None,
        "frame_stack": None,
        "utd": 1,
        "eval_interval": 100_000,
        "eval_episodes": 50,
        "final_eval_episodes": 50,
        "video_episodes": 0,
        "video_frame_skip": 3,
        "gradient_checkpointing": True,
        "ogbench_standard_dataset_dir": standard_dir,
        "dataset_replace_interval": 1000,
        "dataset_paths": dataset_paths,
        "agent": {**run.agent, "gradient_checkpointing": True},
    }
    identity_sha256 = campaign.immutable_identity_sha256(identity)
    completion = {
        "schema_version": 1,
        "status": "complete",
        "identity": identity,
        "identity_sha256": identity_sha256,
        "global_step": 1_000_000,
        "final_eval_step": 1_000_000,
        "final_eval_episodes": 50,
        "protocol_sha256": campaign.manifest_sha256(manifest),
        "code_manifest_sha256": contract["code_manifest_sha256"],
        "runtime_software_sha256": contract["runtime_software_sha256"],
        "run_name": run.run_id,
        "env_name": run.env_name,
        "seed": run.seed,
        "wandb_id": run.wandb_id,
        "final_evaluation": {"success": success},
        "checkpoint": "checkpoint.pkl",
        "upstream_commit": manifest["upstream"]["commit"],
        "gradient_checkpointing": True,
    }
    metadata = {
        "upstream_commit": manifest["upstream"]["commit"],
        "identity": identity,
        "identity_sha256": identity_sha256,
        "protocol_sha256": campaign.manifest_sha256(manifest),
        "code_manifest_sha256": contract["code_manifest_sha256"],
        "code_files": contract["code_files"],
        "runtime_software_sha256": contract["runtime_software_sha256"],
        "runtime": {**contract["runtime_software"], "platform": {}},
        "gradient_checkpointing": {"enabled": True},
    }
    return completion, metadata


def write_completion(manifest, run, run_dir, success=0.5):
    completion, metadata = completion_artifacts(manifest, run, run_dir, success)
    (run_dir / "COMPLETED.json").write_text(json.dumps(completion))
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata))
    return completion


def valid_npz_bytes() -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for name in prepare_data.REQUIRED_NPZ_MEMBERS:
            archive.writestr(name, b"test")
    return output.getvalue()


def test_manifest_is_exact_10_by_5_by_4(manifest):
    runs = campaign.expand_runs(manifest)
    assert len(runs) == 200
    assert len({run.run_id for run in runs}) == 200
    assert len({run.wandb_id for run in runs}) == 200
    assert {run.task_id for run in runs} == {1, 2, 3, 4, 5}
    assert {run.seed for run in runs} == {0, 1, 2, 3}
    assert manifest["training"]["offline_steps"] == 1_000_000
    assert manifest["execution"]["allocation_shards"] == 13
    assert (CAMPAIGN_DIR / "protocol.sha256").read_text().strip() == campaign.manifest_sha256(manifest)


def test_array_ownership_is_exactly_once_without_gaps(manifest):
    runs = campaign.expand_runs(manifest)
    ownership = {}
    active_counts = []
    for allocation_shard in range(13):
        active = 0
        for rank in range(16):
            assigned = campaign.worker_runs(
                runs,
                rank,
                allocation_shard=allocation_shard,
                allocation_shards=13,
            )
            assert len(assigned) <= 1
            if assigned:
                active += 1
                run = assigned[0]
                assert run.index == allocation_shard * 16 + rank
                assert run.index not in ownership
                ownership[run.index] = (allocation_shard, rank)
        active_counts.append(active)
    assert sorted(ownership) == list(range(200))
    assert active_counts == [16] * 12 + [8]
    assert [ownership[index] for index in range(192, 200)] == [(12, rank) for rank in range(8)]
    for rank in range(8, 16):
        assert campaign.worker_runs(
            runs,
            rank,
            allocation_shard=12,
            allocation_shards=13,
        ) == []


def test_array_state_isolation_and_final_idle_ranks_remain_in_barrier(tmp_path):
    roots = [
        dispatcher.allocation_state_root(tmp_path, f"12345_{shard}", shard, 13)
        for shard in range(13)
    ]
    assert len(set(roots)) == 13
    assert roots[12].name == "allocation-shard-12-of-13"
    assert roots[12].parent.name == "array-job-12345_12"

    coordinators = [
        dispatcher.RequeueCoordinator(
            roots[12],
            job_id="12345_12",
            restart_count=0,
            rank=rank,
            workers=16,
            wait_seconds=0,
            allocation_shard=12,
            allocation_shards=13,
        )
        for rank in range(16)
    ]
    for coordinator in coordinators[:15]:
        coordinator.mark_finished()
    assert not coordinators[0].all_finished()
    coordinators[15].mark_finished()
    assert coordinators[0].all_finished()


def test_array_requeue_target_is_one_composite_element():
    assert dispatcher.slurm_requeue_target(
        {
            "SLURM_JOB_ID": "99999",
            "SLURM_ARRAY_JOB_ID": "12345",
            "SLURM_ARRAY_TASK_ID": "12",
        }
    ) == "12345_12"
    assert dispatcher.slurm_requeue_target({"SLURM_JOB_ID": "67890"}) == "67890"
    with pytest.raises(ValueError):
        dispatcher.slurm_requeue_target({"SLURM_ARRAY_JOB_ID": "12345"})


def test_dispatcher_parses_explicit_array_coordinates(monkeypatch):
    monkeypatch.delenv("SLURM_ARRAY_JOB_ID", raising=False)
    monkeypatch.delenv("SLURM_ARRAY_TASK_ID", raising=False)
    args = dispatcher.parse_args(
        [
            "--allocation-shard=12",
            "--allocation-shards=13",
            "--worker-index=15",
            "--workers=16",
            "--job-id=12345_12",
            "--dry-run",
        ]
    )
    assert (args.allocation_shard, args.allocation_shards) == (12, 13)
    assert args.job_id == "12345_12"


def test_per_run_lock_rejects_duplicate_array_owner(manifest, tmp_path):
    run = campaign.expand_runs(manifest)[0]
    run_dir = campaign.run_directory(tmp_path, run)
    with dispatcher.run_lock(
        run_dir,
        run=run,
        job_id="12345_0",
        allocation_shard=0,
        rank=0,
    ):
        with pytest.raises(dispatcher.RunLockUnavailable):
            with dispatcher.run_lock(
                run_dir,
                run=run,
                job_id="54321_0",
                allocation_shard=0,
                rank=0,
            ):
                pytest.fail("a duplicate allocation must never acquire the same run")


def test_all_commands_pin_formal_protocol_and_no_credentials(manifest, tmp_path):
    commands = [
        campaign.build_train_command(
            manifest,
            run,
            python_executable="python",
            upstream_main=CAMPAIGN_DIR / "upstream_rql" / "main.py",
            run_root=tmp_path / "runs",
            data_root=tmp_path / "data",
        )
        for run in campaign.expand_runs(manifest)
    ]
    joined = [" ".join(command) for command in commands]
    protocol = campaign.manifest_sha256(manifest)
    assert all("--offline_steps=1000000" in command for command in joined)
    assert all("--eval_episodes=50" in command for command in joined)
    assert all("--final_eval_episodes=50" in command for command in joined)
    assert all("--gradient_checkpointing=true" in command for command in joined)
    assert all(f"--protocol_sha256={protocol}" in command for command in joined)
    assert sum("--ogbench_dataset_dir=" in command for command in joined) == 40
    assert sum("--ogbench_standard_dataset_dir=" in command for command in joined) == 160
    assert not any("API_KEY" in command or "TOKEN" in command for command in joined)


@pytest.mark.parametrize("field", ["run_name", "env_name", "seed", "wandb_id", "protocol_sha256"])
def test_completion_skip_rejects_misplaced_identity(manifest, tmp_path, field):
    run = campaign.expand_runs(manifest)[0]
    run_dir = campaign.run_directory(tmp_path, run)
    run_dir.mkdir(parents=True)
    payload = write_completion(manifest, run, run_dir)
    assert campaign.completion_is_valid(run_dir, manifest, run)
    payload[field] = "wrong" if field != "seed" else 999
    (run_dir / "COMPLETED.json").write_text(json.dumps(payload))
    assert not campaign.completion_is_valid(run_dir, manifest, run)


def test_data_manifest_expects_416_structural_npz_files(manifest, tmp_path):
    paths = prepare_data.expected_paths(manifest, tmp_path)
    assert len(paths) == 416
    assert len(set(paths)) == 416
    assert len(prepare_data.standard_paths(manifest, tmp_path)) == 16
    assert len(prepare_data.large_paths(manifest, tmp_path)) == 400
    paths[0].parent.mkdir(parents=True)
    paths[0].write_bytes(valid_npz_bytes())
    assert prepare_data.is_valid_npz(paths[0])
    assert len(prepare_data.check_all_data(manifest, tmp_path)) == 415


def test_resumable_download_appends_range_then_atomically_finishes(monkeypatch, tmp_path):
    content = valid_npz_bytes()
    destination = tmp_path / "sample.npz"
    partial = tmp_path / "sample.npz.part"
    partial.write_bytes(content[:17])

    class Response(io.BytesIO):
        status = 206

        def getcode(self):
            return self.status

    def fake_open(url, offset, timeout):
        assert url == "https://example.invalid/sample.npz"
        assert offset == 17
        assert timeout == 9
        return Response(content[offset:])

    monkeypatch.setattr(prepare_data, "_open_resume", fake_open)
    prepare_data.download_file("https://example.invalid/sample.npz", destination, timeout_seconds=9)
    assert destination.read_bytes() == content
    assert not partial.exists()


def test_download_retries_timeout_from_preserved_offset(monkeypatch, tmp_path):
    content = valid_npz_bytes()
    destination = tmp_path / "retry.npz"
    calls = []

    class Response(io.BytesIO):
        status = 206

        def getcode(self):
            return self.status

    def fake_open(url, offset, timeout):
        calls.append(offset)
        if len(calls) == 1:
            # Simulate bytes already made durable by a prior timed-out read.
            destination.with_name(destination.name + ".part").write_bytes(content[:23])
            raise TimeoutError("transient read timeout")
        assert offset == 23
        return Response(content[offset:])

    monkeypatch.setattr(prepare_data, "_open_resume", fake_open)
    monkeypatch.setattr(prepare_data.time, "sleep", lambda seconds: None)
    prepare_data.download_file(
        "https://example.invalid/retry.npz",
        destination,
        timeout_seconds=1,
        retries=2,
    )
    assert calls == [0, 23]
    assert destination.read_bytes() == content


def test_download_retries_incomplete_body_and_preserves_exception_bytes(monkeypatch, tmp_path):
    content = valid_npz_bytes()
    destination = tmp_path / "incomplete.npz"
    calls = []

    class TruncatedResponse:
        status = 200

        def __init__(self):
            self.read_count = 0

        def getcode(self):
            return self.status

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            del size
            self.read_count += 1
            if self.read_count == 1:
                return content[:19]
            raise http.client.IncompleteRead(content[19:31], len(content) - 19)

    class ResumedResponse(io.BytesIO):
        status = 206

        def getcode(self):
            return self.status

    def fake_open(url, offset, timeout):
        del url, timeout
        calls.append(offset)
        if len(calls) == 1:
            return TruncatedResponse()
        return ResumedResponse(content[offset:])

    monkeypatch.setattr(prepare_data, "_open_resume", fake_open)
    monkeypatch.setattr(prepare_data.time, "sleep", lambda seconds: None)
    prepare_data.download_file(
        "https://example.invalid/incomplete.npz",
        destination,
        timeout_seconds=1,
        retries=2,
    )
    assert calls == [0, 31]
    assert destination.read_bytes() == content


def test_download_timeout_honors_signal_before_backoff(monkeypatch, tmp_path):
    destination = tmp_path / "stopped.npz"

    def stopped_open(url, offset, timeout):
        prepare_data.STOP_REQUESTED = True
        raise TimeoutError("socket remained blocked until signal")

    monkeypatch.setattr(prepare_data, "_open_resume", stopped_open)
    monkeypatch.setattr(
        prepare_data.time,
        "sleep",
        lambda seconds: pytest.fail("must not back off after scheduler stop"),
    )
    try:
        with pytest.raises(prepare_data.GracefulDataStop):
            prepare_data.download_file(
                "https://example.invalid/stopped.npz",
                destination,
                timeout_seconds=1,
                retries=20,
            )
    finally:
        prepare_data.STOP_REQUESTED = False


def test_transient_retry_exhaustion_requests_safe_requeue(monkeypatch, tmp_path):
    destination = tmp_path / "source-down.npz"
    attempts = []

    def unavailable(url, offset, timeout):
        attempts.append((offset, timeout))
        raise TimeoutError("upstream unavailable")

    monkeypatch.setattr(prepare_data, "_open_resume", unavailable)
    monkeypatch.setattr(prepare_data.time, "sleep", lambda seconds: None)
    with pytest.raises(prepare_data.RetryableDownloadExhausted):
        prepare_data.download_file(
            "https://example.invalid/source-down.npz",
            destination,
            timeout_seconds=3,
            retries=2,
        )
    assert attempts == [(0, 3), (0, 3)]


def test_requeue_timeout_still_calls_scontrol(monkeypatch, tmp_path):
    coordinator = dispatcher.RequeueCoordinator(
        tmp_path,
        job_id="12345",
        restart_count=2,
        rank=0,
        workers=16,
        wait_seconds=0,
    )
    coordinator.request("preempt")
    coordinator.mark_ready(run_id=None, child_exit_code=75)
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="requeued")

    monkeypatch.setattr(dispatcher.subprocess, "run", fake_run)
    assert coordinator.resolve_as_rank_zero() == 75
    assert calls == [["scontrol", "requeue", "12345"]]
    assert (coordinator.directory / "RENDEZVOUS_FAILED.json").is_file()


def test_abort_suppresses_scontrol(monkeypatch, tmp_path):
    coordinator = dispatcher.RequeueCoordinator(
        tmp_path,
        job_id="54321",
        restart_count=0,
        rank=0,
        workers=16,
        wait_seconds=0,
    )
    coordinator.request("fatal", abort=True)
    monkeypatch.setattr(
        dispatcher.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("scontrol must not run for abort"),
    )
    assert coordinator.resolve_as_rank_zero() == 1


def test_nonzero_rank_observes_scontrol_failure(monkeypatch, tmp_path):
    coordinator = dispatcher.RequeueCoordinator(
        tmp_path,
        job_id="99999",
        restart_count=1,
        rank=3,
        workers=16,
        wait_seconds=0,
    )
    dispatcher.atomic_json(
        coordinator.directory / "REQUEUE_FAILED.json",
        {
            "job_id": "99999",
            "restart_count": 1,
            "rank": 0,
            "status": "scontrol_failed",
        },
    )
    monkeypatch.setattr(
        dispatcher.time,
        "sleep",
        lambda seconds: pytest.fail("failure resolution should be immediate"),
    )
    assert coordinator.wait_for_resolution() == 1


def test_gpu_probe_creates_output_directory_before_smoke(monkeypatch, tmp_path):
    output = tmp_path / "not-created-yet" / "preflight"
    monkeypatch.setenv("SLURM_PROCID", "0")
    monkeypatch.setenv("SLURM_LOCALID", "0")
    monkeypatch.setenv("SLURM_JOB_ID", "123")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    assert gpu_preflight.probe(output, CAMPAIGN_DIR / "upstream_rql", skip_jax=True) == 0
    assert (output / "rank.0.json").is_file()


def test_quick_gpu_verifier_requires_current_two_by_eight_topology(tmp_path):
    for rank in range(16):
        gpu_preflight.atomic_json(
            tmp_path / f"rank.{rank}.json",
            {
                "status": "ok",
                "rank": rank,
                "local_rank": rank % 8,
                "hostname": f"node-{rank // 8}",
                "cuda_visible_devices": [str(rank % 8)],
                "jax_device_count": 1,
                "jax_platform": "gpu",
                "preflight_level": "quick",
            },
        )
    assert gpu_preflight.verify(tmp_path, 16, level="quick") == 0
    payload = json.loads((tmp_path / "rank.15.json").read_text())
    payload["local_rank"] = 6
    gpu_preflight.atomic_json(tmp_path / "rank.15.json", payload)
    assert gpu_preflight.verify(tmp_path, 16, level="quick") == 2


def test_preflight_cache_uses_trainer_curated_runtime_contract():
    software = gpu_preflight.trainer_runtime_software(CAMPAIGN_DIR / "upstream_rql")
    assert set(software["packages"]) == {
        "jax",
        "jaxlib",
        "flax",
        "optax",
        "ogbench",
        "wandb",
        "numpy",
        "mujoco",
        "distrax",
        "einops",
        "ml_collections",
        "gymnasium",
    }
    digest = gpu_preflight.preflight_cache_key(
        CAMPAIGN_DIR / "protocol.sha256",
        CAMPAIGN_DIR / "upstream_rql",
    )
    assert len(digest) == 64
    assert digest == gpu_preflight.preflight_cache_key(
        CAMPAIGN_DIR / "protocol.sha256",
        CAMPAIGN_DIR / "upstream_rql",
    )


def test_strict_aggregate_reads_all_200_and_reports_seed_ci(manifest, tmp_path):
    for run in campaign.expand_runs(manifest):
        run_dir = campaign.run_directory(tmp_path, run)
        run_dir.mkdir(parents=True)
        success = (run.task_id + run.seed) / 10.0
        write_completion(manifest, run, run_dir, success)
    values = aggregate.load_final_successes(manifest, tmp_path)
    report = aggregate.build_report(manifest, values)
    assert report["validated_runs"] == 200
    assert len(report["per_task"]) == 50
    assert len(report["per_setting"]) == 10
    assert report["aggregate_50_task"]["n_seeds"] == 4
    output = tmp_path / "report"
    aggregate.write_report(report, output)
    assert {path.name for path in output.iterdir()} == {"report.json", "per_task.csv", "summary.md"}


def test_slurm_literal_contract():
    submit.validate_slurm(CAMPAIGN_DIR / "train.slurm")
    submit.validate_stage_slurm(CAMPAIGN_DIR / "stage_data.slurm")
    submit.validate_aggregate_slurm(CAMPAIGN_DIR / "aggregate.slurm")


def test_submit_reuses_stage_then_chains_array_and_strict_report(monkeypatch, tmp_path, capsys):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        if len(calls) == 1:
            return subprocess.CompletedProcess(argv, 0, stdout="40000\n")
        if len(calls) == 2:
            return subprocess.CompletedProcess(argv, 0, stdout="40014\n")
        pytest.fail(f"unexpected third submission: {argv}")

    monkeypatch.setattr(submit, "wandb_credentials_available", lambda: True)
    monkeypatch.setattr(submit.subprocess, "run", fake_run)
    status = submit.main(
        [
            "--submit",
            "--data-job-id=32711736",
            f"--repo-root={tmp_path / 'repo'}",
            f"--data-root={tmp_path / 'data'}",
            f"--run-root={tmp_path / 'runs'}",
            "--python=python",
        ]
    )
    assert status == 0
    assert calls[0][0] == [
        "sbatch",
        "--parsable",
        "--dependency=afterok:32711736",
        str((CAMPAIGN_DIR / "train.slurm").resolve()),
    ]
    assert calls[1][0] == [
        "sbatch",
        "--parsable",
        "--dependency=afterok:40000",
        str((CAMPAIGN_DIR / "aggregate.slurm").resolve()),
    ]
    output = capsys.readouterr().out
    assert "submitted 13-element training array job 40000" in output
    assert "submitted strict afterok report job 40014" in output
