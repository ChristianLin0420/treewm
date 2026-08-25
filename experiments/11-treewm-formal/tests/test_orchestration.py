from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CAMPAIGN_DIR.parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import aggregate  # noqa: E402
import campaign  # noqa: E402
import dispatcher  # noqa: E402
import gpu_preflight  # noqa: E402
import submit  # noqa: E402


@pytest.fixture
def manifest():
    return campaign.load_manifest(CAMPAIGN_DIR / "manifest.json")


def test_literal_slurm_contracts_and_treewm_only_launch():
    submit.validate_slurm(
        CAMPAIGN_DIR / "train.slurm",
        CAMPAIGN_DIR / "stage_data.slurm",
        CAMPAIGN_DIR / "aggregate.slurm",
    )
    text = (CAMPAIGN_DIR / "train.slurm").read_text()
    assert "#SBATCH --array=0-2%3" in text
    assert 'REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"' in text
    assert "dispatcher.py" in text
    assert "FALLBACK_REQUEUE_CALLING.json" in text
    assert "BATCH_REQUEUE_CALLING.json" in text
    assert "upstream_rql" not in text
    stage_text = (CAMPAIGN_DIR / "stage_data.slurm").read_text()
    assert "REQUEUE_CALLING.json" in stage_text


def test_composite_array_target_and_isolated_state():
    environment = {
        "SLURM_JOB_ID": "777",
        "SLURM_ARRAY_JOB_ID": "12345",
        "SLURM_ARRAY_TASK_ID": "2",
    }
    assert dispatcher.slurm_requeue_target(environment) == "12345_2"
    root = dispatcher.allocation_state_root(Path("/tmp/state"), "12345_2", 2, 3)
    assert root.parts[-2:] == ("array-job-12345_2", "allocation-shard-02-of-03")
    with pytest.raises(ValueError):
        dispatcher.slurm_requeue_target({"SLURM_ARRAY_JOB_ID": "12345"})


def test_submit_preserves_locked_formal_python_symlink_path():
    args = submit.parse_args([f"--python={submit.FORMAL_PYTHON}"])
    assert args.python == submit.FORMAL_PYTHON
    assert str(args.python) == campaign.load_manifest(args.manifest)["paths"]["python"]


def test_formal_python_symlink_activates_its_venv_site_packages():
    result = subprocess.run(
        [
            str(submit.FORMAL_PYTHON),
            "-c",
            "import hydra, sys; print(sys.executable); print(hydra.__file__)",
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env={
            **submit.scrub_sensitive_environment(os.environ),
            "PYTHONNOUSERSITE": "1",
        },
    )
    assert result.returncode == 0, result.stdout
    lines = result.stdout.splitlines()
    assert lines[0] == str(submit.FORMAL_PYTHON)
    assert "treewm-formal-py311" in lines[1]


def _coordinators(tmp_path: Path, restart: int = 0):
    return [
        dispatcher.RequeueCoordinator(
            tmp_path,
            job_id="12345_0",
            restart_count=restart,
            rank=rank,
            workers=16,
            wait_seconds=1,
            allocation_shard=0,
            allocation_shards=3,
        )
        for rank in range(16)
    ]


def test_usr1_authorizes_only_after_all_sixteen_durable(tmp_path):
    coordinators = _coordinators(tmp_path)
    coordinators[0].request("requeue", "slurm_usr1")
    for rank, coordinator in enumerate(coordinators):
        coordinator.mark_ready(run_id=f"run-{rank}", child_exit_code=75)
    assert coordinators[0].authorize_as_rank_zero() == 75
    payload = json.loads(
        (coordinators[0].directory / "REQUEUE_AUTHORIZED.json").read_text()
    )
    assert payload["ready_ranks"] == list(range(16))


def test_sigterm_overrides_earlier_usr1_and_persists_across_restart(tmp_path):
    coordinators = _coordinators(tmp_path)
    leader = coordinators[0]
    leader.request("requeue", "slurm_usr1")
    leader.cancel("slurm_sigterm")
    for coordinator in coordinators:
        coordinator.mark_ready(run_id=None, child_exit_code=75)
    assert leader.request_payload()["action"] == "cancel"
    assert leader.authorize_as_rank_zero() == 143
    assert not (leader.directory / "REQUEUE_AUTHORIZED.json").exists()
    successor = _coordinators(tmp_path, restart=1)[0]
    assert successor.cancelled


def test_abort_and_incomplete_barrier_never_authorize_requeue(tmp_path):
    coordinators = _coordinators(tmp_path)
    leader = coordinators[0]
    leader.request("abort", "trainer_failed")
    leader.mark_ready(run_id="failed", child_exit_code=2)
    leader.wait_seconds = 0
    assert leader.authorize_as_rank_zero() == 1
    assert not (leader.directory / "REQUEUE_AUTHORIZED.json").exists()


def test_missing_rank_still_explicitly_requeues_latest_periodic_checkpoint(
    tmp_path, monkeypatch
):
    coordinators = _coordinators(tmp_path)
    leader = coordinators[4]
    leader.request("requeue", "allocation_wide_deadline")
    # Rank 7 is dead/hung. Every survivor has a periodic exact checkpoint even
    # though only fifteen can acknowledge this generation's signal checkpoint.
    for rank, coordinator in enumerate(coordinators):
        if rank != 7:
            coordinator.mark_ready(run_id=f"run-{rank}", child_exit_code=75)
    leader.wait_seconds = 0
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, stdout="requeued")

    monkeypatch.setattr(dispatcher.subprocess, "run", fake_run)
    assert leader.coordinate_resolution() == 75
    assert calls == [["scontrol", "requeue", "12345_0"]]
    authorization = json.loads(
        (leader.directory / "REQUEUE_AUTHORIZED.json").read_text()
    )
    assert authorization["all_ready"] is False
    assert authorization["ready_ranks"] == [rank for rank in range(16) if rank != 7]
    assert (
        dispatcher.resolution_status(
            tmp_path,
            job_id="12345_0",
            restart_count=0,
            rank=0,
            workers=16,
            allocation_shard=0,
            allocation_shards=3,
            cancel_latch=tmp_path / "CANCEL_REQUESTED",
        )
        == "called"
    )


def test_missing_rank_fallback_still_honors_cancel_override(tmp_path, monkeypatch):
    coordinators = _coordinators(tmp_path)
    leader = coordinators[2]
    leader.request("requeue", "deadline")
    leader.cancel("scancel")
    leader.mark_ready(run_id=None, child_exit_code=75)
    leader.wait_seconds = 0
    monkeypatch.setattr(
        dispatcher.subprocess,
        "run",
        lambda *a, **k: pytest.fail("cancel must suppress fallback scontrol"),
    )
    assert leader.coordinate_resolution() == 143


def test_fallback_requeue_term_is_intentional_teardown_not_scancel(
    tmp_path, monkeypatch
):
    coordinators = _coordinators(tmp_path)
    leader = coordinators[5]
    leader.request("requeue", "allocation_wide_deadline")
    for rank, coordinator in enumerate(coordinators):
        if rank != 9:
            coordinator.mark_ready(run_id=f"run-{rank}", child_exit_code=75)
    leader.wait_seconds = 0
    signal_state = dispatcher.SignalState()
    leader.signal_state = signal_state

    def fake_run(argv, **kwargs):
        del kwargs
        # Successful live-step scontrol tears down the step with TERM. The marker
        # must already be durable when the handler observes it.
        assert leader.intentional_requeue_teardown
        signal_state.term = True
        leader.sync_signals()
        assert not leader.cancelled
        assert leader.request_payload()["action"] == "requeue"
        return subprocess.CompletedProcess(argv, 0, stdout="requeued")

    monkeypatch.setattr(dispatcher.subprocess, "run", fake_run)
    assert leader.coordinate_resolution() == 75
    assert leader.fallback_calling_path.is_file()
    assert not (tmp_path / "CANCEL_REQUESTED").exists()
    assert not leader.cancelled_path.exists()
    # The marker is scoped to restart generation zero. A successor neither sees
    # it as cancellation nor suppresses a genuine TERM in generation one.
    successor = _coordinators(tmp_path, restart=1)[0]
    assert not successor.intentional_requeue_teardown
    assert not successor.cancelled


def test_failed_fallback_scontrol_removes_intent_marker(tmp_path, monkeypatch):
    coordinators = _coordinators(tmp_path)
    leader = coordinators[3]
    leader.request("requeue", "allocation_wide_deadline")
    leader.mark_ready(run_id="run-3", child_exit_code=75)
    leader.wait_seconds = 0
    monkeypatch.setattr(
        dispatcher.subprocess,
        "run",
        lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 1, stdout="scheduler rejected request"
        ),
    )
    assert leader.coordinate_resolution() == 1
    assert not leader.fallback_calling_path.exists()
    assert (leader.directory / "REQUEUE_FAILED.json").is_file()


def test_per_run_nonblocking_lease_rejects_duplicate(manifest, tmp_path):
    run = campaign.expand_runs(manifest)[0]
    run_dir = campaign.run_directory(tmp_path, run)
    with dispatcher.run_lock(run_dir, run=run, job_id="1_0", allocation_shard=0, rank=0):
        with pytest.raises(dispatcher.RunLockUnavailable):
            with dispatcher.run_lock(
                run_dir, run=run, job_id="2_0", allocation_shard=0, rank=0
            ):
                pytest.fail("duplicate lease must not be acquired")


def test_launch_guard_accepts_named_treewm_and_rejects_wrong_family(monkeypatch, tmp_path):
    args = type(
        "Args",
        (),
        {
            "python": "python",
            "repo_root": REPO_ROOT,
            "run_root": tmp_path / "runs",
            "data_root": tmp_path / "data",
            "cache_root": tmp_path / "cache",
            "wandb_project": "treewm-50task-formal",
            "wandb_mode": "online",
        },
    )()
    manifest = {"logging": {"wandb_project": "treewm-50task-formal"}}
    run = type("Run", (), {})()
    monkeypatch.setattr(
        dispatcher,
        "trainer_command",
        lambda *a, **k: (
            ["python", str(REPO_ROOT / "scripts" / "train.py"), "arm=treewm"],
            {"WANDB_RUN_ID": "stable"},
        ),
    )
    command, _ = dispatcher._launch_spec(args, manifest, run)
    assert command[-1] == "arm=treewm"
    monkeypatch.setattr(
        dispatcher,
        "trainer_command",
        lambda *a, **k: (
            ["/base/interpreter/python", str(REPO_ROOT / "scripts" / "train.py"), "arm=treewm"],
            {},
        ),
    )
    with pytest.raises(ValueError, match="virtual-environment interpreter path"):
        dispatcher._launch_spec(args, manifest, run)
    monkeypatch.setattr(
        dispatcher,
        "trainer_command",
        lambda *a, **k: (["python", "upstream_main.py", "arm=treewm"], {}),
    )
    with pytest.raises(ValueError):
        dispatcher._launch_spec(args, manifest, run)


def _write_gpu_payloads(root: Path, level: str = "quick") -> None:
    for rank in range(16):
        payload = {
            "status": "ok",
            "rank": rank,
            "local_rank": rank % 8,
            "hostname": f"node-{rank // 8}",
            "cuda_visible_devices": [str(rank % 8)],
            "torch_cuda_device_count": 1,
            "preflight_level": level,
        }
        if level == "full":
            payload.update(
                {
                    "treewm_full_update": True,
                    "arm": "treewm",
                    "model_class": "TreeWM",
                    "scorer": "learned",
                    "gradient_checkpointing": True,
                    "checkpoint_restore": True,
                    "ogbench_egl_render": True,
                    "wandb_auth_readonly": True if rank == 0 else None,
                }
            )
        gpu_preflight.atomic_json(root / f"rank.{rank}.json", payload)


def test_every_generation_topology_verifier_is_exact_two_by_eight(tmp_path):
    _write_gpu_payloads(tmp_path)
    assert gpu_preflight.verify(tmp_path, 16, level="quick") == 0
    payload = json.loads((tmp_path / "rank.15.json").read_text())
    payload["local_rank"] = 6
    gpu_preflight.atomic_json(tmp_path / "rank.15.json", payload)
    assert gpu_preflight.verify(tmp_path, 16, level="quick") == 2


def test_full_preflight_requires_named_method_remat_restore_egl_and_wandb(tmp_path):
    _write_gpu_payloads(tmp_path, level="full")
    assert gpu_preflight.verify(tmp_path, 16, level="full") == 0
    payload = json.loads((tmp_path / "rank.3.json").read_text())
    payload["arm"] = "randomtreewm"
    gpu_preflight.atomic_json(tmp_path / "rank.3.json", payload)
    assert gpu_preflight.verify(tmp_path, 16, level="full") == 2


def test_cached_gpu_gate_requires_exact_success_sentinel(tmp_path):
    sentinel = tmp_path / "SUCCESS.json"
    key = "a" * 64
    gpu_preflight.atomic_json(
        sentinel,
        {
            "schema_version": 1,
            "status": "full_gpu_preflight_complete",
            "cache_key": key,
            "workers": 16,
        },
    )
    assert gpu_preflight.validate_success_sentinel(sentinel, key, 16)
    payload = json.loads(sentinel.read_text())
    payload["cache_key"] = "b" * 64
    gpu_preflight.atomic_json(sentinel, payload)
    assert not gpu_preflight.validate_success_sentinel(sentinel, key, 16)


def _write_raw_runs(manifest, run_root: Path, *, reorder_first: bool = False) -> None:
    for run_index, run in enumerate(campaign.expand_runs(manifest)):
        run_dir = campaign.run_directory(run_root, run)
        run_dir.mkdir(parents=True)
        results = []
        metrics = {"eval/num_episodes": 250, "eval/success_rate": 0.0}
        all_successes = []
        for task_index, task_id in enumerate(range(1, 6)):
            successes = []
            for episode_index in range(50):
                success = float((run.seed + task_id + episode_index) % 3 == 0)
                successes.append(success)
                all_successes.append(success)
                results.append(
                    {
                        "task_index": task_index,
                        "task_id": task_id,
                        "episode_index": episode_index,
                        "success": success,
                    }
                )
            metrics[f"eval/task{task_id}/success_rate"] = sum(successes) / 50
            metrics[f"eval/task{task_id}/successes"] = sum(successes)
            metrics[f"eval/task{task_id}/num_episodes"] = 50
        metrics["eval/success_rate"] = sum(all_successes) / 250
        if reorder_first and run_index == 0:
            results[0], results[1] = results[1], results[0]
        progress = {
            "schema_version": 1,
            "status": "complete",
            "identity_sha256": "i" * 64,
            "task_ids": [1, 2, 3, 4, 5],
            "episodes_per_task": 50,
            "completed_results": results,
            "metrics": metrics,
        }
        (run_dir / "final_eval_progress.json").write_text(json.dumps(progress))
        (run_dir / "COMPLETED.json").write_text(
            json.dumps(
                {
                    "wandb_id": run.wandb_id,
                    "final_evaluation": metrics,
                    "final_eval_progress": "final_eval_progress.json",
                }
            )
        )


def test_aggregate_recomputes_all_10000_raw_episodes(manifest, tmp_path, monkeypatch):
    run_root = tmp_path / "runs"
    _write_raw_runs(manifest, run_root)
    monkeypatch.setattr(aggregate, "completion_is_valid", lambda *a, **k: True)
    monkeypatch.setattr(
        aggregate,
        "load_data_contract",
        lambda *a, **k: {"data_manifest_sha256": "d" * 64},
    )
    monkeypatch.setattr(
        aggregate,
        "live_contract",
        lambda *a, **k: {"code_sha256": "c" * 64, "runtime_sha256": "r" * 64},
    )
    values, hashes = aggregate.load_task_seed_values(
        manifest,
        run_root,
        repo_root=REPO_ROOT,
        data_root=tmp_path / "data",
        cache_root=tmp_path / "cache",
    )
    assert len(values) == 200 and len(hashes) == 10
    report = aggregate.build_report(
        manifest, values, repo_root=REPO_ROOT, data_hashes=hashes
    )
    assert report["validated_model_runs"] == 40
    assert report["validated_raw_episodes"] == 10_000
    output = tmp_path / "report"
    aggregate.write_report(report, values, output)
    assert {path.name for path in output.iterdir()} == {
        "report.json",
        "task_seed.csv",
        "per_task.csv",
        "summary.md",
        "REPORT_COMPLETED.json",
    }


def test_aggregate_rejects_reordered_raw_episode_prefix(manifest, tmp_path, monkeypatch):
    run_root = tmp_path / "runs"
    _write_raw_runs(manifest, run_root, reorder_first=True)
    monkeypatch.setattr(aggregate, "completion_is_valid", lambda *a, **k: True)
    monkeypatch.setattr(
        aggregate,
        "load_data_contract",
        lambda *a, **k: {"data_manifest_sha256": "d" * 64},
    )
    with pytest.raises(aggregate.ReportError, match="deterministic 5x50 order"):
        aggregate.load_task_seed_values(
            manifest,
            run_root,
            repo_root=REPO_ROOT,
            data_root=tmp_path / "data",
            cache_root=tmp_path / "cache",
        )


def test_submit_chains_cache_training_and_report_afterok(
    manifest, tmp_path, monkeypatch, capsys
):
    calls = []

    monkeypatch.setattr(submit, "validate", lambda args: (manifest, 416, False))
    monkeypatch.setattr(submit, "wandb_credentials_available", lambda: True)

    def fake_sbatch(argv, **kwargs):
        calls.append(argv)
        return ["30000\n", "30001\n", "30002\n"][len(calls) - 1]

    monkeypatch.setattr(submit, "_run_sbatch", fake_sbatch)
    status = submit.main(
        [
            "--submit",
            "--stage-data",
            f"--repo-root={REPO_ROOT}",
            f"--run-root={tmp_path / 'runs'}",
        ]
    )
    assert status == 0
    assert calls[0] == [str((CAMPAIGN_DIR / "stage_data.slurm").resolve())]
    assert calls[1] == [
        "--dependency=afterok:30000",
        str((CAMPAIGN_DIR / "train.slurm").resolve()),
    ]
    assert calls[2] == [
        "--dependency=afterok:30001",
        str((CAMPAIGN_DIR / "aggregate.slurm").resolve()),
    ]
    output = capsys.readouterr().out
    assert "three-element TreeWM training array 30001" in output
    assert "strict afterok report 30002" in output


def test_every_sbatch_call_scrubs_ambient_credentials(tmp_path, monkeypatch):
    captured = {}

    def fake_run(argv, **kwargs):
        captured["argv"] = argv
        captured["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(argv, 0, stdout="12345\n")

    monkeypatch.setattr(submit.subprocess, "run", fake_run)
    output = submit._run_sbatch(
        ["train.slurm"],
        cwd=tmp_path,
        environment={
            "PATH": "/usr/bin",
            "HOME": "/safe/home",
            "GITLAB_TOKEN": "must-not-reach-slurm",
            "VSCODE_GIT_IPC_AUTH_TOKEN": "must-not-reach-slurm",
            "WANDB_API_KEY": "must-not-reach-slurm",
            "DATABASE_PASSWORD": "must-not-reach-slurm",
            "SERVICE_CREDENTIAL": "must-not-reach-slurm",
        },
        label="test",
    )
    assert output == "12345\n"
    assert captured["environment"] == {"PATH": "/usr/bin", "HOME": "/safe/home"}


def test_wandb_submission_auth_uses_netrc_not_scrubbed_environment(monkeypatch):
    class FakeNetrc:
        def authenticators(self, host):
            assert host == "api.wandb.ai"
            return ("user", None, "netrc-secret")

    monkeypatch.setenv("WANDB_API_KEY", "ambient-key-is-not-the-job-credential")
    monkeypatch.setattr(submit.netrc, "netrc", lambda: FakeNetrc())
    assert submit.wandb_credentials_available()
    assert "WANDB_API_KEY" not in submit.scrub_sensitive_environment(dict(os.environ))


def test_shell_syntax_is_valid():
    result = subprocess.run(
        [
            "bash",
            "-n",
            str(CAMPAIGN_DIR / "train.slurm"),
            str(CAMPAIGN_DIR / "stage_data.slurm"),
            str(CAMPAIGN_DIR / "aggregate.slurm"),
        ],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    assert result.returncode == 0, result.stdout


def test_no_embedded_credentials():
    submit.scan_for_embedded_secrets([CAMPAIGN_DIR, REPO_ROOT / "configs" / "env"])
