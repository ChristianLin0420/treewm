from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CAMPAIGN_DIR.parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))

import aggregate  # noqa: E402
import campaign  # noqa: E402
import dispatcher  # noqa: E402
import gpu_preflight  # noqa: E402
import submit  # noqa: E402
import validate_pilot  # noqa: E402


@pytest.fixture
def manifest():
    return campaign.load_manifest(CAMPAIGN_DIR / "manifest.json")


def test_literal_slurm_and_full_dependency_sources():
    submit.validate_slurm(
        CAMPAIGN_DIR / "train.slurm",
        CAMPAIGN_DIR / "stage_data.slurm",
        CAMPAIGN_DIR / "calibration_gate.slurm",
        CAMPAIGN_DIR / "pilot.slurm",
        CAMPAIGN_DIR / "pilot_gate.slurm",
        CAMPAIGN_DIR / "aggregate.slurm",
    )
    train = (CAMPAIGN_DIR / "train.slurm").read_text()
    assert "#SBATCH --nodes=2" in train
    assert "#SBATCH --ntasks-per-node=8" in train
    assert "#SBATCH --gpus-per-node=8" in train
    assert 'REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"' in train
    assert "FALLBACK_REQUEUE_CALLING.json" in train
    assert "BATCH_REQUEUE_CALLING.json" in train
    pilot = (CAMPAIGN_DIR / "pilot.slurm").read_text()
    assert "--post-training-checkpoint" in pilot
    assert pilot.index("--run-setting") < pilot.index("--audit-setting-index")
    assert pilot.index("--audit-setting-index") < pilot.index("--validate-setting")
    for name in (
        "train.slurm", "stage_data.slurm", "calibration_gate.slurm",
        "pilot.slurm", "pilot_gate.slurm", "aggregate.slurm",
    ):
        text = (CAMPAIGN_DIR / name).read_text()
        assert "--verify-source-snapshot" in text
        assert "PYTHONDONTWRITEBYTECODE=1" in text


def test_protocol_covers_every_executable_orchestration_source():
    expected = {
        "dispatcher.py", "gpu_preflight.py", "train.slurm", "stage_data.slurm",
        "calibration_gate.slurm",
        "pilot.slurm", "pilot_gate.slurm", "validate_pilot.py",
        "aggregate.py", "aggregate.slurm", "submit.py",
    }
    assert expected.issubset(set(campaign.PROTOCOL_SOURCE_FILES))


def test_composite_array_target_and_isolated_generation_state():
    environ = {
        "SLURM_JOB_ID": "777",
        "SLURM_ARRAY_JOB_ID": "12345",
        "SLURM_ARRAY_TASK_ID": "2",
    }
    assert dispatcher.slurm_requeue_target(environ) == "12345_2"
    root = dispatcher.allocation_state_root(Path("/tmp/state"), "12345_2", 2, 3)
    assert root.parts[-2:] == ("array-job-12345_2", "allocation-shard-02-of-03")
    with pytest.raises(ValueError):
        dispatcher.slurm_requeue_target({"SLURM_ARRAY_JOB_ID": "12345"})


def _coordinators(tmp_path: Path, restart: int = 0):
    return [
        dispatcher.RequeueCoordinator(
            tmp_path, job_id="12345_0", restart_count=restart, rank=rank,
            workers=16, wait_seconds=0, allocation_shard=0, allocation_shards=3,
        )
        for rank in range(16)
    ]


def test_usr1_authorizes_only_after_sixteen_durable_checkpoints(tmp_path):
    coordinators = _coordinators(tmp_path)
    leader = coordinators[0]
    leader.request("requeue", "slurm_usr1")
    for rank, coordinator in enumerate(coordinators):
        coordinator.mark_ready(run_id=f"run-{rank}", child_exit_code=75)
    assert leader.authorize_as_rank_zero() == 75
    payload = json.loads((leader.directory / "REQUEUE_AUTHORIZED.json").read_text())
    assert payload["all_ready"] is True
    assert payload["ready_ranks"] == list(range(16))
    assert dispatcher.resolution_status(
        tmp_path, job_id="12345_0", restart_count=0, rank=0, workers=16,
        allocation_shard=0, allocation_shards=3,
        cancel_latch=tmp_path / "CANCEL_REQUESTED",
    ) == "authorized"


def test_sigterm_overrides_usr1_and_persists_into_successor(tmp_path):
    coordinators = _coordinators(tmp_path)
    leader = coordinators[0]
    leader.request("requeue", "slurm_usr1")
    leader.cancel("scancel")
    for coordinator in coordinators:
        coordinator.mark_ready(run_id=None, child_exit_code=75)
    assert leader.authorize_as_rank_zero() == 143
    assert not (leader.directory / "REQUEUE_AUTHORIZED.json").exists()
    assert _coordinators(tmp_path, restart=1)[0].cancelled


def test_missing_rank_fallback_requeues_only_composite_element(tmp_path, monkeypatch):
    coordinators = _coordinators(tmp_path)
    leader = coordinators[4]
    leader.request("requeue", "deadline")
    for rank, coordinator in enumerate(coordinators):
        if rank != 7:
            coordinator.mark_ready(run_id=f"run-{rank}", child_exit_code=75)
    calls = []
    monkeypatch.setattr(
        dispatcher.subprocess,
        "run",
        lambda argv, **kwargs: (
            calls.append(argv) or subprocess.CompletedProcess(argv, 0, stdout="requeued")
        ),
    )
    assert leader.coordinate_resolution() == 75
    assert calls == [["scontrol", "requeue", "12345_0"]]
    assert leader.fallback_calling_path.is_file()


def test_intentional_generation_marker_does_not_poison_successor(tmp_path):
    coordinator = _coordinators(tmp_path)[0]
    coordinator.request("requeue", "deadline")
    for rank, peer in enumerate(_coordinators(tmp_path)):
        peer.mark_ready(run_id=f"run-{rank}", child_exit_code=75)
    assert coordinator.authorize_as_rank_zero() == 75
    assert coordinator.mark_requeue_calling(fallback=False)
    state = dispatcher.SignalState()
    coordinator.signal_state = state
    state.term = True
    coordinator.sync_signals()
    assert not coordinator.cancelled
    successor = _coordinators(tmp_path, restart=1)[0]
    successor.signal_state = dispatcher.SignalState()
    successor.signal_state.term = True
    successor.sync_signals()
    assert successor.cancelled


def test_per_run_nonblocking_lease(manifest, tmp_path):
    run = campaign.expand_runs(manifest)[0]
    run_dir = campaign.run_directory(tmp_path, run)
    with dispatcher.run_lock(run_dir, run=run, job_id="1_0", allocation_shard=0, rank=0):
        with pytest.raises(dispatcher.RunLockUnavailable):
            with dispatcher.run_lock(run_dir, run=run, job_id="2_0", allocation_shard=0, rank=0):
                pass


def _gpu_payloads(root: Path, level: str) -> None:
    for rank in range(16):
        payload = {
            "schema_version": 2, "status": "ok", "rank": rank,
            "local_rank": rank % 8, "hostname": f"node-{rank // 8}",
            "cuda_visible_devices": [str(rank % 8)], "torch_cuda_device_count": 1,
            "preflight_level": level,
        }
        if level == "full":
            payload.update({
                "treewm_v2_real_data_update": True,
                "objective_version": "treewm_v2_rms_rank_v1",
                "gradient_checkpointing": True,
                "active_gain_gradient_audit": True,
                "gradient_audit_batches": 3,
                "gradient_share_bound": 0.80,
                "checkpoint_exact_resume": True,
                "ogbench_egl_render": True,
                "gradient_audit_sha256": "a" * 64,
                "wandb_auth_readonly": True if rank == 0 else None,
            })
        gpu_preflight.atomic_json(root / f"rank.{rank}.json", payload)


def test_topology_and_heavy_contract_verification(tmp_path):
    _gpu_payloads(tmp_path, "quick")
    assert gpu_preflight.verify(tmp_path, 16, level="quick") == 0
    _gpu_payloads(tmp_path, "full")
    assert gpu_preflight.verify(tmp_path, 16, level="full") == 0
    payload = json.loads((tmp_path / "rank.2.json").read_text())
    payload["objective_version"] = "treewm_v1"
    gpu_preflight.atomic_json(tmp_path / "rank.2.json", payload)
    assert gpu_preflight.verify(tmp_path, 16, level="full") == 2


def test_full_preflight_sentinel_is_exact_v2(tmp_path):
    sentinel = tmp_path / "SUCCESS.json"
    payload = {
        "schema_version": 2, "status": "full_gpu_preflight_complete",
        "cache_key": "a" * 64, "workers": 16,
        "objective_version": "treewm_v2_rms_rank_v1", "gradient_share_bound": 0.80,
        "gradient_audit_batches": 3,
    }
    gpu_preflight.atomic_json(sentinel, payload)
    assert gpu_preflight.validate_success_sentinel(sentinel, "a" * 64, 16)
    payload["gradient_share_bound"] = 0.81
    gpu_preflight.atomic_json(sentinel, payload)
    assert not gpu_preflight.validate_success_sentinel(sentinel, "a" * 64, 16)


def _audit_payload(max_encoder_share: float = 0.4):
    modules = validate_pilot.REQUIRED_AUDIT_MODULES
    active_terms = sorted(validate_pilot.FORMAL_ACTIVE_TERMS)
    dataset_size = 1000
    sampled_positions = gpu_preflight.representative_dataset_positions(dataset_size)
    batches = []
    for batch_index, positions in enumerate(sampled_positions):
        anchors = [10_000 + position for position in positions]
        metrics = {
            **{f"gradient_audit/objective_norm/{module}": 1.0 for module in modules},
            "gradient_audit/share/encoder/state": max_encoder_share,
            "gradient_audit/share/encoder/control": 1.0 - max_encoder_share,
        }
        for module in validate_pilot.SHARED_AUDIT_MODULES[1:]:
            metrics[f"gradient_audit/share/{module}/state"] = 0.5
            metrics[f"gradient_audit/share/{module}/control"] = 0.5
        batches.append(
            {
                "batch_index": batch_index,
                "dataset_positions": positions,
                "anchor_indices": anchors,
                "batch_sha256": gpu_preflight.canonical_hash(
                    {"dataset_positions": positions, "anchor_indices": anchors}
                ),
                "active_terms": list(active_terms),
                "metrics": metrics,
            }
        )
    payload = {
        "schema_version": 2, "status": "passed", "setting_id": "scene",
        "audit_step": 5000, "checkpoint_completed_updates": 5000,
        "scheduler_total_steps": 1_000_000,
        "dataset_size": dataset_size, "dataset_selection_seed": 0,
        "dataset_selection_sha256": gpu_preflight.canonical_hash(sampled_positions),
        "active_terms": list(active_terms),
        "batch_audits": batches,
    }
    payload["artifact_sha256"] = gpu_preflight.canonical_hash(payload)
    return payload


def test_representative_audit_sampling_is_deterministic_and_nonprefix():
    first = gpu_preflight.representative_dataset_positions(1000)
    second = gpu_preflight.representative_dataset_positions(1000)
    flattened = [position for batch in first for position in batch]
    assert first == second
    assert len(flattened) == len(set(flattened)) == 48
    assert set(flattened) != set(range(48))
    assert min(flattened) < 250
    assert max(flattened) >= 750


def test_gradient_audit_rejects_prefix_or_tampered_sampling():
    prefix = _audit_payload()
    prefix["batch_audits"][0]["dataset_positions"] = list(range(16))
    prefix["artifact_sha256"] = gpu_preflight.canonical_hash(
        {key: value for key, value in prefix.items() if key != "artifact_sha256"}
    )
    with pytest.raises(validate_pilot.PilotError, match="representative samples"):
        validate_pilot.validate_gradient_audit(prefix, "scene")

    tampered = _audit_payload()
    tampered["batch_audits"][0]["anchor_indices"][0] += 1
    with pytest.raises(validate_pilot.PilotError, match="artifact hash"):
        validate_pilot.validate_gradient_audit(tampered, "scene")


def test_gradient_audit_requires_exact_formal_active_terms():
    assert gpu_preflight.FORMAL_ACTIVE_TERMS == validate_pilot.FORMAL_ACTIVE_TERMS
    missing = _audit_payload()
    missing["batch_audits"][0]["active_terms"].remove("recursive")
    missing["artifact_sha256"] = gpu_preflight.canonical_hash(
        {key: value for key, value in missing.items() if key != "artifact_sha256"}
    )
    with pytest.raises(validate_pilot.PilotError, match="active terms differ"):
        validate_pilot.validate_gradient_audit(missing, "scene")

    forbidden = _audit_payload()
    forbidden["active_terms"].append("mass")
    forbidden["artifact_sha256"] = gpu_preflight.canonical_hash(
        {key: value for key, value in forbidden.items() if key != "artifact_sha256"}
    )
    with pytest.raises(validate_pilot.PilotError, match="wrong active-term union"):
        validate_pilot.validate_gradient_audit(forbidden, "scene")


def test_recent_loss_telemetry_requires_exact_formal_term_set_and_accounting():
    scalars = {"train/loss_total": [(5000, 1.2)]}
    for term in validate_pilot.FORMAL_ACTIVE_TERMS:
        scalars[f"train/loss_raw/{term}"] = [(5000, 0.1)]
        scalars[f"train/loss_effective/{term}"] = [(5000, 0.1)]
    validate_pilot.validate_recent_loss_telemetry(scalars, "scene")

    for forbidden in ("mass", "multistep"):
        invalid = {key: list(value) for key, value in scalars.items()}
        invalid[f"train/loss_raw/{forbidden}"] = [(5000, 0.0)]
        invalid[f"train/loss_effective/{forbidden}"] = [(5000, 0.0)]
        with pytest.raises(validate_pilot.PilotError, match="exact formal objective"):
            validate_pilot.validate_recent_loss_telemetry(invalid, "scene")

    missing = {key: list(value) for key, value in scalars.items()}
    missing.pop("train/loss_raw/reconstruction")
    missing.pop("train/loss_effective/reconstruction")
    with pytest.raises(validate_pilot.PilotError, match="exact formal objective"):
        validate_pilot.validate_recent_loss_telemetry(missing, "scene")


def test_gain_stride_four_never_overlaps_module_log_stride_fifty():
    # TrainingStepModule activates gain at zero-based step % 4 == 0, while the
    # module snapshot is taken when (step + 1) % 50 == 0. The validator must use
    # interval-averaged train/grad_norm_gain for live contextual-gain health.
    overlap = [
        step for step in range(200)
        if step % 4 == 0 and (step + 1) % 50 == 0
    ]
    assert overlap == []
    assert [step for step in range(50) if step % 4 == 0]


def _nondegeneracy_scalars(**overrides):
    values = {
        "control/q_pair_distance_mean": 0.051,
        "control/q_near_collapse_fraction": 0.949,
        "expansion/predicted_gain_std": 1.1e-4,
        "expansion/target_gain_std": 1.1e-4,
        "tree/effective_branching_factor": 1.0,
        "tree/support_recall": 0.101,
        "expansion/nodes_generated": 40.0,
        "expansion/budget_shortfall": 24.0,
    }
    values.update(overrides)
    return {tag: [(5000, value)] for tag, value in values.items()}


def test_preregistered_nondegeneracy_thresholds_are_strict_and_persistable():
    observed = validate_pilot.validate_recent_nondegeneracy(
        _nondegeneracy_scalars(), "scene"
    )
    assert observed["tree/effective_branching_factor"] == 1.0
    assert validate_pilot.NONDEGENERACY_THRESHOLDS == {
        "control/q_pair_distance_mean": {"operator": ">", "threshold": 0.05},
        "control/q_near_collapse_fraction": {"operator": "<", "threshold": 0.95},
        "expansion/predicted_gain_std": {"operator": ">", "threshold": 1e-4},
        "expansion/target_gain_std": {"operator": ">", "threshold": 1e-4},
        "tree/effective_branching_factor": {"operator": ">", "threshold": 0.5},
        "tree/effective_branching_factor:max": {"operator": "<=", "threshold": 4.0},
        "tree/support_recall": {"operator": ">", "threshold": 0.1},
        "expansion/nodes_generated": {"operator": ">=", "threshold": 32.0},
        "expansion/budget_shortfall": {"operator": ">=", "threshold": 0.0},
        "expansion/budget_shortfall:max": {"operator": "<=", "threshold": 32.0},
    }
    failures = (
        {"control/q_pair_distance_mean": 0.05},
        {"control/q_near_collapse_fraction": 0.95},
        {"expansion/predicted_gain_std": 1e-4},
        {"expansion/target_gain_std": 1e-4},
        {"tree/effective_branching_factor": 0.5},
        {"tree/effective_branching_factor": 4.0001},
        {"tree/support_recall": 0.1},
        {"expansion/nodes_generated": 31.999, "expansion/budget_shortfall": 32.001},
    )
    for override in failures:
        with pytest.raises(validate_pilot.PilotError, match="nondegeneracy gate"):
            validate_pilot.validate_recent_nondegeneracy(
                _nondegeneracy_scalars(**override), "scene"
            )


def test_global_pilot_acceptance_binds_recipe_and_audit_hashes(
    manifest, tmp_path, monkeypatch,
):
    settings = [setting["id"] for setting in manifest["settings"]]

    def passed(_manifest, run, _args):
        index = settings.index(run.setting_id)
        return {
            "setting_id": run.setting_id,
            "protocol_sha256": f"{index + 40:064x}",
            "code_sha256": "a" * 64,
            "runtime_sha256": "b" * 64,
            "data_manifest_sha256": f"{index:064x}",
            "calibration_sha256": f"{index + 10:064x}",
            "future_recipe_sha256": f"{index + 20:064x}",
            "gradient_audit_sha256": f"{index + 30:064x}",
            "max_shared_module_gradient_share": 0.5,
        }

    monkeypatch.setattr(validate_pilot, "validate_setting", passed)
    validate_pilot.validate_all(manifest, SimpleNamespace(pilot_root=tmp_path))
    acceptance = json.loads((tmp_path / "state" / "PILOT_ACCEPTED.json").read_text())
    assert acceptance["settings"] == settings
    assert acceptance["future_recipe_sha256_by_setting"] == {
        setting_id: f"{index + 20:064x}"
        for index, setting_id in enumerate(settings)
    }
    assert acceptance["gradient_audit_sha256_by_setting"] == {
        setting_id: f"{index + 30:064x}"
        for index, setting_id in enumerate(settings)
    }


def test_post_training_gradient_audit_gate_is_encoder_specific():
    validate_pilot.validate_gradient_audit(_audit_payload(0.60), "scene")
    with pytest.raises(validate_pilot.PilotError, match="encoder loss-gradient share"):
        validate_pilot.validate_gradient_audit(_audit_payload(0.81), "scene")
    initialization = _audit_payload(0.60)
    initialization["checkpoint_completed_updates"] = 0
    initialization["artifact_sha256"] = gpu_preflight.canonical_hash(
        {key: value for key, value in initialization.items() if key != "artifact_sha256"}
    )
    with pytest.raises(validate_pilot.PilotError, match="5k checkpoint"):
        validate_pilot.validate_gradient_audit(initialization, "scene")


def test_gradient_audit_uses_the_preregistered_validation_anchor_selection():
    source = (CAMPAIGN_DIR / "gpu_preflight.py").read_text(encoding="utf-8")
    assert "max_val_anchors=int(cfg.train.max_val_anchors)" in source
    assert "max_val_anchors=16" not in source


def test_pilot_launch_rewrites_only_lifecycle_overrides(manifest, tmp_path, monkeypatch):
    run = validate_pilot.seed_zero_run(manifest, 0)
    command = [
        "python", str(REPO_ROOT / "scripts" / "train.py"),
        "experiment=treewm_v2", "objective_version=treewm_v2_rms_rank_v1",
        "arm=treewm", "train.gradient_checkpointing=true", "retrieval.enabled=false",
        "train.steps=1000000",
        "train.scheduler_total_steps=1000000",
        "train.ckpt_every=2000", "train.eval_every=100000",
        "eval.final_episodes_per_task=50", "train.viz_every=100000",
        "train.viz_every_early=10000", "resume=auto",
    ]
    environment = {
        "WANDB_RUN_ID": "p" * 32,
        "WANDB_PROJECT": manifest["logging"]["pilot_wandb_project"],
        "WANDB_RUN_GROUP": manifest["logging"]["pilot_wandb_group"],
        "TREEWM_DATA_SHA256": "d" * 64,
        "TREEWM_CALIBRATION_SHA256": "c" * 64,
        "TREEWM_FUTURE_RECIPE_SHA256": "f" * 64,
        "TREEWM_DATA_CONTRACT_SHA256": "e" * 64,
    }
    monkeypatch.setattr(validate_pilot, "trainer_command", lambda *a, **k: (command, environment))
    argv, env = validate_pilot.pilot_launch_spec(
        manifest, run, repo_root=REPO_ROOT,
        data_root=tmp_path / "data", cache_root=tmp_path / "cache",
        pilot_root=Path(manifest["paths"]["pilot_run_root"]), python="python",
    )
    assert "train.steps=5000" in argv
    assert "train.scheduler_total_steps=1000000" in argv
    assert "train.ckpt_every=500" in argv
    assert "eval.final_episodes_per_task=1" in argv
    assert env["WANDB_PROJECT"].endswith("-pilot")


def test_pilot_and_formal_scheduler_factors_match_through_five_thousand():
    def factors(steps: int, scheduler_total_steps: int | None):
        warmup, floor = 1000, 0.1
        total = max(
            warmup + 1,
            scheduler_total_steps if scheduler_total_steps is not None else steps,
        )

        def factor(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, total - warmup)
            return floor + (1 - floor) * 0.5 * (
                1 + math.cos(math.pi * min(progress, 1.0))
            )

        return [factor(step) for step in (0, 999, 1000, 2500, 4999)]

    formal = factors(1_000_000, 1_000_000)
    pilot = factors(5_000, 1_000_000)
    unpinned_pilot = factors(5_000, None)
    assert pilot == formal
    assert unpinned_pilot[-1] != formal[-1]


def test_submit_dependency_chain_includes_pilot_gate(manifest, tmp_path, monkeypatch, capsys):
    calls = []
    submission_cwds = []
    snapshot_root = tmp_path / "immutable-snapshot"
    monkeypatch.setattr(submit, "validate", lambda args: (manifest, 416, False))
    monkeypatch.setattr(submit, "wandb_credentials_available", lambda: True)
    monkeypatch.setattr(
        submit, "prepare_source_snapshot", lambda args, loaded: snapshot_root
    )

    def fake_sbatch(argv, **kwargs):
        calls.append(argv)
        submission_cwds.append(kwargs["cwd"])
        return f"{30000 + len(calls) - 1}\n"

    monkeypatch.setattr(submit, "_run_sbatch", fake_sbatch)
    assert submit.main(["--submit", "--stage-data", f"--repo-root={REPO_ROOT}"]) == 0
    frozen = snapshot_root / "experiments" / "12-treewm-formal-v2"
    assert calls == [
        [str(frozen / "stage_data.slurm")],
        ["--dependency=afterok:30000", str(frozen / "calibration_gate.slurm")],
        ["--dependency=afterok:30001", "--export=ALL,TREEWM_STAGE_PHASE=recipe", str(frozen / "stage_data.slurm")],
        ["--dependency=afterok:30002", str(frozen / "pilot.slurm")],
        ["--dependency=afterok:30003", str(frozen / "pilot_gate.slurm")],
        ["--dependency=afterok:30004", str(frozen / "train.slurm")],
        ["--dependency=afterok:30005", str(frozen / "aggregate.slurm")],
    ]
    assert submission_cwds == [snapshot_root] * 7
    output = capsys.readouterr().out
    assert "5k-step TreeWM-v2 pilot" in output
    assert "three-element 1M TreeWM-v2" in output


def test_protocol_keyed_source_snapshot_is_isolated_readonly_and_verifiable(
    tmp_path, monkeypatch,
):
    live = tmp_path / "live"
    campaign_dir = live / "experiments" / "12-treewm-formal-v2"
    files = {
        live / "treewm" / "unit.py": "VALUE = 1\n",
        live / "configs" / "base.yaml": "train: {}\n",
        live / "scripts" / "train.py": "print('train')\n",
        live / "scripts" / "__init__.py": "",
        campaign_dir / "submit.py": "print('submit')\n",
        campaign_dir / "manifest.json": "{}\n",
        campaign_dir / "protocol.sha256": "placeholder\n",
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    (live / "outputs").mkdir()
    (live / "outputs" / "must-not-copy.txt").write_text("output")
    (live / ".netrc").write_text("secret")
    run_root = tmp_path / "formal-runs"
    manifest = {
        "campaign_id": "test-v2",
        "method": {"objective_version": "treewm_v2_rms_rank_v1"},
        "paths": {"run_root": str(run_root)},
    }

    def fake_code(root):
        root = Path(root)
        digest = hashlib.sha256(
            b"".join(
                (root / relative).read_bytes()
                for relative in (
                    "treewm/unit.py", "configs/base.yaml", "scripts/train.py"
                )
            )
        ).hexdigest()
        return {"manifest_sha256": digest, "files": {}}

    def fake_protocol(_manifest, campaign_dir=None):
        repository = Path(campaign_dir).resolve().parents[1]
        return hashlib.sha256(
            (repository / "treewm" / "unit.py").read_bytes()
        ).hexdigest()

    initial_protocol = fake_protocol(manifest, campaign_dir=campaign_dir)
    (campaign_dir / "protocol.sha256").write_text(initial_protocol + "\n")

    monkeypatch.setattr(submit, "PROTOCOL_SOURCE_FILES", ("submit.py",))
    monkeypatch.setattr(submit, "protocol_sha256", fake_protocol)
    monkeypatch.setattr(submit, "trainer_code_fingerprint", fake_code)
    monkeypatch.setattr(submit, "load_manifest", lambda path: manifest)
    args = SimpleNamespace(
        repo_root=live.resolve(), run_root=run_root.resolve(),
        protocol_lock=(campaign_dir / "protocol.sha256").resolve(),
    )
    snapshot = submit.prepare_source_snapshot(args, manifest)
    try:
        marker = submit.verify_source_snapshot(snapshot)
        assert snapshot == (
            run_root / "state" / "source-snapshots" / initial_protocol / "repo"
        )
        assert marker["source_file_count"] == 7
        assert (snapshot / "logs").is_symlink()
        assert os.access((snapshot / "logs").resolve(), os.W_OK)
        assert not (snapshot / "outputs" / "must-not-copy.txt").exists()
        assert not (snapshot / ".netrc").exists()
        assert all(
            (snapshot / relative).stat().st_mode & 0o222 == 0
            for relative in marker["source_files"]
        )

        # Later live-worktree edits cannot alter a submitted campaign.
        (live / "treewm" / "unit.py").write_text("VALUE = 2\n")
        assert submit.verify_source_snapshot(snapshot) == marker
        changed_protocol = fake_protocol(manifest, campaign_dir=campaign_dir)
        assert changed_protocol != initial_protocol
        assert (
            run_root / "state" / "source-snapshots" / changed_protocol / "repo"
        ) != snapshot

        # Snapshot corruption itself is detected before any allocation does work.
        copied = snapshot / "treewm" / "unit.py"
        copied.chmod(0o644)
        copied.write_text("VALUE = 3\n")
        with pytest.raises(submit.ValidationError, match="snapshot content"):
            submit.verify_source_snapshot(snapshot)
    finally:
        for path in snapshot.rglob("*"):
            if path.is_dir() and not path.is_symlink():
                path.chmod(0o700)
            elif path.is_file() and not path.is_symlink():
                path.chmod(0o600)
        snapshot.chmod(0o700)


def test_sbatch_scrubs_all_ambient_credentials(tmp_path, monkeypatch):
    captured = {}
    monkeypatch.setattr(
        submit.subprocess,
        "run",
        lambda argv, **kwargs: (
            captured.update(argv=argv, environment=kwargs["env"])
            or subprocess.CompletedProcess(argv, 0, stdout="12345\n")
        ),
    )
    output = submit._run_sbatch(
        ["train.slurm"], cwd=tmp_path,
        environment={"PATH": "/bin", "HOME": "/safe", "WANDB_API_KEY": "secret", "HF_TOKEN": "secret"},
        label="test",
    )
    assert output == "12345\n"
    assert captured["environment"] == {"PATH": "/bin", "HOME": "/safe"}


def test_shell_syntax_and_python_syntax():
    shell = subprocess.run(
        ["bash", "-n", *(str(CAMPAIGN_DIR / name) for name in (
            "train.slurm", "stage_data.slurm", "calibration_gate.slurm",
            "pilot.slurm", "pilot_gate.slurm", "aggregate.slurm"
        ))],
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    assert shell.returncode == 0, shell.stdout
    python = subprocess.run(
        [sys.executable, "-m", "py_compile", *(str(CAMPAIGN_DIR / name) for name in (
            "dispatcher.py", "gpu_preflight.py", "validate_pilot.py", "aggregate.py", "submit.py"
        ))],
        check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    assert python.returncode == 0, python.stdout


def test_no_embedded_secrets_and_no_v1_run_or_wandb_namespace():
    submit.scan_for_embedded_secrets([CAMPAIGN_DIR])
    manifest = json.loads((CAMPAIGN_DIR / "manifest.json").read_text())
    assert "v1" not in manifest["paths"]["run_root"].lower()
    assert "v1" not in manifest["paths"]["pilot_run_root"].lower()
    assert "v1" not in manifest["logging"]["wandb_group"].lower()
    assert "v1" not in manifest["logging"]["pilot_wandb_group"].lower()
    assert manifest["paths"]["raw_cache_root"].endswith("treewm-50task-full-cache-v1")


def test_fresh_namespace_validator_distinguishes_v2_contract_schema_suffix(manifest):
    args = SimpleNamespace(
        run_root=Path(manifest["paths"]["run_root"]).resolve(),
        pilot_root=Path(manifest["paths"]["pilot_run_root"]).resolve(),
        contract_root=Path(manifest["paths"]["contract_root"]).resolve(),
        cache_root=Path(manifest["paths"]["raw_cache_root"]).resolve(),
    )
    submit.validate_fresh_v2_paths(args, manifest)
    args.run_root = Path("/tmp/treewm-50task-1m-v1").resolve()
    with pytest.raises(submit.ValidationError, match="formal run root differs"):
        submit.validate_fresh_v2_paths(args, manifest)
