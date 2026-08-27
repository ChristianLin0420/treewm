from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import signal
import sys

import pytest


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CAMPAIGN_DIR.parents[1]


def _load_local_modules():
    names = ("campaign", "metric_boundary", "worker", "stage_gate", "submit")
    previous = {name: sys.modules.get(name) for name in names}
    for name in names:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(CAMPAIGN_DIR))
    try:
        campaign = importlib.import_module("campaign")
        metric_boundary = importlib.import_module("metric_boundary")
        worker = importlib.import_module("worker")
        stage_gate = importlib.import_module("stage_gate")
        submit = importlib.import_module("submit")
        return campaign, metric_boundary, worker, stage_gate, submit
    finally:
        sys.path.remove(str(CAMPAIGN_DIR))
        for name in names:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]


campaign, metric_boundary, worker, stage_gate, submit = _load_local_modules()


@pytest.fixture(scope="module")
def manifest():
    return campaign.load_manifest(CAMPAIGN_DIR / "manifest.json")


def _contract():
    return {
        "chosen_thresholds": {
            "retrieval_radius": 1.0,
            "displacement_threshold": 2.0,
            "cluster_threshold": 3.0,
        },
        "contract_sha256": "1" * 64,
        "calibration_sha256": "2" * 64,
        "future_recipe_sha256": "3" * 64,
    }


def test_exact_stage_mappings_and_fresh_seeds(manifest):
    runs = campaign.expand_runs(manifest)
    assert len(runs) == 30
    assert [run.index for run in runs] == list(range(30))
    assert {run.seed for run in runs} == {108, 109}
    assert {run.arm_id for run in runs} == {"N", "G", "GS"}
    assert len({run.run_name for run in runs}) == 30
    assert len({run.wandb_id for run in runs}) == 30
    assert all(run.run_name.startswith("gauge-v2-") for run in runs)
    assert (runs[0].setting_id, runs[0].arm_id, runs[0].seed) == (
        "antmaze-large",
        "N",
        108,
    )
    assert (runs[-1].setting_id, runs[-1].arm_id, runs[-1].seed) == (
        "cube-quadruple-100m",
        "GS",
        109,
    )
    for run in runs:
        assert run.index == ((run.setting_index * 3) + run.arm_index) * 2 + run.seed_index

    continuation = campaign.continuation_runs(manifest)
    assert len(continuation) == 20
    assert {run.arm_id for run in continuation} == {"G", "GS"}
    assert {run.seed for run in continuation} == {108, 109}
    assert all(run.arm_id != "N" for run in continuation)
    assert campaign.run_at_stage(manifest, 5000, 2) == runs[2]
    assert campaign.run_at_stage(manifest, 25000, 0) == runs[2]


def test_n_g_gs_change_only_preregistered_gauge_and_clip_switches(manifest):
    by_arm = {}
    for arm_id in campaign.ARM_IDS:
        run = next(run for run in campaign.expand_runs(manifest) if run.arm_id == arm_id)
        by_arm[arm_id] = set(campaign.scientific_overrides(manifest, run, _contract()))

    invariants = (
        "experiment=treewm_v2_grounded_gauge_pilot_v2",
        "objective_version=treewm_v2_grounded_gauge_pilot_v2",
        "train.steps=25000",
        "train.scheduler_total_steps=1000000",
        "train.lr=3e-05",
        "train.weight_decay=0.001",
        "train.ckpt_every=1000",
        "train.val_every=1000",
        "train.diag_every=1000",
        "train.eval_every=12500",
        "train.validation_sample_seed=1701",
        "losses.multistep_transition_mode=grounded_execution_v2",
        "losses.grounded_select_action_weight=1.0",
        "losses.grounded_select_endpoint_weight=1.0",
        "losses.grounded_select_horizon_weight=0.25",
        "losses.grounded_loss_latent_weight=0.25",
        "losses.grounded_loss_action_weight=0.5",
        "losses.grounded_loss_horizon_weight=0.25",
        "losses.grounded_loss_endpoint_weight=0.5",
        "losses.scheduled_sampling_p=0.25",
        "losses.scheduled_sampling_warmup=5000",
        "losses.scheduled_sampling_granularity=sequence",
        "planner.require_first_edge_improvement=true",
        "tree.scorer=learned",
        "eval.final_episodes_per_task=5",
        "eval.seed=2718",
    )
    for overrides in by_arm.values():
        assert set(invariants) <= overrides
    assert "losses.enabled.latent_gauge=false" in by_arm["N"]
    assert "losses.weights.latent_gauge=0.0" in by_arm["N"]
    assert "train.separate_branch_transformer_grad_clip=false" in by_arm["N"]
    assert "losses.enabled.latent_gauge=true" in by_arm["G"]
    assert "losses.weights.latent_gauge=1.0" in by_arm["G"]
    assert "train.separate_branch_transformer_grad_clip=false" in by_arm["G"]
    assert "losses.enabled.latent_gauge=true" in by_arm["GS"]
    assert "losses.weights.latent_gauge=1.0" in by_arm["GS"]
    assert "train.separate_branch_transformer_grad_clip=true" in by_arm["GS"]
    assert "train.branch_transformer_grad_clip=1.0" in by_arm["GS"]


def test_actual_outcome_bank_is_exact_25k_periodic_monitor(manifest):
    bank = campaign.actual_evaluation_bank(manifest)
    assert bank["policy"] == "fixed_cfg_eval_seed_fallback_periodic_monitor"
    assert bank["stage_target"] == 25000
    assert bank["task_ids"] == [1, 2, 3, 4, 5]
    assert bank["episodes_per_task"] == 1
    assert bank["seeds"] == [[2718], [3718], [4718], [5718], [6718]]


def test_new_objective_is_available_only_through_sealed_local_entry():
    from scripts import train

    objective = "treewm_v2_grounded_gauge_pilot_v2"
    with pytest.raises(ValueError, match="unsupported objective_version"):
        train.validate_objective_version(objective, 25_000)
    assert {"train_entry.py", "metric_boundary.py"} <= set(campaign.PROTOCOL_FILES)
    config = (
        REPO_ROOT / "configs/experiment/treewm_v2_grounded_gauge_pilot_v2.yaml"
    ).read_text(encoding="utf-8")
    assert f"objective_version: {objective}" in config


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["design"].update(seeds=[102, 103]),
        lambda value: value["design"].update(promotion_precedence=["GS", "G"]),
        lambda value: value["design"].update(nonpromotable_arms=[]),
        lambda value: value["arms"][0].update(promotable=True),
        lambda value: value["arms"][0].update(latent_gauge_enabled=True),
        lambda value: value["arms"][1].update(latent_gauge_weight=0.0),
        lambda value: value["arms"][2].update(separate_branch_transformer_grad_clip=False),
        lambda value: value["scientific_contract"].update(optimizer_updates=5000),
        lambda value: value["scientific_contract"].update(scheduler_total_steps=25000),
        lambda value: value["stage_acceptance"].update(min_scale_ratio=0.79),
        lambda value: value["stage_acceptance"].update(universal_candidate_cells_required=9),
        lambda value: value["lifecycle"].update(stage_targets=[5000]),
        lambda value: value["execution"].update(stage_5000_array="0-29%1"),
        lambda value: value["superseded_design"].update(results_consumed=True),
        lambda value: value["superseded_design"].update(resume_allowed=True),
    ],
)
def test_manifest_rejects_scientific_selection_or_lifecycle_drift(manifest, mutation):
    changed = copy.deepcopy(manifest)
    mutation(changed)
    with pytest.raises(campaign.ContractError):
        campaign.validate_manifest(changed)


def _healthy_metrics(manifest, target=5000, *, gs=False, ratio=0.90, clip=0.50):
    gate = manifest["stage_acceptance"]
    metrics = {tag: {target: 1.0} for tag in gate["required_finite_tags"]}
    validation_steps = range(1000, target + 1, 1000)
    metrics["data/validation_fixed_sample_count"] = {step: 5120.0 for step in validation_steps}
    metrics["val/loss_total"] = {step: 1.0 for step in validation_steps}
    metrics["val/loss_multistep_self_fed"] = {step: 1.0 for step in validation_steps}
    metrics["val/loss_horizon"] = {step: 1.0 for step in validation_steps}
    for horizon in (4, 8, 16, 32, 64):
        metrics[f"data/validation_horizon_label_fraction_h{horizon}"] = {
            step: 0.2 for step in validation_steps
        }
    metrics["control/retrieval_uses_task_metric_endpoint"] = {target: 1.0}
    metrics["control/q_advantage_over_z"] = {target: 0.1}
    metrics["control/q_advantage_over_random_proj"] = {target: 0.1}
    metrics["tree/support_recall"] = {target: 0.8}
    metrics["tree/support_precision"] = {target: 0.8}
    metrics["expansion/gain_rank_correlation"] = {target: 0.5}
    metrics["expansion/gain_pairwise_accuracy"] = {target: 0.8}
    metrics["expansion/gain_eligible_decision_fraction"] = {target: 0.5}
    metrics["expansion/gain_ordered_pair_count"] = {target: 10.0}
    metrics["expansion/gain_pair_coverage_fraction"] = {target: 0.5}

    gauge_axis = stage_gate._expected_axis(
        target,
        gate["training_every_updates"],
        min(gate["gauge_recent_window_updates"], target),
    )
    metrics["latent_gauge/min_ratio"] = {step: ratio for step in gauge_axis}
    metrics["latent_gauge/root/ratio"] = {target: ratio + 0.05}
    metrics["latent_gauge/future/ratio"] = {target: ratio + 0.02}
    metrics["latent_gauge/root/scale"] = {target: ratio + 0.05}
    metrics["latent_gauge/future/scale"] = {target: ratio + 0.02}
    metrics["latent_gauge/root/reference"] = {target: 1.0}
    metrics["latent_gauge/future/reference"] = {target: 1.0}
    metrics["latent_gauge/loss"] = {target: 0.01}
    metrics["latent_gauge/reference_sealed"] = {target: 1.0}
    metrics["latent_gauge/reference_update"] = {target: 0.0}

    gradient_axis = stage_gate._expected_axis(
        target,
        gate["training_every_updates"],
        min(gate["gradient_recent_window_updates"], target),
    )
    metrics["train/grad_norm_world"] = {step: 1.0 for step in gradient_axis}
    metrics["train/grad_norm_gain"] = {step: 1.0 for step in gradient_axis}
    metrics["train/grad_clip_coefficient_world"] = {step: clip for step in gradient_axis}
    metrics["train/grad_clip_coefficient_gain"] = {step: clip for step in gradient_axis}
    if gs:
        metrics["train/grad_norm_world_rest"] = {step: 1.0 for step in gradient_axis}
        metrics["train/grad_norm_branch_transformer"] = {step: 1.0 for step in gradient_axis}
        metrics["train/grad_clip_coefficient_world_rest"] = {
            step: clip for step in gradient_axis
        }
        metrics["train/grad_clip_coefficient_branch_transformer"] = {
            step: clip for step in gradient_axis
        }
    return metrics


def test_realistic_tracker_averages_use_inequality_not_false_equality(manifest):
    metrics = _healthy_metrics(manifest, ratio=0.88)
    metrics["latent_gauge/root/ratio"][5000] = 0.95
    metrics["latent_gauge/future/ratio"][5000] = 0.92
    health = stage_gate.evaluate_metrics(manifest, metrics, 5000, "G")
    assert health["integrity_gates"]["gauge_ratio_consistent"] is True
    assert health["candidate_passed"] is True

    impossible = copy.deepcopy(metrics)
    impossible["latent_gauge/min_ratio"][5000] = 0.94
    health = stage_gate.evaluate_metrics(manifest, impossible, 5000, "G")
    assert health["integrity_gates"]["gauge_ratio_consistent"] is False
    assert health["candidate_passed"] is False


def test_absolute_ratio_and_complete_recent_axis_are_fail_closed(manifest):
    low = _healthy_metrics(manifest, ratio=0.79)
    health = stage_gate.evaluate_metrics(manifest, low, 5000, "G")
    assert health["gauge_absolute_passed"] is False
    assert health["candidate_passed"] is False

    missing = _healthy_metrics(manifest)
    del missing["latent_gauge/min_ratio"][4500]
    health = stage_gate.evaluate_metrics(manifest, missing, 5000, "G")
    assert health["integrity_gates"]["complete_recent_gauge_axis"] is False
    assert health["candidate_passed"] is False


def test_gs_requires_explicit_split_gradient_and_clip_axes(manifest):
    metrics = _healthy_metrics(manifest, gs=False)
    g = stage_gate.evaluate_metrics(manifest, metrics, 5000, "G")
    assert g["candidate_passed"] is True
    gs = stage_gate.evaluate_metrics(manifest, metrics, 5000, "GS")
    assert gs["integrity_gates"]["complete_recent_gradient_axis"] is False
    assert gs["candidate_passed"] is False

    metrics = _healthy_metrics(manifest, gs=True)
    gs = stage_gate.evaluate_metrics(manifest, metrics, 5000, "GS")
    assert gs["candidate_passed"] is True
    metrics["train/grad_norm_branch_transformer"][5000] = 0.0
    gs = stage_gate.evaluate_metrics(manifest, metrics, 5000, "GS")
    assert gs["integrity_gates"]["nonzero_world_gain_and_required_split_gradients"] is False


def test_clipping_fraction_and_nonfinite_values_cannot_be_hidden(manifest):
    metrics = _healthy_metrics(manifest, gs=True)
    axis = sorted(metrics["train/grad_clip_coefficient_branch_transformer"])
    for step in axis[: len(axis) // 2]:
        metrics["train/grad_clip_coefficient_branch_transformer"][step] = 0.01
    health = stage_gate.evaluate_metrics(manifest, metrics, 5000, "GS")
    assert health["integrity_gates"]["bounded_gradient_clipping"] is False
    assert health["candidate_passed"] is False

    nonfinite = _healthy_metrics(manifest)
    nonfinite["train/grad_norm_world"][5000] = float("nan")
    health = stage_gate.evaluate_metrics(manifest, nonfinite, 5000, "G")
    assert health["integrity_gates"]["complete_recent_gradient_axis"] is False


def test_n_is_measured_but_never_candidate_eligible(manifest):
    health = stage_gate.evaluate_metrics(manifest, _healthy_metrics(manifest), 5000, "N")
    assert health["integrity_passed"] is True
    assert health["method_passed"] is True
    assert health["gauge_absolute_passed"] is True
    assert health["candidate_passed"] is False


@pytest.mark.parametrize("low_clip_samples", [33, 29, 36])
def test_frozen_like_n_clipping_and_collapse_are_allowed_causal_outcomes(
    manifest, low_clip_samples
):
    metrics = _healthy_metrics(manifest, ratio=0.30)
    axis = sorted(metrics["train/grad_clip_coefficient_world"])
    assert len(axis) == 100
    for tag in ("train/grad_clip_coefficient_world", "train/grad_clip_coefficient_gain"):
        for step in axis[:low_clip_samples]:
            metrics[tag][step] = 0.01
    metrics["control/q_advantage_over_z"][5000] = -0.5
    metrics["expansion/gain_rank_correlation"][5000] = -0.2
    health = stage_gate.evaluate_metrics(manifest, metrics, 5000, "N")
    assert health["clip_fraction_below_threshold"] == pytest.approx(low_clip_samples / 100)
    assert health["structural_integrity_passed"] is True
    assert health["integrity_passed"] is False
    assert health["method_passed"] is False
    assert health["gauge_absolute_passed"] is False
    assert health["candidate_passed"] is False


def test_corrupt_nonfinite_or_incomplete_n_is_structurally_rejected(manifest):
    nonfinite = _healthy_metrics(manifest, ratio=0.30)
    nonfinite["train/grad_clip_coefficient_world"][5000] = float("nan")
    health = stage_gate.evaluate_metrics(manifest, nonfinite, 5000, "N")
    assert health["structural_integrity_passed"] is False

    incomplete = _healthy_metrics(manifest, ratio=0.30)
    del incomplete["train/grad_norm_world"][2500]
    health = stage_gate.evaluate_metrics(manifest, incomplete, 5000, "N")
    assert health["structural_integrity_passed"] is False


@pytest.mark.parametrize("coefficient", [-0.1, 0.0, 1.1])
def test_n_invalid_clip_coefficient_range_is_structurally_rejected(
    manifest, coefficient
):
    metrics = _healthy_metrics(manifest, ratio=0.30)
    metrics["train/grad_clip_coefficient_world"][5000] = coefficient
    health = stage_gate.evaluate_metrics(manifest, metrics, 5000, "N")
    assert health["clip_coefficients_valid"] is False
    assert health["structural_integrity_gates"]["valid_gradient_clip_coefficients"] is False
    assert health["structural_integrity_passed"] is False


class _Child:
    def __init__(self):
        self.signals = []

    def poll(self):
        return None

    def send_signal(self, signum):
        self.signals.append(signum)


def test_remote_worker_signal_state_writes_latches_and_forwards_to_trainer(tmp_path):
    state = worker.SignalState(tmp_path)
    child = _Child()
    state.child = child
    state.request_requeue()
    assert state.requeue_requested is True
    assert child.signals == [signal.SIGUSR1]
    assert json.loads((tmp_path / "REQUEUE_REQUESTED.json").read_text())["status"] == "checkpoint_requested"

    state.request_cancel()
    assert state.cancel_requested is True
    assert child.signals[-1] == signal.SIGTERM
    assert json.loads((tmp_path / "CANCEL_REQUESTED").read_text())["status"] == "cancel_requested"


def test_latched_signal_before_child_creation_is_forwarded_immediately(tmp_path):
    state = worker.SignalState(tmp_path)
    state.request_requeue()
    child = _Child()
    state.child = child
    state.forward_latched_intent()
    assert child.signals == [signal.SIGUSR1]

    cancelled = worker.SignalState(tmp_path / "cancel")
    cancelled.request_cancel()
    cancel_child = _Child()
    cancelled.child = cancel_child
    cancelled.forward_latched_intent()
    assert cancel_child.signals == [signal.SIGTERM]


def test_cancel_precedence_blocks_requeue(tmp_path):
    state = worker.SignalState(tmp_path)
    state.request_cancel()
    state.request_requeue()
    assert state.cancel_requested is True
    assert state.requeue_requested is False
    assert not (tmp_path / "REQUEUE_REQUESTED.json").exists()
    assert worker.classify_child_exit(
        75,
        cancel_requested=True,
        cancel_latch_exists=True,
        requeue_requested=True,
    ) == "cancelled"
    assert worker.classify_child_exit(
        0,
        cancel_requested=True,
        cancel_latch_exists=True,
        requeue_requested=False,
    ) == "complete"


def _exact_rank_state(metric_tracker):
    return {
        "run_identity": {"world_size": 1},
        "rank_states": [{
            "rank": 0,
            "rng_state": {"python": object()},
            "loader": {
                "epoch": 0,
                "batches_yielded_in_epoch": 17,
                "epoch_generator_state": object(),
            },
            "rng_streams": {"planner": {}, "eval": {}, "viz": {}},
            "horizon_generator": object(),
            "metric_tracker": metric_tracker,
        }],
        "checkpoint_manager": {"best_success": None, "best_val_loss": None},
    }


def test_checkpoint_rank_state_requires_exact_metric_tracker_window():
    valid = {
        "schema_version": 1,
        "sums": {"latent_gauge/min_ratio": 7.25},
        "counts": {"latent_gauge/min_ratio": 8.0},
        "hists": {},
    }
    assert worker._rank_state_complete(_exact_rank_state(valid)) is True

    missing = _exact_rank_state(valid)
    del missing["rank_states"][0]["metric_tracker"]
    assert worker._rank_state_complete(missing) is False


@pytest.mark.parametrize(
    "malformed",
    [
        {"schema_version": 2, "sums": {}, "counts": {}, "hists": {}},
        {"schema_version": 1, "sums": {"x": 1.0}, "counts": {}, "hists": {}},
        {"schema_version": 1, "sums": {"x": float("nan")}, "counts": {"x": 1.0}, "hists": {}},
        {"schema_version": 1, "sums": {"x": 1.0}, "counts": {"x": -1.0}, "hists": {}},
        {"schema_version": 1, "sums": {}, "counts": {}, "hists": {"x": [float("inf")]}},
    ],
)
def test_checkpoint_rank_state_rejects_malformed_metric_tracker(malformed):
    assert worker._rank_state_complete(_exact_rank_state(malformed)) is False


def _metric_boundary_checkpoint(path, *, step, tracker_state):
    import torch

    torch.save({
        "completed_updates": step,
        "reason": "graceful-stop:SIGUSR1",
        "run_identity": {"world_size": 1},
        "rank_states": [{"rank": 0, "metric_tracker": tracker_state}],
    }, path)


def test_exact_metric_tracker_boundary_is_published_reset_and_idempotent(tmp_path):
    import torch
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    from treewm.logging.metrics import MetricTracker

    run_dir = tmp_path / "run"
    checkpoint_path = run_dir / "checkpoints" / "latest.pt"
    checkpoint_path.parent.mkdir(parents=True)
    tracker = MetricTracker()
    tracker.add("train/grad_norm_world", 2.0, count=50)
    tracker.add_hist("train/example_hist", [1.0, 2.0, 3.0])
    _metric_boundary_checkpoint(checkpoint_path, step=50, tracker_state=tracker.state_dict())
    before = campaign.file_sha256(checkpoint_path)
    launch = {
        "campaign_id": "treewm-grounded-gauge-pilot-v2",
        "run": {"run_name": "gauge-v2-test"},
        "launch_sha256": "a" * 64,
        "metric_boundary_required_tags": ["train/grad_norm_world"],
    }

    recovered = metric_boundary.recover_metric_boundary(
        checkpoint_path, run_dir, launch
    )
    assert recovered["status"] == "metric_boundary_recovered"
    assert recovered["checkpoint_before_sha256"] == before
    assert recovered["checkpoint_after_sha256"] != before
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["rank_states"][0]["metric_tracker"] == {
        "schema_version": 1,
        "sums": {},
        "counts": {},
        "hists": {},
    }
    event_path = next(run_dir.glob("events.out.tfevents.*"))
    accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    accumulator.Reload()
    points = accumulator.Scalars("train/grad_norm_world")
    assert [(point.step, point.value) for point in points] == [(50, pytest.approx(2.0))]

    again = metric_boundary.recover_metric_boundary(checkpoint_path, run_dir, launch)
    assert again["status"] == "metric_boundary_previously_recovered"
    assert len(list(run_dir.glob("events.out.tfevents.*"))) == 1


def test_partial_metric_tracker_window_is_preserved_for_normal_resume(tmp_path):
    from treewm.logging.metrics import MetricTracker

    run_dir = tmp_path / "run"
    checkpoint_path = run_dir / "checkpoints" / "latest.pt"
    checkpoint_path.parent.mkdir(parents=True)
    tracker = MetricTracker()
    tracker.add("train/grad_norm_world", 2.0, count=37)
    _metric_boundary_checkpoint(checkpoint_path, step=37, tracker_state=tracker.state_dict())
    before = campaign.file_sha256(checkpoint_path)
    launch = {
        "campaign_id": "treewm-grounded-gauge-pilot-v2",
        "run": {"run_name": "gauge-v2-test"},
        "launch_sha256": "b" * 64,
        "metric_boundary_required_tags": ["train/grad_norm_world"],
    }
    result = metric_boundary.recover_metric_boundary(checkpoint_path, run_dir, launch)
    assert result["status"] == "no_metric_boundary_recovery_needed"
    assert campaign.file_sha256(checkpoint_path) == before
    assert not list(run_dir.glob("events.out.tfevents.*"))


def test_batch_never_signals_local_srun_and_requires_cancelled_artifact(manifest):
    train = (CAMPAIGN_DIR / "train.slurm").read_text(encoding="utf-8")
    assert 'kill -TERM "$step_pid"' not in train
    assert 'kill -USR1 "$step_pid"' not in train
    assert "touch \"$CANCEL_LATCH\"" in train
    assert "touch \"$REQUEUE_LATCH\"" in train
    assert "durable remote-worker CANCELLED.json" in train
    assert train.count('"$SRUN" --ntasks=1') == 2
    assert train.index('if [[ "$status" -eq 0 ]]') < train.index(
        'if [[ -e "$CANCEL_LATCH" || "$status" -eq 143 ]]'
    )
    submit.validate_slurms(CAMPAIGN_DIR)


def _fake_launch(run):
    hashes = {
        "source_sha256": "1" * 64,
        "runtime_sha256": "2" * 64,
        "package_protocol_sha256": "3" * 64,
        "actual_evaluation_bank_sha256": "4" * 64,
        "validation_manifest_sha256": f"{run.setting_index + 10:064x}",
        "evaluation_seed_tables_sha256": f"{run.index + 100:064x}",
        "final_seed_table_sha256": "5" * 64,
        "config_sha256": f"{run.index + 200:064x}",
    }
    launch = {
        "campaign_id": "treewm-grounded-gauge-pilot-v2",
        "formal_validation": False,
        "run": {
            "index": run.index,
            "run_name": run.run_name,
            "setting_id": run.setting_id,
            "arm_id": run.arm_id,
            "seed": run.seed,
            "run_directory": f"/tmp/{run.run_name}",
        },
        "hashes": hashes,
    }
    launch["launch_sha256"] = f"{run.index + 1:064x}"
    return launch


def _mock_stage5(
    monkeypatch,
    manifest,
    *,
    g_pass=True,
    gs_pass=True,
    causal=True,
    n_structural=True,
):
    monkeypatch.setattr(
        stage_gate,
        "trainer_command",
        lambda _manifest, run, repo_root: _fake_launch(run),
    )
    monkeypatch.setattr(
        stage_gate,
        "validate_stage_complete",
        lambda _path, launch, _target, _slot: {
            "identity_sha256": f"{int(launch['run']['index']) + 300:064x}",
            "checkpoint_sha256": f"{int(launch['run']['index']) + 400:064x}",
        },
    )
    monkeypatch.setattr(
        stage_gate,
        "verify_stage_marker",
        lambda _path, _target, launch: {
            "checkpoint_sha256": f"{int(launch['run']['index']) + 400:064x}",
        },
    )
    monkeypatch.setattr(stage_gate, "event_scalars", lambda _path: {})

    def health(_manifest, _metrics, _target, arm_id):
        passed = {"N": False, "G": g_pass, "GS": gs_pass}[arm_id]
        ratio = 0.5 if arm_id == "N" else (0.9 if causal else 0.5)
        return {
            "structural_integrity_passed": n_structural if arm_id == "N" else True,
            "integrity_passed": arm_id != "N",
            "method_passed": arm_id != "N",
            "gauge_absolute_passed": arm_id != "N",
            "candidate_passed": passed,
            "recent_gauge_min_ratio": ratio,
            "last": {"data/validation_fixed_sample_count": 5120.0},
        }

    monkeypatch.setattr(stage_gate, "evaluate_metrics", health)


def test_5k_selection_is_universal_causal_and_g_before_gs(monkeypatch, manifest):
    _mock_stage5(monkeypatch, manifest)
    gate = stage_gate._stage_5000_gate(manifest, REPO_ROOT)
    assert gate["selected_arm"] == "G"
    assert len(gate["runs"]) == 30
    assert gate["candidate_summary"]["G"]["universal_cells_passed"] == 10
    assert gate["candidate_summary"]["G"]["causal_scale_retention_passed"] is True
    assert gate["nonpromotable_arm"] == "N"


def test_5k_allows_n_candidate_failures_but_rejects_structural_n_corruption(
    monkeypatch, manifest
):
    _mock_stage5(monkeypatch, manifest, n_structural=True)
    gate = stage_gate._stage_5000_gate(manifest, REPO_ROOT)
    assert gate["selected_arm"] == "G"
    assert all(
        not row["health"]["integrity_passed"]
        and not row["health"]["method_passed"]
        and row["health"]["structural_integrity_passed"]
        for row in gate["runs"]
        if row["arm_id"] == "N"
    )

    _mock_stage5(monkeypatch, manifest, n_structural=False)
    with pytest.raises(campaign.ContractError, match="30/30 exact rows"):
        stage_gate._stage_5000_gate(manifest, REPO_ROOT)


def test_5k_falls_back_to_gs_only_under_same_universal_contract(monkeypatch, manifest):
    _mock_stage5(monkeypatch, manifest, g_pass=False, gs_pass=True)
    gate = stage_gate._stage_5000_gate(manifest, REPO_ROOT)
    assert gate["selected_arm"] == "GS"
    assert gate["candidate_summary"]["G"]["eligible"] is False
    assert gate["candidate_summary"]["GS"]["universal_cells_passed"] == 10


def test_5k_rejects_no_causal_delta_or_any_candidate_failure(monkeypatch, manifest):
    _mock_stage5(monkeypatch, manifest, causal=False)
    with pytest.raises(campaign.ContractError, match="neither G nor GS"):
        stage_gate._stage_5000_gate(manifest, REPO_ROOT)

    _mock_stage5(monkeypatch, manifest, g_pass=False, gs_pass=False, causal=True)
    with pytest.raises(campaign.ContractError, match="neither G nor GS"):
        stage_gate._stage_5000_gate(manifest, REPO_ROOT)


def _outcome_rows():
    rows = []
    for setting in campaign.SETTING_IDS:
        for seed in campaign.SEEDS:
            rows.append({
                "setting_id": setting,
                "seed": seed,
                "outcome": {
                    "num_episodes": 5.0,
                    "successes": 1.0 if setting == "antmaze-large" else 0.0,
                    "success_rate": 0.2 if setting == "antmaze-large" else 0.0,
                    "distance_reduction_frac": 0.1 if setting in campaign.SETTING_IDS[:3] else -0.01,
                },
            })
    return rows


def test_25k_outcome_requires_replication_by_seed_and_setting(manifest):
    result = stage_gate._outcome_summary(manifest, _outcome_rows())
    assert result["per_seed"]["108"]["successes"] == 1.0
    assert result["per_seed"]["109"]["successes"] == 1.0
    assert result["settings_with_both_seed_nonzero_success"] == 1
    assert result["settings_with_both_seed_positive_progress"] == 3

    broken = _outcome_rows()
    for row in broken:
        if row["seed"] == 109:
            row["outcome"]["successes"] = 0.0
            row["outcome"]["success_rate"] = 0.0
    with pytest.raises(campaign.ContractError, match="seed 109"):
        stage_gate._outcome_summary(manifest, broken)


def test_skip_artifact_is_hash_bound_and_idempotent(monkeypatch, tmp_path):
    run = type("Run", (), {
        "index": 2,
        "setting_index": 0,
        "setting_id": "antmaze-large",
        "arm_id": "G",
        "seed": 108,
        "run_name": "gauge-v2-antmaze-large-armg-seed108",
    })()
    launch = _fake_launch(run)
    launch["run"]["run_directory"] = str(tmp_path / "run")
    checkpoint = {
        "identity_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "completed_updates": 5000,
    }
    monkeypatch.setattr(worker, "verify_checkpoint", lambda *_args, **_kwargs: checkpoint)
    prior = {
        "identity_sha256": "a" * 64,
        "checkpoint_sha256": "b" * 64,
        "selected_arm": "GS",
        "gate_sha256": "c" * 64,
    }
    state_dir = tmp_path / "state"
    run_dir = tmp_path / "run"
    first = worker._write_selection_skip(state_dir, run_dir, launch, prior, 0)
    second = worker._write_selection_skip(state_dir, run_dir, launch, prior, 0)
    assert first == second
    assert first["trainer_launched"] is False
    assert (state_dir / "SKIPPED_BY_SELECTION.json").is_file()
    assert (run_dir / "stage-gates" / "SKIPPED_BY_SELECTION_25000.json").is_file()

    tampered = copy.deepcopy(first)
    tampered["selected_arm"] = "G"
    path = tmp_path / "tampered.json"
    campaign.atomic_json(path, tampered)
    gate = {
        "selected_arm": "GS",
        "gate_sha256": "c" * 64,
        "runs": [{
            "index": 2,
            "identity_sha256": "a" * 64,
            "checkpoint_sha256": "b" * 64,
        }],
    }
    with pytest.raises(campaign.ContractError, match="hash differs"):
        stage_gate.validate_skip(path, launch, 0, gate)


def test_previous_gate_is_immutable_and_selects_per_launch(monkeypatch, tmp_path, manifest):
    changed = copy.deepcopy(manifest)
    changed["paths"]["run_root"] = str(tmp_path / "treewm-grounded-gauge-pilot-v2")
    run = campaign.continuation_runs(changed)[0]
    launch = _fake_launch(run)
    rows = []
    for row_run in campaign.expand_runs(changed):
        row_launch = _fake_launch(row_run)
        rows.append({
            "index": row_run.index,
            "arm_id": row_run.arm_id,
            "launch_sha256": row_launch["launch_sha256"],
            "identity_sha256": f"{row_run.index + 500:064x}",
            "checkpoint_sha256": f"{row_run.index + 600:064x}",
        })
    value = {
        "schema_version": 1,
        "status": "accepted_for_selected_continuation",
        "stage_target": 5000,
        "selected_arm": "G",
        "package_protocol_sha256": "3" * 64,
        "source_sha256": "1" * 64,
        "runtime_sha256": "2" * 64,
        "runs": rows,
    }
    value["gate_sha256"] = campaign.stable_hash(value)
    path = Path(changed["paths"]["run_root"]) / "state" / "stage-gates" / "STAGE_GATE_5000.json"
    campaign.atomic_json(path, value)
    previous = worker.verify_previous_gate(changed, 25000, launch)
    assert previous["selected"] is True
    assert previous["selected_arm"] == "G"

    value["selected_arm"] = "N"
    value["gate_sha256"] = campaign.stable_hash({k: v for k, v in value.items() if k != "gate_sha256"})
    campaign.atomic_json(path, value)
    with pytest.raises(campaign.ContractError, match="ineligible"):
        worker.verify_previous_gate(changed, 25000, launch)


def test_launch_plan_has_exact_staged_dag(monkeypatch, manifest):
    monkeypatch.setattr(submit, "validate_slurms", lambda *_args: None)
    monkeypatch.setattr(submit, "verify_all", lambda *_args, **_kwargs: {"status": "verified"})
    monkeypatch.setattr(
        submit,
        "trainer_command",
        lambda _manifest, run, repo_root: _fake_launch(run),
    )
    plan = submit.launch_plan(
        manifest,
        REPO_ROOT,
        verify_files=False,
        inspect_scheduler=False,
    )
    assert len(plan["stage_5000_runs"]) == 30
    assert len(plan["stage_25000_slots"]) == 20
    assert [node["name"] for node in plan["dag"]] == [
        "train_5000",
        "gate_5000",
        "train_25000",
        "gate_25000",
    ]
    assert plan["dag"][0]["array"] == "0-29%30"
    assert plan["dag"][2]["array"] == "0-19%20"
    assert all(row["arm_id"] != "N" for row in plan["stage_25000_slots"])


def test_protocol_lock_slurms_and_namespace_are_isolated(manifest, tmp_path):
    assert campaign.verify_protocol_lock(CAMPAIGN_DIR) == (
        CAMPAIGN_DIR / "protocol.sha256"
    ).read_text(encoding="utf-8").strip()
    submit.validate_slurms(CAMPAIGN_DIR)
    assert manifest["paths"]["run_root"].endswith("/outputs/treewm-grounded-gauge-pilot-v2")
    changed = copy.deepcopy(manifest)
    changed["paths"]["run_root"] = str(tmp_path / "fresh")
    assert submit.namespace_is_fresh(changed) is True
    launch = tmp_path / "fresh" / "x" / "GAUGE_PILOT_V2_LAUNCH.json"
    launch.parent.mkdir(parents=True)
    launch.write_text("{}")
    assert submit.namespace_is_fresh(changed) is False


def test_publish_gate_is_immutable(tmp_path):
    path = tmp_path / "gate.json"
    value = {"schema_version": 1, "status": "accepted", "gate_sha256": "a" * 64}
    stage_gate.publish_gate(path, value)
    stage_gate.publish_gate(path, value)
    with pytest.raises(campaign.ContractError, match="already differs"):
        stage_gate.publish_gate(path, {**value, "status": "changed"})
