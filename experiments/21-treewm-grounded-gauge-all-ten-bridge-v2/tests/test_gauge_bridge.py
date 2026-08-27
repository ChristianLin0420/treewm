from __future__ import annotations

import copy
import importlib
from pathlib import Path
import signal
import sys

from hydra import compose, initialize_config_dir
import pytest

PACKAGE = Path(__file__).resolve().parents[1]


def _load_local_modules():
    names = ("campaign", "worker", "stage_gate", "submit", "train_entry")
    previous = {name: sys.modules.get(name) for name in names}
    for name in names:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(PACKAGE))
    try:
        modules = tuple(importlib.import_module(name) for name in names)
        return modules
    finally:
        sys.path.remove(str(PACKAGE))
        for name in names:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]


campaign, worker, stage_gate, submit, train_entry = _load_local_modules()


@pytest.fixture(scope="module")
def manifest() -> dict:
    return campaign.load_manifest()


def fake_binding(manifest: dict, arm: str = "G") -> dict:
    recipe = campaign.selected_recipe(manifest, arm)
    binding = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "status": "sealed_exp20_acceptance",
        "launch_allowed": True,
        "selection_policy": "consume_recomputed_exp20_G_then_GS_without_bridge_recipe_selection",
        "exp20": {},
        "selected_arm": arm,
        "selected_recipe": recipe,
        "selected_recipe_sha256": campaign.stable_hash(recipe),
    }
    binding["binding_sha256"] = campaign.stable_hash(binding)
    return binding


def fake_contract() -> dict:
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


def test_fresh_all_ten_mapping_and_unique_identities(manifest: dict) -> None:
    runs = campaign.expand_runs(manifest)
    assert len(runs) == 20
    assert [run.index for run in runs] == list(range(20))
    assert {run.seed for run in runs} == {106, 107}
    assert {run.setting_id for run in runs} == set(campaign.SETTING_IDS)
    assert len({run.run_name for run in runs}) == 20
    assert len({run.wandb_id for run in runs}) == 20
    assert all(run.run_name.startswith("gaugebridge-v2-") for run in runs)
    assert all(run.index == run.setting_index * 2 + run.seed_index for run in runs)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["design"].update(seeds=[104, 105]),
        lambda value: value["method"].update(objective_version="treewm_v2_grounded_gauge_pilot_v2"),
        lambda value: value["scientific_contract"].update(optimizer_updates=1_000_000),
        lambda value: value["scientific_contract"].update(scheduled_sampling_granularity="batch"),
        lambda value: value["stage_acceptance"].update(required_method_runs=19),
        lambda value: value["stage_acceptance"].update(min_settings_with_both_seed_positive_progress=5),
        lambda value: value["stage_acceptance"].update(min_clip_coefficient=0.0),
        lambda value: value["logging"].update(wandb_project="treewm"),
        lambda value: value["prerequisite"].update(forbidden_ancestry_tokens=["exp15", "exp16"]),
    ],
)
def test_manifest_rejects_scientific_or_identity_drift(manifest: dict, mutation) -> None:
    changed = copy.deepcopy(manifest)
    mutation(changed)
    with pytest.raises(campaign.ContractError):
        campaign.validate_manifest(changed)


def test_selected_g_or_gs_changes_only_registered_clipping_switch(manifest: dict) -> None:
    run = campaign.expand_runs(manifest)[0]
    g = set(campaign.scientific_overrides(manifest, run, fake_contract(), fake_binding(manifest, "G")))
    gs = set(campaign.scientific_overrides(manifest, run, fake_contract(), fake_binding(manifest, "GS")))
    recipe_g = {value for value in g if not value.startswith("+campaign_")}
    recipe_gs = {value for value in gs if not value.startswith("+campaign_")}
    assert recipe_g.symmetric_difference(recipe_gs) == {
        "train.separate_branch_transformer_grad_clip=false",
        "train.separate_branch_transformer_grad_clip=true",
    }
    assert "+campaign_factorial_arm=exp20-G-all-ten" in g
    assert "+campaign_factorial_arm=exp20-GS-all-ten" in gs
    for invariant in (
        "objective_version=treewm_v2_grounded_gauge_all_ten_bridge_v2",
        "train.steps=25000",
        "train.scheduler_total_steps=1000000",
        "future_sets.recipe_anchor_policy=published_union",
        "losses.keep_balance=true",
        "losses.scheduled_sampling_granularity=sequence",
        "losses.multistep_transition_mode=grounded_execution_v2",
        "losses.enabled.latent_gauge=true",
        "losses.weights.latent_gauge=1.0",
        "planner.decoded_metric=domain_raw",
        "planner.execute_steps=4",
        "planner.require_first_edge_improvement=true",
    ):
        assert invariant in g and invariant in gs


def test_direct_hydra_composition_is_bounded_but_unsealed(manifest: dict) -> None:
    with initialize_config_dir(config_dir=str(campaign.REPOSITORY_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="base", overrides=["experiment=treewm_v2_grounded_gauge_all_ten_bridge_v2"])
    assert cfg.objective_version == "treewm_v2_grounded_gauge_all_ten_bridge_v2"
    assert int(cfg.train.steps) == 25_000
    assert int(cfg.train.scheduler_total_steps) == 1_000_000
    assert cfg.losses.keep_balance is True
    assert cfg.losses.scheduled_sampling_granularity == "sequence"
    assert cfg.losses.multistep_transition_mode == "grounded_execution_v2"
    assert cfg.get("campaign_prerequisite_binding_sha256") is None
    assert cfg.get("campaign_selected_recipe_sha256") is None
    with pytest.raises(campaign.ContractError):
        train_entry.verify_exact_invocation()


def test_train_entry_registers_exp21_for_strict_cadence_checkpoints() -> None:
    from scripts import train
    from treewm.utils import checkpoint as checkpoint_utils

    original_v2 = train.TREEWM_V2_OBJECTIVES
    original_bounded = train.BOUNDED_PILOT_OBJECTIVES
    original_gauge = train.LATENT_GAUGE_OBJECTIVES
    original_cadence = checkpoint_utils.OBJECTIVES_REQUIRING_POST_UPDATE_CADENCE
    try:
        registered = train_entry.register_objective()
        assert registered is train
        assert train_entry.OBJECTIVE in train.TREEWM_V2_OBJECTIVES
        assert train.BOUNDED_PILOT_OBJECTIVES[train_entry.OBJECTIVE] == 25_000
        assert train_entry.OBJECTIVE in train.LATENT_GAUGE_OBJECTIVES
        assert (
            train_entry.OBJECTIVE
            in checkpoint_utils.OBJECTIVES_REQUIRING_POST_UPDATE_CADENCE
        )
    finally:
        train.TREEWM_V2_OBJECTIVES = original_v2
        train.BOUNDED_PILOT_OBJECTIVES = original_bounded
        train.LATENT_GAUGE_OBJECTIVES = original_gauge
        checkpoint_utils.OBJECTIVES_REQUIRING_POST_UPDATE_CADENCE = original_cadence


def test_binding_state_and_static_test_are_fail_closed(manifest: dict) -> None:
    raw = campaign.read_json(campaign.BINDING_PATH)
    result = submit.static_test(manifest, campaign.REPOSITORY_ROOT)
    if raw["status"] == "unsealed_waiting_for_exp20_acceptance":
        with pytest.raises(campaign.ContractError, match="not sealed"):
            campaign.load_exp20_binding(manifest, verify_external_files=False)
        assert result["status"] == "static_package_verified_blocked_on_exp20"
        assert result["launch_allowed"] is False
    else:
        assert campaign.load_exp20_binding(manifest, verify_external_files=False)["launch_allowed"] is True
        assert result["status"] == "static_package_verified"
        assert result["launch_allowed"] is True
    assert result["jobs_submitted"] == 0
    assert result["snapshot_created"] is False


def test_exact_monitor_bank_is_five_prospective_episodes(manifest: dict) -> None:
    bank = campaign.actual_evaluation_bank(manifest)
    assert bank["policy"] == "prospective_monitor_25000_fixed_cfg_eval_seed_fallback"
    assert bank["episodes_per_task"] == 1
    assert bank["task_ids"] == [1, 2, 3, 4, 5]
    assert bank["seeds"] == [[2718], [3718], [4718], [5718], [6718]]
    assert bank["sha256"] == campaign.stable_hash({key: value for key, value in bank.items() if key != "sha256"})


def healthy_stage_metrics(manifest: dict, arm: str = "G") -> dict[str, dict[int, float]]:
    target = campaign.STAGE_TARGET
    gate = manifest["stage_acceptance"]
    validation_axis = range(1_000, target + 1, 1_000)
    training_axis = range(50, target + 1, 50)
    metrics = {
        tag: {
            step: 1.0
            for step in (training_axis if tag in gate["training_exact_target_tags"] else validation_axis)
        }
        for tag in gate["required_finite_tags"]
    }
    metrics["data/validation_fixed_sample_count"] = {step: 5120.0 for step in validation_axis}
    metrics["val/loss_total"] = {step: 1.0 for step in validation_axis}
    metrics["val/loss_multistep_self_fed"] = {step: 1.0 for step in validation_axis}
    metrics["val/loss_horizon"] = {step: 1.0 for step in validation_axis}
    for horizon in (4, 8, 16, 32, 64):
        metrics[f"data/validation_horizon_label_fraction_h{horizon}"] = {step: 0.2 for step in validation_axis}
    metrics["control/retrieval_uses_task_metric_endpoint"] = {step: 1.0 for step in validation_axis}
    metrics["control/q_advantage_over_z"] = {step: 0.2 for step in validation_axis}
    metrics["control/q_advantage_over_random_proj"] = {step: 0.1 for step in validation_axis}
    metrics["tree/support_recall"] = {step: 0.8 for step in validation_axis}
    metrics["tree/support_precision"] = {step: 0.8 for step in validation_axis}
    metrics["expansion/gain_rank_correlation"] = {step: 0.5 for step in validation_axis}
    metrics["expansion/gain_pairwise_accuracy"] = {step: 0.8 for step in validation_axis}
    metrics["expansion/gain_eligible_decision_fraction"] = {step: 0.5 for step in validation_axis}
    metrics["expansion/gain_ordered_pair_count"] = {step: 10.0 for step in validation_axis}
    metrics["expansion/gain_pair_coverage_fraction"] = {step: 0.5 for step in validation_axis}
    for tag in ("latent_gauge/min_ratio", "latent_gauge/root/ratio", "latent_gauge/future/ratio"):
        metrics[tag] = {step: 1.0 for step in training_axis}
    metrics["latent_gauge/root/reference"] = {step: 1.0 for step in training_axis}
    metrics["latent_gauge/future/reference"] = {step: 1.0 for step in training_axis}
    metrics["latent_gauge/reference_sealed"] = {step: 1.0 for step in training_axis}
    metrics["latent_gauge/reference_update"] = {step: 0.0 for step in training_axis}
    for tag in ("train/grad_norm_world", "train/grad_norm_gain"):
        metrics[tag] = {step: 1.0 for step in training_axis}
    for tag in ("train/grad_clip_coefficient_world", "train/grad_clip_coefficient_gain"):
        metrics[tag] = {step: 0.5 for step in training_axis}
    if arm == "GS":
        metrics["train/grad_norm_world_rest"] = {step: 1.0 for step in training_axis}
        metrics["train/grad_norm_branch_transformer"] = {step: 1.0 for step in training_axis}
        metrics["train/grad_clip_coefficient_world_rest"] = {step: 0.5 for step in training_axis}
        metrics["train/grad_clip_coefficient_branch_transformer"] = {step: 0.5 for step in training_axis}
    return metrics


def test_25k_method_freshness_and_clipping_gates_are_fail_closed(manifest: dict) -> None:
    healthy = stage_gate.evaluate_metrics(manifest, healthy_stage_metrics(manifest), "G")
    assert healthy["candidate_passed"] is True

    stale = healthy_stage_metrics(manifest)
    stale["latent_gauge/min_ratio"].pop(campaign.STAGE_TARGET)
    report = stage_gate.evaluate_metrics(manifest, stale, "G")
    assert report["integrity_gates"]["target_appropriate_telemetry"] is False
    assert report["integrity_gates"]["complete_recent_gauge_axis"] is False

    forged_method = healthy_stage_metrics(manifest)
    forged_method["control/q_advantage_over_z"][campaign.STAGE_TARGET] = -1.0
    report = stage_gate.evaluate_metrics(manifest, forged_method, "G")
    assert report["method_gates"]["q_advantage"] is False
    assert report["candidate_passed"] is False

    saturated = healthy_stage_metrics(manifest, "GS")
    axis = [
        step for step in sorted(saturated["train/grad_clip_coefficient_branch_transformer"])
        if step >= campaign.STAGE_TARGET - manifest["stage_acceptance"]["gradient_recent_window_updates"]
    ]
    for step in axis[: len(axis) // 2]:
        saturated["train/grad_clip_coefficient_branch_transformer"][step] = 0.01
    report = stage_gate.evaluate_metrics(manifest, saturated, "GS")
    assert report["integrity_gates"]["bounded_gradient_clipping"] is False
    assert report["candidate_passed"] is False


def test_outcome_quorum_is_exactly_requested(manifest: dict) -> None:
    rows = []
    for setting_index, setting in enumerate(campaign.SETTING_IDS):
        for seed in campaign.SEEDS:
            positive = setting_index < 6
            success = setting_index == 0
            task_rows = [
                {
                    "task_id": task_id,
                    "episode_seed": campaign.actual_evaluation_bank(manifest)["seeds"][task_index][0],
                    "num_episodes": 1.0,
                    "successes": 1.0 if success and task_index == 0 else 0.0,
                    "success_rate": 1.0 if success and task_index == 0 else 0.0,
                }
                for task_index, task_id in enumerate(campaign.TASK_IDS)
            ]
            successes = sum(task["successes"] for task in task_rows)
            rows.append({
                "setting_id": setting,
                "seed": seed,
                "outcome": {
                    "num_episodes": 5.0,
                    "successes": successes,
                    "success_rate": successes / 5.0,
                    "distance_reduction_frac": 0.1 if positive else -0.01,
                    "prospective_monitor_bank_sha256": campaign.actual_evaluation_bank(manifest)["sha256"],
                    "tasks": task_rows,
                },
            })
    result = stage_gate.outcome_summary(manifest, rows)
    assert result["settings_with_both_seed_positive_progress"] == 6
    assert result["settings_with_both_seed_nonzero_success"] == 1
    assert all(value["successes"] >= 1 for value in result["per_seed"].values())
    assert all(value["mean_distance_reduction_frac"] > 0 for value in result["per_seed"].values())

    broken = copy.deepcopy(rows)
    for row in broken:
        if row["setting_id"] == campaign.SETTING_IDS[5]:
            row["outcome"]["distance_reduction_frac"] = -0.01
    with pytest.raises(campaign.ContractError, match="6/10"):
        stage_gate.outcome_summary(manifest, broken)


class Child:
    def __init__(self) -> None:
        self.signals: list[int] = []

    def poll(self):
        return None

    def send_signal(self, signum: int) -> None:
        self.signals.append(signum)


def test_worker_signal_and_exit_precedence(tmp_path: Path) -> None:
    state = worker.SignalState(tmp_path)
    child = Child()
    state.child = child
    state.request_requeue()
    assert child.signals == [signal.SIGUSR1]
    state.request_cancel()
    assert child.signals[-1] == signal.SIGTERM
    assert worker.classify_child_exit(75, cancel_requested=True, cancel_latch_exists=True, requeue_requested=True) == "cancelled"
    assert worker.classify_child_exit(0, cancel_requested=True, cancel_latch_exists=True, requeue_requested=True) == "complete"

    window = worker.SignalState(tmp_path / "window")
    window.state_dir.mkdir()
    window.cancel_requested = True
    delayed_child = Child()
    window.child = delayed_child
    window.forward_latched_intent()
    assert delayed_child.signals == [signal.SIGTERM]


def test_exact_stage_target_requeue_is_valid_but_past_target_rejects() -> None:
    worker.validate_requeue_checkpoint({
        "completed_updates": campaign.STAGE_TARGET,
        "reason": "graceful-stop:SIGUSR1",
    })
    with pytest.raises(campaign.ContractError, match="passed boundary"):
        worker.validate_requeue_checkpoint({
            "completed_updates": campaign.STAGE_TARGET + 1,
            "reason": "graceful-stop:SIGUSR1",
        })
    with pytest.raises(campaign.ContractError, match="graceful"):
        worker.validate_requeue_checkpoint({
            "completed_updates": campaign.STAGE_TARGET,
            "reason": "periodic",
        })


def test_exp21_checkpoint_cadence_is_mandatory_and_durable() -> None:
    complete = {
        "post_update_cadence": {
            "schema_version": 1,
            "committed_update": 25_000,
            "completed_update": 25_000,
            "replay_action": None,
        }
    }
    assert worker.validate_post_update_cadence(complete, 25_000) == complete[
        "post_update_cadence"
    ]
    worker.validate_complete_post_update_cadence(
        complete["post_update_cadence"],
        25_000,
    )

    replayable = {
        "post_update_cadence": {
            "schema_version": 1,
            "committed_update": 12_500,
            "completed_update": 12_499,
            "replay_action": "evaluation",
        }
    }
    assert worker.validate_post_update_cadence(replayable, 12_500) == replayable[
        "post_update_cadence"
    ]
    with pytest.raises(campaign.ContractError, match="incomplete post-update cadence"):
        worker.validate_complete_post_update_cadence(
            replayable["post_update_cadence"],
            12_500,
        )

    with pytest.raises(campaign.ContractError, match="lacks post_update_cadence"):
        worker.validate_post_update_cadence({}, 25_000)
    incomplete = copy.deepcopy(replayable)
    incomplete["post_update_cadence"]["replay_action"] = None
    with pytest.raises(campaign.ContractError, match="cadence is invalid"):
        worker.validate_post_update_cadence(incomplete, 12_500)


def test_slurm_and_protocol_inventory_are_isolated() -> None:
    submit.validate_slurms(PACKAGE)
    assert len(campaign.PROTOCOL_FILES) == len(set(campaign.PROTOCOL_FILES))
    assert {"train_entry.py", "bind_exp20.py", "train.slurm", "gate.slurm"}.issubset(campaign.PROTOCOL_FILES)
    assert "metric_boundary.py" not in campaign.PROTOCOL_FILES
    train = (PACKAGE / "train.slurm").read_text(encoding="utf-8")
    assert train.index('if [[ "$status" -eq 0 ]]') < train.index('if [[ -e "$CANCEL_LATCH" || "$status" -eq 143 ]]')
    manifest = campaign.load_manifest()
    assert set(manifest["prerequisite"]) == {
        "bindings_file", "campaign_id", "stage_5000_gate_path", "acceptance_path",
        "required_status", "allowed_selected_arms", "selection_precedence",
        "manifest_sha256", "package_protocol_sha256", "source_sha256",
        "runtime_sha256", "actual_evaluation_bank_sha256",
        "forbidden_ancestry_tokens", "raw_recomputation",
    }
