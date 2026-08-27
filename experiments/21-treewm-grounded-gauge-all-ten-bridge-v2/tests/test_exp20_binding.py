from __future__ import annotations

import copy
import importlib
from pathlib import Path
import statistics
import sys

import pytest

PACKAGE = Path(__file__).resolve().parents[1]


def _load_local_modules():
    names = ("campaign", "bind_exp20")
    previous = {name: sys.modules.get(name) for name in names}
    for name in names:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(PACKAGE))
    try:
        campaign_module = importlib.import_module("campaign")
        binding_module = importlib.import_module("bind_exp20")
        return campaign_module, binding_module
    finally:
        sys.path.remove(str(PACKAGE))
        for name in names:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]


campaign, bind_exp20 = _load_local_modules()


EXP20_MANIFEST = campaign.read_json(
    campaign.REPOSITORY_ROOT / "experiments/20-treewm-grounded-gauge-pilot-v2/manifest.json"
)
MANIFEST = campaign.load_manifest()
CONTRACT = MANIFEST["prerequisite"]
RAW = CONTRACT["raw_recomputation"]


def healthy_metrics(*, arm: str, ratio: float, target: int = 25_000) -> dict[str, dict[int, float]]:
    gate = EXP20_MANIFEST["stage_acceptance"]
    validation_axis = range(1_000, target + 1, 1_000)
    training_axis = range(50, target + 1, 50)
    metrics: dict[str, dict[int, float]] = {}
    for tag in gate["required_finite_tags"]:
        axis = training_axis if tag in gate["training_exact_target_tags"] else validation_axis
        metrics[tag] = {step: 1.0 for step in axis}
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
    metrics["latent_gauge/min_ratio"] = {step: ratio for step in training_axis}
    metrics["latent_gauge/root/ratio"] = {step: ratio + 0.05 for step in training_axis}
    metrics["latent_gauge/future/ratio"] = {step: ratio + 0.02 for step in training_axis}
    metrics["latent_gauge/root/scale"] = {step: ratio + 0.05 for step in training_axis}
    metrics["latent_gauge/future/scale"] = {step: ratio + 0.02 for step in training_axis}
    metrics["latent_gauge/root/reference"] = {step: 1.0 for step in training_axis}
    metrics["latent_gauge/future/reference"] = {step: 1.0 for step in training_axis}
    metrics["latent_gauge/loss"] = {step: 0.01 for step in training_axis}
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
    metrics["eval/num_episodes"] = {25_000: 5.0}
    metrics["eval/successes"] = {25_000: 1.0}
    metrics["eval/success_rate"] = {25_000: 0.2}
    metrics["eval/distance_reduction_frac"] = {25_000: 0.1}
    return metrics


def synthetic_stage_5000():
    metrics_by_key = {}
    rows = []
    derived = {}
    for setting_index, setting in enumerate(RAW["settings"]):
        for arm_index, arm in enumerate(RAW["arms"]):
            for seed_index, seed in enumerate(RAW["seeds"]):
                key = (setting, arm, seed)
                metrics = healthy_metrics(arm=arm, ratio=0.90 if arm == "N" else 1.0)
                metrics_by_key[key] = metrics
                health = bind_exp20.evaluate_exp20_metrics(EXP20_MANIFEST, metrics, 5_000, arm)
                derived[key] = health
                index = ((setting_index * 3) + arm_index) * 2 + seed_index
                token = campaign.stable_hash([setting, arm, seed])
                rows.append({
                    "index": index,
                    "stage_slot": index,
                    "run_name": f"gauge-v2-launch2-{setting}-arm{arm.lower()}-seed{seed}",
                    "setting_id": setting,
                    "arm_id": arm,
                    "seed": seed,
                    "launch_sha256": token,
                    "identity_sha256": campaign.stable_hash([token, "identity"]),
                    "checkpoint_sha256": campaign.stable_hash([token, "5k"]),
                    "health": health,
                })
    summary = {}
    for arm in ("G", "GS"):
        cells = [derived[(setting, arm, seed)] for setting in RAW["settings"] for seed in RAW["seeds"]]
        deltas = [
            derived[(setting, arm, seed)]["recent_gauge_min_ratio"]
            - derived[(setting, "N", seed)]["recent_gauge_min_ratio"]
            for setting in RAW["settings"] for seed in RAW["seeds"]
        ]
        summary[arm] = {
            "universal_cells_passed": sum(cell["candidate_passed"] for cell in cells),
            "required_cells": 10,
            "paired_ratio_deltas_vs_n": deltas,
            "paired_mean_ratio_delta_vs_n": statistics.fmean(deltas),
            "causal_scale_retention_passed": statistics.fmean(deltas) > 0.0,
            "eligible": all(cell["candidate_passed"] for cell in cells) and statistics.fmean(deltas) > 0.0,
        }
    gate = {
        "schema_version": 1,
        "status": "accepted_for_selected_continuation",
        "campaign_id": CONTRACT["campaign_id"],
        "formal_validation": False,
        "stage_target": 5_000,
        "selected_arm": "G",
        "selection_precedence": ["G", "GS"],
        "nonpromotable_arm": "N",
        "candidate_summary": summary,
        "runs": rows,
        "package_protocol_sha256": CONTRACT["package_protocol_sha256"],
        "source_sha256": CONTRACT["source_sha256"],
        "runtime_sha256": CONTRACT["runtime_sha256"],
        "actual_evaluation_bank_sha256": CONTRACT["actual_evaluation_bank_sha256"],
    }
    gate["gate_sha256"] = campaign.stable_hash(gate)
    return gate, metrics_by_key


def synthetic_acceptance(stage_gate: dict, metrics_by_key: dict):
    selected_rows = []
    skipped_rows = []
    by_key = {(row["setting_id"], row["arm_id"], row["seed"]): row for row in stage_gate["runs"]}
    for setting_index, setting in enumerate(RAW["settings"]):
        for arm in ("G", "GS"):
            arm_index = RAW["arms"].index(arm)
            for seed_index, seed in enumerate(RAW["seeds"]):
                key = (setting, arm, seed)
                prior = by_key[key]
                stage_slot = setting_index * 4 + (0 if arm == "G" else 2) + seed_index
                if arm == "G":
                    selected_rows.append({
                        "index": ((setting_index * 3) + arm_index) * 2 + seed_index,
                        "stage_slot": stage_slot,
                        "run_name": prior["run_name"],
                        "setting_id": setting,
                        "arm_id": arm,
                        "seed": seed,
                        "launch_sha256": prior["launch_sha256"],
                        "identity_sha256": prior["identity_sha256"],
                        "checkpoint_sha256": campaign.stable_hash([prior["checkpoint_sha256"], "25k"]),
                        "health": bind_exp20.evaluate_exp20_metrics(EXP20_MANIFEST, metrics_by_key[key], 25_000, arm),
                        "outcome": {
                            "num_episodes": 5.0,
                            "successes": 1.0,
                            "success_rate": 0.2,
                            "distance_reduction_frac": 0.1,
                        },
                    })
                else:
                    skipped_rows.append({
                        "index": ((setting_index * 3) + arm_index) * 2 + seed_index,
                        "stage_slot": stage_slot,
                        "setting_id": setting,
                        "arm_id": arm,
                        "seed": seed,
                        "launch_sha256": prior["launch_sha256"],
                        "checkpoint_sha256": prior["checkpoint_sha256"],
                        "skip_sha256": campaign.stable_hash([prior["checkpoint_sha256"], "skip"]),
                    })
    outcome = bind_exp20.recompute_outcome(RAW, selected_rows)
    gate = {
        "schema_version": 1,
        "status": CONTRACT["required_status"],
        "campaign_id": CONTRACT["campaign_id"],
        "formal_validation": False,
        "stage_target": 25_000,
        "selected_arm": "G",
        "stage_5000_gate_sha256": stage_gate["gate_sha256"],
        "selected_runs": selected_rows,
        "skipped_runs": skipped_rows,
        "outcome": outcome,
        "package_protocol_sha256": CONTRACT["package_protocol_sha256"],
        "source_sha256": CONTRACT["source_sha256"],
        "runtime_sha256": CONTRACT["runtime_sha256"],
        "actual_evaluation_bank_sha256": CONTRACT["actual_evaluation_bank_sha256"],
    }
    gate["gate_sha256"] = campaign.stable_hash(gate)
    return gate


def test_exp20_launch_binding_requires_direct_shared_trainer() -> None:
    setting, arm, seed = RAW["settings"][0], "N", RAW["seeds"][0]
    run_name = f"gauge-v2-launch2-{setting}-arm{arm.lower()}-seed{seed}"
    snapshot_repo = bind_exp20._expected_exp20_snapshot_repo(CONTRACT, EXP20_MANIFEST)
    assert snapshot_repo != campaign.REPOSITORY_ROOT
    launch = {
        "campaign_id": CONTRACT["campaign_id"],
        "formal_validation": False,
        "run": {
            "setting_id": setting,
            "arm_id": arm,
            "seed": seed,
            "run_name": run_name,
            "index": 0,
        },
        "hashes": {
            key: CONTRACT[key]
            for key in (
                "manifest_sha256",
                "package_protocol_sha256",
                "source_sha256",
                "runtime_sha256",
                "actual_evaluation_bank_sha256",
            )
        },
        "argv": [
            EXP20_MANIFEST["paths"]["python"],
            str(snapshot_repo / "scripts/train.py"),
            "experiment=treewm_v2_grounded_gauge_pilot_v2",
            "objective_version=treewm_v2_grounded_gauge_pilot_v2",
            "train.steps=25000",
            "train.scheduler_total_steps=1000000",
            "resume=auto",
            "+campaign_factorial_arm=N",
        ],
    }
    launch["launch_sha256"] = campaign.stable_hash(launch)
    bind_exp20._validate_exp20_launch(
        launch, CONTRACT, EXP20_MANIFEST, (setting, arm, seed), run_name,
    )

    arbitrary_snapshot_launch = copy.deepcopy(launch)
    arbitrary_snapshot_launch["argv"][1] = "/some/other/snapshot/repo/scripts/train.py"
    arbitrary_snapshot_launch.pop("launch_sha256")
    arbitrary_snapshot_launch["launch_sha256"] = campaign.stable_hash(arbitrary_snapshot_launch)
    with pytest.raises(campaign.ContractError, match="exact sealed direct"):
        bind_exp20._validate_exp20_launch(
            arbitrary_snapshot_launch, CONTRACT, EXP20_MANIFEST, (setting, arm, seed), run_name,
        )

    wrapper_launch = copy.deepcopy(launch)
    wrapper_launch["argv"][1] = str(
        snapshot_repo / "experiments/20-treewm-grounded-gauge-pilot-v2/train_entry.py"
    )
    wrapper_launch.pop("launch_sha256")
    wrapper_launch["launch_sha256"] = campaign.stable_hash(wrapper_launch)
    with pytest.raises(campaign.ContractError, match="exact sealed direct"):
        bind_exp20._validate_exp20_launch(
            wrapper_launch, CONTRACT, EXP20_MANIFEST, (setting, arm, seed), run_name,
        )


def test_raw_metric_recomputation_rejects_forged_method_boolean() -> None:
    metrics = healthy_metrics(arm="G", ratio=1.0)
    health = bind_exp20.evaluate_exp20_metrics(EXP20_MANIFEST, metrics, 5_000, "G")
    assert health["candidate_passed"] is True
    forged_metrics = copy.deepcopy(metrics)
    forged_metrics["control/q_advantage_over_z"][5_000] = -999.0
    forged = copy.deepcopy(health)
    forged["method_gates"]["q_advantage"] = True
    forged["method_passed"] = True
    forged["candidate_passed"] = True
    with pytest.raises(campaign.ContractError, match="raw TensorBoard"):
        bind_exp20.recompute_health(
            EXP20_MANIFEST,
            forged_metrics,
            {"run_name": "forged", "arm_id": "G", "health": forged},
            target=5_000,
        )


def test_raw_metric_recomputation_rejects_stale_and_low_clip() -> None:
    stale = healthy_metrics(arm="G", ratio=1.0)
    stale["latent_gauge/min_ratio"].pop(5_000)
    result = bind_exp20.evaluate_exp20_metrics(EXP20_MANIFEST, stale, 5_000, "G")
    assert result["integrity_gates"]["target_appropriate_telemetry"] is False
    assert result["integrity_gates"]["complete_recent_gauge_axis"] is False

    clipped = healthy_metrics(arm="GS", ratio=1.0)
    for tag in (
        "train/grad_clip_coefficient_world",
        "train/grad_clip_coefficient_gain",
        "train/grad_clip_coefficient_world_rest",
        "train/grad_clip_coefficient_branch_transformer",
    ):
        clipped[tag] = {step: 0.01 for step in clipped[tag]}
    result = bind_exp20.evaluate_exp20_metrics(EXP20_MANIFEST, clipped, 5_000, "GS")
    assert result["integrity_gates"]["bounded_gradient_clipping"] is False


@pytest.mark.parametrize("low_clip_samples", [33, 29, 36])
def test_raw_exp20_n_saturation_and_collapse_are_causal_not_blocking(
    low_clip_samples: int,
) -> None:
    metrics = healthy_metrics(arm="N", ratio=0.30, target=5_000)
    axis = sorted(metrics["train/grad_clip_coefficient_world"])
    assert len(axis) == 100
    for tag in ("train/grad_clip_coefficient_world", "train/grad_clip_coefficient_gain"):
        for step in axis[:low_clip_samples]:
            metrics[tag][step] = 0.01
    metrics["control/q_advantage_over_z"][5_000] = -1.0
    result = bind_exp20.evaluate_exp20_metrics(EXP20_MANIFEST, metrics, 5_000, "N")
    assert result["clip_fraction_below_threshold"] == pytest.approx(low_clip_samples / 100)
    assert result["structural_integrity_passed"] is True
    assert result["integrity_passed"] is False
    assert result["method_passed"] is False
    assert result["gauge_absolute_passed"] is False


@pytest.mark.parametrize("coefficient", [-0.1, 0.0, 1.1])
def test_raw_exp20_n_invalid_clip_coefficients_are_structural_failures(
    coefficient: float,
) -> None:
    metrics = healthy_metrics(arm="N", ratio=0.30, target=5_000)
    metrics["train/grad_clip_coefficient_world"][5_000] = coefficient
    result = bind_exp20.evaluate_exp20_metrics(EXP20_MANIFEST, metrics, 5_000, "N")
    assert result["clip_coefficients_valid"] is False
    assert result["structural_integrity_passed"] is False


def test_stage_5000_allows_n_saturation_but_requires_all_30_structural_integrity() -> None:
    gate, metrics = synthetic_stage_5000()
    selected, _rows = bind_exp20.recompute_stage_5000(CONTRACT, gate, EXP20_MANIFEST, metrics)
    assert selected == "G"

    saturated_gate = copy.deepcopy(gate)
    saturated_metrics = copy.deepcopy(metrics)
    key = (RAW["settings"][0], "N", RAW["seeds"][0])
    saturated_metrics[key]["train/grad_clip_coefficient_world"] = {
        step: 0.01 for step in saturated_metrics[key]["train/grad_clip_coefficient_world"]
    }
    record = next(row for row in saturated_gate["runs"] if (row["setting_id"], row["arm_id"], row["seed"]) == key)
    record["health"] = bind_exp20.evaluate_exp20_metrics(
        EXP20_MANIFEST, saturated_metrics[key], 5_000, "N"
    )
    assert record["health"]["integrity_passed"] is False
    assert record["health"]["structural_integrity_passed"] is True
    selected, _rows = bind_exp20.recompute_stage_5000(
        CONTRACT, saturated_gate, EXP20_MANIFEST, saturated_metrics
    )
    assert selected == "G"

    corrupt_gate = copy.deepcopy(saturated_gate)
    corrupt_metrics = copy.deepcopy(saturated_metrics)
    corrupt_metrics[key]["train/grad_clip_coefficient_world"][5_000] = 0.0
    record = next(row for row in corrupt_gate["runs"] if (row["setting_id"], row["arm_id"], row["seed"]) == key)
    record["health"] = bind_exp20.evaluate_exp20_metrics(
        EXP20_MANIFEST, corrupt_metrics[key], 5_000, "N"
    )
    with pytest.raises(campaign.ContractError, match="structural integrity failed"):
        bind_exp20.recompute_stage_5000(
            CONTRACT, corrupt_gate, EXP20_MANIFEST, corrupt_metrics
        )


def test_acceptance_enforces_cross_stage_launch_identity_and_slots() -> None:
    stage, metrics = synthetic_stage_5000()
    selected, rows = bind_exp20.recompute_stage_5000(CONTRACT, stage, EXP20_MANIFEST, metrics)
    acceptance = synthetic_acceptance(stage, metrics)
    bind_exp20.recompute_acceptance(
        CONTRACT, acceptance, stage["gate_sha256"], selected, rows, EXP20_MANIFEST, metrics,
    )
    forged = copy.deepcopy(acceptance)
    forged["selected_runs"][0]["launch_sha256"] = "f" * 64
    with pytest.raises(campaign.ContractError, match="launch changed"):
        bind_exp20.recompute_acceptance(
            CONTRACT, forged, stage["gate_sha256"], selected, rows, EXP20_MANIFEST, metrics,
        )


@pytest.mark.parametrize(
    "foreign",
    [
        "exp15",
        "exp16",
        "exp18",
        "15-treewm-grounded-repair-pilot-v1",
        "treewm-grounded-repair-pilot-v1",
        "16-treewm-grounded-repair-all-ten-bridge-v1",
        "treewm-grounded-repair-all-ten-bridge-v1",
        "18-treewm-grounded-gauge-pilot-v1",
        "treewm-grounded-gauge-pilot-v1",
    ],
)
def test_real_exp15_exp16_exp18_ancestry_identities_are_rejected(foreign: str) -> None:
    with pytest.raises(campaign.ContractError, match="ancestry"):
        bind_exp20.reject_forbidden_ancestry(
            {"foreign_prerequisite": foreign}, CONTRACT["forbidden_ancestry_tokens"], "test",
        )


def test_exp21_selected_recipe_matches_pinned_exp20_recipe() -> None:
    for arm in ("G", "GS"):
        assert campaign.selected_recipe(MANIFEST, arm) == bind_exp20._selected_exp20_recipe(EXP20_MANIFEST, arm)
