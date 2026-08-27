from __future__ import annotations

import copy

import pytest

import campaign
import raw_exp20_recompute
import stage_gate


def healthy_metrics(target: int = 25_000, arm: str = "G") -> dict[str, dict[int, float]]:
    manifest = campaign.load_manifest()
    gate = manifest["stage_acceptance"]
    freshness = gate["telemetry_freshness"]
    values = {
        "latent_gauge/root/scale": 1.0,
        "latent_gauge/root/reference": 1.0,
        "latent_gauge/root/ratio": 1.0,
        "latent_gauge/future/scale": 1.0,
        "latent_gauge/future/reference": 1.0,
        "latent_gauge/future/ratio": 1.0,
        "latent_gauge/min_ratio": 1.0,
        "latent_gauge/loss": 0.1,
        "latent_gauge/reference_sealed": 1.0,
        "latent_gauge/reference_update": 0.0,
        "val/loss_total": 1.0,
        "val/loss_horizon": 1.0,
        "val/loss_multistep": 0.5,
        "val/loss_multistep_self_fed": 0.5,
        "data/validation_fixed_sample_count": 512.0,
        "data/validation_horizon_label_fraction_h4": 0.2,
        "data/validation_horizon_label_fraction_h8": 0.2,
        "data/validation_horizon_label_fraction_h16": 0.2,
        "data/validation_horizon_label_fraction_h32": 0.2,
        "data/validation_horizon_label_fraction_h64": 0.2,
        "control/q_advantage_over_z": 0.2,
        "control/q_advantage_over_random_proj": 0.2,
        "control/retrieval_uses_task_metric_endpoint": 1.0,
        "expansion/gain_rank_correlation": 0.2,
        "expansion/gain_pairwise_accuracy": 0.6,
        "expansion/gain_eligible_decision_fraction": 0.3,
        "expansion/gain_ordered_pair_count": 8.0,
        "expansion/gain_pair_coverage_fraction": 0.1,
        "tree/support_recall": 0.6,
        "tree/support_precision": 0.3,
        "train/grad_norm_world": 1.0,
        "train/grad_norm_gain": 1.0,
        "train/grad_clip_coefficient_world": 1.0,
        "train/grad_clip_coefficient_gain": 1.0,
    }
    if arm == "GS":
        values.update({
            "train/grad_norm_world_rest": 1.0,
            "train/grad_norm_branch_transformer": 1.0,
            "train/grad_clip_coefficient_world_rest": 1.0,
            "train/grad_clip_coefficient_branch_transformer": 1.0,
        })
    training = set(freshness["training_exact_target_tags"]) | set(freshness["selected_arm_conditional_training_tags"][arm])
    validation_step = target - target % int(freshness["validation_diagnostic_every_updates"])
    metrics = {tag: {validation_step: value} for tag, value in values.items() if tag not in training}
    gradient_axis = stage_gate._expected_axis(target, 50, min(int(gate["gradient_recent_window_updates"]), target))
    gauge_axis = set(stage_gate._expected_axis(target, 50, min(int(gate["gauge_recent_window_updates"]), target)))
    for tag, value in values.items():
        if tag in training:
            axis = gradient_axis if tag.startswith("train/") else tuple(step for step in gradient_axis if step in gauge_axis)
            metrics[tag] = {step: value for step in axis}
    return metrics


@pytest.mark.parametrize("arm", ["G", "GS"])
def test_target_fresh_healthy_25k_passes(arm: str) -> None:
    report = stage_gate.evaluate_metrics(campaign.load_manifest(), healthy_metrics(25_000, arm), 25_000, arm)
    assert report["integrity_passed"] and report["scientific_passed"]
    assert set(report["last_step"].values()) == {24_000, 25_000}


def test_1m_rejects_stale_2k_telemetry() -> None:
    report = stage_gate.evaluate_metrics(campaign.load_manifest(), healthy_metrics(2_000), 1_000_000, "G")
    assert not report["integrity_gates"]["target_appropriate_telemetry"]
    assert not report["integrity_passed"]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.1, 0.0, 1.1])
def test_clip_coefficients_fail_closed(value: float) -> None:
    metrics = healthy_metrics()
    metrics["train/grad_clip_coefficient_world"][25_000] = value
    report = stage_gate.evaluate_metrics(campaign.load_manifest(), metrics, 25_000, "G")
    assert not report["integrity_gates"]["valid_gradient_clip_coefficients"]
    assert not report["integrity_passed"]


def test_per_tag_saturation_cannot_hide_in_pooled_fraction() -> None:
    metrics = healthy_metrics()
    tag = "train/grad_clip_coefficient_world"
    steps = sorted(metrics[tag])
    for step in steps[: len(steps) // 3]:
        metrics[tag][step] = 0.01
    report = stage_gate.evaluate_metrics(campaign.load_manifest(), metrics, 25_000, "G")
    assert report["clip_fraction_below_threshold_by_tag"][tag] > 0.25
    assert not report["integrity_gates"]["bounded_gradient_clipping_per_tag"]


def test_gs_requires_complete_split_gradient_axis() -> None:
    metrics = healthy_metrics(25_000, "GS")
    metrics["train/grad_norm_branch_transformer"].pop(24_950)
    report = stage_gate.evaluate_metrics(campaign.load_manifest(), metrics, 25_000, "GS")
    assert not report["integrity_gates"]["complete_recent_gradient_axis"]


def test_fabricated_ratio_cannot_mask_collapsed_scale() -> None:
    metrics = healthy_metrics()
    metrics["latent_gauge/root/scale"][25_000] = 0.1
    metrics["latent_gauge/root/reference"][25_000] = 1.0
    metrics["latent_gauge/root/ratio"][25_000] = 1.0
    metrics["latent_gauge/min_ratio"][25_000] = 1.0
    report = stage_gate.evaluate_metrics(campaign.load_manifest(), metrics, 25_000, "G")
    assert not report["integrity_gates"]["gauge_ratio_consistent"]
    assert not report["integrity_passed"]


def test_missing_recent_root_gauge_sample_fails_complete_axis() -> None:
    metrics = healthy_metrics()
    metrics["latent_gauge/root/scale"].pop(24_950)
    report = stage_gate.evaluate_metrics(campaign.load_manifest(), metrics, 25_000, "G")
    assert not report["integrity_gates"]["complete_recent_gauge_axis"]


def test_prerequisite_raw_replay_rejects_fabricated_gauge_ratios() -> None:
    exp20_manifest = campaign.read_json(
        campaign.REPOSITORY_ROOT
        / "experiments/20-treewm-grounded-gauge-pilot-v2/manifest.json"
    )
    metrics = healthy_metrics()
    for tag, samples in list(metrics.items()):
        if set(samples) == {24_000}:
            metrics[tag] = {25_000: next(iter(samples.values()))}
    healthy = raw_exp20_recompute.evaluate_exp20_metrics(
        exp20_manifest, metrics, 25_000, "G"
    )
    assert healthy["integrity_passed"]
    metrics["latent_gauge/future/scale"][24_950] = 0.01
    metrics["latent_gauge/future/ratio"][24_950] = 1.0
    forged = raw_exp20_recompute.evaluate_exp20_metrics(
        exp20_manifest, metrics, 25_000, "G"
    )
    assert not forged["integrity_gates"]["gauge_ratio_consistent"]
    assert not forged["integrity_passed"]


@pytest.mark.parametrize("mutation", ["collapse", "unsealed", "late_reference"])
def test_gauge_integrity_is_required_at_1m(mutation: str) -> None:
    metrics = healthy_metrics(1_000_000)
    if mutation == "collapse":
        metrics["latent_gauge/min_ratio"][1_000_000] = 0.7
    elif mutation == "unsealed":
        metrics["latent_gauge/reference_sealed"][1_000_000] = 0.0
    else:
        metrics["latent_gauge/reference_update"][1_000_000] = 1.0
    report = stage_gate.evaluate_metrics(campaign.load_manifest(), metrics, 1_000_000, "G")
    assert not report["integrity_passed"]


def test_method_outcomes_do_not_stop_after_100k() -> None:
    manifest = campaign.load_manifest()
    metrics = healthy_metrics(1_000_000)
    metrics["control/q_advantage_over_z"][1_000_000] = -99.0
    report = stage_gate.evaluate_metrics(manifest, metrics, 1_000_000, "G")
    assert report["scientific_raw_observations"]["q_advantage"] is False
    assert report["scientific_gates"]["q_advantage"] is True
    assert report["integrity_passed"]


def outcome_rows(*, success: float = 1.0, progress: float = 0.1) -> list[dict]:
    rows = []
    for setting in campaign.SETTING_IDS:
        for seed in campaign.SEEDS:
            first = len(rows) == 0
            successes = success if first else 0.0
            rows.append({
                "setting_id": setting,
                "monitor_seed_table_sha256": campaign.stable_hash({"setting": setting}),
                "monitor": {
                    "num_episodes": 5.0,
                    "successes": successes,
                    "success_rate": successes / 5.0,
                    "distance_reduction_frac": progress,
                },
            })
    return rows


def test_100k_requires_nonzero_success_and_positive_progress_only_at_100k() -> None:
    manifest = campaign.load_manifest()
    with pytest.raises(campaign.ContractError, match="successes"):
        stage_gate.validate_outcome_sanity(manifest, outcome_rows(success=0), 100_000)
    with pytest.raises(campaign.ContractError, match="mean progress"):
        stage_gate.validate_outcome_sanity(manifest, outcome_rows(progress=0), 100_000)
    assert stage_gate.validate_outcome_sanity(manifest, outcome_rows(success=0, progress=0), 1_000_000) is None


def test_25k_quorum_is_three_of_four_per_setting() -> None:
    manifest = campaign.load_manifest()
    rows = [
        {"setting_id": setting, "health": {"scientific_passed": index < 3}}
        for setting in campaign.SETTING_IDS
        for index, _seed in enumerate(campaign.SEEDS)
    ]
    coverage = stage_gate.validate_scientific_setting_coverage(manifest, rows, 25_000)
    assert set(coverage.values()) == {3}
    rows[2]["health"]["scientific_passed"] = False
    with pytest.raises(campaign.ContractError, match="3/4"):
        stage_gate.validate_scientific_setting_coverage(manifest, rows, 25_000)
