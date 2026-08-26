from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
package_path = str(PACKAGE)
while package_path in sys.path:
    sys.path.remove(package_path)
sys.path.insert(0, package_path)
for module_name in ("campaign", "worker", "stage_gate", "final_eval", "aggregate", "submit"):
    module = sys.modules.get(module_name)
    module_file = Path(getattr(module, "__file__", "")).resolve() if module else None
    if module_file is not None and not module_file.is_relative_to(PACKAGE):
        del sys.modules[module_name]

import campaign
import stage_gate


def healthy_metrics(step: int = 25_000) -> dict[str, dict[int, float]]:
    values = {
        "val/loss_total": 1.0,
        "val/loss_horizon": 1.0,
        "val/loss_multistep": 0.5,
        "val/loss_multistep_self_fed": 0.6,
        "data/validation_fixed_sample_count": 512.0,
        "data/validation_horizon_label_fraction_h4": 0.2,
        "data/validation_horizon_label_fraction_h8": 0.2,
        "data/validation_horizon_label_fraction_h16": 0.2,
        "data/validation_horizon_label_fraction_h32": 0.2,
        "data/validation_horizon_label_fraction_h64": 0.2,
        "control/q_advantage_over_z": 0.2,
        "control/q_advantage_over_random_proj": 0.1,
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
    }
    return {tag: {step: value} for tag, value in values.items()}


def test_horizon_and_gain_coverage_gate_passes() -> None:
    report = stage_gate.evaluate_metrics(campaign.load_manifest(), healthy_metrics(), 25_000)
    assert report["integrity_passed"]
    assert report["scientific_passed"]
    assert report["horizon_empirical_prior_entropy"] == pytest.approx(1.6094379124341003)


@pytest.mark.parametrize(
    ("tag", "value"),
    [
        ("val/loss_horizon", 1.7),
        ("expansion/gain_eligible_decision_fraction", 0.01),
        ("expansion/gain_ordered_pair_count", 0.0),
        ("expansion/gain_pair_coverage_fraction", 0.0),
    ],
)
def test_scientific_gate_rejects_known_degenerate_metrics(tag: str, value: float) -> None:
    metrics = healthy_metrics()
    metrics[tag][25_000] = value
    report = stage_gate.evaluate_metrics(campaign.load_manifest(), metrics, 25_000)
    assert report["integrity_passed"]
    assert not report["scientific_passed"]


def test_self_fed_multistep_validation_is_independently_nonregressing() -> None:
    metrics = healthy_metrics()
    metrics["val/loss_multistep_self_fed"] = {20_000: 0.5, 25_000: 0.7}
    report = stage_gate.evaluate_metrics(campaign.load_manifest(), metrics, 25_000)
    assert report["integrity_passed"]
    assert not report["scientific_gates"]["self_fed_multistep_validation_nonregression"]
    assert not report["scientific_passed"]


def test_scientific_coverage_is_three_of_four_per_setting() -> None:
    manifest = campaign.load_manifest()
    rows = []
    for setting in ("scene", "cube-double"):
        rows.extend(
            {"setting_id": setting, "health": {"scientific_passed": seed < 3}}
            for seed in range(4)
        )
    assert stage_gate.validate_scientific_setting_coverage(manifest, rows, 25_000) == {"scene": 3, "cube-double": 3}
    rows[2]["health"]["scientific_passed"] = False
    with pytest.raises(campaign.ContractError, match="3/4"):
        stage_gate.validate_scientific_setting_coverage(manifest, rows, 25_000)
    assert stage_gate.validate_scientific_setting_coverage(manifest, rows, 2_000) is None


def outcome_rows(successes: float) -> list[dict]:
    rows = []
    for setting in campaign.SETTING_IDS:
        for seed in range(4):
            rows.append(
                {
                    "setting_id": setting,
                    "monitor_seed_table_sha256": campaign.stable_hash({"setting": setting}),
                    "monitor": {
                        "num_episodes": 5.0,
                        "successes": successes if len(rows) == 0 else 0.0,
                        "success_rate": successes / 5.0 if len(rows) == 0 else 0.0,
                        "distance_reduction_frac": 0.1,
                    },
                }
            )
    return rows


def test_100k_outcome_sanity_rejects_all_zero_fleet() -> None:
    manifest = campaign.load_manifest()
    with pytest.raises(campaign.ContractError, match="successes=0.0/200"):
        stage_gate.validate_outcome_sanity(manifest, outcome_rows(0.0), 100_000)
    accepted = stage_gate.validate_outcome_sanity(manifest, outcome_rows(1.0), 100_000)
    assert accepted["episodes"] == 200
    assert accepted["successes"] == 1.0
    assert stage_gate.validate_outcome_sanity(manifest, outcome_rows(0.0), 25_000) is None


def test_100k_outcome_sanity_rejects_fractional_or_inconsistent_successes() -> None:
    manifest = campaign.load_manifest()
    with pytest.raises(campaign.ContractError, match="self-consistent"):
        stage_gate.validate_outcome_sanity(manifest, outcome_rows(1.5), 100_000)
    rows = outcome_rows(1.0)
    rows[0]["monitor"]["success_rate"] = 0.3
    with pytest.raises(campaign.ContractError, match="self-consistent"):
        stage_gate.validate_outcome_sanity(manifest, rows, 100_000)
