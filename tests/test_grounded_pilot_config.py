"""Fail-closed contract for the corrected, bounded TreeWM-v2 pilot."""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from scripts.train import validate_objective_version


def _config():
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        return compose(
            config_name="base",
            overrides=["experiment=treewm_v2_grounded_pilot"],
        )


def test_grounded_pilot_contract_is_explicit_and_bounded():
    cfg = _config()
    assert cfg.objective_version == "treewm_v2_grounded_pilot_v1"
    assert cfg.planner.decoded_metric == "domain_raw"
    assert cfg.planner.execute_steps == 4
    assert cfg.tree.max_depth == 3
    assert cfg.model.max_depth == 3
    assert cfg.model.dropout == pytest.approx(0.1)
    assert cfg.train.lr == pytest.approx(1.0e-4)
    assert cfg.train.weight_decay == pytest.approx(1.0e-3)
    assert cfg.train.gain_loss_every == 1
    assert cfg.train.gain_lr == pytest.approx(3.0e-4)
    assert cfg.train.gain_weight_decay == 0.0
    assert list(cfg.train.gain_training_scorers) == ["learned", "novelty_q"]
    assert cfg.losses.enabled.multistep is True
    assert cfg.future_sets.cache is False
    assert cfg.future_sets.shared_cache is True
    assert cfg.future_sets.retrieval_pool == 50_000
    assert cfg.losses.weights.multistep == pytest.approx(1.0)
    assert cfg.losses.scheduled_sampling_p == pytest.approx(0.25)
    assert cfg.losses.scheduled_sampling_granularity == "step"
    assert list(cfg.losses.multistep_depth_weights) == [1.0, 1.0, 1.0]
    assert cfg.losses.gain_rank_weight == pytest.approx(1.0)
    assert cfg.losses.gain_calibration_weight == 0.0

    validate_objective_version(str(cfg.objective_version), 20_000)
    with pytest.raises(ValueError, match="bounded diagnostic objective"):
        validate_objective_version(str(cfg.objective_version), 20_001)


def test_unknown_objective_still_fails_closed():
    with pytest.raises(ValueError, match="unsupported objective_version"):
        validate_objective_version("treewm_v2_unregistered", 5_000)


def test_grounded_repair_pilot_objective_is_recognized_and_capped_at_25k():
    objective = "treewm_v2_grounded_repair_pilot_v1"
    validate_objective_version(objective, 25_000)
    with pytest.raises(ValueError, match="25000-update cap"):
        validate_objective_version(objective, 25_001)


def test_grounded_repair_preset_locks_the_candidate_without_weakening_gates():
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="base",
            overrides=["experiment=treewm_v2_grounded_repair_pilot"],
        )
    assert cfg.objective_version == "treewm_v2_grounded_repair_pilot_v1"
    assert cfg.train.steps == 25_000
    assert cfg.train.scheduler_total_steps == 1_000_000
    assert cfg.train.ckpt_every == cfg.train.val_every == 1_000
    assert cfg.train.validation_sample_seed == 1701
    assert cfg.train.lr == pytest.approx(3.0e-5)
    assert cfg.tree.keep_threshold == pytest.approx(0.5)
    assert cfg.losses.keep_balance is True
    assert cfg.losses.scheduled_sampling_granularity == "sequence"
    assert cfg.losses.multistep_transition_mode == "grounded_execution_v2"
    assert cfg.losses.grounded_select_action_weight == pytest.approx(1.0)
    assert cfg.losses.grounded_select_endpoint_weight == pytest.approx(1.0)
    assert cfg.losses.grounded_select_horizon_weight == pytest.approx(0.25)
    assert cfg.losses.grounded_loss_latent_weight == pytest.approx(0.25)
    assert cfg.losses.grounded_loss_action_weight == pytest.approx(0.5)
    assert cfg.losses.grounded_loss_horizon_weight == pytest.approx(0.25)
    assert cfg.losses.grounded_loss_endpoint_weight == pytest.approx(0.5)
    assert cfg.losses.grounded_detach_self_fed_parent is True
    assert cfg.planner.require_first_edge_improvement is True
    assert cfg.eval.final_episodes_per_task == 5
