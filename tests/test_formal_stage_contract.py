"""Fail-closed lifecycle contract for the fresh grounded formal objective."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

from scripts.train import resolve_stage_stop_after, validate_objective_version
from treewm.utils import config as cfg_utils


def test_direct_grounded_formal_composition_cannot_fall_back_to_base_defaults():
    config_dir = str((Path(__file__).resolve().parents[1] / "configs").resolve())
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(
            config_name="base",
            overrides=["experiment=treewm_v2_grounded_formal"],
        )
    assert cfg.objective_version == "treewm_v2_grounded_formal_v1"
    assert cfg.train.steps == 1_000_000
    assert cfg.train.scheduler_total_steps == 1_000_000
    assert cfg.train.ckpt_every == 1_000
    assert cfg.train.val_every == cfg.train.diag_every == 2_000
    assert cfg.train.eval_every == 25_000
    assert cfg.eval.task_split == "standard"
    assert cfg.eval.episodes_per_task == 1
    assert cfg.eval.final_episodes_per_task == 50
    assert cfg.future_sets.recipe_anchor_policy == "published_union"
    assert cfg.planner.require_first_edge_improvement is True
    typed_future = cfg_utils.future_set_config(cfg)
    assert typed_future.multi_step_depth == 3
    assert typed_future.retrieval_pool == 50_000


def test_grounded_formal_is_not_pilot_capped_and_stage_limits_are_locked():
    validate_objective_version("treewm_v2_grounded_formal_v1", 1_000_000)
    with pytest.raises(ValueError, match="exactly 1,000,000"):
        validate_objective_version("treewm_v2_grounded_formal_v1", 50_000)
    for stage in (2_000, 25_000, 100_000, 1_000_000):
        assert resolve_stage_stop_after(
            "treewm_v2_grounded_formal_v1", 1_000_000, str(stage)
        ) == (stage, True)
    assert resolve_stage_stop_after(
        "treewm_v2_grounded_formal_v1", 1_000_000, None
    ) == (1_000_000, False)


def test_stage_limit_cannot_mutate_an_old_objective_or_scientific_horizon():
    with pytest.raises(ValueError, match="reserved"):
        resolve_stage_stop_after("treewm_v2_rms_rank_v1", 1_000_000, "2000")
    with pytest.raises(ValueError, match="must be"):
        resolve_stage_stop_after("treewm_v2_grounded_formal_v1", 1_000_000, "12000")
    with pytest.raises(ValueError, match="cannot exceed"):
        resolve_stage_stop_after("treewm_v2_grounded_formal_v1", 25_000, "100000")
    with pytest.raises(ValueError, match="bounded diagnostic"):
        validate_objective_version("treewm_v2_grounded_pilot_v1", 1_000_000)
