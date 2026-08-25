"""Static contracts for the checkpoint-only inference ablation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

from scripts.checkpoint_ablation import (
    AblationArm,
    SCREEN_SETTINGS,
    atomic_json_exclusive,
    compact_grid,
    discover_checkpoints,
    factorial_grid,
    preregistered_contrasts,
    select_work,
    evaluate_arm,
    write_or_validate,
)
from treewm.data.ogbench_dataset import Normalizer
from treewm.evaluation.domains import Domain
from treewm.models.treewm import TreeWM, TreeWMConfig
from treewm.planning.goal_planner import PlannerConfig
from treewm.tree.expansion import TreeConfig


def test_compact_grid_is_unique_and_covers_every_preregistered_axis():
    assert SCREEN_SETTINGS == ("antmaze-large", "cube-double", "puzzle-3x3", "scene")
    arms = compact_grid(include_fixed16=True)
    assert len(arms) == 9
    assert len({arm.arm_id for arm in arms}) == len(arms)
    assert {arm.decoded_metric for arm in arms} == {"normalized_l2", "domain_raw"}
    assert {arm.max_depth for arm in arms} == {2, 3, 16}
    assert {arm.execute_steps for arm in arms} == {4, 16}
    assert {arm.scorer for arm in arms} == {"learned", "novelty_q", "random", "bfs"}
    fixed = [arm for arm in arms if arm.horizon_mode == "fixed"]
    assert fixed == [AblationArm("domain_raw", 3, 4, "learned", "fixed", 16)]

    contrasts = preregistered_contrasts(arms)
    assert len(contrasts["decoded_metric_at_d16_e16_learned"]) == 2
    assert len(contrasts["max_depth_at_domain_raw_e16_learned"]) == 3
    assert len(contrasts["execute_steps_at_domain_raw_d3_learned"]) == 2
    assert len(contrasts["frontier_scorer_at_domain_raw_d3_e4"]) == 4
    assert len(contrasts["horizon_selector_at_domain_raw_d3_e4_learned"]) == 2


def test_factorial_grid_has_48_cells_plus_one_fixed_horizon_reference():
    assert len(factorial_grid(include_fixed16=False)) == 2 * 3 * 2 * 4
    assert len(factorial_grid(include_fixed16=True)) == 2 * 3 * 2 * 4 + 1


def _checkpoint(root: Path, setting: str, seed: int) -> Path:
    path = (
        root / "outputs" / "treewm-50task-1m-v2" / setting / "treewm"
        / f"treewm-v2-{setting}-seed{seed}" / "checkpoints" / "latest.pt"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path.resolve()


def test_discovery_filters_setting_and_seed_and_sorts_deterministically(tmp_path):
    expected = _checkpoint(tmp_path, "cube-double", 0)
    _checkpoint(tmp_path, "antmaze-large", 0)
    _checkpoint(tmp_path, "cube-double", 1)
    pattern = "outputs/treewm-50task-1m-v2/*/treewm/*/checkpoints/latest.pt"
    selected = discover_checkpoints(
        [pattern], repo_root=tmp_path, settings=["cube-double"], seeds=[0]
    )
    assert selected == [expected]


def test_flat_work_index_is_checkpoint_major():
    checkpoints = [Path("a.pt"), Path("b.pt")]
    arms = compact_grid(include_fixed16=False)
    last_first_checkpoint = select_work(checkpoints, arms, work_index=len(arms) - 1)
    first_second_checkpoint = select_work(checkpoints, arms, work_index=len(arms))
    assert last_first_checkpoint[0][:2] == (0, len(arms) - 1)
    assert first_second_checkpoint[0][:2] == (1, 0)
    with pytest.raises(IndexError):
        select_work(checkpoints, arms, work_index=len(checkpoints) * len(arms))


def test_exclusive_json_never_replaces_and_resume_requires_same_identity(tmp_path):
    path = tmp_path / "result.json"
    first = {"result_id": "abc", "value": 1}
    assert write_or_validate(path, first, identity_key="result_id", resume=False)
    assert not write_or_validate(path, first, identity_key="result_id", resume=True)
    assert json.loads(path.read_text()) == first
    with pytest.raises(FileExistsError):
        atomic_json_exclusive(path, {"result_id": "abc", "value": 2})
    with pytest.raises(RuntimeError, match="identity differs"):
        write_or_validate(
            path, {"result_id": "different"}, identity_key="result_id", resume=True
        )
    assert json.loads(path.read_text()) == first


def test_slurm_wrapper_separates_immutable_source_from_live_artifacts():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/12-treewm-formal-v2/checkpoint_ablation.slurm"
    )
    text = path.read_text(encoding="utf-8")
    assert 'SOURCE_ROOT="${TREEWM_SOURCE_ROOT' in text
    assert 'PROJECT_ROOT="${TREEWM_PROJECT_ROOT' in text
    assert 'cd "$SOURCE_ROOT"' in text
    assert '--checkpoint-glob "$CHECKPOINT_GLOB"' in text
    assert '--output-root "$OUTPUT_ROOT"' in text

class _OneStepEnv:
    def reset(self, *, options, seed):
        del options, seed
        return np.zeros(2, dtype=np.float32), {"goal": np.ones(2, dtype=np.float32)}

    def step(self, action):
        del action
        return (
            np.ones(2, dtype=np.float32),
            0.0,
            False,
            False,
            {"success": True},
        )


def test_evaluate_arm_applies_inference_overrides_without_checkpoint_mutation():
    model = TreeWM(
        TreeWMConfig(
            obs_dim=2,
            action_dim=2,
            z_dim=8,
            q_dim=4,
            hidden_dim=16,
            encoder_hidden=16,
            num_layers=1,
            num_heads=2,
            branch_factor=2,
            h_max=4,
            horizons=(4,),
            scales=(("mixed", 4, 1.0),),
            max_depth=16,
            use_depth_embedding=False,
        )
    ).eval()
    normalizer = Normalizer(
        obs_mean=np.zeros(2, dtype=np.float32),
        obs_std=np.ones(2, dtype=np.float32),
        act_mean=np.zeros(2, dtype=np.float32),
        act_std=np.ones(2, dtype=np.float32),
    )
    run_cfg = OmegaConf.create(
        {
            "arm": "treewm",
            "seed": 0,
            "tree": vars(TreeConfig(node_budget=4, branch_factor=2, max_depth=16)),
            "planner": vars(PlannerConfig()),
        }
    )
    domain = Domain(
        "unit", "locomotion", (0, 1), "l2", 2, 2,
        subgoals=((0, 2),), max_episode_steps=1,
    )
    original_mode = model.cfg.horizon_mode
    outcome = evaluate_arm(
        model=model,
        normalizer=normalizer,
        run_cfg=run_cfg,
        env=_OneStepEnv(),
        domain=domain,
        tasks=[{"task_id": 1}],
        arm=AblationArm("domain_raw", 2, 4, "bfs"),
        episodes_per_task=1,
        max_env_steps=1,
        eval_seed=0,
        device=torch.device("cpu"),
    )
    assert outcome["status"] == "completed"
    assert outcome["metrics"]["eval/success_rate"] == 1.0
    assert outcome["effective"]["tree_scorer"] == "bfs"
    assert outcome["effective"]["tree_max_depth"] == 2
    assert model.cfg.horizon_mode == original_mode
