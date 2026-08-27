from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
from hydra import compose, initialize_config_dir

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
import submit


def fake_binding(manifest: dict, arm: str = "F") -> dict:
    grounded = manifest["method"]["grounded_multistep"]
    recipe = {
        "transition_mode": grounded["transition_mode"],
        "grounded_select_action_weight": grounded["selector_weights"]["action"],
        "grounded_select_endpoint_weight": grounded["selector_weights"]["endpoint"],
        "grounded_select_horizon_weight": grounded["selector_weights"]["horizon"],
        **manifest["prerequisites"]["allowed_selected_recipes"][arm],
    }
    return {
        "selected_arm": arm,
        "selected_recipe": recipe,
        "selected_recipe_sha256": campaign.stable_hash(recipe),
        "binding_sha256": campaign.stable_hash({"arm": arm, "recipe": recipe}),
    }


def test_exact_fresh_design_and_mapping() -> None:
    manifest = campaign.load_manifest()
    runs = campaign.expand_runs(manifest)
    assert len(runs) == 40
    assert [(run.index, run.setting_id, run.seed) for run in runs[:5]] == [
        (0, "scene", 200),
        (1, "scene", 201),
        (2, "scene", 202),
        (3, "scene", 203),
        (4, "puzzle-3x3", 200),
    ]
    assert campaign.eval_at(manifest, 0).task_id == 1
    assert campaign.eval_at(manifest, 199).training_index == 39
    assert campaign.eval_at(manifest, 199).task_id == 5
    assert manifest["method"]["experiment_config"] == "treewm_v2_grounded_repair_formal"
    assert "pilot" not in manifest["method"]["experiment_config"]
    assert manifest["promotion_authority"]["old_checkpoints_allowed"] is False


def test_scientific_overrides_lock_full_horizon_and_corrections() -> None:
    manifest = campaign.load_manifest()
    run = campaign.run_at(manifest, 0)
    contract = campaign.load_compatible_input(manifest, run)
    overrides = set(campaign.scientific_overrides(manifest, run, contract, fake_binding(manifest)))
    assert "experiment=treewm_v2_grounded_repair_formal" in overrides
    assert "objective_version=treewm_v2_grounded_repair_formal_v1" in overrides
    assert "train.steps=1000000" in overrides
    assert "train.scheduler_total_steps=1000000" in overrides
    assert "train.ckpt_every=1000" in overrides
    assert "train.log_every=50" in overrides
    assert "train.val_every=2000" in overrides
    assert "train.validation_sample_seed=1701" in overrides
    assert "future_sets.recipe_anchor_policy=published_union" in overrides
    assert "planner.require_first_edge_improvement=true" in overrides
    assert "planner.min_first_edge_improvement=0.0" in overrides
    assert "tree.max_depth=3" in overrides
    assert "planner.execute_steps=4" in overrides
    assert "losses.keep_balance=true" in overrides
    assert "losses.scheduled_sampling_granularity=sequence" in overrides
    assert "losses.multistep_transition_mode=grounded_execution_v2" in overrides
    assert "losses.grounded_loss_latent_weight=0.25" in overrides
    assert "losses.grounded_loss_action_weight=0.5" in overrides
    assert "+campaign_factorial_arm=exp16-F" in overrides
    assert manifest["scientific_contract"]["first_edge_guard_extra_root_predictions_per_replan"] == 4
    assert manifest["scientific_contract"]["tree_search_nodes_per_full_budget_replan"] == 64
    assert manifest["scientific_contract"]["effective_world_predictions_per_full_budget_replan"] == 68
    assert "train.max_train_anchors=758084" in overrides
    assert "train.max_val_anchors=75816" in overrides


def test_all_published_recipe_unions_and_split_sources_are_exact() -> None:
    manifest = campaign.load_manifest()
    for setting in manifest["settings"]:
        contract = campaign.load_compatible_input(manifest, setting)
        assert contract["recipe_coverage_audit"]["train"]["source"] == "sealed_recipe_union"
        assert contract["recipe_coverage_audit"]["train"]["recipe_anchor_count"] == setting["published_union_train_anchors"]
        assert contract["recipe_coverage_audit"]["val"]["recipe_anchor_count"] == setting["published_union_validation_anchors"]
        assert contract["source_split_audit"]["train_manifest_sha256"] != contract["source_split_audit"]["validation_manifest_sha256"]
        assert contract["source_split_audit"]["path_overlap_count"] == 0
        assert contract["source_split_audit"]["sha256_overlap_count"] == 0


def test_recipe_audit_samples_recipe_values_not_prefix_positions() -> None:
    anchors = [100, 300, 900, 1200, 4000]
    selected = campaign.recipe_audit_anchors(anchors, 3)
    assert selected == [100, 900, 4000]
    assert set(selected).issubset(anchors)


def test_seed_table_round_trip_ignores_json_object_key_order(tmp_path: Path) -> None:
    manifest = campaign.load_manifest()
    payload = campaign.load_seed_table(manifest)
    path = tmp_path / "seeds.json"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    reloaded = campaign.load_seed_table(manifest, path)
    assert reloaded == payload
    assert set(reloaded["settings"]) == set(campaign.SETTING_IDS)


def test_setting_seed_banks_are_common_across_training_seeds() -> None:
    from treewm.evaluation.rollout import build_evaluation_seed_tables

    manifest = campaign.load_manifest()
    locked = campaign.load_seed_table(manifest)
    for setting in manifest["settings"]:
        generated = [
            build_evaluation_seed_tables(
                setting["evaluation_seed_protocol_sha256"], seed, [1, 2, 3, 4, 5], 1, 50
            )
            for seed in campaign.SEEDS
        ]
        assert len({bundle["monitor"]["sha256"] for bundle in generated}) == 1
        assert len({bundle["final"]["sha256"] for bundle in generated}) == 1
        assert all(bundle["final"] == locked["settings"][setting["id"]] for bundle in generated)


def test_snapshot_rejects_writable_noncritical_source(tmp_path: Path) -> None:
    source = tmp_path / "treewm" / "deep" / "noncritical.py"
    source.parent.mkdir(parents=True)
    source.write_text("x = 1\n", encoding="utf-8")
    source.chmod(0o444)
    assert campaign.assert_snapshot_files_read_only(tmp_path) == 1
    source.chmod(0o644)
    with pytest.raises(campaign.ContractError, match="writable"):
        campaign.assert_snapshot_files_read_only(tmp_path)


def test_evaluation_source_fingerprint_rejects_eval_tamper(tmp_path: Path) -> None:
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "__init__.py").write_text("", encoding="utf-8")
    evaluator = scripts / "eval.py"
    evaluator.write_text("VALUE = 1\n", encoding="utf-8")
    original = campaign.evaluation_source_contract(tmp_path)
    evaluator.write_text("VALUE = 2\n", encoding="utf-8")
    changed = campaign.evaluation_source_contract(tmp_path)
    assert set(original["files"]) == {"scripts/__init__.py", "scripts/eval.py"}
    assert changed["sha256"] != original["sha256"]


def test_submit_interpreter_is_exactly_pinned() -> None:
    manifest = campaign.load_manifest()
    assert submit.verify_submit_interpreter(manifest, manifest["paths"]["python"]) == manifest["paths"]["python"]
    with pytest.raises(campaign.ContractError, match="must run under pinned formal Python"):
        submit.verify_submit_interpreter(manifest, "/usr/bin/env")


def test_every_dag_launcher_and_resource_contract_is_sealed() -> None:
    submit.validate_slurms(campaign.CAMPAIGN_DIR)
    assert len(campaign.PROTOCOL_FILES) == len(set(campaign.PROTOCOL_FILES))
    assert {"gate.slurm", "aggregate.slurm", "train.slurm", "final_eval.slurm"}.issubset(campaign.PROTOCOL_FILES)
    train = (campaign.CAMPAIGN_DIR / "train.slurm").read_text(encoding="utf-8")
    final = (campaign.CAMPAIGN_DIR / "final_eval.slurm").read_text(encoding="utf-8")
    assert '#SBATCH --mem=64G' in train and '#SBATCH --mem=64G' in final
    assert "#SBATCH --partition=cpu" in (campaign.CAMPAIGN_DIR / "gate.slurm").read_text(encoding="utf-8")
    assert "#SBATCH --partition=cpu" in (campaign.CAMPAIGN_DIR / "aggregate.slurm").read_text(encoding="utf-8")
    assert 'REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"' in train
    assert 'REQUEUE_TARGET="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"' in final
    assert 'if [[ -s "$WORKER_COMPLETE_PATH" ]]; then' in train
    assert 'worker returned success without a durable completion artifact' in train
    assert 'if [[ -s "$RESULT_PATH" ]]; then' in final
    assert 'final evaluator returned success without a durable result' in final
    for launcher in (train, final):
        cancel_body = launcher.split("on_cancel() {", 1)[1].split("}", 1)[0]
        assert 'touch "$CANCEL_LATCH"' in cancel_body
        assert 'kill -TERM "$step_pid"' not in cancel_body
    assert "verify_scheduler_dependency_policy" in (campaign.CAMPAIGN_DIR / "submit.py").read_text(encoding="utf-8")
    assert "launch_plan(snapshot_manifest, snapshot, verify_files=True)" in (campaign.CAMPAIGN_DIR / "submit.py").read_text(encoding="utf-8")
    assert submit.verify_scheduler_dependency_policy("/cm/shared/apps/slurm/current/bin/scontrol")["policy"] == "kill_invalid_depend"
    manifest = campaign.load_manifest()
    assert manifest["execution"]["srun"] == "/cm/shared/apps/slurm/current/bin/srun"
    assert manifest["execution"]["scontrol"] == "/cm/shared/apps/slurm/current/bin/scontrol"
    for script in ("train.slurm", "final_eval.slurm", "gate.slurm", "aggregate.slurm"):
        text = (campaign.CAMPAIGN_DIR / script).read_text(encoding="utf-8")
        assert "TREEWM_PYTHON" not in text
        assert f"PYTHON_EXECUTABLE={campaign.PINNED_FORMAL_PYTHON}" in text


def test_formal_hydra_preset_is_not_the_pilot_preset() -> None:
    text = (campaign.REPOSITORY_ROOT / "configs" / "experiment" / "treewm_v2_grounded_repair_formal.yaml").read_text(encoding="utf-8")
    assert "objective_version: treewm_v2_grounded_repair_formal_v1" in text
    assert "recipe_anchor_policy: published_union" in text
    assert "require_first_edge_improvement: true" in text
    assert "steps: 1000000" in text
    assert "bounded repair pilot" not in text.lower()


def test_direct_composition_is_1m_but_cannot_supply_prerequisite_seals() -> None:
    config_root = campaign.REPOSITORY_ROOT / "configs"
    with initialize_config_dir(config_dir=str(config_root), version_base=None):
        cfg = compose(
            config_name="base",
            overrides=["experiment=treewm_v2_grounded_repair_formal"],
        )
    assert cfg.objective_version == "treewm_v2_grounded_repair_formal_v1"
    assert int(cfg.train.steps) == 1_000_000
    assert float(cfg.train.lr) == pytest.approx(3e-5)
    assert cfg.losses.multistep_transition_mode == "grounded_execution_v2"
    assert cfg.losses.scheduled_sampling_granularity == "sequence"
    assert cfg.losses.keep_balance is True
    assert cfg.get("campaign_prerequisite_binding_sha256") is None
    assert cfg.get("campaign_selected_recipe_sha256") is None
    from scripts.train import repaired_formal_recipe_contract, validate_objective_version
    from treewm.utils import config as cfg_utils

    validate_objective_version(cfg.objective_version, int(cfg.train.steps))
    direct_recipe_contract = repaired_formal_recipe_contract(
        cfg_utils.loss_config(cfg), cfg.train, str(cfg.get("campaign_factorial_arm", ""))
    )
    assert direct_recipe_contract["registered_full_or_half_grounded_loss_weights"]
    assert not direct_recipe_contract["selected_arm_matches_grounded_loss_weights"]
    with pytest.raises(ValueError, match="exactly 1,000,000"):
        validate_objective_version(cfg.objective_version, 25_000)


def test_full_or_half_weights_come_only_from_bound_exp16_arm() -> None:
    manifest = campaign.load_manifest()
    run = campaign.run_at(manifest, 0)
    contract = campaign.load_compatible_input(manifest, run)
    full = set(campaign.scientific_overrides(manifest, run, contract, fake_binding(manifest, "F")))
    half = set(campaign.scientific_overrides(manifest, run, contract, fake_binding(manifest, "H")))
    assert "losses.grounded_loss_endpoint_weight=0.5" in full
    assert "losses.grounded_loss_endpoint_weight=0.25" in half
    assert "+campaign_factorial_arm=exp16-F" in full
    assert "+campaign_factorial_arm=exp16-H" in half


def test_unsealed_prerequisites_block_formal_verification() -> None:
    with pytest.raises(campaign.ContractError, match="blocked"):
        campaign.load_prerequisite_bindings(campaign.load_manifest())


def test_test_only_reports_blocked_without_scheduler_or_snapshot() -> None:
    report = submit.test_only_report(campaign.load_manifest(), campaign.REPOSITORY_ROOT)
    assert report["status"] == "blocked_waiting_for_accepted_exp15_and_exp16"
    assert report["submitted"] is False
    assert report["snapshot_created"] is False
    assert report["scheduler_contacted"] is False
    assert report["training_runs"] == 40
    assert report["final_eval_tasks"] == 200


def test_absolute_package_entrypoints_work_outside_repository() -> None:
    for command in (
        [sys.executable, str(campaign.CAMPAIGN_DIR / "campaign.py"), "runs"],
        [sys.executable, str(campaign.CAMPAIGN_DIR / "worker.py"), "--help"],
    ):
        result = subprocess.run(
            command,
            cwd="/tmp",
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert result.returncode == 0, result.stderr
