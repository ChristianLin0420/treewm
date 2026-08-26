from __future__ import annotations

import json
from pathlib import Path
import subprocess
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
import submit


def test_exact_fresh_design_and_mapping() -> None:
    manifest = campaign.load_manifest()
    runs = campaign.expand_runs(manifest)
    assert len(runs) == 40
    assert [(run.index, run.setting_id, run.seed) for run in runs[:5]] == [
        (0, "scene", 0),
        (1, "scene", 1),
        (2, "scene", 2),
        (3, "scene", 3),
        (4, "puzzle-3x3", 0),
    ]
    assert campaign.eval_at(manifest, 0).task_id == 1
    assert campaign.eval_at(manifest, 199).training_index == 39
    assert campaign.eval_at(manifest, 199).task_id == 5
    assert manifest["method"]["experiment_config"] == "treewm_v2_grounded_formal"
    assert "pilot" not in manifest["method"]["experiment_config"]
    assert manifest["promotion_authority"]["old_checkpoints_allowed"] is False


def test_scientific_overrides_lock_full_horizon_and_corrections() -> None:
    manifest = campaign.load_manifest()
    run = campaign.run_at(manifest, 0)
    contract = campaign.load_compatible_input(manifest, run)
    overrides = set(campaign.scientific_overrides(manifest, run, contract))
    assert "experiment=treewm_v2_grounded_formal" in overrides
    assert "objective_version=treewm_v2_grounded_formal_v1" in overrides
    assert "train.steps=1000000" in overrides
    assert "train.scheduler_total_steps=1000000" in overrides
    assert "train.ckpt_every=1000" in overrides
    assert "train.val_every=2000" in overrides
    assert "future_sets.recipe_anchor_policy=published_union" in overrides
    assert "planner.require_first_edge_improvement=true" in overrides
    assert "planner.min_first_edge_improvement=0.0" in overrides
    assert "tree.max_depth=3" in overrides
    assert "planner.execute_steps=4" in overrides
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
            for seed in range(4)
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
    text = (campaign.REPOSITORY_ROOT / "configs" / "experiment" / "treewm_v2_grounded_formal.yaml").read_text(encoding="utf-8")
    assert "objective_version: treewm_v2_grounded_formal_v1" in text
    assert "recipe_anchor_policy: published_union" in text
    assert "require_first_edge_improvement: true" in text


def test_absolute_package_entrypoints_work_outside_repository() -> None:
    for command in (
        [sys.executable, str(campaign.CAMPAIGN_DIR / "campaign.py"), "verify"],
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
