from __future__ import annotations

import copy
from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

import campaign
import bind_prerequisites
import raw_exp20_recompute
import submit


def fake_binding(manifest: dict, arm: str = "G") -> dict:
    recipe = manifest["prerequisites"]["allowed_selected_recipes"][arm]
    return {
        "selected_arm": arm,
        "selected_recipe": recipe,
        "selected_recipe_sha256": campaign.stable_hash(recipe),
        "binding_sha256": "b" * 64,
    }


def test_fresh_matrix_and_namespaces_are_exact() -> None:
    manifest = campaign.load_manifest()
    runs = campaign.expand_runs(manifest)
    assert len(runs) == 40
    assert [run.seed for run in runs[:4]] == [220, 221, 222, 223]
    assert not set(campaign.SEEDS) & (set(range(4)) | set(range(100, 112)))
    assert len({run.run_name for run in runs}) == len({run.wandb_id for run in runs}) == 40
    assert campaign.eval_at(manifest, 199).training_index == 39
    assert campaign.eval_at(manifest, 199).task_id == 5
    assert manifest["paths"]["run_root"].endswith("treewm-grounded-gauge-formal-v1")
    assert manifest["promotion_authority"]["old_checkpoints_allowed"] is False


@pytest.mark.parametrize("arm,separate", [("G", False), ("GS", True)])
def test_selected_recipe_is_the_only_dynamic_method_field(arm: str, separate: bool) -> None:
    manifest = campaign.load_manifest()
    run = campaign.run_at(manifest, 0)
    contract = campaign.load_compatible_input(manifest, run)
    overrides = set(campaign.scientific_overrides(manifest, run, contract, fake_binding(manifest, arm)))
    assert "experiment=treewm_v2_grounded_gauge_formal_v1" in overrides
    assert "objective_version=treewm_v2_grounded_gauge_formal_v1" in overrides
    assert "train.steps=1000000" in overrides
    assert "train.scheduler_total_steps=1000000" in overrides
    assert "train.eval_every=25000" in overrides
    assert f"train.separate_branch_transformer_grad_clip={str(separate).lower()}" in overrides
    assert "train.branch_transformer_grad_clip=1" in overrides
    assert "losses.enabled.latent_gauge=true" in overrides
    assert "losses.weights.latent_gauge=1" in overrides
    assert "losses.scheduled_sampling_granularity=sequence" in overrides
    assert "losses.multistep_transition_mode=grounded_execution_v2" in overrides
    assert "losses.grounded_select_action_weight=1" in overrides
    assert "losses.grounded_select_endpoint_weight=1" in overrides
    assert "losses.grounded_select_horizon_weight=0.25" in overrides
    assert "future_sets.recipe_anchor_policy=published_union" in overrides
    assert "+campaign_factorial_arm=exp21-" + arm in overrides


def test_direct_hydra_composition_has_no_prerequisite_authority() -> None:
    with initialize_config_dir(config_dir=str(campaign.REPOSITORY_ROOT / "configs"), version_base=None):
        cfg = compose(config_name="base", overrides=["experiment=treewm_v2_grounded_gauge_formal_v1"])
    assert cfg.objective_version == "treewm_v2_grounded_gauge_formal_v1"
    assert int(cfg.train.steps) == 1_000_000
    assert cfg.losses.scheduled_sampling_granularity == "sequence"
    assert cfg.losses.multistep_transition_mode == "grounded_execution_v2"
    assert cfg.losses.enabled.latent_gauge is True
    assert cfg.get("campaign_prerequisite_binding_sha256") is None
    assert cfg.get("campaign_selected_recipe_sha256") is None
    from scripts import train
    assert cfg.objective_version in train.TREEWM_V2_OBJECTIVES
    assert cfg.objective_version in train.LATENT_GAUGE_OBJECTIVES
    assert cfg.objective_version in train.GROUNDED_FORMAL_OBJECTIVES
    assert cfg.objective_version in train.FORMAL_RECIPE_AUTHORIZATION_LABELS
    assert cfg.objective_version not in train.BOUNDED_PILOT_OBJECTIVES


def test_seed_bank_is_common_and_monitor_disjoint() -> None:
    from treewm.evaluation.rollout import build_evaluation_seed_tables
    manifest = campaign.load_manifest()
    bundle = campaign.load_seed_table(manifest)
    seen: set[int] = set()
    for setting in manifest["settings"]:
        tables = [build_evaluation_seed_tables(setting["evaluation_seed_protocol_sha256"], seed, campaign.TASK_IDS, 1, 50) for seed in campaign.SEEDS]
        assert len({table["final"]["sha256"] for table in tables}) == 1
        assert len({table["monitor"]["sha256"] for table in tables}) == 1
        assert tables[0]["final"] == bundle["settings"][setting["id"]]
        final = {value for row in tables[0]["final"]["seeds"] for value in row}
        monitor = {value for row in tables[0]["monitor"]["seeds"] for value in row}
        assert not final & monitor
        assert not seen & (final | monitor)
        seen |= final | monitor


def test_binding_placeholder_fails_closed_and_rejects_old_fields(tmp_path: Path) -> None:
    manifest = campaign.load_manifest()
    with pytest.raises(campaign.ContractError, match="blocked"):
        campaign.load_prerequisite_bindings(manifest)
    recipe = manifest["prerequisites"]["allowed_selected_recipes"]["G"]
    forged = {
        "schema_version": 1,
        "campaign_id": manifest["campaign_id"],
        "status": "sealed_accepted_exp20_and_exp21_raw_recomputed",
        "formal_submission_allowed": True,
        "selection_policy": "derive_exact_exp21_recipe_after_independent_exp20_and_exp21_raw_recomputation",
        "selected_arm": "G",
        "selected_recipe": recipe,
        "selected_recipe_sha256": campaign.stable_hash(recipe),
        "exp20": {"campaign_id": manifest["prerequisites"]["exp20"]["campaign_id"]},
        "exp21": {"campaign_id": manifest["prerequisites"]["exp21"]["campaign_id"], "ancestor": "treewm-grounded-formal-v1"},
        "bound_files": {},
    }
    forged["binding_sha256"] = campaign.stable_hash(forged)
    path = tmp_path / "binding.json"
    campaign.atomic_json(path, forged)
    with pytest.raises(campaign.ContractError, match="forbidden"):
        campaign.load_prerequisite_bindings(manifest, path, verify_external_files=False)


def test_slurm_and_static_blocked_report() -> None:
    manifest = campaign.load_manifest()
    submit.validate_slurms(campaign.CAMPAIGN_DIR)
    report = submit.test_only_report(manifest, campaign.REPOSITORY_ROOT)
    assert report["status"] == "blocked_waiting_for_accepted_exp20_and_exp21_raw_evidence"
    assert report["submitted"] is False and report["snapshot_created"] is False
    assert report["training_runs"] == 40 and report["final_eval_tasks"] == 200


def test_manifest_mutations_fail_closed() -> None:
    manifest = campaign.load_manifest()
    for mutate in (
        lambda value: value["design"].update(seeds=[0, 1, 2, 3]),
        lambda value: value["lifecycle"].update(stage_targets=[2_000, 25_000, 1_000_000]),
        lambda value: value["lifecycle"].update(post_update_cadence_policy="invalid"),
        lambda value: value["stage_acceptance"].update(post_100k_policy="outcome_selection"),
        lambda value: value["final_evaluation"].update(episodes_per_task_per_rail=5),
        lambda value: value["method"]["grounded_multistep"]["selector_weights"].update(action=999),
        lambda value: value["prerequisites"]["allowed_selected_recipes"]["G"].update(grounded_loss_action_weight=999),
    ):
        changed = copy.deepcopy(manifest)
        mutate(changed)
        with pytest.raises(campaign.ContractError):
            campaign.validate_manifest(changed)


def test_train_entry_preserves_two_layer_canonical_authorization() -> None:
    import inspect

    entry = (campaign.CAMPAIGN_DIR / "train_entry.py").read_text(encoding="utf-8")
    assert entry.index("verify_exact_invocation()") < entry.index("os.execve(")
    assert "authoritative second" in entry

    from scripts import train

    source = inspect.getsource(train.main.__wrapped__)
    authorization = source.index(
        "validate_formal_recipe_authorization(objective_version, cfg)"
    )
    assert authorization < source.index("build_datasets(") < source.index("build_model(")


def test_exp21_binding_requires_durable_25k_post_update_cadence() -> None:
    valid = {
        "post_update_cadence": {
            "schema_version": 1,
            "committed_update": 25_000,
            "completed_update": 25_000,
            "replay_action": None,
        }
    }
    assert (
        bind_prerequisites._validated_exp21_post_update_cadence(valid)
        == valid["post_update_cadence"]
    )
    with pytest.raises(campaign.ContractError, match="post-update cadence is invalid"):
        bind_prerequisites._validated_exp21_post_update_cadence({})
    wrong_boundary = copy.deepcopy(valid)
    wrong_boundary["post_update_cadence"]["committed_update"] = 24_999
    wrong_boundary["post_update_cadence"]["completed_update"] = 24_999
    wrong_boundary["post_update_cadence"]["replay_action"] = None
    with pytest.raises(campaign.ContractError, match="not committed at 25000"):
        bind_prerequisites._validated_exp21_post_update_cadence(wrong_boundary)
    replayable_but_not_terminal = copy.deepcopy(valid)
    replayable_but_not_terminal["post_update_cadence"]["completed_update"] = 24_999
    replayable_but_not_terminal["post_update_cadence"]["replay_action"] = "evaluation"
    with pytest.raises(campaign.ContractError, match="terminal checkpoint.*incomplete"):
        bind_prerequisites._validated_exp21_post_update_cadence(
            replayable_but_not_terminal
        )

    binder_source = Path(bind_prerequisites.__file__).read_text(encoding="utf-8")
    exp21_start = binder_source.index("def _validate_exp21_checkpoint(")
    exp21_end = binder_source.index("def _validate_exp21_stage_artifacts(")
    exp20_start = binder_source.index("def _validate_exp20_checkpoint(")
    exp20_end = binder_source.index("def _exp20_stage_completion(")
    assert "_validated_exp21_post_update_cadence(payload)" in binder_source[exp21_start:exp21_end]
    assert "PostUpdateCadenceState" not in binder_source[exp20_start:exp20_end]


def test_prerequisite_replayers_require_complete_ordered_scientific_argv() -> None:
    """Unchecked method/cadence tokens must not hide behind a self-consistent hash."""
    exp20 = campaign.read_json(
        campaign.REPOSITORY_ROOT
        / "experiments/20-treewm-grounded-gauge-pilot-v2/manifest.json"
    )
    setting20 = exp20["settings"][0]
    arm20 = next(arm for arm in exp20["arms"] if arm["id"] == "G")
    contract20 = campaign.load_compatible_input(exp20, setting20, verify_files=False)
    tokens20 = raw_exp20_recompute._expected_exp20_overrides(
        exp20, setting20, arm20, 108, contract20
    )
    keys20 = [token.split("=", 1)[0] for token in tokens20]
    assert len(keys20) == len(set(keys20))
    assert {
        "train.ckpt_every",
        "train.val_every",
        "train.diag_every",
        "train.eval_every",
        "train.gain_lr",
        "model.max_depth",
        "tree.scorer",
        "planner.max_env_steps",
        "future_sets.num_neighbors",
        "losses.weights.multistep",
        "losses.multistep_depth_weights",
        "losses.latent_gauge_min_reference_scale",
        "eval.seed",
        "+campaign_input_contract_sha256",
    } <= set(keys20)
    changed20 = copy.deepcopy(exp20)
    changed20["scientific_contract"]["gain_lr"] = 999
    assert raw_exp20_recompute._expected_exp20_overrides(
        changed20, setting20, arm20, 108, contract20
    ) != tokens20
    with pytest.raises(campaign.ContractError, match="duplicate override"):
        raw_exp20_recompute._argv_overrides(
            ["python", "train.py", *tokens20, tokens20[0]], "forged Exp20"
        )

    exp21 = campaign.read_json(
        campaign.REPOSITORY_ROOT
        / "experiments/21-treewm-grounded-gauge-all-ten-bridge-v2/manifest.json"
    )
    setting21 = exp21["settings"][0]
    contract21 = campaign.load_compatible_input(exp21, setting21, verify_files=False)
    recipe = campaign.load_manifest()["prerequisites"]["allowed_selected_recipes"]["G"]
    tokens21 = bind_prerequisites._expected_exp21_overrides(
        exp21,
        setting21,
        112,
        contract21,
        recipe,
        "G",
        "a" * 64,
        campaign.stable_hash(recipe),
    )
    keys21 = [token.split("=", 1)[0] for token in tokens21]
    assert len(keys21) == len(set(keys21))
    assert {
        "train.log_every",
        "train.ckpt_every",
        "train.gain_lr",
        "tree.node_budget",
        "tree.scorer",
        "planner.require_first_edge_improvement",
        "future_sets.num_neighbors",
        "losses.weights.multistep",
        "losses.multistep_depth_weights",
        "losses.latent_gauge_epsilon",
        "eval.seed",
        "+campaign_exp20_binding_sha256",
    } <= set(keys21)
    changed21 = copy.deepcopy(exp21)
    changed21["scientific_contract"]["checkpoint_every_updates"] = 777
    assert bind_prerequisites._expected_exp21_overrides(
        changed21,
        setting21,
        112,
        contract21,
        recipe,
        "G",
        "a" * 64,
        campaign.stable_hash(recipe),
    ) != tokens21
    with pytest.raises(campaign.ContractError, match="duplicate override"):
        bind_prerequisites._argv_overrides(
            ["python", "train_entry.py", *tokens21, tokens21[0]], "forged Exp21"
        )


def test_exp20_continuation_binds_exact_previous_gate(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    gates = run_dir / "stage-gates"
    gates.mkdir(parents=True)
    digest = "d" * 64
    launch = {
        "campaign_id": "treewm-grounded-gauge-pilot-v2-launch2",
        "launch_sha256": "a" * 64,
        "run": {
            "index": 1,
            "setting_id": "scene",
            "arm_id": "G",
            "seed": 108,
        },
        "hashes": {
            "package_protocol_sha256": "b" * 64,
            "source_sha256": "c" * 64,
            "runtime_sha256": digest,
            "evaluation_seed_tables_sha256": "e" * 64,
            "final_seed_table_sha256": "f" * 64,
        },
    }
    row = {"identity_sha256": "1" * 64, "checkpoint_sha256": "2" * 64}
    previous = {
        "stage_target": 5_000,
        "gate_sha256": "3" * 64,
        "selected_arm": "G",
        "selected": True,
        "identity_sha256": "4" * 64,
        "checkpoint_sha256": "5" * 64,
    }
    campaign.atomic_json(
        gates / "AWAITING_GATE_25000.json",
        {
            "schema_version": 1,
            "status": "awaiting_external_stage_gate",
            "objective_version": "treewm_v2_grounded_gauge_pilot_v2",
            "completed_updates": 25_000,
            "step": 25_000,
            "total_steps": 25_000,
            "scheduler_total_steps": 1_000_000,
            "identity_sha256": row["identity_sha256"],
            "checkpoint": "checkpoints/latest.pt",
            "checkpoint_sha256": row["checkpoint_sha256"],
            "evaluation_seed_tables_sha256": launch["hashes"]["evaluation_seed_tables_sha256"],
        },
    )
    complete = {
        "schema_version": 1,
        "status": "stage_complete_awaiting_campaign_gate",
        "campaign_id": launch["campaign_id"],
        "stage_slot": 7,
        "index": 1,
        "setting_id": "scene",
        "arm_id": "G",
        "seed": 108,
        "stage_target": 25_000,
        "launch_sha256": launch["launch_sha256"],
        "package_protocol_sha256": launch["hashes"]["package_protocol_sha256"],
        "source_sha256": launch["hashes"]["source_sha256"],
        "runtime_sha256": launch["hashes"]["runtime_sha256"],
        "identity_sha256": row["identity_sha256"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "evaluation_seed_tables_sha256": launch["hashes"]["evaluation_seed_tables_sha256"],
        "final_seed_table_sha256": launch["hashes"]["final_seed_table_sha256"],
    }
    complete["stage_complete_sha256"] = campaign.stable_hash(complete)
    campaign.atomic_json(gates / "STAGE_COMPLETE_25000.json", complete)

    def write_launch(value: dict) -> None:
        campaign.atomic_json(
            gates / "STAGE_LAUNCH_25000.json",
            {
                "schema_version": 1,
                "status": "stage_launch_sealed",
                "stage_target": 25_000,
                "stage_slot": 7,
                "launch_sha256": launch["launch_sha256"],
                "package_protocol_sha256": launch["hashes"]["package_protocol_sha256"],
                "previous_gate": value,
            },
        )

    write_launch(previous)
    bind_prerequisites._exp20_stage_completion(
        run_dir,
        launch,
        row,
        target=25_000,
        stage_slot=7,
        current_checkpoint=False,
        forbidden_tokens=(),
        expected_previous_gate=previous,
    )
    forged = dict(previous)
    forged["gate_sha256"] = "9" * 64
    write_launch(forged)
    with pytest.raises(campaign.ContractError, match="stage launch differs"):
        bind_prerequisites._exp20_stage_completion(
            run_dir,
            launch,
            row,
            target=25_000,
            stage_slot=7,
            current_checkpoint=False,
            forbidden_tokens=(),
            expected_previous_gate=previous,
        )
