from __future__ import annotations

import copy
import hashlib
import importlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import pytest
import torch


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CAMPAIGN_DIR.parents[1]


def _load_local_modules():
    previous = {name: sys.modules.get(name) for name in ("campaign", "worker", "report", "submit")}
    for name in previous:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(CAMPAIGN_DIR))
    try:
        campaign = importlib.import_module("campaign")
        worker = importlib.import_module("worker")
        report = importlib.import_module("report")
        submit = importlib.import_module("submit")
        return campaign, worker, report, submit
    finally:
        sys.path.remove(str(CAMPAIGN_DIR))
        for name in previous:
            sys.modules.pop(name, None)
            if previous[name] is not None:
                sys.modules[name] = previous[name]


campaign, worker, report, submit = _load_local_modules()
REAL_LOAD_EXP15_PREREQUISITE = campaign.load_exp15_prerequisite


def _accepted_exp15_binding():
    value = {
        "schema_version": 1,
        "status": "accepted_exp15_prerequisite",
        "campaign_id": "treewm-grounded-repair-pilot-v1",
        "accepted_status": "accepted_for_fresh_formal_campaign_design",
        "candidate_arm": "C",
        "integrity_runs_passing": 40,
        "acceptance_path": "/synthetic/exp15/acceptance.json",
        "acceptance_sha256": "1" * 64,
        "acceptance_canonical_sha256": "2" * 64,
        "launch_plan_path": "/synthetic/exp15/LAUNCH_PLAN.json",
        "launch_plan_sha256": "3" * 64,
        "launch_plan_canonical_sha256": "4" * 64,
        "package_protocol_sha256": "ec41f19a97ab0c21d341b00baa69a6f50259408adda2d7bc6428ff46398a4f49",
        "source_sha256": "dc0b5d2c80a25c6ac51495696e83450859de4429fe9e40137572b1e981510d6a",
        "runtime_sha256": "77da91d49a1db99850fbf0632dc02ec58a3209f1a87949d6f5640ae6bf505c6b",
        "actual_evaluation_bank_sha256": "5" * 64,
    }
    value["prerequisite_sha256"] = campaign.stable_hash(value)
    return value


def _accepted_exp15_report():
    records = []
    for index, (setting, arm, seed) in enumerate(
        (setting, arm, seed)
        for setting in campaign.EXP15_SETTING_IDS
        for arm in campaign.EXP15_ARM_IDS
        for seed in campaign.EXP15_SEEDS
    ):
        records.append(
            {
                "index": index,
                "setting_id": setting,
                "arm_id": arm,
                "seed": seed,
                "integrity_gates": {
                    gate: True for gate in campaign.EXP15_INTEGRITY_GATES
                },
                "integrity_pass": True,
                "scientific_gates": {
                    gate: True for gate in campaign.EXP15_SCIENTIFIC_GATES
                },
                "scientific_pass": True,
                "metrics": {
                    "validation_final": 1.0,
                    "validation_min": 1.0,
                    "self_fed_final": 1.0,
                    "self_fed_min": 1.0,
                    "horizon_loss": 1.0,
                    "horizon_empirical_prior": 1.5,
                    "gain_recent_mean": {
                        "expansion/gain_rank_correlation": 0.1,
                        "expansion/gain_pairwise_accuracy": 0.52,
                        "expansion/gain_eligible_decision_fraction": 0.2,
                        "expansion/gain_ordered_pair_count": 1.0,
                        "expansion/gain_pair_coverage_fraction": 0.01,
                    },
                    "support_recall": 0.5,
                    "support_precision": 0.25,
                    "clip_fraction_below_threshold": 0.25,
                    "midpoint": {
                        "num_episodes": 5.0,
                        "successes": 1.0,
                        "success_rate": 0.2,
                        "distance_reduction_frac": 0.2,
                    },
                    "final": {
                        "num_episodes": 25.0,
                        "successes": 1.0,
                        "success_rate": 0.04,
                        "distance_reduction_frac": 0.2,
                    },
                },
                "error": None,
            }
        )
    paired = [
        {
            "setting_id": setting,
            "seed": seed,
            "success_delta_candidate_minus_control": 0.0,
            "distance_reduction_delta_candidate_minus_control": 0.0,
        }
        for setting in campaign.EXP15_SETTING_IDS
        for seed in campaign.EXP15_SEEDS
    ]
    return {
        "schema_version": 1,
        "campaign_id": "treewm-grounded-repair-pilot-v1",
        "status": "accepted_for_fresh_formal_campaign_design",
        "accepted": True,
        "formal_validation": False,
        "preregistered_candidate_arm": "C",
        "matched_control_arm": "A",
        "sensitivity_arms_are_nonpromotable": ["B", "D"],
        "integrity_runs_passing": 40,
        "candidate_setting_pass": {
            setting: True for setting in campaign.EXP15_SETTING_IDS
        },
        "candidate_settings_passing": 5,
        "aggregate_gates": {
            gate: True for gate in campaign.EXP15_AGGREGATE_GATES
        },
        "aggregate_metrics": {
            "candidate_total_successes": 10.0,
            "candidate_mean_distance_reduction": 0.2,
            "candidate_runs_with_positive_progress": 10,
            "paired_mean_success_delta_candidate_minus_control": 0.0,
            "paired_mean_distance_reduction_delta_candidate_minus_control": 0.0,
        },
        "paired_comparisons": paired,
        "actual_evaluation_bank_sha256": "a" * 64,
        "missing_or_extra_keys": [],
        "runs": records,
    }


def _exp15_launch_plan(manifest):
    plan = {
        "schema_version": 1,
        "campaign_id": "treewm-grounded-repair-pilot-v1",
        "status": "sealed_bounded_repair_pilot_plan",
        "formal_validation": False,
        "common_hashes": {
            "package_protocol_sha256": manifest["prerequisite"][
                "package_protocol_sha256"
            ],
            "source_sha256": manifest["prerequisite"]["source_sha256"],
            "runtime_sha256": manifest["prerequisite"]["runtime_sha256"],
            "actual_evaluation_bank_sha256": "a" * 64,
        },
        "runs": [
            {"setting_id": setting, "arm_id": arm, "seed": seed}
            for setting in campaign.EXP15_SETTING_IDS
            for arm in campaign.EXP15_ARM_IDS
            for seed in campaign.EXP15_SEEDS
        ],
    }
    plan["plan_sha256"] = campaign.stable_hash(plan)
    return plan


@pytest.fixture(autouse=True)
def accepted_exp15_prerequisite(monkeypatch):
    monkeypatch.setattr(
        campaign,
        "load_exp15_prerequisite",
        lambda *_args, **_kwargs: _accepted_exp15_binding(),
    )


@pytest.fixture(scope="module")
def manifest():
    return campaign.load_manifest(CAMPAIGN_DIR / "manifest.json")


def test_exact_10x2x2_mapping_is_complete_and_fresh(manifest):
    runs = campaign.expand_runs(manifest)
    assert len(runs) == 40
    assert [run.index for run in runs] == list(range(40))
    assert {run.seed for run in runs} == {102, 103}
    assert len({run.run_name for run in runs}) == 40
    assert len({run.wandb_id for run in runs}) == 40
    assert (runs[0].setting_id, runs[0].arm_id, runs[0].seed) == (
        "scene",
        "F",
        102,
    )
    assert (runs[-1].setting_id, runs[-1].arm_id, runs[-1].seed) == (
        "humanoidmaze-large",
        "H",
        103,
    )
    for run in runs:
        expected = ((run.setting_index * 2) + run.arm_index) * 2 + run.seed_index
        assert run.index == expected


def test_exp15_prerequisite_binds_acceptance_plan_and_pinned_identities(
    manifest, tmp_path
):
    acceptance = _accepted_exp15_report()
    plan = _exp15_launch_plan(manifest)
    acceptance_path = tmp_path / "acceptance.json"
    plan_path = tmp_path / "LAUNCH_PLAN.json"
    campaign.atomic_json(acceptance_path, acceptance)
    campaign.atomic_json(plan_path, plan)
    binding = REAL_LOAD_EXP15_PREREQUISITE(
        manifest,
        acceptance_path=acceptance_path,
        launch_plan_path=plan_path,
    )
    assert binding["status"] == "accepted_exp15_prerequisite"
    assert binding["acceptance_sha256"] == campaign.file_sha256(acceptance_path)
    assert binding["acceptance_canonical_sha256"] == campaign.stable_hash(acceptance)
    assert binding["launch_plan_canonical_sha256"] == plan["plan_sha256"]
    assert binding["package_protocol_sha256"] == manifest["prerequisite"]["package_protocol_sha256"]
    body = dict(binding)
    claimed = body.pop("prerequisite_sha256")
    assert claimed == campaign.stable_hash(body)

    acceptance["aggregate_gates"]["candidate_not_all_zero_success"] = False
    campaign.atomic_json(acceptance_path, acceptance)
    with pytest.raises(campaign.ContractError, match="aggregate gates differ"):
        REAL_LOAD_EXP15_PREREQUISITE(
            manifest,
            acceptance_path=acceptance_path,
            launch_plan_path=plan_path,
        )


def test_exp15_prerequisite_rejects_forged_scientific_quorum_claims(
    manifest, tmp_path
):
    acceptance = _accepted_exp15_report()
    for record in acceptance["runs"]:
        if record["arm_id"] == "C":
            record["scientific_gates"]["validation_regret_le_1p10"] = False
            record["scientific_pass"] = False
    plan = _exp15_launch_plan(manifest)
    acceptance_path = tmp_path / "acceptance.json"
    plan_path = tmp_path / "LAUNCH_PLAN.json"
    campaign.atomic_json(acceptance_path, acceptance)
    campaign.atomic_json(plan_path, plan)
    with pytest.raises(campaign.ContractError, match="setting quorum differs"):
        REAL_LOAD_EXP15_PREREQUISITE(
            manifest,
            acceptance_path=acceptance_path,
            launch_plan_path=plan_path,
        )


def test_exp15_prerequisite_rejects_forged_outcome_claims(manifest, tmp_path):
    acceptance = _accepted_exp15_report()
    for record in acceptance["runs"]:
        if record["arm_id"] == "C":
            record["metrics"]["final"].update(
                successes=0.0,
                success_rate=0.0,
                distance_reduction_frac=0.0,
            )
    plan = _exp15_launch_plan(manifest)
    acceptance_path = tmp_path / "acceptance.json"
    plan_path = tmp_path / "LAUNCH_PLAN.json"
    campaign.atomic_json(acceptance_path, acceptance)
    campaign.atomic_json(plan_path, plan)
    with pytest.raises(campaign.ContractError, match="paired comparisons differ"):
        REAL_LOAD_EXP15_PREREQUISITE(
            manifest,
            acceptance_path=acceptance_path,
            launch_plan_path=plan_path,
        )


def test_arm_overrides_match_preregistered_scientific_matrix(manifest):
    runs = campaign.expand_runs(manifest)
    by_arm = {}
    for arm_id in campaign.ARM_IDS:
        run = next(run for run in runs if run.arm_id == arm_id)
        contract = campaign.load_compatible_input(manifest, run)
        by_arm[arm_id] = set(campaign.scientific_overrides(manifest, run, contract))

    for overrides in by_arm.values():
        for invariant in (
            "train.steps=25000",
            "train.scheduler_total_steps=1000000",
            "train.ckpt_every=1000",
            "train.val_every=1000",
            "train.diag_every=1000",
            "train.eval_every=12500",
            "train.validation_sample_seed=1701",
            "train.weight_decay=0.001",
            "train.gain_lr=0.0003",
            "train.gain_weight_decay=0.0",
            "train.gain_loss_every=1",
            "model.max_depth=3",
            "tree.max_depth=3",
            "tree.keep_threshold=0.5",
            "tree.scorer=learned",
            "planner.decoded_metric=domain_raw",
            "planner.execute_mode=clipped",
            "planner.execute_steps=4",
            "planner.require_first_edge_improvement=true",
            "future_sets.recipe_anchor_policy=published_union",
            "future_sets.multi_step_depth=3",
            "losses.keep_balance=true",
            "losses.scheduled_sampling_p=0.25",
            "losses.scheduled_sampling_warmup=5000",
            "losses.scheduled_sampling_granularity=sequence",
            "losses.grounded_detach_self_fed_parent=true",
            "losses.multistep_depth_weights=[1.0,1.0,1.0]",
            "eval.final_episodes_per_task=5",
            "eval.seed=2718",
            "+campaign_prerequisite_binding_sha256="
            + _accepted_exp15_binding()["prerequisite_sha256"],
        ):
            assert invariant in overrides

    for arm_id in ("F", "H"):
        assert "train.lr=3e-05" in by_arm[arm_id]
        assert "losses.multistep_transition_mode=grounded_execution_v2" in by_arm[arm_id]
        assert "losses.grounded_select_action_weight=1.0" in by_arm[arm_id]
        assert "losses.grounded_select_endpoint_weight=1.0" in by_arm[arm_id]
        assert "losses.grounded_select_horizon_weight=0.25" in by_arm[arm_id]
    assert "losses.grounded_loss_latent_weight=0.25" in by_arm["F"]
    assert "losses.grounded_loss_action_weight=0.5" in by_arm["F"]
    assert "losses.grounded_loss_horizon_weight=0.25" in by_arm["F"]
    assert "losses.grounded_loss_endpoint_weight=0.5" in by_arm["F"]
    assert "losses.grounded_loss_latent_weight=0.125" in by_arm["H"]
    assert "losses.grounded_loss_action_weight=0.25" in by_arm["H"]
    assert "losses.grounded_loss_horizon_weight=0.125" in by_arm["H"]
    assert "losses.grounded_loss_endpoint_weight=0.25" in by_arm["H"]


def test_published_union_counts_and_full_100m_settings_are_locked(manifest):
    assert [setting["id"] for setting in manifest["settings"]] == list(campaign.SETTING_IDS)
    for setting in manifest["settings"]:
        assert (
            setting["published_union_train_anchors"],
            setting["published_union_validation_anchors"],
        ) == campaign.EXPECTED_UNION_COUNTS[setting["id"]]
    by_id = {setting["id"]: setting for setting in manifest["settings"]}
    assert by_id["puzzle-4x4-100m"]["dataset_kind"] == "sharded_100m_full"
    assert by_id["cube-quadruple-100m"]["expected_shards"] == 100


def test_inference_profile_is_explicit_and_prospectively_locked(manifest):
    assert campaign.inference_profile(manifest) == {
        "scorer": "learned",
        "require_first_edge_improvement": True,
    }
    changed = copy.deepcopy(manifest)
    changed["inference_choice"]["profile"] = "bfs_guard_off"
    with pytest.raises(campaign.ContractError, match="inference profile"):
        campaign.validate_manifest(changed)
    broken = copy.deepcopy(changed)
    broken["inference_choice"]["profiles"]["bfs_guard_off"]["scorer"] = "learned"
    with pytest.raises(campaign.ContractError, match="profile definitions"):
        campaign.validate_manifest(broken)


def test_actual_eval_bank_is_identical_and_exact(manifest):
    bank = campaign.actual_evaluation_bank(manifest)
    assert bank["episodes_per_task"] == 5
    assert bank["task_ids"] == [1, 2, 3, 4, 5]
    assert bank["seeds"][0] == [2718, 2719, 2720, 2721, 2722]
    assert bank["seeds"][4] == [6718, 6719, 6720, 6721, 6722]
    assert len({value for row in bank["seeds"] for value in row}) == 25
    assert len(bank["sha256"]) == 64


def test_launch_binds_source_runtime_config_data_recipe_and_eval_bank(manifest):
    run = campaign.expand_runs(manifest)[0]
    launch = campaign.trainer_command(manifest, run, repo_root=REPO_ROOT)
    hashes = launch["hashes"]
    for key in (
        "source_sha256",
        "runtime_sha256",
        "package_protocol_sha256",
        "config_sha256",
        "run_protocol_sha256",
        "input_contract_sha256",
        "data_manifest_sha256",
        "normalizer_sha256",
        "train_manifest_sha256",
        "validation_manifest_sha256",
        "calibration_sha256",
        "future_recipe_sha256",
        "recipe_code_sha256",
        "recipe_runtime_sha256",
        "evaluation_seed_tables_sha256",
        "final_seed_table_sha256",
        "actual_evaluation_bank_sha256",
        "exp15_prerequisite_sha256",
    ):
        assert len(hashes[key]) == 64
    assert launch["formal_validation"] is False
    assert launch["environment"]["TREEWM_CODE_SHA256"] == hashes["source_sha256"]
    assert launch["environment"]["TREEWM_RECIPE_CODE_SHA256"] == hashes["recipe_code_sha256"]
    assert hashes["source_sha256"] != hashes["recipe_code_sha256"]
    assert launch["environment"]["TREEWM_EXPECTED_FINAL_SEED_TABLE_SHA256"] == hashes["final_seed_table_sha256"]
    prerequisite = _accepted_exp15_binding()
    assert launch["exp15_prerequisite"] == prerequisite
    assert hashes["exp15_prerequisite_sha256"] == prerequisite["prerequisite_sha256"]
    assert launch["environment"]["TREEWM_PREREQUISITE_BINDING_SHA256"] == prerequisite["prerequisite_sha256"]
    assert launch["environment"]["TREEWM_EXP15_PREREQUISITE_SHA256"] == prerequisite["prerequisite_sha256"]


def test_protocol_slurm_resources_and_lifecycle_are_locked(manifest):
    assert campaign.verify_protocol_lock(CAMPAIGN_DIR) == (
        CAMPAIGN_DIR / "protocol.sha256"
    ).read_text(encoding="utf-8").strip()
    submit.validate_slurms(CAMPAIGN_DIR)
    pilot = (CAMPAIGN_DIR / "bridge.slurm").read_text(encoding="utf-8")
    assert pilot.splitlines().count("#SBATCH --array=0-39%40") == 1
    assert pilot.splitlines().count("#SBATCH --gpus-per-node=1") == 1
    assert pilot.splitlines().count("#SBATCH --cpus-per-task=12") == 1
    assert pilot.splitlines().count("#SBATCH --mem=64G") == 1
    assert '"$SCONTROL" requeue "$REQUEUE_TARGET"' in pilot
    assert "CANCEL_REQUESTED" in pilot and "READY_FOR_REQUEUE.json" in pilot
    assert "TREEWM_EXPECTED_EXP15_PREREQUISITE_SHA256" in pilot
    assert 'campaign.py" snapshot' in pilot
    report_slurm = (CAMPAIGN_DIR / "report.slurm").read_text(encoding="utf-8")
    assert "report.py" in report_slurm and "--publish" in report_slurm
    assert manifest["paths"]["run_root"].endswith("/outputs/treewm-grounded-repair-all-ten-bridge-v1")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["scientific_contract"].update(optimizer_updates=25_001),
        lambda value: value["scientific_contract"].update(scheduler_total_steps=25_000),
        lambda value: value["scientific_contract"].update(validation_sample_seed=1702),
        lambda value: value["scientific_contract"].update(keep_threshold=0.42),
        lambda value: value["arms"][0].update(world_lr=1e-4),
        lambda value: value["arms"][1].update(grounded_select_endpoint_weight=0.5),
        lambda value: value["design"].update(selection_order=["H", "F"]),
        lambda value: value["acceptance"].update(settings_per_arm_required=9),
        lambda value: value["acceptance"].update(min_arm_scientific_runs_passing=17),
        lambda value: value["acceptance"].update(min_arm_runs_with_positive_progress=11),
        lambda value: value["prerequisite"].update(required=False),
        lambda value: value["prerequisite"].update(source_sha256="0" * 64),
        lambda value: value["execution"].update(array="0-39%1"),
        lambda value: value["execution"].update(memory_per_task="1T"),
        lambda value: value["compatible_v2_recipe_input"].update(read_only=False),
    ],
)
def test_manifest_rejects_scientific_or_execution_drift(manifest, mutation):
    changed = copy.deepcopy(manifest)
    mutation(changed)
    with pytest.raises(campaign.ContractError):
        campaign.validate_manifest(changed)


def test_corrupt_existing_launch_identity_fails_closed(tmp_path):
    path = tmp_path / "launch.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(campaign.ContractError, match="unreadable"):
        worker.read_optional_json(path)


def _valid_exp16_checkpoint():
    hashes = {
        "source_sha256": "1" * 64,
        "runtime_sha256": "2" * 64,
        "recipe_code_sha256": "3" * 64,
        "recipe_runtime_sha256": "4" * 64,
        "package_protocol_sha256": "5" * 64,
        "config_sha256": "6" * 64,
        "input_contract_sha256": "7" * 64,
        "exp15_prerequisite_sha256": "8" * 64,
        "data_manifest_sha256": "9" * 64,
        "calibration_sha256": "a" * 64,
        "future_recipe_sha256": "b" * 64,
        "evaluation_seed_tables_sha256": "c" * 64,
        "final_seed_table_sha256": "d" * 64,
        "run_protocol_sha256": "e" * 64,
    }
    launch = {
        "run": {
            "run_name": "bridge-scene-armf-seed102",
            "setting_id": "scene",
            "seed": 102,
            "arm_id": "F",
        },
        "hashes": hashes,
        "launch_sha256": "f" * 64,
    }
    identity = {
        "world_size": 1,
        "run_name": launch["run"]["run_name"],
        "setting": launch["run"]["setting_id"],
        "seed": launch["run"]["seed"],
        "objective_version": "treewm_v2_grounded_repair_pilot_v1",
        "total_steps": 25_000,
        "scheduler_total_steps": 1_000_000,
        "protocol_sha256": hashes["run_protocol_sha256"],
        "code_sha256": hashes["source_sha256"],
        "runtime_sha256": hashes["runtime_sha256"],
        "recipe_code_sha256": hashes["recipe_code_sha256"],
        "recipe_runtime_sha256": hashes["recipe_runtime_sha256"],
        "campaign_source_sha256": hashes["source_sha256"],
        "campaign_protocol_sha256": hashes["package_protocol_sha256"],
        "campaign_config_sha256": hashes["config_sha256"],
        "campaign_input_contract_sha256": hashes["input_contract_sha256"],
        "campaign_prerequisite_binding_sha256": hashes[
            "exp15_prerequisite_sha256"
        ],
        "campaign_factorial_arm": launch["run"]["arm_id"],
        "data_manifest_sha256": hashes["data_manifest_sha256"],
        "calibration_sha256": hashes["calibration_sha256"],
        "future_recipe_sha256": hashes["future_recipe_sha256"],
        "evaluation_seed_tables_sha256": hashes[
            "evaluation_seed_tables_sha256"
        ],
        "final_seed_table_sha256": hashes["final_seed_table_sha256"],
    }
    generator_state = torch.Generator().get_state()
    rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "torch_cuda": [generator_state.clone()],
    }
    completed = 3
    payload = {
        "schema_version": 2,
        "model": {},
        "optimizer": {
            "state": {0: {"step": torch.tensor(float(completed))}},
            "param_groups": [{"params": [0], "lr": 1e-3}],
        },
        "scheduler": {
            "last_epoch": completed,
            "base_lrs": [1e-3],
            "_last_lr": [1e-3],
            "_step_count": completed + 1,
        },
        "scaler": None,
        "step": completed,
        "epoch": 1,
        "completed_updates": completed,
        "next_step": completed,
        "config": {
            "campaign_source_sha256": hashes["source_sha256"],
            "campaign_protocol_sha256": hashes["package_protocol_sha256"],
            "campaign_config_sha256": hashes["config_sha256"],
            "campaign_prerequisite_binding_sha256": hashes[
                "exp15_prerequisite_sha256"
            ],
        },
        "rng_state": copy.deepcopy(rng_state),
        "run_identity": identity,
        "identity_sha256": campaign.stable_hash(identity),
        "rank_states": [
            {
                "rank": 0,
                "rng_state": copy.deepcopy(rng_state),
                "loader": {
                    "epoch": 1,
                    "batches_yielded_in_epoch": 2,
                    "epoch_generator_state": generator_state.clone(),
                },
                "rng_streams": {
                    "planner": generator_state.clone(),
                    "eval": generator_state.clone(),
                    "viz": generator_state.clone(),
                },
                "horizon_generator": generator_state.clone(),
            }
        ],
        "checkpoint_manager": {"best_success": 0.0, "best_val_loss": 1.0},
        "normalizer": {},
        "latent_index": None,
        "pending_eval_step": None,
        "final_eval": None,
        "phase": "train",
        "gradient_checkpointing": True,
        "reason": "graceful-stop:SIGUSR1",
    }
    return launch, payload


def test_exp16_worker_accepts_complete_exact_resume_checkpoint(monkeypatch, tmp_path):
    launch, payload = _valid_exp16_checkpoint()
    path = tmp_path / "latest.pt"
    path.write_bytes(b"synthetic-checkpoint")
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: payload)
    verified = worker.verify_checkpoint(path, launch)
    assert verified["status"] == "checkpoint_verified"
    assert verified["identity_sha256"] == payload["identity_sha256"]
    assert verified["completed_updates"] == 3


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.__setitem__("optimizer", {}),
        lambda value: value.__setitem__("scheduler", {}),
        lambda value: value.__setitem__("identity_sha256", "not-a-hash"),
        lambda value: value["rng_state"].__setitem__("torch", 7),
        lambda value: value["rank_states"][0]["rng_streams"].__setitem__(
            "planner", 7
        ),
    ],
    ids=(
        "empty-optimizer",
        "empty-scheduler",
        "forged-identity-hash",
        "malformed-global-rng",
        "malformed-generator",
    ),
)
def test_exp16_worker_rejects_malformed_exact_resume_checkpoint(
    monkeypatch, tmp_path, mutation
):
    launch, payload = _valid_exp16_checkpoint()
    mutation(payload)
    path = tmp_path / "latest.pt"
    path.write_bytes(b"synthetic-checkpoint")
    monkeypatch.setattr(torch, "load", lambda *_args, **_kwargs: payload)
    with pytest.raises(campaign.ContractError, match="exact-resume payload is invalid"):
        worker.verify_checkpoint(path, launch)


def _synthetic_records(
    manifest,
    *,
    full_successes: float = 1.0,
    half_successes: float = 1.0,
    full_progress: float = 0.20,
    half_progress: float = 0.20,
):
    records = []
    for run in campaign.expand_runs(manifest):
        successes = full_successes if run.arm_id == "F" else half_successes
        progress = full_progress if run.arm_id == "F" else half_progress
        records.append({
            "setting_id": run.setting_id,
            "arm_id": run.arm_id,
            "seed": run.seed,
            "integrity_pass": True,
            "scientific_pass": True,
            "metrics": {
                "final": {
                    "successes": successes,
                    "success_rate": successes / 25.0,
                    "distance_reduction_frac": progress,
                },
                "midpoint": {},
            },
        })
    return records


def test_report_prefers_full_when_both_global_arms_clear_every_gate(manifest):
    result = report.aggregate_acceptance(manifest, _synthetic_records(manifest))
    assert result["accepted"] is True
    assert result["status"] == "selected_full_for_fresh_formal_campaign_design"
    assert result["formal_validation"] is False
    assert result["selected_arm"] == "F"
    assert result["selected_recipe"]["grounded_loss_scale"] == 1.0
    assert result["aggregate_gates"]["full_eligible"] is True
    assert result["aggregate_gates"]["half_eligible"] is True


def test_report_allows_half_only_as_noninferior_global_fallback(manifest):
    records = _synthetic_records(manifest)
    for record in records:
        if record["arm_id"] == "F" and record["setting_id"] == "scene":
            record["scientific_pass"] = False
    result = report.aggregate_acceptance(manifest, records)
    assert result["accepted"] is True
    assert result["selected_arm"] == "H"
    assert result["status"] == "selected_half_for_fresh_formal_campaign_design"
    assert result["selected_recipe"]["grounded_loss_scale"] == 0.5
    assert result["aggregate_gates"]["full_eligible"] is False
    assert result["aggregate_gates"]["half_eligible"] is True


def test_report_rejects_missing_integrity_and_all_zero_success(manifest):
    records = _synthetic_records(manifest)
    assert report.aggregate_acceptance(manifest, records[:-1])["accepted"] is False

    records[0]["integrity_pass"] = False
    assert report.aggregate_acceptance(manifest, records)["accepted"] is False

    all_zero = _synthetic_records(manifest, full_successes=0.0, half_successes=0.0)
    result = report.aggregate_acceptance(manifest, all_zero)
    assert result["accepted"] is False
    assert result["selected_arm"] is None
    assert result["arm_outcome_gates"]["F"]["not_all_zero_success"] is False
    assert result["arm_outcome_gates"]["H"]["not_all_zero_success"] is False


def test_report_rejects_half_when_it_regresses_full_outcomes(manifest):
    records = _synthetic_records(
        manifest,
        full_successes=2.0,
        half_successes=1.0,
        full_progress=0.20,
        half_progress=0.10,
    )
    for record in records:
        if record["arm_id"] == "F" and record["setting_id"] == "scene":
            record["scientific_pass"] = False
    result = report.aggregate_acceptance(manifest, records)
    assert result["accepted"] is False
    assert result["selected_arm"] is None
    assert result["aggregate_gates"]["half_success_noninferior_to_full"] is False
    assert result["aggregate_gates"]["half_distance_reduction_noninferior_to_full"] is False


def test_report_accepts_exact_evidence_backed_scientific_boundary(manifest):
    records = _synthetic_records(manifest)
    failed = {("scene", 102), ("puzzle-3x3", 102)}
    for record in records:
        if record["arm_id"] == "F" and (record["setting_id"], record["seed"]) in failed:
            record["scientific_pass"] = False
    result = report.aggregate_acceptance(manifest, records)
    assert result["selected_arm"] == "F"
    assert result["arm_scientific_runs_passing"]["F"] == 18
    assert sum(result["arm_setting_pass"]["F"].values()) == 8
    assert all(count >= 1 for count in result["arm_setting_seed_pass_count"]["F"].values())
    assert all(result["arm_scientific_quorum_gates"]["F"].values())
    assert result["aggregate_gates"]["full_scientific_quorum"] is True


def test_report_rejects_zero_covered_setting_even_at_18_of_20(manifest):
    records = _synthetic_records(manifest)
    for record in records:
        if record["setting_id"] == "scene":
            record["scientific_pass"] = False
    result = report.aggregate_acceptance(manifest, records)
    assert result["accepted"] is False
    assert result["arm_scientific_runs_passing"] == {"F": 18, "H": 18}
    for arm in ("F", "H"):
        gates = result["arm_scientific_quorum_gates"][arm]
        assert gates["scientific_runs_at_least_18_of_20"] is True
        assert gates["both_seed_settings_at_least_8_of_10"] is True
        assert gates["every_setting_has_at_least_one_passing_seed"] is False


def test_report_requires_12_positive_progress_runs(manifest):
    records = _synthetic_records(manifest)
    by_arm = {arm: 0 for arm in ("F", "H")}
    for record in records:
        arm = record["arm_id"]
        by_arm[arm] += 1
        if by_arm[arm] > 11:
            record["metrics"]["final"]["distance_reduction_frac"] = 0.0
    result = report.aggregate_acceptance(manifest, records)
    assert result["accepted"] is False
    assert result["arm_outcome_gates"]["F"]["positive_mean_progress"] is True
    assert result["arm_outcome_gates"]["F"]["positive_progress_run_quorum"] is False

    records = _synthetic_records(manifest)
    seen = 0
    for record in records:
        if record["arm_id"] == "F":
            seen += 1
            if seen > 12:
                record["metrics"]["final"]["distance_reduction_frac"] = 0.0
    result = report.aggregate_acceptance(manifest, records)
    assert result["selected_arm"] == "F"
    assert result["aggregate_metrics"]["by_arm"]["F"]["runs_with_positive_progress"] == 12


def test_selected_recipe_artifact_is_self_hashed_and_binds_20_configs(
    manifest, monkeypatch, tmp_path
):
    def fake_launch(_manifest, run, **_kwargs):
        prerequisite = _accepted_exp15_binding()
        return {
            "run": {"arm_id": run.arm_id},
            "exp15_prerequisite": prerequisite,
            "hashes": {
                "source_sha256": "1" * 64,
                "runtime_sha256": "2" * 64,
                "package_protocol_sha256": "3" * 64,
                "actual_evaluation_bank_sha256": "4" * 64,
                "config_sha256": f"{run.index + 1:064x}",
                "input_contract_sha256": f"{run.setting_index + 101:064x}",
                "exp15_prerequisite_sha256": prerequisite["prerequisite_sha256"],
            },
        }

    monkeypatch.setattr(report, "trainer_command", fake_launch)
    sealed, artifact = report.seal_report(
        manifest,
        report.aggregate_acceptance(manifest, _synthetic_records(manifest)),
        repo_root=REPO_ROOT,
    )
    assert artifact["selected"] is True
    assert artifact["selected_arm"] == "F"
    assert sealed["exp15_prerequisite"] == _accepted_exp15_binding()
    assert sealed["provenance"]["exp15_prerequisite"] == _accepted_exp15_binding()
    assert artifact["exp15_prerequisite"] == _accepted_exp15_binding()
    assert artifact["exp15_prerequisite_sha256"] == _accepted_exp15_binding()["prerequisite_sha256"]
    assert artifact["bridge_acceptance_sha256"] == sealed["report_sha256"]
    acceptance_bytes = (
        json.dumps(sealed, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    assert artifact["acceptance_sha256"] == hashlib.sha256(acceptance_bytes).hexdigest()
    acceptance_path = tmp_path / "acceptance.json"
    campaign.atomic_json(acceptance_path, sealed)
    assert artifact["acceptance_sha256"] == campaign.file_sha256(acceptance_path)
    assert len(artifact["selected_run_config_sha256"]) == 20
    artifact_body = dict(artifact)
    artifact_hash = artifact_body.pop("artifact_sha256")
    assert artifact_hash == campaign.stable_hash(artifact_body)
    report_body = dict(sealed)
    report_hash = report_body.pop("report_sha256")
    assert report_hash == campaign.stable_hash(report_body)


def test_launch_plan_has_one_40_element_gpu_array_and_dependent_report(manifest, monkeypatch):
    monkeypatch.setattr(
        submit,
        "verify_all",
        lambda *_args, **_kwargs: {
            "status": "verified",
            "exp15_prerequisite": {"prerequisite_sha256": "6" * 64},
        },
    )
    launches = []
    for run in campaign.expand_runs(manifest):
        launches.append({
            "launch_sha256": f"{run.index + 1:064x}",
            "hashes": {
                "source_sha256": "1" * 64,
                "runtime_sha256": "2" * 64,
                "package_protocol_sha256": "3" * 64,
                "actual_evaluation_bank_sha256": "4" * 64,
                "final_seed_table_sha256": "5" * 64,
                "exp15_prerequisite_sha256": "6" * 64,
                "config_sha256": f"{run.index + 101:064x}",
            },
        })
    iterator = iter(launches)
    monkeypatch.setattr(submit, "trainer_command", lambda *_args, **_kwargs: next(iterator))
    plan = submit.launch_plan(
        manifest,
        REPO_ROOT,
        verify_files=False,
        inspect_scheduler=False,
    )
    assert plan["dag"] == [
        {"name": "bridge", "kind": "gpu_array", "elements": 40, "dependency": None},
        {"name": "strict_report", "kind": "cpu_report", "dependency": "bridge"},
    ]
    assert len(plan["runs"]) == 40
    assert plan["common_hashes"]["exp15_prerequisite_sha256"] == "6" * 64
    assert plan["formal_validation"] is False


def test_objective_is_bounded_at_exactly_25k():
    train = importlib.import_module("scripts.train")
    train.validate_objective_version("treewm_v2_grounded_repair_pilot_v1", 25_000)
    with pytest.raises(ValueError, match="25000-update cap"):
        train.validate_objective_version("treewm_v2_grounded_repair_pilot_v1", 25_001)
