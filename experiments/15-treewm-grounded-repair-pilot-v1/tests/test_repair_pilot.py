from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
import sys

import pytest


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


@pytest.fixture(scope="module")
def manifest():
    return campaign.load_manifest(CAMPAIGN_DIR / "manifest.json")


def test_exact_5x4x2_mapping_is_complete_and_fresh(manifest):
    runs = campaign.expand_runs(manifest)
    assert len(runs) == 40
    assert [run.index for run in runs] == list(range(40))
    assert {run.seed for run in runs} == {100, 101}
    assert len({run.run_name for run in runs}) == 40
    assert len({run.wandb_id for run in runs}) == 40
    assert (runs[0].setting_id, runs[0].arm_id, runs[0].seed) == (
        "antmaze-large",
        "A",
        100,
    )
    assert (runs[-1].setting_id, runs[-1].arm_id, runs[-1].seed) == (
        "cube-quadruple-100m",
        "D",
        101,
    )
    for run in runs:
        expected = ((run.setting_index * 4) + run.arm_index) * 2 + run.seed_index
        assert run.index == expected


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
        ):
            assert invariant in overrides

    assert "train.lr=3e-05" in by_arm["A"]
    assert "losses.multistep_transition_mode=teacher_action" in by_arm["A"]
    assert "losses.grounded_loss_endpoint_weight=0.0" in by_arm["A"]
    assert "train.lr=0.0001" in by_arm["B"]
    assert "losses.grounded_loss_action_weight=0.5" in by_arm["B"]
    assert "train.lr=3e-05" in by_arm["C"]
    assert "losses.grounded_select_endpoint_weight=1.0" in by_arm["C"]
    assert "losses.grounded_select_endpoint_weight=2.0" in by_arm["D"]
    assert "losses.grounded_loss_action_weight=1.0" in by_arm["D"]
    assert "losses.grounded_loss_endpoint_weight=1.0" in by_arm["D"]


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
    ):
        assert len(hashes[key]) == 64
    assert launch["formal_validation"] is False
    assert launch["environment"]["TREEWM_CODE_SHA256"] == hashes["source_sha256"]
    assert launch["environment"]["TREEWM_RECIPE_CODE_SHA256"] == hashes["recipe_code_sha256"]
    assert hashes["source_sha256"] != hashes["recipe_code_sha256"]
    assert launch["environment"]["TREEWM_EXPECTED_FINAL_SEED_TABLE_SHA256"] == hashes["final_seed_table_sha256"]


def test_protocol_slurm_resources_and_lifecycle_are_locked(manifest):
    assert campaign.verify_protocol_lock(CAMPAIGN_DIR) == (
        CAMPAIGN_DIR / "protocol.sha256"
    ).read_text(encoding="utf-8").strip()
    submit.validate_slurms(CAMPAIGN_DIR)
    pilot = (CAMPAIGN_DIR / "pilot.slurm").read_text(encoding="utf-8")
    assert pilot.splitlines().count("#SBATCH --array=0-39%40") == 1
    assert pilot.splitlines().count("#SBATCH --gpus-per-node=1") == 1
    assert pilot.splitlines().count("#SBATCH --cpus-per-task=12") == 1
    assert pilot.splitlines().count("#SBATCH --mem=64G") == 1
    assert '"$SCONTROL" requeue "$REQUEUE_TARGET"' in pilot
    assert "CANCEL_REQUESTED" in pilot and "READY_FOR_REQUEUE.json" in pilot
    assert 'campaign.py" snapshot' in pilot
    report_slurm = (CAMPAIGN_DIR / "report.slurm").read_text(encoding="utf-8")
    assert "report.py" in report_slurm and "--publish" in report_slurm
    assert manifest["paths"]["run_root"].endswith("/outputs/treewm-grounded-repair-pilot-v1")


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value["scientific_contract"].update(optimizer_updates=25_001),
        lambda value: value["scientific_contract"].update(scheduler_total_steps=25_000),
        lambda value: value["scientific_contract"].update(validation_sample_seed=1702),
        lambda value: value["scientific_contract"].update(keep_threshold=0.42),
        lambda value: value["arms"][2].update(world_lr=1e-4),
        lambda value: value["design"].update(preregistered_candidate_arm="D"),
        lambda value: value["acceptance"].update(candidate_settings_required=3),
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


def _synthetic_records(manifest, *, candidate_successes: float = 1.0):
    records = []
    for run in campaign.expand_runs(manifest):
        if run.arm_id == "C":
            successes = candidate_successes
            progress = 0.20
        elif run.arm_id == "A":
            successes = 0.0
            progress = 0.10
        else:
            successes = 0.0
            progress = 0.05
        records.append({
            "setting_id": run.setting_id,
            "arm_id": run.arm_id,
            "seed": run.seed,
            "integrity_pass": True,
            "scientific_pass": run.arm_id == "C",
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


def test_report_only_promotes_preregistered_c_and_never_claims_formal(manifest):
    result = report.aggregate_acceptance(manifest, _synthetic_records(manifest))
    assert result["accepted"] is True
    assert result["status"] == "accepted_for_fresh_formal_campaign_design"
    assert result["formal_validation"] is False
    assert result["preregistered_candidate_arm"] == "C"
    assert result["matched_control_arm"] == "A"
    assert result["sensitivity_arms_are_nonpromotable"] == ["B", "D"]


def test_report_rejects_missing_run_all_zero_and_control_regression(manifest):
    records = _synthetic_records(manifest)
    assert report.aggregate_acceptance(manifest, records[:-1])["accepted"] is False

    all_zero = _synthetic_records(manifest, candidate_successes=0.0)
    result = report.aggregate_acceptance(manifest, all_zero)
    assert result["accepted"] is False
    assert result["aggregate_gates"]["candidate_not_all_zero_success"] is False

    regression = _synthetic_records(manifest)
    for record in regression:
        if record["arm_id"] == "C":
            record["metrics"]["final"]["success_rate"] = 0.0
            record["metrics"]["final"]["distance_reduction_frac"] = 0.01
        if record["arm_id"] == "A":
            record["metrics"]["final"]["success_rate"] = 0.2
            record["metrics"]["final"]["distance_reduction_frac"] = 0.10
    result = report.aggregate_acceptance(manifest, regression)
    assert result["accepted"] is False
    assert result["aggregate_gates"]["candidate_success_noninferior_to_control"] is False
    assert result["aggregate_gates"]["candidate_distance_reduction_noninferior_to_control"] is False


def test_report_requires_both_candidate_seeds_in_four_settings(manifest):
    records = _synthetic_records(manifest)
    failed_settings = {"scene", "puzzle-3x3"}
    for record in records:
        if record["arm_id"] == "C" and record["seed"] == 101 and record["setting_id"] in failed_settings:
            record["scientific_pass"] = False
    result = report.aggregate_acceptance(manifest, records)
    assert result["accepted"] is False
    assert result["candidate_settings_passing"] == 3
    assert result["aggregate_gates"]["preregistered_candidate_setting_quorum"] is False


def test_launch_plan_has_one_40_element_gpu_array_and_dependent_report(manifest, monkeypatch):
    monkeypatch.setattr(submit, "verify_all", lambda *_args, **_kwargs: {"status": "verified"})
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
        {"name": "pilot", "kind": "gpu_array", "elements": 40, "dependency": None},
        {"name": "strict_report", "kind": "cpu_report", "dependency": "pilot"},
    ]
    assert len(plan["runs"]) == 40
    assert plan["formal_validation"] is False


def test_objective_is_bounded_at_exactly_25k():
    train = importlib.import_module("scripts.train")
    train.validate_objective_version("treewm_v2_grounded_repair_pilot_v1", 25_000)
    with pytest.raises(ValueError, match="25000-update cap"):
        train.validate_objective_version("treewm_v2_grounded_repair_pilot_v1", 25_001)
