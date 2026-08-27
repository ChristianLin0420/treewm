from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import sys

import pytest
import torch

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

import aggregate
import campaign
import final_eval
import worker


def test_planner_generator_resume_matches_uninterrupted_suffix() -> None:
    uninterrupted = torch.Generator(device="cpu").manual_seed(913)
    first = torch.rand(37, generator=uninterrupted)
    state = final_eval.encode_generator_state(uninterrupted)
    expected_suffix = torch.rand(53, generator=uninterrupted)

    resumed = torch.Generator(device="cpu").manual_seed(1)
    final_eval.restore_generator_state(resumed, state)
    actual_suffix = torch.rand(53, generator=resumed)
    assert torch.equal(expected_suffix, actual_suffix)
    assert len(first) == 37


def test_progress_resume_requires_self_hash_and_exact_seed_prefix() -> None:
    table = {
        "seeds": [[11, 12, 13]],
    }
    identity = {"task_id": 3, "token": "sealed"}
    progress = {
        "schema_version": 1,
        "status": "in_progress",
        "identity": identity,
        "rails": {
            "learned": {
                "episodes": [
                    {"task_id": 3, "episode_index": 0, "episode_seed": 11}
                ],
                "metrics": None,
                "planner_generator_state": [1, 2, 3],
            },
            "bfs": {"episodes": [], "metrics": None},
        },
    }
    final_eval.seal_progress(progress)
    assert final_eval.validate_progress(progress, identity, table) == progress

    corrupted = copy.deepcopy(progress)
    corrupted["rails"]["learned"]["episodes"][0]["episode_seed"] = 12
    with pytest.raises(campaign.ContractError, match="self-hash"):
        final_eval.validate_progress(corrupted, identity, table)

    corrupted = copy.deepcopy(progress)
    corrupted["rails"]["learned"]["episodes"][0]["episode_seed"] = 12
    final_eval.seal_progress(corrupted)
    with pytest.raises(campaign.ContractError, match="locked seed prefix"):
        final_eval.validate_progress(corrupted, identity, table)


def test_single_task_table_is_locked_final_split() -> None:
    manifest = campaign.load_manifest()
    bundle = campaign.load_seed_table(manifest)
    full = bundle["settings"]["scene"]
    task = final_eval.single_task_seed_table(full, 3)
    assert task["split"] == "final"
    assert task["task_ids"] == [3]
    assert task["seeds"] == [full["seeds"][2]]
    assert len(task["seeds"][0]) == 50
    assert "expected_episode_seed_split=\"final\"" in (campaign.CAMPAIGN_DIR / "final_eval.py").read_text(encoding="utf-8")


def test_sorted_json_result_round_trip_accepts_exact_rail_set(tmp_path: Path) -> None:
    manifest = campaign.load_manifest()
    full = campaign.load_seed_table(manifest)["settings"]["scene"]
    task = final_eval.single_task_seed_table(full, 1)
    identity = {"rails": ["learned", "bfs"], "token": "exact"}
    identity["eval_contract_sha256"] = campaign.stable_hash(identity)
    rails = {}
    for rail in ("learned", "bfs"):
        rails[rail] = {
            "episodes": [
                {"task_id": 1, "episode_index": index, "episode_seed": seed, "success": False}
                for index, seed in enumerate(task["seeds"][0])
            ],
            "metrics": {"eval/num_episodes": 50.0},
        }
    result = {"schema_version": 1, "status": "complete", "identity": identity, "rails": rails}
    result["result_sha256"] = campaign.stable_hash(result)
    path = tmp_path / "result.json"
    campaign.atomic_json(path, result)
    assert list(json.loads(path.read_text(encoding="utf-8"))["rails"]) == ["bfs", "learned"]
    assert final_eval._validate_existing_result(path, identity, task)["status"] == "complete"


def test_seed_level_t_interval_exact_statistics() -> None:
    result = aggregate.seed_t_summary([0.1, 0.2, 0.3, 0.4])
    expected_sd = math.sqrt(0.05 / 3.0)
    expected_half = 3.182446 * expected_sd / 2.0
    assert result["mean"] == pytest.approx(0.25)
    assert result["sample_sd"] == pytest.approx(expected_sd)
    assert result["degrees_of_freedom"] == 3
    assert result["ci95_lower"] == pytest.approx(0.25 - expected_half)
    assert result["ci95_upper"] == pytest.approx(0.25 + expected_half)
    with pytest.raises(campaign.ContractError, match="four"):
        aggregate.seed_t_summary([0.1, 0.2, 0.3])


def test_aggregate_rejects_fractional_reported_success_count() -> None:
    episode_successes = [True, *([False] * 49)]
    assert aggregate.validate_episode_success_accounting(
        episode_successes, 1.0, "unit"
    ) == episode_successes
    with pytest.raises(campaign.ContractError, match="disagree"):
        aggregate.validate_episode_success_accounting(
            episode_successes, 1.5, "unit"
        )
    aggregate.validate_exact_episode_count(50.0, 50, "unit")
    with pytest.raises(campaign.ContractError, match="episode count differs"):
        aggregate.validate_exact_episode_count(50.5, 50, "unit")


def test_aggregate_requires_exact_final_identity_without_extensions() -> None:
    manifest = campaign.load_manifest()
    spec = campaign.eval_at(manifest, 0)
    bundle = campaign.load_seed_table(manifest)
    seed_table = bundle["settings"][spec.run.setting_id]
    launch = {
        "launch_sha256": campaign.stable_hash({"launch": 0}),
        "hashes": {
            "final_seed_table_sha256": seed_table["sha256"],
            "package_seed_table_sha256": bundle["sha256"],
            "package_protocol_sha256": campaign.stable_hash({"protocol": 1}),
            "source_sha256": campaign.stable_hash({"source": 1}),
            "evaluation_source_sha256": campaign.stable_hash({"eval_source": 1}),
            "runtime_sha256": campaign.stable_hash({"runtime": 1}),
            "prerequisite_binding_sha256": campaign.stable_hash({"binding": 1}),
            "selected_recipe_sha256": campaign.stable_hash({"recipe": 1}),
            "selected_arm": "F",
        },
    }
    gate_row = {
        "launch_sha256": launch["launch_sha256"],
        "identity_sha256": campaign.stable_hash({"checkpoint_identity": 1}),
        "checkpoint_sha256": campaign.stable_hash({"checkpoint": 1}),
        "final_seed_table_sha256": seed_table["sha256"],
    }
    task_table = final_eval.single_task_seed_table(seed_table, spec.task_id)
    identity = final_eval._progress_identity(
        manifest, spec, launch, gate_row, task_table
    )
    rails = {
        rail: {
            "episodes": [
                {
                    "task_id": spec.task_id,
                    "episode_index": episode,
                    "episode_seed": seed,
                    "success": False,
                }
                for episode, seed in enumerate(task_table["seeds"][0])
            ],
            "metrics": {
                "eval/num_episodes": 50.0,
                "eval/successes": 0.0,
                "eval/success_rate": 0.0,
                "eval/distance_reduction_frac": 0.0,
            },
        }
        for rail in ("learned", "bfs")
    }
    result = {
        "schema_version": 1,
        "status": "complete",
        "identity": identity,
        "rails": rails,
    }
    result["result_sha256"] = campaign.stable_hash(result)
    aggregate._validate_result(
        result,
        manifest=manifest,
        spec=spec,
        launch=launch,
        gate_row=gate_row,
        seed_table=seed_table,
    )

    for mutation in ("wrong_campaign", "extra_key"):
        changed = copy.deepcopy(result)
        if mutation == "wrong_campaign":
            changed["identity"]["campaign_id"] = "another-campaign"
        else:
            changed["identity"]["unregistered_extension"] = True
        identity_body = dict(changed["identity"])
        identity_body.pop("eval_contract_sha256", None)
        changed["identity"]["eval_contract_sha256"] = campaign.stable_hash(identity_body)
        changed.pop("result_sha256")
        changed["result_sha256"] = campaign.stable_hash(changed)
        with pytest.raises(campaign.ContractError, match="identity differs"):
            aggregate._validate_result(
                changed,
                manifest=manifest,
                spec=spec,
                launch=launch,
                gate_row=gate_row,
                seed_table=seed_table,
            )


def test_worker_delegates_to_shared_exact_resume_validator(monkeypatch, tmp_path: Path) -> None:
    checkpoint = tmp_path / "latest.pt"
    checkpoint.write_bytes(b"not-used")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: {"run_identity": {"world_size": 1}})

    validator_call = {}

    def reject(*args, **kwargs):
        validator_call.update(kwargs)
        raise ValueError("optimizer tensor state malformed")

    monkeypatch.setattr("treewm.utils.checkpoint.validate_exact_resume_payload", reject)
    with pytest.raises(campaign.ContractError, match="optimizer tensor state malformed"):
        worker.verify_checkpoint(checkpoint, {"run": {}, "hashes": {}})
    assert validator_call["expected_world_size"] == 1
    assert validator_call["require_cuda_rng"] is True


def test_prior_boundary_is_checked_only_on_first_stage_invocation(tmp_path: Path) -> None:
    path = tmp_path / "STAGE_LAUNCH_25000.json"
    value = {
        "schema_version": 1,
        "status": "stage_launch_sealed",
        "stage_target": 25_000,
        "previous_gate": {"stage_target": 2_000, "checkpoint_sha256": "a" * 64},
    }
    calls = []
    assert worker.seal_or_validate_stage_launch(
        path, value, validate_first_boundary=lambda: calls.append("checked")
    )
    assert calls == ["checked"]
    # A requeue can now have latest.pt beyond 2k; it validates the sealed launch and
    # must not demand that the live checkpoint still equal the prior boundary.
    assert not worker.seal_or_validate_stage_launch(
        path,
        value,
        validate_first_boundary=lambda: (_ for _ in ()).throw(AssertionError("rechecked")),
    )


def test_successful_stage_exit_wins_late_signal_race() -> None:
    assert worker.classify_child_exit(
        0,
        cancel_requested=False,
        cancel_latch_exists=False,
        requeue_requested=True,
    ) == "complete"
    assert worker.classify_child_exit(
        0,
        cancel_requested=True,
        cancel_latch_exists=True,
        requeue_requested=False,
    ) == "complete"
    assert worker.classify_child_exit(
        worker.GRACEFUL_EXIT_CODE,
        cancel_requested=False,
        cancel_latch_exists=False,
        requeue_requested=True,
    ) == "requeue"


def test_final_eval_requires_all_environment_seals() -> None:
    text = (campaign.CAMPAIGN_DIR / "final_eval.py").read_text(encoding="utf-8")
    assert "if expected is None or expected !=" in text
    for name in (
        "TREEWM_EXPECTED_SOURCE_SHA256",
        "TREEWM_EXPECTED_EVALUATION_SOURCE_SHA256",
        "TREEWM_EXPECTED_RUNTIME_SHA256",
        "TREEWM_EXPECTED_PACKAGE_PROTOCOL_SHA256",
        "TREEWM_EXPECTED_SEED_TABLE_SHA256",
        "TREEWM_EXPECTED_PREREQUISITE_BINDING_SHA256",
        "TREEWM_EXPECTED_SELECTED_RECIPE_SHA256",
    ):
        assert name in text
