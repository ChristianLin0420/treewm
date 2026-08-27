from __future__ import annotations

from pathlib import Path

import pytest

import aggregate
import campaign
import final_eval
import worker


def test_completion_precedes_racing_cancel() -> None:
    assert worker.classify_child_exit(0, cancel_requested=True, cancel_latch_exists=True, requeue_requested=True) == "complete"
    text = (campaign.CAMPAIGN_DIR / "train.slurm").read_text(encoding="utf-8")
    assert text.index('if [[ "$status" -eq 0 ]]') < text.index('if [[ -e "$CANCEL_LATCH" || "$status" -eq 143 ]]')

    final_text = (campaign.CAMPAIGN_DIR / "final_eval.slurm").read_text(encoding="utf-8")
    assert final_text.index('if [[ "$status" -eq 0 ]]') < final_text.index('if [[ -e "$CANCEL_LATCH" || "$status" -eq 143 ]]')
    assert 'touch "$REQUEUE_LATCH"' in final_text
    assert 'kill -USR1 "$step_pid"' in final_text
    assert 'kill -TERM "$step_pid"' in final_text
    assert final_text.index("step_pid=$!") < final_text.index('if [[ -e "$CANCEL_LATCH" ]]', final_text.index("step_pid=$!"))


def test_signal_state_forwards_latched_intent(tmp_path: Path) -> None:
    class Child:
        def __init__(self) -> None:
            self.signals: list[int] = []
        def poll(self):
            return None
        def send_signal(self, signum: int) -> None:
            self.signals.append(signum)
    state = worker.SignalState(tmp_path)
    state.requeue_requested = True
    child = Child()
    state.child = child  # type: ignore[assignment]
    state.forward_latched_intent()
    assert child.signals == [worker.signal.SIGUSR1]


def test_checkpoint_rank_state_requires_metric_tracker_and_rng_streams() -> None:
    from treewm.logging.metrics import MetricTracker
    state = {
        "run_identity": {"world_size": 1},
        "rank_states": [{
            "rank": 0,
            "rng_state": {"python": 1},
            "rng_streams": {"planner": [1], "eval": [2], "viz": [3]},
            "horizon_generator": [1],
            "loader": {"epoch": 0, "batches_yielded_in_epoch": 1, "epoch_generator_state": [1]},
            "metric_tracker": MetricTracker().state_dict(),
        }],
        "checkpoint_manager": {"best_success": 0, "best_val_loss": 1},
    }
    assert worker._rank_state_complete(state)
    del state["rank_states"][0]["rng_streams"]["eval"]
    assert not worker._rank_state_complete(state)


def test_worker_requires_explicit_durable_post_update_cadence() -> None:
    complete = {
        "post_update_cadence": {
            "schema_version": 1,
            "committed_update": 50,
            "completed_update": 50,
            "replay_action": None,
        }
    }
    assert worker._validated_post_update_cadence(complete, 50) == complete["post_update_cadence"]

    replayable = {
        "post_update_cadence": {
            "schema_version": 1,
            "committed_update": 1_000_000,
            "completed_update": 999_999,
            "replay_action": "visualization",
        }
    }
    assert worker._validated_post_update_cadence(replayable, 1_000_000) == replayable["post_update_cadence"]

    with pytest.raises(campaign.ContractError, match="post-update cadence is invalid"):
        worker._validated_post_update_cadence({}, 50)
    incomplete = {
        "post_update_cadence": {
            "schema_version": 1,
            "committed_update": 50,
            "completed_update": 49,
            "replay_action": None,
        }
    }
    with pytest.raises(campaign.ContractError, match="without replay intent"):
        worker._validated_post_update_cadence(incomplete, 50)


def test_stage_marker_requires_complete_target_cadence() -> None:
    complete = {
        "post_update_cadence": {
            "schema_version": 1,
            "committed_update": 25_000,
            "completed_update": 25_000,
            "replay_action": None,
        }
    }
    worker._require_complete_stage_cadence(complete, 25_000)
    replayable = {
        "post_update_cadence": {
            "schema_version": 1,
            "committed_update": 25_000,
            "completed_update": 24_999,
            "replay_action": "visualization",
        }
    }
    with pytest.raises(campaign.ContractError, match="cadence is incomplete"):
        worker._require_complete_stage_cadence(replayable, 25_000)


def test_final_seed_task_table_is_exact_and_paired() -> None:
    manifest = campaign.load_manifest()
    full = campaign.load_seed_table(manifest)["settings"]["scene"]
    task = final_eval.single_task_seed_table(full, 3)
    assert task["task_ids"] == [3]
    assert len(task["seeds"][0]) == 50
    assert len(set(task["seeds"][0])) == 50


def test_seed_level_paired_t_summary_requires_four_replicates() -> None:
    summary = aggregate.seed_t_summary([0.1, 0.2, 0.3, 0.4])
    assert summary["n"] == 4
    assert summary["ci95_lower"] < summary["mean"] < summary["ci95_upper"]
    with pytest.raises(campaign.ContractError, match="exactly four"):
        aggregate.seed_t_summary([0.1, 0.2, 0.3])


def test_protocol_inventory_covers_lifecycle_and_tests() -> None:
    required = {
        "manifest.json", "prerequisite_bindings.json", "bind_prerequisites.py",
        "raw_exp20_recompute.py",
        "worker.py", "stage_gate.py", "final_eval.py",
        "aggregate.py", "tests/conftest.py", "tests/test_formal_campaign.py",
        "tests/test_stage_gate.py", "tests/test_lifecycle_and_eval.py",
    }
    assert required.issubset(set(campaign.PROTOCOL_FILES))
    assert "metric_boundary.py" not in campaign.PROTOCOL_FILES
