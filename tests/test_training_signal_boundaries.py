"""Regression tests for signals delivered on deterministic cadence boundaries."""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest

from scripts import train
from scripts.train import PostUpdateCadenceState
from treewm.utils.checkpoint import StopController


def visualization_config():
    return SimpleNamespace(
        train=SimpleNamespace(
            viz_early_until=25_000,
            viz_every_early=2_000,
            viz_every=25_000,
        )
    )


@pytest.mark.parametrize("reason", ["SIGUSR1", "SIGTERM"])
def test_signal_at_2k_waits_for_telemetry_and_replays_due_visualization(
    reason: str,
) -> None:
    stop = StopController()
    cadence = PostUpdateCadenceState(1_999)
    cadence.begin(2_000)
    stop.request(reason)

    assert stop.requested and stop.reason == reason
    assert not cadence.stop_checkpoint_is_durable(None)
    for _completed_action in ("50-step-log", "diagnostics", "validation"):
        assert not cadence.stop_checkpoint_is_durable(None)

    assert train.should_visualise(2_000, visualization_config())
    cadence.mark_replay("visualization", 2_000)
    assert cadence.stop_checkpoint_is_durable(None)
    resumed = PostUpdateCadenceState.from_state_dict(
        cadence.state_dict(), require_durable=True
    )
    events: list[tuple[str, int] | tuple[str, str]] = []
    train.replay_pending_post_update_cadence(
        resumed,
        None,
        run_evaluation=lambda step: events.append(("evaluation", step)),
        run_visualization=lambda step: events.append(("visualization", step)),
        save_completion=lambda reason: events.append(("checkpoint", reason)),
    )
    assert events == [
        ("visualization", 2_000),
        ("checkpoint", "post-update-cadence-complete"),
    ]
    assert resumed.complete


@pytest.mark.parametrize(
    ("stage_target", "visualization_due"),
    [
        (25_000, False),
        (100_000, True),
        (1_000_000, True),
    ],
)
def test_interrupted_evaluation_empty_loop_resume_runs_entire_remaining_cadence(
    stage_target: int,
    visualization_due: bool,
) -> None:
    cadence = PostUpdateCadenceState(stage_target - 1)
    cadence.begin(stage_target)
    cadence.mark_replay("evaluation", stage_target)
    assert cadence.stop_checkpoint_is_durable(stage_target)

    resumed = PostUpdateCadenceState.from_state_dict(
        cadence.state_dict(), require_durable=True
    )
    assert list(range(resumed.committed_update, stage_target)) == []
    assert train.should_visualise(stage_target, visualization_config()) is visualization_due
    events: list[tuple[str, int] | tuple[str, str]] = []

    def evaluate(step: int) -> None:
        events.append(("evaluation", step))
        if visualization_due:
            resumed.mark_replay("visualization", step)
        else:
            resumed.finish(step)

    train.replay_pending_post_update_cadence(
        resumed,
        stage_target,
        run_evaluation=evaluate,
        run_visualization=lambda step: events.append(("visualization", step)),
        save_completion=lambda reason: events.append(("checkpoint", reason)),
    )

    expected: list[tuple[str, int] | tuple[str, str]] = [
        ("evaluation", stage_target)
    ]
    if visualization_due:
        expected.extend(
            [
                ("visualization", stage_target),
                ("checkpoint", "post-update-cadence-complete"),
            ]
        )
    assert events == expected
    assert resumed.complete


def test_train_loop_has_no_graceful_exit_inside_post_update_cadence() -> None:
    source = inspect.getsource(train.main.__wrapped__)
    committed = source.index("post_update_cadence.begin(completed_updates)")
    logging = source.index(
        "# ------------------------------------------------------------- logging", committed
    )
    diagnostics = source.index(
        "# --------------------------------------------------------- diagnostics", logging
    )
    validation = source.index(
        "# --------------------------------------------------------- validation", diagnostics
    )
    evaluation = source.index(
        "# ---------------------------------------------------- goal evaluation", validation
    )
    visualization = source.index(
        "# ------------------------------------------------------ visualisations", evaluation
    )
    safe_boundary = source.index(
        "# This is the sole ordinary post-update graceful-stop boundary",
        visualization,
    )

    assert committed < logging < diagnostics < validation
    assert validation < evaluation < visualization < safe_boundary
    assert "raise_if_stopping()" not in source[committed:safe_boundary]
    assert "raise_if_stopping()" in source[safe_boundary:]
