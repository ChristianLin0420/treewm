"""Fail-closed scalar identity and training-telemetry namespace regressions."""

from __future__ import annotations

import ast
from collections.abc import Iterator, Mapping
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from scripts import train
from treewm.evaluation.tree_stats import structural_summary
from treewm.logging.tensorboard import ScalarCollisionError, TreeWMLogger
from treewm.models.baselines import build_model, tree_config_for
from treewm.models.treewm import TreeWMConfig
from treewm.tree.expansion import TreeConfig
from treewm.utils.rng import make_generator


COLLIDING_VALIDATION_AUX_TAGS = (
    "bind/negative_margin_loss",
    "control/loss_metric",
    "control/loss_rank",
    "latent_gauge/future/loss",
    "latent_gauge/loss",
    "latent_gauge/root/loss",
)
STRUCTURAL_SUMMARY_TAGS = {
    "tree/max_depth",
    "tree/mean_depth",
    *(f"tree/effective_branching_factor_d{depth}" for depth in range(4)),
    *(f"tree/horizon_d{depth}" for depth in range(4)),
    *(f"tree/count_d{depth}" for depth in range(4)),
    "tree/unique_root_subtrees_explored",
    "tree/top2_root_subtree_fraction",
    *(f"tree/pairwise_endpoint_diversity_d{depth}" for depth in range(1, 4)),
    *(f"tree/pairwise_action_diversity_d{depth}" for depth in range(1, 4)),
    "tree/expansion_order_vs_depth",
}


class _RecordingWriter:
    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, value: float, step: int) -> None:
        self.scalars.append((tag, value, step))

    def close(self) -> None:
        pass


class _RecordingRun:
    def __init__(self) -> None:
        self.payloads: list[dict[str, object]] = []

    def log(self, payload: dict[str, object]) -> None:
        self.payloads.append(dict(payload))

    def finish(self, exit_code: int = 0) -> None:
        del exit_code


class _RepeatedItems(Mapping[str, float]):
    """Adversarial mapping whose item stream repeats one effective tag."""

    def __init__(self, entries: list[tuple[str, float]]) -> None:
        self.entries = entries

    def __getitem__(self, key: str) -> float:
        return next(value for name, value in reversed(self.entries) if name == key)

    def __iter__(self) -> Iterator[str]:
        return iter(dict(self.entries))

    def __len__(self) -> int:
        return len(dict(self.entries))

    def items(self):
        return iter(self.entries)


def _recording_logger(tmp_path: Path) -> tuple[TreeWMLogger, _RecordingWriter, _RecordingRun]:
    logger = TreeWMLogger(tmp_path, is_main=False)
    writer = _RecordingWriter()
    run = _RecordingRun()
    logger._writer = writer
    logger._wandb_run = run
    return logger, writer, run


def _float32_bits(value: float) -> int:
    return int(np.asarray(np.float32(value)).view(np.uint32).item())


def test_scalar_ledger_suppresses_float32_identical_and_rejects_conflict_atomically(
    tmp_path,
):
    logger, writer, run = _recording_logger(tmp_path)
    logger.scalar("train/loss_total", 1.0, 1000)

    # Numerically distinct float64 values with the same serialized float32 identity
    # are one observation and must reach neither sink twice.
    same_float32 = float(np.nextafter(np.float64(1.0), np.float64(2.0)))
    assert same_float32 != 1.0
    logger.scalars({"train/loss_total": same_float32}, 1000)
    assert writer.scalars == [("train/loss_total", 1.0, 1000)]
    assert run.payloads == [{"global_step": 1000, "train/loss_total": 1.0}]

    # The fresh entry precedes the conflict deliberately.  Full-batch preflight means
    # neither it nor any W&B payload is emitted or committed on failure.
    with pytest.raises(ScalarCollisionError, match=r"train/loss_total@1000"):
        logger.scalars({"fresh": 7.0, "train/loss_total": 2.0}, 1000)
    assert writer.scalars == [("train/loss_total", 1.0, 1000)]
    assert len(run.payloads) == 1
    logger.scalar("fresh", 7.0, 1000)
    assert writer.scalars[-1] == ("fresh", 7.0, 1000)
    with pytest.raises(ScalarCollisionError, match=r"out-of-order scalar fresh@999"):
        logger.scalar("fresh", 7.0, 999)
    assert writer.scalars[-1] == ("fresh", 7.0, 1000)
    logger.scalar("fresh", 8.0, 1001)
    assert writer.scalars[-1] == ("fresh", 8.0, 1001)
    for invalid in (True, 1.5, np.int64(1), -1):
        with pytest.raises(ValueError, match="non-negative built-in integer"):
            logger.scalar("invalid-step", 1.0, invalid)


def test_scalar_ledger_rejects_intra_batch_alias_before_any_sink_write(tmp_path):
    logger, writer, run = _recording_logger(tmp_path)
    payload = _RepeatedItems([("same", 1.0), ("same", 2.0)])
    with pytest.raises(ScalarCollisionError, match=r"same@9 within batch"):
        logger.scalars(payload, 9)
    assert writer.scalars == []
    assert run.payloads == []
    assert logger._scalar_ledger == {}


def test_selected_validation_metric_namespace_is_exact_and_injective():
    source = {
        "train/loss_total": 1.0,
        "train/executable_prefix/loss_latent": 2.0,
        **{tag: float(index + 3) for index, tag in enumerate(COLLIDING_VALIDATION_AUX_TAGS)},
    }
    mapped = train.namespace_validation_metrics(source)
    assert mapped["val/loss_total"] == 1.0
    assert mapped["val/executable_prefix/loss_latent"] == 2.0
    assert {
        f"val/aux/{tag}" for tag in COLLIDING_VALIDATION_AUX_TAGS
    }.issubset(mapped)
    assert len(mapped) == len(source)

    with pytest.raises(ValueError, match="namespace collision"):
        train.namespace_validation_metrics({"train/loss_total": 1.0, "val/loss_total": 1.0})


def test_terminal_evaluation_namespace_is_exact_and_strict():
    mapped = train.namespace_terminal_evaluation_metrics(
        {
            "eval/success_rate": 0.5,
            "eval/guard/rejection_rate": 0.25,
        }
    )
    assert mapped == {
        "eval/final/success_rate": 0.5,
        "eval/final/guard/rejection_rate": 0.25,
    }
    for invalid in ("success_rate", "val/success_rate", "eval/", ""):
        with pytest.raises(ValueError, match="terminal evaluation metric"):
            train.namespace_terminal_evaluation_metrics({invalid: 1.0})


def test_monitor_evaluation_resource_namespace_is_exact_and_cache_compatible():
    monitor = {
        "eval/success_rate": 0.5,
        "eval/guard/rejection_rate": 0.25,
    }
    assert train.validated_monitor_evaluation_metrics(monitor) == monitor
    cache = {
        "cache/consumed": 1.0,
        "future_recipe/consumed": 1.0,
    }
    assert train.validated_evaluation_cache_metrics(cache) == cache
    assert train.namespace_monitor_evaluation_resource_metrics(
        {
            "resource/host_rss_gb": 10.0,
            "resource/peak_reserved_gb": 8.0,
        }
    ) == {
        "eval/monitor/resource/host_rss_gb": 10.0,
        "eval/monitor/resource/peak_reserved_gb": 8.0,
    }
    for invalid in ("success_rate", "val/success_rate", "eval/"):
        with pytest.raises(ValueError, match="monitor evaluation metric"):
            train.validated_monitor_evaluation_metrics({invalid: 1.0})
    for invalid in ("resource/host_rss_gb", "cache/", "unexpected"):
        with pytest.raises(ValueError, match="evaluation cache metric"):
            train.validated_evaluation_cache_metrics({invalid: 1.0})
    for invalid in ("cache/consumed", "resource/", "unexpected"):
        with pytest.raises(ValueError, match="monitor evaluation resource metric"):
            train.namespace_monitor_evaluation_resource_metrics({invalid: 1.0})


def test_visualization_anchor_namespace_rejects_lossy_index_and_encodes_name():
    assert train.visualization_anchor_scalar_prefix(1, "same") != (
        train.visualization_anchor_scalar_prefix(2, "same")
    )
    assert train.visualization_anchor_scalar_prefix(3, "task/a\nβ") == (
        "viz/anchor/03/task%2Fa%0A%CE%B2/"
    )
    for invalid in (True, 1.0, np.int64(1), -1):
        with pytest.raises(ValueError, match="non-negative integer"):
            train.visualization_anchor_scalar_prefix(invalid, "anchor")
    for invalid in ("", None, 7):
        with pytest.raises(ValueError, match="name must be non-empty"):
            train.visualization_anchor_scalar_prefix(0, invalid)


def test_production_telemetry_self_test_is_exact_and_observational(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    torch_state = torch.random.get_rng_state().clone()
    numpy_state = np.random.get_state()
    environment = dict(os.environ)
    evidence = train.telemetry_contract_self_test()
    assert evidence == {
        "schema_version": 1,
        "status": "telemetry_contract_verified",
        "validation_namespace_sha256": (
            "8f58904f1d6ead6530902886b6ae24dc9529c58ec3562ee25c79c67369368883"
        ),
        "terminal_evaluation_namespace_sha256": (
            "40b28ebb3e286038da6815396452389d1d78a2ef0d42d44bc5cb149145ed2c54"
        ),
        "monitor_evaluation_namespace_sha256": (
            "0d7ec142f9f0715d9e8f3f2f028ab54f5f6d5b9b2066b46da73d3102c8e53e9b"
        ),
        "visualization_namespace_sha256": (
            "cab5e85de7fb86cdd42757529daf307584886ad0f8349b892b12fbbc58a247e2"
        ),
        "float32_identity_bits": "0x3f800000",
        "identical_duplicate_suppressed": True,
        "conflicting_duplicate_rejected": True,
        "batch_preflight_atomic": True,
        "out_of_order_step_rejected": True,
        "invalid_step_rejected": True,
        "backend_writes_performed": 0,
        "persistent_writes_performed": 0,
    }
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)
    numpy_after = np.random.get_state()
    assert numpy_after[0] == numpy_state[0]
    np.testing.assert_array_equal(numpy_after[1], numpy_state[1])
    assert numpy_after[2:] == numpy_state[2:]
    assert dict(os.environ) == environment
    assert list(tmp_path.iterdir()) == []
    assert capsys.readouterr() == ("", "")


def test_train_scalar_call_sites_bind_validation_and_visualization_namespaces():
    syntax = ast.parse(Path(train.__file__).read_text(encoding="utf-8"))
    validation_adds = []
    structural_summary_assignments = []
    structural_summary_scalars = []
    branch_divergence_scalars = []
    formal_diagnostic_scalars = []
    for assignment in (
        node for node in ast.walk(syntax) if isinstance(node, ast.Assign)
    ):
        if any(
            isinstance(node, ast.Attribute)
            and node.attr == "structural_summary"
            for node in ast.walk(assignment.value)
        ):
            structural_summary_assignments.append(assignment)
    for call in (node for node in ast.walk(syntax) if isinstance(node, ast.Call)):
        if isinstance(call.func, ast.Attribute) and call.func.attr == "add_many":
            if isinstance(call.func.value, ast.Name) and call.func.value.id == "val_tracker":
                validation_adds.append(call)
        if not (
            isinstance(call.func, ast.Attribute)
            and call.func.attr == "scalars"
            and call.args
        ):
            continue
        first = call.args[0]
        names = {node.id for node in ast.walk(first) if isinstance(node, ast.Name)}
        if "structural_scalars" in names:
            structural_summary_scalars.append(call)
        if "dvz" in names:
            branch_divergence_scalars.append(call)
        if "dmetrics" in names:
            formal_diagnostic_scalars.append(call)

    assert len(validation_adds) == 1
    assert isinstance(validation_adds[0].args[0], ast.Call)
    assert isinstance(validation_adds[0].args[0].func, ast.Name)
    assert validation_adds[0].args[0].func.id == "namespace_validation_metrics"
    assert len(structural_summary_assignments) == 2
    assert len(structural_summary_scalars) == 4
    prefixed_structural_calls = [
        call
        for call in structural_summary_scalars
        if any(keyword.arg == "prefix" for keyword in call.keywords)
    ]
    canonical_structural_calls = [
        call
        for call in structural_summary_scalars
        if all(keyword.arg != "prefix" for keyword in call.keywords)
    ]
    assert len(prefixed_structural_calls) == 2
    assert len(canonical_structural_calls) == 2
    for call in prefixed_structural_calls:
        prefix = next(keyword.value for keyword in call.keywords if keyword.arg == "prefix")
        assert isinstance(prefix, ast.Call)
        assert isinstance(prefix.func, ast.Name)
        assert prefix.func.id == "visualization_anchor_scalar_prefix"
    assert len(branch_divergence_scalars) == 1
    assert all(
        keyword.arg != "prefix" for keyword in branch_divergence_scalars[0].keywords
    )
    assert len(formal_diagnostic_scalars) == 1
    assert all(keyword.arg != "prefix" for keyword in formal_diagnostic_scalars[0].keywords)

    final_calls = [
        call
        for call in ast.walk(syntax)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and call.func.attr == "scalars"
        and call.args
        and any(
            isinstance(node, ast.Name) and node.id == "final_eval"
            for node in ast.walk(call.args[0])
        )
    ]
    assert len(final_calls) == 1
    assert isinstance(final_calls[0].args[0], ast.Call)
    assert isinstance(final_calls[0].args[0].func, ast.Name)
    assert final_calls[0].args[0].func.id == "namespace_terminal_evaluation_metrics"

    source = Path(train.__file__).read_text(encoding="utf-8")
    assert source.count(
        "validated_monitor_evaluation_metrics(\n                    evaluate("
    ) == 1
    assert source.count("validated_evaluation_cache_metrics(\n        getattr(") == 1
    assert source.count(
        "namespace_monitor_evaluation_resource_metrics(resource_metrics())"
    ) == 1


def test_visualization_collision_escapes_best_effort_handlers():
    syntax = ast.parse(Path(train.__file__).read_text(encoding="utf-8"))
    visualization_handlers = []
    for node in ast.walk(syntax):
        if not isinstance(node, ast.Try):
            continue
        names = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}
        if "structural_scalars" not in names:
            continue
        visualization_handlers.append(node.handlers)

    assert len(visualization_handlers) == 2
    for handlers in visualization_handlers:
        assert len(handlers) == 2
        collision, best_effort = handlers
        assert isinstance(collision.type, ast.Name)
        assert collision.type.id == "ScalarCollisionError"
        assert len(collision.body) == 1
        assert isinstance(collision.body[0], ast.Raise)
        assert collision.body[0].exc is None
        assert isinstance(best_effort.type, ast.Name)
        assert best_effort.type.id == "Exception"
        assert any(
            isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "print"
            for child in ast.walk(best_effort)
        )


def test_sealed_hydra_smoke_calls_exact_telemetry_self_test_once():
    entry = (
        Path(train.__file__).parents[1]
        / "experiments/23-treewm-executable-prefix-repair-pilot-v1/train_entry.py"
    )
    source = entry.read_text(encoding="utf-8")
    assert source.count("train.telemetry_contract_self_test()") == 1
    call = source.index("train.telemetry_contract_self_test()")
    verified_import = source.rindex("_verify_imported_module", 0, call)
    hydra_argv = source.index("sys.argv =", call)
    assert verified_import < call < hydra_argv

    syntax = ast.parse(source)
    assignment = next(
        node
        for node in syntax.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "TELEMETRY_CONTRACT_EVIDENCE"
            for target in node.targets
        )
    )
    assert ast.literal_eval(assignment.value) == train.telemetry_contract_self_test()


@pytest.fixture(scope="module")
def structural_metrics() -> dict[str, float]:
    cfg = TreeWMConfig(
        obs_dim=2,
        action_dim=2,
        z_dim=32,
        q_dim=16,
        hidden_dim=64,
        num_layers=2,
    )
    torch.manual_seed(23)
    model = build_model("randomtreewm", cfg).eval()
    tree_cfg = tree_config_for(
        "randomtreewm",
        TreeConfig(node_budget=48, branch_factor=4, max_depth=16),
        model,
    )
    tree_cfg = replace(tree_cfg, scorer="bfs")
    with torch.no_grad():
        tree, _ = model.generate(
            torch.randn(2, cfg.z_dim),
            tree_cfg,
            generator=make_generator(23, "eval"),
        )
    metrics = structural_summary(tree, model)
    assert set(metrics) == STRUCTURAL_SUMMARY_TAGS
    assert len(metrics) == 23
    return metrics


def test_representative_cadence_has_exact_tensorboard_scalar_identity(
    tmp_path,
    structural_metrics,
):
    run_dir = tmp_path / "run"
    logger = TreeWMLogger(run_dir, flush_secs=1)
    sampler = {"global_sample_size": 5120, "seed": 1701}
    logger.text(
        "meta/fixed_validation_sample",
        json.dumps(sampler, sort_keys=True, indent=2),
    )
    expected: dict[tuple[str, int], float] = {}

    def emit(values: Mapping[str, float], step: int, prefix: str = "") -> None:
        logger.scalars(dict(values), step, prefix=prefix)
        for tag, value in values.items():
            key = (f"{prefix}{tag}" if prefix else tag, step)
            assert key not in expected
            expected[key] = float(np.float32(value))

    training = {
        "train/loss_total": 1.0,
        **{tag: float(index + 2) for index, tag in enumerate(COLLIDING_VALIDATION_AUX_TAGS)},
    }
    emit(training, 1000)
    emit({"control/q_advantage_over_z": 0.25}, 1000)
    validation = train.namespace_validation_metrics(
        {
            "train/loss_total": 11.0,
            **{
                tag: float(index + 12)
                for index, tag in enumerate(COLLIDING_VALIDATION_AUX_TAGS)
            },
        }
    )
    emit(validation, 1000)
    emit({"data/validation_fixed_sample_count": 5120.0}, 0)
    emit({"data/validation_fixed_sample_count": 5120.0}, 1000)

    for anchor_index in range(4):
        prefix = train.visualization_anchor_scalar_prefix(anchor_index, "duplicate-name")
        # Make each anchor genuinely different: the old shared tree/* namespace would
        # therefore fail the new ledger, not merely create identical duplicates.
        per_anchor = {
            tag: float(value) + anchor_index for tag, value in structural_metrics.items()
        }
        if anchor_index == 0:
            emit(per_anchor, 1000)
        emit(per_anchor, 1000, prefix)
    domain_prefix = train.visualization_anchor_scalar_prefix(0, "domain-task")
    emit(structural_metrics, 2000)
    emit(structural_metrics, 2000, domain_prefix)
    emit(
        {
            "tree/sibling_spread": 0.5,
            "tree/global_spread": 1.5,
            "tree/sibling_spread_ratio": 1.0 / 3.0,
            "tree/num_branch_points": 2.0,
        },
        2000,
    )
    # The ordinary dense observation precedes evaluation at this same boundary.
    # Evaluation may increase the process peak, so it needs a distinct phase identity.
    emit({"resource/host_rss_gb": 10.0}, 25_000)
    emit(train.validated_evaluation_cache_metrics({"cache/consumed": 1.0}), 25_000)
    emit(
        train.validated_monitor_evaluation_metrics(
            {"eval/success_rate": 0.2, "eval/num_episodes": 20.0}
        ),
        25_000,
    )
    emit(
        train.namespace_monitor_evaluation_resource_metrics(
            {"resource/host_rss_gb": 11.0}
        ),
        25_000,
    )
    emit(
        train.namespace_terminal_evaluation_metrics(
            {"eval/success_rate": 0.8, "eval/num_episodes": 1000.0}
        ),
        25_000,
    )

    # Cross-API, bit-identical retry is suppressed rather than serialized twice.
    logger.scalar("train/loss_total", float(np.float32(1.0)), 1000)
    logger.close()

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    actual: dict[tuple[str, int], float] = {}
    for tag in accumulator.Tags()["scalars"]:
        for event in accumulator.Scalars(tag):
            key = (tag, int(event.step))
            assert key not in actual, f"duplicate TensorBoard scalar {tag}@{event.step}"
            actual[key] = float(event.value)

    assert actual.keys() == expected.keys()
    assert len(actual) == len(expected)
    for key, value in expected.items():
        assert _float32_bits(actual[key]) == _float32_bits(value), key
    assert all(not tag.startswith("val/control/q_advantage") for tag, _ in actual)
    assert ("control/q_advantage_over_z", 1000) in actual
    assert actual[("eval/success_rate", 25_000)] == pytest.approx(0.2)
    assert actual[("eval/final/success_rate", 25_000)] == pytest.approx(0.8)
    assert actual[("resource/host_rss_gb", 25_000)] == pytest.approx(10.0)
    assert actual[("eval/monitor/resource/host_rss_gb", 25_000)] == pytest.approx(11.0)
    assert actual[("cache/consumed", 25_000)] == pytest.approx(1.0)
    assert actual[("tree/sibling_spread_ratio", 2000)] == pytest.approx(1.0 / 3.0)
    assert ("tree/max_depth", 1000) in actual
    assert len(
        {
            tag
            for tag, step in actual
            if step == 1000 and tag.endswith("tree/max_depth")
        }
    ) == 5

    report_path = (
        Path(train.__file__).parents[1]
        / "experiments/23-treewm-executable-prefix-repair-pilot-v1/report.py"
    )
    spec = importlib.util.spec_from_file_location("exp23_report_telemetry_test", report_path)
    assert spec is not None and spec.loader is not None
    report = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(report)
    parsed = report.parse_event_files(run_dir, sampler)
    assert not any(tag.startswith("eval/") for tag in parsed["scalars"])
    assert parsed["excluded_eval_tags"] == [
        "eval/final/num_episodes",
        "eval/final/success_rate",
        "eval/monitor/resource/host_rss_gb",
        "eval/num_episodes",
        "eval/success_rate",
    ]


def test_wandb_offline_settings_bound_symlinks_to_isolated_core_log(tmp_path, monkeypatch):
    isolated = {
        name: tmp_path / name
        for name in ("home", "xdg-config", "xdg-cache", "wandb-config", "wandb-cache", "tmp")
    }
    for path in isolated.values():
        path.mkdir()
    monkeypatch.setenv("HOME", str(isolated["home"]))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(isolated["xdg-config"]))
    monkeypatch.setenv("XDG_CACHE_HOME", str(isolated["xdg-cache"]))
    monkeypatch.setenv("WANDB_CONFIG_DIR", str(isolated["wandb-config"]))
    monkeypatch.setenv("WANDB_CACHE_DIR", str(isolated["wandb-cache"]))
    monkeypatch.setenv("TMPDIR", str(isolated["tmp"]))
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_SILENT", "true")
    monkeypatch.setenv("WANDB_CONSOLE", "off")
    monkeypatch.setenv("WANDB_DISABLE_CODE", "true")
    run_dir = tmp_path / "offline"
    logger = TreeWMLogger(
        run_dir,
        flush_secs=1,
        wandb_project="treewm-telemetry-test",
        wandb_id="treewm-telemetry-symlink-test",
        wandb_name="symlink-test",
    )
    try:
        assert logger._wandb_run is not None
        assert logger._wandb_run._settings.symlink is False
        logger.scalar("train/loss_total", 1.0, 1)
    finally:
        logger.close()

    symlinks = [path for path in tmp_path.rglob("*") if path.is_symlink()]
    assert len(symlinks) <= 1
    if symlinks:
        link = symlinks[0]
        relative = link.relative_to(run_dir).as_posix()
        assert relative.startswith("wandb/offline-run-")
        assert relative.endswith("/logs/debug-core.log")
        target = link.resolve(strict=True)
        assert target.is_relative_to(tmp_path)
        assert target.is_file() and not target.is_symlink()
    assert not any(path.name in {"latest-run", "debug.log", "debug-internal.log"} for path in symlinks)
