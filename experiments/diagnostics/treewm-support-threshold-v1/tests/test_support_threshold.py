from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import support_threshold as diagnostic


def test_threshold_grid_is_exact_and_inclusive() -> None:
    assert diagnostic.threshold_grid() == (
        0.25,
        0.275,
        0.3,
        0.325,
        0.35,
        0.375,
        0.4,
        0.425,
        0.45,
        0.475,
        0.5,
        0.525,
        0.55,
    )


def test_fixed_sample_is_unique_representative_and_version_independent() -> None:
    first = diagnostic.fixed_stratified_positions(101_003, 4096)
    second = diagnostic.fixed_stratified_positions(101_003, 4096)
    assert np.array_equal(first, second)
    assert np.all(first[1:] > first[:-1])
    assert int(first.min()) >= 0 and int(first.max()) < 101_003
    starts = np.arange(4096, dtype=np.int64) * 101_003 // 4096
    stops = np.arange(1, 4097, dtype=np.int64) * 101_003 // 4096
    assert np.all(first >= starts)
    assert np.all(first < stops)
    assert diagnostic.array_sha256(first, "<i8") == diagnostic.array_sha256(second, "<i8")


@pytest.mark.parametrize("population,sample", [(0, 1), (10, 0), (3, 4)])
def test_fixed_sample_rejects_invalid_sizes(population: int, sample: int) -> None:
    with pytest.raises(ValueError):
        diagnostic.fixed_stratified_positions(population, sample)


def test_threshold_metrics_match_exact_confusion_counts_and_fallback_proxy() -> None:
    scores = np.asarray([[0.9, 0.6, 0.4, 0.1], [0.49, 0.48, 0.2, 0.1]], dtype=np.float32)
    labels = np.asarray([[1, 0, 1, 0], [1, 0, 0, 0]], dtype=np.bool_)
    row = diagnostic.threshold_metrics(scores, labels, 0.5, max_depth=3, node_budget=64)
    assert (row["true_positive"], row["false_positive"]) == (1, 1)
    assert (row["false_negative"], row["true_negative"]) == (2, 4)
    assert row["support_recall"] == pytest.approx(1 / 3)
    assert row["support_precision"] == pytest.approx(1 / 2)
    assert row["raw_keep_rate"] == pytest.approx(2 / 8)
    assert row["raw_kept_children_per_anchor"] == pytest.approx(1.0)
    assert row["root_children_after_top1_fallback_mean"] == pytest.approx(1.5)
    assert row["root_top1_fallback_fraction"] == pytest.approx(0.5)
    assert row["root_multi_child_fraction"] == pytest.approx(0.5)
    # widths 2 and fallback-1 imply 1+2+4+8=15 and 1+1+1+1=4 nodes.
    assert row["homogeneous_full_depth_nodes_proxy_mean"] == pytest.approx(9.5)


def test_threshold_metrics_zero_prediction_precision_is_zero() -> None:
    scores = np.zeros((2, 4), dtype=np.float32)
    labels = np.asarray([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.bool_)
    row = diagnostic.threshold_metrics(scores, labels, 0.5, max_depth=3, node_budget=64)
    assert row["support_recall"] == 0.0
    assert row["support_precision"] == 0.0
    assert row["root_top1_fallback_fraction"] == 1.0
    assert row["homogeneous_full_depth_nodes_proxy_mean"] == 4.0


def test_output_root_rejects_entire_formal_campaign_tree(tmp_path: Path) -> None:
    formal = tmp_path / "outputs" / "treewm-grounded-formal-v1"
    formal.mkdir(parents=True)
    with pytest.raises(diagnostic.DiagnosticError, match="outside"):
        diagnostic.validate_output_root(formal / "diagnostics", formal)
    outside = tmp_path / "outputs" / "treewm-support-threshold-diagnostic-v1"
    assert diagnostic.validate_output_root(outside, formal) == outside.resolve()


def test_immutable_artifact_is_self_hashed_idempotent_and_rejects_drift(tmp_path: Path) -> None:
    body = {
        "run": {"setting_id": "puzzle-3x3", "seed": 2},
        "checkpoint": {"completed_updates": 25_000},
        "value": 1,
    }
    path, reused = diagnostic.write_immutable_json(tmp_path, body)
    assert reused is False
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("artifact_sha256")
    assert claimed == diagnostic.stable_hash(payload)
    assert path.stat().st_mode & 0o222 == 0
    repeated, reused = diagnostic.write_immutable_json(tmp_path, body)
    assert repeated == path and reused is True
    path.chmod(0o644)
    path.write_text("different\n", encoding="utf-8")
    path.chmod(0o444)
    with pytest.raises(diagnostic.DiagnosticError, match="differs"):
        diagnostic.write_immutable_json(tmp_path, body)


def test_protected_tree_fingerprint_changes_on_material_mutation(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    artifact = run / "FORMAL_LAUNCH.json"
    artifact.write_text("{}\n", encoding="utf-8")
    before = diagnostic.protected_tree_fingerprint(run)
    artifact.write_text('{"changed":true}\n', encoding="utf-8")
    assert diagnostic.protected_tree_fingerprint(run) != before


def test_slurm_wrapper_separates_immutable_source_from_live_outputs() -> None:
    wrapper = (Path(__file__).resolve().parents[1] / "support_threshold.slurm").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --array=0-7%8" in wrapper
    assert 'PACKAGE="$SOURCE_ROOT/experiments/diagnostics/treewm-support-threshold-v1"' in wrapper
    assert 'OUTPUT_ROOT="$PROJECT_ROOT/outputs/treewm-support-threshold-diagnostic-v1"' in wrapper
    assert 'checkpoint="$PROJECT_ROOT/outputs/treewm-grounded-formal-v1/' in wrapper
    assert 'cd "$SOURCE_ROOT"' in wrapper
