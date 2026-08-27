from __future__ import annotations

import json
from pathlib import Path

import pytest

import gradient_audit as diagnostic


PACKAGE = Path(__file__).resolve().parents[1]


def test_locked_recipe_set_is_exact_and_self_hashed() -> None:
    payload = diagnostic.load_locked_recipes(PACKAGE / "recipes.json")
    assert payload["recipes"] == list(diagnostic.LOCKED_RECIPES)
    assert [row["recipe_id"] for row in payload["recipes"]] == [
        "baseline-exact",
        "candidate-conservative",
        "candidate-control",
    ]
    assert payload["gradient_share_bound"] == 0.8
    assert payload["shared_norm_ratio_bound"] == 1.5
    assert payload["module_norm_ratio_bound"] == 2.0


def test_fixed_batches_are_deterministic_disjoint_and_each_representative() -> None:
    first = diagnostic.fixed_representative_batches(100_003, split="train")
    second = diagnostic.fixed_representative_batches(100_003, split="train")
    assert first == second
    flattened = [value for batch in first for value in batch]
    assert len(flattened) == 48 and len(set(flattened)) == 48
    assert all(len(batch) == 16 for batch in first)
    # Each interleaved batch begins near rank zero and reaches near rank one.
    assert all(batch[0] < 0.08 * 100_003 for batch in first)
    assert all(batch[-1] > 0.90 * 100_003 for batch in first)
    assert diagnostic.fixed_representative_batches(100_003, split="validation") != first


@pytest.mark.parametrize("population,batches,batch_size", [(0, 3, 16), (20, 3, 16), (100, 0, 16)])
def test_fixed_batches_reject_invalid_dimensions(
    population: int, batches: int, batch_size: int
) -> None:
    with pytest.raises(ValueError):
        diagnostic.fixed_representative_batches(
            population, split="train", batches=batches, batch_size=batch_size
        )


def _fake_audit(norm: float, share: float = 0.4) -> dict:
    return {
        "modules": {
            name: {
                "objective_norm": norm,
                "max_term_share": share,
                "terms": {
                    "multistep": {"effective_norm": norm},
                    "keep": {"effective_norm": norm},
                },
            }
            for name in (*diagnostic.MODULE_NAMES, "shared")
        }
    }


def test_scale_gate_applies_all_preregistered_bounds() -> None:
    key = ("train", 0)
    baseline = {key: _fake_audit(2.0)}
    candidate = {key: _fake_audit(2.5)}
    result = diagnostic._candidate_scale_gate(baseline, candidate)
    assert result["passed"]
    shared = [
        row
        for row in result["checks"]
        if row["check"] == "candidate_to_baseline_objective_norm_ratio"
        and row["module"] == "shared"
    ]
    assert shared == [
        {
            "split": "train",
            "batch_index": 0,
            "check": "candidate_to_baseline_objective_norm_ratio",
            "module": "shared",
            "value": 1.25,
            "baseline_value": 2.0,
            "candidate_value": 2.5,
            "bound": 1.5,
            "passed": True,
        }
    ]


def test_scale_gate_rejects_ratio_share_and_dead_intended_path() -> None:
    key = ("validation", 2)
    baseline = {key: _fake_audit(1.0)}
    candidate = {key: _fake_audit(1.0)}
    candidate[key]["modules"]["shared"]["objective_norm"] = 1.6
    candidate[key]["modules"]["encoder"]["max_term_share"] = 0.81
    candidate[key]["modules"]["decoder"]["terms"]["multistep"][
        "effective_norm"
    ] = 1e-9
    result = diagnostic._candidate_scale_gate(baseline, candidate)
    assert not result["passed"]
    assert result["failure_count"] == 3


def test_scale_gate_records_zero_baseline_without_nonfinite_json() -> None:
    key = ("train", 1)
    result = diagnostic._candidate_scale_gate(
        {key: _fake_audit(0.0)}, {key: _fake_audit(1.0)}
    )
    assert not result["passed"]
    ratio = next(
        row
        for row in result["failures"]
        if row["check"] == "candidate_to_baseline_objective_norm_ratio"
    )
    assert ratio["value"] is None
    json.dumps(result, allow_nan=False)


def test_output_root_rejects_formal_tree(tmp_path: Path) -> None:
    formal = tmp_path / "outputs" / "treewm-grounded-formal-v1"
    formal.mkdir(parents=True)
    with pytest.raises(diagnostic.DiagnosticError, match="outside"):
        diagnostic.validate_output_root(formal / "audit", formal)
    outside = tmp_path / "outputs" / "treewm-gradient-audit-v1"
    assert diagnostic.validate_output_root(outside, formal) == outside.resolve()


def test_immutable_artifact_is_self_hashed_and_idempotent(tmp_path: Path) -> None:
    body = {
        "run": {"setting_id": "scene", "seed": 0},
        "checkpoint": {"completed_updates": 25_000},
        "value": 7,
    }
    path, reused = diagnostic.write_immutable_json(tmp_path, body)
    assert not reused
    payload = json.loads(path.read_text(encoding="utf-8"))
    claimed = payload.pop("artifact_sha256")
    assert claimed == diagnostic.stable_hash(payload)
    assert path.stat().st_mode & 0o222 == 0
    repeated, reused = diagnostic.write_immutable_json(tmp_path, body)
    assert repeated == path and reused


def test_slurm_wrapper_maps_ten_jobs_and_separates_roots() -> None:
    wrapper = (PACKAGE / "gradient_audit.slurm").read_text(encoding="utf-8")
    assert "#SBATCH --array=0-9%10" in wrapper
    assert 'PACKAGE="$SOURCE_ROOT/experiments/diagnostics/treewm-gradient-audit-v1"' in wrapper
    assert 'checkpoint="$PROJECT_ROOT/outputs/treewm-grounded-formal-v1/' in wrapper
    assert 'OUTPUT_ROOT="$PROJECT_ROOT/outputs/treewm-gradient-audit-v1"' in wrapper
    assert 'cd "$SOURCE_ROOT"' in wrapper
