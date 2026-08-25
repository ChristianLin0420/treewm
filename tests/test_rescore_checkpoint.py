"""Focused contracts for the read-only checkpoint validation rescorer."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch
from torch.utils.data import Dataset

from scripts import rescore_checkpoint as rescore


def _checkpoint(root: Path, name: str) -> Path:
    path = root / name / "checkpoints" / "latest.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(name.encode())
    return path.resolve()


def test_checkpoint_discovery_and_work_index_are_deterministic(tmp_path):
    second = _checkpoint(tmp_path, "z-run")
    first = _checkpoint(tmp_path, "a-run")
    found = rescore.discover_checkpoints(
        [str(second)], [str(tmp_path / "*-run/checkpoints/latest.pt")], repo_root=tmp_path
    )
    assert found == [first, second]
    assert rescore.select_work(found, 1) == [second]
    with pytest.raises(IndexError):
        rescore.select_work(found, 2)


def _identity_payload() -> dict:
    config = {
        "resume": "auto",
        "campaign_data_contract_sha256": "9" * 64,
    }
    identity = {
        "setting": "cube-double",
        "objective_version": "treewm_v2_rms_rank_v1",
        "protocol_sha256": "1" * 64,
        "code_sha256": "2" * 64,
        "runtime_sha256": "3" * 64,
        "data_manifest_sha256": "4" * 64,
        "calibration_sha256": "5" * 64,
        "future_recipe_sha256": "6" * 64,
    }
    identity_config = dict(config)
    identity_config["resume"] = None
    identity["config_sha256"] = rescore.stable_hash(identity_config)
    return {
        "config": config,
        "run_identity": identity,
        "identity_sha256": rescore.stable_hash(identity),
    }


def _write_contract(root: Path) -> None:
    (root / "data").mkdir(parents=True)
    recipe = root / "future-recipes" / "cube-double"
    recipe.mkdir(parents=True)
    (root / "data" / "cube-double.json").write_text(
        json.dumps(
            {
                "setting_id": "cube-double",
                "objective_version": "treewm_v2_rms_rank_v1",
                "contract_sha256": "9" * 64,
                "data_manifest_sha256": "4" * 64,
                "calibration_sha256": "5" * 64,
                "future_recipe_sha256": "6" * 64,
                "normalizer_sha256": "8" * 64,
            }
        )
    )
    (recipe / "manifest.json").write_text(
        json.dumps(
            {
                "source_manifest_sha256": "4" * 64,
                "calibration_sha256": "5" * 64,
                "recipe_sha256": "6" * 64,
                "code_sha256": "2" * 64,
                "runtime_sha256": "3" * 64,
            }
        )
    )


def test_dataset_environment_separates_live_code_from_recipe_producer(tmp_path):
    payload = _identity_payload()
    contract_root = tmp_path / "contracts"
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    _write_contract(contract_root)
    environment, provenance = rescore.dataset_environment(
        payload,
        contract_root,
        cache_root,
        live_code_sha256="a" * 64,
        live_runtime_sha256="b" * 64,
    )
    assert environment["TREEWM_CODE_SHA256"] == "a" * 64
    assert environment["TREEWM_RUNTIME_SHA256"] == "b" * 64
    assert environment["TREEWM_RECIPE_CODE_SHA256"] == "2" * 64
    assert environment["TREEWM_RECIPE_RUNTIME_SHA256"] == "3" * 64
    assert provenance["live_trainer_code_sha256"] == "a" * 64
    assert provenance["recipe_code_sha256"] == "2" * 64
    assert provenance["normalizer_sha256"] == "8" * 64


def test_checkpoint_identity_rejects_resolved_config_drift():
    payload = _identity_payload()
    rescore.validate_checkpoint_identity(payload)
    payload["config"]["new_value"] = 1
    with pytest.raises(ValueError, match="config does not match"):
        rescore.validate_checkpoint_identity(payload)


def test_content_addressed_output_cannot_live_inside_run(tmp_path):
    checkpoint = _checkpoint(tmp_path, "run")
    with pytest.raises(ValueError, match="outside checkpoint run"):
        rescore.assert_output_outside_runs(checkpoint.parents[1] / "analysis", [checkpoint])
    output = tmp_path / "independent-analysis"
    rescore.assert_output_outside_runs(output, [checkpoint])
    artifact_id = "d" * 64
    path = rescore.result_path(output, checkpoint, 123, artifact_id)
    assert path.name == f"step123-{artifact_id}.json"
    assert output in path.parents


class _ValidationDataset(Dataset):
    def __init__(self) -> None:
        self.anchors = torch.arange(40).numpy() * 10

    def __len__(self) -> int:
        return len(self.anchors)

    def __getitem__(self, index: int):
        return {
            "anchor_index": torch.tensor(int(self.anchors[index])),
            "obs": torch.tensor([float(index), 0.0]),
            "fut_horizon_idx": torch.tensor([0, 1]),
            "fut_valid": torch.ones(2),
            "num_modes": torch.tensor(index % 3),
            "future_diversity": torch.tensor(float(index)),
        }


def test_fixed_rescore_reports_losses_diagnostics_labels_and_sample(monkeypatch):
    monkeypatch.setattr(rescore.cfg_utils, "loss_config", lambda cfg: object())
    monkeypatch.setattr(rescore.cfg_utils, "matching_config", lambda cfg: object())

    def fake_losses(model, batch, loss_cfg, match_cfg, step):
        del model, loss_cfg, match_cfg, step
        value = float(batch["obs"][:, 0].mean())
        return torch.tensor(value), {
            "train/loss_total": value,
            "train/loss_state": value + 1.0,
            "model/matched_fraction": 0.5,
        }, {}

    monkeypatch.setattr(rescore, "compute_branch_losses", fake_losses)
    monkeypatch.setattr(
        rescore.diag,
        "q_vs_z_retrieval",
        lambda model, batch: {
            "control/retrieval_uses_task_metric_endpoint": 1.0,
            "control/retrieval_precision_q": 0.75,
        },
    )
    monkeypatch.setattr(
        rescore.diag,
        "branching_diversity_correlation",
        lambda model, batch: {"control/branching_future_diversity_corr": 0.25},
    )
    cfg = OmegaConf.create({"seed": 3, "train": {"batch_size": 4}})
    result = rescore.evaluate_fixed_validation(
        object(),
        _ValidationDataset(),
        cfg,
        SimpleNamespace(horizons=(4, 8), max_modes=2),
        device=torch.device("cpu"),
        val_batches=2,
        num_workers=0,
        step=50,
    )
    assert result["evaluated_batches"] == 2
    assert result["evaluated_examples"] == 8
    assert set(result["branch_loss_components"]) == {"val/loss_state", "val/loss_total"}
    assert result["diagnostics"]["control/retrieval_uses_task_metric_endpoint"] == 1.0
    assert result["label_distribution"][
        "data/validation_horizon_label_normalized_entropy"
    ] == pytest.approx(1.0)
    assert result["sample"]["anchor_rank_fraction_quantiles"]["q00"] < 0.2
    assert result["sample"]["anchor_rank_fraction_quantiles"]["q100"] > 0.8
    assert len(result["sample"]["diagnostic_anchor_indices"]) == 4


def test_slurm_wrapper_uses_absolute_cm_client_and_only_maps_work():
    path = (
        Path(__file__).resolve().parents[1]
        / "experiments/12-treewm-formal-v2/rescore_checkpoint.slurm"
    )
    text = path.read_text(encoding="utf-8")
    assert 'SLURM_SRUN="/cm/shared/apps/slurm/current/bin/srun"' in text
    assert "#SBATCH --array=0-39%40" in text
    assert '--work-index "$SLURM_ARRAY_TASK_ID"' in text
    assert "scripts/rescore_checkpoint.py" in text
    assert 'SOURCE_ROOT="${TREEWM_SOURCE_ROOT' in text
    assert 'PROJECT_ROOT="${TREEWM_PROJECT_ROOT' in text
    assert 'cd "$SOURCE_ROOT"' in text
    executable_lines = [
        line.strip() for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert not any("sbatch" in line for line in executable_lines)
