from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CAMPAIGN_DIR.parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))
import campaign  # noqa: E402


@pytest.fixture
def manifest():
    return campaign.load_manifest(CAMPAIGN_DIR / "manifest.json")


def test_manifest_is_exact_treewm_40_model_200_evaluation_protocol(manifest):
    assert (CAMPAIGN_DIR / "protocol.sha256").read_text().strip() == campaign.protocol_sha256(manifest)
    assert manifest["method"] == {
        "arm": "treewm",
        "model_class": "TreeWM",
        "scorer": "learned",
        "node_budget": 64,
        "branch_factor": 4,
        "gradient_checkpointing": True,
    }
    assert manifest["training"]["optimizer_updates"] == 1_000_000
    assert manifest["training"]["future_set_cache"] is False
    assert manifest["training"]["shared_cache"] is True
    assert manifest["evaluation"]["final_episodes_per_task"] == 50
    runs = campaign.expand_runs(manifest)
    assert len(runs) == 40
    assert len({(run.setting_id, run.seed) for run in runs}) == 40
    assert len({run.run_name for run in runs}) == 40
    assert len({run.wandb_id for run in runs}) == 40
    assert all(len(run.wandb_id) == 32 for run in runs)


def test_three_allocation_mapping_owns_every_run_exactly_once(manifest):
    assert campaign.all_worker_ownership(manifest) == list(range(40))
    assert campaign.run_for_worker(manifest, 0, 0).global_index == 0
    assert campaign.run_for_worker(manifest, 1, 0).global_index == 16
    assert campaign.run_for_worker(manifest, 2, 0).global_index == 32
    assert campaign.run_for_worker(manifest, 2, 7).global_index == 39
    assert all(campaign.run_for_worker(manifest, 2, rank) is None for rank in range(8, 16))


def test_environment_configs_explicitly_match_endpoint_and_episode_policy(manifest):
    for setting in manifest["settings"]:
        text = (REPO_ROOT / "configs" / "env" / f"{setting['env_config']}.yaml").read_text()
        relative = str(setting["relative_endpoints"]).lower()
        assert f"relative_endpoints: {relative}" in text
        assert f"max_episode_steps: {setting['max_episode_steps']}" in text


def test_absolute_campaign_import_bootstraps_this_repository(tmp_path, manifest):
    python = manifest["paths"]["python"]
    code = """
import pathlib
import sys
sys.path.insert(0, sys.argv[1])
import campaign
import treewm
print(pathlib.Path(treewm.__file__).resolve())
print(campaign.REPOSITORY_ROOT)
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [python, "-c", code, str(CAMPAIGN_DIR)],
        cwd=tmp_path,
        env=env,
        check=True,
        text=True,
        capture_output=True,
    )
    lines = completed.stdout.splitlines()
    assert Path(lines[-2]) == (REPO_ROOT / "treewm" / "__init__.py").resolve()
    assert Path(lines[-1]) == REPO_ROOT.resolve()


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["method"].update(arm="randomtreewm"),
        lambda value: value["method"].update(model_class="RandomTreeWM"),
        lambda value: value["method"].update(scorer="random"),
        lambda value: value["training"].update(optimizer_updates=999_999),
        lambda value: value["training"].update(future_set_cache=True),
        lambda value: value["training"].update(shared_cache=False),
        lambda value: value["evaluation"].update(final_episodes_per_task=5),
        lambda value: value["settings"][2].update(dataset_kind="sharded_100m_sample"),
        lambda value: value["settings"][2].update(expected_train_shards=99),
        lambda value: value["settings"][2].update(expected_train_transitions=99_999_999),
        lambda value: value["settings"][0].update(relative_endpoints=True),
        lambda value: value["execution"].update(allocation_shards=13),
        lambda value: value["paths"].update(
            python="/lustre/fsw/portfolios/edgeai/users/chrislin/envs/maniskill-conda/bin/python3.11"
        ),
    ],
)
def test_manifest_validator_rejects_scientific_drift(manifest, mutate):
    changed = copy.deepcopy(manifest)
    mutate(changed)
    with pytest.raises(campaign.ManifestError):
        campaign.validate_manifest(changed)


def _make_standard_contract(manifest, tmp_path: Path, setting_index: int = 0):
    manifest = copy.deepcopy(manifest)
    data_root = tmp_path / "data"
    cache_root = tmp_path / "cache"
    run_root = tmp_path / "runs"
    manifest["paths"].update(
        data_root=str(data_root), cache_root=str(cache_root), run_root=str(run_root)
    )
    setting = manifest["settings"][setting_index]
    paths = campaign.required_dataset_files(manifest, data_root, setting)
    entries = []
    for i, path in enumerate(paths):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"source-{i}".encode())
        stat = path.stat()
        entries.append(
            {
                "path": str(path.relative_to(data_root)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    cache_manifest = cache_root / "materialized" / "manifest.json"
    cache_manifest.parent.mkdir(parents=True)
    cache_manifest.write_text("{}")
    data_sha = "d" * 64
    contract = {
        "schema_version": 1,
        "status": "complete",
        "protocol_sha256": campaign.protocol_sha256(manifest),
        "setting_id": setting["id"],
        "dataset_kind": setting["dataset_kind"],
        "data_manifest_sha256": data_sha,
        "cache_manifest": str(cache_manifest),
        "source_files": entries,
    }
    path = campaign.data_contract_path(cache_root, setting["id"])
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(contract))
    return manifest, setting, data_sha, data_root, cache_root, run_root


def test_trainer_command_is_treewm_only_and_fully_pinned(manifest, tmp_path, monkeypatch):
    manifest, _, data_sha, data_root, cache_root, run_root = _make_standard_contract(
        manifest, tmp_path
    )
    monkeypatch.setattr(
        campaign, "live_contract", lambda repo_root: {"code_sha256": "c" * 64, "runtime_sha256": "r" * 64}
    )
    run = campaign.expand_runs(manifest)[0]
    argv, env = campaign.trainer_command(
        manifest,
        run,
        python_executable=manifest["paths"]["python"],
        repo_root=REPO_ROOT,
        run_root=run_root,
        data_root=data_root,
        cache_root=cache_root,
    )
    joined = " ".join(argv)
    assert argv[1] == str(REPO_ROOT / "scripts" / "train.py")
    assert "upstream_rql" not in joined and "randomtreewm" not in joined
    for expected in (
        "arm=treewm",
        "train.steps=1000000",
        "tree.node_budget=64",
        "tree.scorer=learned",
        "eval.task_split=standard",
        "eval.final_episodes_per_task=50",
        "eval.episodes_per_task=1",
        "future_sets.cache=false",
        "future_sets.shared_cache=true",
        "planner.max_env_steps=750",
    ):
        assert expected in argv
    assert env["TREEWM_DATA_SHA256"] == data_sha
    assert env["TREEWM_CODE_SHA256"] == "c" * 64
    assert env["TREEWM_RUNTIME_SHA256"] == "r" * 64
    assert env["WANDB_RUN_ID"] == run.wandb_id
    assert not any("KEY" in key or "TOKEN" in key for key in env)


def test_all_trainer_commands_preserve_formal_venv_symlink_and_import_hydra(
    manifest, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        campaign,
        "load_data_contract",
        lambda *args, **kwargs: {"data_manifest_sha256": "d" * 64},
    )
    monkeypatch.setattr(
        campaign,
        "live_contract",
        lambda repo_root: {"code_sha256": "c" * 64, "runtime_sha256": "r" * 64},
    )
    commands = [
        campaign.trainer_command(
            manifest,
            run,
            python_executable=manifest["paths"]["python"],
            repo_root=REPO_ROOT,
            run_root=tmp_path / "runs",
            data_root=tmp_path / "data",
            cache_root=tmp_path / "cache",
        )[0]
        for run in campaign.expand_runs(manifest)
    ]
    assert len(commands) == 40
    assert {argv[0] for argv in commands} == {manifest["paths"]["python"]}
    assert Path(commands[0][0]).is_symlink()
    with pytest.raises(ValueError, match="formal Python path is immutable"):
        campaign.trainer_command(
            manifest,
            campaign.expand_runs(manifest)[0],
            python_executable=Path(manifest["paths"]["python"]).resolve(),
            repo_root=REPO_ROOT,
            run_root=tmp_path / "runs",
            data_root=tmp_path / "data",
            cache_root=tmp_path / "cache",
        )

    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONNOUSERSITE"] = "1"
    completed = subprocess.run(
        [commands[0][0], "-c", "import hydra; print(hydra.__version__)"],
        cwd=tmp_path,
        env=environment,
        check=True,
        text=True,
        capture_output=True,
    )
    assert completed.stdout.strip() == "1.3.2"


def _write_valid_completion(run_dir, manifest, run, data_sha, code_sha="c" * 64, runtime_sha="r" * 64):
    run_dir.mkdir(parents=True)
    identity = {
        "schema_version": 1,
        "run_dir": str(run_dir.resolve()),
        "run_name": run.run_name,
        "arm": "treewm",
        "env_name": run.env_name,
        "setting": run.setting_id,
        "dataset_kind": run.dataset_kind,
        "source_name": run.source_name,
        "seed": run.seed,
        "total_steps": 1_000_000,
        "world_size": 1,
        "model_class": "TreeWM",
        "scorer": "learned",
        "node_budget": 64,
        "branch_factor": 4,
        "gradient_checkpointing": True,
        "future_set_cache": False,
        "shared_cache": True,
        "task_ids": [1, 2, 3, 4, 5],
        "final_episodes_per_task": 50,
        "config_sha256": "f" * 64,
        "protocol_sha256": campaign.protocol_sha256(manifest),
        "code_sha256": code_sha,
        "runtime_sha256": runtime_sha,
        "data_manifest_sha256": data_sha,
        "wandb_project": manifest["logging"]["wandb_project"],
        "wandb_entity": "",
        "wandb_group": manifest["logging"]["wandb_group"],
        "wandb_mode": "online",
        "wandb_id": run.wandb_id,
    }
    identity_sha = campaign.stable_hash(identity)
    metrics = {"eval/num_episodes": 250, "eval/success_rate": 0.2}
    for task_id in range(1, 6):
        metrics[f"eval/task{task_id}/num_episodes"] = 50
        metrics[f"eval/task{task_id}/success_rate"] = task_id / 10
    results = [{"task_id": task_id, "success": 0.0} for task_id in range(1, 6) for _ in range(50)]
    progress = {
        "schema_version": 1,
        "status": "complete",
        "identity_sha256": identity_sha,
        "task_ids": [1, 2, 3, 4, 5],
        "episodes_per_task": 50,
        "completed_results": results,
        "metrics": metrics,
    }
    (run_dir / "final_eval_progress.json").write_text(json.dumps(progress))
    payload = {
        "schema_version": 1,
        "status": "complete",
        "run_identity": identity,
        "identity_sha256": identity_sha,
        "protocol_sha256": campaign.protocol_sha256(manifest),
        "code_sha256": code_sha,
        "runtime_sha256": runtime_sha,
        "data_manifest_sha256": data_sha,
        "arm": "treewm",
        "model_class": "TreeWM",
        "scorer": "learned",
        "setting": run.setting_id,
        "env_name": run.env_name,
        "dataset_kind": run.dataset_kind,
        "source_name": run.source_name,
        "seed": run.seed,
        "wandb_id": run.wandb_id,
        "completed_updates": 1_000_000,
        "final_eval_step": 1_000_000,
        "task_ids": [1, 2, 3, 4, 5],
        "episodes_per_task": 50,
        "node_budget": 64,
        "branch_factor": 4,
        "gradient_checkpointing": True,
        "future_set_cache": False,
        "shared_cache": True,
        "wandb_group": manifest["logging"]["wandb_group"],
        "final_evaluation": metrics,
        "final_eval_progress": "final_eval_progress.json",
    }
    (run_dir / "COMPLETED.json").write_text(json.dumps(payload))
    return payload


def test_completion_validation_checks_full_identity_and_250_final_episodes(manifest, tmp_path, monkeypatch):
    run = campaign.expand_runs(manifest)[0]
    run_dir = tmp_path / "run"
    payload = _write_valid_completion(run_dir, manifest, run, "d" * 64)
    monkeypatch.setattr(
        campaign, "live_contract", lambda repo_root: {"code_sha256": "c" * 64, "runtime_sha256": "r" * 64}
    )
    assert campaign.completion_is_valid(
        run_dir, manifest, run, repo_root=REPO_ROOT, data_manifest_sha256="d" * 64
    )

    for mutation in (
        lambda value: value.update(arm="randomtreewm"),
        lambda value: value.update(completed_updates=999_999),
        lambda value: value["final_evaluation"].update({"eval/task4/num_episodes": 49}),
        lambda value: value["run_identity"].update(seed=3),
        lambda value: value.update(data_manifest_sha256="e" * 64),
    ):
        changed = copy.deepcopy(payload)
        mutation(changed)
        (run_dir / "COMPLETED.json").write_text(json.dumps(changed))
        assert not campaign.completion_is_valid(
            run_dir, manifest, run, repo_root=REPO_ROOT, data_manifest_sha256="d" * 64
        )
    (run_dir / "COMPLETED.json").write_text(json.dumps(payload))


def test_data_contract_rejects_source_drift(manifest, tmp_path):
    manifest, setting, _, data_root, cache_root, _ = _make_standard_contract(manifest, tmp_path)
    campaign.load_data_contract(manifest, setting, data_root=data_root, cache_root=cache_root)
    source = campaign.required_dataset_files(manifest, data_root, setting)[0]
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        campaign.load_data_contract(manifest, setting, data_root=data_root, cache_root=cache_root)
