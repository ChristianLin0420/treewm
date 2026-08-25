from __future__ import annotations

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from omegaconf import OmegaConf
import pytest


CAMPAIGN_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CAMPAIGN_DIR.parents[1]
sys.path.insert(0, str(CAMPAIGN_DIR))
sys.path.insert(0, str(REPO_ROOT))

import calibration  # noqa: E402
import campaign  # noqa: E402
from treewm.data import future_recipe  # noqa: E402


@pytest.fixture
def manifest():
    return campaign.load_manifest(CAMPAIGN_DIR / "manifest.json")


def test_manifest_locks_v2_objective_matrix_and_fresh_namespaces(manifest):
    assert manifest["method"]["objective_version"] == "treewm_v2_rms_rank_v1"
    assert manifest["objective"] == campaign.EXPECTED_OBJECTIVE
    assert manifest["calibration"] == campaign.EXPECTED_CALIBRATION
    assert manifest["training"]["optimizer_updates"] == 1_000_000
    assert manifest["training"]["scheduler_total_steps"] == 1_000_000
    assert manifest["training"]["data_loader_workers"] == 10
    assert manifest["training"]["loader_thread_limit"] == 1
    assert manifest["training"]["future_set_cache"] is False
    assert manifest["training"]["latent_retrieval_enabled"] is False
    assert manifest["training"]["latent_retrieval_keys"] == 0
    assert manifest["logging"]["wandb_project"].endswith("-v2")
    assert manifest["logging"]["wandb_group"] == manifest["campaign_id"]
    assert "1m-v1" not in manifest["paths"]["run_root"]
    assert "1m-v1" not in manifest["paths"]["pilot_run_root"]
    # This is the sole explicitly approved legacy artifact and it is read-only.
    assert manifest["paths"]["raw_cache_root"].endswith("full-cache-v1")
    runs = campaign.expand_runs(manifest)
    assert len(runs) == 40
    assert len({run.run_name for run in runs}) == 40
    assert len({run.wandb_id for run in runs}) == 40
    assert all(run.run_name.startswith("treewm-v2-") for run in runs)


def test_three_allocations_own_every_model_once_with_eight_idle_tail_ranks(manifest):
    runs = campaign.expand_runs(manifest)
    assert campaign.all_worker_ownership(manifest) == list(range(40))
    assert campaign.run_for_worker(runs, 0, allocation_shard=0).index == 0
    assert campaign.run_for_worker(runs, 0, allocation_shard=1).index == 16
    assert campaign.run_for_worker(runs, 7, allocation_shard=2).index == 39
    assert all(
        campaign.run_for_worker(runs, rank, allocation_shard=2) is None
        for rank in range(8, 16)
    )


def test_protocol_hash_binds_core_loss_and_config_sources(manifest, tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    campaign_dir = repo / "experiments" / "12-treewm-formal-v2"
    loss_path = repo / "treewm" / "losses" / "world_losses.py"
    config_path = repo / "configs" / "base.yaml"
    files = {
        campaign_dir / "manifest.json": json.dumps(manifest),
        campaign_dir / "campaign.py": "CAMPAIGN = 1\n",
        loss_path: "LOSS = 1\n",
        config_path: "train: {}\n",
        repo / "scripts" / "train.py": "TRAIN = 1\n",
        repo / "scripts" / "__init__.py": "",
    }
    for path, content in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    monkeypatch.setattr(campaign, "PROTOCOL_SOURCE_FILES", ("campaign.py",))

    original = campaign.protocol_sha256(manifest, campaign_dir=campaign_dir)
    loss_path.write_text("LOSS = 2\n")
    changed_loss = campaign.protocol_sha256(manifest, campaign_dir=campaign_dir)
    assert changed_loss != original

    loss_path.write_text("LOSS = 1\n")
    config_path.write_text("train: {lr: 0.1}\n")
    changed_config = campaign.protocol_sha256(manifest, campaign_dir=campaign_dir)
    assert changed_config != original


def test_manifest_loader_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text('{"schema_version":2,"schema_version":2}')
    with pytest.raises(campaign.ManifestError, match="duplicate JSON key"):
        campaign.load_manifest(path)


def test_expected_objective_source_has_no_duplicate_literal_keys():
    tree = ast.parse((CAMPAIGN_DIR / "campaign.py").read_text())
    assignment = next(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "EXPECTED_OBJECTIVE"
            for target in node.targets
        )
    )
    for mapping in ast.walk(assignment.value):
        if not isinstance(mapping, ast.Dict):
            continue
        keys = [
            key.value
            for key in mapping.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        ]
        assert len(keys) == len(set(keys))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["method"].update(objective_version="treewm_v1"),
        lambda value: value["method"].update(arm="randomtreewm"),
        lambda value: value["objective"].update(future_metric_mode="legacy_l2"),
        lambda value: value["objective"].update(detach_world_targets=False),
        lambda value: value["objective"].update(gain_set_context=False),
        lambda value: value["objective"].update(keep_threshold=None),
        lambda value: value["objective"]["matching_contract"].update(lambda_action=0.7),
        lambda value: value["objective"]["loss_contract"]["weights"].update(state=2.0),
        lambda value: value["objective"]["loss_contract"]["enabled"].update(recursive=False),
        lambda value: value["objective"]["loss_contract"]["warmup"].update(expand=0),
        lambda value: value["objective"]["optimizer_contract"].update(lr=1e-3),
        lambda value: value["objective"]["gain_training_contract"].update(loss_every=1),
        lambda value: value["objective"]["model_contract"].update(reconstruction=False),
        lambda value: value["objective"]["model_contract"].update(normalize_q=False),
        lambda value: value["objective"]["model_contract"].update(novelty_space="z"),
        lambda value: value["objective"]["model_contract"].update(horizon_mode="fixed"),
        lambda value: value["objective"]["planner_contract"].update(execute_steps=8),
        lambda value: value["training"].update(optimizer_updates=999_999),
        lambda value: value["training"].update(scheduler_total_steps=5_000),
        lambda value: value["training"].update(future_set_cache=True),
        lambda value: value["training"].update(latent_retrieval_enabled=True),
        lambda value: value["training"].update(latent_retrieval_keys=100_000),
        lambda value: value["calibration"].update(split="validation"),
        lambda value: value["calibration"].update(sample_size=2_000),
        lambda value: value["logging"].update(wandb_group="treewm-50task-1m-v1"),
        lambda value: value["paths"].update(run_root=value["paths"]["run_root"].replace("v2", "v1")),
        lambda value: value["settings"][0].update(task_metric_dims=[0, 1]),
        lambda value: value["settings"][2].update(dataset_kind="sharded_100m_sample"),
    ],
)
def test_manifest_rejects_v1_and_v2_scientific_drift(manifest, mutate):
    changed = copy.deepcopy(manifest)
    mutate(changed)
    with pytest.raises(campaign.ManifestError):
        campaign.validate_manifest(changed)


def _contract(**updates):
    value = {
        "contract_sha256": "1" * 64,
        "data_manifest_sha256": "2" * 64,
        "train_manifest_sha256": "3" * 64,
        "normalizer_sha256": "4" * 64,
        "calibration_sha256": "5" * 64,
        "future_recipe_sha256": "6" * 64,
        "chosen_thresholds": {
            "retrieval_radius": 0.75,
            "displacement_threshold": 0.25,
            "cluster_threshold": 0.5,
        },
    }
    value.update(updates)
    return value


def test_trainer_command_pins_calibration_recipe_objective_and_pilot_namespace(
    manifest, monkeypatch
):
    contract = _contract()
    monkeypatch.setattr(campaign, "load_data_contract", lambda *a, **k: contract)
    monkeypatch.setattr(
        campaign,
        "live_contract",
        lambda root: {"code_sha256": "7" * 64, "runtime_sha256": "8" * 64},
    )
    run = campaign.expand_runs(manifest)[0]
    argv, env = campaign.trainer_command(
        run,
        manifest=manifest,
        repo_root=REPO_ROOT,
        run_root=manifest["paths"]["run_root"],
        data_root=manifest["paths"]["data_root"],
        cache_root=manifest["paths"]["raw_cache_root"],
    )
    assert argv[0] == manifest["paths"]["python"]
    assert argv[1] == str(REPO_ROOT / "scripts" / "train.py")
    for required in (
        "experiment=treewm_v2",
        "objective_version=treewm_v2_rms_rank_v1",
        "arm=treewm",
        "train.steps=1000000",
        "train.scheduler_total_steps=1000000",
        "train.num_workers=10",
        "model.scales=[[mixed,32,1.0]]",
        "model.use_depth_embedding=false",
        "future_sets.metric_mode=rms_v2",
        "future_sets.retrieval_radius=0.75",
        "future_sets.displacement_threshold=0.25",
        "future_sets.cluster_threshold=0.5",
        "future_sets.max_modes=4",
        "retrieval.enabled=false",
        "retrieval.num_keys=0",
        "matching.normalization_version=rms_v2",
        "losses.control_target_transform=rms_tanh",
        "losses.detach_world_targets=true",
        "losses.gain_set_context=true",
        "losses.gain_branch_prior_weight=0.0",
        "losses.enabled.mass=false",
        "losses.weights.mass=0.0",
        "+campaign_calibration_sha256=" + "5" * 64,
        "+campaign_future_recipe_sha256=" + "6" * 64,
    ):
        assert required in argv
    full_contract_overrides = {
        *(campaign._override(f"train.{key}", value) for key, value in manifest["objective"]["optimizer_contract"].items()),
        campaign._override(
            "train.gain_loss_every",
            manifest["objective"]["gain_training_contract"]["loss_every"],
        ),
        campaign._override(
            "train.gain_batch_size",
            manifest["objective"]["gain_training_contract"]["batch_size"],
        ),
        campaign._override(
            "train.gain_tree_budget",
            manifest["objective"]["gain_training_contract"]["tree_budget"],
        ),
        *(campaign._override(f"model.{key}", value) for key, value in manifest["objective"]["model_contract"].items()),
        *(campaign._override(f"tree.{key}", value) for key, value in manifest["objective"]["tree_contract"].items()),
        *(campaign._override(f"planner.{key}", value) for key, value in manifest["objective"]["planner_contract"].items()),
        *(campaign._override(f"matching.{key}", value) for key, value in manifest["objective"]["matching_contract"].items()),
    }
    loss_contract = manifest["objective"]["loss_contract"]
    full_contract_overrides.update(
        campaign._override(f"losses.{key}", loss_contract[key])
        for key in (
            "scheduled_sampling_p",
            "scheduled_sampling_warmup",
            "multistep_depth_weights",
            "redundancy_temperature",
            "contrastive_temperature",
            "coverage_space",
            "future_scale",
            "control_batch",
            "recursive_batch",
        )
    )
    for section in ("warmup", "decay", "weights", "enabled"):
        full_contract_overrides.update(
            campaign._override(f"losses.{section}.{key}", value)
            for key, value in loss_contract[section].items()
        )
    assert full_contract_overrides.issubset(set(argv))
    assert not any("randomtreewm" in item or "treewm_v1" in item for item in argv)
    assert env["TREEWM_DATA_SHA256"] == "2" * 64
    assert env["TREEWM_CALIBRATION_SHA256"] == "5" * 64
    assert env["TREEWM_FUTURE_RECIPE_SHA256"] == "6" * 64
    assert env["WANDB_PROJECT"] == "treewm-50task-formal-v2"
    assert env["WANDB_RUN_ID"] == run.wandb_id
    assert env["OMP_NUM_THREADS"] == env["MKL_NUM_THREADS"] == "1"
    assert not any("KEY" in key or "TOKEN" in key or "SECRET" in key for key in env)

    pilot_argv, pilot_env = campaign.trainer_command(
        run,
        manifest=manifest,
        repo_root=REPO_ROOT,
        run_root=manifest["paths"]["pilot_run_root"],
        data_root=manifest["paths"]["data_root"],
        cache_root=manifest["paths"]["raw_cache_root"],
    )
    assert "train.steps=5000" in pilot_argv
    assert "train.scheduler_total_steps=1000000" in pilot_argv
    assert "train.ckpt_every=500" in pilot_argv
    assert "eval.final_episodes_per_task=1" in pilot_argv
    assert pilot_env["WANDB_PROJECT"].endswith("-pilot")
    assert pilot_env["WANDB_RUN_ID"] != run.wandb_id
    assert pilot_env["TREEWM_PROTOCOL_SHA256"] != env["TREEWM_PROTOCOL_SHA256"]


def test_all_commands_preserve_formal_venv_symlink_and_it_imports_hydra(
    manifest, monkeypatch, tmp_path
):
    monkeypatch.setattr(campaign, "load_data_contract", lambda *a, **k: _contract())
    monkeypatch.setattr(
        campaign,
        "live_contract",
        lambda root: {"code_sha256": "7" * 64, "runtime_sha256": "8" * 64},
    )
    commands = [
        campaign.trainer_command(
            run,
            manifest=manifest,
            repo_root=REPO_ROOT,
            run_root=manifest["paths"]["run_root"],
            data_root=manifest["paths"]["data_root"],
            cache_root=manifest["paths"]["raw_cache_root"],
        )[0]
        for run in campaign.expand_runs(manifest)
    ]
    assert {command[0] for command in commands} == {manifest["paths"]["python"]}
    assert Path(commands[0][0]).is_symlink()
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
    with pytest.raises(ValueError, match="immutable"):
        campaign.trainer_command(
            campaign.expand_runs(manifest)[0],
            manifest=manifest,
            python_executable=Path(manifest["paths"]["python"]).resolve(),
            repo_root=REPO_ROOT,
            run_root=manifest["paths"]["run_root"],
            data_root=manifest["paths"]["data_root"],
            cache_root=manifest["paths"]["raw_cache_root"],
        )


def test_data_contract_binds_raw_train_calibration_recipe_and_detects_source_drift(
    manifest, tmp_path, monkeypatch
):
    setting = manifest["settings"][0]
    data_root = tmp_path / "data"
    cache_root = tmp_path / "raw-cache"
    contract_root = tmp_path / "contracts-v2"
    source_paths = campaign.required_dataset_files(manifest, data_root, setting)
    raw_entries = []
    for split, path in zip(("train", "val"), source_paths, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(split.encode())
        stat = path.stat()
        raw_entries.append(
            {
                "split": split,
                "path": path.name,
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    raw_dir = cache_root / f"{setting['source_name']}__test"
    raw_dir.mkdir(parents=True)
    raw = {
        "dataset": setting["source_name"],
        "source_manifest_sha256": "2" * 64,
        "source_files": raw_entries,
        "norm_stats": {"obs_mean": [0.0], "obs_std": [1.0], "act_mean": [0.0], "act_std": [1.0]},
    }
    raw_path = raw_dir / "manifest.json"
    raw_path.write_text(json.dumps(raw))
    normalizer_sha = campaign.normalizer_sha256(raw["norm_stats"])
    calibration_path = campaign.calibration_contract_path(contract_root, setting["id"])
    calibration_path.parent.mkdir(parents=True)
    calibration_payload = {"contract_sha256": "5" * 64, "chosen": _contract()["chosen_thresholds"]}
    calibration_path.write_text(json.dumps(calibration_payload))
    recipe_path = campaign.recipe_root_path(contract_root, setting["id"]) / "manifest.json"
    recipe_path.parent.mkdir(parents=True)
    recipe_path.write_text(json.dumps({"recipe_sha256": "6" * 64}))
    monkeypatch.setattr(calibration, "validate_contract", lambda *a, **k: None)
    recipe_validation = {}

    def capture_recipe_validation(*args, **kwargs):
        recipe_validation.update(kwargs)

    monkeypatch.setattr(future_recipe, "validate_recipe_manifest", capture_recipe_validation)
    current = campaign._validate_current_source_files(manifest, setting, data_root, raw_entries)
    body = {
        "schema_version": 2,
        "status": "complete",
        "campaign_id": manifest["campaign_id"],
        "objective_version": manifest["method"]["objective_version"],
        "campaign_protocol_sha256": campaign.protocol_sha256(manifest),
        "setting_id": setting["id"],
        "dataset_kind": setting["dataset_kind"],
        "raw_cache_read_only": True,
        "data_manifest_sha256": "2" * 64,
        "train_manifest_sha256": campaign.train_inventory_sha256(raw_entries),
        "validation_manifest_sha256": campaign.stable_hash(
            [
                {
                    "split": "val",
                    "index": raw_entries[1].get("index"),
                    "path": raw_entries[1]["path"],
                    "size": raw_entries[1]["size"],
                    "sha256": raw_entries[1]["sha256"],
                }
            ]
        ),
        "normalizer_sha256": normalizer_sha,
        "raw_cache_manifest": str(raw_path),
        "raw_cache_manifest_file_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "source_files": current,
        "calibration_path": str(calibration_path),
        "calibration_sha256": "5" * 64,
        "chosen_thresholds": calibration_payload["chosen"],
        "future_recipe_manifest": str(recipe_path),
        "future_recipe_sha256": "6" * 64,
    }
    body["contract_sha256"] = campaign.stable_hash(body)
    path = campaign.data_contract_path(contract_root, setting["id"])
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(body))
    assert campaign.load_data_contract(
        manifest,
        setting,
        data_root=data_root,
        cache_root=cache_root,
        contract_root=contract_root,
    )["calibration_sha256"] == "5" * 64
    assert recipe_validation["verify_file_hash"] is False
    campaign.load_data_contract(
        manifest,
        setting,
        data_root=data_root,
        cache_root=cache_root,
        contract_root=contract_root,
        verify_recipe_files=True,
    )
    assert recipe_validation["verify_file_hash"] is True
    source_paths[0].write_bytes(b"drift")
    with pytest.raises(ValueError, match="changed"):
        campaign.load_data_contract(
            manifest,
            setting,
            data_root=data_root,
            cache_root=cache_root,
            contract_root=contract_root,
        )


def _resolved_config(manifest, run, contract):
    objective = manifest["objective"]
    training = manifest["training"]
    loss_contract = objective["loss_contract"]
    setting = campaign.setting_for_run(manifest, run)
    return {
        "objective_version": manifest["method"]["objective_version"],
        "arm": "treewm",
        "seed": run.seed,
        "run_name": run.run_name,
        "resume": "auto",
        "train": {
            "steps": 1_000_000,
            "scheduler_total_steps": 1_000_000,
            "gradient_checkpointing": True,
            "batch_size": training["batch_size"],
            "separate_gain_grad_clip": True,
            "world_grad_clip": objective["world_grad_clip"],
            "gain_grad_clip": objective["gain_grad_clip"],
            "gain_loss_every": objective["gain_training_contract"]["loss_every"],
            "gain_batch_size": objective["gain_training_contract"]["batch_size"],
            "gain_tree_budget": objective["gain_training_contract"]["tree_budget"],
            **objective["optimizer_contract"],
        },
        "model": {
            "branch_factor": 4,
            "scales": objective["q_scales"],
            "use_depth_embedding": False,
            **objective["model_contract"],
        },
        "tree": {
            "node_budget": 64,
            "scorer": "learned",
            "keep_threshold": 0.5,
            **objective["tree_contract"],
        },
        "future_sets": {
            "metric_mode": "rms_v2",
            "max_modes": 4,
            "cache": False,
            "shared_cache": True,
            "num_neighbors": training["future_num_neighbors"],
            "query_multiplier": training["future_query_multiplier"],
            "time_exclusion": training["future_time_exclusion"],
            "include_self": training["future_include_self"],
            "horizons": training["future_horizons"],
            "h_max": training["future_h_max"],
            "horizon_rule": training["future_horizon_rule"],
            "fixed_horizon": training["future_fixed_horizon"],
            "cluster_method": training["future_cluster_method"],
            "multi_step_depth": training["future_multi_step_depth"],
            "retrieval_pool": training["future_retrieval_pool"],
            **contract["chosen_thresholds"],
        },
        "env": {"task_metric_dims": setting["task_metric_dims"]},
        "matching": {
            "normalization_version": "rms_v2",
            "num_horizons": 5,
            **objective["matching_contract"],
        },
        "retrieval": {"enabled": False, "num_keys": 0},
        "losses": {
            "control_objective": objective["control_objective"],
            "control_target_transform": "rms_tanh",
            "control_endpoint_key": "fut_metric_endpoint",
            "control_allow_endpoint_fallback": False,
            "control_require_single_scale": True,
            "control_metric_weight": objective["control_metric_weight"],
            "control_rank_weight": objective["control_rank_weight"],
            "control_rank_temperature": objective["control_rank_temperature"],
            "detach_world_targets": True,
            "bind_negative_margin": objective["bind_negative_margin"],
            "gain_target": objective["gain_target"],
            "gain_set_context": True,
            "gain_rank_weight": objective["gain_rank_weight"],
            "gain_calibration_weight": objective["gain_calibration_weight"],
            "gain_branch_prior_weight": 0.0,
            "keep_balance": False,
            **{
                key: loss_contract[key]
                for key in (
                    "scheduled_sampling_p",
                    "scheduled_sampling_warmup",
                    "multistep_depth_weights",
                    "redundancy_temperature",
                    "contrastive_temperature",
                    "coverage_space",
                    "future_scale",
                    "control_batch",
                    "recursive_batch",
                )
            },
            "warmup": loss_contract["warmup"],
            "decay": loss_contract["decay"],
            "enabled": loss_contract["enabled"],
            "weights": loss_contract["weights"],
        },
        "planner": {**objective["planner_contract"], "max_env_steps": setting["max_episode_steps"]},
        "campaign_calibration_sha256": contract["calibration_sha256"],
        "campaign_data_contract_sha256": contract["contract_sha256"],
        "campaign_future_recipe_sha256": contract["future_recipe_sha256"],
    }


def test_completion_requires_named_v2_calibration_recipe_and_resolved_config(
    manifest, tmp_path, monkeypatch
):
    run = campaign.expand_runs(manifest)[0]
    run_dir = tmp_path / "formal-run"
    monkeypatch.setattr(campaign, "run_directory", lambda root, spec: run_dir.absolute())
    contract = _contract()
    monkeypatch.setattr(campaign, "load_data_contract", lambda *a, **k: contract)
    monkeypatch.setattr(
        campaign,
        "live_contract",
        lambda root: {"code_sha256": "7" * 64, "runtime_sha256": "8" * 64},
    )
    config = _resolved_config(manifest, run, contract)
    hydra_path = run_dir / "hydra" / ".hydra" / "config.yaml"
    hydra_path.parent.mkdir(parents=True)
    OmegaConf.save(OmegaConf.create(config), hydra_path)
    identity_config = copy.deepcopy(config)
    identity_config["resume"] = None
    identity = {
        "schema_version": 1,
        "objective_version": manifest["method"]["objective_version"],
        "run_dir": str(run_dir.absolute()),
        "run_name": run.run_name,
        "arm": "treewm",
        "env_name": run.env_name,
        "setting": run.setting_id,
        "dataset_kind": run.dataset_kind,
        "source_name": run.source_name,
        "seed": run.seed,
        "total_steps": 1_000_000,
        "scheduler_total_steps": 1_000_000,
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
        "config_sha256": campaign.stable_hash(identity_config),
        "protocol_sha256": campaign.run_protocol_sha256(manifest, contract),
        "code_sha256": "7" * 64,
        "runtime_sha256": "8" * 64,
        "data_manifest_sha256": "2" * 64,
        "calibration_sha256": "5" * 64,
        "future_recipe_sha256": "6" * 64,
        "retrieval_enabled": False,
        "retrieval_num_keys": 0,
        "wandb_project": manifest["logging"]["wandb_project"],
        "wandb_group": manifest["logging"]["wandb_group"],
        "wandb_mode": "online",
        "wandb_id": run.wandb_id,
    }
    identity_sha = campaign.stable_hash(identity)
    metrics = {"eval/num_episodes": 250}
    for task in range(1, 6):
        metrics[f"eval/task{task}/num_episodes"] = 50
        metrics[f"eval/task{task}/success_rate"] = 0.2
    results = [{"task_id": task, "success": 0.0} for task in range(1, 6) for _ in range(50)]
    progress = {
        "schema_version": 1,
        "objective_version": manifest["method"]["objective_version"],
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
        "objective_version": manifest["method"]["objective_version"],
        "status": "complete",
        "run_identity": identity,
        "identity_sha256": identity_sha,
        "protocol_sha256": identity["protocol_sha256"],
        "code_sha256": "7" * 64,
        "runtime_sha256": "8" * 64,
        "data_manifest_sha256": "2" * 64,
        "calibration_sha256": "5" * 64,
        "future_recipe_sha256": "6" * 64,
        "retrieval_enabled": False,
        "retrieval_num_keys": 0,
        "arm": "treewm",
        "model_class": "TreeWM",
        "scorer": "learned",
        "setting": run.setting_id,
        "seed": run.seed,
        "wandb_id": run.wandb_id,
        "completed_updates": 1_000_000,
        "scheduler_total_steps": 1_000_000,
        "final_eval_step": 1_000_000,
        "task_ids": [1, 2, 3, 4, 5],
        "episodes_per_task": 50,
        "node_budget": 64,
        "branch_factor": 4,
        "gradient_checkpointing": True,
        "future_set_cache": False,
        "shared_cache": True,
        "final_evaluation": metrics,
        "final_eval_progress": "final_eval_progress.json",
    }
    (run_dir / "COMPLETED.json").write_text(json.dumps(payload))
    assert campaign.completion_is_valid(
        run_dir,
        run,
        manifest=manifest,
        repo_root=REPO_ROOT,
    )
    drifted_config = copy.deepcopy(config)
    drifted_config["losses"]["weights"]["state"] = 2.0
    drifted_path = run_dir / "drifted-config.yaml"
    OmegaConf.save(OmegaConf.create(drifted_config), drifted_path)
    drifted_identity_config = copy.deepcopy(drifted_config)
    drifted_identity_config["resume"] = None
    assert not campaign._config_contract_is_valid(
        drifted_path,
        campaign.stable_hash(drifted_identity_config),
        manifest,
        run,
        contract,
    )
    payload["scheduler_total_steps"] = 5_000
    (run_dir / "COMPLETED.json").write_text(json.dumps(payload))
    assert not campaign.completion_is_valid(
        run_dir,
        run,
        manifest=manifest,
        repo_root=REPO_ROOT,
    )
    payload["scheduler_total_steps"] = 1_000_000
    payload["calibration_sha256"] = "9" * 64
    (run_dir / "COMPLETED.json").write_text(json.dumps(payload))
    assert not campaign.completion_is_valid(
        run_dir,
        run,
        manifest=manifest,
        repo_root=REPO_ROOT,
    )


def test_normalizer_hash_is_shared_exactly_with_recipe_layer():
    stats = {
        "obs_mean": [0.1, -0.2],
        "obs_std": [1.0, 2.0],
        "act_mean": [0.0],
        "act_std": [0.5],
    }
    assert campaign.normalizer_sha256(stats) == future_recipe.normalizer_state_sha256(stats)
