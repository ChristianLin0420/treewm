#!/usr/bin/env python3
"""Manifest loading and command construction for the RQL reproduction.

This module deliberately contains no JAX, OGBench, W&B, or Slurm imports.  It is
the stable interface between the campaign layer and ``upstream_rql/main.py``.
The trainer owns checkpoint internals; this layer owns run identity, sharding,
completion checks, and exact command-line flags.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_SETTING_COUNT = 10
EXPECTED_TASK_IDS = (1, 2, 3, 4, 5)
EXPECTED_SEEDS = (0, 1, 2, 3)
EXPECTED_RUN_COUNT = 200
EXPECTED_WORKERS = 16
EXPECTED_ALLOCATION_SHARDS = 13
EXPECTED_UPSTREAM_COMMIT = "229c956efb4494c2b9bb0bbddbd67b761c93f1cc"
SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
IMMUTABLE_IDENTITY_KEYS = {
    "upstream_commit",
    "run_name",
    "run_dir",
    "wandb_id",
    "wandb_project",
    "protocol_sha256",
    "code_manifest_sha256",
    "code_files",
    "runtime_software",
    "run_group",
    "env_name",
    "seed",
    "offline_steps",
    "online_steps",
    "sparse",
    "p_aug",
    "frame_stack",
    "utd",
    "eval_interval",
    "eval_episodes",
    "final_eval_episodes",
    "video_episodes",
    "video_frame_skip",
    "gradient_checkpointing",
    "ogbench_standard_dataset_dir",
    "dataset_replace_interval",
    "dataset_paths",
    "agent",
}


class ManifestError(ValueError):
    """Raised when the campaign manifest is internally inconsistent."""


@dataclass(frozen=True)
class RunSpec:
    """One setting/task/seed training run."""

    index: int
    setting_id: str
    task_id: int
    seed: int
    env_name: str
    dataset_kind: str
    dataset_name: str
    dataset_directory: str | None
    sparse: bool
    agent: Mapping[str, Any]
    run_id: str
    wandb_id: str


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash parsed semantics, independent of whitespace or JSON key order."""

    return hashlib.sha256(_canonical_json(manifest).encode("utf-8")).hexdigest()


def immutable_identity_sha256(identity: Mapping[str, Any]) -> str:
    """Use the trainer's stable JSON encoding for a persisted identity."""

    return hashlib.sha256(_canonical_json(identity).encode("utf-8")).hexdigest()


def current_trainer_contract(upstream_dir: os.PathLike[str] | str | None = None) -> dict[str, Any]:
    """Load the trainer's authoritative dependency-light provenance helpers.

    Loading ``utils/resume.py`` directly avoids importing JAX, OGBench, W&B,
    or the trainer's global Abseil flags.  The shared helper owns the exact
    runtime-critical file list, so the campaign cannot silently diverge from
    the trainer when deciding whether a completed run may be skipped.
    """

    root = (
        Path(upstream_dir).resolve()
        if upstream_dir is not None
        else Path(__file__).resolve().parent / "upstream_rql"
    )
    resume_path = root / "utils" / "resume.py"
    spec = importlib.util.spec_from_file_location("_rql_completion_contract", resume_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load trainer completion contract: {resume_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fingerprint = module.trainer_code_fingerprint(root)
    provenance = module.collect_runtime_provenance()
    runtime_software = {
        "python": provenance["python"],
        "packages": provenance["packages"],
    }
    return {
        "completion_schema_version": module.COMPLETION_SCHEMA_VERSION,
        "code_files": fingerprint["files"],
        "code_manifest_sha256": fingerprint["manifest_sha256"],
        "runtime_software": runtime_software,
        "runtime_software_sha256": module.stable_json_hash(runtime_software),
    }


def load_manifest(path: os.PathLike[str] | str) -> dict[str, Any]:
    manifest_path = Path(path)
    try:
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ManifestError(f"manifest does not exist: {manifest_path}") from exc
    except json.JSONDecodeError as exc:
        raise ManifestError(f"invalid JSON in {manifest_path}: {exc}") from exc
    validate_manifest(data)
    return data


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed if the formal protocol has drifted."""

    if manifest.get("schema_version") != 1:
        raise ManifestError("schema_version must be 1")
    if manifest.get("expected_run_count") != EXPECTED_RUN_COUNT:
        raise ManifestError(f"expected_run_count must be {EXPECTED_RUN_COUNT}")

    axes = manifest.get("axes", {})
    if tuple(axes.get("task_ids", ())) != EXPECTED_TASK_IDS:
        raise ManifestError(f"task_ids must be exactly {list(EXPECTED_TASK_IDS)}")
    if tuple(axes.get("seeds", ())) != EXPECTED_SEEDS:
        raise ManifestError(f"seeds must be exactly {list(EXPECTED_SEEDS)}")

    execution = manifest.get("execution", {})
    if execution.get("workers") != EXPECTED_WORKERS:
        raise ManifestError(f"execution.workers must be {EXPECTED_WORKERS}")
    if execution.get("allocation_shards") != EXPECTED_ALLOCATION_SHARDS:
        raise ManifestError(f"execution.allocation_shards must be {EXPECTED_ALLOCATION_SHARDS}")
    if execution.get("assignment") != "manifest_index_equals_allocation_shard_times_workers_plus_rank":
        raise ManifestError("execution.assignment does not describe the formal contiguous array mapping")
    if (EXPECTED_RUN_COUNT + EXPECTED_WORKERS - 1) // EXPECTED_WORKERS != EXPECTED_ALLOCATION_SHARDS:
        raise ManifestError("allocation shard count is not the minimal exact cover for the campaign")
    if manifest.get("upstream", {}).get("commit") != EXPECTED_UPSTREAM_COMMIT:
        raise ManifestError(f"upstream.commit must be {EXPECTED_UPSTREAM_COMMIT}")

    training = manifest.get("training", {})
    required_training = {
        "offline_steps": 1_000_000,
        "online_steps": 0,
        "eval_episodes": 50,
        "final_eval_episodes": 50,
        "eval_interval": 100_000,
        "gradient_checkpointing": True,
    }
    for key, expected in required_training.items():
        if training.get(key) != expected:
            raise ManifestError(f"training.{key} must be {expected!r}")

    settings = manifest.get("settings")
    if not isinstance(settings, list) or len(settings) != EXPECTED_SETTING_COUNT:
        raise ManifestError(f"settings must contain exactly {EXPECTED_SETTING_COUNT} entries")
    seen_ids: set[str] = set()
    for setting in settings:
        setting_id = setting.get("id")
        if not isinstance(setting_id, str) or not SAFE_ID.fullmatch(setting_id):
            raise ManifestError(f"unsafe setting id: {setting_id!r}")
        if setting_id in seen_ids:
            raise ManifestError(f"duplicate setting id: {setting_id}")
        seen_ids.add(setting_id)

        template = setting.get("env_template")
        if not isinstance(template, str) or template.count("{task_id}") != 1:
            raise ManifestError(f"{setting_id}: env_template needs one {{task_id}} placeholder")
        rendered = [template.format(task_id=task_id) for task_id in EXPECTED_TASK_IDS]
        if any(f"-singletask-task{task_id}-v0" not in env for task_id, env in zip(EXPECTED_TASK_IDS, rendered)):
            raise ManifestError(f"{setting_id}: task-specific OGBench environment names are malformed")

        dataset = setting.get("dataset", {})
        kind = dataset.get("kind")
        if kind not in {"standard", "sharded_100m"}:
            raise ManifestError(f"{setting_id}: unsupported dataset kind {kind!r}")
        if not isinstance(dataset.get("name"), str) or not dataset["name"].endswith("-v0"):
            raise ManifestError(f"{setting_id}: dataset.name must be an OGBench v0 name")
        if kind == "sharded_100m":
            if dataset.get("expected_train_shards") != 100:
                raise ManifestError(f"{setting_id}: 100M dataset must declare 100 train shards")
            if dataset.get("expected_validation_shards") != 100:
                raise ManifestError(f"{setting_id}: 100M dataset must declare 100 validation shards")
            if dataset.get("replace_interval") != 1000:
                raise ManifestError(f"{setting_id}: 100M shard replacement interval must be 1000")

        agent = setting.get("agent", {})
        for key in ("alpha", "expectile", "ensemble_ct", "rho", "h", "discount", "batch_size"):
            if key not in agent:
                raise ManifestError(f"{setting_id}: missing agent.{key}")
        if agent["ensemble_ct"] != 10 or agent["batch_size"] != 256:
            raise ManifestError(f"{setting_id}: official ensemble/batch settings have drifted")

    if len(expand_runs_unchecked(manifest)) != EXPECTED_RUN_COUNT:
        raise ManifestError(f"expanded campaign must contain exactly {EXPECTED_RUN_COUNT} runs")


def expand_runs_unchecked(manifest: Mapping[str, Any]) -> list[RunSpec]:
    task_ids = tuple(manifest["axes"]["task_ids"])
    seeds = tuple(manifest["axes"]["seeds"])
    protocol_tag = str(manifest["campaign_id"])
    runs: list[RunSpec] = []
    for setting in manifest["settings"]:
        for task_id in task_ids:
            for seed in seeds:
                run_key = {
                    "protocol": protocol_tag,
                    "setting": setting["id"],
                    "task_id": task_id,
                    "seed": seed,
                    "env_name": setting["env_template"].format(task_id=task_id),
                    "dataset": setting["dataset"],
                    "sparse": bool(setting["sparse"]),
                    "agent": setting["agent"],
                    "training": manifest["training"],
                }
                identity = hashlib.sha256(_canonical_json(run_key).encode("utf-8")).hexdigest()
                run_id = f"{setting['id']}-task{task_id}-seed{seed}"
                runs.append(
                    RunSpec(
                        index=len(runs),
                        setting_id=setting["id"],
                        task_id=task_id,
                        seed=seed,
                        env_name=run_key["env_name"],
                        dataset_kind=setting["dataset"]["kind"],
                        dataset_name=setting["dataset"]["name"],
                        dataset_directory=setting["dataset"].get("directory"),
                        sparse=bool(setting["sparse"]),
                        agent=dict(setting["agent"]),
                        run_id=run_id,
                        wandb_id=f"rql-{identity[:24]}",
                    )
                )
    return runs


def expand_runs(manifest: Mapping[str, Any]) -> list[RunSpec]:
    validate_manifest(manifest)
    runs = expand_runs_unchecked(manifest)
    run_ids = [run.run_id for run in runs]
    wandb_ids = [run.wandb_id for run in runs]
    if len(set(run_ids)) != len(run_ids) or len(set(wandb_ids)) != len(wandb_ids):
        raise ManifestError("expanded run or W&B identities are not unique")
    return runs


def worker_runs(
    runs: Sequence[RunSpec],
    worker_index: int,
    workers: int = EXPECTED_WORKERS,
    *,
    allocation_shard: int | None,
    allocation_shards: int = EXPECTED_ALLOCATION_SHARDS,
) -> list[RunSpec]:
    """Return the allocation/rank's sole run, or empty for final idle ranks.

    Ownership is intentionally contiguous and stateless:
    ``global_index = allocation_shard * workers + worker_index``.  Thus all
    200 runs can execute concurrently across 13 two-node allocations without
    duplicate work, while the final allocation retains all 16 ranks for its
    requeue barrier even though only ranks 0--7 own runs.
    """

    if workers != EXPECTED_WORKERS:
        raise ManifestError(f"formal dispatch requires exactly {EXPECTED_WORKERS} workers")
    if allocation_shards != EXPECTED_ALLOCATION_SHARDS:
        raise ManifestError(
            f"formal dispatch requires exactly {EXPECTED_ALLOCATION_SHARDS} allocation shards"
        )
    if allocation_shard is None:
        raise ManifestError("allocation_shard is required; use the Slurm array task index")
    if not 0 <= allocation_shard < allocation_shards:
        raise ManifestError(
            f"allocation shard {allocation_shard} is outside [0, {allocation_shards})"
        )
    if not 0 <= worker_index < workers:
        raise ManifestError(f"worker index {worker_index} is outside [0, {workers})")
    global_index = allocation_shard * workers + worker_index
    if global_index >= len(runs):
        return []
    run = runs[global_index]
    if run.index != global_index:
        raise ManifestError("expanded runs are not in canonical contiguous index order")
    return [run]


def run_directory(run_root: os.PathLike[str] | str, run: RunSpec) -> Path:
    return Path(run_root).resolve() / "runs" / run.setting_id / f"task{run.task_id}" / f"seed{run.seed}"


def completion_is_valid(
    run_dir: os.PathLike[str] | str,
    manifest: Mapping[str, Any],
    run: RunSpec,
) -> bool:
    """Fail closed unless trainer, runtime, run identity, and final eval match."""

    completion = Path(run_dir) / "COMPLETED.json"
    try:
        payload = json.loads(completion.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return False
    try:
        metadata = json.loads((Path(run_dir) / "run_metadata.json").read_text(encoding="utf-8"))
        contract = current_trainer_contract()
    except (FileNotFoundError, OSError, json.JSONDecodeError, ImportError, AttributeError, KeyError):
        return False

    training = manifest["training"]
    protocol_sha256 = manifest_sha256(manifest)
    identity = payload.get("identity")
    identity_sha256 = payload.get("identity_sha256")
    if not isinstance(identity, dict) or set(identity) != IMMUTABLE_IDENTITY_KEYS:
        return False
    if not isinstance(identity_sha256, str) or SHA256.fullmatch(identity_sha256) is None:
        return False
    if immutable_identity_sha256(identity) != identity_sha256:
        return False

    agent = identity.get("agent")
    if not isinstance(agent, dict) or any(agent.get(key) != value for key, value in run.agent.items()):
        return False
    if agent.get("gradient_checkpointing") is not True:
        return False

    dataset_paths = identity.get("dataset_paths")
    standard_dir = identity.get("ogbench_standard_dataset_dir")
    if not isinstance(dataset_paths, list) or not all(isinstance(path, str) for path in dataset_paths):
        return False
    if run.dataset_kind == "standard":
        if dataset_paths or not isinstance(standard_dir, str):
            return False
        if Path(standard_dir).name != manifest["data"]["standard_subdir"]:
            return False
    else:
        setting = next(item for item in manifest["settings"] if item["id"] == run.setting_id)
        file_stem = setting["dataset"]["file_stem"]
        expected_names = {f"{file_stem}-{index:03d}.npz" for index in range(100)}
        paths = [Path(path) for path in dataset_paths]
        if standard_dir is not None or len(paths) != 100 or not all(path.is_absolute() for path in paths):
            return False
        if {path.name for path in paths} != expected_names:
            return False
        if len({path.parent for path in paths}) != 1 or paths[0].parent.name != run.dataset_directory:
            return False

    expected_identity = {
        "upstream_commit": manifest["upstream"]["commit"],
        "run_name": run.run_id,
        "run_dir": str(Path(run_dir).resolve()),
        "wandb_id": run.wandb_id,
        "wandb_project": manifest["logging"]["wandb_project"],
        "protocol_sha256": protocol_sha256,
        "code_manifest_sha256": contract["code_manifest_sha256"],
        "code_files": contract["code_files"],
        "runtime_software": contract["runtime_software"],
        "run_group": manifest["logging"]["wandb_group"],
        "env_name": run.env_name,
        "seed": run.seed,
        "offline_steps": training["offline_steps"],
        "online_steps": training["online_steps"],
        "sparse": run.sparse,
        "p_aug": None,
        "frame_stack": None,
        "utd": 1,
        "eval_interval": training["eval_interval"],
        "eval_episodes": training["eval_episodes"],
        "final_eval_episodes": training["final_eval_episodes"],
        "video_episodes": 0,
        "video_frame_skip": 3,
        "gradient_checkpointing": True,
        "dataset_replace_interval": 1000,
    }
    if any(identity.get(key) != value for key, value in expected_identity.items()):
        return False

    return (
        payload.get("schema_version") == contract["completion_schema_version"] == 1
        and payload.get("status") == "complete"
        and payload.get("identity_sha256") == metadata.get("identity_sha256")
        and payload.get("identity") == metadata.get("identity")
        and payload.get("global_step") == training["offline_steps"]
        and payload.get("final_eval_step") == training["offline_steps"]
        and payload.get("final_eval_episodes") == training["final_eval_episodes"]
        and payload.get("protocol_sha256") == metadata.get("protocol_sha256") == protocol_sha256
        and payload.get("code_manifest_sha256")
        == metadata.get("code_manifest_sha256")
        == contract["code_manifest_sha256"]
        and metadata.get("code_files") == contract["code_files"]
        and payload.get("runtime_software_sha256")
        == metadata.get("runtime_software_sha256")
        == contract["runtime_software_sha256"]
        and isinstance(metadata.get("runtime"), dict)
        and metadata["runtime"].get("python") == contract["runtime_software"]["python"]
        and metadata["runtime"].get("packages") == contract["runtime_software"]["packages"]
        and payload.get("run_name") == run.run_id
        and payload.get("env_name") == run.env_name
        and payload.get("seed") == run.seed
        and payload.get("wandb_id") == run.wandb_id
        and payload.get("upstream_commit")
        == metadata.get("upstream_commit")
        == EXPECTED_UPSTREAM_COMMIT
        and payload.get("gradient_checkpointing") is True
        and isinstance(metadata.get("gradient_checkpointing"), dict)
        and metadata["gradient_checkpointing"].get("enabled") is True
        and payload.get("checkpoint") == "checkpoint.pkl"
        and isinstance(payload.get("final_evaluation"), dict)
        and bool(payload["final_evaluation"])
    )


def _flag(name: str, value: Any) -> str:
    if isinstance(value, bool):
        rendered = "true" if value else "false"
    elif isinstance(value, float):
        rendered = format(value, ".12g")
    else:
        rendered = str(value)
    return f"--{name}={rendered}"


def build_train_command(
    manifest: Mapping[str, Any],
    run: RunSpec,
    *,
    python_executable: os.PathLike[str] | str,
    upstream_main: os.PathLike[str] | str,
    run_root: os.PathLike[str] | str,
    data_root: os.PathLike[str] | str,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_mode: str = "online",
    walltime_seconds_override: int | None = None,
) -> list[str]:
    """Build one argv list without shell interpolation or credentials."""

    validate_manifest(manifest)
    main_path = Path(upstream_main).resolve()
    upstream_dir = main_path.parent
    run_dir = run_directory(run_root, run)
    data_dir = Path(data_root).resolve()
    training = manifest["training"]
    project = wandb_project or manifest["logging"]["wandb_project"]

    command = [
        str(python_executable),
        str(main_path),
        _flag("agent", upstream_dir / "agents" / "rql.py"),
        _flag("env_name", run.env_name),
        _flag("seed", run.seed),
        _flag("run_group", manifest["logging"]["wandb_group"]),
        _flag("run_dir", run_dir),
        _flag("run_name", run.run_id),
        _flag("wandb_id", run.wandb_id),
        _flag("wandb_project", project),
        _flag("wandb_mode", wandb_mode),
        _flag("protocol_sha256", manifest_sha256(manifest)),
        _flag("resume", True),
        _flag("gradient_checkpointing", training["gradient_checkpointing"]),
        _flag(
            "walltime_seconds",
            training["walltime_seconds"] if walltime_seconds_override is None else walltime_seconds_override,
        ),
        _flag("checkpoint_interval", training["checkpoint_interval"]),
        _flag("offline_steps", training["offline_steps"]),
        _flag("online_steps", training["online_steps"]),
        _flag("eval_episodes", training["eval_episodes"]),
        _flag("final_eval_episodes", training["final_eval_episodes"]),
        _flag("eval_interval", training["eval_interval"]),
        _flag("log_interval", training["log_interval"]),
        _flag("save_interval", training["save_interval"]),
        _flag("sparse", run.sparse),
    ]
    if wandb_entity:
        command.append(_flag("wandb_entity", wandb_entity))

    if run.dataset_kind == "standard":
        command.append(_flag("ogbench_standard_dataset_dir", data_dir / manifest["data"]["standard_subdir"]))
    elif run.dataset_kind == "sharded_100m":
        if not run.dataset_directory:
            raise ManifestError(f"{run.setting_id}: 100M dataset directory is missing")
        command.extend(
            [
                _flag("ogbench_dataset_dir", data_dir / manifest["data"]["large_subdir"] / run.dataset_directory),
                _flag("dataset_replace_interval", 1000),
                _flag("buffer_size", 100_000_000),
            ]
        )
    else:  # pragma: no cover - validate_manifest rejects this first.
        raise ManifestError(f"unsupported dataset kind: {run.dataset_kind}")

    for key in ("alpha", "expectile", "ensemble_ct", "rho", "h", "discount", "batch_size"):
        command.append(_flag(f"agent.{key}", run.agent[key]))
    return command


def standard_dataset_names(manifest: Mapping[str, Any]) -> list[str]:
    validate_manifest(manifest)
    return sorted(
        {
            setting["dataset"]["name"]
            for setting in manifest["settings"]
            if setting["dataset"]["kind"] == "standard"
        }
    )


def large_dataset_specs(manifest: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    validate_manifest(manifest)
    return [
        dict(setting["dataset"])
        for setting in manifest["settings"]
        if setting["dataset"]["kind"] == "sharded_100m"
    ]


def redact_command(command: Iterable[str]) -> str:
    """Human-readable command; argv never contains API tokens by design."""

    import shlex

    return shlex.join(list(command))
