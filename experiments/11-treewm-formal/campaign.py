#!/usr/bin/env python3
"""Dependency-light scientific contract for the formal TreeWM campaign.

TreeWM learns one goal-agnostic world model per dataset/seed and evaluates that model on
all five built-in tasks.  Consequently this campaign has 40 model runs and 200
task-seed evaluations; it must never be expanded into 200 task-specific TreeWM models.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any, Mapping, Sequence


# Slurm invokes the sibling entrypoints by absolute path with no promise that the
# repository is installed or present in PYTHONPATH.  Importing this shared contract is
# therefore also the single fail-closed bootstrap for every campaign executable.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if not (REPOSITORY_ROOT / "treewm" / "__init__.py").is_file():
    raise RuntimeError(f"TreeWM repository package is missing beneath {REPOSITORY_ROOT}")
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


SHA256 = re.compile(r"^[0-9a-f]{64}$")
SETTING_ID = re.compile(r"^[a-z0-9][a-z0-9-]*$")
FORMAL_UPDATES = 1_000_000
FORMAL_TASK_IDS = (1, 2, 3, 4, 5)
FORMAL_SEEDS = (0, 1, 2, 3)
FORMAL_PYTHON = (
    "/lustre/fs11/portfolios/edgeai/projects/edgeai_tao-ptm_image-foundation-model-clip/"
    "users/chrislin/envs/treewm-formal-py311/bin/python"
)
WORKERS_PER_ALLOCATION = 16
ALLOCATION_SHARDS = 3
# Scheduler/dispatcher-facing names retained as an explicit public interface.
EXPECTED_WORKERS = WORKERS_PER_ALLOCATION
EXPECTED_ALLOCATION_SHARDS = ALLOCATION_SHARDS


class ManifestError(ValueError):
    """The immutable campaign protocol is malformed or internally inconsistent."""


@dataclass(frozen=True)
class RunSpec:
    global_index: int
    setting_index: int
    setting_id: str
    env_config: str
    env_name: str
    source_name: str
    dataset_kind: str
    seed: int
    run_name: str
    wandb_id: str

    @property
    def index(self) -> int:
        return self.global_index

    @property
    def run_id(self) -> str:
        return self.run_name


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def load_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_manifest(manifest)
    return manifest


def manifest_sha256(manifest: Mapping[str, Any]) -> str:
    validate_manifest(manifest)
    return stable_hash(manifest)


PROTOCOL_SOURCE_FILES = (
    "campaign.py",
    "prepare_cache.py",
    "dispatcher.py",
    "gpu_preflight.py",
    "train.slurm",
    "stage_data.slurm",
    "aggregate.py",
    "aggregate.slurm",
    "submit.py",
)


def protocol_sha256(
    manifest: Mapping[str, Any], campaign_dir: str | Path | None = None
) -> str:
    """Hash manifest semantics plus every executable campaign orchestration source.

    The lock file itself is deliberately excluded. Documentation and tests are also
    excluded because they cannot alter a submitted run. Missing allowlisted sources are
    represented explicitly, so adding one invalidates a provisional lock.
    """
    validate_manifest(manifest)
    root = Path(campaign_dir or Path(__file__).resolve().parent).resolve()
    sources: dict[str, str] = {}
    for relative in PROTOCOL_SOURCE_FILES:
        path = root / relative
        sources[relative] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else "missing"
    return stable_hash({"manifest_sha256": manifest_sha256(manifest), "sources": sources})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ManifestError(message)


def validate_manifest(manifest: Mapping[str, Any]) -> None:
    """Fail closed on every formal scientific/execution invariant."""
    _require(manifest.get("schema_version") == 1, "schema_version must be 1")
    _require(manifest.get("campaign_id") == "treewm-50task-1m-v1", "unexpected campaign_id")
    _require(manifest.get("expected_model_runs") == 40, "expected_model_runs must be 40")
    _require(
        manifest.get("expected_task_seed_evaluations") == 200,
        "expected_task_seed_evaluations must be 200",
    )
    axes = manifest.get("axes") or {}
    _require(tuple(axes.get("seeds", ())) == FORMAL_SEEDS, "seeds must be exactly 0..3")
    _require(tuple(axes.get("task_ids", ())) == FORMAL_TASK_IDS, "task IDs must be exactly 1..5")

    method = manifest.get("method") or {}
    _require(method.get("arm") == "treewm", "formal arm must be treewm")
    _require(method.get("model_class") == "TreeWM", "formal model class must be TreeWM")
    _require(method.get("scorer") == "learned", "formal scorer must be learned")
    _require(method.get("node_budget") == 64, "formal node budget must be 64")
    _require(method.get("branch_factor") == 4, "formal branch factor must be 4")
    _require(method.get("gradient_checkpointing") is True, "gradient checkpointing is mandatory")

    training = manifest.get("training") or {}
    expected_training = {
        "optimizer_updates": FORMAL_UPDATES,
        "batch_size": 256,
        "max_train_anchors": 300_000,
        "max_validation_anchors": 30_000,
        "shared_cache": True,
        "future_set_cache": False,
        "future_retrieval_pool": 50_000,
        "redundancy_decay_updates": 50_000,
        "checkpoint_every_updates": 2_000,
        "periodic_evaluation_every_updates": 100_000,
        "periodic_episodes_per_task": 1,
        "data_loader_workers": 2,
    }
    for key, value in expected_training.items():
        _require(training.get(key) == value, f"training.{key} must be {value!r}")
    _require(
        training.get("anchor_sampling")
        == "uniform_without_replacement_over_full_valid_transition_universe",
        "anchor sampling must be uniform over the complete valid universe",
    )
    _require(
        training.get("anchor_sampling_algorithm")
        == "numpy_choice_le10m_else_uniform_rejection_v1",
        "anchor sampling implementation drifted",
    )
    _require(training.get("future_horizons") == [4, 8, 16, 32, 64], "learned horizon grid drifted")
    _require(training.get("future_h_max") == 64, "future h_max must be 64")
    _require(training.get("future_horizon_rule") == "displacement", "horizon rule must be displacement")

    evaluation = manifest.get("evaluation") or {}
    _require(evaluation.get("task_split") == "standard", "evaluation split must be standard")
    _require(tuple(evaluation.get("task_ids", ())) == FORMAL_TASK_IDS, "evaluation task IDs drifted")
    _require(evaluation.get("final_episodes_per_task") == 50, "final evaluation must use 50 episodes/task")
    _require(evaluation.get("node_budget") == 64, "evaluation node budget must be 64")
    _require(evaluation.get("seed_rule") == "training_seed", "evaluation seed must equal training seed")

    execution = manifest.get("execution") or {}
    _require(execution.get("allocation_shards") == ALLOCATION_SHARDS, "must use three allocations")
    _require(execution.get("workers_per_allocation") == WORKERS_PER_ALLOCATION, "must use 16 workers")
    _require(execution.get("nodes") == 2, "must use two nodes/allocation")
    _require(execution.get("workers_per_node") == 8, "must use eight workers/node")
    _require(execution.get("gpus_per_worker") == 1, "each worker must own one GPU")
    _require(
        execution.get("mapping") == "global_index=16*allocation_shard+rank",
        "allocation mapping drifted",
    )
    logging = manifest.get("logging") or {}
    _require(logging.get("wandb_project") == "treewm-50task-formal", "W&B project drifted")
    _require(logging.get("wandb_group") == "treewm-50task-1m-v1", "W&B group drifted")
    _require(logging.get("wandb_mode") == "online", "formal W&B mode must be online")

    paths = manifest.get("paths") or {}
    _require(paths.get("python") == FORMAL_PYTHON, "formal Python symlink path drifted")
    _require(Path(FORMAL_PYTHON).is_absolute(), "formal Python path must be absolute")

    settings = manifest.get("settings") or []
    _require(len(settings) == 10, "manifest must contain exactly ten settings")
    ids: set[str] = set()
    full_sources = 0
    for index, setting in enumerate(settings):
        setting_id = setting.get("id")
        _require(isinstance(setting_id, str) and SETTING_ID.fullmatch(setting_id) is not None,
                 f"setting {index} has invalid id")
        _require(setting_id not in ids, f"duplicate setting id {setting_id}")
        ids.add(setting_id)
        _require(setting.get("env_config"), f"{setting_id}: env_config missing")
        _require(str(setting.get("env_name", "")).endswith("-v0"), f"{setting_id}: bad env name")
        _require(str(setting.get("source_name", "")).endswith("-v0"), f"{setting_id}: bad source name")
        _require(setting.get("obs_dim", 0) > 0 and setting.get("action_dim", 0) > 0,
                 f"{setting_id}: invalid dimensions")
        _require(setting.get("max_episode_steps") in {500, 750, 1000, 2000},
                 f"{setting_id}: unverified episode limit")
        expected_relative = setting_id not in {"scene", "puzzle-3x3", "puzzle-4x4-100m"}
        _require(
            setting.get("relative_endpoints") is expected_relative,
            f"{setting_id}: relative endpoint semantics drifted",
        )
        kind = setting.get("dataset_kind")
        _require(kind in {"standard", "sharded_100m_full"}, f"{setting_id}: unsupported data kind")
        if kind == "sharded_100m_full":
            full_sources += 1
            _require(setting.get("source_name", "").endswith("-100m-v0"),
                     f"{setting_id}: 100M source label missing")
            _require(setting.get("env_name") != setting.get("source_name"),
                     f"{setting_id}: 100M source cannot be used as simulator ID")
            _require(setting.get("expected_train_shards") == 100, f"{setting_id}: need 100 train shards")
            _require(setting.get("expected_validation_shards") == 100,
                     f"{setting_id}: need 100 validation shards")
            _require(setting.get("expected_train_transitions") == 100_000_000,
                     f"{setting_id}: need the exact 100M train transitions")
            _require(setting.get("expected_validation_transitions") == 10_000_000,
                     f"{setting_id}: need the exact 10M validation transitions")
            _require(setting.get("expected_train_trajectories") == 100_000,
                     f"{setting_id}: train trajectory count drifted")
            _require(setting.get("expected_validation_trajectories") == 10_000,
                     f"{setting_id}: validation trajectory count drifted")
    _require(full_sources == 2, "exactly two settings must consume full 100M releases")
    _require(len(settings) * len(FORMAL_SEEDS) == 40, "setting/seed product must be 40")
    _require(len(settings) * len(FORMAL_SEEDS) * len(FORMAL_TASK_IDS) == 200,
             "task-evaluation product must be 200")


def _wandb_id(campaign_id: str, setting_id: str, seed: int) -> str:
    return stable_hash({"campaign_id": campaign_id, "setting_id": setting_id, "seed": seed})[:32]


def expand_runs(manifest: Mapping[str, Any]) -> list[RunSpec]:
    validate_manifest(manifest)
    runs: list[RunSpec] = []
    for setting_index, setting in enumerate(manifest["settings"]):
        for seed in manifest["axes"]["seeds"]:
            run_name = f"treewm-{setting['id']}-seed{seed}"
            runs.append(
                RunSpec(
                    global_index=len(runs),
                    setting_index=setting_index,
                    setting_id=setting["id"],
                    env_config=setting["env_config"],
                    env_name=setting["env_name"],
                    source_name=setting["source_name"],
                    dataset_kind=setting["dataset_kind"],
                    seed=int(seed),
                    run_name=run_name,
                    wandb_id=_wandb_id(manifest["campaign_id"], setting["id"], int(seed)),
                )
            )
    if len(runs) != 40 or [run.global_index for run in runs] != list(range(40)):
        raise ManifestError("run expansion is not the exact contiguous 40-run matrix")
    return runs


def run_for_worker(
    manifest: Mapping[str, Any], allocation_shard: int, rank: int
) -> RunSpec | None:
    if not 0 <= allocation_shard < ALLOCATION_SHARDS:
        raise ValueError("allocation_shard must be in [0, 3)")
    if not 0 <= rank < WORKERS_PER_ALLOCATION:
        raise ValueError("rank must be in [0, 16)")
    index = WORKERS_PER_ALLOCATION * allocation_shard + rank
    runs = expand_runs(manifest)
    return runs[index] if index < len(runs) else None


def run_directory(run_root: str | Path, run: RunSpec) -> Path:
    # Must exactly match scripts/train.py's explicit-run-name layout.
    return Path(run_root).expanduser().resolve() / run.setting_id / "treewm" / run.run_name


def setting_for_run(manifest: Mapping[str, Any], run: RunSpec) -> Mapping[str, Any]:
    setting = manifest["settings"][run.setting_index]
    if setting["id"] != run.setting_id:
        raise ManifestError("RunSpec setting index/id mismatch")
    return setting


def required_dataset_files(
    manifest: Mapping[str, Any], data_root: str | Path, setting: Mapping[str, Any]
) -> list[Path]:
    validate_manifest(manifest)
    directory = Path(data_root).expanduser().resolve() / setting["data_subdir"]
    name = setting["source_name"]
    if setting["dataset_kind"] == "standard":
        return [directory / f"{name}.npz", directory / f"{name}-val.npz"]
    stem = name.removesuffix("-100m-v0") + "-v0"
    paths: list[Path] = []
    for shard in range(100):
        paths.extend(
            (directory / f"{stem}-{shard:03d}.npz", directory / f"{stem}-{shard:03d}-val.npz")
        )
    return paths


def data_contract_path(cache_root: str | Path, setting_id: str) -> Path:
    return Path(cache_root).expanduser().resolve() / "contracts" / f"{setting_id}.json"


def load_data_contract(
    manifest: Mapping[str, Any],
    setting: Mapping[str, Any],
    *,
    data_root: str | Path,
    cache_root: str | Path,
) -> dict[str, Any]:
    """Load the stage-produced source/cache identity and revalidate current files."""
    path = data_contract_path(cache_root, setting["id"])
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"missing or invalid data contract {path}: {exc}") from exc
    expected_files = required_dataset_files(manifest, data_root, setting)
    expected_relative = [str(path.relative_to(Path(data_root).resolve())) for path in expected_files]
    entries = payload.get("source_files") or []
    if (
        payload.get("schema_version") != 1
        or payload.get("status") != "complete"
        or payload.get("protocol_sha256") != protocol_sha256(manifest)
        or payload.get("setting_id") != setting["id"]
        or payload.get("dataset_kind") != setting["dataset_kind"]
        or SHA256.fullmatch(str(payload.get("data_manifest_sha256", ""))) is None
        or [entry.get("path") for entry in entries] != expected_relative
    ):
        raise ValueError(f"data contract does not match formal setting: {path}")
    if setting["dataset_kind"] == "sharded_100m_full" and (
        payload.get("full_transition_corpus") is not True
        or payload.get("train_transitions") != setting["expected_train_transitions"]
        or payload.get("validation_transitions") != setting["expected_validation_transitions"]
        or payload.get("train_trajectories") != setting["expected_train_trajectories"]
        or payload.get("validation_trajectories") != setting["expected_validation_trajectories"]
    ):
        raise ValueError(f"full-shard cache counts drifted: {path}")
    for expected_path, entry in zip(expected_files, entries, strict=True):
        try:
            stat = expected_path.stat()
        except OSError as exc:
            raise ValueError(f"source file vanished after cache stage: {expected_path}") from exc
        if stat.st_size != entry.get("size") or stat.st_mtime_ns != entry.get("mtime_ns"):
            raise ValueError(f"source file changed after cache stage: {expected_path}")
        if SHA256.fullmatch(str(entry.get("sha256", ""))) is None:
            raise ValueError(f"source digest missing from data contract: {expected_path}")
    cache_manifest = Path(payload.get("cache_manifest", ""))
    if not cache_manifest.is_file():
        raise ValueError(f"materialized cache manifest vanished: {cache_manifest}")
    return payload


def live_contract(repo_root: str | Path) -> dict[str, Any]:
    from treewm.utils.provenance import runtime_fingerprint, trainer_code_fingerprint

    code = trainer_code_fingerprint(repo_root)
    runtime = runtime_fingerprint()
    return {"code_sha256": code["manifest_sha256"], "runtime_sha256": runtime["sha256"]}


def trainer_command(
    manifest: Mapping[str, Any],
    run: RunSpec,
    *,
    python_executable: str | Path,
    repo_root: str | Path,
    run_root: str | Path,
    data_root: str | Path,
    cache_root: str | Path,
    wandb_project: str | None = None,
    wandb_mode: str = "online",
) -> tuple[list[str], dict[str, str]]:
    """Return exact trainer argv and non-secret stable environment overrides."""
    validate_manifest(manifest)
    expected_project = manifest["logging"]["wandb_project"]
    if wandb_project is not None and wandb_project != expected_project:
        raise ValueError(
            f"formal W&B project is immutable ({expected_project!r}), got {wandb_project!r}"
        )
    if wandb_mode != manifest["logging"]["wandb_mode"]:
        raise ValueError("formal W&B mode is immutable and must be online")
    setting = setting_for_run(manifest, run)
    training = manifest["training"]
    evaluation = manifest["evaluation"]
    method = manifest["method"]
    data_contract = load_data_contract(
        manifest, setting, data_root=data_root, cache_root=cache_root
    )
    contract = live_contract(repo_root)
    final_run_dir = run_directory(run_root, run)
    # A venv may itself be selected by the path used to invoke Python. Resolving this
    # stable symlink to its base interpreter silently drops the formal venv's
    # site-packages (including Hydra), so preserve and enforce the locked path exactly.
    locked_python = os.path.abspath(os.fspath(Path(manifest["paths"]["python"]).expanduser()))
    requested_python = os.path.abspath(os.fspath(Path(python_executable).expanduser()))
    if requested_python != locked_python:
        raise ValueError(
            f"formal Python path is immutable ({locked_python!r}), got {requested_python!r}"
        )
    if not Path(locked_python).is_file() or not os.access(locked_python, os.X_OK):
        raise ValueError(f"formal Python is not executable: {locked_python}")

    def override(name: str, value: object) -> str:
        if isinstance(value, bool):
            rendered = str(value).lower()
        elif isinstance(value, (list, tuple)):
            rendered = "[" + ",".join(str(item) for item in value) + "]"
        else:
            rendered = str(value)
        return f"{name}={rendered}"

    argv = [
        locked_python,
        str(Path(repo_root).resolve() / "scripts" / "train.py"),
        override("env", run.env_config),
        override("arm", method["arm"]),
        override("seed", run.seed),
        override("run_root", Path(run_root).expanduser().resolve()),
        override("run_name", run.run_name),
        override("resume", "auto"),
        override("train.steps", training["optimizer_updates"]),
        override("train.gradient_checkpointing", method["gradient_checkpointing"]),
        override("train.batch_size", training["batch_size"]),
        override("train.max_train_anchors", training["max_train_anchors"]),
        override("train.max_val_anchors", training["max_validation_anchors"]),
        override("train.num_workers", training["data_loader_workers"]),
        override("train.ckpt_every", training["checkpoint_every_updates"]),
        override("train.eval_every", training["periodic_evaluation_every_updates"]),
        override("train.viz_every", training["visualization_every_updates"]),
        override("train.viz_every_early", training["early_visualization_every_updates"]),
        override("train.viz_early_until", training["early_visualization_until_update"]),
        override("tree.node_budget", method["node_budget"]),
        override("tree.scorer", method["scorer"]),
        override("model.branch_factor", method["branch_factor"]),
        override("future_sets.shared_cache", training["shared_cache"]),
        override("future_sets.cache", training["future_set_cache"]),
        override("future_sets.horizons", training["future_horizons"]),
        override("future_sets.h_max", training["future_h_max"]),
        override("future_sets.horizon_rule", training["future_horizon_rule"]),
        override("future_sets.retrieval_pool", training["future_retrieval_pool"]),
        override("future_sets.relative_endpoints", setting["relative_endpoints"]),
        override("retrieval.num_keys", training["latent_retrieval_keys"]),
        override("losses.decay.redundancy", training["redundancy_decay_updates"]),
        override("eval.task_split", evaluation["task_split"]),
        override("eval.episodes_per_task", training["periodic_episodes_per_task"]),
        override("eval.final_episodes_per_task", evaluation["final_episodes_per_task"]),
        override("eval.seed", run.seed),
        override("planner.max_env_steps", setting["max_episode_steps"]),
        override("hydra.run.dir", final_run_dir / "hydra"),
        override("hydra.job.chdir", False),
    ]
    env = {
        "TREEWM_PROTOCOL_SHA256": protocol_sha256(manifest),
        "TREEWM_CODE_SHA256": contract["code_sha256"],
        "TREEWM_RUNTIME_SHA256": contract["runtime_sha256"],
        "TREEWM_DATA_SHA256": data_contract["data_manifest_sha256"],
        "TREEWM_DATA_ROOT": str(Path(data_root).expanduser().resolve()),
        "TREEWM_CACHE": str(Path(cache_root).expanduser().resolve()),
        "TREEWM_RUN_NAME": run.run_name,
        "WANDB_PROJECT": expected_project,
        "WANDB_RUN_GROUP": manifest["logging"]["wandb_group"],
        "WANDB_RUN_ID": run.wandb_id,
        "WANDB_MODE": wandb_mode,
    }
    return argv, env


def completion_is_valid(
    run_dir: str | Path,
    manifest: Mapping[str, Any],
    run: RunSpec,
    *,
    repo_root: str | Path,
    data_manifest_sha256: str | None = None,
    cache_root: str | Path | None = None,
) -> bool:
    """Strictly validate a completion before skip or aggregation."""
    try:
        payload = json.loads((Path(run_dir) / "COMPLETED.json").read_text(encoding="utf-8"))
        live = live_contract(repo_root)
        metrics = payload["final_evaluation"]
        identity = payload["run_identity"]
        if data_manifest_sha256 is None:
            setting = setting_for_run(manifest, run)
            data_manifest_sha256 = load_data_contract(
                manifest,
                setting,
                data_root=manifest["paths"]["data_root"],
                cache_root=cache_root or manifest["paths"]["cache_root"],
            )["data_manifest_sha256"]
        expected_data = data_manifest_sha256
        expected_identity_fields = {
            "run_dir": str(Path(run_dir).resolve()),
            "run_name": run.run_name,
            "arm": "treewm",
            "env_name": run.env_name,
            "setting": run.setting_id,
            "dataset_kind": run.dataset_kind,
            "source_name": run.source_name,
            "seed": run.seed,
            "total_steps": FORMAL_UPDATES,
            "world_size": 1,
            "model_class": "TreeWM",
            "scorer": "learned",
            "node_budget": 64,
            "branch_factor": 4,
            "gradient_checkpointing": True,
            "future_set_cache": False,
            "shared_cache": True,
            "task_ids": list(FORMAL_TASK_IDS),
            "final_episodes_per_task": 50,
            "protocol_sha256": protocol_sha256(manifest),
            "code_sha256": live["code_sha256"],
            "runtime_sha256": live["runtime_sha256"],
            "data_manifest_sha256": expected_data,
            "wandb_project": manifest["logging"]["wandb_project"],
            "wandb_group": manifest["logging"]["wandb_group"],
            "wandb_mode": "online",
            "wandb_id": run.wandb_id,
        }
        if any(identity.get(key) != value for key, value in expected_identity_fields.items()):
            return False
        if SHA256.fullmatch(str(identity.get("config_sha256", ""))) is None:
            return False
        if stable_hash(identity) != payload.get("identity_sha256"):
            return False
        if (
            payload.get("schema_version") != 1
            or payload.get("status") != "complete"
            or payload.get("protocol_sha256") != protocol_sha256(manifest)
            or payload.get("code_sha256") != live["code_sha256"]
            or payload.get("runtime_sha256") != live["runtime_sha256"]
            or payload.get("data_manifest_sha256") != expected_data
            or payload.get("arm") != "treewm"
            or payload.get("model_class") != "TreeWM"
            or payload.get("scorer") != "learned"
            or payload.get("setting") != run.setting_id
            or payload.get("env_name") != run.env_name
            or payload.get("dataset_kind") != run.dataset_kind
            or payload.get("source_name") != run.source_name
            or payload.get("seed") != run.seed
            or payload.get("wandb_id") != run.wandb_id
            or payload.get("completed_updates") != FORMAL_UPDATES
            or payload.get("final_eval_step") != FORMAL_UPDATES
            or payload.get("task_ids") != list(FORMAL_TASK_IDS)
            or payload.get("episodes_per_task") != 50
            or payload.get("node_budget") != 64
            or payload.get("branch_factor") != 4
            or payload.get("gradient_checkpointing") is not True
            or payload.get("future_set_cache") is not False
            or payload.get("shared_cache") is not True
            or payload.get("wandb_group") != manifest["logging"]["wandb_group"]
            or metrics.get("eval/num_episodes") != 250
        ):
            return False
        for task_id in FORMAL_TASK_IDS:
            success = metrics.get(f"eval/task{task_id}/success_rate")
            episodes = metrics.get(f"eval/task{task_id}/num_episodes")
            if episodes != 50 or not isinstance(success, (int, float)) or not math.isfinite(success):
                return False
            if not 0.0 <= float(success) <= 1.0:
                return False
        progress_path = Path(run_dir) / str(payload.get("final_eval_progress", ""))
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if (
            progress.get("schema_version") != 1
            or progress.get("status") != "complete"
            or progress.get("identity_sha256") != payload.get("identity_sha256")
            or progress.get("task_ids") != list(FORMAL_TASK_IDS)
            or progress.get("episodes_per_task") != 50
            or len(progress.get("completed_results", ())) != 250
            or progress.get("metrics") != metrics
        ):
            return False
        return True
    except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
        return False


def all_worker_ownership(manifest: Mapping[str, Any]) -> list[int]:
    """Test/audit helper: list every owned global run index across three allocations."""
    return [
        run.global_index
        for shard in range(ALLOCATION_SHARDS)
        for rank in range(WORKERS_PER_ALLOCATION)
        if (run := run_for_worker(manifest, shard, rank)) is not None
    ]
