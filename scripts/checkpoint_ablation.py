#!/usr/bin/env python3
"""Read-only, paired inference ablations for TreeWM checkpoints.

The default screen is deliberately small enough to fan out as a 36-element Slurm
array: four representative seed-0 formal-v2 checkpoints times nine preregistered
inference arms, with one episode on each built-in task ID 1..5.  It changes no model weights and never writes
beside a checkpoint.  Each result is created atomically under a content-addressed study
directory; an existing result is either verified and skipped with ``--resume`` or is a
hard error.

Examples::

    # Show the exact checkpoint/arm work map without loading a model.
    python scripts/checkpoint_ablation.py --dry-run

    # Run one item from the default 4 x 9 screen work map.
    python scripts/checkpoint_ablation.py --work-index 0 --resume

    # Evaluate selected checkpoints serially with a subset of preregistered arms.
    python scripts/checkpoint_ablation.py \
      --settings antmaze-large,cube-double --seeds 0 \
      --arms normalized_l2-d16-e16-learned-hlearned,domain_raw-d16-e16-learned-hlearned

The compact contrasts are fixed before evaluation:

* legacy normalised-L2 versus raw domain scoring at depth 16 / execute 16;
* raw-domain depth 2, 3, and 16 at execute 16;
* execute 4 versus 16 at raw-domain depth 3;
* learned, exact q-novelty, random, and BFS allocation at raw-domain depth 3 /
  execute 4; and
* learned horizon selection versus a fixed 16-step edge at that same reference arm.

Use ``--stage full`` only after the preregistered screen criterion in
``experiments/12-treewm-formal-v2/CHECKPOINT_ABLATION.md`` passes. It expands the same
nine arms to all ten settings. Use ``--grid factorial`` only when the full 2 x 3 x 2 x 4 diagnostic is affordable.
The fixed-16 contrast remains one additional paired arm rather than doubling the whole
factorial.
"""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict, dataclass, replace
import datetime as dt
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Mapping, Sequence
import uuid

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
from omegaconf import OmegaConf
import torch

from scripts.eval import load_run
from treewm.data.ogbench_dataset import load_ogbench
from treewm.evaluation.domains import get_domain
from treewm.evaluation.rollout import evaluate
from treewm.evaluation.tasks import build_tasks
from treewm.models.baselines import tree_config_for
from treewm.planning.goal_planner import GoalPlanner
from treewm.utils import config as cfg_utils
from treewm.utils.provenance import runtime_fingerprint
from treewm.utils.rng import make_generator
from treewm.utils.seeding import seed_everything


SCHEMA_VERSION = 1
DEFAULT_CHECKPOINT_GLOB = (
    "outputs/treewm-50task-1m-v2/*/treewm/*-seed0/checkpoints/latest.pt"
)
DEFAULT_OUTPUT_ROOT = "outputs/treewm-v2-checkpoint-ablation"
SCREEN_SETTINGS = ("antmaze-large", "cube-double", "puzzle-3x3", "scene")
SCORERS = ("learned", "novelty_q", "random", "bfs")
DECODED_METRICS = ("normalized_l2", "domain_raw")
DEPTHS = (2, 3, 16)
EXECUTE_STEPS = (4, 16)
RUN_NAME_RE = re.compile(r"^(?P<prefix>.+)-seed(?P<seed>[0-9]+)$")
SAFE_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_.-]+")


@dataclass(frozen=True)
class AblationArm:
    """One immutable inference-only intervention."""

    decoded_metric: str
    max_depth: int
    execute_steps: int
    scorer: str
    horizon_mode: str = "learned"
    fixed_horizon: int | None = None
    keep_threshold: float | None = None
    require_first_edge_improvement: bool | None = None

    def __post_init__(self) -> None:
        if self.decoded_metric not in DECODED_METRICS:
            raise ValueError(f"unknown decoded metric {self.decoded_metric!r}")
        if self.max_depth not in DEPTHS:
            raise ValueError(f"max depth must be one of {DEPTHS}, got {self.max_depth}")
        if self.execute_steps not in EXECUTE_STEPS:
            raise ValueError(
                f"execute steps must be one of {EXECUTE_STEPS}, got {self.execute_steps}"
            )
        if self.scorer not in SCORERS:
            raise ValueError(f"unknown scorer {self.scorer!r}")
        if self.horizon_mode not in {"learned", "fixed"}:
            raise ValueError(f"unknown horizon mode {self.horizon_mode!r}")
        if self.horizon_mode == "fixed" and self.fixed_horizon is None:
            raise ValueError("fixed horizon mode requires fixed_horizon")
        if self.horizon_mode != "fixed" and self.fixed_horizon is not None:
            raise ValueError("fixed_horizon is only valid in fixed horizon mode")
        if self.keep_threshold is not None and not 0.0 <= self.keep_threshold <= 1.0:
            raise ValueError("keep_threshold must lie in [0, 1]")

    @property
    def arm_id(self) -> str:
        horizon = (
            f"fixed{self.fixed_horizon}"
            if self.horizon_mode == "fixed"
            else self.horizon_mode
        )
        base = (
            f"{self.decoded_metric}-d{self.max_depth}-e{self.execute_steps}-"
            f"{self.scorer}-h{horizon}"
        )
        if self.keep_threshold is not None:
            base += f"-k{int(round(100 * self.keep_threshold)):02d}"
        if self.require_first_edge_improvement is not None:
            base += "-guard" if self.require_first_edge_improvement else "-noguard"
        return base


def _unique_arms(values: Iterable[AblationArm]) -> tuple[AblationArm, ...]:
    seen: set[str] = set()
    out: list[AblationArm] = []
    for arm in values:
        if arm.arm_id not in seen:
            seen.add(arm.arm_id)
            out.append(arm)
    return tuple(out)


def compact_grid(include_fixed16: bool = True) -> tuple[AblationArm, ...]:
    """Nine-arm, one-factor-at-a-time preregistration used by default."""
    reference = AblationArm("domain_raw", 3, 4, "learned")
    arms = [
        # Historical checkpoint inference and its one-variable metric correction.
        AblationArm("normalized_l2", 16, 16, "learned"),
        AblationArm("domain_raw", 16, 16, "learned"),
        # Reliable-depth contrast, holding metric/scorer/execution fixed.
        AblationArm("domain_raw", 2, 16, "learned"),
        AblationArm("domain_raw", 3, 16, "learned"),
        # Replanning contrast at the depth-3 reference.
        reference,
        # Allocation contrast, all at the exact same metric/depth/execution budget.
        AblationArm("domain_raw", 3, 4, "novelty_q"),
        AblationArm("domain_raw", 3, 4, "random"),
        AblationArm("domain_raw", 3, 4, "bfs"),
    ]
    if include_fixed16:
        # Uses the already-trained action and dynamics heads. Only the inference-time
        # horizon selector changes, so checkpoint shapes and weights are untouched.
        arms.append(AblationArm("domain_raw", 3, 4, "learned", "fixed", 16))
    return _unique_arms(arms)


def factorial_grid(include_fixed16: bool = True) -> tuple[AblationArm, ...]:
    arms = [
        AblationArm(metric, depth, execute, scorer)
        for metric in DECODED_METRICS
        for depth in DEPTHS
        for execute in EXECUTE_STEPS
        for scorer in SCORERS
    ]
    if include_fixed16:
        arms.append(AblationArm("domain_raw", 3, 4, "learned", "fixed", 16))
    return _unique_arms(arms)


def grounded_repair_grid() -> tuple[AblationArm, ...]:
    """Frozen-checkpoint screen for support admission and executable-edge gating.

    The current arm is retained verbatim.  Every intervention changes only inference;
    no checkpoint tensor is modified.  A 0.42 KEEP threshold is the sealed midpoint of
    the failed puzzle settings' observed positive priors (0.415 and 0.442), while the
    learned/BFS and guard contrasts isolate allocation from action execution.
    """
    return _unique_arms(
        (
            AblationArm("domain_raw", 3, 4, "learned"),
            AblationArm("domain_raw", 3, 4, "learned", keep_threshold=0.42,
                        require_first_edge_improvement=True),
            AblationArm("domain_raw", 3, 4, "learned", keep_threshold=0.50,
                        require_first_edge_improvement=False),
            AblationArm("domain_raw", 3, 4, "learned", keep_threshold=0.42,
                        require_first_edge_improvement=False),
            AblationArm("domain_raw", 3, 4, "bfs", keep_threshold=0.42,
                        require_first_edge_improvement=True),
            AblationArm("domain_raw", 3, 4, "bfs", keep_threshold=0.42,
                        require_first_edge_improvement=False),
            AblationArm("domain_raw", 3, 4, "novelty_q", keep_threshold=0.42,
                        require_first_edge_improvement=False),
            AblationArm("domain_raw", 3, 4, "bfs", "fixed", 16,
                        keep_threshold=0.42, require_first_edge_improvement=False),
        )
    )


def preregistered_contrasts(arms: Sequence[AblationArm]) -> dict[str, list[str]]:
    """Named paired contrasts; no result-dependent arm selection is permitted."""
    available = {arm.arm_id for arm in arms}

    def ids(*values: AblationArm) -> list[str]:
        return [value.arm_id for value in values if value.arm_id in available]

    return {
        "decoded_metric_at_d16_e16_learned": ids(
            AblationArm("normalized_l2", 16, 16, "learned"),
            AblationArm("domain_raw", 16, 16, "learned"),
        ),
        "max_depth_at_domain_raw_e16_learned": ids(
            *(AblationArm("domain_raw", depth, 16, "learned") for depth in DEPTHS)
        ),
        "execute_steps_at_domain_raw_d3_learned": ids(
            *(AblationArm("domain_raw", 3, execute, "learned") for execute in EXECUTE_STEPS)
        ),
        "frontier_scorer_at_domain_raw_d3_e4": ids(
            *(AblationArm("domain_raw", 3, 4, scorer) for scorer in SCORERS)
        ),
        "horizon_selector_at_domain_raw_d3_e4_learned": ids(
            AblationArm("domain_raw", 3, 4, "learned"),
            AblationArm("domain_raw", 3, 4, "learned", "fixed", 16),
        ),
    }


def parse_csv(value: str | None, cast=str) -> tuple[Any, ...]:
    if value is None or not value.strip():
        return ()
    return tuple(cast(item.strip()) for item in value.split(",") if item.strip())


def _run_descriptor(path: Path) -> tuple[str, int | None]:
    run_name = path.parents[1].name if len(path.parents) >= 2 else path.stem
    match = RUN_NAME_RE.match(run_name)
    if match is None:
        return run_name, None
    prefix = match.group("prefix")
    # Formal names are treewm-v2-<setting>-seedN. Keep arbitrary names usable too.
    setting = prefix.removeprefix("treewm-v2-").removeprefix("grounded-formal-")
    return setting, int(match.group("seed"))


def discover_checkpoints(
    patterns: Sequence[str],
    *,
    repo_root: Path = REPOSITORY_ROOT,
    settings: Sequence[str] = (),
    seeds: Sequence[int] = (),
) -> list[Path]:
    """Resolve globs deterministically and filter by formal run setting/seed."""
    found: set[Path] = set()
    for pattern in patterns:
        expanded = Path(pattern).expanduser()
        query = str(expanded if expanded.is_absolute() else repo_root / expanded)
        for value in glob.glob(query, recursive=True):
            path = Path(value).resolve()
            if path.is_file():
                found.add(path)
    wanted_settings = set(settings)
    wanted_seeds = {int(seed) for seed in seeds}
    selected = []
    for path in found:
        setting, seed = _run_descriptor(path)
        if wanted_settings and setting not in wanted_settings:
            continue
        if wanted_seeds and seed not in wanted_seeds:
            continue
        selected.append(path)
    return sorted(selected, key=lambda path: (*_run_descriptor(path), str(path)))


def select_work(
    checkpoints: Sequence[Path],
    arms: Sequence[AblationArm],
    *,
    checkpoint_index: int | None = None,
    arm_index: int | None = None,
    work_index: int | None = None,
) -> list[tuple[int, int, Path, AblationArm]]:
    """Return ``(checkpoint index, arm index, path, arm)`` work items."""
    if work_index is not None and (checkpoint_index is not None or arm_index is not None):
        raise ValueError("--work-index cannot be combined with checkpoint/arm indices")
    if not checkpoints:
        return []
    if not arms:
        return []
    if work_index is not None:
        total = len(checkpoints) * len(arms)
        if not 0 <= work_index < total:
            raise IndexError(f"work index {work_index} outside [0, {total})")
        ci, ai = divmod(work_index, len(arms))
        return [(ci, ai, checkpoints[ci], arms[ai])]
    checkpoint_indices = (
        range(len(checkpoints)) if checkpoint_index is None else (checkpoint_index,)
    )
    arm_indices = range(len(arms)) if arm_index is None else (arm_index,)
    out = []
    for ci in checkpoint_indices:
        if not 0 <= ci < len(checkpoints):
            raise IndexError(f"checkpoint index {ci} outside [0, {len(checkpoints)})")
        for ai in arm_indices:
            if not 0 <= ai < len(arms):
                raise IndexError(f"arm index {ai} outside [0, {len(arms)})")
            out.append((ci, ai, checkpoints[ci], arms[ai]))
    return out


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if torch.is_tensor(value):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(
        _json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False,
    )


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def evaluation_code_fingerprint(repo_root: Path) -> dict[str, Any]:
    """Hash every live source/config file that can affect this evaluation."""
    root = repo_root.resolve()
    candidates = {
        Path(__file__).resolve(),
        root / "scripts" / "eval.py",
        *(root / "treewm").rglob("*.py"),
        *(root / "configs").rglob("*.yaml"),
    }
    files: dict[str, str] = {}
    manifest = hashlib.sha256()
    for path in sorted(candidate.resolve() for candidate in candidates):
        if not path.is_file() or not path.is_relative_to(root):
            raise RuntimeError(f"evaluation source missing or outside repository: {path}")
        relative = path.relative_to(root).as_posix()
        digest = sha256_file(path)
        files[relative] = digest
        manifest.update(relative.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
    return {"manifest_sha256": manifest.hexdigest(), "files": files}


def git_provenance(repo_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        completed = subprocess.run(
            ["git", *args], cwd=repo_root, check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        )
        return completed.stdout.strip() if completed.returncode == 0 else "unavailable"

    status = run("status", "--porcelain", "--untracked-files=all")
    return {
        "commit": run("rev-parse", "HEAD"),
        "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status and status != "unavailable"),
        "status": status.splitlines() if status and status != "unavailable" else [],
    }


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError:
        pass


def atomic_json_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    """Atomically create ``path`` and refuse to replace any existing artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(_json_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Hard-link publication has no replacement semantics and is atomic on the
            # same filesystem, unlike os.replace which could hide a concurrent result.
            os.link(temporary, path)
        except FileExistsError:
            raise
        _fsync_directory(path.parent)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def write_or_validate(
    path: Path,
    payload: Mapping[str, Any],
    *,
    identity_key: str,
    resume: bool,
) -> bool:
    """Create an artifact, or validate an identical identity when resuming.

    Returns ``True`` when a new artifact was created and ``False`` when a compatible
    existing artifact was retained.
    """
    try:
        atomic_json_exclusive(path, payload)
        return True
    except FileExistsError:
        if not resume:
            raise FileExistsError(
                f"refusing to overwrite existing result {path}; use --resume only to "
                "verify and skip an identical artifact"
            )
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"existing artifact is unreadable: {path}") from exc
        if existing.get(identity_key) != payload.get(identity_key):
            raise RuntimeError(
                f"existing artifact identity differs at {path}: "
                f"{existing.get(identity_key)!r} != {payload.get(identity_key)!r}"
            )
        return False


def _stat_identity(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def checkpoint_provenance(
    path: Path,
    checkpoint_sha256: str,
    payload: Mapping[str, Any],
    run_cfg,
) -> dict[str, Any]:
    resolved_config = OmegaConf.to_container(run_cfg, resolve=True)
    return {
        "path": str(path.resolve()),
        "sha256": checkpoint_sha256,
        "size_bytes": path.stat().st_size,
        "schema_version": payload.get("schema_version"),
        "step": int(payload.get("step", -1)),
        "completed_updates": int(payload.get("completed_updates", payload.get("step", -1))),
        "reason": payload.get("reason"),
        "phase": payload.get("phase"),
        "identity_sha256": payload.get("identity_sha256"),
        "run_identity": _json_safe(payload.get("run_identity")),
        "resolved_config_sha256": stable_hash(resolved_config),
        "resolved_config": _json_safe(resolved_config),
    }


def arm_support(model, arm: AblationArm) -> tuple[bool, str | None, int | None]:
    if arm.scorer == "learned" and getattr(model, "gain_head", None) is None:
        return False, "checkpoint model has no learned gain head", None
    if arm.scorer == "novelty_q" and not callable(getattr(model, "q_cdist", None)):
        return False, "checkpoint model has no exact q-distance operator", None
    fixed_index = None
    if arm.horizon_mode == "fixed":
        horizons = [int(value) for value in model.cfg.horizons]
        if int(arm.fixed_horizon) not in horizons:
            return (
                False,
                f"fixed horizon {arm.fixed_horizon} absent from checkpoint horizons {horizons}",
                None,
            )
        fixed_index = horizons.index(int(arm.fixed_horizon))
    return True, None, fixed_index


@contextlib.contextmanager
def inference_horizon(model, arm: AblationArm, fixed_index: int | None):
    previous_mode = model.cfg.horizon_mode
    previous_index = model.cfg.fixed_horizon_index
    model.cfg.horizon_mode = arm.horizon_mode
    if fixed_index is not None:
        model.cfg.fixed_horizon_index = int(fixed_index)
    try:
        yield
    finally:
        model.cfg.horizon_mode = previous_mode
        model.cfg.fixed_horizon_index = previous_index


def _select_standard_tasks(env, task_ids: Sequence[int]) -> list[dict[str, Any]]:
    standard = build_tasks(env, split="standard")
    by_id = {int(task.get("task_id", index + 1)): task for index, task in enumerate(standard)}
    missing = [task_id for task_id in task_ids if task_id not in by_id]
    if missing:
        raise ValueError(
            f"requested task IDs {missing} are unavailable; environment provides {sorted(by_id)}"
        )
    return [by_id[int(task_id)] for task_id in task_ids]


def evaluate_arm(
    *,
    model,
    normalizer,
    run_cfg,
    env,
    domain,
    tasks: Sequence[dict[str, Any]],
    arm: AblationArm,
    episodes_per_task: int,
    max_env_steps: int,
    eval_seed: int,
    device: torch.device,
) -> dict[str, Any]:
    supported, reason, fixed_index = arm_support(model, arm)
    if not supported:
        return {"status": "unsupported", "reason": reason, "metrics": {}, "episodes": []}

    tree_cfg = cfg_utils.tree_config(run_cfg)
    tree_cfg = replace(
        tree_cfg,
        max_depth=int(arm.max_depth),
        scorer=str(arm.scorer),
        scorer_override=str(arm.scorer),
        **(
            {"keep_threshold": float(arm.keep_threshold)}
            if arm.keep_threshold is not None
            else {}
        ),
    )
    tree_cfg = tree_config_for(str(run_cfg.arm), tree_cfg, model)
    planner_cfg = replace(
        cfg_utils.planner_config(run_cfg),
        score_space="decoded",
        decoded_metric=str(arm.decoded_metric),
        execute_mode="clipped",
        execute_steps=int(arm.execute_steps),
        max_env_steps=int(max_env_steps),
        **(
            {"require_first_edge_improvement": bool(arm.require_first_edge_improvement)}
            if arm.require_first_edge_improvement is not None
            else {}
        ),
    )
    # Every arm starts from identical global and owned RNG streams. In particular, the
    # random frontier control is paired rather than inheriting state from a prior arm.
    seed_everything(int(eval_seed))
    generator = make_generator(int(eval_seed), "planner", device)
    planner = GoalPlanner(
        model, normalizer, tree_cfg, planner_cfg, device=device,
        generator=generator, domain=domain,
    )
    episodes: list[dict[str, Any]] = []
    with inference_horizon(model, arm, fixed_index):
        metrics = evaluate(
            env,
            planner,
            list(tasks),
            episodes_per_task=int(episodes_per_task),
            max_steps=int(max_env_steps),
            seed=int(eval_seed),
            domain=domain,
            episode_callback=lambda result: episodes.append(_json_safe(result)),
        )
    return {
        "status": "completed",
        "reason": None,
        "effective": {
            "score_space": "decoded",
            "decoded_metric": arm.decoded_metric,
            "tree_max_depth": tree_cfg.max_depth,
            "tree_node_budget": tree_cfg.node_budget,
            "tree_scorer": tree_cfg.scorer,
            "tree_keep_threshold": tree_cfg.keep_threshold,
            "planner_execute_mode": planner_cfg.execute_mode,
            "planner_execute_steps": planner_cfg.execute_steps,
            "require_first_edge_improvement": planner_cfg.require_first_edge_improvement,
            "horizon_mode": arm.horizon_mode,
            "fixed_horizon": arm.fixed_horizon,
            "fixed_horizon_index": fixed_index,
            "available_horizons": [int(value) for value in model.cfg.horizons],
        },
        "metrics": _json_safe(metrics),
        "episodes": episodes,
    }


def _safe_component(value: str) -> str:
    return SAFE_COMPONENT_RE.sub("_", value).strip("._") or "unnamed"


def result_path(
    study_dir: Path,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_step: int,
    arm: AblationArm,
    result_id: str,
) -> Path:
    run_name = checkpoint_path.parents[1].name
    checkpoint_key = _safe_component(
        f"{run_name}-{checkpoint_path.stem}-step{checkpoint_step}-{checkpoint_sha256[:12]}"
    )
    return study_dir / "results" / checkpoint_key / f"{arm.arm_id}-{result_id[:12]}.json"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def maybe_write_checkpoint_bundle(
    *,
    study_dir: Path,
    study_id: str,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    checkpoint_step: int,
    arms: Sequence[AblationArm],
    resume: bool,
) -> Path | None:
    records: list[dict[str, Any]] = []
    paths = []
    for arm in arms:
        result_id = stable_hash(
            {"study_id": study_id, "checkpoint_sha256": checkpoint_sha256, "arm": asdict(arm)}
        )
        path = result_path(
            study_dir, checkpoint_path, checkpoint_sha256, checkpoint_step, arm, result_id
        )
        if not path.is_file():
            return None
        paths.append(path)
        records.append(_load_json(path))
    bundle_id = stable_hash(
        {
            "study_id": study_id,
            "checkpoint_sha256": checkpoint_sha256,
            "result_ids": [record["result_id"] for record in records],
        }
    )
    bundle = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "checkpoint_ablation_bundle",
        "bundle_id": bundle_id,
        "study_id": study_id,
        "checkpoint": records[0]["checkpoint"],
        "arm_results": [
            {
                "arm": record["arm"],
                "status": record["status"],
                "reason": record["reason"],
                "effective": record.get("effective"),
                "metrics": record["metrics"],
                "episodes": record["episodes"],
                "result_id": record["result_id"],
                "result_path": str(path),
            }
            for path, record in zip(paths, records, strict=True)
        ],
    }
    path = paths[0].parent / f"bundle-{bundle_id[:12]}.json"
    write_or_validate(path, bundle, identity_key="bundle_id", resume=resume)
    return path


def maybe_write_study_summary(
    *, study_dir: Path, study_id: str, expected_checkpoints: int, resume: bool
) -> Path | None:
    bundles = sorted((study_dir / "results").glob("*/bundle-*.json"))
    valid = []
    seen_checkpoints: set[str] = set()
    for path in bundles:
        value = _load_json(path)
        if value.get("study_id") != study_id:
            continue
        checkpoint_sha = value.get("checkpoint", {}).get("sha256")
        if checkpoint_sha in seen_checkpoints:
            raise RuntimeError(f"multiple bundles for checkpoint {checkpoint_sha}")
        seen_checkpoints.add(checkpoint_sha)
        valid.append((path, value))
    if len(valid) != expected_checkpoints:
        return None
    summary_id = stable_hash(
        {"study_id": study_id, "bundle_ids": [value["bundle_id"] for _, value in valid]}
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "checkpoint_ablation_summary",
        "summary_id": summary_id,
        "study_id": study_id,
        "num_checkpoints": len(valid),
        "bundles": [
            {
                "path": str(path),
                "bundle_id": value["bundle_id"],
                "checkpoint": value["checkpoint"],
                "arm_results": value["arm_results"],
            }
            for path, value in valid
        ],
    }
    path = study_dir / f"summary-{summary_id[:12]}.json"
    write_or_validate(path, payload, identity_key="summary_id", resume=resume)
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-glob", action="append", default=None,
        help=f"repeatable checkpoint glob (default: {DEFAULT_CHECKPOINT_GLOB})",
    )
    parser.add_argument(
        "--stage", choices=("screen", "full"), default="screen",
        help="screen defaults to four representative settings; full uses all ten",
    )
    parser.add_argument(
        "--settings",
        help="comma-separated setting IDs, or 'all'; overrides the stage's setting set",
    )
    parser.add_argument("--seeds", default="0", help="comma-separated training seeds")
    parser.add_argument(
        "--grid", choices=("compact", "factorial", "grounded-repair"),
        default="compact",
    )
    parser.add_argument(
        "--fixed16", action=argparse.BooleanOptionalAction, default=True,
        help="include the paired fixed-16 horizon arm (default: enabled)",
    )
    parser.add_argument("--arms", help="comma-separated preregistered arm IDs")
    parser.add_argument("--task-ids", default="1,2,3,4,5")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument(
        "--max-env-steps", type=int,
        help="override the checkpoint's domain-specific episode limit",
    )
    parser.add_argument(
        "--eval-seed", type=int,
        help="fixed evaluation seed; default pairs on each checkpoint's training seed",
    )
    parser.add_argument("--checkpoint-index", type=int)
    parser.add_argument("--arm-index", type=int)
    parser.add_argument(
        "--work-index", type=int,
        help="flattened checkpoint-major index used by the default Slurm array",
    )
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="cuda")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--resume", action="store_true",
        help="verify and skip exact completed artifacts; never overwrites",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return torch.device(requested)


def _validate_args(args: argparse.Namespace) -> None:
    if args.episodes_per_task < 1:
        raise ValueError("episodes per task must be positive")
    if args.max_env_steps is not None and args.max_env_steps < 1:
        raise ValueError("max environment steps must be positive")
    if len({value for value in (args.work_index,) if value is not None}) and (
        args.checkpoint_index is not None or args.arm_index is not None
    ):
        raise ValueError("--work-index cannot be combined with checkpoint/arm indices")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_args(args)
    if args.settings is None:
        settings = SCREEN_SETTINGS if args.stage == "screen" else ()
    elif args.settings.strip().lower() == "all":
        settings = ()
    else:
        settings = parse_csv(args.settings)
    seeds = parse_csv(args.seeds, int)
    task_ids = parse_csv(args.task_ids, int)
    if not task_ids or len(set(task_ids)) != len(task_ids) or any(value < 1 for value in task_ids):
        raise ValueError("--task-ids must contain unique positive IDs")

    patterns = tuple(args.checkpoint_glob or (DEFAULT_CHECKPOINT_GLOB,))
    checkpoints = discover_checkpoints(patterns, settings=settings, seeds=seeds)
    if not checkpoints:
        raise FileNotFoundError(
            f"no checkpoints matched patterns={patterns}, settings={settings}, seeds={seeds}"
        )
    if args.grid == "compact":
        arms = compact_grid(args.fixed16)
    elif args.grid == "factorial":
        arms = factorial_grid(args.fixed16)
    else:
        arms = grounded_repair_grid()
    if args.arms:
        requested = parse_csv(args.arms)
        by_id = {arm.arm_id: arm for arm in arms}
        unknown = [arm_id for arm_id in requested if arm_id not in by_id]
        if unknown:
            raise ValueError(f"unknown/non-preregistered arm IDs {unknown}; choices={sorted(by_id)}")
        arms = tuple(by_id[arm_id] for arm_id in requested)
    work = select_work(
        checkpoints,
        arms,
        checkpoint_index=args.checkpoint_index,
        arm_index=args.arm_index,
        work_index=args.work_index,
    )

    code = evaluation_code_fingerprint(REPOSITORY_ROOT)
    study_protocol = {
        "schema_version": SCHEMA_VERSION,
        "study": "treewm_v2_paused_checkpoint_inference_ablation",
        "stage": args.stage,
        "grid": args.grid,
        "arms": [asdict(arm) | {"arm_id": arm.arm_id} for arm in arms],
        "preregistered_contrasts": preregistered_contrasts(arms),
        "primary_metric": "eval/success_rate",
        "secondary_metrics": [
            "eval/distance_reduction_frac",
            "eval/goal_distance_best",
            "eval/progress/subgoal_gain",
            "eval/progress/best_subgoal_gain",
            "eval/action_magnitude",
            "eval/selected_leaf_depth",
            "eval/no_action_plan_rate",
            "eval/no_action_episode_fraction",
            "eval/guard/rejection_rate",
            "eval/guard/candidate_acceptance_rate",
            "eval/guard/best_predicted_executable_improvement",
            "eval/guard/selected_predicted_executable_improvement",
        ],
        "pairing": "same checkpoint, task ID, episode index, environment seed, and node budget",
        "adaptive_arm_selection": False,
        "task_split": "standard",
        "task_ids": list(task_ids),
        "episodes_per_task": int(args.episodes_per_task),
        "max_env_steps": args.max_env_steps or "checkpoint_domain_default",
        "eval_seed_rule": args.eval_seed if args.eval_seed is not None else "training_seed",
        "checkpoint_patterns": list(patterns),
        "settings": list(settings),
        "seeds": list(seeds),
        "checkpoint_paths": [str(path) for path in checkpoints],
        "code_manifest_sha256": code["manifest_sha256"],
    }
    study_id = stable_hash(study_protocol)
    output_root = Path(args.output_root).expanduser()
    if not output_root.is_absolute():
        output_root = REPOSITORY_ROOT / output_root
    output_root = output_root.resolve()
    if any(output_root == checkpoint.parent or checkpoint.parent in output_root.parents
           for checkpoint in checkpoints):
        raise ValueError("output root must not be inside a checkpoint directory")
    study_dir = output_root / study_id[:16]
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "checkpoint_ablation_manifest",
        "study_id": study_id,
        "protocol": study_protocol,
        "code": code,
        "git": git_provenance(REPOSITORY_ROOT),
        "work_map": [
            {
                "work_index": ci * len(arms) + ai,
                "checkpoint_index": ci,
                "arm_index": ai,
                "checkpoint": str(checkpoint),
                "setting": _run_descriptor(checkpoint)[0],
                "seed": _run_descriptor(checkpoint)[1],
                "arm_id": arm.arm_id,
            }
            for ci, checkpoint in enumerate(checkpoints)
            for ai, arm in enumerate(arms)
        ],
    }

    print(f"[ablation] study_id={study_id}")
    print(
        f"[ablation] checkpoints={len(checkpoints)} arms={len(arms)} "
        f"total_work={len(checkpoints) * len(arms)} selected_work={len(work)}"
    )
    for ci, ai, checkpoint, arm in work:
        print(
            f"  work={ci * len(arms) + ai:03d} checkpoint[{ci}]={checkpoint} "
            f"arm[{ai}]={arm.arm_id}"
        )
    if args.dry_run:
        print(json.dumps(_json_safe(manifest), sort_keys=True, indent=2, allow_nan=False))
        return 0

    write_or_validate(
        study_dir / "manifest.json", manifest, identity_key="study_id", resume=True
    )
    device = _resolve_device(args.device)
    runtime = runtime_fingerprint()
    loaded_checkpoint: Path | None = None
    model = normalizer = run_cfg = payload = env = domain = tasks = None
    checkpoint_sha = ""
    checkpoint_before: tuple[int, int, int, int] | None = None
    checkpoint_prov: dict[str, Any] | None = None

    for ci, ai, checkpoint, arm in work:
        if loaded_checkpoint != checkpoint:
            if env is not None and hasattr(env, "close"):
                env.close()
            del model, normalizer, run_cfg, payload, env, domain, tasks
            if device.type == "cuda":
                torch.cuda.empty_cache()
            checkpoint_before = _stat_identity(checkpoint)
            print(f"[ablation] hashing checkpoint {checkpoint}", flush=True)
            checkpoint_sha = sha256_file(checkpoint)
            if _stat_identity(checkpoint) != checkpoint_before:
                raise RuntimeError(f"checkpoint changed while it was being hashed: {checkpoint}")
            print(f"[ablation] loading {checkpoint}", flush=True)
            model, normalizer, run_cfg, payload = load_run(str(checkpoint), device)
            checkpoint_prov = checkpoint_provenance(
                checkpoint, checkpoint_sha, payload, run_cfg
            )
            # The optimizer/scheduler/RNG tensors are irrelevant to read-only inference
            # and can be large. Keep only the provenance already copied above.
            for key in (
                "optimizer", "scheduler", "scaler", "rng_state", "rank_states",
                "model", "normalizer", "latent_index",
            ):
                payload.pop(key, None)
            env = load_ogbench(
                str(run_cfg.env.name), dataset_dir=str(run_cfg.env.dataset_dir), env_only=True
            )
            domain = get_domain(str(run_cfg.env.name))
            tasks = _select_standard_tasks(env, task_ids)
            loaded_checkpoint = checkpoint

        assert checkpoint_prov is not None and checkpoint_before is not None
        eval_seed = int(args.eval_seed if args.eval_seed is not None else run_cfg.seed)
        max_env_steps = int(
            args.max_env_steps
            if args.max_env_steps is not None
            else run_cfg.planner.max_env_steps
        )
        result_id = stable_hash(
            {"study_id": study_id, "checkpoint_sha256": checkpoint_sha, "arm": asdict(arm)}
        )
        path = result_path(
            study_dir,
            checkpoint,
            checkpoint_sha,
            int(checkpoint_prov["completed_updates"]),
            arm,
            result_id,
        )
        if path.exists() and args.resume:
            existing = _load_json(path)
            if existing.get("result_id") != result_id:
                raise RuntimeError(f"result identity mismatch at {path}")
            print(f"[ablation] exact result already complete; skipping {path}")
        else:
            print(f"[ablation] evaluating {arm.arm_id}", flush=True)
            outcome = evaluate_arm(
                model=model,
                normalizer=normalizer,
                run_cfg=run_cfg,
                env=env,
                domain=domain,
                tasks=tasks,
                arm=arm,
                episodes_per_task=int(args.episodes_per_task),
                max_env_steps=max_env_steps,
                eval_seed=eval_seed,
                device=device,
            )
            if _stat_identity(checkpoint) != checkpoint_before:
                raise RuntimeError(f"checkpoint changed during evaluation: {checkpoint}")
            result = {
                "schema_version": SCHEMA_VERSION,
                "artifact_type": "checkpoint_ablation_arm_result",
                "result_id": result_id,
                "study_id": study_id,
                "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "work_index": ci * len(arms) + ai,
                "checkpoint_index": ci,
                "arm_index": ai,
                "checkpoint": checkpoint_prov,
                "code": code,
                "git": manifest["git"],
                "runtime": runtime,
                "evaluation": {
                    "task_split": "standard",
                    "task_ids": list(task_ids),
                    "episodes_per_task": int(args.episodes_per_task),
                    "max_env_steps": max_env_steps,
                    "eval_seed": eval_seed,
                },
                "arm": asdict(arm) | {"arm_id": arm.arm_id},
                **outcome,
            }
            created = write_or_validate(
                path, result, identity_key="result_id", resume=args.resume
            )
            print(f"[ablation] {'wrote' if created else 'retained'} {path}", flush=True)

        bundle = maybe_write_checkpoint_bundle(
            study_dir=study_dir,
            study_id=study_id,
            checkpoint_path=checkpoint,
            checkpoint_sha256=checkpoint_sha,
            checkpoint_step=int(checkpoint_prov["completed_updates"]),
            arms=arms,
            resume=True,
        )
        if bundle is not None:
            print(f"[ablation] checkpoint bundle complete: {bundle}", flush=True)

    if env is not None and hasattr(env, "close"):
        env.close()
    summary = maybe_write_study_summary(
        study_dir=study_dir,
        study_id=study_id,
        expected_checkpoints=len(checkpoints),
        resume=True,
    )
    if summary is not None:
        print(f"[ablation] study summary complete: {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
