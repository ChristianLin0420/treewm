"""Self-describing result artifacts (spec section 16).

Written after a stale-result bug: killed jobs left an earlier day's ``budget_sweep.json``
on disk and a collector merged it into a table labelled with a different scoring rule.
Every artifact now records what produced it, and collectors compare metadata instead of
trusting filenames.
"""

from __future__ import annotations

import datetime
import hashlib
from importlib import metadata
import json
import platform
from pathlib import Path
from typing import Any


RUNTIME_DISTRIBUTIONS = (
    "torch",
    "numpy",
    "scipy",
    "ogbench",
    "gymnasium",
    "mujoco",
    "hydra-core",
    "omegaconf",
    "tensorboard",
    "wandb",
    "matplotlib",
    "tqdm",
)


def trainer_code_fingerprint(repo_root: str | Path) -> dict[str, Any]:
    """Hash every live source/config file that can affect formal training.

    Hydra's resolved config hash protects checkpoints after the process has started,
    but a campaign dispatcher can only validate an existing completion sentinel against
    the files that are live *before* launching Hydra. Include the complete config tree
    so changing an inherited model/loss/default also invalidates the campaign's injected
    ``TREEWM_CODE_SHA256``.
    """
    root = Path(repo_root).resolve()
    paths = [
        root / "scripts" / "train.py",
        *sorted((root / "treewm").rglob("*.py")),
        *sorted((root / "configs").rglob("*.yaml")),
    ]
    files: dict[str, str] = {}
    manifest = hashlib.sha256()
    for path in paths:
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        files[relative] = digest
        manifest.update(relative.encode("utf-8") + b"\0" + digest.encode("ascii") + b"\n")
    return {"manifest_sha256": manifest.hexdigest(), "files": files}


def runtime_fingerprint() -> dict[str, Any]:
    packages = {}
    for distribution in RUNTIME_DISTRIBUTIONS:
        try:
            packages[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            packages[distribution] = "missing"
    # Checkpoint identity must survive a Slurm requeue onto another healthy node.
    # ``platform.platform()`` includes the host kernel/build string, which is useful
    # provenance but is not part of the Python execution environment.  Hash only the
    # interpreter and imported distribution versions; retain host facts as descriptive
    # metadata so heterogeneous allocations remain auditable without making exact
    # resume spuriously node-specific.
    software = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "packages": packages,
    }
    encoded = json.dumps(software, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "software": software,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }


def checkpoint_hash(path: str | Path, chunk_mb: int = 8) -> str:
    """Hash of the first ``chunk_mb`` of the checkpoint -- enough to catch a swap."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            h.update(fh.read(chunk_mb * 1024 * 1024))
        return h.hexdigest()[:16]
    except OSError:
        return "unavailable"


def provenance(
    checkpoint: str | Path | None = None,
    cfg: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Full description of the run that produced an artifact."""
    from treewm.utils.meta import git_commit, hostname

    repo = Path(__file__).resolve().parents[2]
    out: dict[str, Any] = {
        "git_commit": git_commit(repo),
        "hostname": hostname(),
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    if checkpoint is not None:
        out["checkpoint"] = str(checkpoint)
        out["checkpoint_sha256_head"] = checkpoint_hash(checkpoint)
    if cfg is not None:
        def g(path: str, default=None):
            node = cfg
            for part in path.split("."):
                node = getattr(node, part, None) if not isinstance(node, dict) else node.get(part)
                if node is None:
                    return default
            return node

        out.update({
            "arm": str(g("arm")),
            "seed": int(g("seed", -1)),
            "env": str(g("env.name")),
            "dataset": str(g("env.short_name")),
            "train_steps": int(g("train.steps", -1)),
            "score_space": str(g("planner.score_space")),
            "decoded_metric": str(g("planner.decoded_metric", "normalized_l2")),
            "planner_execute_steps": int(g("planner.execute_steps", -1)),
            "node_budget": int(g("tree.node_budget", -1)),
            "branch_factor": int(g("model.branch_factor", -1)),
            "horizons": list(g("future_sets.horizons", [])),
            "h_max": int(g("future_sets.h_max", -1)),
            "scorer": str(g("tree.scorer")),
        })
    if extra:
        out.update(extra)
    return out


def write_artifact(path: str | Path, payload: dict[str, Any], prov: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"provenance": prov, **payload}, indent=2, default=str))
    return path


def compatible(artifacts: list[dict[str, Any]], keys: tuple[str, ...]) -> tuple[bool, str]:
    """Do these artifacts agree on the fields that must match to be merged?"""
    if not artifacts:
        return False, "no artifacts"
    for key in keys:
        values = {json.dumps(a.get("provenance", {}).get(key), default=str) for a in artifacts}
        if len(values) > 1:
            return False, f"mixed {key}: {sorted(values)}"
    return True, "compatible"


def load_checked(paths: list[str | Path], keys: tuple[str, ...] = ("score_space",)):
    """Load artifacts, returning ``(loaded, missing, mismatched)``.

    Missing files are reported as missing -- never silently replaced by whatever else is
    on disk.
    """
    loaded, missing, mismatched = [], [], []
    for p in paths:
        p = Path(p)
        if not p.exists():
            missing.append(str(p))
            continue
        try:
            loaded.append(json.loads(p.read_text()))
        except (OSError, json.JSONDecodeError) as exc:
            mismatched.append(f"{p}: {exc}")
    ok, reason = compatible(loaded, keys)
    if not ok:
        mismatched.append(reason)
    return loaded, missing, mismatched
