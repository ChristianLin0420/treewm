"""Self-describing result artifacts (spec section 16).

Written after a stale-result bug: killed jobs left an earlier day's ``budget_sweep.json``
on disk and a collector merged it into a table labelled with a different scoring rule.
Every artifact now records what produced it, and collectors compare metadata instead of
trusting filenames.
"""

from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any


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
