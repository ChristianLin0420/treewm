"""Run metadata: git commit, hostname, parameter counts, run directory naming.

Nothing here may raise if git is unavailable (spec section 21).
"""

from __future__ import annotations

import datetime
import platform
import socket
import subprocess
from pathlib import Path

import torch


def git_commit(repo_root: Path | str | None = None) -> str:
    """Short git SHA, with ``-dirty`` suffix if the tree has uncommitted changes."""
    cwd = str(repo_root) if repo_root is not None else None
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=cwd, stderr=subprocess.DEVNULL, timeout=5
        )
        sha = sha.decode().strip()
        dirty = subprocess.call(
            ["git", "diff", "--quiet"], cwd=cwd, stderr=subprocess.DEVNULL, timeout=5
        )
        return f"{sha}-dirty" if dirty != 0 else sha
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return "unavailable"


def hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "unknown"


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    params = model.parameters()
    if trainable_only:
        return sum(p.numel() for p in params if p.requires_grad)
    return sum(p.numel() for p in params)


def build_run_dir(root: str | Path, dataset: str, arm: str, seed: int, timestamp: str | None = None) -> Path:
    """``runs/{dataset}/{arm}/{YYYYMMDD}_seed{N}`` -- deterministic and readable."""
    stamp = timestamp or datetime.datetime.now().strftime("%Y%m%d")
    return Path(root) / dataset / arm / f"{stamp}_seed{seed}"


def env_summary() -> dict[str, str]:
    return {
        "hostname": hostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda or "cpu",
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none",
    }
