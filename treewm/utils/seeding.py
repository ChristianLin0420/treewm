"""Deterministic seeding and RNG state capture/restore."""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int, rank: int = 0, deterministic: bool = False) -> None:
    """Seed python / numpy / torch.

    Each rank gets a distinct stream (``seed + rank``) so that data augmentation and
    dropout differ across ranks, while the *config* seed still identifies the run.
    """
    effective = seed + rank
    os.environ["PYTHONHASHSEED"] = str(effective)
    random.seed(effective)
    np.random.seed(effective)
    torch.manual_seed(effective)
    torch.cuda.manual_seed_all(effective)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_rng_state() -> dict[str, Any]:
    """Capture every RNG stream needed for exact resume."""
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG streams captured by :func:`get_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu() if torch.is_tensor(state["torch"]) else state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state["torch_cuda"])
        except (RuntimeError, ValueError):
            # Different device count on resume -- non-fatal, the run is still seeded.
            pass
