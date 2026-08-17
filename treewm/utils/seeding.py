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


def _as_byte_cpu(t: Any) -> Any:
    """RNG states must be CPU uint8 tensors.

    ``load_checkpoint`` passes ``map_location=<device>``, which moves *every* tensor in
    the payload to CUDA -- including these. ``set_rng_state`` then raises
    ``TypeError: RNG state must be a torch.ByteTensor``, which the old handler did not
    catch, so every resumed job died at startup.
    """
    if torch.is_tensor(t):
        return t.detach().to("cpu", torch.uint8)
    return t


def set_rng_state(state: dict[str, Any]) -> None:
    """Restore RNG streams captured by :func:`get_rng_state`."""
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(_as_byte_cpu(state["torch"]))
    if "torch_cuda" in state and torch.cuda.is_available():
        try:
            saved = [_as_byte_cpu(s) for s in state["torch_cuda"]]
            # The fleet pins each job to one GPU via CUDA_VISIBLE_DEVICES, so a
            # checkpoint written under a different visible-device count would otherwise
            # mismatch here.
            n = torch.cuda.device_count()
            if len(saved) >= n:
                for i in range(n):
                    torch.cuda.set_rng_state(saved[i], i)
            else:
                torch.cuda.set_rng_state_all(saved)
        except (RuntimeError, ValueError, TypeError) as exc:
            # Non-fatal: the run is still seeded, only bit-exact resume is lost.
            print(f"[treewm] CUDA RNG state not restored ({exc}); continuing seeded")
