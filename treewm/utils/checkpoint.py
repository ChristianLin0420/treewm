"""Checkpoint save/load with exact-resume support.

Three checkpoints are maintained (spec section 22): ``latest.pt``, ``best_success.pt``
and ``best_validation_loss.pt``. Each carries model/optimizer/scheduler/scaler state,
step, epoch, config and RNG state, so resuming reproduces the same stream of batches and
dropout masks rather than merely restarting from the same weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import signal
from typing import Any

import torch

from treewm.utils.distributed import unwrap_model
from treewm.utils.seeding import get_rng_state, set_rng_state


CHECKPOINT_SCHEMA_VERSION = 2
GRACEFUL_EXIT_CODE = 75


class GracefulStop(RuntimeError):
    """Raised only at a safe training boundary after a deferred signal."""


class StopController:
    """Async-signal-safe request latch.

    Signal handlers only assign Python fields.  Model/device synchronisation and file
    I/O happen later at a normal control-flow boundary.
    """

    def __init__(self) -> None:
        self.reason: str | None = None

    @property
    def requested(self) -> bool:
        return self.reason is not None

    def request(self, reason: str) -> None:
        if self.reason is None:
            self.reason = str(reason)

    def install(self) -> None:
        for sig in (signal.SIGUSR1, signal.SIGTERM):
            signal.signal(sig, lambda signum, _frame: self.request(signal.Signals(signum).name))


def build_checkpoint(
    model,
    optimizer=None,
    scheduler=None,
    scaler=None,
    step: int = 0,
    epoch: int = 0,
    config: Any = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": int(step),
        "epoch": int(epoch),
        "config": config,
        "rng_state": get_rng_state(),
    }
    if extra:
        payload.update(extra)
    return payload


def save_checkpoint(path: str | Path, **kwargs: Any) -> Path:
    """Atomic write: a crash mid-save must not destroy the previous checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as handle:
            torch.save(build_checkpoint(**kwargs), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            # Some network filesystems do not support directory fsync. The atomic
            # replace still protects the previous checkpoint.
            pass
        return path
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def atomic_json_dump(value: dict[str, Any], path: str | Path) -> Path:
    """Durably replace a small JSON sentinel without exposing a partial file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass
        return path
    finally:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass


def load_checkpoint(
    path: str | Path,
    model=None,
    optimizer=None,
    scheduler=None,
    scaler=None,
    map_location: str = "cpu",
    restore_rng: bool = True,
    strict: bool = True,
    rank: int = 0,
    expected_identity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema {payload.get('schema_version')!r}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    if expected_identity is not None and payload.get("run_identity") != expected_identity:
        raise ValueError(f"checkpoint run identity does not match requested run: {path}")
    if model is not None:
        unwrap_model(model).load_state_dict(payload["model"], strict=strict)
    if optimizer is not None and payload.get("optimizer"):
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler"):
        scheduler.load_state_dict(payload["scheduler"])
    if scaler is not None and payload.get("scaler"):
        scaler.load_state_dict(payload["scaler"])
    if restore_rng:
        rank_states = payload.get("rank_states") or []
        selected = next((state for state in rank_states if int(state.get("rank", -1)) == rank), None)
        rng_state = selected.get("rng_state") if selected is not None else payload.get("rng_state")
        if rng_state:
            set_rng_state(rng_state)
    return payload


@dataclass
class CheckpointManager:
    """Tracks the three checkpoint slots. Rank-0 only -- callers must guard."""

    directory: Path
    enabled: bool = True
    best_success: float = field(default=-float("inf"))
    best_val_loss: float = field(default=float("inf"))

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        if self.enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    def save_latest(self, **kwargs: Any) -> Path | None:
        if not self.enabled:
            return None
        return save_checkpoint(self.directory / "latest.pt", **kwargs)

    def maybe_save_success(self, success: float, **kwargs: Any) -> Path | None:
        if not self.enabled or success <= self.best_success:
            return None
        self.best_success = success
        return save_checkpoint(self.directory / "best_success.pt", **kwargs)

    def maybe_save_val_loss(self, val_loss: float, **kwargs: Any) -> Path | None:
        if not self.enabled or val_loss >= self.best_val_loss:
            return None
        self.best_val_loss = val_loss
        return save_checkpoint(self.directory / "best_validation_loss.pt", **kwargs)

    def state_dict(self) -> dict[str, float]:
        return {"best_success": self.best_success, "best_val_loss": self.best_val_loss}

    def load_state_dict(self, state: dict[str, float]) -> None:
        self.best_success = float(state.get("best_success", -float("inf")))
        self.best_val_loss = float(state.get("best_val_loss", float("inf")))
