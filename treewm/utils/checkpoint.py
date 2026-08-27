"""Checkpoint save/load with exact-resume support.

Three checkpoints are maintained (spec section 22): ``latest.pt``, ``best_success.pt``
and ``best_validation_loss.pt``. Each carries model/optimizer/scheduler/scaler state,
step, epoch, config and RNG state, so resuming reproduces the same stream of batches and
dropout masks rather than merely restarting from the same weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
import os
from pathlib import Path
import signal
from typing import Any, Mapping

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
    return save_checkpoint_payload(path, build_checkpoint(**kwargs))


def save_checkpoint_payload(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Atomically persist an already-built payload to one checkpoint slot.

    Building once lets ``latest`` and a newly selected ``best`` slot carry precisely
    the same collective resume boundary, including manager state after the selection.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as handle:
            torch.save(dict(payload), handle)
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


def _stable_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _validate_global_rng_state(
    value: Any,
    label: str,
    *,
    require_cuda_rng: bool = False,
) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint {label} is invalid")
    missing = {"python", "numpy", "torch"}.difference(value)
    if missing:
        raise ValueError(
            f"checkpoint {label} lacks " + ", ".join(sorted(missing))
        )
    if not isinstance(value["python"], tuple):
        raise ValueError(f"checkpoint {label} Python state is invalid")
    if not isinstance(value["numpy"], tuple):
        raise ValueError(f"checkpoint {label} NumPy state is invalid")
    if not isinstance(value["torch"], torch.Tensor) or value["torch"].numel() == 0:
        raise ValueError(f"checkpoint {label} torch state is invalid")
    cuda_states = value.get("torch_cuda")
    if require_cuda_rng and cuda_states is None:
        raise ValueError(f"checkpoint {label} lacks CUDA RNG state")
    if cuda_states is not None and (
        not isinstance(cuda_states, (list, tuple))
        or not cuda_states
        or any(not isinstance(state, torch.Tensor) or state.numel() == 0 for state in cuda_states)
    ):
        raise ValueError(f"checkpoint {label} CUDA state is invalid")


def _validate_generator_state(value: Any, label: str) -> None:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        raise ValueError(f"checkpoint {label} is invalid")


def validate_exact_resume_payload(
    payload: Mapping[str, Any],
    *,
    expected_identity: Mapping[str, Any] | None = None,
    expected_world_size: int | None = None,
    require_cuda_rng: bool = False,
) -> None:
    """Validate the complete collective boundary before mutating live objects."""
    required = {
        "schema_version",
        "model",
        "optimizer",
        "scheduler",
        "scaler",
        "step",
        "epoch",
        "completed_updates",
        "next_step",
        "config",
        "rng_state",
        "run_identity",
        "identity_sha256",
        "rank_states",
        "checkpoint_manager",
        "normalizer",
        "latent_index",
        "pending_eval_step",
        "final_eval",
        "phase",
        "gradient_checkpointing",
        "reason",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"checkpoint lacks exact-resume fields: {', '.join(missing)}")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("checkpoint schema differs")
    identity = payload.get("run_identity")
    if not isinstance(identity, Mapping):
        raise ValueError("checkpoint run identity is invalid")
    if expected_identity is not None and dict(identity) != dict(expected_identity):
        raise ValueError("checkpoint run identity does not match requested run")
    if payload.get("identity_sha256") != _stable_hash(identity):
        raise ValueError("checkpoint run-identity hash differs")
    if not isinstance(payload.get("config"), Mapping):
        raise ValueError("checkpoint resolved config is invalid")
    for name in ("step", "completed_updates", "next_step"):
        value = payload.get(name)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"checkpoint {name} is invalid")
    completed = payload["completed_updates"]
    if payload["step"] != completed:
        raise ValueError("checkpoint step/completed_updates mismatch")
    if payload["next_step"] != completed:
        raise ValueError("checkpoint next_step is not the next update boundary")
    if (
        not isinstance(payload.get("epoch"), int)
        or isinstance(payload.get("epoch"), bool)
        or payload["epoch"] < 0
    ):
        raise ValueError("checkpoint epoch is invalid")
    _validate_global_rng_state(
        payload.get("rng_state"),
        "global RNG state",
        require_cuda_rng=require_cuda_rng,
    )
    if not isinstance(payload.get("model"), Mapping):
        raise ValueError("checkpoint model state is invalid")
    optimizer_state = payload.get("optimizer")
    if not isinstance(optimizer_state, Mapping):
        raise ValueError("checkpoint optimizer state is unavailable")
    if not isinstance(optimizer_state.get("state"), Mapping) or not isinstance(
        optimizer_state.get("param_groups"), list
    ):
        raise ValueError("checkpoint optimizer structure is invalid")
    if not optimizer_state["param_groups"]:
        raise ValueError("checkpoint optimizer parameter groups are unavailable")
    if completed > 0 and not optimizer_state["state"]:
        raise ValueError("checkpoint optimizer state is empty after completed updates")
    scheduler_state = payload.get("scheduler")
    if not isinstance(scheduler_state, Mapping):
        raise ValueError("checkpoint scheduler state is unavailable")
    scheduler_required = {"last_epoch", "base_lrs", "_last_lr"}
    missing_scheduler = scheduler_required.difference(scheduler_state)
    if missing_scheduler:
        raise ValueError(
            "checkpoint scheduler state lacks "
            + ", ".join(sorted(missing_scheduler))
        )
    last_epoch = scheduler_state["last_epoch"]
    if (
        not isinstance(last_epoch, int)
        or isinstance(last_epoch, bool)
        or last_epoch != completed
    ):
        raise ValueError("checkpoint scheduler last_epoch differs from completed updates")
    optimizer_group_count = len(optimizer_state["param_groups"])
    for name in ("base_lrs", "_last_lr"):
        values = scheduler_state[name]
        if not isinstance(values, (list, tuple)) or len(values) != optimizer_group_count:
            raise ValueError(f"checkpoint scheduler {name} group count differs")
        try:
            finite = all(math.isfinite(float(value)) and float(value) >= 0.0 for value in values)
        except (TypeError, ValueError):
            finite = False
        if not finite:
            raise ValueError(f"checkpoint scheduler {name} contains invalid learning rates")
    if "_step_count" in scheduler_state:
        step_count = scheduler_state["_step_count"]
        if (
            not isinstance(step_count, int)
            or isinstance(step_count, bool)
            or step_count != completed + 1
        ):
            raise ValueError("checkpoint scheduler _step_count differs from update boundary")
    if payload.get("scaler") is not None and not isinstance(
        payload.get("scaler"), Mapping
    ):
        raise ValueError("checkpoint scaler state is invalid")
    identity_world_size = identity.get("world_size")
    if (
        not isinstance(identity_world_size, int)
        or isinstance(identity_world_size, bool)
        or identity_world_size <= 0
    ):
        raise ValueError("checkpoint world size is invalid")
    world_size = (
        int(expected_world_size)
        if expected_world_size is not None
        else identity_world_size
    )
    if world_size <= 0 or identity_world_size != world_size:
        raise ValueError("checkpoint world size differs")
    rank_states = payload.get("rank_states")
    if not isinstance(rank_states, list) or len(rank_states) != world_size:
        raise ValueError("checkpoint does not contain every rank state")
    ranks: list[int] = []
    for state in rank_states:
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint rank state is invalid")
        missing_rank = {
            "rank",
            "rng_state",
            "loader",
            "rng_streams",
            "horizon_generator",
        }.difference(state)
        if missing_rank:
            raise ValueError(
                "checkpoint rank state lacks " + ", ".join(sorted(missing_rank))
            )
        if not isinstance(state["rank"], int) or isinstance(state["rank"], bool):
            raise ValueError("checkpoint rank identifier is invalid")
        _validate_global_rng_state(
            state["rng_state"],
            "rank RNG state",
            require_cuda_rng=require_cuda_rng,
        )
        loader_state = state["loader"]
        if not isinstance(loader_state, Mapping):
            raise ValueError("checkpoint rank loader state is invalid")
        missing_loader = {
            "epoch",
            "batches_yielded_in_epoch",
            "epoch_generator_state",
        }.difference(loader_state)
        if missing_loader:
            raise ValueError(
                "checkpoint rank loader state lacks "
                + ", ".join(sorted(missing_loader))
            )
        for name in ("epoch", "batches_yielded_in_epoch"):
            value = loader_state[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"checkpoint rank loader {name} is invalid")
        epoch_generator = loader_state["epoch_generator_state"]
        if completed > 0:
            _validate_generator_state(
                epoch_generator, "rank loader epoch-generator state"
            )
        elif epoch_generator is not None:
            _validate_generator_state(
                epoch_generator, "rank loader epoch-generator state"
            )
        rng_streams = state["rng_streams"]
        if not isinstance(rng_streams, Mapping):
            raise ValueError("checkpoint rank RNG streams are invalid")
        missing_streams = {"planner", "eval", "viz"}.difference(rng_streams)
        if missing_streams:
            raise ValueError(
                "checkpoint rank RNG streams lack "
                + ", ".join(sorted(missing_streams))
            )
        for stream_name in ("planner", "eval", "viz"):
            _validate_generator_state(
                rng_streams[stream_name], f"rank {stream_name} RNG stream"
            )
        _validate_generator_state(
            state["horizon_generator"], "horizon-generator state"
        )
        if "metric_tracker" in state:
            # New checkpoints may preserve an in-flight scalar aggregation window.
            # Validate it here, before the trainer mutates any resumed state.
            from treewm.logging.metrics import MetricTracker

            try:
                MetricTracker().load_state_dict(state["metric_tracker"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "checkpoint rank metric-tracker state is invalid"
                ) from exc
        ranks.append(state["rank"])
    if sorted(ranks) != list(range(world_size)) or len(set(ranks)) != world_size:
        raise ValueError("checkpoint rank-state coverage differs")
    manager_state = payload.get("checkpoint_manager")
    if not isinstance(manager_state, Mapping):
        raise ValueError("checkpoint manager state is invalid")
    try:
        best_success = float(manager_state["best_success"])
        best_val_loss = float(manager_state["best_val_loss"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("checkpoint manager metrics are invalid") from exc
    if math.isnan(best_success) or best_success == float("inf"):
        raise ValueError("checkpoint best-success metric is invalid")
    if math.isnan(best_val_loss) or best_val_loss == -float("inf"):
        raise ValueError("checkpoint best-validation metric is invalid")
    if not isinstance(payload.get("normalizer"), Mapping):
        raise ValueError("checkpoint normalizer state is invalid")
    if payload.get("latent_index") is not None and not isinstance(
        payload.get("latent_index"), Mapping
    ):
        raise ValueError("checkpoint latent-index state is invalid")
    pending = payload.get("pending_eval_step")
    if pending is not None and (
        not isinstance(pending, int) or isinstance(pending, bool) or pending < 0
    ):
        raise ValueError("checkpoint pending evaluation step is invalid")
    if payload.get("final_eval") is not None and not isinstance(
        payload.get("final_eval"), Mapping
    ):
        raise ValueError("checkpoint final evaluation state is invalid")
    if payload.get("phase") not in {"train", "final_eval"}:
        raise ValueError("checkpoint phase is invalid")
    if not isinstance(payload.get("gradient_checkpointing"), bool):
        raise ValueError("checkpoint gradient-checkpointing state is invalid")
    if not isinstance(payload.get("reason"), str) or not payload["reason"]:
        raise ValueError("checkpoint reason is invalid")


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


def _validate_live_state_compatibility(
    payload: Mapping[str, Any],
    *,
    model: Any,
    optimizer: Any,
    scheduler: Any,
    strict_model: bool,
) -> None:
    """Reject incompatible live objects before any ``load_state_dict`` mutates them."""
    if model is not None:
        saved_model = payload.get("model")
        live_model = unwrap_model(model).state_dict()
        if not isinstance(saved_model, Mapping):
            raise ValueError("checkpoint model state is invalid")
        saved_keys = set(saved_model)
        live_keys = set(live_model)
        if strict_model and saved_keys != live_keys:
            raise ValueError("checkpoint model state keys differ from the live model")
        if not saved_keys.issubset(live_keys):
            raise ValueError("checkpoint model contains unknown live-state keys")
        for key in saved_keys:
            saved_value = saved_model[key]
            live_value = live_model[key]
            if not torch.is_tensor(saved_value) or not torch.is_tensor(live_value):
                raise ValueError(f"checkpoint model state {key!r} is not a tensor")
            if saved_value.shape != live_value.shape:
                raise ValueError(f"checkpoint model state {key!r} shape differs")
            if saved_value.dtype != live_value.dtype:
                raise ValueError(f"checkpoint model state {key!r} dtype differs")

    if optimizer is not None:
        saved_optimizer = payload.get("optimizer")
        live_optimizer = optimizer.state_dict()
        if not isinstance(saved_optimizer, Mapping):
            raise ValueError("checkpoint optimizer state is invalid")
        saved_groups = saved_optimizer.get("param_groups")
        live_groups = live_optimizer.get("param_groups")
        if not isinstance(saved_groups, list) or not isinstance(live_groups, list):
            raise ValueError("checkpoint optimizer parameter groups are invalid")
        if len(saved_groups) != len(live_groups):
            raise ValueError("checkpoint optimizer group count differs")
        saved_id_to_parameter: dict[Any, torch.Tensor] = {}
        for group_index, (saved_group, live_group, object_group) in enumerate(
            zip(saved_groups, live_groups, optimizer.param_groups, strict=True)
        ):
            saved_ids = saved_group.get("params")
            live_ids = live_group.get("params")
            live_parameters = object_group.get("params")
            if not all(
                isinstance(value, (list, tuple))
                for value in (saved_ids, live_ids, live_parameters)
            ):
                raise ValueError(
                    f"checkpoint optimizer group {group_index} parameter list is invalid"
                )
            if not (len(saved_ids) == len(live_ids) == len(live_parameters)):
                raise ValueError(
                    f"checkpoint optimizer group {group_index} parameter count differs"
                )
            saved_id_to_parameter.update(zip(saved_ids, live_parameters, strict=True))
        saved_state = saved_optimizer.get("state")
        if not isinstance(saved_state, Mapping):
            raise ValueError("checkpoint optimizer moment state is invalid")
        for parameter_id, state in saved_state.items():
            if parameter_id not in saved_id_to_parameter or not isinstance(state, Mapping):
                raise ValueError("checkpoint optimizer state references an unknown parameter")
            parameter = saved_id_to_parameter[parameter_id]
            for state_name, state_value in state.items():
                if not torch.is_tensor(state_value) or state_name == "step":
                    continue
                if state_value.shape != parameter.shape:
                    raise ValueError(
                        f"checkpoint optimizer {state_name!r} shape differs from its parameter"
                    )

    if scheduler is not None:
        saved_scheduler = payload.get("scheduler")
        live_scheduler = scheduler.state_dict()
        if not isinstance(saved_scheduler, Mapping) or not isinstance(
            live_scheduler, Mapping
        ):
            raise ValueError("checkpoint scheduler state is invalid")
        for key in ("base_lrs", "_last_lr"):
            if key not in live_scheduler or len(saved_scheduler[key]) != len(
                live_scheduler[key]
            ):
                raise ValueError(f"checkpoint scheduler {key} structure differs")


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
    require_exact_resume: bool = False,
    expected_world_size: int | None = None,
    require_cuda_rng: bool = False,
) -> dict[str, Any]:
    payload = torch.load(str(path), map_location=map_location, weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported checkpoint schema {payload.get('schema_version')!r}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    if expected_identity is not None and payload.get("run_identity") != expected_identity:
        raise ValueError(f"checkpoint run identity does not match requested run: {path}")
    if require_exact_resume:
        validate_exact_resume_payload(
            payload,
            expected_identity=expected_identity,
            expected_world_size=expected_world_size,
            require_cuda_rng=require_cuda_rng,
        )
        _validate_live_state_compatibility(
            payload,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            strict_model=strict,
        )
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
            set_rng_state(rng_state, strict_cuda=require_cuda_rng)
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

    def save_latest_payload(self, payload: Mapping[str, Any]) -> Path | None:
        if not self.enabled:
            return None
        validate_exact_resume_payload(payload)
        return save_checkpoint_payload(self.directory / "latest.pt", payload)

    def save_best_success_payload(self, payload: Mapping[str, Any]) -> Path | None:
        if not self.enabled:
            return None
        validate_exact_resume_payload(payload)
        return save_checkpoint_payload(self.directory / "best_success.pt", payload)

    def save_best_val_loss_payload(self, payload: Mapping[str, Any]) -> Path | None:
        if not self.enabled:
            return None
        validate_exact_resume_payload(payload)
        return save_checkpoint_payload(
            self.directory / "best_validation_loss.pt", payload
        )

    def record_success(self, success: float) -> bool:
        value = float(success)
        if not math.isfinite(value):
            raise ValueError("best-success metric must be finite")
        if value <= self.best_success:
            return False
        self.best_success = value
        return True

    def record_val_loss(self, val_loss: float) -> bool:
        value = float(val_loss)
        if not math.isfinite(value):
            raise ValueError("best-validation metric must be finite")
        if value >= self.best_val_loss:
            return False
        self.best_val_loss = value
        return True

    def maybe_save_success(self, success: float, **kwargs: Any) -> Path | None:
        if not math.isfinite(float(success)):
            raise ValueError("best-success metric must be finite")
        if not self.enabled or not self.record_success(success):
            return None
        return save_checkpoint(self.directory / "best_success.pt", **kwargs)

    def maybe_save_val_loss(self, val_loss: float, **kwargs: Any) -> Path | None:
        if not math.isfinite(float(val_loss)):
            raise ValueError("best-validation metric must be finite")
        if not self.enabled or not self.record_val_loss(val_loss):
            return None
        return save_checkpoint(self.directory / "best_validation_loss.pt", **kwargs)

    def state_dict(self) -> dict[str, float]:
        return {"best_success": self.best_success, "best_val_loss": self.best_val_loss}

    def load_state_dict(self, state: dict[str, float]) -> None:
        best_success = float(state.get("best_success", -float("inf")))
        best_val_loss = float(state.get("best_val_loss", float("inf")))
        if math.isnan(best_success) or best_success == float("inf"):
            raise ValueError("checkpoint best-success metric is invalid")
        if math.isnan(best_val_loss) or best_val_loss == -float("inf"):
            raise ValueError("checkpoint best-validation metric is invalid")
        self.best_success = best_success
        self.best_val_loss = best_val_loss
