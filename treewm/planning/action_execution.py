"""Canonical conversion from model actions to actions the environment receives.

TreeWM predicts actions in the training normalizer's coordinates.  Execution first
restores environment units and then clips to the environment's sealed action bounds.
The prospective executable-prefix objective must use that exact projection before it
asks the latent dynamics model where the controller will land; otherwise training and
planning reason about different actions.

Both NumPy (the historical planner path) and torch (the differentiable training path)
are supported by one public function.  Bounds are always explicit here.  Legacy
fallback policy, where needed for old runs, remains the caller's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import numpy as np
import torch


Array = TypeVar("Array", np.ndarray, torch.Tensor)


@dataclass(frozen=True)
class ExecutableActionProjection(Generic[Array]):
    """Raw and applied actions in environment and normalized coordinates."""

    raw_env: Array
    applied_env: Array
    applied_normalized: Array


def _validate_numpy_inputs(
    actions: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> None:
    if actions.ndim < 1:
        raise ValueError("normalized actions must have a trailing action dimension")
    action_dim = actions.shape[-1]
    for name, value in (
        ("action_mean", mean),
        ("action_std", std),
        ("action_lower_bound", lower),
        ("action_upper_bound", upper),
    ):
        if value.shape != (action_dim,):
            raise ValueError(f"{name} must have shape [{action_dim}]")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must be finite")
    if np.any(std <= 0):
        raise ValueError("action_std must be strictly positive")
    if np.any(lower >= upper):
        raise ValueError("every action lower bound must be below its upper bound")


def _validate_torch_inputs(
    actions: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
) -> None:
    if actions.ndim < 1:
        raise ValueError("normalized actions must have a trailing action dimension")
    action_dim = actions.shape[-1]
    for name, value in (
        ("action_mean", mean),
        ("action_std", std),
        ("action_lower_bound", lower),
        ("action_upper_bound", upper),
    ):
        if value.shape != (action_dim,):
            raise ValueError(f"{name} must have shape [{action_dim}]")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} must be finite")
    if bool((std <= 0).any()):
        raise ValueError("action_std must be strictly positive")
    if bool((lower >= upper).any()):
        raise ValueError("every action lower bound must be below its upper bound")


def project_normalized_actions(
    normalized_actions: Array,
    *,
    action_mean: Array,
    action_std: Array,
    action_lower_bound: Array,
    action_upper_bound: Array,
) -> ExecutableActionProjection[Array]:
    """Apply the controller's canonical denormalize-and-clip projection.

    The torch branch intentionally computes in float32 even under autocast.  Hard
    clipping matches environment execution exactly.  A separate raw-action objective
    supplies gradients when a prediction is outside the interval and this projection's
    derivative is therefore zero.
    """

    if torch.is_tensor(normalized_actions):
        if not all(
            torch.is_tensor(value)
            for value in (
                action_mean,
                action_std,
                action_lower_bound,
                action_upper_bound,
            )
        ):
            raise TypeError("torch actions require torch normalizer statistics and bounds")
        actions_t = normalized_actions.float()
        mean_t = action_mean.to(device=actions_t.device, dtype=torch.float32)
        std_t = action_std.to(device=actions_t.device, dtype=torch.float32)
        lower_t = action_lower_bound.to(device=actions_t.device, dtype=torch.float32)
        upper_t = action_upper_bound.to(device=actions_t.device, dtype=torch.float32)
        _validate_torch_inputs(actions_t, mean_t, std_t, lower_t, upper_t)
        raw_t = actions_t * std_t + mean_t
        applied_t = torch.maximum(torch.minimum(raw_t, upper_t), lower_t)
        normalized_t = (applied_t - mean_t) / std_t
        return ExecutableActionProjection(raw_t, applied_t, normalized_t)

    if not isinstance(normalized_actions, np.ndarray):
        raise TypeError("normalized_actions must be a numpy array or torch tensor")
    if not all(
        isinstance(value, np.ndarray)
        for value in (
            action_mean,
            action_std,
            action_lower_bound,
            action_upper_bound,
        )
    ):
        raise TypeError("numpy actions require numpy normalizer statistics and bounds")
    # Preserve the historical planner's float32 normalizer arithmetic exactly.
    actions_n = np.asarray(normalized_actions, dtype=np.float32)
    mean_n = np.asarray(action_mean, dtype=np.float32)
    std_n = np.asarray(action_std, dtype=np.float32)
    lower_n = np.asarray(action_lower_bound, dtype=np.float32)
    upper_n = np.asarray(action_upper_bound, dtype=np.float32)
    _validate_numpy_inputs(actions_n, mean_n, std_n, lower_n, upper_n)
    raw_n = (actions_n * std_n + mean_n).astype(np.float32, copy=False)
    applied_n = np.clip(raw_n, lower_n, upper_n).astype(np.float32, copy=False)
    normalized_n = ((applied_n - mean_n) / std_n).astype(np.float32, copy=False)
    return ExecutableActionProjection(raw_n, applied_n, normalized_n)


def uniform_action_bounds(
    action_dim: int,
    lower: float,
    upper: float,
    *,
    like: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
    """Expand sealed scalar bounds without silently choosing default values."""

    if action_dim < 1:
        raise ValueError("action_dim must be positive")
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("sealed action bounds must be finite and ordered")
    if torch.is_tensor(like):
        return (
            torch.full(
                (action_dim,), float(lower), device=like.device, dtype=torch.float32
            ),
            torch.full(
                (action_dim,), float(upper), device=like.device, dtype=torch.float32
            ),
        )
    return (
        np.full(action_dim, float(lower), dtype=np.float32),
        np.full(action_dim, float(upper), dtype=np.float32),
    )
