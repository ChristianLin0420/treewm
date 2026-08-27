"""A sealed scale gauge for the otherwise affine-invariant v2 latent objective.

The corrected v2 world losses deliberately divide by a detached target scale.  That
makes their geometry well conditioned, but also leaves a global ``z -> c z`` gauge:
the encoder and decoder can jointly shrink ``c`` without changing those losses.  This
module fixes only the dangerous (shrinking) half of that gauge.  It anchors the root
and encoded-future populations to their outcome-blind update-zero scales while leaving
latent expansion completely unpenalised.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch import nn


def _flatten_population(
    values: torch.Tensor,
    valid: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return float32 ``[N, D]`` values and nonnegative ``[N]`` weights."""
    if values.ndim < 2 or values.shape[-1] < 1:
        raise ValueError("latent gauge requires a non-empty population and latent dimension")
    flat = values.float().reshape(-1, values.shape[-1])
    if valid is None:
        weight = torch.ones(flat.shape[0], device=flat.device, dtype=torch.float32)
    else:
        if tuple(valid.shape) != tuple(values.shape[:-1]):
            raise ValueError(
                f"valid shape {tuple(valid.shape)} does not match latent population "
                f"{tuple(values.shape[:-1])}"
            )
        weight = valid.detach().float().reshape(-1)
        if not bool(torch.isfinite(weight).all()) or bool((weight < 0).any()):
            raise ValueError("latent-gauge validity weights must be finite and nonnegative")
    return flat, weight


def centered_rms(
    values: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Differentiable RMS after centering every latent coordinate over a population.

    The statistic is translation invariant, uses one scalar for the whole population,
    and remains in float32 under autocast.  ``epsilon`` is inside the square root so a
    constant synthetic population stays finite without introducing a hard clamp.
    """
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("latent-gauge epsilon must be finite and positive")
    flat, weight = _flatten_population(values, valid)
    count = weight.sum()
    if not bool(torch.isfinite(flat).all()):
        raise FloatingPointError("latent-gauge population contains non-finite values")
    if float(count.detach().item()) <= 0.0:
        raise ValueError("latent gauge requires at least one valid population member")
    mean = (flat * weight.unsqueeze(-1)).sum(0) / count
    centered_square = (flat - mean).square() * weight.unsqueeze(-1)
    variance = centered_square.sum() / (count * flat.shape[-1])
    return (variance + float(epsilon) ** 2).sqrt()


def distributed_centered_rms(
    values: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Differentiable centered RMS of the union of all DDP populations.

    A global update-zero reference must be compared with the same global statistic at
    every later update.  Reducing only the detached reference but using rank-local
    current scales would create a nonzero loss at update zero whenever rank populations
    differ.  The functional collectives below retain autograd edges; DDP's subsequent
    gradient averaging then yields the gradient of the single global-population loss.
    """
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("latent-gauge epsilon must be finite and positive")
    flat, weight = _flatten_population(values, valid)
    if not bool(torch.isfinite(flat).all()):
        raise FloatingPointError("latent-gauge population contains non-finite values")
    # Use the same stable two-pass statistic as the sealed reference.  The global
    # mean may be detached: its omitted derivative term is proportional to the global
    # sum of centered values, which is zero at that mean.  The centered-square SUM
    # remains autograd-aware so every rank contributes to the active loss.
    coordinate_sum = (flat.detach() * weight.unsqueeze(-1)).sum(0)
    count = weight.sum()
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(coordinate_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    if float(count.detach().item()) <= 0.0:
        raise ValueError("latent gauge requires at least one globally valid member")
    global_mean = coordinate_sum / count
    centered_square_sum = (
        (flat - global_mean).square() * weight.unsqueeze(-1)
    ).sum()
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        centered_square_sum = dist_nn.all_reduce(
            centered_square_sum, op=dist.ReduceOp.SUM
        )
    variance = centered_square_sum / (count * flat.shape[-1])
    return (variance + float(epsilon) ** 2).sqrt()


@torch.no_grad()
def distributed_centered_rms_reference(
    values: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    epsilon: float = 1.0e-8,
) -> torch.Tensor:
    """Exact centered RMS of the union of equal-step DDP rank populations.

    Only detached sufficient statistics are reduced.  The resulting initialization
    reference is identical on every rank and cannot create a cross-rank autograd edge.
    """
    if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
        raise ValueError("latent-gauge epsilon must be finite and positive")
    flat, weight = _flatten_population(values.detach(), valid)
    if not bool(torch.isfinite(flat).all()):
        raise FloatingPointError("latent-gauge reference population is non-finite")
    weighted = flat * weight.unsqueeze(-1)
    coordinate_sum = weighted.sum(0)
    count = weight.sum()
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(coordinate_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    if float(count.item()) <= 0.0:
        raise ValueError("latent gauge requires at least one globally valid member")
    global_mean = coordinate_sum / count
    centered_square_sum = (
        (flat - global_mean).square() * weight.unsqueeze(-1)
    ).sum()
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(centered_square_sum, op=dist.ReduceOp.SUM)
    variance = centered_square_sum / (count * flat.shape[-1])
    return (variance + float(epsilon) ** 2).sqrt()


class LatentGauge(nn.Module):
    """Persistent update-zero root/future scale references and shrink-only loss."""

    def __init__(
        self,
        *,
        epsilon: float = 1.0e-8,
        min_reference_scale: float = 1.0e-4,
    ) -> None:
        super().__init__()
        if not math.isfinite(float(epsilon)) or float(epsilon) <= 0.0:
            raise ValueError("latent-gauge epsilon must be finite and positive")
        if (
            not math.isfinite(float(min_reference_scale))
            or float(min_reference_scale) <= float(epsilon)
        ):
            raise ValueError("latent-gauge minimum reference must exceed epsilon")
        self.epsilon = float(epsilon)
        self.min_reference_scale = float(min_reference_scale)
        # Zero is a finite unsealed checkpoint value; ``sealed_update`` is the sole
        # state sentinel. Avoiding NaN here lets an update-zero lifecycle checkpoint
        # pass generic tensor-finiteness audits before the first batch is consumed.
        self.register_buffer("root_reference", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("future_reference", torch.tensor(0.0, dtype=torch.float32))
        self.register_buffer("sealed_update", torch.tensor(-1, dtype=torch.int64))

    @property
    def is_sealed(self) -> bool:
        return (
            int(self.sealed_update.item()) == 0
            and bool(torch.isfinite(self.root_reference))
            and bool(torch.isfinite(self.future_reference))
            and float(self.root_reference.item()) >= self.min_reference_scale
            and float(self.future_reference.item()) >= self.min_reference_scale
        )

    @torch.no_grad()
    def seal(
        self,
        root: torch.Tensor,
        future: torch.Tensor,
        future_valid: torch.Tensor,
        *,
        step: int,
    ) -> None:
        """Seal both references atomically, and only on the first update."""
        if self.is_sealed:
            return
        if int(self.sealed_update.item()) != -1:
            raise ValueError("latent-gauge reference state is partially initialized")
        if int(step) != 0:
            raise ValueError(
                "unsealed latent-gauge reference after update zero; exact resume state is missing"
            )
        root_reference = distributed_centered_rms_reference(
            root, epsilon=self.epsilon
        )
        future_reference = distributed_centered_rms_reference(
            future, future_valid, epsilon=self.epsilon
        )
        for name, value in (
            ("root", root_reference),
            ("future", future_reference),
        ):
            scalar = float(value.item())
            if not math.isfinite(scalar) or scalar < self.min_reference_scale:
                raise ValueError(
                    f"latent-gauge {name} initialization scale {scalar:.8g} is below "
                    f"the sealed minimum {self.min_reference_scale:.8g}"
                )
        self.root_reference.copy_(root_reference)
        self.future_reference.copy_(future_reference)
        self.sealed_update.fill_(0)

    def forward(
        self,
        root: torch.Tensor,
        future: torch.Tensor,
        future_valid: torch.Tensor,
        *,
        step: int,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        self.seal(root, future, future_valid, step=step)
        if not self.is_sealed:
            raise RuntimeError("latent-gauge initialization reference was not sealed")

        root_scale = distributed_centered_rms(root, epsilon=self.epsilon)
        future_scale = distributed_centered_rms(
            future, future_valid, epsilon=self.epsilon
        )
        root_ratio = root_scale / self.root_reference.detach()
        future_ratio = future_scale / self.future_reference.detach()
        root_log_ratio = root_ratio.clamp_min(self.epsilon).log()
        future_log_ratio = future_ratio.clamp_min(self.epsilon).log()
        root_loss = torch.relu(-root_log_ratio).square()
        future_loss = torch.relu(-future_log_ratio).square()
        loss = 0.5 * (root_loss + future_loss)

        # Both current scales and both references are global, so every rank already
        # observes the same ratio.  No detached metric-only reduction is necessary.
        conservative_ratio = torch.minimum(root_ratio, future_ratio).detach().float()
        metrics = {
            "latent_gauge/root/scale": float(root_scale.detach().item()),
            "latent_gauge/root/reference": float(self.root_reference.item()),
            "latent_gauge/root/ratio": float(root_ratio.detach().item()),
            "latent_gauge/root/log_ratio": float(root_log_ratio.detach().item()),
            "latent_gauge/root/loss": float(root_loss.detach().item()),
            "latent_gauge/future/scale": float(future_scale.detach().item()),
            "latent_gauge/future/reference": float(self.future_reference.item()),
            "latent_gauge/future/ratio": float(future_ratio.detach().item()),
            "latent_gauge/future/log_ratio": float(future_log_ratio.detach().item()),
            "latent_gauge/future/loss": float(future_loss.detach().item()),
            "latent_gauge/min_ratio": float(conservative_ratio.item()),
            "latent_gauge/loss": float(loss.detach().item()),
            "latent_gauge/reference_sealed": 1.0,
            "latent_gauge/reference_update": float(self.sealed_update.item()),
        }
        return loss, metrics
