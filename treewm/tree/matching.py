"""Matching predicted branches to retrieved future modes.

Given ``K`` predicted branches and up to ``C`` retrieved modes, decide which branch is
responsible for which mode. Everything supervised per-branch -- action chunk, horizon,
successor latent, KEEP, mass -- keys off this assignment, so it is the hinge between the
data-side definition of a mode and the model-side notion of a branch.

Cost (spec section 12)::

    C_ij = lam_z d_z(z_i, z_j) + lam_q d_q(q_i, q_j) + lam_A d_A(A_i, A_j) + lam_h d_h(h_i, h_j)

Hungarian assignment is the default: it is optimal, and at K<=8 / C<=8 the cost is
negligible next to a forward pass. Greedy matching is kept for ablations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

LARGE_COST = 1e6


@dataclass
class MatchingConfig:
    lambda_z: float = 1.0
    lambda_q: float = 1.0
    lambda_action: float = 0.5
    lambda_horizon: float = 0.1
    method: str = "hungarian"  # hungarian | greedy
    # ``rms_v2`` puts every component in dimensionless, dimension-invariant units.
    # ``legacy`` is retained so old checkpoints/ablations can reproduce their original
    # assignment exactly.
    # Historical callers remain legacy unless the v2 protocol opts in explicitly.
    normalization_version: str = "legacy"  # rms_v2 | legacy
    num_horizons: int = 5

    def __post_init__(self) -> None:
        if self.normalization_version not in {"rms_v2", "legacy"}:
            raise ValueError(
                "normalization_version must be 'rms_v2' or 'legacy', got "
                f"{self.normalization_version!r}"
            )
        if self.num_horizons < 1:
            raise ValueError("num_horizons must be at least 1")


def _target_centered_rms(
    target: torch.Tensor,
    valid: torch.Tensor,
    floor: float = 1e-3,
) -> torch.Tensor:
    """Detached batch-global centred RMS of the valid target vectors.

    Centring makes the matching units invariant to a global latent translation, while
    RMS normalisation makes them invariant to a global latent rescaling.  A scalar is
    deliberately shared by the whole batch: per-mode scales would change the relative
    assignment costs rather than merely choosing their units.
    """
    with torch.no_grad():
        selected = target.detach().float()[valid.detach() > 0]
        if selected.numel() == 0:
            rms = target.detach().float().sum() * 0.0
        else:
            centered = selected - selected.mean(dim=0, keepdim=True)
            rms = centered.square().mean().sqrt()
        return rms.clamp_min(floor)


def branch_mode_cost(
    pred_z: torch.Tensor,
    pred_q: torch.Tensor,
    pred_action: torch.Tensor,
    pred_horizon_idx: torch.Tensor,
    tgt_z: torch.Tensor,
    tgt_q: torch.Tensor,
    tgt_action: torch.Tensor,
    tgt_action_mask: torch.Tensor,
    tgt_horizon_idx: torch.Tensor,
    tgt_valid: torch.Tensor,
    cfg: MatchingConfig,
    q_cdist=None,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Full cost matrix ``[B, K, C]``.

    Action distance is masked to each *target's* horizon: comparing predicted actions
    beyond the target chunk's length would penalise a branch for padding.
    """
    b, k = pred_z.shape[:2]
    c = tgt_z.shape[1]

    pred_z_f, tgt_z_f = pred_z.float(), tgt_z.float()
    pred_q_f, tgt_q_f = pred_q.float(), tgt_q.float()
    pred_action_f, tgt_action_f = pred_action.float(), tgt_action.float()

    if cfg.normalization_version == "legacy":
        d_z = torch.cdist(pred_z_f, tgt_z_f)  # [B, K, C]
        latent_scale = tgt_z_f.detach().new_tensor(1.0)
        if q_cdist is not None:
            d_q = q_cdist(pred_q_f, tgt_q_f)
        else:
            d_q = torch.cdist(pred_q_f.flatten(2), tgt_q_f.flatten(2))

        # Historical action cost: mean over timesteps but sum over action dimensions.
        diff = pred_action_f.unsqueeze(2) - tgt_action_f.unsqueeze(1)
        mask = tgt_action_mask.float().unsqueeze(1).unsqueeze(-1)
        d_a = ((diff**2) * mask).sum((-1, -2)) / mask.sum((-1, -2)).clamp_min(1.0)
        d_h = (
            pred_horizon_idx.float().unsqueeze(2) - tgt_horizon_idx.float().unsqueeze(1)
        ).abs()
    else:
        # RMS L2 (rather than Euclidean L2) removes latent-dimension dependence.  The
        # detached target scale then gives globally scale/translation-invariant units.
        latent_scale = _target_centered_rms(tgt_z_f, tgt_valid)
        z_diff = pred_z_f.unsqueeze(2) - tgt_z_f.unsqueeze(1)
        d_z = (z_diff.square().mean(-1) + 1e-12).sqrt() / latent_scale

        if q_cdist is not None:
            d_q = q_cdist(pred_q_f, tgt_q_f) / 2.0
        else:
            d_q = torch.cdist(pred_q_f.flatten(2), tgt_q_f.flatten(2)) / 2.0

        # The target mask defines the executable prefix. Divide by both valid time and
        # action dimension, then take RMS, so padding and action width cannot alter the
        # component's units.
        diff = pred_action_f.unsqueeze(2) - tgt_action_f.unsqueeze(1)
        mask = tgt_action_mask.float().unsqueeze(1).unsqueeze(-1)
        action_dim = max(int(pred_action.shape[-1]), 1)
        denom = mask.sum((-1, -2)).clamp_min(1.0) * action_dim
        d_a = (((diff**2) * mask).sum((-1, -2)) / denom + 1e-12).sqrt()

        horizon_denom = max(int(cfg.num_horizons) - 1, 1)
        d_h = (
            pred_horizon_idx.float().unsqueeze(2) - tgt_horizon_idx.float().unsqueeze(1)
        ).abs() / horizon_denom

    components = {
        "z": d_z,
        "q": d_q,
        "action": d_a,
        "horizon": d_h,
        "latent_target_centered_rms": latent_scale,
    }
    cost = (
        cfg.lambda_z * d_z
        + cfg.lambda_q * d_q
        + cfg.lambda_action * d_a
        + cfg.lambda_horizon * d_h
    )
    invalid = (tgt_valid <= 0).view(b, 1, c).expand(b, k, c)
    cost = cost.masked_fill(invalid, LARGE_COST)
    if not return_components:
        return cost
    # Returned components follow the same validity contract as the total cost.  The
    # scalar scale is telemetry, not a pairwise component, and remains unmasked.
    components = {
        name: value.masked_fill(invalid, LARGE_COST) if value.ndim == 3 else value
        for name, value in components.items()
    }
    return cost, components


@torch.no_grad()
def assigned_cost_metrics(
    components: dict[str, torch.Tensor],
    branch_to_mode: torch.Tensor,
    matched: torch.Tensor,
    cfg: MatchingConfig,
) -> dict[str, float]:
    """Summarise component costs on the assignment actually used for supervision.

    Pairwise matrix averages are misleading because most entries were never assigned.
    This gathers one target per matched branch, reports raw component units, weighted
    contributions, their shares of the assigned total, and the latent unit scale.
    """
    idx = branch_to_mode.long().clamp_min(0).unsqueeze(-1)
    weight = (matched.detach() > 0).float()
    denom = weight.sum()

    def assigned_mean(value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3:
            raise ValueError("assigned matching components must have shape [B, K, C]")
        selected = torch.gather(value.detach().float(), 2, idx).squeeze(-1)
        if float(denom) == 0.0:
            return selected.sum() * 0.0
        return (selected * weight).sum() / denom

    names = ("z", "q", "action", "horizon")
    lambdas = {
        "z": float(cfg.lambda_z),
        "q": float(cfg.lambda_q),
        "action": float(cfg.lambda_action),
        "horizon": float(cfg.lambda_horizon),
    }
    raw = {name: assigned_mean(components[name]) for name in names}
    weighted = {name: raw[name] * lambdas[name] for name in names}
    total = sum(weighted.values())
    total_value = float(total.item())

    out: dict[str, float] = {}
    for name in names:
        raw_value = float(raw[name].item())
        weighted_value = float(weighted[name].item())
        out[f"matching/assigned_{name}_cost"] = raw_value
        out[f"matching/assigned_{name}_weighted_cost"] = weighted_value
        out[f"matching/assigned_{name}_weighted_share"] = (
            weighted_value / total_value if total_value > 0.0 else 0.0
        )
    out["matching/assigned_total_cost"] = total_value
    scale = components.get("latent_target_centered_rms")
    out["matching/latent_target_centered_rms"] = (
        float(scale.detach().float().item()) if scale is not None else 1.0
    )
    return out


def hungarian_match(cost: torch.Tensor, tgt_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimal one-to-one assignment.

    Returns:
        ``branch_to_mode [B, K]``  mode index per branch, ``-1`` if unmatched
        ``mode_to_branch [B, C]``  branch index per mode, ``-1`` if uncovered
    """
    from scipy.optimize import linear_sum_assignment

    b, k, c = cost.shape
    cost_np = cost.detach().float().cpu().numpy()
    valid_np = tgt_valid.detach().float().cpu().numpy()

    branch_to_mode = np.full((b, k), -1, dtype=np.int64)
    mode_to_branch = np.full((b, c), -1, dtype=np.int64)
    for i in range(b):
        usable = np.flatnonzero(valid_np[i] > 0)
        if len(usable) == 0:
            continue
        sub = cost_np[i][:, usable]
        rows, cols = linear_sum_assignment(sub)
        for r, col in zip(rows, cols):
            if sub[r, col] >= LARGE_COST:
                continue
            mode = int(usable[col])
            branch_to_mode[i, r] = mode
            mode_to_branch[i, mode] = r

    dev = cost.device
    return (
        torch.from_numpy(branch_to_mode).to(dev),
        torch.from_numpy(mode_to_branch).to(dev),
    )


def greedy_match(cost: torch.Tensor, tgt_valid: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Nearest branch per mode with coverage balancing (each branch used at most once)."""
    b, k, c = cost.shape
    work = cost.clone()
    work = work.masked_fill((tgt_valid <= 0).view(b, 1, c).expand(b, k, c), LARGE_COST)

    branch_to_mode = torch.full((b, k), -1, dtype=torch.long, device=cost.device)
    mode_to_branch = torch.full((b, c), -1, dtype=torch.long, device=cost.device)
    for _ in range(min(k, c)):
        flat = work.view(b, k * c)
        best = flat.argmin(dim=1)
        bi, ci = best // c, best % c
        val = flat.gather(1, best.unsqueeze(1)).squeeze(1)
        ok = val < LARGE_COST
        rows = torch.arange(b, device=cost.device)
        branch_to_mode[rows[ok], bi[ok]] = ci[ok]
        mode_to_branch[rows[ok], ci[ok]] = bi[ok]
        work[rows, bi, :] = LARGE_COST
        work[rows, :, ci] = LARGE_COST
    return branch_to_mode, mode_to_branch


def match(cost: torch.Tensor, tgt_valid: torch.Tensor, cfg: MatchingConfig):
    if cfg.method == "hungarian":
        return hungarian_match(cost, tgt_valid)
    if cfg.method == "greedy":
        return greedy_match(cost, tgt_valid)
    raise ValueError(f"unknown matching method {cfg.method!r}")


def gather_matched_targets(
    targets: dict[str, torch.Tensor], branch_to_mode: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Reindex ``[B, C, ...]`` mode targets onto ``[B, K, ...]`` branch slots.

    Unmatched branches receive slot-0 values with ``matched = 0``; every loss that
    consumes these must multiply by ``matched``.
    """
    idx = branch_to_mode.clamp_min(0)
    out: dict[str, torch.Tensor] = {}
    for key, value in targets.items():
        if value.dim() < 2:
            continue
        expand = idx.view(*idx.shape, *([1] * (value.dim() - 2))).expand(
            *idx.shape, *value.shape[2:]
        )
        out[key] = torch.gather(value, 1, expand)
    out["matched"] = (branch_to_mode >= 0).float()
    return out
