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
) -> torch.Tensor:
    """Full cost matrix ``[B, K, C]``.

    Action distance is masked to each *target's* horizon: comparing predicted actions
    beyond the target chunk's length would penalise a branch for padding.
    """
    b, k = pred_z.shape[:2]
    c = tgt_z.shape[1]

    d_z = torch.cdist(pred_z.float(), tgt_z.float())  # [B, K, C]

    if q_cdist is not None:
        d_q = q_cdist(pred_q.float(), tgt_q.float())
    else:
        d_q = torch.cdist(pred_q.float().flatten(2), tgt_q.float().flatten(2))

    # [B, K, 1, H, A] vs [B, 1, C, H, A]
    diff = pred_action.float().unsqueeze(2) - tgt_action.float().unsqueeze(1)
    mask = tgt_action_mask.float().unsqueeze(1).unsqueeze(-1)  # [B, 1, C, H, 1]
    d_a = ((diff**2) * mask).sum((-1, -2)) / mask.sum((-1, -2)).clamp_min(1.0)

    d_h = (pred_horizon_idx.float().unsqueeze(2) - tgt_horizon_idx.float().unsqueeze(1)).abs()

    cost = cfg.lambda_z * d_z + cfg.lambda_q * d_q + cfg.lambda_action * d_a + cfg.lambda_horizon * d_h
    invalid = (tgt_valid <= 0).view(b, 1, c).expand(b, k, c)
    return cost.masked_fill(invalid, LARGE_COST)


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
