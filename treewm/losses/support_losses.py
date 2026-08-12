"""Support, coverage, redundancy and mass.

The distinction this file exists to enforce (spec section 13):

    support (kappa)  "does this branch cover a distinct supported controllability mode"
    mass    (rho)    "how common is this mode in the offline data"

Given retrieved futures that split ``left 80% / straight 15% / right 5%``, the targets
are ``support = [1, 1, 1]`` and ``mass = [.80, .15, .05]``. The support objective is
mode-balanced, so a 5% mode contributes exactly as much gradient as an 80% one and
cannot be optimised away for being rare. Only the mass objective is frequency-aware.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def coverage_loss(
    pred_q: torch.Tensor,
    mode_q: torch.Tensor,
    mode_valid: torch.Tensor,
    q_cdist,
) -> torch.Tensor:
    """``L_coverage = (1/M) sum_j min_i d_q(q_j, q_hat_i)``.

    Averaged over *modes*, not over dataset samples: every retrieved mode gets equal
    weight regardless of how many neighbours landed in it.
    """
    dist = q_cdist(mode_q, pred_q)  # [B, C, K]
    nearest = dist.min(dim=-1).values  # [B, C]
    w = (mode_valid > 0).float()
    return (nearest * w).sum() / w.sum().clamp_min(1.0)


def redundancy_loss(
    pred_q: torch.Tensor,
    keep: torch.Tensor,
    q_cdist,
    temperature: float = 0.25,
) -> torch.Tensor:
    """``L_red = sum_{i<j} kappa_i kappa_j exp(-d_q(q_i, q_j) / tau)``.

    Penalises *keeping* two branches that lead to controllability-equivalent futures.
    Note it operates in q-space: two branches with different action chunks and different
    endpoint coordinates are not redundant unless their future controllability matches
    (spec section 10).
    """
    b, k = keep.shape
    if k < 2:
        return keep.sum() * 0.0
    dist = q_cdist(pred_q, pred_q)  # [B, K, K]
    sim = torch.exp(-dist / temperature)
    pair_keep = keep.unsqueeze(2) * keep.unsqueeze(1)
    triu = torch.triu(torch.ones(k, k, device=keep.device, dtype=torch.bool), diagonal=1)
    num_pairs = triu.sum().clamp_min(1)
    return (sim * pair_keep * triu.unsqueeze(0)).sum() / (b * num_pairs)


def keep_loss(keep_logit: torch.Tensor, matched: torch.Tensor, balance: bool = True) -> torch.Tensor:
    """Binary supervision: does this branch cover a distinct supported mode?

    Class-balanced by default. Typically ~3 of K=4 branches match a mode, so an
    unbalanced BCE would drift toward predicting "keep everything".
    """
    bce = F.binary_cross_entropy_with_logits(keep_logit, matched, reduction="none")
    if not balance:
        return bce.mean()
    pos = matched.sum().clamp_min(1.0)
    neg = (1.0 - matched).sum().clamp_min(1.0)
    weight = torch.where(matched > 0, 0.5 / pos, 0.5 / neg)
    return (bce * weight).sum()


def mass_loss(
    mass_logit: torch.Tensor,
    target_mass: torch.Tensor,
    matched: torch.Tensor,
) -> torch.Tensor:
    """Predict empirical mode prevalence, separately from support.

    Soft cross-entropy between the sibling-normalised predicted distribution and the
    matched modes' empirical masses. Unmatched branches carry zero target mass.
    """
    target = target_mass * matched
    total = target.sum(-1, keepdim=True)
    valid = (total.squeeze(-1) > 0).float()
    target = target / total.clamp_min(1e-8)
    log_pred = torch.log_softmax(mass_logit, dim=-1)
    ce = -(target * log_pred).sum(-1)
    return (ce * valid).sum() / valid.sum().clamp_min(1.0)


@torch.no_grad()
def support_metrics(
    keep: torch.Tensor,
    mass_pred: torch.Tensor,
    matched: torch.Tensor,
    target_mass: torch.Tensor,
    branch_target_mass: torch.Tensor,
    mode_valid: torch.Tensor,
    mode_to_branch: torch.Tensor,
    pred_q: torch.Tensor,
    pred_z: torch.Tensor,
    q_cdist,
    keep_threshold: float = 0.5,
    rare_threshold: float = 0.1,
) -> dict[str, float]:
    """``tree/*`` and ``stochastic/*`` scalars.

    ``rare_mode_recall`` is the headline: of the modes whose empirical mass is below
    ``rare_threshold``, how many are both matched to a branch *and* given a high KEEP
    score? If support has collapsed onto frequency this number falls while overall
    support recall stays high, which is precisely the failure the project is built to
    detect.

    ``target_mass`` is per-mode ``[B, C]``; ``branch_target_mass`` is the same quantity
    reindexed onto branches ``[B, K]``. Both are needed: rare-mode statistics are defined
    over modes, calibration error over branches.
    """
    kept = (keep > keep_threshold).float()
    eff_branching = kept.sum(-1)

    # Support precision/recall are measured against the matching, not against mass.
    covered = (mode_to_branch >= 0).float() * (mode_valid > 0).float()
    recall = covered.sum() / (mode_valid > 0).float().sum().clamp_min(1.0)
    precision = (kept * matched).sum() / kept.sum().clamp_min(1.0)

    rare = ((target_mass < rare_threshold) & (mode_valid > 0)).float()
    rare_covered = rare * (mode_to_branch >= 0).float()
    rare_recall = rare_covered.sum() / rare.sum().clamp_min(1.0)

    common = ((target_mass >= rare_threshold) & (mode_valid > 0)).float()
    common_recall = (common * (mode_to_branch >= 0).float()).sum() / common.sum().clamp_min(1.0)

    # Decoupling: correlation between KEEP and predicted mass across branches. Near zero
    # is the goal -- support that tracks frequency is support in name only.
    k_flat = keep.flatten().float()
    m_flat = mass_pred.flatten().float()
    if k_flat.numel() > 1 and k_flat.std() > 1e-6 and m_flat.std() > 1e-6:
        corr = float(
            ((k_flat - k_flat.mean()) * (m_flat - m_flat.mean())).mean() / (k_flat.std() * m_flat.std())
        )
    else:
        corr = 0.0

    dist_q = q_cdist(pred_q, pred_q)
    dist_z = torch.cdist(pred_z.float(), pred_z.float())
    k = keep.shape[-1]
    if k > 1:
        triu = torch.triu(torch.ones(k, k, device=keep.device, dtype=torch.bool), diagonal=1)
        mean_q = dist_q[:, triu].mean()
        mean_z = dist_z[:, triu].mean()
        redundancy_rate = ((dist_q[:, triu] < 0.1).float()).mean()
    else:
        mean_q = dist_q.mean() * 0
        mean_z = dist_z.mean() * 0
        redundancy_rate = dist_q.mean() * 0

    probs = keep / keep.sum(-1, keepdim=True).clamp_min(1e-8)
    branch_entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-1).mean()

    return {
        "tree/keep_rate": float(kept.mean().item()),
        "tree/effective_branching_factor": float(eff_branching.mean().item()),
        "tree/mean_num_supported_children": float(matched.sum(-1).mean().item()),
        "tree/branch_entropy": float(branch_entropy.item()),
        "tree/redundancy_rate": float(redundancy_rate.item()),
        "tree/mean_pairwise_q_distance": float(mean_q.item()),
        "tree/mean_pairwise_z_distance": float(mean_z.item()),
        "tree/support_recall": float(recall.item()),
        "tree/support_precision": float(precision.item()),
        "tree/rare_mode_recall": float(rare_recall.item()),
        "stochastic/rare_mode_recall": float(rare_recall.item()),
        "stochastic/mode_recall": float(recall.item()),
        "stochastic/mode_precision": float(precision.item()),
        "stochastic/support_frequency_decoupling": float(1.0 - abs(corr)),
        "stochastic/mass_calibration_error": float(
            (((mass_pred - branch_target_mass) * matched).abs().sum(-1)).mean().item()
        ),
        "stochastic/common_mode_recall": float(common_recall.item()),
    }
