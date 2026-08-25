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
    normalization_version: str = "rms_v2",
    space: str = "q",
    branch_to_mode: torch.Tensor | None = None,
) -> torch.Tensor:
    """Detached one-to-one mode coverage in q- or z-space.

    Averaged over *modes*, not over dataset samples: every retrieved mode gets equal
    weight regardless of how many neighbours landed in it.  The assignment and targets
    are detached: coverage trains predictions toward distinct targets, never the target
    encoder toward whichever prediction happened to win the assignment.

    ``rms_v2`` puts the bounded q-distance in ``[0, 1]`` by dividing its ``[0, 2]``
    contract by two.  The z-space ablation instead uses relative RMS distance with the
    same detached, batch-global centred target scale as matching. ``legacy`` preserves
    the old component units, while still using the corrected one-to-one assignment.
    """
    if normalization_version not in {"rms_v2", "legacy"}:
        raise ValueError("normalization_version must be 'rms_v2' or 'legacy'")
    if space not in {"q", "z"}:
        raise ValueError("coverage space must be 'q' or 'z'")

    target = mode_q.detach().float()
    pred = pred_q.float()
    if normalization_version == "rms_v2" and space == "z":
        # Flatten the optional scale axis so this accepts both [B, C, D] and the
        # historical z-ablation shape [B, C, 1, D].
        target_flat = target.flatten(2)
        pred_flat = pred.flatten(2)
        selected = target_flat[mode_valid.detach() > 0]
        if selected.numel() == 0:
            scale = target_flat.sum() * 0.0
        else:
            centered = selected - selected.mean(dim=0, keepdim=True)
            scale = centered.square().mean().sqrt()
        scale = scale.clamp_min(1e-3)
        diff = target_flat.unsqueeze(2) - pred_flat.unsqueeze(1)
        dist = (diff.square().mean(-1) + 1e-12).sqrt() / scale
    else:
        dist = q_cdist(target, pred)  # [B, C, K]
        if normalization_version == "rms_v2":
            dist = dist / 2.0

    # Formal v2 supplies the *canonical* joint z/q/action/horizon assignment used by
    # every other branch objective.  Re-matching here in q-space can assign a branch to
    # a different mode than its state/action target, producing contradictory gradients.
    # The internal Hungarian fallback is retained only for historical direct callers.
    if branch_to_mode is None:
        from treewm.tree.matching import hungarian_match

        assignment, _ = hungarian_match(
            dist.detach().transpose(1, 2), mode_valid.detach()
        )
    else:
        assignment = branch_to_mode.detach().long()
        expected = (pred.shape[0], pred.shape[1])
        if assignment.shape != expected:
            raise ValueError(
                f"branch_to_mode shape {tuple(assignment.shape)} != {expected}"
            )
        if bool((assignment >= mode_q.shape[1]).any()) or bool((assignment < -1).any()):
            raise ValueError("branch_to_mode contains an out-of-range mode index")
        selected_valid = torch.gather(
            mode_valid.detach(), 1, assignment.clamp_min(0)
        )
        if bool(((assignment >= 0) & (selected_valid <= 0)).any()):
            raise ValueError("branch_to_mode selects an invalid mode")

    matched = assignment >= 0
    if not bool(matched.any()):
        return pred.sum() * 0.0
    idx = assignment.clamp_min(0).unsqueeze(-1)
    assigned = torch.gather(dist.transpose(1, 2), 2, idx).squeeze(-1)
    weight = matched.float()
    return (assigned * weight).sum() / weight.sum()


def redundancy_loss(
    pred_q: torch.Tensor,
    keep: torch.Tensor,
    q_cdist,
    temperature: float = 0.25,
    matched: torch.Tensor | None = None,
    keep_threshold: float = 0.5,
) -> torch.Tensor:
    """Similarity of distinct target-supported branch pairs.

    In v2 the active gate is ``matched.detach()`` only.  Predicted KEEP must not be able
    to reduce this loss by pruning a duplicate, and the redundancy term must not send a
    gradient into the KEEP head.  Each eligible anchor is normalised by its own active
    pair count before anchors are averaged, making the objective invariant to K and to
    inactive padding branches.

    ``matched=None`` retains call compatibility for old code by deriving a *detached*
    active gate from KEEP. New training code should always pass ``matched`` explicitly.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    b, k = keep.shape
    if k < 2:
        return pred_q.sum() * 0.0
    dist = q_cdist(pred_q, pred_q)  # [B, K, K]
    sim = torch.exp(-dist / temperature)
    active = (
        (keep.detach() >= keep_threshold)
        if matched is None
        else (matched.detach() > 0)
    )
    triu = torch.triu(torch.ones(k, k, device=keep.device, dtype=torch.bool), diagonal=1)
    pair_active = active.unsqueeze(2) & active.unsqueeze(1) & triu.unsqueeze(0)
    pair_count = pair_active.sum((1, 2))
    eligible = pair_count > 0
    if not bool(eligible.any()):
        return pred_q.sum() * 0.0
    per_anchor = (sim * pair_active).sum((1, 2)) / pair_count.clamp_min(1)
    return per_anchor[eligible].mean()


def keep_loss(
    keep_logit: torch.Tensor,
    matched: torch.Tensor,
    balance: bool | str = True,
) -> torch.Tensor:
    """Binary supervision: does this branch cover a distinct supported mode?

    ``balance=False`` (or ``balance="standard"``) is ordinary BCE and therefore the
    calibrated-probability option.  ``balance=True`` retains the historical class-
    balanced objective for existing experiments.
    """
    if isinstance(balance, str):
        if balance in {"standard", "calibrated", "unbalanced"}:
            balance = False
        elif balance == "balanced":
            balance = True
        else:
            raise ValueError(f"unknown KEEP BCE mode {balance!r}")
    target = matched.detach().float()
    bce = F.binary_cross_entropy_with_logits(keep_logit, target, reduction="none")
    if not balance:
        return bce.mean()
    pos = target.sum().clamp_min(1.0)
    neg = (1.0 - target).sum().clamp_min(1.0)
    weight = torch.where(target > 0, 0.5 / pos, 0.5 / neg)
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
    target = target_mass.detach() * matched.detach()
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
    mass_enabled: bool = True,
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
    keep_f = keep.float()
    kept_b = keep_f >= keep_threshold
    kept = kept_b.float()
    matched_b = matched > 0
    valid_mode = mode_valid > 0
    eff_branching = kept.sum(-1)

    # Matching alone is only an assignment-capacity diagnostic. Genuine support recall
    # additionally requires the assigned branch to survive the KEEP decision.
    assigned_mode = (mode_to_branch >= 0) & valid_mode
    mode_branch = mode_to_branch.long().clamp_min(0)
    assigned_keep = torch.gather(kept_b, 1, mode_branch)
    mode_kept = assigned_mode & assigned_keep

    num_modes = valid_mode.sum()
    num_assigned_modes = assigned_mode.sum()
    num_mode_kept = mode_kept.sum()
    assignment_recall = num_assigned_modes.float() / num_modes.clamp_min(1).float()
    mode_keep_recall = num_mode_kept.float() / num_modes.clamp_min(1).float()

    true_positive = kept_b & matched_b
    branch_precision = true_positive.sum().float() / kept_b.sum().clamp_min(1).float()
    branch_recall = true_positive.sum().float() / matched_b.sum().clamp_min(1).float()
    keep_brier = (keep_f - matched.float()).square().mean()

    rare = (target_mass < rare_threshold) & valid_mode
    common = (target_mass >= rare_threshold) & valid_mode
    rare_recall = (mode_kept & rare).sum().float() / rare.sum().clamp_min(1).float()
    common_recall = (mode_kept & common).sum().float() / common.sum().clamp_min(1).float()

    # Decoupling: correlation between KEEP and predicted mass across branches. Near zero
    # is the goal -- support that tracks frequency is support in name only.
    k_flat = keep_f.flatten()
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

    probs = keep_f / keep_f.sum(-1, keepdim=True).clamp_min(1e-8)
    branch_entropy = -(probs * probs.clamp_min(1e-8).log()).sum(-1).mean()

    # Mass calibration is evaluated against the same conditional distribution used by
    # mass_loss. Uncovered target probability is reported separately, so assignment
    # capacity cannot disappear inside a renormalised KL/TV score.
    eps = 1e-8
    pred_mass = mass_pred.float().clamp_min(0)
    pred_mass = pred_mass / pred_mass.sum(-1, keepdim=True).clamp_min(eps)
    branch_target = branch_target_mass.float().clamp_min(0) * matched.float()
    branch_total = branch_target.sum(-1, keepdim=True)
    mass_anchor_valid = branch_total.squeeze(-1) > 0
    branch_target = branch_target / branch_total.clamp_min(eps)

    log_pred_mass = pred_mass.clamp_min(eps).log()
    log_target_mass = branch_target.clamp_min(eps).log()
    mass_ce_per = -(branch_target * log_pred_mass).sum(-1)
    mass_entropy_per = -(branch_target * log_target_mass).sum(-1)
    mass_kl_per = (branch_target * (log_target_mass - log_pred_mass)).sum(-1).clamp_min(0)
    mass_tv_per = 0.5 * (pred_mass - branch_target).abs().sum(-1)
    mass_brier_per = (pred_mass - branch_target).square().sum(-1)

    def _valid_anchor_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        weight = valid.float()
        return (value * weight).sum() / weight.sum().clamp_min(1.0)

    mass_ce = _valid_anchor_mean(mass_ce_per, mass_anchor_valid)
    mass_target_entropy = _valid_anchor_mean(mass_entropy_per, mass_anchor_valid)
    mass_kl = _valid_anchor_mean(mass_kl_per, mass_anchor_valid)
    mass_tv = _valid_anchor_mean(mass_tv_per, mass_anchor_valid)
    mass_brier = _valid_anchor_mean(mass_brier_per, mass_anchor_valid)

    full_target = target_mass.float().clamp_min(0) * valid_mode.float()
    full_total = full_target.sum(-1, keepdim=True)
    full_anchor_valid = full_total.squeeze(-1) > 0
    full_target = full_target / full_total.clamp_min(eps)
    uncovered_per = (full_target * (~assigned_mode).float()).sum(-1)
    mass_uncovered = _valid_anchor_mean(uncovered_per, full_anchor_valid)

    metrics = {
        "tree/keep_rate": float(kept.mean().item()),
        "tree/effective_branching_factor": float(eff_branching.mean().item()),
        "tree/mean_num_supported_children": float(matched.sum(-1).mean().item()),
        "tree/branch_entropy": float(branch_entropy.item()),
        "tree/redundancy_rate": float(redundancy_rate.item()),
        "tree/mean_pairwise_q_distance": float(mean_q.item()),
        "tree/mean_pairwise_z_distance": float(mean_z.item()),
        "tree/support_recall": float(mode_keep_recall.item()),
        "tree/support_precision": float(branch_precision.item()),
        "tree/assignment_capacity_recall": float(assignment_recall.item()),
        "tree/mode_keep_recall": float(mode_keep_recall.item()),
        "tree/branch_keep_precision": float(branch_precision.item()),
        "tree/branch_keep_recall": float(branch_recall.item()),
        "tree/keep_brier": float(keep_brier.item()),
        "tree/rare_mode_recall": float(rare_recall.item()),
        "tree/common_mode_recall": float(common_recall.item()),
        "tree/num_modes": float(num_modes.item()),
        "tree/num_rare_modes": float(rare.sum().item()),
        "tree/num_common_modes": float(common.sum().item()),
        "tree/num_multimode_anchors": float((valid_mode.sum(-1) > 1).sum().item()),
        "tree/num_assigned_modes": float(num_assigned_modes.item()),
        "tree/num_mode_kept": float(num_mode_kept.item()),
        "tree/num_kept_branches": float(kept_b.sum().item()),
        "tree/num_keep_true_positives": float(true_positive.sum().item()),
        "stochastic/rare_mode_recall": float(rare_recall.item()),
        "stochastic/mode_recall": float(mode_keep_recall.item()),
        "stochastic/mode_precision": float(branch_precision.item()),
        "stochastic/common_mode_recall": float(common_recall.item()),
    }
    # Formal v2 deliberately disables and freezes the mass head because its output is
    # not consumed by the planner.  Emitting calibration scores from that frozen,
    # random head would make healthy runs look scientifically meaningful.  Keep the
    # legacy telemetry only when the objective is actually enabled.
    if mass_enabled:
        metrics.update(
            {
                "tree/mass_ce": float(mass_ce.item()),
                "tree/mass_target_entropy": float(mass_target_entropy.item()),
                "tree/mass_kl": float(mass_kl.item()),
                "tree/mass_tv": float(mass_tv.item()),
                "tree/mass_brier": float(mass_brier.item()),
                "tree/mass_uncovered_target": float(mass_uncovered.item()),
                "stochastic/support_frequency_decoupling": float(1.0 - abs(corr)),
                "stochastic/mass_calibration_error": float(mass_tv.item()),
            }
        )
    return metrics
