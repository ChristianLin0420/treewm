"""World-model losses: state, action, horizon, bind, reconstruction, recursion.

All of these are computed on *matched* branches only. A branch that the Hungarian
assignment left unmatched has no target future, so supervising it with somebody else's
target would teach every branch the dominant mode -- the exact mode collapse FlatKWM is
supposed to avoid.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _masked_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    total = weight.sum()
    if total <= 0:
        return value.sum() * 0.0
    return (value * weight).sum() / total


def state_loss(pred_z: torch.Tensor, target_z: torch.Tensor, matched: torch.Tensor) -> torch.Tensor:
    """``L_state = d_z(z', z*)`` over matched branches. ``[B, K, D]``."""
    per_branch = F.mse_loss(pred_z, target_z, reduction="none").mean(-1)
    return _masked_mean(per_branch, matched)


def action_loss(
    pred_action: torch.Tensor,
    target_action: torch.Tensor,
    target_mask: torch.Tensor,
    matched: torch.Tensor,
) -> torch.Tensor:
    """Masked MSE up to the target horizon. ``[B, K, H, A]``.

    The mask is the *target's* horizon, so predictions past the end of the target chunk
    are neither rewarded nor penalised.
    """
    err = F.mse_loss(pred_action, target_action, reduction="none").mean(-1)  # [B, K, H]
    per_branch = (err * target_mask).sum(-1) / target_mask.sum(-1).clamp_min(1.0)
    return _masked_mean(per_branch, matched)


def horizon_loss(
    horizon_logits: torch.Tensor, target_idx: torch.Tensor, matched: torch.Tensor
) -> torch.Tensor:
    """Categorical cross-entropy over candidate horizons. ``[B, K, n_h]``."""
    b, k, n = horizon_logits.shape
    ce = F.cross_entropy(
        horizon_logits.reshape(b * k, n), target_idx.reshape(b * k).long().clamp(0, n - 1), reduction="none"
    ).view(b, k)
    return _masked_mean(ce, matched)


def bind_loss(
    model,
    z: torch.Tensor,
    branch_embedding: torch.Tensor,
    target_action: torch.Tensor,
    target_mask: torch.Tensor,
    target_horizon_idx: torch.Tensor,
    target_z: torch.Tensor,
    matched: torch.Tensor,
) -> torch.Tensor:
    """Action-consequence binding.

    Runs the *target* action chunk through the dynamics model and requires it to land on
    the target successor latent. Without this, ``F`` can learn to ignore ``A`` entirely
    and predict ``z'`` from ``z`` and the branch token alone -- which would make every
    "executable future hypothesis" a fiction, since the actions would not be what
    produces the predicted consequence (spec section 6).
    """
    z_teacher = model.dynamics(z, target_action, target_mask, target_horizon_idx, branch_embedding)
    per_branch = F.mse_loss(z_teacher, target_z, reduction="none").mean(-1)
    return _masked_mean(per_branch, matched)


def uncertainty_loss(
    sigma: torch.Tensor,
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
    matched: torch.Tensor,
    unmatched_quantile: float = 0.9,
) -> torch.Tensor:
    """Train ``sigma`` to estimate model error / lack of transition support.

    Without this the uncertainty head receives no gradient anywhere, and
    ``UncertaintyTreeWM`` would expand on an untrained output -- a random baseline
    wearing a different name, which would quietly corrupt the arm comparison.

    Matched branches regress onto their own detached prediction error. Unmatched
    branches have no supported target future at all, so they are pushed toward a high
    value taken from the upper quantile of the matched errors rather than an arbitrary
    constant that would depend on the loss scale.
    """
    with torch.no_grad():
        # float32 throughout: torch.quantile has no bf16 kernel, and this runs inside
        # an autocast region during training.
        err = (pred_z.float() - target_z.float()).pow(2).mean(-1)  # [B, K]
        matched_err = err[matched > 0]
        high = (
            torch.quantile(matched_err, unmatched_quantile)
            if matched_err.numel() > 0
            else err.max()
        )
        target = torch.where(matched > 0, err, high.expand_as(err))
    return F.smooth_l1_loss(sigma.float(), target)


def reconstruction_loss(decoder, z: torch.Tensor, obs: torch.Tensor) -> torch.Tensor:
    """``D(E(s)) ~ s``. Optional diagnostic decoder."""
    return F.mse_loss(decoder(z), obs)


def recursive_loss(
    model,
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
    matched: torch.Tensor,
    depth: torch.Tensor | None = None,
    max_nodes: int = 256,
) -> torch.Tensor:
    """Multi-level rollout stability.

    Applies the branch operator to the *predicted* successor latent and to the *encoded
    ground-truth* successor, and penalises disagreement. The operator is reused at every
    depth, so if it only behaves on encoded latents the tree degrades with depth and
    "recursive prediction" buys nothing (spec section 26, stage 5).

    Subsampled to ``max_nodes`` successors: this is O(B * K) applications of a K-token
    transformer, i.e. O(B * K^2) overall, which at FlatKWM's K=256 tries to allocate tens
    of gigabytes. A random subset each step gives the same objective in expectation.
    """
    b, k, d = pred_z.shape
    flat_pred = pred_z.reshape(b * k, d)
    flat_tgt = target_z.reshape(b * k, d)
    matched = matched.reshape(b * k)

    if b * k > max_nodes:
        sel = torch.randperm(b * k, device=pred_z.device)[:max_nodes]
        flat_pred, flat_tgt, matched = flat_pred[sel], flat_tgt[sel], matched[sel]
        depth = depth.reshape(b * k)[sel] if depth is not None else None

    n = flat_pred.shape[0]
    dep = (
        torch.ones(n, device=pred_z.device, dtype=torch.long)
        if depth is None
        else depth.reshape(n).long()
    )

    out_pred = model.branch(flat_pred, dep)
    with torch.no_grad():
        out_tgt = model.branch(flat_tgt, dep)

    # branch() applied to B*K latents returns K sub-branches per latent, so both terms
    # reduce over the sub-branch axis as well to give one scalar per predicted node.
    emb = F.mse_loss(out_pred.embedding, out_tgt.embedding, reduction="none").mean((-1, -2))
    act = F.mse_loss(out_pred.action, out_tgt.action, reduction="none").mean((-1, -2, -3))
    per = emb + act  # [N]
    return _masked_mean(per, matched)


@torch.no_grad()
def prediction_metrics(
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
    pred_action: torch.Tensor,
    target_action: torch.Tensor,
    target_mask: torch.Tensor,
    horizon_logits: torch.Tensor,
    target_horizon_idx: torch.Tensor,
    matched: torch.Tensor,
    horizon_values: torch.Tensor,
) -> dict[str, float]:
    """``model/*`` scalars."""
    m = matched
    denom = m.sum().clamp_min(1.0)

    latent_mse = ((pred_z - target_z).pow(2).mean(-1) * m).sum() / denom
    act_err = F.mse_loss(pred_action, target_action, reduction="none").mean(-1)
    act_mse = ((act_err * target_mask).sum(-1) / target_mask.sum(-1).clamp_min(1.0) * m).sum() / denom

    pred_h = horizon_logits.argmax(-1)
    acc = ((pred_h == target_horizon_idx.long()).float() * m).sum() / denom
    pred_len = horizon_values[pred_h.clamp(0, len(horizon_values) - 1)]
    tgt_len = horizon_values[target_horizon_idx.long().clamp(0, len(horizon_values) - 1)]
    mae = ((pred_len - tgt_len).abs() * m).sum() / denom

    # Does the *last* predicted action land where the successor latent says it should?
    # A cheap consistency check between the two halves of a branch.
    endpoint_consistency = (
        (pred_action[..., -1, :] - target_action[..., -1, :]).pow(2).mean(-1) * m
    ).sum() / denom

    return {
        "model/state_latent_mse": float(latent_mse.item()),
        "model/action_mse": float(act_mse.item()),
        "model/horizon_accuracy": float(acc.item()),
        "model/horizon_mae": float(mae.item()),
        "model/action_endpoint_consistency": float(endpoint_consistency.item()),
        "model/matched_fraction": float((m.sum() / m.numel()).item()),
    }
