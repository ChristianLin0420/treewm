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


def detached_target_scale(
    target_z: torch.Tensor,
    valid: torch.Tensor | None = None,
    *,
    min_scale: float = 1e-3,
) -> torch.Tensor:
    """Detached centered RMS scale for a latent target population.

    Dividing latent squared errors by this scale squared makes them invariant to a
    global affine reparameterization ``z -> c*z + b``.  The scale is one scalar per
    objective, not per example, so it does not erase meaningful relative errors within
    a batch. The 1e-3 floor matches v2 latent matching and makes constant/single-target
    batches finite; callers still mask objectives with no valid targets to zero.
    """
    if target_z.ndim == 0 or target_z.shape[-1] == 0:
        raise ValueError("target_z must have a non-empty latent dimension")
    if min_scale <= 0:
        raise ValueError("min_scale must be positive")
    with torch.no_grad():
        detached = target_z.detach().float()
        if valid is None:
            selected = detached.reshape(-1, detached.shape[-1])
        else:
            weight = valid.detach().float()
            if weight.shape != detached.shape[:-1]:
                raise ValueError(
                    f"valid shape {tuple(weight.shape)} does not match latent population "
                    f"{tuple(detached.shape[:-1])}"
                )
            selected = detached[weight > 0]
        if selected.numel() == 0:
            scale = detached.sum() * 0.0
        else:
            centered = selected - selected.mean(dim=0, keepdim=True)
            scale = centered.pow(2).mean().sqrt()
        return scale.clamp_min(float(min_scale)).detach()


def scale_invariant_latent_error(
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
    target_scale: torch.Tensor,
) -> torch.Tensor:
    """Per-item latent MSE expressed in detached target-RMS-squared units."""
    scale_sq = target_scale.detach().float().pow(2)
    return (pred_z.float() - target_z.float()).pow(2).mean(-1) / scale_sq


def state_loss(
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
    matched: torch.Tensor,
    *,
    target_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Scale-invariant ``d_z(z', z*)`` over matched branches. ``[B, K, D]``."""
    scale = (
        detached_target_scale(target_z, matched)
        if target_scale is None
        else target_scale.detach()
    )
    per_branch = scale_invariant_latent_error(pred_z, target_z, scale)
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
    *,
    target_scale: torch.Tensor | None = None,
    bind_negative_margin: float = 0.0,
    return_metrics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """Action-consequence binding.

    Runs the *target* action chunk through the dynamics model and requires it to land on
    the target successor latent. Without this, ``F`` can learn to ignore ``A`` entirely
    and predict ``z'`` from ``z`` and the branch token alone -- which would make every
    "executable future hypothesis" a fiction, since the actions would not be what
    produces the predicted consequence (spec section 6).
    """
    if bind_negative_margin < 0:
        raise ValueError("bind_negative_margin must be nonnegative")
    scale = (
        detached_target_scale(target_z, matched)
        if target_scale is None
        else target_scale.detach()
    )
    z_teacher = model.dynamics(
        z, target_action, target_mask, target_horizon_idx, branch_embedding
    )
    positive = scale_invariant_latent_error(z_teacher, target_z, scale)
    positive_loss = _masked_mean(positive, matched)

    margin_loss = positive_loss * 0.0
    negative_error = positive_loss * 0.0
    achieved_margin = positive_loss * 0.0
    eligible_anchors = matched.sum(-1) >= 2
    eligible = matched * eligible_anchors.unsqueeze(-1).to(matched.dtype)

    # A zero margin is the exact backward-compatible path: avoid a second dynamics
    # forward entirely.  With a positive margin, cyclically swap executable chunks
    # among the matched modes of each eligible anchor while retaining each receiving
    # branch embedding and target endpoint.
    if bind_negative_margin > 0 and bool(eligible_anchors.any()):
        b, k = matched.shape
        swap = torch.arange(k, device=matched.device).view(1, k).expand(b, k).clone()
        for row in torch.nonzero(eligible_anchors, as_tuple=False).flatten():
            positions = torch.nonzero(matched[row] > 0, as_tuple=False).flatten()
            swap[row, positions] = positions.roll(-1)

        def _swap_modes(value: torch.Tensor) -> torch.Tensor:
            index = swap.view(b, k, *([1] * (value.ndim - 2))).expand_as(value)
            return torch.gather(value, 1, index)

        z_swapped = model.dynamics(
            z,
            _swap_modes(target_action),
            _swap_modes(target_mask),
            _swap_modes(target_horizon_idx),
            branch_embedding,
        )
        negative = scale_invariant_latent_error(z_swapped, target_z, scale)
        negative_error = _masked_mean(negative, eligible)
        achieved_margin = _masked_mean(negative - positive, eligible)
        margin_loss = _masked_mean(
            F.relu(float(bind_negative_margin) + positive - negative), eligible
        )

    loss = positive_loss + margin_loss
    if not return_metrics:
        return loss
    metrics = {
        "bind/positive_error": float(positive_loss.detach().item()),
        "bind/negative_error": float(negative_error.detach().item()),
        "bind/negative_margin_loss": float(margin_loss.detach().item()),
        "bind/achieved_margin": float(achieved_margin.detach().item()),
        "bind/eligible_anchors": float(eligible_anchors.float().sum().item()),
        "bind/eligible_pairs": float(eligible.sum().item()),
        "bind/latent_target_scale": float(scale.detach().item()),
    }
    return loss, metrics


def uncertainty_loss(
    sigma: torch.Tensor,
    pred_z: torch.Tensor,
    target_z: torch.Tensor,
    matched: torch.Tensor,
    unmatched_quantile: float = 0.9,
    *,
    target_scale: torch.Tensor | None = None,
    balance_groups: bool = True,
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
    if not 0 <= unmatched_quantile <= 1:
        raise ValueError("unmatched_quantile must lie in [0, 1]")
    matched_bool = matched > 0
    if not bool(matched_bool.any()):
        # There is no supported error distribution from which to construct the high
        # target. Training on gathered slot-zero placeholders would be arbitrary.
        return sigma.sum() * 0.0
    scale = (
        detached_target_scale(target_z, matched)
        if target_scale is None
        else target_scale.detach()
    )
    with torch.no_grad():
        # float32 throughout: torch.quantile has no bf16 kernel, and this runs inside
        # an autocast region during training.
        err = scale_invariant_latent_error(pred_z, target_z, scale)  # [B, K]
        matched_err = err[matched_bool]
        high = torch.quantile(matched_err, unmatched_quantile)
        target = torch.where(matched_bool, err, high.expand_as(err))

    per = F.smooth_l1_loss(sigma.float(), target, reduction="none")
    matched_loss = per[matched_bool].mean()
    unmatched_bool = ~matched_bool
    if not bool(unmatched_bool.any()):
        return matched_loss
    unmatched_loss = per[unmatched_bool].mean()
    if balance_groups:
        # Equal group weight prevents K-1 unsupported branches from overwhelming the
        # one supported branch merely because the branch budget is larger.
        return 0.5 * (matched_loss + unmatched_loss)
    return per.mean()


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
    *,
    return_metrics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """Multi-level rollout stability.

    Applies the complete child predictor to the *predicted* successor latent and to the
    *encoded ground-truth* successor, then compares the child latents actually consumed
    by tree recursion plus their executable action chunks. The operator is reused at
    every depth, so if it only behaves on encoded latents the tree degrades with depth
    and "recursive prediction" buys nothing (spec section 26, stage 5).

    Subsampled to ``max_nodes`` successors: this is O(B * K) applications of a K-token
    transformer, i.e. O(B * K^2) overall, which at FlatKWM's K=256 tries to allocate tens
    of gigabytes. A random subset each step gives the same objective in expectation.
    """
    if max_nodes < 1:
        raise ValueError("max_nodes must be at least one")
    b, k, d = pred_z.shape
    flat_pred_all = pred_z.reshape(b * k, d)
    flat_tgt_all = target_z.reshape(b * k, d)
    flat_matched = matched.reshape(b * k)

    # Select from supported nodes directly. Sampling B*K first wastes about 75% of
    # formal K=4 batches and makes the effective batch depend on the number of modes.
    eligible = torch.nonzero(flat_matched > 0, as_tuple=False).flatten()
    candidate_nodes = int(eligible.numel())
    if candidate_nodes == 0:
        zero = pred_z.sum() * 0.0
        metrics = {
            "recursive/latent_component": 0.0,
            "recursive/action_component": 0.0,
            "recursive/matched_nodes": 0.0,
            "recursive/sampled_nodes": 0.0,
            "recursive/candidate_nodes": 0.0,
            "recursive/sampling_fraction": 0.0,
        }
        return (zero, metrics) if return_metrics else zero
    if candidate_nodes > max_nodes:
        order = torch.randperm(candidate_nodes, device=pred_z.device)[:max_nodes]
        eligible = eligible[order]

    flat_pred = flat_pred_all[eligible]
    flat_tgt = flat_tgt_all[eligible]
    if depth is not None:
        depth = depth.reshape(b * k)[eligible]

    n = flat_pred.shape[0]
    dep = (
        torch.ones(n, device=pred_z.device, dtype=torch.long)
        if depth is None
        else depth.reshape(n).long()
    )

    out_pred = model.predict_children(flat_pred, dep)
    with torch.no_grad():
        out_tgt = model.predict_children(flat_tgt, dep)

    # The tree recursively consumes predicted child *latents*, not intermediate branch
    # embeddings. Compare that actual recursive state in affine-invariant target units.
    # Action error averages action coordinates and valid target horizon steps, then K;
    # padded tails must not become training targets.
    child_scale = detached_target_scale(out_tgt["latent"])
    latent = scale_invariant_latent_error(
        out_pred["latent"], out_tgt["latent"], child_scale
    ).mean(-1)
    action_err = F.mse_loss(
        out_pred["branch"].action, out_tgt["branch"].action, reduction="none"
    ).mean(-1)
    action_mask = out_tgt["action_mask"].float()
    act_per_branch = (action_err * action_mask).sum(-1) / action_mask.sum(-1).clamp_min(1.0)
    act = act_per_branch.mean(-1)
    per = latent + act  # [N]
    loss = per.mean()
    if not return_metrics:
        return loss
    sampled_nodes = int(eligible.numel())
    metrics = {
        "recursive/latent_component": float(latent.mean().detach().item()),
        "recursive/action_component": float(act.mean().detach().item()),
        "recursive/latent_target_scale": float(child_scale.detach().item()),
        "recursive/matched_nodes": float(candidate_nodes),
        "recursive/sampled_nodes": float(sampled_nodes),
        "recursive/candidate_nodes": float(candidate_nodes),
        "recursive/sampling_fraction": float(sampled_nodes / candidate_nodes),
    }
    return loss, metrics


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

    # Compare the final *valid* target timestep, not padded slot H_max-1. Most formal
    # targets are shorter than H_max, so the historical last-slot metric mostly measured
    # agreement between two padding values.
    valid_lengths = target_mask.float().sum(-1).long()
    endpoint_weight = m * (valid_lengths > 0).to(m.dtype)
    endpoint_denom = endpoint_weight.sum().clamp_min(1.0)
    last = (valid_lengths - 1).clamp_min(0)
    gather_index = last.unsqueeze(-1).unsqueeze(-1).expand(
        *last.shape, 1, pred_action.shape[-1]
    )
    pred_last = torch.gather(pred_action, -2, gather_index).squeeze(-2)
    target_last = torch.gather(target_action, -2, gather_index).squeeze(-2)
    endpoint_consistency = (
        (pred_last - target_last).pow(2).mean(-1) * endpoint_weight
    ).sum() / endpoint_denom

    out = {
        "model/state_latent_mse": float(latent_mse.item()),
        "model/action_mse": float(act_mse.item()),
        "model/horizon_accuracy": float(acc.item()),
        "model/horizon_mae": float(mae.item()),
        "model/action_endpoint_consistency": float(endpoint_consistency.item()),
        "model/action_endpoint_consistency_count": float(endpoint_weight.sum().item()),
        "model/matched_fraction": float((m.sum() / m.numel()).item()),
    }
    target_prob = []
    for index, horizon in enumerate(horizon_values):
        target_fraction = (
            ((target_horizon_idx.long() == index).float() * m).sum() / denom
        )
        pred_fraction = ((pred_h == index).float() * m).sum() / denom
        target_prob.append(target_fraction)
        horizon_name = int(horizon.item())
        out[f"data/horizon_target_fraction_h{horizon_name}"] = float(target_fraction.item())
        out[f"model/horizon_pred_fraction_h{horizon_name}"] = float(pred_fraction.item())
    probability = torch.stack(target_prob)
    entropy = -(probability * probability.clamp_min(1e-8).log()).sum()
    normalizer = torch.log(probability.new_tensor(max(len(target_prob), 1))).clamp_min(1.0)
    out["data/horizon_target_entropy"] = float(entropy.item())
    out["data/horizon_target_normalized_entropy"] = float((entropy / normalizer).item())
    out["data/horizon_target_occupied_classes"] = float((probability > 0).sum().item())
    return out
