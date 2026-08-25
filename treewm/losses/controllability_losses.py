"""Training the controllability encoder ``q = C(z)``, plus the expansion-gain loss.

Two targets are supported and can be compared directly (spec section 15):

  ``future_set``   States are close in q iff their *retrieved future sets* are close.
                   The set-to-set distance is a symmetric Chamfer distance between
                   endpoint clouds -- data-derived, available at step 1, no bootstrap.

  ``bootstrap``    States are close in q iff the model's own deeper expansion yields a
                   similar tree signature. Self-referential, so it is off by default;
                   it exists so the two can be compared rather than assumed.

An important caveat that shapes how the result may be read: ``q = C(z)`` is a
deterministic function of ``z``, so q *cannot* carry more information about the future
than z does. Any q-beats-z result is a statement about metric geometry, which is why the
diagnostics compare ``d_q`` against both ``d_z`` and a frozen random projection of z
down to ``q_dim``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def chamfer_set_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    a_valid: torch.Tensor,
    b_valid: torch.Tensor,
    *,
    rms: bool = False,
) -> torch.Tensor:
    """Symmetric Chamfer distance between point sets. ``[B, M, D]`` -> ``[B, B]``.

    Compares every anchor's future-endpoint cloud with every other anchor's in the
    batch, which is the data-side notion of "these two states have the same options".
    """
    bsz, m, d = a.shape
    n = b.shape[1]
    # [B, 1, M, D] vs [1, B, N, D] -> [B, B, M, N]
    a_exp = a.unsqueeze(1).expand(bsz, bsz, m, d).reshape(bsz * bsz, m, d)
    b_exp = b.unsqueeze(0).expand(bsz, bsz, n, d).reshape(bsz * bsz, n, d)
    dist = torch.cdist(a_exp, b_exp).view(bsz, bsz, m, n)
    if rms:
        # Euclidean distance grows as sqrt(D) even when every coordinate has the
        # same physical/statistical scale.  RMS units make the future-set target
        # invariant to duplicating equivalent task-metric coordinates.
        dist = dist / float(max(d, 1)) ** 0.5

    av = (a_valid > 0).view(bsz, 1, m, 1)
    bv = (b_valid > 0).view(1, bsz, 1, n)
    big = torch.finfo(dist.dtype).max / 4
    masked = dist.masked_fill(~(av & bv), big)

    a_to_b = masked.min(-1).values  # [B, B, M]
    b_to_a = masked.min(-2).values  # [B, B, N]
    a_w = av.squeeze(-1).float()
    b_w = bv.squeeze(-2).float()
    a_term = (a_to_b.clamp(max=big / 2) * a_w).sum(-1) / a_w.sum(-1).clamp_min(1.0)
    b_term = (b_to_a.clamp(max=big / 2) * b_w).sum(-1) / b_w.sum(-1).clamp_min(1.0)
    return 0.5 * (a_term + b_term)


def future_set_distance_loss(
    q: torch.Tensor,
    endpoints: torch.Tensor,
    endpoint_valid: torch.Tensor,
    q_distance,
    scale: float = 1.0,
    detach_target: bool = True,
    target_transform: str = "linear",
    metric_weight: float = 1.0,
    rank_weight: float = 0.0,
    rank_temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Regress ``d_q`` onto the data-derived future-set distance.

    A metric-matching objective rather than a contrastive one: it supervises the whole
    distance profile instead of a positive/negative split, which avoids having to pick a
    similarity threshold that would itself encode the answer.
    """
    if target_transform not in {"linear", "rms_tanh"}:
        raise ValueError(
            f"unknown control target transform {target_transform!r}; "
            "options: linear | rms_tanh"
        )
    if metric_weight < 0 or rank_weight < 0:
        raise ValueError("control metric/rank weights must be non-negative")
    if rank_temperature <= 0:
        raise ValueError("control rank temperature must be positive")

    b = q.shape[0]
    if endpoint_valid.shape[:2] != endpoints.shape[:2]:
        raise ValueError("endpoint_valid must match endpoints [B, M]")
    if endpoints.dim() != 3 or endpoints.shape[-1] < 1:
        raise ValueError("control endpoints must have shape [B, M, D_metric], D_metric >= 1")
    if not bool((endpoint_valid > 0).any(dim=1).all()):
        raise ValueError("every control anchor needs at least one valid future endpoint")

    with torch.set_grad_enabled(not detach_target):
        d_future = chamfer_set_distance(
            endpoints,
            endpoints,
            endpoint_valid,
            endpoint_valid,
            rms=target_transform == "rms_tanh",
        )
        if target_transform == "rms_tanh":
            # q is L2-normalised, hence d_q is bounded in [0, 2].  This monotonic
            # map retains future-set ordering while making every calibration target
            # feasible in exactly the same range.
            target_matrix = 2.0 * torch.tanh(d_future / 2.0)
        else:
            # Exact v1 behaviour, retained for old configs/checkpoints.
            target_matrix = d_future * scale
    if detach_target:
        d_future = d_future.detach()
        target_matrix = target_matrix.detach()

    d_q = q_distance(
        q.unsqueeze(1).expand(b, b, *q.shape[1:]),
        q.unsqueeze(0).expand(b, b, *q.shape[1:]),
    ).float()

    off_diag = ~torch.eye(b, device=q.device, dtype=torch.bool)
    target = target_matrix.float()[off_diag]
    pred = d_q[off_diag]
    if pred.numel() == 0:
        metric_loss = q.sum() * 0.0
        rank_loss = q.sum() * 0.0
    else:
        metric_loss = (
            F.smooth_l1_loss(pred, target)
            if target_transform == "rms_tanh"
            else F.mse_loss(pred, target)
        )

        # Per-anchor neighbour distributions prevent the O(B^2) far pairs from
        # overwhelming the local geometry.  The data distribution is detached;
        # gradients flow only through q distances.
        diagonal = torch.eye(b, device=q.device, dtype=torch.bool)
        target_logits = (-target_matrix.float() / rank_temperature).masked_fill(diagonal, -1e9)
        pred_logits = (-d_q / rank_temperature).masked_fill(diagonal, -1e9)
        target_prob = torch.softmax(target_logits, dim=1).detach()
        pred_log_prob = torch.log_softmax(pred_logits, dim=1)
        rank_loss = F.kl_div(pred_log_prob, target_prob, reduction="batchmean")

    loss = metric_weight * metric_loss + rank_weight * rank_loss

    with torch.no_grad():
        if target.numel():
            median = target.median()
            pos = pred[target <= median].mean() if (target <= median).any() else pred.mean() * 0
            neg = pred[target > median].mean() if (target > median).any() else pred.mean() * 0
            raw = d_future.float()[off_diag]
            quantiles = torch.tensor((0.5, 0.9, 0.99), device=target.device)
            raw_q = torch.quantile(raw, quantiles)
            target_q = torch.quantile(target, quantiles)
            target_max = target.max()
            target_mean = target.mean()
            target_std = target.std(unbiased=False)
            q_mean = pred.mean()
            q_std = pred.std(unbiased=False)
            q_near_zero = (pred < 0.05).float().mean()
            saturation = (target >= 1.9).float().mean()
        else:
            zero = d_q.sum() * 0.0
            pos = neg = zero
            raw_q = target_q = torch.zeros(3, device=q.device)
            target_max = target_mean = target_std = zero
            q_mean = q_std = q_near_zero = saturation = zero

        if b > 1:
            k = min(5, b - 1)
            diagonal = torch.eye(b, device=q.device, dtype=torch.bool)
            future_order = target_matrix.float().masked_fill(diagonal, float("inf"))
            q_order = d_q.masked_fill(diagonal, float("inf"))
            true_nn = future_order.topk(k, dim=1, largest=False).indices
            pred_nn = q_order.topk(k, dim=1, largest=False).indices
            agreement = (
                (pred_nn.unsqueeze(-1) == true_nn.unsqueeze(-2)).any(-1).float().mean()
            )
            top1 = (pred_nn[:, 0] == true_nn[:, 0]).float().mean()

            order_scores: list[torch.Tensor] = []
            for row in range(b):
                mask = ~diagonal[row]
                t_row = target_matrix[row, mask].float()
                p_row = d_q[row, mask]
                if t_row.numel() < 2:
                    continue
                t_delta = t_row.unsqueeze(1) - t_row.unsqueeze(0)
                p_delta = p_row.unsqueeze(1) - p_row.unsqueeze(0)
                comparable = t_delta.abs() > 1e-6
                if bool(comparable.any()):
                    order_scores.append(
                        ((t_delta * p_delta) > 0)[comparable].float().mean()
                    )
            order_accuracy = (
                torch.stack(order_scores).mean() if order_scores else d_q.sum() * 0.0
            )
        else:
            agreement = top1 = order_accuracy = d_q.sum() * 0.0

    metrics = {
        "model/q_positive_distance": float(pos.item()),
        "model/q_negative_distance": float(neg.item()),
        "control/q_same_future_distance": float(pos.item()),
        "control/q_different_future_distance": float(neg.item()),
        "control/separation_ratio": float((neg / pos.clamp_min(1e-6)).item()),
        "control/loss_metric": float(metric_loss.detach().item()),
        "control/loss_rank": float(rank_loss.detach().item()),
        "control/effective_metric": float((metric_weight * metric_loss).detach().item()),
        "control/effective_rank": float((rank_weight * rank_loss).detach().item()),
        "control/raw_rms_p50": float(raw_q[0].item()),
        "control/raw_rms_p90": float(raw_q[1].item()),
        "control/raw_rms_p99": float(raw_q[2].item()),
        "control/target_p50": float(target_q[0].item()),
        "control/target_p90": float(target_q[1].item()),
        "control/target_p99": float(target_q[2].item()),
        "control/target_max": float(target_max.item()),
        "control/target_mean": float(target_mean.item()),
        "control/target_std": float(target_std.item()),
        "control/target_saturation_fraction": float(saturation.item()),
        "control/q_pair_distance_mean": float(q_mean.item()),
        "control/q_pair_distance_std": float(q_std.item()),
        "control/q_near_collapse_fraction": float(q_near_zero.item()),
        "control/neighbor_top1_agreement": float(top1.item()),
        "control/neighbor_topk_agreement": float(agreement.item()),
        "control/neighbor_pairwise_order_accuracy": float(order_accuracy.item()),
        "control/target_transform_rms_tanh": float(target_transform == "rms_tanh"),
    }
    return loss, metrics


def future_set_contrastive_loss(
    q: torch.Tensor,
    endpoints: torch.Tensor,
    endpoint_valid: torch.Tensor,
    q_distance,
    temperature: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """InfoNCE variant: the nearest future-set in the batch is the positive."""
    b = q.shape[0]
    with torch.no_grad():
        d_future = chamfer_set_distance(endpoints, endpoints, endpoint_valid, endpoint_valid)
        d_future.fill_diagonal_(float("inf"))
        positive = d_future.argmin(dim=1)

    d_q = q_distance(q.unsqueeze(1).expand(b, b, *q.shape[1:]), q.unsqueeze(0).expand(b, b, *q.shape[1:]))
    logits = -d_q / temperature
    logits.fill_diagonal_(-1e9)
    loss = F.cross_entropy(logits, positive)

    with torch.no_grad():
        pos_d = d_q.gather(1, positive.view(-1, 1)).mean()
        mask = ~torch.eye(b, device=q.device, dtype=torch.bool)
        neg_d = d_q[mask].mean()
    return loss, {
        "model/q_positive_distance": float(pos_d.item()),
        "model/q_negative_distance": float(neg_d.item()),
        "control/q_same_future_distance": float(pos_d.item()),
        "control/q_different_future_distance": float(neg_d.item()),
        "control/separation_ratio": float((neg_d / pos_d.clamp_min(1e-6)).item()),
    }


def bootstrap_signature_loss(
    signature_module,
    q: torch.Tensor,
    child_q: torch.Tensor,
    child_valid: torch.Tensor,
) -> torch.Tensor:
    """Option 1: q must predict the model's own subtree signature.

    Self-referential by construction, hence disabled by default.
    """
    target = signature_module(child_q.detach(), child_valid.detach())
    pred = q.flatten(1)
    if pred.shape[-1] != target.shape[-1]:
        pred = pred[..., : target.shape[-1]]
    return F.mse_loss(pred, target)


def expansion_gain_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
) -> torch.Tensor:
    """Regress predicted marginal coverage gain onto the data-grounded target."""
    err = F.smooth_l1_loss(predicted, target, reduction="none")
    w = (valid > 0).float()
    return (err * w).sum() / w.sum().clamp_min(1.0)


@torch.no_grad()
def retrieval_precision(
    embedding: torch.Tensor,
    endpoints: torch.Tensor,
    endpoint_valid: torch.Tensor,
    distance_fn,
    k: int = 5,
) -> float:
    """How often are an embedding's k nearest neighbours also its future-set neighbours?

    The dimension-matched q-vs-z diagnostic runs this three times -- with ``d_q``, with
    ``d_z``, and with a frozen random projection of z to ``q_dim`` -- so that "q wins"
    cannot simply mean "fewer dimensions retrieve better".
    """
    b = embedding.shape[0]
    if b <= k + 1:
        return 0.0
    d_future = chamfer_set_distance(endpoints, endpoints, endpoint_valid, endpoint_valid)
    d_future.fill_diagonal_(float("inf"))
    true_nn = d_future.topk(k, dim=1, largest=False).indices

    d_emb = distance_fn(embedding)
    d_emb.fill_diagonal_(float("inf"))
    pred_nn = d_emb.topk(k, dim=1, largest=False).indices

    hits = 0.0
    for i in range(b):
        hits += len(set(true_nn[i].tolist()) & set(pred_nn[i].tolist()))
    return hits / (b * k)
