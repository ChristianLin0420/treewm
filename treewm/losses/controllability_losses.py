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


def chamfer_set_distance(a: torch.Tensor, b: torch.Tensor, a_valid: torch.Tensor, b_valid: torch.Tensor) -> torch.Tensor:
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
) -> tuple[torch.Tensor, dict[str, float]]:
    """Regress ``d_q`` onto the data-derived future-set distance.

    A metric-matching objective rather than a contrastive one: it supervises the whole
    distance profile instead of a positive/negative split, which avoids having to pick a
    similarity threshold that would itself encode the answer.
    """
    b = q.shape[0]
    with torch.set_grad_enabled(not detach_target):
        d_future = chamfer_set_distance(endpoints, endpoints, endpoint_valid, endpoint_valid)
    if detach_target:
        d_future = d_future.detach()

    d_q = q_distance(q.unsqueeze(1).expand(b, b, *q.shape[1:]), q.unsqueeze(0).expand(b, b, *q.shape[1:]))

    off_diag = ~torch.eye(b, device=q.device, dtype=torch.bool)
    target = (d_future * scale)[off_diag]
    pred = d_q[off_diag]
    loss = F.mse_loss(pred, target)

    with torch.no_grad():
        median = target.median()
        pos = pred[target <= median].mean() if (target <= median).any() else pred.mean() * 0
        neg = pred[target > median].mean() if (target > median).any() else pred.mean() * 0
    return loss, {
        "model/q_positive_distance": float(pos.item()),
        "model/q_negative_distance": float(neg.item()),
        "control/q_same_future_distance": float(pos.item()),
        "control/q_different_future_distance": float(neg.item()),
        "control/separation_ratio": float((neg / pos.clamp_min(1e-6)).item()),
    }


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
