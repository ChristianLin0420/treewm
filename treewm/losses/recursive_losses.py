"""Track A -- recursive robustness.

Recursion is the strongest positive result in the project, so this trains it directly
rather than relying on single-edge supervision to generalise.

A1 **multi-step supervision**: roll the operator forward along the anchor's own
trajectory, ``z_t -> z_{t+h} -> z_{t+2h} -> z_{t+3h}``, and supervise every predicted
latent against the encoded true state at that time. Single-edge training only ever sees
depth 1, which is exactly the regime where the measured error was small (0.1 units) --
the failures live at depth 4+ (1.3 -> 4+ units).

A2 **scheduled sampling**: with probability ``p`` feed the model its own predicted latent
instead of the encoded truth, so it is trained on the input distribution it actually
faces during tree expansion. ``p`` warms up from 0 to avoid destabilising early training.

Losses are reported per depth so "does depth-2 improve while depth-3 degrades" is
answerable rather than hidden inside one scalar.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def multi_step_recursive_loss(
    model,
    batch: dict[str, torch.Tensor],
    scheduled_sampling_p: float = 0.0,
    depth_weights: tuple[float, ...] | None = None,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Chained latent prediction along the anchor's own future.

    Returns ``(loss, metrics)``. Metrics include per-depth state error, per-depth loss and
    the divergence between the self-fed and teacher-fed latents.
    """
    obs = batch["obs"]
    ms_actions = batch["ms_actions"]  # [B, D, h_max, dA]
    ms_mask = batch["ms_action_mask"]  # [B, D, h_max]
    ms_obs = batch["ms_obs"]  # [B, D, obs_dim]
    ms_h = batch["ms_horizon_idx"]  # [B, D]
    ms_valid = batch["ms_valid"]  # [B, D]

    b, d_max = ms_valid.shape
    z = model.encode(obs)
    with torch.no_grad():
        z_true_all = model.encode(ms_obs)  # [B, D, z_dim]

    total = z.sum() * 0.0
    metrics: dict[str, float] = {}
    weight_sum = 0.0
    z_cur = z
    z_teacher = z

    for d in range(d_max):
        valid = ms_valid[:, d]
        if float(valid.sum()) == 0:
            break

        act = ms_actions[:, d]  # [B, h_max, dA]
        mask = ms_mask[:, d]
        h_idx = ms_h[:, d]

        # Pick the branch whose proposed action best matches the executed one, then push
        # the *true* action through the dynamics with that branch's embedding. This keeps
        # the rollout consistent with how the bind loss defines an executable branch.
        out = model.branch(z_cur)
        with torch.no_grad():
            diff = (out.action - act.unsqueeze(1)).pow(2).mean((-1, -2))  # [B, K]
            pick = diff.argmin(dim=1)
        emb = out.embedding[torch.arange(b, device=z.device), pick]  # [B, H]

        z_next = model.dynamics(
            z_cur, act.unsqueeze(1), mask.unsqueeze(1), h_idx.unsqueeze(1), emb.unsqueeze(1)
        ).squeeze(1)

        target = z_true_all[:, d]
        per = F.mse_loss(z_next, target, reduction="none").mean(-1)
        w = float(depth_weights[d]) if depth_weights is not None and d < len(depth_weights) else 1.0
        step_loss = (per * valid).sum() / valid.sum().clamp_min(1.0)
        total = total + w * step_loss
        weight_sum += w

        with torch.no_grad():
            metrics[f"recursive/loss_depth{d + 1}"] = float(step_loss.item())
            metrics[f"recursive/state_error_depth{d + 1}"] = float(
                ((z_next - target).pow(2).mean(-1) * valid).sum().item()
                / max(float(valid.sum().item()), 1.0)
            )

        # Teacher-forced reference chain, for the drift diagnostic.
        with torch.no_grad():
            out_t = model.branch(z_teacher)
            diff_t = (out_t.action - act.unsqueeze(1)).pow(2).mean((-1, -2))
            emb_t = out_t.embedding[torch.arange(b, device=z.device), diff_t.argmin(dim=1)]
            z_teacher_next = model.dynamics(
                z_teacher, act.unsqueeze(1), mask.unsqueeze(1), h_idx.unsqueeze(1), emb_t.unsqueeze(1)
            ).squeeze(1)
            metrics[f"recursive/self_vs_teacher_depth{d + 1}"] = float(
                (z_next - z_teacher_next).pow(2).mean().item()
            )
        z_teacher = torch.where(valid.unsqueeze(-1) > 0, z_true_all[:, d], z_teacher)

        # Scheduled sampling: feed our own prediction some of the time.
        if scheduled_sampling_p > 0.0:
            u = torch.rand(b, 1, device=z.device, generator=generator)
            use_pred = (u < scheduled_sampling_p).float()
            nxt = use_pred * z_next.detach() + (1.0 - use_pred) * z_true_all[:, d]
        else:
            nxt = z_true_all[:, d]
        z_cur = torch.where(valid.unsqueeze(-1) > 0, nxt, z_cur)

    if weight_sum > 0:
        total = total / weight_sum
    metrics["recursive/scheduled_sampling_p"] = float(scheduled_sampling_p)
    metrics["recursive/mean_chain_depth"] = float(ms_valid.sum(1).float().mean().item())
    return total, metrics


def scheduled_sampling_schedule(step: int, final_p: float, warmup: int) -> float:
    """Linear warm-up from 0 to ``final_p`` over ``warmup`` steps."""
    if final_p <= 0.0:
        return 0.0
    if warmup <= 0:
        return final_p
    return float(min(final_p, final_p * step / warmup))
