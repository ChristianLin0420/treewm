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

import math

import torch
import torch.nn.functional as F

from treewm.losses.world_losses import detached_target_scale, scale_invariant_latent_error


def _masked_action_distance(
    predicted: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Per-branch action MSE over valid time and action coordinates.

    ``predicted`` is ``[B,K,H,A]`` while target/mask are ``[B,H,A]`` and ``[B,H]``.
    Averaging action coordinates before the time reduction makes the distance invariant
    to both padded horizon length and action dimensionality.
    """
    per_step = (predicted.float() - target.float().unsqueeze(1)).pow(2).mean(-1)
    weight = mask.float().unsqueeze(1)
    return (per_step * weight).sum(-1) / weight.sum(-1).clamp_min(1.0)


def _validated_task_metric_dims(
    batch: dict[str, torch.Tensor], obs_dim: int
) -> torch.Tensor:
    """Return the one task-coordinate vector shared by every batch item."""
    if "task_metric_dims" not in batch:
        raise ValueError(
            "grounded decoded recursive terms require batch task_metric_dims"
        )
    batched_dims = batch["task_metric_dims"]
    if (
        batched_dims.ndim != 2
        or batched_dims.shape[0] != batch["obs"].shape[0]
        or batched_dims.shape[1] == 0
    ):
        raise ValueError("task_metric_dims must have shape [B, nonzero_dims]")
    if batched_dims.is_floating_point() and not bool(
        (batched_dims == batched_dims.long()).all()
    ):
        raise ValueError("task_metric_dims must contain integer coordinates")
    dims = batched_dims[0].long()
    if not bool((batched_dims.long() == dims.unsqueeze(0)).all()):
        raise ValueError("task_metric_dims must be identical within a batch")
    if bool((dims < 0).any()) or bool((dims >= int(obs_dim)).any()):
        raise ValueError("task_metric_dims are outside the decoded observation width")
    if int(torch.unique(dims).numel()) != int(dims.numel()):
        raise ValueError("task_metric_dims must not contain duplicates")
    return dims


def _decoded_task_endpoint_rms(
    decoded: torch.Tensor,
    target_obs: torch.Tensor,
    task_metric_dims: torch.Tensor,
) -> torch.Tensor:
    """RMS endpoint error in normalized task coordinates, excluding nuisance dims."""
    predicted_metric = decoded.float().index_select(-1, task_metric_dims)
    target_metric = target_obs.float().index_select(-1, task_metric_dims)
    while target_metric.ndim < predicted_metric.ndim:
        target_metric = target_metric.unsqueeze(1)
    difference = predicted_metric - target_metric
    return torch.linalg.vector_norm(difference, dim=-1) / math.sqrt(
        int(task_metric_dims.numel())
    )


def _grounded_execution_candidates(
    model,
    z_parent: torch.Tensor,
    parent_depth: torch.Tensor,
    target_action: torch.Tensor,
    target_mask: torch.Tensor,
    target_horizon_idx: torch.Tensor,
    target_obs: torch.Tensor,
    task_metric_dims: torch.Tensor | None,
    *,
    selection_action_weight: float,
    selection_endpoint_weight: float,
    selection_horizon_weight: float,
) -> tuple[object, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Build predicted-action successors and deterministically select one branch.

    The ground-truth mask/horizon define which logged endpoint is available; actions
    always come from each branch head. Selection combines normalized action RMS,
    decoded task-coordinate endpoint RMS, and bounded horizon error ``1-p(h*)``. It is
    deliberately non-differentiable and first-index tie stable. Gradients are applied
    later only to the selected branch, whose supervised horizon term remains CE.
    """
    branch = model.branch(z_parent, parent_depth)
    batch_size, branch_factor = branch.action.shape[:2]
    expanded_mask = target_mask.unsqueeze(1).expand(
        batch_size, branch_factor, target_mask.shape[-1]
    )
    expanded_horizon = target_horizon_idx.unsqueeze(1).expand(
        batch_size, branch_factor
    )
    successors = model.dynamics(
        z_parent,
        branch.action,
        expanded_mask,
        expanded_horizon,
        branch.embedding,
    )

    with torch.no_grad():
        action_rms = _masked_action_distance(
            branch.action.detach(), target_action, target_mask
        ).clamp_min(0.0).sqrt()
        horizon_logits = branch.horizon_logits.detach().float().reshape(
            batch_size * branch_factor, -1
        )
        horizon_target = expanded_horizon.reshape(-1).long()
        horizon_ce = F.cross_entropy(
            horizon_logits,
            horizon_target,
            reduction="none",
        ).view(batch_size, branch_factor)
        horizon_target_probability = torch.softmax(horizon_logits, dim=-1).gather(
            1, horizon_target.unsqueeze(1)
        )
        horizon_error = (1.0 - horizon_target_probability).view(
            batch_size, branch_factor
        )
        if selection_endpoint_weight > 0.0:
            assert task_metric_dims is not None
            decoded = model.decoder(successors.detach())
            endpoint_rms = _decoded_task_endpoint_rms(
                decoded, target_obs, task_metric_dims
            )
        else:
            endpoint_rms = torch.zeros_like(action_rms)
        composite = (
            float(selection_action_weight) * action_rms
            + float(selection_endpoint_weight) * endpoint_rms
            + float(selection_horizon_weight) * horizon_error
        )
        # torch.argmin returns the first minimum, making exact ties branch-index stable.
        selected = composite.argmin(dim=1)
    return branch, successors, selected, {
        "action_rms": action_rms,
        "endpoint_rms": endpoint_rms,
        "horizon_error": horizon_error,
        "horizon_ce": horizon_ce,
        "composite": composite,
    }


def multi_step_recursive_loss(
    model,
    batch: dict[str, torch.Tensor],
    scheduled_sampling_p: float = 0.0,
    scheduled_sampling_granularity: str = "step",
    depth_weights: tuple[float, ...] | None = None,
    generator: torch.Generator | None = None,
    transition_mode: str = "teacher_action",
    grounded_select_action_weight: float = 0.0,
    grounded_select_endpoint_weight: float = 0.0,
    grounded_select_horizon_weight: float = 0.0,
    grounded_loss_latent_weight: float = 0.0,
    grounded_loss_action_weight: float = 0.0,
    grounded_loss_horizon_weight: float = 0.0,
    grounded_loss_endpoint_weight: float = 0.0,
    grounded_detach_self_fed_parent: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Chained latent prediction along the anchor's own future.

    Returns ``(loss, metrics)``. Metrics include per-depth state error, per-depth loss and
    the divergence between the self-fed and teacher-fed latents.
    """
    if scheduled_sampling_granularity not in {"step", "sequence"}:
        raise ValueError(
            "scheduled_sampling_granularity must be 'step' or 'sequence'"
        )
    if transition_mode not in {"teacher_action", "grounded_execution_v2"}:
        raise ValueError(
            "transition_mode must be 'teacher_action' or 'grounded_execution_v2'"
        )

    grounded = transition_mode == "grounded_execution_v2"
    grounded_weights = {
        "grounded_select_action_weight": grounded_select_action_weight,
        "grounded_select_endpoint_weight": grounded_select_endpoint_weight,
        "grounded_select_horizon_weight": grounded_select_horizon_weight,
        "grounded_loss_latent_weight": grounded_loss_latent_weight,
        "grounded_loss_action_weight": grounded_loss_action_weight,
        "grounded_loss_horizon_weight": grounded_loss_horizon_weight,
        "grounded_loss_endpoint_weight": grounded_loss_endpoint_weight,
    }
    task_metric_dims = None
    decoded_terms_active = False
    if grounded:
        invalid_weights = [
            name
            for name, value in grounded_weights.items()
            if not math.isfinite(float(value)) or float(value) < 0.0
        ]
        if invalid_weights:
            raise ValueError(
                "grounded recursive weights must be finite and nonnegative: "
                + ", ".join(invalid_weights)
            )
        if (
            float(grounded_select_action_weight)
            + float(grounded_select_endpoint_weight)
            + float(grounded_select_horizon_weight)
            <= 0.0
        ):
            raise ValueError("grounded recursive branch selection requires a positive weight")
        if (
            float(grounded_loss_latent_weight)
            + float(grounded_loss_action_weight)
            + float(grounded_loss_horizon_weight)
            + float(grounded_loss_endpoint_weight)
            <= 0.0
        ):
            raise ValueError("grounded recursive objective requires a positive loss weight")
        decoded_terms_active = (
            float(grounded_select_endpoint_weight) > 0.0
            or float(grounded_loss_endpoint_weight) > 0.0
        )
        if decoded_terms_active and getattr(model, "decoder", None) is None:
            raise ValueError("grounded decoded recursive terms require model.decoder")

    obs = batch["obs"]
    ms_actions = batch["ms_actions"]  # [B, D, h_max, dA]
    ms_mask = batch["ms_action_mask"]  # [B, D, h_max]
    ms_obs = batch["ms_obs"]  # [B, D, obs_dim]
    ms_h = batch["ms_horizon_idx"]  # [B, D]
    ms_valid = batch["ms_valid"]  # [B, D]

    if decoded_terms_active:
        task_metric_dims = _validated_task_metric_dims(batch, int(obs.shape[-1]))

    b, d_max = ms_valid.shape
    z = model.encode(obs)
    with torch.no_grad():
        # The target encoder is a teacher for this objective. Detaching is explicit even
        # though no_grad already enforces it, making the contract robust to refactors.
        z_true_all = model.encode(ms_obs).detach()  # [B, D, z_dim]
    target_scale = detached_target_scale(z_true_all, ms_valid)

    total = z.sum() * 0.0
    metrics: dict[str, float] = {}
    weighted_valid_count = z.new_tensor(0.0, dtype=torch.float32)
    z_cur = z
    z_teacher = z
    # Sequence-level sampling deliberately reuses one decision for the complete chain.
    # At p=.25 and depth three this exposes 25% of examples to a fully self-fed chain,
    # versus p**2=6.25% under independent step-level decisions.  The historical
    # step-level path remains the default and retains its exact RNG draw locations.
    sequence_use_pred = None
    if scheduled_sampling_p > 0.0 and scheduled_sampling_granularity == "sequence":
        sequence_use_pred = (
            torch.rand(b, 1, device=z.device, generator=generator)
            < scheduled_sampling_p
        ).float()
    feed_decisions: list[torch.Tensor] = []
    grounded_component_sums = (
        {
            name: z.new_tensor(0.0, dtype=torch.float32)
            for name in ("latent", "action", "horizon", "endpoint")
        }
        if grounded
        else None
    )

    for d in range(d_max):
        valid = ms_valid[:, d]
        if float(valid.sum()) == 0:
            # Validity is applied independently at every depth. Builder-produced chains
            # are prefixes, but accepting holes keeps this loss correct for adapters and
            # makes the mask contract testable.
            if d + 1 < d_max:
                feed_decisions.append(torch.zeros(b, device=z.device))
            continue

        act = ms_actions[:, d]  # [B, h_max, dA]
        mask = ms_mask[:, d]
        h_idx = ms_h[:, d]

        parent_depth = torch.full(
            (b,), d, device=z.device, dtype=torch.long
        )
        target = z_true_all[:, d]
        if not grounded:
            # Pick the branch whose proposed action best matches the executed one, then
            # push the *true* action through dynamics with that branch's embedding. This
            # is the historical objective and deliberately remains operation-for-operation
            # unchanged behind the default transition mode.
            out = model.branch(z_cur, parent_depth)
            with torch.no_grad():
                diff = _masked_action_distance(out.action, act, mask)  # [B, K]
                pick = diff.argmin(dim=1)
            emb = out.embedding[torch.arange(b, device=z.device), pick]  # [B, H]

            z_next = model.dynamics(
                z_cur, act.unsqueeze(1), mask.unsqueeze(1), h_idx.unsqueeze(1), emb.unsqueeze(1)
            ).squeeze(1)
            per = scale_invariant_latent_error(z_next, target, target_scale)
            grounded_components = None
            grounded_selection = None
        else:
            out, successors, pick, grounded_selection = _grounded_execution_candidates(
                model,
                z_cur,
                parent_depth,
                act,
                mask,
                h_idx,
                ms_obs[:, d],
                task_metric_dims,
                selection_action_weight=float(grounded_select_action_weight),
                selection_endpoint_weight=float(grounded_select_endpoint_weight),
                selection_horizon_weight=float(grounded_select_horizon_weight),
            )
            rows = torch.arange(b, device=z.device)
            z_next = successors[rows, pick]
            selected_action = out.action[rows, pick]
            selected_horizon_logits = out.horizon_logits[rows, pick]
            latent_per = scale_invariant_latent_error(z_next, target, target_scale)
            action_per = _masked_action_distance(
                selected_action.unsqueeze(1), act, mask
            ).squeeze(1)
            horizon_per = F.cross_entropy(
                selected_horizon_logits.float(), h_idx.long(), reduction="none"
            )
            if float(grounded_loss_endpoint_weight) > 0.0:
                assert task_metric_dims is not None
                endpoint_per = _decoded_task_endpoint_rms(
                    model.decoder(z_next), ms_obs[:, d], task_metric_dims
                )
            elif float(grounded_select_endpoint_weight) > 0.0:
                endpoint_per = grounded_selection["endpoint_rms"][rows, pick]
            else:
                endpoint_per = torch.zeros_like(latent_per)
            grounded_components = {
                "latent": latent_per,
                "action": action_per,
                "horizon": horizon_per,
                "endpoint": endpoint_per,
            }
            weighted_components = [
                float(weight) * grounded_components[name]
                for name, weight in (
                    ("latent", grounded_loss_latent_weight),
                    ("action", grounded_loss_action_weight),
                    ("horizon", grounded_loss_horizon_weight),
                    ("endpoint", grounded_loss_endpoint_weight),
                )
                if float(weight) > 0.0
            ]
            # Configuration validation guarantees at least one active component.
            per = sum(weighted_components[1:], weighted_components[0])

        w = float(depth_weights[d]) if depth_weights is not None and d < len(depth_weights) else 1.0
        step_loss = (per * valid).sum() / valid.sum().clamp_min(1.0)
        # Reduce over valid (example, depth) transitions rather than first averaging
        # each depth equally. Sparse late depths therefore do not get accidental extra
        # weight; explicit depth_weights remain the sole depth reweighting mechanism.
        # This denominator is rank-local. Current campaigns assign one rank per model;
        # a future multi-rank-per-model recipe must globally reduce valid counts before
        # claiming an exact global transition mean.
        total = total + w * (per * valid).sum()
        weighted_valid_count = weighted_valid_count + w * valid.sum()

        if grounded:
            assert grounded_components is not None
            assert grounded_component_sums is not None
            assert grounded_selection is not None
            for name, component in grounded_components.items():
                grounded_component_sums[name] = (
                    grounded_component_sums[name] + w * (component * valid).sum()
                )

        with torch.no_grad():
            metrics[f"recursive/loss_depth{d + 1}"] = float(step_loss.item())
            metrics[f"recursive/state_error_depth{d + 1}"] = float(
                ((z_next - target).pow(2).mean(-1) * valid).sum().item()
                / max(float(valid.sum().item()), 1.0)
            )
            if grounded:
                rows = torch.arange(b, device=z.device)
                valid_denominator = max(float(valid.sum().item()), 1.0)
                for name, component in grounded_components.items():
                    metrics[f"recursive/grounded/loss_{name}_depth{d + 1}"] = float(
                        (component * valid).sum().item() / valid_denominator
                    )
                for name, component in grounded_selection.items():
                    selected_component = component[rows, pick]
                    metrics[
                        f"recursive/grounded/selection_{name}_depth{d + 1}"
                    ] = float(
                        (selected_component * valid).sum().item() / valid_denominator
                    )
                selected_horizon = out.horizon_logits[rows, pick].argmax(dim=-1)
                metrics[
                    f"recursive/grounded/selection_horizon_accuracy_depth{d + 1}"
                ] = float(
                    ((selected_horizon == h_idx).float() * valid).sum().item()
                    / valid_denominator
                )
                metrics[
                    f"recursive/grounded/selected_branch_mean_depth{d + 1}"
                ] = float((pick.float() * valid).sum().item() / valid_denominator)

        # Teacher-forced reference chain, for the drift diagnostic.
        with torch.no_grad():
            if not grounded:
                out_t = model.branch(z_teacher, parent_depth)
                diff_t = _masked_action_distance(out_t.action, act, mask)
                emb_t = out_t.embedding[torch.arange(b, device=z.device), diff_t.argmin(dim=1)]
                z_teacher_next = model.dynamics(
                    z_teacher, act.unsqueeze(1), mask.unsqueeze(1), h_idx.unsqueeze(1), emb_t.unsqueeze(1)
                ).squeeze(1)
            else:
                _, teacher_successors, teacher_pick, _ = _grounded_execution_candidates(
                    model,
                    z_teacher,
                    parent_depth,
                    act,
                    mask,
                    h_idx,
                    ms_obs[:, d],
                    task_metric_dims,
                    selection_action_weight=float(grounded_select_action_weight),
                    selection_endpoint_weight=float(grounded_select_endpoint_weight),
                    selection_horizon_weight=float(grounded_select_horizon_weight),
                )
                z_teacher_next = teacher_successors[
                    torch.arange(b, device=z.device), teacher_pick
                ]
            metrics[f"recursive/self_vs_teacher_depth{d + 1}"] = float(
                (z_next - z_teacher_next).pow(2).mean().item()
            )
        z_teacher = torch.where(valid.unsqueeze(-1) > 0, z_true_all[:, d], z_teacher)

        # Scheduled sampling: feed our own prediction some of the time.
        if scheduled_sampling_p > 0.0:
            if sequence_use_pred is None:
                u = torch.rand(b, 1, device=z.device, generator=generator)
                use_pred = (u < scheduled_sampling_p).float()
            else:
                use_pred = sequence_use_pred
            predicted_parent = (
                z_next.detach()
                if not grounded or grounded_detach_self_fed_parent
                else z_next
            )
            nxt = use_pred * predicted_parent + (1.0 - use_pred) * z_true_all[:, d]
        else:
            use_pred = torch.zeros(b, 1, device=z.device)
            nxt = z_true_all[:, d]
        if d + 1 < d_max:
            feed_decisions.append(use_pred.squeeze(-1))
        z_cur = torch.where(valid.unsqueeze(-1) > 0, nxt, z_cur)

    if float(weighted_valid_count.item()) > 0:
        total = total / weighted_valid_count
        if grounded:
            assert grounded_component_sums is not None
            for name, component_sum in grounded_component_sums.items():
                metrics[f"recursive/grounded/loss_{name}"] = float(
                    (component_sum / weighted_valid_count).detach().item()
                )
    elif grounded:
        for name in ("latent", "action", "horizon", "endpoint"):
            metrics[f"recursive/grounded/loss_{name}"] = 0.0
    if grounded:
        metrics["recursive/transition_mode_grounded_execution_v2"] = 1.0
    metrics["recursive/scheduled_sampling_p"] = float(scheduled_sampling_p)
    metrics["recursive/scheduled_sampling_sequence_level"] = float(
        scheduled_sampling_granularity == "sequence"
    )
    if feed_decisions:
        decisions = torch.stack(feed_decisions, dim=1) > 0
        # A feed decision affects the objective only when both its source and its next
        # target depth are valid. Builder chains are prefixes, but the adjacent mask
        # keeps the telemetry correct for adapters with holes as well.
        feed_valid = (ms_valid[:, :-1] > 0) & (ms_valid[:, 1:] > 0)
        if bool(feed_valid.any()):
            metrics["recursive/predicted_feed_fraction"] = float(
                decisions[feed_valid].float().mean().item()
            )
            eligible_chains = feed_valid.any(dim=1)
            fully_self_fed = ((~feed_valid) | decisions).all(dim=1)
            metrics["recursive/fully_self_fed_chain_fraction"] = float(
                fully_self_fed[eligible_chains].float().mean().item()
            )
        else:
            metrics["recursive/predicted_feed_fraction"] = 0.0
            metrics["recursive/fully_self_fed_chain_fraction"] = 0.0
    else:
        metrics["recursive/predicted_feed_fraction"] = 0.0
        metrics["recursive/fully_self_fed_chain_fraction"] = 0.0
    metrics["recursive/mean_chain_depth"] = float(ms_valid.sum(1).float().mean().item())
    metrics["recursive/valid_transitions"] = float(ms_valid.sum().item())
    metrics["recursive/latent_target_scale"] = float(target_scale.item())
    return total, metrics


def scheduled_sampling_schedule(step: int, final_p: float, warmup: int) -> float:
    """Linear warm-up from 0 to ``final_p`` over ``warmup`` steps."""
    if final_p <= 0.0:
        return 0.0
    if warmup <= 0:
        return final_p
    return float(min(final_p, final_p * step / warmup))
