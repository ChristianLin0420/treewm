"""Prospective supervision for the action prefix the controller actually executes."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from treewm.losses.world_losses import (
    detached_target_scale,
    scale_invariant_latent_error,
)
from treewm.planning.action_execution import (
    ExecutableActionProjection,
    project_normalized_actions,
    uniform_action_bounds,
)
from treewm.planning.goal_planner import decoded_goal_scores


EXECUTABLE_PREFIX_SCHEMA_VERSION = 1
EXECUTABLE_PREFIX_TRAIN_METRIC_PREFIX = "train/executable_prefix/"
EXECUTABLE_PREFIX_VALIDATION_METRIC_PREFIX = "val/executable_prefix/"


@dataclass(frozen=True)
class ExecutablePrefixLossResult:
    """Three independent raw losses plus telemetry and audit tensors."""

    action: torch.Tensor
    latent: torch.Tensor
    endpoint: torch.Tensor
    metrics: dict[str, float]
    artifacts: dict[str, torch.Tensor]


def _uniform_batch_vector(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim != 2 or value.shape[0] == 0 or value.shape[1] == 0:
        raise ValueError(f"{name} must have shape [B, nonzero_width]")
    first = value[0]
    if not bool((value == first.unsqueeze(0)).all()):
        raise ValueError(f"{name} must be identical within a batch")
    return first


def _uniform_batch_subgoals(value: torch.Tensor) -> tuple[tuple[int, int], ...]:
    if value.ndim != 3 or value.shape[-1] != 2 or value.shape[0] == 0:
        raise ValueError("executable_task_subgoals must have shape [B, S, 2]")
    first = value[0].long()
    if not bool((value.long() == first.unsqueeze(0)).all()):
        raise ValueError("executable_task_subgoals must be identical within a batch")
    return tuple((int(pair[0]), int(pair[1])) for pair in first.tolist())


def _equal_anchor_branch_mean(
    value: torch.Tensor, matched: torch.Tensor
) -> torch.Tensor:
    """Mean branches within anchor, then mean anchors (empty anchors contribute zero)."""

    count = matched.float().sum(-1)
    per_anchor = (value.float() * matched.float()).sum(-1) / count.clamp_min(1.0)
    return per_anchor.mean()


def _matched_branch_mean(value: torch.Tensor, matched: torch.Tensor) -> torch.Tensor:
    """Give every valid matched branch equal objective weight."""

    denominator = matched.float().sum()
    if float(denominator.detach().item()) == 0.0:
        return value.sum() * 0.0
    return (value.float() * matched.float()).sum() / denominator


def _equal_anchor_action_mean(
    value: torch.Tensor,
    prefix_mask: torch.Tensor,
    matched: torch.Tensor,
    *,
    scalar_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean valid action scalars within anchor, then mean anchors."""

    scalar_weight = (
        prefix_mask.float().unsqueeze(-1)
        * matched.float().unsqueeze(-1).unsqueeze(-1)
    )
    scalar_weight = scalar_weight.expand_as(value)
    if scalar_valid is not None:
        if scalar_valid.shape != value.shape:
            raise ValueError("action telemetry validity mask has the wrong shape")
        scalar_weight = scalar_weight * scalar_valid.float()
    count = scalar_weight.sum((1, 2, 3))
    per_anchor = (value.float() * scalar_weight).sum((1, 2, 3)) / count.clamp_min(1.0)
    return per_anchor.mean()


def _onehot_hamming(
    left: torch.Tensor,
    right: torch.Tensor,
    dims: torch.Tensor,
    subgoals: tuple[tuple[int, int], ...],
) -> torch.Tensor:
    left_metric = left.float().index_select(-1, dims)
    right_metric = right.float().index_select(-1, dims)
    result = torch.zeros(left_metric.shape[:-1], device=left.device, dtype=torch.float32)
    for lower, upper in subgoals:
        result = result + (
            left_metric[..., lower:upper].argmax(-1)
            != right_metric[..., lower:upper].argmax(-1)
        ).float()
    return result


def executable_prefix_losses(
    model,
    *,
    parent_z: torch.Tensor,
    parent_obs: torch.Tensor,
    branch_embedding: torch.Tensor,
    raw_predicted_action: torch.Tensor,
    target_action: torch.Tensor,
    prefix_action_mask: torch.Tensor,
    prefix_horizon_idx: torch.Tensor,
    prefix_target_endpoint: torch.Tensor,
    prefix_target_metric_endpoint: torch.Tensor,
    prefix_length: torch.Tensor,
    matched: torch.Tensor,
    task_metric_dims: torch.Tensor,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    observation_mean: torch.Tensor,
    observation_std: torch.Tensor,
    task_metric_kind: torch.Tensor,
    task_subgoals: torch.Tensor,
    action_lower_bound: float,
    action_upper_bound: float,
) -> ExecutablePrefixLossResult:
    """Supervise every matched branch at its fixed logged executable prefix.

    Assignment is supplied by the caller and is never recomputed here. The learned
    horizon is deliberately absent: the data mask and data horizon index select exactly
    ``min(configured prefix, logged horizon)`` actions. Three returned losses remain
    independent so their effective gradients can be audited and weighted explicitly.
    """

    if getattr(model, "decoder", None) is None:
        raise ValueError("executable-prefix endpoint supervision requires model.decoder")
    if parent_z.ndim != 2 or parent_obs.ndim != 2:
        raise ValueError("parent latent/observation must have shape [B, D]")
    if raw_predicted_action.ndim != 4:
        raise ValueError("raw_predicted_action must have shape [B, K, H, A]")
    batch, branches, horizon, action_dim = raw_predicted_action.shape
    if target_action.shape != raw_predicted_action.shape:
        raise ValueError("target and predicted actions must have identical shapes")
    if prefix_action_mask.shape != (batch, branches, horizon):
        raise ValueError("prefix action mask shape does not match predicted actions")
    if prefix_horizon_idx.shape != (batch, branches):
        raise ValueError("prefix horizon index must have shape [B, K]")
    if prefix_length.shape != (batch, branches):
        raise ValueError("prefix length must have shape [B, K]")
    if matched.shape != (batch, branches):
        raise ValueError("matched must have shape [B, K]")
    if prefix_target_endpoint.shape[:2] != (batch, branches):
        raise ValueError("prefix target endpoint must have shape [B, K, obs_dim]")
    if prefix_target_endpoint.shape[-1] != parent_obs.shape[-1]:
        raise ValueError("prefix target and parent observation widths differ")
    mask_float = prefix_action_mask.float()
    length_float = prefix_length.float()
    matched_float = matched.float()
    horizon_index_float = prefix_horizon_idx.float()
    if (
        not bool(torch.isfinite(mask_float).all())
        or not bool(torch.isfinite(length_float).all())
        or not bool(torch.isfinite(matched_float).all())
        or not bool(torch.isfinite(horizon_index_float).all())
        or bool(((mask_float != 0.0) & (mask_float != 1.0)).any())
        or bool(((matched_float != 0.0) & (matched_float != 1.0)).any())
        or bool((length_float != length_float.round()).any())
        or bool((horizon_index_float != horizon_index_float.round()).any())
        or bool((length_float < 0).any())
        or bool((length_float > horizon).any())
    ):
        raise ValueError(
            "executable prefix masks/matches/lengths/indices must be finite integers"
        )
    expected_mask = (
        torch.arange(horizon, device=prefix_action_mask.device)
        .view(1, 1, horizon)
        .expand(batch, branches, horizon)
        < length_float.long().unsqueeze(-1)
    )
    if not bool((expected_mask == (mask_float > 0)).all()):
        raise ValueError("executable prefix mask must be one contiguous exact prefix")
    matched_bool = matched_float > 0
    if bool(((length_float <= 0) & matched_bool).any()):
        raise ValueError("every matched executable prefix must contain an action")
    if bool(((horizon_index_float < 0) & matched_bool).any()):
        raise ValueError("matched executable prefix horizon index is invalid")
    safe_prefix_horizon_idx = torch.where(
        matched_bool,
        prefix_horizon_idx.long(),
        torch.zeros_like(prefix_horizon_idx, dtype=torch.long),
    )
    model_horizons = getattr(model, "horizons", None)
    if model_horizons is not None:
        horizon_values = torch.as_tensor(
            model_horizons, device=prefix_horizon_idx.device
        ).long()
        if (
            horizon_values.ndim != 1
            or horizon_values.numel() == 0
            or bool(((prefix_horizon_idx < 0) & matched_bool).any())
            or bool(
                ((prefix_horizon_idx >= horizon_values.numel()) & matched_bool).any()
            )
        ):
            raise ValueError("executable prefix horizon index is invalid")
        selected_length = horizon_values[safe_prefix_horizon_idx]
        if bool(((selected_length != length_float.long()) & matched_bool).any()):
            raise ValueError(
                "executable prefix horizon index must encode its exact available length"
            )

    dims = _uniform_batch_vector(task_metric_dims, "task_metric_dims").long()
    if bool((dims < 0).any()) or bool((dims >= parent_obs.shape[-1]).any()):
        raise ValueError("task_metric_dims are outside observation width")
    if int(torch.unique(dims).numel()) != int(dims.numel()):
        raise ValueError("task_metric_dims must not contain duplicates")
    if prefix_target_metric_endpoint.shape != (batch, branches, dims.numel()):
        raise ValueError("prefix target metric endpoint has the wrong shape")

    mean = _uniform_batch_vector(action_mean, "executable_action_mean").float()
    std = _uniform_batch_vector(action_std, "executable_action_std").float()
    obs_mean = _uniform_batch_vector(
        observation_mean, "executable_observation_mean"
    ).float()
    obs_std = _uniform_batch_vector(
        observation_std, "executable_observation_std"
    ).float()
    if mean.numel() != action_dim or std.numel() != action_dim:
        raise ValueError("action normalizer width differs from model action width")
    if obs_mean.numel() != parent_obs.shape[-1] or obs_std.numel() != parent_obs.shape[-1]:
        raise ValueError("observation normalizer width differs from observation width")
    if task_metric_kind.ndim != 1 or task_metric_kind.shape[0] != batch:
        raise ValueError("executable_task_metric_kind must have shape [B]")
    kind = int(task_metric_kind[0].item())
    if kind not in {0, 1} or not bool((task_metric_kind == kind).all()):
        raise ValueError("task metric kind must be one uniform l2/onehot code")
    subgoals = _uniform_batch_subgoals(task_subgoals)
    if kind == 1 and not subgoals:
        raise ValueError("onehot executable telemetry requires non-empty subgoals")
    for lower, upper in subgoals:
        if not 0 <= lower < upper <= dims.numel():
            raise ValueError("executable task subgoal is outside task metric width")

    lower, upper = uniform_action_bounds(
        action_dim,
        float(action_lower_bound),
        float(action_upper_bound),
        like=raw_predicted_action,
    )
    projection: ExecutableActionProjection[torch.Tensor] = project_normalized_actions(
        raw_predicted_action,
        action_mean=mean,
        action_std=std,
        action_lower_bound=lower,
        action_upper_bound=upper,
    )

    predicted_prefix_z = model.dynamics(
        parent_z,
        projection.applied_normalized,
        prefix_action_mask,
        safe_prefix_horizon_idx,
        branch_embedding,
    )
    with torch.no_grad():
        target_prefix_z = model.encode(prefix_target_endpoint).detach()
    latent_scale = detached_target_scale(target_prefix_z, matched)
    latent_per = scale_invariant_latent_error(
        predicted_prefix_z, target_prefix_z, latent_scale
    )

    action_per_step = (
        raw_predicted_action.float() - target_action.detach().float()
    ).square().mean(-1)
    action_per = (
        action_per_step * prefix_action_mask.float()
    ).sum(-1) / prefix_action_mask.float().sum(-1).clamp_min(1.0)

    predicted_endpoint = model.decoder(predicted_prefix_z)
    predicted_metric = predicted_endpoint.float().index_select(-1, dims)
    parent_metric = parent_obs.float().index_select(-1, dims).unsqueeze(1)
    endpoint_delta = (
        predicted_metric - prefix_target_metric_endpoint.detach().float()
    )
    endpoint_per = endpoint_delta.square().mean(-1)
    predicted_normalized_displacement = (
        predicted_metric - parent_metric
    ).square().mean(-1).clamp_min(0.0).sqrt()
    actual_normalized_displacement = (
        prefix_target_metric_endpoint.detach().float() - parent_metric
    ).square().mean(-1).clamp_min(0.0).sqrt()

    action_loss = _matched_branch_mean(action_per, matched)
    latent_loss = _matched_branch_mean(latent_per, matched)
    endpoint_loss = _matched_branch_mean(endpoint_per, matched)

    with torch.no_grad():
        flat_predicted = predicted_endpoint.detach().float().reshape(
            batch * branches, 1, -1
        )
        flat_actual = prefix_target_endpoint.detach().float().reshape(
            batch * branches, -1
        )
        flat_parent = parent_obs.detach().float().unsqueeze(1).expand(
            batch, branches, parent_obs.shape[-1]
        ).reshape(batch * branches, -1)
        goal_metric = "onehot" if kind == 1 else "l2"

        def guard_score(nodes: torch.Tensor, goals: torch.Tensor) -> torch.Tensor:
            return decoded_goal_scores(
                nodes,
                goals,
                decoded_metric="domain_raw",
                goal_dims=dims,
                goal_metric=goal_metric,
                subgoals=subgoals,
                obs_mean=obs_mean,
                obs_std=obs_std,
            ).reshape(batch, branches)

        predicted_vs_actual_guard = guard_score(flat_predicted, flat_actual)
        predicted_displacement_guard = guard_score(flat_predicted, flat_parent)
        actual_displacement_guard = guard_score(
            flat_actual.unsqueeze(1), flat_parent
        )
        endpoint_rms = endpoint_per.clamp_min(0.0).sqrt()

        raw_env = projection.raw_env.detach().float()
        applied_env = projection.applied_env.detach().float()
        target_env = target_action.detach().float() * std + mean
        finite = torch.isfinite(raw_env)
        applied_finite = torch.isfinite(applied_env)
        target_finite = torch.isfinite(target_env)
        raw_safe = torch.where(finite, raw_env, torch.zeros_like(raw_env))
        applied_safe = torch.where(
            applied_finite, applied_env, torch.zeros_like(applied_env)
        )
        target_safe = torch.where(
            target_finite, target_env, torch.zeros_like(target_env)
        )
        clipped = finite & (raw_env != applied_env)

        matched_count = matched.float().sum(-1)
        valid_anchor = matched_count > 0
        # Remove unmatched branches for the actual denominator reported to monitoring.
        scalar_count = (
            prefix_action_mask.float()
            * matched.float().unsqueeze(-1)
        ).sum((1, 2)) * float(action_dim)
        action_scalar_weight = (
            prefix_action_mask.float().unsqueeze(-1)
            * matched.float().unsqueeze(-1).unsqueeze(-1)
        ).expand_as(raw_env)
        raw_finite_scalar_count = (action_scalar_weight * finite.float()).sum(
            (1, 2, 3)
        )
        applied_finite_scalar_count = (
            action_scalar_weight * applied_finite.float()
        ).sum((1, 2, 3))
        logged_finite_scalar_count = (
            action_scalar_weight * target_finite.float()
        ).sum((1, 2, 3))

        metrics = {
            "train/executable_prefix/schema_version": float(
                EXECUTABLE_PREFIX_SCHEMA_VERSION
            ),
            "train/executable_prefix/loss_action_normalized": float(
                action_loss.detach().item()
            ),
            "train/executable_prefix/loss_latent": float(
                latent_loss.detach().item()
            ),
            "train/executable_prefix/loss_endpoint_normalized_task": float(
                endpoint_loss.detach().item()
            ),
            "train/executable_prefix/action_raw_env_abs_mean": float(
                _equal_anchor_action_mean(
                    raw_safe.abs(),
                    prefix_action_mask,
                    matched,
                    scalar_valid=finite,
                ).item()
            ),
            "train/executable_prefix/action_raw_env_rms": float(
                _equal_anchor_action_mean(
                    raw_safe.square(),
                    prefix_action_mask,
                    matched,
                    scalar_valid=finite,
                ).clamp_min(0.0).sqrt().item()
            ),
            "train/executable_prefix/action_applied_env_abs_mean": float(
                _equal_anchor_action_mean(
                    applied_safe.abs(),
                    prefix_action_mask,
                    matched,
                    scalar_valid=applied_finite,
                ).item()
            ),
            "train/executable_prefix/action_applied_env_rms": float(
                _equal_anchor_action_mean(
                    applied_safe.square(),
                    prefix_action_mask,
                    matched,
                    scalar_valid=applied_finite,
                ).clamp_min(0.0).sqrt().item()
            ),
            "train/executable_prefix/action_logged_env_abs_mean": float(
                _equal_anchor_action_mean(
                    target_safe.abs(),
                    prefix_action_mask,
                    matched,
                    scalar_valid=target_finite,
                ).item()
            ),
            "train/executable_prefix/action_logged_env_rms": float(
                _equal_anchor_action_mean(
                    target_safe.square(),
                    prefix_action_mask,
                    matched,
                    scalar_valid=target_finite,
                ).clamp_min(0.0).sqrt().item()
            ),
            "train/executable_prefix/action_clipped_fraction": float(
                _equal_anchor_action_mean(
                    clipped.float(),
                    prefix_action_mask,
                    matched,
                    scalar_valid=finite,
                ).item()
            ),
            "train/executable_prefix/action_finite_fraction": float(
                _equal_anchor_action_mean(
                    finite.float(), prefix_action_mask, matched
                ).item()
            ),
            "train/executable_prefix/action_applied_finite_fraction": float(
                _equal_anchor_action_mean(
                    applied_finite.float(), prefix_action_mask, matched
                ).item()
            ),
            "train/executable_prefix/action_logged_finite_fraction": float(
                _equal_anchor_action_mean(
                    target_finite.float(), prefix_action_mask, matched
                ).item()
            ),
            "train/executable_prefix/predicted_vs_actual_normalized_task_rms": float(
                _equal_anchor_branch_mean(endpoint_rms, matched).item()
            ),
            "train/executable_prefix/predicted_normalized_task_displacement_rms": float(
                _equal_anchor_branch_mean(
                    predicted_normalized_displacement, matched
                ).item()
            ),
            "train/executable_prefix/actual_normalized_task_displacement_rms": float(
                _equal_anchor_branch_mean(
                    actual_normalized_displacement, matched
                ).item()
            ),
            "train/executable_prefix/predicted_vs_actual_guard_metric_error": float(
                _equal_anchor_branch_mean(
                    predicted_vs_actual_guard, matched
                ).item()
            ),
            "train/executable_prefix/predicted_guard_metric_displacement": float(
                _equal_anchor_branch_mean(
                    predicted_displacement_guard, matched
                ).item()
            ),
            "train/executable_prefix/actual_guard_metric_displacement": float(
                _equal_anchor_branch_mean(
                    actual_displacement_guard, matched
                ).item()
            ),
            "train/executable_prefix/prefix_steps_mean": float(
                _equal_anchor_branch_mean(prefix_length.float(), matched).item()
            ),
            "train/executable_prefix/valid_anchor_fraction": float(
                valid_anchor.float().mean().item()
            ),
            "train/executable_prefix/matched_branches_per_anchor": float(
                matched_count.mean().item()
            ),
            "train/executable_prefix/action_scalars_per_anchor": float(
                scalar_count.mean().item()
            ),
            "train/executable_prefix/action_raw_finite_scalars_per_anchor": float(
                raw_finite_scalar_count.mean().item()
            ),
            "train/executable_prefix/action_applied_finite_scalars_per_anchor": float(
                applied_finite_scalar_count.mean().item()
            ),
            "train/executable_prefix/action_logged_finite_scalars_per_anchor": float(
                logged_finite_scalar_count.mean().item()
            ),
            "train/executable_prefix/goal_metric_onehot": float(kind == 1),
            "train/executable_prefix/latent_target_scale": float(
                latent_scale.detach().item()
            ),
        }
        if kind == 1:
            # Categorical semantics live in raw domain coordinates. Independent
            # standardization can change an argmax, so denormalize exactly as the
            # planner guard does before computing Hamming-equivalent telemetry.
            predicted_full = predicted_endpoint.detach().float() * obs_std + obs_mean
            actual_full = prefix_target_endpoint.detach().float() * obs_std + obs_mean
            parent_full = (
                parent_obs.detach().float() * obs_std + obs_mean
            ).unsqueeze(1).expand_as(actual_full)
            hamming_error = _onehot_hamming(
                predicted_full, actual_full, dims, subgoals
            )
            predicted_hamming_displacement = _onehot_hamming(
                predicted_full, parent_full, dims, subgoals
            )
            actual_hamming_displacement = _onehot_hamming(
                actual_full, parent_full, dims, subgoals
            )
            hamming_denom = float(len(subgoals))
            metrics.update(
                {
                    "train/executable_prefix/predicted_vs_actual_hamming": float(
                        _equal_anchor_branch_mean(hamming_error, matched).item()
                    ),
                    "train/executable_prefix/predicted_vs_actual_hamming_fraction": float(
                        (
                            _equal_anchor_branch_mean(hamming_error, matched)
                            / hamming_denom
                        ).item()
                    ),
                    "train/executable_prefix/predicted_hamming_displacement": float(
                        _equal_anchor_branch_mean(
                            predicted_hamming_displacement, matched
                        ).item()
                    ),
                    "train/executable_prefix/actual_hamming_displacement": float(
                        _equal_anchor_branch_mean(
                            actual_hamming_displacement, matched
                        ).item()
                    ),
                }
            )

    artifacts = {
        "raw_action_env": projection.raw_env,
        "applied_action_env": projection.applied_env,
        "applied_action_normalized": projection.applied_normalized,
        "target_prefix_action_normalized": target_action,
        "target_prefix_action_env": target_env,
        "predicted_prefix_latent": predicted_prefix_z,
        "predicted_prefix_endpoint": predicted_endpoint,
        "target_prefix_endpoint": prefix_target_endpoint,
        "predicted_prefix_metric_endpoint": predicted_metric,
        "target_prefix_metric_endpoint": prefix_target_metric_endpoint,
        "predicted_normalized_task_displacement_rms": (
            predicted_normalized_displacement
        ),
        "actual_normalized_task_displacement_rms": actual_normalized_displacement,
        "predicted_vs_actual_guard_metric_error": predicted_vs_actual_guard,
        "predicted_guard_metric_displacement": predicted_displacement_guard,
        "actual_guard_metric_displacement": actual_displacement_guard,
        "prefix_length": prefix_length,
        "prefix_action_mask": prefix_action_mask,
        "matched": matched,
    }
    if kind == 1:
        artifacts.update(
            {
                "predicted_vs_actual_hamming": hamming_error,
                "predicted_hamming_displacement": predicted_hamming_displacement,
                "actual_hamming_displacement": actual_hamming_displacement,
            }
        )

    return ExecutablePrefixLossResult(
        action=action_loss,
        latent=latent_loss,
        endpoint=endpoint_loss,
        metrics=metrics,
        artifacts=artifacts,
    )
