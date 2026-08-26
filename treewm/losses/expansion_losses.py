"""Novelty-target supervision for the expansion-gain head.

Replaces the retrieval-based marginal-coverage target with the signal the winning
heuristic actually acts on:

    G*(n | T) = min_{j in T} d(q_n, q_j)      (or d(z_n, z_j))

Two properties matter for isolating the previous failure:

1. **Targets come from the partial tree.** The head is supervised at each expansion
   iteration against the tree as it exists at that moment -- the same state it faces at
   inference. Regressing against the *finished* tree would train it on a distribution it
   never sees, which is a plausible contributor to the earlier anti-scaling.

2. **Only the head learns.** Snapshot features are detached, so the encoder, branch
   transformer and dynamics receive no gradient from this loss. Every arm therefore
   shares identical world-model training and differs *only* in how the frontier is
   ranked, which is the one variable under test.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import replace

import torch
import torch.nn.functional as F

from treewm.logging.metrics import pearson_correlation, rank_correlation


@torch.no_grad()
def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Tie-aware average ranks without a GPU-to-SciPy synchronization per decision."""
    _, inverse, counts = torch.unique(
        values.float(), sorted=True, return_inverse=True, return_counts=True
    )
    ends = counts.cumsum(0).float()
    starts = ends - counts.float()
    midpoints = (starts + ends - 1.0) * 0.5
    return midpoints[inverse]


@torch.no_grad()
def _tensor_correlation(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.float() - a.float().mean()
    b = b.float() - b.float().mean()
    numerator = (a * b).sum()
    denominator = a.square().sum().sqrt() * b.square().sum().sqrt()
    return numerator / denominator.clamp_min(1e-12)


def frontier_gain_objective(
    predicted: torch.Tensor,
    target: torch.Tensor,
    frontier: torch.Tensor,
    *,
    rank_weight: float = 1.0,
    calibration_weight: float = 0.0,
    target_tie_absolute: float = 1e-4,
    target_tie_relative: float = 0.05,
    prediction_tie_tolerance: float = 0.0,
    tie_tolerance: float | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Eligible-only pairwise ranking plus optional stable calibration.

    A singleton or an effectively tied frontier contains no ranking supervision. Such
    rows are counted for coverage telemetry but contribute neither a synthetic zero to
    ranking metrics nor a raw-zero calibration target.  This distinction is essential:
    a chance-level pairwise gate is meaningful only when its denominator contains
    actual comparisons.

    Eligible rows remain equally weighted regardless of frontier size.  Target ties use
    a predeclared adaptive band ``max(1e-4, 0.05 * target spread)`` by default, while a
    predicted tie earns half credit (standard concordance/AUC semantics).  When optional
    calibration is enabled, the detached target is min-max normalised within each
    eligible frontier so moving q-distance scale cannot dominate the ranking objective.
    """
    if predicted.shape != target.shape or predicted.shape != frontier.shape:
        raise ValueError("predicted, target and frontier must have identical [B, N] shape")
    if rank_weight < 0 or calibration_weight < 0:
        raise ValueError("gain rank/calibration weights must be non-negative")
    # Compatibility for diagnostic callers written against v1. Formal v2 omits this
    # argument and therefore always receives the adaptive policy above.
    if tie_tolerance is not None:
        target_tie_absolute = float(tie_tolerance)
        target_tie_relative = 0.0
    if target_tie_absolute < 0 or target_tie_relative < 0:
        raise ValueError("gain target tie tolerances must be non-negative")
    if prediction_tie_tolerance < 0:
        raise ValueError("gain prediction tie tolerance must be non-negative")

    rank_losses: list[torch.Tensor] = []
    calibration_losses: list[torch.Tensor] = []
    calibration_mae: list[torch.Tensor] = []
    rank_scores: list[torch.Tensor] = []
    pearson_scores: list[torch.Tensor] = []
    top1_scores: list[torch.Tensor] = []
    regrets: list[torch.Tensor] = []
    pair_accuracy: list[torch.Tensor] = []
    pred_spreads: list[torch.Tensor] = []
    target_spreads: list[torch.Tensor] = []
    pred_stds: list[torch.Tensor] = []
    target_stds: list[torch.Tensor] = []
    pred_means: list[torch.Tensor] = []
    target_means: list[torch.Tensor] = []
    frontier_sizes: list[float] = []
    tie_bands: list[float] = []
    total_pair_count = 0
    ordered_pair_count = 0
    target_tie_count = 0
    predicted_tie_count = 0
    pairwise_credit_sum = 0.0

    for row in range(predicted.shape[0]):
        valid = frontier[row] > 0
        if not bool(valid.any()):
            continue
        pred = predicted[row, valid].float()
        tgt = target[row, valid].float().detach()
        frontier_sizes.append(float(pred.numel()))

        if pred.numel() < 2:
            continue

        best = tgt.max()
        worst = tgt.min()
        target_spread = best - worst
        tie_band = max(
            float(target_tie_absolute),
            float(target_tie_relative) * float(target_spread.detach().item()),
        )
        tie_bands.append(tie_band)
        pair_i, pair_j = torch.triu_indices(
            pred.numel(), pred.numel(), offset=1, device=pred.device
        )
        target_delta = tgt[pair_i] - tgt[pair_j]
        pred_delta = pred[pair_i] - pred[pair_j]
        comparable = target_delta.abs() > tie_band
        row_pairs = int(pair_i.numel())
        row_ordered = int(comparable.sum().item())
        total_pair_count += row_pairs
        ordered_pair_count += row_ordered
        target_tie_count += row_pairs - row_ordered
        if row_ordered == 0:
            continue

        # Orient every margin so a positive value means the prediction agrees with the
        # target. Exactly tied predictions are neither right nor wrong and earn 0.5.
        margin = pred_delta[comparable] * target_delta[comparable].sign()
        predicted_tie = margin.abs() <= prediction_tie_tolerance
        accuracy_values = torch.where(
            predicted_tie,
            torch.full_like(margin, 0.5),
            (margin > 0).to(margin.dtype),
        )
        rank = F.softplus(-margin).mean()
        accuracy = accuracy_values.mean()
        rank_losses.append(rank)
        pair_accuracy.append(accuracy)
        predicted_tie_count += int(predicted_tie.sum().item())
        pairwise_credit_sum += float(accuracy_values.detach().sum().item())

        # Calibration is deliberately eligible-only and scale-stable.  The formal v2
        # recipe is rank-only, but this remains useful for controlled ablations.
        stable_target = (tgt - worst) / target_spread.clamp_min(tie_band)
        calibration = F.smooth_l1_loss(pred, stable_target)
        calibration_losses.append(calibration)
        calibration_mae.append((pred - stable_target).abs().mean())

        with torch.no_grad():
            choice = pred.argmax()
            top1_scores.append((tgt[choice] >= best - tie_band).float())
            regret = best - tgt[choice]
            regret = torch.where(regret <= tie_band, torch.zeros_like(regret), regret)
            regrets.append(regret / target_spread.clamp_min(tie_band))
            pred_spreads.append(pred.max() - pred.min())
            target_spreads.append(target_spread)
            pred_stds.append(pred.std(unbiased=False))
            target_stds.append(tgt.std(unbiased=False))
            pred_means.append(pred.mean())
            target_means.append(tgt.mean())
            # Spearman is Pearson over exact tie-aware ranks. Do not form tolerance
            # groups by chaining adjacent gaps: a smooth frontier whose every adjacent
            # gap is below the adaptive pair band can span a large range and must not
            # collapse into one giant artificial tie. Pairwise accuracy separately
            # applies the preregistered target-tie band.
            rank_scores.append(
                _tensor_correlation(
                    _average_ranks(pred),
                    _average_ranks(tgt),
                )
            )
            pearson_scores.append(_tensor_correlation(pred, tgt))

    decision_count = len(frontier_sizes)
    ranking_decision_count = len(rank_losses)
    if decision_count == 0:
        zero = predicted.sum() * 0.0
        return zero, {
            "decision_count": 0.0,
            "ranking_decision_count": 0.0,
            "calibration_decision_count": 0.0,
            "eligible_decision_fraction": 0.0,
            "total_pair_count": 0.0,
            "ordered_pair_count": 0.0,
            "pair_coverage_fraction": 0.0,
            "target_tie_fraction": 0.0,
            "predicted_tie_fraction": 0.0,
            "mean_target_tie_band": 0.0,
            "mean_frontier_size": 0.0,
            "loss_rank": 0.0,
            "loss_calibration": 0.0,
            "calibration_mae": 0.0,
            "rank_correlation": 0.0,
            "pearson_correlation": 0.0,
            "top1_accuracy": 0.0,
            "normalized_regret": 0.0,
            "pairwise_accuracy": 0.0,
            "pairwise_accuracy_pair_weighted": 0.0,
            "predicted_spread": 0.0,
            "target_spread": 0.0,
            "predicted_std": 0.0,
            "target_std": 0.0,
            "predicted_mean": 0.0,
            "target_mean": 0.0,
        }

    zero = predicted.sum() * 0.0

    def mean_tensor(values: list[torch.Tensor]) -> torch.Tensor:
        return torch.stack(values).mean() if values else zero

    rank_loss = mean_tensor(rank_losses)
    calibration_loss = mean_tensor(calibration_losses)
    loss = rank_weight * rank_loss + calibration_weight * calibration_loss
    metrics = {
        "decision_count": float(decision_count),
        "ranking_decision_count": float(ranking_decision_count),
        "calibration_decision_count": float(len(calibration_losses)),
        "eligible_decision_fraction": float(ranking_decision_count / decision_count),
        "total_pair_count": float(total_pair_count),
        "ordered_pair_count": float(ordered_pair_count),
        "pair_coverage_fraction": float(ordered_pair_count / max(total_pair_count, 1)),
        "target_tie_fraction": float(target_tie_count / max(total_pair_count, 1)),
        "predicted_tie_fraction": float(predicted_tie_count / max(ordered_pair_count, 1)),
        "mean_target_tie_band": float(sum(tie_bands) / max(len(tie_bands), 1)),
        "mean_frontier_size": float(sum(frontier_sizes) / decision_count),
        "loss_rank": float(rank_loss.detach().item()),
        "loss_calibration": float(calibration_loss.detach().item()),
        "calibration_mae": float(mean_tensor(calibration_mae).detach().item()),
        "rank_correlation": float(mean_tensor(rank_scores).item()),
        "pearson_correlation": float(mean_tensor(pearson_scores).item()),
        "top1_accuracy": float(mean_tensor(top1_scores).item()),
        "normalized_regret": float(mean_tensor(regrets).item()),
        "pairwise_accuracy": float(mean_tensor(pair_accuracy).item()),
        "pairwise_accuracy_pair_weighted": float(
            pairwise_credit_sum / max(ordered_pair_count, 1)
        ),
        "predicted_spread": float(mean_tensor(pred_spreads).item()),
        "target_spread": float(mean_tensor(target_spreads).item()),
        "predicted_std": float(mean_tensor(pred_stds).item()),
        "target_std": float(mean_tensor(target_stds).item()),
        "predicted_mean": float(mean_tensor(pred_means).item()),
        "target_mean": float(mean_tensor(target_means).item()),
    }
    return loss, metrics


def _root_branch_prior_objective(
    model,
    z0: torch.Tensor,
    space: str,
    *,
    rank_weight: float,
    calibration_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train the context-free branch prior on root-frontier novelty only."""
    with torch.no_grad():
        root_children = model.predict_children(z0)
        branch_embedding = root_children["branch"].embedding.detach()
        if space == "q":
            root = model.q_of(z0).unsqueeze(1)
            nodes = torch.cat((root, root_children["q"]), dim=1)
            distance = model.q_cdist(nodes.float(), nodes.float())
        elif space == "z":
            nodes = torch.cat((z0.unsqueeze(1), root_children["latent"]), dim=1)
            distance = torch.cdist(nodes.float(), nodes.float())
        else:
            raise ValueError(f"unknown novelty space {space!r}; options: q | z")
        k = branch_embedding.shape[1]
        child_rows = distance[:, 1 : k + 1].clone()
        child_index = torch.arange(k, device=z0.device)
        child_rows[:, child_index, child_index + 1] = float("inf")
        prior_target = child_rows.min(-1).values.detach()

    prior_pred = model.heads.gain_head(branch_embedding).squeeze(-1)
    valid = torch.ones_like(prior_target, dtype=torch.bool)
    return frontier_gain_objective(
        prior_pred,
        prior_target,
        valid,
        rank_weight=rank_weight,
        calibration_weight=calibration_weight,
    )


def novelty_gain_loss(
    model,
    z0: torch.Tensor,
    tree_cfg,
    space: str = "q",
    generator: torch.Generator | None = None,
    rank_weight: float = 0.0,
    calibration_weight: float = 1.0,
    branch_prior_weight: float = 0.0,
    training_scorers: tuple[str, ...] | list[str] | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Train ``g_psi`` to predict min-novelty over the frontier.

    Returns ``(loss, metrics)``. Metrics carry both Pearson and Spearman correlation --
    the head can be well calibrated in value yet wrong in *ordering*, and best-first
    expansion only consumes the ordering.
    """
    scorers = tuple(str(value) for value in (training_scorers or (tree_cfg.scorer,)))
    if not scorers:
        raise ValueError("gain training needs at least one behavior scorer")
    if len(set(scorers)) != len(scorers):
        raise ValueError("gain training behavior scorers must be unique")
    if "novelty_q" in scorers and space != "q":
        raise ValueError("novelty_q gain-training behavior requires novelty_space=q")

    # A deterministic round-robin split gives the learned head both its on-policy
    # states and target-consistent novelty_q states without adding another RNG stream.
    # The exact scorer list is configuration/provenance; the observed fractions below
    # make accidental data-policy drift visible in every run.
    generated: list[tuple[object, object, str, int]] = []
    row = torch.arange(z0.shape[0], device=z0.device)
    with torch.no_grad():
        for scorer_index, scorer in enumerate(scorers):
            selected = row.remainder(len(scorers)) == scorer_index
            count = int(selected.sum().item())
            if count == 0:
                continue
            behavior_cfg = replace(tree_cfg, scorer=scorer, scorer_override=None)
            tree, trace = model.generate(
                z0[selected],
                behavior_cfg,
                generator=generator,
                collect_snapshots=True,
                track_novelty=True,
            )
            generated.append((tree, trace, scorer, count))

    if not any(trace.snapshots for _, trace, _, _ in generated):
        zero = z0.sum() * 0.0
        return zero, {}

    preds: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    decision_predictions: list[torch.Tensor] = []
    decision_targets: list[torch.Tensor] = []
    decision_frontiers: list[torch.Tensor] = []

    set_aware_gain = bool(getattr(model.gain_head, "set_aware_enabled", False))
    for _, trace, _, _ in generated:
        for snap in trace.snapshots:
            if set_aware_gain:
                # ``.float()`` alone does not override an enclosing autocast region for
                # Linear or MultiheadAttention. Disable it for repaired v2 so small,
                # ordered gain differences cannot become exact BF16 ties.
                with torch.autocast(device_type=z0.device.type, enabled=False):
                    pred = model.gain_head(
                        snap["feats"].float(),
                        snap["feats"].float(),
                        snap["depth"],
                        snap["keep"].float(),
                        snap["sigma"].float(),
                        context_valid=snap["valid"],
                        exclude_self=True,
                    )
            else:
                # Preserve the historical v1 dtype/autocast path exactly.
                pred = model.gain_head(
                    snap["feats"],
                    snap["context"],
                    snap["depth"],
                    snap["keep"],
                    snap["sigma"],
                )
            sel = snap["frontier"]
            if not bool(sel.any()):
                continue
            preds.append(pred[sel])
            targets.append(snap["target"][sel])
            decision_predictions.append(pred)
            decision_targets.append(snap["target"])
            decision_frontiers.append(sel)

    if set_aware_gain:
        # The repaired rank objective is also evaluated in FP32. Its targets and
        # denominators are meaningful only for the set-aware v2 head.
        autocast_context = torch.autocast(device_type=z0.device.type, enabled=False)
    else:
        autocast_context = nullcontext()
    with autocast_context:
        if rank_weight > 0 and decision_predictions:
            # Every snapshot has the fixed tree capacity, so one concatenated call
            # gives exact eligible-only denominators across iterations and behavior
            # policies. Averaging already-averaged snapshot losses would reintroduce
            # the original singleton/frontier-size bias.
            contextual_loss, decision_metrics = frontier_gain_objective(
                torch.cat(decision_predictions, dim=0),
                torch.cat(decision_targets, dim=0),
                torch.cat(decision_frontiers, dim=0),
                rank_weight=rank_weight,
                calibration_weight=calibration_weight,
            )
        elif rank_weight > 0:
            contextual_loss = z0.float().sum() * 0.0
            decision_metrics = {}
        else:
            decision_metrics = {}

    if not preds:
        zero = z0.sum() * 0.0
        return zero, {}

    pred_flat = torch.cat(preds)
    tgt_flat = torch.cat(targets)
    if rank_weight <= 0:
        # Exact v1 pointwise objective and node-weighted reduction.
        contextual_loss = calibration_weight * F.smooth_l1_loss(pred_flat, tgt_flat)

    prior_loss = contextual_loss.detach() * 0.0
    prior_metrics: dict[str, float] = {}
    if branch_prior_weight > 0:
        with torch.autocast(device_type=z0.device.type, enabled=False):
            prior_loss, prior_metrics = _root_branch_prior_objective(
                model,
                z0.float(),
                space,
                rank_weight=rank_weight,
                calibration_weight=calibration_weight,
            )
    loss = contextual_loss + branch_prior_weight * prior_loss

    with torch.no_grad():
        metrics = {
            "expansion/loss_novelty_gain": float(loss.item()),
            "expansion/loss_novelty_gain_contextual": float(contextual_loss.item()),
            "expansion/loss_branch_gain_prior": float(prior_loss.item()),
            "expansion/effective_branch_gain_prior": float((branch_prior_weight * prior_loss).item()),
            "expansion/predicted_gain_mean": float(pred_flat.mean().item()),
            "expansion/target_gain_mean": float(tgt_flat.mean().item()),
            "expansion/gain_mae": float((pred_flat - tgt_flat).abs().mean().item()),
            "expansion/gain_rank_correlation": rank_correlation(pred_flat, tgt_flat),
            "expansion/gain_pearson_correlation": pearson_correlation(pred_flat, tgt_flat),
            "expansion/target_gain_std": float(tgt_flat.std(unbiased=False).item()),
            "expansion/predicted_gain_std": float(pred_flat.std(unbiased=False).item()),
            "expansion/gain_objective_fp32": float(contextual_loss.dtype == torch.float32),
        }
        total_roots = max(sum(count for _, _, _, count in generated), 1)
        for scorer in scorers:
            scorer_roots = sum(
                count for _, _, observed, count in generated if observed == scorer
            )
            metrics[f"expansion/gain_training_scorer_{scorer}_fraction"] = float(
                scorer_roots / total_roots
            )
        if rank_weight > 0 and decision_metrics:
            metrics.update(
                {
                    "expansion/gain_decision_count": decision_metrics["decision_count"],
                    "expansion/gain_ranking_decision_count": decision_metrics[
                        "ranking_decision_count"
                    ],
                    "expansion/gain_calibration_decision_count": decision_metrics[
                        "calibration_decision_count"
                    ],
                    "expansion/gain_eligible_decision_fraction": decision_metrics[
                        "eligible_decision_fraction"
                    ],
                    "expansion/gain_total_pair_count": decision_metrics["total_pair_count"],
                    "expansion/gain_ordered_pair_count": decision_metrics[
                        "ordered_pair_count"
                    ],
                    "expansion/gain_pair_coverage_fraction": decision_metrics[
                        "pair_coverage_fraction"
                    ],
                    "expansion/gain_target_tie_fraction": decision_metrics[
                        "target_tie_fraction"
                    ],
                    "expansion/gain_predicted_tie_fraction": decision_metrics[
                        "predicted_tie_fraction"
                    ],
                    "expansion/gain_target_tie_band": decision_metrics[
                        "mean_target_tie_band"
                    ],
                    "expansion/gain_mean_frontier_size": decision_metrics[
                        "mean_frontier_size"
                    ],
                    "expansion/loss_novelty_gain_rank": decision_metrics["loss_rank"],
                    "expansion/loss_novelty_gain_calibration": decision_metrics[
                        "loss_calibration"
                    ],
                    "expansion/gain_calibration_mae": decision_metrics[
                        "calibration_mae"
                    ],
                    "expansion/gain_rank_correlation": decision_metrics[
                        "rank_correlation"
                    ],
                    "expansion/gain_pearson_correlation": decision_metrics[
                        "pearson_correlation"
                    ],
                    "expansion/gain_top1_accuracy": decision_metrics["top1_accuracy"],
                    "expansion/gain_normalized_regret": decision_metrics[
                        "normalized_regret"
                    ],
                    "expansion/gain_pairwise_accuracy": decision_metrics[
                        "pairwise_accuracy"
                    ],
                    "expansion/gain_pairwise_accuracy_pair_weighted": decision_metrics[
                        "pairwise_accuracy_pair_weighted"
                    ],
                    "expansion/predicted_gain_spread": decision_metrics[
                        "predicted_spread"
                    ],
                    "expansion/target_gain_spread": decision_metrics["target_spread"],
                    "expansion/predicted_gain_std": decision_metrics["predicted_std"],
                    "expansion/target_gain_std": decision_metrics["target_std"],
                }
            )
        if prior_metrics:
            metrics.update(
                {
                    f"expansion/branch_prior_{key}": value
                    for key, value in prior_metrics.items()
                }
            )
        traces_with_novelty = [
            (trace, count)
            for _, trace, _, count in generated
            if trace.frontier_novelty_before
        ]
        if traces_with_novelty:
            novelty_weight = sum(count for _, count in traces_with_novelty)
            metrics["expansion/frontier_novelty_before"] = float(
                sum(
                    count * sum(trace.frontier_novelty_before)
                    / len(trace.frontier_novelty_before)
                    for trace, count in traces_with_novelty
                )
                / novelty_weight
            )
            metrics["expansion/frontier_novelty_after"] = float(
                sum(
                    count
                    * sum(trace.frontier_novelty_after or trace.frontier_novelty_before)
                    / len(trace.frontier_novelty_after or trace.frontier_novelty_before)
                    for trace, count in traces_with_novelty
                )
                / novelty_weight
            )
            metrics["expansion/frontier_novelty_first"] = float(
                sum(count * trace.frontier_novelty_before[0] for trace, count in traces_with_novelty)
                / novelty_weight
            )
            metrics["expansion/frontier_novelty_last"] = float(
                sum(count * trace.frontier_novelty_before[-1] for trace, count in traces_with_novelty)
                / novelty_weight
            )
            # Falling frontier novelty means the allocator is running out of distinct
            # places to go -- the signature of the collapse seen with the old target.
            metrics["expansion/frontier_novelty_decay"] = (
                metrics["expansion/frontier_novelty_last"]
                - metrics["expansion/frontier_novelty_first"]
            )
        tree_metric_sums: dict[str, float] = {}
        for tree, _, _, count in generated:
            for key, value in tree_expansion_metrics(model, tree, space).items():
                tree_metric_sums[key] = tree_metric_sums.get(key, 0.0) + count * value
        metrics.update({key: value / total_roots for key, value in tree_metric_sums.items()})
    return loss, metrics


@torch.no_grad()
def tree_expansion_metrics(model, tree, space: str = "q") -> dict[str, float]:
    """Depth, budget use and redundancy for a generated tree.

    Physical coverage is intentionally not inferred here.  The old implementation
    quantised decoded observation dimensions ``(0, 1)`` at a hard-coded resolution,
    which happened to resemble maze coordinates but was meaningless for manipulation
    and puzzle domains.  Task-aware endpoint fidelity is reported by ``total.py`` from
    the explicit ``task_metric_dims`` contract instead.
    """
    from treewm.tree.novelty import novelty_of, redundant_fraction

    valid = tree.valid
    nodes = valid.float().sum(1).clamp_min(1.0)
    depth = tree.depth.float()
    capacity = float(tree.capacity)

    out = {
        # These are acceptance-critical for the KEEP-gated v2 allocator.  A model
        # whose KEEP logits are all low still grows a depth-limited top-1 chain;
        # depth/entropy alone therefore cannot distinguish that collapse from a
        # tree that actually uses its advertised node budget.
        "expansion/nodes_generated": float(nodes.mean().item()),
        "expansion/budget_shortfall": float((capacity - nodes).clamp_min(0).mean().item()),
        "expansion/budget_fill_fraction": float((nodes / capacity).mean().item()),
        "expansion/mean_depth": float(((depth * valid.float()).sum(1) / nodes).mean().item()),
        "expansion/max_depth": float(depth.masked_fill(~valid, 0).max(1).values.mean().item()),
        "expansion/depth_std": float(
            (((depth - (depth * valid.float()).sum(1, keepdim=True) / nodes.unsqueeze(1)) ** 2
              * valid.float()).sum(1) / nodes).sqrt().mean().item()
        ),
        "expansion/redundant_expansion_fraction": redundant_fraction(
            tree, space, model.q_cdist if space == "q" else None
        ),
        "expansion/frontier_novelty_mean": float(
            (novelty_of(tree, space, model.q_cdist if space == "q" else None) * valid.float()).sum(1).div(nodes).mean().item()
        ),
    }

    return out
