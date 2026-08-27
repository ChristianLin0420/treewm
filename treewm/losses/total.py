"""Total loss assembly.

Every term is independently switchable and independently weighted (spec section 14), so
the causal chain

    multimodal -> recursive -> controllability-aware support -> adaptive compute

can be ablated one link at a time. Disabled terms are not merely zero-weighted: their
computation is skipped, so an ablation cannot pay for a loss it is not using.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import torch

from treewm.data.future_sets import gather_mode_targets
from treewm.losses import controllability_losses as cl
from treewm.losses import support_losses as sl
from treewm.losses import world_losses as wl
from treewm.tree.matching import (
    MatchingConfig,
    assigned_cost_metrics,
    branch_mode_cost,
    gather_matched_targets,
    match,
)


@dataclass
class LossWeights:
    state: float = 1.0
    action: float = 1.0
    horizon: float = 0.5
    coverage: float = 1.0
    redundancy: float = 0.1
    mass: float = 0.2
    keep: float = 0.5
    expand: float = 0.5
    bind: float = 1.0
    control: float = 0.5
    reconstruction: float = 0.1
    recursive: float = 0.2
    uncertainty: float = 0.2
    multistep: float = 0.0  # Track A1; 0 keeps the exp-1..4 baseline exactly
    # Opt-in only. V2's scale-invariant latent losses otherwise leave a shrinkable
    # encoder/decoder gauge; the dedicated bounded gauge objective pins this to 1.0.
    latent_gauge: float = 0.0


@dataclass
class LossConfig:
    weights: LossWeights = field(default_factory=LossWeights)
    enabled: dict[str, bool] = field(
        default_factory=lambda: {
            "state": True,
            "action": True,
            "horizon": True,
            "coverage": True,
            "redundancy": True,
            "mass": True,
            "keep": True,
            "expand": True,
            "bind": True,
            "control": True,
            "reconstruction": True,
            "recursive": True,
            "uncertainty": True,
            "multistep": False,
            "latent_gauge": False,
        }
    )
    control_objective: str = "future_set"  # future_set | contrastive | bootstrap
    gain_target: str = "novelty"  # novelty | retrieval
    # Track A2: probability of feeding the model its own predicted latent during the
    # multi-step rollout, warmed up linearly over scheduled_sampling_warmup steps.
    scheduled_sampling_p: float = 0.0
    scheduled_sampling_warmup: int = 2000
    # Historical experiments sample an independent Bernoulli decision at each depth.
    # Revised objectives may opt into one decision reused across the whole sequence so
    # the configured fraction of examples sees a complete self-fed chain.
    scheduled_sampling_granularity: str = "step"  # step | sequence
    # The historical recursive objective selects a branch by action proximity but
    # executes the logged action. A fresh v2 objective can instead execute every
    # branch's predicted action, select one against logged physical targets, and train
    # explicit recursive action/horizon/endpoint terms. All defaults below leave the
    # historical graph untouched.
    multistep_transition_mode: str = "teacher_action"  # teacher_action | grounded_execution_v2
    grounded_select_action_weight: float = 0.0
    grounded_select_endpoint_weight: float = 0.0
    grounded_select_horizon_weight: float = 0.0
    grounded_loss_latent_weight: float = 0.0
    grounded_loss_action_weight: float = 0.0
    grounded_loss_horizon_weight: float = 0.0
    grounded_loss_endpoint_weight: float = 0.0
    grounded_detach_self_fed_parent: bool = True
    # The gauge reference is sealed from the DDP-global first training batch at update
    # zero. These numerical guards are explicit config/identity fields; old objectives
    # leave the term disabled and never execute this path.
    latent_gauge_epsilon: float = 1.0e-8
    latent_gauge_min_reference_scale: float = 1.0e-4
    # Track H2: relative weight of each recursive depth (empty -> uniform).
    multistep_depth_weights: tuple[float, ...] = ()
    redundancy_temperature: float = 0.25
    contrastive_temperature: float = 0.1
    keep_balance: bool = True
    coverage_space: str = "q"  # q | z  (ablation axis, spec section 18)
    future_scale: float = 1.0
    # V2 controllability geometry. Defaults retain the historical objective exactly;
    # formal v2 explicitly selects the task-metric endpoint and bounded RMS target.
    control_target_transform: str = "linear"  # linear | rms_tanh
    control_endpoint_key: str = "fut_endpoint"
    control_allow_endpoint_fallback: bool = True
    control_require_single_scale: bool = False
    control_metric_weight: float = 1.0
    control_rank_weight: float = 0.0
    control_rank_temperature: float = 0.1
    # V2 freezes target-side world representations and enables the identifiable
    # set-aware expansion scorer. Both remain opt-in for checkpoint compatibility.
    detach_world_targets: bool = False
    bind_negative_margin: float = 0.0
    gain_set_context: bool = False
    gain_rank_weight: float = 0.0
    gain_calibration_weight: float = 1.0
    gain_branch_prior_weight: float = 0.0
    # Anchors used for the controllability objective. The future-set Chamfer distance is
    # O(B^2 * M^2) and materialises a [B, B, M, M] tensor, which at B=256 dominates the
    # whole training step (140 ms of 195 ms measured). A fresh random subset each step
    # gives the same objective in expectation at a fraction of the cost.
    control_batch: int = 64
    # Successors sampled for the recursive-consistency loss. O(B * K^2) otherwise, which
    # OOMs at FlatKWM's K=256.
    recursive_batch: int = 256
    # Linear ramps, in steps. Auxiliary terms must not run at full strength from step 0
    # (spec section 14). Redundancy in particular is degenerate early: before the branch
    # heads differentiate, every sibling pair has d_q ~ 0, so exp(-d_q/tau) ~ 1 and the
    # penalty sum_{i<j} kappa_i kappa_j is minimised by driving every KEEP score to zero.
    # That collapses the effective branching factor to 0 and it never recovers.
    warmup: dict[str, int] = field(
        default_factory=lambda: {"redundancy": 5000, "expand": 2000, "mass": 1000}
    )
    # Step by which a term's weight has decayed linearly back to zero (0 = never).
    #
    # The warm-up above guards only the first 5000 steps, but the pressure never stops,
    # and over a long run the optimiser satisfies the redundancy penalty the easy way --
    # by killing branches rather than diversifying them. Measured on three environments
    # at 200k steps: effective branching factor 1.7 -> 1.06 (a tree with one branch per
    # node is a SingleWM, which scores 0.000 everywhere) with success falling ~10x in
    # lockstep, spearman(success, keep_rate) = +0.70 on scene.
    #
    # So the term is annealed out once it has done its job of separating the branch
    # heads. This is an untested fix applied directly to the formal run at the user's
    # direction; effective_branching_factor is logged every diagnostic step so collapse
    # remains visible if it recurs.
    decay: dict[str, int] = field(default_factory=dict)

    def on(self, name: str) -> bool:
        return bool(self.enabled.get(name, True)) and getattr(self.weights, name, 0.0) != 0.0

    def scale(self, name: str, step: int) -> float:
        """Ramp multiplier for ``name`` at ``step``: warm up, then optionally anneal out."""
        warm = int(self.warmup.get(name, 0))
        up = min(1.0, step / warm) if warm > 0 else 1.0
        dec = int(self.decay.get(name, 0))
        if dec <= 0:
            return up
        if step >= dec:
            return 0.0
        # Decay measured from the end of warm-up so the two schedules do not overlap.
        start = warm
        down = 1.0 - max(0.0, (step - start) / max(dec - start, 1))
        return up * max(0.0, min(1.0, down))


@dataclass
class LossTermTensors:
    """Differentiable audit view of the exact objective assembled for backward."""

    raw: dict[str, torch.Tensor]
    effective: dict[str, torch.Tensor]
    weights: dict[str, float]
    schedules: dict[str, float]
    total: torch.Tensor


def assemble_loss_terms(
    raw: dict[str, torch.Tensor], loss_cfg: LossConfig, step: int
) -> LossTermTensors:
    """Apply configured weights/schedules without detaching the component graphs."""
    effective: dict[str, torch.Tensor] = {}
    weights: dict[str, float] = {}
    schedules: dict[str, float] = {}
    for name, value in raw.items():
        weight = float(getattr(loss_cfg.weights, name))
        schedule = float(loss_cfg.scale(name, step))
        weights[name] = weight
        schedules[name] = schedule
        effective[name] = value * weight * schedule
    if effective:
        total = sum(effective.values())
    elif raw:
        total = next(iter(raw.values())).sum() * 0.0
    else:
        raise ValueError("cannot assemble an empty loss dictionary")
    return LossTermTensors(raw, effective, weights, schedules, total)


def loss_term_metrics(terms: LossTermTensors, prefix: str = "train") -> dict[str, float]:
    """Detached telemetry whose sum is exactly ``terms.total``."""
    metrics: dict[str, float] = {}
    for name, value in terms.raw.items():
        metrics[f"{prefix}/loss_{name}"] = float(value.detach().item())
        metrics[f"{prefix}/loss_raw/{name}"] = float(value.detach().item())
        metrics[f"{prefix}/loss_effective/{name}"] = float(
            terms.effective[name].detach().item()
        )
        metrics[f"{prefix}/loss_weight/{name}"] = terms.weights[name]
        metrics[f"{prefix}/loss_schedule/{name}"] = terms.schedules[name]
    metrics[f"{prefix}/loss_total"] = float(terms.total.detach().item())
    return metrics


@dataclass(frozen=True)
class GradientAuditResult:
    """Outcome-blind pilot gate for effective per-term gradient dominance."""

    metrics: dict[str, float]
    active_terms: tuple[str, ...]
    max_share: float
    passed: bool


def audit_effective_loss_gradients(
    terms: LossTermTensors,
    shared_modules: Mapping[str, torch.nn.Module],
    *,
    term_names: Iterable[str] | None = None,
    max_terms: int = 32,
    max_gradient_share: float = 1.0,
    fail_on_excess: bool = False,
    preserve_graph: bool = False,
) -> GradientAuditResult:
    """Measure effective term gradients without touching ``Parameter.grad``.

    This deliberately belongs to a bounded pilot/preflight, not the distributed hot
    path: it traverses one already-built objective graph once per active term.  The
    caller supplies the shared modules of scientific interest (formal v2 uses encoder,
    controllability and branch transformer). For each module and their union it reports
    the effective gradient L2 norm, cosine against the summed backward objective, and
    ``norm / sum(term norms)``. The latter is a bounded dominance share suitable for a
    preregistered initialization gate.

    ``term_names`` can exclude a stride-inactive synthetic zero (for example expansion
    on a non-gain update). A formal pilot should instead evaluate an active gain step so
    the default audits every effective configured term. By default the final autograd
    traversal releases the graph and no production graph is retained.
    """
    if max_terms < 1:
        raise ValueError("gradient audit max_terms must be positive")
    if not 0 < max_gradient_share <= 1:
        raise ValueError("max_gradient_share must lie in (0, 1]")
    if not shared_modules:
        raise ValueError("gradient audit requires at least one shared module")

    requested = set(term_names) if term_names is not None else None
    missing = requested.difference(terms.effective) if requested is not None else set()
    if missing:
        raise KeyError(f"unknown gradient-audit terms: {sorted(missing)}")
    active = [
        name
        for name, value in terms.effective.items()
        if (requested is None or name in requested)
        and terms.weights[name] * terms.schedules[name] != 0.0
        and value.requires_grad
    ]
    if not active:
        raise ValueError("gradient audit has no active differentiable loss terms")
    if len(active) > max_terms:
        raise ValueError(
            f"gradient audit has {len(active)} terms, exceeding max_terms={max_terms}"
        )

    parameters: list[torch.nn.Parameter] = []
    parameter_index: dict[int, int] = {}
    module_indices: dict[str, list[int]] = {}
    for module_name, module in shared_modules.items():
        indices: list[int] = []
        for parameter in module.parameters():
            if not parameter.requires_grad:
                continue
            identity = id(parameter)
            if identity not in parameter_index:
                parameter_index[identity] = len(parameters)
                parameters.append(parameter)
            indices.append(parameter_index[identity])
        if not indices:
            raise ValueError(f"gradient audit module {module_name!r} has no trainable parameters")
        module_indices[str(module_name)] = indices
    module_indices["shared"] = list(range(len(parameters)))

    objective_grads = torch.autograd.grad(
        terms.total,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )

    def norm(grads, indices: list[int]) -> torch.Tensor:
        values = [
            grads[index].detach().float().pow(2).sum()
            for index in indices
            if grads[index] is not None
        ]
        if not values:
            return terms.total.detach().new_zeros((), dtype=torch.float32)
        return torch.stack(values).sum().sqrt()

    def cosine(grads, reference, indices: list[int]) -> torch.Tensor:
        products = [
            (grads[index].detach().float() * reference[index].detach().float()).sum()
            for index in indices
            if grads[index] is not None and reference[index] is not None
        ]
        if not products:
            return terms.total.detach().new_zeros((), dtype=torch.float32)
        numerator = torch.stack(products).sum()
        denominator = norm(grads, indices) * norm(reference, indices)
        return numerator / denominator.clamp_min(1e-12)

    objective_norms = {
        module_name: norm(objective_grads, indices)
        for module_name, indices in module_indices.items()
    }
    term_norms: dict[str, dict[str, torch.Tensor]] = {}
    term_cosines: dict[str, dict[str, torch.Tensor]] = {}
    for offset, term_name in enumerate(active):
        retain = preserve_graph or offset < len(active) - 1
        grads = torch.autograd.grad(
            terms.effective[term_name],
            parameters,
            retain_graph=retain,
            allow_unused=True,
        )
        term_norms[term_name] = {
            module_name: norm(grads, indices)
            for module_name, indices in module_indices.items()
        }
        term_cosines[term_name] = {
            module_name: cosine(grads, objective_grads, indices)
            for module_name, indices in module_indices.items()
        }

    denominators = {
        module_name: torch.stack(
            [term_norms[term_name][module_name] for term_name in active]
        ).sum()
        for module_name in module_indices
    }
    metrics: dict[str, float] = {
        "gradient_audit/num_terms": float(len(active)),
        "gradient_audit/share_bound": float(max_gradient_share),
    }
    shared_gate_shares: list[float] = []
    for module_name in module_indices:
        metrics[f"gradient_audit/objective_norm/{module_name}"] = float(
            objective_norms[module_name].item()
        )
        for term_name in active:
            term_norm = term_norms[term_name][module_name]
            share = term_norm / denominators[module_name].clamp_min(1e-12)
            share_value = float(share.item())
            if module_name == "shared":
                shared_gate_shares.append(share_value)
            metrics[f"gradient_audit/effective_norm/{module_name}/{term_name}"] = float(
                term_norm.item()
            )
            metrics[f"gradient_audit/cosine_total/{module_name}/{term_name}"] = float(
                term_cosines[term_name][module_name].item()
            )
            metrics[f"gradient_audit/share/{module_name}/{term_name}"] = share_value
    max_share = max(shared_gate_shares, default=0.0)
    passed = max_share <= max_gradient_share
    metrics["gradient_audit/max_share"] = max_share
    metrics["gradient_audit/passed"] = float(passed)
    result = GradientAuditResult(metrics, tuple(active), max_share, passed)
    if fail_on_excess and not passed:
        raise ValueError(
            "effective loss gradient-share gate failed: "
            f"max={max_share:.6f} > bound={max_gradient_share:.6f}"
        )
    return result


def compute_branch_losses(
    model,
    batch: dict[str, torch.Tensor],
    loss_cfg: LossConfig,
    match_cfg: MatchingConfig,
    step: int = 10**9,
    return_loss_terms: bool = False,
) -> tuple:
    """Level-1 branch losses for a batch of anchors.

    Returns ``(total_loss, metrics, artifacts)``. ``artifacts`` carries tensors the
    trainer reuses for histograms and for the expansion-gain stage.
    """
    obs = batch["obs"]
    z = model.encode(obs)
    child = model.predict_children(z)
    branch = child["branch"]

    modes = gather_mode_targets(batch)
    b, c = modes["valid"].shape
    tgt_z = model.encode(modes["endpoint"])  # [B, C, D]
    tgt_q = model.q_of(tgt_z)  # [B, C, S, qd]

    cost, cost_components = branch_mode_cost(
        pred_z=child["latent"],
        pred_q=child["q"],
        pred_action=branch.action,
        pred_horizon_idx=child["horizon_idx"],
        tgt_z=tgt_z,
        tgt_q=tgt_q,
        tgt_action=modes["actions"],
        tgt_action_mask=modes["action_mask"],
        tgt_horizon_idx=modes["horizon_idx"],
        tgt_valid=modes["valid"],
        cfg=match_cfg,
        q_cdist=model.q_cdist,
        return_components=True,
    )
    branch_to_mode, mode_to_branch = match(cost, modes["valid"], match_cfg)
    tgt = gather_matched_targets(
        {
            "z": tgt_z,
            "q": tgt_q,
            "actions": modes["actions"],
            "action_mask": modes["action_mask"],
            "horizon_idx": modes["horizon_idx"],
            "mass": modes["mass"].unsqueeze(-1),
            **(
                {"metric_endpoint": modes["metric_endpoint"]}
                if "metric_endpoint" in modes
                else {}
            ),
        },
        branch_to_mode,
    )
    matched = tgt["matched"]
    target_mass = tgt["mass"].squeeze(-1)

    losses: dict[str, torch.Tensor] = {}
    auxiliary_metrics: dict[str, float] = {}
    latent_gauge = getattr(model, "latent_gauge", None)
    if latent_gauge is not None:
        gauge_loss, gauge_metrics = latent_gauge(
            z,
            tgt_z,
            modes["valid"],
            step=step,
        )
        if loss_cfg.on("latent_gauge"):
            losses["latent_gauge"] = gauge_loss
        auxiliary_metrics.update(gauge_metrics)
    elif loss_cfg.on("latent_gauge"):
        raise ValueError("active latent-gauge objective has no sealed gauge module")
    world_target_z = tgt["z"].detach() if loss_cfg.detach_world_targets else tgt["z"]
    target_scale = wl.detached_target_scale(world_target_z, matched)

    if loss_cfg.on("state"):
        losses["state"] = wl.state_loss(
            child["latent"], world_target_z, matched, target_scale=target_scale
        )
    if loss_cfg.on("action"):
        losses["action"] = wl.action_loss(branch.action, tgt["actions"], tgt["action_mask"], matched)
    if loss_cfg.on("horizon"):
        losses["horizon"] = wl.horizon_loss(branch.horizon_logits, tgt["horizon_idx"], matched)
    if loss_cfg.on("bind"):
        bind_result = wl.bind_loss(
            model,
            z,
            branch.embedding,
            tgt["actions"],
            tgt["action_mask"],
            tgt["horizon_idx"],
            world_target_z,
            matched,
            target_scale=target_scale,
            bind_negative_margin=float(loss_cfg.bind_negative_margin),
            return_metrics=True,
        )
        losses["bind"], bind_metrics = bind_result
        auxiliary_metrics.update(bind_metrics)
    if loss_cfg.on("coverage"):
        if loss_cfg.coverage_space == "q":
            losses["coverage"] = sl.coverage_loss(
                child["q"],
                tgt_q,
                modes["valid"],
                model.q_cdist,
                normalization_version=match_cfg.normalization_version,
                space="q",
                branch_to_mode=branch_to_mode,
            )
        else:
            losses["coverage"] = sl.coverage_loss(
                child["latent"].unsqueeze(2),
                tgt_z.detach().unsqueeze(2),
                modes["valid"],
                lambda a, bb: torch.cdist(a.squeeze(2), bb.squeeze(2)),
                normalization_version=match_cfg.normalization_version,
                space="z",
                branch_to_mode=branch_to_mode,
            )
    if loss_cfg.on("redundancy"):
        losses["redundancy"] = sl.redundancy_loss(
            child["q"],
            branch.keep,
            model.q_cdist,
            loss_cfg.redundancy_temperature,
            matched=matched,
        )
    if loss_cfg.on("keep"):
        losses["keep"] = sl.keep_loss(branch.keep_logit, matched, loss_cfg.keep_balance)
    if loss_cfg.on("mass"):
        losses["mass"] = sl.mass_loss(branch.mass_logit, target_mass, matched)
    if loss_cfg.on("uncertainty"):
        losses["uncertainty"] = wl.uncertainty_loss(
            branch.uncertainty,
            child["latent"],
            world_target_z,
            matched,
            target_scale=target_scale,
            balance_groups=True,
        )
    if loss_cfg.on("recursive"):
        recursive_result = wl.recursive_loss(
            model,
            child["latent"],
            world_target_z,
            matched,
            max_nodes=int(loss_cfg.recursive_batch),
            return_metrics=True,
        )
        losses["recursive"], recursive_metrics = recursive_result
        auxiliary_metrics.update(recursive_metrics)
    if loss_cfg.on("reconstruction") and model.decoder is not None:
        losses["reconstruction"] = wl.reconstruction_loss(model.decoder, z, obs)

    control_metrics: dict[str, float] = {}
    if loss_cfg.on("control"):
        n_ctrl = min(int(loss_cfg.control_batch), z.shape[0])
        sub = torch.randperm(z.shape[0], device=z.device)[:n_ctrl]
        q_anchor = model.q_of(z[sub])
        endpoint_key = str(loss_cfg.control_endpoint_key)
        used_fallback = False
        if endpoint_key not in batch:
            if not loss_cfg.control_allow_endpoint_fallback or "fut_endpoint" not in batch:
                raise KeyError(
                    f"control endpoint {endpoint_key!r} is absent and fallback is disabled"
                )
            endpoint_key = "fut_endpoint"
            used_fallback = True
        endpoints = batch[endpoint_key][sub]
        valid = batch["fut_valid"][sub]
        if loss_cfg.control_require_single_scale and q_anchor.shape[-2] != 1:
            raise ValueError(
                "formal v2 control requires exactly one q scale; got "
                f"{q_anchor.shape[-2]}"
            )
        if loss_cfg.control_objective == "contrastive":
            loss_q, control_metrics = cl.future_set_contrastive_loss(
                q_anchor, endpoints, valid, model.q_distance, loss_cfg.contrastive_temperature
            )
        elif loss_cfg.control_objective == "bootstrap":
            loss_q = cl.bootstrap_signature_loss(
                model.tree_signature, q_anchor, child["q"], matched
            )
        else:
            loss_q, control_metrics = cl.future_set_distance_loss(
                q_anchor,
                endpoints,
                valid,
                model.q_distance,
                loss_cfg.future_scale,
                target_transform=loss_cfg.control_target_transform,
                metric_weight=float(loss_cfg.control_metric_weight),
                rank_weight=float(loss_cfg.control_rank_weight),
                rank_temperature=float(loss_cfg.control_rank_temperature),
            )
        losses["control"] = loss_q
        control_metrics["control/endpoint_fallback"] = float(used_fallback)
        control_metrics["control/endpoint_dimension"] = float(endpoints.shape[-1])
        control_metrics["control/valid_endpoints_per_anchor"] = float(
            valid.float().sum(-1).mean().item()
        )

    terms = assemble_loss_terms(losses, loss_cfg, step)
    total = terms.total

    metrics = loss_term_metrics(terms)
    metrics["train/redundancy_warmup_scale"] = loss_cfg.scale("redundancy", step)
    metrics["world/latent_target_scale"] = float(target_scale.detach().item())
    metrics.update(control_metrics)
    metrics.update(auxiliary_metrics)
    metrics.update(assigned_cost_metrics(cost_components, branch_to_mode, matched, match_cfg))
    metrics.update(
        wl.prediction_metrics(
            child["latent"].detach(), tgt["z"].detach(), branch.action.detach(),
            tgt["actions"], tgt["action_mask"], branch.horizon_logits.detach(),
            tgt["horizon_idx"], matched, model.horizon_selector.horizon_values,
        )
    )
    if (
        model.decoder is not None
        and "metric_endpoint" in tgt
        and "task_metric_dims" in batch
    ):
        dims = batch["task_metric_dims"][0].long()
        if not bool((batch["task_metric_dims"] == dims.unsqueeze(0)).all()):
            raise ValueError("task_metric_dims must be identical within a batch")
        with torch.no_grad():
            decoded_child = model.decoder(child["latent"].detach())
            predicted_metric = decoded_child.index_select(-1, dims)
            endpoint_rms = (
                predicted_metric.float() - tgt["metric_endpoint"].detach().float()
            ).square().mean(-1).sqrt()
            denom = matched.sum().clamp_min(1.0)
            metrics["model/assigned_task_endpoint_rms"] = float(
                ((endpoint_rms * matched).sum() / denom).item()
            )
    metrics.update(
        sl.support_metrics(
            keep=branch.keep.detach(),
            mass_pred=branch.mass.detach(),
            matched=matched,
            target_mass=modes["mass"],
            branch_target_mass=target_mass,
            mode_valid=modes["valid"],
            mode_to_branch=mode_to_branch,
            pred_q=child["q"].detach(),
            pred_z=child["latent"].detach(),
            q_cdist=model.q_cdist,
            mass_enabled=loss_cfg.on("mass"),
        )
    )
    for key in (
        "num_modes",
        "future_diversity",
        "num_retrieved",
        "retrieval_num_candidates",
        "retrieval_num_valid",
        "retrieval_mean_distance",
        "retrieval_fallback",
        "retrieval_truncated",
        "retrieval_query_saturated",
        "modes_raw",
        "modes_retained",
        "modes_truncated",
    ):
        if key in batch:
            metrics[f"data/{key}"] = float(batch[key].float().mean().item())
    if "modes_truncated" in batch:
        metrics["data/mode_truncation_fraction"] = float(
            (batch["modes_truncated"] > 0).float().mean().item()
        )
    if "num_modes" in batch:
        metrics["data/multimode_anchor_fraction"] = float(
            (batch["num_modes"] > 1).float().mean().item()
        )

    if model.decoder is not None and "reconstruction" in losses:
        metrics["model/state_reconstruction_mse"] = metrics.pop("train/loss_reconstruction", 0.0)
        metrics["train/loss_reconstruction"] = metrics["model/state_reconstruction_mse"]

    artifacts = {
        "z": z.detach(),
        "child_q": child["q"].detach(),
        "child_z": child["latent"].detach(),
        "keep": branch.keep.detach(),
        "mass": branch.mass.detach(),
        "uncertainty": branch.uncertainty.detach(),
        "gain_prior": branch.gain_prior.detach(),
        "horizon_pred": branch.horizon_index().detach(),
        "matched": matched.detach(),
        "num_modes": batch["num_modes"],
        "future_diversity": batch["future_diversity"],
    }
    if return_loss_terms:
        return total, metrics, artifacts, terms
    return total, metrics, artifacts


def compute_expansion_gain_loss(
    model,
    z0: torch.Tensor,
    tree_cfg,
    latent_index,
    quantizer,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Supervise ``g_psi`` against data-grounded marginal coverage gain.

    The tree itself is generated without gradients; only the gain head is trained here.
    Backpropagating through a 64-node expansion would be expensive and would let the
    model reduce the loss by degrading its own tree instead of by predicting better.
    """
    from treewm.data.retrieval_index import gain_targets
    from treewm.logging.metrics import rank_correlation

    with torch.no_grad():
        tree, trace = model.generate(z0, tree_cfg, generator=generator)

        node_states = model.decoder(tree.latent) if model.decoder is not None else None
        if node_states is None:
            covered_cells, covered_valid = latent_index.query_cells(
                tree.latent.reshape(-1, tree.latent.shape[-1])
            )
            covered_cells = covered_cells[:, 0].view(tree.latent.shape[0], -1)
            covered_valid = tree.valid.float()
        else:
            covered_cells = quantizer.cell_ids(node_states)
            covered_valid = tree.valid.float()

        target = gain_targets(latent_index, tree.latent, covered_cells, covered_valid)
        valid = tree.valid.float()

    context = tree.context(tree_cfg.context_pooling) if tree_cfg.context_pooling != "none" else None
    predicted = model.gain_head(
        tree.q.float(), context.float() if context is not None else None,
        tree.depth, tree.keep_score.float(), tree.uncertainty.float(),
    )
    loss = cl.expansion_gain_loss(predicted, target, valid)

    # The per-branch G_i head of spec section 6 is the context-free prior. It is
    # supervised directly against root-child gain rather than read off tree slots,
    # because add_children reorders children by KEEP score and slot order therefore
    # does not correspond to branch order.
    root_children = model.predict_children(z0)
    with torch.no_grad():
        if model.decoder is not None:
            root_cells = quantizer.cell_ids(model.decoder(z0)).unsqueeze(1)
        else:
            root_cells, _ = latent_index.query_cells(z0)
            root_cells = root_cells[:, :1]
        root_valid = torch.ones_like(root_cells, dtype=torch.float32)
        prior_target = gain_targets(latent_index, root_children["latent"], root_cells, root_valid)
    loss = loss + cl.expansion_gain_loss(
        root_children["branch"].gain_prior, prior_target, torch.ones_like(prior_target)
    )

    with torch.no_grad():
        from treewm.evaluation.coverage import unique_cells_per_row

        sel = valid > 0
        covered = unique_cells_per_row(covered_cells, covered_valid).float()  # [B]
        nodes = valid.sum(1).clamp_min(1.0)
        metrics = {
            "expansion/predicted_gain_mean": float(predicted[sel].mean().item()),
            "expansion/target_gain_mean": float(target[sel].mean().item()),
            "expansion/gain_mae": float((predicted[sel] - target[sel]).abs().mean().item()),
            "expansion/gain_rank_correlation": rank_correlation(predicted[sel], target[sel]),
            # Distinct regions reached by the tree, and how efficiently the budget
            # bought them -- the primary compute-normalised quantity.
            "expansion/controllability_coverage": float(covered.mean().item()),
            "expansion/controllability_coverage_per_node": float((covered / nodes).mean().item()),
            "expansion/coverage_per_node": float((covered / nodes).mean().item()),
            "expansion/redundant_expansion_fraction": float(
                (1.0 - covered / nodes).clamp_min(0).mean().item()
            ),
        }
    return loss, metrics
