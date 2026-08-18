"""Total loss assembly.

Every term is independently switchable and independently weighted (spec section 14), so
the causal chain

    multimodal -> recursive -> controllability-aware support -> adaptive compute

can be ablated one link at a time. Disabled terms are not merely zero-weighted: their
computation is skipped, so an ablation cannot pay for a loss it is not using.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from treewm.data.future_sets import gather_mode_targets
from treewm.losses import controllability_losses as cl
from treewm.losses import support_losses as sl
from treewm.losses import world_losses as wl
from treewm.tree.matching import MatchingConfig, branch_mode_cost, gather_matched_targets, match


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
        }
    )
    control_objective: str = "future_set"  # future_set | contrastive | bootstrap
    gain_target: str = "novelty"  # novelty | retrieval
    # Track A2: probability of feeding the model its own predicted latent during the
    # multi-step rollout, warmed up linearly over scheduled_sampling_warmup steps.
    scheduled_sampling_p: float = 0.0
    scheduled_sampling_warmup: int = 2000
    # Track H2: relative weight of each recursive depth (empty -> uniform).
    multistep_depth_weights: tuple[float, ...] = ()
    redundancy_temperature: float = 0.25
    contrastive_temperature: float = 0.1
    keep_balance: bool = True
    coverage_space: str = "q"  # q | z  (ablation axis, spec section 18)
    future_scale: float = 1.0
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


def compute_branch_losses(
    model,
    batch: dict[str, torch.Tensor],
    loss_cfg: LossConfig,
    match_cfg: MatchingConfig,
    step: int = 10**9,
) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
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

    cost = branch_mode_cost(
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
        },
        branch_to_mode,
    )
    matched = tgt["matched"]
    target_mass = tgt["mass"].squeeze(-1)

    losses: dict[str, torch.Tensor] = {}
    w = loss_cfg.weights

    if loss_cfg.on("state"):
        losses["state"] = wl.state_loss(child["latent"], tgt["z"], matched)
    if loss_cfg.on("action"):
        losses["action"] = wl.action_loss(branch.action, tgt["actions"], tgt["action_mask"], matched)
    if loss_cfg.on("horizon"):
        losses["horizon"] = wl.horizon_loss(branch.horizon_logits, tgt["horizon_idx"], matched)
    if loss_cfg.on("bind"):
        losses["bind"] = wl.bind_loss(
            model, z, branch.embedding, tgt["actions"], tgt["action_mask"],
            tgt["horizon_idx"], tgt["z"], matched,
        )
    if loss_cfg.on("coverage"):
        if loss_cfg.coverage_space == "q":
            losses["coverage"] = sl.coverage_loss(child["q"], tgt_q, modes["valid"], model.q_cdist)
        else:
            # z-space coverage ablation: identical objective, different metric space.
            losses["coverage"] = sl.coverage_loss(
                child["latent"].unsqueeze(2), tgt_z.unsqueeze(2), modes["valid"],
                lambda a, bb: torch.cdist(a.squeeze(2), bb.squeeze(2)),
            )
    if loss_cfg.on("redundancy"):
        losses["redundancy"] = sl.redundancy_loss(
            child["q"], branch.keep, model.q_cdist, loss_cfg.redundancy_temperature
        )
    if loss_cfg.on("keep"):
        losses["keep"] = sl.keep_loss(branch.keep_logit, matched, loss_cfg.keep_balance)
    if loss_cfg.on("mass"):
        losses["mass"] = sl.mass_loss(branch.mass_logit, target_mass, matched)
    if loss_cfg.on("uncertainty"):
        losses["uncertainty"] = wl.uncertainty_loss(
            branch.uncertainty, child["latent"], tgt["z"], matched
        )
    if loss_cfg.on("recursive"):
        losses["recursive"] = wl.recursive_loss(
            model, child["latent"], tgt["z"], matched, max_nodes=int(loss_cfg.recursive_batch)
        )
    if loss_cfg.on("reconstruction") and model.decoder is not None:
        losses["reconstruction"] = wl.reconstruction_loss(model.decoder, z, obs)

    control_metrics: dict[str, float] = {}
    if loss_cfg.on("control"):
        n_ctrl = min(int(loss_cfg.control_batch), z.shape[0])
        sub = torch.randperm(z.shape[0], device=z.device)[:n_ctrl]
        q_anchor = model.q_of(z[sub])
        endpoints = batch["fut_endpoint"][sub]
        valid = batch["fut_valid"][sub]
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
                q_anchor, endpoints, valid, model.q_distance, loss_cfg.future_scale
            )
        losses["control"] = loss_q

    total = sum(
        getattr(w, name) * loss_cfg.scale(name, step) * value for name, value in losses.items()
    )

    metrics = {f"train/loss_{k}": float(v.detach().item()) for k, v in losses.items()}
    metrics["train/redundancy_warmup_scale"] = loss_cfg.scale("redundancy", step)
    metrics["train/loss_total"] = float(total.detach().item())
    metrics.update(control_metrics)
    metrics.update(
        wl.prediction_metrics(
            child["latent"].detach(), tgt["z"].detach(), branch.action.detach(),
            tgt["actions"], tgt["action_mask"], branch.horizon_logits.detach(),
            tgt["horizon_idx"], matched, model.horizon_selector.horizon_values,
        )
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
        )
    )
    metrics["data/num_modes"] = float(batch["num_modes"].float().mean().item())
    metrics["data/future_diversity"] = float(batch["future_diversity"].float().mean().item())

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
