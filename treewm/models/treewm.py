"""TreeWM: the recursive branch operator plus tree generation.

One branch-generation network is reused at every depth (spec section 7): there are no
per-depth parameters, which is what makes "recursive prediction" a real claim rather
than a stack of independently trained levels. Depth enters only as an embedding.

Tree construction never sees a goal. ``generate`` takes a latent and a budget; the goal
appears only in :mod:`treewm.planning.goal_planner`, after the tree exists
(spec sections 8 and 28).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from treewm.models.branch_transformer import (
    BranchHeads,
    BranchOutputs,
    BranchTransformer,
    ExpansionGainHead,
)
from treewm.models.controllability_encoder import ControllabilityEncoder, ScaleSpec, TreeSignature
from treewm.models.latent_dynamics import LatentDynamics, SoftHorizonSelector
from treewm.models.state_encoder import RandomProjection, StateDecoder, StateEncoder
from treewm.tree.expansion import TreeConfig, generate_tree


@dataclass
class TreeWMConfig:
    obs_dim: int = 2
    action_dim: int = 2
    z_dim: int = 128
    q_dim: int = 64
    hidden_dim: int = 256
    encoder_hidden: int = 256
    num_layers: int = 3
    num_heads: int = 4
    branch_factor: int = 4
    h_max: int = 64
    horizons: tuple[int, ...] = (4, 8, 16, 32, 64)
    scales: tuple[tuple[str, int, float], ...] = (("short", 8, 1.0), ("medium", 32, 1.0))
    max_depth: int = 16
    dropout: float = 0.0
    reconstruction: bool = True
    residual_dynamics: bool = True
    normalize_q: bool = True
    use_tree_context: bool = True
    use_depth_embedding: bool = True

    @property
    def num_horizons(self) -> int:
        return len(self.horizons)


def horizon_mask(h_idx: torch.Tensor, horizons: torch.Tensor, h_max: int) -> torch.Tensor:
    """Build ``[..., h_max]`` masks from horizon indices."""
    lengths = horizons[h_idx.long().clamp(0, len(horizons) - 1)]
    steps = torch.arange(h_max, device=h_idx.device).view(*([1] * h_idx.dim()), h_max)
    return (steps < lengths.unsqueeze(-1)).float()


class TreeWM(nn.Module):
    """The full model. Baselines subclass or reuse its components with matched capacity."""

    arm_name: str = "treewm"
    default_scorer: str = "learned"

    def __init__(self, cfg: TreeWMConfig) -> None:
        super().__init__()
        self.cfg = cfg
        scales = tuple(ScaleSpec(name, steps, weight) for name, steps, weight in cfg.scales)

        self.encoder = StateEncoder(cfg.obs_dim, cfg.z_dim, cfg.encoder_hidden)
        self.decoder = StateDecoder(cfg.z_dim, cfg.obs_dim, cfg.encoder_hidden) if cfg.reconstruction else None
        self.controllability = ControllabilityEncoder(
            cfg.z_dim, cfg.q_dim, cfg.hidden_dim, scales=scales, normalize=cfg.normalize_q
        )
        self.branch_transformer = BranchTransformer(
            z_dim=cfg.z_dim,
            hidden_dim=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            num_heads=cfg.num_heads,
            branch_factor=cfg.branch_factor,
            dropout=cfg.dropout,
            max_depth=cfg.max_depth,
            use_depth_embedding=cfg.use_depth_embedding,
        )
        self.heads = BranchHeads(
            hidden_dim=cfg.hidden_dim,
            action_dim=cfg.action_dim,
            h_max=cfg.h_max,
            num_horizons=cfg.num_horizons,
        )
        self.dynamics = LatentDynamics(
            z_dim=cfg.z_dim,
            action_dim=cfg.action_dim,
            branch_dim=cfg.hidden_dim,
            h_max=cfg.h_max,
            num_horizons=cfg.num_horizons,
            hidden_dim=cfg.hidden_dim,
            residual=cfg.residual_dynamics,
        )
        self.gain_head = ExpansionGainHead(
            q_dim=cfg.q_dim,
            num_scales=self.controllability.num_scales,
            use_context=cfg.use_tree_context,
            max_depth=cfg.max_depth,
        )
        self.horizon_selector = SoftHorizonSelector(cfg.horizons)
        # Only used when losses.control_objective == "bootstrap" (spec section 15,
        # option 1); built unconditionally so checkpoints stay interchangeable.
        self.tree_signature = TreeSignature(
            q_dim=cfg.q_dim,
            num_scales=self.controllability.num_scales,
            out_dim=cfg.q_dim * self.controllability.num_scales,
        )
        # Dimension-matched control for the q-vs-z retrieval diagnostic; frozen, never
        # trained, never used by any loss.
        self.z_control_projection = RandomProjection(cfg.z_dim, cfg.q_dim * len(scales))
        self.register_buffer("horizons", torch.tensor(cfg.horizons, dtype=torch.long))

    # ------------------------------------------------------------------ encoders

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def q_of(self, z: torch.Tensor) -> torch.Tensor:
        return self.controllability(z)

    def q_distance(self, qa: torch.Tensor, qb: torch.Tensor) -> torch.Tensor:
        return self.controllability.distance(qa, qb)

    def q_cdist(self, qa: torch.Tensor, qb: torch.Tensor) -> torch.Tensor:
        return self.controllability.cdist(qa, qb)

    # -------------------------------------------------------------------- branch

    def branch(self, z: torch.Tensor, depth: torch.Tensor | None = None) -> BranchOutputs:
        if depth is None:
            depth = torch.zeros(z.shape[0], device=z.device, dtype=torch.long)
        emb = self.branch_transformer(z, depth=depth)
        return self.heads(emb)

    def predict_children(
        self,
        z: torch.Tensor,
        depth: torch.Tensor | None = None,
        horizon_override: torch.Tensor | None = None,
        action_override: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """One application of the recursive operator ``T_theta(z)``.

        ``horizon_override`` / ``action_override`` are the teacher-forcing hooks used by
        the bind loss: feeding the *target* action chunk through the same dynamics model
        checks that the successor prediction genuinely depends on the actions rather
        than being read off ``z`` alone.
        """
        out = self.branch(z, depth)
        h_idx = out.horizon_index() if horizon_override is None else horizon_override
        actions = out.action if action_override is None else action_override
        mask = horizon_mask(h_idx, self.horizons, self.cfg.h_max)

        z_next = self.dynamics(z, actions, mask, h_idx, out.embedding)
        q_next = self.q_of(z_next)
        return {
            "branch": out,
            "latent": z_next,
            "q": q_next,
            "action_chunk": actions,
            "action_mask": mask,
            "horizon_idx": h_idx,
            "keep_score": out.keep,
            "mass": out.mass,
            "uncertainty": out.uncertainty,
            "expansion_gain": out.gain_prior,
        }

    @torch.no_grad()
    def expand_nodes(self, z: torch.Tensor, depth: torch.Tensor) -> dict[str, torch.Tensor]:
        """Tree-facing wrapper: detached tensors only, shaped ``[M, K, ...]``."""
        child = self.predict_children(z, depth)
        return {
            "latent": child["latent"],
            "q": child["q"],
            "action_chunk": child["action_chunk"],
            "action_mask": child["action_mask"],
            "horizon_idx": child["horizon_idx"],
            "keep_score": child["keep_score"],
            "mass": child["mass"],
            "uncertainty": child["uncertainty"],
            "expansion_gain": child["expansion_gain"],
        }

    # ---------------------------------------------------------------- generation

    @torch.no_grad()
    def generate(
        self,
        z0: torch.Tensor,
        tree_cfg: TreeConfig,
        generator: torch.Generator | None = None,
        on_iteration=None,
    ):
        """Goal-independent tree generation under a node budget."""
        q0 = self.q_of(z0)
        return generate_tree(
            self,
            z0,
            q0,
            tree_cfg,
            q_distance=self.q_distance,
            gain_head=self.gain_head,
            generator=generator,
            on_iteration=on_iteration,
            h_max=self.cfg.h_max,
            action_dim=self.cfg.action_dim,
        )

    def generate_from_obs(self, obs: torch.Tensor, tree_cfg: TreeConfig, **kwargs):
        return self.generate(self.encode(obs), tree_cfg, **kwargs)

    # --------------------------------------------------------------------- misc

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
