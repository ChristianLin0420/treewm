"""Branch transformer and the per-branch prediction heads.

At a node with latent ``z`` the model creates ``K`` learned branch-query tokens and lets
them interact through self-attention, so siblings are *contextualised*: branch 3 can
learn "go right" partly because branches 1 and 2 already took the other options. Without
sibling attention the K heads collapse onto the dominant mode, which is exactly the
failure FlatKWM is supposed to avoid.

This module is **never** conditioned on a goal. The tree is goal-independent
(spec sections 5 and 8); the goal enters only at leaf selection in the planner.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


@dataclass
class BranchOutputs:
    """Per-branch predictions. Shapes are ``[B, K, ...]`` throughout.

    ``keep`` and ``mass`` are kept as separate fields with separate heads on purpose:
    support is "does this branch cover a distinct supported controllability future" and
    mass is "how common is this mode in the data". Conflating them is an anti-goal
    (spec sections 6, 13, 28).
    """

    embedding: torch.Tensor  # [B, K, H]      contextualised branch embeddings
    action: torch.Tensor  # [B, K, H_max, dA] action chunk A_i
    horizon_logits: torch.Tensor  # [B, K, n_h]      categorical over candidate horizons
    keep_logit: torch.Tensor  # [B, K]           kappa_i (support, NOT probability)
    mass_logit: torch.Tensor  # [B, K]           rho_i (empirical prevalence)
    uncertainty: torch.Tensor  # [B, K]           sigma_i, positive
    gain_prior: torch.Tensor  # [B, K]           G_i without tree context

    @property
    def keep(self) -> torch.Tensor:
        return torch.sigmoid(self.keep_logit)

    @property
    def mass(self) -> torch.Tensor:
        """Normalised across siblings so it is a distribution over modes."""
        return torch.softmax(self.mass_logit, dim=-1)

    def horizon_probs(self) -> torch.Tensor:
        return torch.softmax(self.horizon_logits, dim=-1)

    def horizon_index(self) -> torch.Tensor:
        return self.horizon_logits.argmax(-1)


class BranchTransformer(nn.Module):
    """``z -> K contextualised branch embeddings``."""

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
        branch_factor: int = 4,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        max_depth: int = 16,
        use_depth_embedding: bool = True,
        num_scale_embeddings: int = 0,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        self.z_dim = z_dim
        self.hidden_dim = hidden_dim
        self.branch_factor = branch_factor
        self.use_depth_embedding = use_depth_embedding

        self.node_proj = nn.Linear(z_dim, hidden_dim)
        self.branch_tokens = nn.Parameter(torch.randn(branch_factor, hidden_dim) * 0.02)
        if use_depth_embedding:
            self.depth_embedding = nn.Embedding(max_depth + 1, hidden_dim)
        self.scale_embedding = (
            nn.Embedding(num_scale_embeddings, hidden_dim) if num_scale_embeddings > 0 else None
        )

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * ffn_mult,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first and would only warn;
        # all sequences here are the same length anyway (1 + K tokens).
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.max_depth = max_depth
        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, enabled: bool = True) -> None:
        """Rematerialise transformer activations during backward when enabled."""
        self.gradient_checkpointing = bool(enabled)

    def forward(
        self,
        z: torch.Tensor,
        depth: torch.Tensor | None = None,
        scale_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """``z``: ``[B, z_dim]``; returns ``[B, K, hidden_dim]``.

        Token 0 is the node-latent token and is dropped from the output; the remaining
        K tokens are the branch embeddings.
        """
        assert z.dim() == 2, f"expected [B, z_dim], got {tuple(z.shape)}"
        b = z.shape[0]
        node_tok = self.node_proj(z).unsqueeze(1)  # [B, 1, H]
        branch_tok = self.branch_tokens.unsqueeze(0).expand(b, -1, -1)  # [B, K, H]
        tokens = torch.cat([node_tok, branch_tok], dim=1)  # [B, 1+K, H]

        if self.use_depth_embedding and depth is not None:
            d = depth.clamp(0, self.max_depth).long().view(b, 1, 1)
            tokens = tokens + self.depth_embedding(d.squeeze(-1))
        if self.scale_embedding is not None and scale_idx is not None:
            tokens = tokens + self.scale_embedding(scale_idx.long().view(b, 1))

        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            # Non-reentrant checkpointing supports modern autograd/DDP semantics and
            # preserves RNG state, so dropout masks match the non-checkpointed graph.
            out = checkpoint(self.encoder, tokens, use_reentrant=False, preserve_rng_state=True)
        else:
            out = self.encoder(tokens)
        return self.out_norm(out[:, 1:, :])


class BranchHeads(nn.Module):
    """Maps branch embeddings to the branch tuple ``(A, h, kappa, rho, sigma, G)``.

    Plain MLP heads by design: diffusion / DiT action decoders are explicitly forbidden
    before simple heads have been tested (spec sections 6 and 28).
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        h_max: int = 64,
        num_horizons: int = 5,
        head_hidden: int = 256,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.h_max = h_max
        self.num_horizons = num_horizons

        def head(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(hidden_dim, head_hidden), nn.SiLU(), nn.Linear(head_hidden, out_dim)
            )

        self.action_head = head(h_max * action_dim)
        self.horizon_head = head(num_horizons)
        self.keep_head = head(1)
        self.mass_head = head(1)
        self.uncertainty_head = head(1)
        self.gain_head = head(1)

    def forward(self, b: torch.Tensor) -> BranchOutputs:
        """``b``: ``[B, K, H]``."""
        assert b.dim() == 3, f"expected [B, K, H], got {tuple(b.shape)}"
        bsz, k, _ = b.shape
        action = self.action_head(b).view(bsz, k, self.h_max, self.action_dim)
        return BranchOutputs(
            embedding=b,
            action=action,
            horizon_logits=self.horizon_head(b),
            keep_logit=self.keep_head(b).squeeze(-1),
            mass_logit=self.mass_head(b).squeeze(-1),
            # softplus keeps sigma positive without saturating like exp at init
            uncertainty=torch.nn.functional.softplus(self.uncertainty_head(b).squeeze(-1)) + 1e-4,
            gain_prior=self.gain_head(b).squeeze(-1),
        )


class ExpansionGainHead(nn.Module):
    """Query-conditioned set scorer ``g_psi(feat_n, {feat_j}, ...) -> G``.

    Nearest-set novelty is not identifiable from a mean-pooled context: two node sets
    can have the same mean and different nearest neighbours.  The v2 head therefore
    cross-attends every candidate query to the valid tree-node feature set.  It never
    receives the exact novelty scalar as an input; that quantity remains supervision.

    A single pooled context token is still accepted for old callers, but formal v2
    passes the complete set and its validity mask.

    ``feat_dim`` selects the metric space: ``q_dim * num_scales`` for controllability
    features, ``z_dim`` for state features. The head must consume the same representation
    whose novelty it is trained to predict, or the learned arm is handicapped relative to
    its own direct heuristic for a reason unrelated to learning.
    """

    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 128,
        use_context: bool = True,
        max_depth: int = 16,
        num_attention_heads: int = 4,
    ) -> None:
        super().__init__()
        if hidden_dim % num_attention_heads != 0:
            raise ValueError("gain hidden_dim must be divisible by num_attention_heads")
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_attention_heads = num_attention_heads
        self.use_context = use_context
        self.max_depth = max_depth
        # Preserve the v1 pointwise head exactly until set-aware v2 is explicitly
        # enabled.  The extra modules are created lazily before optimiser/checkpoint
        # construction, so old configs neither change behaviour nor acquire dead
        # parameters in their state dict.
        legacy_in_dim = feat_dim + (feat_dim if use_context else 0) + 3
        self.net = nn.Sequential(
            nn.Linear(legacy_in_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.set_aware_enabled = False
        self.query_proj: nn.Linear | None = None
        self.context_proj: nn.Linear | None = None
        self.cross_attention: nn.MultiheadAttention | None = None
        self.attention_norm: nn.LayerNorm | None = None
        self.set_net: nn.Sequential | None = None

    def set_set_aware(self, enabled: bool = True) -> None:
        """Explicitly select the v2 set-aware scorer architecture.

        This must be called before constructing the optimiser or loading a v2
        checkpoint. Calling it with ``False`` retains exact v1 pointwise semantics.
        """
        enabled = bool(enabled)
        if enabled and self.query_proj is None:
            self.query_proj = nn.Linear(self.feat_dim, self.hidden_dim)
            self.context_proj = nn.Linear(self.feat_dim, self.hidden_dim)
            self.cross_attention = nn.MultiheadAttention(
                self.hidden_dim,
                self.num_attention_heads,
                dropout=0.0,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(self.hidden_dim)
            self.set_net = nn.Sequential(
                nn.Linear(2 * self.hidden_dim + 4, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Linear(self.hidden_dim, 1),
            )
            reference = next(self.net.parameters())
            for module in (
                self.query_proj,
                self.context_proj,
                self.cross_attention,
                self.attention_norm,
                self.set_net,
            ):
                module.to(device=reference.device, dtype=reference.dtype)
        self.net.requires_grad_(not enabled)
        for module in (
            self.query_proj,
            self.context_proj,
            self.cross_attention,
            self.attention_norm,
            self.set_net,
        ):
            if module is not None:
                module.requires_grad_(enabled)
        self.set_aware_enabled = enabled

    def forward(
        self,
        node_feats: torch.Tensor,
        context: torch.Tensor | None,
        depth: torch.Tensor,
        keep: torch.Tensor,
        uncertainty: torch.Tensor,
        context_valid: torch.Tensor | None = None,
        exclude_self: bool = False,
    ) -> torch.Tensor:
        """Score candidate nodes against a valid set-valued context.

        ``node_feats`` is ``[B,N,F]`` (or ``[B,N,S,D]``). ``context`` may be the
        full node set with the same trailing feature product, or one legacy pooled
        token containing exactly ``F`` values per batch item. ``context_valid`` is
        ``[B,T]`` for the context tokens.
        """
        b, n = node_feats.shape[:2]
        node_flat = node_feats.reshape(b, n, -1).float()
        if node_flat.shape[-1] != self.feat_dim:
            raise ValueError(
                f"gain node feature width {node_flat.shape[-1]} != configured {self.feat_dim}"
            )

        if not self.set_aware_enabled:
            legacy: list[torch.Tensor] = [node_flat]
            if self.use_context:
                assert context is not None, "use_context=True requires tree context"
                context_flat = context.reshape(b, -1, self.feat_dim).float()
                if context_valid is None or context_flat.shape[1] == 1:
                    pooled = context_flat.mean(1)
                else:
                    valid = context_valid.to(context.device, dtype=torch.bool)
                    weight = valid.float().unsqueeze(-1)
                    pooled = (context_flat * weight).sum(1) / weight.sum(1).clamp_min(1.0)
                legacy.append(pooled.unsqueeze(1).expand(b, n, self.feat_dim))
            legacy.extend(
                (
                    (depth.float() / max(float(self.max_depth), 1.0)).unsqueeze(-1),
                    keep.float().unsqueeze(-1),
                    uncertainty.float().unsqueeze(-1),
                )
            )
            return self.net(torch.cat(legacy, dim=-1)).squeeze(-1)

        assert self.query_proj is not None
        assert self.context_proj is not None
        assert self.cross_attention is not None
        assert self.attention_norm is not None
        assert self.set_net is not None
        if self.use_context:
            assert context is not None, "use_context=True requires a pooled tree context"
            if context.numel() % (b * self.feat_dim) != 0:
                raise ValueError("gain context cannot be reshaped into feature tokens")
            context_flat = context.reshape(b, -1, self.feat_dim).float()
            tokens = context_flat.shape[1]
            if context_valid is None:
                if tokens == 1:
                    context_valid = torch.ones(b, 1, device=context.device, dtype=torch.bool)
                else:
                    raise ValueError(
                        "multi-token gain context requires an explicit validity mask"
                    )
            else:
                context_valid = context_valid.to(device=context.device, dtype=torch.bool)
                if context_valid.shape != (b, tokens):
                    raise ValueError(
                        f"gain context_valid {tuple(context_valid.shape)} != {(b, tokens)}"
                    )
                if not bool(context_valid.any(dim=1).all()):
                    raise ValueError("every gain context needs at least one valid node")

            query = self.query_proj(node_flat)
            key_value = self.context_proj(context_flat)
            attention_mask = None
            if exclude_self:
                if n != tokens:
                    raise ValueError(
                        "exclude_self requires aligned query and context node sets"
                    )
                # Novelty excludes the candidate itself.  Mask the aligned diagonal
                # whenever another valid token exists, but leave the sole valid root
                # unmasked so MultiheadAttention never receives an all-masked row.
                attention_mask = torch.zeros(
                    b, n, tokens, device=context.device, dtype=torch.bool
                )
                diagonal = torch.arange(n, device=context.device)
                has_alternative = context_valid.sum(-1) > 1
                attention_mask[:, diagonal, diagonal] = (
                    has_alternative.unsqueeze(-1) & context_valid
                )
                attention_mask = attention_mask.repeat_interleave(
                    self.num_attention_heads, dim=0
                )
            attended, _ = self.cross_attention(
                query,
                key_value,
                key_value,
                key_padding_mask=~context_valid,
                attn_mask=attention_mask,
                need_weights=False,
            )
            attended = self.attention_norm(query + attended)
            context_fraction = context_valid.float().mean(-1, keepdim=True)
        else:
            query = self.query_proj(node_flat)
            attended = torch.zeros_like(query)
            context_fraction = torch.zeros(b, 1, device=node_flat.device)

        metadata = torch.stack(
            (
                depth.float() / max(float(self.max_depth), 1.0),
                keep.float(),
                uncertainty.float(),
            ),
            dim=-1,
        )
        context_fraction = context_fraction.unsqueeze(1).expand(b, n, 1)
        return self.set_net(
            torch.cat((query, attended, metadata, context_fraction), dim=-1)
        ).squeeze(-1)
