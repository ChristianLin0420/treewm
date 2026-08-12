"""Latent dynamics ``z' = F_theta(z, A, h, b)``.

Future-state prediction is deliberately *not* independent of the action prediction. The
factorisation is

    branch token b  ->  action chunk A  ->  F(z, A, h, b)  ->  z'

so a branch cannot claim a consequence that its own actions do not produce
(spec section 6). The action chunk is a real input, which is what makes the bind loss
meaningful: feeding a *different* branch's action chunk must move the prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from treewm.models.state_encoder import mlp


class ActionChunkEncoder(nn.Module):
    """Encodes a padded action chunk plus its horizon into a fixed vector.

    The mask is applied before flattening: padded timesteps must contribute nothing, or
    the dynamics model can read the chunk length off the padding pattern instead of the
    horizon embedding.
    """

    def __init__(self, action_dim: int, h_max: int, out_dim: int = 128, hidden_dim: int = 256) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.h_max = h_max
        self.out_dim = out_dim
        self.net = mlp(h_max * action_dim, hidden_dim, out_dim, num_hidden=1)

    def forward(self, actions: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """``actions``: ``[..., h_max, dA]``, ``mask``: ``[..., h_max]``."""
        assert actions.shape[-2:] == (self.h_max, self.action_dim), (
            f"expected [..., {self.h_max}, {self.action_dim}], got {tuple(actions.shape)}"
        )
        if mask is not None:
            actions = actions * mask.unsqueeze(-1)
        lead = actions.shape[:-2]
        flat = actions.reshape(-1, self.h_max * self.action_dim)
        return self.net(flat).view(*lead, self.out_dim)


class LatentDynamics(nn.Module):
    """``z'_i = F_theta(z, A_i, h_i, b_i)``."""

    def __init__(
        self,
        z_dim: int,
        action_dim: int,
        branch_dim: int,
        h_max: int = 64,
        num_horizons: int = 5,
        hidden_dim: int = 256,
        action_embed_dim: int = 128,
        num_hidden: int = 2,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.residual = residual
        self.num_horizons = num_horizons
        self.action_encoder = ActionChunkEncoder(action_dim, h_max, action_embed_dim, hidden_dim)
        self.horizon_embedding = nn.Embedding(num_horizons, action_embed_dim)
        in_dim = z_dim + action_embed_dim * 2 + branch_dim
        self.net = mlp(in_dim, hidden_dim, z_dim, num_hidden=num_hidden)

    def forward(
        self,
        z: torch.Tensor,
        actions: torch.Tensor,
        action_mask: torch.Tensor,
        horizon_idx: torch.Tensor,
        branch_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Predict successor latents.

        Args:
            z: ``[B, z_dim]`` parent latent (broadcast across branches).
            actions: ``[B, K, h_max, dA]``
            action_mask: ``[B, K, h_max]``
            horizon_idx: ``[B, K]`` index into the candidate-horizon list.
            branch_embedding: ``[B, K, branch_dim]``

        Returns:
            ``[B, K, z_dim]``
        """
        assert z.dim() == 2 and actions.dim() == 4, "z must be [B, D] and actions [B, K, h_max, dA]"
        b, k = actions.shape[:2]
        a_emb = self.action_encoder(actions, action_mask)  # [B, K, E]
        h_emb = self.horizon_embedding(horizon_idx.long().clamp(0, self.num_horizons - 1))  # [B, K, E]
        z_exp = z.unsqueeze(1).expand(b, k, self.z_dim)
        feats = torch.cat([z_exp, a_emb, h_emb, branch_embedding], dim=-1)
        delta = self.net(feats.reshape(b * k, -1)).view(b, k, self.z_dim)
        return z_exp + delta if self.residual else delta


class SoftHorizonSelector(nn.Module):
    """Differentiable expected horizon, for metrics and for the horizon MAE.

    Selecting a horizon by ``argmax`` is non-differentiable, so the horizon head is
    trained with cross-entropy against the matched target while this expectation is used
    where a continuous value is wanted (``model/horizon_mae``).
    """

    def __init__(self, horizons: tuple[int, ...]) -> None:
        super().__init__()
        self.register_buffer("horizon_values", torch.tensor(horizons, dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return (torch.softmax(logits, dim=-1) * self.horizon_values).sum(-1)
