"""Controllability representation ``q = C(z)``.

``q`` describes *future controllability structure*, not physical state identity: two
states with ``z_a != z_b`` should have ``q_a ~ q_b`` when their reachable-future sets
look alike. Coverage and redundancy therefore operate primarily in q-space
(spec section 10).

Multi-scale by construction: one head per temporal scale (short/medium/long ~ 8/32/128
steps). The first experiments may use one or two scales, but the API is multi-scale
from the start so nothing downstream needs restructuring later (section 4).

Two training signals are supported and can be compared directly (section 15):

  ``future_set_contrastive``  states are close in q iff their retrieved future sets are
                              close -- supervised by the data, available immediately.
  ``bootstrap``               close in q iff a deeper teacher expansion produces a
                              similar tree signature -- self-referential, so it is off
                              by default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from treewm.models.state_encoder import mlp


@dataclass
class ScaleSpec:
    """One temporal scale of the controllability representation."""

    name: str
    steps: int
    weight: float = 1.0


DEFAULT_SCALES: tuple[ScaleSpec, ...] = (
    ScaleSpec("short", 8, 1.0),
    ScaleSpec("medium", 32, 1.0),
    ScaleSpec("long", 128, 1.0),
)


class ControllabilityEncoder(nn.Module):
    """``q = C_eta(z)``, returning one embedding per temporal scale.

    Output shape is ``[..., S, q_dim]`` where ``S`` is the number of scales. Embeddings
    are L2-normalised when ``normalize`` is set, which keeps ``d_q`` bounded in
    ``[0, 2]`` and stops the redundancy and coverage losses from being gamed by
    inflating the norm of q.
    """

    def __init__(
        self,
        z_dim: int,
        q_dim: int = 64,
        hidden_dim: int = 256,
        num_hidden: int = 2,
        scales: tuple[ScaleSpec, ...] = DEFAULT_SCALES,
        normalize: bool = True,
        shared_trunk: bool = True,
    ) -> None:
        super().__init__()
        assert len(scales) >= 1, "at least one controllability scale is required"
        self.z_dim = z_dim
        self.q_dim = q_dim
        self.scales = tuple(scales)
        self.normalize = normalize
        self.shared_trunk = shared_trunk

        if shared_trunk:
            self.trunk = mlp(z_dim, hidden_dim, hidden_dim, num_hidden=num_hidden)
            self.heads = nn.ModuleList([nn.Linear(hidden_dim, q_dim) for _ in self.scales])
        else:
            self.trunk = nn.Identity()
            self.heads = nn.ModuleList(
                [mlp(z_dim, hidden_dim, q_dim, num_hidden=num_hidden) for _ in self.scales]
            )

        weights = torch.tensor([s.weight for s in self.scales], dtype=torch.float32)
        self.register_buffer("scale_weights", weights / weights.sum())

    @property
    def num_scales(self) -> int:
        return len(self.scales)

    @property
    def scale_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.scales)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``[..., z_dim] -> [..., S, q_dim]``."""
        flat = z.reshape(-1, self.z_dim)
        h = self.trunk(flat)
        qs = [head(h if self.shared_trunk else flat) for head in self.heads]
        q = torch.stack(qs, dim=-2)  # [N, S, q_dim]
        if self.normalize:
            q = F.normalize(q, dim=-1)
        return q.view(*z.shape[:-1], self.num_scales, self.q_dim)

    def as_dict(self, q: torch.Tensor) -> dict[str, torch.Tensor]:
        """Convenience view: ``{"short": ..., "medium": ..., "long": ...}``."""
        return {name: q[..., i, :] for i, name in enumerate(self.scale_names)}

    def distance(self, qa: torch.Tensor, qb: torch.Tensor) -> torch.Tensor:
        """Scale-weighted controllability distance.

        ``d_q = w_s d(q_s^a, q_s^b) + w_m d(q_m^a, q_m^b) + w_l d(q_l^a, q_l^b)``
        with the per-scale distances being plain L2. Inputs broadcast over leading dims
        and must agree on ``[..., S, q_dim]``.
        """
        assert qa.shape[-2:] == qb.shape[-2:] == (self.num_scales, self.q_dim)
        per_scale = torch.linalg.vector_norm(qa - qb, dim=-1)  # [..., S]
        return (per_scale * self.scale_weights).sum(-1)

    def cdist(self, qa: torch.Tensor, qb: torch.Tensor) -> torch.Tensor:
        """Pairwise distances: ``[B, N, S, D] x [B, M, S, D] -> [B, N, M]``."""
        assert qa.dim() == 4 and qb.dim() == 4, "expected [B, N, S, D] inputs"
        b, n, s, d = qa.shape
        m = qb.shape[1]
        # cdist per scale, then weighted sum -- cheaper than materialising [B,N,M,S,D].
        out = qa.new_zeros(b, n, m)
        for i in range(s):
            out = out + self.scale_weights[i] * torch.cdist(qa[:, :, i, :], qb[:, :, i, :])
        return out


class TreeSignature(nn.Module):
    """Summarises a node's subtree into a fixed vector (bootstrap q target, option 1).

    Kept deliberately trivial -- mean/max pooling over child q embeddings plus the
    child count -- because a graph network here is explicitly out of scope for v1
    (spec section 9).
    """

    def __init__(self, q_dim: int, num_scales: int, out_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.q_dim = q_dim
        self.num_scales = num_scales
        self.net = mlp(2 * num_scales * q_dim + 1, hidden_dim, out_dim, num_hidden=1)

    def forward(self, child_q: torch.Tensor, child_valid: torch.Tensor) -> torch.Tensor:
        """``child_q``: ``[B, N, S, D]``; ``child_valid``: ``[B, N]``."""
        b, n, s, d = child_q.shape
        mask = (child_valid > 0).float().view(b, n, 1, 1)
        count = mask.sum((1, 2, 3)).clamp_min(1.0)
        mean = (child_q * mask).sum(1) / count.view(b, 1, 1)
        maxed = (child_q.masked_fill(mask == 0, -1e4)).max(1).values
        feat = torch.cat([mean.reshape(b, -1), maxed.reshape(b, -1), count.view(b, 1)], dim=-1)
        return self.net(feat)
