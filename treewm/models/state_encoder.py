"""State encoder / decoder.

``StateEncoder`` is deliberately a plain MLP over a flat observation vector. The only
contract the rest of the codebase relies on is ``(obs_dim, z_dim, forward(obs) -> z)``,
so swapping in a CNN for pixel observations later is a drop-in replacement and requires
no changes to the tree, losses or planner (spec section 1).

``z`` must stay rich enough for dynamics, future-state prediction and goal matching, so
it is **not** normalised or bottlenecked toward controllability -- that is ``q``'s job,
and collapsing the two is an explicit anti-goal (section 28).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def mlp(
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    num_hidden: int = 2,
    layer_norm_first: bool = True,
    activation: type[nn.Module] = nn.SiLU,
) -> nn.Sequential:
    """``Linear -> [LayerNorm] -> SiLU -> (Linear -> SiLU) * (n-1) -> Linear``."""
    layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dim)]
    if layer_norm_first:
        layers.append(nn.LayerNorm(hidden_dim))
    layers.append(activation())
    for _ in range(max(0, num_hidden - 1)):
        layers += [nn.Linear(hidden_dim, hidden_dim), activation()]
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


class StateEncoder(nn.Module):
    """``z_t = E_phi(s_t)``."""

    def __init__(self, obs_dim: int, z_dim: int = 128, hidden_dim: int = 256, num_hidden: int = 2) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.z_dim = z_dim
        self.net = mlp(obs_dim, hidden_dim, z_dim, num_hidden=num_hidden)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        assert obs.shape[-1] == self.obs_dim, f"expected obs dim {self.obs_dim}, got {obs.shape[-1]}"
        flat = obs.reshape(-1, self.obs_dim)
        z = self.net(flat)
        return z.view(*obs.shape[:-1], self.z_dim)


class StateDecoder(nn.Module):
    """``D(E(s)) ~ s``. Optional; used only when ``model.reconstruction`` is enabled.

    Its purpose is diagnostic -- it makes ``model/state_reconstruction_mse`` and the XY
    tree plots possible by mapping predicted latents back to physical positions.
    """

    def __init__(self, z_dim: int, obs_dim: int, hidden_dim: int = 256, num_hidden: int = 2) -> None:
        super().__init__()
        self.z_dim = z_dim
        self.obs_dim = obs_dim
        self.net = mlp(z_dim, hidden_dim, obs_dim, num_hidden=num_hidden)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        flat = z.reshape(-1, self.z_dim)
        s = self.net(flat)
        return s.view(*z.shape[:-1], self.obs_dim)


class RandomProjection(nn.Module):
    """Frozen random projection ``z -> R^{out_dim}``.

    This is the dimension-matched control for the q-vs-z comparison. ``q = C(z)`` is a
    deterministic function of ``z`` and therefore cannot contain more information about
    the future than ``z`` does; any retrieval advantage is a claim about *metric
    geometry*. Comparing ``d_q`` against ``d_z`` alone would largely measure that 32
    dimensions beat 128 at nearest-neighbour retrieval, so the honest baseline is a
    random projection of ``z`` to exactly ``q_dim``.
    """

    def __init__(self, in_dim: int, out_dim: int, seed: int = 0) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        weight = torch.randn(out_dim, in_dim, generator=generator) / (in_dim**0.5)
        self.register_buffer("weight", weight)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(z, self.weight)
