"""Batched tree storage.

Trees are represented as pre-allocated tensors with integer parent indices rather than
python objects with child pointers (spec section 24). A whole batch of trees -- one per
anchor state -- expands in lockstep, so every operation is a tensor op and nothing
recurses in the hot loop.

Slot ``0`` of every tree is the root. ``parent_index`` is ``-1`` at the root and a valid
slot index elsewhere, which is what lets the planner trace a path back with a simple
while loop over integers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class BatchedTree:
    """``[B, capacity, ...]`` node storage. Field names follow spec section 24."""

    latent: torch.Tensor  # [B, N, z_dim]
    q: torch.Tensor  # [B, N, S, q_dim]
    parent_index: torch.Tensor  # [B, N] long, -1 at root
    depth: torch.Tensor  # [B, N] long
    action_chunk: torch.Tensor  # [B, N, h_max, dA]  action that leads INTO this node
    action_mask: torch.Tensor  # [B, N, h_max]
    horizon_idx: torch.Tensor  # [B, N] long
    keep_score: torch.Tensor  # [B, N]
    mass: torch.Tensor  # [B, N]
    uncertainty: torch.Tensor  # [B, N]
    expansion_gain: torch.Tensor  # [B, N]
    expanded: torch.Tensor  # [B, N] bool
    valid: torch.Tensor  # [B, N] bool
    order: torch.Tensor  # [B, N] long, -1 until created; expansion order for viz
    num_nodes: torch.Tensor  # [B] long

    @classmethod
    def initialize(
        cls,
        root_z: torch.Tensor,
        root_q: torch.Tensor,
        capacity: int,
        h_max: int,
        action_dim: int,
    ) -> "BatchedTree":
        b, z_dim = root_z.shape
        s, q_dim = root_q.shape[-2:]
        dev = root_z.device
        # Latents and q keep the model's compute dtype (bf16 under autocast), but every
        # bookkeeping field is float32. Frontier scorers are a mix -- bfs/random/
        # uncertainty produce float32 while the learned head produces bf16 under
        # autocast -- so a dtype that follows the root latent makes scatter_ fail for
        # some arms and not others.
        f32 = torch.float32

        latent = torch.zeros(b, capacity, z_dim, device=dev, dtype=root_z.dtype)
        q = torch.zeros(b, capacity, s, q_dim, device=dev, dtype=root_q.dtype)
        latent[:, 0] = root_z
        q[:, 0] = root_q

        tree = cls(
            latent=latent,
            q=q,
            parent_index=torch.full((b, capacity), -1, device=dev, dtype=torch.long),
            depth=torch.zeros(b, capacity, device=dev, dtype=torch.long),
            action_chunk=torch.zeros(b, capacity, h_max, action_dim, device=dev, dtype=f32),
            action_mask=torch.zeros(b, capacity, h_max, device=dev, dtype=f32),
            horizon_idx=torch.zeros(b, capacity, device=dev, dtype=torch.long),
            keep_score=torch.zeros(b, capacity, device=dev, dtype=f32),
            mass=torch.zeros(b, capacity, device=dev, dtype=f32),
            uncertainty=torch.zeros(b, capacity, device=dev, dtype=f32),
            expansion_gain=torch.zeros(b, capacity, device=dev, dtype=f32),
            expanded=torch.zeros(b, capacity, device=dev, dtype=torch.bool),
            valid=torch.zeros(b, capacity, device=dev, dtype=torch.bool),
            order=torch.full((b, capacity), -1, device=dev, dtype=torch.long),
            num_nodes=torch.ones(b, device=dev, dtype=torch.long),
        )
        tree.valid[:, 0] = True
        tree.order[:, 0] = 0
        return tree

    # ------------------------------------------------------------------ queries

    @property
    def batch_size(self) -> int:
        return self.latent.shape[0]

    @property
    def capacity(self) -> int:
        return self.latent.shape[1]

    def expandable_frontier(self, max_depth: int | None = None) -> torch.Tensor:
        """``[B, N]`` bool mask of nodes that exist and have not been expanded."""
        mask = self.valid & ~self.expanded
        if max_depth is not None:
            mask = mask & (self.depth < max_depth)
        return mask

    def leaves(self) -> torch.Tensor:
        """Valid nodes with no children."""
        has_child = torch.zeros_like(self.valid)
        parent = self.parent_index.clamp_min(0)
        has_child.scatter_(1, parent, self.valid & (self.parent_index >= 0))
        return self.valid & ~has_child

    def context(self, pooling: str = "mean") -> torch.Tensor:
        """Pooled tree context ``c_T = Pool({q_j})``. Returns ``[B, S, q_dim]``."""
        if pooling == "none":
            return torch.zeros_like(self.q[:, 0])
        mask = self.valid.float().view(self.batch_size, self.capacity, 1, 1)
        if pooling == "mean":
            return (self.q * mask).sum(1) / mask.sum(1).clamp_min(1.0)
        if pooling == "max":
            return self.q.masked_fill(mask == 0, -1e4).max(1).values
        raise ValueError(f"unknown pooling {pooling!r}")

    def path_to_root(self, node_idx: torch.Tensor) -> list[torch.Tensor]:
        """Trace ``[B]`` node indices back to the root.

        Returns a list of ``[B]`` index tensors ordered root -> node. Trees in the batch
        can have different depths, so shallower ones simply repeat their root index.
        """
        chains: list[torch.Tensor] = []
        cur = node_idx.clone()
        for _ in range(self.capacity):
            chains.append(cur.clone())
            parent = torch.gather(self.parent_index, 1, cur.unsqueeze(1)).squeeze(1)
            if bool((parent < 0).all()):
                break
            cur = torch.where(parent >= 0, parent, cur)
        chains.reverse()
        return chains

    def gather_nodes(self, idx: torch.Tensor) -> dict[str, torch.Tensor]:
        """Gather per-node fields at ``[B, M]`` indices."""

        def g(x: torch.Tensor) -> torch.Tensor:
            shape = idx.shape + x.shape[2:]
            expanded = idx.view(*idx.shape, *([1] * (x.dim() - 2))).expand(shape)
            return torch.gather(x, 1, expanded)

        return {
            "latent": g(self.latent),
            "q": g(self.q),
            "depth": g(self.depth),
            "action_chunk": g(self.action_chunk),
            "action_mask": g(self.action_mask),
            "horizon_idx": g(self.horizon_idx),
            "keep_score": g(self.keep_score),
            "mass": g(self.mass),
            "uncertainty": g(self.uncertainty),
            "expansion_gain": g(self.expansion_gain),
            "valid": g(self.valid),
        }

    # ------------------------------------------------------------------ mutation

    def add_children(
        self,
        parent_idx: torch.Tensor,
        child: dict[str, torch.Tensor],
        budget: int,
        step: int,
        child_valid: torch.Tensor | None = None,
        parent_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Append children, truncating so no tree ever exceeds ``budget`` nodes.

        Args:
            parent_idx: ``[B, E]`` slots being expanded.
            child: dict of ``[B, E, K, ...]`` predictions.
            budget: hard node cap for every tree in the batch.
            step: expansion iteration, recorded in ``order`` for visualisation.
            child_valid: ``[B, E, K]`` mask; children marked invalid are never admitted.
                Needed because a tree whose frontier is smaller than the expansion batch
                produces padded selections that must not turn into real nodes.
            parent_valid: ``[B, E]`` mask; only these parents are flagged expanded.

        Returns:
            ``[B, E*K]`` bool mask of children that were actually inserted.
        """
        b, e = parent_idx.shape
        k = child["latent"].shape[2]
        flat = e * k

        parent_flat = parent_idx.unsqueeze(-1).expand(b, e, k).reshape(b, flat)
        keep = child["keep_score"].reshape(b, flat).float()
        if child_valid is not None:
            keep = keep.masked_fill(child_valid.reshape(b, flat) <= 0, float("-inf"))

        # Rank children by support so that, when the last batch overflows the budget,
        # the branches dropped are the ones the model considers least supported --
        # never a silent tail truncation by slot order.
        rank = torch.argsort(keep, dim=1, descending=True)
        inv_rank = torch.empty_like(rank)
        inv_rank.scatter_(1, rank, torch.arange(flat, device=rank.device).expand(b, flat))

        room = (budget - self.num_nodes).clamp_min(0)  # [B]
        admitted = inv_rank < room.unsqueeze(1)  # [B, flat]
        if child_valid is not None:
            admitted = admitted & (child_valid.reshape(b, flat) > 0)

        # Destination slot = num_nodes + position among admitted children.
        pos = (admitted.long().cumsum(1) - 1).clamp_min(0)
        slot = (self.num_nodes.unsqueeze(1) + pos).clamp(max=self.capacity - 1)

        def _scatter(dst: torch.Tensor, src: torch.Tensor) -> None:
            src = src.reshape(b, flat, *dst.shape[2:])
            idx = slot.view(b, flat, *([1] * (dst.dim() - 2))).expand_as(src)
            mask = admitted.view(b, flat, *([1] * (dst.dim() - 2))).expand_as(src)
            current = torch.gather(dst, 1, idx)
            dst.scatter_(1, idx, torch.where(mask, src.to(dst.dtype), current))

        _scatter(self.latent, child["latent"])
        _scatter(self.q, child["q"])
        _scatter(self.action_chunk, child["action_chunk"])
        _scatter(self.action_mask, child["action_mask"])
        _scatter(self.keep_score, child["keep_score"])
        _scatter(self.mass, child["mass"])
        _scatter(self.uncertainty, child["uncertainty"])
        _scatter(self.expansion_gain, child["expansion_gain"])
        _scatter(self.horizon_idx, child["horizon_idx"])
        _scatter(self.parent_index, parent_flat)

        parent_depth = torch.gather(self.depth, 1, parent_flat)
        _scatter(self.depth, parent_depth + 1)
        _scatter(self.valid, torch.ones_like(admitted))
        _scatter(self.order, torch.full_like(parent_flat, step))

        self.num_nodes = (self.num_nodes + admitted.sum(1)).clamp(max=budget)
        flag = (
            torch.ones_like(parent_idx, dtype=torch.bool)
            if parent_valid is None
            else (parent_valid > 0)
        )
        current_expanded = torch.gather(self.expanded, 1, parent_idx)
        self.expanded.scatter_(1, parent_idx, current_expanded | flag)
        return admitted

    def to(self, device: torch.device) -> "BatchedTree":
        return BatchedTree(**{k: v.to(device) for k, v in self.__dict__.items()})

    def detach(self) -> "BatchedTree":
        return BatchedTree(
            **{k: (v.detach() if torch.is_floating_point(v) else v) for k, v in self.__dict__.items()}
        )
