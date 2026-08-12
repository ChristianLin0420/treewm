"""Goal-conditioned planning over an already-generated tree.

Deliberately simple (spec section 16). The world model builds the tree without ever
seeing the goal; the goal enters only here, to score already-generated nodes:

    J(n) = d_z(z_n, z_g)        n* = argmin_n J(n)

then the path from root to ``n*`` is traced and its first action chunk executed, after
which the environment is observed again and the tree is rebuilt. No CEM, no MCTS -- the
claim under test is about how prediction compute is *allocated*, and a strong search
procedure on top would confound it (section 28).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from treewm.tree.expansion import TreeConfig


@dataclass
class PlannerConfig:
    execute_steps: int = 16
    max_env_steps: int = 500
    use_uncertainty: bool = False
    uncertainty_weight: float = 0.0
    exclude_root: bool = True


@dataclass
class PlanResult:
    actions: np.ndarray  # [T, dA] primitive actions, in env units
    selected_node: int
    path_length: int
    num_nodes: int
    goal_distance: float
    tree: object = None


class GoalPlanner:
    """Wraps a model + normaliser into a replanning controller."""

    def __init__(self, model, normalizer, tree_cfg: TreeConfig, cfg: PlannerConfig, device=None) -> None:
        self.model = model
        self.normalizer = normalizer
        self.tree_cfg = tree_cfg
        self.cfg = cfg
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def plan(self, obs: np.ndarray, goal: np.ndarray, generator=None, return_tree: bool = False) -> PlanResult:
        model = self.model
        obs_n = torch.from_numpy(self.normalizer.norm_obs(np.asarray(obs, dtype=np.float32)[None])).to(self.device)
        goal_n = torch.from_numpy(self.normalizer.norm_obs(np.asarray(goal, dtype=np.float32)[None])).to(self.device)

        z = model.encode(obs_n)
        z_goal = model.encode(goal_n)

        tree, _ = model.generate(z, self.tree_cfg, generator=generator)

        # Score every generated node by latent distance to the goal.
        score = torch.linalg.vector_norm(tree.latent - z_goal.unsqueeze(1), dim=-1)  # [1, N]
        if self.cfg.use_uncertainty and self.cfg.uncertainty_weight != 0.0:
            score = score + self.cfg.uncertainty_weight * tree.uncertainty

        invalid = ~tree.valid
        if self.cfg.exclude_root:
            # The root is the current state: selecting it would mean "do nothing".
            invalid = invalid.clone()
            invalid[:, 0] = True
        score = score.masked_fill(invalid, float("inf"))

        best = int(score.argmin(dim=1).item())
        chain = tree.path_to_root(torch.tensor([best], device=self.device))
        path = [int(c.item()) for c in chain]
        # Drop the root and any repeats introduced by padding shallower paths.
        trimmed: list[int] = []
        for node in path:
            if node != 0 and (not trimmed or trimmed[-1] != node):
                trimmed.append(node)

        if not trimmed:
            actions = np.zeros((0, model.cfg.action_dim), dtype=np.float32)
        else:
            first = trimmed[0]
            chunk = tree.action_chunk[0, first]  # [h_max, dA]
            mask = tree.action_mask[0, first]
            length = int(mask.sum().item())
            length = max(1, min(length, self.cfg.execute_steps))
            actions = chunk[:length].float().cpu().numpy()
            actions = self.normalizer.denorm_act(actions)
            actions = np.clip(actions, -1.0, 1.0)

        return PlanResult(
            actions=actions.astype(np.float32),
            selected_node=best,
            path_length=len(trimmed),
            num_nodes=int(tree.num_nodes[0].item()),
            goal_distance=float(score[0, best].item()),
            tree=tree if return_tree else None,
        )
