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
from treewm.tree.frontier import GOAL_AWARE_SCORERS


@dataclass
class PlannerConfig:
    # How a generated node is scored against the goal.
    #   decoded -- decode both to observation space and compare there (DEFAULT)
    #   latent  -- d_z(z_n, z_g), the original formulation
    # Measured on stitch over 90 episodes/cell: decoded nearly doubles success
    # (noveltyq .144 -> .367, random .256 -> .489). z is trained for dynamics and future
    # prediction, never for metric goal matching, so distances in it do not track
    # spatial proximity. The decoder is part of the model, so this is not privileged.
    score_space: str = "decoded"
    # Track F -- how a node's goal score is formed.
    #   endpoint  F0: decoded endpoint distance only
    #   path_aware F2: endpoint distance + lambda_c * cumulative path cost
    #   ancestor   F3: blend of endpoint distance with the best distance along the
    #              root-to-node path, so a deep leaf cannot win on its endpoint alone
    #              after a poor path
    score_mode: str = "endpoint"
    path_cost_weight: float = 0.02
    ancestor_weight: float = 0.5
    # Track E -- how much of the selected chunk to execute before replanning.
    #   fixed   execute_steps primitive actions
    #   clipped min(chunk length, execute_steps)  (previous behaviour)
    #   full    the whole predicted chunk
    execute_mode: str = "clipped"
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
    selected_depth: int
    goal_distance: float
    tree: object = None


class GoalPlanner:
    """Wraps a model + normaliser into a replanning controller."""

    def __init__(self, model, normalizer, tree_cfg: TreeConfig, cfg: PlannerConfig, device=None,
                 generator: torch.Generator | None = None) -> None:
        self.model = model
        self.normalizer = normalizer
        self.tree_cfg = tree_cfg
        self.cfg = cfg
        self.device = device or next(model.parameters()).device
        # Own stream: a diagnostic render must not change what the planner does.
        from treewm.utils.rng import make_generator

        self.generator = generator or make_generator(0, "planner", self.device)

    def _path_aware_score(self, tree, endpoint_score: torch.Tensor) -> torch.Tensor:
        """Aggregate goal score along each root-to-node path.

        Parent slots always precede their children, so a single forward pass over slots
        propagates cumulative path quantities without recursion.
        """
        b, n = endpoint_score.shape
        cost = tree.action_mask.sum(-1).float()  # primitive actions into each node
        cum_cost = torch.zeros_like(endpoint_score)
        best_on_path = endpoint_score.clone()
        for slot in range(1, n):
            parent = tree.parent_index[:, slot].clamp_min(0)
            rows = torch.arange(b, device=endpoint_score.device)
            cum_cost[:, slot] = cum_cost[rows, parent] + cost[:, slot]
            best_on_path[:, slot] = torch.minimum(
                best_on_path[rows, parent], endpoint_score[:, slot]
            )
        if self.cfg.score_mode == "path_aware":
            return endpoint_score + self.cfg.path_cost_weight * cum_cost
        if self.cfg.score_mode == "ancestor":
            w = self.cfg.ancestor_weight
            return (1.0 - w) * endpoint_score + w * best_on_path
        raise ValueError(f"unknown score_mode {self.cfg.score_mode!r}")

    @torch.no_grad()
    def plan(self, obs: np.ndarray, goal: np.ndarray, generator=None, return_tree: bool = False) -> PlanResult:
        generator = generator or self.generator
        model = self.model
        obs_n = torch.from_numpy(self.normalizer.norm_obs(np.asarray(obs, dtype=np.float32)[None])).to(self.device)
        goal_n = torch.from_numpy(self.normalizer.norm_obs(np.asarray(goal, dtype=np.float32)[None])).to(self.device)

        z = model.encode(obs_n)
        z_goal = model.encode(goal_n)

        tree, _ = model.generate(
            z, self.tree_cfg, generator=generator,
            goal_obs=goal_n if self.tree_cfg.scorer in GOAL_AWARE_SCORERS else None,
        )

        if self.cfg.score_space == "decoded" and model.decoder is not None:
            # Compare in observation space, where the goal metric is meaningful.
            node_obs = model.decoder(tree.latent)  # [1, N, obs_dim]
            score = torch.linalg.vector_norm(node_obs - goal_n.unsqueeze(1), dim=-1)
        else:
            score = torch.linalg.vector_norm(tree.latent - z_goal.unsqueeze(1), dim=-1)  # [1, N]

        if self.cfg.score_mode != "endpoint":
            score = self._path_aware_score(tree, score)
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
            chunk_len = int(mask.sum().item())
            if self.cfg.execute_mode == "full":
                length = max(1, chunk_len)
            elif self.cfg.execute_mode == "fixed":
                length = max(1, min(self.cfg.execute_steps, chunk_len))
            else:  # clipped
                length = max(1, min(chunk_len, self.cfg.execute_steps))
            self.last_planned_chunk = chunk_len
            actions = chunk[:length].float().cpu().numpy()
            actions = self.normalizer.denorm_act(actions)
            actions = np.clip(actions, -1.0, 1.0)

        return PlanResult(
            actions=actions.astype(np.float32),
            selected_node=best,
            path_length=len(trimmed),
            num_nodes=int(tree.num_nodes[0].item()),
            selected_depth=int(tree.depth[0, best].item()),
            goal_distance=float(score[0, best].item()),
            tree=tree if return_tree else None,
        )
