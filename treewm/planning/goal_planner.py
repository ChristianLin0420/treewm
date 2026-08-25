"""Goal-conditioned planning over an already-generated tree.

Deliberately simple (spec section 16). The world model builds the tree without ever
seeing the goal; the goal enters only here, to score already-generated nodes. The
historical planner uses L2 distance between normalised decoded observations, while the
``domain_raw`` metric uses each domain's task coordinates and units. The path from the
best node back to the root is then traced and its first action chunk executed, after
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


_ONEHOT_TIEBREAK_SCALE = 1.0e-3


def decoded_goal_scores(
    node_obs_n: torch.Tensor,
    goal_obs_n: torch.Tensor,
    *,
    decoded_metric: str,
    goal_dims: torch.Tensor | None = None,
    goal_metric: str = "l2",
    subgoals: tuple[tuple[int, int], ...] = (),
    obs_mean: torch.Tensor | None = None,
    obs_std: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score batched decoded nodes against batched goals.

    Args:
        node_obs_n: Normalised decoded observations with shape ``[B, N, D]``.
        goal_obs_n: Normalised goal observations with shape ``[B, D]``.
        decoded_metric: ``normalized_l2`` preserves the historical planner exactly;
            ``domain_raw`` first restores raw observation units and then applies the
            domain's metric.
        goal_dims: Task-coordinate indices. ``None`` means all observation dimensions.
        goal_metric: Domain metric, either ``l2`` or ``onehot``.
        subgoals: Slices into the selected task coordinates, one per categorical cell.
        obs_mean: Per-coordinate observation mean, required by ``domain_raw``.
        obs_std: Per-coordinate observation standard deviation, required by
            ``domain_raw``.

    The one-hot score is lexicographic: its integer part is the exact Hamming distance
    used by :class:`treewm.evaluation.domains.Domain`, and a bounded value below
    ``1e-3`` breaks ties using decoded confidence. Consequently no confidence difference
    can make an extra categorical mismatch look preferable.
    """
    if node_obs_n.ndim != 3:
        raise ValueError(f"node_obs_n must have shape [B, N, D], got {node_obs_n.shape}")
    if goal_obs_n.ndim != 2:
        raise ValueError(f"goal_obs_n must have shape [B, D], got {goal_obs_n.shape}")
    if node_obs_n.shape[0] != goal_obs_n.shape[0]:
        raise ValueError("node and goal batch sizes differ")
    if node_obs_n.shape[-1] != goal_obs_n.shape[-1]:
        raise ValueError("node and goal observation dimensions differ")

    if decoded_metric == "normalized_l2":
        node_metric = node_obs_n
        goal_metric_obs = goal_obs_n
    elif decoded_metric == "domain_raw":
        if obs_mean is None or obs_std is None:
            raise ValueError("domain_raw decoded scoring requires obs_mean and obs_std")
        # Planner evaluation may run under bf16 autocast. Raw distances and the small
        # categorical tie-break need fp32 precision, but remain on the model's device.
        node_metric = node_obs_n.float()
        goal_metric_obs = goal_obs_n.float()
        mean = obs_mean.to(device=node_obs_n.device, dtype=torch.float32)
        std = obs_std.to(device=node_obs_n.device, dtype=torch.float32)
        if (
            mean.ndim != 1
            or std.ndim != 1
            or mean.shape[0] != node_obs_n.shape[-1]
            or std.shape[0] != node_obs_n.shape[-1]
        ):
            raise ValueError("normalizer statistics must be vectors of observation dimension")
        node_metric = node_metric * std + mean
        goal_metric_obs = goal_metric_obs * std + mean
    else:
        raise ValueError(f"unknown decoded_metric {decoded_metric!r}")

    if goal_dims is not None:
        dims = goal_dims.to(device=node_obs_n.device, dtype=torch.long)
        node_metric = node_metric.index_select(-1, dims)
        goal_metric_obs = goal_metric_obs.index_select(-1, dims)
    goal_metric_obs = goal_metric_obs.unsqueeze(1)

    # The legacy mode intentionally ignores categorical domain semantics. This is the
    # exact normalised L2 behaviour used by historical checkpoints and evaluations.
    if decoded_metric == "normalized_l2" or goal_metric == "l2":
        return torch.linalg.vector_norm(node_metric - goal_metric_obs, dim=-1)

    if goal_metric != "onehot":
        raise ValueError(f"unknown domain goal_metric {goal_metric!r}")
    if not subgoals:
        raise ValueError("onehot decoded scoring requires non-empty domain subgoals")

    leading = node_metric.shape[:-1]
    hamming = torch.zeros(leading, dtype=node_metric.dtype, device=node_metric.device)
    soft_error = torch.zeros_like(hamming)
    for lo, hi in subgoals:
        if not (0 <= lo < hi <= node_metric.shape[-1]):
            raise ValueError(f"invalid onehot subgoal slice {(lo, hi)}")
        node_block = node_metric[..., lo:hi]
        goal_block = goal_metric_obs[..., lo:hi]
        hamming = hamming + (node_block.argmax(-1) != goal_block.argmax(-1)).to(hamming.dtype)
        soft_error = soft_error + (node_block - goal_block).square().mean(-1)

    # Map arbitrary decoder errors into [0, 1), then reserve less than 1e-3 for the
    # tie-break. A one-cell Hamming advantage therefore always dominates confidence.
    soft_error = soft_error / float(len(subgoals))
    bounded_tiebreak = soft_error / (1.0 + soft_error)
    return hamming + _ONEHOT_TIEBREAK_SCALE * bounded_tiebreak


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
    # Metric used when score_space=decoded.
    #   normalized_l2 -- historical L2 over normalised task coordinates
    #   domain_raw    -- domain metric over denormalised task coordinates
    decoded_metric: str = "normalized_l2"
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
                 generator: torch.Generator | None = None, domain=None) -> None:
        self.model = model
        self.normalizer = normalizer
        self.tree_cfg = tree_cfg
        self.cfg = cfg
        self.device = device or next(model.parameters()).device
        # Restrict decoded scoring to the dims the task constrains (see plan()).
        self.domain = domain
        self.goal_dims = (
            torch.tensor(domain.goal_dims, dtype=torch.long, device=self.device)
            if domain is not None else None
        )
        self.obs_mean = (
            torch.as_tensor(normalizer.obs_mean, dtype=torch.float32, device=self.device)
            if normalizer is not None else None
        )
        self.obs_std = (
            torch.as_tensor(normalizer.obs_std, dtype=torch.float32, device=self.device)
            if normalizer is not None else None
        )
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
            # Compare in observation space, optionally in the domain's raw units and
            # categorical semantics. Goal dimensions exclude unconstrained state such as
            # puzzle proprioception and velocities.
            node_obs = model.decoder(tree.latent)  # [1, N, obs_dim]
            if self.cfg.decoded_metric == "domain_raw" and self.domain is None:
                raise ValueError("planner.decoded_metric=domain_raw requires a Domain adapter")
            score = decoded_goal_scores(
                node_obs,
                goal_n,
                decoded_metric=self.cfg.decoded_metric,
                goal_dims=self.goal_dims,
                goal_metric=self.domain.goal_metric if self.domain is not None else "l2",
                subgoals=self.domain.subgoals if self.domain is not None else (),
                obs_mean=self.obs_mean,
                obs_std=self.obs_std,
            )
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
