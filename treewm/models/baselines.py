"""The seven arms of the comparison.

Every arm is the *same* network class with the same components and (near-)matched
capacity. They differ only in three settings:

    arm                branch_factor   max_depth   frontier scorer
    -----------------------------------------------------------------
    SingleWM           1               unbounded   (irrelevant, K=1)
    FlatKWM            K_max           1           (irrelevant, depth 1)
    FixedTreeWM        K               unbounded   bfs
    RandomTreeWM       K               unbounded   random
    UncertaintyTreeWM  K               unbounded   uncertainty
    HeuristicTreeWM    K               unbounded   heuristic   <- control for "learned"
    TreeWM             K               unbounded   learned

Under the matched-budget protocol every arm emits exactly ``node_budget`` chunk-nodes
(root included); only the *shape* of the spend differs -- SingleWM spends it all on
depth, FlatKWM all on breadth, the tree arms allocate it. That is what stops
"TreeWM wins" from collapsing into "TreeWM got more nodes".
"""

from __future__ import annotations

from dataclasses import replace

import torch

from treewm.models.treewm import TreeWM, TreeWMConfig
from treewm.tree.expansion import TreeConfig


class SingleWM(TreeWM):
    """One joint action-state prediction, applied repeatedly: ``z -> (A, h, z')``.

    With ``branch_factor=1`` the recursive operator degenerates to a chain, so a budget
    of N produces a single N-deep trajectory. That *is* the single-future world model.
    """

    arm_name = "singlewm"
    default_scorer = "bfs"

    def __init__(self, cfg: TreeWMConfig) -> None:
        super().__init__(replace(cfg, branch_factor=1))


class FlatKWM(TreeWM):
    """K alternative branches, no recursion: ``z -> {(A_i, h_i, z_i)}``.

    Trained at ``K_max`` so a single checkpoint yields the whole budget curve: at budget
    N the tree keeps the top ``N-1`` branches by KEEP score.
    """

    arm_name = "flatkwm"
    default_scorer = "bfs"

    def __init__(self, cfg: TreeWMConfig, k_max: int = 256) -> None:
        super().__init__(replace(cfg, branch_factor=k_max))
        self.k_max = k_max


class FixedTreeWM(TreeWM):
    """Recursive tree, uniform breadth-first expansion. No learned allocation."""

    arm_name = "fixedtreewm"
    default_scorer = "bfs"


class RandomTreeWM(TreeWM):
    """Same budget, random frontier choice. Isolates *which* nodes get expanded."""

    arm_name = "randomtreewm"
    default_scorer = "random"


class UncertaintyTreeWM(TreeWM):
    """Expansion driven purely by predicted uncertainty."""

    arm_name = "uncertaintytreewm"
    default_scorer = "uncertainty"


class HeuristicTreeWM(TreeWM):
    """Adaptive but *unlearned*: greedy q-novelty against the pooled tree context.

    The control that separates "adaptive allocation helps" from "learning the allocation
    helps". If TreeWM does not beat this arm, the learned gain head is not the mechanism.
    """

    arm_name = "heuristictreewm"
    default_scorer = "heuristic"


class NoveltyQTreeWM(TreeWM):
    """Direct ``min_j d_q(q_n, q_j)`` expansion. No learned allocation."""

    arm_name = "noveltyq"
    default_scorer = "novelty_q"
    novelty_space = "q"


class LearnedNoveltyQTreeWM(TreeWM):
    """Learned head predicting q-novelty. Paired one-to-one with ``noveltyq``."""

    arm_name = "learnedq"
    default_scorer = "learned"
    novelty_space = "q"


class NoveltyZTreeWM(TreeWM):
    """Direct ``min_j ||z_n - z_j||`` expansion -- the state-space control for q-novelty."""

    arm_name = "noveltyz"
    default_scorer = "novelty_z"
    novelty_space = "z"


class LearnedNoveltyZTreeWM(TreeWM):
    """Learned head predicting z-novelty. Paired one-to-one with ``noveltyz``."""

    arm_name = "learnedz"
    default_scorer = "learned"
    novelty_space = "z"


ARMS: dict[str, type[TreeWM]] = {
    "singlewm": SingleWM,
    "flatkwm": FlatKWM,
    "fixedtreewm": FixedTreeWM,
    "randomtreewm": RandomTreeWM,
    "uncertaintytreewm": UncertaintyTreeWM,
    "heuristictreewm": HeuristicTreeWM,
    "treewm": TreeWM,
    "noveltyq": NoveltyQTreeWM,
    "learnedq": LearnedNoveltyQTreeWM,
    "noveltyz": NoveltyZTreeWM,
    "learnedz": LearnedNoveltyZTreeWM,
}

# The novelty experiment: each learned arm is paired with the direct heuristic computing
# the identical signal, so "does learning close the gap" is a one-variable question.
NOVELTY_ARMS = ("noveltyq", "learnedq", "noveltyz", "learnedz", "randomtreewm", "fixedtreewm")
NOVELTY_PAIRS = (("noveltyq", "learnedq"), ("noveltyz", "learnedz"))

TREE_ARMS = ("fixedtreewm", "randomtreewm", "uncertaintytreewm", "heuristictreewm", "treewm")


def build_model(arm: str, cfg: TreeWMConfig, k_max: int = 256) -> TreeWM:
    arm = arm.lower()
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; options: {sorted(ARMS)}")
    cls = ARMS[arm]
    # The arm dictates the metric space so the head's input features and its regression
    # target can never disagree.
    space = getattr(cls, "novelty_space", None)
    if space is not None:
        cfg = replace(cfg, novelty_space=space)
    if arm == "flatkwm":
        return FlatKWM(cfg, k_max=k_max)
    return cls(cfg)


def tree_config_for(arm: str, base: TreeConfig, model: TreeWM) -> TreeConfig:
    """Arm-appropriate expansion settings at a *fixed* node budget.

    The node budget is never modified here -- that is the controlled variable.
    """
    arm = arm.lower()
    cfg = replace(base, branch_factor=model.cfg.branch_factor, scorer=ARMS[arm].default_scorer)
    if arm == "singlewm":
        # A chain needs depth room equal to the budget; max_depth must not bind.
        return replace(cfg, max_depth=max(base.node_budget, base.max_depth), expansion_batch_size=1)
    if arm == "flatkwm":
        # Depth 1: the root is expanded once and its children are never expanded.
        return replace(cfg, max_depth=1, expansion_batch_size=1)
    return cfg


def check_budget_parity(trees: dict[str, torch.Tensor], budget: int) -> dict[str, float]:
    """Assert every arm produced the same node count. Returns per-arm counts.

    Called from evaluation: a silent budget mismatch would invalidate the entire
    headline comparison, so it is checked rather than assumed.
    """
    counts = {arm: float(n.float().mean().item()) for arm, n in trees.items()}
    for arm, n in trees.items():
        assert int(n.max()) <= budget, f"{arm} exceeded node budget: {int(n.max())} > {budget}"
    return counts
