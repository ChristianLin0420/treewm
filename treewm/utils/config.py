"""Translate OmegaConf configs into the typed dataclasses the library uses.

The library modules take plain dataclasses, not DictConfigs, so that they are usable
from a notebook or a unit test without Hydra in the picture. This file is the single
boundary where the two representations meet.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf

from treewm.data.future_sets import FutureSetConfig
from treewm.data.retrieval_index import RetrievalConfig
from treewm.losses.total import LossConfig, LossWeights
from treewm.models.treewm import TreeWMConfig
from treewm.planning.goal_planner import PlannerConfig
from treewm.tree.expansion import TreeConfig
from treewm.tree.matching import MatchingConfig


def to_container(cfg: Any) -> Any:
    if isinstance(cfg, (DictConfig,)) or OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def future_set_config(cfg: DictConfig) -> FutureSetConfig:
    raw = to_container(cfg.future_sets)
    raw["horizons"] = tuple(raw["horizons"])
    # These are dataset-loading concerns, not part of the future-set definition itself.
    raw.pop("cache", None)
    raw.pop("shared_cache", None)
    raw.pop("recipe_anchor_policy", None)
    return FutureSetConfig(**raw)


def retrieval_config(cfg: DictConfig) -> RetrievalConfig:
    raw = to_container(cfg.retrieval)
    # endpoint_horizon and grid_resolution belong to the quantiser, not the index.
    raw.pop("endpoint_horizon", None)
    raw.pop("grid_resolution", None)
    return RetrievalConfig(**raw)


def model_config(cfg: DictConfig) -> TreeWMConfig:
    raw = to_container(cfg.model)
    raw.pop("flatk_max", None)
    raw["horizons"] = tuple(raw["horizons"])
    raw["scales"] = tuple((str(n), int(s), float(w)) for n, s, w in raw["scales"])
    raw["obs_dim"] = int(cfg.env.obs_dim)
    raw["action_dim"] = int(cfg.env.action_dim)
    # Chunk length and candidate horizons must agree with how the data was built, or
    # the horizon head would be predicting into a different index space.
    raw["h_max"] = int(cfg.future_sets.h_max)
    raw["horizons"] = tuple(int(h) for h in cfg.future_sets.horizons)
    return TreeWMConfig(**raw)


def tree_config(cfg: DictConfig) -> TreeConfig:
    raw = to_container(cfg.tree)
    scorer = raw.pop("scorer", None)
    tc = TreeConfig(**raw)
    if scorer:
        # Recorded as an override so tree_config_for cannot silently replace it with the
        # arm's default scorer.
        tc.scorer = scorer
        tc.scorer_override = scorer
    return tc


def matching_config(cfg: DictConfig) -> MatchingConfig:
    return MatchingConfig(**to_container(cfg.matching))


def loss_config(cfg: DictConfig) -> LossConfig:
    raw = to_container(cfg.losses)
    weights = LossWeights(**raw.pop("weights"))
    enabled = raw.pop("enabled")
    if raw.get("multistep_depth_weights"):
        raw["multistep_depth_weights"] = tuple(float(x) for x in raw["multistep_depth_weights"])
    else:
        raw["multistep_depth_weights"] = ()
    return LossConfig(weights=weights, enabled=dict(enabled), **raw)


def planner_config(cfg: DictConfig) -> PlannerConfig:
    return PlannerConfig(**to_container(cfg.planner))


def config_text(cfg: DictConfig) -> str:
    return OmegaConf.to_yaml(cfg, resolve=True)


def flat_hparams(cfg: DictConfig, prefix: str = "") -> dict[str, Any]:
    """Flatten a config into scalar hparams for ``add_hparams``."""
    out: dict[str, Any] = {}
    raw = to_container(cfg)

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(node, (list, tuple)):
            out[path] = str(list(node))
        else:
            out[path] = node

    walk(raw, prefix)
    return out
