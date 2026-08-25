"""Evaluate a checkpoint at one node budget.

    python scripts/eval.py checkpoint=runs/.../latest.pt tree.node_budget=64

The checkpoint carries its own config and normaliser, so the only things worth
overriding here are the budget, the task split and the episode count.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treewm.data.ogbench_dataset import Normalizer, load_ogbench
from treewm.evaluation.rollout import evaluate
from treewm.evaluation.domains import get_domain
from treewm.evaluation.tasks import build_tasks, describe_tasks
from treewm.models.baselines import build_model, tree_config_for
from treewm.planning.goal_planner import GoalPlanner
from treewm.utils import config as cfg_utils
from treewm.utils.seeding import seed_everything


def load_run(checkpoint: str, device: torch.device, overrides: DictConfig | None = None):
    """Rebuild model + normaliser + config from a checkpoint."""
    payload = torch.load(checkpoint, map_location=str(device), weights_only=False)
    cfg = OmegaConf.create(payload["config"])
    if overrides is not None:
        cfg = OmegaConf.merge(cfg, overrides)

    model = build_model(cfg.arm, cfg_utils.model_config(cfg), k_max=int(cfg.model.flatk_max)).to(device)
    # V2 adds set-attention parameters lazily for v1 checkpoint compatibility. Create
    # the architecture recorded in the checkpoint before loading its state dict; every
    # standalone analysis imports this helper, so the rule is centralised here.
    model.gain_head.set_set_aware(bool(cfg.losses.get("gain_set_context", False)))
    model.load_state_dict(payload["model"])
    model.eval()

    if "normalizer" in payload:
        normalizer = Normalizer.from_state_dict(payload["normalizer"])
    else:
        _, train, _ = load_ogbench(cfg.env.name, dataset_dir=cfg.env.dataset_dir)
        normalizer = Normalizer.fit(train["observations"], train["actions"])
    return model, normalizer, cfg, payload


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    checkpoint = cfg.get("checkpoint")
    if not checkpoint:
        raise SystemExit("usage: python scripts/eval.py checkpoint=<path> [tree.node_budget=64]")

    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu")
    seed_everything(int(cfg.seed))

    overrides = OmegaConf.create(
        {"tree": OmegaConf.to_container(cfg.tree, resolve=True),
         "eval": OmegaConf.to_container(cfg.eval, resolve=True),
         "planner": OmegaConf.to_container(cfg.planner, resolve=True)}
    )
    model, normalizer, run_cfg, _ = load_run(str(checkpoint), device, overrides)

    env = load_ogbench(run_cfg.env.name, dataset_dir=run_cfg.env.dataset_dir, env_only=True)
    tasks = build_tasks(
        env, str(run_cfg.eval.task_split), int(run_cfg.eval.num_hard_tasks),
        float(run_cfg.eval.hard_percentile), int(run_cfg.eval.seed),
    )
    tree_cfg = tree_config_for(run_cfg.arm, cfg_utils.tree_config(run_cfg), model)
    domain = get_domain(run_cfg.env.name)
    planner = GoalPlanner(
        model, normalizer, tree_cfg, cfg_utils.planner_config(run_cfg), device, domain=domain
    )

    print(f"[eval] arm={run_cfg.arm} budget={tree_cfg.node_budget} scorer={tree_cfg.scorer}")
    print(f"[eval] tasks ({run_cfg.eval.task_split}):\n{describe_tasks(tasks)}")

    metrics = evaluate(
        env, planner, tasks, int(run_cfg.eval.episodes_per_task),
        int(run_cfg.planner.max_env_steps), int(run_cfg.eval.seed), domain=domain,
    )
    for key in sorted(metrics):
        print(f"  {key:44s} {metrics[key]:.4f}")

    out = Path(checkpoint).parent.parent / f"eval_budget{tree_cfg.node_budget}.json"
    out.write_text(json.dumps({"arm": str(run_cfg.arm), "budget": tree_cfg.node_budget, **metrics}, indent=2))
    print(f"[eval] wrote {out}")


if __name__ == "__main__":
    main()
