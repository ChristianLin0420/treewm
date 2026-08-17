"""Render the full tree-visualisation suite for a checkpoint into a TensorBoard run.

Runs standalone so any checkpoint -- including screening runs that trained before the
suite existed -- can be visualised with the *same fixed anchors*, which is what makes two
runs comparable by flipping between them in TensorBoard.

    python scripts/render_trees.py --checkpoint <ck> --scorer random --budget 64
    python scripts/render_trees.py --runs runs_screen --all --budget 64
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treewm.data.maze_utils import MazeSpec
from treewm.data.ogbench_dataset import load_ogbench
from treewm.evaluation import tree_viz as tv
from treewm.evaluation.grounding import ground_tree, restore_state, save_state
from treewm.evaluation.tree_stats import depth_histogram, structural_summary
from treewm.logging.tensorboard import TreeWMLogger
from treewm.models.baselines import tree_config_for
from treewm.utils import config as cfg_utils
from treewm.utils.provenance import provenance, write_artifact
from treewm.utils.rng import make_generator
from treewm.utils.seeding import seed_everything
from scripts.eval import load_run

REPO = Path(__file__).resolve().parents[1]


@torch.no_grad()
def render_checkpoint(ck: Path, scorer: str | None, budget: int, num_anchors: int,
                      device, ground_subset: int = 2, video: bool = True, step: int = 0):
    model, normalizer, cfg, payload = load_run(str(ck), device)
    if model.decoder is None:
        print(f"  skip {ck}: no decoder")
        return None
    arm = ck.parts[-4]
    env = load_ogbench(cfg.env.name, dataset_dir=cfg.env.dataset_dir, env_only=True)
    spec = MazeSpec.from_env(env)
    anchors = tv.build_anchors(spec, num=num_anchors)

    tc = tree_config_for(arm, replace(cfg_utils.tree_config(cfg), node_budget=budget), model)
    if scorer:
        tc = replace(tc, scorer=scorer)

    run_dir = ck.parent.parent / f"viz_{tc.scorer}_b{budget}"
    logger = TreeWMLogger(run_dir, is_main=True)
    stats_acc: list[dict] = []

    for a in range(len(anchors)):
        start, goal = anchors.starts[a], anchors.goals[a]
        obs = torch.from_numpy(normalizer.norm_obs(start[None])).to(device)
        goal_n = torch.from_numpy(normalizer.norm_obs(goal[None])).to(device)
        z = model.encode(obs)
        tree, _ = model.generate(
            z, tc, generator=make_generator(0, 'viz', device),
            goal_obs=goal_n if tc.scorer in
            ("goal", "goal_novelty", "diverse_goal", "broad_to_focused") else None
        )

        # leaf selection exactly as the planner does it (decoded position)
        node_obs = model.decoder(tree.latent)
        d = torch.linalg.vector_norm(node_obs - goal_n.unsqueeze(1), dim=-1)
        d = d.masked_fill(~tree.valid, float("inf"))
        d[:, 0] = float("inf")
        selected = int(d.argmin(dim=1).item())

        r = tv.TreeRender.from_tree(model, tree, normalizer, goal, start, 0, selected)
        name = anchors.names[a]
        logger.figure(f"viz/tree_xy_depth/{name}", tv.view_depth(r, spec, name), step)
        logger.figure(f"viz/tree_xy_expansion_order/{name}", tv.view_expansion_order(r, spec, name), step)
        logger.figure(f"viz/tree_xy_goal_distance/{name}", tv.view_goal_distance(r, spec, name), step)
        logger.figure(f"viz/tree_xy_root_subtree/{name}", tv.view_root_subtree(r, spec, name), step)
        logger.figure(f"viz/tree_horizon/{name}", tv.view_horizon(r, spec, name), step)
        logger.figure(f"viz/tree_selected_path/{name}", tv.view_selected_path(r, spec, name), step)
        logger.figure(f"viz/tree_topology/{name}", tv.view_topology(r, name), step)

        # grounded comparison on a small fixed subset only
        if a < ground_subset:
            env.reset(seed=a)
            env.unwrapped.set_xy(np.asarray(start, dtype=np.float64))
            entry = save_state(env)
            actual, grounded = ground_tree(env, tree, normalizer)
            restore_state(env, entry)
            logger.figure(f"viz/predicted_vs_grounded/{name}",
                          tv.view_predicted_vs_grounded(r, actual, grounded, spec, name), step)

        # expansion animation: rebuild the tree batch-by-batch
        if video and a < ground_subset:
            frames = []
            for b_stop in range(1, int(tree.order[0][tree.valid[0]].max().item()) + 2):
                sub = tree.order[0].cpu().numpy()
                keep = (sub >= 0) & (sub < b_stop) & tree.valid[0].cpu().numpy()
                rr = tv.TreeRender.from_tree(model, tree, normalizer, goal, start, 0, selected)
                rr.valid = keep
                frames.append(rr)
            vid = tv.expansion_video(model, frames, normalizer, spec, goal, start)
            if vid is not None:
                try:
                    logger._writer.add_video(f"viz/expansion/{name}", vid, step, fps=2)
                except Exception as exc:
                    print(f"    video skipped ({exc})")

        stats_acc.append(structural_summary(tree, model, normalizer))
        logger.histogram(f"tree/depth_histogram/{name}", depth_histogram(tree), step)

    merged: dict[str, float] = {}
    for k in {k for s in stats_acc for k in s}:
        vals = [s[k] for s in stats_acc if k in s]
        merged[k] = float(np.mean(vals))
    logger.scalars(merged, step)
    logger.close()

    prov = provenance(ck, cfg, extra={"scorer": tc.scorer, "node_budget": budget,
                                      "num_anchors": len(anchors)})
    write_artifact(ck.parent.parent / f"tree_stats_{tc.scorer}_b{budget}.json", merged, prov)
    print(f"  {arm}/{ck.parts[-3]} scorer={tc.scorer} -> {run_dir}")
    return merged


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--runs", default="experiments/05-design-space/runs/screen")
    p.add_argument("--all", action="store_true")
    p.add_argument("--scorer", default=None)
    p.add_argument("--budget", type=int, default=64)
    p.add_argument("--anchors", type=int, default=8)
    p.add_argument("--ground-subset", type=int, default=2)
    p.add_argument("--no-video", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(0)

    if args.checkpoint:
        cks = [Path(args.checkpoint)]
    else:
        cks = sorted((REPO / args.runs).glob("*/*/*/checkpoints/latest.pt"))
        if not args.all:
            cks = cks[:1]
    print(f"[viz] {len(cks)} checkpoints, budget {args.budget}")
    for ck in cks:
        try:
            render_checkpoint(ck, args.scorer, args.budget, args.anchors, device,
                              args.ground_subset, not args.no_video)
        except Exception as exc:
            print(f"  FAILED {ck}: {exc}")


if __name__ == "__main__":
    main()
