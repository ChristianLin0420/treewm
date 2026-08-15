"""TreeWM training entry point.

    torchrun --nproc_per_node=2 scripts/train.py experiment=pointmaze_treewm seed=0

Runs single-process too (no torchrun) with identical semantics. Only rank 0 creates the
TensorBoard writer, saves checkpoints, renders plots and prints progress; every logged
scalar is reduced across ranks first (spec section 23).
"""

from __future__ import annotations

import math
import os
from dataclasses import replace
import sys
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treewm.data.ogbench_dataset import build_datasets
from treewm.data.retrieval_index import LatentIndex, compute_endpoint_cells
from treewm.data.samplers import InfiniteLoader, build_dataloader, to_device
from treewm.evaluation import diagnostics as diag
from treewm.evaluation import tree_stats as tstats
from treewm.evaluation import tree_viz as tv
from treewm.tree.frontier import GOAL_AWARE_SCORERS
from treewm.evaluation.rollout import evaluate
from treewm.evaluation.tasks import build_tasks, describe_tasks
from treewm.evaluation.coverage import StateQuantizer
from treewm.data.maze_utils import MazeSpec
from treewm.logging.metrics import MetricTracker
from treewm.logging.tensorboard import TreeWMLogger
from treewm.losses.expansion_losses import novelty_gain_loss
from treewm.losses.recursive_losses import multi_step_recursive_loss, scheduled_sampling_schedule
from treewm.losses.total import compute_branch_losses, compute_expansion_gain_loss
from treewm.models.baselines import build_model, tree_config_for
from treewm.planning.goal_planner import GoalPlanner
from treewm.utils import config as cfg_utils
from treewm.utils.checkpoint import CheckpointManager, load_checkpoint
from treewm.utils.distributed import (
    all_reduce_mean,
    cleanup_distributed,
    is_distributed,
    setup_distributed,
    unwrap_model,
)
from treewm.utils.meta import build_run_dir, count_parameters, env_summary, git_commit, hostname
from treewm.utils.rng import RngStreams, make_generator
from treewm.utils.seeding import seed_everything


def should_visualise(step: int, cfg) -> bool:
    """Dense visualisation early, sparser later (spec section 12).

    Tree structure changes fastest in the first couple of thousand steps, so a single
    fixed stride either floods the run or misses the interesting phase entirely.
    """
    if step == 0:
        return False
    early_until = int(cfg.train.viz_early_until)
    every = int(cfg.train.viz_every_early) if step <= early_until else int(cfg.train.viz_every)
    return step % max(every, 1) == 0


def build_scheduler(optimizer, cfg):
    """Linear warmup then cosine decay to ``min_lr_scale`` of the peak."""
    warmup = max(1, int(cfg.train.warmup_steps))
    total = max(warmup + 1, int(cfg.train.steps))
    floor = float(cfg.train.min_lr_scale)

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total - warmup)
        return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@hydra.main(version_base=None, config_path="../configs", config_name="base")
def main(cfg: DictConfig) -> None:
    dist_info = setup_distributed()
    seed_everything(int(cfg.seed), rank=dist_info.rank)
    device = torch.device(
        f"cuda:{dist_info.local_rank}" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    )

    run_dir = build_run_dir(cfg.run_root, cfg.env.short_name, cfg.arm, int(cfg.seed))
    if cfg.run_name:
        run_dir = Path(cfg.run_root) / cfg.env.short_name / cfg.arm / str(cfg.run_name)
    logger = TreeWMLogger(run_dir, is_main=dist_info.is_main)
    ckpt = CheckpointManager(run_dir / "checkpoints", enabled=dist_info.is_main)

    if dist_info.is_main:
        print(f"[treewm] arm={cfg.arm} env={cfg.env.name} seed={cfg.seed}")
        print(f"[treewm] run_dir={run_dir}")

    # ---------------------------------------------------------------- data
    future_cfg = cfg_utils.future_set_config(cfg)
    env, train_ds, val_ds, normalizer = build_datasets(
        cfg.env.name,
        future_cfg,
        dataset_dir=cfg.env.dataset_dir,
        xy_dims=tuple(cfg.env.xy_dims),
        max_train_anchors=int(cfg.train.max_train_anchors),
        max_val_anchors=int(cfg.train.max_val_anchors),
        seed=int(cfg.seed),
    )
    # AntMaze env configs leave obs/action dims null; fill them from the loaded data so
    # a new environment needs no hand-edited dimensions.
    if cfg.env.obs_dim is None or cfg.env.action_dim is None:
        with open_dict(cfg):
            cfg.env.obs_dim = int(train_ds.obs_dim)
            cfg.env.action_dim = int(train_ds.act_dim)
        if dist_info.is_main:
            print(f"[treewm] inferred obs_dim={cfg.env.obs_dim} action_dim={cfg.env.action_dim}")

    if dist_info.is_main:
        print(f"[treewm] train {train_ds.summary()}")
        print(f"[treewm] val   {val_ds.summary()}")

    # Separate loader generators: re-creating the val iterator inside a diagnostic must
    # not advance the stream the training loader samples from.
    train_loader, train_sampler = build_dataloader(
        train_ds, int(cfg.train.batch_size), shuffle=True,
        num_workers=int(cfg.train.num_workers), seed=int(cfg.seed),
        generator=make_generator(int(cfg.seed), "train"),
    )
    val_loader, _ = build_dataloader(
        val_ds, int(cfg.train.batch_size), shuffle=False,
        num_workers=max(2, int(cfg.train.num_workers) // 4), seed=int(cfg.seed),
        generator=make_generator(int(cfg.seed), "viz"),
    )
    train_iter = InfiniteLoader(train_loader, train_sampler)

    # ---------------------------------------------------------------- model
    model_cfg = cfg_utils.model_config(cfg)
    model = build_model(cfg.arm, model_cfg, k_max=int(cfg.model.flatk_max)).to(device)
    base_tree_cfg = cfg_utils.tree_config(cfg)
    tree_cfg = tree_config_for(cfg.arm, base_tree_cfg, model)
    # Small, uniform tree for gain-head supervision; evaluation still uses tree_cfg.
    gain_tree_cfg = tree_config_for(
        cfg.arm, replace(base_tree_cfg, node_budget=int(cfg.train.gain_tree_budget)), model
    )
    match_cfg = cfg_utils.matching_config(cfg)
    loss_cfg = cfg_utils.loss_config(cfg)
    planner_cfg = cfg_utils.planner_config(cfg)

    # Four isolated streams so diagnostics cannot perturb training or planning.
    rng = RngStreams(seed=int(cfg.seed), device=device)
    model._horizon_gen = make_generator(int(cfg.seed), "train", device)

    ddp_model = model
    if is_distributed():
        # find_unused_parameters is required: the expansion-gain stage runs on a stride
        # and the bootstrap-only tree_signature module may never be touched.
        ddp_model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[dist_info.local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg.train.lr), weight_decay=float(cfg.train.weight_decay)
    )
    scheduler = build_scheduler(optimizer, cfg)
    use_bf16 = bool(cfg.train.bf16) and device.type == "cuda"
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float32

    # ------------------------------------------------- retrieval / gain target
    quantizer = StateQuantizer(
        resolution=float(cfg.retrieval.grid_resolution), dims=tuple(cfg.env.xy_dims)
    )
    endpoint_cells, endpoint_valid = compute_endpoint_cells(
        train_ds.obs_norm, train_ds.index, quantizer, int(cfg.retrieval.endpoint_horizon)
    )
    latent_index = LatentIndex(
        train_ds.obs_norm, endpoint_cells, endpoint_valid,
        cfg_utils.retrieval_config(cfg), device, seed=int(cfg.seed),
    )

    maze_spec = MazeSpec.from_env(env)
    anchors = tv.build_anchors(maze_spec, num=int(cfg.train.viz_anchors))
    tasks = build_tasks(
        env, str(cfg.eval.task_split), int(cfg.eval.num_hard_tasks),
        float(cfg.eval.hard_percentile), int(cfg.eval.seed),
    )

    # --------------------------------------------------------------- metadata
    if dist_info.is_main:
        logger.text("config/full", cfg_utils.config_text(cfg))
        logger.text("meta/tasks", describe_tasks(tasks))
        info = env_summary()
        for key, value in {
            "meta/git_commit": git_commit(Path(__file__).resolve().parents[1]),
            "meta/hostname": hostname(),
            "meta/arm": str(cfg.arm),
            "meta/env": str(cfg.env.name),
            **{f"meta/{k}": v for k, v in info.items()},
        }.items():
            logger.text(key, str(value))
        logger.scalar("meta/num_parameters", count_parameters(model), 0)
        logger.scalar("meta/world_size", dist_info.world_size, 0)
        logger.scalar("meta/seed", int(cfg.seed), 0)
        print(f"[treewm] parameters: {count_parameters(model)/1e6:.2f}M | scorer={tree_cfg.scorer}")

    # ----------------------------------------------------------------- resume
    start_step = 0
    resume_path = None
    if cfg.resume == "auto":
        candidate = run_dir / "checkpoints" / "latest.pt"
        resume_path = candidate if candidate.exists() else None
    elif cfg.resume:
        resume_path = Path(cfg.resume)
    if resume_path is not None and resume_path.exists():
        payload = load_checkpoint(resume_path, model, optimizer, scheduler, map_location=str(device))
        start_step = int(payload.get("step", 0))
        train_iter.load_state_dict(payload.get("loader", {}))
        ckpt.load_state_dict(payload.get("checkpoint_manager", {}))
        if dist_info.is_main:
            print(f"[treewm] resumed from {resume_path} at step {start_step}")

    # -------------------------------------------------------------- train loop
    tracker = MetricTracker(device)
    accum = max(1, int(cfg.train.grad_accum))
    total_steps = int(cfg.train.steps)
    progress = tqdm(
        range(start_step, total_steps), initial=start_step, total=total_steps,
        disable=not dist_info.is_main, desc=cfg.arm,
    )
    last_log = time.perf_counter()
    examples_since_log = 0

    for step in progress:
        model.train()
        latent_index.refresh(model.encoder, step)
        optimizer.zero_grad(set_to_none=True)

        for micro in range(accum):
            batch = to_device(next(train_iter), device)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_bf16):
                loss, metrics, artifacts = compute_branch_losses(
                    model, batch, loss_cfg, match_cfg, step=step
                )

                if loss_cfg.on("multistep"):
                    p_ss = scheduled_sampling_schedule(
                        step, float(loss_cfg.scheduled_sampling_p),
                        int(loss_cfg.scheduled_sampling_warmup),
                    )
                    ms_loss, ms_metrics = multi_step_recursive_loss(
                        model, batch, scheduled_sampling_p=p_ss,
                        depth_weights=loss_cfg.multistep_depth_weights or None,
                    )
                    loss = loss + loss_cfg.weights.multistep * ms_loss
                    metrics["train/loss_multistep"] = float(ms_loss.detach().item())
                    metrics.update(ms_metrics)

                if loss_cfg.on("expand") and step % int(cfg.train.gain_loss_every) == 0:
                    n_gain = min(int(cfg.train.gain_batch_size), batch["obs"].shape[0])
                    if str(cfg.losses.gain_target) == "novelty":
                        gain_loss, gain_metrics = novelty_gain_loss(
                            model, artifacts["z"][:n_gain], gain_tree_cfg,
                            space=str(cfg.model.novelty_space), generator=rng.planner,
                        )
                    else:
                        gain_loss, gain_metrics = compute_expansion_gain_loss(
                            model, artifacts["z"][:n_gain], gain_tree_cfg, latent_index, quantizer,
                        )
                    loss = loss + loss_cfg.weights.expand * loss_cfg.scale('expand', step) * gain_loss
                    metrics["train/loss_expand"] = float(gain_loss.detach().item())
                    metrics.update(gain_metrics)

            (loss / accum).backward()
            tracker.add_many(metrics, count=batch["obs"].shape[0])
            examples_since_log += batch["obs"].shape[0]

            if micro == accum - 1 and step % int(cfg.train.hist_every) == 0:
                tracker.add_hist("tree/keep_scores", artifacts["keep"])
                tracker.add_hist("tree/mass_scores", artifacts["mass"])
                tracker.add_hist("tree/uncertainty", artifacts["uncertainty"])
                tracker.add_hist("tree/expansion_gain", artifacts["gain_prior"])
                tracker.add_hist("tree/predicted_horizons", artifacts["horizon_pred"].float())
                tracker.add_hist(
                    "tree/effective_branching_factor_hist", (artifacts["keep"] > 0.5).float().sum(-1)
                )

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg.train.grad_clip))
        optimizer.step()
        scheduler.step()
        tracker.add("train/grad_norm", float(grad_norm))
        tracker.add("train/learning_rate", scheduler.get_last_lr()[0])
        tracker.add("train/weight_decay", float(cfg.train.weight_decay))

        # ------------------------------------------------------------- logging
        if step % int(cfg.train.log_every) == 0:
            elapsed = max(time.perf_counter() - last_log, 1e-6)
            scalars = tracker.compute(reduce=True)
            steps_per_s = int(cfg.train.log_every) / elapsed
            scalars["train/steps_per_second"] = steps_per_s
            scalars["train/examples_per_second"] = all_reduce_mean(
                examples_since_log / elapsed, device
            ) * max(dist_info.world_size, 1)
            if device.type == "cuda":
                scalars["train/gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(device) / 1e9
                scalars["train/gpu_memory_reserved_gb"] = torch.cuda.memory_reserved(device) / 1e9
            logger.scalars(scalars, step)
            logger.histograms(tracker.histograms(), step)
            if dist_info.is_main:
                progress.set_postfix(
                    loss=f"{scalars.get('train/loss_total', 0):.3f}",
                    ebf=f"{scalars.get('tree/effective_branching_factor', 0):.2f}",
                    rare=f"{scalars.get('tree/rare_mode_recall', 0):.2f}",
                )
            tracker.reset()
            last_log = time.perf_counter()
            examples_since_log = 0

        # --------------------------------------------------------- diagnostics
        if step > 0 and step % int(cfg.train.diag_every) == 0:
            model.eval()
            with torch.no_grad():
                dbatch = to_device(next(iter(val_loader)), device)
                dmetrics = {}
                dmetrics.update(diag.q_vs_z_retrieval(model, dbatch))
                dmetrics.update(diag.branching_diversity_correlation(model, dbatch))
                dmetrics.update(diag.geometry_sanity(model, dbatch, maze_spec, normalizer))
            logger.scalars({k: all_reduce_mean(v, device) for k, v in dmetrics.items()}, step)

        # --------------------------------------------------------- validation
        if step > 0 and step % int(cfg.train.ckpt_every) == 0:
            val_tracker = MetricTracker(device)
            model.eval()
            with torch.no_grad():
                for i, vbatch in enumerate(val_loader):
                    if i >= int(cfg.train.val_batches):
                        break
                    vbatch = to_device(vbatch, device)
                    _, vmetrics, _ = compute_branch_losses(model, vbatch, loss_cfg, match_cfg)
                    val_tracker.add_many(
                        {k.replace("train/", "val/"): v for k, v in vmetrics.items() if "loss" in k},
                        count=vbatch["obs"].shape[0],
                    )
            vscalars = val_tracker.compute(reduce=True)
            logger.scalars(vscalars, step)
            if dist_info.is_main:
                payload = dict(
                    model=model, optimizer=optimizer, scheduler=scheduler, step=step,
                    epoch=train_iter.epoch, config=OmegaConf.to_container(cfg, resolve=True),
                    extra={
                        "loader": train_iter.state_dict(),
                        "checkpoint_manager": ckpt.state_dict(),
                        "normalizer": normalizer.state_dict(),
                    },
                )
                ckpt.save_latest(**payload)
                ckpt.maybe_save_val_loss(vscalars.get("val/loss_total", float("inf")), **payload)

        # ---------------------------------------------------- goal evaluation
        if step > 0 and step % int(cfg.train.eval_every) == 0 and dist_info.is_main:
            model.eval()
            planner = GoalPlanner(model, normalizer, tree_cfg, planner_cfg, device,
                                  generator=rng.reset("eval"))
            emetrics = evaluate(
                env, planner, tasks, int(cfg.eval.episodes_per_task),
                int(cfg.planner.max_env_steps), int(cfg.eval.seed),
            )
            logger.scalars(emetrics, step)
            ckpt.maybe_save_success(
                emetrics["eval/success_rate"], model=model, optimizer=optimizer,
                scheduler=scheduler, step=step, epoch=train_iter.epoch,
                config=OmegaConf.to_container(cfg, resolve=True),
                extra={"normalizer": normalizer.state_dict()},
            )
            print(f"[treewm] step {step} success={emetrics['eval/success_rate']:.3f}")

        # ------------------------------------------------------ visualisations
        if should_visualise(step, cfg) and dist_info.is_main:
            model.eval()
            try:
                logger.figure(
                    "viz/branching_factor_heatmap",
                    diag.branching_factor_heatmap(model, maze_spec, normalizer, device), step,
                )
                logger.figure(
                    "viz/expansion_gain_heatmap",
                    diag.expansion_gain_heatmap(model, maze_spec, normalizer, device), step,
                )
                with torch.no_grad():
                    vbatch = to_device(next(iter(val_loader)), device)
                    fig = diag.q_pca_plot(model, vbatch, normalizer, maze_spec)
                    if fig is not None:
                        logger.figure("viz/q_pca", fig, step)
                    # Fixed anchors, identical across every run, so two runs can be
                    # compared by flipping between them in TensorBoard.
                    for a in range(min(int(cfg.train.viz_anchors), len(anchors))):
                        start, goal = anchors.starts[a], anchors.goals[a]
                        obs_a = torch.from_numpy(normalizer.norm_obs(start[None])).to(device)
                        goal_a = torch.from_numpy(normalizer.norm_obs(goal[None])).to(device)
                        tree, _ = model.generate(
                            model.encode(obs_a), tree_cfg, generator=rng.viz,
                            goal_obs=goal_a if tree_cfg.scorer in GOAL_AWARE_SCORERS else None,
                        )
                        node_obs = model.decoder(tree.latent)
                        gd = torch.linalg.vector_norm(node_obs - goal_a.unsqueeze(1), dim=-1)
                        gd = gd.masked_fill(~tree.valid, float("inf")); gd[:, 0] = float("inf")
                        r = tv.TreeRender.from_tree(model, tree, normalizer, goal, start, 0,
                                                    int(gd.argmin(dim=1).item()))
                        nm = anchors.names[a]
                        logger.figure(f"viz/tree_xy_depth/{nm}", tv.view_depth(r, maze_spec, nm), step)
                        logger.figure(f"viz/tree_xy_expansion_order/{nm}",
                                      tv.view_expansion_order(r, maze_spec, nm), step)
                        logger.figure(f"viz/tree_xy_goal_distance/{nm}",
                                      tv.view_goal_distance(r, maze_spec, nm), step)
                        logger.figure(f"viz/tree_xy_root_subtree/{nm}",
                                      tv.view_root_subtree(r, maze_spec, nm), step)
                        logger.figure(f"viz/tree_horizon/{nm}", tv.view_horizon(r, maze_spec, nm), step)
                        logger.figure(f"viz/tree_selected_path/{nm}",
                                      tv.view_selected_path(r, maze_spec, nm), step)
                        logger.scalars(tstats.structural_summary(tree, model, normalizer), step)
                        logger.histogram(f"tree/horizon_hist/{nm}",
                                         tree.action_mask.sum(-1)[tree.valid].float(), step)
            except Exception as exc:  # visualisation must never kill a run
                print(f"[treewm] visualisation skipped at step {step}: {exc}")

    # ------------------------------------------------------------ final eval
    if dist_info.is_main:
        model.eval()
        planner = GoalPlanner(model, normalizer, tree_cfg, planner_cfg, device,
                              generator=rng.reset("eval"))
        final = evaluate(
            env, planner, tasks, int(cfg.eval.episodes_per_task),
            int(cfg.planner.max_env_steps), int(cfg.eval.seed),
        )
        logger.scalars(final, total_steps)
        logger.hparams(
            {
                "arm": str(cfg.arm), "env": str(cfg.env.name), "seed": int(cfg.seed),
                "node_budget": int(cfg.tree.node_budget),
                "branch_factor": int(model.cfg.branch_factor),
                "z_dim": int(cfg.model.z_dim), "q_dim": int(cfg.model.q_dim),
                "coverage_space": str(cfg.losses.coverage_space),
                "context_pooling": str(cfg.tree.context_pooling),
                "scorer": str(tree_cfg.scorer),
            },
            {k: v for k, v in final.items() if np.isfinite(v)},
        )
        ckpt.save_latest(
            model=model, optimizer=optimizer, scheduler=scheduler, step=total_steps,
            epoch=train_iter.epoch, config=OmegaConf.to_container(cfg, resolve=True),
            extra={"normalizer": normalizer.state_dict(), "final_eval": final},
        )
        print(f"[treewm] done. final success={final['eval/success_rate']:.3f}")

    logger.close()
    cleanup_distributed()


if __name__ == "__main__":
    main()
